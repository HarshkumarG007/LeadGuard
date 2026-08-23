"""Offline Policy Optimization Job for LeadGuard.

Runs the expensive prediction, SHAP, and EVI pipeline for all properties and outputs 
a `decision_snapshot.parquet` for O(1) online API retrieval.

Usage:
    python scripts/run_offline_policy_job.py
"""

import logging
import json
import uuid
import time
from datetime import UTC, datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
import shap
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.absolute()))

from leadguard.data.features import build_features
from leadguard.evaluation.fairness import compute_equity_boost
from leadguard.models.uncertainty import compute_predictive_entropy
from leadguard.models.serving import predict_proba, load_serving_model
from leadguard.evaluation.explainability import extract_top_shap_features
from leadguard.models.active_learning import compute_approximate_evi
from leadguard.data.validation import validate_features
from leadguard.models.xgboost_model import XGB_FEATURES
from api.model_loader import get_state

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

def run_job(
    features_path="data/processed/features.parquet",
    output_path="data/processed/decision_snapshot.parquet",
    explanation_path="data/processed/explanation_snapshot.parquet",
    metadata_path="data/processed/decision_snapshot_meta.json",
    scoring_config="configs/scoring.yaml"
):
    logger.info("Starting Offline Policy Job")
    start_time = time.time()
    
    # 1. Load Data
    t0 = time.time()
    df = None
    for p in [features_path, "data/processed/features_sample.parquet", "data/sample/sample_properties.parquet"]:
        if Path(p).exists():
            df = pd.read_parquet(p)
            logger.info(f"Loaded {len(df)} properties from {p}")
            break
            
    if df is None:
        logger.error("No features data found.")
        return
    logger.info(f"Loaded {len(df)} properties in {time.time() - t0:.2f}s")
    
    # 2. Model Inference
    t0 = time.time()
    state = get_state()
    if not state.is_ready:
        logger.error("Model artifacts not loaded.")
        return
        
    X_all = validate_features(df, state.feature_names).values
    probas = predict_proba(state.model, X_all)
    p_lead = probas[:, -1]
    uncertainty = compute_predictive_entropy(probas)
    logger.info(f"Inference completed in {time.time() - t0:.2f}s")
    
    # 3. SHAP
    t0 = time.time()
    try:
        raw_model = state.model
        if hasattr(state.model, "calibrated_classifiers_"):
            # Unwrap CalibratedClassifierCV
            raw_model = state.model.calibrated_classifiers_[0].estimator
        elif hasattr(state.model, "estimator"):
            raw_model = getattr(state.model, "estimator")

        explainer = shap.TreeExplainer(raw_model)
        shap_values = explainer.shap_values(X_all)
        
        # Serialize top SHAP features
        shap_json_list = []
        for i in range(len(X_all)):
            sv_row = shap_values[i] if shap_values.ndim > 1 else shap_values
            top_f = extract_top_shap_features(sv_row, state.feature_names, top_n=5)
            shap_json_list.append(json.dumps(top_f))
    except Exception as e:
        logger.error(f"SHAP failed: {e}")
        shap_json_list = ["[]"] * len(X_all)
    logger.info(f"SHAP completed in {time.time() - t0:.2f}s")
        
    # 4. Equity Boost
    t0 = time.time()
    equity_boost = np.zeros(len(df))
    if not state.fairness_reference.empty and "census_tract" in df.columns:
        from api.main import _read_inspections
        inspections = _read_inspections()
        insp_df = pd.DataFrame([{"census_tract": i.get("census_tract")} for i in inspections if "census_tract" in i])
        
        pred_df = df[["census_tract"]].copy()
        pred_df["p_lead_calibrated"] = p_lead
        boost_series = compute_equity_boost(pred_df, insp_df)
        equity_boost = df["census_tract"].map(boost_series).fillna(0.0).values
    logger.info(f"Equity completed in {time.time() - t0:.2f}s")
        
    # 5. EVI Calculation
    t0 = time.time()
    cfg = yaml.safe_load(Path(scoring_config).read_text())
    base_cfg = cfg.get("scenarios", {}).get("base", {})
    intervention_value = base_cfg.get("intervention", {}).get("value_usd", 5000.0)
    equity_weight = base_cfg.get("equity", {}).get("weight", 1000.0)
    
    evi = compute_approximate_evi(
        p_lead=p_lead,
        intervention_value=intervention_value,
        intervention_cost=0.0,
        equity_boost=equity_boost,
        equity_weight=equity_weight
    )
    logger.info(f"EVI completed in {time.time() - t0:.2f}s")
    
    # 6. Save Decision Snapshot (No SHAP)
    t0 = time.time()
    df["p_lead_calibrated"] = p_lead
    df["uncertainty_score"] = uncertainty
    df["equity_boost"] = equity_boost
    df["evi"] = evi
    
    # Store decisions in memory compact format
    cols_to_keep = ["property_id", "census_tract", "ward", "p_lead_calibrated", "uncertainty_score", "equity_boost", "evi"]
    # keep cols that exist
    cols_to_keep = [c for c in cols_to_keep if c in df.columns]
    decision_df = df[cols_to_keep]
    
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    decision_df.to_parquet(output_path)
    logger.info(f"Decision snapshot saved in {time.time() - t0:.2f}s")
    
    # 7. Save Explanation Snapshot
    t0 = time.time()
    explanation_df = pd.DataFrame({
        "property_id": df["property_id"],
        "shap_features_json": shap_json_list
    })
    Path(explanation_path).parent.mkdir(parents=True, exist_ok=True)
    explanation_df.to_parquet(explanation_path)
    logger.info(f"Explanation snapshot saved in {time.time() - t0:.2f}s")
    
    # 8. Save Metadata with Full Provenance
    snapshot_id = f"snap-{uuid.uuid4().hex[:8]}"
    generated_at = datetime.now(UTC)
    valid_until = generated_at + timedelta(hours=24)
    
    cfg = yaml.safe_load(Path(scoring_config).read_text())
    import hashlib
    policy_hash = hashlib.sha256(json.dumps(cfg, sort_keys=True).encode()).hexdigest()[:12]
    
    # Simple code version via git hash if available
    try:
        import subprocess
        code_version = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"]).decode("ascii").strip()
    except Exception:
        code_version = "unknown"
        
    data_cutoff = datetime.now(UTC).isoformat()
    
    meta = {
        "snapshot_id": snapshot_id,
        "generated_at": generated_at.isoformat(),
        "valid_until": valid_until.isoformat(),
        "model_version": getattr(state, "model_version", "v1.0"),
        "feature_version": getattr(state, "feature_version", "v1.0"),
        "policy_version": "v1.0",
        "policy_parameters_hash": policy_hash,
        "data_cutoff": data_cutoff,
        "code_version": code_version,
    }
    Path(metadata_path).write_text(json.dumps(meta, indent=2))
    logger.info(f"Metadata saved: {snapshot_id}")
    logger.info(f"Total Offline Job Time: {time.time() - start_time:.2f}s")

if __name__ == "__main__":
    run_job()
