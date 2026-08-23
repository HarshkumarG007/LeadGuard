"""LeadGuard FastAPI application.

Implements all 7 endpoints from Architecture §8:
  GET  /v1/health
  POST /v1/predict
  GET  /v1/properties/{property_id}/prediction
  GET  /v1/priority-queue
  POST /v1/inspections
  GET  /v1/fairness-report
  GET  /v1/model/metadata

Error convention: all errors return {\"error\": ..., \"detail\": ...} with
the appropriate HTTP status. No bare stack traces are returned to callers.
"""

from __future__ import annotations

import hashlib
import json
import logging
from datetime import UTC, datetime
from pathlib import Path
from typing import Annotated

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import JSONResponse

from api.model_loader import ModelState, get_state
from api.schemas import (
    FairnessReportResponse,
    HealthResponse,
    InspectionSubmitRequest,
    InspectionSubmitResponse,
    ModelMetadataResponse,
    PredictionResult,
    PredictRequest,
    PredictResponse,
    PriorityQueueItem,
    PriorityQueueResponse,
    SHAPFeature,
    DecisionIssueRequest,
    DecisionIssueResponse,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# App initialization
# ---------------------------------------------------------------------------

app = FastAPI(
    title="LeadGuard API",
    description="Lead service line prediction, uncertainty quantification, and inspection prioritization",
    version="0.1.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# ---------------------------------------------------------------------------
# Demo In-Memory Store
# 
# WARNING: This is a non-persistent demo store designed for the Hugging Face 
# free tier environment. Submitted inspections will NOT persist across restarts.
# In a production environment, this should be replaced with a persistent 
# database (e.g. PostgreSQL or SQLite) integrated with the active learning pipeline.
# ---------------------------------------------------------------------------
ACTIVE_LEARNING_LOG_PATH = Path("data/processed/active_learning_log.jsonl")

def _read_inspections() -> list[dict]:
    if not ACTIVE_LEARNING_LOG_PATH.exists():
        return []
    try:
        with open(ACTIVE_LEARNING_LOG_PATH, "r") as f:
            return [json.loads(line) for line in f if line.strip()]
    except Exception as e:
        logger.error(f"Failed to read active learning log: {e}")
        return []

def _append_inspection(inspection: dict):
    ACTIVE_LEARNING_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(ACTIVE_LEARNING_LOG_PATH, "a") as f:
        f.write(json.dumps(inspection) + "\n")

# Properties store (loaded lazily from sample/processed data)
_properties_cache: pd.DataFrame | None = None


def _get_properties() -> pd.DataFrame:
    """Load precomputed decision snapshot from disk, cached in memory.

    Raises:
        HTTPException: 503 if snapshot is missing or stale.
    """
    global _properties_cache
    if _properties_cache is not None:
        return _properties_cache

    snapshot_path = Path("data/processed/decision_snapshot.parquet")
    meta_path = Path("data/processed/decision_snapshot_meta.json")
    
    if not snapshot_path.exists() or not meta_path.exists():
        raise HTTPException(
            status_code=503, 
            detail={"error": "Decision snapshot unavailable. Run offline policy job."}
        )
        
    meta = json.loads(meta_path.read_text())
    valid_until_str = meta.get("valid_until")
    if valid_until_str:
        valid_until = datetime.fromisoformat(valid_until_str)
        if datetime.now(UTC) > valid_until:
            raise HTTPException(
                status_code=503,
                detail={"error": "Decision snapshot is stale", "generated_at": meta.get("generated_at"), "valid_until": valid_until_str}
            )

    _properties_cache = pd.read_parquet(snapshot_path)
    logger.info("Decision snapshot loaded (%d rows)", len(_properties_cache))
    return _properties_cache


def _model_version(state: ModelState) -> str:
    """Build a model version string.

    Args:
        state: Loaded model state.

    Returns:
        Version string.
    """
    return state.metrics.get("model_version", f"xgb-{datetime.now(UTC).strftime('%Y.%m.%d')}")


def _predict_single(
    property_row: pd.Series,
    state: ModelState,
    features_df: pd.DataFrame | None = None,
) -> dict:
    """Generate a full prediction for one property.

    Args:
        property_row: A single property Series with all feature columns.
        state: Loaded model state.
        features_df: Full features DataFrame (for equity_boost context).

    Returns:
        Prediction dictionary matching PredictionResult schema.
    """
    # O(1) Online Path Lookups ONLY. No dynamic feature evaluation.
    if "p_lead_calibrated" not in property_row or "uncertainty_score" not in property_row:
        # This shouldn't happen if they came from the snapshot, but just in case
        raise HTTPException(status_code=503, detail={"error": "Snapshot missing required fields", "detail": "Run offline job."})
        
    p_lead = float(property_row["p_lead_calibrated"])
    uncertainty = float(property_row["uncertainty_score"])
    
    # Priority score is EVI if available, else fallback
    priority = float(property_row.get("evi", 0.0))

    # Conformal set
    proba = np.array([1 - p_lead, p_lead])
    if state.conformal_global is not None:
        conformal_sets = state.conformal_global.predict_set(np.array([proba]))
        conformal_set = list(conformal_sets[0])
    else:
        conformal_set = ["Non-Lead", "Lead"]

    # SHAP top features (loaded dynamically from explanation snapshot)
    top_features_list = []
    try:
        import pyarrow.parquet as pq
        expl_path = Path("data/processed/explanation_snapshot.parquet")
        if expl_path.exists():
            table = pq.read_table(expl_path, filters=[("property_id", "==", str(property_row.get("property_id")))])
            if table.num_rows > 0:
                json_str = table.column("shap_features_json")[0].as_py()
                top_features_list = json.loads(json_str)
    except Exception as e:
        logger.error(f"Failed to fetch explanation: {e}")

    return {
        "property_id": str(property_row.get("property_id", "unknown")),
        "p_lead_calibrated": round(p_lead, 4),
        "conformal_set": conformal_set,
        "confidence_level": 0.90,
        "uncertainty_score": round(uncertainty, 4),
        "priority_score": round(float(priority), 4),
        "shap_top_features": [SHAPFeature(**f) for f in top_features_list],
        "model_version": _model_version(state),
        "predicted_at": datetime.now(UTC),
    }


def _error_503(detail: str) -> JSONResponse:
    """Return a 503 response with structured error body.

    Args:
        detail: Human-readable error detail.

    Returns:
        JSONResponse with 503 status.
    """
    return JSONResponse(
        status_code=503,
        content={"error": "Service unavailable", "detail": detail},
    )


def _error_404(detail: str) -> HTTPException:
    """Raise a 404 HTTPException.

    Args:
        detail: Error detail message.

    Returns:
        HTTPException with 404 status.
    """
    return HTTPException(status_code=404, detail={"error": "Not found", "detail": detail})


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@app.get("/v1/health", response_model=HealthResponse, tags=["system"])
async def health() -> HealthResponse:
    """Liveness and readiness check.

    Returns 503 if the model artifacts failed to load.
    """
    state = get_state()
    if not state.is_ready:
        return JSONResponse(  # type: ignore[return-value]
            status_code=503,
            content={
                "status": "degraded",
                "model_loaded": False,
                "conformal_loaded": state.conformal_ready,
                "fairness_ref_loaded": state.fairness_ready,
                "errors": state.load_errors,
            },
        )
    return HealthResponse(
        status="ok",
        model_loaded=state.is_ready,
        conformal_loaded=state.conformal_ready,
        fairness_ref_loaded=state.fairness_ready,
    )


@app.post("/v1/predict", response_model=PredictResponse, tags=["prediction"])
async def batch_predict(request: PredictRequest) -> PredictResponse:
    """Batch predict for a list of property IDs.

    Returns calibrated probability, conformal set, uncertainty score,
    priority score, and top SHAP features for each property.
    """
    state = get_state()
    if not state.is_ready:
        return _error_503("Model not loaded")  # type: ignore[return-value]

    props = _get_properties()
    if props.empty:
        return _error_503("Property data not available")  # type: ignore[return-value]

    results = []
    for pid in request.property_ids:
        row_mask = props["property_id"].astype(str) == str(pid)
        if not row_mask.any():
            # Unknown property — return 404 per Architecture §8 error convention
            raise _error_404(f"property_id '{pid}' not found")
        row = props[row_mask].iloc[0]
        pred = _predict_single(row, state, features_df=props)
        results.append(PredictionResult(**pred))

    return PredictResponse(predictions=results)


@app.get(
    "/v1/properties/{property_id}/prediction",
    response_model=PredictionResult,
    tags=["prediction"],
)
async def single_prediction(property_id: str) -> PredictionResult:
    """Single prediction with full SHAP explanation."""
    state = get_state()
    if not state.is_ready:
        return _error_503("Model not loaded")  # type: ignore[return-value]

    props = _get_properties()
    row_mask = props["property_id"] == property_id
    if not row_mask.any():
        raise _error_404(f"property_id '{property_id}' not found")

    row = props[row_mask].iloc[0]
    pred = _predict_single(row, state, features_df=props)
    return PredictionResult(**pred)


@app.get("/v1/priority-queue", response_model=PriorityQueueResponse, tags=["queue"])
async def priority_queue(
    scenario: str = "base",
    budget_usd: Annotated[float, Query(ge=0)] = 100000.0,
    limit: Annotated[int, Query(ge=1, le=10000)] = 500,
) -> PriorityQueueResponse:
    """Ranked inspection queue using EVI and budget constraints."""
    import yaml
    from leadguard.models.active_learning import compute_approximate_evi
    
    state = get_state()
    if not state.is_ready:
        return _error_503("Model not loaded")

    props = _get_properties()
    if props.empty or "p_lead_calibrated" not in props.columns:
        return _error_503("Precomputed property data not available")

    # 1. Load policy scenario
    cfg_path = Path("configs/scoring.yaml")
    if not cfg_path.exists():
        return _error_503("scoring.yaml not found")
        
    cfg = yaml.safe_load(cfg_path.read_text())
    scenario_cfg = cfg.get("scenarios", {}).get(scenario, cfg.get("scenarios", {}).get("base", {}))
    
    inspection_cost = scenario_cfg.get("inspection", {}).get("cost_usd", 500.0)
    intervention_value = scenario_cfg.get("intervention", {}).get("value_usd", 5000.0)
    equity_weight = scenario_cfg.get("equity", {}).get("weight", 1000.0)

    p_lead_all = props["p_lead_calibrated"].values
    uncertainty_all = props["uncertainty_score"].values
    equity_all = props.get("equity_boost", pd.Series(np.zeros(len(props)))).values
    wards_all = props.get("ward", pd.Series(np.zeros(len(props)))).values

    # 2. Read Precomputed EVI
    if "evi" not in props:
        return _error_503("Precomputed EVI not available in snapshot. Run offline job.")
    evi_all = props["evi"].values
    
    # Net EVI
    net_evi = evi_all - inspection_cost
    
    # Sort candidates
    sorted_idx = np.argsort(net_evi)[::-1]
    
    # 3. Geographic Diversification
    # Soft constraint: no more than (limit / max(1, n_wards)) * 2 properties per ward
    unique_wards = len(np.unique(wards_all))
    max_per_ward = max(1, int((limit / max(1, unique_wards)) * 2.0))
    ward_counts = {}

    items = []
    remaining_budget = budget_usd
    
    for idx in sorted_idx:
        if net_evi[idx] <= 0 or remaining_budget < inspection_cost or len(items) >= limit:
            break
            
        w = str(wards_all[idx])
        if ward_counts.get(w, 0) >= max_per_ward:
            continue
            
        ward_counts[w] = ward_counts.get(w, 0) + 1
        
        row = props.iloc[idx]
        items.append(
            PriorityQueueItem(
                rank=len(items) + 1,
                property_id=str(row.get("property_id", "")),
                priority_score=round(float(net_evi[idx]), 4),
                p_lead_calibrated=round(float(p_lead_all[idx]), 4),
                uncertainty_score=round(float(uncertainty_all[idx]), 4),
                conformal_set=list(state.conformal_global.predict_set(np.array([[[1-p_lead_all[idx], p_lead_all[idx]]]]))[0]) if state.conformal_global else [],
                estimated_cost_usd=inspection_cost,
            )
        )
        remaining_budget -= inspection_cost

    return PriorityQueueResponse(
        budget_usd=budget_usd,
        total_properties_ranked=len(sorted_idx),
        properties_within_budget=len(items),
        items=items,
    )


_decision_idempotency_cache = set()

@app.post("/v1/decisions/issue", response_model=DecisionIssueResponse, tags=["decisions"])
async def issue_decisions(
    request: DecisionIssueRequest,
    idempotency_key: Annotated[str | None, Header(alias="Idempotency-Key")] = None,
) -> DecisionIssueResponse:
    """Formally issue a policy decision and record it to the immutable ledger."""
    
    # SAFETY INVARIANT (S0.4): Absence of operational capability is default.
    # We must explicitly enable PRODUCTION_MODE to dispatch operational decisions.
    import os
    production_mode = os.environ.get("LEADGUARD_PRODUCTION_MODE", "false").lower() == "true"
    if not production_mode:
        raise HTTPException(
            status_code=403, 
            detail={
                "error": "Operational dispatch disabled", 
                "detail": "LEADGUARD_PRODUCTION_MODE is not explicitly enabled. Operational dispatch is structurally prohibited."
            }
        )

    from leadguard.data.ledger import ImmutableLedger, LedgerEntry
    ledger = ImmutableLedger()
    
    if idempotency_key:
        if idempotency_key in _decision_idempotency_cache:
            return DecisionIssueResponse(
                decision_id=f"dec-cached-{idempotency_key[:8]}",
                property_ids=request.property_ids,
                status="success",
                message="Idempotency key matched. Decision already recorded."
            )
        _decision_idempotency_cache.add(idempotency_key)

    decision_id = "dec-" + hashlib.sha256(f"{request.property_ids}{datetime.now(UTC).isoformat()}".encode()).hexdigest()[:12]
    
    payload = {
        "decision_id": decision_id,
        "property_ids": request.property_ids,
        "decision_type": request.decision_type,
        "reasoning": request.reasoning,
        "issued_at": datetime.now(UTC).isoformat(),
        "idempotency_key": idempotency_key,
    }
    
    entry = LedgerEntry(
        event_type="DecisionIssued",
        decision_id=decision_id,
        model_version=_model_version(get_state()),
        payload=payload
    )
    ledger.append_event(entry)
    
    return DecisionIssueResponse(
        decision_id=decision_id,
        property_ids=request.property_ids,
        status="success",
        message=f"Issued {request.decision_type} decision for {len(request.property_ids)} properties."
    )

@app.post("/v1/inspections", response_model=InspectionSubmitResponse, tags=["feedback"])
async def submit_inspection(request: InspectionSubmitRequest) -> InspectionSubmitResponse:
    """Submit a new ground-truth inspection result and update the Decision Ledger."""
    props = _get_properties()
    census_tract = None
    if not props.empty:
        row_mask = props["property_id"] == request.property_id
        if row_mask.any():
            census_tract = props[row_mask].iloc[0].get("census_tract")

    from leadguard.data.ledger import ImmutableLedger, LedgerEntry
    ledger = ImmutableLedger()
    
    # 1. Update legacy JSON store to ensure backwards compatibility with older components
    inspection_id = (
        "insp-"
        + hashlib.sha256(
            f"{request.property_id}{datetime.now(UTC).isoformat()}".encode()
        ).hexdigest()[:12]
    )

    inspection_payload = {
        "inspection_id": inspection_id,
        "property_id": request.property_id,
        "inspected_material": request.inspected_material,
        "census_tract": census_tract,
        "submitted_at": datetime.now(UTC).isoformat(),
    }
    
    _append_inspection(inspection_payload)
    
    # 2. Append Canonical Ledger Event
    entry = LedgerEntry(
        event_type="InspectionCompleted",
        entity_id=request.property_id,
        model_version=_model_version(get_state()),
        payload=inspection_payload
    )
    ledger.append_event(entry)
    
    logger.info("Inspection recorded: %s for property %s", inspection_id, request.property_id)

    return InspectionSubmitResponse(
        inspection_id=inspection_id,
        property_id=request.property_id,
        inspected_material=request.inspected_material,
        inspected_at=request.inspected_at or datetime.now(UTC),
        message="Inspection recorded in Immutable Ledger.",
    )


@app.get("/v1/fairness-report", response_model=FairnessReportResponse, tags=["fairness"])
async def fairness_report() -> FairnessReportResponse:
    """Latest fairness audit metrics."""
    report_path = Path("reports/fairness_report.json")
    if not report_path.exists():
        return _error_503(
            "Fairness report not yet generated. Run Phase 6 (evaluation/fairness.py) first."
        )  # type: ignore[return-value]

    report = json.loads(report_path.read_text())
    return FairnessReportResponse(
        **report,
        generated_at=datetime.fromtimestamp(report_path.stat().st_mtime, tz=UTC),
    )


@app.get("/v1/model/metadata", response_model=ModelMetadataResponse, tags=["system"])
async def model_metadata() -> ModelMetadataResponse:
    """Active model version, training date, and headline metrics."""
    state = get_state()
    metrics = state.metrics

    return ModelMetadataResponse(
        model_version=_model_version(state),
        training_date=metrics.get("training_date"),
        pr_auc_geo=metrics.get("pr_auc_geo"),
        pr_auc_random=metrics.get("pr_auc_random"),
        features=state.feature_names,
        confidence_level=0.90,
    )
