"""Active learning simulation for LeadGuard.

Implements Architecture §7.4–7.5:
  - priority_score = λ1·p_lead + λ2·uncertainty_score + λ3·equity_boost
  - Simulation: 10% initial labels, batch=500, 10 rounds
  - Retrains model and rebuilds spatial features after each round.
  - Benchmarks: random, risk, uncertainty, risk_uncertainty, risk_uncertainty_equity, oracle.

Usage:
    python -m leadguard.models.active_learning
    python -m leadguard.models.active_learning --fast
"""

from __future__ import annotations

import logging
import json
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
import yaml
from sklearn.metrics import average_precision_score

from leadguard.data.features import build_features
from leadguard.evaluation.fairness import compute_equity_boost
from leadguard.models.uncertainty import compute_predictive_entropy
from leadguard.models.xgboost_model import XGB_FEATURES
from leadguard.utils.seed import SEED

logger = logging.getLogger(__name__)


def _load_scoring_config(config_path: Path = Path("configs/scoring.yaml")) -> dict:
    if config_path.exists():
        return yaml.safe_load(config_path.read_text())
    logger.warning("scoring.yaml not found; using hardcoded defaults")
    return {"lambda1": 0.60, "lambda2": 0.25, "lambda3": 0.15}


def compute_expected_utility(
    p_lead: np.ndarray,
    intervention_value: float | np.ndarray,
    cost: float | np.ndarray,
    equity_boost: np.ndarray,
    equity_weight: float = 1.0,
) -> np.ndarray:
    """Compute expected utility of an intervention.
    
    Formula: EU = P(Lead) * intervention_value - cost + equity_boost * equity_weight
    
    Args:
        p_lead: Probability of lead.
        intervention_value: Monetized or util-based value of intervention (V_i).
        cost: Cost of intervention (C_i).
        equity_boost: Tract-level equity weight.
        equity_weight: Scaling factor for equity.
        
    Returns:
        Array of expected utilities.
    """
    if np.isscalar(intervention_value):
        intervention_value = np.full_like(p_lead, intervention_value)
    if np.isscalar(cost):
        cost = np.full_like(p_lead, cost)
        
    return (p_lead * intervention_value) - cost + (equity_boost * equity_weight)

def compute_priority_score(*args, **kwargs):
    """Deprecated. Use compute_expected_utility instead."""
    import warnings
    warnings.warn("compute_priority_score is deprecated. Use compute_expected_utility.", DeprecationWarning)
    # Temporary fallback for any old callers
    p_lead = args[0] if args else kwargs.get('p_lead')
    return p_lead


def compute_approximate_evi(
    p_lead: np.ndarray,
    intervention_value: float | np.ndarray,
    intervention_cost: float | np.ndarray,
    equity_boost: np.ndarray,
    equity_weight: float = 1.0,
) -> np.ndarray:
    """Compute Approximate Expected Value of Information (EVI).
    
    Gross EVI is the expected improvement in decision utility from learning the true label.
    
    Current policy: Replace if EU_replace > 0.
    Current utility = max(0, EU_replace(p_lead))
    
    If we inspect, we learn Y (Lead=1 or NotLead=0).
    Expected future utility = p_lead * max(0, EU_replace(1)) + (1 - p_lead) * max(0, EU_replace(0))
    
    Approximate EVI = Expected future utility - Current utility
    
    Args:
        p_lead: Prior probability of lead.
        intervention_value: Value of intervention (V_i).
        intervention_cost: Cost of intervention (C_i).
        equity_boost: Tract-level equity weight.
        equity_weight: Scaling factor for equity.
        
    Returns:
        Array of Gross EVI (in utility units).
    """
    eu_current = compute_expected_utility(p_lead, intervention_value, intervention_cost, equity_boost, equity_weight)
    u_current = np.maximum(0, eu_current)
    
    eu_if_lead = compute_expected_utility(np.ones_like(p_lead), intervention_value, intervention_cost, equity_boost, equity_weight)
    u_if_lead = np.maximum(0, eu_if_lead)
    
    eu_if_not_lead = compute_expected_utility(np.zeros_like(p_lead), intervention_value, intervention_cost, equity_boost, equity_weight)
    u_if_not_lead = np.maximum(0, eu_if_not_lead)
    
    expected_future_u = (p_lead * u_if_lead) + ((1 - p_lead) * u_if_not_lead)
    
    evi = expected_future_u - u_current
    
    # Due to floating point math, ensure EVI >= 0
    return np.maximum(0, evi)

def simulate_active_learning(
    features_path: Path | str = "data/processed/features.parquet",
    fairness_ref_path: Path | str = "data/fairness_reference.parquet",
    output_path: Path | str = "reports/active_learning_curve.csv",
    config_path: Path | str = "configs/scoring.yaml",
    fast: bool = False,
) -> pd.DataFrame:
    features_path = Path(features_path)
    cfg = _load_scoring_config(Path(config_path))
    lambda1 = cfg.get("lambda1", 0.60)
    lambda2 = cfg.get("lambda2", 0.25)
    lambda3 = cfg.get("lambda3", 0.15)
    al_cfg = cfg.get("active_learning", {})
    initial_frac = al_cfg.get("initial_labeled_fraction", 0.10)
    batch_size = al_cfg.get("batch_size", 500)
    n_rounds = al_cfg.get("n_rounds", 10)
    cost_per_inspection = cfg.get("budget", {}).get("default_inspection_cost_usd", 500)

    df = pd.read_parquet(features_path)
    labeled_mask = df["service_line_material"].isin(["Lead", "Copper", "Galvanized"])
    labeled = df[labeled_mask].copy()

    if len(labeled) == 0:
        logger.error("No labeled data found")
        return pd.DataFrame()

    fairness_ref = pd.DataFrame()
    if Path(fairness_ref_path).exists():
        fairness_ref = pd.read_parquet(fairness_ref_path)

    rng = np.random.default_rng(SEED)
    all_labeled_idx = labeled.index.tolist()
    rng.shuffle(all_labeled_idx)
    n_init = max(1, int(len(all_labeled_idx) * initial_frac))

    strategies = [
        "random",
        "risk",
        "uncertainty",
        "risk_uncertainty",
        "risk_uncertainty_equity",
        "oracle",
    ]
    results = []

    for strategy in strategies:
        logger.info("=== Strategy: %s ===", strategy)
        currently_labeled_idx = set(all_labeled_idx[:n_init])
        cumulative_cost = 0.0
        cumulative_discoveries = 0

        for round_num in range(0, n_rounds + 1):  # 0 is initial eval, 1-N are acquisition
            # Define L_i (reference df)
            L_df = labeled.loc[list(currently_labeled_idx)]

            # Rebuild spatial features for L_df
            L_df_feats = build_features(L_df, reference_df=L_df, include_label_dependent=True)
            from leadguard.data.validation import validate_features
            X_train = validate_features(L_df_feats, XGB_FEATURES).values
            y_train = (L_df_feats["service_line_material"] == "Lead").astype(int).values

            # Retrain model
            n_estimators = 50 if fast else 300
            model = xgb.XGBClassifier(
                n_estimators=n_estimators, max_depth=6, verbosity=0, random_state=SEED
            )
            # Add simple class weights
            n_neg = (y_train == 0).sum()
            n_pos = (y_train == 1).sum()
            scale_pos_weight = float(n_neg / n_pos) if n_pos > 0 else 1.0
            model.set_params(scale_pos_weight=scale_pos_weight)
            model.fit(X_train, y_train)

            # Evaluate on remaining hidden
            test_idx = [i for i in all_labeled_idx if i not in currently_labeled_idx]
            if not test_idx:
                break

            test_df = labeled.loc[test_idx]
            test_df_feats = build_features(test_df, reference_df=L_df, include_label_dependent=True)
            X_test = (
                validate_features(test_df_feats, XGB_FEATURES).values
            )
            y_test = (test_df_feats["service_line_material"] == "Lead").astype(int).values

            proba_test = model.predict_proba(X_test)[:, 1]
            pr_auc = (
                float(average_precision_score(y_test, proba_test))
                if len(np.unique(y_test)) > 1
                else float("nan")
            )

            results.append(
                {
                    "round": round_num,
                    "strategy": strategy,
                    "cumulative_inspections": len(currently_labeled_idx),
                    "cumulative_cost_usd": cumulative_cost,
                    "cumulative_discoveries": cumulative_discoveries,
                    "discoveries_per_inspection": cumulative_discoveries
                    / max(1, len(currently_labeled_idx) - n_init),
                    "discoveries_per_usd": cumulative_discoveries / max(1.0, cumulative_cost),
                    "pr_auc_remaining": pr_auc,
                }
            )

            if round_num == n_rounds:
                break

            # ACQUISITION
            if strategy == "oracle":
                # Oracle acquires true leads first
                lead_indices = np.where(y_test == 1)[0]
                if len(lead_indices) > 0:
                    acq_indices = lead_indices[:batch_size]
                else:
                    acq_indices = np.arange(min(batch_size, len(y_test)))
            else:
                pred_entropy = compute_predictive_entropy(proba_test)

                equity_boost = np.zeros(len(test_idx))
                if not fairness_ref.empty and "census_tract" in test_df.columns:
                    test_with_tract = test_df.merge(fairness_ref, on="census_tract", how="left")
                    test_with_tract["p_lead_calibrated"] = proba_test
                    equity_boost = compute_equity_boost(test_with_tract, labeled)

                scores = compute_priority_score(
                    p_lead=proba_test,
                    uncertainty_score=pred_entropy,
                    equity_boost=equity_boost,
                    strategy=strategy,
                    lambda1=lambda1,
                    lambda2=lambda2,
                    lambda3=lambda3,
                )
                acq_indices = np.argsort(scores)[::-1][:batch_size]

            # Update L_i
            new_idx = [test_idx[i] for i in acq_indices]
            currently_labeled_idx.update(new_idx)
            cumulative_cost += len(new_idx) * cost_per_inspection
            cumulative_discoveries += sum(y_test[acq_indices])

    res_df = pd.DataFrame(results)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    res_df.to_csv(output_path, index=False)
    
    # Write JSONL decision log
    log_path = Path(output_path).with_suffix(".jsonl")
    with open(log_path, "w") as f:
        for r in results:
            f.write(json.dumps({
                "round": r["round"],
                "strategy": r["strategy"],
                "model_version": "xgboost_v1",
                "feature_version": "intrinsic_geo_v1",
                "policy_version": "scoring_v1",
                "outcome_available_at": pd.Timestamp.now().isoformat(),
                "cumulative_inspections": r["cumulative_inspections"],
                "pr_auc_remaining": r["pr_auc_remaining"]
            }) + "\n")
            
    logger.info("Active learning simulation written to %s and %s", output_path, log_path)
    return res_df


def main():
    import argparse

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", default="data/processed/features.parquet")
    parser.add_argument("--fairness-ref", default="data/interim/fairness_reference.parquet")
    parser.add_argument(
        "--fast", action="store_true", help="Use faster XGBoost with fewer estimators for testing"
    )
    parser.add_argument("--sample", action="store_true", help="Use sample dataset")
    args = parser.parse_args()

    features_path = "data/processed/features_sample.parquet" if args.sample else args.features
    simulate_active_learning(
        features_path=features_path, fairness_ref_path=args.fairness_ref, fast=args.fast
    )


if __name__ == "__main__":
    main()
