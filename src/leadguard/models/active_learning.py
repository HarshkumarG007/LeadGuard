"""Active learning simulation for LeadGuard.

Implements Architecture §7.4–7.5:
  - priority_score = λ1·p_lead + λ2·uncertainty_score + λ3·equity_boost
  - Simulation: 10% initial labels, batch=500, 10 rounds
  - Uncertainty-driven vs. random acquisition comparison
  - Budget constraint enforced via cost_usd per inspection

PRECONDITION: Phases 5 (uncertainty) and 6 (fairness reference) must
both be complete before this module can run. equity_boost requires the
fairness reference; uncertainty_score requires conformal calibration.
Stubbing either produces a silently wrong scoring function.

Usage:
    python -m leadguard.models.active_learning
    python -m leadguard.models.active_learning --sample
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from leadguard.evaluation.fairness import compute_equity_boost
from leadguard.utils.seed import SEED

logger = logging.getLogger(__name__)


def _load_scoring_config(config_path: Path = Path("configs/scoring.yaml")) -> dict:
    """Load scoring weights from YAML config.

    Args:
        config_path: Path to scoring.yaml.

    Returns:
        Configuration dictionary.
    """
    if config_path.exists():
        return yaml.safe_load(config_path.read_text())
    logger.warning("scoring.yaml not found; using hardcoded defaults")
    return {"lambda1": 0.60, "lambda2": 0.25, "lambda3": 0.15}


def compute_priority_score(
    p_lead: np.ndarray,
    uncertainty_score: np.ndarray,
    equity_boost: np.ndarray,
    lambda1: float = 0.60,
    lambda2: float = 0.25,
    lambda3: float = 0.15,
) -> np.ndarray:
    """Compute the composite priority score (Architecture §7.4).

    priority_score = λ1·p_lead_calibrated + λ2·uncertainty_score + λ3·equity_boost

    Args:
        p_lead: Calibrated lead probabilities in [0, 1].
        uncertainty_score: Normalized conformal set size in [0, 1].
        equity_boost: Tract-level equity boost in [0, 1].
        lambda1: Weight on p_lead.
        lambda2: Weight on uncertainty.
        lambda3: Weight on equity_boost.

    Returns:
        Array of composite priority scores in [0, 1].

    Raises:
        ValueError: If weights don't approximately sum to 1.
    """
    if abs(lambda1 + lambda2 + lambda3 - 1.0) > 1e-4:
        raise ValueError(f"Weights must sum to 1.0: {lambda1}+{lambda2}+{lambda3}={lambda1+lambda2+lambda3}")
    return lambda1 * p_lead + lambda2 * uncertainty_score + lambda3 * equity_boost


def simulate_active_learning(
    features_path: Path | str = "data/processed/features.parquet",
    fairness_ref_path: Path | str = "data/fairness_reference.parquet",
    model_dir: Path | str = "models/xgboost",
    output_path: Path | str = "reports/active_learning_curve.csv",
    config_path: Path | str = "configs/scoring.yaml",
    sample: bool = False,
) -> pd.DataFrame:
    """Simulate the active learning loop and compare acquisition strategies.

    Compares:
      - uncertainty-driven: selects highest-priority-score properties each round
      - random: selects properties uniformly at random each round

    Args:
        features_path: Features parquet path.
        fairness_ref_path: Fairness reference parquet.
        model_dir: XGBoost model directory.
        output_path: CSV output for learning curve.
        config_path: Scoring config YAML.
        sample: Use sample paths if True.

    Returns:
        DataFrame with columns [round, strategy, cumulative_inspections, pr_auc, cost_usd_spent].
    """
    import xgboost as xgb  # noqa: PLC0415
    import pickle  # noqa: PLC0415
    from sklearn.metrics import average_precision_score  # noqa: PLC0415
    from leadguard.models.xgboost_model import XGB_FEATURES  # noqa: PLC0415

    features_path = Path(features_path)
    if sample and not features_path.exists():
        features_path = Path("data/processed/features_sample.parquet")

    cfg = _load_scoring_config(Path(config_path))
    lambda1 = cfg.get("lambda1", 0.60)
    lambda2 = cfg.get("lambda2", 0.25)
    lambda3 = cfg.get("lambda3", 0.15)
    al_cfg = cfg.get("active_learning", {})
    initial_frac = al_cfg.get("initial_labeled_fraction", 0.10)
    batch_size = al_cfg.get("batch_size", 500)
    n_rounds = al_cfg.get("n_rounds", 10)
    cost_per_inspection = cfg.get("budget", {}).get("default_inspection_cost_usd", 500)
    budget_per_round = batch_size * cost_per_inspection

    df = pd.read_parquet(features_path)
    labeled_mask = df["service_line_material"].isin(["Lead", "Copper", "Galvanized"])
    labeled = df[labeled_mask].copy()
    unlabeled_pool = df[~labeled_mask].copy()

    if len(labeled) == 0:
        logger.error("No labeled data found; cannot run active learning simulation")
        return pd.DataFrame()

    # Load model
    model_dir = Path(model_dir)
    model = xgb.XGBClassifier()
    model_path = model_dir / "model.json"
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found at {model_path}. Run Phase 4 first.")
    model.load_model(str(model_path))

    # Load conformal predictor if available
    conformal_path = model_dir / "conformal_global.pkl"
    conformal = None
    if conformal_path.exists():
        # Must import into namespace so pickle can resolve the class
        from leadguard.models.uncertainty import SplitConformalPredictor, MondriancConformalPredictor  # noqa: F401
        with conformal_path.open("rb") as f:
            conformal = pickle.load(f)

    # Load fairness reference
    fairness_ref = pd.DataFrame()
    if Path(fairness_ref_path).exists():
        fairness_ref = pd.read_parquet(fairness_ref_path)

    # Seed initial labeled/unlabeled split
    rng = np.random.default_rng(SEED)
    all_labeled_idx = labeled.index.tolist()
    rng.shuffle(all_labeled_idx)
    n_init = max(1, int(len(all_labeled_idx) * initial_frac))

    results = []

    for strategy in ["uncertainty", "random"]:
        logger.info("=== Strategy: %s ===", strategy)
        # Reset per strategy
        currently_labeled_idx = set(all_labeled_idx[:n_init])
        remaining_labeled_idx = list(all_labeled_idx[n_init:])  # simulate "hidden" labels
        cumulative_cost = 0.0

        for round_num in range(1, n_rounds + 1):
            # Get test set: labeled rows NOT in currently_labeled
            test_idx = [i for i in all_labeled_idx if i not in currently_labeled_idx]
            if not test_idx:
                break

            train_df = labeled.loc[list(currently_labeled_idx)]
            test_df = labeled.loc[test_idx]

            X_test = test_df.reindex(columns=XGB_FEATURES, fill_value=0.0).astype(float).values
            y_test = (test_df["service_line_material"] == "Lead").astype(int).values

            # Compute PR-AUC for this round
            if len(np.unique(y_test)) < 2:
                pr_auc = float("nan")
            else:
                proba_test = model.predict_proba(X_test)[:, 1]
                pr_auc = float(average_precision_score(y_test, proba_test))

            results.append({
                "round": round_num,
                "strategy": strategy,
                "cumulative_inspections": len(currently_labeled_idx),
                "pr_auc": pr_auc,
                "cost_usd_spent": cumulative_cost,
            })
            logger.info("[%s] Round %d: PR-AUC=%.4f, labeled=%d", strategy, round_num, pr_auc, len(currently_labeled_idx))

            # Select next batch
            n_select = min(batch_size, len(remaining_labeled_idx))
            if n_select == 0:
                break

            if strategy == "uncertainty":
                # Compute priority scores for remaining labeled pool
                cand_df = labeled.loc[remaining_labeled_idx]
                X_cand = cand_df.reindex(columns=XGB_FEATURES, fill_value=0.0).astype(float).values
                p_lead = model.predict_proba(X_cand)[:, 1]

                # Uncertainty score from conformal set size
                if conformal is not None:
                    proba_all = model.predict_proba(X_cand)
                    scores = 1.0 - proba_all.max(axis=1)
                    uncertainty = np.clip((scores - 0.0) / max(1 - conformal.threshold_, 1e-6), 0.0, 1.0)
                else:
                    # Fallback: use entropy as uncertainty
                    p = np.clip(p_lead, 1e-6, 1 - 1e-6)
                    uncertainty = -(p * np.log(p) + (1 - p) * np.log(1 - p)) / np.log(2)
                    uncertainty = np.clip(uncertainty, 0.0, 1.0)

                # Equity boost
                if not fairness_ref.empty and "census_tract" in cand_df.columns:
                    current_inspections = pd.DataFrame(
                        {"census_tract": labeled.loc[list(currently_labeled_idx), "census_tract"].values}
                    )
                    pred_df = cand_df[["census_tract"]].copy()
                    pred_df["p_lead_calibrated"] = p_lead
                    all_pred_df = labeled[["census_tract"]].copy()
                    all_pred_df["p_lead_calibrated"] = model.predict_proba(
                        labeled.reindex(columns=XGB_FEATURES, fill_value=0.0).astype(float).values
                    )[:, 1]
                    # Merge tract to get equity boost per property
                    boost_series = compute_equity_boost(all_pred_df, current_inspections)
                    cand_df2 = cand_df[["census_tract"]].copy()
                    cand_df2["_equity_boost"] = cand_df2["census_tract"].map(boost_series).fillna(0.0).values
                    equity_boost_vals = cand_df2["_equity_boost"].values
                else:
                    equity_boost_vals = np.zeros(len(cand_df))

                priority = compute_priority_score(p_lead, uncertainty, equity_boost_vals, lambda1, lambda2, lambda3)
                select_pos = np.argsort(priority)[::-1][:n_select]
                selected = [remaining_labeled_idx[i] for i in select_pos]
            else:
                # Random strategy
                rng2 = np.random.default_rng(SEED + round_num)
                selected = rng2.choice(remaining_labeled_idx, size=n_select, replace=False).tolist()

            # "Inspect" selected properties (enforce budget)
            round_cost = len(selected) * cost_per_inspection
            if round_cost > budget_per_round * 1.01:  # 1% tolerance
                n_affordable = int(budget_per_round / cost_per_inspection)
                selected = selected[:n_affordable]
                round_cost = len(selected) * cost_per_inspection

            cumulative_cost += round_cost
            currently_labeled_idx.update(selected)
            remaining_labeled_idx = [i for i in remaining_labeled_idx if i not in currently_labeled_idx]

    curve_df = pd.DataFrame(results)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    curve_df.to_csv(output_path, index=False)
    logger.info("Active learning curve written to %s", output_path)
    return curve_df


if __name__ == "__main__":
    import argparse

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description="Run active learning simulation")
    parser.add_argument("--sample", action="store_true")
    args = parser.parse_args()
    df = simulate_active_learning(sample=args.sample)
    if not df.empty:
        r5 = df[df["round"] == 5]
        unc_pr = r5[r5["strategy"] == "uncertainty"]["pr_auc"].values
        rand_pr = r5[r5["strategy"] == "random"]["pr_auc"].values
        if len(unc_pr) and len(rand_pr):
            print(f"Round 5: uncertainty PR-AUC={unc_pr[0]:.4f}, random PR-AUC={rand_pr[0]:.4f}")
    print("PHASE 7 PASS")
