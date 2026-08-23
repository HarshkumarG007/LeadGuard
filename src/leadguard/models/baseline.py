"""Baseline model training for LeadGuard.

Implements three baselines from Architecture §7.2:
  - Baseline 0: Year-built heuristic (year_built < 1950 → Lead)
  - Baseline 1: Logistic regression (L2, C=1.0)
  - Baseline 2: Random forest (300 estimators)

All baselines are scored on both random and geographic splits.
Results are written to reports/baseline_metrics.json.

Usage:
    python -m leadguard.models.baseline --config configs/train.yaml
    python -m leadguard.models.baseline --config configs/train.yaml --sample
"""

from __future__ import annotations

import json
import logging
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from leadguard.data.features import build_features
from leadguard.data.split import split_dataset
from leadguard.evaluation.metrics import (
    compute_metrics,
    write_metrics,
)
from leadguard.utils.seed import SEED

logger = logging.getLogger(__name__)

# Feature columns used by baseline models (subset — no leakage-sensitive spatial lags)
BASELINE_FEATURES = [
    "year_built",
    "lot_size_sqft",
    "building_sqft",
    "stories",
    "has_basement",
    "dist_to_nearest_hydrant_m",
    "dist_to_nearest_known_lead_m",
]

# Full feature set including spatial lags (used for RF and logistic regression)
FULL_FEATURES = [
    "year_built",
    "lot_size_sqft",
    "building_sqft",
    "stories",
    "has_basement",
    "dist_to_nearest_hydrant_m",
    "dist_to_nearest_known_lead_m",
    "neighbor_lead_rate_h3res8",
    "knn10_lead_rate",
    "known_lead_rate_in_ward",
]


def _prep_xy(
    df: pd.DataFrame,
    features: list[str],
    target_col: str = "service_line_material",
    positive_class: str = "Lead",
) -> tuple[np.ndarray, np.ndarray]:
    """Extract feature matrix and binary target from DataFrame.

    Args:
        df: Labeled DataFrame.
        features: List of feature column names.
        target_col: Name of the target column.
        positive_class: Class to treat as positive.

    Returns:
        Tuple of (X, y) where y is binary (1 = Lead).
    """
    from leadguard.data.validation import validate_features
    from leadguard.models.serving import encode_target

    X = validate_features(df, features).values
    y = encode_target(df[target_col]).values
    return X, y


class YearBuiltHeuristic:
    """Baseline 0: year_built < 1950 → predict Lead.

    Architecture §7.2: Baseline 0.
    """

    threshold: int = 1950

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return binary probabilities based on year_built threshold.

        Args:
            X: Feature matrix; first column must be year_built.

        Returns:
            Array of shape (n, 2) with [P(not lead), P(lead)].
        """
        year_built = X[:, 0]
        p_lead = (year_built < self.threshold).astype(float)
        return np.column_stack([1 - p_lead, p_lead])


class PrevalenceBaseline:
    """M0 Baseline: Predicts the training set prevalence (intercept only)."""

    def __init__(self):
        self.prevalence = 0.5

    def fit(self, X: np.ndarray, y: np.ndarray):
        self.prevalence = float(np.mean(y))

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        n = X.shape[0]
        p_lead = np.full(n, self.prevalence)
        return np.column_stack([1 - p_lead, p_lead])


def train_baselines(
    features_path: Path | str = "data/processed/features.parquet",
    output_dir: Path | str = "models/baseline",
    reports_dir: Path | str = "reports",
    config_path: Path | str = "configs/train.yaml",
    sample: bool = False,
    split_mode: str = "geographic",
    cutoff_date: str | None = None,
    test_start_date: str | None = None,
    test_end_date: str | None = None,
    holdout_wards: list[int] | None = None,
    min_test_rows: int | None = 100,
) -> dict:
    """Train all three baselines and write results.

    Args:
        features_path: Path to features.parquet.
        output_dir: Where to save model artifacts.
        reports_dir: Where to write baseline_metrics.json.
        config_path: Training config YAML.
        sample: If True, use sample features path.
        split_mode: 'geographic', 'temporal', or 'spatial-temporal'
        cutoff_date: Cutoff date for temporal splits.
        holdout_wards: Holdout wards for spatial splits.

    Returns:
        Dictionary of all baseline metrics.
    """
    features_path = Path(features_path)
    output_dir = Path(output_dir)
    reports_dir = Path(reports_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    # Load config
    cfg = yaml.safe_load(Path(config_path).read_text()) if Path(config_path).exists() else {}

    # Load features
    if sample and not features_path.exists():
        sample_path = Path("data/processed/features_sample.parquet")
        if sample_path.exists():
            features_path = sample_path

    df = pd.read_parquet(features_path)
    labeled = df[df["service_line_material"].isin(["Lead", "Copper", "Galvanized"])].copy()
    logger.info("Loaded %d labeled rows from %s", len(labeled), features_path)

    # Centralized Split
    split_res = split_dataset(
        labeled,
        mode=split_mode,
        cutoff_date=cutoff_date,
        test_start_date=test_start_date,
        test_end_date=test_end_date,
        holdout_wards=holdout_wards,
        seed=SEED,
        min_test_rows=min_test_rows,
    )

    train_df = split_res.train
    test_df = split_res.test

    # Feature generation with proper as_of_date
    as_of_date = split_res.metadata.get("cutoff_date")

    train_f = build_features(train_df, reference_df=train_df, include_label_dependent=True, as_of_date=as_of_date)
    test_f = build_features(test_df, reference_df=train_df, include_label_dependent=True, as_of_date=as_of_date)

    all_metrics: dict = {"split_metadata": split_res.metadata}

    # -----------------------------------------------------------------------
    # Baseline 0 — Year-built heuristic
    # -----------------------------------------------------------------------
    assert set(test_df["property_id"]) == set(test_f["property_id"]), "Model IDs must match heuristic IDs perfectly"
    
    heuristic = YearBuiltHeuristic()
    X_test_full, y_test_full = _prep_xy(test_f, ["year_built"])

    m_eval = compute_metrics(
        y_test_full, heuristic.predict_proba(X_test_full)[:, 1], prefix="test_eval_"
    )
    all_metrics["heuristic"] = {
        "pr_auc": m_eval["test_eval_pr_auc"],
        **m_eval,
    }

    # -----------------------------------------------------------------------
    # Baseline M0 — Prevalence intercept
    # -----------------------------------------------------------------------
    prevalence_model = PrevalenceBaseline()
    
    X_train_full, y_train_full = _prep_xy(train_f, FULL_FEATURES)
    prevalence_model.fit(X_train_full, y_train_full)

    m_eval_prev = compute_metrics(
        y_test_full, prevalence_model.predict_proba(X_test_full)[:, 1], prefix="test_eval_"
    )
    all_metrics["prevalence"] = {
        "pr_auc": m_eval_prev["test_eval_pr_auc"],
        **m_eval_prev,
    }

    # -----------------------------------------------------------------------
    # Baseline 1 — Logistic regression
    # -----------------------------------------------------------------------
    scaler = StandardScaler()
    X_test_f_full, y_test_f_full = _prep_xy(test_f, FULL_FEATURES)

    lr_cfg = cfg.get("baseline", {}).get("logistic_regression", {})
    lr = LogisticRegression(
        C=lr_cfg.get("C", 1.0),
        penalty=lr_cfg.get("penalty", "l2"),
        max_iter=lr_cfg.get("max_iter", 1000),
        solver=lr_cfg.get("solver", "lbfgs"),
        random_state=SEED,
        class_weight="balanced",
    )
    X_train_scaled = scaler.fit_transform(X_train_full)
    lr.fit(X_train_scaled, y_train_full)

    m_eval_lr = compute_metrics(
        y_test_f_full,
        lr.predict_proba(scaler.transform(X_test_f_full))[:, 1],
        prefix="test_eval_",
    )
    all_metrics["logistic_regression"] = {
        "pr_auc": m_eval_lr["test_eval_pr_auc"],
        **m_eval_lr,
    }

    # Save LR artifact
    with (output_dir / "logistic_regression.pkl").open("wb") as f:
        pickle.dump({"model": lr, "scaler": scaler, "features": FULL_FEATURES}, f)

    # -----------------------------------------------------------------------
    # Baseline 2 — Random forest
    # -----------------------------------------------------------------------
    rf_cfg = cfg.get("baseline", {}).get("random_forest", {})
    rf = RandomForestClassifier(
        n_estimators=rf_cfg.get("n_estimators", 300),
        max_depth=rf_cfg.get("max_depth", 10),
        min_samples_leaf=rf_cfg.get("min_samples_leaf", 5),
        random_state=SEED,
        class_weight="balanced",
        n_jobs=-1,
    )
    rf.fit(X_train_full, y_train_full)
    m_eval_rf = compute_metrics(
        y_test_f_full, rf.predict_proba(X_test_f_full)[:, 1], prefix="test_eval_"
    )
    all_metrics["random_forest"] = {
        "pr_auc": m_eval_rf["test_eval_pr_auc"],
        **m_eval_rf,
    }

    # Save RF artifact
    with (output_dir / "random_forest.pkl").open("wb") as f:
        pickle.dump({"model": rf, "features": FULL_FEATURES}, f)

    # Write metrics
    write_metrics(all_metrics, reports_dir / "baseline_metrics.json")
    logger.info("Phase 3 PASS — RF PR-AUC: %.4f", m_eval_rf["test_eval_pr_auc"])
    return all_metrics


def main():
    import argparse

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description="Train LeadGuard baseline models")
    parser.add_argument("--output-dir", default="models/baseline")
    parser.add_argument("--reports-dir", default="reports")
    parser.add_argument("--config", default="configs/train.yaml")
    parser.add_argument("--features", default="data/processed/features.parquet")
    parser.add_argument("--sample", action="store_true", help="Use sample features")
    parser.add_argument("--min-test-rows", type=int, default=100)
    parser.add_argument("--split-mode", default="geographic", help="geographic, temporal, or spatial-temporal")
    parser.add_argument("--cutoff-date", default=None, help="YYYY-MM-DD for temporal splits")
    parser.add_argument("--test-start-date", default=None)
    parser.add_argument("--test-end-date", default=None)
    parser.add_argument("--holdout-wards", type=int, nargs="*", default=None, help="Specific wards to hold out")
    args = parser.parse_args()
    metrics = train_baselines(
        features_path=args.features,
        output_dir=args.output_dir,
        reports_dir=args.reports_dir,
        config_path=args.config,
        sample=args.sample,
        split_mode=args.split_mode,
        cutoff_date=args.cutoff_date,
        test_start_date=args.test_start_date,
        test_end_date=args.test_end_date,
        holdout_wards=args.holdout_wards,
        min_test_rows=args.min_test_rows,
    )
    print(
        json.dumps(
            {"random_forest": {"pr_auc": metrics["random_forest"]["pr_auc"]}}, indent=2
        )
    )


if __name__ == "__main__":
    main()
