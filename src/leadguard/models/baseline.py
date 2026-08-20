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
from leadguard.evaluation.metrics import (
    check_leakage_gap,
    compute_metrics,
    geographic_split,
    random_split,
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
FULL_FEATURES = BASELINE_FEATURES + ["neighbor_lead_rate_h3res8", "knn10_lead_rate"]


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
    avail = [f for f in features if f in df.columns]
    if len(avail) < len(features):
        missing = set(features) - set(avail)
        logger.warning("Missing features (will fill 0): %s", missing)

    X = df.reindex(columns=features, fill_value=0.0).astype(float).values
    y = (df[target_col] == positive_class).astype(int).values
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


def train_baselines(
    features_path: Path | str = "data/processed/features.parquet",
    output_dir: Path | str = "models/baseline",
    reports_dir: Path | str = "reports",
    config_path: Path | str = "configs/train.yaml",
    sample: bool = False,
) -> dict:
    """Train all three baselines and write results.

    Args:
        features_path: Path to features.parquet.
        output_dir: Where to save model artifacts.
        reports_dir: Where to write baseline_metrics.json.
        config_path: Training config YAML.
        sample: If True, use sample features path.

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

    # 3-way Splits
    train_rand, cal_rand, test_rand = random_split(labeled, seed=SEED)
    train_geo, cal_geo, test_geo = geographic_split(labeled)

    # Feature generation
    train_geo_f = build_features(train_geo, reference_df=train_geo, include_label_dependent=True)
    test_geo_f = build_features(test_geo, reference_df=train_geo, include_label_dependent=True)

    train_r_f = build_features(train_rand, reference_df=train_rand, include_label_dependent=True)
    test_r_f = build_features(test_rand, reference_df=train_rand, include_label_dependent=True)

    all_metrics: dict = {}

    # -----------------------------------------------------------------------
    # Baseline 0 — Year-built heuristic
    # -----------------------------------------------------------------------
    heuristic = YearBuiltHeuristic()
    X_test_rand, y_test_rand = _prep_xy(test_rand, ["year_built"])
    X_test_geo, y_test_geo = _prep_xy(test_geo, ["year_built"])
    m_rand = compute_metrics(
        y_test_rand, heuristic.predict_proba(X_test_rand)[:, 1], prefix="test_rand_"
    )
    m_geo = compute_metrics(
        y_test_geo, heuristic.predict_proba(X_test_geo)[:, 1], prefix="test_geo_"
    )
    all_metrics["heuristic"] = {
        "pr_auc_random": m_rand["test_rand_pr_auc"],
        "pr_auc_geo": m_geo["test_geo_pr_auc"],
        **m_rand,
        **m_geo,
    }

    # -----------------------------------------------------------------------
    # Baseline 1 — Logistic regression
    # -----------------------------------------------------------------------
    scaler = StandardScaler()
    X_train_rand, y_train_rand = _prep_xy(train_r_f, FULL_FEATURES)
    X_test_rand_full, y_test_rand_full = _prep_xy(test_r_f, FULL_FEATURES)
    X_train_geo, y_train_geo = _prep_xy(train_geo_f, FULL_FEATURES)
    X_test_geo_full, y_test_geo_full = _prep_xy(test_geo_f, FULL_FEATURES)

    lr_cfg = cfg.get("baseline", {}).get("logistic_regression", {})
    lr = LogisticRegression(
        C=lr_cfg.get("C", 1.0),
        penalty=lr_cfg.get("penalty", "l2"),
        max_iter=lr_cfg.get("max_iter", 1000),
        solver=lr_cfg.get("solver", "lbfgs"),
        random_state=SEED,
        class_weight="balanced",
    )
    X_train_scaled = scaler.fit_transform(X_train_rand)
    lr.fit(X_train_scaled, y_train_rand)
    m_rand_lr = compute_metrics(
        y_test_rand_full,
        lr.predict_proba(scaler.transform(X_test_rand_full))[:, 1],
        prefix="test_rand_",
    )

    # Retrain on geo train split
    scaler_geo = StandardScaler()
    X_train_geo_scaled = scaler_geo.fit_transform(X_train_geo)
    lr_geo = LogisticRegression(
        C=1.0, max_iter=1000, solver="lbfgs", random_state=SEED, class_weight="balanced"
    )
    lr_geo.fit(X_train_geo_scaled, y_train_geo)
    m_geo_lr = compute_metrics(
        y_test_geo_full,
        lr_geo.predict_proba(scaler_geo.transform(X_test_geo_full))[:, 1],
        prefix="test_geo_",
    )
    all_metrics["logistic_regression"] = {
        "pr_auc_random": m_rand_lr["test_rand_pr_auc"],
        "pr_auc_geo": m_geo_lr["test_geo_pr_auc"],
        **m_rand_lr,
        **m_geo_lr,
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
    rf.fit(X_train_rand, y_train_rand)
    m_rand_rf = compute_metrics(
        y_test_rand_full, rf.predict_proba(X_test_rand_full)[:, 1], prefix="test_rand_"
    )

    rf_geo = RandomForestClassifier(
        n_estimators=300,
        max_depth=10,
        min_samples_leaf=5,
        random_state=SEED,
        class_weight="balanced",
        n_jobs=-1,
    )
    rf_geo.fit(X_train_geo, y_train_geo)
    m_geo_rf = compute_metrics(
        y_test_geo_full, rf_geo.predict_proba(X_test_geo_full)[:, 1], prefix="test_geo_"
    )
    all_metrics["random_forest"] = {
        "pr_auc_random": m_rand_rf["test_rand_pr_auc"],
        "pr_auc_geo": m_geo_rf["test_geo_pr_auc"],
        **m_rand_rf,
        **m_geo_rf,
    }

    # Leakage check on RF
    check_leakage_gap(m_rand_rf["test_rand_pr_auc"], m_geo_rf["test_geo_pr_auc"])

    # Save RF artifact
    with (output_dir / "random_forest.pkl").open("wb") as f:
        pickle.dump({"model": rf_geo, "features": FULL_FEATURES}, f)

    # Write metrics
    write_metrics(all_metrics, reports_dir / "baseline_metrics.json")
    logger.info("Phase 3 PASS — RF geo PR-AUC: %.4f", m_geo_rf["pr_auc"])
    return all_metrics


def main():
    import argparse

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description="Train LeadGuard baseline models")
    parser.add_argument("--config", default="configs/train.yaml")
    parser.add_argument("--features", default="data/processed/features.parquet")
    parser.add_argument("--sample", action="store_true", help="Use sample features")
    args = parser.parse_args()
    metrics = train_baselines(
        features_path=args.features, config_path=args.config, sample=args.sample
    )
    print(
        json.dumps(
            {"random_forest": {"pr_auc_geo": metrics["random_forest"]["pr_auc_geo"]}}, indent=2
        )
    )


if __name__ == "__main__":
    main()
