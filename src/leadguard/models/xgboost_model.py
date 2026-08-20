"""XGBoost advanced model training for LeadGuard.

Implements Architecture §7.2:
  - XGBoost with monotonic constraint +1 on year_built
  - scale_pos_weight from class ratio
  - Optuna hyperparameter search (100 trials / 60-min budget)
  - Geographic holdout training + leakage gap check

Usage:
    python -m leadguard.models.xgboost_model --config configs/train.yaml
    python -m leadguard.models.xgboost_model --config configs/train.yaml --sample
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path

import matplotlib
import numpy as np
import optuna
import pandas as pd
import xgboost as xgb
import yaml

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.calibration import CalibratedClassifierCV

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
optuna.logging.set_verbosity(optuna.logging.WARNING)

# Feature columns (must not include any demographic or forbidden columns)
XGB_FEATURES = [
    "year_built",
    "lot_size_sqft",
    "building_sqft",
    "stories",
    "has_basement",
    "dist_to_nearest_hydrant_m",
    "dist_to_nearest_known_lead_m",
    "neighbor_lead_rate_h3res8",
    "knn10_lead_rate",
]

# Monotonic constraint: +1 means older year_built → higher lead probability
# year_built is the 0th feature in XGB_FEATURES → index 0
MONOTONE_CONSTRAINTS = {feat: 0 for feat in XGB_FEATURES}
MONOTONE_CONSTRAINTS["year_built"] = -1  # Lower year_built (older) → higher probability


def _prep_xy(df: pd.DataFrame, features: list[str]) -> tuple[np.ndarray, np.ndarray]:
    """Extract feature matrix and binary target.

    Args:
        df: Labeled DataFrame.
        features: Feature column names.

    Returns:
        Tuple of (X float array, y binary array).
    """
    X = df.reindex(columns=features, fill_value=0.0).astype(float).values
    y = (df["service_line_material"] == "Lead").astype(int).values
    return X, y


def _build_xgb_params(trial: optuna.Trial, cfg: dict, scale_pos_weight: float) -> dict:
    """Build XGBoost params from an Optuna trial.

    Args:
        trial: Optuna trial object.
        cfg: Training config dict.
        scale_pos_weight: Class imbalance weight.

    Returns:
        XGBoost parameter dictionary.
    """
    ss = cfg.get("optuna", {}).get("search_space", {})
    return {
        "max_depth": trial.suggest_int("max_depth", *ss.get("max_depth", [3, 8])),
        "learning_rate": trial.suggest_float(
            "learning_rate", *ss.get("learning_rate", [0.01, 0.3]), log=True
        ),
        "subsample": trial.suggest_float("subsample", *ss.get("subsample", [0.6, 1.0])),
        "colsample_bytree": trial.suggest_float(
            "colsample_bytree", *ss.get("colsample_bytree", [0.6, 1.0])
        ),
        "min_child_weight": trial.suggest_int(
            "min_child_weight", *ss.get("min_child_weight", [1, 10])
        ),
        "n_estimators": cfg.get("xgboost", {}).get("n_estimators", 500),
        "early_stopping_rounds": cfg.get("xgboost", {}).get("early_stopping_rounds", 30),
        "eval_metric": "aucpr",
        "tree_method": "hist",
        "device": "cpu",
        "scale_pos_weight": scale_pos_weight,
        "monotone_constraints": tuple(MONOTONE_CONSTRAINTS[f] for f in XGB_FEATURES),
        "random_state": SEED,
        "use_label_encoder": False,
    }


def train_xgboost(
    features_path: Path | str = "data/processed/features.parquet",
    output_dir: Path | str = "models/xgboost",
    reports_dir: Path | str = "reports",
    config_path: Path | str = "configs/train.yaml",
    baseline_metrics_path: Path | str = "reports/baseline_metrics.json",
    sample: bool = False,
) -> dict:
    """Train XGBoost with Optuna search and geographic holdout.

    Args:
        features_path: Path to features parquet.
        output_dir: Where to save model artifact.
        reports_dir: Where to write metrics and plots.
        config_path: Training config YAML.
        baseline_metrics_path: Baseline metrics for gate check.
        sample: Use sample data paths if True.

    Returns:
        Dictionary of XGBoost metrics.
    """
    features_path = Path(features_path)
    output_dir = Path(output_dir)
    reports_dir = Path(reports_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    cfg = yaml.safe_load(Path(config_path).read_text()) if Path(config_path).exists() else {}

    if sample and not features_path.exists():
        features_path = Path("data/processed/features_sample.parquet")

    df = pd.read_parquet(features_path)
    labeled = df[df["service_line_material"].isin(["Lead", "Copper", "Galvanized"])].copy()
    logger.info("Loaded %d labeled rows", len(labeled))

    # Geo 3-way split
    train_geo, cal_geo, test_geo = geographic_split(labeled)

    # Generate label-dependent spatial features (Leakage Guard!)
    logger.info("Building label-dependent features for Geo Split")
    train_geo_f = build_features(train_geo, reference_df=train_geo, include_label_dependent=True)
    cal_geo_f = build_features(cal_geo, reference_df=train_geo, include_label_dependent=True)
    test_geo_f = build_features(test_geo, reference_df=train_geo, include_label_dependent=True)

    X_train, y_train = _prep_xy(train_geo_f, XGB_FEATURES)
    X_cal, y_cal = _prep_xy(cal_geo_f, XGB_FEATURES)
    X_test, y_test = _prep_xy(test_geo_f, XGB_FEATURES)

    # Random split for leakage gap check
    # Random split for leakage gap check
    train_r, cal_r, test_r = random_split(labeled, seed=SEED)
    test_r_f = build_features(test_r, reference_df=train_r, include_label_dependent=True)
    X_test_rand, y_test_rand = _prep_xy(test_r_f, XGB_FEATURES)

    # class imbalance weight
    n_neg = (y_train == 0).sum()
    n_pos = (y_train == 1).sum()
    scale_pos_weight = float(n_neg / n_pos) if n_pos > 0 else 1.0
    logger.info("scale_pos_weight = %.2f (neg=%d, pos=%d)", scale_pos_weight, n_neg, n_pos)

    # Split off an early-stopping set from train (10%) to preserve CAL strictly for calibration
    es_size = max(1, int(len(X_train) * 0.10))
    rng = np.random.default_rng(SEED)
    idx = rng.permutation(len(X_train))
    es_idx, tr_idx = idx[:es_size], idx[es_size:]
    X_es, y_es = X_train[es_idx], y_train[es_idx]
    X_tr, y_tr = X_train[tr_idx], y_train[tr_idx]

    # -----------------------------------------------------------------------
    # Optuna hyperparameter search
    # -----------------------------------------------------------------------
    n_trials = cfg.get("optuna", {}).get("n_trials", 20)
    timeout = cfg.get("optuna", {}).get("timeout_seconds", 3600)
    start_time = time.time()

    def objective(trial: optuna.Trial) -> float:
        """Optuna objective: PR-AUC on calibration set."""
        params = _build_xgb_params(trial, cfg, scale_pos_weight)
        n_est = params.pop("n_estimators")
        early = params.pop("early_stopping_rounds")
        model = xgb.XGBClassifier(
            **params,
            n_estimators=n_est,
            early_stopping_rounds=early,
            verbosity=0,
        )
        model.fit(
            X_tr,
            y_tr,
            eval_set=[(X_es, y_es)],
            verbose=False,
        )
        from sklearn.metrics import average_precision_score  # noqa: PLC0415

        proba = model.predict_proba(X_es)[:, 1]
        return float(average_precision_score(y_es, proba))

    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=SEED))
    study.optimize(objective, n_trials=n_trials, timeout=timeout - (time.time() - start_time))

    best_params = study.best_params
    logger.info("Best Optuna params: %s  (val PR-AUC=%.4f)", best_params, study.best_value)

    # Retrain best model on full TRAIN
    best_params = _build_xgb_params(study.best_trial, cfg, scale_pos_weight)
    n_est = best_params.pop("n_estimators")
    early = best_params.pop("early_stopping_rounds")
    best_model = xgb.XGBClassifier(
        **best_params,
        n_estimators=n_est,
        early_stopping_rounds=early,
        verbosity=0,
    )
    best_model.fit(
        X_tr,
        y_tr,
        eval_set=[(X_es, y_es)],
        verbose=False,
    )

    # Apply Probability Calibration on CAL
    logger.info("Fitting CalibratedClassifierCV on CAL set")
    import sklearn
    from sklearn.utils.fixes import parse_version

    # Handle sklearn version differences for prefit calibration API
    if parse_version(sklearn.__version__) >= parse_version("1.6"):
        from sklearn.calibration import FrozenEstimator

        calibrated_model = CalibratedClassifierCV(
            estimator=FrozenEstimator(best_model), method="sigmoid", cv="prefit"
        )
    else:
        calibrated_model = CalibratedClassifierCV(
            estimator=best_model, method="sigmoid", cv="prefit"
        )

    # In sklearn 1.9, cv='prefit' is completely removed
    if parse_version(sklearn.__version__) >= parse_version("1.9"):
        from sklearn.calibration import FrozenEstimator

        calibrated_model = CalibratedClassifierCV(
            estimator=FrozenEstimator(best_model), method="sigmoid", cv=None
        )
    calibrated_model.fit(X_cal, y_cal)

    # Predict on TEST using calibrated model
    test_proba = calibrated_model.predict_proba(X_test)[:, 1]
    metrics_geo = compute_metrics(y_test, test_proba, prefix="test_geo_")

    # Predict on random TEST (for leakage gap check)
    test_proba_rand = calibrated_model.predict_proba(X_test_rand)[:, 1]
    metrics_rand = compute_metrics(y_test_rand, test_proba_rand, prefix="test_rand_")

    # Leakage gap check (Architecture §7.6)
    leakage_ok = check_leakage_gap(metrics_rand["test_rand_pr_auc"], metrics_geo["test_geo_pr_auc"])
    if not leakage_ok:
        logger.error("LEAKAGE CHECK FAILED — investigate before proceeding to Phase 5")

    # Gate check vs baseline
    baseline_path = Path(baseline_metrics_path)
    if baseline_path.exists():
        baseline = json.loads(baseline_path.read_text())
        rf_geo = baseline.get("random_forest", {}).get("pr_auc_geo", None)
        if rf_geo is not None:
            improvement = (metrics_geo["test_geo_pr_auc"] - rf_geo) / max(rf_geo, 1e-9)
            logger.info("XGBoost improvement over RF geo: %.1f%% (gate: >10%%)", improvement * 100)
            if improvement <= 0.10:
                logger.warning("GATE NOT MET: improvement=%.1f%% <= 10%%", improvement * 100)

    # Export the CALIBRATED model artifact for serving
    import pickle

    model_artifact_path = output_dir / "xgb_model.pkl"
    with open(model_artifact_path, "wb") as f:
        pickle.dump(calibrated_model, f)
    logger.info("Saved calibrated XGBoost pipeline to %s", model_artifact_path)

    # Save metrics
    result = {
        "pr_auc_geo": metrics_geo["test_geo_pr_auc"],
        "pr_auc_random": metrics_rand["test_rand_pr_auc"],
        "best_optuna_params": best_params,
        "scale_pos_weight": scale_pos_weight,
        "features": XGB_FEATURES,
        "leakage_check_passed": leakage_ok,
        **{f"{k}_random": v for k, v in metrics_rand.items()},
    }
    write_metrics(result, output_dir / "metrics.json")

    # Feature importance plot
    _plot_feature_importance(best_model, XGB_FEATURES, reports_dir / "feature_importance.png")

    logger.info("Phase 4 model saved to %s", model_artifact_path)
    return result


def _plot_feature_importance(
    model: xgb.XGBClassifier, features: list[str], output_path: Path
) -> None:
    """Save a feature importance bar chart.

    Args:
        model: Trained XGBoost model.
        features: Feature names.
        output_path: PNG output path.
    """
    importances = model.feature_importances_
    sorted_idx = np.argsort(importances)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(np.array(features)[sorted_idx], importances[sorted_idx], color="#4C72B0")
    ax.set_title("XGBoost Feature Importance (gain)")
    ax.set_xlabel("Importance")
    plt.tight_layout()
    fig.savefig(output_path, dpi=120)
    plt.close(fig)
    logger.info("Feature importance plot saved to %s", output_path)


def load_model(model_dir: Path | str = "models/xgboost") -> xgb.XGBClassifier:
    """Load a trained XGBoost model from disk.

    Args:
        model_dir: Directory containing model.json.

    Returns:
        Loaded XGBoost classifier.
    """
    model_path = Path(model_dir) / "model.json"
    model = xgb.XGBClassifier()
    model.load_model(str(model_path))
    return model


def main():
    import argparse

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description="Train LeadGuard XGBoost model")
    parser.add_argument("--config", default="configs/train.yaml")
    parser.add_argument("--features", default="data/processed/features.parquet")
    parser.add_argument("--sample", action="store_true")
    args = parser.parse_args()
    result = train_xgboost(features_path=args.features, config_path=args.config, sample=args.sample)
    print(f"XGBoost geo PR-AUC: {result['pr_auc_geo']:.4f}")


if __name__ == "__main__":
    main()
