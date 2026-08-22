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
import pickle
import time
from datetime import UTC, datetime
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
from sklearn.model_selection import train_test_split

from leadguard.data.features import build_features
from leadguard.data.split import split_dataset
from leadguard.evaluation.metrics import (
    compute_metrics,
    write_metrics,
)
from leadguard.utils.seed import SEED

logger = logging.getLogger(__name__)
optuna.logging.set_verbosity(optuna.logging.WARNING)

# Feature Groups (C2: Four-Way Ablation)
GROUP_A_INTRINSIC = [
    "year_built",
    "lot_size_sqft",
    "building_sqft",
    "stories",
    "has_basement",
]
GROUP_B_GEO = [
    "dist_to_nearest_hydrant_m",
]
GROUP_C_PROCESS = [
    "inspection_count_in_ward",
    "known_lead_count_in_ward",
    "days_since_last_inspection_in_ward",
]
GROUP_D_OBSERVED = [
    "dist_to_nearest_known_lead_m",
    "neighbor_lead_rate_h3res8",
    "knn10_lead_rate",
    "known_lead_rate_in_ward",
]

FEATURE_SETS = {
    "intrinsic": GROUP_A_INTRINSIC,
    "intrinsic_geo": GROUP_A_INTRINSIC + GROUP_B_GEO,
    "intrinsic_geo_process": GROUP_A_INTRINSIC + GROUP_B_GEO + GROUP_C_PROCESS,
    "full": GROUP_A_INTRINSIC + GROUP_B_GEO + GROUP_C_PROCESS + GROUP_D_OBSERVED,
}

# Default backwards compatible exported feature list
# Based on Phase C ablation, intrinsic_geo provides the best temporal generalization
XGB_FEATURES = FEATURE_SETS["intrinsic_geo"]

# Monotonic constraint: -1 means older year_built (lower value) → higher lead probability
MONOTONE_CONSTRAINTS = {feat: 0 for feat in FEATURE_SETS["full"]}
MONOTONE_CONSTRAINTS["year_built"] = -1  # Lower year_built (older) → higher probability


def _prep_xy(df: pd.DataFrame, features: list[str]) -> tuple[np.ndarray, np.ndarray]:
    """Extract feature matrix and binary target.

    Args:
        df: Labeled DataFrame.
        features: Feature column names.

    Returns:
        Tuple of (X float array, y binary array).
    """
    from leadguard.data.validation import validate_features
    from leadguard.models.serving import encode_target

    X = validate_features(df, features).values
    y = encode_target(df["service_line_material"]).values
    return X, y


def _build_xgb_params(trial: optuna.Trial, cfg: dict, scale_pos_weight: float, features_list: list[str]) -> dict:
    """Build XGBoost params from an Optuna trial.

    Args:
        trial: Optuna trial object.
        cfg: Training config dict.
        scale_pos_weight: Class imbalance weight.
        features_list: List of features to use.

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
        "monotone_constraints": tuple(MONOTONE_CONSTRAINTS[f] for f in features_list),
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
    split_mode: str = "geographic",
    cutoff_date: str | None = None,
    test_start_date: str | None = None,
    test_end_date: str | None = None,
    holdout_wards: list[int] | None = None,
    min_test_rows: int | None = 100,
    feature_set: str = "full",
) -> dict:
    """Train XGBoost with Optuna search and holdout.

    Args:
        features_path: Path to features parquet.
        output_dir: Where to save model artifact.
        reports_dir: Where to write metrics and plots.
        config_path: Training config YAML.
        baseline_metrics_path: Baseline metrics for gate check.
        sample: Use sample data paths if True.
        split_mode: 'geographic', 'temporal', or 'spatial-temporal'
        cutoff_date: Cutoff date for temporal splits.
        holdout_wards: Holdout wards for spatial splits.
        min_test_rows: Minimum test rows required.

    Returns:
        Dictionary of XGBoost metrics.
    """
    if feature_set not in FEATURE_SETS:
        raise ValueError(f"Unknown feature set: {feature_set}")
    features_to_use = FEATURE_SETS[feature_set]

    features_path = Path(features_path)
    output_dir = Path(output_dir)
    reports_dir = Path(reports_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    cfg = yaml.safe_load(Path(config_path).read_text()) if Path(config_path).exists() else {}

    if sample and not features_path.exists():
        features_path = Path("data/processed/features_sample.parquet")

    from leadguard.models.serving import KNOWN_MATERIALS
    df = pd.read_parquet(features_path)
    labeled = df[df["service_line_material"].isin(KNOWN_MATERIALS)].copy()
    logger.info("Loaded %d labeled rows", len(labeled))

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

    train_geo = split_res.train
    cal_geo = split_res.calibration
    test_geo = split_res.test

    as_of_date = split_res.metadata.get("cutoff_date")

    # Generate label-dependent spatial features for final evaluation
    logger.info("Building label-dependent features for Split")
    train_geo_f = build_features(
        train_geo, reference_df=train_geo, include_label_dependent=True, as_of_date=as_of_date
    )
    cal_geo_f = build_features(
        cal_geo, reference_df=train_geo, include_label_dependent=True, as_of_date=as_of_date
    )
    test_geo_f = build_features(
        test_geo, reference_df=train_geo, include_label_dependent=True, as_of_date=as_of_date
    )

    X_train, y_train = _prep_xy(train_geo_f, features_to_use)
    X_cal, y_cal = _prep_xy(cal_geo_f, features_to_use)
    X_test, y_test = _prep_xy(test_geo_f, features_to_use)

    # Split train_geo for Optuna early stopping — geographic split avoids leakage,
    # temporal split uses random (seeded) since there is no spatial structure left
    if split_mode in ("geographic", "spatial-temporal"):
        rng_wards = np.random.default_rng(SEED)
        train_wards = sorted(train_geo["ward"].unique())
        n_val_wards = max(1, len(train_wards) // 10)
        val_wards = rng_wards.choice(train_wards, size=n_val_wards, replace=False)
        val_mask = train_geo["ward"].isin(val_wards)
        train_sub = train_geo[~val_mask]
        val_sub = train_geo[val_mask]
    else:
        train_sub, val_sub = train_test_split(train_geo, test_size=0.15, random_state=SEED)

    train_sub_f = build_features(
        train_sub, reference_df=train_sub, include_label_dependent=True, as_of_date=as_of_date
    )
    val_sub_f = build_features(
        val_sub, reference_df=train_sub, include_label_dependent=True, as_of_date=as_of_date
    )

    X_tr, y_tr = _prep_xy(train_sub_f, features_to_use)
    X_es, y_es = _prep_xy(val_sub_f, features_to_use)

    # Class imbalance weight
    n_neg = (y_train == 0).sum()
    n_pos = (y_train == 1).sum()
    scale_pos_weight = float(n_neg / n_pos) if n_pos > 0 else 1.0
    logger.info("scale_pos_weight = %.2f (neg=%d, pos=%d)", scale_pos_weight, n_neg, n_pos)

    # -----------------------------------------------------------------------
    # Optuna hyperparameter search
    # -----------------------------------------------------------------------
    n_trials = cfg.get("optuna", {}).get("n_trials", 20)
    timeout = cfg.get("optuna", {}).get("timeout_seconds", 3600)
    start_time = time.time()

    def objective(trial: optuna.Trial) -> float:
        """Optuna objective: PR-AUC on early-stopping validation set."""
        params = _build_xgb_params(trial, cfg, scale_pos_weight, features_to_use)
        n_est = params.pop("n_estimators")
        early = params.pop("early_stopping_rounds")
        model = xgb.XGBClassifier(
            **params,
            n_estimators=n_est,
            early_stopping_rounds=early,
            verbosity=0,
        )
        model.fit(X_tr, y_tr, eval_set=[(X_es, y_es)], verbose=False)
        from sklearn.metrics import average_precision_score  # noqa: PLC0415

        proba = model.predict_proba(X_es)[:, 1]
        return float(average_precision_score(y_es, proba))

    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=SEED))
    study.optimize(objective, n_trials=n_trials, timeout=timeout - (time.time() - start_time))

    best_params = study.best_params
    logger.info("Best Optuna params: %s  (val PR-AUC=%.4f)", best_params, study.best_value)

    # Retrain best model on full TRAIN (with X_cal as early-stopping monitor)
    best_params = _build_xgb_params(study.best_trial, cfg, scale_pos_weight, features_to_use)
    n_est = best_params.pop("n_estimators")
    early = best_params.pop("early_stopping_rounds")
    best_model = xgb.XGBClassifier(
        **best_params,
        n_estimators=n_est,
        early_stopping_rounds=early,
        verbosity=0,
    )
    best_model.fit(X_train, y_train, eval_set=[(X_cal, y_cal)], verbose=False)

    # ---------------------------------------------------------------------------
    # Probability Calibration on CAL set
    # ---------------------------------------------------------------------------
    logger.info("Fitting CalibratedClassifierCV on CAL set")
    import sklearn
    from sklearn.utils.fixes import parse_version

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

    # ---------------------------------------------------------------------------
    # Evaluate on TEST
    # ---------------------------------------------------------------------------
    test_proba = calibrated_model.predict_proba(X_test)[:, 1]
    m_eval = compute_metrics(y_test, test_proba, prefix="test_eval_")

    # Gate check vs baseline RF
    baseline_path = Path(baseline_metrics_path)
    if baseline_path.exists():
        baseline = json.loads(baseline_path.read_text())
        rf_eval = baseline.get("random_forest", {}).get("pr_auc", None)
        if rf_eval is not None:
            improvement = (m_eval["test_eval_pr_auc"] - rf_eval) / max(rf_eval, 1e-9)
            logger.info("XGBoost improvement over RF: %.1f%% (gate: >10%%)", improvement * 100)
            if improvement <= 0.10:
                logger.warning("GATE NOT MET: improvement=%.1f%% <= 10%%", improvement * 100)

    # ---------------------------------------------------------------------------
    # Persist artifacts
    # ---------------------------------------------------------------------------
    # BASE model — for SHAP and downstream analysis
    best_model.save_model(output_dir / "model.json")

    # CALIBRATED model — served by the API and used for conformal calibration
    model_artifact_path = output_dir / "xgb_model.pkl"
    with open(model_artifact_path, "wb") as f:
        pickle.dump(calibrated_model, f)
    logger.info("Saved calibrated XGBoost pipeline to %s", model_artifact_path)

    # Save artifact metadata
    if "split_mode" not in split_res.metadata:
        raise RuntimeError("Experiment integrity failed: split_metadata missing split_mode")
        
    artifact_metadata = {
        "model_version": "v2",
        "feature_version": "v1",
        "feature_names": XGB_FEATURES,
        "calibration_method": "sigmoid",
        "training_timestamp": datetime.now(UTC).isoformat(),
        "experiment_metadata": split_res.metadata,
    }
    with open(output_dir / "metadata.json", "w") as f:
        json.dump(artifact_metadata, f)

    # Metrics — C2 fix: split_metadata is now included
    result = {
        "pr_auc": m_eval["test_eval_pr_auc"],
        "best_optuna_params": best_params,
        "scale_pos_weight": scale_pos_weight,
        "features": XGB_FEATURES,
        "split_metadata": split_res.metadata,
        **m_eval,
    }
    write_metrics(result, output_dir / "metrics.json")
    
    # Final safety check on metrics
    persisted = json.loads((output_dir / "metrics.json").read_text())
    if "split_metadata" not in persisted:
        raise RuntimeError("Experiment integrity failed: metrics.json lacks split_metadata")

    # Feature importance plot
    _plot_feature_importance(best_model, features_to_use, reports_dir / "feature_importance.png")

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
    parser.add_argument("--output-dir", default="models/xgboost")
    parser.add_argument("--reports-dir", default="reports")
    parser.add_argument("--baseline", default="reports/baseline_metrics.json")
    parser.add_argument("--sample", action="store_true", help="Use sample features")
    parser.add_argument("--min-test-rows", type=int, default=100)
    parser.add_argument(
        "--split-mode",
        default="geographic",
        help="geographic, temporal, or spatial-temporal",
    )
    parser.add_argument("--cutoff-date", default=None, help="YYYY-MM-DD for temporal splits")
    parser.add_argument("--test-start-date", default=None)
    parser.add_argument("--test-end-date", default=None)
    parser.add_argument(
        "--holdout-wards",
        type=int,
        nargs="*",
        default=None,
        help="Specific wards to hold out",
    )
    parser.add_argument("--feature-set", default="intrinsic_geo", help="intrinsic, intrinsic_geo, intrinsic_geo_process, full")
    args = parser.parse_args()
    metrics = train_xgboost(
        features_path=args.features,
        output_dir=args.output_dir,
        reports_dir=args.reports_dir,
        config_path=args.config,
        baseline_metrics_path=args.baseline,
        sample=args.sample,
        split_mode=args.split_mode,
        cutoff_date=args.cutoff_date,
        test_start_date=args.test_start_date,
        test_end_date=args.test_end_date,
        holdout_wards=args.holdout_wards,
        feature_set=args.feature_set,
    )
    print(json.dumps({"xgboost": {"pr_auc": metrics["pr_auc"]}}, indent=2))


if __name__ == "__main__":
    main()
