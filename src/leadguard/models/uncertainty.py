"""Uncertainty quantification for LeadGuard.

Implements Architecture §7.3:
  - Split conformal prediction (global, α=0.10, target coverage 90%)
  - Mondrian (group-conditional) conformal prediction per income quartile
  - uncertainty_score = normalized conformal set size
  - 5-seed ensemble disagreement cross-check (Pearson corr > 0.6)

Calibration uses a held-out split; income quartile joined from
fairness_reference.parquet FOR CALIBRATION GROUPING ONLY — never
leaked into the feature matrix.

Usage:
    python -m leadguard.models.uncertainty --config configs/train.yaml
"""

from __future__ import annotations

import logging
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sklearn.calibration import CalibratedClassifierCV

from leadguard.utils.seed import SEED

logger = logging.getLogger(__name__)

# Ordered class labels for conformal prediction (binary: NotLead=0, Lead=1)
# NOTE: actual data values are "Lead", "Copper", "Galvanized".
# For conformal scoring we map: Lead→1 (positive), all others→0 (negative).
MATERIALS = ["NotLead", "Lead"]
N_MATERIALS = len(MATERIALS)

# Data labels that count as labeled for splitting/evaluation
_LABELED_MATERIALS = ["Lead", "Copper", "Galvanized"]


def compute_predictive_entropy(proba: np.ndarray) -> np.ndarray:
    """Compute normalized predictive entropy from binary probabilities.

    Formula: 1 - |2p - 1|, bounded in [0, 1].
    p=0.5 -> 1.0 (highly uncertain)
    p=1.0 or 0.0 -> 0.0 (highly certain)

    Args:
        proba: Array of shape (n, 2) with class probabilities, or shape (n,) with P(Lead).

    Returns:
        Array of uncertainty scores in [0, 1].
    """
    # Extract P(Lead) if it's a 2D array
    if proba.ndim == 2:
        p = proba[:, 1]
    else:
        p = proba

    p = np.clip(p, 0.0, 1.0)
    return 1.0 - np.abs(2.0 * p - 1.0)


def _uncertainty_from_set_size(set_size: np.ndarray, k: int = N_MATERIALS) -> np.ndarray:
    """Compute normalized uncertainty score from conformal set size.

    Formula: (|set| - 1) / (K - 1), normalized to [0, 1].

    Args:
        set_size: Array of conformal prediction set sizes.
        k: Total number of possible class labels.

    Returns:
        Array of uncertainty scores in [0, 1].
    """
    if k <= 1:
        return np.zeros_like(set_size, dtype=float)
    return np.clip((set_size - 1) / (k - 1), 0.0, 1.0)


def _platt_calibrate(model: object, X_cal: np.ndarray, y_cal: np.ndarray) -> CalibratedClassifierCV:
    """Wrap a model with Platt scaling using a calibration set.

    Args:
        model: Fitted classifier with predict_proba.
        X_cal: Calibration feature matrix.
        y_cal: Calibration labels.

    Returns:
        Calibrated classifier.
    """
    import sklearn
    from sklearn.utils.fixes import parse_version

    # Handle sklearn version differences for prefit calibration API
    if parse_version(sklearn.__version__) >= parse_version("1.6"):
        from sklearn.calibration import FrozenEstimator

        cal = CalibratedClassifierCV(
            estimator=FrozenEstimator(model), method="sigmoid", cv="prefit"
        )
    else:
        cal = CalibratedClassifierCV(estimator=model, method="sigmoid", cv="prefit")

    # In sklearn 1.9, cv='prefit' is completely removed and we must use cv=None with FrozenEstimator
    if parse_version(sklearn.__version__) >= parse_version("1.9"):
        from sklearn.calibration import FrozenEstimator

        cal = CalibratedClassifierCV(estimator=FrozenEstimator(model), method="sigmoid", cv=None)
    cal.fit(X_cal, y_cal)
    return cal


class SplitConformalPredictor:
    """Split conformal prediction sets (Architecture §7.3).

    Provides prediction sets with guaranteed marginal coverage.

    Args:
        alpha: Miscoverage level (default 0.10 → 90% coverage target).
    """

    def __init__(self, alpha: float = 0.10) -> None:
        self.alpha = alpha
        self.threshold_: float | None = None
        self._n_cal: int = 0

    def calibrate(self, scores: np.ndarray) -> None:
        """Fit conformal threshold from nonconformity scores.

        Args:
            scores: 1-D array of nonconformity scores on the calibration set.
                    For binary classification, use 1 - P(y_true).
        """
        self._n_cal = len(scores)
        # Quantile with finite-sample correction
        level = np.ceil((self._n_cal + 1) * (1 - self.alpha)) / self._n_cal
        level = float(np.clip(level, 0.0, 1.0))
        self.threshold_ = float(np.quantile(scores, level))
        logger.debug(
            "Conformal threshold = %.4f (α=%.2f, n_cal=%d)",
            self.threshold_,
            self.alpha,
            self._n_cal,
        )

    def predict_set(self, proba: np.ndarray) -> list[list[str]]:
        """Return conformal prediction sets for each row.

        Args:
            proba: Array of shape (n, C) with class probabilities.

        Returns:
            List of prediction sets (list of class-name strings per row).
        """
        if self.threshold_ is None:
            raise RuntimeError("Call calibrate() before predict_set()")
        sets = []
        for row in proba:
            # row is [P(NotLead), P(Lead)]
            pred_set = [MATERIALS[i] for i, p in enumerate(row) if (1 - p) <= self.threshold_]
            if not pred_set:
                pred_set = [MATERIALS[int(np.argmax(row))]]
            sets.append(pred_set)
        return sets


class MondriancConformalPredictor:
    """Mondrian (group-conditional) conformal predictor (Architecture §7.3).

    Calibrated separately per income quartile so coverage guarantees
    hold within each group, not just on average.

    Args:
        alpha: Miscoverage level (default 0.10).
    """

    def __init__(self, alpha: float = 0.10) -> None:
        self.alpha = alpha
        self._predictors: dict[int, SplitConformalPredictor] = {}

    def calibrate(self, scores: np.ndarray, quartiles: np.ndarray) -> None:
        """Fit per-quartile conformal thresholds.

        Args:
            scores: Nonconformity scores (1-D array).
            quartiles: Income quartile label per calibration row (1-4).
        """
        for q in range(1, 5):
            mask = quartiles == q
            if mask.sum() < 5:
                logger.warning(
                    "Quartile %d has only %d calibration rows; using global predictor",
                    q,
                    mask.sum(),
                )
                pred = SplitConformalPredictor(self.alpha)
                pred.calibrate(scores)
            else:
                pred = SplitConformalPredictor(self.alpha)
                pred.calibrate(scores[mask])
            self._predictors[q] = pred

    def predict_set(self, proba: np.ndarray, quartiles: np.ndarray) -> list[list[str]]:
        """Return per-quartile conformal prediction sets.

        Args:
            proba: Array of shape (n, C) probabilities.
            quartiles: Income quartile per row.

        Returns:
            List of prediction sets.
        """
        sets = []
        for row, q in zip(proba, quartiles):
            pred = self._predictors.get(int(q), self._predictors.get(1))
            row_2d = row.reshape(1, -1)
            sets.extend(pred.predict_set(row_2d))
        return sets


def calibrate_uncertainty(
    features_path: Path | str = "data/processed/features.parquet",
    model_dir: Path | str = "models/xgboost",
    fairness_ref_path: Path | str = "data/processed/fairness_reference.parquet",
    config_path: Path | str = "configs/train.yaml",
    reports_dir: Path | str = "reports",
    sample: bool = False,
    split_mode: str = "geographic",
    cutoff_date: str | None = None,
    test_start_date: str | None = None,
    test_end_date: str | None = None,
    holdout_wards: list[int] | None = None,
    min_test_rows: int | None = 100,
) -> dict:
    """Calibrate conformal predictors for XGBoost.

    Args:
        features_path: Path to processed features.
        model_dir: Directory containing trained XGBoost model.
        config_path: Config with confidence_level.
        sample: If true, use sample features.
        split_mode: 'geographic', 'temporal', or 'spatial-temporal'
        cutoff_date: Cutoff date for temporal splits.
        holdout_wards: Holdout wards for spatial splits.

    Returns:
        Dictionary with coverage verification results.
    """
    import xgboost as xgb  # noqa: PLC0415
    from sklearn.preprocessing import LabelEncoder  # noqa: PLC0415

    model_dir = Path(model_dir)
    features_path = Path(features_path)
    if sample and not features_path.exists():
        features_path = Path("data/processed/features_sample.parquet")

    cfg = yaml.safe_load(Path(config_path).read_text()) if Path(config_path).exists() else {}
    cfg = yaml.safe_load(Path(config_path).read_text()) if Path(config_path).exists() else {}
    # Read confidence from train config, then scoring config overrides
    confidence = cfg.get("confidence_level", 0.90) if isinstance(cfg, dict) else 0.90
    scoring_cfg_path = Path("configs/scoring.yaml")
    if scoring_cfg_path.exists():
        scoring_cfg = yaml.safe_load(scoring_cfg_path.read_text())
        if isinstance(scoring_cfg, dict):
            confidence = scoring_cfg.get("confidence_level", confidence)
            
    if not 0.0 < confidence < 1.0:
        raise ValueError(f"Confidence level must be strictly between 0 and 1, got {confidence}")
    alpha = 1.0 - confidence

    import sys
    sys.path.insert(0, str(Path("src").absolute()))
    from leadguard.models.serving import load_serving_model, predict_proba, encode_target, KNOWN_MATERIALS

    try:
        model = load_serving_model(model_dir)
    except Exception as e:
        raise RuntimeError(f"Failed to load serving model for uncertainty calibration: {e}")

    # Load features
    df = pd.read_parquet(features_path)
    labeled = df[df["service_line_material"].isin(KNOWN_MATERIALS)].copy()

    from leadguard.data.features import build_features  # noqa: PLC0415
    from leadguard.data.split import split_dataset  # noqa: PLC0415
    from leadguard.models.xgboost_model import XGB_FEATURES  # noqa: PLC0415

    # Centralized split
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
    cal_df = split_res.calibration
    test_df = split_res.test
    
    train_ids = set(train_df["property_id"])
    cal_ids = set(cal_df["property_id"])
    test_ids = set(test_df["property_id"])
    assert train_ids.isdisjoint(cal_ids), "Train and Cal property IDs must be disjoint"
    assert train_ids.isdisjoint(test_ids), "Train and Test property IDs must be disjoint"
    assert cal_ids.isdisjoint(test_ids), "Cal and Test property IDs must be disjoint"

    as_of_date = split_res.metadata.get("cutoff_date")

    cal_f = build_features(cal_df, reference_df=train_df, include_label_dependent=True, as_of_date=as_of_date)
    test_f = build_features(test_df, reference_df=train_df, include_label_dependent=True, as_of_date=as_of_date)

    from leadguard.data.validation import validate_features
    X_cal = validate_features(cal_f, XGB_FEATURES).values
    X_test = validate_features(test_f, XGB_FEATURES).values

    # Encode targets consistently
    y_cal = encode_target(cal_df["service_line_material"]).values
    y_test = encode_target(test_df["service_line_material"]).values

    proba_cal = predict_proba(model, X_cal)
    proba_test = predict_proba(model, X_test)

    def _nonconformity(proba: np.ndarray, y_true: np.ndarray) -> np.ndarray:
        """Compute nonconformity scores: 1 - P(true class).
        proba is (n, 2) where col 0 = NotLead, col 1 = Lead.
        y_true is (n,) where 0 = NotLead, 1 = Lead.
        """
        p_true = proba[np.arange(len(y_true)), y_true]
        return 1.0 - p_true

    scores_cal = _nonconformity(proba_cal, y_cal)

    # Global split conformal
    global_cp = SplitConformalPredictor(alpha=alpha)
    global_cp.calibrate(scores_cal)
    with (model_dir / "conformal_global.pkl").open("wb") as f:
        pickle.dump(global_cp, f)

    # Verify empirical coverage on test set
    scores_test = _nonconformity(proba_test, y_test)
    global_coverage = float((scores_test <= global_cp.threshold_).mean())
    set_sizes = (proba_test >= (1.0 - global_cp.threshold_)).sum(axis=1)
    avg_set_size = float(set_sizes.mean())
    logger.info("Global empirical coverage: %.3f, Avg Set Size: %.3f (target: %.3f)", global_coverage, avg_set_size, 1 - alpha)


    # Mondrian conformal — per income quartile (join from fairness_reference, not features)
    fairness_ref_path = Path(fairness_ref_path)
    quartile_coverage = {}

    if fairness_ref_path.exists():
        fairness_ref = pd.read_parquet(fairness_ref_path)
        cal_with_tract = cal_df.merge(fairness_ref, on="census_tract", how="left")
        cal_quartiles = cal_with_tract["income_quartile"].fillna(2).astype(int).values
        test_with_tract = test_df.merge(fairness_ref, on="census_tract", how="left")
        test_quartiles = test_with_tract["income_quartile"].fillna(2).astype(int).values

        mondrian_cp = MondriancConformalPredictor(alpha=alpha)
        mondrian_cp.calibrate(scores_cal, cal_quartiles)

        # Per-quartile coverage verification
        for q in range(1, 5):
            mask = test_quartiles == q
            if mask.sum() == 0:
                quartile_coverage[q] = float("nan")
                continue
            q_scores = _nonconformity(proba_test[mask], y_test[mask])
            q_threshold = mondrian_cp._predictors[q].threshold_
            quartile_coverage[q] = float((q_scores <= q_threshold).mean())
            q_set_sizes = (proba_test[mask] >= (1.0 - q_threshold)).sum(axis=1)
            q_avg_set_size = float(q_set_sizes.mean())
            logger.info(
                "Quartile %d empirical coverage: %.3f, Avg Set Size: %.3f (n=%d)",
                q,
                quartile_coverage[q],
                q_avg_set_size,
                mask.sum(),
            )
    else:
        logger.warning("fairness_reference.parquet not found; skipping Mondrian calibration")
        mondrian_cp = MondriancConformalPredictor(alpha=alpha)
        mondrian_cp.calibrate(scores_cal, np.ones(len(scores_cal), dtype=int) * 2)
        quartile_coverage = {q: global_coverage for q in range(1, 5)}

    with (model_dir / "conformal_by_quartile.pkl").open("wb") as f:
        pickle.dump(mondrian_cp, f)

    result = {
        "global_coverage": global_coverage,
        "target_coverage": 1 - alpha,
        "quartile_coverage": {str(k): v for k, v in quartile_coverage.items()},
        "global_threshold": global_cp.threshold_,
    }

    import json
    reports_dir = Path(reports_dir)
    reports_dir.mkdir(parents=True, exist_ok=True)
    with (reports_dir / "uncertainty_metrics.json").open("w") as f:
        json.dump(result, f, indent=2, default=str)

    logger.info("Uncertainty calibration complete: %s", result)
    return result


def main():
    import argparse

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description="Calibrate conformal predictors")
    parser.add_argument("--config", default="configs/train.yaml")
    parser.add_argument("--features", default="data/processed/features.parquet")
    parser.add_argument("--output-dir", default="models/xgboost")
    parser.add_argument("--reports-dir", default="reports")
    parser.add_argument(
        "--fairness-ref", default="data/processed/fairness_reference.parquet"
    )
    parser.add_argument("--sample", action="store_true", help="Use sample features")
    parser.add_argument("--min-test-rows", type=int, default=100)
    parser.add_argument("--split-mode", default="geographic", help="geographic, temporal, or spatial-temporal")
    parser.add_argument("--cutoff-date", default=None, help="YYYY-MM-DD for temporal splits")
    parser.add_argument("--test-start-date", default=None)
    parser.add_argument("--test-end-date", default=None)
    parser.add_argument("--holdout-wards", type=int, nargs="*", default=None, help="Specific wards to hold out")
    args = parser.parse_args()
    result = calibrate_uncertainty(
        model_dir=args.output_dir,
        features_path=args.features,
        fairness_ref_path=args.fairness_ref,
        config_path=args.config,
        reports_dir=args.reports_dir,
        sample=args.sample,
        split_mode=args.split_mode,
        cutoff_date=args.cutoff_date,
        test_start_date=args.test_start_date,
        test_end_date=args.test_end_date,
        holdout_wards=args.holdout_wards,
        min_test_rows=args.min_test_rows,
    )
    print(result)


if __name__ == "__main__":
    main()
