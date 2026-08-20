"""Evaluation metrics for LeadGuard.

Provides geographic and random train/test splits, and computes
PR-AUC, ROC-AUC, F2, Brier score, and cost-sensitive metrics.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    fbeta_score,
    roc_auc_score,
)

from leadguard.utils.seed import SEED

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Split strategies
# ---------------------------------------------------------------------------


def random_split(
    df: pd.DataFrame,
    test_fraction: float = 0.15,
    cal_fraction: float = 0.15,
    seed: int = SEED,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Random 3-way split (train/cal/test) stratified on the target.

    Args:
        df: Full labeled DataFrame.
        test_fraction: Fraction reserved for final testing.
        cal_fraction: Fraction reserved for calibration.
        seed: Random seed.

    Returns:
        Tuple of (train_df, cal_df, test_df).
    """
    from sklearn.model_selection import train_test_split  # noqa: PLC0415

    labeled = df[df["service_line_material"].notna()].copy()
    labeled["_is_lead"] = (labeled["service_line_material"] == "Lead").astype(int)
    
    train_cal, test = train_test_split(
        labeled,
        test_size=test_fraction,
        random_state=seed,
        stratify=labeled["_is_lead"],
    )
    
    # Calculate relative fraction for cal from the train_cal remainder
    rel_cal_fraction = cal_fraction / (1.0 - test_fraction)
    train, cal = train_test_split(
        train_cal,
        test_size=rel_cal_fraction,
        random_state=seed + 1,
        stratify=train_cal["_is_lead"],
    )
    
    return train.drop(columns=["_is_lead"]), cal.drop(columns=["_is_lead"]), test.drop(columns=["_is_lead"])


def geographic_split(
    df: pd.DataFrame,
    holdout_test_wards: list[int] | None = None,
    holdout_cal_wards: list[int] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Geographic 3-way holdout split: entire wards excluded for testing and calibration.

    Args:
        df: Full labeled DataFrame.
        holdout_test_wards: Ward IDs to hold out for testing.
        holdout_cal_wards: Ward IDs to hold out for calibration.

    Returns:
        Tuple of (train_df, cal_df, test_df).
    """
    labeled = df[df["service_line_material"].notna()].copy()

    ward_counts = labeled["ward"].value_counts()
    
    if holdout_test_wards is None or holdout_cal_wards is None:
        # Auto-select holdout wards: pick ~15% for test, ~15% for cal
        total = len(labeled)
        selected_test = []
        selected_cal = []
        cum_test = 0
        cum_cal = 0
        
        for ward, count in ward_counts.items():
            if cum_test < total * 0.15:
                selected_test.append(ward)
                cum_test += count
            elif cum_cal < total * 0.15:
                selected_cal.append(ward)
                cum_cal += count
                
        holdout_test_wards = selected_test if selected_test else [int(ward_counts.index[0])]
        holdout_cal_wards = selected_cal if selected_cal else [int(ward_counts.index[1])]

    test = labeled[labeled["ward"].isin(holdout_test_wards)]
    cal = labeled[labeled["ward"].isin(holdout_cal_wards)]
    train = labeled[~labeled["ward"].isin(holdout_test_wards + holdout_cal_wards)]
    
    logger.info(
        "Geographic split: TEST wards %s, CAL wards %s — train=%d, cal=%d, test=%d",
        holdout_test_wards,
        holdout_cal_wards,
        len(train),
        len(cal),
        len(test),
    )
    return train, cal, test



# ---------------------------------------------------------------------------
# Metric computation
# ---------------------------------------------------------------------------


def compute_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float = 0.5,
    split_name: str = "",
) -> dict[str, float]:
    """Compute all evaluation metrics for a binary lead prediction.

    Args:
        y_true: Binary ground-truth labels (1 = Lead).
        y_prob: Predicted probabilities for the positive class.
        threshold: Decision threshold for F2 / precision.
        split_name: Optional prefix for log messages.

    Returns:
        Dictionary of metric name → value.
    """
    y_pred = (y_prob >= threshold).astype(int)

    metrics: dict[str, float] = {}

    # Primary
    if len(np.unique(y_true)) > 1:
        metrics["pr_auc"] = float(average_precision_score(y_true, y_prob))
        metrics["roc_auc"] = float(roc_auc_score(y_true, y_prob))
    else:
        logger.warning("Only one class in y_true; PR-AUC and ROC-AUC set to NaN")
        metrics["pr_auc"] = float("nan")
        metrics["roc_auc"] = float("nan")

    # F2 (β=2, recall-weighted)
    metrics["f2"] = float(fbeta_score(y_true, y_pred, beta=2, zero_division=0))

    # Calibration
    metrics["brier_score"] = float(brier_score_loss(y_true, y_prob))

    if split_name:
        logger.info(
            "[%s] PR-AUC=%.4f  ROC-AUC=%.4f  F2=%.4f  Brier=%.4f",
            split_name,
            metrics["pr_auc"],
            metrics["roc_auc"],
            metrics["f2"],
            metrics["brier_score"],
        )
    return metrics


def check_leakage_gap(
    random_pr_auc: float,
    geo_pr_auc: float,
    max_gap: float = 0.15,
) -> bool:
    """Check if random-split vs geographic-split PR-AUC gap is within bounds.

    A gap > max_gap signals spatial leakage (Architecture §7.6).

    Args:
        random_pr_auc: PR-AUC on random split.
        geo_pr_auc: PR-AUC on geographic holdout.
        max_gap: Maximum allowed relative gap (default 0.15 = 15%).

    Returns:
        True if gap is acceptable, False if leakage is suspected.
    """
    if geo_pr_auc == 0:
        return False
    relative_gap = (random_pr_auc - geo_pr_auc) / geo_pr_auc
    if relative_gap > max_gap:
        logger.error(
            "LEAKAGE SUSPECTED: random-split PR-AUC (%.4f) exceeds geo-split (%.4f) "
            "by %.1f%% > %.0f%% threshold",
            random_pr_auc,
            geo_pr_auc,
            relative_gap * 100,
            max_gap * 100,
        )
        return False
    logger.info(
        "Leakage check PASS: random/geo gap = %.1f%% (threshold %.0f%%)",
        relative_gap * 100,
        max_gap * 100,
    )
    return True


def write_metrics(metrics: dict[str, Any], output_path: Path | str) -> None:
    """Write metrics dictionary to JSON.

    Args:
        metrics: Metrics to write.
        output_path: Destination JSON file path.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w") as f:
        json.dump(metrics, f, indent=2, default=str)
    logger.info("Metrics written to %s", output_path)
