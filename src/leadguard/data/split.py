"""Centralized split engine for LeadGuard data.

Implements geographic, temporal, and spatial-temporal holdouts
with strict invariants protecting against label leakage.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from leadguard.utils.seed import SEED

logger = logging.getLogger(__name__)


@dataclass
class SplitResult:
    """Standardized split output with metadata."""
    train: pd.DataFrame
    calibration: pd.DataFrame
    test: pd.DataFrame
    metadata: dict[str, Any]


def split_dataset(
    df: pd.DataFrame,
    mode: str = "geographic",
    cutoff_date: str | None = None,
    test_start_date: str | None = None,
    test_end_date: str | None = None,
    holdout_wards: list[int] | None = None,
    seed: int = SEED,
    min_test_rows: int | None = None,
) -> SplitResult:
    """Split the dataset according to the specified mode.

    Modes:
      - 'geographic': Random wards are held out for cal/test.
      - 'temporal': Train on past, cal/test on future.
      - 'spatial-temporal': Train on past known wards, cal/test on future unseen wards.
      - 'random': Pure random split (not recommended for spatial data).

    Args:
        df: Input DataFrame with labels.
        mode: Split mode.
        cutoff_date: ISO date string for temporal splits (e.g., '2024-01-01').
        holdout_wards: Specific wards to hold out. If None, chosen pseudo-randomly.
        seed: Random seed.
        min_test_rows: Minimum required rows for test set. If test set is smaller, 
            a warning is logged and the experiment status is set to INSUFFICIENT_SAMPLE.

    Returns:
        SplitResult containing train, cal, test DataFrames and metadata.
    """
    if mode not in ("geographic", "temporal", "spatial-temporal", "random"):
        raise ValueError(f"Unknown split mode: {mode}")

    labeled = df[df["service_line_material"].notna()].copy()
    if len(labeled) == 0:
        raise ValueError("No labeled data found for splitting")

    metadata: dict[str, Any] = {
        "split_mode": mode,
        "seed": seed,
        "total_labeled_rows": len(labeled),
    }

    if mode == "random":
        labeled["_is_lead"] = (labeled["service_line_material"] == "Lead").astype(int)
        train_cal, test = train_test_split(labeled, test_size=0.15, random_state=seed, stratify=labeled["_is_lead"])
        train, cal = train_test_split(train_cal, test_size=0.15/0.85, random_state=seed, stratify=train_cal["_is_lead"])

        train = train.drop(columns=["_is_lead"])
        cal = cal.drop(columns=["_is_lead"])
        test = test.drop(columns=["_is_lead"])

    elif mode == "geographic":
        if holdout_wards is None:
            rng = np.random.default_rng(seed)
            wards = sorted(labeled["ward"].unique())
            n_holdout = max(1, int(len(wards) * 0.25))
            holdout_wards = list(rng.choice(wards, size=n_holdout, replace=False))
            # further split holdout into cal and test
            n_cal = max(1, len(holdout_wards) // 2)
            cal_wards = holdout_wards[:n_cal]
            test_wards = holdout_wards[n_cal:]
        else:
            n_cal = max(1, len(holdout_wards) // 2)
            cal_wards = holdout_wards[:n_cal]
            test_wards = holdout_wards[n_cal:]

        metadata["cal_wards"] = [int(w) for w in cal_wards]
        metadata["test_wards"] = [int(w) for w in test_wards]

        train = labeled[~labeled["ward"].isin(holdout_wards)].copy()
        cal = labeled[labeled["ward"].isin(cal_wards)].copy()
        test = labeled[labeled["ward"].isin(test_wards)].copy()

    elif mode == "temporal":
        if cutoff_date is None:
            raise ValueError("cutoff_date required for temporal split")

        cutoff = pd.Timestamp(cutoff_date)
        metadata["cutoff_date"] = cutoff_date

        if "inspected_at" not in labeled.columns:
            raise ValueError("inspected_at column missing for temporal split")

        train = labeled[labeled["inspected_at"] < cutoff].copy()
        future = labeled[labeled["inspected_at"] >= cutoff].copy()

        if len(future) == 0:
            raise ValueError(f"No future data found after {cutoff_date}")

        if test_start_date and test_end_date:
            t_start = pd.Timestamp(test_start_date)
            t_end = pd.Timestamp(test_end_date)
            cal = future[future["inspected_at"] < t_start].copy()
            test = future[(future["inspected_at"] >= t_start) & (future["inspected_at"] < t_end)].copy()
        else:
            # Fallback to naive halving
            future = future.sort_values("inspected_at")
            mid_idx = len(future) // 2
            cal = future.iloc[:mid_idx].copy()
            test = future.iloc[mid_idx:].copy()

        metadata["train_max_inspected_at"] = str(train["inspected_at"].max()) if len(train) else None
        metadata["cal_max_inspected_at"] = str(cal["inspected_at"].max()) if len(cal) else None
        metadata["test_min_inspected_at"] = str(test["inspected_at"].min()) if len(test) else None

    elif mode == "spatial-temporal":
        if cutoff_date is None:
            raise ValueError("cutoff_date required for spatial-temporal split")

        cutoff = pd.Timestamp(cutoff_date)
        metadata["cutoff_date"] = cutoff_date

        if holdout_wards is None:
            rng = np.random.default_rng(seed)
            wards = sorted(labeled["ward"].unique())
            n_holdout = max(1, int(len(wards) * 0.25))
            holdout_wards = list(rng.choice(wards, size=n_holdout, replace=False))
            n_cal = max(1, len(holdout_wards) // 2)
            cal_wards = holdout_wards[:n_cal]
            test_wards = holdout_wards[n_cal:]
        else:
            n_cal = max(1, len(holdout_wards) // 2)
            cal_wards = holdout_wards[:n_cal]
            test_wards = holdout_wards[n_cal:]

        metadata["cal_wards"] = [int(w) for w in cal_wards]
        metadata["test_wards"] = [int(w) for w in test_wards]

        # Train on past known wards
        train = labeled[(~labeled["ward"].isin(holdout_wards)) & (labeled["inspected_at"] < cutoff)].copy()

        # Cal on future cal unseen wards
        cal = labeled[(labeled["ward"].isin(cal_wards)) & (labeled["inspected_at"] >= cutoff)].copy()

        # Test on future test unseen wards
        test = labeled[(labeled["ward"].isin(test_wards)) & (labeled["inspected_at"] >= cutoff)].copy()

        metadata["train_max_inspected_at"] = str(train["inspected_at"].max()) if len(train) else None
        metadata["test_min_inspected_at"] = str(test["inspected_at"].min()) if len(test) else None

    metadata["train_rows"] = len(train)
    metadata["calibration_rows"] = len(cal)
    metadata["test_rows"] = len(test)

    # Automated invariants
    if mode in ("temporal", "spatial-temporal"):
        if train["inspected_at"].max() >= pd.Timestamp(cutoff_date):
            raise ValueError("Temporal leakage: Train contains future data")
        if len(cal) > 0 and cal["inspected_at"].min() < pd.Timestamp(cutoff_date):
            raise ValueError("Temporal leakage: Cal contains past data")
        if len(test) > 0 and test["inspected_at"].min() < pd.Timestamp(cutoff_date):
            raise ValueError("Temporal leakage: Test contains past data")

    if mode in ("geographic", "spatial-temporal"):
        train_wards = set(train["ward"].unique())
        test_wards = set(test["ward"].unique())
        if len(test_wards) > 0 and train_wards.intersection(test_wards):
            raise ValueError("Spatial leakage: Overlapping wards between train and test")

    if min_test_rows is not None and len(test) < min_test_rows:
        logger.warning(
            "Test set size %d is smaller than min_test_rows %d. Marking INSUFFICIENT_SAMPLE.",
            len(test),
            min_test_rows,
        )
        metadata["STATUS"] = "INSUFFICIENT_SAMPLE"
    else:
        metadata["STATUS"] = "OK"

    return SplitResult(
        train=train,
        calibration=cal,
        test=test,
        metadata=metadata
    )
