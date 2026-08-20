"""Feature engineering pipeline for LeadGuard.

Reads the interim table, adds H3 spatial indices, distance features,
and leakage-safe spatial-lag features. Outputs features.parquet
matching the Property schema (Architecture §6.2).

LEAKAGE GUARD: neighbor_lead_rate and knn_lead_rate MUST be computed
only from the training partition within each CV fold. This module exposes
a fold-aware wrapper `build_spatial_features_for_fold` for that purpose.

Usage:
    python -m leadguard.data.features
    python -m leadguard.data.features --input data/interim/properties.parquet
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from leadguard.data.validation import FEATURES_SCHEMA
from leadguard.utils.geospatial import (
    add_h3_columns,
    compute_dist_to_nearest,
    compute_knn_lead_rate,
    compute_neighbor_lead_rate_h3,
)
from leadguard.utils.seed import SEED

logger = logging.getLogger(__name__)

# Forbidden demographic columns — double-checked at write time
_FORBIDDEN_COLUMNS = frozenset(
    [
        "income_quartile",
        "median_household_income",
        "race",
        "ethnicity",
        "pct_nonwhite",
        "pct_minority",
    ]
)


def _impute_numerics(df: pd.DataFrame) -> pd.DataFrame:
    """Apply median/mode imputation to numeric feature columns.

    Args:
        df: DataFrame with raw numeric columns.

    Returns:
        DataFrame with imputed values.
    """
    df = df.copy()
    for col in ["year_built", "lot_size_sqft", "building_sqft"]:
        if col in df.columns and df[col].isna().any():
            median_val = df[col].median()
            df[col] = df[col].fillna(median_val)

    if "stories" in df.columns:
        mode_val = df["stories"].mode()
        df["stories"] = df["stories"].fillna(mode_val[0] if len(mode_val) > 0 else 1)

    if "has_basement" in df.columns:
        df["has_basement"] = df["has_basement"].fillna(False).astype(bool)

    if "property_class" in df.columns:
        df["property_class"] = df["property_class"].fillna("Unknown")

    return df


def _load_osm_hydrants(raw_dir: Path) -> pd.DataFrame:
    """Load OSM hydrant points from the raw JSON download.

    Args:
        raw_dir: Raw data directory.

    Returns:
        DataFrame with latitude/longitude columns, or empty DataFrame.
    """
    hydrant_path = raw_dir / "osm_hydrants_chicago.json"
    if not hydrant_path.exists():
        logger.warning(
            "OSM hydrant file not found at %s; dist_to_nearest_hydrant_m will default", hydrant_path
        )
        return pd.DataFrame(columns=["latitude", "longitude"])
    import json

    data = json.loads(hydrant_path.read_text())
    elements = data.get("elements", [])
    rows = [{"latitude": e["lat"], "longitude": e["lon"]} for e in elements if "lat" in e]
    return pd.DataFrame(rows)


def build_base_features(
    df: pd.DataFrame,
    raw_dir: Path = Path("data/raw"),
) -> pd.DataFrame:
    """Add H3 indices and distance features that don't require train/test split.

    Safe to compute on the full dataset — these features don't leak label info.

    Args:
        df: Interim properties DataFrame.
        raw_dir: Raw data directory (for OSM hydrant file).

    Returns:
        DataFrame with H3 and distance columns added.
    """
    df = df.copy()
    df = _impute_numerics(df)

    # H3 spatial indices
    df = add_h3_columns(df)

    # Distance to nearest hydrant
    hydrants = _load_osm_hydrants(raw_dir)
    df = compute_dist_to_nearest(df, hydrants, "dist_to_nearest_hydrant_m", default_m=5000.0)

    # Distance to nearest known-lead property (from all rows — no label leak here)
    known_lead = df[df["service_line_material"] == "Lead"][["latitude", "longitude"]]
    df = compute_dist_to_nearest(df, known_lead, "dist_to_nearest_known_lead_m", default_m=10000.0)

    return df


def build_spatial_features_for_fold(
    df: pd.DataFrame,
    train_indices: pd.Index,
    knn_k: int = 10,
) -> pd.DataFrame:
    """Compute leakage-sensitive spatial-lag features using training rows only.

    This is the core leakage-safe wrapper. It MUST be called with only the
    training indices so that spatial-lag rates don't incorporate held-out rows.

    Args:
        df: Full DataFrame (train + eval rows), already has H3 columns.
        train_indices: pandas Index of training rows (used to define the label distribution).
        knn_k: Number of nearest neighbors for KNN lead rate.

    Returns:
        Full DataFrame with spatial-lag columns computed from training data only.
    """
    train_df = df.loc[train_indices]
    df = compute_neighbor_lead_rate_h3(
        df, train_df, resolution=8, output_col="neighbor_lead_rate_h3res8"
    )
    df = compute_knn_lead_rate(df, train_df, k=knn_k, output_col="knn10_lead_rate")
    return df


def build_features(
    input_path: Path | str = "data/interim/properties.parquet",
    output_path: Path | str = "data/processed/features.parquet",
    raw_dir: Path | str = "data/raw",
    knn_k: int = 10,
    train_fraction: float = 0.70,
    seed: int = SEED,
) -> pd.DataFrame:
    """Full feature engineering pipeline for a single train/test split.

    For use in training. For CV, use build_spatial_features_for_fold within each fold.

    Args:
        input_path: Path to interim parquet.
        output_path: Destination features parquet.
        raw_dir: Raw data directory.
        knn_k: KNN neighbor count for lead-rate feature.
        train_fraction: Fraction used to define training rows for spatial-lag computation.
        seed: Random seed for train/test split.

    Returns:
        Validated features DataFrame.
    """
    input_path = Path(input_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    raw_dir = Path(raw_dir)

    df = pd.read_parquet(input_path)
    logger.info("Loaded %d rows from %s", len(df), input_path)

    # Base features (no leakage risk)
    df = build_base_features(df, raw_dir=raw_dir)

    # Leakage-safe: use only training rows for spatial-lag computation
    rng = np.random.default_rng(seed)
    train_mask = rng.random(len(df)) < train_fraction
    train_indices = df.index[train_mask]
    logger.info(
        "Computing spatial-lag features from %d training rows (%d total)",
        len(train_indices),
        len(df),
    )
    df = build_spatial_features_for_fold(df, train_indices, knn_k=knn_k)

    # Enforce no demographic columns
    forbidden_present = _FORBIDDEN_COLUMNS & set(df.columns)
    if forbidden_present:
        raise ValueError(
            f"DEMOGRAPHIC LEAKAGE: forbidden columns in feature table: {forbidden_present}"
        )

    # Validate against schema
    logger.info("Validating feature schema (%d rows)", len(df))
    validated = FEATURES_SCHEMA.validate(df, lazy=True)

    validated.to_parquet(output_path, index=False)
    logger.info(
        "Features written to %s (%d rows, %d columns)",
        output_path,
        len(validated),
        len(validated.columns),
    )
    return validated


def main():
    import argparse

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description="Build LeadGuard feature table")
    parser.add_argument("--input", default="data/interim/properties.parquet")
    parser.add_argument("--output", default="data/processed/features.parquet")
    parser.add_argument("--raw-dir", default="data/raw")
    args = parser.parse_args()
    build_features(input_path=args.input, output_path=args.output, raw_dir=args.raw_dir)
    print("FEATURES DONE")


if __name__ == "__main__":
    main()
