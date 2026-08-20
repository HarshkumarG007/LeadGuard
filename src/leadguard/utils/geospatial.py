"""Geospatial utilities for LeadGuard.

Provides H3 indexing, distance computations, and spatial-lag helpers.
All functions are pure (no side effects) to allow safe use in CV folds.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# H3 library — try importing; version 3.x and 4.x have different APIs
try:
    import h3  # type: ignore

    H3_AVAILABLE = True
    # Detect API version
    _H3_NEW_API = hasattr(h3, "latlng_to_cell")  # h3 >= 4.0
except ImportError:
    H3_AVAILABLE = False
    logger.warning("h3 not available; H3 features will be zero-filled")


def latlng_to_h3(lat: float, lng: float, resolution: int) -> str:
    """Convert a lat/lng coordinate to an H3 cell index.

    Args:
        lat: Latitude in decimal degrees.
        lng: Longitude in decimal degrees.
        resolution: H3 resolution (8 or 9 for LeadGuard).

    Returns:
        H3 cell index string, or "0" if h3 is not available.
    """
    if not H3_AVAILABLE:
        return "0"
    if _H3_NEW_API:
        return h3.latlng_to_cell(lat, lng, resolution)
    else:
        return h3.geo_to_h3(lat, lng, resolution)  # type: ignore[attr-defined]


def add_h3_columns(
    df: pd.DataFrame, res8_col: str = "h3_index_res8", res9_col: str = "h3_index_res9"
) -> pd.DataFrame:
    """Add H3 index columns at resolutions 8 and 9 to a DataFrame.

    Args:
        df: DataFrame containing ``latitude`` and ``longitude`` columns.
        res8_col: Output column name for resolution-8 index.
        res9_col: Output column name for resolution-9 index.

    Returns:
        DataFrame with two new H3 index columns appended (string dtype).
    """
    df = df.copy()
    df[res8_col] = [latlng_to_h3(lat, lng, 8) for lat, lng in zip(df["latitude"], df["longitude"])]
    df[res9_col] = [latlng_to_h3(lat, lng, 9) for lat, lng in zip(df["latitude"], df["longitude"])]
    return df


def haversine_distance_m(
    lat1: np.ndarray,
    lon1: np.ndarray,
    lat2: float,
    lon2: float,
) -> np.ndarray:
    """Compute Haversine distance in metres between arrays of points and a reference point.

    Args:
        lat1: Array of latitudes (decimal degrees).
        lon1: Array of longitudes (decimal degrees).
        lat2: Reference point latitude.
        lon2: Reference point longitude.

    Returns:
        Array of distances in metres.
    """
    R = 6_371_000.0  # Earth radius in metres
    phi1 = np.radians(lat1)
    phi2 = np.radians(lat2)
    d_phi = np.radians(lat2 - lat1)
    d_lam = np.radians(lon2 - lon1)
    a = np.sin(d_phi / 2) ** 2 + np.cos(phi1) * np.cos(phi2) * np.sin(d_lam / 2) ** 2
    return R * 2 * np.arcsin(np.sqrt(a))


def compute_dist_to_nearest(
    df: pd.DataFrame,
    reference_df: pd.DataFrame,
    output_col: str,
    default_m: float = 5000.0,
) -> pd.DataFrame:
    """Add a column with distance-to-nearest-point in ``reference_df``.

    Args:
        df: Properties DataFrame with ``latitude`` and ``longitude``.
        reference_df: Reference points with ``latitude`` and ``longitude``.
        output_col: Name of the new distance column (metres).
        default_m: Default distance when ``reference_df`` is empty.

    Returns:
        Copy of ``df`` with ``output_col`` appended.
    """
    df = df.copy()
    if reference_df.empty:
        logger.warning(
            "Reference DataFrame is empty; defaulting %s to %.0f m", output_col, default_m
        )
        df[output_col] = default_m
        return df

    ref_lats = reference_df["latitude"].values
    ref_lons = reference_df["longitude"].values

    distances = []
    for lat, lon in zip(df["latitude"], df["longitude"]):
        dists = haversine_distance_m(ref_lats, ref_lons, lat, lon)
        distances.append(float(dists.min()))
    df[output_col] = distances
    return df


def compute_neighbor_lead_rate_h3(
    df: pd.DataFrame,
    train_df: pd.DataFrame,
    resolution: int = 8,
    h3_col: str | None = None,
    output_col: str = "neighbor_lead_rate_h3res8",
) -> pd.DataFrame:
    """Compute the fraction of Lead service lines in each H3 cell.

    LEAKAGE GUARD: ``train_df`` must be the training partition only.
    Never pass the full dataset — this would cause data leakage in CV folds.

    Args:
        df: All properties to annotate.
        train_df: Training-only rows whose label distribution defines the rate.
        resolution: H3 resolution (default 8).
        h3_col: Column name for H3 index. Inferred from resolution if None.
        output_col: Name of the output rate column.

    Returns:
        Copy of ``df`` with ``output_col`` appended.
    """
    if h3_col is None:
        h3_col = f"h3_index_res{resolution}"

    df = df.copy()

    # Compute per-cell lead rate from training partition only
    labeled_train = train_df[train_df["service_line_material"].notna()].copy()
    labeled_train["_is_lead"] = (labeled_train["service_line_material"] == "Lead").astype(float)

    cell_rates = labeled_train.groupby(h3_col)["_is_lead"].mean().to_dict()
    global_rate = labeled_train["_is_lead"].mean() if len(labeled_train) > 0 else 0.0

    df[output_col] = df[h3_col].map(cell_rates).fillna(global_rate).clip(0.0, 1.0)
    return df


def compute_knn_lead_rate(
    df: pd.DataFrame,
    train_df: pd.DataFrame,
    k: int = 10,
    output_col: str = "knn10_lead_rate",
) -> pd.DataFrame:
    """Compute the fraction of Lead among the K nearest training neighbours.

    LEAKAGE GUARD: ``train_df`` must be the training partition only.

    Args:
        df: Properties to annotate.
        train_df: Training-only rows for KNN lookup.
        k: Number of nearest neighbours.
        output_col: Output column name.

    Returns:
        Copy of ``df`` with ``output_col`` appended.
    """
    from sklearn.neighbors import BallTree  # noqa: PLC0415

    df = df.copy()
    labeled_train = train_df[train_df["service_line_material"].notna()].copy()

    if len(labeled_train) < k:
        logger.warning("Fewer than k=%d labeled training rows; defaulting %s to 0", k, output_col)
        df[output_col] = 0.0
        return df

    is_lead = (labeled_train["service_line_material"] == "Lead").astype(float).values
    train_coords = np.radians(labeled_train[["latitude", "longitude"]].values)
    query_coords = np.radians(df[["latitude", "longitude"]].values)

    tree = BallTree(train_coords, metric="haversine")
    k_actual = min(k, len(labeled_train))
    indices = tree.query(query_coords, k=k_actual, return_distance=False)

    df[output_col] = [is_lead[idx].mean() for idx in indices]
    return df
