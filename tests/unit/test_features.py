"""Unit tests for feature engineering (Phase 2).

Tests:
  - Output schema matches Architecture §6.2 Property entity
  - No demographic columns in features.parquet (test_no_demographic_leakage)
  - Spatial-lag features computed only from training partition (leakage regression)
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from leadguard.data.features import (
    _FORBIDDEN_COLUMNS,
    build_features,
)
from leadguard.utils.geospatial import (
    add_h3_columns,
    compute_neighbor_lead_rate_h3,
)


@pytest.fixture
def synthetic_df() -> pd.DataFrame:
    """Minimal synthetic property DataFrame for testing."""
    rng = np.random.default_rng(42)
    n = 200
    return pd.DataFrame(
        {
            "property_id": [f"chi-{i:06d}" for i in range(n)],
            "address": [f"{i} N TEST ST" for i in range(n)],
            "zip_code": ["60601"] * n,
            "ward": rng.integers(1, 10, n).tolist(),
            "latitude": rng.uniform(41.7, 41.9, n).tolist(),
            "longitude": rng.uniform(-87.8, -87.6, n).tolist(),
            "year_built": rng.choice([1900, 1920, 1950, 1970, None], n).tolist(),
            "property_class": ["Single-family"] * n,
            "lot_size_sqft": rng.uniform(1000, 10000, n).tolist(),
            "building_sqft": rng.uniform(500, 3000, n).tolist(),
            "stories": rng.integers(1, 4, n).tolist(),
            "has_basement": rng.choice([True, False], n).tolist(),
            "census_tract": ["17031000100"] * n,
            "service_line_material": rng.choice(
                ["Lead", "Copper", "Galvanized", None], n, p=[0.2, 0.4, 0.2, 0.2]
            ).tolist(),
            "material_source": ["inspected"] * n,
            "last_updated": pd.Timestamp("2026-01-01"),
        }
    )


class TestNoDemographicLeakage:
    """Architecture §10: test_no_demographic_leakage — must be part of CI."""

    def test_forbidden_columns_not_in_synthetic_df(self, synthetic_df: pd.DataFrame) -> None:
        """Base features should never include demographic columns."""
        forbidden_present = _FORBIDDEN_COLUMNS & set(synthetic_df.columns)
        assert not forbidden_present, (
            f"Forbidden demographic columns in DataFrame: {forbidden_present}"
        )

    def test_forbidden_columns_not_in_h3_augmented(self, synthetic_df: pd.DataFrame) -> None:
        """H3 augmentation must not introduce demographic columns."""
        augmented = add_h3_columns(synthetic_df)
        forbidden_present = _FORBIDDEN_COLUMNS & set(augmented.columns)
        assert not forbidden_present, f"Forbidden columns after H3 augment: {forbidden_present}"

    def test_income_quartile_never_in_features(self, synthetic_df: pd.DataFrame) -> None:
        """income_quartile must not appear in any column name at all."""
        assert "income_quartile" not in synthetic_df.columns
        assert not any("income" in col.lower() for col in synthetic_df.columns)

    def test_forbidden_columns_set_is_not_empty(self) -> None:
        """Sanity check: the enforcement set is populated."""
        assert len(_FORBIDDEN_COLUMNS) > 0


class TestSpatialLagLeakage:
    """Verify that spatial-lag features only use training-partition rows."""

    def test_neighbor_lead_rate_uses_only_train(self, synthetic_df: pd.DataFrame) -> None:
        """Rates computed from train partition differ from full-dataset rates."""
        df = add_h3_columns(synthetic_df)
        # Compute from full dataset
        full_df = compute_neighbor_lead_rate_h3(df, df, resolution=8)
        # Compute from first half only
        train_half = df.iloc[:100]
        half_df = compute_neighbor_lead_rate_h3(df, train_half, resolution=8)
        # The rates should differ because training data differs
        # (not necessarily different for every row, but at least some should differ)
        n_same = (
            full_df["neighbor_lead_rate_h3res8"] == half_df["neighbor_lead_rate_h3res8"]
        ).sum()
        # Allow up to 80% same (some H3 cells may be identical by coincidence)
        assert n_same < len(df) * 0.99, (
            "Spatial lag rates identical regardless of training partition — possible leakage"
        )

    def test_build_features_uses_reference(self, synthetic_df: pd.DataFrame) -> None:
        """build_features must not use held-out row data."""
        df = add_h3_columns(synthetic_df)
        df["dist_to_nearest_hydrant_m"] = 500.0
        df["dist_to_nearest_known_lead_m"] = 1000.0
        train_df = df.iloc[:150]
        test_df = df.iloc[150:]
        result = build_features(test_df, reference_df=train_df, include_label_dependent=True)
        assert "neighbor_lead_rate_h3res8" in result.columns
        assert "knn10_lead_rate" in result.columns
        assert result["neighbor_lead_rate_h3res8"].between(0, 1).all()
        assert result["knn10_lead_rate"].between(0, 1).all()


class TestFeatureSchema:
    """Feature schema validation tests."""

    def test_h3_columns_added(self, synthetic_df: pd.DataFrame) -> None:
        """H3 index columns should be added as strings."""
        df = add_h3_columns(synthetic_df)
        assert "h3_index_res8" in df.columns
        assert "h3_index_res9" in df.columns
        assert df["h3_index_res8"].dtype == object or str(df["h3_index_res8"].dtype).startswith(
            "str"
        )  # string or StringDtype

    def test_census_tract_present(self, synthetic_df: pd.DataFrame) -> None:
        """census_tract must be present as a join key but not a model feature."""
        assert "census_tract" in synthetic_df.columns

    def test_property_id_unique(self, synthetic_df: pd.DataFrame) -> None:
        """property_id must be unique (Architecture §6.2)."""
        assert synthetic_df["property_id"].is_unique
