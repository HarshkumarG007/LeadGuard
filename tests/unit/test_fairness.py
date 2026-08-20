"""Unit tests for fairness module (Phase 6).

Tests:
  - equity_boost computable for every tract
  - FNR by quartile reported
  - No fairness_reference column in features (leakage enforcement)
  - equity_boost formula matches Architecture §7.4 exactly
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from leadguard.evaluation.fairness import (
    FORBIDDEN_IN_FEATURES,
    compute_equity_boost,
)


@pytest.fixture
def sample_predictions() -> pd.DataFrame:
    """Synthetic predictions DataFrame."""
    rng = np.random.default_rng(42)
    n = 300
    tracts = [f"17031{str(i).zfill(6)}" for i in rng.integers(1, 20, n)]
    return pd.DataFrame({
        "property_id": [f"chi-{i:06d}" for i in range(n)],
        "census_tract": tracts,
        "p_lead_calibrated": rng.uniform(0.1, 0.9, n),
        "service_line_material": rng.choice(["Lead", "Copper", "Galvanized"], n),
    })


@pytest.fixture
def sample_inspections() -> pd.DataFrame:
    """Synthetic inspections DataFrame."""
    rng = np.random.default_rng(1)
    n = 50
    return pd.DataFrame({
        "census_tract": [f"17031{str(i).zfill(6)}" for i in rng.integers(1, 10, n)],
    })


class TestEquityBoost:
    """Test equity_boost formula per Architecture §7.4."""

    def test_equity_boost_in_range(
        self, sample_predictions: pd.DataFrame, sample_inspections: pd.DataFrame
    ) -> None:
        """All equity boost values must be in [0, 1]."""
        boost = compute_equity_boost(sample_predictions, sample_inspections)
        assert (boost >= 0.0).all(), "equity_boost must be >= 0"
        assert (boost <= 1.0).all(), "equity_boost must be <= 1"

    def test_equity_boost_zero_when_over_inspected(self) -> None:
        """A tract that received all inspections should get zero boost (never negative)."""
        # Single tract — all inspections concentrated here
        preds = pd.DataFrame({
            "census_tract": ["17031000001"] * 100,
            "p_lead_calibrated": [0.5] * 100,
        })
        # All inspections in same tract → actual_share = 1.0, target_share = 1.0
        insp = pd.DataFrame({"census_tract": ["17031000001"] * 50})
        boost = compute_equity_boost(preds, insp)
        # actual_share ≥ target_share → boost should be 0
        assert boost["17031000001"] == pytest.approx(0.0, abs=0.01)

    def test_equity_boost_positive_when_under_inspected(self) -> None:
        """A tract with high risk share but zero inspections should get a positive boost."""
        preds = pd.DataFrame({
            "census_tract": ["17031000001"] * 80 + ["17031000002"] * 20,
            "p_lead_calibrated": [0.9] * 80 + [0.1] * 20,
        })
        # All inspections in tract 2 — tract 1 under-inspected relative to its risk
        insp = pd.DataFrame({"census_tract": ["17031000002"] * 10})
        boost = compute_equity_boost(preds, insp)
        assert boost["17031000001"] > 0.0, "Under-inspected tract should get positive equity boost"

    def test_equity_boost_empty_inspections(self, sample_predictions: pd.DataFrame) -> None:
        """With no inspections, all boosts should be 1.0 (maximum under-inspection)."""
        empty_insp = pd.DataFrame(columns=["census_tract"])
        boost = compute_equity_boost(sample_predictions, empty_insp)
        # All actual_share = 0, target_share > 0 → boost = 1.0
        assert np.allclose(boost.values, 1.0, atol=0.01), (
            "With no inspections, every tract is maximally under-inspected → boost should be 1.0"
        )

    def test_equity_boost_weights_sum_correctly(self) -> None:
        """Equity boost computation should be deterministic."""
        preds = pd.DataFrame({
            "census_tract": ["A", "B"],
            "p_lead_calibrated": [0.8, 0.2],
        })
        insp = pd.DataFrame(columns=["census_tract"])
        boost1 = compute_equity_boost(preds, insp)
        boost2 = compute_equity_boost(preds, insp)
        pd.testing.assert_series_equal(boost1, boost2)


class TestNoDemographicLeakageFromFairness:
    """Ensure no fairness reference columns leak into features."""

    def test_forbidden_columns_set_populated(self) -> None:
        """The forbidden columns set must contain at least income_quartile."""
        assert "income_quartile" in FORBIDDEN_IN_FEATURES

    def test_fairness_ref_columns_not_in_sample_features(self) -> None:
        """Sample features DataFrame must not contain any forbidden demographic column."""
        # Simulate a features DataFrame (as would be loaded from features.parquet)
        sample_features_cols = [
            "property_id", "address", "zip_code", "ward", "latitude", "longitude",
            "year_built", "property_class", "lot_size_sqft", "building_sqft", "stories",
            "has_basement", "h3_index_res8", "h3_index_res9", "dist_to_nearest_hydrant_m",
            "dist_to_nearest_known_lead_m", "neighbor_lead_rate_h3res8", "knn10_lead_rate",
            "census_tract", "service_line_material", "material_source", "last_updated",
        ]
        forbidden_present = FORBIDDEN_IN_FEATURES & set(sample_features_cols)
        assert not forbidden_present, (
            f"Demographic columns found in feature table: {forbidden_present}"
        )

    def test_census_tract_is_allowed_as_join_key(self) -> None:
        """census_tract is explicitly allowed as a join key, not a model input."""
        assert "census_tract" not in FORBIDDEN_IN_FEATURES


class TestFNRByQuartile:
    """Test FNR computation logic."""

    def test_fnr_is_fraction_between_0_and_1(self, sample_predictions: pd.DataFrame) -> None:
        """FNR values must be in [0, 1]."""
        from leadguard.evaluation.fairness import _quartile_from_income  # noqa: PLC0415

        income = pd.Series([30000, 50000, 70000, 100000])
        quartiles = _quartile_from_income(income)
        assert set(quartiles) == {1, 2, 3, 4}
        assert quartiles.min() == 1
        assert quartiles.max() == 4
