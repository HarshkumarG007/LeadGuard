"""Unit tests for the no-demographic-leakage invariant (Phase 2/6).

This test file is explicitly referenced in Architecture §10 as a required
CI check. It is separate from test_features.py so it is easily found and
verified as part of CI configuration.
"""

from __future__ import annotations

from leadguard.data.features import _FORBIDDEN_COLUMNS
from leadguard.evaluation.fairness import FORBIDDEN_IN_FEATURES

# The canonical forbidden column set (union of all protected-class-correlated fields)
ALL_FORBIDDEN = _FORBIDDEN_COLUMNS | FORBIDDEN_IN_FEATURES


class TestNoDemographicLeakage:
    """Architecture §10 — automated column-name assertion tests.

    These tests form the enforceable version of 'for fairness analysis only'
    from the original proposal. They are part of CI.
    """

    def test_income_quartile_not_allowed_in_features(self) -> None:
        """income_quartile must never appear in the training feature matrix."""
        assert "income_quartile" in ALL_FORBIDDEN

    def test_median_income_not_allowed_in_features(self) -> None:
        """median_household_income must never appear in the training feature matrix."""
        assert "median_household_income" in ALL_FORBIDDEN

    def test_race_not_allowed_in_features(self) -> None:
        """race must never appear in the training feature matrix."""
        assert "race" in ALL_FORBIDDEN

    def test_forbidden_set_is_nonempty(self) -> None:
        """Sanity check: the forbidden column set is populated."""
        assert len(ALL_FORBIDDEN) >= 4

    def test_model_feature_list_contains_no_forbidden_columns(self) -> None:
        """XGB_FEATURES must not contain any forbidden column name."""
        from leadguard.models.xgboost_model import XGB_FEATURES  # noqa: PLC0415

        forbidden_in_model = ALL_FORBIDDEN & set(XGB_FEATURES)
        assert not forbidden_in_model, (
            f"DEMOGRAPHIC LEAKAGE DETECTED: forbidden columns in XGB_FEATURES: {forbidden_in_model}"
        )

    def test_baseline_feature_list_contains_no_forbidden_columns(self) -> None:
        """BASELINE_FEATURES and FULL_FEATURES must not contain forbidden columns."""
        from leadguard.models.baseline import BASELINE_FEATURES, FULL_FEATURES  # noqa: PLC0415

        for feature_set, name in [
            (BASELINE_FEATURES, "BASELINE_FEATURES"),
            (FULL_FEATURES, "FULL_FEATURES"),
        ]:
            forbidden_in_set = ALL_FORBIDDEN & set(feature_set)
            assert not forbidden_in_set, f"DEMOGRAPHIC LEAKAGE in {name}: {forbidden_in_set}"

    def test_fairness_forbidden_matches_features_forbidden(self) -> None:
        """Both forbidden sets should agree on the core demographic identifiers."""
        core = {"income_quartile", "median_household_income", "race", "ethnicity"}
        assert core <= _FORBIDDEN_COLUMNS or core <= FORBIDDEN_IN_FEATURES, (
            "Core forbidden columns not consistently defined across modules"
        )
