"""Unit tests for SHAP explainability (Phase 8)."""

from __future__ import annotations

import numpy as np

from leadguard.evaluation.explainability import extract_top_shap_features


class TestExtractTopSHAPFeatures:
    """Tests for per-prediction SHAP extraction."""

    def test_returns_top_n_features(self) -> None:
        """Should return exactly top_n features."""
        shap_row = np.array([0.1, -0.5, 0.3, -0.2, 0.8, 0.05])
        features = [
            "year_built",
            "lot_size",
            "building_sqft",
            "stories",
            "dist_hydrant",
            "knn_rate",
        ]
        result = extract_top_shap_features(shap_row, features, top_n=3)
        assert len(result) == 3

    def test_sorted_by_absolute_value(self) -> None:
        """Features must be sorted by |contribution| descending."""
        shap_row = np.array([0.1, -0.5, 0.3, -0.2, 0.8])
        features = ["a", "b", "c", "d", "e"]
        result = extract_top_shap_features(shap_row, features, top_n=5)
        abs_vals = [abs(r["contribution"]) for r in result]
        assert abs_vals == sorted(abs_vals, reverse=True)

    def test_output_schema(self) -> None:
        """Each result must have 'feature' and 'contribution' keys."""
        shap_row = np.array([0.3, -0.1])
        features = ["x", "y"]
        result = extract_top_shap_features(shap_row, features, top_n=2)
        for item in result:
            assert "feature" in item
            assert "contribution" in item
            assert isinstance(item["feature"], str)
            assert isinstance(item["contribution"], float)

    def test_top_n_greater_than_features(self) -> None:
        """If top_n > len(features), return all features."""
        shap_row = np.array([0.5, -0.3])
        features = ["a", "b"]
        result = extract_top_shap_features(shap_row, features, top_n=10)
        assert len(result) == 2

    def test_default_top_5(self) -> None:
        """Default top_n=5."""
        shap_row = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7])
        features = [f"f{i}" for i in range(7)]
        result = extract_top_shap_features(shap_row, features)
        assert len(result) == 5
