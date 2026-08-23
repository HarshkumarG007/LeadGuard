"""Unit tests for active learning loop (Phase 10/11)."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd

from leadguard.models.active_learning import compute_priority_score, simulate_active_learning


# Obsolete heuristic scoring tests removed (LeadGuard 1.0 transitioned to EVI)


def test_active_learning_rebuilds_features_after_each_round(tmp_path):
    """Verify that build_features is called with the updated labeled set in each round."""
    # Create fake initial data
    np.random.seed(42)
    df = pd.DataFrame(
        {
            "property_id": [f"id-{i}" for i in range(20)],
            "service_line_material": ["Lead"] * 5 + ["Unknown"] * 15,
            "inspected_at": ["2020-01-01"] * 5 + [None] * 15,
            "census_tract": ["17031010100"] * 20,
            "year_built": [1950] * 20,
            "lot_size_sqft": [5000] * 20,
            "building_sqft": [2000] * 20,
            "stories": [2] * 20,
            "has_basement": [1] * 20,
            "dist_to_nearest_hydrant_m": [50] * 20,
            "dist_to_nearest_known_lead_m": [100] * 20,
            "neighbor_lead_rate_h3res8": [0.5] * 20,
            "knn10_lead_rate": [0.5] * 20,
            "known_lead_rate_in_ward": [0.5] * 20,
        }
    )
    features_path = tmp_path / "features.parquet"
    df.to_parquet(features_path)

    with (
        patch("leadguard.models.active_learning.build_features") as mock_build_features,
        patch("leadguard.models.active_learning.xgb.XGBClassifier") as mock_xgb,
        patch("leadguard.models.active_learning.pd.read_parquet") as mock_read_parquet,
    ):
        # Mock read_parquet to return our df
        mock_read_parquet.return_value = df

        # Mock build_features to just return the subset (it returns features df, we just return dummy)
        mock_build_features.side_effect = lambda df, reference_df, include_label_dependent: (
            df.copy()
        )

        # Mock model
        mock_model_instance = MagicMock()
        mock_model_instance.predict_proba.side_effect = lambda X: np.random.rand(len(X), 2)
        mock_xgb.return_value = mock_model_instance

        with patch("leadguard.models.active_learning._load_scoring_config") as mock_cfg:
            mock_cfg.return_value = {"active_learning": {"n_rounds": 2, "batch_size": 2}}
            # Run simulation for 2 rounds
            simulate_active_learning(
                features_path=str(features_path),
                fairness_ref_path=str(tmp_path / "fairness_ref.parquet"),
                output_path=str(tmp_path / "al_results.csv"),
            )

        # In round 0, it calls build_features for training set (5 items) and pool (15 items)
        # In round 1, it calls build_features with updated labeled set (7 items) and pool (13 items)
        # So reference_df size should increase
        reference_sizes = []
        for call in mock_build_features.call_args_list:
            kwargs = call.kwargs
            reference_df = kwargs.get("reference_df")
            if reference_df is not None:
                reference_sizes.append(len(reference_df))

        # We should see increasing reference sizes: 1, then 3, then 5 (2 rounds)
        assert len(reference_sizes) >= 4
        assert reference_sizes[0] == 1
        assert reference_sizes[-1] == 5

        # Check model fit is called with increasing data sizes
        fit_sizes = []
        for call in mock_model_instance.fit.call_args_list:
            X, y = call.args
            fit_sizes.append(len(X))

        assert len(fit_sizes) >= 3
        assert fit_sizes[0] == 1
        assert fit_sizes[-1] == 5
