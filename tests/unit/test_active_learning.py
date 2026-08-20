"""Unit tests for active learning loop (Phase 10/11)."""

from __future__ import annotations

import pandas as pd
import numpy as np
import pytest
from leadguard.models.active_learning import compute_priority_score

def test_random_acquisition():
    """Random strategy should ignore predictions and return random score."""
    p_lead = np.array([0.9, 0.1, 0.5])
    uncertainty_score = np.array([0.2, 0.2, 0.8])
    equity_boost = np.array([0.0, 0.0, 0.0])
    
    # We test that random acquisition just returns a random array
    # We can't predict exact scores, but we can verify it doesn't just sort by risk
    np.random.seed(42)
    scores1 = compute_priority_score(p_lead, uncertainty_score, equity_boost, strategy="random")
    np.random.seed(43)
    scores2 = compute_priority_score(p_lead, uncertainty_score, equity_boost, strategy="random")
    
    assert not np.allclose(scores1, scores2), "Random strategy is deterministic when it shouldn't be."
    assert not np.allclose(scores1, p_lead), "Random strategy is just returning risk."

def test_highest_risk_acquisition():
    """Highest risk strategy should return exactly the p_lead array."""
    p_lead = np.array([0.9, 0.1, 0.5])
    uncertainty_score = np.array([0.2, 0.2, 0.8])
    equity_boost = np.array([0.0, 0.0, 0.0])
    
    scores = compute_priority_score(p_lead, uncertainty_score, equity_boost, strategy="highest_risk")
    np.testing.assert_array_equal(scores, p_lead)

def test_highest_uncertainty_acquisition():
    """Highest uncertainty strategy should return exactly the uncertainty array."""
    p_lead = np.array([0.9, 0.1, 0.5])
    uncertainty_score = np.array([0.2, 0.2, 0.8])
    equity_boost = np.array([0.0, 0.0, 0.0])
    
    scores = compute_priority_score(p_lead, uncertainty_score, equity_boost, strategy="highest_uncertainty")
    np.testing.assert_array_equal(scores, uncertainty_score)

def test_risk_and_uncertainty_acquisition():
    """Risk + Uncertainty strategy should combine properly."""
    p_lead = np.array([0.9, 0.1, 0.5])
    uncertainty_score = np.array([0.2, 0.2, 0.8])
    equity_boost = np.array([0.0, 0.0, 0.0])
    
    lambda1 = 0.5
    lambda2 = 0.5
    
    scores = compute_priority_score(
        p_lead, uncertainty_score, equity_boost, strategy="risk_uncertainty",
        lambda1=lambda1, lambda2=lambda2
    )
    expected = (0.5 * p_lead) + (0.5 * uncertainty_score)
    np.testing.assert_array_equal(scores, expected)

def test_risk_uncertainty_equity_acquisition():
    """Full strategy should combine risk, uncertainty, and equity boost."""
    p_lead = np.array([0.9, 0.1, 0.5])
    uncertainty_score = np.array([0.2, 0.2, 0.8])
    equity_boost = np.array([0.5, 0.0, 1.0])
    
    lambda1 = 0.4
    lambda2 = 0.3
    lambda3 = 0.3
    
    scores = compute_priority_score(
        p_lead, uncertainty_score, equity_boost, strategy="risk_uncertainty_equity",
        lambda1=lambda1, lambda2=lambda2, lambda3=lambda3
    )
    expected = (0.4 * p_lead) + (0.3 * uncertainty_score) + (0.3 * equity_boost)
    np.testing.assert_array_equal(scores, expected)

from unittest.mock import patch, MagicMock
from leadguard.models.active_learning import simulate_active_learning

def test_active_learning_rebuilds_features_after_each_round(tmp_path):
    """Verify that build_features is called with the updated labeled set in each round."""
    # Create fake initial data
    np.random.seed(42)
    df = pd.DataFrame({
        "property_id": [f"chi-{i}" for i in range(20)],
        "service_line_material": ["Lead"] * 5 + ["Unknown"] * 15,
        "census_tract": ["17031000100"] * 20
    })
    features_path = tmp_path / "features.parquet"
    df.to_parquet(features_path)

    with patch("leadguard.models.active_learning.build_features") as mock_build_features, \
         patch("leadguard.models.active_learning.xgb.XGBClassifier") as mock_xgb, \
         patch("leadguard.models.active_learning.pd.read_parquet") as mock_read_parquet:
        
        # Mock read_parquet to return our df
        mock_read_parquet.return_value = df
        
        # Mock build_features to just return the subset (it returns features df, we just return dummy)
        mock_build_features.side_effect = lambda df, reference_df, include_label_dependent: pd.DataFrame(np.random.rand(len(df), 5), index=df.index)
        
        # Mock model
        mock_model_instance = MagicMock()
        mock_model_instance.predict_proba.return_value = np.random.rand(15, 2)
        mock_xgb.return_value = mock_model_instance

        # Run simulation for 2 rounds
        simulate_active_learning(
            features_path=str(features_path),
            n_rounds=2,
            batch_size=2,
            strategies=["random"],
            output_path=str(tmp_path / "al_results.csv")
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
        
        # We should see increasing reference sizes: 5, then 7
        assert len(reference_sizes) >= 2
        assert reference_sizes[0] == 5
        assert reference_sizes[1] == 5 + 2

        # Check model fit is called with increasing data sizes
        fit_sizes = []
        for call in mock_model_instance.fit.call_args_list:
            X, y = call.args
            fit_sizes.append(len(X))
        
        assert len(fit_sizes) == 2
        assert fit_sizes[0] == 5
        assert fit_sizes[1] == 7
