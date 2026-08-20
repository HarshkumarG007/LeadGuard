"""Unit tests for active learning simulation (Phase 7)."""

from __future__ import annotations

import numpy as np
import pytest

from leadguard.models.active_learning import compute_priority_score


class TestPriorityScore:
    """Test priority_score formula per Architecture §7.4."""

    def test_formula_correctness(self) -> None:
        """priority = λ1·p_lead + λ2·uncertainty + λ3·equity_boost."""
        p_lead = np.array([0.8])
        uncertainty = np.array([0.3])
        equity_boost = np.array([0.5])
        score = compute_priority_score(p_lead, uncertainty, equity_boost, 0.60, 0.25, 0.15)
        expected = 0.60 * 0.8 + 0.25 * 0.3 + 0.15 * 0.5
        assert score[0] == pytest.approx(expected, abs=1e-6)

    def test_weights_must_sum_to_one(self) -> None:
        """ValueError if weights don't sum to 1."""
        with pytest.raises(ValueError, match="sum to 1"):
            compute_priority_score(
                np.array([0.5]), np.array([0.5]), np.array([0.5]),
                lambda1=0.5, lambda2=0.5, lambda3=0.5,
            )

    def test_output_range(self) -> None:
        """Priority scores must be in [0, 1] when inputs are in [0, 1]."""
        rng = np.random.default_rng(42)
        n = 100
        p_lead = rng.uniform(0, 1, n)
        uncertainty = rng.uniform(0, 1, n)
        equity = rng.uniform(0, 1, n)
        scores = compute_priority_score(p_lead, uncertainty, equity)
        assert (scores >= 0.0).all()
        assert (scores <= 1.0).all()

    def test_high_uncertainty_increases_priority(self) -> None:
        """Higher uncertainty should yield higher priority (all else equal)."""
        base = compute_priority_score(np.array([0.5]), np.array([0.3]), np.array([0.2]))[0]
        high_unc = compute_priority_score(np.array([0.5]), np.array([0.9]), np.array([0.2]))[0]
        assert high_unc > base

    def test_high_equity_boost_increases_priority(self) -> None:
        """Higher equity boost should yield higher priority (all else equal)."""
        base = compute_priority_score(np.array([0.5]), np.array([0.3]), np.array([0.1]))[0]
        high_eq = compute_priority_score(np.array([0.5]), np.array([0.3]), np.array([0.9]))[0]
        assert high_eq > base

    def test_default_weights_match_config(self) -> None:
        """Default weights should match Architecture §7.4: λ1=0.60, λ2=0.25, λ3=0.15."""
        # Default call should not raise (weights sum to 1.0)
        score = compute_priority_score(np.array([0.7]), np.array([0.4]), np.array([0.3]))
        expected = 0.60 * 0.7 + 0.25 * 0.4 + 0.15 * 0.3
        assert score[0] == pytest.approx(expected, abs=1e-6)
