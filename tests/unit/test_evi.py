import numpy as np
from leadguard.models.active_learning import compute_approximate_evi, compute_expected_utility

def test_evi_perfect_certainty():
    """If outcome is already known, EVI -> 0."""
    p_lead = np.array([0.0, 1.0])
    evi = compute_approximate_evi(
        p_lead=p_lead,
        intervention_value=5000.0,
        intervention_cost=2000.0,
        equity_boost=np.zeros(2),
        equity_weight=1000.0
    )
    np.testing.assert_allclose(evi, [0.0, 0.0], atol=1e-7)

def test_evi_zero_downstream_value():
    """If learning the label cannot change any useful decision, EVI = 0."""
    # Case 1: intervention cost is so high that even if P(Lead)=1, EU < 0
    p_lead = np.array([0.5, 0.5])
    evi = compute_approximate_evi(
        p_lead=p_lead,
        intervention_value=1000.0,
        intervention_cost=5000.0, # U_if_lead = 1000 - 5000 < 0
        equity_boost=np.zeros(2),
        equity_weight=1.0
    )
    np.testing.assert_allclose(evi, [0.0, 0.0], atol=1e-7)
    
    # Case 2: intervention value is so high that even if P(Lead)=0, EU > 0
    # Wait, if P(Lead)=0, EU = 0 * V - C + E = -C + E.
    # If equity is extremely high, we might always replace.
    evi2 = compute_approximate_evi(
        p_lead=p_lead,
        intervention_value=1000.0,
        intervention_cost=5000.0,
        equity_boost=np.array([10000.0, 10000.0]), # U_if_not_lead > 0
        equity_weight=1.0
    )
    np.testing.assert_allclose(evi2, [0.0, 0.0], atol=1e-7)

def test_evi_non_negative():
    """EVI should always be >= 0."""
    rng = np.random.default_rng(42)
    p_lead = rng.uniform(0, 1, 1000)
    costs = rng.uniform(500, 5000, 1000)
    values = rng.uniform(500, 10000, 1000)
    equity = rng.uniform(0, 1, 1000)
    
    evi = compute_approximate_evi(
        p_lead=p_lead,
        intervention_value=values,
        intervention_cost=costs,
        equity_boost=equity,
        equity_weight=1000.0
    )
    assert np.all(evi >= -1e-7)

def test_eu_derivatives():
    """EU increases with Value, decreases with Cost."""
    p_lead = np.array([0.5])
    eu1 = compute_expected_utility(p_lead, intervention_value=5000, cost=2000, equity_boost=np.zeros(1))
    eu2 = compute_expected_utility(p_lead, intervention_value=6000, cost=2000, equity_boost=np.zeros(1))
    eu3 = compute_expected_utility(p_lead, intervention_value=5000, cost=3000, equity_boost=np.zeros(1))
    
    assert eu2[0] > eu1[0]
    assert eu3[0] < eu1[0]
