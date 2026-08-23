import pytest
import numpy as np
from itertools import product

# IMPORT ONLY FROM PRODUCTION TO GET THE POLICY BEING TESTED
from leadguard.models.active_learning import compute_approximate_evi

def brute_force_oracle(
    p_lead: np.ndarray,
    inspection_costs: np.ndarray,
    intervention_values: np.ndarray,
    intervention_costs: np.ndarray,
    budget: float,
) -> tuple[float, np.ndarray]:
    """
    Independent O(2^N) oracle that calculates the optimal policy.
    This does NOT use any code from active_learning.py.
    """
    N = len(p_lead)
    best_u = -np.inf
    best_policy = np.zeros(N, dtype=bool)
    
    # 2^N possible policies (1 = inspect, 0 = don't inspect)
    for policy_tuple in product([0, 1], repeat=N):
        policy = np.array(policy_tuple, dtype=bool)
        
        # Total inspection cost of this policy
        total_inspection_cost = np.sum(inspection_costs[policy])
        if total_inspection_cost > budget:
            continue
            
        expected_u = 0.0
        for i in range(N):
            if policy[i]:
                u_if_lead = max(0.0, intervention_values[i] - intervention_costs[i])
                u_if_not_lead = 0.0
                eu = (p_lead[i] * u_if_lead) + ((1.0 - p_lead[i]) * u_if_not_lead) - inspection_costs[i]
                expected_u += eu
            else:
                eu_intervene = p_lead[i] * intervention_values[i] - intervention_costs[i]
                eu = max(0.0, eu_intervene)
                expected_u += eu
                
        if expected_u > best_u:
            best_u = expected_u
            best_policy = policy
            
    return float(best_u), best_policy


def test_1000_random_worlds():
    """Test EVI against 1000 randomized 10-property worlds."""
    np.random.seed(42)
    N = 10
    
    zero_regret_count = 0
    
    for _ in range(1000):
        # Randomize world
        p_lead = np.random.uniform(0.01, 0.99, size=N)
        insp_costs = np.random.uniform(100, 1000, size=N)
        interv_values = np.random.uniform(5000, 15000, size=N)
        interv_costs = np.random.uniform(1000, 5000, size=N)
        budget = np.random.uniform(500, 5000)
        
        # 1. Oracle (Ground truth maximum)
        oracle_u, oracle_policy = brute_force_oracle(
            p_lead, insp_costs, interv_values, interv_costs, budget
        )
        
        # 2. LeadGuard Approximation (using compute_approximate_evi)
        evi = compute_approximate_evi(
            p_lead=p_lead,
            intervention_value=interv_values,
            intervention_cost=interv_costs,
            equity_boost=np.zeros(N),
            equity_weight=0.0
        )
        
        # LeadGuard Policy Optimizer (Greedy Knapsack over Net EVI)
        net_evi = evi - insp_costs
        sorted_idx = np.argsort(net_evi)[::-1]
        
        leadguard_policy = np.zeros(N, dtype=bool)
        spent = 0.0
        
        for idx in sorted_idx:
            if net_evi[idx] <= 0:
                break
            if spent + insp_costs[idx] <= budget:
                leadguard_policy[idx] = True
                spent += insp_costs[idx]
                
        # Calculate utility of leadguard policy
        leadguard_u = 0.0
        for i in range(N):
            if leadguard_policy[i]:
                u_if_lead = max(0.0, interv_values[i] - interv_costs[i])
                u_if_not_lead = 0.0
                eu = (p_lead[i] * u_if_lead) + ((1.0 - p_lead[i]) * u_if_not_lead) - insp_costs[i]
                leadguard_u += eu
            else:
                eu_intervene = p_lead[i] * interv_values[i] - interv_costs[i]
                eu = max(0.0, eu_intervene)
                leadguard_u += eu
                
        regret = oracle_u - leadguard_u
        
        # Knapsack greediness can introduce regret compared to optimal discrete knapsack,
        # but EVI correctly values the information. For this test, if regret is 0, we increment.
        if abs(regret) < 1e-5:
            zero_regret_count += 1
            
    # Allow some regret due to greedy knapsack approximation of discrete knapsack problem,
    # but the vast majority should be optimal.
    assert zero_regret_count > 900, f"Only {zero_regret_count}/1000 worlds had zero regret."


def test_zero_information_invariant():
    """EVI must be 0 if information cannot change the decision (P=0 or P=1)."""
    p_lead = np.array([0.0, 1.0])
    values = np.array([5000.0, 5000.0])
    costs = np.array([1000.0, 1000.0])
    evi = compute_approximate_evi(p_lead, values, costs, np.zeros(2), 0.0)
    
    assert np.allclose(evi, 0.0), f"EVI should be 0 for perfect info, got {evi}"


def test_no_policy_change_invariant():
    """EVI must be 0 if the optimal action is the same regardless of the outcome."""
    # Intervention value = 10000, cost = 0.
    # Since cost is 0, we ALWAYS intervene (EU >= 0 for all P).
    # Since we always intervene even if we know it's not lead (no downside),
    # information has no value in changing our decision.
    p_lead = np.array([0.5, 0.1, 0.9])
    evi = compute_approximate_evi(p_lead, np.array([10000.0, 10000.0, 10000.0]), np.array([0.0, 0.0, 0.0]), np.zeros(3), 0.0)
    assert np.allclose(evi, 0.0), f"EVI should be 0 when decision doesn't change, got {evi}"


def test_negative_evi_bounds():
    """EVI must never be negative."""
    p_lead = np.random.uniform(0, 1, size=100)
    values = np.random.uniform(5000, 15000, size=100)
    costs = np.random.uniform(1000, 5000, size=100)
    
    evi = compute_approximate_evi(p_lead, values, costs, np.zeros(100), 0.0)
    
    assert np.all(evi >= -1e-9), "EVI must be >= 0"


def test_decision_boundary_sensitivity():
    """EVI should peak near the decision boundary, not universally at 0.5."""
    # Decision boundary is where P * Value - Cost = 0 => P = Cost / Value
    values = np.array([10000.0]*99)
    costs = np.array([2000.0]*99)
    # Boundary at P = 0.2
    
    ps = np.linspace(0.01, 0.99, 99)
    evi = compute_approximate_evi(ps, values, costs, np.zeros(99), 0.0)
    
    max_evi_idx = np.argmax(evi)
    max_p = ps[max_evi_idx]
    
    assert 0.15 < max_p < 0.25, f"EVI should peak near boundary P=0.2, got P={max_p}"
