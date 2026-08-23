import numpy as np
import itertools
from leadguard.models.active_learning import compute_approximate_evi, compute_expected_utility

def brute_force_optimal_inspections(p_lead, intervention_value, intervention_cost, inspection_cost, budget, equity_boost, equity_weight):
    """
    Oracle: Find the inspection subset that maximizes expected utility.
    We iterate over all possible subsets of properties to inspect.
    """
    n = len(p_lead)
    best_subset = tuple()
    best_eu = -np.inf
    
    # Pre-calculate base utility without inspections
    # U_base = sum( max(0, EU(p_i)) )
    eu_current_all = compute_expected_utility(p_lead, intervention_value, intervention_cost, equity_boost, equity_weight)
    
    for k in range(n + 1):
        for subset in itertools.combinations(range(n), k):
            cost_of_inspections = k * inspection_cost
            if cost_of_inspections > budget:
                continue
                
            # For each property in the subset, we observe Y
            # Expected utility for inspected properties:
            # P_i * max(0, EU(1)) + (1-P_i) * max(0, EU(0))
            expected_u = 0.0
            
            for i in range(n):
                if i in subset:
                    # Inspected
                    eu_if_lead = compute_expected_utility(np.array([1.0]), intervention_value[i], intervention_cost[i], np.array([equity_boost[i]]), equity_weight)[0]
                    eu_if_not_lead = compute_expected_utility(np.array([0.0]), intervention_value[i], intervention_cost[i], np.array([equity_boost[i]]), equity_weight)[0]
                    u_i = p_lead[i] * max(0, eu_if_lead) + (1 - p_lead[i]) * max(0, eu_if_not_lead)
                else:
                    # Not inspected
                    u_i = max(0, eu_current_all[i])
                    
                expected_u += u_i
                
            # Subtract inspection cost
            expected_u -= cost_of_inspections
            
            if expected_u > best_eu:
                best_eu = expected_u
                best_subset = subset
                
    return best_subset, best_eu

def test_synthetic_evi_oracle_regret():
    """Killer validation experiment: Compare Approximate EVI vs Oracle on a small world."""
    n_props = 10
    rng = np.random.default_rng(42)
    
    # Synthetic world
    p_lead = rng.uniform(0.1, 0.9, n_props)
    intervention_value = rng.uniform(2000, 8000, n_props)
    intervention_cost = np.full(n_props, 4000.0)
    inspection_cost = 500.0
    budget = 1500.0  # Can inspect 3 properties
    equity_boost = np.zeros(n_props)
    equity_weight = 1.0
    
    # 1. Oracle exact EVI subset selection
    oracle_subset, oracle_eu = brute_force_optimal_inspections(
        p_lead, intervention_value, intervention_cost, inspection_cost, budget, equity_boost, equity_weight
    )
    
    # 2. Approximate EVI greedy selection
    approx_evi = compute_approximate_evi(
        p_lead, intervention_value, intervention_cost, equity_boost, equity_weight
    )
    
    # We greedily select properties with the highest Approximate EVI that fit in budget
    # EVI is marginal value. We subtract inspection cost to get net value
    net_evi = approx_evi - inspection_cost
    
    # Sort by net EVI descending
    ranked_indices = np.argsort(net_evi)[::-1]
    
    greedy_subset = []
    spent = 0.0
    for idx in ranked_indices:
        if net_evi[idx] > 0 and spent + inspection_cost <= budget:
            greedy_subset.append(idx)
            spent += inspection_cost
            
    # Compute the expected utility achieved by the greedy subset
    _, greedy_eu = brute_force_optimal_inspections(
        p_lead, intervention_value, intervention_cost, inspection_cost, budget, equity_boost, equity_weight
    )
    
    # Wait, the greedy subset EU can be evaluated exactly
    # Let's write a small helper to evaluate a specific subset
    def evaluate_subset(subset):
        expected_u = 0.0
        eu_current_all = compute_expected_utility(p_lead, intervention_value, intervention_cost, equity_boost, equity_weight)
        for i in range(n_props):
            if i in subset:
                eu_if_lead = compute_expected_utility(np.array([1.0]), intervention_value[i], intervention_cost[i], np.array([equity_boost[i]]), equity_weight)[0]
                eu_if_not_lead = compute_expected_utility(np.array([0.0]), intervention_value[i], intervention_cost[i], np.array([equity_boost[i]]), equity_weight)[0]
                u_i = p_lead[i] * max(0, eu_if_lead) + (1 - p_lead[i]) * max(0, eu_if_not_lead)
            else:
                u_i = max(0, eu_current_all[i])
            expected_u += u_i
        return expected_u - (len(subset) * inspection_cost)

    greedy_eu_actual = evaluate_subset(greedy_subset)
    
    regret = oracle_eu - greedy_eu_actual
    
    print(f"\nSynthetic World EVI Evaluation")
    print(f"Oracle Subset: {oracle_subset}, EU: {oracle_eu:.2f}")
    print(f"Greedy Subset: {tuple(greedy_subset)}, EU: {greedy_eu_actual:.2f}")
    print(f"Regret: {regret:.2f}")
    
    # For a good heuristic, regret should be low (e.g. within 5% of optimal)
    # The Approximate EVI is actually EXACT for independent properties, so Regret should be EXACTLY 0
    # when properties are independent and costs are constant (no knapsack packing issues).
    np.testing.assert_allclose(regret, 0.0, atol=1e-5)
