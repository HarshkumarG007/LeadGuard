import numpy as np
import pandas as pd
import json
import time
from itertools import product
from pathlib import Path
from leadguard.models.active_learning import compute_approximate_evi

def calc_utility(policy, p_lead, insp_costs, interv_values, interv_costs):
    N = len(p_lead)
    u = 0.0
    for j in range(N):
        if policy[j]:
            u_if_lead = max(0.0, interv_values[j] - interv_costs[j])
            eu = (p_lead[j] * u_if_lead) - insp_costs[j]
            u += eu
        else:
            eu_intervene = p_lead[j] * interv_values[j] - interv_costs[j]
            u += max(0.0, eu_intervene)
    return u

def brute_force_oracle(p_lead, insp_costs, interv_values, interv_costs, budget):
    N = len(p_lead)
    best_u = -np.inf
    best_policy = np.zeros(N, dtype=bool)
    evals = 0
    
    t0 = time.time()
    for policy_tuple in product([0, 1], repeat=N):
        policy = np.array(policy_tuple, dtype=bool)
        if np.sum(insp_costs[policy]) <= budget:
            evals += 1
            expected_u = calc_utility(policy, p_lead, insp_costs, interv_values, interv_costs)
            if expected_u > best_u:
                best_u = expected_u
                best_policy = policy
                
    return float(best_u), best_policy, time.time() - t0, evals

def greedy_optimizer(p_lead, insp_costs, interv_values, interv_costs, budget):
    N = len(p_lead)
    t0 = time.time()
    evi = compute_approximate_evi(p_lead, interv_values, interv_costs, np.zeros(N), 0.0)
    net_evi = evi - insp_costs
    sorted_idx = np.argsort(net_evi)[::-1]
    
    policy = np.zeros(N, dtype=bool)
    spent = 0.0
    for idx in sorted_idx:
        if net_evi[idx] <= 0: break
        if spent + insp_costs[idx] <= budget:
            policy[idx] = True
            spent += insp_costs[idx]
            
    u = calc_utility(policy, p_lead, insp_costs, interv_values, interv_costs)
    return u, policy, time.time() - t0, 1

def local_search_optimizer(initial_policy, p_lead, insp_costs, interv_values, interv_costs, budget, max_k=1):
    N = len(p_lead)
    t0 = time.time()
    best_policy = initial_policy.copy()
    best_u = calc_utility(best_policy, p_lead, insp_costs, interv_values, interv_costs)
    evals = 1
    
    improved = True
    while improved:
        improved = False
        current_spent = np.sum(insp_costs[best_policy])
        
        # 1-for-1 swap
        if max_k >= 1:
            for i in range(N):
                if best_policy[i]:
                    for j in range(N):
                        if not best_policy[j]:
                            new_cost = current_spent - insp_costs[i] + insp_costs[j]
                            if new_cost <= budget:
                                candidate = best_policy.copy()
                                candidate[i] = False
                                candidate[j] = True
                                u = calc_utility(candidate, p_lead, insp_costs, interv_values, interv_costs)
                                evals += 1
                                if u > best_u + 1e-9:
                                    best_u = u
                                    best_policy = candidate
                                    improved = True
                                    break
                if improved: break
        
        if improved: continue
        
        # 2-for-1 swap
        if max_k >= 2:
            for i1 in range(N):
                if not best_policy[i1]: continue
                for i2 in range(i1 + 1, N):
                    if not best_policy[i2]: continue
                    
                    for j in range(N):
                        if best_policy[j]: continue
                        
                        new_cost = current_spent - insp_costs[i1] - insp_costs[i2] + insp_costs[j]
                        if new_cost <= budget:
                            candidate = best_policy.copy()
                            candidate[i1] = False
                            candidate[i2] = False
                            candidate[j] = True
                            u = calc_utility(candidate, p_lead, insp_costs, interv_values, interv_costs)
                            evals += 1
                            if u > best_u + 1e-9:
                                best_u = u
                                best_policy = candidate
                                improved = True
                                break
                        
                        # 1-for-2 swap (drop 1, pick 2)
                        for j2 in range(j + 1, N):
                            if best_policy[j2]: continue
                            new_cost2 = current_spent - insp_costs[i1] + insp_costs[j] + insp_costs[j2]
                            if new_cost2 <= budget:
                                candidate = best_policy.copy()
                                candidate[i1] = False
                                candidate[j] = True
                                candidate[j2] = True
                                u = calc_utility(candidate, p_lead, insp_costs, interv_values, interv_costs)
                                evals += 1
                                if u > best_u + 1e-9:
                                    best_u = u
                                    best_policy = candidate
                                    improved = True
                                    break
                        if improved: break
                    if improved: break
                if improved: break

    return best_u, best_policy, time.time() - t0, evals

def analyze_regret():
    corpus_path = "data/test_fixtures/f7_regret_corpus_v1.json"
    with open(corpus_path, "r") as f:
        worlds = json.load(f)
        
    results = []
    
    for w in worlds:
        world_id = w["world_id"]
        p_lead = np.array(w["p_lead"])
        insp_costs = np.array(w["insp_costs"])
        interv_values = np.array(w["interv_values"])
        interv_costs = np.array(w["interv_costs"])
        budget = w["budget"]
        
        o_u, o_pol, o_t, o_eval = brute_force_oracle(p_lead, insp_costs, interv_values, interv_costs, budget)
        g_u, g_pol, g_t, g_eval = greedy_optimizer(p_lead, insp_costs, interv_values, interv_costs, budget)
        s1_u, s1_pol, s1_t, s1_eval = local_search_optimizer(g_pol, p_lead, insp_costs, interv_values, interv_costs, budget, max_k=1)
        s2_u, s2_pol, s2_t, s2_eval = local_search_optimizer(g_pol, p_lead, insp_costs, interv_values, interv_costs, budget, max_k=2)
        
        # Guard assertions
        assert g_u <= o_u + 1e-9, f"Greedy exceeded Oracle: {g_u} > {o_u}"
        assert s1_u <= o_u + 1e-9, f"Swap-1 exceeded Oracle: {s1_u} > {o_u}"
        assert s2_u <= o_u + 1e-9, f"Swap-2 exceeded Oracle: {s2_u} > {o_u}"
        
        max_u = max(abs(o_u), 1e-9)
        
        res = {
            "world_id": world_id,
            "budget_tightness": budget / np.sum(insp_costs),
            "oracle_u": o_u,
            "greedy": {"u": g_u, "r_rel": (o_u - g_u) / max_u, "t": g_t, "evals": g_eval},
            "swap1": {"u": s1_u, "r_rel": (o_u - s1_u) / max_u, "t": s1_t + g_t, "evals": s1_eval + g_eval},
            "swap2": {"u": s2_u, "r_rel": (o_u - s2_u) / max_u, "t": s2_t + g_t, "evals": s2_eval + g_eval},
            "oracle": {"u": o_u, "r_rel": 0.0, "t": o_t, "evals": o_eval}
        }
        
        for k in ["greedy", "swap1", "swap2"]:
            assert res[k]["r_rel"] >= -1e-9, f"Negative regret for {k}: {res[k]['r_rel']}"
            
        results.append(res)
        
    df = pd.DataFrame([{
        "world_id": r["world_id"],
        "budget_tightness": r["budget_tightness"],
        "greedy_r_rel": r["greedy"]["r_rel"],
        "greedy_t": r["greedy"]["t"],
        "greedy_evals": r["greedy"]["evals"],
        "swap1_r_rel": r["swap1"]["r_rel"],
        "swap1_t": r["swap1"]["t"],
        "swap1_evals": r["swap1"]["evals"],
        "swap2_r_rel": r["swap2"]["r_rel"],
        "swap2_t": r["swap2"]["t"],
        "swap2_evals": r["swap2"]["evals"],
        "oracle_t": r["oracle"]["t"],
        "oracle_evals": r["oracle"]["evals"],
    } for r in results])
    
    print("\n| Optimizer | Mean regret | Median | P95 | P99 | Max | >1% Failures | Runtime (s) | Evals |")
    print("|---|---|---|---|---|---|---|---|---|")
    for opt in ["greedy", "swap1", "swap2", "oracle"]:
        if opt == "oracle":
            print(f"| Exact Oracle | 0.000% | 0.000% | 0.000% | 0.000% | 0.000% | 0.0% | {df['oracle_t'].sum():.2f} | {int(df['oracle_evals'].mean())} |")
            continue
            
        r_rel = df[f"{opt}_r_rel"] * 100
        mean_r = r_rel.mean()
        median_r = r_rel.median()
        p95_r = np.percentile(r_rel, 95)
        p99_r = np.percentile(r_rel, 99)
        max_r = r_rel.max()
        fail_rate = (r_rel > 1.0).mean() * 100
        t_sum = df[f"{opt}_t"].sum()
        evals = int(df[f"{opt}_evals"].mean())
        
        name = "Greedy" if opt == "greedy" else ("+ swap (1-for-1)" if opt == "swap1" else "+ local search (2-for-2)")
        print(f"| {name} | {mean_r:.3f}% | {median_r:.3f}% | {p95_r:.3f}% | {p99_r:.3f}% | {max_r:.3f}% | {fail_rate:.1f}% | {t_sum:.3f} | {evals} |")
        
    # Worst 10 worlds for Swap2
    worst_10 = df.nlargest(10, "swap2_r_rel")
    print("\nWorst 10 Worlds for Local Search (2-for-2):")
    print(worst_10[["world_id", "budget_tightness", "swap2_r_rel", "greedy_r_rel"]])

if __name__ == "__main__":
    analyze_regret()
