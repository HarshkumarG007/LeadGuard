"""Final Red-Team Pass: Calibration x EVI.

Measures the impact of probability calibration error and ranking error on EVI optimization.
"""

import numpy as np
import pandas as pd
import json
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import brier_score_loss
from pathlib import Path

# EVI & Utility functions
def compute_expected_utility(policy, p_lead, insp_costs, interv_values, interv_costs):
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

def compute_evi(p_lead, interv_values, interv_costs):
    """Computes pure approximate EVI for ranking."""
    u_if_lead = np.maximum(0, interv_values - interv_costs)
    eu_inspect = p_lead * u_if_lead
    eu_no_inspect = np.maximum(0, p_lead * interv_values - interv_costs)
    return eu_inspect - eu_no_inspect

def get_greedy_policy(p_lead, insp_costs, interv_values, interv_costs, budget):
    N = len(p_lead)
    evi = compute_evi(p_lead, interv_values, interv_costs)
    net_evi = evi - insp_costs
    sorted_idx = np.argsort(net_evi)[::-1]
    
    policy = np.zeros(N, dtype=bool)
    spent = 0.0
    for idx in sorted_idx:
        if net_evi[idx] <= 0: break
        if spent + insp_costs[idx] <= budget:
            policy[idx] = True
            spent += insp_costs[idx]
    return policy

def expected_calibration_error(y_true, p_pred, n_bins=10):
    bins = np.linspace(0, 1, n_bins + 1)
    bin_ids = np.digitize(p_pred, bins) - 1
    ece = 0.0
    for i in range(n_bins):
        mask = bin_ids == i
        if np.any(mask):
            acc = np.mean(y_true[mask])
            conf = np.mean(p_pred[mask])
            ece += np.abs(acc - conf) * np.sum(mask) / len(y_true)
    return ece

def logit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))

def expit(x):
    return 1 / (1 + np.exp(-x))

def run_calibration_test():
    corpus_path = "data/test_fixtures/f7_regret_corpus_v1.json"
    with open(corpus_path, "r") as f:
        worlds = json.load(f)
        
    alphas = [0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 3.0]
    results = []
    
    for w in worlds:
        N = w["N"]
        p_true = np.array(w["p_lead"])
        insp_costs = np.array(w["insp_costs"])
        interv_values = np.array(w["interv_values"])
        interv_costs = np.array(w["interv_costs"])
        budget = w["budget"]
        
        # Ground truth Oracle
        oracle_policy = get_greedy_policy(p_true, insp_costs, interv_values, interv_costs, budget)
        oracle_u = compute_expected_utility(oracle_policy, p_true, insp_costs, interv_values, interv_costs)
        
        # We need a holdout set to train the calibrator, or we just simulate y_true.
        # To avoid noise dominating, we calculate expected metrics directly using p_true for Brier/ECE,
        # but isotonic regression needs binary labels.
        np.random.seed(w["world_id"])
        y_true_sample = np.random.binomial(1, p_true)
        
        for alpha in alphas:
            # 1. Rank-preserving transformation (A/B/C)
            p_est = expit(alpha * logit(p_true))
            
            # Policy derived from p_est
            est_policy = get_greedy_policy(p_est, insp_costs, interv_values, interv_costs, budget)
            
            # EVI estimated vs realized
            evi_est = compute_expected_utility(est_policy, p_est, insp_costs, interv_values, interv_costs)
            evi_real = compute_expected_utility(est_policy, p_true, insp_costs, interv_values, interv_costs)
            
            regret = oracle_u - evi_real
            ecevi = evi_est - evi_real
            
            # Calibration metrics
            brier = np.mean((p_est - p_true)**2) # Expected brier relative to true prob
            
            # Jaccard
            intersect = np.sum(est_policy & oracle_policy)
            union = np.sum(est_policy | oracle_policy)
            jaccard = intersect / union if union > 0 else 1.0
            
            results.append({
                "world_id": w["world_id"],
                "regime": "Rank-Preserving",
                "alpha": alpha,
                "brier": brier,
                "jaccard": jaccard,
                "regret": regret,
                "ecevi": ecevi,
                "rel_ecevi": ecevi / max(abs(evi_est), 1e-9)
            })
            
        # 2. Regime D: Systematically Biased (Ranking Changed)
        # We inject an arbitrary feature bias: property ID modulo 2 gets a boost.
        bias = np.where(np.arange(N) % 2 == 0, 0.3, -0.3)
        p_est_D = np.clip(p_true + bias, 0.01, 0.99)
        
        policy_D = get_greedy_policy(p_est_D, insp_costs, interv_values, interv_costs, budget)
        evi_est_D = compute_expected_utility(policy_D, p_est_D, insp_costs, interv_values, interv_costs)
        evi_real_D = compute_expected_utility(policy_D, p_true, insp_costs, interv_values, interv_costs)
        
        brier_D = np.mean((p_est_D - p_true)**2)
        intersect = np.sum(policy_D & oracle_policy)
        union = np.sum(policy_D | oracle_policy)
        jaccard_D = intersect / union if union > 0 else 1.0
        
        results.append({
            "world_id": w["world_id"],
            "regime": "Biased (Rank Changed)",
            "alpha": "N/A",
            "brier": brier_D,
            "jaccard": jaccard_D,
            "regret": oracle_u - evi_real_D,
            "ecevi": evi_est_D - evi_real_D,
            "rel_ecevi": (evi_est_D - evi_real_D) / max(abs(evi_est_D), 1e-9)
        })
        
        # 3. Calibration Repair (Isotonic on D)
        iso = IsotonicRegression(out_of_bounds='clip')
        # We need a distinct train set to simulate holding out data for calibration.
        # We'll just generate a large surrogate holdout to get a good calibrator.
        p_true_holdout = np.random.choice(p_true, 1000)
        p_est_holdout = np.clip(p_true_holdout + np.random.choice([0.3, -0.3], 1000), 0.01, 0.99)
        y_holdout = np.random.binomial(1, p_true_holdout)
        iso.fit(p_est_holdout, y_holdout)
        
        p_est_rep = iso.predict(p_est_D)
        
        policy_rep = get_greedy_policy(p_est_rep, insp_costs, interv_values, interv_costs, budget)
        evi_est_rep = compute_expected_utility(policy_rep, p_est_rep, insp_costs, interv_values, interv_costs)
        evi_real_rep = compute_expected_utility(policy_rep, p_true, insp_costs, interv_values, interv_costs)
        
        brier_rep = np.mean((p_est_rep - p_true)**2)
        intersect = np.sum(policy_rep & oracle_policy)
        union = np.sum(policy_rep | oracle_policy)
        jaccard_rep = intersect / union if union > 0 else 1.0
        
        results.append({
            "world_id": w["world_id"],
            "regime": "Repaired",
            "alpha": "N/A",
            "brier": brier_rep,
            "jaccard": jaccard_rep,
            "regret": oracle_u - evi_real_rep,
            "ecevi": evi_est_rep - evi_real_rep,
            "rel_ecevi": (evi_est_rep - evi_real_rep) / max(abs(evi_est_rep), 1e-9)
        })
        
    df = pd.DataFrame(results)
    
    print("=== LeadGuard Final Red-Team: Calibration x EVI ===")
    
    # Analyze Alpha Sweep
    print("\n1. Rank-Preserving Calibration Error (Alpha Sweep)")
    print(df[df["regime"] == "Rank-Preserving"].groupby("alpha")[["brier", "jaccard", "rel_ecevi", "regret"]].mean())
    
    # Analyze Regimes
    print("\n2. Regime Comparison")
    regime_df = []
    
    def add_row(name, mask):
        sub = df[mask]
        regime_df.append({
            "Regime": name,
            "Brier": sub["brier"].mean(),
            "Jaccard": sub["jaccard"].mean(),
            "Rel_ECEVI": sub["rel_ecevi"].mean(),
            "Regret": sub["regret"].mean()
        })
        
    add_row("A (Well Calibrated, a=1)", (df["regime"] == "Rank-Preserving") & (df["alpha"] == 1.0))
    add_row("B (Overconfident, a=3)", (df["regime"] == "Rank-Preserving") & (df["alpha"] == 3.0))
    add_row("C (Underconfident, a=0.5)", (df["regime"] == "Rank-Preserving") & (df["alpha"] == 0.5))
    add_row("D (Systematically Biased)", df["regime"] == "Biased (Rank Changed)")
    add_row("Repaired (Isotonic on D)", df["regime"] == "Repaired")
    
    res_df = pd.DataFrame(regime_df)
    print(res_df.to_string(index=False))

if __name__ == "__main__":
    run_calibration_test()
