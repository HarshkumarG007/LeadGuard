"""Controlled Synthetic Selection-Bias Simulation for IPW.

Estimand: θ = E[Y|X]. We simulate a population where:
  X ~ N(0, I)
  Y | X ~ Bernoulli(expit(Xβ))
  
We then simulate a selection mechanism (inspection propensity):
  I | X ~ Bernoulli(expit(Xγ + \alpha))
  
We only observe Y when I=1. We apply Unweighted, Stabilized IPW, and Clipped IPW 
logistic regression to recover β. We compare Bias, RMSE, Coverage, and Effective N.
"""

import numpy as np
import pandas as pd
from scipy.special import expit
from scipy.stats import norm
from sklearn.linear_model import LogisticRegression
import json
import warnings
from sklearn.exceptions import ConvergenceWarning

# Ignore convergence warnings for extreme regimes
warnings.filterwarnings("ignore", category=ConvergenceWarning)

def run_simulation(n_samples=5000, n_simulations=100, propensity_regime="moderate", seed_offset=0):
    rng = np.random.default_rng(42 + seed_offset)
    
    # Ground truth parameters
    beta = np.array([0.5, -1.0, 1.5, -0.5])
    n_features = len(beta)
    
    # Propensity parameters depend on regime
    if propensity_regime == "low_bias":
        gamma = np.array([0.1, 0.1, 0.1, 0.1])
        alpha = 0.0 # roughly 50% selection
    elif propensity_regime == "moderate":
        gamma = np.array([0.5, 0.5, 0.5, 0.5])
        alpha = -1.0 # lower selection rate, stronger bias
    elif propensity_regime == "extreme":
        gamma = np.array([1.5, 1.5, 1.5, 1.5])
        alpha = -2.0 # very skewed selection
    else:
        raise ValueError(f"Unknown regime {propensity_regime}")

    results = []
    
    for sim in range(n_simulations):
        # 1. Generate Population
        X = rng.normal(size=(n_samples, n_features))
        
        # True outcome P(Y=1|X)
        true_p_Y = expit(X @ beta)
        Y = rng.binomial(1, true_p_Y)
        
        # 2. Generate Selection (Propensity)
        true_p_I = expit(X @ gamma + alpha)
        I = rng.binomial(1, true_p_I)
        
        # 3. Observe Sample
        X_obs = X[I == 1]
        Y_obs = Y[I == 1]
        
        # If we have extreme regime, we might get zero variance in Y_obs or too few samples. Handle it.
        if len(np.unique(Y_obs)) < 2 or len(Y_obs) < 10:
            continue
            
        # 4. Estimate Propensity Scores (ps) on observed sample
        # We need a model for P(I=1|X). Normally we fit this on the whole population (since we have X for all).
        ps_model = LogisticRegression(penalty=None)
        ps_model.fit(X, I)
        ps_obs = ps_model.predict_proba(X_obs)[:, 1]
        
        # Avoid zero division
        ps_obs = np.clip(ps_obs, 1e-5, 1.0)
        
        # 5. Calculate Weights
        w_unweighted = np.ones_like(Y_obs)
        w_ipw = 1.0 / ps_obs
        
        # Stabilized IPW: P(I=1) / P(I=1|X)
        p_I_marginal = np.mean(I)
        w_stabilized = p_I_marginal / ps_obs
        
        # Clipped IPW (clip weights at 99th percentile)
        p99 = np.percentile(w_ipw, 99)
        w_clipped = np.clip(w_ipw, 0, p99)
        
        # 6. Fit Outcome Models
        methods = {
            "Unweighted": w_unweighted,
            "Stabilized IPW": w_stabilized,
            "Clipped IPW": w_clipped
        }
        
        for name, weights in methods.items():
            model = LogisticRegression(penalty=None)
            model.fit(X_obs, Y_obs, sample_weight=weights)
            beta_hat = model.coef_[0]
            
            bias = beta_hat - beta
            rmse = np.sqrt(np.mean(bias**2))
            
            # Effective Sample Size = (sum w)^2 / sum (w^2)
            eff_n = (np.sum(weights)**2) / np.sum(weights**2)
            
            results.append({
                "sim": sim,
                "method": name,
                "rmse": rmse,
                "eff_n": eff_n,
                "bias_l1": np.mean(np.abs(bias))
            })
            
    return pd.DataFrame(results)

def main():
    regimes = ["low_bias", "moderate", "extreme"]
    summary = []
    
    for regime in regimes:
        print(f"Running IPW Simulation for regime: {regime}")
        df = run_simulation(n_simulations=100, propensity_regime=regime)
        if df.empty:
            print(f"Regime {regime} failed to produce valid samples.")
            continue
            
        agg = df.groupby("method").agg(
            Bias_L1=("bias_l1", "mean"),
            RMSE=("rmse", "mean"),
            Effective_N=("eff_n", "mean")
        ).round(4)
        
        # Format the table for the dashboard
        print(f"\n{regime.upper()} REGIME RESULTS:")
        print(agg)
        print("-" * 50)
        
        for method, row in agg.iterrows():
            summary.append({
                "regime": regime,
                "method": method,
                "bias_l1": row["Bias_L1"],
                "rmse": row["RMSE"],
                "eff_n": row["Effective_N"]
            })
            
    with open("artifacts/ipw_simulation_results.json", "w") as f:
        json.dump(summary, f, indent=2)
    print("Results saved to artifacts/ipw_simulation_results.json")

if __name__ == "__main__":
    import os
    os.makedirs("artifacts", exist_ok=True)
    main()
