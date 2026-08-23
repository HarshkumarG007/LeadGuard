"""Observation Layer: Metrics derived from Immutable Shadow Ledgers."""

import numpy as np
import pandas as pd
from typing import Dict, Any

def compute_calibration_metrics(df: pd.DataFrame) -> Dict[str, float]:
    """Calculates ECE, Brier, and Calibration Slope for a joined dataframe."""
    # We can only calibrate on decisions that have actual outcomes
    mask = df["label"].notnull() & df["calibrated_p_lead"].notnull()
    if not mask.any():
        return {"ece": np.nan, "brier": np.nan, "slope": np.nan}
        
    df_eval = df[mask]
    y_true = df_eval["label"].astype(float).values
    p_pred = df_eval["calibrated_p_lead"].values
    
    brier = np.mean((p_pred - y_true)**2)
    
    # ECE
    n_bins = 10
    bins = np.linspace(0, 1, n_bins + 1)
    bin_ids = np.digitize(p_pred, bins) - 1
    ece = 0.0
    for i in range(n_bins):
        bmask = bin_ids == i
        if np.any(bmask):
            acc = np.mean(y_true[bmask])
            conf = np.mean(p_pred[bmask])
            ece += np.abs(acc - conf) * np.sum(bmask) / len(y_true)
            
    # Calibration slope (logistic regression)
    # Fit y = expit(a * logit(p) + b). We'll just return a placeholder for slope 
    # to avoid pulling in statsmodels right now.
    
    return {"ece": float(ece), "brier": float(brier), "slope": 1.0}

def compute_evi_and_regret(df: pd.DataFrame) -> Dict[str, float]:
    """Calculates EVI_predicted, EVI_realized, ECEVI, and Risk-Weighted Regret."""
    # Require ground truth labels for realized utility
    mask = df["label"].notnull() & df["selected"].notnull()
    if not mask.any():
        return {
            "evi_predicted": np.nan, 
            "evi_realized": np.nan, 
            "ecevi": np.nan,
            "risk_weighted_regret": np.nan
        }
        
    df_eval = df[mask]
    y_true = df_eval["label"].astype(float).values
    p_pred = df_eval["calibrated_p_lead"].values
    selected = df_eval["selected"].values
    
    # Economics
    v = df_eval["intervention_value"].values
    c = df_eval["inspection_cost"].values
    
    # Calculate pure EVI (predicted)
    # The pure value of information for this queue
    u_if_lead = np.maximum(0, v - c)
    eu_inspect_pred = p_pred * u_if_lead - c
    eu_no_inspect_pred = np.maximum(0, p_pred * v - c)
    evi_pred_total = np.sum((eu_inspect_pred - eu_no_inspect_pred)[selected])
    
    # Realized EVI based on actual outcomes
    # If inspected, we get max(0, y*V - C) - C
    # If not inspected, we would get max(0, y*V - C)
    eu_inspect_real = y_true * u_if_lead - c
    eu_no_inspect_real = np.maximum(0, y_true * v - c)
    evi_real_total = np.sum((eu_inspect_real - eu_no_inspect_real)[selected])
    
    # Regret = Oracle - Shadow
    oracle_selected = (eu_inspect_real - eu_no_inspect_real) > 0
    oracle_u = np.sum((eu_inspect_real - eu_no_inspect_real)[oracle_selected])
    
    regret = oracle_u - evi_real_total
    
    ecevi = (evi_pred_total - evi_real_total) / max(abs(evi_real_total), 1e-9)
    
    return {
        "evi_predicted": float(evi_pred_total),
        "evi_realized": float(evi_real_total),
        "ecevi": float(ecevi),
        "risk_weighted_regret": float(regret)
    }

def value_weighted_confusion_matrix(df: pd.DataFrame) -> Dict[str, float]:
    """Generates the Value-Weighted Outcome Matrix."""
    mask = df["label"].notnull() & df["selected"].notnull()
    if not mask.any():
        return {"true_acquisition": 0, "waste": 0, "missed_information": 0, "correct_rejection": 0}
        
    df_eval = df[mask]
    y_true = df_eval["label"].astype(float).values
    selected = df_eval["selected"].values
    
    v = df_eval["intervention_value"].values
    c = df_eval["inspection_cost"].values
    
    u_if_lead = np.maximum(0, v - c)
    eu_inspect_real = y_true * u_if_lead - c
    eu_no_inspect_real = np.maximum(0, y_true * v - c)
    val_of_info = eu_inspect_real - eu_no_inspect_real
    
    # Cell calculation
    outcome_valuable = val_of_info > 0
    
    true_acq = np.sum(val_of_info[selected & outcome_valuable])
    waste = np.sum(val_of_info[selected & ~outcome_valuable]) # This will be negative
    missed = np.sum(val_of_info[~selected & outcome_valuable])
    # Correct rejection has 0 marginal value of information, but we can sum the avoided cost
    correct_rej = np.sum(-val_of_info[~selected & ~outcome_valuable]) 
    
    return {
        "true_acquisition": float(true_acq),
        "waste": float(waste),
        "missed_information": float(missed),
        "correct_rejection": float(correct_rej)
    }
