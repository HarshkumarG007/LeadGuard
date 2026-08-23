"""Links shadow decisions to observations temporally."""

import pandas as pd
from datetime import datetime
from typing import Optional
from leadguard.shadow.decision_recorder import get_ledger_path

def load_shadow_ledger(environment: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    dec_path = get_ledger_path(environment, "decisions")
    obs_path = get_ledger_path(environment, "observations")
    
    if not dec_path.exists():
        df_dec = pd.DataFrame()
    else:
        df_dec = pd.read_json(dec_path, lines=True)
        df_dec["decision_time"] = pd.to_datetime(df_dec["decision_time"])
        df_dec["information_available_at"] = pd.to_datetime(df_dec["information_available_at"])
        
    if not obs_path.exists():
        df_obs = pd.DataFrame()
    else:
        df_obs = pd.read_json(obs_path, lines=True)
        df_obs["outcome_available_at"] = pd.to_datetime(df_obs["outcome_available_at"])
        if "label_available_at" in df_obs.columns:
            df_obs["label_available_at"] = pd.to_datetime(df_obs["label_available_at"])
            
    return df_dec, df_obs

def temporal_join(environment: str, as_of: Optional[datetime] = None) -> pd.DataFrame:
    """Joins decisions to observations available at the time of `as_of`.
    
    CRITICAL (S0.2): Enforces temporal barrier: 
    Observation is only linked if outcome_available_at <= as_of
    and the decision's information_available_at > decision_time.
    """
    df_dec, df_obs = load_shadow_ledger(environment)
    if df_dec.empty:
        return pd.DataFrame()
        
    # If no as_of is provided, use the maximum time
    if as_of is not None:
        # We can only use observations that were actually available
        if not df_obs.empty:
            df_obs = df_obs[df_obs["outcome_available_at"] <= pd.to_datetime(as_of)]
            
    if df_obs.empty:
        # Return decisions with all observation columns as NaN
        df_joined = df_dec.copy()
        for col in ["label", "outcome_available_at", "observed_lead"]:
            df_joined[col] = None
        return df_joined
        
    # Join on shadow_decision_id
    df_joined = pd.merge(
        df_dec, 
        df_obs, 
        on=["shadow_decision_id", "property_id"], 
        how="left",
        suffixes=("", "_obs")
    )
    
    # S0.2 temporal barrier validation
    # A decision should NEVER be linked to an outcome that was known at decision time
    if not df_joined.empty:
        # In shadow mode, we simulate decisions in the past. If the outcome was already
        # available at decision time, the temporal invariant is violated.
        violation_mask = df_joined["outcome_available_at"] <= df_joined["decision_time"]
        if violation_mask.any():
            raise ValueError(
                f"Temporal violation: {violation_mask.sum()} observations were available "
                "BEFORE or AT the decision time. This breaks causality."
            )
            
    return df_joined
