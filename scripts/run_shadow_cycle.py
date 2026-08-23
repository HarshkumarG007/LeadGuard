"""Simulates the S1 Cohort cycle to prove the S0 infrastructure."""

import uuid
from datetime import datetime, UTC, timedelta
from leadguard.shadow.schemas import ShadowDecisionRecord, ObservationRecord, EligibilityState
from leadguard.shadow.decision_recorder import record_shadow_decision, record_observation
from leadguard.shadow.outcome_linker import temporal_join
from leadguard.monitoring.dashboard import (
    compute_calibration_metrics,
    compute_evi_and_regret,
    value_weighted_confusion_matrix
)
import numpy as np
import pandas as pd

def run_s0_proof():
    print("=== LeadGuard 2.0 Phase S0: Shadow-Mode Proof ===")
    
    # 1. Simulate Frozen LeadGuard 1.0 making a decision at T0
    t0 = datetime.now(UTC) - timedelta(days=60)
    decision_id = f"shadow-{uuid.uuid4().hex[:8]}"
    prop_id = "prop-12345"
    
    dec = ShadowDecisionRecord(
        shadow_decision_id=decision_id,
        property_id=prop_id,
        snapshot_id="snap-v1.0",
        model_version="xgb-v1.0",
        feature_version="feat-v1.0",
        calibration_version="cal-v1.0",
        calibration_dataset_cutoff=(t0 - timedelta(days=1)).isoformat(),
        policy_version="pol-v1.0",
        policy_parameters_hash="hash-abc",
        optimizer_version="opt-v1.0",
        p_lead=0.6,
        calibrated_p_lead=0.7,
        uncertainty=0.1,
        evi=3000.0,
        expected_utility=3500.0,
        inspection_cost=500.0,
        intervention_value=5000.0,
        selected=True,
        rank=1,
        eligibility_state=EligibilityState.ELIGIBLE,
        decision_time=t0,
        information_cutoff=t0 - timedelta(days=1),
        information_available_at=t0, # Information known at decision time
        canonical_hash="placeholder",
        environment="synthetic",
        simulation_version="s0-proof"
    )
    
    # 2. Immutable Record
    print(f"\n[1] Recording Shadow Decision at {t0.strftime('%Y-%m-%d')}")
    record_shadow_decision(dec)
    
    # 3. 30-day information delay
    t_obs = t0 + timedelta(days=30)
    
    # 4. Observation
    print(f"[2] Recording Delayed Observation at {t_obs.strftime('%Y-%m-%d')}")
    obs = ObservationRecord(
        observation_id=f"obs-{uuid.uuid4().hex[:8]}",
        shadow_decision_id=decision_id,
        property_id=prop_id,
        inspection_performed=True,
        inspection_at=t0 + timedelta(days=5),
        label=True, # Ground truth is LEAD
        label_available_at=t_obs,
        intervention_performed=True,
        intervention_at=t0 + timedelta(days=10),
        intervention_cost=5000.0,
        observed_lead=True,
        remediation_completed=True,
        remediation_cost=5000.0,
        outcome_available_at=t_obs, # Not available until T0 + 30
        outcome_source="synthetic-field-ops",
        environment="synthetic",
        simulation_version="s0-proof"
    )
    record_observation(obs)
    
    # 5. Temporal Join (Attempt to join at T0 + 15 days, should fail to see outcome)
    t_early = t0 + timedelta(days=15)
    print(f"\n[3] Temporal Join at {t_early.strftime('%Y-%m-%d')} (Expected: outcome hidden)")
    df_early = temporal_join("synthetic", as_of=t_early)
    assert df_early["label"].isnull().all(), "Temporal violation! Outcome leaked early."
    print("    -> SUCCESS: Temporal barrier held. Outcome safely hidden.")
    
    # Temporal Join (At T0 + 35 days, outcome should be visible)
    t_late = t0 + timedelta(days=35)
    print(f"[4] Temporal Join at {t_late.strftime('%Y-%m-%d')} (Expected: outcome visible)")
    df_late = temporal_join("synthetic", as_of=t_late)
    assert df_late["label"].notnull().all(), "Outcome failed to join after availability."
    print("    -> SUCCESS: Outcome correctly linked.")
    
    # 6. Independent Metric Reconstruction & 7. Dashboard
    print("\n[5] Independent Metric Reconstruction")
    cal_metrics = compute_calibration_metrics(df_late)
    evi_metrics = compute_evi_and_regret(df_late)
    cm = value_weighted_confusion_matrix(df_late)
    
    print("\n--- S0 DASHBOARD ---")
    print("Calibration:")
    for k, v in cal_metrics.items(): print(f"  {k}: {v:.4f}")
    
    print("\nDecision Quality:")
    for k, v in evi_metrics.items(): print(f"  {k}: {v:.4f}")
    
    print("\nValue-Weighted Outcome Matrix:")
    for k, v in cm.items(): print(f"  {k}: ${v:,.2f}")
    
    print("\n[6] Phase S0 Complete: The Observational Infrastructure is ready.")

if __name__ == "__main__":
    run_s0_proof()
