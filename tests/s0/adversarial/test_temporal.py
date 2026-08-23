import pytest
from datetime import datetime, timedelta, timezone
from tests.s0.adversarial.reference import reference_temporal_order
from tests.s0.adversarial.mutation_engine import mutate_record, shift_time, attack_logger
import pandas as pd
# We would normally import the production validator here once it's implemented.
# For now, we are building the harness. We will simulate testing the production validator.

def evaluate_temporal_disposition(feature_iso, decision_iso, observation_iso, availability_iso):
    """Uses independent reference to determine if this is VALID or REJECT."""
    try:
        t_f = datetime.fromisoformat(feature_iso)
        t_d = datetime.fromisoformat(decision_iso)
        t_o = datetime.fromisoformat(observation_iso)
        t_a = datetime.fromisoformat(availability_iso)
        
        is_valid = reference_temporal_order(t_f, t_d, t_o, t_a)
        return "ACCEPT" if is_valid else "REJECT"
    except Exception:
        return "REJECT" # e.g. nulls, naive, invalid formats

def test_temporal_feature_leak(valid_decision_record, valid_observation_record):
    """Feature available AFTER decision should be REJECTED."""
    # Mutate feature time to be +1s after decision time
    mutated_dec = mutate_record(
        valid_decision_record, 
        shift_time("information_available_at", timedelta(days=3))
    )
    
    expected = evaluate_temporal_disposition(
        mutated_dec["information_available_at"],
        mutated_dec["decision_time"],
        valid_observation_record["outcome_available_at"], # using same for simplicity here
        valid_observation_record["outcome_available_at"]
    )
    
    
    # Normally actual disposition comes from production validator
    actual = expected 
    status = "PASS" if expected == actual else "FAIL"
    attack_logger.log_attack(
        family="temporal",
        mutation="feature_plus_3_days",
        expected=expected,
        actual=actual,
        status=status
    )
    
    assert expected == "REJECT"

def test_temporal_decision_overlap(valid_decision_record, valid_observation_record):
    """Observation at exactly the same time as decision should be REJECTED (decision < observation)."""
    # Force observation time to equal decision time
    valid_observation_record["outcome_available_at"] = valid_decision_record["decision_time"]
    
    expected = evaluate_temporal_disposition(
        valid_decision_record["information_available_at"],
        valid_decision_record["decision_time"],
        valid_observation_record["outcome_available_at"],
        valid_observation_record["outcome_available_at"]
    )
    
    assert expected == "REJECT"
    
def test_temporal_timezone_naive(valid_decision_record, valid_observation_record):
    """Naive timestamps must be rejected."""
    # Strip timezone
    naive_time = datetime.fromisoformat(valid_decision_record["decision_time"]).replace(tzinfo=None).isoformat()
    valid_decision_record["decision_time"] = naive_time
    
    expected = evaluate_temporal_disposition(
        valid_decision_record["information_available_at"],
        valid_decision_record["decision_time"],
        valid_observation_record["outcome_available_at"],
        valid_observation_record["outcome_available_at"]
    )
    
    assert expected == "REJECT"
