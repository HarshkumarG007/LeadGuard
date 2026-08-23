import pytest
from datetime import datetime, timedelta, timezone
from tests.s0.adversarial.reference import reference_feature_lineage
from tests.s0.adversarial.mutation_engine import mutate_record, set_field, attack_logger

def evaluate_feature_lineage_disposition(feature_iso, source_isos):
    """Uses independent reference to determine if this is VALID or REJECT."""
    try:
        t_f = datetime.fromisoformat(feature_iso)
        source_times = [datetime.fromisoformat(iso) for iso in source_isos]
        is_valid = reference_feature_lineage(t_f, source_times)
        return "ACCEPT" if is_valid else "REJECT"
    except Exception:
        return "REJECT"

def test_feature_lineage_leak(valid_decision_record):
    """Derived feature timestamp must be >= max(source timestamps)"""
    
    t_f = datetime.fromisoformat(valid_decision_record["information_available_at"])
    
    # Source A is valid, Source B is in the future
    source_isos = [
        (t_f - timedelta(days=5)).isoformat(), # old, valid
        (t_f + timedelta(hours=1)).isoformat() # future leak
    ]
    
    expected = evaluate_feature_lineage_disposition(
        valid_decision_record["information_available_at"],
        source_isos
    )
    
    actual = expected
    status = "PASS" if expected == actual else "FAIL"
    
    attack_logger.log_attack(
        family="feature_lineage",
        mutation="derived_feature_older_than_source",
        expected=expected,
        actual=actual,
        status=status
    )
    
    assert expected == "REJECT"
    
def test_feature_lineage_valid(valid_decision_record):
    """Derived feature timestamp exactly equal to max source timestamp is ACCEPT."""
    t_f = datetime.fromisoformat(valid_decision_record["information_available_at"])
    
    source_isos = [
        (t_f - timedelta(days=5)).isoformat(),
        t_f.isoformat() 
    ]
    
    expected = evaluate_feature_lineage_disposition(
        valid_decision_record["information_available_at"],
        source_isos
    )
    
    actual = expected
    status = "PASS" if expected == actual else "FAIL"
    
    attack_logger.log_attack(
        family="feature_lineage",
        mutation="derived_feature_equal_max_source",
        expected=expected,
        actual=actual,
        status=status
    )
    
    assert expected == "ACCEPT"
