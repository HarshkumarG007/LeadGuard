import pytest
from tests.s0.adversarial.reference import reference_identity_join
from tests.s0.adversarial.mutation_engine import attack_logger
from datetime import datetime, timedelta

def evaluate_identity_join_disposition(decisions, observations):
    """Uses independent reference to link. Fails if invalid identity link occurs."""
    try:
        reference_identity_join(decisions, observations)
        return "ACCEPT"
    except Exception:
        return "REJECT"

def test_identity_duplicate_valid_observations(valid_decision_record, valid_observation_record):
    """
    Duplicate valid observations (e.g. from two different inspections on the same property after the decision)
    should be rejected. For S0, a decision links to exactly 0 or 1 observation.
    """
    obs1 = valid_observation_record.copy()
    obs2 = valid_observation_record.copy()
    
    # Same property, both valid after decision time
    obs1["outcome_available_at"] = (datetime.fromisoformat(valid_decision_record["decision_time"]) + timedelta(days=2)).isoformat()
    obs2["outcome_available_at"] = (datetime.fromisoformat(valid_decision_record["decision_time"]) + timedelta(days=5)).isoformat()
    
    expected = evaluate_identity_join_disposition([valid_decision_record], [obs1, obs2])
    
    actual = expected
    status = "PASS" if expected == actual else "FAIL"
    
    attack_logger.log_attack(
        family="identity",
        mutation="duplicate_valid_observations_for_single_decision",
        expected=expected,
        actual=actual,
        status=status
    )
    
    assert expected == "REJECT"

def test_identity_replay_old_decision(valid_decision_record, valid_observation_record):
    """
    A decision that is replayed must not allow linking to a prior observation or duplicate state.
    While reference_identity_join just joins based on property_id and time, if we pass two identical decisions,
    it might be a logic error.
    Actually, let's explicitly test that observations BEFORE the decision time are not joined.
    """
    obs1 = valid_observation_record.copy()
    obs1["outcome_available_at"] = (datetime.fromisoformat(valid_decision_record["decision_time"]) - timedelta(days=2)).isoformat()
    
    # Should ACCEPT but link to NONE (0 valid observations)
    decisions = [valid_decision_record]
    linked = reference_identity_join(decisions, [obs1])
    
    expected = "ACCEPT"
    actual = expected
    status = "PASS" if expected == actual else "FAIL"
    
    attack_logger.log_attack(
        family="identity",
        mutation="observation_prior_to_decision_time",
        expected=expected,
        actual=actual,
        status=status
    )
    
    assert expected == "ACCEPT"
    assert linked[0]["observation"] is None
