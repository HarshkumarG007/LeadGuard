import pytest
from datetime import timedelta
from tests.s0.adversarial.mutation_engine import mutate_record, shift_time, nullify_field, swap_property_id
from tests.s0.adversarial.reference import reference_temporal_order

def test_mutation_effectiveness_temporal_shift(valid_decision_record, valid_observation_record):
    """
    Prove that shifting the decision time into the future actually breaks the invariant.
    """
    # Baseline: must be true
    from datetime import datetime
    t_f = datetime.fromisoformat(valid_decision_record["information_available_at"])
    t_d = datetime.fromisoformat(valid_decision_record["decision_time"])
    t_o = datetime.fromisoformat(valid_observation_record["outcome_available_at"])
    t_a = datetime.fromisoformat(valid_observation_record["outcome_available_at"])
    
    assert reference_temporal_order(t_f, t_d, t_o, t_a) is True

    # Mutate
    mutated_dec = mutate_record(
        valid_decision_record, 
        shift_time("decision_time", timedelta(days=50))
    )
    
    # Mutated check
    t_d_mutated = datetime.fromisoformat(mutated_dec["decision_time"])
    
    # Must be false, because t_d (decision) > t_o (observation)
    assert reference_temporal_order(t_f, t_d_mutated, t_o, t_a) is False

def test_mutation_effectiveness_identity_swap(valid_decision_record):
    """
    Prove that identity swap changes the property_id.
    """
    original_id = valid_decision_record["property_id"]
    new_id = "malicious_id"
    mutated = mutate_record(valid_decision_record, swap_property_id(new_id))
    
    assert mutated["property_id"] == new_id
    assert mutated["property_id"] != original_id
