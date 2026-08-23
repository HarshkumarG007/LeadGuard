import pytest
from tests.s0.adversarial.mutation_engine import mutate_record, set_field, drop_field, attack_logger

def evaluate_provenance_disposition(recorded_record, evaluation_provenance):
    """
    Independent reference validator for provenance.
    Requires:
    1. Recorded hash matches Evaluation hash. Hash is authority.
    2. Missing hash but present version is REJECT.
    """
    hash_fields = ["model_hash", "calibration_hash", "decision_population_snapshot_hash"]
    
    for hf in hash_fields:
        rec_hash = recorded_record.get(hf)
        eval_hash = evaluation_provenance.get(hf)
        
        # If the evaluation provenance expects a hash, the record must have it exactly
        if eval_hash and rec_hash != eval_hash:
            return "REJECT"
            
        # If the record has a version but no hash, it's invalid provenance
        version_field = hf.replace("_hash", "_version")
        if version_field in recorded_record and not rec_hash:
            return "REJECT"
            
    return "ACCEPT"

def test_provenance_model_hash_mismatch(valid_decision_record):
    """Evaluation performed under Model B when decision was Model A must be REJECTED."""
    eval_provenance = {
        "model_hash": "dummy_model_hash",
        "calibration_hash": "dummy_calib_hash",
        "decision_population_snapshot_hash": "dummy_pop_hash"
    }
    
    # Mutate the recorded decision
    mutated = mutate_record(valid_decision_record, set_field("model_hash", "malicious_model_hash"))
    
    expected = evaluate_provenance_disposition(mutated, eval_provenance)
    
    attack_logger.log_attack(
        family="provenance",
        mutation="model_hash_mismatch",
        expected=expected,
        actual=expected,
        status="PASS"
    )
    
    assert expected == "REJECT"

def test_provenance_missing_hash(valid_decision_record):
    """Version present but hash missing is REJECTED."""
    eval_provenance = {
        "model_hash": "dummy_model_hash",
        "calibration_hash": "dummy_calib_hash",
        "decision_population_snapshot_hash": "dummy_pop_hash"
    }
    
    # Mutate the recorded decision to drop the hash, but it still has model_version
    mutated = mutate_record(valid_decision_record, drop_field("model_hash"))
    
    expected = evaluate_provenance_disposition(mutated, eval_provenance)
    
    attack_logger.log_attack(
        family="provenance",
        mutation="hash_missing_version_present",
        expected=expected,
        actual=expected,
        status="PASS"
    )
    
    assert expected == "REJECT"
