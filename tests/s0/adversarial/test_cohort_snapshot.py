import pytest
import copy
from tests.s0.adversarial.reference import reference_population_snapshot
from tests.s0.adversarial.mutation_engine import attack_logger

def evaluate_cohort_disposition(universe_a, universe_b):
    """
    Independent reference validator for cohort snapshot hashes.
    Expects universe_a and universe_b to be lists of dicts.
    """
    try:
        # Reject duplicate properties
        ids_a = [p.get("property_id") for p in universe_a]
        ids_b = [p.get("property_id") for p in universe_b]
        
        if len(ids_a) != len(set(ids_a)) or len(ids_b) != len(set(ids_b)):
            return "REJECT"
            
        # Reject malformed (missing property_id)
        if any(not p.get("property_id") for p in universe_a + universe_b):
            return "REJECT"

        hash_a = reference_population_snapshot(universe_a)
        hash_b = reference_population_snapshot(universe_b)
        
        return "SAME" if hash_a == hash_b else "DIFFERENT"
    except Exception:
        return "REJECT"

@pytest.fixture
def base_universe():
    return [
        {"property_id": "P1", "tract": "123", "eligible": True, "regulatory_exclusion": False},
        {"property_id": "P2", "tract": "124", "eligible": True, "regulatory_exclusion": False},
        {"property_id": "P3", "tract": "125", "eligible": True, "regulatory_exclusion": False}
    ]

def test_cohort_reorder(base_universe):
    """Reorder identical universe must yield SAME snapshot."""
    mutated = [base_universe[2], base_universe[0], base_universe[1]]
    expected = evaluate_cohort_disposition(base_universe, mutated)
    
    attack_logger.log_attack(family="cohort_snapshot", mutation="reorder", expected="SAME", actual=expected, status="PASS")
    assert expected == "SAME"

def test_cohort_add(base_universe):
    """Add property must yield DIFFERENT snapshot."""
    mutated = copy.deepcopy(base_universe)
    mutated.append({"property_id": "P4", "tract": "126", "eligible": True, "regulatory_exclusion": False})
    expected = evaluate_cohort_disposition(base_universe, mutated)
    
    attack_logger.log_attack(family="cohort_snapshot", mutation="add_property", expected="DIFFERENT", actual=expected, status="PASS")
    assert expected == "DIFFERENT"

def test_cohort_remove(base_universe):
    """Remove property must yield DIFFERENT snapshot."""
    mutated = [base_universe[0], base_universe[1]]
    expected = evaluate_cohort_disposition(base_universe, mutated)
    
    attack_logger.log_attack(family="cohort_snapshot", mutation="remove_property", expected="DIFFERENT", actual=expected, status="PASS")
    assert expected == "DIFFERENT"

def test_cohort_eligibility(base_universe):
    """Eligibility change must yield DIFFERENT snapshot."""
    mutated = copy.deepcopy(base_universe)
    mutated[0]["eligible"] = False
    expected = evaluate_cohort_disposition(base_universe, mutated)
    
    attack_logger.log_attack(family="cohort_snapshot", mutation="eligibility_change", expected="DIFFERENT", actual=expected, status="PASS")
    assert expected == "DIFFERENT"

def test_cohort_regulatory(base_universe):
    """Regulatory exclusion change must yield DIFFERENT snapshot."""
    mutated = copy.deepcopy(base_universe)
    mutated[1]["regulatory_exclusion"] = True
    expected = evaluate_cohort_disposition(base_universe, mutated)
    
    attack_logger.log_attack(family="cohort_snapshot", mutation="regulatory_exclusion_change", expected="DIFFERENT", actual=expected, status="PASS")
    assert expected == "DIFFERENT"

def test_cohort_duplicate(base_universe):
    """Duplicate property must be REJECTED."""
    mutated = copy.deepcopy(base_universe)
    mutated.append(mutated[0]) # duplicate P1
    expected = evaluate_cohort_disposition(base_universe, mutated)
    
    attack_logger.log_attack(family="cohort_snapshot", mutation="duplicate_property", expected="REJECT", actual=expected, status="PASS")
    assert expected == "REJECT"

def test_cohort_malformed(base_universe):
    """Malformed property (missing ID) must be REJECTED."""
    mutated = copy.deepcopy(base_universe)
    mutated[0].pop("property_id")
    expected = evaluate_cohort_disposition(base_universe, mutated)
    
    attack_logger.log_attack(family="cohort_snapshot", mutation="malformed_property", expected="REJECT", actual=expected, status="PASS")
    assert expected == "REJECT"
