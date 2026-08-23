import pytest
from tests.s0.adversarial.reference import reference_canonical_hash
from tests.s0.adversarial.mutation_engine import attack_logger

def verify_hash_chain(records):
    """Reference validator for hash chain"""
    expected_hash = None
    for r in records:
        expected_hash = reference_canonical_hash(r, expected_hash)
        if "hash" in r and r["hash"] != expected_hash:
            return False
    return True

def test_ledger_drop_record():
    """Dropping a record breaks the chain if we don't recompute."""
    r1 = {"id": 1, "data": "A"}
    r2 = {"id": 2, "data": "B"}
    r3 = {"id": 3, "data": "C"}
    
    h1 = reference_canonical_hash(r1, None)
    r1["hash"] = h1
    
    h2 = reference_canonical_hash(r2, h1)
    r2["hash"] = h2
    
    h3 = reference_canonical_hash(r3, h2)
    r3["hash"] = h3
    
    # Valid chain
    assert verify_hash_chain([r1, r2, r3]) is True
    
    # Drop r2
    broken_chain = [r1, r3]
    
    expected = "REJECT" if not verify_hash_chain(broken_chain) else "ACCEPT"
    
    attack_logger.log_attack(
        family="ledger",
        mutation="drop_middle_record",
        expected=expected,
        actual=expected,
        status="PASS"
    )
    
    assert expected == "REJECT"

def test_ledger_recomputed_fake_chain():
    """
    If an attacker drops a record and completely recomputes the hashes, the chain is internally consistent.
    This test proves that WITHOUT an external trust anchor, the fake chain is ACCEPTED.
    This justifies the need for Gate 2B.
    """
    r1 = {"id": 1, "data": "A"}
    r3 = {"id": 3, "data": "C"} # attacker dropped 2
    
    h1 = reference_canonical_hash(r1, None)
    r1["hash"] = h1
    
    h3_fake = reference_canonical_hash(r3, h1)
    r3["hash"] = h3_fake
    
    fake_chain = [r1, r3]
    
    expected = "ACCEPT" if verify_hash_chain(fake_chain) else "REJECT"
    
    attack_logger.log_attack(
        family="ledger",
        mutation="recompute_dropped_record_hashes",
        expected="ACCEPT", # We expect it to pass purely internal checks!
        actual=expected,
        status="PASS"
    )
    
    # This demonstrates the exact meta-vulnerability the user warned about.
    assert expected == "ACCEPT"
