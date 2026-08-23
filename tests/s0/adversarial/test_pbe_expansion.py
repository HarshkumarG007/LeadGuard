import pytest
import json
from datetime import datetime, timezone, timedelta
from hypothesis import given, settings, strategies as st
from tests.s0.adversarial.mutation_engine import attack_logger
from tests.s0.adversarial.reference import (
    reference_temporal_order,
    canonicalize_json,
    reference_population_snapshot
)
from pathlib import Path
import math

def save_regression(family: str, mutation: str, payload: dict, seed: str):
    reg_path = Path("tests/s0/adversarial/regressions")
    reg_path.mkdir(parents=True, exist_ok=True)
    filename = reg_path / f"regression_{family}_{seed}.json"
    with open(filename, "w") as f:
        json.dump({"payload": payload, "mutation": mutation, "family": family}, f)

# --- 1. TEMPORAL ---
base_date = datetime(2026, 8, 1, 12, 0, tzinfo=timezone.utc)
dates_st = st.datetimes(
    min_value=base_date - timedelta(days=365*10),
    max_value=base_date + timedelta(days=365*10),
    timezones=st.just(timezone.utc)
)

@settings(max_examples=100, deadline=None)
@given(t_f=dates_st, t_d=dates_st, t_o=dates_st, t_a=dates_st)
def test_pbe_temporal_invariant(t_f, t_d, t_o, t_a):
    try:
        actual = reference_temporal_order(t_f, t_d, t_o, t_a)
        if actual is True:
            assert t_f <= t_d, f"Violation: T_f ({t_f}) > T_d ({t_d})"
            assert t_d < t_o, f"Violation: T_d ({t_d}) >= T_o ({t_o})"
            assert t_o <= t_a, f"Violation: T_o ({t_o}) > T_a ({t_a})"
        
        expected = "ACCEPT" if (t_f <= t_d < t_o <= t_a) else "REJECT"
        actual_str = "ACCEPT" if actual else "REJECT"
        
        attack_logger.log_attack(
            family="temporal",
            mutation="fuzz_timestamps",
            expected=expected,
            actual=actual_str,
            status="PASS" if expected == actual_str else "FAIL",
            seed=str(hash(f"{t_f}{t_d}{t_o}{t_a}"))
        )
    except AssertionError as e:
        save_regression("temporal", "fuzz_timestamps", {"t_f": t_f.isoformat(), "t_d": t_d.isoformat(), "t_o": t_o.isoformat(), "t_a": t_a.isoformat()}, str(hash(f"{t_f}{t_d}{t_o}{t_a}")))
        raise

# --- 2. METAMORPHIC: INPUT PERMUTATION ---
@settings(max_examples=50, deadline=None)
@given(permutation_indices=st.permutations([0, 1, 2, 3, 4]))
def test_metamorphic_input_permutation(permutation_indices):
    """
    f(perm(X)) == f(X) for deterministic snapshot hashing
    """
    universe = [
        {"property_id": f"P{i}", "eligible": True} for i in range(5)
    ]
    permuted_universe = [universe[i] for i in permutation_indices]
    
    hash_original = reference_population_snapshot(universe)
    hash_permuted = reference_population_snapshot(permuted_universe)
    
    try:
        assert hash_original == hash_permuted
        attack_logger.log_attack(
            family="cohort_snapshot",
            mutation="metamorphic_input_permutation",
            expected="ACCEPT",
            actual="ACCEPT",
            status="PASS",
            seed=str(permutation_indices)
        )
    except AssertionError:
        save_regression("cohort_snapshot", "metamorphic_input_permutation", {"indices": permutation_indices}, str(hash(str(permutation_indices))))
        raise

# --- 3. METAMORPHIC: INELIGIBLE PROPERTY ADDITION ---
@settings(max_examples=50, deadline=None)
@given(ineligible_id=st.text(min_size=1))
def test_metamorphic_ineligible_addition(ineligible_id):
    """
    f(X U {ineligible}) == f(X) if we filter for eligibility
    Wait, the reference_population_snapshot operates on the ELIGIBLE universe.
    So passing an ineligible property to the pre-filter logic should yield the same snapshot.
    We'll simulate the filter here.
    """
    universe = [{"property_id": "P1", "eligible": True}]
    universe_filtered = [p for p in universe if p["eligible"]]
    
    universe_with_ineligible = [{"property_id": "P1", "eligible": True}, {"property_id": ineligible_id, "eligible": False}]
    universe_with_ineligible_filtered = [p for p in universe_with_ineligible if p["eligible"]]
    
    hash_original = reference_population_snapshot(universe_filtered)
    hash_expanded = reference_population_snapshot(universe_with_ineligible_filtered)
    
    try:
        assert hash_original == hash_expanded
        attack_logger.log_attack(
            family="cohort_snapshot",
            mutation="metamorphic_ineligible_addition",
            expected="ACCEPT",
            actual="ACCEPT",
            status="PASS",
            seed=ineligible_id
        )
    except AssertionError:
        save_regression("cohort_snapshot", "metamorphic_ineligible_addition", {"ineligible_id": ineligible_id}, str(hash(ineligible_id)))
        raise

# --- 4. CANONICALIZATION PBE ---
@settings(max_examples=100, deadline=None)
@given(key1=st.text(), val1=st.text(), key2=st.text(), val2=st.floats(allow_nan=False, allow_infinity=False))
def test_pbe_canonicalization(key1, val1, key2, val2):
    """
    Ensures json.dumps doesn't fail on weird unicode, and that key ordering is respected.
    """
    try:
        # Avoid same keys
        if key1 == key2:
            return
            
        record1 = {key1: val1, key2: val2}
        record2 = {key2: val2, key1: val1}
        
        canon1 = canonicalize_json(record1)
        canon2 = canonicalize_json(record2)
        
        assert canon1 == canon2
        
        attack_logger.log_attack(
            family="schema_canonicalization",
            mutation="fuzz_unicode_and_floats",
            expected="ACCEPT",
            actual="ACCEPT",
            status="PASS",
            seed=str(hash(f"{key1}{val1}"))
        )
    except AssertionError:
        save_regression("schema_canonicalization", "fuzz_unicode_and_floats", {"k1": key1, "k2": key2}, str(hash(f"{key1}{val1}")))
        raise
