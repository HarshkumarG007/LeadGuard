import pytest
import math
from tests.s0.adversarial.reference import canonicalize_json

def test_canonicalize_json_key_order():
    record1 = {"b": 2, "a": 1, "c": 3}
    record2 = {"c": 3, "b": 2, "a": 1}
    assert canonicalize_json(record1) == canonicalize_json(record2)
    assert canonicalize_json(record1) == '{"a":1,"b":2,"c":3}'

def test_canonicalize_json_spacing():
    record1 = {"a": 1, "b": 2}
    # Testing that serialization has no spaces
    assert canonicalize_json(record1) == '{"a":1,"b":2}'

def test_canonicalize_json_utf8():
    # Test ensure_ascii=False
    record1 = {"name": "Café"}
    assert canonicalize_json(record1) == '{"name":"Café"}'

def test_canonicalize_json_nested():
    record1 = {"outer": {"z": 1, "y": 2}, "list": [3, 1, 2]}
    record2 = {"list": [3, 1, 2], "outer": {"y": 2, "z": 1}}
    assert canonicalize_json(record1) == canonicalize_json(record2)
    assert canonicalize_json(record1) == '{"list":[3,1,2],"outer":{"y":2,"z":1}}'

def test_canonicalize_null_vs_absent():
    # Null and absent MUST hash differently
    record1 = {"a": 1, "b": None}
    record2 = {"a": 1}
    assert canonicalize_json(record1) != canonicalize_json(record2)

def test_canonicalize_negative_zero():
    # Canonical JSON usually normalizes -0.0 to 0.0 or at least they should be considered if we want exact bytes.
    # Python json module handles -0.0 as "-0.0". 
    # For now, ensure we understand the strict string representation.
    record1 = {"a": -0.0}
    record2 = {"a": 0.0}
    # In python json dumps, -0.0 becomes "-0.0"
    assert canonicalize_json(record1) != canonicalize_json(record2)

def test_canonicalize_nan_infinity_rejected():
    # JSON standard does not support NaN or Infinity. Python's json does by default.
    # We should strictly enforce allow_nan=False to prevent structural corruption.
    with pytest.raises(ValueError):
        canonicalize_json({"a": math.nan})
        
    with pytest.raises(ValueError):
        canonicalize_json({"a": math.inf})
