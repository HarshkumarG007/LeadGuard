import pytest
import os
import json
from pathlib import Path
from leadguard.data.ledger import ImmutableLedger, LedgerEntry

@pytest.fixture
def temp_ledger(tmp_path):
    ledger_path = tmp_path / "decision_ledger.jsonl"
    ledger = ImmutableLedger(log_path=ledger_path)
    
    # Add 3 valid events
    ledger.append_event(LedgerEntry(event_type="DecisionIssued", payload={"msg": "first"}))
    ledger.append_event(LedgerEntry(event_type="InspectionCompleted", payload={"msg": "second"}))
    ledger.append_event(LedgerEntry(event_type="DecisionIssued", payload={"msg": "third"}))
    
    return ledger, ledger_path


def test_verify_chain_valid(temp_ledger):
    ledger, _ = temp_ledger
    assert ledger.verify_chain() is True


def test_event_mutation(temp_ledger):
    ledger, path = temp_ledger
    
    # Read, mutate, write back
    lines = path.read_text().splitlines()
    data = json.loads(lines[1])
    data["payload"]["msg"] = "hacked"
    lines[1] = json.dumps(data)
    
    path.write_text("\n".join(lines) + "\n")
    
    # Validation should catch hash mismatch
    assert ledger.verify_chain() is False


def test_event_deletion(temp_ledger):
    ledger, path = temp_ledger
    
    # Read, delete middle event, write back
    lines = path.read_text().splitlines()
    del lines[1]
    
    path.write_text("\n".join(lines) + "\n")
    
    # Validation should catch broken link (prev_hash mismatch)
    assert ledger.verify_chain() is False


def test_event_reordering(temp_ledger):
    ledger, path = temp_ledger
    
    # Read, swap last two events, write back
    lines = path.read_text().splitlines()
    lines[1], lines[2] = lines[2], lines[1]
    
    path.write_text("\n".join(lines) + "\n")
    
    # Validation should catch broken link
    assert ledger.verify_chain() is False
