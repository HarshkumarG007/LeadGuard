"""Append-only ledger logic for Shadow Decisions."""

import json
import hashlib
from pathlib import Path
from leadguard.shadow.schemas import ShadowDecisionRecord, ObservationRecord

def get_ledger_path(environment: str, record_type: str) -> Path:
    assert environment in ("production", "synthetic")
    assert record_type in ("decisions", "observations")
    
    path = Path(f"data/processed/shadow/{environment}/{record_type}.jsonl")
    path.parent.mkdir(parents=True, exist_ok=True)
    return path

def _hash_payload(payload: dict) -> str:
    canonical = json.dumps(payload, sort_keys=True, separators=(',', ':'))
    return hashlib.sha256(canonical.encode('utf-8')).hexdigest()

def record_shadow_decision(record: ShadowDecisionRecord):
    """Writes an immutable shadow decision to the correct JSONL ledger."""
    path = get_ledger_path(record.environment, "decisions")
    
    payload = record.model_dump(mode="json")
    
    # Calculate canonical hash for integrity
    hash_payload = payload.copy()
    hash_payload.pop("canonical_hash", None)
    payload["canonical_hash"] = _hash_payload(hash_payload)
    
    with open(path, "a") as f:
        f.write(json.dumps(payload) + "\n")

def record_observation(record: ObservationRecord):
    """Writes an immutable observation to the correct JSONL ledger."""
    path = get_ledger_path(record.environment, "observations")
    payload = record.model_dump(mode="json")
    
    with open(path, "a") as f:
        f.write(json.dumps(payload) + "\n")
