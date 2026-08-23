import copy
import json
from datetime import datetime, timedelta, timezone
from typing import Dict, Any, Callable
from pathlib import Path

import hashlib

class AttackLogger:
    def __init__(self, manifest_path: str = "reports/s0/S0_ATTACK_MANIFEST.jsonl"):
        self.manifest_path = Path(manifest_path)
        self.manifest_path.parent.mkdir(parents=True, exist_ok=True)
        # Frozen harness metadata
        self.harness_version = "1.0.0"
        self.contract_hash = "3080c2e617687c2f3ddc4edca30d8aad31575dc2" # from git rev-parse HEAD at Gate 0
        
        # Clear the manifest on start
        if self.manifest_path.exists():
            self.manifest_path.unlink()

    def _generate_attack_id(self, family: str, mutation: str, expected: str, seed: str) -> str:
        # AttackID = SHA256(contract_hash || family || mutation || parameters || seed)
        hasher = hashlib.sha256()
        payload = f"{self.contract_hash}|{family}|{mutation}|{expected}|{seed}"
        hasher.update(payload.encode('utf-8'))
        # Return a human readable prefix with the deterministic hash
        return f"{family[:4].upper()}-{hasher.hexdigest()[:8]}"

    def log_attack(self, family: str, mutation: str, expected: str, actual: str, status: str, seed: str = "N/A"):
        attack_id = self._generate_attack_id(family, mutation, expected, seed)
        record = {
            "attack_id": attack_id,
            "harness_version": self.harness_version,
            "contract_hash": self.contract_hash,
            "family": family,
            "mutation": mutation,
            "seed": seed,
            "expected_disposition": expected,
            "actual_disposition": actual,
            "status": status,
            "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%f") + "Z"
        }
        with open(self.manifest_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")

attack_logger = AttackLogger()

def mutate_record(record: Dict[str, Any], mutator: Callable[[Dict[str, Any]], None]) -> Dict[str, Any]:
    """Applies a mutation to a copy of a record."""
    mutated = copy.deepcopy(record)
    mutator(mutated)
    return mutated

# Temporal Operators
def shift_time(field: str, delta: timedelta) -> Callable[[Dict[str, Any]], None]:
    def _mutate(r: Dict[str, Any]):
        t = datetime.fromisoformat(r[field])
        r[field] = (t + delta).isoformat()
    return _mutate

def nullify_field(field: str) -> Callable[[Dict[str, Any]], None]:
    def _mutate(r: Dict[str, Any]):
        r[field] = None
    return _mutate

# Identity Operators
def swap_property_id(new_id: str) -> Callable[[Dict[str, Any]], None]:
    def _mutate(r: Dict[str, Any]):
        r["property_id"] = new_id
    return _mutate

def swap_decision_id(new_id: str) -> Callable[[Dict[str, Any]], None]:
    def _mutate(r: Dict[str, Any]):
        r["shadow_decision_id"] = new_id
    return _mutate

# Provenance Operators
def set_field(field: str, value: Any) -> Callable[[Dict[str, Any]], None]:
    def _mutate(r: Dict[str, Any]):
        r[field] = value
    return _mutate

# Structural Operators
def drop_field(field: str) -> Callable[[Dict[str, Any]], None]:
    def _mutate(r: Dict[str, Any]):
        r.pop(field, None)
    return _mutate
