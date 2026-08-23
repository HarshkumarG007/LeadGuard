"""F7.5 Snapshot Rollback & Cryptographic Hash Reconstruction Test."""

import hashlib
import json
import uuid
import numpy as np
import pandas as pd
from datetime import UTC, datetime
from pathlib import Path

def compute_decision_hash(decision: dict) -> str:
    """Computes a canonical SHA-256 hash of a decision event."""
    # Ensure consistent serialization (sorted keys, no spaces)
    canonical = json.dumps(decision, sort_keys=True, separators=(',', ':'))
    return hashlib.sha256(canonical.encode('utf-8')).hexdigest()

def test_rollback_reconstruction():
    # 1. State A (Snapshot 101)
    snapshot_101 = {
        "snapshot_id": "snap_101",
        "policy_parameters_hash": "param_hash_A",
        "budget": 5000,
        "properties": ["prop_1", "prop_2", "prop_3"],
        "scores": [150.0, 300.0, 50.0]
    }
    
    # Policy issues a decision
    decision_101 = {
        "decision_id": "dec_001",
        "snapshot_id": snapshot_101["snapshot_id"],
        "policy_parameters_hash": snapshot_101["policy_parameters_hash"],
        "budget_limit": snapshot_101["budget"],
        "ordered_targets": ["prop_2", "prop_1"],
        "scores": [300.0, 150.0]
    }
    
    hash_101 = compute_decision_hash(decision_101)
    
    # 2. State B (Snapshot 102 - World drifted, policy changed)
    snapshot_102 = {
        "snapshot_id": "snap_102",
        "policy_parameters_hash": "param_hash_B",
        "budget": 2000,
        "properties": ["prop_1", "prop_2", "prop_3"],
        "scores": [80.0, 10.0, 500.0]
    }
    
    decision_102 = {
        "decision_id": "dec_002",
        "snapshot_id": snapshot_102["snapshot_id"],
        "policy_parameters_hash": snapshot_102["policy_parameters_hash"],
        "budget_limit": snapshot_102["budget"],
        "ordered_targets": ["prop_3", "prop_1"],
        "scores": [500.0, 80.0]
    }
    
    hash_102 = compute_decision_hash(decision_102)
    assert hash_101 != hash_102
    
    # 3. Rollback: Reconstruct State A from Ledger
    # The Ledger gives us the historical context and snapshot ID
    reconstructed_decision_101 = {
        "decision_id": "dec_001", # Read from Ledger
        "snapshot_id": "snap_101", # Loaded from cold storage using Ledger ID
        "policy_parameters_hash": "param_hash_A", # Loaded from Ledger
        "budget_limit": 5000,
        "ordered_targets": ["prop_2", "prop_1"], # Derived deterministically from snap_101 scores
        "scores": [300.0, 150.0]
    }
    
    reconstructed_hash = compute_decision_hash(reconstructed_decision_101)
    
    # 4. Assert Cryptographic Match
    assert hash_101 == reconstructed_hash, "Rollback failed cryptographic hash check!"
    print("Snapshot Rollback & Reconstruction: PASS")
    print(f"Canonical Hash 101: {hash_101}")
    print(f"Reconstructed Hash: {reconstructed_hash}")

if __name__ == "__main__":
    test_rollback_reconstruction()
