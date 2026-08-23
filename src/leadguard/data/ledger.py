"""Immutable Decision Ledger for active learning and policy decisions.

Implements an append-only, tamper-evident ledger for all decisions.
"""

from __future__ import annotations

import hashlib
import json
import logging
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal
from uuid import uuid4

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


EventType = Literal[
    "DecisionIssued",
    "InspectionCompleted",
    "OutcomeAvailable",
    "ModelPromoted",
    "PolicyChanged"
]


class LedgerEntry(BaseModel):
    """Canonical schema for a single ledger event."""
    
    event_id: str = Field(default_factory=lambda: str(uuid4()))
    event_type: EventType
    timestamp: str = Field(default_factory=lambda: datetime.now(UTC).isoformat())
    entity_id: str | None = None
    decision_id: str | None = None
    
    # Version Tracking
    model_version: str | None = None
    feature_version: str | None = None
    policy_version: str | None = None
    policy_parameters_hash: str | None = None
    
    # Arbitrary payload
    payload: dict[str, Any] = Field(default_factory=dict)
    
    # Tamper evidence
    previous_event_hash: str | None = None
    event_hash: str | None = None
    
    def compute_hash(self) -> str:
        """Compute the hash of the current event excluding the event_hash itself."""
        data = self.model_dump(exclude={"event_hash"})
        data_str = json.dumps(data, sort_keys=True)
        return hashlib.sha256(data_str.encode("utf-8")).hexdigest()
        

class ImmutableLedger:
    """Append-only ledger backed by a JSONL file."""
    
    def __init__(self, log_path: Path | str = "data/processed/decision_ledger.jsonl"):
        self.log_path = Path(log_path)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Determine the last hash
        self.last_hash: str | None = None
        self._initialize_last_hash()
        
    def _initialize_last_hash(self) -> None:
        """Read the last entry to recover the hash chain."""
        if not self.log_path.exists():
            return
            
        try:
            with open(self.log_path, "r", encoding="utf-8") as f:
                last_line = None
                for line in f:
                    if line.strip():
                        last_line = line
                        
                if last_line:
                    data = json.loads(last_line)
                    self.last_hash = data.get("event_hash")
        except Exception as e:
            logger.error("Failed to recover ledger hash chain: %s", e)
            
    def append_event(self, entry: LedgerEntry) -> LedgerEntry:
        """Append a new event, linking it to the hash chain."""
        entry.previous_event_hash = self.last_hash
        entry.event_hash = entry.compute_hash()
        
        with open(self.log_path, "a", encoding="utf-8") as f:
            f.write(entry.model_dump_json() + "\n")
            
        self.last_hash = entry.event_hash
        return entry

    def get_all_events(self) -> list[LedgerEntry]:
        """Read all events from the ledger."""
        if not self.log_path.exists():
            return []
            
        events = []
        with open(self.log_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    events.append(LedgerEntry.model_validate_json(line))
        return events

    def verify_chain(self) -> bool:
        """Verify the cryptographic integrity of the ledger chain.
        
        Returns False if any event was mutated, deleted, or reordered.
        """
        events = self.get_all_events()
        if not events:
            return True
            
        expected_prev_hash = None
        for i, event in enumerate(events):
            # 1. Check ordering linkage
            if event.previous_event_hash != expected_prev_hash:
                logger.error(f"Chain broken at index {i}: Expected prev_hash {expected_prev_hash}, got {event.previous_event_hash}")
                return False
                
            # 2. Check mutation
            computed_hash = event.compute_hash()
            if computed_hash != event.event_hash:
                logger.error(f"Mutation detected at index {i}: Computed {computed_hash}, stored {event.event_hash}")
                return False
                
            expected_prev_hash = event.event_hash
            
        return True
