"""
Independent Reference Implementation for S0 Adversarial Verification Harness.
This is completely independent of the application logic.
"""
import hashlib
import json
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

def canonicalize_json(record: Dict[str, Any]) -> str:
    """
    Independent reference for Canonical JSON serialization.
    Must ensure deterministic ordering, UTF-8, no extra spaces.
    """
    return json.dumps(record, sort_keys=True, separators=(',', ':'), ensure_ascii=False, allow_nan=False)

def reference_temporal_order(feature_time: datetime, decision_time: datetime, observation_time: datetime, availability_time: datetime) -> bool:
    """
    The grand causal invariant: T_feature_availability <= T_decision < T_observation <= T_availability
    """
    if not all(isinstance(t, datetime) and t.tzinfo is not None for t in [feature_time, decision_time, observation_time, availability_time]):
        return False # All must be timezone aware
        
    return (feature_time <= decision_time) and (decision_time < observation_time) and (observation_time <= availability_time)

def reference_feature_lineage(feature_time: datetime, source_times: List[datetime]) -> bool:
    """
    T_feature_availability >= max(T_source)
    """
    if not source_times:
        return True
    return feature_time >= max(source_times)

def reference_canonical_hash(record: Dict[str, Any], previous_hash: Optional[str]) -> str:
    """
    H_i = SHA256(Canonical(record_i) || H_{i-1})
    """
    record_copy = record.copy()
    record_copy.pop("hash", None)
    record_copy.pop("previous_record_hash", None)
    
    canonical_payload = canonicalize_json(record_copy)
    hasher = hashlib.sha256()
    hasher.update(canonical_payload.encode('utf-8'))
    if previous_hash:
        hasher.update(previous_hash.encode('utf-8'))
    return hasher.hexdigest()

def reference_identity_join(decisions: List[Dict], observations: List[Dict]) -> List[Dict]:
    """
    Reference linker. A decision can only join to exactly 1 observation where:
    property_id matches, AND outcome_available_at > decision_time.
    If multiple match, this is an identity error in S1 context unless explicitly handled.
    For reference, we enforce 0 or 1 strict match.
    """
    linked = []
    for d in decisions:
        matches = [
            o for o in observations 
            if o['property_id'] == d['property_id'] 
            and o['outcome_available_at'] > d['decision_time']
        ]
        if len(matches) > 1:
            raise ValueError("Duplicate valid observations for a single decision")
        
        linked_record = d.copy()
        if matches:
            linked_record.update({'observation': matches[0]})
        else:
            linked_record.update({'observation': None})
        linked.append(linked_record)
    return linked

def reference_population_snapshot(eligible_universe: List[Dict]) -> str:
    """
    Reference snapshot hashing. Deterministic set hash of the eligible universe.
    Sorts by property ID, then canonicalizes and hashes.
    """
    sorted_universe = sorted(eligible_universe, key=lambda x: x['property_id'])
    canonical_list = [canonicalize_json(p) for p in sorted_universe]
    
    hasher = hashlib.sha256()
    for item in canonical_list:
        hasher.update(item.encode('utf-8'))
    return hasher.hexdigest()

def reference_metric_denominators(decisions: List[Dict], linked_observations: List[Dict], cutoff_time: datetime):
    """
    Calculates reference denominators for metrics.
    """
    eligible = len(decisions)
    
    # Selected by LeadGuard
    selected = len([d for d in decisions if d.get('shadow_selected', False)])
    
    # Actually inspected
    inspected = len([d for d in decisions if d.get('actually_inspected', False)])
    
    # Observed (linked)
    observed = len([o for o in linked_observations if o is not None])
    
    # Unknowns
    unknown = len([o for o in linked_observations if o is not None and o.get('material') == 'Unknown'])
    
    # Valid outcomes (Lead, Copper, Galvanized)
    valid_outcomes = len([o for o in linked_observations if o is not None and o.get('material') in ('Lead', 'Copper', 'Galvanized')])
    
    # Censored (no valid outcome AND decision age >= 30 days)
    # We use cutoff_time for 'now'
    censored = 0
    for d, o in zip(decisions, linked_observations):
        d_time = datetime.fromisoformat(d['decision_time'])
        if (cutoff_time - d_time).days >= 30:
            if o is None:
                censored += 1

    return {
        "eligible": eligible,
        "selected": selected,
        "actually_inspected": inspected,
        "observed": observed,
        "unknown": unknown,
        "valid_outcomes": valid_outcomes,
        "censored": censored,
        "cost_denominator": inspected, 
        "evi_denominator": valid_outcomes
    }
