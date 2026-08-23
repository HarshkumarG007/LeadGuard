import pytest
from datetime import datetime, timezone, timedelta
import uuid

@pytest.fixture
def base_property_id():
    return str(uuid.uuid4())

@pytest.fixture
def base_decision_id():
    return str(uuid.uuid4())

@pytest.fixture
def valid_times():
    base_time = datetime(2026, 8, 1, 12, 0, tzinfo=timezone.utc)
    return {
        "feature_availability": base_time - timedelta(days=2),
        "decision": base_time,
        "observation": base_time + timedelta(days=35),  # 35 days later (past operational horizon)
        "availability": base_time + timedelta(days=36)
    }

@pytest.fixture
def valid_decision_record(base_property_id, base_decision_id, valid_times):
    return {
        "shadow_decision_id": base_decision_id,
        "property_id": base_property_id,
        "decision_time": valid_times["decision"].isoformat(),
        "information_available_at": valid_times["feature_availability"].isoformat(),
        "model_version": "v1.0",
        "model_hash": "dummy_model_hash",
        "feature_schema_version": "v1.0",
        "calibration_version": "v1.0",
        "calibration_hash": "dummy_calib_hash",
        "policy_version": "v1.0",
        "decision_population_snapshot_hash": "dummy_pop_hash",
        "shadow_selected": True,
        "actually_inspected": True,
        "selection_source": "leadguard",
        "reason_not_inspected": None,
        "p_lead": 0.8,
        "calibrated_p_lead": 0.85,
        "uncertainty": 0.1,
        "evi": 1200.0,
        "expected_utility": 1100.0,
        "inspection_cost": 250.0,
        "intervention_value": 5000.0,
        "selected": True,
        "rank": 1,
        "environment": "production"
    }

@pytest.fixture
def valid_observation_record(base_property_id, base_decision_id, valid_times):
    return {
        "shadow_decision_id": base_decision_id,
        "property_id": base_property_id,
        "outcome_available_at": valid_times["availability"].isoformat(),
        "label_available_at": valid_times["availability"].isoformat(),
        "material": "Lead",
        "outcome_source": "field_inspection",
        "outcome_version": "v1",
        "environment": "production"
    }
