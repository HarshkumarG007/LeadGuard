"""Data models for LeadGuard 2.0 Shadow Mode."""

from pydantic import BaseModel, Field
from typing import Optional, List
from datetime import datetime
from enum import Enum

class EligibilityState(str, Enum):
    ELIGIBLE = "ELIGIBLE"
    WITHHELD = "WITHHELD"
    EXPIRED = "EXPIRED"
    SUPERSEDED = "SUPERSEDED"
    OUTCOME_PENDING = "OUTCOME_PENDING"
    OUTCOME_AVAILABLE = "OUTCOME_AVAILABLE"

class ShadowDecisionRecord(BaseModel):
    """An immutable record of a shadow decision for a single property."""
    
    schema_version: str = "1.0"
    shadow_decision_id: str
    property_id: str
    snapshot_id: str
    
    # Provenance
    model_version: str
    feature_version: str
    calibration_version: str
    calibration_dataset_cutoff: str
    policy_version: str
    policy_parameters_hash: str
    optimizer_version: str
    
    # Beliefs
    p_lead: float
    calibrated_p_lead: float
    uncertainty: float
    evi: float
    expected_utility: float
    
    # Economics
    inspection_cost: float
    intervention_value: float
    
    # Decision
    selected: bool
    rank: int
    eligibility_state: EligibilityState = EligibilityState.ELIGIBLE
    
    # Temporal
    decision_time: datetime
    information_cutoff: datetime
    information_available_at: datetime
    
    # Integrity
    canonical_hash: str
    previous_record_hash: Optional[str] = None
    
    # Metadata
    environment: str = Field("synthetic", description="synthetic or production")
    simulation_version: Optional[str] = None
    random_seed: Optional[int] = None
    ground_truth_version: Optional[str] = None

class ObservationRecord(BaseModel):
    """An immutable record of a delayed outcome."""
    
    schema_version: str = "1.0"
    observation_id: str
    shadow_decision_id: str
    property_id: str
    
    inspection_performed: bool
    inspection_at: Optional[datetime] = None
    label: Optional[bool] = None
    label_available_at: Optional[datetime] = None
    
    intervention_performed: bool
    intervention_at: Optional[datetime] = None
    intervention_cost: Optional[float] = None
    
    observed_lead: Optional[bool] = None
    remediation_completed: bool = False
    remediation_cost: Optional[float] = None
    
    outcome_available_at: datetime
    outcome_source: str
    
    # Metadata
    environment: str = Field("synthetic", description="synthetic or production")
    simulation_version: Optional[str] = None
    random_seed: Optional[int] = None
    ground_truth_version: Optional[str] = None
