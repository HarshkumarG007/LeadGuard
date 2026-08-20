"""Pydantic schemas for the LeadGuard API.

Defines all request and response models for Architecture §8 endpoints.
Every field matches the Prediction and Inspection entities from Architecture §6.2.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# Shared sub-models
# ---------------------------------------------------------------------------


class SHAPFeature(BaseModel):
    """A single SHAP feature contribution."""

    feature: str = Field(..., description="Feature name")
    contribution: float = Field(..., description="SHAP contribution value")


# ---------------------------------------------------------------------------
# Prediction models
# ---------------------------------------------------------------------------


class PredictRequest(BaseModel):
    """Request body for POST /v1/predict."""

    property_ids: list[str] = Field(
        ..., min_length=1, max_length=10000, description="List of property IDs to predict"
    )


class PredictionResult(BaseModel):
    """A single prediction result (Architecture §6.2 Prediction entity)."""

    property_id: str
    p_lead_calibrated: float = Field(..., ge=0.0, le=1.0)
    conformal_set: list[str] = Field(..., description="Materials not ruled out at confidence_level")
    confidence_level: float = Field(default=0.90, ge=0.0, le=1.0)
    uncertainty_score: float = Field(..., ge=0.0, le=1.0)
    priority_score: float = Field(..., ge=0.0, le=1.0)
    shap_top_features: list[SHAPFeature]
    model_version: str
    predicted_at: datetime


class PredictResponse(BaseModel):
    """Response body for POST /v1/predict."""

    predictions: list[PredictionResult]


class PropertyPredictionResponse(PredictionResult):
    """Full prediction for GET /v1/properties/{property_id}/prediction."""

    pass


# ---------------------------------------------------------------------------
# Priority queue
# ---------------------------------------------------------------------------


class PriorityQueueItem(BaseModel):
    """A single item in the inspection priority queue."""

    rank: int
    property_id: str
    address: str | None = None
    priority_score: float
    p_lead_calibrated: float
    uncertainty_score: float
    conformal_set: list[str]
    estimated_cost_usd: float


class PriorityQueueResponse(BaseModel):
    """Response for GET /v1/priority-queue."""

    budget_usd: float
    total_properties_ranked: int
    properties_within_budget: int
    items: list[PriorityQueueItem]


# ---------------------------------------------------------------------------
# Inspections (feedback loop)
# ---------------------------------------------------------------------------


class InspectionSubmitRequest(BaseModel):
    """Request body for POST /v1/inspections."""

    property_id: str
    inspected_material: str = Field(..., pattern="^(Lead|Copper|Galvanized|Unknown)$")
    source: str = Field(
        default="field_inspection", pattern="^(field_inspection|self_report_verified)$"
    )
    cost_usd: float = Field(default=500.0, ge=0.0)
    inspected_at: datetime | None = None


class InspectionSubmitResponse(BaseModel):
    """Response for POST /v1/inspections."""

    inspection_id: str
    property_id: str
    inspected_material: str
    inspected_at: datetime
    message: str


# ---------------------------------------------------------------------------
# Fairness report
# ---------------------------------------------------------------------------


class FairnessReportResponse(BaseModel):
    """Response for GET /v1/fairness-report."""

    fnr_by_quartile: dict[str, Any]
    fnr_disparity_pp: float
    disparity_flagged: bool
    equity_boost_sample: dict[str, Any]
    n_labeled_properties: int
    n_properties_with_quartile: int
    generated_at: datetime


# ---------------------------------------------------------------------------
# Model metadata
# ---------------------------------------------------------------------------


class ModelMetadataResponse(BaseModel):
    """Response for GET /v1/model/metadata."""

    model_version: str
    training_date: str | None
    pr_auc_geo: float | None
    pr_auc_random: float | None
    features: list[str]
    confidence_level: float


# ---------------------------------------------------------------------------
# Health check
# ---------------------------------------------------------------------------


class HealthResponse(BaseModel):
    """Response for GET /v1/health."""

    status: str = Field(..., description="'ok' or 'degraded'")
    model_loaded: bool
    conformal_loaded: bool
    fairness_ref_loaded: bool


# ---------------------------------------------------------------------------
# Error response
# ---------------------------------------------------------------------------


class ErrorResponse(BaseModel):
    """Standard error body (Architecture §8 error convention)."""

    error: str
    detail: str
