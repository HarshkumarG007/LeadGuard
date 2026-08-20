"""LeadGuard FastAPI application.

Implements all 7 endpoints from Architecture §8:
  GET  /v1/health
  POST /v1/predict
  GET  /v1/properties/{property_id}/prediction
  GET  /v1/priority-queue
  POST /v1/inspections
  GET  /v1/fairness-report
  GET  /v1/model/metadata

Error convention: all errors return {\"error\": ..., \"detail\": ...} with
the appropriate HTTP status. No bare stack traces are returned to callers.
"""

from __future__ import annotations

import hashlib
import json
import logging
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Annotated

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import JSONResponse

from api.model_loader import ModelState, get_state
from api.schemas import (
    ErrorResponse,
    FairnessReportResponse,
    HealthResponse,
    InspectionSubmitRequest,
    InspectionSubmitResponse,
    ModelMetadataResponse,
    PredictRequest,
    PredictResponse,
    PredictionResult,
    PriorityQueueItem,
    PriorityQueueResponse,
    SHAPFeature,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# App initialization
# ---------------------------------------------------------------------------

app = FastAPI(
    title="LeadGuard API",
    description="Lead service line prediction, uncertainty quantification, and inspection prioritization",
    version="0.1.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# In-memory inspection store (per-session, not persistent on HF free tier)
_inspection_store: list[dict] = []

# Properties store (loaded lazily from sample/processed data)
_properties_cache: pd.DataFrame | None = None


def _get_properties() -> pd.DataFrame:
    """Load properties from disk, cached in memory.

    Returns:
        DataFrame of properties with all feature columns.
    """
    global _properties_cache
    if _properties_cache is not None:
        return _properties_cache

    for path in [
        Path("data/processed/features.parquet"),
        Path("data/processed/features_sample.parquet"),
        Path("data/sample/sample_properties.parquet"),
    ]:
        if path.exists():
            _properties_cache = pd.read_parquet(path)
            logger.info("Properties loaded from %s (%d rows)", path, len(_properties_cache))
            return _properties_cache

    _properties_cache = pd.DataFrame()
    logger.warning("No properties data found; prediction endpoints will be limited")
    return _properties_cache


def _model_version(state: ModelState) -> str:
    """Build a model version string.

    Args:
        state: Loaded model state.

    Returns:
        Version string.
    """
    return state.metrics.get("model_version", f"xgb-{datetime.now(timezone.utc).strftime('%Y.%m.%d')}")


def _predict_single(
    property_row: pd.Series,
    state: ModelState,
    features_df: pd.DataFrame | None = None,
) -> dict:
    """Generate a full prediction for one property.

    Args:
        property_row: A single property Series with all feature columns.
        state: Loaded model state.
        features_df: Full features DataFrame (for equity_boost context).

    Returns:
        Prediction dictionary matching PredictionResult schema.
    """
    import shap  # noqa: PLC0415
    from leadguard.evaluation.explainability import extract_top_shap_features  # noqa: PLC0415
    from leadguard.evaluation.fairness import compute_equity_boost  # noqa: PLC0415
    from leadguard.models.active_learning import compute_priority_score  # noqa: PLC0415
    from leadguard.models.uncertainty import (  # noqa: PLC0415
        _uncertainty_from_set_size,
        MATERIALS,
    )

    feature_names = state.feature_names
    X = property_row.reindex(feature_names, fill_value=0.0).values.astype(float).reshape(1, -1)

    # P(Lead)
    proba = state.model.predict_proba(X)[0]  # shape (C,)
    p_lead = float(proba[-1])  # last column is Lead in binary classifier

    # Conformal set
    if state.conformal_global is not None:
        score = 1.0 - proba.max()
        if score <= state.conformal_global.threshold_:
            conformal_set = [MATERIALS[int(np.argmax(proba))]]
        else:
            conformal_set = MATERIALS.copy()
    else:
        conformal_set = [MATERIALS[int(np.argmax(proba))]]

    # Uncertainty score
    set_size = np.array([len(conformal_set)])
    uncertainty = float(_uncertainty_from_set_size(set_size, k=len(MATERIALS))[0])

    # Equity boost
    census_tract = property_row.get("census_tract", None)
    equity_boost = 0.0
    if census_tract and not state.fairness_reference.empty and features_df is not None:
        all_pred = features_df[["census_tract"]].copy()
        all_pred["p_lead_calibrated"] = state.model.predict_proba(
            features_df.reindex(columns=feature_names, fill_value=0.0).astype(float).values
        )[:, -1]
        inspections_df = pd.DataFrame(
            [{"census_tract": i["census_tract"]} for i in _inspection_store if "census_tract" in i]
        )
        boost_series = compute_equity_boost(all_pred, inspections_df)
        equity_boost = float(boost_series.get(census_tract, 0.0))

    # Priority score
    priority = compute_priority_score(
        np.array([p_lead]),
        np.array([uncertainty]),
        np.array([equity_boost]),
    )[0]

    # SHAP top features
    try:
        explainer = shap.TreeExplainer(state.model)
        sv = explainer.shap_values(X)
        shap_row = sv[0] if sv.ndim > 1 else sv
        top_features = extract_top_shap_features(shap_row, feature_names, top_n=5)
    except Exception as e:
        logger.warning("SHAP failed for property %s: %s", property_row.get("property_id"), e)
        top_features = []

    return {
        "property_id": str(property_row.get("property_id", "unknown")),
        "p_lead_calibrated": round(p_lead, 4),
        "conformal_set": conformal_set,
        "confidence_level": 0.90,
        "uncertainty_score": round(uncertainty, 4),
        "priority_score": round(float(priority), 4),
        "shap_top_features": [SHAPFeature(**f) for f in top_features],
        "model_version": _model_version(state),
        "predicted_at": datetime.now(timezone.utc),
    }


def _error_503(detail: str) -> JSONResponse:
    """Return a 503 response with structured error body.

    Args:
        detail: Human-readable error detail.

    Returns:
        JSONResponse with 503 status.
    """
    return JSONResponse(
        status_code=503,
        content={"error": "Service unavailable", "detail": detail},
    )


def _error_404(detail: str) -> HTTPException:
    """Raise a 404 HTTPException.

    Args:
        detail: Error detail message.

    Returns:
        HTTPException with 404 status.
    """
    return HTTPException(status_code=404, detail={"error": "Not found", "detail": detail})


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@app.get("/v1/health", response_model=HealthResponse, tags=["system"])
async def health() -> HealthResponse:
    """Liveness and readiness check.

    Returns 503 if the model artifacts failed to load.
    """
    state = get_state()
    if not state.is_ready:
        return JSONResponse(  # type: ignore[return-value]
            status_code=503,
            content={
                "status": "degraded",
                "model_loaded": False,
                "conformal_loaded": state.conformal_ready,
                "fairness_ref_loaded": state.fairness_ready,
                "errors": state.load_errors,
            },
        )
    return HealthResponse(
        status="ok",
        model_loaded=state.is_ready,
        conformal_loaded=state.conformal_ready,
        fairness_ref_loaded=state.fairness_ready,
    )


@app.post("/v1/predict", response_model=PredictResponse, tags=["prediction"])
async def batch_predict(request: PredictRequest) -> PredictResponse:
    """Batch predict for a list of property IDs.

    Returns calibrated probability, conformal set, uncertainty score,
    priority score, and top SHAP features for each property.
    """
    state = get_state()
    if not state.is_ready:
        return _error_503("Model not loaded")  # type: ignore[return-value]

    props = _get_properties()
    if props.empty:
        return _error_503("Property data not available")  # type: ignore[return-value]

    results = []
    for pid in request.property_ids:
        row_mask = props["property_id"] == pid
        if not row_mask.any():
            # Unknown property — return 404 per Architecture §8 error convention
            raise _error_404(f"property_id '{pid}' not found")
        row = props[row_mask].iloc[0]
        pred = _predict_single(row, state, features_df=props)
        results.append(PredictionResult(**pred))

    return PredictResponse(predictions=results)


@app.get(
    "/v1/properties/{property_id}/prediction",
    response_model=PredictionResult,
    tags=["prediction"],
)
async def single_prediction(property_id: str) -> PredictionResult:
    """Single prediction with full SHAP explanation."""
    state = get_state()
    if not state.is_ready:
        return _error_503("Model not loaded")  # type: ignore[return-value]

    props = _get_properties()
    row_mask = props["property_id"] == property_id
    if not row_mask.any():
        raise _error_404(f"property_id '{property_id}' not found")

    row = props[row_mask].iloc[0]
    pred = _predict_single(row, state, features_df=props)
    return PredictionResult(**pred)


@app.get("/v1/priority-queue", response_model=PriorityQueueResponse, tags=["queue"])
async def priority_queue(
    budget_usd: Annotated[float, Query(ge=0)] = 100000.0,
    limit: Annotated[int, Query(ge=1, le=10000)] = 500,
    cost_per_inspection: Annotated[float, Query(ge=0)] = 500.0,
) -> PriorityQueueResponse:
    """Ranked inspection queue under a budget constraint."""
    state = get_state()
    if not state.is_ready:
        return _error_503("Model not loaded")  # type: ignore[return-value]

    props = _get_properties()
    if props.empty:
        return _error_503("Property data not available")  # type: ignore[return-value]

    from leadguard.evaluation.fairness import compute_equity_boost  # noqa: PLC0415
    from leadguard.models.active_learning import compute_priority_score  # noqa: PLC0415
    from leadguard.models.uncertainty import _uncertainty_from_set_size, MATERIALS  # noqa: PLC0415

    feature_names = state.feature_names
    X_all = props.reindex(columns=feature_names, fill_value=0.0).astype(float).values
    probas = state.model.predict_proba(X_all)
    p_lead_all = probas[:, -1]

    # Uncertainty
    if state.conformal_global is not None:
        scores = 1.0 - probas.max(axis=1)
        thr = state.conformal_global.threshold_
        set_sizes = np.where(scores <= thr, 1, len(MATERIALS))
    else:
        set_sizes = np.ones(len(props), dtype=int)
    uncertainty_all = _uncertainty_from_set_size(set_sizes, k=len(MATERIALS))

    # Equity boost
    pred_df = props[["census_tract"]].copy()
    pred_df["p_lead_calibrated"] = p_lead_all
    inspections_df = pd.DataFrame(
        [{"census_tract": i.get("census_tract", "")} for i in _inspection_store]
    )
    boost_series = compute_equity_boost(pred_df, inspections_df)
    equity_all = props["census_tract"].map(boost_series).fillna(0.0).values

    priority_all = compute_priority_score(p_lead_all, uncertainty_all, equity_all)

    # Sort and apply budget
    sorted_idx = np.argsort(priority_all)[::-1]
    items = []
    remaining_budget = budget_usd
    for rank, idx in enumerate(sorted_idx[:limit], start=1):
        if remaining_budget < cost_per_inspection:
            break
        row = props.iloc[idx]
        items.append(PriorityQueueItem(
            rank=rank,
            property_id=str(row.get("property_id", "")),
            address=str(row.get("address", "")) or None,
            priority_score=round(float(priority_all[idx]), 4),
            p_lead_calibrated=round(float(p_lead_all[idx]), 4),
            uncertainty_score=round(float(uncertainty_all[idx]), 4),
            conformal_set=[MATERIALS[int(np.argmax(probas[idx]))]] if int(set_sizes[idx]) == 1 else MATERIALS,
            estimated_cost_usd=cost_per_inspection,
        ))
        remaining_budget -= cost_per_inspection

    return PriorityQueueResponse(
        budget_usd=budget_usd,
        total_properties_ranked=len(sorted_idx),
        properties_within_budget=len(items),
        items=items,
    )


@app.post("/v1/inspections", response_model=InspectionSubmitResponse, tags=["feedback"])
async def submit_inspection(request: InspectionSubmitRequest) -> InspectionSubmitResponse:
    """Submit a new ground-truth inspection result (feedback loop entry)."""
    props = _get_properties()
    census_tract = None
    if not props.empty:
        row_mask = props["property_id"] == request.property_id
        if row_mask.any():
            census_tract = props[row_mask].iloc[0].get("census_tract")

    inspection_id = "insp-" + hashlib.sha256(
        f"{request.property_id}{datetime.now(timezone.utc).isoformat()}".encode()
    ).hexdigest()[:12]

    inspection = {
        "inspection_id": inspection_id,
        "property_id": request.property_id,
        "inspected_material": request.inspected_material,
        "inspected_at": (request.inspected_at or datetime.now(timezone.utc)).isoformat(),
        "source": request.source,
        "cost_usd": request.cost_usd,
        "census_tract": census_tract,
        "used_in_training": False,
    }
    _inspection_store.append(inspection)
    logger.info("Inspection recorded: %s for property %s", inspection_id, request.property_id)

    return InspectionSubmitResponse(
        inspection_id=inspection_id,
        property_id=request.property_id,
        inspected_material=request.inspected_material,
        inspected_at=request.inspected_at or datetime.now(timezone.utc),
        message=f"Inspection recorded. Total inspections this session: {len(_inspection_store)}",
    )


@app.get("/v1/fairness-report", response_model=FairnessReportResponse, tags=["fairness"])
async def fairness_report() -> FairnessReportResponse:
    """Latest fairness audit metrics."""
    report_path = Path("reports/fairness_report.json")
    if not report_path.exists():
        return _error_503("Fairness report not yet generated. Run Phase 6 (evaluation/fairness.py) first.")  # type: ignore[return-value]

    report = json.loads(report_path.read_text())
    return FairnessReportResponse(
        **report,
        generated_at=datetime.fromtimestamp(report_path.stat().st_mtime, tz=timezone.utc),
    )


@app.get("/v1/model/metadata", response_model=ModelMetadataResponse, tags=["system"])
async def model_metadata() -> ModelMetadataResponse:
    """Active model version, training date, and headline metrics."""
    state = get_state()
    metrics = state.metrics

    return ModelMetadataResponse(
        model_version=_model_version(state),
        training_date=metrics.get("training_date"),
        pr_auc_geo=metrics.get("pr_auc_geo"),
        pr_auc_random=metrics.get("pr_auc_random"),
        features=state.feature_names,
        confidence_level=0.90,
    )
