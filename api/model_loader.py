"""Model loader for LeadGuard API.

Loads all model artifacts and reference tables exactly once at startup.
Per Architecture §8: artifacts must be loaded at startup, not per-request.

A failed load does NOT crash the process — instead, the health endpoint
returns 503 and all prediction endpoints return 503 with a structured error.
"""

from __future__ import annotations

import json
import logging
import pickle
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class ModelState:
    """Container for all loaded model artifacts.

    Attributes:
        model: Fitted XGBoost model (or None if load failed).
        conformal_global: Global conformal predictor.
        conformal_by_quartile: Mondrian conformal predictor per quartile.
        fairness_reference: Census tract → income quartile mapping.
        feature_names: Ordered list of feature column names.
        model_version: Version string (git SHA or semantic).
        metrics: Training metrics dictionary.
        load_errors: List of load failures (empty if all OK).
    """

    model: object = None
    conformal_global: object = None
    conformal_by_quartile: object = None
    fairness_reference: pd.DataFrame = field(default_factory=pd.DataFrame)
    feature_names: list[str] = field(default_factory=list)
    model_version: str = "unknown"
    metrics: dict = field(default_factory=dict)
    load_errors: list[str] = field(default_factory=list)
    shap_explainer: object = None

    @property
    def is_ready(self) -> bool:
        """True if minimum artifacts required for prediction are loaded."""
        return self.model is not None and len(self.feature_names) > 0

    @property
    def conformal_ready(self) -> bool:
        """True if conformal predictors are loaded."""
        return self.conformal_global is not None

    @property
    def fairness_ready(self) -> bool:
        """True if fairness reference is loaded."""
        return not self.fairness_reference.empty


# ---------------------------------------------------------------------------
# Singleton state — loaded once at process startup
# ---------------------------------------------------------------------------

_state: ModelState | None = None


def get_state() -> ModelState:
    """Return the loaded model state, initializing on first call.

    Returns:
        The global ModelState instance.
    """
    global _state
    if _state is None:
        _state = load_artifacts()
    return _state


def load_artifacts(
    model_dir: Path | str = "models/xgboost",
    fairness_ref_path: Path | str = "data/fairness_reference.parquet",
) -> ModelState:
    """Load all model artifacts from disk.

    Failures are recorded in state.load_errors rather than raising,
    so the API process can start and return 503 gracefully.

    Args:
        model_dir: Directory containing model.json, conformal pickles, metrics.json.
        fairness_ref_path: Path to fairness reference parquet.

    Returns:
        Populated ModelState.
    """
    import xgboost as xgb  # noqa: PLC0415

    state = ModelState()
    model_dir = Path(model_dir)

    import pickle
    
    # Load Metadata first to validate compatibility
    metadata_path = model_dir / "metadata.json"
    if metadata_path.exists():
        try:
            metadata = json.loads(metadata_path.read_text())
            state.feature_names = metadata.get("feature_names", [])
            state.model_version = metadata.get("model_version", "unknown")
            logger.info("Metadata loaded: feature_version=%s", metadata.get("feature_version"))
        except Exception as e:
            logger.warning("Could not load metadata: %s", e)
            state.load_errors.append(f"Metadata load failed: {e}")
    else:
        logger.warning("No metadata.json found in %s", model_dir)
        state.load_errors.append("metadata.json missing")

    # Load Calibrated XGBoost model
    try:
        import sys
        sys.path.insert(0, str(Path("src").absolute()))
        from leadguard.models.serving import load_serving_model
        
        state.model = load_serving_model(model_dir)
        logger.info("Calibrated model loaded from %s", model_dir)
        
        try:
            import shap
            # Initialize explainer on the raw XGBoost model to save time later
            # (assuming model is calibrated, the base estimator is often the target for SHAP)
            # Actually, TreeExplainer needs the raw booster
            if hasattr(state.model, "estimator"):
                # if it's a CalibratedClassifierCV
                raw_model = state.model.estimator
            else:
                raw_model = state.model
            state.shap_explainer = shap.TreeExplainer(raw_model)
            logger.info("SHAP TreeExplainer initialized and cached.")
        except Exception as shap_err:
            logger.warning("Failed to initialize SHAP TreeExplainer: %s", shap_err)

    except Exception as e:
        msg = f"Failed to load serving model: {e}"
        logger.error(msg)
        state.load_errors.append(msg)

    # Load feature names from metrics (fallback)
    metrics_path = model_dir / "metrics.json"
    if metrics_path.exists() and not state.feature_names:
        try:
            state.metrics = json.loads(metrics_path.read_text())
            state.feature_names = state.metrics.get("features", [])
            state.model_version = state.metrics.get("model_version", "unknown")
            logger.info("Features fallback loaded from metrics: %d features", len(state.feature_names))
        except Exception as e:
            logger.warning("Could not load metrics: %s", e)

    # Fallback feature names if metrics missing
    if not state.feature_names:
        try:
            from leadguard.models.xgboost_model import XGB_FEATURES  # noqa: PLC0415

            state.feature_names = XGB_FEATURES
        except ImportError:
            state.feature_names = []

    # Load global conformal predictor
    cp_path = model_dir / "conformal_global.pkl"
    if cp_path.exists():
        try:
            with cp_path.open("rb") as f:
                state.conformal_global = pickle.load(f)
            logger.info("Global conformal predictor loaded")
        except Exception as e:
            msg = f"Failed to load conformal_global: {e}"
            logger.warning(msg)
            state.load_errors.append(msg)

    # Load Mondrian conformal predictor
    mcp_path = model_dir / "conformal_by_quartile.pkl"
    if mcp_path.exists():
        try:
            with mcp_path.open("rb") as f:
                state.conformal_by_quartile = pickle.load(f)
            logger.info("Mondrian conformal predictor loaded")
        except Exception as e:
            logger.warning("Failed to load conformal_by_quartile: %s", e)

    # Load fairness reference
    fr_path = Path(fairness_ref_path)
    if fr_path.exists():
        try:
            state.fairness_reference = pd.read_parquet(fr_path)
            logger.info("Fairness reference loaded: %d tracts", len(state.fairness_reference))
        except Exception as e:
            logger.warning("Failed to load fairness_reference: %s", e)

    logger.info(
        "ModelState ready=%s, conformal=%s, fairness=%s, errors=%d",
        state.is_ready,
        state.conformal_ready,
        state.fairness_ready,
        len(state.load_errors),
    )
    return state


def reset_state() -> None:
    """Reset the global model state (used in tests)."""
    global _state
    _state = None
