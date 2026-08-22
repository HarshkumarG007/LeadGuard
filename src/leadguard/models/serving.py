"""Canonical model serving and probability contract.

This module provides exactly one definition of target encoding,
model loading, and probability prediction. All other modules (API,
uncertainty, evaluation, explainability) must consume this contract.

Implements the invariant:
    Lead -> 1
    Copper/Galvanized -> 0
    Unknown -> excluded
"""

from __future__ import annotations

import logging
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# The strict ordered classes for the model probability output
CLASS_ORDER = ("NotLead", "Lead")
LEAD_CLASS_INDEX = 1

KNOWN_MATERIALS = frozenset(["Lead", "Copper", "Galvanized"])


def encode_target(series: pd.Series) -> pd.Series:
    """Encode material labels into binary targets.
    
    Args:
        series: A pandas Series containing `service_line_material`.
        
    Returns:
        A pandas Series of integers: 1 for Lead, 0 for Copper/Galvanized.
        
    Raises:
        ValueError: If any unknown labels are present in the series.
    """
    unknowns = set(series.dropna().unique()) - KNOWN_MATERIALS
    if unknowns:
        raise ValueError(f"Unknown materials must be filtered out before encoding: {unknowns}")
        
    return (series == "Lead").astype(int)


def load_serving_model(model_dir: Path | str) -> object:
    """Load the canonical calibrated model artifact.
    
    Args:
        model_dir: Directory containing `xgb_model.pkl`.
        
    Returns:
        The CalibratedClassifierCV object.
        
    Raises:
        FileNotFoundError: If the artifact does not exist.
        ValueError: If the model classes do not match [0, 1].
    """
    artifact_path = Path(model_dir) / "xgb_model.pkl"
    if not artifact_path.exists():
        raise FileNotFoundError(f"Serving model artifact not found: {artifact_path}")
        
    with open(artifact_path, "rb") as f:
        model = pickle.load(f)
        
    # Validate the class contract
    if not hasattr(model, "classes_"):
        raise ValueError("Loaded model does not have classes_ attribute.")
        
    classes = list(model.classes_)
    if classes != [0, 1]:
        raise ValueError(f"Expected binary classes [0, 1], got {classes!r}")
        
    return model


def predict_proba(model: object, X: np.ndarray) -> np.ndarray:
    """Predict probabilities using the canonical contract.
    
    Args:
        model: The loaded serving model.
        X: Feature matrix.
        
    Returns:
        np.ndarray of shape (n_samples, 2).
        Column 0: P(NotLead)
        Column 1: P(Lead)
    """
    proba = model.predict_proba(X)
    if proba.shape[1] != 2:
        raise ValueError(f"Expected model to output 2 class probabilities, got {proba.shape[1]}")
    return proba
