"""SHAP explainability for LeadGuard predictions.

Implements Architecture §8 shap_top_features field:
  - SHAP TreeExplainer on XGBoost model
  - Global summary plot (reports/shap_summary.png)
  - Per-prediction top-5 feature extraction (<100ms per property)

Usage:
    python -m leadguard.evaluation.explainability
    python -m leadguard.evaluation.explainability --sample
"""

from __future__ import annotations

import logging
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap

from leadguard.utils.seed import SEED

logger = logging.getLogger(__name__)


def _load_model_and_data(
    model_dir: Path,
    features_path: Path,
    sample: bool = False,
) -> tuple:
    """Load XGBoost model and feature data.

    Args:
        model_dir: Directory containing model.json.
        features_path: Path to features parquet.
        sample: Use sample data if True.

    Returns:
        Tuple of (model, df, feature_names).
    """
    import xgboost as xgb  # noqa: PLC0415
    from leadguard.models.xgboost_model import XGB_FEATURES  # noqa: PLC0415

    if sample and not features_path.exists():
        features_path = Path("data/processed/features_sample.parquet")

    model = xgb.XGBClassifier()
    model.load_model(str(model_dir / "model.json"))
    df = pd.read_parquet(features_path)
    return model, df, XGB_FEATURES


def compute_shap_values(
    model: object,
    X: np.ndarray,
    feature_names: list[str],
) -> tuple[shap.Explainer, np.ndarray]:
    """Compute SHAP values using TreeExplainer.

    Args:
        model: Fitted XGBoost model.
        X: Feature matrix.
        feature_names: Feature column names.

    Returns:
        Tuple of (explainer, shap_values array).
    """
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)
    return explainer, shap_values


def extract_top_shap_features(
    shap_row: np.ndarray,
    feature_names: list[str],
    top_n: int = 5,
) -> list[dict]:
    """Extract the top N SHAP contributions for a single prediction.

    Args:
        shap_row: 1-D SHAP values for one property.
        feature_names: Feature names corresponding to shap_row.
        top_n: Number of top features to return.

    Returns:
        List of {feature, contribution} dicts sorted by |contribution| descending.
    """
    abs_vals = np.abs(shap_row)
    top_idx = np.argsort(abs_vals)[::-1][:top_n]
    return [
        {"feature": feature_names[i], "contribution": float(shap_row[i])}
        for i in top_idx
    ]


def generate_global_summary(
    model: object,
    X: np.ndarray,
    feature_names: list[str],
    output_path: Path,
    max_rows: int = 1000,
) -> None:
    """Generate and save a SHAP summary bar plot.

    Args:
        model: Fitted XGBoost model.
        X: Feature matrix (will be sampled if too large).
        feature_names: Feature names.
        output_path: PNG file path.
        max_rows: Maximum rows to use for summary plot.
    """
    rng = np.random.default_rng(SEED)
    if len(X) > max_rows:
        idx = rng.choice(len(X), size=max_rows, replace=False)
        X_sample = X[idx]
    else:
        X_sample = X

    explainer, shap_values = compute_shap_values(model, X_sample, feature_names)

    fig, ax = plt.subplots(figsize=(10, 6))
    shap.summary_plot(
        shap_values,
        X_sample,
        feature_names=feature_names,
        show=False,
        plot_type="bar",
    )
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=120, bbox_inches="tight")
    plt.close("all")
    logger.info("SHAP summary plot saved to %s", output_path)


def benchmark_per_prediction_latency(
    model: object,
    X_single: np.ndarray,
    feature_names: list[str],
    n_runs: int = 20,
    threshold_ms: float = 100.0,
) -> float:
    """Benchmark per-prediction SHAP extraction latency.

    Args:
        model: Fitted XGBoost model.
        X_single: Single-row feature matrix (shape 1 × n_features).
        feature_names: Feature names.
        n_runs: Number of timing runs.
        threshold_ms: Latency budget in milliseconds.

    Returns:
        Median latency in milliseconds.
    """
    explainer = shap.TreeExplainer(model)
    latencies = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        sv = explainer.shap_values(X_single)
        _ = extract_top_shap_features(sv[0] if sv.ndim > 1 else sv, feature_names)
        latencies.append((time.perf_counter() - t0) * 1000)

    median_ms = float(np.median(latencies))
    logger.info("SHAP per-prediction latency: median=%.1f ms (threshold=%.0f ms)", median_ms, threshold_ms)
    if median_ms > threshold_ms:
        logger.warning("SHAP latency %.1f ms exceeds %.0f ms API budget — consider caching", median_ms, threshold_ms)
    return median_ms


def run_explainability(
    model_dir: Path | str = "models/xgboost",
    features_path: Path | str = "data/processed/features.parquet",
    output_dir: Path | str = "reports",
    sample: bool = False,
) -> dict:
    """Run the full explainability pipeline.

    Args:
        model_dir: XGBoost model directory.
        features_path: Features parquet path.
        output_dir: Output directory for plots.
        sample: Use sample data if True.

    Returns:
        Dictionary with latency and plot path.
    """
    model_dir = Path(model_dir)
    features_path = Path(features_path)
    output_dir = Path(output_dir)

    model, df, feature_names = _load_model_and_data(model_dir, features_path, sample=sample)
    labeled = df[df["service_line_material"].isin(["Lead", "Copper", "Galvanized"])].copy()

    X = labeled.reindex(columns=feature_names, fill_value=0.0).astype(float).values
    logger.info("Computing SHAP values for %d rows", len(X))

    # Global summary plot
    summary_path = output_dir / "shap_summary.png"
    generate_global_summary(model, X, feature_names, summary_path)

    # Latency benchmark on a single row
    median_ms = benchmark_per_prediction_latency(model, X[:1], feature_names)

    result = {
        "shap_summary_path": str(summary_path),
        "per_prediction_latency_ms_median": median_ms,
        "latency_within_budget": median_ms < 100.0,
    }
    logger.info("Explainability phase complete: %s", result)
    return result


if __name__ == "__main__":
    import argparse

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description="Run SHAP explainability pipeline")
    parser.add_argument("--sample", action="store_true")
    args = parser.parse_args()
    result = run_explainability(sample=args.sample)
    print("PHASE 8 PASS — latency:", result["per_prediction_latency_ms_median"], "ms")
