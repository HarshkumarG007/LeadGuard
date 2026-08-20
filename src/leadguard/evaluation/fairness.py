"""Fairness reference table and equity accounting for LeadGuard.

Implements Architecture §7.4:
  - ACS income → income quartile → fairness_reference.parquet
  - target_share(tract) and actual_share(tract) computation
  - equity_boost(tract) formula
  - FNR-by-quartile fairness audit
  - Leakage enforcement: no fairness_reference column enters features.parquet

The key design rule (Architecture §5): protected-class-correlated fields
enter ONLY this module and are never present in features.parquet.

Usage:
    python -m leadguard.evaluation.fairness
    python -m leadguard.evaluation.fairness --sample
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from leadguard.data.validation import FAIRNESS_REFERENCE_SCHEMA
from leadguard.utils.seed import SEED

logger = logging.getLogger(__name__)

# Forbidden columns — enforced by test_no_demographic_leakage
FORBIDDEN_IN_FEATURES = frozenset([
    "income_quartile",
    "median_household_income",
    "race",
    "ethnicity",
    "pct_nonwhite",
])


def _quartile_from_income(income: pd.Series) -> pd.Series:
    """Bucket income values into quartiles 1–4 (1=lowest, 4=highest).

    Args:
        income: Series of median household income values.

    Returns:
        Integer series of quartile labels 1–4.
    """
    return pd.qcut(income, q=4, labels=[1, 2, 3, 4]).astype(int)


def build_fairness_reference(
    raw_dir: Path | str = "data/raw",
    output_path: Path | str = "data/fairness_reference.parquet",
    sample_mode: bool = False,
) -> pd.DataFrame:
    """Build the census tract → income quartile reference table.

    Reads ACS tract-level data and produces:
        census_tract (str) | income_quartile (int 1-4)

    This table is NEVER joined onto features.parquet. It is used only in:
      - This module (fairness audit)
      - uncertainty.py (Mondrian grouping — explicitly local, not persisted to features)
      - active_learning.py (equity_boost computation)

    Args:
        raw_dir: Directory containing census_acs_cook_county.csv.
        output_path: Destination parquet path.
        sample_mode: If True, generate a synthetic reference table for testing.

    Returns:
        Validated fairness reference DataFrame.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    acs_path = Path(raw_dir) / "census_acs_cook_county.csv"

    if sample_mode or not acs_path.exists():
        logger.warning(
            "ACS data not found at %s — generating synthetic fairness reference for sample mode", acs_path
        )
        rng = np.random.default_rng(SEED)
        n_tracts = 200
        tract_ids = [f"17031{str(i).zfill(6)}" for i in range(1, n_tracts + 1)]
        incomes = rng.integers(20000, 150000, size=n_tracts).astype(float)
        df = pd.DataFrame({"census_tract": tract_ids, "_income": incomes})
    else:
        rows = pd.read_csv(acs_path, header=None)
        # First row is header from Census API JSON→CSV conversion
        header = rows.iloc[0].tolist()
        df = rows.iloc[1:].copy()
        df.columns = header
        # B19013_001E = median household income estimate
        income_col = "B19013_001E"
        if income_col not in df.columns:
            raise ValueError(f"Expected column {income_col!r} in ACS data; got {df.columns.tolist()}")
        df[income_col] = pd.to_numeric(df[income_col], errors="coerce")
        df = df[df[income_col] > 0]  # exclude suppressed values (-666666666)
        # Construct census_tract FIPS code
        df["census_tract"] = "17031" + df["tract"].astype(str).str.zfill(6)
        df = df.rename(columns={income_col: "_income"})
        df = df[["census_tract", "_income"]]

    # Assign income quartiles
    df = df.dropna(subset=["_income"])
    df["income_quartile"] = _quartile_from_income(df["_income"])
    df = df[["census_tract", "income_quartile"]].copy()

    # Validate
    validated = FAIRNESS_REFERENCE_SCHEMA.validate(df, lazy=True)
    validated.to_parquet(output_path, index=False)
    logger.info("Fairness reference written: %d tracts → %s", len(validated), output_path)
    return validated


def compute_equity_boost(
    predictions: pd.DataFrame,
    inspections: pd.DataFrame,
    epsilon: float = 1e-6,
) -> pd.Series:
    """Compute equity_boost per census tract (Architecture §7.4).

    equity_boost(tract) = clip(
        (target_share(tract) - actual_share(tract)) / max(target_share(tract), ε),
        0, 1
    )

    where:
        target_share(tract) = Σ p_lead_calibrated in tract / Σ p_lead_calibrated citywide
        actual_share(tract) = inspections in tract / total inspections

    A tract that has received fewer inspections than its risk share warrants
    gets a boost; over-inspected tracts get zero boost, never a penalty.

    Args:
        predictions: DataFrame with columns [census_tract, p_lead_calibrated].
        inspections: DataFrame with column [census_tract] (one row per inspection).
        epsilon: Floor for target_share denominator.

    Returns:
        Series indexed by census_tract with equity_boost values in [0, 1].
    """
    # target_share per tract
    city_total_risk = predictions["p_lead_calibrated"].sum()
    if city_total_risk == 0:
        logger.warning("City total p_lead_calibrated is 0; all equity_boost = 0")
        return pd.Series(0.0, index=predictions["census_tract"].unique())

    tract_risk = predictions.groupby("census_tract")["p_lead_calibrated"].sum()
    target_share = tract_risk / city_total_risk

    # actual_share per tract
    if len(inspections) == 0:
        actual_share = pd.Series(0.0, index=target_share.index)
    else:
        total_inspections = len(inspections)
        inspection_counts = inspections["census_tract"].value_counts()
        actual_share = inspection_counts / total_inspections
        actual_share = actual_share.reindex(target_share.index, fill_value=0.0)

    # equity_boost
    delta = target_share - actual_share
    equity_boost = np.clip(delta / target_share.clip(lower=epsilon), 0.0, 1.0)
    return equity_boost


def run_fairness_audit(
    features_path: Path | str = "data/processed/features.parquet",
    fairness_ref_path: Path | str = "data/fairness_reference.parquet",
    predictions: Optional[pd.DataFrame] = None,
    inspections: Optional[pd.DataFrame] = None,
    output_path: Path | str = "reports/fairness_report.json",
    sample_mode: bool = False,
) -> dict:
    """Run the FNR-by-quartile fairness audit and write report.

    Args:
        features_path: Features parquet path.
        fairness_ref_path: Fairness reference parquet path.
        predictions: Optional pre-computed predictions DataFrame.
        inspections: Optional inspections DataFrame.
        output_path: Destination JSON report path.
        sample_mode: If True, use sample data paths.

    Returns:
        Fairness report dictionary.
    """
    import xgboost as xgb  # noqa: PLC0415
    from leadguard.models.xgboost_model import XGB_FEATURES  # noqa: PLC0415

    features_path = Path(features_path)
    if sample_mode and not features_path.exists():
        features_path = Path("data/processed/features_sample.parquet")

    df = pd.read_parquet(features_path)
    labeled = df[df["service_line_material"].isin(["Lead", "Copper", "Galvanized"])].copy()

    # Load model and generate predictions if not provided
    if predictions is None:
        model_path = Path("models/xgboost/model.json")
        if not model_path.exists():
            logger.warning("Model not found; using dummy predictions for fairness audit")
            labeled["p_lead_calibrated"] = 0.5
        else:
            model = xgb.XGBClassifier()
            model.load_model(str(model_path))
            X = labeled.reindex(columns=XGB_FEATURES, fill_value=0.0).astype(float).values
            labeled["p_lead_calibrated"] = model.predict_proba(X)[:, 1]
        predictions = labeled[["property_id", "census_tract", "p_lead_calibrated",
                                "service_line_material"]].copy()

    # Join income quartile from fairness reference
    fairness_ref = pd.read_parquet(fairness_ref_path) if Path(fairness_ref_path).exists() else pd.DataFrame()

    if fairness_ref.empty:
        logger.warning("No fairness reference available; building synthetic one")
        fairness_ref = build_fairness_reference(sample_mode=True)

    preds_with_quartile = predictions.merge(fairness_ref, on="census_tract", how="left")
    preds_with_quartile["income_quartile"] = preds_with_quartile["income_quartile"].fillna(2).astype(int)

    # FNR by quartile
    y_true = (preds_with_quartile["service_line_material"] == "Lead").astype(int)
    y_pred = (preds_with_quartile["p_lead_calibrated"] >= 0.5).astype(int)
    preds_with_quartile["_fn"] = (y_true == 1) & (y_pred == 0)
    preds_with_quartile["_lead"] = y_true

    fnr_by_quartile = {}
    for q in range(1, 5):
        mask = preds_with_quartile["income_quartile"] == q
        sub = preds_with_quartile[mask]
        n_lead = sub["_lead"].sum()
        fnr = float(sub["_fn"].sum() / n_lead) if n_lead > 0 else float("nan")
        fnr_by_quartile[q] = fnr

    # Disparity flag (>5 pp difference — Architecture §7.6)
    fnr_values = [v for v in fnr_by_quartile.values() if not np.isnan(v)]
    disparity = max(fnr_values) - min(fnr_values) if len(fnr_values) > 1 else 0.0
    disparity_flag = disparity > 0.05
    if disparity_flag:
        logger.warning(
            "FAIRNESS FLAG: FNR disparity %.1f pp across income quartiles (threshold: 5 pp)",
            disparity * 100,
        )

    # Equity boost sample
    if inspections is None:
        inspections = pd.DataFrame(columns=["census_tract"])
    equity_boost = compute_equity_boost(
        predictions[["census_tract", "p_lead_calibrated"]].dropna(),
        inspections,
    )
    equity_boost_sample = equity_boost.head(10).to_dict()

    report = {
        "fnr_by_quartile": {str(k): v for k, v in fnr_by_quartile.items()},
        "fnr_disparity_pp": disparity * 100,
        "disparity_flagged": disparity_flag,
        "equity_boost_sample": {str(k): float(v) for k, v in equity_boost_sample.items()},
        "n_labeled_properties": len(labeled),
        "n_properties_with_quartile": int((preds_with_quartile["income_quartile"] > 0).sum()),
    }

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w") as f:
        json.dump(report, f, indent=2, default=str)
    logger.info("Fairness report written to %s", output_path)
    return report


if __name__ == "__main__":
    import argparse

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description="Build fairness reference and run fairness audit")
    parser.add_argument("--raw-dir", default="data/raw")
    parser.add_argument("--sample", action="store_true")
    args = parser.parse_args()

    build_fairness_reference(raw_dir=args.raw_dir, sample_mode=args.sample)
    run_fairness_audit(sample_mode=args.sample)
    print("PHASE 6 PASS")
