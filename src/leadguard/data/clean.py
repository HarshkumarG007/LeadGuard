"""Data cleaning pipeline for LeadGuard.

Reads raw CSVs, deduplicates, normalizes, joins assessor data,
validates with Pandera, and writes to data/interim/.

Usage:
    python -m leadguard.data.clean
    python -m leadguard.data.clean --input data/sample --output data/interim/sample_interim.parquet
"""

from __future__ import annotations

import hashlib
import logging
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from leadguard.data.validation import INTERIM_SCHEMA
from leadguard.utils.seed import SEED

logger = logging.getLogger(__name__)


def _make_property_id(address: str, pin: str = "") -> str:
    """Create a deterministic property ID from normalized address + PIN.

    Args:
        address: Normalized property address string.
        pin: Cook County PIN (may be empty).

    Returns:
        12-character hex hash string prefixed with 'chi-'.
    """
    raw = f"{address.strip().lower()}|{pin.strip()}"
    return "chi-" + hashlib.sha256(raw.encode()).hexdigest()[:12]


def normalize_address(s: str) -> str:
    """Normalize an address string to a canonical form.

    Args:
        s: Raw address string.

    Returns:
        Normalized lowercase address with extra whitespace collapsed.
    """
    return " ".join(str(s).strip().lower().split())


def load_chicago_water(path: Path) -> pd.DataFrame:
    """Load the Chicago Water Service Line Inventory CSV.

    Args:
        path: Path to the raw CSV file.

    Returns:
        DataFrame with standardized column names.
    """
    rename_map = {
        "Street Address": "address",
        "ZIP Code": "zip_code",
        "Ward": "ward",
        "Property Type": "property_type",
        "Service Line Material": "service_line_material",
        "Latitude": "latitude",
        "Longitude": "longitude",
    }
    df = pd.read_csv(path, dtype=str)
    # Normalize column names — handle multiple possible header formats
    df.columns = [c.strip() for c in df.columns]
    df = df.rename(columns=rename_map)

    # Coerce types
    df["latitude"] = pd.to_numeric(df.get("latitude", pd.Series()), errors="coerce")
    df["longitude"] = pd.to_numeric(df.get("longitude", pd.Series()), errors="coerce")
    df["ward"] = pd.to_numeric(df.get("ward", pd.Series()), errors="coerce").astype("Int64")

    # Standardize material labels
    mat_map = {
        "lead": "Lead",
        "copper": "Copper",
        "galvanized": "Galvanized",
        "galvanized steel": "Galvanized",
        "unknown": "Unknown",
        "not determined": "Unknown",
    }
    if "service_line_material" in df.columns:
        df["service_line_material"] = df["service_line_material"].str.strip().str.lower().map(mat_map)

    return df


def load_assessor(path: Path) -> pd.DataFrame:
    """Load Cook County Assessor CSV.

    Args:
        path: Path to the raw assessor CSV.

    Returns:
        DataFrame with standardized column names.
    """
    rename_map = {
        "PIN": "pin",
        "Address": "address",
        "Year Built": "year_built",
        "Class": "property_class",
        "Land Area (Sq Ft)": "lot_size_sqft",
        "Building Sq Ft": "building_sqft",
        "Stories": "stories",
    }
    df = pd.read_csv(path, dtype=str)
    df.columns = [c.strip() for c in df.columns]
    df = df.rename(columns=rename_map)

    for col in ["year_built", "lot_size_sqft", "building_sqft", "stories"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    if "has_basement" not in df.columns:
        df["has_basement"] = False

    return df


def determine_material_source(row: pd.Series) -> str:
    """Determine the material_source category for a row.

    Args:
        row: A property record Series.

    Returns:
        One of 'inspected', 'self_reported', or 'unknown'.
    """
    mat = row.get("service_line_material")
    prop_type = str(row.get("property_type", "")).lower()

    if mat in ("Lead", "Copper", "Galvanized") and "inspect" in prop_type:
        return "inspected"
    elif mat in ("Lead", "Copper", "Galvanized"):
        return "self_reported"
    return "unknown"


def clean(
    input_dir: Path | str = "data/raw",
    output_path: Path | str = "data/interim/properties.parquet",
    sample_mode: bool = False,
) -> pd.DataFrame:
    """Run the full cleaning pipeline.

    Args:
        input_dir: Directory containing raw CSV files (or data/sample for sample mode).
        output_path: Destination parquet file path.
        sample_mode: If True, reads from sample directory directly.

    Returns:
        Cleaned and validated interim DataFrame.
    """
    input_dir = Path(input_dir)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # --- Load source data ---
    if sample_mode or (input_dir / "sample_properties.parquet").exists():
        # Sample mode: load the pre-generated sample parquet
        sample_files = list(input_dir.glob("*.parquet"))
        if sample_files:
            logger.info("Sample mode: loading %d parquet file(s) from %s", len(sample_files), input_dir)
            df = pd.concat([pd.read_parquet(f) for f in sample_files], ignore_index=True)
        else:
            raise FileNotFoundError(f"No parquet files found in sample dir: {input_dir}")
    else:
        water_path = input_dir / "chicago_water_service_lines.csv"
        if not water_path.exists():
            raise FileNotFoundError(
                f"Chicago water inventory not found at {water_path}. "
                "Run `python -m leadguard.data.download` first."
            )
        df = load_chicago_water(water_path)
        logger.info("Loaded water inventory: %d rows", len(df))

        # Try to join assessor data if available
        assessor_path = input_dir / "cook_county_assessor.csv"
        if assessor_path.exists():
            assessor = load_assessor(assessor_path)
            # Normalize addresses for join
            df["_addr_norm"] = df["address"].apply(normalize_address)
            assessor["_addr_norm"] = assessor["address"].apply(normalize_address)
            df = df.merge(
                assessor[["_addr_norm", "pin", "year_built", "property_class",
                          "lot_size_sqft", "building_sqft", "stories", "has_basement"]],
                on="_addr_norm",
                how="left",
            ).drop(columns=["_addr_norm"])
            logger.info("Joined assessor data: %d rows", len(df))
        else:
            logger.warning("Assessor data not found at %s — proceeding without it", assessor_path)
            for col in ["pin", "year_built", "property_class", "lot_size_sqft", "building_sqft", "stories"]:
                df[col] = None
            df["has_basement"] = False

    # --- Generate property_id ---
    df["pin"] = df.get("pin", pd.Series([""] * len(df))).fillna("")
    df["_addr_norm"] = df["address"].apply(normalize_address)
    df["property_id"] = df.apply(
        lambda r: _make_property_id(r["_addr_norm"], str(r.get("pin", ""))), axis=1
    )
    df = df.drop(columns=["_addr_norm"])

    # --- Ensure required columns ---
    for col in ["zip_code", "ward", "latitude", "longitude"]:
        if col not in df.columns:
            df[col] = None

    # Ensure ward is int
    df["ward"] = pd.to_numeric(df.get("ward"), errors="coerce").fillna(0).astype(int)

    # Drop rows without lat/lon (cannot geocode)
    n_before = len(df)
    df = df.dropna(subset=["latitude", "longitude"])
    n_dropped = n_before - len(df)
    if n_dropped > 0:
        logger.warning("Dropped %d rows missing lat/lon", n_dropped)

    # --- Deduplicate on property_id ---
    n_before = len(df)
    df = df.drop_duplicates(subset="property_id", keep="first")
    n_dupes = n_before - len(df)
    if n_dupes > 0:
        logger.info("Deduped %d duplicate property_id rows", n_dupes)

    # --- material_source ---
    df["service_line_material"] = df.get("service_line_material", pd.Series([None] * len(df)))
    df["material_source"] = df.apply(determine_material_source, axis=1)

    # --- census_tract placeholder ---
    if "census_tract" not in df.columns:
        df["census_tract"] = None

    # --- timestamp ---
    df["last_updated"] = datetime.now(timezone.utc).replace(tzinfo=None)

    # --- has_basement coerce ---
    df["has_basement"] = df.get("has_basement", pd.Series([False] * len(df))).fillna(False).astype(bool)

    # --- Validate with Pandera ---
    logger.info("Validating against INTERIM_SCHEMA (%d rows)", len(df))
    validated = INTERIM_SCHEMA.validate(df, lazy=True)

    # --- Write ---
    validated.to_parquet(output_path, index=False)
    logger.info("Wrote %d rows to %s", len(validated), output_path)
    return validated


if __name__ == "__main__":
    import argparse

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description="Clean and validate LeadGuard raw data")
    parser.add_argument("--input", default="data/raw", help="Input directory (raw or sample)")
    parser.add_argument("--output", default="data/interim/properties.parquet", help="Output parquet path")
    args = parser.parse_args()

    # Auto-detect sample mode
    sample_mode = "sample" in str(args.input)
    clean(input_dir=args.input, output_path=args.output, sample_mode=sample_mode)
    print("CLEAN DONE")
