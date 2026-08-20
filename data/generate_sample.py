"""Script to generate the synthetic sample dataset.

Creates a 7,500-row, seeded, stratified sample committed to data/sample/.
This replaces real downloads for CI, testing, and demo purposes.

Architecture §1.4: The sample must be:
  - Seeded (reproducible)
  - Stratified to keep some of every material class
  - De-identified (no real addresses in git)
  - Small enough to stay well under 2GB repo limit
"""

from __future__ import annotations

import sys
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

# Add src/ to path so leadguard imports work without install
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from leadguard.utils.seed import SEED


def generate_sample(n_rows: int = 7500, output_dir: Path = Path("data/sample")) -> pd.DataFrame:
    """Generate a synthetic property dataset for sample/CI use.

    Args:
        n_rows: Number of synthetic properties to generate.
        output_dir: Directory to write sample_properties.parquet.

    Returns:
        Generated DataFrame.
    """
    rng = np.random.default_rng(SEED)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Chicago lat/lon bounding box
    lat_min, lat_max = 41.64, 42.02
    lon_min, lon_max = -87.94, -87.52

    # Property IDs — synthetic, not real
    property_ids = [f"chi-{i:08d}" for i in range(n_rows)]

    # Lat/lon (random within Chicago)
    lats = rng.uniform(lat_min, lat_max, n_rows)
    lons = rng.uniform(lon_min, lon_max, n_rows)

    # Year built — bimodal: pre-1950 (older, more likely lead) and post-1950
    year_built = np.where(
        rng.random(n_rows) < 0.40,
        rng.integers(1880, 1950, n_rows),  # older buildings
        rng.integers(1950, 2005, n_rows),  # newer buildings
    ).astype(float)
    # 7% missing
    missing_mask = rng.random(n_rows) < 0.07
    year_built[missing_mask] = np.nan

    # Wards 1–50
    wards = rng.integers(1, 51, n_rows)

    # ZIP codes — simplified set for Chicago
    zip_codes = rng.choice(
        [
            "60601",
            "60602",
            "60603",
            "60604",
            "60607",
            "60608",
            "60609",
            "60610",
            "60611",
            "60612",
            "60613",
            "60614",
            "60615",
            "60616",
            "60617",
            "60618",
            "60619",
            "60620",
            "60621",
            "60622",
        ],
        n_rows,
    )

    # Property class
    prop_classes = rng.choice(
        ["Single-family", "Multi-family", "Commercial", "Unknown"], n_rows, p=[0.5, 0.3, 0.15, 0.05]
    )

    # Lot/building size
    lot_size = rng.uniform(1000, 20000, n_rows)
    building_sqft = rng.uniform(500, 5000, n_rows)
    stories = rng.integers(1, 4, n_rows).astype(float)
    has_basement = rng.random(n_rows) < 0.60

    # Service line material — stratified
    # Pre-1950 → ~60% lead; post-1950 → ~10% lead
    p_lead = np.where(year_built < 1950, 0.60, 0.10)
    p_lead = np.where(np.isnan(year_built), 0.30, p_lead)  # unknown year → 30%
    materials = []
    for i in range(n_rows):
        r = rng.random()
        pl = p_lead[i]
        if r < pl:
            materials.append("Lead")
        elif r < pl + 0.50:
            materials.append("Copper")
        elif r < pl + 0.65:
            materials.append("Galvanized")
        else:
            materials.append(None)  # Unknown — treated as missing

    materials_arr = np.array(materials, dtype=object)

    # material_source
    sources = []
    for m in materials_arr:
        if m in ("Lead", "Copper", "Galvanized"):
            sources.append(rng.choice(["inspected", "self_reported"], p=[0.30, 0.70]))
        else:
            sources.append("unknown")

    # Census tracts — synthetic FIPS codes for Cook County
    n_tracts = 100
    tract_pool = [f"17031{str(i).zfill(6)}" for i in range(1, n_tracts + 1)]
    census_tracts = rng.choice(tract_pool, n_rows)

    # Spatial features — synthetic but correlated
    # neighbor_lead_rate and knn10_lead_rate will be recomputed in features.py
    # Here we include them as placeholders at 0.0 (to be overwritten)
    dist_hydrant = rng.uniform(50, 2000, n_rows)
    dist_lead = rng.uniform(10, 5000, n_rows)
    neighbor_lead_rate = rng.uniform(0.0, 0.5, n_rows)
    knn10_lead_rate = rng.uniform(0.0, 0.5, n_rows)

    # H3 indices — placeholder strings (real ones computed in features.py)
    h3_res8 = [f"881f1d4{i % 10}09fffff" for i in range(n_rows)]
    h3_res9 = [f"891f1d4{i % 10}09fffff" for i in range(n_rows)]

    df = pd.DataFrame(
        {
            "property_id": property_ids,
            "address": [
                f"{rng.integers(100, 9999)} N SYNTHETIC ST UNIT {i % 100 + 1}"
                for i in range(n_rows)
            ],
            "zip_code": zip_codes,
            "ward": wards,
            "latitude": lats,
            "longitude": lons,
            "year_built": year_built,
            "property_class": prop_classes,
            "lot_size_sqft": lot_size,
            "building_sqft": building_sqft,
            "stories": stories,
            "has_basement": has_basement,
            "h3_index_res8": h3_res8,
            "h3_index_res9": h3_res9,
            "dist_to_nearest_hydrant_m": dist_hydrant,
            "dist_to_nearest_known_lead_m": dist_lead,
            "neighbor_lead_rate_h3res8": neighbor_lead_rate,
            "knn10_lead_rate": knn10_lead_rate,
            "census_tract": census_tracts,
            "service_line_material": materials_arr,
            "material_source": sources,
            "last_updated": datetime.now(UTC).replace(tzinfo=None),
        }
    )

    # Stats
    n_labeled = df["service_line_material"].notna().sum()
    n_lead = (df["service_line_material"] == "Lead").sum()
    n_copper = (df["service_line_material"] == "Copper").sum()
    n_galv = (df["service_line_material"] == "Galvanized").sum()
    n_unknown = df["service_line_material"].isna().sum()
    n_missing_year = df["year_built"].isna().sum()

    print(f"Generated {n_rows} synthetic properties:")
    print(f"  Labeled: {n_labeled} ({n_labeled / n_rows:.1%})")
    print(f"  Lead: {n_lead} | Copper: {n_copper} | Galvanized: {n_galv} | Unknown: {n_unknown}")
    print(f"  Missing year_built: {n_missing_year} ({n_missing_year / n_rows:.1%})")

    out_path = output_dir / "sample_properties.parquet"
    df.to_parquet(out_path, index=False)
    print(f"  Written to {out_path}")
    return df


if __name__ == "__main__":
    generate_sample()
