# LeadGuard — Data Card

**Version:** 1.0  
**Date:** 2026-08-20  
**Scope:** Sample dataset (7,500-row synthetic) + real data sources description

---

## Sample Dataset Statistics

| Metric | Value |
|--------|-------|
| Total properties | 7,500 (synthetic) |
| Unique `property_id`s | 7,461 (39 deduped) |
| Properties with known material | 6,439 (85.9%) |
| Lead | 2,194 (29.3%) |
| Copper | 3,540 (47.2%) |
| Galvanized | 705 (9.4%) |
| Unknown / unlabeled | 1,061 (14.1%) |
| Missing `year_built` | 511 (6.8%) |
| Missing `lot_size_sqft` | 0 (synthetic) |
| Census tracts | 100 (synthetic) |
| Wards | 50 (synthetic) |

---

## Real Data Sources (Production Use)

### Chicago Water Service Line Inventory
- **URL:** https://data.cityofchicago.org
- **License:** CC0 (public domain)
- **Size:** ~412,000 properties as of 2025
- **Known quality issues:**
  - ~15% of records have `service_line_material = Unknown` — these are excluded from the training target but retained for spatial-lag computation
  - Self-reported materials may be inaccurate (captured in `material_source` field with `self_reported` tag)
  - Geocoding errors exist at <1% of records (dropped during cleaning)

### Cook County Property Assessor
- **URL:** https://datacatalog.cookcountyil.gov
- **License:** Open Data
- **Known quality issues:**
  - ~5-10% missing `year_built` — imputed with median during feature engineering
  - Property class codes require mapping (handled in `clean.py`)

### OpenStreetMap (Geofabrik Chicago metro)
- **URL:** https://download.geofabrik.de/north-america/us/illinois.html
- **License:** ODbL
- **Used for:** Fire hydrant locations for `dist_to_nearest_hydrant_m`

### US Census ACS (5-year estimates)
- **URL:** https://api.census.gov
- **License:** Public domain
- **Used for:** Tract-level median household income → income quartile
- **Design rule:** This data enters ONLY `fairness_reference.parquet`, never the feature matrix

---

## Class Balance Note

The ~34% Lead rate in the synthetic sample reflects realistic Chicago conditions (~30-40% estimated lead lines in older neighborhoods). The `scale_pos_weight` parameter in XGBoost is computed automatically from the training partition ratio.

---

## Fairness Constraints

- Income and demographic data are NEVER used as model inputs
- `census_tract` is present as a join key but excluded from the feature matrix
- The automated `test_no_demographic_leakage` test enforces this in CI

---

## Known Limitations

1. The sample dataset is synthetic — metrics on it do not represent real Chicago performance
2. The model is designed for Chicago (Cook County) and has not been validated on other cities
3. Self-reported service line materials in the inventory may have higher error rates than field-inspected records
