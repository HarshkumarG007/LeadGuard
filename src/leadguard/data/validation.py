"""Pandera schemas for raw and interim LeadGuard data.

Every schema is used to validate pipeline stage outputs before they are written to disk.
A stage that writes unvalidated data is not done, per the spec §2.
"""

from __future__ import annotations

import pandera as pa
from pandera import Column, DataFrameSchema, Check


# ---------------------------------------------------------------------------
# Raw Chicago Water Service Line Inventory schema
# ---------------------------------------------------------------------------

RAW_WATER_SCHEMA = DataFrameSchema(
    {
        "address": Column(str, nullable=False),
        "zip_code": Column(str, nullable=False),
        "ward": Column(int, nullable=False),
        "property_type": Column(str, nullable=True),
        "service_line_material": Column(
            str,
            checks=Check.isin(["Lead", "Copper", "Galvanized", "Unknown"]),
            nullable=True,
        ),
        "latitude": Column(float, checks=Check.in_range(-90, 90), nullable=False),
        "longitude": Column(float, checks=Check.in_range(-180, 180), nullable=False),
    },
    coerce=True,
    strict=False,  # allow extra columns from raw source
)


# ---------------------------------------------------------------------------
# Raw Cook County Assessor schema
# ---------------------------------------------------------------------------

RAW_ASSESSOR_SCHEMA = DataFrameSchema(
    {
        "pin": Column(str, nullable=False),
        "address": Column(str, nullable=True),
        "year_built": Column(float, nullable=True),
        "property_class": Column(str, nullable=True),
        "lot_size_sqft": Column(float, nullable=True),
        "building_sqft": Column(float, nullable=True),
        "stories": Column(float, nullable=True),
        "has_basement": Column(bool, nullable=True),
    },
    coerce=True,
    strict=False,
)


# ---------------------------------------------------------------------------
# Interim (post-clean) schema — used after clean.py writes
# ---------------------------------------------------------------------------

INTERIM_SCHEMA = DataFrameSchema(
    {
        "property_id": Column(str, checks=Check.str_length(min_value=1), nullable=False),
        "address": Column(str, nullable=False),
        "zip_code": Column(str, nullable=False),
        "ward": Column(int, nullable=False),
        "latitude": Column(float, checks=Check.in_range(-90, 90), nullable=False),
        "longitude": Column(float, checks=Check.in_range(-180, 180), nullable=False),
        "year_built": Column(float, nullable=True),
        "property_class": Column(str, nullable=True),
        "lot_size_sqft": Column(float, nullable=True),
        "building_sqft": Column(float, nullable=True),
        "stories": Column(float, nullable=True),
        "has_basement": Column(bool, nullable=True),
        "service_line_material": Column(
            str,
            checks=Check.isin(["Lead", "Copper", "Galvanized", "Unknown"]),
            nullable=True,
        ),
        "material_source": Column(
            str,
            checks=Check.isin(["inspected", "self_reported", "unknown"]),
            nullable=False,
        ),
        "census_tract": Column(str, nullable=True),
        "last_updated": Column("datetime64[ns]", nullable=False),
    },
    checks=[
        # Property IDs must be unique
        Check(lambda df: df["property_id"].is_unique, error="duplicate property_id found"),
    ],
    coerce=True,
    strict=False,
)


# ---------------------------------------------------------------------------
# Feature table schema (Architecture §6.2 Property entity)
# ---------------------------------------------------------------------------

FEATURES_SCHEMA = DataFrameSchema(
    {
        "property_id": Column(str, nullable=False),
        "address": Column(str, nullable=False),
        "zip_code": Column(str, nullable=False),
        "ward": Column(int, nullable=False),
        "latitude": Column(float, nullable=False),
        "longitude": Column(float, nullable=False),
        "year_built": Column(float, nullable=True),
        "property_class": Column(str, nullable=True),
        "lot_size_sqft": Column(float, nullable=True),
        "building_sqft": Column(float, nullable=True),
        "stories": Column(float, nullable=True),
        "has_basement": Column(bool, nullable=True),
        "h3_index_res8": Column(str, nullable=False),
        "h3_index_res9": Column(str, nullable=False),
        "dist_to_nearest_hydrant_m": Column(float, nullable=False),
        "dist_to_nearest_known_lead_m": Column(float, nullable=False),
        "neighbor_lead_rate_h3res8": Column(
            float, checks=Check.in_range(0.0, 1.0), nullable=False
        ),
        "knn10_lead_rate": Column(float, checks=Check.in_range(0.0, 1.0), nullable=False),
        "census_tract": Column(str, nullable=True),
        "service_line_material": Column(str, nullable=True),
        "material_source": Column(str, nullable=False),
        "last_updated": Column("datetime64[ns]", nullable=False),
    },
    checks=[
        Check(lambda df: df["property_id"].is_unique, error="duplicate property_id in features"),
        # Hard constraint: no demographic columns allowed
        Check(
            lambda df: not any(
                col in df.columns
                for col in ["income_quartile", "median_household_income", "race", "ethnicity"]
            ),
            error="DEMOGRAPHIC LEAKAGE: protected-class column found in feature table",
        ),
    ],
    coerce=True,
    strict=False,
)


# ---------------------------------------------------------------------------
# Fairness reference schema
# ---------------------------------------------------------------------------

FAIRNESS_REFERENCE_SCHEMA = DataFrameSchema(
    {
        "census_tract": Column(str, nullable=False),
        "income_quartile": Column(
            int, checks=Check.isin([1, 2, 3, 4]), nullable=False
        ),
    },
    checks=[
        Check(
            lambda df: df["census_tract"].is_unique,
            error="duplicate census_tract in fairness_reference",
        )
    ],
    coerce=True,
)
