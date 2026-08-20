import json

import pandas as pd

from leadguard.data.clean import (
    _make_property_id,
    load_assessor,
    load_chicago_water,
    normalize_address,
)
from leadguard.data.download import _load_manifest, _save_manifest
from leadguard.data.features import _impute_numerics, _load_osm_hydrants


def test_clean_utils():
    assert normalize_address(" 123 MAIN  St ") == "123 main st"
    pid = _make_property_id("123 main st", "12345")
    assert pid.startswith("chi-")
    assert len(pid) == 16


def test_load_chicago_water(tmp_path):
    p = tmp_path / "water.csv"
    p.write_text(
        "Street Address,ZIP Code,Ward,Property Type,Service Line Material,Latitude,Longitude\n123 Main St,60601,1,RES,lead,41.8,-87.6\n"
    )
    df = load_chicago_water(p)
    assert len(df) == 1
    assert df["service_line_material"].iloc[0] == "Lead"


def test_load_assessor(tmp_path):
    p = tmp_path / "assessor.csv"
    p.write_text("PIN,Address,Year Built,Building Sq Ft\n12345,123 Main St,1910,1000\n")
    df = load_assessor(p)
    assert len(df) == 1
    assert "year_built" in df.columns


def test_download_manifest(tmp_path):
    m = tmp_path / "manifest.json"
    _save_manifest(m, {"test": "val"})
    d = _load_manifest(m)
    assert d["test"] == "val"
    assert _load_manifest(tmp_path / "missing.json") == {}


def test_impute_numerics():
    df = pd.DataFrame({"year_built": [1900.0, None]})
    res = _impute_numerics(df)
    assert res["year_built"].iloc[1] == 1900.0


def test_load_osm_hydrants(tmp_path):
    p = tmp_path / "osm_hydrants_chicago.json"
    data = {"elements": [{"lat": 41.8, "lon": -87.6}]}
    p.write_text(json.dumps(data))
    df = _load_osm_hydrants(tmp_path)
    assert len(df) == 1
    assert df["latitude"].iloc[0] == 41.8

    # missing file
    df2 = _load_osm_hydrants(tmp_path / "other")
    assert len(df2) == 0
