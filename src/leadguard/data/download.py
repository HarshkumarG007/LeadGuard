"""Data download module for LeadGuard.

Fetches the four public data sources defined in Architecture §6.1.
Idempotent: checks file checksums before re-downloading.

Sources:
    1. Chicago Water Service Line Inventory (Chicago Data Portal)
    2. Cook County Property Assessor (Cook County Open Data)
    3. OpenStreetMap (Geofabrik Chicago metro extract)
    4. US Census ACS 5-year estimates (Census API)
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
from pathlib import Path

import requests

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Public endpoints (all free, no auth required except optional Census API key)
# ---------------------------------------------------------------------------

SOURCES = {
    "chicago_water": {
        "url": "https://data.cityofchicago.org/api/views/x2n5-8w5q/rows.csv?accessType=DOWNLOAD",
        "filename": "chicago_water_service_lines.csv",
        "description": "Chicago Water Service Line Inventory",
    },
    "cook_county_assessor": {
        "url": "https://datacatalog.cookcountyil.gov/api/views/uzyt-m557/rows.csv?accessType=DOWNLOAD",
        "filename": "cook_county_assessor.csv",
        "description": "Cook County Property Assessor",
    },
}

CENSUS_BASE = "https://api.census.gov/data/2022/acs/acs5"
OSM_HYDRANTS_OVERPASS = "https://overpass-api.de/api/interpreter"


def _sha256(path: Path) -> str:
    """Compute SHA-256 hex digest of a file.

    Args:
        path: Path to the file.

    Returns:
        Hex-encoded SHA-256 digest string.
    """
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _load_manifest(manifest_path: Path) -> dict:
    """Load the download manifest (maps filename → sha256).

    Args:
        manifest_path: Path to the JSON manifest file.

    Returns:
        Dictionary mapping filename to SHA-256 hash.
    """
    if manifest_path.exists():
        return json.loads(manifest_path.read_text())
    return {}


def _save_manifest(manifest_path: Path, manifest: dict) -> None:
    """Save the download manifest.

    Args:
        manifest_path: Path to the JSON manifest file.
        manifest: Dictionary mapping filename to SHA-256 hash.
    """
    manifest_path.write_text(json.dumps(manifest, indent=2))


def download_file(url: str, dest: Path, session: requests.Session) -> None:
    """Stream-download a URL to a file.

    Args:
        url: Source URL.
        dest: Destination file path.
        session: requests.Session to use.
    """
    logger.info("Downloading %s → %s", url, dest)
    with session.get(url, stream=True, timeout=120) as r:
        r.raise_for_status()
        with dest.open("wb") as f:
            for chunk in r.iter_content(chunk_size=65536):
                f.write(chunk)


def download_census_acs(raw_dir: Path, api_key: str = "") -> Path:
    """Download ACS 5-year tract-level median household income for Cook County, IL.

    Args:
        raw_dir: Directory to write raw CSV.
        api_key: Optional Census API key (uses anonymous access if empty).

    Returns:
        Path to the written CSV file.
    """
    dest = raw_dir / "census_acs_cook_county.csv"
    params = {
        "get": "NAME,B19013_001E",  # Median household income
        "for": "tract:*",
        "in": "state:17 county:031",  # Illinois (17), Cook County (031)
    }
    if api_key:
        params["key"] = api_key

    logger.info("Fetching ACS tract-level income for Cook County, IL")
    resp = requests.get(CENSUS_BASE, params=params, timeout=60)
    resp.raise_for_status()

    import csv

    data = resp.json()
    with dest.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerows(data)
    logger.info("Census ACS data written to %s (%d tracts)", dest, len(data) - 1)
    return dest


def download_osm_hydrants(raw_dir: Path) -> Path:
    """Download fire hydrant locations in Chicago via Overpass API.

    Args:
        raw_dir: Directory to write raw JSON.

    Returns:
        Path to the written JSON file.
    """
    dest = raw_dir / "osm_hydrants_chicago.json"
    query = """
    [out:json][timeout:120];
    area["name"="Chicago"]["admin_level"="8"]->.chicago;
    node["emergency"="fire_hydrant"](area.chicago);
    out body;
    """
    logger.info("Fetching OSM fire hydrant locations via Overpass API")
    resp = requests.post(OSM_HYDRANTS_OVERPASS, data={"data": query}, timeout=180)
    resp.raise_for_status()
    dest.write_bytes(resp.content)
    logger.info("OSM hydrant data written to %s", dest)
    return dest


def download_all(raw_dir: Path | str = "data/raw", census_api_key: str = "") -> None:
    """Download all four data sources idempotently.

    Skips files whose SHA-256 matches the previously recorded manifest.

    Args:
        raw_dir: Directory to store raw downloads.
        census_api_key: Optional Census API key. Reads CENSUS_API_KEY env var if empty.
    """
    raw_dir = Path(raw_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = raw_dir / "manifest.json"
    manifest = _load_manifest(manifest_path)

    if not census_api_key:
        census_api_key = os.getenv("CENSUS_API_KEY", "")

    session = requests.Session()
    updated = False

    for key, source in SOURCES.items():
        dest = raw_dir / source["filename"]
        current_hash = manifest.get(source["filename"])

        if dest.exists() and current_hash and _sha256(dest) == current_hash:
            logger.info("Skipping %s (checksum match)", source["filename"])
            continue

        try:
            download_file(source["url"], dest, session)
            manifest[source["filename"]] = _sha256(dest)
            updated = True
            logger.info("Downloaded %s", source["description"])
        except Exception as exc:
            logger.error("Failed to download %s: %s", source["description"], exc)
            raise

    # Census ACS
    census_file = "census_acs_cook_county.csv"
    census_dest = raw_dir / census_file
    if (
        census_dest.exists()
        and manifest.get(census_file)
        and _sha256(census_dest) == manifest[census_file]
    ):
        logger.info("Skipping Census ACS (checksum match)")
    else:
        try:
            download_census_acs(raw_dir, api_key=census_api_key)
            manifest[census_file] = _sha256(raw_dir / census_file)
            updated = True
        except Exception as exc:
            logger.warning("Census ACS download failed: %s — continuing without it", exc)

    # OSM hydrants
    osm_file = "osm_hydrants_chicago.json"
    osm_dest = raw_dir / osm_file
    if osm_dest.exists() and manifest.get(osm_file) and _sha256(osm_dest) == manifest[osm_file]:
        logger.info("Skipping OSM hydrants (checksum match)")
    else:
        try:
            download_osm_hydrants(raw_dir)
            manifest[osm_file] = _sha256(raw_dir / osm_file)
            updated = True
        except Exception as exc:
            logger.warning("OSM hydrant download failed: %s — continuing without it", exc)

    if updated:
        _save_manifest(manifest_path, manifest)
        logger.info("Manifest updated at %s", manifest_path)
    else:
        logger.info("All files up to date — no downloads needed")


def main():
    import argparse

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description="Download LeadGuard raw data")
    parser.add_argument("--raw-dir", default="data/raw", help="Raw data directory")
    parser.add_argument("--census-api-key", default="", help="Census API key (optional)")
    args = parser.parse_args()
    download_all(raw_dir=args.raw_dir, census_api_key=args.census_api_key)


if __name__ == "__main__":
    main()
