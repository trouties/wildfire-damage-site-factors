"""Download CORINE Land Cover 2018 polygons around every study event.

Queries the EEA map service once per event, over the exposure domain's
bounding box grown by the wildland-fraction radius. A polygon reaching into
several boxes is kept once. CORINE 2018 shows the land cover before every
study fire (see min_activation_date in config/events.yaml).

Output: data/raw/corine/clc2018.gpkg (EPSG:4326, integer column ``code``)
        + manifest.json
"""
from __future__ import annotations

from pathlib import Path

import geopandas as gpd
import pandas as pd
import requests

from src.acquire.ems import domain_boxes
from src.utils.config import get_project_root, load_config, resolve_path
from src.utils.download import fetch, file_record, full_url, http_get, write_manifest

GPKG_NAME = "clc2018.gpkg"
PAGE_SIZE = 1000  # the service's maxRecordCount


def _no_error(resp: requests.Response) -> None:
    """ArcGIS reports a failed query as HTTP 200 with an ``error`` member."""
    body = resp.json()
    if "error" in body:
        raise RuntimeError(f"CORINE query failed: {body['error']}")


def _envelope(bounds: tuple[float, float, float, float]) -> dict[str, str]:
    return {
        "where": "1=1",
        "geometry": ",".join(f"{v:.6f}" for v in bounds),
        "geometryType": "esriGeometryEnvelope",
        "inSR": "4326",
        "spatialRel": "esriSpatialRelIntersects",
    }


def _count(url: str, envelope: dict[str, str]) -> int:
    resp = http_get(url, params={**envelope, "returnCountOnly": "true", "f": "json"})
    resp.raise_for_status()
    _no_error(resp)
    return resp.json()["count"]


def download_corine(cfg: dict) -> Path:
    src = cfg["sources"]["corine"]
    url, code_field = src["query_url"], src["code_field"]
    out_dir = resolve_path(src["output_dir"])
    root = get_project_root()
    pad_m = cfg["features"]["corine"]["wildland_fraction_radius_m"]

    records, pages = [], []
    for event, box in domain_boxes(cfg, pad_m).to_crs(4326).items():
        envelope = _envelope(box.bounds)
        count = _count(url, envelope)
        if count == 0:
            raise RuntimeError(f"{event}: no CORINE polygons in {envelope['geometry']}")
        got = 0
        for offset in range(0, count, PAGE_SIZE):
            params = {**envelope, "outFields": f"OBJECTID,{code_field}", "outSR": "4326",
                      "orderByFields": "OBJECTID", "resultOffset": str(offset),
                      "resultRecordCount": str(PAGE_SIZE), "f": "geojson"}
            dest = fetch(url, out_dir / f"{event}_{offset:05d}.geojson", params=params,
                         check=_no_error)
            records.append(file_record(dest, full_url(url, params), root))
            pages.append(gpd.read_file(dest))
            got += len(pages[-1])
        if got != count:
            raise RuntimeError(f"{event}: service reports {count} polygons, pages hold {got}")

    clc = pd.concat(pages, ignore_index=True).drop_duplicates("OBJECTID")
    out = gpd.GeoDataFrame({"code": clc[code_field].astype(int).values},
                           geometry=clc.geometry.values, crs="EPSG:4326")
    gpkg = out_dir / GPKG_NAME
    out.to_file(gpkg, layer="clc2018", driver="GPKG")
    write_manifest(out_dir, src, records, n_polygons=len(out))
    print(f"CORINE: {len(out):,} polygons from {len(records)} pages -> {gpkg.name}")
    return gpkg


if __name__ == "__main__":
    download_corine(load_config())
