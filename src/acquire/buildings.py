"""Download Microsoft ML building footprints around every study event.

Covers each event's exposure domain, as its bounding box grown by
``context_buffer_m``, so neighbour counts near the domain edge see the
buildings just outside it. Needs the event domains written by src.acquire.ems.

Output: data/raw/buildings_msft/footprints.gpkg (EPSG:4326) + manifest.json
"""
from __future__ import annotations

import gzip
import json
import math
from pathlib import Path

import geopandas as gpd
import pandas as pd
from shapely.geometry import shape

from src.acquire.ems import domain_boxes
from src.utils.config import get_project_root, load_config, resolve_path
from src.utils.download import fetch, file_record, write_manifest

GPKG_NAME = "footprints.gpkg"


def tile_xy(lon: float, lat: float, zoom: int) -> tuple[int, int]:
    """Web-Mercator tile containing (lon, lat)."""
    n = 2 ** zoom
    x = int((lon + 180.0) / 360.0 * n)
    y = int((1.0 - math.asinh(math.tan(math.radians(lat))) / math.pi) / 2.0 * n)
    return min(x, n - 1), min(y, n - 1)


def quadkey(x: int, y: int, zoom: int) -> str:
    """Bing Maps quadkey of tile (x, y)."""
    digits = []
    for level in range(zoom, 0, -1):
        mask = 1 << (level - 1)
        digits.append(str((1 if x & mask else 0) + (2 if y & mask else 0)))
    return "".join(digits)


def quadkeys_for_bbox(west: float, south: float, east: float, north: float,
                      zoom: int) -> list[str]:
    x0, y0 = tile_xy(west, north, zoom)   # tile y grows southwards
    x1, y1 = tile_xy(east, south, zoom)
    return [quadkey(x, y, zoom) for x in range(x0, x1 + 1) for y in range(y0, y1 + 1)]


def read_tile(path: Path, boxes: list[tuple[float, float, float, float]]) -> gpd.GeoDataFrame:
    """Footprints of one tile whose first vertex lies in any (west, south, east,
    north) box. A tile holds up to ~10^6 footprints, so the test runs on the
    raw coordinates before any geometry is built."""
    geoms = []
    with gzip.open(path, "rt", encoding="utf-8") as fh:
        for line in fh:
            geom = json.loads(line)["geometry"]
            if geom["type"] != "Polygon":
                raise ValueError(f"{path.name}: unexpected geometry type {geom['type']}")
            lon, lat = geom["coordinates"][0][0]
            if any(w <= lon <= e and s <= lat <= n for w, s, e, n in boxes):
                geoms.append(shape(geom))
    return gpd.GeoDataFrame(geometry=geoms, crs="EPSG:4326")


def download_buildings(cfg: dict) -> Path:
    src = cfg["sources"]["buildings"]
    out_dir = resolve_path(src["output_dir"])
    root = get_project_root()

    index_path = fetch(src["index_url"], out_dir / "dataset-links.csv")
    index = pd.read_csv(index_path, dtype=str)
    index = index[index["Location"] == src["region"]]
    zooms = index["QuadKey"].str.len().unique()
    if len(zooms) != 1:
        raise RuntimeError(f"Mixed quadkey zoom levels in index: {zooms}")
    zoom = int(zooms[0])

    pad_m = cfg["features"]["buildings"]["context_buffer_m"]
    boxes = [tuple(g.bounds) for g in domain_boxes(cfg, pad_m).to_crs(4326)]
    needed = {q for b in boxes for q in quadkeys_for_bbox(*b, zoom)}
    rows = index[index["QuadKey"].isin(needed)]
    # Quadkeys over open sea have no tile; a land tile missing from the index
    # would show up as unmatched EMS points in the label summary.
    print(f"Buildings: {len(rows)} of {len(needed)} quadkeys have a {src['region']} tile")

    records, parts = [], []
    for _, row in rows.iterrows():
        dest = out_dir / f"{row['QuadKey']}_{Path(row['Url']).name}"
        fetch(row["Url"], dest)
        records.append(file_record(dest, row["Url"], root) | {"upload_date": row["UploadDate"]})
        parts.append(read_tile(dest, boxes))
    out = gpd.GeoDataFrame(pd.concat(parts, ignore_index=True), crs="EPSG:4326")

    gpkg = out_dir / GPKG_NAME
    out.to_file(gpkg, layer="footprints", driver="GPKG")
    write_manifest(out_dir, src, [file_record(index_path, src["index_url"], root), *records],
                   n_footprints=len(out))
    print(f"Buildings: {len(out):,} footprints -> {gpkg.name}")
    return gpkg


if __name__ == "__main__":
    download_buildings(load_config())
