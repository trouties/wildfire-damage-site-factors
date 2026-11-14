"""Download the ESA WorldCover 2020 tiles under every study event.

Covers each exposure domain's bounding box plus the distance cap of the
wildland features.

Output: data/raw/worldcover/<tile>.tif (EPSG:4326, 10 m classes) + manifest.json
"""
from __future__ import annotations

import math
from pathlib import Path

from src.acquire.ems import domain_boxes
from src.utils.config import get_project_root, load_config, resolve_path
from src.utils.download import fetch, file_record, write_manifest


def tile_name(lat: int, lon: int) -> str:
    """WorldCover tile whose south-west corner is (lat, lon), both multiples of 3."""
    ns = f"N{lat:02d}" if lat >= 0 else f"S{-lat:02d}"
    ew = f"E{lon:03d}" if lon >= 0 else f"W{-lon:03d}"
    return f"ESA_WorldCover_10m_2020_v100_{ns}{ew}_Map"


def tiles_for_bbox(west: float, south: float, east: float, north: float) -> list[str]:
    return [tile_name(lat, lon)
            for lat in range(math.floor(south / 3) * 3, math.floor(north / 3) * 3 + 1, 3)
            for lon in range(math.floor(west / 3) * 3, math.floor(east / 3) * 3 + 1, 3)]


def download_worldcover(cfg: dict) -> list[Path]:
    src = cfg["sources"]["worldcover"]
    out_dir = resolve_path(src["output_dir"])
    root = get_project_root()
    # the grid of src.features.build: distance cap plus one cell
    pad = (cfg["features"]["corine"]["wildland_fraction_radius_m"]
           + cfg["features"]["worldcover"]["cell_m"])
    boxes = domain_boxes(cfg, pad).to_crs(4326)
    tiles = sorted({t for g in boxes for t in tiles_for_bbox(*g.bounds)})
    records, paths = [], []
    for t in tiles:
        url = f"{src['base_url']}/{t}.tif"
        paths.append(fetch(url, out_dir / f"{t}.tif"))
        records.append(file_record(paths[-1], url, root))
    write_manifest(out_dir, src, records)
    print(f"WorldCover: {len(tiles)} tiles -> {out_dir}")
    return paths


if __name__ == "__main__":
    download_worldcover(load_config())
