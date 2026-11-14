"""Download the Copernicus GLO-30 DEM tiles under every study event.

Covers each exposure domain's bounding box plus the TPI radius. GLO-30 has no
tiles over open sea; the bucket's tile list says which 1-degree cells exist.

Output: data/raw/dem/<tile>.tif (EPSG:4326, 1 arc-second) + manifest.json
"""
from __future__ import annotations

import math
from pathlib import Path

from src.acquire.ems import domain_boxes
from src.utils.config import get_project_root, load_config, resolve_path
from src.utils.download import fetch, file_record, write_manifest


def tile_name(lat: int, lon: int) -> str:
    """GLO-30 tile whose south-west corner is (lat, lon)."""
    ns = f"N{lat:02d}" if lat >= 0 else f"S{-lat:02d}"
    ew = f"E{lon:03d}" if lon >= 0 else f"W{-lon:03d}"
    return f"Copernicus_DSM_COG_10_{ns}_00_{ew}_00_DEM"


def tiles_for_bbox(west: float, south: float, east: float, north: float) -> list[str]:
    return [tile_name(lat, lon)
            for lat in range(math.floor(south), math.floor(north) + 1)
            for lon in range(math.floor(west), math.floor(east) + 1)]


def download_dem(cfg: dict) -> list[Path]:
    src = cfg["sources"]["dem"]
    out_dir = resolve_path(src["output_dir"])
    root = get_project_root()

    list_url = f"{src['base_url']}/tileList.txt"
    list_path = fetch(list_url, out_dir / "tileList.txt")
    available = set(list_path.read_text().split())

    boxes = domain_boxes(cfg, cfg["features"]["terrain"]["tpi_radius_m"]).to_crs(4326)
    needed = sorted({t for g in boxes for t in tiles_for_bbox(*g.bounds)})
    tiles = [t for t in needed if t in available]
    print(f"DEM: {len(tiles)} of {len(needed)} 1-degree cells have a tile")

    records, paths = [file_record(list_path, list_url, root)], []
    for t in tiles:
        url = f"{src['base_url']}/{t}/{t}.tif"
        paths.append(fetch(url, out_dir / f"{t}.tif"))
        records.append(file_record(paths[-1], url, root))
    write_manifest(out_dir, src, records)
    return paths


if __name__ == "__main__":
    download_dem(load_config())
