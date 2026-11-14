"""Clean and reproject the footprints and assign ids.

Input:  data/raw/buildings_msft/footprints.gpkg
Output: data/processed/buildings.gpkg (working CRS): building_id, centroid x
        and y, area_m2, geometry
"""
from __future__ import annotations

import geopandas as gpd
import pandas as pd
import shapely

from src.acquire.buildings import GPKG_NAME
from src.utils.config import load_config, resolve_path

MIN_AREA_M2 = 5.0  # smaller polygons are segmentation slivers, not buildings
PROCESSED = "data/processed/buildings.gpkg"


def clean(gdf: gpd.GeoDataFrame, crs: str) -> gpd.GeoDataFrame:
    gdf = gdf.to_crs(crs)
    gdf["geometry"] = shapely.make_valid(gdf.geometry.values)
    gdf = gdf[~gdf.geometry.is_empty]
    gdf = gdf.assign(area_m2=gdf.geometry.area)
    gdf = gdf[gdf["area_m2"] >= MIN_AREA_M2]
    n_dup = gdf.geometry.to_wkb().duplicated().sum()
    if n_dup:
        raise ValueError(f"{n_dup} duplicate footprints; download tiles overlap")
    cent = gdf.geometry.centroid
    gdf = gdf.assign(x=cent.x, y=cent.y).sort_values(["x", "y"], ignore_index=True)
    gdf.insert(0, "building_id", [f"B{i:07d}" for i in range(len(gdf))])
    return gdf[["building_id", "x", "y", "area_m2", "geometry"]]


def read_buildings() -> pd.DataFrame:
    """building_id, x, y and area_m2 of every footprint, without geometry."""
    return gpd.read_file(resolve_path(PROCESSED), columns=["building_id", "x", "y", "area_m2"],
                         ignore_geometry=True)


def main() -> None:
    cfg = load_config()
    raw = resolve_path(cfg["sources"]["buildings"]["output_dir"]) / GPKG_NAME
    gdf = clean(gpd.read_file(raw), cfg["pipeline"]["crs"])
    out = resolve_path(PROCESSED)
    out.parent.mkdir(parents=True, exist_ok=True)
    gdf.to_file(out, layer="buildings", driver="GPKG")
    print(f"buildings: {len(gdf):,} footprints -> {out.name}")


if __name__ == "__main__":
    main()
