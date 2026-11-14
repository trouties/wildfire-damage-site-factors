"""Static per-building features, written with the labels to the analysis table.

Terrain (GLO-30) and land cover (CORINE 2018) predate every study fire. The
footprints are a 2026 release; buildings that vanished after a fire show up
as unlinked damage points (see the link-rate rule in src.labels.build).

data/analysis_table.csv: one row per row of data/processed/labels.csv, with
its label columns, the footprint centroid (x, y, working CRS) and area, and
  slope_deg        terrain slope at the footprint centroid (Horn, on the DEM grid)
  northness        north and east components of the unit surface normal:
  eastness         0 on flat ground, sin(slope) on a slope facing due north/east
  tpi_m            elevation minus the mean elevation within ``tpi_radius_m``
  wildland_frac    share of the disc of ``wildland_fraction_radius_m`` that is
                   CORINE forest or shrub
  wildland_dist_m  distance to the nearest forest or shrub cell, capped at
                   ``wildland_fraction_radius_m``; wildland outside the event
                   grid is always farther than that, so the cap is exact
  wildland_dist_wc_m  the same from ESA WorldCover 2020 at 10 m (``worldcover``)
  n_neighbours     other footprint centroids within ``neighbour_radius_m``
  dnbr_100m        burn severity around the building, the positive control, from
                   src.features.dnbr; empty where an event has no Sentinel-2 pair

Rasters are built per event on a grid around its exposure domain and are not
written to disk. A missing static feature raises. The table is committed so
that src.analysis.evaluate runs without the downloads.
"""
from __future__ import annotations

import geopandas as gpd
import numpy as np
import pandas as pd
import shapely
from affine import Affine
from rasterio.features import rasterize
from rasterio.merge import merge
from rasterio.transform import array_bounds
from rasterio.warp import Resampling, reproject, transform_bounds
from scipy.spatial import cKDTree

from src.acquire.corine import GPKG_NAME as CORINE_GPKG
from src.acquire.dem import tiles_for_bbox
from src.acquire.ems import load_domains
from src.acquire.worldcover import tiles_for_bbox as worldcover_tiles
from src.features.dnbr import COLUMN as DNBR_COLUMN
from src.features.dnbr import DNBR_CSV
from src.features.rasters import disc_mean, distance_to, grid, horn_gradient, sample
from src.labels.build import labelled_buildings
from src.preprocess.buildings import read_buildings
from src.utils.config import load_config, resolve_path

TABLE_CSV = "data/analysis_table.csv"
# Decimals kept in the table; finer digits are below the inputs' resolution.
DECIMALS = {"x": 1, "y": 1, "area_m2": 1, "slope_deg": 2, "northness": 4, "eastness": 4,
            "tpi_m": 2, "wildland_frac": 4, "wildland_dist_m": 1, "wildland_dist_wc_m": 1}


def count_neighbours(all_xy: np.ndarray, query_xy: np.ndarray, radius_m: float) -> np.ndarray:
    """Points of ``all_xy`` within ``radius_m`` of each query point, the point
    itself excluded (query points must be part of ``all_xy``)."""
    return cKDTree(all_xy).query_ball_point(query_xy, radius_m, return_length=True) - 1


def read_dem(cfg: dict, transform: Affine, shape: tuple[int, int]) -> np.ndarray:
    """GLO-30 resampled (bilinear) onto the grid. Cells without a GLO-30 tile
    are open sea, which GLO-30 stores as 0 m inside its tiles too."""
    dem_dir = resolve_path(cfg["sources"]["dem"]["output_dir"])
    available = set((dem_dir / "tileList.txt").read_text().split())
    crs = cfg["pipeline"]["crs"]
    w, s, e, n = transform_bounds(crs, "EPSG:4326", *array_bounds(*shape, transform))
    paths = [dem_dir / f"{t}.tif" for t in tiles_for_bbox(w, s, e, n) if t in available]
    missing = [p.name for p in paths if not p.exists()]
    if missing:
        raise FileNotFoundError(f"DEM tiles not downloaded: {missing}")
    margin = 0.01  # degrees, so bilinear resampling has neighbours at the grid edge
    mosaic, src_transform = merge([str(p) for p in paths], nodata=0,
                                  bounds=(w - margin, s - margin, e + margin, n + margin))
    dem = np.full(shape, np.nan, dtype="float32")
    reproject(mosaic[0], dem, src_transform=src_transform, src_crs="EPSG:4326",
              dst_transform=transform, dst_crs=crs, dst_nodata=np.nan,
              resampling=Resampling.bilinear)
    if np.isnan(dem).any():
        raise ValueError("DEM does not cover the event grid")
    return dem


def terrain_features(cfg: dict, bounds, x: np.ndarray, y: np.ndarray) -> dict[str, np.ndarray]:
    t = cfg["features"]["terrain"]
    cell, radius = t["cell_m"], t["tpi_radius_m"]
    transform, shape = grid(bounds, cell, radius + cell)
    dem = read_dem(cfg, transform, shape)
    dx, dy = horn_gradient(dem, cell)
    norm = np.sqrt(1 + dx**2 + dy**2)
    layers = {
        "slope_deg": np.degrees(np.arctan(np.hypot(dx, dy))),
        "northness": -dy / norm,
        "eastness": -dx / norm,
        "tpi_m": dem - disc_mean(dem, radius / cell),
    }
    return {k: sample(v, transform, x, y) for k, v in layers.items()}


def landcover_features(cfg: dict, bounds, x: np.ndarray, y: np.ndarray,
                       clc: gpd.GeoDataFrame) -> dict[str, np.ndarray]:
    c = cfg["features"]["corine"]
    cell, radius = c["cell_m"], c["wildland_fraction_radius_m"]
    transform, shape = grid(bounds, cell, radius + cell)
    near = clc.iloc[clc.sindex.query(shapely.box(*array_bounds(*shape, transform)))]
    codes = rasterize(zip(near.geometry, near["code"]), out_shape=shape,
                      transform=transform, fill=0, dtype="int16")
    covered = sample(disc_mean((codes > 0).astype(float), radius / cell), transform, x, y)
    if (covered < 1 - 1e-6).any():
        raise ValueError(f"CORINE has gaps within {radius} m of {(covered < 1 - 1e-6).sum()} "
                         "buildings")
    wild = np.isin(codes, c["forest"] + c["shrub"])
    frac = np.clip(disc_mean(wild.astype(float), radius / cell), 0, 1)
    return {"wildland_frac": sample(frac, transform, x, y),
            "wildland_dist_m": sample(distance_to(wild, cell, radius), transform, x, y)}


def read_worldcover(cfg: dict, transform: Affine, shape: tuple[int, int]) -> np.ndarray:
    """WorldCover classes on the grid (nearest neighbour; 0 = no data)."""
    wc_dir = resolve_path(cfg["sources"]["worldcover"]["output_dir"])
    crs = cfg["pipeline"]["crs"]
    w, s, e, n = transform_bounds(crs, "EPSG:4326", *array_bounds(*shape, transform))
    paths = [wc_dir / f"{t}.tif" for t in worldcover_tiles(w, s, e, n)]
    missing = [p.name for p in paths if not p.exists()]
    if missing:
        raise FileNotFoundError(f"WorldCover tiles not downloaded: {missing}")
    margin = 0.001  # degrees
    mosaic, src_transform = merge([str(p) for p in paths], nodata=0,
                                  bounds=(w - margin, s - margin, e + margin, n + margin))
    codes = np.zeros(shape, dtype="uint8")
    reproject(mosaic[0], codes, src_transform=src_transform, src_crs="EPSG:4326",
              dst_transform=transform, dst_crs=crs, src_nodata=0, dst_nodata=0,
              resampling=Resampling.nearest)
    if (codes == 0).any():
        raise ValueError("WorldCover does not cover the event grid")
    return codes


def worldcover_features(cfg: dict, bounds, x: np.ndarray, y: np.ndarray) -> dict[str, np.ndarray]:
    wc = cfg["features"]["worldcover"]
    cap = cfg["features"]["corine"]["wildland_fraction_radius_m"]
    transform, shape = grid(bounds, wc["cell_m"], cap + wc["cell_m"])
    wild = np.isin(read_worldcover(cfg, transform, shape), wc["wildland"])
    return {"wildland_dist_wc_m": sample(distance_to(wild, wc["cell_m"], cap), transform, x, y)}


def main() -> None:
    cfg = load_config()
    # EMS activation codes are issued in date order, so sorting by event is
    # chronological.
    rows = labelled_buildings()[["event", "activation", "building_id", "damage_grade",
                                 "damaged", "x", "y", "area_m2"]]
    all_xy = read_buildings()[["x", "y"]].to_numpy()

    domains = load_domains(cfg)
    clc_path = resolve_path(cfg["sources"]["corine"]["output_dir"]) / CORINE_GPKG
    clc = gpd.read_file(clc_path).to_crs(cfg["pipeline"]["crs"])

    rows["n_neighbours"] = count_neighbours(all_xy, rows[["x", "y"]].to_numpy(),
                                            cfg["features"]["buildings"]["neighbour_radius_m"])
    frames = []
    for event, grp in rows.groupby("event", sort=False):
        bounds = domains.geometry[event].bounds
        x, y = grp["x"].to_numpy(), grp["y"].to_numpy()
        feats = {**terrain_features(cfg, bounds, x, y),
                 **landcover_features(cfg, bounds, x, y, clc),
                 **worldcover_features(cfg, bounds, x, y)}
        frames.append(grp.assign(**feats))
        print(f"{event}: {len(grp):,} buildings")
    table = pd.concat(frames, ignore_index=True).round(DECIMALS)
    dnbr = pd.read_csv(resolve_path(DNBR_CSV))
    table = table.merge(dnbr, on=["event", "building_id"], how="left", validate="one_to_one")

    # Only the labels (not graded, or "Possibly damaged") and the control may be NULL.
    bad = table.drop(columns=["damage_grade", "damaged", DNBR_COLUMN]).isna().any(axis=1)
    if bad.any():
        raise ValueError(f"{bad.sum()} buildings with a missing feature")
    table.to_csv(resolve_path(TABLE_CSV), index=False, lineterminator="\n")
    print(f"{len(table):,} rows -> {TABLE_CSV}")


if __name__ == "__main__":
    main()
