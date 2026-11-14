"""Label assignment on hand-built geometries, and download tile lookup."""
from __future__ import annotations

import gzip
import json

import geopandas as gpd
import pandas as pd
import pytest
from shapely.geometry import Point, box, mapping

from src.acquire.buildings import quadkey, quadkeys_for_bbox, read_tile, tile_xy
from src.acquire.dem import tile_name, tiles_for_bbox
from src.acquire.ems import PKG_RE, geometry_source, merge_versions
from src.acquire.worldcover import tile_name as worldcover_tile
from src.acquire.worldcover import tiles_for_bbox as worldcover_tiles
from src.labels.build import grade_from_points, most_severe

CRS = "EPSG:2100"
ORDER = ["Not Analysed", "No visible damage", "Possibly damaged", "Damaged", "Destroyed"]


def test_quadkey_matches_bing_documentation_example():
    # Bing Maps tile system docs: tile (3, 5) at level 3 has quadkey "213".
    assert quadkey(3, 5, 3) == "213"


def test_tile_xy_at_zoom_1_quadrants():
    # At zoom 1 the world is 2x2 tiles split at lon 0 and lat 0.
    assert tile_xy(-90.0, 45.0, 1) == (0, 0)
    assert tile_xy(90.0, -45.0, 1) == (1, 1)


def test_quadkeys_for_bbox_spanning_origin_covers_all_four_tiles():
    assert sorted(quadkeys_for_bbox(-1.0, -1.0, 1.0, 1.0, 1)) == ["0", "1", "2", "3"]


def test_quadkeys_for_bbox_inside_one_tile():
    assert quadkeys_for_bbox(10.0, 10.0, 11.0, 11.0, 1) == ["1"]


def _write_tile(path, geoms):
    with gzip.open(path, "wt", encoding="utf-8") as fh:
        for g in geoms:
            fh.write(json.dumps({"type": "Feature", "properties": {}, "geometry": mapping(g)}) + "\n")


def test_read_tile_keeps_footprints_in_any_box(tmp_path):
    tile = tmp_path / "t.csv.gz"
    _write_tile(tile, [box(0.1, 0.1, 0.2, 0.2),     # in the first box
                       box(5.1, 5.1, 5.2, 5.2),     # in the second box
                       box(2.0, 2.0, 2.1, 2.1)])    # in neither
    gdf = read_tile(tile, [(0, 0, 1, 1), (5, 5, 6, 6)])
    assert sorted(gdf.geometry.bounds["minx"].round(1)) == [0.1, 5.1]


def test_read_tile_rejects_non_polygons(tmp_path):
    tile = tmp_path / "t.csv.gz"
    _write_tile(tile, [box(0, 0, 1, 1).union(box(2, 2, 3, 3))])
    with pytest.raises(ValueError, match="MultiPolygon"):
        read_tile(tile, [(0, 0, 9, 9)])


def test_dem_tiles_are_named_by_south_west_corner():
    assert tiles_for_bbox(23.5, 37.5, 24.2, 38.5) == [
        tile_name(37, 23), tile_name(37, 24), tile_name(38, 23), tile_name(38, 24)]
    assert tile_name(38, 23) == "Copernicus_DSM_COG_10_N38_00_E023_00_DEM"
    # A cell west of Greenwich and south of the equator: corner is (-1, -1).
    assert tiles_for_bbox(-0.5, -0.5, -0.1, -0.1) == ["Copernicus_DSM_COG_10_S01_00_W001_00_DEM"]


def test_worldcover_tiles_are_3_degree_cells_named_by_south_west_corner():
    # A box from 37.9 to 39.1 N and 20.5 to 21.2 E spans two latitude bands
    # (36-39, 39-42) and two longitude bands (18-21, 21-24).
    assert worldcover_tiles(20.5, 37.9, 21.2, 39.1) == [
        worldcover_tile(36, 18), worldcover_tile(36, 21),
        worldcover_tile(39, 18), worldcover_tile(39, 21)]
    assert worldcover_tile(36, 21) == "ESA_WorldCover_10m_2020_v100_N36E021_Map"


@pytest.mark.parametrize("name, expected", [
    # old portal naming: release r2, monitoring 0
    ("EMSR527_AOI01_GRA_PRODUCT_r2_RTP01_v1_vector.zip", ("EMSR527", "01", None, "2", "1")),
    ("EMSR593_AOI01_GRA_MONIT01_r1_RTP01_v2_vector.zip", ("EMSR593", "01", "01", "1", "2")),
    ("EMSR508_AOI01_GRA_PRODUCT_r1_VECTORS_v1_vector.zip", ("EMSR508", "01", None, "1", "1")),
    # rapid-mapping API naming
    ("EMSR746_AOI01_GRA_PRODUCT_v1.zip", ("EMSR746", "01", None, None, "1")),
    ("EMSR807_AOI01_GRA_MONIT01_v2.zip", ("EMSR807", "01", "01", None, "2")),
])
def test_package_name_parsing(name, expected):
    m = PKG_RE.search(name)
    assert (m["code"], m["aoi"], m["mon"], m["rel"], m["ver"]) == expected


def test_package_name_rejects_delineation_products():
    assert PKG_RE.search("EMSR527_AOI01_DEL_MONIT02_r1_RTP01_v1_vector.zip") is None


def _points(grades, xy):
    return gpd.GeoDataFrame({"damage_gra": grades},
                            geometry=[Point(x, y) for x, y in xy], crs=CRS)


def test_merge_versions_newer_grade_replaces_older_at_same_building():
    first = _points(["Possibly damaged", "Possibly damaged"], [(0, 0), (100, 0)])
    monit = _points(["Destroyed"], [(1, 0)])        # re-graded the first building
    merged = merge_versions([first, monit], tolerance_m=2)
    assert sorted(zip(merged.geometry.x, merged["damage_gra"])) == [
        (1.0, "Destroyed"), (100.0, "Possibly damaged")]


def test_merge_versions_keeps_buildings_outside_a_partial_monitoring_product():
    first = _points(["Damaged", "Damaged"], [(0, 0), (500, 0)])
    monit = _points(["Damaged"], [(900, 0)])        # new area only
    assert len(merge_versions([first, monit], tolerance_m=2)) == 3


def test_geometry_source_looks_up_the_package_source_table():
    # src_id is numeric in some packages' tables; an id the table lacks has no source.
    sources = pd.DataFrame({"src_id": [994, 1], "source_nam": ["Open Street Map", "WorldView-3"]})
    points = pd.DataFrame({"or_src_id": ["1", "994", "997"]})
    assert geometry_source(points, sources).tolist()[:2] == ["WorldView-3", "Open Street Map"]
    assert pd.isna(geometry_source(points, sources).iloc[2])


def test_most_severe_uses_grade_order_not_alphabet():
    assert most_severe(pd.Series(["Damaged", "Possibly damaged"]), ORDER) == "Damaged"
    assert most_severe(pd.Series(["Damaged", "Destroyed"]), ORDER) == "Destroyed"
    assert most_severe(pd.Series(["Not Analysed", "No visible damage"]), ORDER) == "No visible damage"


def test_most_severe_rejects_unknown_grade():
    with pytest.raises(KeyError):
        most_severe(pd.Series(["Damaged", "Burnt"]), ORDER)


@pytest.fixture
def footprints():
    # Three 10 m x 10 m houses, 100 m apart along x.
    return gpd.GeoDataFrame(
        {"building_id": ["A", "B", "C"]},
        geometry=[box(0, 0, 10, 10), box(100, 0, 110, 10), box(200, 0, 210, 10)],
        crs=CRS,
    )


def test_grade_from_points(footprints):
    points = gpd.GeoDataFrame(
        {"damage_gra": ["Possibly damaged", "Damaged", "Destroyed", "Damaged"]},
        geometry=[
            Point(5, 5),       # inside A
            Point(2, 8),       # inside A, more severe
            Point(113, 5),     # 3 m east of B
            Point(150, 50),    # > 10 m from every footprint
        ],
        crs=CRS,
    )
    grade, linked = grade_from_points(footprints, points, max_dist_m=10, order=ORDER)
    assert grade.tolist()[:2] == ["Damaged", "Destroyed"]
    assert pd.isna(grade.iloc[2])
    assert linked.tolist() == [True, True, True, False]


def test_grade_from_points_respects_distance_limit(footprints):
    points = gpd.GeoDataFrame({"damage_gra": ["Destroyed"]},
                              geometry=[Point(113, 5)], crs=CRS)
    grade, linked = grade_from_points(footprints, points, max_dist_m=2, order=ORDER)
    assert grade.isna().all()
    assert not linked.any()
