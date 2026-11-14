"""Per-building damage labels for every study event.

data/processed/labels.csv: one row per (event, building) whose footprint
centroid lies in the event's exposure domain (graded AOI intersected with the
activation's burnt area; see src.acquire.ems).
  damage_grade  most severe EMS grade linked to the footprint; empty = not graded
  damaged       damage_grade is one of ``damaged_grades``; empty for
                ``unlabelled_grades``. "Not Analysed" buildings are left out.

Also writes outputs/labels/label_summary.csv with per-event place name and
counts, the share of damaged points that found a footprint, the shares of
damaged and of possibly damaged points in the domain whose geometry EMS took
from OpenStreetMap, and why an event was dropped.
"""
from __future__ import annotations

import geopandas as gpd
import numpy as np
import pandas as pd
import shapely

from src.acquire.ems import aoi_locality, event_points, load_domains, load_events, load_packages
from src.preprocess.buildings import PROCESSED, read_buildings
from src.utils.config import load_config, resolve_path

SUMMARY_CSV = "outputs/labels/label_summary.csv"
LABELS_CSV = "data/processed/labels.csv"


def labelled_buildings() -> pd.DataFrame:
    """The labels with each footprint's centroid and area, sorted by event and
    building. A building can appear in several activations' events."""
    labels = pd.read_csv(resolve_path(LABELS_CSV), dtype={"damaged": "boolean"})
    rows = labels.merge(read_buildings(), on="building_id", how="left", validate="many_to_one")
    if rows["x"].isna().any():
        raise ValueError(f"{rows['x'].isna().sum()} labelled buildings missing from {PROCESSED}")
    return rows.sort_values(["event", "building_id"], ignore_index=True)


def most_severe(grades: pd.Series, order: list[str]) -> str:
    rank = {g: i for i, g in enumerate(order)}
    return max(grades, key=rank.__getitem__)


def grade_from_points(footprints: gpd.GeoDataFrame, points: gpd.GeoDataFrame,
                      max_dist_m: float, order: list[str]) -> tuple[pd.Series, pd.Series]:
    """Link each graded point to the footprint containing it, else to the
    nearest footprint within ``max_dist_m``.

    Returns the most severe grade per footprint (aligned to ``footprints``,
    NaN where nothing is linked) and a boolean Series over ``points`` saying
    whether each point found a footprint.
    """
    j = gpd.sjoin_nearest(points[["damage_gra", "geometry"]], footprints[["geometry"]],
                          how="left", max_distance=max_dist_m)
    j = j[~j.index.duplicated(keep="first")]  # equidistant ties: keep one footprint
    linked = j.dropna(subset=["index_right"])
    grade = linked.groupby(linked["index_right"].astype(footprints.index.dtype))[
        "damage_gra"].agg(lambda g: most_severe(g, order))
    return grade.reindex(footprints.index), j["index_right"].notna().reindex(points.index)


def label_event(cfg: dict, packages: pd.DataFrame, event: pd.Series, domain,
                buildings: gpd.GeoDataFrame) -> tuple[pd.DataFrame, dict]:
    ev = cfg["events"]
    points = event_points(cfg, packages, event["event"], cfg["pipeline"]["crs"])

    # Footprints a point could link to: anything in the bounding box of the
    # domain and the points (no buffer of the outline; see ems.domain_boxes).
    pad = ev["point_match_max_m"] + 1
    b = np.vstack([domain.bounds, points.total_bounds])
    reach = shapely.box(*(b[:, :2].min(axis=0) - pad), *(b[:, 2:].max(axis=0) + pad))
    near = buildings.iloc[buildings.sindex.query(reach)]
    grade, point_linked = grade_from_points(near, points, ev["point_match_max_m"],
                                            ev["grade_order"])

    shapely.prepare(domain)  # point-in-polygon tests against up to 1.3 M vertices
    in_domain = shapely.contains_xy(domain, near["x"].values, near["y"].values)
    df = pd.DataFrame({
        "event": event["event"],
        "activation": event["activation"],
        "building_id": near["building_id"].values,
        "damage_grade": grade.values,
    })[in_domain]
    df = df[df["damage_grade"] != "Not Analysed"].copy()
    df["damaged"] = df["damage_grade"].isin(ev["damaged_grades"]).astype("boolean")
    df.loc[df["damage_grade"].isin(ev["unlabelled_grades"]), "damaged"] = pd.NA

    is_dmg_point = points["damage_gra"].isin(ev["damaged_grades"])
    pxy = points.geometry.centroid  # a few layers store MultiPoints
    point_in_domain = shapely.contains_xy(domain, pxy.x.values, pxy.y.values)
    link_rate = point_linked[is_dmg_point].mean()
    # If one class of points came from OpenStreetMap and the other were
    # digitised by the analysts, the two classes would differ in where they
    # are mapped, not only in damage.
    from_osm = points["geometry_source"].eq("Open Street Map")
    is_possibly = points["damage_gra"].isin(ev["unlabelled_grades"])
    counts = df["damage_grade"].fillna("not graded").value_counts()
    summary = {
        "event": event["event"],
        "activation": event["activation"],
        "activation_date": event["activation_date"],
        "locality": aoi_locality(cfg, packages, event["event"]),
        "n_buildings": len(df),
        "n_labelled": int(df["damaged"].notna().sum()),
        "n_damaged": int(df["damaged"].sum()),
        **{f"grade_{g.lower().replace(' ', '_')}": int(counts.get(g, 0))
           for g in [*ev["grade_order"][1:], "not graded"]},
        "n_damaged_points": int(is_dmg_point.sum()),
        "damaged_point_link_rate": round(float(link_rate), 3),
        "damaged_points_outside_domain": int((is_dmg_point & ~point_in_domain).sum()),
        "osm_share_damaged": round(from_osm[is_dmg_point & point_in_domain].mean(), 2),
        "osm_share_possibly": round(from_osm[is_possibly & point_in_domain].mean(), 2),
        "domain_ha": round(domain.area / 1e4, 1),
    }
    if link_rate < ev["min_damaged_link_rate"]:
        summary["dropped"] = f"damaged-point link rate below {ev['min_damaged_link_rate']}"
        return df.iloc[0:0], summary
    summary["dropped"] = ""
    return df, summary


def main() -> None:
    cfg = load_config()
    packages = load_packages(cfg)
    domains = load_domains(cfg)
    buildings = gpd.read_file(resolve_path(PROCESSED))

    frames, summaries = [], []
    for _, event in load_events(cfg).iterrows():
        df, summary = label_event(cfg, packages, event, domains.geometry[event["event"]],
                                  buildings)
        frames.append(df)
        summaries.append(summary)
        print(f"{event['event']}: {summary['n_buildings']:,} buildings, "
              f"{summary['n_labelled']:,} labelled, {summary['n_damaged']} damaged, "
              f"link rate {summary['damaged_point_link_rate']} {summary['dropped']}")
    labels = pd.concat(frames, ignore_index=True)

    dup = labels.duplicated(["activation", "building_id"])
    if dup.any():
        raise ValueError(f"{dup.sum()} buildings fall in two AOIs of the same activation")

    labels.to_csv(resolve_path(LABELS_CSV), index=False, lineterminator="\n")

    table = pd.DataFrame(summaries)
    out = resolve_path(SUMMARY_CSV)
    out.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(out, index=False, lineterminator="\n")
    kept = table[table["dropped"] == ""]
    print(f"\n{len(kept)} events kept, {int(kept['n_buildings'].sum()):,} buildings, "
          f"{int(kept['n_labelled'].sum()):,} labelled, "
          f"{int(kept['n_damaged'].sum()):,} damaged -> {out.name}")


if __name__ == "__main__":
    main()
