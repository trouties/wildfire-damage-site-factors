"""Copernicus EMS grading products for the study events.

``python -m src.acquire.ems`` lists every wildfire activation in the study
country since ``min_activation_date``, downloads all grading (GRA) vector
packages, merges each AOI's products point by point, applies the selection
rules in config/events.yaml and writes
  outputs/labels/ems_packages.csv  one row per package
  outputs/labels/ems_events.csv    one row per AOI, with the selection reason
  data/processed/event_domains.gpkg  exposure domain of each selected event

An event is one grading AOI, keyed ``<activation>_AOI<nn>``.
"""
from __future__ import annotations

import re
import zipfile
from pathlib import Path

import geopandas as gpd
import pandas as pd
import shapely

from src.utils.config import get_project_root, load_config, resolve_path
from src.utils.download import fetch, file_record, http_get, write_manifest

PACKAGES_CSV = "outputs/labels/ems_packages.csv"
EVENTS_CSV = "outputs/labels/ems_events.csv"
DOMAINS_GPKG = "data/processed/event_domains.gpkg"

PKG_RE = re.compile(
    r"(?P<code>EMSR\d+)_AOI(?P<aoi>\d+)_GRA_(?:PRODUCT|MONIT(?P<mon>\d+))"
    r"_(?:r(?P<rel>\d+)_[A-Z0-9]+_)?v(?P<ver>\d+)(?:_vector)?\.zip$"
)
# Layer names differ by prefix and version suffix only, e.g.
# EMSR527_AOI03_GRA_PRODUCT_builtUpP_r1_v2 and EMSR746_AOI01_GRA_PRODUCT_builtUpP_v1.
LAYERS = {"points": "builtUpP", "burnt": "observedEventA", "aoi": "areaOfInterestA",
          "source": "source"}


def event_key(code: str, aoi: int) -> str:
    return f"{code}_AOI{aoi:02d}"


def list_activations(cfg: dict) -> pd.DataFrame:
    ev, src = cfg["events"], cfg["sources"]["ems"]
    resp = http_get(src["activations_api"], params={"limit": "5000"})
    resp.raise_for_status()
    body = resp.json()
    if body["next"]:
        raise RuntimeError("EMS activation list is paginated beyond limit=5000")
    rows = [
        {"code": a["code"], "name": a["name"], "activation_date": a["activationTime"][:10]}
        for a in body["results"]
        if a["code"].startswith("EMSR")
        and a["category"]["name"] == ev["category"]
        and ev["country"] in {c["short_name"] for c in a["countries"]}
        and a["activationTime"][:10] >= ev["min_activation_date"]
    ]
    return pd.DataFrame(rows).sort_values("code", ignore_index=True)


def gra_package_urls(cfg: dict, code: str) -> list[str]:
    src = cfg["sources"]["ems"]
    resp = http_get(src["products_api"], params={"code": code})
    if resp.status_code == 200 and resp.json().get("results"):
        return [p["downloadPath"]
                for aoi in resp.json()["results"][0]["aois"]
                for p in aoi["products"]
                if p["type"] == "GRA" and p.get("downloadPath")]
    if resp.status_code not in (200, 401, 403):
        resp.raise_for_status()
    # Not in the rapid-mapping API: older activation, links on its portal page.
    page = http_get(src["activation_page"].format(code=code))
    page.raise_for_status()
    links = re.findall(r'https://[^"\s<>]+?/static/activations/[^"\s<>]+?_vector\.zip', page.text)
    return sorted({u for u in links if PKG_RE.search(u.rsplit("/", 1)[-1])})


def list_packages(cfg: dict) -> pd.DataFrame:
    """One row per GRA package of every candidate activation, oldest product
    of each AOI first."""
    rows = []
    for act in list_activations(cfg).itertuples():
        for url in gra_package_urls(cfg, act.code):
            m = PKG_RE.search(url.rsplit("/", 1)[-1])
            if m["code"] != act.code:
                raise ValueError(f"{url} listed under activation {act.code}")
            rows.append({
                "event": event_key(act.code, int(m["aoi"])),
                "activation": act.code,
                "activation_name": act.name,
                "activation_date": act.activation_date,
                "monitoring": int(m["mon"] or 0),
                "release": int(m["rel"] or 1),
                "version": int(m["ver"]),
                "url": url,
            })
    return (pd.DataFrame(rows).drop_duplicates("url")
            .sort_values(["event", "monitoring", "release", "version"], ignore_index=True))


def package_dir(cfg: dict, url: str) -> Path:
    name = url.rsplit("/", 1)[-1]
    code = PKG_RE.search(name)["code"]
    return resolve_path(cfg["sources"]["ems"]["output_dir"]) / code / Path(name).stem


def download_package(cfg: dict, url: str) -> dict:
    folder = package_dir(cfg, url)
    zpath = fetch(url, folder.with_suffix(".zip"))
    if not folder.exists():
        with zipfile.ZipFile(zpath) as zf:
            for member in zf.namelist():
                if member.lower().endswith((".shp", ".shx", ".dbf", ".prj", ".cpg")):
                    zf.extract(member, folder)
        folder.mkdir(exist_ok=True)
    return file_record(zpath, url, get_project_root())


def find_layer(folder: Path, kind: str, suffix: str = ".shp") -> Path | None:
    pattern = re.compile(rf"_{LAYERS[kind]}(_|$)")
    matches = [p for p in folder.rglob(f"*{suffix}") if pattern.search(p.stem)]
    if len(matches) > 1:
        raise ValueError(f"Several {kind} layers in {folder}: {matches}")
    return matches[0] if matches else None


def read_layer(cfg: dict, url: str, kind: str, crs: str) -> gpd.GeoDataFrame | None:
    path = find_layer(package_dir(cfg, url), kind)
    if path is None:
        return None
    gdf = gpd.read_file(path)
    return gdf[gdf.geometry.notna() & ~gdf.geometry.is_empty].to_crs(crs)


def load_packages(cfg: dict) -> pd.DataFrame:
    return pd.read_csv(resolve_path(PACKAGES_CSV))


def load_events(cfg: dict) -> pd.DataFrame:
    events = pd.read_csv(resolve_path(EVENTS_CSV))
    return events[events["selected"]].reset_index(drop=True)


def load_domains(cfg: dict) -> gpd.GeoDataFrame:
    """Exposure domain per selected event (working CRS), indexed by event."""
    return gpd.read_file(resolve_path(DOMAINS_GPKG)).set_index("event")


def aoi_locality(cfg: dict, packages: pd.DataFrame, event: str) -> str:
    """Place name EMS gives the event's AOI."""
    for url in packages.loc[packages["event"] == event, "url"]:
        path = find_layer(package_dir(cfg, url), "aoi")
        if path is not None:
            name = gpd.read_file(path, ignore_geometry=True)["locality"].iloc[0]
            return name.title() if name.isupper() else name
    raise ValueError(f"No AOI layer for {event}")


def domain_boxes(cfg: dict, pad_m: float) -> gpd.GeoSeries:
    """Bounding box of every exposure domain grown by ``pad_m`` (working CRS).

    Use these instead of buffering the domains: some burnt-area outlines have
    over a million vertices (EMSR909: 1.3 M, 1.6 m median segment), and a
    300 m buffer of such an outline exhausts memory.
    """
    b = load_domains(cfg).bounds
    return gpd.GeoSeries(
        shapely.box(b.minx - pad_m, b.miny - pad_m, b.maxx + pad_m, b.maxy + pad_m),
        index=b.index, crs=cfg["pipeline"]["crs"])


def write_domains(cfg: dict, packages: pd.DataFrame, events: pd.DataFrame) -> Path:
    crs = cfg["pipeline"]["crs"]
    chosen = events[events["selected"]]
    gdf = gpd.GeoDataFrame(
        chosen[["event", "activation", "activation_date"]].reset_index(drop=True),
        geometry=[exposure_domain(cfg, packages, e, crs) for e in chosen["event"]], crs=crs)
    if gdf.geometry.is_empty.any():
        raise ValueError(f"Empty exposure domain: {list(gdf.loc[gdf.geometry.is_empty, 'event'])}")
    out = resolve_path(DOMAINS_GPKG)
    out.parent.mkdir(parents=True, exist_ok=True)
    gdf.to_file(out, layer="domains", driver="GPKG")
    return out


def _union_layer(cfg: dict, urls, kind: str, crs: str):
    parts = [g.union_all() for u in urls if (g := read_layer(cfg, u, kind, crs)) is not None]
    return shapely.union_all(parts) if parts else None


def activation_burnt_area(cfg: dict, packages: pd.DataFrame, activation: str, crs: str):
    """Union of the burnt-area polygons of every package in one activation.
    Some AOIs ship their building points without a burnt-area layer; a
    neighbouring AOI of the same activation covers them."""
    urls = packages.loc[packages["activation"] == activation, "url"]
    burnt = _union_layer(cfg, urls, "burnt", crs)
    if burnt is None:
        raise ValueError(f"No burnt-area polygons in any package of {activation}")
    return burnt


def exposure_domain(cfg: dict, packages: pd.DataFrame, event: str, crs: str):
    """Union of the event's graded AOIs, intersected with the activation's burnt area."""
    pkgs = packages[packages["event"] == event]
    aoi = _union_layer(cfg, pkgs["url"], "aoi", crs)
    return aoi.intersection(activation_burnt_area(cfg, packages, pkgs["activation"].iloc[0], crs))


def merge_versions(layers: list[gpd.GeoDataFrame], tolerance_m: float) -> gpd.GeoDataFrame:
    """Merge point layers given oldest first: a point is kept unless a newer
    layer has a point within ``tolerance_m`` of it."""
    kept: list[gpd.GeoDataFrame] = []
    for layer in reversed(layers):
        if kept:
            newer = pd.concat(kept, ignore_index=True)
            near = layer.sjoin_nearest(newer[["geometry"]], how="left",
                                       max_distance=tolerance_m)
            layer = layer[~near.groupby(level=0)["index_right"].first().notna()]
        kept.append(layer)
    return gpd.GeoDataFrame(pd.concat(kept, ignore_index=True), crs=layers[0].crs)


def geometry_source(points: pd.DataFrame, sources: pd.DataFrame) -> pd.Series:
    """Name of the source each point's geometry was taken from: the package's
    source table entry for its ``or_src_id``. NaN for ids missing from the
    table (EMSR588 uses 997, "Not Applicable" elsewhere, without listing it)."""
    names = dict(zip(sources["src_id"].astype(str), sources["source_nam"]))
    return points["or_src_id"].astype(str).map(names)


def _point_layers(cfg: dict, packages: pd.DataFrame, event: str,
                  crs: str) -> list[gpd.GeoDataFrame]:
    urls = packages.loc[packages["event"] == event, "url"]  # oldest product first
    layers = []
    for u in urls:
        g = read_layer(cfg, u, "points", crs)
        if g is None or "damage_gra" not in g:
            continue
        sources = gpd.read_file(find_layer(package_dir(cfg, u), "source", ".dbf"),
                                ignore_geometry=True)
        g["geometry_source"] = geometry_source(g, sources)
        layers.append(g[["damage_gra", "geometry_source", "geometry"]])
    return layers


def event_points(cfg: dict, packages: pd.DataFrame, event: str, crs: str) -> gpd.GeoDataFrame:
    """Graded building points of one event, merged across its products, with
    the source EMS took each point's geometry from."""
    layers = _point_layers(cfg, packages, event, crs)
    if not layers:
        raise ValueError(f"No building-point grading for {event}")
    return merge_versions(layers, cfg["events"]["version_merge_tolerance_m"])


def grade_counts(cfg: dict, points: gpd.GeoDataFrame) -> dict:
    grades = points["damage_gra"]
    unknown = set(grades) - set(cfg["events"]["grade_order"])
    if unknown:
        raise ValueError(f"Unexpected EMS damage grades: {unknown}")
    vc = grades.value_counts()
    return {
        "n_points": len(points),
        "n_damaged_points": int(grades.isin(cfg["events"]["damaged_grades"]).sum()),
        **{f"n_{g.lower().replace(' ', '_')}": int(vc.get(g, 0))
           for g in cfg["events"]["grade_order"]},
    }


def select(cfg: dict, packages: pd.DataFrame) -> pd.DataFrame:
    crs = cfg["pipeline"]["crs"]
    min_dmg = cfg["events"]["min_damaged_points"]
    rows = []
    for event, pkgs in packages.groupby("event", sort=True):
        layers = _point_layers(cfg, packages, event, crs)
        row = {
            "event": event,
            "activation": pkgs["activation"].iloc[0],
            "activation_name": pkgs["activation_name"].iloc[0],
            "activation_date": pkgs["activation_date"].iloc[0],
            "n_products": len(pkgs),
            "has_points": bool(layers),
            "has_aoi": any(find_layer(package_dir(cfg, u), "aoi") for u in pkgs["url"]),
        }
        if layers:
            merged = merge_versions(layers, cfg["events"]["version_merge_tolerance_m"])
            row |= grade_counts(cfg, merged)
        if not row["has_points"]:
            reason = "no building-point grading"
        elif not row["has_aoi"]:
            reason = "no AOI polygon"
        elif row["n_damaged_points"] < min_dmg:
            reason = f"fewer than {min_dmg} damaged points"
        else:
            reason = "selected"
        rows.append(row | {"selected": reason == "selected", "reason": reason})
    return pd.DataFrame(rows)


def main() -> None:
    cfg = load_config()
    packages = list_packages(cfg)
    records = [download_package(cfg, u) for u in packages["url"]]
    events = select(cfg, packages)

    for df, rel in ((packages, PACKAGES_CSV), (events, EVENTS_CSV)):
        out = resolve_path(rel)
        out.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out, index=False, lineterminator="\n")
    write_manifest(resolve_path(cfg["sources"]["ems"]["output_dir"]), cfg["sources"]["ems"],
                   records)
    write_domains(cfg, packages, events)

    chosen = events[events["selected"]]
    print(events["reason"].value_counts().to_string())
    print(f"\n{len(chosen)} events in {chosen['activation'].nunique()} activations, "
          f"{int(chosen['n_damaged_points'].sum())} damaged points")


if __name__ == "__main__":
    main()
