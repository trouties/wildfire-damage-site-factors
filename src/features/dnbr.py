"""Burn severity around every study building from Sentinel-2: the positive
control. Rules in config/features.yaml (``dnbr``), source in
config/data_sources.yaml (``sentinel2``).

Per event, a pre- and a post-fire acquisition are chosen by those rules, and
B8A, B12 and SCL are read window by window from the Cloud-Optimized GeoTIFFs
onto a grid in the working CRS (bilinear for reflectance, nearest for SCL).
Nothing is stored but the results:
  data/processed/dnbr.csv        event, building_id, dnbr_100m (read by
                                 src.features.build)
  outputs/sentinel2/scenes.csv   the acquisitions used per event, or why none
  outputs/sentinel2/dnbr_map.png the event with the most damaged buildings
  data/raw/sentinel2/manifest.json
"""
from __future__ import annotations

import re
from datetime import datetime, timezone

import geopandas as gpd
import matplotlib
import numpy as np
import pandas as pd
import rasterio
import requests
from affine import Affine
from rasterio.enums import Resampling
from rasterio.plot import plotting_extent
from rasterio.transform import array_bounds
from rasterio.vrt import WarpedVRT
from rasterio.warp import transform_bounds

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

from src.acquire.ems import load_domains  # noqa: E402
from src.features.rasters import disc_kernel, disc_mean, grid, sample  # noqa: E402
from src.labels.build import SUMMARY_CSV, labelled_buildings  # noqa: E402
from src.labels.maps import scale_bar_m  # noqa: E402
from src.utils.config import load_config, resolve_path  # noqa: E402
from src.utils.download import write_manifest  # noqa: E402

DNBR_CSV = "data/processed/dnbr.csv"
SCENES_CSV = "outputs/sentinel2/scenes.csv"
MAP_PNG = "outputs/sentinel2/dnbr_map.png"
COLUMN = "dnbr_100m"
GDAL_ENV = {"GDAL_DISABLE_READDIR_ON_OPEN": "EMPTY_DIR",
            "CPL_VSIL_CURL_ALLOWED_EXTENSIONS": ".tif",
            "GDAL_HTTP_MERGE_CONSECUTIVE_RANGES": "YES",
            "GDAL_HTTP_MULTIRANGE": "YES",
            "GDAL_HTTP_MAX_RETRY": "5",
            "GDAL_HTTP_RETRY_DELAY": "2"}
ORBIT_RE = re.compile(r"_R(\d{3})_")


def search(src: dict, bbox: list[float], start: pd.Timestamp, end: pd.Timestamp) -> list[dict]:
    body = {"collections": [src["collection"]], "bbox": bbox, "limit": 100,
            "datetime": f"{start:%Y-%m-%d}T00:00:00Z/{end:%Y-%m-%d}T23:59:59Z"}
    resp = requests.post(src["stac_search"], json=body, timeout=120)
    resp.raise_for_status()
    page = resp.json()
    # Earth Search links a next page even when this one holds every match.
    if page["numberMatched"] > len(page["features"]):
        raise RuntimeError(f"STAC search {body} has more than one page")
    return page["features"]


def acquisitions(items: list[dict], max_cloud_pct: float) -> list[dict]:
    """Items grouped by datatake (one pass of one satellite), oldest first.
    Tiles above the cloud prefilter are dropped before grouping."""
    groups: dict[str, list[dict]] = {}
    for it in items:
        if it["properties"]["eo:cloud_cover"] <= max_cloud_pct:
            groups.setdefault(it["properties"]["s2:datatake_id"], []).append(it)
    acqs = [{"datatake": k,
             "time": pd.Timestamp(v[0]["properties"]["datetime"]),
             "orbit": int(ORBIT_RE.search(v[0]["properties"]["s2:product_uri"])[1]),
             "items": v} for k, v in groups.items()]
    return sorted(acqs, key=lambda a: a["time"])


def choose_pair(pre: list[dict], post: list[dict], usable) -> tuple[dict | None, dict | None, str]:
    """The usable post-fire acquisition closest after the fire, and the usable
    pre-fire acquisition closest before it on the same relative orbit.
    ``pre`` and ``post`` are oldest first; ``usable(acq)`` decides."""
    if not post:
        return None, None, "no post-fire acquisition in the catalogue"
    after = next((a for a in post if usable(a)), None)
    if after is None:
        return None, None, "no usable post-fire acquisition"
    before = next((a for a in reversed(pre) if a["orbit"] == after["orbit"] and usable(a)), None)
    if before is None:
        return None, after, "no usable pre-fire acquisition on the post-fire orbit"
    return before, after, ""


def read_asset(acq: dict, asset: str, crs: str, transform: Affine, shape: tuple[int, int],
               resampling: Resampling) -> np.ndarray:
    """One asset of an acquisition on the grid. Tiles fill each other's gaps;
    0 is no data in every asset."""
    out = None
    with rasterio.Env(**GDAL_ENV):
        for it in acq["items"]:
            with (rasterio.open(it["assets"][asset]["href"]) as src,
                  WarpedVRT(src, crs=crs, transform=transform, width=shape[1],
                            height=shape[0], resampling=resampling, nodata=0) as vrt):
                a = vrt.read(1)
            out = a if out is None else np.where(out == 0, a, out)
    return out


def reflectance(acq: dict, asset: str, dn: np.ndarray) -> np.ndarray:
    """Digital numbers to reflectance with the scale and offset the items
    declare, which must agree across the acquisition's tiles."""
    bands = {(b["scale"], b["offset"]) for it in acq["items"]
             for b in it["assets"][asset]["raster:bands"]}
    if len(bands) != 1:
        raise ValueError(f"{acq['datatake']} {asset}: tiles disagree on scale/offset {bands}")
    scale, offset = bands.pop()
    return np.where(dn == 0, np.nan, dn * scale + offset)


def nbr(nir: np.ndarray, swir: np.ndarray) -> np.ndarray:
    total = nir + swir
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(total > 0, (nir - swir) / total, np.nan)


def disc_average(values: np.ndarray, valid: np.ndarray, radius_px: float) -> np.ndarray:
    """Mean of ``values`` over the valid cells within ``radius_px`` of every
    cell; NaN where the disc holds no valid cell."""
    num = disc_mean(np.where(valid, values, 0.0), radius_px)
    den = disc_mean(valid.astype(float), radius_px)
    # den is (valid cells) / (cells in the disc); one valid cell gives at least
    # 1 / kernel size, and fftconvolve leaves round-off far below half of that.
    has_valid = den > 0.5 / disc_kernel(radius_px).sum()
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(has_valid, num / den, np.nan)


def event_dnbr(cfg: dict, event: str, activation_date: str, domain, xy: np.ndarray,
               cache: dict) -> dict:
    """Chosen acquisitions and the per-building dNBR of one event."""
    d, src, crs = cfg["features"]["dnbr"], cfg["sources"]["sentinel2"], cfg["pipeline"]["crs"]
    transform, shape = grid(domain.bounds, d["cell_m"], d["radius_m"] + d["cell_m"])
    bbox = list(transform_bounds(crs, "EPSG:4326", *array_bounds(*shape, transform)))
    act = pd.Timestamp(activation_date)
    windows = {k: (act + pd.Timedelta(days=d[f"{k}_days"][0]),
                   act + pd.Timedelta(days=d[f"{k}_days"][1])) for k in ("pre", "post")}
    acqs = {k: acquisitions(search(src, bbox, *w), d["max_tile_cloud_pct"])
            for k, w in windows.items()}
    x, y = xy[:, 0], xy[:, 1]

    def scl(acq):
        if acq["datatake"] not in cache:
            cache[acq["datatake"]] = read_asset(acq, src["assets"]["scl"], crs, transform,
                                                shape, Resampling.nearest)
        return cache[acq["datatake"]]

    def valid_share(acq):
        valid = ~np.isin(scl(acq), d["invalid_scl"])
        acq["valid_share"] = float(sample(valid, transform, x, y).mean())
        return acq["valid_share"]

    before, after, reason = choose_pair(
        acqs["pre"], acqs["post"], lambda a: valid_share(a) >= d["min_valid_building_share"])
    out = {"event": event, "before": before, "after": after, "reason": reason,
           "transform": transform, "dnbr": None}
    if reason:
        out["values"] = np.full(len(xy), np.nan)
        return out

    ratio = {}
    for name, acq in (("pre", before), ("post", after)):
        bands = {k: reflectance(acq, src["assets"][k],
                                read_asset(acq, src["assets"][k], crs, transform, shape,
                                           Resampling.bilinear)) for k in ("nir", "swir")}
        ratio[name] = nbr(bands["nir"], bands["swir"])
    dnbr = ratio["pre"] - ratio["post"]
    valid = (~np.isin(scl(before), d["invalid_scl"] + d["invalid_scl_pre_fire"])
             & ~np.isin(scl(after), d["invalid_scl"]) & np.isfinite(dnbr))
    out["dnbr"] = np.where(valid, dnbr, np.nan)
    out["values"] = sample(disc_average(dnbr, valid, d["radius_m"] / d["cell_m"]),
                           transform, x, y)
    return out


def scene_row(res: dict, n_buildings: int) -> dict:
    row = {"event": res["event"], "status": res["reason"] or "ok"}
    for name, acq in (("pre", res["before"]), ("post", res["after"])):
        row[f"{name}_datatake"] = acq["datatake"] if acq else ""
        row[f"{name}_utc"] = acq["time"].strftime("%Y-%m-%dT%H:%M:%SZ") if acq else ""
        row[f"{name}_items"] = " ".join(it["id"] for it in acq["items"]) if acq else ""
        row[f"{name}_valid_share"] = round(acq["valid_share"], 3) if acq else np.nan
    row["relative_orbit"] = res["after"]["orbit"] if res["after"] else pd.NA
    row["n_buildings"] = n_buildings
    row["n_without_value"] = int(np.isnan(res["values"]).sum())
    return row


def plot_map(res: dict, domain, buildings: pd.DataFrame, title: str, path) -> None:
    """dNBR of one event with the EMS-graded buildings on top."""
    fig, ax = plt.subplots(figsize=(8.5, 7.5))
    grey = LinearSegmentedColormap.from_list("grey", ["#f0efec", "#3d3d3a"])
    grey.set_bad("#ffffff")
    img = ax.imshow(res["dnbr"], cmap=grey, vmin=0, vmax=0.8, interpolation="nearest",
                    extent=plotting_extent(res["dnbr"], res["transform"]))
    gpd.GeoSeries([domain.simplify(5)]).boundary.plot(ax=ax, color="#8a8984", linewidth=0.6)
    styles = [("possibly", buildings["damaged"].isna() & buildings["damage_grade"].notna(),
               "#2a78d6", 6, "Possibly damaged"),
              ("damaged", buildings["damaged"].eq(True), "#e34948", 12, "Damaged or destroyed")]
    for _, mask, colour, size, _ in styles:
        ax.scatter(buildings.loc[mask, "x"], buildings.loc[mask, "y"], s=size, c=colour,
                   edgecolors="#ffffff", linewidths=0.4, zorder=3)
    x0, y0, x1, y1 = domain.bounds
    ax.set_xlim(x0, x1)
    ax.set_ylim(y0, y1)
    bar = scale_bar_m(x1 - x0)
    ax.plot([x0 + 0.03 * (x1 - x0), x0 + 0.03 * (x1 - x0) + bar], [y0, y0], color="#3d3d3a",
            linewidth=3, solid_capstyle="butt")
    ax.text(x0 + 0.03 * (x1 - x0), y0, f"{bar / 1000:g} km\n", va="bottom", fontsize=8,
            color="#52514e")
    ax.set_axis_off()
    ax.set_aspect("equal")
    ax.set_title(title, fontsize=10, loc="left")
    handles = [Line2D([], [], marker="o", linestyle="", markersize=6, markerfacecolor=c,
                      markeredgewidth=0, label=label) for _, _, c, _, label in styles]
    handles.append(Line2D([], [], color="#8a8984", linewidth=0.8, label="Exposure domain"))
    ax.legend(handles=handles, loc="upper right", fontsize=8, frameon=True, framealpha=0.9)
    cbar = fig.colorbar(img, ax=ax, shrink=0.6, extend="both")
    cbar.set_label("dNBR (pre-fire NBR minus post-fire NBR)", fontsize=8)
    cbar.ax.tick_params(labelsize=8)
    fig.text(0.01, 0.01, "Footprints EMS did not grade are not shown. White: no valid "
             "Sentinel-2 cell (cloud, water before the fire).", fontsize=7.5, color="#52514e")
    fig.tight_layout(rect=(0, 0.02, 1, 1))
    fig.savefig(path, dpi=120)


def main() -> None:
    cfg = load_config()
    src = cfg["sources"]["sentinel2"]
    rows = labelled_buildings()
    domains = load_domains(cfg)
    summary = pd.read_csv(resolve_path(SUMMARY_CSV)).set_index("event")
    map_event = rows.groupby("event")["damaged"].sum().idxmax()

    values, scenes, records = [], [], []
    scale_offset: set[tuple[float, float]] = set()
    retrieved = datetime.now(timezone.utc).isoformat(timespec="seconds")
    for event, grp in rows.groupby("event", sort=False):
        res = event_dnbr(cfg, event, domains.at[event, "activation_date"],
                         domains.geometry[event], grp[["x", "y"]].to_numpy(), cache={})
        for acq in (res["before"], res["after"]):
            for it in (acq["items"] if acq and not res["reason"] else []):
                scale_offset |= {(b["scale"], b["offset"]) for k in ("nir", "swir")
                                 for b in it["assets"][src["assets"][k]]["raster:bands"]}
        values.append(grp[["event", "building_id"]].assign(**{COLUMN: res["values"]}))
        scenes.append(scene_row(res, len(grp)))
        for acq in (res["before"], res["after"]):
            for it in (acq["items"] if acq else []):
                records.append({"id": it["id"], "datetime": it["properties"]["datetime"],
                                "stac_item": next(lk["href"] for lk in it["links"]
                                                  if lk["rel"] == "self"),
                                "assets": {k: it["assets"][a]["href"]
                                           for k, a in src["assets"].items()},
                                "retrieved_utc": retrieved})
        print(f"{event}: {scenes[-1]['status']}, {scenes[-1]['pre_utc'][:10]} -> "
              f"{scenes[-1]['post_utc'][:10]}, {scenes[-1]['n_without_value']} of "
              f"{len(grp)} buildings without a value")
        if event == map_event and res["dnbr"] is not None:
            title = (f"{summary.at[event, 'locality']} {summary.at[event, 'activation_date'][:4]}: "
                     f"burn severity from Sentinel-2 and EMS building grades")
            out = resolve_path(MAP_PNG)
            out.parent.mkdir(parents=True, exist_ok=True)
            plot_map(res, domains.geometry[event], grp, title, out)

    # Before processing baseline 04.00 L2A had no reflectance offset; a pair
    # mixing the two would shift dNBR. Collection 1 should have one throughout.
    if len(scale_offset) != 1:
        raise ValueError(f"Reflectance scale/offset differ between scenes: {scale_offset}")
    table = pd.concat(values, ignore_index=True)
    table[COLUMN] = table[COLUMN].round(3)
    table.to_csv(resolve_path(DNBR_CSV), index=False, lineterminator="\n")
    scenes_out = resolve_path(SCENES_CSV)
    scenes_out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(scenes).astype({"relative_orbit": "Int64"}).to_csv(
        scenes_out, index=False, lineterminator="\n")
    write_manifest(resolve_path(src["output_dir"]), src, records,
                   stac_search=src["stac_search"], collection=src["collection"])
    print(f"{table[COLUMN].notna().sum():,} of {len(table):,} buildings with {COLUMN} "
          f"-> {DNBR_CSV}")


if __name__ == "__main__":
    main()
