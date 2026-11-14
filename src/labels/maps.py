"""Map of every study event: exposure domain and labelled building centroids.

A visual check of the labels against the published EMS grading maps.
Output: outputs/labels/label_maps.png
"""
from __future__ import annotations

import math

import geopandas as gpd
import matplotlib
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

from src.acquire.ems import load_domains  # noqa: E402
from src.labels.build import SUMMARY_CSV, labelled_buildings  # noqa: E402
from src.utils.config import load_config, resolve_path  # noqa: E402

OUT_PNG = "outputs/labels/label_maps.png"
COLS = 4
# Footprints EMS did not grade recede; the two EMS-graded classes carry identity.
STYLE = {
    "undamaged": dict(color="#7d7c77", s=1.5, label="Not graded by EMS, or no visible damage"),
    "possibly": dict(color="#2a78d6", s=2.0, label="Possibly damaged"),
    "damaged": dict(color="#e34948", s=5.0, label="Damaged or destroyed"),
}
DOMAIN = dict(facecolor="#efe9dc", edgecolor="#b8ae98", linewidth=0.3)


def scale_bar_m(width_m: float) -> int:
    """Largest 1-2-5 step that fits in a quarter of the panel width."""
    target = width_m / 4
    step = 10 ** math.floor(math.log10(target))
    return max(m * step for m in (1, 2, 5) if m * step <= target)


def main() -> None:
    cfg = load_config()
    rows = labelled_buildings()
    domains = load_domains(cfg)
    locality = pd.read_csv(resolve_path(SUMMARY_CSV)).set_index("event")["locality"]
    events = sorted(rows["event"].unique(), key=lambda e: (domains.loc[e, "activation_date"], e))

    n_rows = math.ceil((len(events) + 1) / COLS)
    fig, axes = plt.subplots(n_rows, COLS, figsize=(3.4 * COLS, 3.6 * n_rows))
    for ax in axes.flat:
        ax.set_axis_off()
    for ax, event in zip(axes.flat, events):
        dom = gpd.GeoSeries([domains.geometry[event].simplify(5)], crs=domains.crs)
        dom.plot(ax=ax, **DOMAIN)
        g = rows[rows["event"] == event]
        classes = {
            "undamaged": g["damaged"].eq(False),
            "possibly": g["damaged"].isna(),
            "damaged": g["damaged"].eq(True),
        }
        for name, mask in classes.items():
            st = STYLE[name]
            ax.scatter(g.loc[mask, "x"], g.loc[mask, "y"], s=st["s"], c=st["color"],
                       linewidths=0, rasterized=True)
        x0, y0, x1, y1 = dom.total_bounds
        bar = scale_bar_m(x1 - x0)
        ax.plot([x0, x0 + bar], [y0, y0], color="#3d3d3a", linewidth=2, solid_capstyle="butt")
        ax.text(x0, y0, f"\n{bar / 1000:g} km", va="top", ha="left", fontsize=7,
                color="#52514e")
        ax.set_aspect("equal")
        year = domains.loc[event, "activation_date"][:4]
        n_dmg = int(classes["damaged"].sum())
        ax.set_title(f"{locality[event]} {year}\n"
                     f"{event.replace('_', ' ')} · {n_dmg} damaged of {len(g):,}",
                     fontsize=8, color="#0b0b0b")
    legend_ax = axes.flat[len(events)]
    handles = [Line2D([], [], marker="o", linestyle="", markersize=6,
                      markerfacecolor=s["color"], markeredgewidth=0, label=s["label"])
               for s in STYLE.values()]
    handles.append(plt.Rectangle((0, 0), 1, 1, **DOMAIN,
                                 label="Exposure domain (graded AOI ∩ burnt area)"))
    legend_ax.legend(handles=handles, loc="center", frameon=False, fontsize=8)
    fig.tight_layout()
    out = resolve_path(OUT_PNG)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=110)
    print(f"{len(events)} event maps -> {out.name}")


if __name__ == "__main__":
    main()
