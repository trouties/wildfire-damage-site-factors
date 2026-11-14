"""Per-event AUC of distance to wildland (CORINE and WorldCover), of a logistic
model on all static features and of the burn-severity control, for each choice
of undamaged buildings (``labellings``); method in README.md. Reads
data/analysis_table.csv and outputs/labels/label_summary.csv; writes
outputs/results/auc_by_event.csv and .png and the README results block.

A held-out event is scored with the slopes alone: the model has one intercept
per training event, and the event's own intercept would not change its AUC.
"""
from __future__ import annotations

import warnings

import matplotlib
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from scipy.stats import chi2, rankdata
from sklearn.linear_model import LogisticRegression

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from src.features.build import TABLE_CSV  # noqa: E402
from src.labels.build import SUMMARY_CSV  # noqa: E402
from src.utils.config import load_config, resolve_path  # noqa: E402

RESULTS_CSV = "outputs/results/auc_by_event.csv"
RESULTS_PNG = "outputs/results/auc_by_event.png"
CORRELATION_CSV = "outputs/results/label_correlation.csv"
COEFFICIENTS_CSV = "outputs/results/coefficients.csv"
README = "README.md"
BLOCK_BEGIN, BLOCK_END = "<!-- results:begin -->", "<!-- results:end -->"
BASELINE = "wildland_dist_m"
LOG1P = {"n_neighbours"}  # counts are right-skewed
SCORES = {"distance": "Distance to wildland, CORINE",
          "distance_wc": "Distance to wildland, WorldCover 10 m",
          "static": "All static features",
          "control": "Burn severity (dNBR), control"}
# The two distances measure one factor: one hue, told apart by marker shape.
COLOURS = {"distance": "#eb6834", "distance_wc": "#eb6834", "static": "#2a78d6",
           "control": "#1baf7a"}
MARKERS = {"distance": "o", "distance_wc": "D", "static": "o", "control": "o"}


def auc(score: np.ndarray, damaged: np.ndarray) -> float:
    """Mann-Whitney AUC; ties count one half."""
    damaged = np.asarray(damaged, bool)
    n1 = damaged.sum()
    n0 = damaged.size - n1
    if n1 == 0 or n0 == 0:
        raise ValueError("AUC needs damaged and undamaged buildings")
    ranks = rankdata(score)
    return float((ranks[damaged].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def block_ids(x: np.ndarray, y: np.ndarray, size_m: float) -> np.ndarray:
    """0..k-1 id of the ``size_m`` grid square containing each point."""
    cells = np.stack([np.floor(np.asarray(x) / size_m), np.floor(np.asarray(y) / size_m)], 1)
    return np.unique(cells, axis=0, return_inverse=True)[1].ravel()


def block_resamples(blocks: np.ndarray, damaged: np.ndarray, n_boot: int,
                    rng: np.random.Generator):
    """Index arrays of ``n_boot`` resamples, each as many blocks as there are,
    drawn with replacement. A draw without both classes is redrawn."""
    members = [np.flatnonzero(blocks == b) for b in range(blocks.max() + 1)]
    k, done = len(members), 0
    while done < n_boot:
        idx = np.concatenate([members[i] for i in rng.integers(0, k, k)])
        if 0 < damaged[idx].sum() < idx.size:
            done += 1
            yield idx


def design(df: pd.DataFrame, features: list[str]) -> np.ndarray:
    return np.column_stack([np.log1p(df[f]) if f in LOG1P else df[f] for f in features]
                           ).astype(float)


def fit_slopes(X: np.ndarray, damaged: np.ndarray, event: np.ndarray):
    """Unpenalised logistic regression with one intercept per event.
    Returns (mean, sd, slopes) for scoring standardised features."""
    mean, sd = X.mean(axis=0), X.std(axis=0)
    dummies = pd.get_dummies(event).to_numpy(float)
    model = LogisticRegression(C=np.inf, fit_intercept=False, max_iter=5000)
    with warnings.catch_warnings():
        # scikit-learn 1.8 documents C=np.inf as the way to turn the penalty
        # off, then warns that the C it was given is ignored.
        warnings.filterwarnings("ignore", "Setting penalty=None will ignore")
        model.fit(np.hstack([(X - mean) / sd, dummies]), damaged)
    return mean, sd, model.coef_[0, :X.shape[1]]


def score_slopes(slopes, X: np.ndarray) -> np.ndarray:
    mean, sd, beta = slopes
    return ((X - mean) / sd) @ beta


def held_out_scores(df: pd.DataFrame, damaged: np.ndarray, features: list[str]) -> np.ndarray:
    X = design(df, features)
    out = np.empty(len(df))
    for act in df["activation"].unique():
        test = (df["activation"] == act).to_numpy()
        slopes = fit_slopes(X[~test], damaged[~test], df["event"].to_numpy()[~test])
        out[test] = score_slopes(slopes, X[test])
    return out


def score_auc(score: np.ndarray, damaged: np.ndarray) -> float:
    """AUC over the buildings that have a score; NaN if none has (an event
    without a Sentinel-2 pair has no control score)."""
    has = ~np.isnan(score)
    return auc(score[has], damaged[has]) if has.any() else np.nan


def interval(a: np.ndarray) -> tuple[float, float]:
    if np.isnan(a).all():
        return np.nan, np.nan
    return tuple(np.percentile(a, [2.5, 97.5]))


def evaluate(df: pd.DataFrame, damaged: pd.Series, cfg: dict) -> pd.DataFrame:
    """One row per event plus a final ``mean`` row."""
    a = cfg["analysis"]
    keep = damaged.notna().to_numpy()
    df, y = df[keep].reset_index(drop=True), damaged[keep].astype(bool).to_numpy()
    scores = {"distance": -df[BASELINE].to_numpy(float),
              "distance_wc": -df[a["fine_baseline"]].to_numpy(float),
              "static": held_out_scores(df, y, a["features"]),
              "control": df[a["control"]].to_numpy(float)}
    rng = np.random.default_rng(a["seed"])
    rows, boots = [], []
    for event in df["event"].unique():
        m = (df["event"] == event).to_numpy()
        ye = y[m]
        s = {k: v[m] for k, v in scores.items()}
        blocks = block_ids(df.loc[m, "x"], df.loc[m, "y"], a["block_m"])
        b = np.array([[score_auc(s[k][i], ye[i]) for k in SCORES]
                      for i in block_resamples(blocks, ye, a["n_boot"], rng)])
        boots.append(b)
        rows.append({"event": event, "n_labelled": int(m.sum()), "n_damaged": int(ye.sum()),
                     "n_blocks": int(blocks.max() + 1),
                     **{k: score_auc(s[k], ye) for k in SCORES}, "boot": b})
    # Events without a control score drop out of the control's mean only.
    rows.append({"event": "mean", "n_labelled": len(df), "n_damaged": int(y.sum()),
                 "n_blocks": sum(r["n_blocks"] for r in rows),
                 **{k: np.nanmean([r[k] for r in rows]) for k in SCORES},
                 "boot": np.nanmean(boots, axis=0)})
    out, keys = [], list(SCORES)
    for r in rows:
        b = r.pop("boot")
        diff = b[:, keys.index("static")] - b[:, keys.index("distance")]
        r["diff"] = r["static"] - r["distance"]
        for j, k in enumerate(SCORES):
            r[f"{k}_lo"], r[f"{k}_hi"] = interval(b[:, j])
            r[f"{k}_se"] = np.nan if np.isnan(b[:, j]).all() else float(np.nanstd(b[:, j]))
        r["diff_lo"], r["diff_hi"] = interval(diff)
        out.append(r)
    return pd.DataFrame(out)


def heterogeneity(auc_by_event: pd.Series, se: pd.Series) -> tuple[float, float]:
    """Cochran's Q test of whether events' AUCs differ by more than their
    standard errors allow. Returns I² (share of the spread beyond sampling
    error) and the p-value."""
    w = 1 / se**2
    mean = (w * auc_by_event).sum() / w.sum()
    q = float((w * (auc_by_event - mean) ** 2).sum())
    dof = len(auc_by_event) - 1
    return (max(0.0, (q - dof) / q) if q > 0 else 0.0), float(chi2.sf(q, dof))


def label_correlation(df: pd.DataFrame, damaged: pd.Series, bin_m: float,
                      max_m: float) -> pd.DataFrame:
    """Correlation of the labels of two buildings of the same event, by their
    distance: per event and distance class, the mean over the pairs of
    z_i * z_j with z = (y - p) / sqrt(p (1 - p)) and p the event's damage share;
    then the mean over the events with pairs in the class."""
    keep = damaged.notna().to_numpy()
    df, y = df[keep], damaged[keep].astype(float).to_numpy()
    n_bins = int(np.ceil(max_m / bin_m))
    per_event, n_pairs = [], np.zeros(n_bins, int)
    for event in df["event"].unique():
        m = (df["event"] == event).to_numpy()
        xy, ye = df.loc[m, ["x", "y"]].to_numpy(), y[m]
        p = ye.mean()
        z = (ye - p) / np.sqrt(p * (1 - p))
        i, j = cKDTree(xy).query_pairs(max_m, output_type="ndarray").T
        k = (np.hypot(*(xy[i] - xy[j]).T) // bin_m).astype(int).clip(max=n_bins - 1)
        count = np.bincount(k, minlength=n_bins)
        with np.errstate(invalid="ignore", divide="ignore"):
            per_event.append(np.bincount(k, z[i] * z[j], minlength=n_bins) / count)
        n_pairs += count
    per_event = np.array(per_event)
    n_events = np.isfinite(per_event).sum(axis=0)
    mean = np.where(n_events > 0, np.nansum(per_event, axis=0) / np.maximum(n_events, 1), np.nan)
    edges = np.arange(n_bins) * bin_m
    return pd.DataFrame({"from_m": edges, "to_m": edges + bin_m, "correlation": mean.round(4),
                         "n_events": n_events, "n_pairs": n_pairs})


def coefficients(df: pd.DataFrame, damaged: pd.Series, features: list[str]) -> pd.DataFrame:
    """Standardised slopes of the model fitted on all events, and their range
    over the fits that leave one activation out."""
    keep = damaged.notna().to_numpy()
    df, y = df[keep].reset_index(drop=True), damaged[keep].astype(bool).to_numpy()
    X, event = design(df, features), df["event"].to_numpy()
    folds = np.array([fit_slopes(X[~t], y[~t], event[~t])[2]
                      for t in (df["activation"] == a for a in df["activation"].unique())])
    return pd.DataFrame({"feature": features, "all_events": fit_slopes(X, y, event)[2],
                         "fold_min": folds.min(axis=0), "fold_max": folds.max(axis=0)})


def graded_only_labels(df: pd.DataFrame, min_per_class: int) -> pd.Series:
    """Damaged against every other EMS grade; footprints EMS did not grade get
    no label, and so does every event with fewer than ``min_per_class``
    buildings in either class."""
    y = df["damaged"].fillna(False).where(df["damage_grade"].notna())
    n_pos = y.eq(True).groupby(df["event"]).transform("sum")
    n_neg = y.eq(False).groupby(df["event"]).transform("sum")
    return y.where((n_pos >= min_per_class) & (n_neg >= min_per_class))


def labellings(df: pd.DataFrame, cfg: dict, min_area_m2: float) -> dict[str, pd.Series]:
    """Label per building for each analysis (NA = left out). Damaged is the
    same in all of them; they differ in which buildings count as undamaged."""
    return {
        "graded_only": graded_only_labels(df, cfg["analysis"]["graded_only_min_per_class"]),
        "ungraded": df["damaged"],
        "ungraded_or_possibly": df["damaged"].fillna(False),
        "ungraded_min_area": df["damaged"].mask(df["area_m2"] < min_area_m2),
    }


def area_auc(df: pd.DataFrame) -> dict[str, float]:
    """Mean over events of the AUC of footprint area alone, for graded vs.
    ungraded and for damaged vs. possibly damaged footprints."""
    out = {"graded_vs_ungraded": [], "damaged_vs_possibly": []}
    for _, g in df.groupby("event", sort=False):
        area = g["area_m2"].to_numpy(float)
        graded = g["damage_grade"].notna().to_numpy(bool)
        damaged = g["damaged"].fillna(False).to_numpy(bool)
        possibly = g["damaged"].isna().to_numpy(bool)
        out["graded_vs_ungraded"].append(auc(area, graded))
        out["damaged_vs_possibly"].append(auc(area[damaged | possibly], damaged[damaged | possibly]))
    return {k: float(np.mean(v)) for k, v in out.items()}


def ungraded_share(df: pd.DataFrame) -> tuple[float, pd.Series]:
    """Median area of the EMS-graded footprints, and per event the share of
    footprints at least that large that EMS did not grade."""
    median = float(df.loc[df["damage_grade"].notna(), "area_m2"].median())
    large = df[df["area_m2"] >= median]
    return median, large["damage_grade"].isna().groupby(large["event"], sort=False).mean()


def plot(r: pd.DataFrame, title: str, path) -> None:
    """Per-event AUCs of one analysis, mean row last."""
    r = r.reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(7.5, 6.5))
    ypos = np.arange(len(r))[::-1].astype(float)
    ypos[-1] -= 0.6  # gap above the mean row
    for k, dy in zip(SCORES, (0.3, 0.1, -0.1, -0.3)):
        ax.hlines(ypos + dy, r[f"{k}_lo"], r[f"{k}_hi"], color=COLOURS[k], linewidth=1.2)
        ax.plot(r[k], ypos + dy, MARKERS[k], color=COLOURS[k], markersize=5.5,
                markeredgecolor="white", markeredgewidth=0.8, label=SCORES[k])
    ax.axvline(0.5, color="#8a8984", linewidth=1, linestyle="--", zorder=0)
    ax.axhline(ypos[-1] + 0.8, color="#d4d3cd", linewidth=0.8)
    ax.set_title(title, fontsize=9.5, loc="left", pad=44)
    ax.set_xlabel("AUC (0.5 = no better than chance)", fontsize=9)
    ax.set_xlim(0, 1)
    ax.grid(axis="x", color="#ebeae5", linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.tick_params(axis="y", length=0)
    ax.set_yticks(ypos, r["label"], fontsize=8.5)
    ax.legend(loc="lower left", bbox_to_anchor=(0, 1), ncol=2, fontsize=8, frameon=False,
              borderaxespad=0.2)
    fig.text(0.01, 0.01, "Bars: 95% interval from a spatial block bootstrap (1 km squares). "
             "Burn severity is a control, not a model input;\nit is missing where an event "
             "has no Sentinel-2 pair.", fontsize=7.5, color="#52514e")
    fig.tight_layout(rect=(0, 0.045, 1, 1))
    fig.savefig(path, dpi=120)


def fmt_ci(v: float, lo: float, hi: float, sign: bool = False) -> str:
    f = "{:+.2f}" if sign else "{:.2f}"
    return f"{f.format(v)} [{f.format(lo)}, {f.format(hi)}]"


def readme_block(results: pd.DataFrame, rows: dict[str, str], area: dict[str, float],
                 ungraded: tuple[float, pd.Series], n_no_visible: int,
                 labels: dict[str, str], fine_dist: pd.Series, coef: pd.DataFrame,
                 corr: pd.DataFrame) -> str:
    def ci(m: pd.Series, k: str) -> str:
        return fmt_ci(m[k], m[f"{k}_lo"], m[f"{k}_hi"])

    per_event = results[results["event"] != "mean"]
    n_events = per_event.groupby("analysis").size()
    lines = ["| Counted as undamaged | Buildings (damaged) | Distance, CORINE "
             "| Distance, WorldCover 10 m | All static features "
             "| Static minus CORINE distance (events > 0 / < 0) | Burn severity (control) |",
             "|---|---:|---:|---:|---:|---:|---:|"]
    for a, title in rows.items():
        ev = per_event[per_event["analysis"] == a]
        m = results[(results["analysis"] == a) & (results["event"] == "mean")].iloc[0]
        if n_events.nunique() > 1:
            title = f"{title}, {n_events[a]} events"
        lines.append(f"| {title} | {m.n_labelled:,} ({m.n_damaged:,}) | {ci(m, 'distance')} "
                     f"| {ci(m, 'distance_wc')} | {ci(m, 'static')} "
                     f"| {fmt_ci(m['diff'], m.diff_lo, m.diff_hi, sign=True)} "
                     f"({int((ev.diff_lo > 0).sum())} / {int((ev.diff_hi < 0).sum())}) "
                     f"| {ci(m, 'control')} |")
    median_m2, share = ungraded
    lo, hi = share.idxmin(), share.idxmax()
    head = per_event[per_event["analysis"] == "graded_only"]
    no_control = [labels[e] for e in head.loc[head["control"].isna(), "event"]]

    def beyond_chance(analysis: str, score: str) -> str:
        """Events whose interval lies wholly above / below 0.5, of those with the score."""
        ev = per_event[(per_event["analysis"] == analysis) & per_event[score].notna()]
        return (f"{int((ev[f'{score}_lo'] > 0.5).sum())} / "
                f"{int((ev[f'{score}_hi'] < 0.5).sum())} of {len(ev)}")

    counts = ["| Score | Events above / below 0.5, undamaged = *Possibly damaged* "
              "| Events above / below 0.5, undamaged = not graded "
              "| Spread between events beyond sampling error, headline: I² (p) |",
              "|---|---:|---:|---:|"]
    for k in SCORES:
        e = head[head[k].notna()]
        i2, p = heterogeneity(e[k], e[f"{k}_se"])
        counts.append(f"| {SCORES[k]} | {beyond_chance('graded_only', k)} "
                      f"| {beyond_chance('ungraded', k)} "
                      f"| {i2:.0%} ({'< 0.001' if p < 0.001 else f'{p:.3f}'}) |")
    coef_lines = ["| Feature | Slope, all events | Range over the held-out fits |",
                  "|---|---:|---:|"]
    for c in coef.itertuples():
        coef_lines.append(f"| `{c.feature}` | {c.all_events:+.2f} "
                          f"| {c.fold_min:+.2f} to {c.fold_max:+.2f} |")
    corr_at = corr.set_index("from_m")["correlation"]
    bin_m = int(corr["to_m"].iloc[0] - corr["from_m"].iloc[0])

    n_note = (f"All rows cover the same {n_events.iloc[0]} events. " if n_events.nunique() == 1
              else "")
    return "\n".join([
        f"Mean AUC over the events, with 95% intervals. {n_note}In every row the damaged "
        "buildings are those EMS graded *Damaged* or *Destroyed*; the rows differ in which "
        f"buildings count as undamaged. The {n_no_visible} buildings EMS graded *No visible "
        "damage* count as undamaged in every row. The distance from WorldCover was added "
        "after the first results. In brackets after the difference, the number of events "
        "whose interval of the difference lies wholly above or below zero. Burn severity is "
        "the control, not a model input; "
        + (f"{' and '.join(no_control)} have no Sentinel-2 pair and are left out of its mean."
           if no_control else "every event has a Sentinel-2 pair."),
        "",
        *lines,
        "",
        *counts,
        "",
        "The first two columns count the events whose interval lies wholly above or below "
        "0.5, of the events with that score. I² is the share of the spread of the per-event "
        "AUCs that their bootstrap errors do not explain (Cochran's Q). The labels of two "
        f"buildings less than {bin_m} m apart correlate at {corr_at[0]:.2f}, at "
        f"{corr_at[500]:.2f} for buildings 500-{500 + bin_m} m apart and at "
        f"{corr_at[1000]:.2f} for 1000-{1000 + bin_m} m (headline labels, "
        "`label_correlation.csv`).",
        "",
        "Model slopes in the headline comparison, in log-odds per standard deviation of the "
        "feature (`n_neighbours` as log(1 + n)); a negative slope for `wildland_dist_m` means "
        "nearer to wildland, more damage.",
        "",
        *coef_lines,
        "",
        f"At 10 m, {(fine_dist == 0).mean():.0%} of all {len(fine_dist):,} buildings stand on "
        f"a WorldCover tree, shrub or grass cell and {(fine_dist <= 20).mean():.0%} lie within "
        "20 m of one, so the finer distance mostly separates buildings on such a cell from "
        "those next to one.",
        "",
        f"Footprint area alone, which is not one of the features, gives a mean AUC of "
        f"{area['graded_vs_ungraded']:.2f} for EMS-graded vs. ungraded footprints but "
        f"{area['damaged_vs_possibly']:.2f} for damaged vs. possibly damaged ones. Of the "
        f"footprints at least as large as the median EMS-graded one ({median_m2:.0f} m²), "
        f"the share EMS did not grade ranges by event from {share[lo]:.0%} ({labels[lo]}) "
        f"to {share[hi]:.0%} ({labels[hi]}).",
    ])


def write_readme_block(text: str) -> None:
    path = resolve_path(README)
    readme = path.read_text(encoding="utf-8")
    head, sep1, rest = readme.partition(BLOCK_BEGIN)
    _, sep2, tail = rest.partition(BLOCK_END)
    if not (sep1 and sep2):
        raise ValueError(f"{README} lacks the {BLOCK_BEGIN} ... {BLOCK_END} markers")
    path.write_text(f"{head}{BLOCK_BEGIN}\n{text}\n{BLOCK_END}{tail}", encoding="utf-8",
                    newline="\n")


def main() -> None:
    cfg = load_config()
    df = pd.read_csv(resolve_path(TABLE_CSV), dtype={"damaged": "boolean"})
    graded_area = df.loc[df["damage_grade"].notna(), "area_m2"]
    min_area_m2 = float(np.quantile(graded_area, cfg["analysis"]["graded_area_quantile"]))
    rows = {
        "graded_only": "*Possibly damaged* (EMS-graded buildings only; added after the "
                       "first results)",
        "ungraded": "Footprints EMS did not grade",
        "ungraded_or_possibly": "Footprints EMS did not grade, and *Possibly damaged*",
        "ungraded_min_area": f"Footprints EMS did not grade, none below {min_area_m2:.0f} m² "
                             "(check added after the first run)",
    }
    events = pd.read_csv(resolve_path(SUMMARY_CSV)).set_index("event")
    labels = {e: f"{events.at[e, 'locality']} {events.at[e, 'activation_date'][:4]}"
              for e in df["event"].unique()}
    labels["mean"] = "Mean of events"

    a = cfg["analysis"]
    label_sets = labellings(df, cfg, min_area_m2)
    results = pd.concat([evaluate(df, y, cfg).assign(analysis=name)
                         for name, y in label_sets.items()], ignore_index=True)
    results.insert(1, "label", results["event"].map(labels))
    results = results[["analysis", *results.columns.drop("analysis")]]
    coef = coefficients(df, label_sets["graded_only"], a["features"])
    corr = label_correlation(df, label_sets["graded_only"], a["label_correlation_bin_m"],
                             a["label_correlation_max_m"])

    csv = resolve_path(RESULTS_CSV)
    csv.parent.mkdir(parents=True, exist_ok=True)
    results.round(4).to_csv(csv, index=False, lineterminator="\n")
    coef.round(4).to_csv(resolve_path(COEFFICIENTS_CSV), index=False, lineterminator="\n")
    corr.to_csv(resolve_path(CORRELATION_CSV), index=False, lineterminator="\n")
    plot(results[results["analysis"] == "graded_only"],
         "EMS-graded buildings: Damaged or Destroyed vs. Possibly damaged\n"
         "inside the burnt area; comparison added after the first results",
         resolve_path(RESULTS_PNG))
    n_no_visible = int((df["damage_grade"] == "No visible damage").sum())
    block = readme_block(results, rows, area_auc(df), ungraded_share(df), n_no_visible, labels,
                         df[a["fine_baseline"]], coef, corr)
    write_readme_block(block)
    print(block)


if __name__ == "__main__":
    main()
