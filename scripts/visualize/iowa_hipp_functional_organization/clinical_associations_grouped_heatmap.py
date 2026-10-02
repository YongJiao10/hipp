"""Side-by-side HC-only / patient-only clinical association heatmap.

Selection (self-consistent with what is displayed, NOT the pooled figure):
  - columns: fields with >= 1 nominal p<0.05 cell in standard/hc:sensitivity
    OR standard/patient:sensitivity (metrics area_percent|segregation|margin).
    Order: scale block -> level (total/subscale before items) ->
    min within-group raw p across both groups.
  - panels: standard/hc (top) vs standard/patient (bottom),
    same column order. Asterisk = raw within-group p < 0.05, no FDR.
  - MADRS-SR recorded totals are quarantined from inference by pipeline
    policy: excluded from selection even if nominally hit.
  - the group-difference screen (clinical_associations_group_screen.csv)
    is reported separately and plays no role here.

Reads only manifest-declared results/*.csv.
Writes clinical_associations_grouped.{csv,png}.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
PREFIX = "clinical_associations_grouped"
PRIMARY_METRICS = ["area_percent", "segregation", "margin"]
METRIC_SHORT = {"area_percent": "Area", "segregation": "Segr", "margin": "Marg"}
SYSTEMS = ["A", "B", "P"]
HEMIS = ["L", "R"]
SYSTEM_LABELS = {"A": "Ant", "B": "Mid", "P": "Post"}
SCALE_ORDER = ["PHQ", "BDI-II", "MADRS-SR", "STAI", "PANAS",
               "DSM-5 Cross-cutting", "PCL-5", "NIH Emotion", "WHODAS"]
SCALE_COLORS = {
    "PHQ": "#b42318", "BDI-II": "#e97316", "MADRS-SR": "#c77700",
    "STAI": "#175cd3", "PANAS": "#0e7490",
    "DSM-5 Cross-cutting": "#6941c6", "PCL-5": "#a1225c",
    "NIH Emotion": "#067647", "WHODAS": "#475467",
}
LEVEL_ORDER = {"total": 0, "subscale": 1, "item": 2}
FALLBACK_GRAY = "#98a2b3"


def load() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    stats = pd.read_csv(RESULTS / "clinical_associations_statistics.csv")
    screen = pd.read_csv(RESULTS / "clinical_associations_group_screen.csv")
    valuable = pd.read_csv(RESULTS / "clinical_associations_valuable.csv")
    return stats, screen, valuable


def grouped_fields(stats: pd.DataFrame) -> list[dict]:
    """Union of within-group hit fields, hierarchically ordered.

    Keeps fields with >= 1 nominal p<0.05 cell in standard/hc or
    standard/patient (primary metrics). MADRS-SR recorded totals are
    quarantined (never selected). Order: scale block -> level ->
    min within-group raw p across both groups.
    """
    sub = stats.loc[
        (stats["branch"] == "standard")
        & (stats["population"].isin(["hc", "patient"]))
        & (stats["metric"].isin(PRIMARY_METRICS))].copy()
    sub = sub.loc[~((sub["scale"] == "MADRS-SR") & (sub["level"] == "total"))]
    sig = sub.loc[sub["p"] < 0.05]
    if sig.empty:
        raise ValueError("no nominal within-group hits; nothing to plot")
    rows = []
    for field, group in sig.groupby("field"):
        first = group.iloc[0]
        rows.append({"field": str(field), "scale": str(first["scale"]),
                     "level": str(first.get("level", "")),
                     "label": str(first["label"]),
                     "min_p": float(group["p"].min())})

    def key(row: dict):
        try:
            scale_idx = SCALE_ORDER.index(row["scale"])
        except ValueError:
            scale_idx = len(SCALE_ORDER)
        return (scale_idx, LEVEL_ORDER.get(row["level"], 9), row["min_p"])

    return sorted(rows, key=key)


def heatmap_png(stats: pd.DataFrame, order: list[dict], path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors
    from matplotlib.patches import Rectangle
    try:
        from clinical_associations_valuable import _display
    except ImportError:  # fallback: local shorten (labels may be longer)
        import textwrap

        def _display(field: str, label: str) -> str:
            return textwrap.shorten(str(label), width=40, placeholder="…")

    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
        "axes.linewidth": 0.8, "axes.edgecolor": "#344054",
        "xtick.color": "#344054", "ytick.color": "#101828",
    })
    fields = [r["field"] for r in order]
    n = len(fields)
    row_keys = [(m, h, s) for m in PRIMARY_METRICS
                for h in HEMIS for s in SYSTEMS]
    nrows = len(row_keys)
    panels = ["hc", "patient"]
    mats, pmats = {}, {}
    for pop in panels:
        sub = stats.loc[(stats["branch"] == "standard")
                        & (stats["population"] == pop)
                        & (stats["metric"].isin(PRIMARY_METRICS))]
        lookup = {(str(r["field"]), r["metric"], r["hemi"], r["system"]): r
                  for r in sub.to_dict("records")}
        mat = np.full((nrows, n), np.nan)
        pmat = np.full((nrows, n), np.nan)
        for i, (metric, hemi, system) in enumerate(row_keys):
            for j, field in enumerate(fields):
                rec = lookup.get((field, metric, hemi, system))
                if rec is not None:
                    mat[i, j] = pd.to_numeric(rec.get("rho"), errors="coerce")
                    pmat[i, j] = pd.to_numeric(rec.get("p"), errors="coerce")
        mats[pop], pmats[pop] = mat, pmat

    vmin, vmax = -0.6, 0.6
    cmap = plt.get_cmap("RdBu_r")
    norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
    faint = np.array(mcolors.to_rgb("#eef2f7"))
    missing_rgb = np.array(mcolors.to_rgb("#e5e7eb"))

    def cell_rgb(rho: float, is_sig: bool):
        if np.isnan(rho):
            return missing_rgb
        rgb = np.array(cmap(norm(rho))[:3])
        return rgb if is_sig else faint + (rgb - faint) * 0.35

    runs: list[tuple[str, int, int]] = []
    pos = 0
    for r in order:
        if runs and runs[-1][0] == r["scale"]:
            s, a, _ = runs[-1]
            runs[-1] = (s, a, pos + 1)
        else:
            runs.append((r["scale"], pos, pos + 1))
        pos += 1

    fig = plt.figure(figsize=(30, 18))
    gs = fig.add_gridspec(3, 1, height_ratios=[0.55, 18, 18],
                          hspace=0.10,
                          left=0.095, right=0.92, top=0.92, bottom=0.16)
    ax_strip = fig.add_subplot(gs[0, 0])
    ax_top = fig.add_subplot(gs[1, 0])
    ax_bottom = fig.add_subplot(gs[2, 0], sharex=ax_top)
    axes = [ax_top, ax_bottom]
    for ax in (ax_strip, *axes):
        ax.set_xlim(-0.5, n - 0.5)
        for spine in ax.spines.values():
            spine.set_visible(False)
    ax_strip.set_ylim(0, 1)
    ax_strip.set_xticks([])
    ax_strip.set_yticks([])
    for scale, a, b in runs:
        ax_strip.add_patch(Rectangle((a - 0.5, 0), b - a, 1,
                                     facecolor=SCALE_COLORS.get(scale, FALLBACK_GRAY),
                                     edgecolor="none"))
        if b - a >= 3:
            ax_strip.text((a + b - 1) / 2, 0.5, scale, va="center", ha="center",
                          fontsize=11, fontweight=700, color="white")

    titles = {"hc": "HC-only (n\u226467; age/sex-adjusted)",
              "patient": "Patient-only (n\u226426; age/sex-adjusted)"}
    for ax, pop in zip(axes, panels):
        ax.set_ylim(nrows - 0.5, -0.5)
        ax.set_title(titles[pop], fontsize=11, fontweight=700, loc="left",
                     pad=4)
        mat, pmat = mats[pop], pmats[pop]
        for k, (scale, a, b) in enumerate(runs):
            if k % 2 == 1:
                ax.axvspan(a - 0.5, b - 0.5, color="#f8fafc", zorder=0)
        for i in range(nrows):
            for j in range(n):
                rho = mat[i, j]
                is_sig = bool(pmat[i, j] < 0.05)
                ax.add_patch(Rectangle((j - 0.5, i - 0.5), 1, 1,
                                       facecolor=cell_rgb(rho, is_sig),
                                       edgecolor="white", linewidth=0.8, zorder=1))
                if is_sig and not np.isnan(rho):
                    tiny = pmat[i, j] < 0.01
                    ax.text(j, i, "**" if tiny else "*", ha="center", va="center",
                            fontsize=10 if tiny else 9, fontweight=700,
                            color="white" if abs(rho) > 0.30 else "#101828", zorder=2)
        for k in range(1, len(PRIMARY_METRICS)):
            y = 6 * k - 0.5
            ax.axhline(y, color="white", linewidth=2.0, zorder=3)
            ax.axhline(y, color="#475467", linewidth=0.9, zorder=4)
        for k in range(len(PRIMARY_METRICS)):
            y = 6 * k + 2.5
            ax.axhline(y, color="white", linewidth=1.2, zorder=3)
            ax.axhline(y, color="#98a2b3", linewidth=0.6,
                       linestyle=(0, (4, 3)), zorder=4)
        for k, (scale, a, b) in enumerate(runs):
            if k:
                ax.axvline(a - 0.5, color="white", linewidth=1.8, zorder=3)
                ax.axvline(a - 0.5, color="#98a2b3", linewidth=0.6,
                           linestyle=(0, (4, 3)), zorder=4)
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_edgecolor("#344054")
            spine.set_linewidth(0.9)

    row_labels = [f"{METRIC_SHORT[m]}  {h}\u00b7{SYSTEM_LABELS[s]}"
                  for m, h, s in row_keys]
    for ax in axes:
        ax.set_yticks(range(nrows), labels=row_labels, fontsize=11)
        ax.tick_params(axis="y", length=0, pad=14)
    col_labels = [f"{r['field']} \u00b7 {_display(r['field'], r['label'])}"
                  for r in order]
    # Shared x axis: labels only on the bottom (patient) panel.
    ax_top.set_xticks(range(n))
    ax_top.tick_params(axis="x", which="both", bottom=False, top=False,
                       labelbottom=False)
    plt.setp(ax_top.get_xticklabels(), visible=False)
    ax_bottom.set_xticks(range(n), labels=col_labels, fontsize=9.5,
                         rotation=45, ha="right", rotation_mode="anchor")
    ax_bottom.tick_params(axis="x", length=0, pad=4)
    for lbl, r in zip(ax_bottom.get_xticklabels(), order):
        if r["level"] in ("total", "subscale"):
            lbl.set_fontweight(800)
    ax_strip.tick_params(axis="x", which="both", bottom=False, top=False,
                         labelbottom=False, labeltop=False)
    plt.setp(ax_strip.get_xticklabels(), visible=False)

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cax = fig.add_axes([0.925, 0.22, 0.012, 0.55])
    cbar = fig.colorbar(sm, cax=cax)
    cbar.set_label("Within-group partial rank \u03c1", fontsize=9)
    cbar.ax.tick_params(labelsize=8)

    n_tot = sum(1 for r in order if r["level"] in ("total", "subscale"))
    fig.suptitle(
        "Hippocampal functional organization \u00b7 clinical associations "
        f"by group (standard / HC-only vs patient-only; {n} hit fields, "
        f"{n_tot} totals in bold)",
        fontsize=13, fontweight=700, x=0.095, ha="left")
    fig.text(0.095, 0.030,
             "Columns = union of HC-only / patient-only hits "
             "(>=1 nominal within-group p<0.05; age/sex-adjusted).  "
             "Top = HC-only, bottom = patient-only.  Bold = total/subscale.  "
             "* raw within-group p < 0.05, ** p < 0.01 \u2014 no FDR.  "
             "Pale = same-field context (n.s.); gray = not estimable.",
             fontsize=9, color="#475467", va="bottom", ha="left")
    fig.text(0.095, 0.008,
             "Union is HC-heavy by design (HC n\u226467 vs patient n\u226426 "
             "power difference, not bias).  Suicide/self-harm columns have no "
             "HC data (HC n=0): top-panel cells render gray; patient panel "
             "(n\u226424) is valid.  MADRS totals quarantined.  "
             "N: HC\u226467, patient\u226426 (NIH 90, PCL 48).",
             fontsize=9, color="#475467", va="bottom", ha="left")
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()
    stats, screen, valuable = load()
    order = grouped_fields(stats)
    if not order:
        raise ValueError("no nominal primary hits; nothing to plot")
    out_csv = RESULTS / f"{PREFIX}.csv"
    out_png = RESULTS / f"{PREFIX}.png"
    for p in (out_csv, out_png):
        if p.exists() and not args.overwrite:
            raise FileExistsError(f"exists, pass --overwrite to replace: {p}")
    keep = [r["field"] for r in order]
    sub = stats.loc[(stats["branch"] == "standard")
                    & (stats["population"].isin(["hc", "patient"]))
                    & (stats["metric"].isin(PRIMARY_METRICS))
                    & (stats["field"].astype(str).isin(keep))].copy()
    cols = ["field", "scale", "label", "population", "metric", "hemi",
            "system", "n", "rho", "ci_low", "ci_high", "p", "status"]
    sub[[c for c in cols if c in sub.columns]].to_csv(out_csv, index=False)
    heatmap_png(stats, order, out_png)
    print(f"grouped: {len(order)} within-group hit fields (union).")


if __name__ == "__main__":
    main()
