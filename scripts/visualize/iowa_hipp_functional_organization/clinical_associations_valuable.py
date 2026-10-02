"""Valuable-only static clinical-association plots (no FDR, primary family only).

口径 (user-confirmed):
- primary only: branch=standard, population=all_adjusted,
  metric in {area_percent, segregation, margin}; raw two-sided permutation p.
- left block: EVERY declared total/subscale field (complete, nulls included).
- right block: items with >= 1 primary cell with nominal p < 0.05.
- drop null items and whole scales with zero hits (currently GAD-7).
- no FDR filtering; tables/CSVs state raw p explicitly.

Figure style: landscape; columns are fields with 45° labels; rows are the
18 primary cells (Area % / Segregation / Margin × L/R × Ant/Mid/Post).
separates TOTALS & SUBSCALES from ITEMS at a glance — no label reading
needed. Significant cells carry an asterisk; non-significant cells are
desaturated context.

Reads only manifest-declared results/*.csv; never selects substitutes.
Writes clinical_associations_valuable.{csv,png} and _dropped.csv.
"""
from __future__ import annotations

import argparse
import textwrap
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
PREFIX = "clinical_associations_valuable"
PRIMARY_METRICS = ["area_percent", "segregation", "margin"]
METRIC_SHORT = {"area_percent": "Area", "segregation": "Segr", "margin": "Marg"}
SYSTEMS = ["A", "B", "P"]  # Ant / Mid / Post
HEMIS = ["L", "R"]
SYSTEM_LABELS = {"A": "Ant", "B": "Mid", "P": "Post"}

# Clinical hierarchy: depressive core first, then anxiety/affect,
# cross-cutting, trauma, emotion, function.
SCALE_ORDER = ["PHQ", "BDI-II", "MADRS-SR", "STAI", "PANAS",
               "DSM-5 Cross-cutting", "PCL-5", "NIH Emotion", "WHODAS"]
SCALE_COLORS = {
    "PHQ": "#b42318",
    "BDI-II": "#e97316",
    "MADRS-SR": "#c77700",
    "STAI": "#175cd3",
    "PANAS": "#0e7490",
    "DSM-5 Cross-cutting": "#6941c6",
    "PCL-5": "#a1225c",
    "NIH Emotion": "#067647",
    "WHODAS": "#475467",
}
LEVEL_ORDER = {"total": 0, "subscale": 1, "item": 2}
FALLBACK_GRAY = "#98a2b3"

# Concise display names for fields whose dictionary labels are sentences.
# Short items (STAI/PANAS/BDI/NIH) use their labels verbatim.
FIELD_SHORT = {
    "dsm5crossad_4": "sleep less, still energetic",
    "dsm5crossad_5": "many projects / risk-taking",
    "dsm5crossad_7": "panic / fright",
    "dsm5crossad_11": "self-harm thoughts",
    "dsm5crossad_13": "thought reading experience",
    "dsm5crossad_16": "intrusive thoughts/urges",
    "dsm5crossad_18": "detachment / dissociation",
    "dsm5crossad_19": "identity uncertainty",
    "dsm5crossad_21": "heavy drinking day",
    "dsm5crossad_23": "unsupervised meds/drugs",
    "madrs_1": "Mood (sad/gloomy)",
    "madrs_6": "Initiative / inertia",
    "madrs_8": "Pessimism / guilt",
    "madrs_9": "Zest for life",
    "pcl_3": "reliving trauma",
    "pcl_4": "upset at reminders",
    "pcl_7": "avoiding reminders",
    "pcl_8": "memory gaps (trauma)",
    "pcl_9": "strong negative beliefs",
    "pcl_16": "risk-taking",
    "pcl_17": "hypervigilance",
    "pcl_18": "startle",
    "phq_2": "feeling down / hopeless",
    "phq_3": "sleep problems",
    "phq_8": "psychomotor slowing/agitation",
    "phq_suicide": "suicidal thoughts",
    "phq_total_score_w_suicide": "PHQ-9 sum + suicide",
    "stai_t9": "worry over trifles",
    "stai_t18": "keen disappointment",
    "stai_t20": "tension/turmoil",
    "stai_t8": "difficulties pile up",
    "whodas_1": "standing long",
    "whodas_3": "learning new task",
    "whodas_4": "joining community activities",
    "whodas_5": "emotionally affected",
    "whodas_9": "getting dressed",
    "whodas_total": "WHODAS total",
    "phq_total_score": "PHQ-8 sum",
    "gad_total_score": "GAD-7 total",
    "bdi_total": "BDI-20 sum",
    "bdi_total_w_suicide": "BDI-21 sum + suicide",
    "madrs_total": "MADRS-8 recorded",
    "madrs_total_w_suicide": "MADRS-9 recorded",
    "stai_s_total": "State anxiety",
    "stai_t_total": "Trait anxiety",
    "pcl_total": "PCL total",
    "panas_positive": "Positive affect",
    "panas_negative": "Negative affect",
    "nih_angerhostile_tscore": "Anger/Hostility",
    "nih_fearanxiety_tscore": "Fear/Anxiety",
    "nih_stress_tscore": "Stress",
    "nih_sad_tscore": "Sadness",
    "nih_genlifesat_tscore": "Life satisfaction",
    "nih_meaning_tscore": "Meaning & purpose",
    "nih_posaffect_tscore": "Positive affect",
    "nih_emosupp_tscore": "Emotional support",
    "nih_friendship_tscore": "Friendship",
    "nih_lonely_tscore": "Loneliness",
    "nih_aggression_tscore": "Aggression",
    "nih_selfefficacy_tscore": "Self-efficacy",
    "nih_negaffect_tscore": "Negative affect",
    "nih_psychwell_tscore": "Well-being",
    "nih_socialsat_tscore": "Social satisfaction",
    "nih_hostility_tscore": "Hostility",
    "nih_anger_tscore": "Anger",
    "nih_fearsa_tscore": "Fear / somatic arousal",
    "nih_rejection_tscore": "Rejection",
    "nih_instrusupp_tscore": "Instrumental support",
}
# Abbreviated scale names for the narrow group-label column.
SCALE_SHORT = {"DSM-5 Cross-cutting": "DSM-5", "NIH Emotion": "NIH"}


def load() -> tuple[pd.DataFrame, pd.DataFrame]:
    stats_path = RESULTS / "clinical_associations_statistics.csv"
    scales_path = RESULTS / "clinical_associations_scales.csv"
    if not stats_path.is_file():
        raise FileNotFoundError(f"missing required input: {stats_path}")
    if not scales_path.is_file():
        raise FileNotFoundError(f"missing required input: {scales_path}")
    stats = pd.read_csv(stats_path)
    scales = pd.read_csv(scales_path)
    required = {"branch", "population", "metric", "hemi", "system", "field",
                "scale", "label", "n", "rho", "ci_low", "ci_high", "p"}
    missing = required - set(stats.columns)
    if missing:
        raise ValueError(f"statistics.csv missing columns: {sorted(missing)}")
    return stats, scales


def _primary(stats: pd.DataFrame) -> pd.DataFrame:
    prim = stats.loc[
        (stats["branch"] == "standard")
        & (stats["population"] == "all_adjusted")
        & (stats["metric"].isin(PRIMARY_METRICS))
    ].copy()
    if prim.empty:
        raise ValueError("no primary standard/all_adjusted rows found")
    return prim


def valuable(stats: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    prim = _primary(stats)
    sig = prim.loc[prim["p"] < 0.05].copy().sort_values("p").reset_index(drop=True)
    if sig.empty:
        raise ValueError("no nominal p<0.05 primary cells; nothing valuable to plot")
    kept_fields = set(sig["field"].astype(str))
    kept = prim.loc[prim["field"].astype(str).isin(kept_fields)].copy()
    dropped = prim.loc[~prim["field"].astype(str).isin(kept_fields),
                       ["field", "scale", "label"]].drop_duplicates().copy()
    return kept, sig, dropped


def ordered_fields(sig: pd.DataFrame) -> list[dict]:
    """Hierarchical row order: scale block → level → min raw p."""
    rows = []
    for field, group in sig.groupby("field"):
        first = group.iloc[0]
        rows.append({
            "field": str(field),
            "scale": str(first["scale"]),
            "level": str(first.get("level", "")),
            "label": str(first["label"]),
            "min_p": float(group["p"].min()),
            "min_n": int(group["n"].min()),
        })

    def key(row: dict):
        try:
            scale_idx = SCALE_ORDER.index(row["scale"])
        except ValueError:
            scale_idx = len(SCALE_ORDER)
        return (scale_idx, LEVEL_ORDER.get(row["level"], 9), row["min_p"])

    return sorted(rows, key=key)


def _short(label: str, width: int = 44) -> str:
    return textwrap.shorten(str(label), width=width, placeholder="…")


def _display(field: str, label: str) -> str:
    return FIELD_SHORT.get(field, _short(label, 40))


def _style():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
        "axes.linewidth": 0.8,
        "axes.edgecolor": "#344054",
        "xtick.color": "#344054",
        "ytick.color": "#101828",
    })
    return plt


def heatmap_png(stats: pd.DataFrame, sig: pd.DataFrame, path: Path) -> None:
    """Landscape grouped heatmap: fields on x (rotated), anatomy on y.

    Columns form two unmissable super-blocks: every declared total/subscale
    field (complete, including nulls) on the left, hit items on the right.
    Rows are the 18 primary cells (Area % / Segregation / Margin × L/R × Ant/Mid/Post).
    Significant cells carry an asterisk; non-significant cells are
    desaturated context.
    """
    plt = _style()
    import matplotlib.colors as mcolors
    from matplotlib.patches import Rectangle

    prim = _primary(stats)
    order = ordered_fields(sig)
    n = len(order)
    n_tot = sum(1 for r in order if r["level"] in ("total", "subscale"))
    fields = [r["field"] for r in order]
    lookup = {(str(r["field"]), r["metric"], r["hemi"], r["system"]): r
              for r in prim.to_dict("records")}
    row_keys = [(m, h, s) for m in PRIMARY_METRICS
                for h in HEMIS for s in SYSTEMS]
    nrows = len(row_keys)
    mat = np.full((nrows, n), np.nan)
    pmat = np.full((nrows, n), np.nan)
    for i, (metric, hemi, system) in enumerate(row_keys):
        for j, field in enumerate(fields):
            rec = lookup.get((field, metric, hemi, system))
            if rec is not None:
                mat[i, j] = pd.to_numeric(rec.get("rho"), errors="coerce")
                pmat[i, j] = pd.to_numeric(rec.get("p"), errors="coerce")

    vmin, vmax = -0.6, 0.6
    cmap = plt.get_cmap("RdBu_r")
    norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
    faint = np.array(mcolors.to_rgb("#eef2f7"))
    missing_rgb = np.array(mcolors.to_rgb("#e5e7eb"))

    def cell_rgb(rho: float, sig_cell: bool):
        if np.isnan(rho):
            return missing_rgb
        rgb = np.array(cmap(norm(rho))[:3])
        return rgb if sig_cell else faint + (rgb - faint) * 0.35

    # Scale blocks along the x axis.
    runs: list[tuple[str, int, int]] = []
    pos = 0
    for r in order:
        if runs and runs[-1][0] == r["scale"]:
            s, a, _ = runs[-1]
            runs[-1] = (s, a, pos + 1)
        else:
            runs.append((r["scale"], pos, pos + 1))
        pos += 1

    fig = plt.figure(figsize=(30, 10.5))
    gs = fig.add_gridspec(2, 1, height_ratios=[0.55, 18],
                          hspace=0.04, left=0.095, right=0.93,
                          top=0.90, bottom=0.36)
    ax_strip = fig.add_subplot(gs[0, 0])
    ax_main = fig.add_subplot(gs[1, 0], sharex=ax_strip)
    for ax in (ax_strip, ax_main):
        ax.set_xlim(-0.5, n - 0.5)
        for spine in ax.spines.values():
            spine.set_visible(False)

    ax_strip.set_ylim(0, 1)
    ax_strip.set_xticks([])
    ax_strip.set_yticks([])
    for k, (scale, a, b) in enumerate(runs):
        color = SCALE_COLORS.get(scale, FALLBACK_GRAY)
        ax_strip.add_patch(Rectangle((a - 0.5, 0), b - a, 1,
                                     facecolor=color, edgecolor="none"))
        if b - a < 3:
            continue  # segment too narrow for a label; super-block says it all
        ax_strip.text((a + b - 1) / 2, 0.5, scale, va="center", ha="center",
                      fontsize=11, fontweight=700, color="white")

    ax_main.set_ylim(nrows - 0.5, -0.5)
    for k, (scale, a, b) in enumerate(runs):
        if k % 2 == 1:
            ax_main.axvspan(a - 0.5, b - 0.5, color="#f8fafc", zorder=0)
    for i in range(nrows):
        for j in range(n):
            rho = mat[i, j]
            is_sig = bool(pmat[i, j] < 0.05)
            ax_main.add_patch(Rectangle((j - 0.5, i - 0.5), 1, 1,
                                        facecolor=cell_rgb(rho, is_sig),
                                        edgecolor="white", linewidth=0.8,
                                        zorder=1))
            if is_sig and not np.isnan(rho):
                tiny = pmat[i, j] < 0.01
                ax_main.text(j, i, "**" if tiny else "*", ha="center",
                             va="center", fontsize=10 if tiny else 9,
                             fontweight=700,
                             color="white" if abs(rho) > 0.30 else "#101828",
                             zorder=2)
    # Block separators between metrics, and L | R within each block.
    for k in range(1, len(PRIMARY_METRICS)):
        y = 6 * k - 0.5
        ax_main.axhline(y, color="white", linewidth=2.0, zorder=3)
        ax_main.axhline(y, color="#475467", linewidth=0.9, zorder=4)
    for k in range(len(PRIMARY_METRICS)):
        y = 6 * k + 2.5
        ax_main.axhline(y, color="white", linewidth=1.2, zorder=3)
        ax_main.axhline(y, color="#98a2b3", linewidth=0.6,
                        linestyle=(0, (4, 3)), zorder=4)
    for k, (scale, a, b) in enumerate(runs):
        if k:
            ax_main.axvline(a - 0.5, color="white", linewidth=1.8, zorder=3)
            ax_main.axvline(a - 0.5, color="#98a2b3", linewidth=0.6,
                            linestyle=(0, (4, 3)), zorder=4)
    for spine in ax_main.spines.values():
        spine.set_visible(True)
        spine.set_edgecolor("#344054")
        spine.set_linewidth(0.9)

    col_labels = [f"{r['field']} · {_display(r['field'], r['label'])}"
                  for r in order]
    # Single-tier 45° labels: each label ends at its own tick, so the
    # display-name tail is always readable at the column. Bold totals.
    ax_main.set_xticks(range(n), labels=col_labels, fontsize=9.5,
                       rotation=45, ha="right", rotation_mode="anchor")
    ax_main.tick_params(axis="x", length=0, pad=4)
    for lbl, r in zip(ax_main.get_xticklabels(), order):
        if r["level"] in ("total", "subscale"):
            lbl.set_fontweight(800)
            lbl.set_color("#101828")
    row_labels = [f"{METRIC_SHORT[m]}  {h}·{SYSTEM_LABELS[s]}"
                  for m, h, s in row_keys]
    ax_main.set_yticks(range(nrows), labels=row_labels, fontsize=11)
    ax_main.tick_params(axis="y", length=0, pad=16)
    for lbl in ax_main.get_yticklabels():
        lbl.set_bbox(dict(facecolor="white", edgecolor="none", pad=1.5))
    # The strip shares the x axis: setting the main ticks repopulates the
    # strip ticks as well, so hide the strip's own copies explicitly here.
    ax_strip.tick_params(axis="x", which="both", bottom=False, top=False,
                         labelbottom=False, labeltop=False)
    plt.setp(ax_strip.get_xticklabels(), visible=False)

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    # Explicit small colorbar confined to the rows zone: never touches
    # the label zone below or the strip above, and its label can't clip.
    cax = fig.add_axes([0.935, 0.52, 0.013, 0.30])
    cbar = fig.colorbar(sm, cax=cax)
    cbar.set_label("Partial rank ρ", fontsize=9)
    cbar.ax.tick_params(labelsize=8)

    fig.suptitle(
        "Hippocampal functional organization · clinical associations "
        f"(primary: standard / all-adjusted for age, sex, HC-patient; {n} hit fields, "
        f"{n_tot} totals in bold)",
        fontsize=13, fontweight=700, x=0.10, ha="left")
    fig.text(0.10, 0.030,
             "All-adjusted = partial rank ρ adjusted for age, sex, HC/patient status.  "
             "Bold columns = total/subscale fields; regular = items.  "
             "* raw p < 0.05, ** raw p < 0.01 — no FDR.  "
             "Pale cells: same-field context (n.s.).",
             fontsize=9, color="#475467", va="bottom", ha="left")
    fig.text(0.10, 0.008,
             "Null fields omitted (see valuable_dropped.csv); GAD-7 has zero hits anywhere.  "
             "N = 93 (NIH = 90, PCL = 48, PHQ-suicide items = 24).",
             fontsize=9, color="#475467", va="bottom", ha="left")
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()
    stats, _ = load()
    kept, sig, dropped = valuable(stats)
    out_csv = RESULTS / f"{PREFIX}.csv"
    out_drop = RESULTS / f"{PREFIX}_dropped.csv"
    out_heat = RESULTS / f"{PREFIX}.png"
    for p in (out_csv, out_drop, out_heat):
        if p.exists() and not args.overwrite:
            raise FileExistsError(f"exists, pass --overwrite to replace: {p}")
    cols = ["field", "scale", "label", "metric", "hemi", "system", "n",
            "rho", "ci_low", "ci_high", "p", "status", "loo_min", "loo_max",
            "loo_sign_flip", "delta_r2"]
    sig_out = sig[[c for c in cols if c in sig.columns]]
    sig_out.to_csv(out_csv, index=False)
    dropped.to_csv(out_drop, index=False)
    heatmap_png(stats, sig, out_heat)
    n_hit = sig["field"].nunique()
    print(f"valuable: {len(sig)} nominal cells across {n_hit} hit fields; "
          f"dropped {len(dropped)} null fields.")


if __name__ == "__main__":
    main()
