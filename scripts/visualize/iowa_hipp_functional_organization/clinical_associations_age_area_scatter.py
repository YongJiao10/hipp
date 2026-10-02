"""Age vs area-percent scatter panels (standalone figure, no heatmap changes).

Rows: HC (67) / patient (26) / comorbid (19). Columns: L-A, L-B, L-P,
R-A, R-B, R-P (standard branch, area_percent). Each panel: raw scatter
of age (x) vs area % (y) with OLS line on raw units for display only;
title carries the sex-adjusted partial rank rho, Freedman-Lane p and n
(same convention as the clinical pipeline). Rows: HC (67) / patient (26) / pure MDD (7) / comorbid (19).
Modes: default sex-adjusted partial rank rho; --unadjusted gives plain
Spearman (no covariates, permutation by shuffling age) for comparison.

Reads only manifest-declared results/*.csv + subject_metrics.csv.
Writes clinical_associations_age_area_scatter.png.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
OUT_NAME = "clinical_associations_age_area_scatter.png"
TOL = 1e-12
PERMUTATIONS = 10000
SEED = 20260917

GROUPS = [("hc", "HC (n=67)", "#6090c0"),
          ("patient", "Patient (n=26)", "#b42318"),
          ("pure", "Pure MDD (n=7)", "#e97316"),
          ("comorbid", "Comorbid MDD (n=19)", "#067647")]
CELLS = [("L", "A"), ("L", "B"), ("L", "P"),
         ("R", "A"), ("R", "B"), ("R", "P")]
CELL_LABELS = {"A": "Ant", "B": "Mid", "P": "Post"}


def _rank(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranked = np.empty(values.size, dtype=float)
    sv = values[order]
    starts = np.r_[0, np.flatnonzero(sv[1:] != sv[:-1]) + 1]
    stops = np.r_[starts[1:], values.size]
    for start, stop in zip(starts, stops):
        ranked[order[start:stop]] = (start + 1 + stop) / 2.0
    return ranked


def _resid(values: np.ndarray, design: np.ndarray):
    coef, _, _, _ = np.linalg.lstsq(design, values, rcond=None)
    fitted = design @ coef
    return values - fitted, fitted


def _corr(left: np.ndarray, right: np.ndarray) -> float:
    left = left - left.mean()
    right = right - right.mean()
    denom = np.linalg.norm(left) * np.linalg.norm(right)
    scale = max(np.linalg.norm(left), np.linalg.norm(right), 1.0)
    if denom <= TOL * scale:
        return float("nan")
    return float(np.dot(left, right) / denom)


def age_partial_rho(area: np.ndarray, age: np.ndarray, sex: np.ndarray,
                    seed: int, unadjusted: bool = False) -> tuple[float, float, int]:
    """Rank rho between area% and age + permutation p.

    Default: sex-adjusted partial rank rho (residualize ranks on
    [1, sex], Freedman-Lane). Unadjusted: plain Spearman; null by
    shuffling age labels (equivalent to shuffling ranks).
    """
    n = area.size
    yr, ar = _rank(area), _rank(age)
    if unadjusted:
        rho = _corr(yr, ar)
        if not np.isfinite(rho):
            return rho, float("nan"), n
        rng = np.random.default_rng(seed)
        extreme = 0
        for _ in range(PERMUTATIONS):
            rho_star = _corr(yr, rng.permutation(ar))
            if np.isfinite(rho_star) and abs(rho_star) >= abs(rho) - TOL:
                extreme += 1
        return rho, float((1 + extreme) / (PERMUTATIONS + 1)), n
    design = np.column_stack([np.ones(n), sex])
    if np.linalg.matrix_rank(design) < design.shape[1]:
        return float("nan"), float("nan"), n
    resid_y, _ = _resid(yr, design)
    resid_a, _ = _resid(ar, design)
    rho = _corr(resid_y, resid_a)
    if not np.isfinite(rho):
        return rho, float("nan"), n
    coef, _, _, _ = np.linalg.lstsq(design, ar, rcond=None)
    fitted = design @ coef
    resid = ar - fitted
    rng = np.random.default_rng(seed)
    extreme = 0
    for _ in range(PERMUTATIONS):
        r_star, _ = _resid(fitted + rng.permutation(resid), design)
        rho_star = _corr(resid_y, r_star)
        if np.isfinite(rho_star) and abs(rho_star) >= abs(rho) - TOL:
            extreme += 1
    return rho, float((1 + extreme) / (PERMUTATIONS + 1)), n


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--unadjusted", action="store_true",
                    help="plain Spearman, no covariates (separate output)")
    args = ap.parse_args()
    out = RESULTS / (OUT_NAME.replace(".png", "_unadjusted.png")
                     if args.unadjusted else OUT_NAME)
    if out.exists() and not args.overwrite:
        raise FileExistsError(f"exists, pass --overwrite to replace: {out}")

    import sys
    sys.path.insert(0, str(HERE))
    from clinical_associations_data import deduplicate_metrics

    subjects = pd.read_csv(RESULTS / "clinical_associations_subjects.csv")
    imaging = deduplicate_metrics(pd.read_csv(RESULTS / "subject_metrics.csv"))
    img = imaging.loc[(imaging["branch"] == "standard")
                      & (imaging["metric"] == "area_percent")]

    group_frames = {
        "hc": subjects.loc[subjects["group"] == 0],
        "patient": subjects.loc[subjects["group"] == 1],
        "pure": subjects.loc[subjects["subgroup"] == "pure_mdd"],
        "comorbid": subjects.loc[subjects["subgroup"] == "comorbid_mdd"],
    }

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
        "axes.linewidth": 0.8, "axes.edgecolor": "#344054",
    })
    fig, axes = plt.subplots(len(GROUPS), len(CELLS), figsize=(24, 15),
                             sharex=True, sharey=False,
                             gridspec_kw={"hspace": 0.45, "wspace": 0.25},
                             squeeze=False)
    seed = SEED
    for i, (gkey, gtitle, color) in enumerate(GROUPS):
        gdf = group_frames[gkey]
        for j, (hemi, sys_) in enumerate(CELLS):
            ax = axes[i][j]
            v = img.loc[(img["hemi"] == hemi) & (img["system"] == sys_),
                        ["subject", "value"]]
            d = gdf.merge(v, on="subject", how="left").dropna(
                subset=["value", "age", "sex"]).copy()
            area = d["value"].to_numpy(float)
            age = d["age"].to_numpy(float)
            sex = d["sex"].to_numpy(float)
            seed += 1
            rho, p, n = age_partial_rho(area, age, sex, seed,
                                       unadjusted=args.unadjusted)
            ax.scatter(age, area, s=28, alpha=0.8, color=color,
                       edgecolors="white", linewidths=0.5, zorder=2)
            if np.ptp(age) > TOL and len(d) >= 3:
                xs = np.linspace(age.min(), age.max(), 50)
                b, a = np.polyfit(age, area, 1)
                ax.plot(xs, a + b * xs, color="#101828", linewidth=1.2,
                        zorder=3)
            sig = bool(np.isfinite(p) and p < 0.05)
            ax.set_title(f"{hemi}\u00b7{CELL_LABELS[sys_]}\n"
                         f"\u03c1 = {rho:.2f}  (p = {p:.3g})",
                         fontsize=11, fontweight=700,
                         color="#b42318" if sig else "#101828")
            if j == 0:
                ax.set_ylabel(f"{gtitle}\narea %", fontsize=12,
                              fontweight=700)
            if i == len(GROUPS) - 1:
                ax.set_xlabel("age (years)", fontsize=11)
    stat_desc = ("\u03c1 = Spearman rank correlation (no covariates); "
                 "p by 10k age-label permutations"
                 if args.unadjusted else
                 "\u03c1 = sex-adjusted partial rank correlation, "
                 "10k Freedman-Lane permutations")
    fig.suptitle("Area % vs age by group (standard branch; " + stat_desc +
                 "; red title: p<0.05, no FDR)",
                 fontsize=13, fontweight=700)
    fig.text(0.5, 0.01,
             "Display line = raw-unit OLS (visual guide only); statistics are rank-based.  "
             "Pure MDD has n=7: trend shown for completeness, interpret cautiously.  "
             "Area % shares the hemispheric denominator (compositional).",
             fontsize=10, color="#475467", ha="center", va="bottom")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
