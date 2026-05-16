#!/usr/bin/env python3
from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
AUDIT_TSV = (
    REPO_ROOT
    / "outputs_experiments"
    / "hippunfold_recommended_t1w_20260423"
    / "vertex_qc_t1only"
    / "qc"
    / "hcp7t_t1only_vertex_qc_166_audit.tsv"
)
OUT_PNG = REPO_ROOT / "present_fixedk" / "null_pct_distribution_166_to_151_explainer.png"


def as_float(v: str) -> float | None:
    t = (v or "").strip()
    return None if not t else float(t)


def as_bool(v: str) -> bool:
    return (v or "").strip().lower() == "true"


def main() -> int:
    rows = list(csv.DictReader(AUDIT_TSV.open("r", encoding="utf-8"), delimiter="\t"))
    if len(rows) != 166:
        raise RuntimeError(f"Expected 166 rows in audit TSV, got {len(rows)}")

    pass_points: list[tuple[float, float]] = []
    dice_fail_points: list[tuple[float, float]] = []
    tsnr_fail_points: list[tuple[float, float]] = []
    missing_tsnr_rows: list[dict[str, str]] = []

    for r in rows:
        l = as_float(r["hipp_tsnr_invalid_fraction_L"])
        rr = as_float(r["hipp_tsnr_invalid_fraction_R"])
        pass_manifest = as_bool(r["in_pass_151_manifest"])
        pass_dice = as_bool(r["pass_dice_derived"])
        tsnr_ok = (r["pass_tsnr_derived"] or "").strip().lower() == "true"
        tsnr_evidence = as_bool(r["tsnr_evidence_available"])

        if l is None or rr is None:
            missing_tsnr_rows.append(r)
            continue

        lp = 100.0 * l
        rp = 100.0 * rr
        if pass_manifest:
            pass_points.append((lp, rp))
        elif not pass_dice:
            dice_fail_points.append((lp, rp))
        elif not tsnr_ok:
            tsnr_fail_points.append((lp, rp))
        else:
            # Kept for robustness if future rows have unexpected flags.
            pass_points.append((lp, rp))

    pass_l = [x for x, _ in pass_points]
    pass_r = [y for _, y in pass_points]
    rng = np.random.default_rng(42)

    fig = plt.figure(figsize=(8.2, 7.2), dpi=190)
    ax = fig.add_subplot(111)

    x_pass_l = rng.normal(0.0, 0.03, size=len(pass_points))
    x_pass_r = rng.normal(1.0, 0.03, size=len(pass_points))
    for i, (l, r) in enumerate(pass_points):
        ax.plot([x_pass_l[i], x_pass_r[i]], [l, r], color="#9AA3AE", alpha=0.25, lw=0.7, zorder=1)
    ax.scatter(x_pass_l, pass_l, s=12, alpha=0.62, color="#4C72B0", label=f"Pass-151 (n={len(pass_points)})", zorder=4)
    ax.scatter(x_pass_r, pass_r, s=12, alpha=0.62, color="#DD8452", zorder=4)

    if tsnr_fail_points:
        x_tsnr_l = rng.normal(0.0, 0.03, size=len(tsnr_fail_points))
        x_tsnr_r = rng.normal(1.0, 0.03, size=len(tsnr_fail_points))
        for i, (l, r) in enumerate(tsnr_fail_points):
            ax.plot([x_tsnr_l[i], x_tsnr_r[i]], [l, r], color="#B22222", alpha=0.62, lw=1.0, zorder=2)
        ax.scatter(x_tsnr_l, [x for x, _ in tsnr_fail_points], s=36, marker="x", linewidths=1.2, color="#B22222", alpha=0.95, label=f"tSNR fail with values (n={len(tsnr_fail_points)})", zorder=6)
        ax.scatter(x_tsnr_r, [y for _, y in tsnr_fail_points], s=36, marker="x", linewidths=1.2, color="#B22222", alpha=0.95, zorder=6)

    if dice_fail_points:
        x_dice_l = rng.normal(0.0, 0.03, size=len(dice_fail_points))
        x_dice_r = rng.normal(1.0, 0.03, size=len(dice_fail_points))
        for i, (l, r) in enumerate(dice_fail_points):
            ax.plot([x_dice_l[i], x_dice_r[i]], [l, r], color="#6A3D9A", alpha=0.78, lw=1.25, zorder=3)
        ax.scatter(x_dice_l, [x for x, _ in dice_fail_points], s=48, marker="D", color="#6A3D9A", alpha=0.95, label=f"Dice fail with values (n={len(dice_fail_points)})", zorder=7)
        ax.scatter(x_dice_r, [y for _, y in dice_fail_points], s=48, marker="D", color="#6A3D9A", alpha=0.95, zorder=7)

    if missing_tsnr_rows:
        n = len(missing_tsnr_rows)
        base = np.linspace(20.15, 21.35, n)
        y_l = base + rng.normal(0.0, 0.05, size=n)
        y_r = base + rng.normal(0.0, 0.05, size=n)
        x_l = rng.normal(0.0, 0.03, size=n)
        x_r = rng.normal(1.0, 0.03, size=n)
        for i in range(n):
            ax.plot([x_l[i], x_r[i]], [y_l[i], y_r[i]], color="#B22222", alpha=0.55, lw=1.0, zorder=5)
        ax.scatter(x_l, y_l, s=32, marker="o", facecolors="none", edgecolors="#B22222", linewidths=1.0, alpha=0.95, label=f"Excluded without L/R tSNR summaries (n={n})", zorder=8)
        ax.scatter(x_r, y_r, s=32, marker="o", facecolors="none", edgecolors="#B22222", linewidths=1.0, alpha=0.95, zorder=8)

    bp = ax.boxplot([pass_l, pass_r], positions=[0, 1], widths=0.24, patch_artist=True, showfliers=False)
    bp["boxes"][0].set(facecolor="#4C72B0", alpha=0.24, edgecolor="#4C72B0")
    bp["boxes"][1].set(facecolor="#DD8452", alpha=0.24, edgecolor="#DD8452")
    for k in ("whiskers", "caps", "medians"):
        for item in bp[k]:
            item.set(color="#333333", linewidth=1.0 if k != "medians" else 1.6)

    ax.axhline(20.0, color="#B22222", lw=1.4, ls="--", label="QC threshold (20%)")
    ax.set_xlim(-0.35, 1.35)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Left hemi", "Right hemi"], fontsize=10)
    ax.set_ylabel("Invalid vertex percentage (tSNR < 25 or NaN)", fontsize=10)
    ax.set_title("All 166 subjects: L/R paired null% (vertex-level QC audit)", fontsize=12)
    ax.grid(axis="y", color="#dddddd", lw=0.6, alpha=0.8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(loc="best", fontsize=8.0, framealpha=0.92)

    fig.suptitle(
        "Hippocampal fMRI QC explanation (T1only)\nWhy the analysis cohort is 151 instead of 166",
        fontsize=13,
    )
    OUT_PNG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PNG, dpi=220, bbox_inches="tight")
    plt.close(fig)
    print(str(OUT_PNG))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
