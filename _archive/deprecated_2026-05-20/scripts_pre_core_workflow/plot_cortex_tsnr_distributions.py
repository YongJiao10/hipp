#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
from scipy.stats import gaussian_kde


HCP_MODE = 10000.0
TSNR_THRESHOLD = 25.0
COLORS = {
    "100610": "#4C72B0",
    "102311": "#DD8452",
    "102816": "#55A868",
}


def extract_structure_data(
    dt_axis: nib.cifti2.cifti2_axes.BrainModelAxis,
    dt_data_t: np.ndarray,
    structure_name: str,
) -> np.ndarray:
    for name, slc, _subaxis in dt_axis.iter_structures():
        if name == structure_name or name.endswith(structure_name):
            return dt_data_t[slc, :]
    raise RuntimeError(f"Could not find structure {structure_name} in dtseries")


def compute_tsnr(metric: np.ndarray) -> np.ndarray:
    sd = np.nanstd(metric.astype(np.float32, copy=False), axis=1, ddof=1)
    return np.where(sd > 0, HCP_MODE / sd, np.nan).astype(np.float32)


def load_cortex_tsnr(dtseries_path: Path) -> np.ndarray:
    img = nib.load(str(dtseries_path))
    dt_data = np.asarray(img.dataobj, dtype=np.float32)
    dt_axis = img.header.get_axis(1)
    dt_data_t = dt_data.T
    left = compute_tsnr(extract_structure_data(dt_axis, dt_data_t, "CORTEX_LEFT"))
    right = compute_tsnr(extract_structure_data(dt_axis, dt_data_t, "CORTEX_RIGHT"))
    return np.concatenate([left[np.isfinite(left)], right[np.isfinite(right)]])


def main() -> int:
    parser = argparse.ArgumentParser(description="Plot cortex tSNR distributions (combined hemispheres) for formal network-first outputs")
    parser.add_argument("--input-root", default="/Users/jy/Documents/HippoMaps-network-first/data/hippunfold_input")
    parser.add_argument("--subjects", nargs="+", default=["100610", "102311", "102816"])
    parser.add_argument("--out", default="/Users/jy/Documents/HippoMaps-network-first/outputs_migration/cortex_tsnr_distributions.png")
    args = parser.parse_args()

    input_root = Path(args.input_root).resolve()
    out_path = Path(args.out).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    subject_data: dict[str, np.ndarray] = {}
    for subject in args.subjects:
        dtseries_path = input_root / f"sub-{subject}" / "func" / f"sub-{subject}_task-rest_run-concat.dtseries.nii"
        subject_data[str(subject)] = load_cortex_tsnr(dtseries_path)

    fig = plt.figure(figsize=(14, 13))
    fig.patch.set_facecolor("#F8F8F8")
    gs = gridspec.GridSpec(
        len(args.subjects), 1,
        figure=fig,
        hspace=0.45,
        left=0.08, right=0.96,
        top=0.93, bottom=0.06,
    )
    rng = np.random.default_rng(42)

    for row, subject in enumerate(args.subjects):
        ax = fig.add_subplot(gs[row, 0])
        vals = subject_data[str(subject)]
        color = COLORS.get(str(subject), "#4C72B0")
        jitter_y = rng.uniform(0.42, 0.68, size=len(vals))
        ax.scatter(vals, jitter_y, s=7, alpha=0.28, color=color, linewidths=0, zorder=2)

        x_min = float(vals.min())
        x_max = float(np.percentile(vals, 99.5))
        xs = np.linspace(x_min, x_max, 400)
        kde = gaussian_kde(vals, bw_method="scott")
        ys = kde(xs)
        ys_norm = ys / max(float(ys.max()), 1e-8)
        kde_y = 0.70 + ys_norm * 0.28
        ax.fill_between(xs, 0.70, kde_y, alpha=0.30, color=color, zorder=3)
        ax.plot(xs, kde_y, color=color, lw=1.8, zorder=4)

        ax.boxplot(
            vals,
            vert=False,
            positions=[0.20],
            widths=[0.18],
            patch_artist=True,
            notch=False,
            zorder=5,
            manage_ticks=False,
            flierprops=dict(marker=".", markersize=3, alpha=0.35, markerfacecolor=color, markeredgecolor=color),
            medianprops=dict(color="white", lw=2.0),
            boxprops=dict(facecolor=color, alpha=0.75, edgecolor=color),
            whiskerprops=dict(color=color, lw=1.4),
            capprops=dict(color=color, lw=1.4),
        )

        ax.axvline(TSNR_THRESHOLD, color="#666666", lw=1.2, ls="--", zorder=1)
        ax.text(TSNR_THRESHOLD, 1.01, "tSNR=25", ha="left", va="bottom", fontsize=8, color="#555555")

        ax.set_ylim(0.0, 1.02)
        ax.set_xlim(x_min - (x_max - x_min) * 0.04, x_max + (x_max - x_min) * 0.04)
        ax.set_yticks([])
        ax.set_xlabel("tSNR", fontsize=10)
        ax.tick_params(axis="x", labelsize=9)
        for y in [0.38, 0.68]:
            ax.axhline(y, color="grey", lw=0.5, ls="--", alpha=0.4, zorder=1)

        med = float(np.nanmedian(vals))
        iqr = float(np.percentile(vals, 75) - np.percentile(vals, 25))
        n_masked = int(np.count_nonzero(vals < TSNR_THRESHOLD))
        ax.text(
            0.985, 0.97,
            f"median={med:.1f}\nIQR={iqr:.1f}\nn={len(vals)}\n<25={n_masked}",
            transform=ax.transAxes,
            ha="right", va="top", fontsize=8,
            color="#333333",
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="none", alpha=0.7),
        )
        ax.set_title(f"sub-{subject}  ·  Cortex (L+R combined)", fontsize=11, fontweight="bold", pad=4)
        ax.spines[["top", "right", "left"]].set_visible(False)
        ax.spines["bottom"].set_color("#AAAAAA")

    fig.suptitle(
        "Cortical tSNR — HCP 7T dtseries (L+R combined by subject)\n"
        r"tSNR = 10000 / $\sigma_t$  [formal threshold = 25]",
        fontsize=13, fontweight="bold", y=0.98,
    )
    plt.savefig(out_path, dpi=180, bbox_inches="tight", facecolor=fig.get_facecolor())
    print(str(out_path))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
