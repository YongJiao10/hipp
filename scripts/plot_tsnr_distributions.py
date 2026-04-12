"""
Plot tSNR distributions for hippocampal 512-vertex surfaces.
6 subplots: 3 subjects × 2 hemispheres (L/R).
Each subplot: scatter + KDE + boxplot.
"""

import argparse
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.stats import gaussian_kde
from pathlib import Path

# ── Config ──────────────────────────────────────────────────────────────────
_REPO_ROOT = Path(__file__).resolve().parents[1]
_p = argparse.ArgumentParser()
_p.add_argument("--batch-dir", default=str(_REPO_ROOT / "outputs_migration" / "dense_corobl_batch"))
_p.add_argument("--out", default=str(_REPO_ROOT / "outputs_migration" / "tsnr_distributions.png"))
_args = _p.parse_args()
BATCH_DIR = Path(_args.batch_dir)
OUT_FIG   = Path(_args.out)

SUBJECTS = ["sub-100610", "sub-102311", "sub-102816"]
HEMIS    = ["L", "R"]

COLORS = {
    "sub-100610": "#4C72B0",
    "sub-102311": "#DD8452",
    "sub-102816": "#55A868",
}

# ── Helper: load bold timeseries (prefer .npy, fallback to .func.gii) ───────
def load_bold(subject: str, hemi: str) -> np.ndarray:
    """Return array of shape (512, T)."""
    surface_dir = BATCH_DIR / subject / "post_dense_corobl" / "surface"
    npy = surface_dir / f"{subject}_hemi-{hemi}_space-corobl_den-512_label-hipp_bold.npy"
    gii = surface_dir / f"{subject}_hemi-{hemi}_space-corobl_den-512_label-hipp_bold.func.gii"
    if npy.exists():
        return np.load(npy).astype(np.float64)          # (512, T)
    img = nib.load(str(gii))
    return np.stack([d.data for d in img.darrays], axis=1).astype(np.float64)  # (512, T)

# ── Compute tSNR ─────────────────────────────────────────────────────────────
# HCP 7T data is intensity-normalized to mode=10000 before hp2000 temporal
# filtering. The hp filter removes the DC component (mean → 0), but the std
# reflects noise relative to the original 10000 baseline.
# Therefore the correct tSNR proxy for HCP hp-filtered data is 10000 / std(t).
# Reference: HCP minimal preprocessing pipeline (Glasser et al. 2013);
#            standard practice in WashU / Petersen Lab FC analyses.
HCP_MODE = 10000.0

def compute_tsnr(bold: np.ndarray) -> np.ndarray:
    """tSNR = 10000 / std(t), per vertex.  Returns (512,).

    Valid for HCP data that has been intensity-normalised to mode=10000 prior
    to hp2000 temporal filtering (which removes the mean).  Using the known
    normalisation constant as the effective mean recovers a physically
    meaningful SNR measure without requiring the original un-filtered signal.
    """
    sd   = bold.std(axis=1, ddof=1)
    tsnr = np.where(sd > 0, HCP_MODE / sd, np.nan)
    return tsnr

# ── Collect data ─────────────────────────────────────────────────────────────
print("Loading data and computing tSNR …")
data = {}   # (subject, hemi) → tsnr array (512,)
for sub in SUBJECTS:
    for hemi in HEMIS:
        bold = load_bold(sub, hemi)
        data[(sub, hemi)] = compute_tsnr(bold)
        vals = data[(sub, hemi)]
        print(f"  {sub} {hemi}: mean={np.nanmean(vals):.2f}  median={np.nanmedian(vals):.2f}  "
              f"std={np.nanstd(vals):.2f}  n_nan={np.isnan(vals).sum()}")

# ── Plot ─────────────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(16, 11))
fig.patch.set_facecolor("#F8F8F8")

gs = gridspec.GridSpec(
    3, 2,
    figure=fig,
    hspace=0.52,
    wspace=0.32,
    left=0.08, right=0.96,
    top=0.92,  bottom=0.06,
)

HEMI_LABEL = {"L": "Left", "R": "Right"}

rng = np.random.default_rng(42)

for row, sub in enumerate(SUBJECTS):
    for col, hemi in enumerate(HEMIS):
        ax = fig.add_subplot(gs[row, col])

        vals = data[(sub, hemi)]
        vals_clean = vals[~np.isnan(vals)]
        color = COLORS[sub]

        # ── Scatter (jittered vertically around y=0.55) ───────────────────
        jitter_y = rng.uniform(0.42, 0.68, size=len(vals_clean))
        ax.scatter(
            vals_clean, jitter_y,
            s=8, alpha=0.35, color=color, linewidths=0,
            zorder=2, label="vertex tSNR",
        )

        # ── KDE ───────────────────────────────────────────────────────────
        kde = gaussian_kde(vals_clean, bw_method="scott")
        x_min = np.percentile(vals_clean, 0.5)
        x_max = np.percentile(vals_clean, 99.5)
        xs = np.linspace(x_min, x_max, 400)
        ys = kde(xs)
        # Normalise KDE to [0, 1] then scale to occupy upper band (0.7 – 1.0)
        ys_norm = ys / ys.max()
        kde_y = 0.70 + ys_norm * 0.28
        ax.fill_between(xs, 0.70, kde_y, alpha=0.30, color=color, zorder=3)
        ax.plot(xs, kde_y, color=color, lw=1.8, zorder=4)

        # ── Boxplot (lower band, y ~0.1–0.35) ────────────────────────────
        bp = ax.boxplot(
            vals_clean,
            vert=False,
            positions=[0.20],
            widths=[0.18],
            patch_artist=True,
            notch=False,
            zorder=5,
            manage_ticks=False,
            flierprops=dict(marker=".", markersize=3, alpha=0.4,
                            markerfacecolor=color, markeredgecolor=color),
            medianprops=dict(color="white", lw=2.0),
            boxprops=dict(facecolor=color, alpha=0.75, edgecolor=color),
            whiskerprops=dict(color=color, lw=1.4),
            capprops=dict(color=color, lw=1.4),
        )

        # ── Axes cosmetics ────────────────────────────────────────────────
        ax.set_ylim(0.0, 1.02)
        ax.set_xlim(x_min - (x_max - x_min) * 0.04,
                    x_max + (x_max - x_min) * 0.04)

        ax.set_yticks([])
        ax.set_xlabel("tSNR", fontsize=10)
        ax.tick_params(axis="x", labelsize=9)

        # Horizontal reference lines (subtle)
        for y in [0.38, 0.68]:
            ax.axhline(y, color="grey", lw=0.5, ls="--", alpha=0.4, zorder=1)

        # Stats annotation
        med  = np.nanmedian(vals_clean)
        iqr  = np.percentile(vals_clean, 75) - np.percentile(vals_clean, 25)
        ax.text(
            0.97, 0.97,
            f"median={med:.1f}\nIQR={iqr:.1f}\nn={len(vals_clean)}",
            transform=ax.transAxes,
            ha="right", va="top", fontsize=8,
            color="#333333",
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="none", alpha=0.7),
        )

        title = f"{sub}  ·  {HEMI_LABEL[hemi]} Hippocampus"
        ax.set_title(title, fontsize=10.5, fontweight="bold", pad=4)

        ax.spines[["top", "right", "left"]].set_visible(False)
        ax.spines["bottom"].set_color("#AAAAAA")

        # Band labels
        ax.text(-0.01, 0.84, "KDE", transform=ax.transAxes,
                ha="right", va="center", fontsize=7.5, color="grey", rotation=90)
        ax.text(-0.01, 0.55, "scatter", transform=ax.transAxes,
                ha="right", va="center", fontsize=7.5, color="grey", rotation=90)
        ax.text(-0.01, 0.20, "box", transform=ax.transAxes,
                ha="right", va="center", fontsize=7.5, color="grey", rotation=90)

fig.suptitle(
    "Hippocampal tSNR — 512-vertex surface (space-corobl)\n"
    r"tSNR = 10000 / $\sigma_t$  [HCP mode-normalised, hp2000-filtered]",
    fontsize=12, fontweight="bold", y=0.975,
)

plt.savefig(OUT_FIG, dpi=180, bbox_inches="tight", facecolor=fig.get_facecolor())
print(f"\nSaved → {OUT_FIG}")
plt.show()
