"""
Plot low-tSNR vertex percentage distribution across all 166 subjects.
2 subplots: Left / Right hippocampus.
Each subplot: scatter + KDE + boxplot (style mirrors plot_tsnr_distributions.py).

"Null" vertex = tSNR < 25.
"""

import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.stats import gaussian_kde
from pathlib import Path

# ── Paths ────────────────────────────────────────────────────────────────────
REPO_ROOT  = Path(__file__).resolve().parents[1]
SHARED_DIR = REPO_ROOT / "outputs_migration" / "hipp_group_inputs_shared"
OUT_FIG    = REPO_ROOT / "outputs_migration" / "null_pct_distribution.png"

HCP_MODE   = 10000.0
HEMIS      = ["L", "R"]
HEMI_LABEL = {"L": "Left", "R": "Right"}
COLORS     = {"L": "#4C72B0", "R": "#DD8452"}

# ── Discover subjects ─────────────────────────────────────────────────────────
subjects = sorted([
    d.name for d in SHARED_DIR.iterdir()
    if d.is_dir() and d.name.startswith("sub-")
])
print(f"Found {len(subjects)} subjects")

# ── Compute null % per subject × hemi ────────────────────────────────────────
null_pct = {h: [] for h in HEMIS}   # hemi → list of floats (one per subject)
skipped  = {h: [] for h in HEMIS}

for sub in subjects:
    for hemi in HEMIS:
        gii_path = (SHARED_DIR / sub / "surface" / "raw" /
                    f"{sub}_hemi-{hemi}_space-corobl_den-512_label-hipp_bold.func.gii")
        if not gii_path.exists():
            print(f"  MISSING: {sub} {hemi}")
            skipped[hemi].append(sub)
            continue

        img  = nib.load(str(gii_path))
        bold = np.stack([d.data for d in img.darrays], axis=1).astype(np.float64)
        sd   = bold.std(axis=1, ddof=1)
        tsnr = np.where(sd > 0, HCP_MODE / sd, np.nan)
        n_null = (~(tsnr >= 25)).sum()      # tSNR<25 + NaN (sd==0) both counted as bad
        pct    = 100.0 * n_null / bold.shape[0]
        null_pct[hemi].append(pct)

for h in HEMIS:
    arr = np.array(null_pct[h])
    print(f"\n{HEMI_LABEL[h]} hemi  (n={len(arr)}, skipped={len(skipped[h])})")
    print(f"  mean={arr.mean():.2f}%  median={np.median(arr):.2f}%  "
          f"std={arr.std():.2f}%  max={arr.max():.2f}%  min={arr.min():.2f}%")

# ── Plot ──────────────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(14, 4.5))
fig.patch.set_facecolor("#F8F8F8")

gs = gridspec.GridSpec(
    1, 2,
    figure=fig,
    hspace=0.4, wspace=0.32,
    left=0.08, right=0.96,
    top=0.84, bottom=0.14,
)

rng = np.random.default_rng(42)

for col, hemi in enumerate(HEMIS):
    ax    = fig.add_subplot(gs[0, col])
    vals  = np.array(null_pct[hemi])
    color = COLORS[hemi]

    # ── Scatter (jittered vertically) ─────────────────────────────────────
    jitter_y = rng.uniform(0.42, 0.68, size=len(vals))
    ax.scatter(
        vals, jitter_y,
        s=10, alpha=0.45, color=color, linewidths=0,
        zorder=2,
    )

    # ── KDE ───────────────────────────────────────────────────────────────
    if len(vals) > 1 and vals.std() > 0:
        kde  = gaussian_kde(vals, bw_method="scott")
        x_lo = max(0.0, np.percentile(vals, 0.5))
        x_hi = np.percentile(vals, 99.5)
        xs   = np.linspace(x_lo, x_hi, 400)
        ys   = kde(xs)
        ys_norm = ys / ys.max()
        kde_y   = 0.70 + ys_norm * 0.28
        ax.fill_between(xs, 0.70, kde_y, alpha=0.30, color=color, zorder=3)
        ax.plot(xs, kde_y, color=color, lw=1.8, zorder=4)
    else:
        x_lo, x_hi = vals.min() - 0.5, vals.max() + 0.5

    # ── Boxplot ───────────────────────────────────────────────────────────
    ax.boxplot(
        vals,
        vert=False,
        positions=[0.20],
        widths=[0.18],
        patch_artist=True,
        notch=False,
        zorder=5,
        manage_ticks=False,
        flierprops=dict(marker=".", markersize=4, alpha=0.5,
                        markerfacecolor=color, markeredgecolor=color),
        medianprops=dict(color="white", lw=2.0),
        boxprops=dict(facecolor=color, alpha=0.75, edgecolor=color),
        whiskerprops=dict(color=color, lw=1.4),
        capprops=dict(color=color, lw=1.4),
    )

    # ── Axes cosmetics ────────────────────────────────────────────────────
    ax.set_ylim(0.0, 1.02)
    pad = (x_hi - x_lo) * 0.05
    ax.set_xlim(x_lo - pad, x_hi + pad)

    ax.set_yticks([])
    ax.set_xlabel("% vertices with tSNR < 25", fontsize=10)
    ax.tick_params(axis="x", labelsize=9)

    for y in [0.38, 0.68]:
        ax.axhline(y, color="grey", lw=0.5, ls="--", alpha=0.4, zorder=1)

    # Stats annotation
    med = np.median(vals)
    iqr = np.percentile(vals, 75) - np.percentile(vals, 25)
    ax.text(
        0.97, 0.97,
        f"median={med:.2f}%\nIQR={iqr:.2f}%\nn={len(vals)}",
        transform=ax.transAxes,
        ha="right", va="top", fontsize=8.5,
        color="#333333",
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="none", alpha=0.7),
    )

    ax.set_title(
        f"{HEMI_LABEL[hemi]} Hippocampus",
        fontsize=11, fontweight="bold", pad=5,
    )
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.spines["bottom"].set_color("#AAAAAA")

    # Band labels
    ax.text(-0.01, 0.84, "KDE",     transform=ax.transAxes,
            ha="right", va="center", fontsize=7.5, color="grey", rotation=90)
    ax.text(-0.01, 0.55, "scatter", transform=ax.transAxes,
            ha="right", va="center", fontsize=7.5, color="grey", rotation=90)
    ax.text(-0.01, 0.20, "box",     transform=ax.transAxes,
            ha="right", va="center", fontsize=7.5, color="grey", rotation=90)

fig.suptitle(
    "tSNR < 25 vertex % per subject — 512-vertex hippocampal surface (space-corobl)\n"
    r"tSNR = 10000 / $\sigma_t$  [HCP mode-normalised, hp2000-filtered]",
    fontsize=11.5, fontweight="bold", y=0.97,
)

OUT_FIG.parent.mkdir(parents=True, exist_ok=True)
plt.savefig(OUT_FIG, dpi=180, bbox_inches="tight", facecolor=fig.get_facecolor())
print(f"\nSaved → {OUT_FIG}")
