#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.append(str(REPO_ROOT / "scripts"))
from common.compute_fc_gradients import corrcoef_rows

SUBJECTS = ["100610", "102311", "102816"]
ATLASES = ["lynch2024", "hermosillo2024", "kong2019"]
HEMIS = ["L", "R"]

# Combine atlas and hemi for columns
# columns = [(lynch2024, L), (lynch2024, R), ...]
COLUMNS = [(atlas, hemi) for atlas in ATLASES for hemi in HEMIS]

def main() -> None:
    out_dir = REPO_ROOT / "outputs" / "hipp_functional_parcellation_network" / "network-prob-cluster-nonneg"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "nonneg_fc_sum_distributions.png"

    # 3 rows (Subjects) x 6 columns (Atlas x Hemi)
    fig, axes = plt.subplots(nrows=len(SUBJECTS), ncols=len(COLUMNS), figsize=(24, 12), squeeze=False)
    
    shared_dir = REPO_ROOT / "outputs" / "hipp_functional_parcellation_network" / "_shared"
    
    for row_idx, subject in enumerate(SUBJECTS):
        for col_idx, (atlas, hemi) in enumerate(COLUMNS):
            ax = axes[row_idx, col_idx]
            print(f"Processing sub-{subject} | {atlas} | {hemi}")
            
            ts_path = shared_dir / f"sub-{subject}" / "surface" / "2mm" / f"sub-{subject}_hemi-{hemi}_timeseries.npy"
            network_ts_path = shared_dir / f"sub-{subject}" / "reference" / atlas / "cortex_canonical_network_timeseries.npy"
            
            if not ts_path.exists():
                print(f"Warning: {ts_path} not found.")
                continue
            if not network_ts_path.exists():
                print(f"Warning: {network_ts_path} not found.")
                continue
            
            ts = np.load(ts_path)
            network_ts = np.load(network_ts_path)
            
            # Compute vertex-to-network FC
            fc = corrcoef_rows(ts, network_ts)
            
            # Fisher-z transform and non-negative clipping
            fisher = np.arctanh(np.clip(fc, -0.999999, 0.999999)).astype(np.float32)
            fisher_nonneg = np.clip(fisher, 0.0, None)
            
            # Sum across networks for each vertex (axis=1 is the network dimension)
            fc_sum = np.nansum(fisher_nonneg, axis=1)
            
            # Plot Histogram + KDE
            sns.histplot(fc_sum, bins=50, kde=True, color='skyblue', edgecolor='white', ax=ax, zorder=2)
            
            # Determine maximum height to scale scatter plot at the bottom
            max_height = 0
            for patch in ax.patches:
                if patch.get_height() > max_height:
                    max_height = patch.get_height()
            
            if max_height == 0:
                max_height = 10  # fallback
                
            # Boxplot at the bottom using matplotlib directly to control its exact y-position
            ax.boxplot(fc_sum, vert=False, positions=[-0.05 * max_height], widths=[0.04 * max_height],
                       patch_artist=True, manage_ticks=False,
                       boxprops=dict(facecolor="lightgray", color="gray", alpha=0.6),
                       medianprops=dict(color="black"),
                       whiskerprops=dict(color="gray"),
                       capprops=dict(color="gray"),
                       flierprops=dict(marker='o', markersize=2, markeredgecolor='none', markerfacecolor='gray', alpha=0.5))
            
            # Expand y-axis downwards on the main axis to fit the boxplot
            ax.set_ylim(bottom=-0.1 * max_height)
            
            # Set titles and labels
            if row_idx == 0:
                ax.set_title(f"{atlas}\nHemi-{hemi}", fontsize=14, fontweight="bold")
            if col_idx == 0:
                ax.set_ylabel(f"Sub-{subject}\nFrequency", fontsize=12, fontweight="bold")
            else:
                ax.set_ylabel("")
                
            if row_idx == len(SUBJECTS) - 1:
                ax.set_xlabel(r"$\sum$ non-negative Fisher-z FC", fontsize=12)
            else:
                ax.set_xlabel("")
            
            # Grid and styling
            ax.grid(axis='y', linestyle='--', alpha=0.5, zorder=1)
            sns.despine(ax=ax, left=False, bottom=False)

    plt.tight_layout()
    plt.savefig(out_file, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved figure to {out_file}")

if __name__ == "__main__":
    main()
