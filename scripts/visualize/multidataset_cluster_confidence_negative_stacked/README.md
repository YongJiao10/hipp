# Multi-Dataset Cluster Confidence Stacked Negative FC Visualization

This directory provides the production-grade visualization script for multi-dataset hippocampal negative functional connectivity (anticorrelated FC magnitude) profiles using a compact **5 rows × 2 columns (10 groups)** layout with locked Workbench 3D overlays.

## Script Overview

- **Script:** [`plot_multidataset_cluster_negative_stacked.py`](plot_multidataset_cluster_negative_stacked.py)
- **Output:** [`multidataset_cluster_confidence_negative_stacked.png`](multidataset_cluster_confidence_negative_stacked.png)

## Layout & Design Specifications

1. **5 Rows × 2 Columns Format:**
   - **Column 1 (5 cohorts):** HCP 7T, Iowa 3T HC (N=67), Iowa 3T Patient (N=26; sub-3003 excluded), Stanford NTHC (N=51), Stanford TEHC (N=40).
   - **Column 2 (5 cohorts):** Stanford NTS (N=56), Stanford NTS Any MDD (N=43), Stanford NTS Pure MDD (N=11), Stanford TIS (N=47), Stanford TIS PTSD (N=35).
2. **Compact & Snug Geometry (`38.0 × 13.0` inches @ 300 DPI):**
   - **Snug 3D Hippocampus Overlay:** Width ratio `0.430` with `wspace=0.005` ensures the 4-view composite sits flush against both the left and right midline y-axes with zero lateral gaps.
   - **Natural Aspect Ratio:** Renders via `aspect="equal"` with zero distortion, mathematically matching the exact vertical height of the 3 stacked bar rows.
   - **Dataset Badge Distances:** Column 1 badges are placed tightly at `x = -0.250`, while Column 2 is placed at `x = -0.350` to leave comfortable clear space from long labels (e.g. `NTS Pure MDD`).
   - **Symmetrical Top/Bottom Margins:** Y-axis bounds (`y_min=-1.10`, `y_max=22.60`) provide clean 0.60 units of breathing room at both the top and bottom, eliminating meaningless bottom voids and clipping.
3. **Typography & Styling:**
   - Group/Cohort names: `fontsize=16.0` (Heavy).
   - Dataset badges: `fontsize=16.5` (Heavy).
   - Negative FC bar text: `fontsize=11.0` (|z| ≥ 0.020) and `fontsize=9.0` (|z| ≥ 0.012) with outline stroke effects.

## Performance & Panel Caching (`panels/`)

Connectome Workbench 3D renders are persistently cached under:
- `panels/wb_{cohort_name}_2mm_tight4view.png`

With persistent panel caching enabled, subsequent script executions complete in ~4.0s (down from ~28.7s), drastically accelerating iteration on visualization styling without re-rendering 3D hippocampal meshes.
