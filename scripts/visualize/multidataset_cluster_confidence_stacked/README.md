# Multi-Dataset Cluster Confidence Stacked Positive FC Visualization

This directory provides the production-grade visualization script for multi-dataset hippocampal functional connectivity (FC) profiles using a compact **5 rows × 2 columns (10 groups)** layout with locked Workbench 3D overlays.

## Script Overview

Panel reuse requires matching source-content and output SHA-256 fingerprints
in `.sources.json`; source labels/manifests/surfaces and grid are checked first.
Unverified historical PNGs are rebuilt, never accepted by file size alone.

Standalone rendering has been removed. Refresh all four grids in 4-way parallel through
`scripts/visualize/run_multidataset_suite.py --figures cluster_confidence_stacked`
using `/home/yojiao/.conda/envs/py314/bin/python`. Add `--version hcp_jjtms_7net`
for the method comparison. Single-grid diagnostic rendering is not supported.

Branch-pair mode renders the canonical 2mm/36p standard and ctxreg cells with
disjoint names (`..._hcp_jjtms_7net_2mm_36p_standard.png` and
`..._hcp_jjtms_7net_2mm_36p_ctxreg.png`):
`scripts/visualize/run_multidataset_suite.py --version hcp_jjtms_7net --figures cluster_confidence_stacked --signal-branch all`.
The renderer accepts `--signal-branch standard` (default) or `ctxreg`; ctxreg
is only valid with the hcp_jjtms_7net comparison version. Pre-signal_branch
HCP Kong manifests are accepted only under the explicit named legacy
provenance contract; any other branch mismatch fails closed.

- **Script:** [`plot_multidataset_cluster_positive_stacked.py`](plot_multidataset_cluster_positive_stacked.py)
- **Output:** [`multidataset_cluster_confidence_stacked.png`](multidataset_cluster_confidence_stacked.png)
- **Data Exporter:** [`export_multigroup_hipp_net_profiles.py`](export_multigroup_hipp_net_profiles.py) extracts the multi-dataset (HCP 7T, Iowa 3T, Stanford) 3-cluster × 7-network Fisher-z profiles into a single clean CSV (`multigroup_hipp_net_profiles.csv`).

## Layout & Design Specifications

1. **5 Rows × 2 Columns Format:**
   - **Column 1 (5 cohorts):** HCP 7T (N=151), Iowa 3T HC (N=67), Iowa 3T Patient (N=26; sub-3003 excluded), Stanford NTHC (N=51), Stanford TEHC (N=40).
   - **Column 2 (5 cohorts):** Stanford NTS (N=56), Stanford NTS Any MDD (N=43), Stanford NTS Pure MDD (N=11), Stanford TIS (N=47), Stanford TIS PTSD (N=35).
   - **Additive HCP/JJ_TMS comparison:** The two method columns are headed by `Kong2022 7-Network` and `Lynch2024 7-Network`; within each column, the original dataset-band convention is retained for `HCP 7T` and `Iowa 3T` with separate background colors and side labels. Stanford remains only in the legacy renderer.
   - The figure title is `Cross-Dataset Hippocampal Positive Functional Network Profiles`; it does not append `(Stacked)`.
2. **Compact & Snug Geometry (`38.0 × 13.0` inches @ 300 DPI):**
   - **Snug 3D Hippocampus Overlay:** Width ratio `0.430` with `wspace=0.005` ensures the 4-view composite sits flush against both the left and right midline y-axes with zero lateral gaps.
   - **Natural Aspect Ratio:** Renders via `aspect="equal"` with zero distortion, mathematically matching the exact vertical height of the 3 stacked bar rows.
   - **Dataset Badge Distances:** Column 1 badges are placed tightly at `x = -0.250`, while Column 2 is placed at `x = -0.350` to leave comfortable clear space from long labels (e.g. `NTS Pure MDD`).
   - **Symmetrical Top/Bottom Margins:** Y-axis bounds (`y_min=-1.10`, `y_max=22.60`) provide clean 0.60 units of breathing room at both the top and bottom, eliminating meaningless bottom voids and clipping.
   - **Adaptive Positive-FC X-Axis:** Each rendered column derives one shared left/right x-range from its largest stacked positive Fisher-z profile, rounds to readable tick steps, and adds a small (~8%) margin so small profiles do not leave excess blank space while larger profiles are not clipped.
3. **Typography & Styling:**
   - Group/Cohort names: `fontsize=16.0` (Heavy).
   - Dataset badges: `fontsize=16.5` (Heavy).
   - Positive FC bar text: `fontsize=11.0` (z ≥ 0.020) and `fontsize=9.0` (z ≥ 0.012) with outline stroke effects and `clip_on=True`.
4. **Performance & Caching:**
   - 中间 4 视角海马表面合成图内置 `panels/` 持久化缓存机制（`wb_{cohort}_{smoothing}_{variant}_tight4view.png`）。首次运行后自动缓存，微调坐标轴刻度、字体或间距时直接读取缓存，运行耗时由 ~35 秒降至 **4 秒**。
