# Multi-Dataset Hippocampal Visualization Suite & 2x2 Grid Architecture

Unified visualization system for multi-dataset hippocampal functional connectivity, gradient parcellations, Winner-Take-All (WTA) maps, cortical projections, anatomical subfield compositions, cortical proximity controls, and template-matching dominant subregions across 12 cohorts ($N=554$) in 8 master combination figure suites.

---

## 1. The 2x2 Matrix Architecture

The pipeline unifies multi-dataset visualizations across a $2 \times 2$ matrix combining hippocampal smoothing resolution and fMRI preprocessing denoising strategy:

| Grid Cell | Smoothing | Iowa 3T & Stanford Denoising | HCP 7T Baseline Denoising | Status |
| --- | --- | --- | --- | --- |
| **`2mm_36p`** | 2mm | 36P (36-parameter confound regression) | ICA-FIX (Canonical Golden Standard) | **Fully Materialized** (10/10 Cohorts Ready, Canonical Default) |
| **`2mm_22p`** | 2mm | 22P (22-parameter confound regression) | ICA-FIX (Canonical Golden Standard) | **Materialized** (HCP + Iowa HC/Patient Ready; Stanford 22P pending upstream) |
| **`4mm_36p`** | 4mm | 36P (36-parameter confound regression) | ICA-FIX (Canonical Golden Standard) | **Fully Materialized** (10/10 Cohorts Ready) |
| **`4mm_22p`** | 4mm | 22P (22-parameter confound regression) | ICA-FIX (Canonical Golden Standard) | **Materialized** (HCP + Iowa HC/Patient Ready; Stanford 22P pending upstream) |

> [!IMPORTANT]
> **HCP 7T 降噪属性说明（关键事实）**:
>
> - **HCP 7T 没有 22P 或 36P 变体**，其时间序列由 HCP 官方流水线采用 **ICA-FIX** 进行高阶独立成分降噪。
> - `22P` 与 `36P` 的预处理降噪策略对比**仅适用于 Iowa 3T 与 Stanford 两个数据集**（fMRIPrep/xcp-d 管道）。
> - 在所有 4 组网格图中，HCP 7T 均作为一致的金标准参考队列（FIX Reference Baseline），数据仅受海马平滑分辨率（`2mm` vs `4mm`）影响。

### Cohort Matrix & Baseline Rules

- **Column 1 (Discovery & Baseline Controls, 6 cohorts, N=392)**:
  1. `hcp_7t`: HCP 7T Young Adults ($N=151$) — **无 22P/36P 区分**，恒定使用官方 ICA-FIX 金标准基准；仅随平滑分辨率 (`2mm` vs `4mm`) 切换。
  2. `iowa_hc`: Iowa 3T Healthy Controls ($N=67$) — `36p` 采用 `kong2022_7net_hc_full_k3_fisherz`；`22p` 采用 `kong2022_7net_hc_22p_full_k3_fisherz`。
  3. `iowa_pat`: Iowa 3T Depression Patients ($N=26$, sub-3003 excluded) — 默认 canonical7 分支使用已恢复的 `routea_patient_full_k3_fisherz` 与 `routea_patient_22p_full_k3_fisherz`；Lynch2024 7net 使用对应的 `lynch2024_7net_iowa3t_patient*` roots。所有 comparison overview 都使用固定的 26 人 Patient roster。
  4. `stan_nthc`: Stanford Non-Trauma Controls ($N=51$) — `36p` 采用 `routea_7net_nthc_full_k3_fisherz`；`22p` 采用 `routea_7net_nthc_22p_full_k3_fisherz`。
  5. `stan_tehc`: Stanford Trauma-Exposed Controls ($N=40$) — `36p` 采用 `routea_7net_tehc_full_k3_fisherz`；`22p` 采用 `routea_7net_tehc_22p_full_k3_fisherz`。
  6. `stan_nts`: Stanford Non-Trauma Symptomatic ($N=56$) — `36p` 采用 `routea_7net_nts_full_k3_fisherz`；`22p` 采用 `routea_7net_nts_22p_full_k3_fisherz`。
- **Column 2 (Clinical Subgroups & Comorbidities, 6 cohorts, N=162)**:
  7. `iowa_pure_mdd`: Iowa 3T Pure MDD ($N=7$) — `36p` 采用 `kong2022_7net_pure_mdd_full_k3_fisherz`；`22p` 采用 `kong2022_7net_pure_mdd_22p_full_k3_fisherz`。
  8. `iowa_comorbid_mdd`: Iowa 3T Comorbid MDD ($N=19$) — `36p` 采用 `kong2022_7net_comorbid_mdd_full_k3_fisherz`；`22p` 采用 `kong2022_7net_comorbid_mdd_22p_full_k3_fisherz`。
  9. `stan_nts_mdd`: Stanford NTS Any MDD ($N=43$) — `36p` vs `22p`。
  10. `stan_nts_pure`: Stanford NTS Pure MDD ($N=11$) — `36p` vs `22p`。
  11. `stan_tis`: Stanford Trauma-Informed Symptomatic ($N=47$) — `36p` vs `22p`。
  12. `stan_tis_ptsd`: Stanford TIS PTSD ($N=35$) — `36p` vs `22p`。

---

---

## 2. Directory & Component Layout

```
scripts/visualize/
├── lib/                                          # Centralized Single Source of Truth
│   ├── __init__.py                               # Package exports
│   ├── multidataset_cohorts.py                   # GridSpec, 12 cohorts SSOT, path resolution & preflight audit
│   └── visual_style.py                           # Network colors, abbreviations, subregion styling
├── run_multidataset_suite.py                     # Master CLI Orchestrator & Batch Runner
├── multidataset_cluster_confidence_stacked/      # Figure 1: 6x2 cluster-network positive FC stacked profiles
├── multidataset_gradient_subregion_top20_stacked_2col/ # Figure 2: Top-20 cortical network composition
├── multidataset_hipp_cortex_gradient_wta/        # Figure 3: Hippocampus + cortical projection overview
├── multidataset_subfield_composition_2mm_2col/  # Figure 4: MultiHist7 subfield compositions & Stanford differences
├── multidataset_workbench_hippocampus_overlay/   # Figure 5: 3D Workbench parcellation overlays & vertex stats
├── multidataset_wta_normalized_longaxis_stacked_2col/ # Figure 6: WTA long-axis composition
├── hcp_cortex_control_gradient_wta/              # Figure 7: HCP cortex proximity control (Standard vs CtxDrop vs CtxReg)
└── template_matching_dominant_subregions/        # Figure 8: Iowa 3T 4-grid dominant subregions & dominance heatmaps (4 cohorts)
```

---

## 3. Network Taxonomies: Canonical7 Default; Canonical8 Historical

All 8 maintained visualization figure suites (the 6 multi-cohort suite visualizations, Figure 7 HCP cortex proximity control, and Figure 8 Iowa 3T template-matching dominant subregions) use semantic `7net` or canonical functional network taxonomies by default. Their group roots are explicit canonical7 roots (`kong2022_7net_*` for HCP/Iowa and `routea_7net_*` for Stanford), while the inner gradient/WTA producer directories are the bare `kong2022` namespace. The `kong2022_7net` name is a presentation/taxonomy label; it never selects a seven-column gradient input. Every active gradient profile is `(3,400)` and every active WTA vertex profile has 400 ROI columns. Canonical8 is historical and is not a default:

| Taxonomy / Atlas | Flag (`--atlas`) | Networks (ROIs / 400) | Notes | Output Tag |
| --- | --- | --- | --- | --- |
| **Kong2022 (Canonical 7)** | `7net` (default; `kong2022` alias) | DMN (77), SMN (68), VAN (52), DAN (51), CON (73), LAN (18), VN (61) | AUD merged into SMN (49+19=68) | `{figure}_{grid}.png` |
| **Kong2022 (Canonical 8, historical)** | explicit `canonical8` resolver only | DMN (77), SMN (49), AUD (19), VAN (52), DAN (51), CON (73), LAN (18), VN (61) | Archived under `outputs_*/work/archive/8net_20260909/` | `8net_{grid}` |

- **Readiness**: The default canonical7 branch is 100% materialized and verified across all 12 cohorts and all 4 grid combinations (`2mm_36p`, `2mm_22p`, `4mm_36p`, `4mm_22p`).
- **Top WTA Networks in Figure 3**: canonical7 DMN, VAN, SMN, and LAN. Kong2022 WTA is always the ROI-registry `normalized_top20_count` assignment; canonical8 is only a historical taxonomy view of that assignment.
- **Figure 2 Subregion Top-20 Capacity Normalization**:
  - In the default canonical7 view, Somatomotor normalizes by 68 cortical ROIs (instead of 49).

---

## 4. Additive HCP/JJ_TMS 7-Network Comparison

The six active renderers also provide an additive HCP/JJ_TMS comparison branch via `--version hcp_jjtms_7net`. It changes the displayed comparison content while preserving the existing renderer layouts, dataset bands, side labels, Workbench rendering, and panel styling.

- The maintained comparison method is **Kong2022 7-Network**; its canonical WTA atlas is always the normalized Top-20 ROI-count product.
- Each method column keeps the original dataset grouping: **HCP 7T** and **Iowa 3T**, with separate background bands and dataset side labels. Stanford cohorts are not included in this branch.
- Both methods use the same fixed roster: HCP 7T `N=151`, Iowa HC `N=67`, Iowa Patient `N=26` (sub-3003 excluded), Pure MDD `N=7`, and Comorbid MDD `N=19`.
- Canonical 8-network and canonical 7-network products are independent branches; neither is recomputed from or dependent on the other.

Render all four grids simultaneously in 4-way parallel with:

```bash
/home/yojiao/.conda/envs/py314/bin/python scripts/visualize/run_multidataset_suite.py --version hcp_jjtms_7net --figures all
```

## 5. Excluded Visualizations (Hard Guardrail)

Per strict repository requirements, the following 5 visualizations/directories **DO NOT** participate in the $2 \times 2$ grid combinations and remain untouched:

1. `scripts/visualize/multidataset_gradient_subregion_top20_fixed_hcp7t_2col` (Fixed HCP-7T template variant)
2. `scripts/visualize/iowa7t_cortex_master_2row` (Iowa 7T cortex-specific display)
3. `scripts/visualize/kong2022_standard_lateral` (Standard cortical lateral & medial surfaces: Kong2022 17-Net & Canonical 7-Net, Lynch2024 20-Net & Canonical 7-Net comparison)
4. `scripts/visualize/multidataset_cluster_confidence_negative_stacked` (Negative cluster confidence analysis)
5. `scripts/visualize/multidataset_subfield_composition_2mm_2col/archive_v1_v3` (Archival runs)

---

## 6. Execution & Usage

Always use the validated explicit Python interpreter:

```bash
/home/yojiao/.conda/envs/py314/bin/python
```

### Preflight Data Audit

Verify data availability before rendering:

```bash
# Audit the canonical7 branch (the maintained default)
/home/yojiao/.conda/envs/py314/bin/python scripts/visualize/run_multidataset_suite.py --atlas 7net --check-readiness
```

The comparison preflight validates every manifest-declared gradient/WTA input
and overview PNG, and rejects missing or temporary (`.tmp*`/`temporary*`)
overview paths before any renderer starts. It never substitutes an older
overview or selects a file by directory order.
For the Iowa Lynch2024 comparison column, the same preflight also checks every
roster subject's PFM manifest, run-level CIFTI inputs, declared surfaces/priors,
and component outputs; temporary or pre-concatenated provenance is rejected.

### Batch Suite Execution

Surface panels in the confidence, Workbench overlay, and subfield-composition
figures require `.sources.json` provenance: exact source manifests, labels,
surfaces, scene/spec, rendering code/style, grid selection, and output SHA-256.
An old PNG without matching provenance is rebuilt from its canonical inputs;
missing canonical inputs raise before cache lookup. File size alone never
authorizes reuse. Subfield comparison labels use the explicitly registered
assignment variant (Kong normalized-top20; Lynch Fisher-z).

The excluded negative cluster-confidence renderer also validates each Kong
`fisherz` assignment manifest before rendering and uses only its validated
`template_functional_atlas_label`; a mismatched or malformed assignment fails
closed instead of selecting the manifest field directly.

The unified suite refreshes all four grid cells simultaneously in 4-way parallel
execution (`max_workers=4`) by default. Standalone renderer CLIs and single-grid CLI flags
have been removed; direct execution exits with an error. The suite invokes internal
rendering functions in isolated subprocesses concurrently across the 4 grid combinations.
No single-grid bypass, serial execution option, or environment-variable override is provided.

```bash
# Default canonical7 branch: all 4 grid cells in 4-way parallel
/home/yojiao/.conda/envs/py314/bin/python scripts/visualize/run_multidataset_suite.py --atlas 7net
```

### Select a Figure Type, Always Refresh Four Grids in Parallel

Use `--figures` to select a figure type. All four grid cells are always refreshed simultaneously in 4-way parallel:

```bash
# Figure 1 (Cluster Confidence Stacked) 7net
/home/yojiao/.conda/envs/py314/bin/python scripts/visualize/run_multidataset_suite.py --figures cluster_confidence_stacked

# Figure 2 (Gradient Subregion Top-20) 7net
/home/yojiao/.conda/envs/py314/bin/python scripts/visualize/run_multidataset_suite.py --figures gradient_subregion_top20_stacked_2col

# Figure 3 (Hippocampus + Cortex Gradient & WTA) 7net
/home/yojiao/.conda/envs/py314/bin/python scripts/visualize/run_multidataset_suite.py --figures hipp_cortex_gradient_wta

# Figure 4 (Subfield Composition) 7net
/home/yojiao/.conda/envs/py314/bin/python scripts/visualize/run_multidataset_suite.py --figures subfield_composition_2col

# Figure 5 (Workbench Hippocampus Overlay) 7net
/home/yojiao/.conda/envs/py314/bin/python scripts/visualize/run_multidataset_suite.py --figures workbench_hippocampus_overlay
```

### Additive HCP/JJ_TMS Kong2022 version

The original 12-cohort/Stanford suite remains the default (`--version legacy`).
The additive version below is selected explicitly with
`--version hcp_jjtms_7net`; it does not include Stanford and uses the
maintained canonical Kong2022 comparison method:

| Method column | Cohort rows |
|---|---|
| `Kong2022 7-network` | HCP 7T, JJ_TMS HC, Patient, Pure MDD, Comorbid MDD |

Run the four grid cells simultaneously in 4-way parallel with:

```bash
/home/yojiao/.conda/envs/py314/bin/python scripts/visualize/run_multidataset_suite.py \
  --version hcp_jjtms_7net
```

This creates 24 PNGs (six participating figure types × four grid cells) with
the disjoint tag `hcp_jjtms_7net_{2mm|4mm}_{36p|22p}` in the existing figure
directories. Use `--check-readiness` first when validating upstream products:

```bash
/home/yojiao/.conda/envs/py314/bin/python scripts/visualize/run_multidataset_suite.py \
  --version hcp_jjtms_7net --check-readiness
```

The explicit comparison branch resolves the `kong2022_7net` presentation
namespace to the canonical internal `gradient-spectral/kong2022` producer path
and the canonical `wta-top/kong2022` normalized Top-20 WTA path independently.
Comparison rendering fails on a missing upstream product instead of creating a
transparent placeholder or substituting another cohort.
Overlay panels are derived from the requested grid's functional labels. The
branch changes only the displayed method/cohort content; figure layout and the
existing Workbench/original-renderer paths remain unchanged.

All figure refresh and rendering operations must be executed through `run_multidataset_suite.py` across all 4 grid combinations simultaneously (`2mm_36p`, `2mm_22p`, `4mm_36p`, `4mm_22p`) in 4-way parallel (`max_workers=4`). Standalone rendering and single-figure CLI flags have been removed:

```bash
/home/yojiao/.conda/envs/py314/bin/python scripts/visualize/run_multidataset_suite.py \
  --version hcp_jjtms_7net --figures cluster_confidence_stacked
```

The current release audit is recorded in
`docs/kong400_recovery_audit/visualization_7net_audit.json`. It checks all 24
active comparison PNGs, the four statistics-producing figure families, the
40 Kong comparison gradient/WTA source manifests, and the 20 all-grid subject
profiles. The HCP ctxdrop/ctxreg release is separately checked by
`docs/kong400_recovery_audit/hcp_control_overview_400roi_audit.json`: two
route manifests, six overview items, and all gradient/WTA source profiles
prove the 400-ROI contract. Historical pseudo-cluster sidecars remain
traceability records and are not accepted as active comparison inputs.

---

## 7. Figure 7: HCP Cortex Proximity Control Combination Visualization (`hcp_cortex_control_gradient_wta`)

Comprehensive multi-modal combination figure directly evaluating hippocampal functional connectivity under cortical proximity controls in HCP 7T Young Adults ($N=151$) across the $2\times2$ grid matrix (`2mm_36p`, `2mm_22p`, `4mm_36p`, `4mm_22p`):

- **Branches Compared**:
  1. `Standard`: Unconstrained RouteA baseline reference (`outputs_hcp/work/group_atlas/kong2022_7net_full_k3_fisherz`)
  2. `CtxDrop`: Proximity control dropping adjacent cortical vertices (`outputs_hcp/work/group_atlas/routea_ctxdrop`)
  3. `CtxReg`: Proximity control regressing adjacent cortical vertex signals (`outputs_hcp/work/group_atlas/routea_ctxreg`)

- **Subfigures & Modalities Displayed**:
  1. **Subfig 1: Gradient Spectral Clustering — Positive Fisher-z Profiles**:
     - Matches `multidataset_cluster_confidence_stacked` layout (`width_ratios = [0.41, 0.18, 0.41]`, `wspace = 0.012`)
     - `[Left Hipp Stacked Bar] | [2x2 Surface Overlay] | [Right Hipp Stacked Bar]`
     - Dynamic positive Fisher-z scale per branch (Standard: ~0.25, CtxDrop: ~0.06, CtxReg: ~0.03)
     - Vertical order: `C1 (Ant)` at top, `C2 (Mid)` in center, `C3 (Post)` at bottom with cohort badge maintaining $+26\text{ px}$ clearance
  2. **Subfig 2: Gradient Spectral Clustering — Normalized Top-20 Network Recruitment (%)**:
     - Matches `multidataset_gradient_subregion_top20_stacked_2col` layout (`width_ratios = [0.12, 0.18, 0.35, 0.35]`, `wspace = 0.024, hspace = 0.14`)
     - `[Cohort Card] | [2x2 Surface Overlay] | [Left Hipp Stacked Bar] | [Right Hipp Stacked Bar]`
     - Normalized 0–100% capacity-adjusted cortical ROI recruitment
     - Vertical order: Anterior at top, Body in center, Posterior at bottom
  3. **Subfig 3: Winner-Take-All (WTA) Parcellation — 20 Long-Axis Bins (%)**:
     - Matches `multidataset_wta_normalized_longaxis_stacked_2col` layout with dedicated spacer column (`width_ratios = [0.11, 0.15, 0.085, 0.3275, 0.3275]`, `wspace = 0.015, hspace = 0.12`)
     - `[Cohort Card] | [2x2 Surface Overlay] | [Spacer] | [Left Hipp Stacked Bar] | [Right Hipp Stacked Bar]`
     - Dedicated spacer provides $+48\text{ px}$ clearance between 2x2 surface and `Long-Axis Bin` labels
     - Vertex percentage (0–100%) belonging to each canonical 7 network
     - Vertical order: Bin 1 (Anterior) at top, Bin 20 (Posterior) at bottom

Execute single grid or batch all 4 grids:

```bash
# Render single grid (e.g. 2mm_36p)
/home/yojiao/.conda/envs/py314/bin/python scripts/visualize/hcp_cortex_control_gradient_wta/render_hcp_cortex_control_gradient_wta.py --grid 2mm_36p

# Render all four grid combinations
/home/yojiao/.conda/envs/py314/bin/python scripts/visualize/hcp_cortex_control_gradient_wta/render_hcp_cortex_control_gradient_wta.py --all-grids

# Force re-rendering of all 3D surface panels from Connectome Workbench
/home/yojiao/.conda/envs/py314/bin/python scripts/visualize/hcp_cortex_control_gradient_wta/render_hcp_cortex_control_gradient_wta.py --all-grids --force
```

Outputs:

- 4 Master figures ($10780 \times 3960\text{ px}$, $49\times18\text{ in}$ at 220 DPI):
  - `hcp_cortex_control_gradient_wta_2mm_36p.png`
  - `hcp_cortex_control_gradient_wta_2mm_22p.png`
  - `hcp_cortex_control_gradient_wta_4mm_36p.png`
  - `hcp_cortex_control_gradient_wta_4mm_22p.png`
- Summary statistics TSVs under `stats/`:
  - `hcp_cortex_control_gradient_wta_{grid}_stats.tsv`
- Panel cache under `panels/` with SHA-256 `.sources.json` provenance sidecars.

---

## 8. Figure 8: Iowa 3T Template-Matching Dominant Subregions Visualization (`template_matching_dominant_subregions`)

The formal renderer is maintained independently of `ctxreg_worktree`.
The original four-grid implementation is preserved as
`render_template_matching_dominant_subregions_legacy.py`; the unsuffixed
renderer is the maintained formal entry point and adds an explicit
Standard/CtxReg 2mm/36P branch mode. The branch figures append two Welch
t-test columns per cohort row (per-subject vertex count and area-% of the
subject's own hemisphere total vs HC, two-sided, uncorrected with a
`+ 0.05≤p<0.10` trend band); these t-test figures ARE the canonical branch
products and carry no suffix. Grid-mode t-test outputs keep the `_ttest`
suffix so they never overwrite the canonical tracked grid products.

Computes and displays the vertex-wise dominant functional subregion atlas (`Anterior / VAN`, `Middle / SMN`, `Posterior / DMN`) and subject dominance agreement heatmaps across the 4 standard grid combinations (`2mm_36p`, `2mm_22p`, `4mm_36p`, `4mm_22p`).

Each grid figure presents all **4 Iowa 3T cohorts** in stacked comparative rows:

1. Healthy Controls (`HC`, $N=67$)
2. Depression Patients (`Patient`, $N=26$)
3. Pure MDD Subgroup (`Pure MDD`, $N=7$)
4. Comorbid MDD Subgroup (`Comorbid MDD`, $N=19$)

Each row displays:

- Left metadata card ($w=280\text{ px}$)
- Center Dominant Atlas 4-view surface ($w=790\text{ px}$)
- Right Dominance Agreement Heatmap 4-view surface ($w=870\text{ px}$, `magma` scaled $33.3\% - 80.0\%$)

Run with the repository's explicit Python interpreter:

```bash
/home/yojiao/.conda/envs/py314/bin/python scripts/visualize/template_matching_dominant_subregions/render_template_matching_dominant_subregions.py
```

Outputs:

- 4 Master figures ($1940 \times 2580\text{ px}$, 300 DPI):
  - `template_matching_dominant_subregions_2mm_36p.png`
  - `template_matching_dominant_subregions_2mm_22p.png`
  - `template_matching_dominant_subregions_4mm_36p.png`
  - `template_matching_dominant_subregions_4mm_22p.png`
- Summary statistics: `template_matching_dominant_subregions_stats.csv` (all 16 conditions)
- Provenance manifest: `template_matching_dominant_subregions_manifest.json`
- Panel cache: `panels/` (32 intermediate 4-view composite captures).

The explicit branch mode is run with:

```bash
/home/yojiao/.conda/envs/py314/bin/python \\
  scripts/visualize/template_matching_dominant_subregions/render_template_matching_dominant_subregions.py \\
  --signal-branch all \\
  --repo-root /home/yojiao/hpchome/hipp \\
  --data-root /home/yojiao/hpchome/hipp
```

It writes branch-qualified PNGs, a comparison CSV, and a provenance
manifest. It validates exact combo manifests and their declared label/score
artifacts; it does not copy or execute worktree products.

---

## 9. Presentation Slides Synchronization

The master PowerPoint presentation (`Cross-Dataset Hippocampal Parcellation.pptx` in repository root) can be updated or synchronized with:

```bash
# Update HCP/JJ_TMS 7-network comparison sections (default, 2mm_36p, 4mm_36p, 2mm_22p, 4mm_22p)
/home/yojiao/.conda/envs/py314/bin/python scripts/visualize/update_cross_dataset_presentation.py --version hcp_jjtms_7net

# Update only a specific figure across all sections (e.g., Page 6 Subfield Composition)
/home/yojiao/.conda/envs/py314/bin/python scripts/visualize/update_cross_dataset_presentation.py --version hcp_jjtms_7net --page 6

# Synchronize legacy 12-cohort suite (both canonical 8-network and 7-network sections)
/home/yojiao/.conda/envs/py314/bin/python scripts/visualize/update_cross_dataset_presentation.py --version legacy --atlas all

# Dry-run validation without modifying PPTX
/home/yojiao/.conda/envs/py314/bin/python scripts/visualize/update_cross_dataset_presentation.py --dry-run
```

Slide & Section Structure:

- **Slide 1 (Default / No Section)**: Standalone Kong2022 vs Lynch2024 native canonical 7-network comparison (`kong_lynch_native_canonical7_comparison.png`), centered with clean default white master background, unassigned to any PowerPoint section.
- **HCP/JJ_TMS 7-network comparison & Canonical7 default** (`7net_2mm_36p`, `7net_4mm_36p`, `7net_2mm_22p`, `7net_4mm_22p`): 8 slides per section using the maintained figures. All figures are scaled to maximum slide coverage while preserving aspect ratios with zero title overlap and zero boundary overflow:
  - Page 1: Hippocampal Functional Network Profiles (Gradient)
  - Page 2: Subregion Vertices Composition (gradient)
  - Page 3: WTA Atlas & Long-Axis Network Composition (WTA)
  - Page 4: Cortical Functional Projections (gradient & WTA)
  - Page 5: Network Composition of Hippocampal Gradient Atlas
  - Page 6: Hippocampal Subfield Composition
  - Page 7: HCP 7T Cortical Proximity Control (Standard vs CtxDrop vs CtxReg, annotated with concise CtxDrop and CtxReg scientific definitions)
  - Page 8: Iowa 3T Template Matching Dominant Subregions & Heatmaps
- **Canonical 8-network historical** (`8net_2mm_36p`, `8net_4mm_36p`, `8net_2mm_22p`, `8net_4mm_22p`): 6 or 7 slides per section when explicitly requested.
- **Cortical network area** (`cortical network area`, final section): 3 slides (Kong2022 patient forest comparison, Lynch2024 patient forest comparison, and healthy control baseline). Figures are placed side by side without cropping.
- Total presentation slides: 36 slides (1 cover/comparison + 4 sections × 8 slides + 3 cortical network area slides). Both `Hippocampal Parcellation 7net.pptx` and `Cross-Dataset Hippocampal Parcellation.pptx` are kept in lockstep synchronization.
- A full synchronization (without `--page`) refreshes all sections. Supplying `--page` updates only the selected figure page across sections.
- Existing historical `7net_<grid>` section names are explicitly recognized and updated in place; if both a canonical and historical name exist for the same grid, synchronization stops with an ambiguity error instead of creating duplicate sections.

---

## 10. Concurrency & Safety Guardrails

- **IDAS Cap**: Hard limit of **10** concurrent processes.
- The suite orchestrator operates sequentially or bounded within $\le 4$ workers to strictly respect system load constraints.

---

## 11. Cross-Grid Diversity, Freshness & Cache Isolation Integrity Contract

To fundamentally prevent cross-variant / cross-grid data leakage, stale cache reuse, or identical panel collisions (e.g. 36P inadvertently reusing 22P assets):

1. **Strict Cache Namespacing**:
   - Every intermediate panel PNG, NPY array, or stats TSV MUST strictly incorporate `{cohort}_{variant}_{atlas}_{smoothing}` in its directory path and filename.
   - Bare `kong2022/` and `routea/` are the canonical7 default namespace; scripts must not fall back from an explicit historical alias to another namespace or cohort root.
2. **Dynamic Atlas Hierarchy**:
   - Default gradient products use `kong2022`; only explicit legacy comparison products may use `kong2022_7net`. WTA products always use the canonical `wta-top/kong2022` directory and never select an atlas by filesystem existence.
3. **Automated Continuous Verification Suite**:
   - Run the dedicated regression suite to enforce mathematical and visual diversity:

   ```bash
   /home/yojiao/.conda/envs/py314/bin/pytest tests/visualize/test_cross_grid_diversity.py
   ```

   - Verifies 100% of the 24 canonical7 master figures exist (six figure groups × four grids), are strictly newer than upstream rebuilds, and exhibit non-zero, significant cross-variant pixel differences ($36\text{P} \neq 22\text{P}$).

---

## 12. Standalone Cortical Network Surface-Area Analysis (`cortex_network_area`)

The independent `scripts/visualize/cortex_network_area/` module measures each
network's area on the subject's individual fsLR-32k midthickness surfaces. The
primary value is the percentage of valid cortical area; the CSV also retains
mm², Noise, and Unassigned area. It has four separate outputs: Kong2022 native
17-network and canonical7, plus Lynch2024 native20 (functional labels; Noise
reported separately) and canonical7. The existing 2x2 suite and its launchers
do not call this module.

Run the readiness audit first, then compute the tables and redraw figures with
the repository's explicit Python interpreter:

```bash
/home/yojiao/.conda/envs/py314/bin/python scripts/visualize/cortex_network_area/run_area_analysis.py --variant all --check-readiness --iowa-surface-manifest scripts/visualize/cortex_network_area/stats/iowa_surface_manifest.json
/home/yojiao/.conda/envs/py314/bin/python scripts/visualize/cortex_network_area/run_area_analysis.py --variant all --iowa-surface-manifest scripts/visualize/cortex_network_area/stats/iowa_surface_manifest.json
/home/yojiao/.conda/envs/py314/bin/python scripts/visualize/cortex_network_area/plot_area_figures.py --stats scripts/visualize/cortex_network_area/stats/network_area_statistics.csv
```

`--variant` accepts `22p`, `36p`, or `all`. HCP resolves one ICA-FIX input for
both variant labels; Iowa resolves the explicit 22P and 36P roots. Missing or
incompatible surfaces, labels, manifests, and BrainModel mappings remain in
`input_audit.csv` and are not replaced by another preprocessing branch. The
analysis output directory contains `subject_network_area_metrics.csv`,
`network_area_statistics.csv`, `input_audit.csv`, and `run_manifest.json`;
the plotting entry point independently reads the statistics CSV and writes PNG forest plots under `figures/`.

The formal rosters are HCP HC 151, Iowa HC 67, Patient 26 (sub-3003 excluded),
Pure MDD 7, and Comorbid MDD 19. The six patient-group contrasts compare each
patient group with Iowa HC and HCP HC; the seventh is Iowa HC versus HCP HC.
Patient and HC–HC plots retain every network and use uncorrected two-sided
Welch significance symbols (* / ** / ***; p≥0.05 is blank). BH-FDR q values are auxiliary and do not
select figures. The HC–HC contrast is a cross-dataset baseline comparison, not
an equivalence test, and overlapping patient subgroups are not independent
replications.

The cortical network area module also exports four separate patient PNG plates (Kong/Lynch × native/canonical7), with Iowa HC/HCP HC control rows, patient-group columns, paired 22P/36P markers, all networks, and aligned significance symbols. The stacked overview is retired. Prepare the complete-roster Iowa surface manifest using the module README before analysis; missing inputs now stop the run. Its HC baseline PNG pairs the two preprocessing variants within each network. See [figure layout and redraw instructions](cortex_network_area/README.md#figure-layout-and-independent-redraw).
