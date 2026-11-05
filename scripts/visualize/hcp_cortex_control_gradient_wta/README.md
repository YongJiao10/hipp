# HCP 7T Cortical Proximity Control Combination Visualization

Unified multi-modal combination visualization comparing the effect of cortical proximity control across **Standard (Baseline)**, **CtxDrop (Adjacent Dropped)**, and **CtxReg (Adjacent Regressed)** in HCP 7T Young Adults ($N=151$).

---

## 1. Overview & Architecture

Cortical functional connectivity profiles of the hippocampus can be influenced by volume conduction and spatial proximity to the adjacent parahippocampal and medial temporal cortex. To rigorously assess whether hippocampal gradient organization and Winner-Take-All (WTA) functional parcellation are driven by cortical proximity bias or true long-range functional networks, this visualization unifies three formal HCP cortex-control branches across three complementary analytical modalities:

### The 3 Cortex-Control Branches
1. **Standard (Baseline Reference)**:
   - Group Atlas Root: `outputs_hcp/work/group_atlas/pfm`
   - Overview Root: `outputs_hcp/present/overviews/pfm`
   - Description: Unconstrained full-cortex connectivity matrix ($N=151$, Young Adults, ICA-FIX).
2. **CtxDrop (Adjacent Vertices Dropped)**:
   - Group Atlas Root: `outputs_hcp/work/group_atlas/pfm_ctxdrop`
   - Overview Root: `outputs_hcp/present/overviews/pfm_ctxdrop`
   - Description: Cortical vertices within spatial proximity to the hippocampus are completely excluded prior to gradient decomposition and parcellation.
3. **CtxReg (Adjacent Vertices Regressed)**:
   - Group Atlas Root: `outputs_hcp/work/group_atlas/pfm_ctxreg`
   - Overview Root: `outputs_hcp/present/overviews/pfm_ctxreg`
   - Description: Cortical time-series are residualized against adjacent proximity signals prior to gradient calculation and parcellation.

---

## 2. The 3 Analytical Modalities

Each branch column presents three vertically stacked analytical modalities:

1. **Modality 1: Gradient Spectral Clustering — Positive Fisher-z FC Profiles**:
   - **Visual Style**: Aligned with `multidataset_cluster_confidence_stacked`.
   - **Hippocampus 3D Surface**: 4-view locked Workbench composite (`Left Front`, `Right Front`, `Left Back`, `Right Back`) showing $k=3$ cluster assignments labeled by dominant positive Fisher-z network.
   - **Stacked Bar Panels**: Left and Right hippocampus horizontal stacked bars showing cumulative positive Fisher-z functional connectivity values across canonical 7 networks for C1 (Anterior), C2 (Middle), and C3 (Posterior).
   - **Annotations**: Dominant network label, margin to second-highest network ($\Delta$), and segment abbreviations (`DMN`, `SMN`, `VAN`, `DAN`, `CON`, `LAN`, `VN`).
   - **X-axis Scale**: Dynamically harmonized positive Fisher-z scale.

2. **Modality 2: Gradient Spectral Clustering — Normalized Top-20 Network Recruitment (%)**:
   - **Visual Style**: Aligned with `multidataset_gradient_subregion_top20_stacked_2col`.
   - **Hippocampus 3D Surface**: 4-view locked Workbench composite showing $k=3$ cluster assignments labeled by normalized Top-20 count dominant network.
   - **Stacked Bar Panels**: Left and Right hippocampus horizontal stacked bars showing percentage composition ($0\% - 100\%$) of normalized Top-20 connected cortical ROIs for Anterior, Middle, and Posterior subregions.
   - **Normalization**: Normalized by available network ROI capacity (canonical 7: DMN=77, CON=73, VN=61, VAN=52, DAN=51, SMN=68, LAN=18).

3. **Modality 3: Winner-Take-All (WTA) Parcellation — 20 Long-Axis Bins Network Composition (%)**:
   - **Visual Style**: Aligned with `multidataset_wta_normalized_longaxis_stacked_2col`.
   - **Hippocampus 3D Surface**: 4-view locked Workbench composite showing canonical 7-network WTA vertex assignments.
   - **Stacked Bar Panels**: Left and Right hippocampus 20 horizontal stacked bars ordered along the anterior-to-posterior longitudinal axis (Bin 1 Anterior at top, Bin 20 Posterior at bottom), showing vertex percentage ($0\% - 100\%$) belonging to each network.

---

## 3. The 36P Grid Outputs

The 36P grid uses smoothing resolutions $2\text{mm}$ and $4\text{mm}$:

| Grid Cell | Resolution | Preprocessing Denoising | Output Image | Status |
|---|---|---|---|---|
| **`2mm_36p`** | 2mm | ICA-FIX Reference Baseline | `hcp_cortex_control_gradient_wta_2mm_36p.png` | **Materialized** (10780 × 3960 px, 220 DPI) |
| **`4mm_36p`** | 4mm | ICA-FIX Reference Baseline | `hcp_cortex_control_gradient_wta_4mm_36p.png` | **Materialized** (10780 × 3960 px, 220 DPI) |

> [!NOTE]
> Canvas geometry is $49.0 \times 18.0\text{ inches}$ ($10,780 \times 3,960\text{ px}$ at 220 DPI). The layout preserves the exact $1.00 : 1.00$ height ratio between the locked 2x2 hippocampal surfaces and stacked bars. It eliminates all left-side text collisions: in Column 1 (Fisher-$z$), `C1 (Ant)`, `C2 (Mid)`, `C3 (Post)` labels and the cohort badges maintain $+26\text{ px}$ of clear separation; in Column 3 (WTA), a dedicated spacer column provides $+48\text{ px}$ of clearance between the 2x2 surface and `Long-Axis Bin` y-axis labels.

---

## 4. Execution & CLI Options

Use the validated repository Python environment:

```bash
# Render single grid
/home/yojiao/.conda/envs/py314/bin/python scripts/visualize/hcp_cortex_control_gradient_wta/render_hcp_cortex_control_gradient_wta.py --grid 2mm_36p

# Render both grids in sequence
/home/yojiao/.conda/envs/py314/bin/python scripts/visualize/hcp_cortex_control_gradient_wta/render_hcp_cortex_control_gradient_wta.py --all-grids
```

### CLI Arguments
- `--smoothing {2mm,4mm}`: Select hippocampal smoothing resolution.
- `--variant {36p}`: Select grid variant tag.
- `--grid {2mm_36p,4mm_36p}`: Shorthand grid specifier.
- `--all-grids`: Batch execute both grid combinations.
- `--force`: Force re-rendering of surface panels and composite figures even if cache is valid.
- `--out-png <path>`: Custom output image destination.
- `--stats-tsv <path>`: Custom output statistics TSV destination.

---

## 5. Artifact Provenance & Caching Integrity

Surface panels are cached under `panels/` with strict SHA-256 `.sources.json` provenance sidecars verifying:
- Upstream surface paths (`template_left_or_right_surface`)
- Label GIFTI paths (`template_functional_atlas_label` and `template_functional_atlas_label_variant`)
- Label GIFTI file contents via SHA-256 (`left_label_sha256`, `right_label_sha256`)
- Specification file (`template_spec_path`)
- Locked view scene (`config/wb_locked_native_view_lateral_medial.scene`)
- Output PNG SHA-256 hash

If upstream labels or manifests change (e.g. following pipeline recalculation of `routea_ctxdrop` or `routea_ctxreg`), the cache is automatically invalidated and panels are cleanly re-rendered via Connectome Workbench. Passing `--force` overrides the cache unconditionally.
