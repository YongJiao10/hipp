# HCP 7T Hippocampus Gradient and Silhouette Confidence Map Replotting Design

**Date:** 2026-11-06
**Status:** Approved

## 1. Background & Objectives

Following the update to hippocampal functional QC (enforcing hippocampal coverage threshold of $\le 10$ unusable vertices per den-512 hemisphere matching Kember et al. 2026's $20\text{ mm}^2$ anatomical criterion), the latest authorized HCP 7T functional QC pass roster is `outputs_hcp/inputs/kong2022/manifests/hcp_7t_hippocampus_fmri_qc_pass_t1only_latest.txt` with $N=166$ subjects.

This design specifies the end-to-end execution of the core workflow stages to recompute:
1. HippoMaps continuous functional gradients (G1, G2, G3) for the primary PFM (Route A) branch under dual smoothing kernels (2 mm and 4 mm FWHM).
2. Group atlas spatial clustering (`gradient-spectral`, fixed $K=3$) and gradient feature extraction.
3. Vertex-level silhouette confidence maps in the G1–G3 feature space.
4. Final presentation figures:
   - Hippocampus continuous gradient surfaces, explained variance plots, and endpoint FC contrasts.
   - Hippocampus silhouette confidence maps on folded and unfolded surfaces.

All outputs will be placed under the standard formal output roots: `outputs_hcp/work/` and `outputs_hcp/present/gradients/pfm/`.

## 2. Input Contracts & Data Assets

- **Subject Roster:** `outputs_hcp/inputs/kong2022/manifests/hcp_7t_hippocampus_fmri_qc_pass_t1only_latest.txt` (166 subjects).
- **Subject FC Data:** `outputs_hcp/inputs/kong2022/shared_inputs/pfm/<sub-id>/fc/kong2022/<2mm|4mm>/hemi_<L|R>/hipp_vertex_to_roi_fc_active.npy`.
- **Group Average Surfaces:** `outputs_hcp/inputs/kong2022/templates/hipp_group_average/group_average_surfaces/group-average_hemi-<L|R>_space-corobl_den-512_label-hipp_midthickness.surf.gii`.
- **HippoMaps Package:** `external/HippoMaps`.

## 3. Workflow Pipeline & Execution Steps

All commands explicitly execute with `/home/yojiao/.conda/envs/py314/bin/python`.

### Step 1: Compute HippoMaps Continuous Gradients (`05_01`)

- **Script:** `scripts/core_workflow/05_01_compute_hippomaps_gradients.py`
- **Arguments:**
  - `--routes pfm`
  - `--smoothings 2mm 4mm`
  - `--hemis L R`
  - `--n-components 5`
  - `--selected-components 1,2,3`
  - `--gradient-random-state 0`
  - `--out-root outputs_hcp/work/hippomaps_gradients`
- **Output Artifacts:**
  - `outputs_hcp/work/hippomaps_gradients/pfm/<2mm|4mm>/hemi_<L|R>/hippomaps_gradients_active.npy`
  - `outputs_hcp/work/hippomaps_gradients/pfm/<2mm|4mm>/hemi_<L|R>/hippomaps_gradients_full.npy`
  - `outputs_hcp/work/hippomaps_gradients/pfm/<2mm|4mm>/hemi_<L|R>/hippomaps_ev_fractions.npy`
  - `outputs_hcp/work/hippomaps_gradients/pfm/<2mm|4mm>/hemi_<L|R>/hippomaps_gradient_manifest.json`

### Step 2: Build Group Atlas and Extract Gradient Features (`05_02`)

- **Script:** `scripts/core_workflow/05_02_run_group_atlas.py`
- **Arguments:**
  - `--shared-root outputs_hcp/inputs/kong2022/shared_inputs/pfm`
  - `--surface-root outputs_hcp/inputs/kong2022/shared_inputs/surface`
  - `--reference-root outputs_hcp/inputs/kong2022/shared_inputs/pfm`
  - `--out-root outputs_hcp/work/group_atlas/pfm`
  - `--methods gradient-spectral`
  - `--atlases kong2022`
  - `--gradient-root outputs_hcp/work/hippomaps_gradients`
  - `--route-tag pfm`
  - `--network-taxonomy canonical7`
  - `--smoothings 2mm 4mm`
  - `--k-min 2 --k-max 8 --default-k 3 --force-final-k 3`
- **Output Artifacts:**
  - `outputs_hcp/work/group_atlas/pfm/gradient-spectral/kong2022/<2mm|4mm>/cov_0p80/g1g2g3/hemi_<L|R>/group_gradient_features_active.npy`
  - `outputs_hcp/work/group_atlas/pfm/gradient-spectral/kong2022/<2mm|4mm>/cov_0p80/g1g2g3/hemi_<L|R>/group_labels_final.npy`
  - `outputs_hcp/work/group_atlas/pfm/gradient-spectral/kong2022/<2mm|4mm>/cov_0p80/g1g2g3/hemi_<L|R>/group_support_mask.npy`
  - `outputs_hcp/work/group_atlas/pfm/gradient-spectral/kong2022/<2mm|4mm>/cov_0p80/g1g2g3/hemi_<L|R>/group_atlas_manifest.json`

### Step 3: Compute Silhouette Confidence Maps (`05_04`)

- **Script:** `scripts/core_workflow/05_04_compute_confidence_maps.py`
- **Arguments:**
  - `--out-root outputs_hcp/work/group_atlas/pfm`
  - `--methods gradient-spectral`
  - `--smoothings 2mm 4mm`
  - `--hemis L R`
  - `--coverage-thresholds 0.80`
  - `--gradient-components 1,2,3`
  - `--metric cosine`
- **Output Artifacts:**
  - `outputs_hcp/work/group_atlas/pfm/gradient-spectral/kong2022/<2mm|4mm>/cov_0p80/g1g2g3/hemi_<L|R>/confidence_silhouette_gradient_feature.npy`
  - `outputs_hcp/work/group_atlas/pfm/gradient-spectral/kong2022/<2mm|4mm>/cov_0p80/g1g2g3/hemi_<L|R>/confidence_silhouette_gradient_feature_summary.json`
  - `outputs_hcp/work/group_atlas/pfm/gradient-spectral/kong2022/<2mm|4mm>/cov_0p80/g1g2g3/hemi_<L|R>/confidence_silhouette_gradient_feature_manifest.json`

### Step 4: Render Continuous Gradient Figures (`06_04`)

- **Script:** `scripts/core_workflow/06_04_render_hippomaps_gradient_products.py`
- **Arguments:**
  - `--gradient-root outputs_hcp/work/hippomaps_gradients`
  - `--present-root outputs_hcp/present/gradients`
  - `--routes pfm`
  - `--smoothings 2mm 4mm`
  - `--hemis L R`
  - `--components 1,2,3`
- **Output Artifacts:**
  - `outputs_hcp/present/gradients/pfm/<2mm|4mm>/hippomaps_gradients_pfm_<smoothing>_bilateral_g1g2g3.png`
  - `outputs_hcp/present/gradients/pfm/<2mm|4mm>/hippomaps_gradients_pfm_<smoothing>_bilateral_g1g2g3_hemi-<L|R>.png`
  - `outputs_hcp/present/gradients/pfm/<2mm|4mm>/hippomaps_ev_pfm_<smoothing>_bilateral.png`
  - `outputs_hcp/present/gradients/pfm/<2mm|4mm>/hippomaps_endpoint_fc_contrast_pfm_<smoothing>_bilateral.png`
  - `outputs_hcp/present/gradients/pfm/<2mm|4mm>/hippomaps_gradient_products_pfm_<smoothing>_bilateral_manifest.json`

### Step 5: Render Silhouette Confidence Figures (`06_06`)

- **Script:** `scripts/core_workflow/06_06_render_confidence_maps.py`
- **Arguments:**
  - `--out-root outputs_hcp/work/group_atlas/pfm`
  - `--present-root outputs_hcp/present/gradients`
  - `--route-tag pfm`
  - `--smoothings 2mm 4mm`
  - `--coverage-thresholds 0.80`
  - `--gradient-components 1,2,3`
  - `--method gradient-spectral`
- **Output Artifacts:**
  - `outputs_hcp/present/gradients/pfm/<2mm|4mm>/confidence_silhouette_gradient_feature_pfm_<smoothing>_bilateral.png`
  - `outputs_hcp/present/gradients/pfm/<2mm|4mm>/confidence_silhouette_gradient_feature_pfm_<smoothing>_bilateral_hemi-<L|R>.png`

## 4. Constraints & Execution Rules

1. **Python Environment:** Must strictly use `/home/yojiao/.conda/envs/py314/bin/python`.
2. **Concurrency Limit:** Up to 10 parallel processes total.
3. **Traceability:** Maintain manifest lineage for all intermediate and presentation stages.
4. **Verification:**
   - Confirm all expected `.png` and `.json` files exist and have non-zero file sizes.
   - Verify image dimensions and visual layout integrity.
   - Confirm mean silhouette values and explained variance fractions are valid and non-degenerate.
