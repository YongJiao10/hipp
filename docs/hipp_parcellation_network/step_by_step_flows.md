# Hipp Parcellation Step-by-Step Flow (Canonical)

This is the execution-ordered flow for active workflows. The first numeric field is the hard pipeline layer: outputs from layer `NN` are prerequisites for layer `NN+1`. The second numeric field is for parallel branch work or branch-specific products within that layer.

## 1. HippUnfold Runtime and Validation

1. `01_01_run_hippunfold_t1w_local.py`
2. `01_02_check_hippunfold_outputs.py`

## 2. Group Inputs and QC Cohort

1. `02_01_prepare_group_inputs.py`
2. `02_02_build_qc_manifest.py`

`02_01` derives shared surfaces, tSNR summaries, and group-average templates from validated HippUnfold outputs. `02_02` combines HippUnfold Dice and tSNR gate summaries into the pinned QC-pass cohort manifest.

## 3. Cortex References

- `03_01_build_kong2022_reference.py` for shared Kong2022 branch usage
- `03_02_build_kong2022_routea_reference.py` for RouteA-specific reference source

## 4. Hippocampal FC Materialization

1. `04_01_materialize_kong2022_roi_fc.py`
2. `04_02_materialize_intrinsic_fc.py`
3. `04_03_materialize_kong2019_fc.py` (Kong2019 branch only)

## 5. Group Atlas Construction

1. `05_01_compute_hippomaps_gradients.py` for Kong2022 gradient artifacts
2. `05_02_run_group_atlas.py` for group atlas outputs and K=2..10 diagnostics with fixed final K=3
3. `05_03_run_spectral_fast_rerun.py` for optional rerun orchestration after the primary atlas step
4. `05_04_compute_confidence_maps.py` for vertex-level silhouette confidence maps in G1/G2/G3 feature space

Kong2022 HippoMaps gradient artifact example:

```bash
/home/yojiao/.conda/envs/py314/bin/python scripts/core_workflow/05_01_compute_hippomaps_gradients.py   --routes routea routeb   --smoothings 2mm 4mm   --hemis L R   --n-components 5   --selected-components 1,2,3
```

Kong2022 group-gradient clustering example:

```bash
/home/yojiao/.conda/envs/py314/bin/python scripts/core_workflow/05_02_run_group_atlas.py   --methods intrinsic-spectral network-spectral gradient-spectral wta-top   --atlases kong2022   --gradient-root outputs_core/work/hippomaps_gradients   --route-tag routea   --k-min 2 --k-max 10 --force-final-k 3
```

`05_01` computes raw G1-G5 from the column-wise z-scored group mean Fisher-z vertex-to-Kong2022 ROI FC matrix using `GradientMaps(n_components=N, kernel="pearson")`, aligns component order and sign within each route to `4mm/hemi_L` when present, and writes only aligned artifacts. `05_02` loads those aligned artifacts for `gradient-spectral` and selects `G1,G2,G3` by default. The `--gradient-components` argument accepts any ordered non-empty subset of `{1,2,3}` (e.g. `--gradient-components 1` for G1-only clustering). Output paths for `gradient-spectral` encode the selected components as a `g{components}` subdirectory (e.g. `gradient-spectral/kong2022/2mm/cov_0p80/g1g2g3/hemi_L/`) so that different component selections do not overwrite each other. The legacy auto-K logic remains behind `--auto-k`, but the maintained default is full K-curve evaluation (`K=2..10`) with fixed final `K=3`.

## 6. Presentation Products

These are presentation products derived from group atlas and gradient work. They are in the same hard layer because they can be refreshed from layer 5 products.

1. `06_01_build_group_overview.py`
2. `06_02_build_wta_long_axis_profiles.py`
3. `06_03_materialize_group_labels.py`
4. `06_04_build_cluster_cortex_panels.py`
5. `06_05_render_hippomaps_gradient_products.py`
6. `06_06_render_group_gradient_heatmaps.py` (deprecated retired; exits with guidance to use `06_05` routea/routeb outputs)
7. `06_07_render_confidence_maps.py` for silhouette confidence map rendering on folded/unfolded surfaces

## 7. Presentation Index

1. `07_01_build_group_matrix.py`

`07_01` reads refreshed `06_01` method overviews and `06_02` WTA long-axis profile products, then writes `outputs_core/present/overviews/group_overview_matrix_manifest.json` and the overview README.

## Utilities

- `smoke_prepare_run.py` prepares isolated smoke-run command plans under `outputs_core/smoke_runs/<run_id>/`. It is not part of the numbered pipeline.

Maintained active output roots:

- Work: `outputs_core/work/`
- Presentation: `outputs_core/present/`
- Smoke tests: `outputs_core/smoke_runs/<run_id>/`
- Non-official reruns: `outputs_core/reruns/<run_id>/`

For branch flags, path roots, and exact artifact contracts, use:

- `core_workflow_step_registry.md`
- `scripts/core_workflow/step_registry.json`
