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
- RouteA ArealMSHBM individualized cortex label generation on Argon; this is an external-backend artifact prerequisite documented in `core_workflow_artifact_lineage.md`, not a local numbered entrypoint
- `03_02_build_kong2022_routea_reference.py` for RouteA-specific reference source, including compact 17-net summaries used by `kong2022_routea_17net`, consuming the generated RouteA label files above

## 4. Hippocampal Time-Series Control and FC Materialization

1. `04_00_materialize_cortex_regressed_surface_timeseries.py` for the isolated `kong2022_routea_ctxreg` branch
2. `04_01_materialize_kong2022_roi_fc.py --ctxdrop` for the isolated `kong2022_routea_ctxdrop` branch
3. `04_01_materialize_kong2022_roi_fc.py`
4. `04_02_materialize_intrinsic_fc.py`
5. `04_03_materialize_kong2019_fc.py` (Kong2019 branch only)

`04_00` regresses local valid cortex signal from hippocampal time series within each run segment and writes only controlled time series plus diagnostics under `shared_inputs/routea_ctxreg_151`. RouteA references and `surface` masks are not copied; downstream `04_01` and `04_02` write FC into the controlled root while reading those static inputs via `--reference-root` and `--mask-root`, so FC filenames remain unchanged but isolated from RouteA.

`04_01 --ctxdrop` keeps hippocampal time series unchanged, reloads cortex dtseries plus RouteA individualized Kong2022 labels, and for each hippocampal vertex excludes valid cortex vertices within 20 mm before computing the target ROI mean time series. It writes standard Kong2022 FC cache filenames and stores smoothing-independent local cortex-neighbor diagnostics under `shared_inputs/routea_ctxdrop_151/sub-*/surface/local_cortex_neighbors/hemi_*/`. The maintained ctxdrop downstream scope is `gradient-spectral` and `wta-top`.

## 5. Group Atlas Construction

1. `05_01_compute_hippomaps_gradients.py` for Kong2022 gradient artifacts
2. `05_02_run_group_atlas.py` for group atlas outputs and K=2..10 diagnostics with fixed final K=3
3. `05_03_run_spectral_fast_rerun.py` for optional rerun orchestration after the primary atlas step
4. `05_04_compute_confidence_maps.py` for vertex-level silhouette confidence maps in G1/G2/G3 feature space

Kong2022 HippoMaps gradient artifact example. The default `--gradient-random-state 0` fixes BrainSpace diffusion initialization and is recorded in each `05_01` manifest:

```bash
/home/yojiao/.conda/envs/py314/bin/python scripts/core_workflow/05_01_compute_hippomaps_gradients.py --routes routea routea_ctxreg routea_ctxdrop routeb --smoothings 2mm 4mm --hemis L R --n-components 5 --gradient-random-state 0 --selected-components 1,2,3
```

Kong2022 RouteA cortex-control group clustering example:

```bash
/home/yojiao/.conda/envs/py314/bin/python scripts/core_workflow/05_02_run_group_atlas.py --shared-root outputs_hcp/inputs/hcp/kong2022/shared_inputs/routea_ctxreg_151 --surface-root outputs_hcp/inputs/hcp/kong2022/shared_inputs/surface --reference-root outputs_hcp/inputs/hcp/kong2022/shared_inputs/routea --out-root outputs_hcp/work/group_atlas/routea_ctxreg_full_k3_fisherz --methods intrinsic-spectral network-spectral gradient-spectral wta-top --atlases kong2022 --gradient-root outputs_hcp/work/hippomaps_gradients --route-tag routea_ctxreg --k-min 2 --k-max 10 --force-final-k 3
/home/yojiao/.conda/envs/py314/bin/python scripts/core_workflow/05_02_run_group_atlas.py --shared-root outputs_hcp/inputs/hcp/kong2022/shared_inputs/routea_ctxdrop_151 --surface-root outputs_hcp/inputs/hcp/kong2022/shared_inputs/surface --reference-root outputs_hcp/inputs/hcp/kong2022/shared_inputs/routea --out-root outputs_hcp/work/group_atlas/routea_ctxdrop_full_k3_fisherz --methods gradient-spectral wta-top --atlases kong2022 --gradient-root outputs_hcp/work/hippomaps_gradients --route-tag routea_ctxdrop --k-min 2 --k-max 10 --force-final-k 3
/home/yojiao/.conda/envs/py314/bin/python scripts/core_workflow/05_02_run_group_atlas.py --shared-root outputs_hcp/inputs/hcp/kong2022/shared_inputs/routeb_151 --surface-root outputs_hcp/inputs/hcp/kong2022/shared_inputs/surface --out-root outputs_hcp/work/group_atlas/routeb_full_k3_fisherz --methods intrinsic-spectral network-spectral gradient-spectral wta-top --atlases kong2022 --gradient-root outputs_hcp/work/hippomaps_gradients --route-tag routeb --k-min 2 --k-max 10 --force-final-k 3
/home/yojiao/.conda/envs/py314/bin/python scripts/core_workflow/05_02_run_group_atlas.py --shared-root outputs_hcp/inputs/hcp/kong2022/shared_inputs/routea --surface-root outputs_hcp/inputs/hcp/kong2022/shared_inputs/surface --reference-root outputs_hcp/inputs/hcp/kong2022/shared_inputs/routea --out-root outputs_hcp/work/group_atlas/routea_17net_full_k3_fisherz --methods gradient-spectral wta-top --atlases kong2022 --gradient-root outputs_hcp/work/hippomaps_gradients --route-tag routea_17net --network-taxonomy family17 --k-min 2 --k-max 10 --force-final-k 3
/home/yojiao/.conda/envs/py314/bin/python scripts/core_workflow/06_07_render_ctxdrop_neighbor_counts.py --shared-root outputs_hcp/inputs/hcp/kong2022/shared_inputs/routea_ctxdrop_151 --source-branch routea_ctxdrop
```

WTA normalized top20-count comparison branches can be generated after the canonical RouteA and RouteA 17-net WTA roots exist. These runs preserve the canonical WTA roots and write isolated comparison roots. Normalized means top-20 positive ROI counts divided by finite available ROI counts in each network, not Fisher-z averaging. The maintained HCP roots and overview directories are:

- `outputs_hcp/work/group_atlas/routea_wta_normalized_top20_count_full_k3/` rendered as `outputs_hcp/present/overviews/wta_normalized/top20_count.png`
- `outputs_hcp/work/group_atlas/routea_17net_wta_normalized_top20_count_full_k3/` rendered as `outputs_hcp/present/overviews/wta_normalized/family17.png`

## 6. Presentation Products

These products are derived from group atlas and gradient work and can be refreshed from layer 5 products.

1. `06_01_build_group_overview.py`
2. `06_02_materialize_group_labels.py`
3. `06_03_build_cluster_cortex_panels.py`
4. `06_04_render_hippomaps_gradient_products.py` (defaults to RouteA-referenced display sign alignment and shared per-component color ranges across rendered gradient branches)
5. `06_05_render_group_gradient_heatmaps.py` (deprecated retired; exits with guidance to use `06_04` route outputs)
6. `06_06_render_confidence_maps.py` for silhouette confidence map rendering on folded/unfolded surfaces
7. `06_07_render_ctxdrop_neighbor_counts.py` for shared ctxreg/ctxdrop local cortex-neighbor overlays

## 7. Presentation Index

1. `07_01_build_group_matrix.py`

`07_01` reads refreshed `06_01` method overviews from canonical group-atlas roots, including `routea_17net`, `routea_ctxreg`, `routea_ctxdrop`, and `routeb` when requested, then writes `outputs_hcp/present/overviews/group_overview_matrix_manifest.json` and the overview README. The WTA-top overview itself contains the fixed 20-bin transposed long-axis network profile; standalone WTA profile code and PNG/TSV products are removed from the maintained workflow. Use `--methods` to restrict exports to completed methods (for example `gradient-spectral wta-top`).

## Utilities

- `tests/core_workflow/smoke_prepare_run.py` prepares isolated smoke-run command plans under `outputs_hcp/smoke_runs/<run_id>/`. It is not part of the numbered pipeline.

Maintained active output roots:

- Work: `outputs_hcp/work/`
- Presentation: `outputs_hcp/present/`
- Smoke tests: `outputs_hcp/smoke_runs/<run_id>/`
- Non-official reruns: `outputs_hcp/reruns/<run_id>/`

For branch flags, path roots, and exact artifact contracts, use:

- `core_workflow_step_registry.md`
- `core_workflow_artifact_lineage.md`
- `scripts/core_workflow/step_registry.json`
