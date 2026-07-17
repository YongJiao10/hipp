# Hipp Parcellation Step-by-Step Flow (Canonical)

This is the execution-ordered flow for active workflows. The first numeric field is the hard pipeline layer: outputs from layer `NN` are prerequisites for layer `NN+1`. The second numeric field is for parallel branch work or branch-specific products within that layer.

## Canonical HCP Shared-Input Paths

The maintained Kong2022 branch names are cohort-independent: `routea`,
`surface`, `routeb`, `routea_ctxreg`, and `routea_ctxdrop`. For the one-time
2026-07-11 migration, first audit and then apply:

```bash
/home/yojiao/.conda/envs/py314/bin/python scripts/core_workflow/migrate_hcp_shared_input_paths.py --repo-root /home/yojiao/hpchome/hipp --dry-run
/home/yojiao/.conda/envs/py314/bin/python scripts/core_workflow/migrate_hcp_shared_input_paths.py --repo-root /home/yojiao/hpchome/hipp --apply
```

The migration performs same-filesystem directory renames and does not create
legacy symlinks. Historical runtime provenance remains immutable because it
records the paths used when those analyses ran; current code and all new
outputs use canonical paths. The temporary old-to-new runtime mapping manifest
was removed after the migration at the user's direction.

## 1. HippUnfold Runtime and Validation

1. `01_01_run_hippunfold_t1w_local.py`
2. `01_02_check_hippunfold_outputs.py`

## 2. Group Inputs and QC Cohort

1. `02_01_prepare_group_inputs.py`
2. `02_02_build_qc_manifest.py`
3. `02_03_build_lynch2024_nsi_manifest.py` for the Lynch2024 NSI-filtered comparison cohort

`02_01` derives shared surfaces, tSNR summaries, and group-average templates from validated HippUnfold outputs. `02_02` combines HippUnfold Dice and tSNR gate summaries into the pinned QC-pass cohort manifest. `02_03` filters that base cohort with an NSI TSV/CSV containing `subject,nsi`, normalizes `sub-` prefixes, applies `nsi >= 0.43`, writes pass/excluded/run manifests under `outputs_hcp/inputs/hcp/lynch2024_fast_pfm_t1only/manifests/`, and fails by default unless the pass count is 78.

```bash
/home/yojiao/.conda/envs/py314/bin/python scripts/core_workflow/02_03_build_lynch2024_nsi_manifest.py --base-subjects-manifest outputs_hcp/inputs/hcp/kong2022/manifests/hcp_7t_hippocampus_fmri_qc_pass_t1only_latest.txt --nsi-tsv <path/to/nsi.tsv> --nsi-threshold 0.43 --expected-count 78
```

## 3. Cortex References

- `03_01_build_kong2022_reference.py` for shared Kong2022 branch usage
- RouteA ArealMSHBM individualized cortex label generation on Argon; this is an external-backend artifact prerequisite documented in `core_workflow_artifact_lineage.md`, not a local numbered entrypoint
- `03_02_build_kong2022_routea_reference.py` for RouteA-specific reference source, including compact 17-net summaries used by `kong2022_routea_17net`, consuming the generated RouteA label files above
- `03_04_run_lynch2024_fastans_pfm.py` for official FASTANS Lynch2024 hard-label PFM generation from local FASTANS priors and Workbench
- `03_03_build_lynch2024_reference.py` for hard-label Lynch2024 component references, exact Lynch20 summaries containing 20 unique networks after excluding `Noise`, and canonical8 merged summaries from `config/cross_atlas_network_merge.json`

```bash
/home/yojiao/.conda/envs/py314/bin/python scripts/core_workflow/03_04_run_lynch2024_fastans_pfm.py --subjects-manifest outputs_hcp/inputs/hcp/lynch2024_fast_pfm_t1only/manifests/hcp_7t_lynch2024_nsi043_pass_78.txt --wb-command /home/yojiao/FASTANS/workbench/bin_linux64/wb_command --jobs 6
/home/yojiao/.conda/envs/py314/bin/python scripts/core_workflow/03_03_build_lynch2024_reference.py --shared-root outputs_hcp/inputs/hcp/lynch2024_fast_pfm_t1only/shared_inputs/pfm_nsi043_78 --subjects-manifest outputs_hcp/inputs/hcp/lynch2024_fast_pfm_t1only/manifests/hcp_7t_lynch2024_nsi043_pass_78.txt --cortex-pfm-root outputs_hcp/work/cortex_pfm --jobs 8
```

## 4. Hippocampal Time-Series Control and FC Materialization

1. `04_00_materialize_cortex_regressed_surface_timeseries.py` for the isolated `kong2022_routea_ctxreg` branch
2. `04_01_materialize_kong2022_roi_fc.py --ctxdrop` for the isolated `kong2022_routea_ctxdrop` branch
3. `04_01_materialize_kong2022_roi_fc.py`
4. `04_01_materialize_kong2022_roi_fc.py --atlas lynch2024_20net` and `--atlas lynch2024_8net` for Lynch2024 ROI/component FC caches
5. `04_02_materialize_intrinsic_fc.py`
6. `04_03_materialize_kong2019_fc.py` (Kong2019 branch only)

`04_00` regresses local valid cortex signal from hippocampal time series within each run segment and writes only controlled time series plus diagnostics under `shared_inputs/routea_ctxreg`. RouteA references and `surface` masks are not copied; downstream `04_01` and `04_02` write FC into the controlled root while reading those static inputs via `--reference-root` and `--mask-root`, so FC filenames remain unchanged but isolated from RouteA.

`04_01 --ctxdrop` keeps hippocampal time series unchanged, reloads cortex dtseries plus RouteA individualized Kong2022 labels, and for each hippocampal vertex excludes valid cortex vertices within 20 mm before computing the target ROI mean time series. It writes standard Kong2022 FC cache filenames and stores smoothing-independent local cortex-neighbor diagnostics under `shared_inputs/routea_ctxdrop/sub-*/surface/local_cortex_neighbors/hemi_*/`. The maintained ctxdrop downstream scope is `gradient-spectral` and `wta-top`.

Lynch2024 FC materialization reuses the existing `04_01` implementation with `--atlas`; run it once per Lynch atlas slug so downstream route validators never fall back to Kong2022 paths:

```bash
/home/yojiao/.conda/envs/py314/bin/python scripts/core_workflow/04_01_materialize_kong2022_roi_fc.py --shared-root outputs_hcp/inputs/hcp/lynch2024_fast_pfm_t1only/shared_inputs/pfm_nsi043_78 --surface-root outputs_hcp/inputs/hcp/kong2022/shared_inputs/surface --reference-root outputs_hcp/inputs/hcp/lynch2024_fast_pfm_t1only/shared_inputs/pfm_nsi043_78 --subjects-manifest outputs_hcp/inputs/hcp/lynch2024_fast_pfm_t1only/manifests/hcp_7t_lynch2024_nsi043_pass_78.txt --atlas lynch2024_20net
/home/yojiao/.conda/envs/py314/bin/python scripts/core_workflow/04_01_materialize_kong2022_roi_fc.py --shared-root outputs_hcp/inputs/hcp/lynch2024_fast_pfm_t1only/shared_inputs/pfm_nsi043_78 --surface-root outputs_hcp/inputs/hcp/kong2022/shared_inputs/surface --reference-root outputs_hcp/inputs/hcp/lynch2024_fast_pfm_t1only/shared_inputs/pfm_nsi043_78 --subjects-manifest outputs_hcp/inputs/hcp/lynch2024_fast_pfm_t1only/manifests/hcp_7t_lynch2024_nsi043_pass_78.txt --atlas lynch2024_8net
```

## 5. Group Atlas Construction

1. `05_01_compute_hippomaps_gradients.py` for Kong2022 and Lynch2024 gradient artifacts
2. `05_02_run_group_atlas.py` for group atlas outputs and K=2..8 diagnostics with fixed final K=3
3. `05_03_run_spectral_fast_rerun.py` for optional rerun orchestration after the primary atlas step
4. `05_04_compute_confidence_maps.py` for vertex-level silhouette confidence maps in G1/G2/G3 feature space

Kong2022 HippoMaps gradient artifact example. The default `--gradient-random-state 0` fixes BrainSpace diffusion initialization and is recorded in each `05_01` manifest:

```bash
/home/yojiao/.conda/envs/py314/bin/python scripts/core_workflow/05_01_compute_hippomaps_gradients.py --routes routea routea_ctxreg routea_ctxdrop routeb --smoothings 2mm 4mm --hemis L R --n-components 5 --gradient-random-state 0 --selected-components 1,2,3
```

Kong2022 RouteA cortex-control group clustering example:

```bash
/home/yojiao/.conda/envs/py314/bin/python scripts/core_workflow/05_02_run_group_atlas.py --shared-root outputs_hcp/inputs/hcp/kong2022/shared_inputs/routea_ctxreg --surface-root outputs_hcp/inputs/hcp/kong2022/shared_inputs/surface --reference-root outputs_hcp/inputs/hcp/kong2022/shared_inputs/routea --out-root outputs_hcp/work/group_atlas/routea_ctxreg_full_k3_fisherz --methods intrinsic-spectral network-spectral gradient-spectral wta-top --atlases kong2022 --gradient-root outputs_hcp/work/hippomaps_gradients --route-tag routea_ctxreg --k-min 2 --k-max 8 --default-k 3
/home/yojiao/.conda/envs/py314/bin/python scripts/core_workflow/05_02_run_group_atlas.py --shared-root outputs_hcp/inputs/hcp/kong2022/shared_inputs/routea_ctxdrop --surface-root outputs_hcp/inputs/hcp/kong2022/shared_inputs/surface --reference-root outputs_hcp/inputs/hcp/kong2022/shared_inputs/routea --out-root outputs_hcp/work/group_atlas/routea_ctxdrop_full_k3_fisherz --methods gradient-spectral wta-top --atlases kong2022 --gradient-root outputs_hcp/work/hippomaps_gradients --route-tag routea_ctxdrop --k-min 2 --k-max 8 --default-k 3
/home/yojiao/.conda/envs/py314/bin/python scripts/core_workflow/05_02_run_group_atlas.py --shared-root outputs_hcp/inputs/hcp/kong2022/shared_inputs/routeb --surface-root outputs_hcp/inputs/hcp/kong2022/shared_inputs/surface --out-root outputs_hcp/work/group_atlas/routeb_full_k3_fisherz --methods intrinsic-spectral network-spectral gradient-spectral wta-top --atlases kong2022 --gradient-root outputs_hcp/work/hippomaps_gradients --route-tag routeb --k-min 2 --k-max 8 --default-k 3
/home/yojiao/.conda/envs/py314/bin/python scripts/core_workflow/05_02_run_group_atlas.py --shared-root outputs_hcp/inputs/hcp/kong2022/shared_inputs/routea --surface-root outputs_hcp/inputs/hcp/kong2022/shared_inputs/surface --reference-root outputs_hcp/inputs/hcp/kong2022/shared_inputs/routea --out-root outputs_hcp/work/group_atlas/routea_17net_full_k3_fisherz --methods gradient-spectral wta-top --atlases kong2022 --gradient-root outputs_hcp/work/hippomaps_gradients --route-tag routea_17net --network-taxonomy family17 --k-min 2 --k-max 8 --default-k 3
/home/yojiao/.conda/envs/py314/bin/python scripts/core_workflow/06_07_render_ctxdrop_neighbor_counts.py --shared-root outputs_hcp/inputs/hcp/kong2022/shared_inputs/routea_ctxdrop --source-branch routea_ctxdrop
```

Lynch2024 group-atlas commands use the NSI-filtered shared root, route-specific atlas slug, and matching taxonomy. `05_01` generates one shared Lynch20 non-Noise component gradient bank with BrainSpace full-graph affinity (`brainspace_sparsity=0.0`); `lynch2024_8net` gradient-spectral reads that bank while WTA reads canonical8 references. Maintained methods are `gradient-spectral` and `wta-top`; WTA defaults to normalized top20-count labeling. Lynch gradient diagnostics evaluate K=2..8; `05_02` uses shifted-cosine surface affinity and the same full-sample cluster-to-network FC redundancy metric as Kong2022. K=3 is the explicit primary default, while automatic K selection remains recorded separately.

```bash
/home/yojiao/.conda/envs/py314/bin/python scripts/core_workflow/05_01_compute_hippomaps_gradients.py --routes lynch2024_20net --shared-inputs-root outputs_hcp/inputs/hcp/lynch2024_fast_pfm_t1only/shared_inputs --surface-root outputs_hcp/inputs/hcp/lynch2024_fast_pfm_t1only/shared_inputs/pfm_nsi043_78 --subjects-manifest outputs_hcp/inputs/hcp/lynch2024_fast_pfm_t1only/manifests/hcp_7t_lynch2024_nsi043_pass_78.txt --smoothings 2mm 4mm --hemis L R --group-average-surfaces-dir outputs_hcp/inputs/hcp/kong2022/templates/hipp_group_average_t1only_151/group_average_surfaces --n-components 5 --gradient-random-state 0 --selected-components 1,2,3 --force
/home/yojiao/.conda/envs/py314/bin/python scripts/core_workflow/05_02_run_group_atlas.py --shared-root outputs_hcp/inputs/hcp/lynch2024_fast_pfm_t1only/shared_inputs/pfm_nsi043_78 --surface-root outputs_hcp/inputs/hcp/lynch2024_fast_pfm_t1only/shared_inputs/pfm_nsi043_78 --profile-surface-root outputs_hcp/inputs/hcp/kong2022/shared_inputs/routea_ctxreg --reference-root outputs_hcp/inputs/hcp/lynch2024_fast_pfm_t1only/shared_inputs/pfm_nsi043_78 --out-root outputs_hcp/work/group_atlas/lynch2024_20net_nsi043_full_k3_normalized_wta --methods gradient-spectral wta-top --atlases lynch2024_20net --gradient-root outputs_hcp/work/hippomaps_gradients --route-tag lynch2024_20net --network-taxonomy lynch20 --subjects-manifest outputs_hcp/inputs/hcp/lynch2024_fast_pfm_t1only/manifests/hcp_7t_lynch2024_nsi043_pass_78.txt --k-min 2 --k-max 8 --default-k 3 --jobs 1 --group-average-surfaces-dir outputs_hcp/inputs/hcp/kong2022/templates/hipp_group_average_t1only_151/group_average_surfaces
/home/yojiao/.conda/envs/py314/bin/python scripts/core_workflow/05_02_run_group_atlas.py --shared-root outputs_hcp/inputs/hcp/lynch2024_fast_pfm_t1only/shared_inputs/pfm_nsi043_78 --surface-root outputs_hcp/inputs/hcp/lynch2024_fast_pfm_t1only/shared_inputs/pfm_nsi043_78 --profile-surface-root outputs_hcp/inputs/hcp/kong2022/shared_inputs/routea_ctxreg --reference-root outputs_hcp/inputs/hcp/lynch2024_fast_pfm_t1only/shared_inputs/pfm_nsi043_78 --out-root outputs_hcp/work/group_atlas/lynch2024_8net_nsi043_full_k3_normalized_wta --methods gradient-spectral wta-top --atlases lynch2024_8net --gradient-root outputs_hcp/work/hippomaps_gradients --route-tag lynch2024_8net --network-taxonomy lynch8 --subjects-manifest outputs_hcp/inputs/hcp/lynch2024_fast_pfm_t1only/manifests/hcp_7t_lynch2024_nsi043_pass_78.txt --k-min 2 --k-max 8 --default-k 3 --jobs 1 --group-average-surfaces-dir outputs_hcp/inputs/hcp/kong2022/templates/hipp_group_average_t1only_151/group_average_surfaces
```

For all ROI-registry `wta-top` runs, normalized top20-count is the default assignment. Normalized means top-20 positive ROI/component counts divided by finite available ROI/component counts in each network, not Fisher-z averaging. Lynch gradient overviews use the manifest default `fisherz` cluster labels; Lynch WTA overviews use `normalized_top20_count`. Render them from their isolated group roots with stable filenames containing `lynch2024_20net` or `lynch2024_8net`; the visible gradient title identifies shifted-cosine and the manifest records `surface_affinity`, `affinity_description`, and `source_group_manifests`:

```bash
/home/yojiao/.conda/envs/py314/bin/python scripts/core_workflow/06_01_build_group_overview.py --group-root outputs_hcp/work/group_atlas/lynch2024_20net_nsi043_full_k3_normalized_wta --route-tag lynch2024_20net --methods gradient-spectral --atlases lynch2024_20net --assignment-variant fisherz
/home/yojiao/.conda/envs/py314/bin/python scripts/core_workflow/06_01_build_group_overview.py --group-root outputs_hcp/work/group_atlas/lynch2024_20net_nsi043_full_k3_normalized_wta --route-tag lynch2024_20net --methods wta-top --atlases lynch2024_20net --assignment-variant normalized_top20_count
/home/yojiao/.conda/envs/py314/bin/python scripts/core_workflow/06_01_build_group_overview.py --group-root outputs_hcp/work/group_atlas/lynch2024_8net_nsi043_full_k3_normalized_wta --route-tag lynch2024_8net --methods gradient-spectral --atlases lynch2024_8net --assignment-variant fisherz
/home/yojiao/.conda/envs/py314/bin/python scripts/core_workflow/06_01_build_group_overview.py --group-root outputs_hcp/work/group_atlas/lynch2024_8net_nsi043_full_k3_normalized_wta --route-tag lynch2024_8net --methods wta-top --atlases lynch2024_8net --assignment-variant normalized_top20_count
```

## Formal Gradient-Spectral Release

The formal release comprises exactly nine audited groups: five HCP groups
(RouteA canonical8, RouteA family17, RouteA weighted-fisherz, Lynch20, and
Lynch8) and four Stanford groups (HC, Symp, and both weighted-fisherz
derivatives). Audit the complete set without modifying it:

```bash
/home/yojiao/.conda/envs/py314/bin/python scripts/core_workflow/audit_formal_gradient_spectral_release.py --repo-root "$PWD"
```

The supported refresh orchestration is a fixed 19-stage, stop-on-first-failure
worker. Its launch path accepts only one job and preflights exact input leaves
before detaching:

```bash
/home/yojiao/.conda/envs/py314/bin/python scripts/core_workflow/run_formal_gradient_spectral_refresh.py --launch --repo-root "$PWD" --jobs 1
```

That production command is not part of repository verification and must not be
run without a separate request. Likewise, do not move or archive products under
the misnamed taxonomy until the exact inventory and destination have been shown
to the user and explicitly confirmed.

## 6. Presentation Products

These products are derived from group atlas and gradient work and can be refreshed from layer 5 products.

1. `06_01_build_group_overview.py`
2. `06_02_materialize_group_labels.py`
3. `06_03_build_cluster_cortex_panels.py`
4. `06_04_render_hippomaps_gradient_products.py` (defaults to RouteA-referenced display sign alignment and shared per-component color ranges across rendered gradient branches)
5. `06_05_render_group_gradient_heatmaps.py` (deprecated retired; exits with guidance to use `06_04` route outputs)
6. `06_06_render_confidence_maps.py` for silhouette confidence map rendering on folded/unfolded surfaces
7. `06_07_render_ctxdrop_neighbor_counts.py` for shared ctxreg/ctxdrop local cortex-neighbor overlays
8. `06_08_render_hcp_subfield_composition.py` for subfield composition plots
9. `06_09_plot_cortex_kong2022_mshbm.py` for subject-level Kong2022 MSHBM cortical parcellation maps

## 7. Presentation Index

1. `07_01_build_group_matrix.py`

`07_01` reads refreshed `06_01` method overviews from canonical group-atlas roots, including `routea_17net`, `routea_ctxreg`, `routea_ctxdrop`, and `routeb` when requested, then writes `outputs_hcp/present/overviews/group_overview_matrix_manifest.json` and the overview README. The WTA-top overview itself contains the fixed 20-bin transposed long-axis network profile; standalone WTA profile code and PNG/TSV products are removed from the maintained workflow. Use `--methods` to restrict exports to completed methods (for example `gradient-spectral wta-top`).

## Utilities

- `tests/core_workflow/smoke_prepare_run.py` prepares isolated smoke-run command plans under `outputs_hcp/smoke_runs/<run_id>/`. It is not part of the numbered pipeline.

Maintained active output roots:

- Work: `outputs_hcp/work/`
- Lynch2024 inputs: `outputs_hcp/inputs/hcp/lynch2024_fast_pfm_t1only/`
- Lynch2024 cortex PFM work: `outputs_hcp/work/cortex_pfm/`
- Presentation: `outputs_hcp/present/`
- Smoke tests: `outputs_hcp/smoke_runs/<run_id>/`
- Non-official reruns: `outputs_hcp/reruns/<run_id>/`

For branch flags, path roots, and exact artifact contracts, use:

- `core_workflow_step_registry.md`
- `core_workflow_artifact_lineage.md`
- `scripts/core_workflow/step_registry.json`
