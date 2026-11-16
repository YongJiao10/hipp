# Hipp Parcellation Step-by-Step Flow (Canonical)

This is the execution-ordered flow for active workflows. The first numeric field is the hard pipeline layer: outputs from layer `NN` are prerequisites for layer `NN+1`. The second numeric field is for parallel branch work or branch-specific products within that layer.

## Canonical HCP Shared-Input Paths

The maintained Kong2022 branch names are cohort-independent: `pfm`,
`surface`, `fixed`, `pfm_ctxreg`, and `pfm_ctxdrop`. Current commands use
canonical roots directly; no migration utility is part of the active workflow.

## 1. HippUnfold Runtime and Validation

1. `01_01_run_hippunfold_t1w_local.py`
2. `01_02_check_hippunfold_outputs.py`

`01_01` uses the selected HippUnfold release with the T1w/512 preset and invokes `01_02`
with `--fail-on-missing --no-cleanup`. Validation checks the full output
contract and retains `anat/`, `coords/`, `warps/`, `cifti/`, `surf/`, `metric/`
and `qc/`. Resume requires `output_check.json` to match the current run
contract and all declared files to remain present; an older cleanup receipt
does not establish completion. The runtime gate, resource patches, dataset
presets and current batch command are maintained in
[the hippo2 runtime manifest](../environment/hippo2_reproducibility_manifest.md).

## 2. Group Inputs and QC Cohort

1. `02_01_prepare_group_inputs.py`
2. `02_02_build_qc_manifest.py`
3. `02_03_build_lynch2024_nsi_manifest.py` for the Lynch2024 all-base-subject comparison cohort and NSI audit table

`02_01` derives shared NIfTI-sampled signals, vertex masks, and display templates from validated HippUnfold outputs. Native T1w vertices are registered to MNI152NLin6Asym before volume sampling; the mapped surfaces are saved in `surface/raw/sampling_surfaces/`. Metrics use `space-T1w` filenames and retain the original vertex numbering. The native T1w mesh defines 2/4-mm smoothing, physical cortical-neighbour distances and subject-level vertex areas; corobl group-average surfaces serve display and vertex correspondence. Dataset registration is selected by `--dataset`, with an optional explicit `--registration-root` Jianglab prefix. Existing corobl raw-cache summaries fail resume. `02_02` applies physical-run Power FD, bilateral HippUnfold Dice, and NIfTI coverage using current sampled signals and NIfTI support arrays. Coverage requires positive image-grid support and a finite, nonconstant sampled signal in every physical run; CIFTI hippocampal membership is not required. The measured roster has no fixed historical subject count. Rebuild raw coverage and masks before recomputing a roster under the NIfTI policy. `02_03` reads an NSI TSV/CSV containing `subject,nsi`, normalizes `sub-` prefixes, includes every subject from that base manifest, and writes `hcp_7t_lynch2024_pass_latest.txt`, `hcp_7t_lynch2024_nsi_audit_latest.tsv`, and the run manifest under `outputs_hcp/inputs/lynch2024/manifests/`. NSI is audit metadata only; there is no NSI threshold or expected-count gate.

`04_00` and `04_01 --ctxdrop` take `--cortex-coordinate-space`. The default
`MNI152NLin6Asym` denotes the HCP MSMAll cortical surfaces; their coordinates
are transformed to native T1w with the subject's FNIRT image pullback before
distance queries. Caller-selected fMRIPrep native cortical meshes require
`--cortex-coordinate-space T1w`. fsLR denotes vertex correspondence and does
not itself specify the physical coordinate frame.

```bash
/home/yojiao/.conda/envs/py314/bin/python scripts/core_workflow/02_03_build_lynch2024_nsi_manifest.py --base-subjects-manifest outputs_hcp/inputs/kong2022/manifests/hcp_7t_hippocampus_fmri_qc_pass_t1only_latest.txt --nsi-tsv <path/to/nsi.tsv>
```

## 3. Cortex References

- `03_01_build_kong2022_reference.py` for shared Kong2022 branch usage
- RouteA ArealMSHBM individualized cortex label generation on Argon; this is an external-backend artifact prerequisite documented in `core_workflow_artifact_lineage.md`, not a local numbered entrypoint
- `03_02_build_kong2022_routea_reference.py --dataset hcp` for individualized PFM references, including compact 17-net summaries used by `kong2022_pfm_17net`, consuming the generated PFM label files above
- `03_04_run_lynch2024_fastans_pfm.py` for official FASTANS Lynch2024 hard-label PFM generation from local FASTANS priors and Workbench
- `03_03_build_lynch2024_reference.py` for hard-label Lynch2024 component references, exact Lynch20 summaries containing 20 unique networks after excluding `Noise`, and canonical7 merged summaries from `config/cross_atlas_network_merge.json`

```bash
/home/yojiao/.conda/envs/py314/bin/python scripts/core_workflow/03_04_run_lynch2024_fastans_pfm.py --subjects-manifest outputs_hcp/inputs/lynch2024/manifests/hcp_7t_lynch2024_pass_latest.txt --wb-command /home/yojiao/FASTANS/workbench/bin_linux64/wb_command --jobs 6
/home/yojiao/.conda/envs/py314/bin/python scripts/core_workflow/03_03_build_lynch2024_reference.py --shared-root outputs_hcp/inputs/lynch2024/shared_inputs/pfm --subjects-manifest outputs_hcp/inputs/lynch2024/manifests/hcp_7t_lynch2024_pass_latest.txt --cortex-pfm-root outputs_hcp/work/cortex_pfm --jobs 8
```

## 4. Hippocampal Time-Series Control and FC Materialization

1. `04_00_materialize_cortex_regressed_surface_timeseries.py` for the isolated `kong2022_pfm_ctxreg` branch
2. `04_01_materialize_kong2022_roi_fc.py --dataset hcp --ctxdrop` for the isolated `kong2022_pfm_ctxdrop` branch
3. `04_01_materialize_kong2022_roi_fc.py --dataset hcp`
4. `04_01_materialize_kong2022_roi_fc.py --dataset hcp --atlas lynch2024_20net` and `--atlas lynch2024_7net` for Lynch2024 ROI/component FC caches
5. `04_02_materialize_intrinsic_fc.py`

`04_00` regresses local valid cortex signal from hippocampal time series within each run segment and writes only controlled time series plus diagnostics under the formal root `outputs_hcp/inputs/kong2022/shared_inputs/pfm_ctxreg`. PFM references and `surface` masks are not copied; downstream `04_01` and `04_02` write FC into the controlled root while reading those static inputs via `--reference-root` and `--mask-root`, so FC filenames remain unchanged but isolated from PFM.

`04_01 --ctxdrop` keeps hippocampal time series unchanged, reloads cortex dtseries plus PFM individualized Kong2022 labels, and for each hippocampal vertex excludes valid cortex vertices within 20 mm before computing the target ROI mean time series. It writes standard Kong2022 FC cache filenames and stores smoothing-independent local cortex-neighbor diagnostics under `outputs_hcp/inputs/kong2022/shared_inputs/pfm_ctxdrop/sub-*/surface/local_cortex_neighbors/hemi_*/`. The maintained ctxdrop downstream scope is `gradient-spectral` and `wta-top`.

All pooled FC removes each declared run mean at computation time, while stored baselines and run variances are preserved. Empty Kong2022 parcels retain all 400 ordered rows as NaN. Kong2022 FC consumers require matching dataset/task/QC policy and cutoff in the selected reference and both cortex gate summaries. Run IDs and lengths come from the exact reference summary or a caller-selected metadata file; historical untyped references must be rebuilt.

Lynch2024 FC materialization reuses the existing `04_01` implementation with `--atlas`; run it once per Lynch atlas slug so downstream route validators never fall back to Kong2022 paths:

```bash
/home/yojiao/.conda/envs/py314/bin/python scripts/core_workflow/04_01_materialize_kong2022_roi_fc.py --dataset hcp --shared-root outputs_hcp/inputs/lynch2024/shared_inputs/pfm --surface-root outputs_hcp/inputs/kong2022/shared_inputs/surface --reference-root outputs_hcp/inputs/lynch2024/shared_inputs/pfm --mask-root outputs_hcp/inputs/kong2022/shared_inputs/surface --subjects-manifest outputs_hcp/inputs/lynch2024/manifests/hcp_7t_lynch2024_pass_latest.txt --atlas lynch2024_20net
/home/yojiao/.conda/envs/py314/bin/python scripts/core_workflow/04_01_materialize_kong2022_roi_fc.py --dataset hcp --shared-root outputs_hcp/inputs/lynch2024/shared_inputs/pfm --surface-root outputs_hcp/inputs/kong2022/shared_inputs/surface --reference-root outputs_hcp/inputs/lynch2024/shared_inputs/pfm --mask-root outputs_hcp/inputs/kong2022/shared_inputs/surface --subjects-manifest outputs_hcp/inputs/lynch2024/manifests/hcp_7t_lynch2024_pass_latest.txt --atlas lynch2024_7net
```

## 5. Group Atlas Construction

1. `05_01_compute_hippomaps_gradients.py` for Kong2022 and Lynch2024 gradient artifacts
2. `05_02_run_group_atlas.py` for group atlas outputs and K=2..8 diagnostics with fixed final K=3
3. `05_03_run_spectral_fast_rerun.py` for optional rerun orchestration after the primary atlas step
4. `05_04_compute_confidence_maps.py` for vertex-level silhouette confidence maps in G1/G2/G3 feature space

For multi-cohort gradients, use the same `--cohort <name> --variant 36p`
selection in `05_01`, `05_02`, and `06_04`. The gradient base root is the stage
root; `--cohort nthc --variant 36p` with `pfm` resolves to
`<gradient-stage-root>/pfm/nthc/<smoothing>/hemi_<hemi>/`.
Select the matching upstream roots and subject manifest explicitly. HCP omits
`--cohort` and uses the default `36p` variant. Group-atlas output roots remain
explicit, for example `work/group_atlas/pfm/nthc`.

Kong2022 HippoMaps gradient artifact example. The exact diffusion-map solver takes no seed; each `05_01` manifest records it as `gradient_preprocessing.eigensolver`:

```bash
/home/yojiao/.conda/envs/py314/bin/python scripts/core_workflow/05_01_compute_hippomaps_gradients.py --routes pfm fixed --smoothings 2mm 4mm --hemis L R --n-components 5 --selected-components 1,2,3
/home/yojiao/.conda/envs/py314/bin/python scripts/core_workflow/05_01_compute_hippomaps_gradients.py --routes pfm_ctxreg --shared-inputs-root outputs_hcp/inputs/kong2022/shared_inputs --reference-inputs-root outputs_hcp/inputs/kong2022/shared_inputs/pfm --surface-root outputs_hcp/inputs/kong2022/shared_inputs/surface --out-root outputs_hcp/work/hippomaps_gradients --smoothings 2mm 4mm --hemis L R --n-components 5 --selected-components 1,2,3
/home/yojiao/.conda/envs/py314/bin/python scripts/core_workflow/05_01_compute_hippomaps_gradients.py --routes pfm_ctxdrop --shared-inputs-root outputs_hcp/inputs/kong2022/shared_inputs --reference-inputs-root outputs_hcp/inputs/kong2022/shared_inputs/pfm --surface-root outputs_hcp/inputs/kong2022/shared_inputs/surface --out-root outputs_hcp/work/hippomaps_gradients --smoothings 2mm 4mm --hemis L R --n-components 5 --selected-components 1,2,3
```

Kong2022 PFM and Fixed group clustering example:

```bash
/home/yojiao/.conda/envs/py314/bin/python scripts/core_workflow/05_02_run_group_atlas.py --shared-root outputs_hcp/inputs/kong2022/shared_inputs/pfm_ctxreg --surface-root outputs_hcp/inputs/kong2022/shared_inputs/surface --profile-surface-root outputs_hcp/inputs/kong2022/shared_inputs/pfm_ctxreg --reference-root outputs_hcp/inputs/kong2022/shared_inputs/pfm --out-root outputs_hcp/work/group_atlas/pfm_ctxreg --methods gradient-spectral wta-top --atlases kong2022 --gradient-root outputs_hcp/work/hippomaps_gradients --route-tag pfm_ctxreg --network-taxonomy canonical7 --k-min 2 --k-max 8 --default-k 3
/home/yojiao/.conda/envs/py314/bin/python scripts/core_workflow/05_02_run_group_atlas.py --shared-root outputs_hcp/inputs/kong2022/shared_inputs/pfm_ctxdrop --surface-root outputs_hcp/inputs/kong2022/shared_inputs/surface --profile-surface-root outputs_hcp/inputs/kong2022/shared_inputs/surface --reference-root outputs_hcp/inputs/kong2022/shared_inputs/pfm --out-root outputs_hcp/work/group_atlas/pfm_ctxdrop --methods gradient-spectral wta-top --atlases kong2022 --gradient-root outputs_hcp/work/hippomaps_gradients --route-tag pfm_ctxdrop --network-taxonomy canonical7 --k-min 2 --k-max 8 --default-k 3
/home/yojiao/.conda/envs/py314/bin/python scripts/core_workflow/05_02_run_group_atlas.py --shared-root outputs_hcp/inputs/kong2022/shared_inputs/fixed --surface-root outputs_hcp/inputs/kong2022/shared_inputs/surface --out-root outputs_hcp/work/group_atlas/fixed --methods intrinsic-spectral network-spectral gradient-spectral wta-top --atlases kong2022 --gradient-root outputs_hcp/work/hippomaps_gradients --route-tag fixed --k-min 2 --k-max 8 --default-k 3
/home/yojiao/.conda/envs/py314/bin/python scripts/core_workflow/05_02_run_group_atlas.py --shared-root outputs_hcp/inputs/kong2022/shared_inputs/pfm --surface-root outputs_hcp/inputs/kong2022/shared_inputs/surface --reference-root outputs_hcp/inputs/kong2022/shared_inputs/pfm --out-root outputs_hcp/work/group_atlas/pfm_17net/topsim --methods gradient-ward --profile topsim --atlases kong2022 --gradient-root outputs_hcp/work/hippomaps_gradients --route-tag pfm_17net --network-taxonomy family17 --reuse-fixed-cluster-root outputs_hcp/work/group_atlas/pfm/topsim --k-min 2 --k-max 8 --default-k 3
/home/yojiao/.conda/envs/py314/bin/python scripts/core_workflow/05_02_run_group_atlas.py --shared-root outputs_hcp/inputs/kong2022/shared_inputs/pfm --surface-root outputs_hcp/inputs/kong2022/shared_inputs/surface --reference-root outputs_hcp/inputs/kong2022/shared_inputs/pfm --out-root outputs_hcp/work/group_atlas/pfm_17net --methods wta-top --atlases kong2022 --route-tag pfm_17net --network-taxonomy family17
/home/yojiao/.conda/envs/py314/bin/python scripts/core_workflow/05_02_run_group_atlas.py --shared-root outputs_hcp/inputs/kong2022/shared_inputs/fixed --surface-root outputs_hcp/inputs/kong2022/shared_inputs/surface --reference-root outputs_hcp/inputs/kong2022/shared_inputs/fixed --out-root outputs_hcp/work/group_atlas/fixed_17net --methods gradient-spectral --atlases kong2022 --gradient-root outputs_hcp/work/hippomaps_gradients --route-tag fixed_17net --network-taxonomy family17 --reuse-fixed-cluster-root outputs_hcp/work/group_atlas/fixed --k-min 2 --k-max 8 --default-k 3
/home/yojiao/.conda/envs/py314/bin/python scripts/core_workflow/06_07_render_ctxdrop_neighbor_counts.py --shared-root outputs_hcp/inputs/kong2022/shared_inputs/pfm_ctxdrop --source-branch pfm_ctxdrop --out-dir outputs_hcp/present/local_cortex_neighbors/pfm_ctxdrop
```

Kong2022 WTA runs use the single `normalized_top20_count` assignment. For the
`pfm_17net`/`fixed_17net` (`family17`) branches, gradient methods reuse the
canonical7 spatial clusters, while `wta-top` is a separate clusterless run
(`--methods wta-top`, no `--reuse-fixed-cluster-root`) that counts votes over
the `family17` registry column. `06_01` requires each Kong WTA source manifest
to declare the route's taxonomy (`family17` for `*_17net`, otherwise
`canonical7`).

Lynch2024 group-atlas commands use the all-base-subject shared root, route-specific atlas slug, and matching taxonomy. `05_01` generates one shared Lynch20 non-Noise component gradient bank with BrainSpace full-graph affinity (`brainspace_sparsity=0.0`); `lynch2024_7net` gradient-spectral reads that bank. The maintained Lynch methods are `gradient-spectral` and direct-network `wta-top`. Lynch gradient diagnostics evaluate K=2..8; `05_02` uses shifted-cosine surface affinity for gradient clusters and direct top-1 Fisher-z network assignments for WTA, with the same full-sample cluster-to-network FC redundancy metric as Kong2022. K=3 is the explicit primary default, while automatic K selection remains recorded separately. If `--methods` is omitted, both Lynch methods are generated.

```bash
/home/yojiao/.conda/envs/py314/bin/python scripts/core_workflow/05_01_compute_hippomaps_gradients.py --routes lynch2024_20net --shared-inputs-root outputs_hcp/inputs/lynch2024/shared_inputs --surface-root outputs_hcp/inputs/kong2022/shared_inputs/surface --subjects-manifest outputs_hcp/inputs/lynch2024/manifests/hcp_7t_lynch2024_pass_latest.txt --smoothings 2mm 4mm --hemis L R --group-average-surfaces-dir outputs_hcp/inputs/kong2022/templates/hipp_group_average/group_average_surfaces --n-components 5 --selected-components 1,2,3 --out-root outputs_hcp/work/hippomaps_gradients --force
/home/yojiao/.conda/envs/py314/bin/python scripts/core_workflow/05_02_run_group_atlas.py --shared-root outputs_hcp/inputs/lynch2024/shared_inputs/pfm --surface-root outputs_hcp/inputs/kong2022/shared_inputs/surface --profile-surface-root outputs_hcp/inputs/kong2022/shared_inputs/pfm_ctxreg --reference-root outputs_hcp/inputs/lynch2024/shared_inputs/pfm --out-root outputs_hcp/work/group_atlas/lynch2024_20net --atlases lynch2024_20net --gradient-root outputs_hcp/work/hippomaps_gradients --route-tag lynch2024_20net --network-taxonomy lynch20 --subjects-manifest outputs_hcp/inputs/lynch2024/manifests/hcp_7t_lynch2024_pass_latest.txt --k-min 2 --k-max 8 --default-k 3 --jobs 1 --group-average-surfaces-dir outputs_hcp/inputs/kong2022/templates/hipp_group_average/group_average_surfaces
/home/yojiao/.conda/envs/py314/bin/python scripts/core_workflow/05_02_run_group_atlas.py --shared-root outputs_hcp/inputs/lynch2024/shared_inputs/pfm --surface-root outputs_hcp/inputs/kong2022/shared_inputs/surface --profile-surface-root outputs_hcp/inputs/kong2022/shared_inputs/pfm_ctxreg --reference-root outputs_hcp/inputs/lynch2024/shared_inputs/pfm --out-root outputs_hcp/work/group_atlas/lynch2024_7net --atlases lynch2024_7net --gradient-root outputs_hcp/work/hippomaps_gradients --route-tag lynch2024_7net --network-taxonomy lynch7 --subjects-manifest outputs_hcp/inputs/lynch2024/manifests/hcp_7t_lynch2024_pass_latest.txt --k-min 2 --k-max 8 --default-k 3 --jobs 1 --group-average-surfaces-dir outputs_hcp/inputs/kong2022/templates/hipp_group_average/group_average_surfaces
```

For Kong2022 cluster-based ROI-registry runs, `fisherz` and `normalized_top20_count` remain available as cluster assignment variants, with `fisherz` (`top1_fisher_z`) as the default; normalized Top-20 count is an explicit comparison, and raw `count` is not generated by `core_workflow`. Kong ROI-registry `wta-top` uses only `normalized_top20_count`, meaning top-20 positive ROI/component counts divided by finite available ROI/component counts in each network. Lynch direct-network `wta-top` uses only `fisherz`, assigning each supported vertex to the strongest finite network Fisher-z value. Render Lynch outputs from their isolated group roots; gradient titles identify shifted-cosine and manifests record `surface_affinity`, `affinity_description`, and `source_group_manifests`:

```bash
/home/yojiao/.conda/envs/py314/bin/python scripts/core_workflow/06_01_build_group_overview.py --group-root outputs_hcp/work/group_atlas/lynch2024_20net --present-dir outputs_hcp/present/overviews/lynch2024_20net --route-tag lynch2024_20net --atlases lynch2024_20net --assignment-variant both
/home/yojiao/.conda/envs/py314/bin/python scripts/core_workflow/06_01_build_group_overview.py --group-root outputs_hcp/work/group_atlas/lynch2024_7net --present-dir outputs_hcp/present/overviews/lynch2024_7net --route-tag lynch2024_7net --atlases lynch2024_7net --assignment-variant both
```

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

`07_01` reads refreshed `06_01` method overviews from canonical group-atlas roots, including `pfm_17net`, `pfm_ctxreg`, `pfm_ctxdrop`, `fixed`, and `fixed_17net` when requested, writes each PNG under `outputs_hcp/present/overviews/<route>/`, and writes `outputs_hcp/present/overviews/group_overview_matrix_manifest.json` plus the overview README at the stage root. WTA matrix jobs include `pfm_17net` and `fixed_17net`; request only routes whose `wta-top` products exist. The WTA-top overview itself contains the fixed 20-bin transposed long-axis network profile. Use `--methods` to restrict exports to completed methods (for example `gradient-spectral wta-top`).

## Utilities

- `tests/core_workflow/smoke_prepare_run.py` prepares smoke command plans and run-owned outputs under `outputs_hcp/runs/<run_id>/`. Stage 01 retains its canonical HippUnfold batch root. The utility is outside the numbered pipeline.

Maintained active output roots:

- Work: `outputs_hcp/work/`
- Lynch2024 inputs: `outputs_hcp/inputs/lynch2024/`
- Lynch2024 cortex PFM work: `outputs_hcp/work/cortex_pfm/`
- Presentation: `outputs_hcp/present/`
- Execution records and non-official products: `outputs_hcp/runs/<run_id>/`

The complete directory contract is maintained in [Output Layout](output_layout.md).

For branch flags, path roots, and exact artifact contracts, use:

- `core_workflow_step_registry.md`
- `core_workflow_artifact_lineage.md`
- `scripts/core_workflow/step_registry.json`
