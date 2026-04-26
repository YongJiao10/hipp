# Kong2022 Route B Flow

## Summary

Route B uses the public atlas label file directly:

- `external/Schaefer2018_Kong2022/Schaefer2018_400Parcels_Kong2022_17Networks_order.dlabel.nii`

No individualized cortex PFM is run in this route. Each subject gets:

- 400-ROI cortex timeseries (after cortex tSNR gate)
- merged canonical-8 cortex timeseries derived from those ROI timeseries

Atlas slug is fixed to `kong2022`.

For T1only, Route B must compute its own cortex tSNR gate and run slicing from the subject dtseries. Do not consume T2only Kong2019/PFM cortex reference outputs.

## Pipeline Steps

1. Build per-subject Kong2022 reference:
   - script: `scripts/cortex/extract_cortex_kong2022_roi_timeseries.py`
   - batch entrypoint: `scripts/cortex/run_cortex_kong2022_reference_batch.py`
   - output root: `outputs_migration/hipp_group_inputs_shared/sub-{sid}/reference/kong2022/`
2. Materialize hippocampal vertex-to-ROI FC:
   - script: `scripts/hipp_parcellation_network/materialize_kong2022_vertex_to_roi_fc.py`
   - requires `reference/common/runs/run-{1..4}/reference_summary.json` from Step 1
   - output root: `outputs_migration/hipp_group_inputs_shared/sub-{sid}/fc/kong2022/`
3. Run group atlas driver:
   - script: `scripts/hipp_parcellation_network/run_merged_canonical_group_atlas.py`
   - include `--atlases kong2022`
4. Render present overview:
   - script: `scripts/hipp_parcellation_network/build_present_group_overview_merged_canonical.py`

## Upstream Common Inputs

Route B now stores atlas-invariant cortex upstream products in `reference/common/`:

- cortex tSNR masks:
  - `reference/common/tsnr_gate/cortex_left_valid_mask.npy`
  - `reference/common/tsnr_gate/cortex_right_valid_mask.npy`
- run boundaries:
  - `reference/common/runs/run-{1..4}/reference_summary.json`

These products are atlas-invariant and are consumed by downstream FC stages.

Historical legacy outputs under `reference/kong2022/{tsnr_gate,runs}` are migrated into `reference/common/` and removed by the batch entrypoint.

Run storage policy for Route B is metadata slicing, not duplicate arrays:

- keep full-length arrays once:
  - `reference/kong2022/cortex_roi_parcel_timeseries.npy`
  - `reference/kong2022/cortex_canonical_network_timeseries.npy`
- keep run boundaries in:
  - `reference/common/runs/run-{1..4}/reference_summary.json`
- downstream scripts must read full arrays and slice `[run_start:run_stop]` at runtime.

## Group Outputs Specific to Kong2022

For each `(method, smoothing, coverage, hemi)` combo:

- `cluster_network_profile.npy` still reports merged canonical-8 cluster profiles
- `cluster_roi_profile.npy` stores `K_final x 400` cluster-to-ROI Fisher-z profile
- `cluster_roi_top20.json` stores per-cluster:
  - strongest positive 20 ROI
  - strongest negative 20 ROI
  - each ROI annotated with `family17` and `canonical8`
- `cluster_roi_registry.tsv` stores the full 400-ROI registry

Cluster network assignment is based on merged canonical-8 mean timeseries, where each canonical network timeseries is the mean of all corresponding Kong2022 ROI timeseries (for example `Default*` ROI mean for `Default`).

## Method Support

Active methods for Kong2022:

- `group-network-spectral`
- `group-intrinsic-spectral`

Partial methods are unsupported for Kong2022 in this route and are warning+skip in the group driver.
