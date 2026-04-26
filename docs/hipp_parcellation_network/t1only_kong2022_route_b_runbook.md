# T1only Kong2022 Route B Runbook

## Scope

This runbook is the canonical path for the T1only HippUnfold branch:

- HippUnfold batch root: `outputs_experiments/hippunfold_recommended_t1w_20260423/dense_corobl_batch/`
- Route root: `outputs_experiments/hippunfold_recommended_t1w_20260423/kong2022_route_b/`
- Shared T1only group inputs: `outputs_experiments/hippunfold_recommended_t1w_20260423/kong2022_route_b/hipp_group_inputs_shared/`
- Group atlas outputs: `outputs_experiments/hippunfold_recommended_t1w_20260423/kong2022_route_b/hipp_group_merged_canonical_atlas/`
- Presentation overview outputs: `group_present_group_t1only/`

The old `manifests/hcp_7t_hippocampus_fmri_qc_pass_153.txt` is not valid for this branch. T1only must derive a new pass manifest from its own HippUnfold Dice QC and its own hippocampal surface tSNR QC.

## Non-Reuse Rule

Kong2022 in this run uses Route B, the public 400-ROI dlabel:

- `external/Schaefer2018_Kong2022/Schaefer2018_400Parcels_Kong2022_17Networks_order.dlabel.nii`

Do not reuse T2only cortex PFM, Kong2019 reference outputs, or old `reference/kong2019` run summaries. Route B computes its own cortex tSNR gate directly from each subject's concat dtseries and extracts Kong2022 ROI/canonical-8 timeseries from the public dlabel.

## Command

Run the full flow with:

```bash
/home/yojiao/.conda/envs/py314/bin/python scripts/hipp_parcellation_network/run_t1only_kong2022_route_b_pipeline.py \
  --jobs 8 \
  --resume
```

The repository-wide concurrency cap is 10. Keep `--jobs` at or below 10.

Both launchers (`run_t1only_kong2022_route_b_pipeline.py`, `prepare_t1only_group_inputs.py`) enforce this cap by default.
For an explicitly approved one-off session, use `HIPP_MAX_CONCURRENCY` to raise the runtime cap:

```bash
HIPP_MAX_CONCURRENCY=12 \
/home/yojiao/.conda/envs/py314/bin/python scripts/hipp_parcellation_network/run_t1only_kong2022_route_b_pipeline.py \
  --jobs 12 \
  --resume
```

Useful restarts:

```bash
/home/yojiao/.conda/envs/py314/bin/python scripts/hipp_parcellation_network/run_t1only_kong2022_route_b_pipeline.py \
  --from-step kong2022-reference \
  --jobs 8 \
  --resume

/home/yojiao/.conda/envs/py314/bin/python scripts/hipp_parcellation_network/run_t1only_kong2022_route_b_pipeline.py \
  --from-step group-atlas \
  --jobs 8 \
  --resume
```

## Runtime Counting Rule

When monitoring `wb_command` concurrency, use this process only:

- `.../exe_linux64/wb_command -volume-to-surface-mapping` (real compute workers)

Do not use `/bin/bash ... wb_command ...` wrapper count as compute concurrency; those are launcher shells.

## Stage Contract

1. `prepare`
   - Checks Dice files under T1only HippUnfold `qc/`.
   - Samples `rsfmri_volume/<sid>/func_uncompressed.nii` onto T1only `space-corobl_den-512` hippocampal surfaces.
   - Builds hippocampal tSNR gates with `tSNR=10000/std`, threshold `25`.
   - Writes `manifests/hcp_7t_hippocampus_fmri_qc_pass_t1only_<N>.txt` and `..._latest.txt`.
   - Builds T1only group-average midthickness surfaces from pass subjects only.
   - Generates `surface/2mm` and `surface/4mm` func.gii plus `*_timeseries.npy`.

2. `kong2022-reference`
   - Extracts cortex ROI timeseries from the Kong2022 dlabel.
   - Computes cortex tSNR masks within `reference/common/tsnr_gate/` (atlas-invariant upstream).
   - Uses explicit/equal 4-run slicing from the concat dtseries; the current HCP 7T concat input is expected to be `3600 = 4 x 900`.
   - Stores full-length atlas arrays once under `reference/kong2022/` (`cortex_roi_parcel_timeseries.npy`, `cortex_canonical_network_timeseries.npy`) and stores shared run boundaries in `reference/common/runs/run-*/reference_summary.json` instead of duplicating run timeseries arrays.
   - Legacy `reference/kong2022/{tsnr_gate,runs}` content is migrated to `reference/common/` and removed.

3. `kong2022-fc`
   - Materializes hippocampal vertex-to-Kong2022-400ROI Pearson FC for `2mm` and `4mm`, both hemispheres.

4. `intrinsic-fc`
   - Materializes atlas-invariant hippocampal vertex-to-vertex intrinsic FC for `2mm` and `4mm`, both hemispheres.

5. `group-atlas`
   - Runs `group-network-spectral` and `group-intrinsic-spectral` for atlas `kong2022`.
   - Uses coverage threshold `0.80` and the T1only group-average surfaces.

6. `present`
   - Builds overview figures in `group_present_group_t1only/`.

## Traceability

Every stage writes a `stage_manifest.json` or top-level manifest containing:

- script path and command
- parameters
- input file metadata
- output file metadata
- timestamp
- git commit when available

The top-level pipeline manifest is:

- `outputs_experiments/hippunfold_recommended_t1w_20260423/kong2022_route_b/pipeline_manifest.json`

If an expected product exists without the corresponding manifest, treat it as non-authoritative and rerun that stage.
