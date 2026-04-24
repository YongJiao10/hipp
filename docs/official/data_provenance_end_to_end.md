# End-to-End Data Provenance (Including LSS External Sources)

This document records the data lineage for the hippocampal network-first workflow, including external LSS paths, so provenance does not start only at `data/hippunfold_input`.

## Why This Exists

- The runtime pipeline mostly consumes local `data/hippunfold_input`.
- But upstream materialization and some batch utilities can read external data trees.
- Provenance must cover both layers.

## Canonical Runtime Input Layer (Local)

Main pipeline scripts read local staged inputs:

- `data/hippunfold_input/sub-<id>/anat/sub-<id>_T1w.nii.gz`
- `data/hippunfold_input/sub-<id>/anat/sub-<id>_T2w.nii.gz`
- `data/hippunfold_input/sub-<id>/func/sub-<id>_task-rest_run-concat.dtseries.nii`
- `data/hippunfold_input/sub-<id>/func/sub-<id>_task-rest_run-concat_bold.nii.gz` (when raw surface sampling is needed)

Primary code entrypoint:

- `scripts/hipp_parcellation_network/run_subject.py` (`--input-root` default: `data/hippunfold_input`)

## Upstream External Source Layer

### A) Copy-to-local path (manifest-recorded)

`scripts/copy_hcp_minimal.py` copies required files into local input trees and records provenance in manifest JSONs.

- The script default `--remote-root` is `/Volumes/Elements/HCP-YA-2025` (override-able by CLI).
- `sub-*_task-rest_run-concat_bold.nii.gz` is extracted from `Rest7TRecommended.zip` member:
  - `MNINonLinear/Results/rfMRI_REST_7T/rfMRI_REST_7T_hp2000_clean_rclean_tclean.nii.gz`

Manifest evidence examples:

- `manifests/raw_copy/sub-102311_copy_manifest.json`
- `manifests/raw_copy/sub-102816_copy_manifest.json`

### B) External LSS batch utility path (hard-coded in script)

There is an external-data batch utility that directly points to:

- `/home/yojiao/LSS/jianglab/3_Data_Working/HCP_7T_Hippocampus/rsfmri_volume`

Script:

- `scripts/batch_surface_sampling_tsnr.py`

This script uses:

- `BOLD_ROOT = Path("/home/yojiao/LSS/jianglab/3_Data_Working/HCP_7T_Hippocampus/rsfmri_volume")`
- per-subject file: `<BOLD_ROOT>/<subject>/func_uncompressed.nii`

It then writes hippocampal raw surface timeseries into:

- `outputs_migration/hipp_functional_parcellation_network/_shared/sub-<id>/surface/raw/*.func.gii`

## Relationship Between Volume and Surface Timeseries

- Upstream signal source can be volume rs-fMRI (`concat_bold` or `func_uncompressed.nii`).
- Analysis-stage hippocampal features are surface timeseries (`*.func.gii` / `*.npy`) after volume-to-surface sampling.

So the workflow is:

1. volume rs-fMRI source
2. sample to hippocampal surface
3. consume surface timeseries for FC / clustering

## Operational Rule

When reporting provenance, always state both:

1. upstream source path (external and/or copied-from location)
2. runtime consumed local path

Do not report only `data/hippunfold_input` when external source tracing is requested.

