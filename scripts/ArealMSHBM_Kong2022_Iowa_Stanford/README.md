# Iowa/Stanford Kong2022 ArealMSHBM

This directory contains dataset-specific wrappers for subject-level
Kong2022_ArealMSHBM parcellation. It does not modify `scripts/core_workflow` or
Kong2022_ArealMSHBM algorithm logic.

## Datasets

Select the dataset with `DATASET=jj_tms` or `DATASET=stanford`. If unset,
`DATASET=jj_tms` is used for backward compatibility.

| Dataset | Default input root | Default output root | Rest dtseries pattern |
| --- | --- | --- | --- |
| `jj_tms` | `/Shared/jianglab/3_Data_Working/fmriprep_processed_JJ_TMS/derivatives_surf` | `/Shared/jianglab/3_Data_Working/fmriprep_processed_JJ_TMS/ArealMSHBM_Kong2022` | `func/*_task-rest_run-0*_space-fsLR_den-91k_bold.dtseries.nii` |
| `stanford` | `/Shared/jianglab/3_Data_Working/fmriprep_processed_CausalConnectome/rest_fmriprep/DataOrg_PFM_36P` | `/Shared/jianglab/3_Data_Working/fmriprep_processed_CausalConnectome/rest_fmriprep/ArealMSHBM_Kong2022` | `func/*_task-rest_space-fsLR_den-91k_desc-denoised_bold.dtseries.nii` |

Each task scans one `sub-*` directory for the selected dataset's rest dtseries
pattern and passes matching dtseries directly to
`CBIG_ArealMSHBM_parcellation_single_subject` as separate fsLR sessions in
`params.lh_fMRI_list`. Runs are not merged or concatenated before MATLAB. For
each dtseries, the task writes an all-ones censor file with the CIFTI timepoint
count.

Per-subject outputs are written directly under:

```bash
${OUT_ROOT}/sub-<id>
```

Key RouteA outputs:

- `lh_labels.txt`
- `rh_labels.txt`

The wrapper keeps CBIG working files and prepared input lists under a dataset
scoped `/tmp/<dataset>_kong2022_<job>_<task>_<subject>` directory during
execution and removes that work directory automatically when the task exits.

The task script prepends `${REPO_ROOT}/software/workbench/bin_linux64` to
`PATH` and passes `WB_COMMAND` through qsub so CBIG/MATLAB calls to
`wb_command` resolve on compute nodes. Override with `WB_COMMAND=/path/to/wb_command`
only when needed.

## Submit

Dry-run JJ_TMS without submitting:

```bash
DRY_RUN=1 scripts/ArealMSHBM_Kong2022_Iowa_Stanford/submit_kong2022_areal_mshbm.sh
```

Dry-run Stanford without submitting:

```bash
DATASET=stanford DRY_RUN=1 scripts/ArealMSHBM_Kong2022_Iowa_Stanford/submit_kong2022_areal_mshbm.sh
```

Submit qsub tasks that validate Python, CBIG, and subject dtseries inputs, then
stop before MATLAB:

```bash
DATASET=stanford TASK_DRY_RUN=1 scripts/ArealMSHBM_Kong2022_Iowa_Stanford/submit_kong2022_areal_mshbm.sh
```

Smoke one real subject:

```bash
DATASET=stanford SMOKE_N=1 scripts/ArealMSHBM_Kong2022_Iowa_Stanford/submit_kong2022_areal_mshbm.sh
```

Use `DRY_RUN=1` with any example above to print qsub commands only.

The default batch scans all subjects under `DATA_ROOT` and follows the source
Argon launcher submission pattern:

- deterministic UI/PINC split by subject index, ratio 1:2
- UI uses `-tc 5`
- PINC uses `-tc 15`

Runtime subject lists and scheduler logs are written under `${OUT_ROOT}/_manifests`
and `${OUT_ROOT}/_sge_logs`; the script directory should contain source files
only.

## Controls

- `DATASET=jj_tms|stanford`: dataset preset and default path selector.
- `DATA_ROOT`: override the selected dataset input root.
- `OUT_ROOT`: override the selected dataset output root.
- `WB_COMMAND`: override Workbench command path; default is
  `${REPO_ROOT}/software/workbench/bin_linux64/wb_command`.
- `SMOKE_N=N`: submit only the first N selected subjects.
- `SUBJECT_IDS_CSV`: comma-separated subject IDs.
- `SUBJECT_IDS_FILE`: subject manifest, one ID per line.
- `UI_TC=N`: UI array throttle, default `5`.
- `PINC_TC=N`: PINC array throttle, default `15`.
- `TASK_DRY_RUN=1`: task preflight only, no MATLAB.
- `DRY_RUN=1`: launcher dry-run only, no qsub submit.
- `OVERWRITE=1`: recompute existing outputs.

## Status

After a submitted smoke or batch job, summarize scheduler state, logs, and
expected label text files with:

```bash
DATASET=stanford scripts/ArealMSHBM_Kong2022_Iowa_Stanford/report_kong2022_areal_mshbm_status.sh <job_id> sub-1001
```

Only submit a full batch after a real smoke run succeeds and both label text
files are present for the smoke subject.
