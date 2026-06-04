# JJ_TMS Kong2022 ArealMSHBM

This directory contains the JJ_TMS-specific wrapper for subject-level
Kong2022_ArealMSHBM parcellation. It does not modify `scripts/core_workflow`.

## Inputs and Outputs

Default input root:

```bash
/Shared/jianglab/3_Data_Working/fmriprep_processed_JJ_TMS/derivatives_surf
```

Each task scans one `sub-*` directory for all available
`func/*task-rest*space-fsLR*den-91k*bold.dtseries.nii` runs and passes those
dtseries directly to `CBIG_ArealMSHBM_parcellation_single_subject` as separate
fsLR sessions in `params.lh_fMRI_list`. The runs are not merged or concatenated
before MATLAB. For each run, the task writes an all-ones censor file with the
run's CIFTI timepoint count.

Default output root:

```bash
/Shared/jianglab/3_Data_Working/fmriprep_processed_JJ_TMS/ArealMSHBM_Kong2022
```

Per-subject outputs are written directly under:

```bash
${OUT_ROOT}/sub-<id>
```

Key RouteA outputs:

- `lh_labels.txt`
- `rh_labels.txt`

The wrapper keeps CBIG working files and prepared input lists under
`/tmp/jj_tms_kong2022_<job>_<task>_<subject>` during execution and removes that
work directory automatically when the task exits.

## Submit

Default batch scans all JJ_TMS subjects under `DATA_ROOT` and follows the
`scripts/core_workflow/sge_argon/submit_kong2022_route_a_dual_queues.sh`
submission pattern:

- deterministic UI/PINC split by subject index, ratio 1:2
- UI uses `-tc 5`
- PINC uses `-tc 15`

```bash
scripts/ArealMSHBM_Kong2022_Iowa_Stanford/submit_jj_tms_kong2022_areal_mshbm.sh
```

Dry-run the launcher without submitting:

```bash
DRY_RUN=1 scripts/ArealMSHBM_Kong2022_Iowa_Stanford/submit_jj_tms_kong2022_areal_mshbm.sh
```

Submit qsub tasks that validate Python, CBIG, and subject dtseries inputs, then stop before MATLAB:

```bash
TASK_DRY_RUN=1 scripts/ArealMSHBM_Kong2022_Iowa_Stanford/submit_jj_tms_kong2022_areal_mshbm.sh
```

Smoke one real subject, defaulting to `sub-1006`:

```bash
SMOKE_N=1 scripts/ArealMSHBM_Kong2022_Iowa_Stanford/submit_jj_tms_kong2022_areal_mshbm.sh
```

Use `DRY_RUN=1` with any of the examples above to print the qsub command only.

Runtime subject lists and scheduler logs are written under `${OUT_ROOT}/_manifests`
and `${OUT_ROOT}/_sge_logs`; the script directory should contain source files only.

## Controls

- `SMOKE_N=N`: submit only the first N selected subjects.
- `SUBJECT_IDS_CSV`: comma-separated subject IDs.
- `SUBJECT_IDS_FILE`: subject manifest, one ID per line.
- `TASK_DRY_RUN=1`: task preflight only, no MATLAB.
- `DRY_RUN=1`: launcher dry-run only, no qsub submit.
- `OVERWRITE=1`: recompute existing outputs.

## Smoke Status

After a submitted smoke job, summarize scheduler state, logs, and expected label text files with:

```bash
scripts/ArealMSHBM_Kong2022_Iowa_Stanford/report_jj_tms_smoke_status.sh <job_id> sub-1006
```

Only submit the full batch after a real smoke run succeeds and both label text files are present.
