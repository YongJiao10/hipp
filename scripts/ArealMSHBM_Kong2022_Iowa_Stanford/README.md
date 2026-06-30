# Iowa/Stanford Kong2022 ArealMSHBM

This directory contains dataset-specific wrappers for subject-level
Kong2022_ArealMSHBM parcellation. It does not modify `scripts/core_workflow` or
Kong2022_ArealMSHBM algorithm logic.

## Datasets

Select the dataset with `DATASET=jj_tms`, `DATASET=stanford`, `DATASET=iowa7t`,
or `DATASET=iowa_hcp500`. If unset,
`DATASET=jj_tms` is used for backward compatibility.

| Dataset | Default input root | Default output root | Rest dtseries pattern |
| --- | --- | --- | --- |
| `jj_tms` | `/Shared/jianglab/3_Data_Working/fmriprep_processed_JJ_TMS/xcpd_surf` | `/Shared/jianglab/3_Data_Working/fmriprep_processed_JJ_TMS/ArealMSHBM_Kong2022` | `func/*_task-rest_run-*_space-fsLR_den-91k_desc-denoised_bold.dtseries.nii` |
| `stanford` | `/Shared/jianglab/3_Data_Working/fmriprep_processed_CausalConnectome/rest_fmriprep/DataOrg_PFM_36P` | `/Shared/jianglab/3_Data_Working/fmriprep_processed_CausalConnectome/rest_fmriprep/ArealMSHBM_Kong2022` | `func/*_task-rest_space-fsLR_den-91k_desc-denoised_bold.dtseries.nii` |
| `iowa7t` | `/Shared/jianglab/3_Data_Working/fmriprep_processed_Iowa7T/DataOrg_PFM_36P` | `/Shared/jianglab/3_Data_Working/fmriprep_processed_Iowa7T/DataOrg_PFM_36P/hipp_outputs` | selected by `IOWA7T_BRANCH` and `IOWA7T_VARIANT` |
| `iowa_hcp500` | `/old_Users/yojiao/HCPpipeline500/work/qunex_iowa7t_hcppipeline500/sessions` | `${REPO_ROOT}/outputs_iowa_hcp500/inputs/iowa_hcp500/kong2022/cortex_references/routea_areal_mshbm` | `<subject>/hcp/<subject>/MNINonLinear/Results/REST/REST_Atlas_MSMAll_hp0_clean_rclean.dtseries.nii` |

Each task scans one `sub-*` directory for the selected dataset's rest dtseries
pattern and passes matching dtseries directly to
`CBIG_ArealMSHBM_parcellation_single_subject` as separate fsLR sessions in
`params.lh_fMRI_list`. Runs are not merged or concatenated before MATLAB. For
each dtseries, the task writes an all-ones censor file with the CIFTI timepoint
count.
The `iowa_hcp500` task resolves its fixed HCP REST path directly and obtains
the timepoint count with `wb_command -file-information`; it does not require a
Python runtime on Argon. The `iowa7t` task also resolves its fixed branch CIFTI
path directly and obtains the timepoint count with `wb_command -file-information`;
it does not require a Python runtime on Argon. Existing JJ_TMS and Stanford
presets continue to use the Python input preparer.

For `DATASET=iowa7t`, `IOWA7T_BRANCH` is required and `IOWA7T_VARIANT` selects
the fMRIPrep denoising tree. `IOWA7T_VARIANT=36p` is the default. If
`DATA_ROOT` ends in `DataOrg_PFM_NoXCPD`, the launcher auto-selects
`IOWA7T_VARIANT=noxcpd`; callers can also set it explicitly.

With `IOWA7T_VARIANT=36p`, the branch controls the exact subject CIFTI consumed
by the task:

- `IOWA7T_BRANCH=multiband`:
  `sub-<ID>/func/sub-<ID>_task-rest_dir-concat_space-fsLR_den-91k_desc-denoised_bold.dtseries.nii`
- `IOWA7T_BRANCH=multi_echo`:
  source CIFTI
  `sub-<ID>/func/sub-<ID>_task-rest_run-concat_space-fsLR_den-91k_desc-MEICA_XCPD36P_denoised_bold.dtseries.nii`
  is split into left/right cortex metrics and rebuilt as a persistent
  cortex-only fsLR32k dtseries before MATLAB.

With `IOWA7T_VARIANT=noxcpd`, the task consumes the NoXCPD equivalents:

- `IOWA7T_BRANCH=multiband`:
  `sub-<ID>/func/sub-<ID>_task-rest_dir-concat_space-fsLR_den-91k_desc-NoXCPD_bold.dtseries.nii`
- `IOWA7T_BRANCH=multi_echo`:
  source CIFTI
  `sub-<ID>/func/sub-<ID>_task-rest_run-concat_space-fsLR_den-91k_desc-MEICA_NoXCPD_bold.dtseries.nii`
  is split into left/right cortex metrics and rebuilt as a persistent
  cortex-only fsLR32k dtseries before MATLAB.

Per-subject outputs are written directly under:

```bash
${OUT_ROOT}/sub-<id>
```

For `DATASET=iowa7t`, `OUT_ROOT` is the selected tree's production
`hipp_outputs` root and per-branch outputs are written under:

```bash
${OUT_ROOT}/subjects/${subject}/${iowa7t_branch}/cortex/areal_mshbm
```

The Iowa7T `multi_echo` branch writes its derived cortex-only input under:

```bash
${OUT_ROOT}/runs/derived_inputs/${RUN_ID}/subjects/${subject}/multi_echo/cortex_only
```

Key RouteA outputs:

- `lh_labels.txt`
- `rh_labels.txt`
- `provenance.json` after successful non-dry-run MATLAB completion

The wrapper keeps CBIG working files and prepared input lists under a dataset
scoped `/tmp/<dataset>_kong2022_<job>_<task>_<subject>` directory during
execution and removes that work directory automatically when the task exits.
For `iowa_hcp500`, `KEEP_TMP=1` is the default so validation and smoke runs do
not delete temporary files. Set `KEEP_TMP=0` only when deletion has been
explicitly approved.

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

Iowa HCP500 single-subject launcher dry-run, task preflight, and real smoke:

```bash
DATASET=iowa_hcp500 SUBJECT_IDS_CSV=04172026 DRY_RUN=1 \
  scripts/ArealMSHBM_Kong2022_Iowa_Stanford/submit_kong2022_areal_mshbm.sh
DATASET=iowa_hcp500 SUBJECT_IDS_CSV=04172026 TASK_DRY_RUN=1 \
  scripts/ArealMSHBM_Kong2022_Iowa_Stanford/submit_kong2022_areal_mshbm.sh
DATASET=iowa_hcp500 SUBJECT_IDS_CSV=04172026 \
  scripts/ArealMSHBM_Kong2022_Iowa_Stanford/submit_kong2022_areal_mshbm.sh
```

Iowa7T 36P branch dry-runs for the two requested production subjects:

```bash
DATASET=iowa7t IOWA7T_BRANCH=multiband SUBJECT_IDS_CSV=06112026,06122026 \
  RUN_ID=iowa7t_36p_single_06112026_06122026 DRY_RUN=1 \
  scripts/ArealMSHBM_Kong2022_Iowa_Stanford/submit_kong2022_areal_mshbm.sh
DATASET=iowa7t IOWA7T_BRANCH=multi_echo SUBJECT_IDS_CSV=06112026,06122026 \
  RUN_ID=iowa7t_36p_multi_06112026_06122026 DRY_RUN=1 \
  scripts/ArealMSHBM_Kong2022_Iowa_Stanford/submit_kong2022_areal_mshbm.sh
```

Iowa7T NoXCPD branch dry-runs for the two requested production subjects:

```bash
DATASET=iowa7t IOWA7T_VARIANT=noxcpd IOWA7T_BRANCH=multiband \
  SUBJECT_IDS_CSV=06112026,06122026 RUN_ID=iowa7t_noxcpd_single_06112026_06122026 \
  DRY_RUN=1 scripts/ArealMSHBM_Kong2022_Iowa_Stanford/submit_kong2022_areal_mshbm.sh
DATASET=iowa7t IOWA7T_VARIANT=noxcpd IOWA7T_BRANCH=multi_echo \
  SUBJECT_IDS_CSV=06112026,06122026 RUN_ID=iowa7t_noxcpd_multi_06112026_06122026 \
  DRY_RUN=1 scripts/ArealMSHBM_Kong2022_Iowa_Stanford/submit_kong2022_areal_mshbm.sh
```

Use `DRY_RUN=1` with any example above to print qsub commands only.

The default batch scans all subjects under `DATA_ROOT` and follows the source
Argon launcher submission pattern:

- deterministic UI/PINC split by subject index, ratio 1:2
- each qsub task requests `-pe smp 4`
- `iowa_hcp500` uses `UI -tc 3` and `PINC -tc 7`, capping total concurrency at 10
- other presets retain `UI -tc 5` and `PINC -tc 15`

Runtime subject lists and scheduler logs are written under `${OUT_ROOT}/_manifests`
and `${OUT_ROOT}/_sge_logs` for legacy presets; for `DATASET=iowa7t`, they are
written under `${OUT_ROOT}/runs/manifests/${RUN_ID}` and
`${OUT_ROOT}/runs/logs/${RUN_ID}`. The script directory should contain source files
only. The launcher writes a queue-specific qsub wrapper under the manifest root that
exports the selected subject manifest before executing the task script. Task
logs show the consumed manifest as `SubjectIDsFile=...`; verify this path when
auditing a batch.

## Controls

- `DATASET=jj_tms|stanford|iowa7t|iowa_hcp500`: dataset preset and default path selector.
- `IOWA7T_BRANCH=multiband|multi_echo`: required with `DATASET=iowa7t`.
- `DATA_ROOT`: override the selected dataset input root.
- `OUT_ROOT`: override the selected dataset output root.
- `RUN_ID`: run identifier for manifest/log roots; Iowa7T defaults to branch plus timestamp.
- `IOWA7T_VARIANT=36p|noxcpd`: Iowa7T input naming/root selector; defaults to
  `36p` unless `DATA_ROOT` ends in `DataOrg_PFM_NoXCPD`.
- `WB_COMMAND`: override Workbench command path; default is
  `${REPO_ROOT}/software/workbench/bin_linux64/wb_command`.
- `SMOKE_N=N`: submit only the first N selected subjects.
- `SUBJECT_IDS_CSV`: comma-separated subject IDs.
- `SUBJECT_IDS_FILE`: subject manifest, one ID per line.
- `UI_TC=N`: UI array throttle, default `3` for `iowa_hcp500`, otherwise `5`.
- `PINC_TC=N`: PINC array throttle, default `7` for `iowa_hcp500`, otherwise `15`.
- `TASK_DRY_RUN=1`: task preflight only, no MATLAB.
- `DRY_RUN=1`: launcher dry-run only, no qsub submit.
- `OVERWRITE=1`: recompute existing outputs.
- `KEEP_TMP=1`: retain task temporary files; default for `iowa_hcp500`.

## Status

After a submitted smoke or batch job, summarize scheduler state, logs, and
expected label text files with:

```bash
DATASET=stanford scripts/ArealMSHBM_Kong2022_Iowa_Stanford/report_kong2022_areal_mshbm_status.sh <job_id> sub-1001
```

For Iowa HCP500:

```bash
DATASET=iowa_hcp500 scripts/ArealMSHBM_Kong2022_Iowa_Stanford/report_kong2022_areal_mshbm_status.sh <job_id> sub-04172026
```

For Iowa7T 36P:

```bash
DATASET=iowa7t IOWA7T_BRANCH=multiband \
  scripts/ArealMSHBM_Kong2022_Iowa_Stanford/report_kong2022_areal_mshbm_status.sh <job_id> sub-06112026
```

For Iowa7T NoXCPD:

```bash
DATASET=iowa7t IOWA7T_VARIANT=noxcpd IOWA7T_BRANCH=multiband \
  scripts/ArealMSHBM_Kong2022_Iowa_Stanford/report_kong2022_areal_mshbm_status.sh <job_id> sub-06112026
```

Only submit a full batch after a real smoke run succeeds and both label text
files are present for the smoke subject.
