# NIH rTMS Preprocessing Scripts (Argon SGE)

This directory contains the adapted Argon SGE array batch launchers for **fMRIPrep** and **XCP-D (36P)** on the merged NIH rTMS cohort (16 HV + 28 TRD, total 44 subjects).

These scripts adapt the established Jianglab pipeline from `fmriprep_processed_JJ_TMS/src/` and the workflow documentation `WorkingFlow_LSS_updated_07222026.pptx`.

---

## 1. Directory & File Overview

| File | Purpose | Description |
| :--- | :--- | :--- |
| `fmriprep_nih_rtms_restfmri.sh` | fMRIPrep SGE launcher | Runs fMRIPrep Singularity container on Argon (`UI,PINC`), outputting both `MNI152NLin6Asym:res-2` volumetric and `fsLR den-91k` CIFTI/surface derivatives with FreeSurfer `recon-all`, slice timing correction, and `--longitudinal` T1w templates. |
| `xcpd_36P_nih_rtms_restfmri.sh` | XCP-D SGE launcher | Runs XCP-D 36P nuisance regression, filtering (0.01-0.10 Hz), and censoring (`FD > 0.5 mm`) in both `cifti` (`xcpd_surf/restfmri`) and `nifti` (`xcpd/restfmri`) modes with retained per-subject scratch caches and atomic shared QC metadata writes. |
| `environment_manifest.md` | Preprocessing runtime contract | Pins container sources and resources, documents the read-only runtime overlays and validation checks. |
| `finalize_fmriprep_nih_rtms.sh` | Explicit fMRIPrep finalization repair | Validates a scientifically successful source array and publishes the exact FreeSurfer label dictionaries; records a completion receipt in its SGE log. |
| `sublist.txt` | Unified subject roster | 44 subjects (`sub-D01`..`sub-D28`, `sub-HVS1`..`sub-HVS17`). |

Submit the launchers from this Git checkout. The cluster shared `src/` directory holds the subject roster and logs; do not copy launchers there as part of a run.

---

## 2. Key Adaptations Made

1. **Unified Subject Roster (`sublist.txt`) & Fail-Closed Run Verification**:
   - Both fMRIPrep and XCP-D read from a single, unified `sublist.txt` (44 deduplicated subjects: 28 TRD from `ds008064` and 16 HV from `ds006731`, totaling 66 resting-state runs across 43 functional subjects).
   - Only `sub-D16` is permitted to have zero `task-rest` runs (`--anat-only` in fMRIPrep; clean skip `exit 0` in XCP-D). Any other subject missing raw `task-rest` runs fails closed.
   - Before and after running XCP-D, every expected raw `sub-*_ses-*_task-rest_bold.nii.gz` run is verified one-by-one against its paired `MNI152NLin6Asym:res-2` volumetric and `fsLR den-91k` CIFTI derivatives.
   - Hidden macOS dot files (`._*`, `.DS_Store`) in `bids_input` are checked fail-closed without automatic deletion.
2. **Simultaneous Cortical Surface (`fsLR 91k`) + Volumetric (`MNI152NLin6Asym:res-2`) Outputs**:
   - **fMRIPrep** runs with FreeSurfer `recon-all` enabled (utilizing 1mm isotropic T1w + 1mm isotropic T2w CUBE where available), `--longitudinal` unbiased multi-session T1w template construction, slice timing correction (`TR = 3.0s` interleaved 48 slices), and `--output-spaces MNI152NLin6Asym:res-2 fsLR --cifti-output 91k`.
   - **XCP-D** runs both `--file-format cifti` (writing cortical + subcortical `space-fsLR_den-91k_desc-denoised_bold.dtseries.nii` to `xcpd_surf/restfmri`) and `--file-format nifti` (writing full denoised volumes `space-MNI152NLin6Asym_res-2_desc-denoised_bold.nii.gz` to `xcpd/restfmri` for downstream HippUnfold surface sampling).
3. **Input / Output Paths (Argon Environment Contract)**:
   - Root: `/Shared/jianglab/3_Data_Working/Depression/NIH_rTMS`
   - BIDS Input: `/Shared/jianglab/3_Data_Working/Depression/NIH_rTMS/bids_input`
   - fMRIPrep Derivatives: `/Shared/jianglab/3_Data_Working/Depression/NIH_rTMS/derivatives/restfmri`
   - Per-subject FreeSurfer outputs: `/Shared/jianglab/3_Data_Working/Depression/NIH_rTMS/derivatives/restfmri/sourcedata/freesurfer_by_subject/sub-<ID>/sub-<ID>`
   - XCP-D Surface/CIFTI Output: `/Shared/jianglab/3_Data_Working/Depression/NIH_rTMS/xcpd_surf/restfmri`
   - XCP-D Volumetric Output: `/Shared/jianglab/3_Data_Working/Depression/NIH_rTMS/xcpd/restfmri`
   - Logs: `/Shared/jianglab/3_Data_Working/Depression/NIH_rTMS/src/logs`
   - fMRIPrep Scratch WorkDir: `/nfsscratch/Users/yojiao/fmriprep_NIH_rTMS_work/sub-${subject}`
   - XCP-D Scratch WorkDir: `/nfsscratch/Users/yojiao/xcpd_NIH_rTMS_work/sub-${subject}/{cifti,nifti}`
4. **Flexible Subject Execution**:
   - Single-subject mode: accepts subject ID as argument `$1` when submitted as a 1-task job (`qsub -t 1-1 -tc 1 script.sh sub-D01` or `bash script.sh sub-D01`). Combining `$1` with a multi-task SGE array is rejected fail-closed to prevent duplicate concurrent runs of the same subject.
   - SGE array mode: reads line `$SGE_TASK_ID` from the shared `sublist.txt`. Both arrays must use the same roster and task range so their task IDs identify the same participant.
5. **Argon Concurrency and FreeSurfer Outputs**:
   - fMRIPrep uses a separate persistent FreeSurfer `SUBJECTS_DIR` for each participant under `derivatives/restfmri/sourcedata/freesurfer_by_subject/sub-<ID>/`. This keeps concurrent template initialization isolated while retaining each subject's recon-all outputs.
   - Submit XCP-D with `-hold_jid_ad` so each task waits only for the matching participant's fMRIPrep task. Completed participants can start XCP-D while other participants are still in fMRIPrep.
   - Across both stages and other active repository jobs, keep the total concurrency at most 20. The submission example reserves 10 tasks for each stage. Rebalance caps with `qalter -tc` as fMRIPrep drains; with one remaining fMRIPrep task and no other active jobs, XCP-D may use 19 tasks, then 20 after fMRIPrep ends.
   - Use explicit user holds as an additional admission gate: SGE has been observed starting more tasks than `-tc` when multiple ready task groups are released. Submit XCP-D held, then release only validated ready participants up to the available concurrency budget. Count running, transferring, deletion-pending, and already released queued tasks before releasing more. Keep other pending tasks held and admit replacements as slots become free; do not release an entire sparse array at once. Decrease the other stage's cap before increasing one, and confirm actual scheduler state after every release.
6. **Runtime and Scratch Caches**:
   - fMRIPrep loads the exact FreeSurfer table resolver documented in [the environment manifest](environment_manifest.md), avoiding an ambiguous broad TemplateFlow lookup during finalization.
   - XCP-D loads the source-validated atomic QC overlay described in [the environment manifest](environment_manifest.md). Each pass retains a log in the subject scratch root and fails on node errors even if the container returns zero.
   - XCP-D retains its scratch caches for resumable execution. Cleanup requires explicit user approval for exact paths. fMRIPrep retains its configured `--clean-workdir` option; obtain explicit cleanup approval before submitting it, as required by the repository deletion safeguard.

---

## Retained data and reruns

Participant/visit motion selection uses unfiltered mean Power FD <=0.50 mm
before censoring and after dummy removal (`mean_fd` in the XCP-D LINC QC TSV).
Evaluate treatment sessions separately; paired pre/post analyses require both
sessions to pass. This mean-FD selection is separate from the existing XCP-D
per-frame censoring at FD >0.50 mm. The current data contain 43 functional
participants and 66 scans, with 64 scans passing this mean-FD threshold and
21 of the 23 complete TRD pairs passing motion in both visits. The full
HippUnfold functional QC below retains 61 scans and 18 complete pairs. See the
[fMRI QC policy](../../docs/hipp_parcellation_network/fmri_qc.md).

The active dataset retains the 132 fMRIPrep functional BOLD intermediates,
fMRIPrep-exported anatomical images, surfaces and transforms, both XCP-D output
trees, cohort mapping metadata and processing logs. The raw `bids_input/` tree
and the complete FreeSurfer
`derivatives/restfmri/sourcedata/freesurfer_by_subject/` tree are not retained.

The submission commands below require their declared inputs to be present.
Before rerunning XCP-D, reconstruct the merged raw BIDS inputs from OpenNeuro
`ds008064` v1.0.0 and `ds006731` v1.0.1 using the dataset's
`metadata/subject_id_mapping.tsv`; the launcher checks expected runs against raw
BIDS even when the fMRIPrep intermediates are retained. Those retained
intermediates allow new XCP-D variants without rerunning fMRIPrep. Analyses or
preprocessing steps that require complete FreeSurfer subjects need those
outputs rebuilt first. Existing XCP-D results remain available for downstream
analysis.

## 3. How to Submit on Argon

### Downstream HippUnfold v2.3.0 QC

The local reconstruction has 44 official v2.3.0 completion records under
`outputs_nih_rtms/inputs/hippunfold/dense_corobl_batch/`. The shared
`scripts/core_workflow/02_01_prepare_group_inputs.py` and `02_02_build_qc_manifest.py`
accept `--dataset nih_rtms`. Functional QC uses separate visit routes at
`outputs_nih_rtms/inputs/kong2022/ses-1/` and `ses-2/`, without pooling visits.
Each route's `manifests/nih_rtms_qc_runs.tsv` declares exact denoised NIfTI,
original motion and `temporal_mask_path` outlier inputs.
Supply the complete visit roster with `--subjects` or `--subjects-manifest`
when using this table, including visits without a declared run.

Motion selection uses original post-dummy, pre-censor frames; coverage uses
the shorter retained NIfTI series. The outlier table binds those two lengths.
The registration transform is paired with the exact fMRIPrep T1w consumed by
HippUnfold, supporting both longitudinal subject-level and single-visit
session-level anatomical exports. Use `--require-hippunfold-version 2.3.0`; the remaining criteria are bilateral
Dice >=0.70 and <=20 unusable den-512 vertices per hemisphere.
Reports are `qc/t1only_hippunfold_qc.{json,tsv}` and pass lists are
`manifests/nih_rtms_hippocampus_fmri_qc_pass_t1only_latest.txt` within each route.
Sampling, vertex masks and final assessment are complete: ses-1 retains 37/42
scans and ses-2 retains 24/24. Ses-1 excludes D11/D12 for mean FD and D08/D09/D28
for unusable sampled signals above the 20-vertex limit. Both reports have no
unassessable scans. Treatment analyses retain 18 of the 23 complete TRD pairs
after requiring both visits to pass all three criteria.
See the [shared QC policy and current results](../../docs/hipp_parcellation_network/fmri_qc.md).

Log into Argon (`ssh argon.hpc.uiowa.edu`) and set the path to this repository checkout:

```bash
repo_root=$(git -C /Users/yojiao/hipp rev-parse --show-toplevel)
```

### Step 1: Run fMRIPrep Array (44 subjects)
```bash
# Reserve 10 tasks for fMRIPrep and 10 for dependent XCP-D (20 total)
fmriprep_submission=$(qsub -terse -t 1-44 -tc 10 "${repo_root}/scripts/nih_rtms/fmriprep_nih_rtms_restfmri.sh")
fmriprep_job_id=${fmriprep_submission%%.*}
printf 'fMRIPrep job: %s\n' "${fmriprep_job_id}"

# Or run a single test subject (override embedded -t 1-44 with -t 1-1)
qsub -t 1-1 -tc 1 "${repo_root}/scripts/nih_rtms/fmriprep_nih_rtms_restfmri.sh" sub-D01
```

### Step 2: Run XCP-D Array (44 tasks, auto-skips sub-D16)
Queue XCP-D after submitting fMRIPrep, using matching task dependencies:
```bash
# Each XCP-D task waits only for its corresponding fMRIPrep task
xcpd_submission=$(qsub -terse -h -t 1-44 -tc 10 -hold_jid_ad "${fmriprep_job_id}" "${repo_root}/scripts/nih_rtms/xcpd_36P_nih_rtms_restfmri.sh")
xcpd_job_id=${xcpd_submission%%.*}
printf 'Held XCP-D job: %s\n' "${xcpd_job_id}"

# Or run a single test subject (override embedded -t 1-44 with -t 1-1)
qsub -t 1-1 -tc 1 "${repo_root}/scripts/nih_rtms/xcpd_36P_nih_rtms_restfmri.sh" sub-D01
```

Logs will be written to `/Shared/jianglab/3_Data_Working/Depression/NIH_rTMS/src/logs/`.
The array is initially user-held. Inspect `qstat -u yojiao`, validate each ready participant's exact inputs, and release individual tasks with `qrls JOB_ID.TASK_ID` only while capacity remains. With the example's 10-task fMRIPrep reservation, allow at most 10 unfinished released XCP-D tasks. Admit more ready participants as those tasks finish; a slow participant does not block other validated participants.
The task dependency normally waits for completion regardless of exit status; SGE exit code 100 keeps the corresponding dependent task held. Each XCP-D task checks its exact expected fMRIPrep derivatives and fails if an input is missing. Do not use `-hold_jid` for this paired workflow: it blocks all participants behind the slowest fMRIPrep task. Argon's JSV policy rejects changing `hold_jid_ad` with `qalter`, so set task dependencies at submission. When replacing an unstarted array, submit its replacement with a user hold (`-h`), verify the dependencies and concurrency cap, cancel the old unstarted array, then release only ready replacement tasks within the available budget.

### Explicit finalization repair

For an array that completed scientific processing and reports but lacks the
FreeSurfer root dictionaries, submit the validated finalization command with
the exact source array ID. Reserve one task within the global concurrency cap:

```bash
qsub -t 1-1 -tc 1 "${repo_root}/scripts/nih_rtms/finalize_fmriprep_nih_rtms.sh" "${fmriprep_job_id}"
```

The command validates the full cohort before publishing the missing dictionaries.
It rejects conflicting existing metadata and records source provenance and hashes
in its scheduler log. See the environment manifest for the pinned source and
resource contracts.
