# NIH rTMS Preprocessing Scripts (Argon SGE)

This directory contains the adapted Argon SGE array batch launchers for **fMRIPrep** and **XCP-D (36P)** on the merged NIH rTMS cohort (16 HV + 28 TRD, total 44 subjects).

These scripts adapt the established Jianglab pipeline from `fmriprep_processed_JJ_TMS/src/` and the workflow documentation `WorkingFlow_LSS_updated_07222026.pptx`.

---

## 1. Directory & File Overview

| File | Purpose | Description |
| :--- | :--- | :--- |
| `fmriprep_nih_rtms_restfmri.sh` | fMRIPrep SGE launcher | Runs fMRIPrep Singularity container on Argon (`UI,PINC`), outputting both `MNI152NLin6Asym:res-2` volumetric and `fsLR den-91k` CIFTI/surface derivatives with FreeSurfer `recon-all`, slice timing correction, and `--longitudinal` T1w templates. |
| `xcpd_36P_nih_rtms_restfmri.sh` | XCP-D SGE launcher | Runs XCP-D 36P nuisance regression, filtering (0.01-0.10 Hz), and censoring (`FD > 0.5 mm`) in both `cifti` (`xcpd_surf/restfmri`) and `nifti` (`xcpd/restfmri`) modes with isolated per-subject scratch workdirs. |
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
   - SGE array mode: reads line `$SGE_TASK_ID` from the shared `sublist.txt`. Both arrays use `-tc 20` on Argon.
5. **Argon Concurrency and FreeSurfer Outputs**:
   - fMRIPrep uses a separate persistent FreeSurfer `SUBJECTS_DIR` for each participant under `derivatives/restfmri/sourcedata/freesurfer_by_subject/sub-<ID>/`. This keeps concurrent template initialization isolated while retaining each subject's recon-all outputs.
   - fMRIPrep and XCP-D arrays may each use `-tc 20`; submit XCP-D with a whole-array dependency on fMRIPrep so the stages do not overlap. Account for other active jobs so the total task count remains at most 20.
6. **Scratch Cleanup**:
   - Both launchers use `--clean-workdir` for automatic cleanup of their scratch workdirs after successful processing. Obtain explicit user approval for that cleanup before submitting a batch, as required by the repository deletion safeguard.

---

## 3. How to Submit on Argon

Log into Argon (`ssh argon.hpc.uiowa.edu`) and set the path to this repository checkout:

```bash
repo_root=$(git -C /Users/yojiao/hipp rev-parse --show-toplevel)
```

### Step 1: Run fMRIPrep Array (44 subjects)
```bash
# Submit the array (up to 20 concurrent subjects) and capture its job ID
fmriprep_submission=$(qsub -terse -t 1-44 -tc 20 "${repo_root}/scripts/nih_rtms/fmriprep_nih_rtms_restfmri.sh")
fmriprep_job_id=${fmriprep_submission%%.*}
printf 'fMRIPrep job: %s\n' "${fmriprep_job_id}"

# Or run a single test subject (override embedded -t 1-44 with -t 1-1)
qsub -t 1-1 -tc 1 "${repo_root}/scripts/nih_rtms/fmriprep_nih_rtms_restfmri.sh" sub-D01
```

### Step 2: Run XCP-D Array (44 tasks, auto-skips sub-D16)
Queue XCP-D after submitting fMRIPrep, using its whole-array dependency:
```bash
# Hold all XCP-D tasks until the entire fMRIPrep array has finished
qsub -t 1-44 -tc 20 -hold_jid "${fmriprep_job_id}" "${repo_root}/scripts/nih_rtms/xcpd_36P_nih_rtms_restfmri.sh"

# Or run a single test subject (override embedded -t 1-44 with -t 1-1)
qsub -t 1-1 -tc 1 "${repo_root}/scripts/nih_rtms/xcpd_36P_nih_rtms_restfmri.sh" sub-D01
```

Logs will be written to `/Shared/jianglab/3_Data_Working/Depression/NIH_rTMS/src/logs/`.
The dependency waits for job completion, regardless of exit status. Each XCP-D task checks its exact expected fMRIPrep derivatives and fails if an input is missing.
