# NIH rTMS Preprocessing Scripts (Argon SGE)

This directory contains the adapted Argon SGE array batch launchers for **fMRIPrep** and **XCP-D (36P)** on the merged NIH rTMS cohort (16 HV + 28 TRD, total 44 subjects).

These scripts adapt the established Jianglab pipeline from `fmriprep_processed_JJ_TMS/src/` and the workflow documentation `WorkingFlow_LSS_updated_07222026.pptx`.

---

## 1. Directory & File Overview

| File | Purpose | Description |
| :--- | :--- | :--- |
| `fmriprep_nih_rtms_restfmri.sh` | fMRIPrep SGE launcher | Runs fMRIPrep Singularity container on Argon compute nodes (`UI,PINC`). |
| `xcpd_36P_nih_rtms_restfmri.sh` | XCP-D SGE launcher | Runs XCP-D 36P nuisance regression, filtering (0.01-0.10 Hz), and censoring. |
| `sublist.txt` | Unified subject roster | 44 subjects (`sub-D01`..`sub-D28`, `sub-HVS1`..`sub-HVS17`). |

Both sets of files are synchronized between the Git repository (`scripts/nih_rtms/`) and the cluster shared storage:
`/Shared/jianglab/3_Data_Working/Depression/NIH_rTMS/src/`

---

## 2. Key Adaptations Made

1. **Unified Subject Roster (`sublist.txt`)**:
   - Both fMRIPrep and XCP-D read from a single, unified `sublist.txt` (44 subjects).
   - In XCP-D, an automated pre-check tests for preprocessed functional BOLD data. For anatomical-only subjects (e.g. `sub-D16`), XCP-D logs an informative message and exits cleanly (`exit 0`), ensuring the array job succeeds without manual list splitting.
2. **Input / Output Paths (Argon Environment Contract)**:
   - Root: `/Shared/jianglab/3_Data_Working/Depression/NIH_rTMS`
   - BIDS Input: `/Shared/jianglab/3_Data_Working/Depression/NIH_rTMS/bids_input`
   - fMRIPrep Derivatives: `/Shared/jianglab/3_Data_Working/Depression/NIH_rTMS/derivatives/restfmri`
   - XCP-D Output: `/Shared/jianglab/3_Data_Working/Depression/NIH_rTMS/xcpd/restfmri`
   - Logs: `/Shared/jianglab/3_Data_Working/Depression/NIH_rTMS/src/logs`
   - Working Directory: `/nfsscratch/Users/yojiao/fmriprep_NIH_rTMS_work/sub-${subject}`
3. **Flexible Subject Execution**:
   - Single-subject mode: accepts subject ID as argument `$1` (e.g., `qsub script.sh sub-D01` or `bash script.sh sub-D01`).
   - SGE array mode: reads line `$SGE_TASK_ID` from `sublist.txt` (`qsub -t 1-44 -tc 20 script.sh`).
4. **Argon Concurrency Cap Compliance**:
   - Strictly enforces `#$ -tc 20` for array jobs on Argon (compliant with repository rules).

---

## 3. How to Submit on Argon

Log into Argon (`ssh argon.hpc.uiowa.edu`), navigate to the dataset source directory:

```bash
cd /Shared/jianglab/3_Data_Working/Depression/NIH_rTMS/src
```

### Step 1: Run fMRIPrep Array (44 subjects)
```bash
# Submit full array (up to 20 concurrent tasks)
qsub fmriprep_nih_rtms_restfmri.sh

# Or run a single test subject
qsub fmriprep_nih_rtms_restfmri.sh sub-D01
```

### Step 2: Run XCP-D Array (44 tasks, auto-skips sub-D16)
Once fMRIPrep completes for the cohort:
```bash
# Submit full XCP-D array
qsub xcpd_36P_nih_rtms_restfmri.sh

# Or run a single test subject
qsub xcpd_36P_nih_rtms_restfmri.sh sub-D01
```

Logs will be written to `/Shared/jianglab/3_Data_Working/Depression/NIH_rTMS/src/logs/`.
