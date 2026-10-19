# HCP 3T task inputs and current completeness

EMOTION and WM use the complete 144-participant task cohort at
`/home/yojiao/LSS/jianglab/3_Data_Working/HCP_tast_3T/`.
Every participant has LR and RL. The task cohort is a subset of the 151-person
HCP 7T resting-state reference cohort; 151 is not the task denominator.

Resolve every run from its exact subject declaration:

- `outputs_hcp/inputs/hcp/kong2022/manifests/task-emotion/sub-<id>.json`
- `outputs_hcp/inputs/hcp/kong2022/manifests/task-wm/sub-<id>.json`

Do not select another variant by filename order or existence. Current task
rosters live in each task's `shared_inputs/task-<task>/manifests/` directory
and are named `hcp_7t_hippocampus_fmri_qc_pass_t1only_144.txt`.

## Required inputs per declared run

| Input | Exact variant / contract | Purpose |
| --- | --- | --- |
| Hippocampal source BOLD | `tfMRI_<TASK>_<LR-or-RL>_hp0_clean_rclean_tclean.nii.gz`; 176 frames for EMOTION, 405 for WM | Sampling to hippocampal surfaces |
| Cortex CIFTI | `tfMRI_<TASK>_<LR-or-RL>_Atlas_MSMAll_hp0_clean_rclean_tclean.dtseries.nii`; same frames, 91,282 grayordinates | Cortex ROI time series |
| EMOTION conditions | `EVs/fear.txt`, `EVs/neut.txt` | Faces/shapes regressors |
| WM conditions | `EVs/0bk_<category>.txt`, `EVs/2bk_<category>.txt`; categories faces, places, tools, body | Eight category-by-load regressors |
| Movement regressors | `Movement_Regressors.txt`; frames × 12 finite values | Model nuisance regressors |
| Displacement traces | `Movement_RelativeRMS.txt`, `Movement_AbsoluteRMS.txt`; one finite value per frame | Motion assessment; RMS is not Power FD |

Condition EVs contain three finite columns: onset, duration, and weight,
with times in seconds. WM cue and correct/error/no-response EVs are additional
inputs for a model that explicitly includes those distinctions. Subject-level
behavior is available in `data/HCP_YA_subjects_2026_04_09_16_01_42.csv`;
condition accuracy, response times, and completion need separate assessment.

## Observed state and unresolved timing

The complete-cohort inspection found all required source files, the expected
image and movement-array dimensions, finite condition EVs, and matching
LR/RL sampling declarations. Sampling summaries record trilinear mapping on
den-512 hippocampal surfaces.

All consumed task CIFTIs currently declare a one-second SeriesAxis step,
whereas every paired BOLD header and the current hippocampal GLM use 0.72 seconds.
Reconcile source/acquisition timing before using a header-driven task model.
EMOTION's final fear EV extends beyond the acquisition window; the official
HCP protocol documents missing final trials. Interpret the actual final-block
timing without imposing an unsupported global event shift.

Current task masks use `--tsnr-threshold none`; residual-tSNR scores were not
computed. File completeness and basic signal validity are prerequisites,
not evidence of acceptable motion, local signal, or task reliability.
See [the complete QC review](tsnr_qc_literature_review.md) for the primary
sources, precise observed scope, and proposed next analysis, and
[the task workflow](hcp_3t_task_fmri_hipp_memory_emotion_plan.md) for current
analysis and output paths.
