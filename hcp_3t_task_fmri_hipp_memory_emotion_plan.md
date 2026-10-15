# HCP 3T hippocampal task analysis

## Current data and cohorts

EMOTION and WM each use the complete 144-participant task dataset at
`/home/yojiao/LSS/jianglab/3_Data_Working/HCP_tast_3T/`.
The independent HCP 7T resting-state reference cohort contains 151 participants.
HippUnfold anatomy uses 512 vertices per hippocampal hemisphere.

| Task | Runs | Frames per run | Primary comparison |
| --- | --- | --- | --- |
| EMOTION | LR, RL | 176 | Faces > shapes |
| WM | LR, RL | 405 | 2-back > 0-back, category effects, and load interactions |

The task run manifests are the authoritative declarations of run identity,
order, length, BOLD/CIFTI paths, EVs, and motion files:

- `outputs_hcp/inputs/hcp/kong2022/manifests/task-emotion/sub-*.json`
- `outputs_hcp/inputs/hcp/kong2022/manifests/task-wm/sub-*.json`

Each directory contains the full task cohort. Use the complete QC-passed
subject file for all subject-level and group commands:
`outputs_hcp/inputs/hcp/kong2022/shared_inputs/task-<task>/manifests/hcp_7t_hippocampus_fmri_qc_pass_t1only_144.txt`.

The HCP 7T reference roster is
`outputs_hcp/inputs/hcp/qc/vertex_qc_t1only/manifests/hcp_7t_hippocampus_fmri_qc_pass_t1only_151.txt`.

## Input and output layout

For `task-emotion` or `task-wm`, use:

- Task root: `outputs_hcp/inputs/hcp/kong2022/shared_inputs/task-<task>/`.
- Hippocampal preparation root: `<task-root>/surface/`.
- Activation products: `<task-root>/sub-<id>/task/<task>/<smoothing>/hemi_<L|R>/`.
- QC reports: `<task-root>/qc/`.
- Task subject lists: `<task-root>/manifests/`.
- Group statistics: `outputs_hcp/work/task_activation/task-<task>/group/`.
- Presentation: `outputs_hcp/present/task_activation/task-<task>/`.

Hippocampal time series are sampled from each declared BOLD volume onto the
subject's HippUnfold midthickness surface using Workbench enclosing mapping.
Cortical time series use task MSMAll CIFTIs and the subject's Kong2022 400-ROI
labels. Runs are concatenated in memory in the declared order. Missing,
misordered, or mismatched inputs are errors.

Task preparation uses its own `--route-root` and `--shared-root`. The
151-participant reference surfaces are reused from
`outputs_hcp/inputs/hcp/kong2022/templates/hipp_group_average_t1only_151/group_average_surfaces/`.

## Preparation and QC

Run `02_01_prepare_group_inputs.py` with the complete task roster and the
corresponding `--task-manifest-root`. The preparation stages are `raw`,
`tsnr`, `qc`, and `smooth`, using the same task-specific roots throughout.

Current task surfaces use `--tsnr-threshold none`: finite, nonconstant signal
is required in every declared run. Residual-tSNR measurements are recorded
under `runwise_baseline_over_task_residual_sd`, with no numerical tSNR cutoff.
QC additionally requires HippUnfold Dice >= 0.70 and an invalid-vertex
fraction < 0.20 in both hemispheres. Motion and model quality require their
own assessment.

Formal resting-state QC outputs and the reference average require the exact
HCP151 subject IDs. Task QC evaluates and writes the complete task cohort in
its task-specific directories.

## Activation model and regional analysis

`04_03_materialize_task_activation.py` fits subject-level hippocampal GLMs
using the declared LR/RL designs, the SPM canonical HRF, motion regressors,
run intercepts, and linear drifts. The model uses OLS without autoregressive
prewhitening. Invalid vertices remain NaN. Saved products include condition
betas, combined contrasts, and run-specific contrasts.

EMOTION compares the `fear` and `neut` conditions, displayed as Faces and
Shapes. WM models eight conditions: 0-back and 2-back for Faces, Places,
Tools, and Body. Its 13 maintained contrasts comprise the load main effect,
four category preferences, four load-by-category interactions, and four
category-specific load effects.

`05_06_build_task_activation_group.py` requires an explicit complete roster
through `--subjects-manifest` or `--subjects`. It uses all maintained
contrasts by default and requires every selected subject's condition betas
and selected contrasts. Missing required products cause an error.

The two regional representations are:

- `hcp151_atlas`: shared resting-state Anterior, Body, and Posterior labels.
- `template_matching`: individualized crossfit labels with the same functional
  identities; their spatial locations may vary across participants.

Regional measurements use each subject's native midthickness vertex areas.
A subject-region measurement enters statistics only when finite tissue covers
at least 10% of the region's supported area. Group tables record each
measurement's effective sample size and BH-FDR-adjusted p values. Paired
subregion comparisons use their complete paired observations.

Profile curves use 20 equal-vertex-count bins along the group surface's PCA
long axis, numbered anterior to posterior. These bins describe surface order.
Condition curves retain the contrast's fixed weights; missing required values
remain NaN. Beta values use the input signal's arbitrary units.

## Run the complete group analysis and presentation

Run from the repository root. Set `task_name` to `emotion` or `wm`:

```bash
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
task_python=/home/yojiao/.conda/envs/py314/bin/python
task_name=emotion
task_root="outputs_hcp/inputs/hcp/kong2022/shared_inputs/task-${task_name}"
subject_roster="${task_root}/manifests/hcp_7t_hippocampus_fmri_qc_pass_t1only_144.txt"

"$task_python" scripts/core_workflow/05_06_build_task_activation_group.py \
  --task "$task_name" --shared-root "$task_root" \
  --subjects-manifest "$subject_roster" --smoothings 2mm

"$task_python" scripts/core_workflow/06_11_render_task_activation_products.py \
  --task "$task_name"
```

The group directory contains `subject_cluster_table.tsv`,
`group_cluster_summary.tsv`, `coverage_exclusions.tsv`, group maps, and the
existing `group_build_manifest.json`.

The maintained 2mm figures are:

- `task-emotion/task_emotion.png`
- `task-wm/task_wm_load.png`
- `task-wm/task_wm_categories.png`
- `task-wm/task_wm_load_by_category.png`

Each task's presentation directory also contains its GIFTI overlays and
`task_activation_products_manifest.json`. Figure sample sizes come from the
corresponding statistics. Group activation and paired regional differences
do not establish that individualized parcellations are superior.

## Other analyses

`04_04_materialize_background_fc.py` supports exploratory hippocampus–cortex
background FC after task-design regression. It requires the matching task
cortical references from `03_02`. Its results have a separate interpretation
from activation contrasts.

Behavioral measures are available in
`data/HCP_YA_subjects_2026_04_09_16_01_42.csv`: task accuracy, response times,
completion measures, and hippocampal volumes. Brain–behavior models should
account for motion and family structure as appropriate to the analysis.
