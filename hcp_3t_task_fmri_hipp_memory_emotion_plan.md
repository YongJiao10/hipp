# HCP 3T hippocampal task analysis

## Current data and cohorts

EMOTION and WM each use the complete 159-participant task dataset at
`/home/yojiao/LSS/jianglab/3_Data_Working/HCP_tast_3T/`.
All 159 task participants have anatomy in the 166-participant HCP reconstruction.
The current HCP7T functional reference roster contains 162 participants.
The explicitly selected HCP151 atlas and crossfit products remain separate
analysis references; changing QC does not rebuild those downstream products.
HippUnfold anatomy uses 512 vertices per hippocampal hemisphere.

| Task | Runs | Frames per run | Primary comparison |
| --- | --- | --- | --- |
| EMOTION | LR, RL | 176 | Faces > shapes |
| WM | LR, RL | 405 | 2-back > 0-back, category effects, and load interactions |

QC consumes the exact run identities, BOLD paths, frame counts and motion files
in `outputs_hcp/inputs/kong2022/shared_inputs/task-<task>/manifests/hcp_3t_<task>_qc_runs.tsv`.
The task activation entrypoints separately consume their run-artifact manifests
for BOLD/CIFTI paths and EVs:

- `outputs_hcp/inputs/kong2022/manifests/task-emotion/sub-*.json`
- `outputs_hcp/inputs/kong2022/manifests/task-wm/sub-*.json`

Use the task-specific final QC roster for subject-level and group commands:
`outputs_hcp/inputs/kong2022/shared_inputs/task-<task>/manifests/hcp_3t_<task>_hippocampus_fmri_qc_pass_t1only_latest.txt`.
A QC roster does not establish that the selected activation inputs and products
have been rebuilt; those entrypoints still require their exact declared artifacts.

The HCP 7T reference roster is
`outputs_hcp/inputs/kong2022/manifests/hcp_7t_hippocampus_fmri_qc_pass_t1only_latest.txt`.

## Input and output layout

For `task-emotion` or `task-wm`, use:

- Task root: `outputs_hcp/inputs/kong2022/shared_inputs/task-<task>/`.
- Hippocampal preparation root: `<task-root>/surface/`.
- Activation products: `<task-root>/sub-<id>/task/<task>/<smoothing>/hemi_<L|R>/`.
- QC reports: `<task-root>/qc/`.
- Task subject lists: `<task-root>/manifests/`.
- Group statistics: `outputs_hcp/work/task_activation/task-<task>/group/`.
- Presentation: `outputs_hcp/present/task_activation/task-<task>/`.

Hippocampal time series require Workbench trilinear sampling on midthickness
surfaces transformed into the declared BOLD volume space. Coordinate sampling
uses the same registered NIfTI coverage calculation for rest and tasks.
Task sampling and QC write their own branch roots. Activation, smoothing and
group/presentation products require their separate maintained stages.
Cortical time series use task MSMAll CIFTIs and the subject's Kong2022 400-ROI
labels. Runs are concatenated in memory in the declared order. Missing,
misordered, or mismatched inputs are errors.

Task preparation uses its own `--route-root` and `--shared-root`. The
151-participant reference surfaces are reused from
`outputs_hcp/inputs/kong2022/templates/hipp_group_average_t1only_151/group_average_surfaces/`.

## Preparation and QC

Apply [the shared fMRI QC policy](docs/hipp_parcellation_network/fmri_qc.md)
separately within each task: overall mean Power FD <=0.50 mm, recorded official
v2.3.0 reconstruction plus bilateral Dice >=0.70, and <=20 unusable
NIfTI-sampled vertices per hemisphere. Never pool EMOTION and WM to pass FD.
All 159 participants pass task motion; 155 have completed v2.3.0 anatomy and
pass bilateral Dice before coverage selection. The other four are 182739
(right Dice <0.70), 233326, 249947 and 942658 (official workflow failures).

`02_01_prepare_group_inputs.py` and `02_02_build_qc_manifest.py` use the
shared `--qc-run-inputs` table. Set `--dataset hcp --task emotion|wm`, explicit
`--route-root` and `--shared-root`, `--tsnr-threshold none`, and
`--require-hippunfold-version 2.3.0`. Run preparation stages `raw` and `tsnr`
followed by `qc`; no task-design EVs or residual-tSNR computation are needed
for coverage QC. Both the participant count and persisted vertex masks use
positive NIfTI trilinear support and finite, nonconstant signal within each
physical run. Missing completion or source inputs remain unassessable.

EMOTION and WM sampling, vertex masks and final participant QC are complete:
all 155 motion/anatomy passes in each task also pass coverage, with zero
unusable vertices on each side. These QC runs do not rebuild
activation GLMs, smoothed signals or group statistics.

The task CIFTI headers declare a one-second SeriesAxis step, while paired BOLD
headers and the hippocampal GLM use TR=0.72 seconds. The known EMOTION terminal
truncation does not justify a uniform event shift. Neither observation is an
instruction to rerun the task model as part of documentation consolidation.

## Activation model and regional analysis

`04_03_materialize_task_activation.py` fits subject-level hippocampal GLMs
using the declared LR/RL designs, the SPM canonical HRF, motion regressors,
run intercepts, and per-run DCT drift terms with a fixed 128-second high-pass
cutoff period. The model uses OLS without autoregressive
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
long axis, numbered 1–20 from anterior to posterior. These bins describe
surface order. Vertices are averaged within each subject's bin, then subjects
are averaged for the plotted curves. Condition curves retain the contrast's
fixed weights; missing required values remain NaN. Beta values use the input
signal's arbitrary units.

Each contrast's bin means are tested against zero across subjects with a
two-sided one-sample t-test. BH-FDR correction covers all selected 2mm task
contrasts, both hemispheres, and all 20 bins as one family: 40 tests for the
full EMOTION build and 520 for WM. A narrow ribbon at the bottom of each
profile panel shows white for q ≥ .05 or untested bins, and progressively
darker gray for q < .05, .01, .001, and .0001. Adjacent bins at the same
level form continuous segments, and each figure has one shared legend
using `ns`, `+`, `*`, `**`, and `***`, consistent with bar/bracket symbols;
curves carry no significance symbols. The manifest-declared
`group_bin_summary.tsv` contains effective n, mean, t, p, and q per bin;
condition-beta rows provide the colored curves and are not in the correction
family. Cluster-bar and paired-subregion corrections remain separate.

## Run the complete group analysis and presentation

Run from the repository root. Set `task_name` to `emotion` or `wm`:

```bash
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
task_python=/home/yojiao/.conda/envs/py314/bin/python
task_name=emotion
task_root="outputs_hcp/inputs/kong2022/shared_inputs/task-${task_name}"
subject_roster="${task_root}/manifests/hcp_3t_${task_name}_hippocampus_fmri_qc_pass_t1only_latest.txt"

"$task_python" scripts/core_workflow/05_06_build_task_activation_group.py \
  --task "$task_name" --shared-root "$task_root" \
  --subjects-manifest "$subject_roster" --smoothings 2mm

"$task_python" scripts/core_workflow/06_11_render_task_activation_products.py \
  --task "$task_name"
```

The group directory contains `subject_cluster_table.tsv`,
`group_cluster_summary.tsv`, `group_bin_summary.tsv`, `coverage_exclusions.tsv`, group maps, and the
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
