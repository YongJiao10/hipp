# fMRI QC

The active default hippocampal group-analysis cohort is the T1only 151-subject QC-passed set:

- `outputs_experiments/hippunfold_recommended_t1w_20260423/kong2022_route_b/manifests/hcp_7t_hippocampus_fmri_qc_pass_t1only_151.txt`
- `outputs_experiments/hippunfold_recommended_t1w_20260423/kong2022_route_b/manifests/hcp_7t_hippocampus_fmri_qc_pass_t1only_latest.txt`

Legacy `manifests/hcp_7t_hippocampus_fmri_qc_pass_153.txt` is T2only-era and is not the active default for T1only workflows.

QC criteria:

1. HippUnfold Dice `>= 0.70` for both hemispheres
2. fMRI null-vertex rate `< 20%` for both hemispheres

T1only uses the same thresholds, but the consumed HippUnfold surfaces and Dice TSVs come from:

- `outputs_experiments/hippunfold_recommended_t1w_20260423/dense_corobl_batch/`

Excluded subjects:

- `sub-102816`
- `sub-109123`
- `sub-155938`
- `sub-158136`
- `sub-164131`
- `sub-173334`
- `sub-182739`
- `sub-198653`
- `sub-550439`
- `sub-680957`
- `sub-783462`
- `sub-878776`
- `sub-878877`
- `sub-898176`
- `sub-901442`

Active shared group-input root:

- `outputs_migration/hipp_group_inputs_shared/`

Example active raw surface path:

- `outputs_migration/hipp_group_inputs_shared/sub-<id>/surface/raw/sub-<id>_hemi-{L,R}_space-corobl_den-512_label-hipp_bold.func.gii`
