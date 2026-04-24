# fMRI QC

The default hippocampal group-analysis cohort is the 153-subject QC-passed set:

- `manifests/hcp_7t_hippocampus_fmri_qc_pass_153.txt`

QC criteria:

1. HippUnfold Dice `>= 0.70` for both hemispheres
2. fMRI null-vertex rate `< 20%` for both hemispheres

Excluded subjects:

- `sub-102816`
- `sub-109123`
- `sub-146129`
- `sub-155938`
- `sub-158136`
- `sub-164131`
- `sub-173334`
- `sub-198653`
- `sub-550439`
- `sub-680957`
- `sub-783462`
- `sub-898176`
- `sub-901442`

Active shared group-input root:

- `outputs_migration/hipp_group_inputs_shared/`

Example active raw surface path:

- `outputs_migration/hipp_group_inputs_shared/sub-<id>/surface/raw/sub-<id>_hemi-{L,R}_space-corobl_den-512_label-hipp_bold.func.gii`
