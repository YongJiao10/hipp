# Network-First End-to-End: Raw to Migration Overview

This runbook is the single execution source for producing migration overviews from raw HCP 7T archives without re-analyzing code.

## Scope and Fixed Parameters

```text
subjects            102311, 102816
branch              network-prob-cluster-nonneg
atlas               kong2019
density             512
k-selection mode    updated (default) or legacy (explicit)
raw source          remote Mac external drive archive zip
raw landing         data/raw/<subject>/
staging input       data/hippunfold_input/sub-<subject>/
dense output        outputs/dense_corobl_batch/sub-<subject>/
migration output    outputs_migration/hipp_functional_parcellation_network/
present output      present_network_migration/
log file            logs/network_migration_102311_102816_<timestamp>.md
python env          py314
```

## K Selection Modes (Do Not Mix)

`updated` (post-update, default):
- Selection rule: local-minimum + 1-SE + non-triviality constraints.
- Includes `V_min` and connectivity constraints.
- Use when you explicitly want the new protocol.

`legacy` (pre-update):
- Selection rule: pick smallest `K` whose `null_corrected_score` is within `0.02` of best and `min_cluster_size_fraction >= 0.05`.
- No local-minimum / non-triviality gate.
- Use when you explicitly request old protocol reproduction.

Mode selection must be passed in commands via `--k-selection-mode`.
Never assume from context.

## Remote Access Rule

1. Resolve `MPM619.local` current IP first.
2. Connect by IP directly: `ssh yojiao@<ip>`.
3. Pull data from `Resting State fMRI 7T Preprocessed Recommended archive/*.zip`.

Example resolution and check:

```bash
IP=$(dns-sd -G v4v6 MPM619.local | awk '/Add/ && $NF ~ /([0-9]{1,3}\.){3}[0-9]{1,3}/ {ip=$NF} END{print ip}')
ssh yojiao@"${IP}" 'hostname; ls -la "/Volumes/Elements/HCP-YA-2025/Resting State fMRI 7T Preprocessed Recommended archive" | head'
```

## Step 1: Pull Raw Files From Archive Zip

Input:
- Remote archive zip for each subject

Command template:

```bash
source /opt/miniconda3/bin/activate py314
python scripts/copy_hcp_minimal.py \
  --remote-host yojiao@<ip> \
  --remote-root "/Volumes/Elements/HCP-YA-2025" \
  --subject <subject> \
  --outdir data/raw/<subject> \
  --manifest manifests/raw_copy/sub-<subject>_copy_manifest.json
```

Output:
- `data/raw/<subject>/sub-<subject>_rfMRI_REST_7T_Atlas_MSMAll_hp2000_clean_rclean_tclean.dtseries.nii`
- `data/raw/<subject>/sub-<subject>_rfMRI_REST_7T_hp2000_clean_rclean_tclean.nii.gz`
- `data/raw/<subject>/sub-<subject>_rfMRI_REST_7T_brain_mask.nii.gz`
- `data/raw/<subject>/sub-<subject>_T1w_acpc_dc_restore.nii.gz`
- `data/raw/<subject>/sub-<subject>_T2w_acpc_dc_restore.nii.gz`

Verify:

```bash
ls -lh data/raw/<subject>
```

## Step 2: Stage HippUnfold-Compatible Inputs

Input:
- `data/raw/<subject>/` files from Step 1

Command template:

```bash
source /opt/miniconda3/bin/activate py314
python scripts/stage_hippunfold_inputs.py \
  --subject <subject> \
  --source-dir data/raw/<subject> \
  --input-dir data/hippunfold_input
```

Disk-safe canonicalization (required after staging):

```bash
for s in 102311 102816; do
  rm -f data/hippunfold_input/sub-${s}/func/sub-${s}_task-rest_run-concat.dtseries.nii
  rm -f data/hippunfold_input/sub-${s}/func/sub-${s}_task-rest_run-concat_bold.nii.gz
  rm -f data/hippunfold_input/sub-${s}/func/sub-${s}_task-rest_run-concat_desc-brain_mask.nii.gz
  ln data/raw/${s}/sub-${s}_rfMRI_REST_7T_Atlas_MSMAll_hp2000_clean_rclean_tclean.dtseries.nii \
     data/hippunfold_input/sub-${s}/func/sub-${s}_task-rest_run-concat.dtseries.nii
  ln data/raw/${s}/sub-${s}_rfMRI_REST_7T_hp2000_clean_rclean_tclean.nii.gz \
     data/hippunfold_input/sub-${s}/func/sub-${s}_task-rest_run-concat_bold.nii.gz
  ln data/raw/${s}/sub-${s}_rfMRI_REST_7T_brain_mask.nii.gz \
     data/hippunfold_input/sub-${s}/func/sub-${s}_task-rest_run-concat_desc-brain_mask.nii.gz
done
```

Output:
- `data/hippunfold_input/sub-<subject>/func/sub-<subject>_task-rest_run-concat.dtseries.nii`
- `data/hippunfold_input/sub-<subject>/func/sub-<subject>_task-rest_run-concat_bold.nii.gz`
- `data/hippunfold_input/sub-<subject>/func/sub-<subject>_task-rest_run-concat_desc-brain_mask.nii.gz`

Verify:

```bash
ls -lh data/hippunfold_input/sub-<subject>/func
ls -li data/raw/<subject>/sub-<subject>_rfMRI_REST_7T_hp2000_clean_rclean_tclean.nii.gz \
       data/hippunfold_input/sub-<subject>/func/sub-<subject>_task-rest_run-concat_bold.nii.gz
```

## Step 3: Produce Dense Corobl Intermediate Assets

Input:
- Staged `data/hippunfold_input/sub-<subject>/`

Command:

```bash
source /opt/miniconda3/bin/activate py314
python scripts/run_dense_corobl_batch.py \
  --subjects 102311 102816 \
  --input-dir data/hippunfold_input \
  --out-root outputs/dense_corobl_batch \
  --density 512
```

Output (required intermediates):
- `outputs/dense_corobl_batch/sub-<subject>/hippunfold/sub-<subject>/surf/sub-<subject>_hemi-L_space-corobl_den-512_label-hipp_midthickness.surf.gii`
- `outputs/dense_corobl_batch/sub-<subject>/hippunfold/sub-<subject>/surf/sub-<subject>_hemi-R_space-corobl_den-512_label-hipp_midthickness.surf.gii`
- `outputs/dense_corobl_batch/sub-<subject>/post_dense_corobl/surface/sub-<subject>_hemi-L_space-corobl_den-512_label-hipp_bold.func.gii`
- `outputs/dense_corobl_batch/sub-<subject>/post_dense_corobl/surface/sub-<subject>_hemi-R_space-corobl_den-512_label-hipp_bold.func.gii`

Verify:

```bash
find outputs/dense_corobl_batch/sub-<subject> -maxdepth 6 -type f | rg 'den-512.*(midthickness\.surf\.gii|bold\.func\.gii)'
```

## Step 4: Run Network Migration and Copy Overviews

Input:
- Step 3 dense assets + staged dtseries

Command:

```bash
source /opt/miniconda3/bin/activate py314
python scripts/experiments/hipp_functional_parcellation_network/run_batch.py \
  --branches network-prob-cluster-nonneg \
  --atlases kong2019 \
  --subjects 102311 102816 \
  --k-selection-mode updated \
  --resume-mode force \
  --retain-level render \
  --cleanup-level none \
  --clear-present
```

Legacy reproduction command:

```bash
source /opt/miniconda3/bin/activate py314
python scripts/experiments/hipp_functional_parcellation_network/run_batch.py \
  --branches network-prob-cluster-nonneg \
  --atlases kong2019 \
  --subjects 102311 102816 \
  --k-selection-mode legacy \
  --resume-mode force \
  --retain-level render \
  --cleanup-level none \
  --clear-present
```

Output:
- `outputs_migration/hipp_functional_parcellation_network/network-prob-cluster-nonneg/kong2019/sub-<subject>/hipp_functional_parcellation_network_overview.png`
- `present_network_migration/sub-<subject>_kong2019_network-prob-cluster-nonneg_overview.png`

## Step 5: Final Acceptance

Required files for both subjects:
- `present_network_migration/sub-102311_kong2019_network-prob-cluster-nonneg_overview.png`
- `present_network_migration/sub-102816_kong2019_network-prob-cluster-nonneg_overview.png`
- `outputs_migration/.../sub-<subject>/final_selection_summary.json`
- `outputs_migration/.../sub-<subject>/summary_manifest.json`
- `outputs_migration/.../sub-<subject>/hipp_functional_parcellation_network_overview.png`

Verification command:

```bash
for s in 102311 102816; do
  test -f "present_network_migration/sub-${s}_kong2019_network-prob-cluster-nonneg_overview.png" || echo "missing present ${s}"
  root="outputs_migration/hipp_functional_parcellation_network/network-prob-cluster-nonneg/kong2019/sub-${s}"
  test -f "${root}/final_selection_summary.json" || echo "missing final_selection_summary ${s}"
  test -f "${root}/summary_manifest.json" || echo "missing summary_manifest ${s}"
  test -f "${root}/hipp_functional_parcellation_network_overview.png" || echo "missing overview ${s}"
done
```

## Failure Triage (Fast Map)

```text
Step 1 failure  -> remote archive zip missing or ssh/ip issue
Step 2 failure  -> raw files incomplete; concat_bold not staged
Step 3 failure  -> hippunfold/post step missing required staged files or den-512 assets
Step 4 failure  -> missing dense intermediates or atlas/cortex dependencies
Step 5 failure  -> run_batch incomplete copy/summary outputs
```

## Validated Example (This Round)

- Run date: 2026-04-10
- Subjects: 102311, 102816
- Present overview paths:
  - `present_network_migration/sub-102311_kong2019_network-prob-cluster-nonneg_overview.png`
  - `present_network_migration/sub-102816_kong2019_network-prob-cluster-nonneg_overview.png`
- Notes:
  - staging concat files are hard-linked to `data/raw` to avoid duplicate disk usage.
  - K selection was aligned to pre-update `legacy` mode (git `c3a1289` style) for this run.
