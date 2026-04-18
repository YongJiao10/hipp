# Network-First End-to-End: Data Input to Migration Overview

This document is synced to the **current executable code paths** in this repository.
It does **not** use remote archive pulling as part of this runbook.

## Scope and Fixed Parameters

```text
subjects            102311, 102816
branch              network-prob-cluster-nonneg
atlas               kong2019
k-selection mode    fixed by launcher (mainline vs experimental)
run split mode      fixed by launcher: runwise
primary input       data/hippunfold_input/sub-<subject>/
input subdirs        anat/, func/ only
dense prerequisite  outputs_migration/dense_corobl_batch/sub-<subject>/hippunfold/sub-<subject>/...
cortex prerequisite outputs_migration/cortex_pfm/sub-<subject>/<atlas>/roi_components/...
migration output    outputs_migration/hipp_functional_parcellation_network/     (run_mainline.sh)
present output      present_network_migration/                                   (run_mainline.sh)
experimental output outputs_experimental/hipp_functional_parcellation_network/   (run_experimental.sh)
experimental present present_network_experimental/                                (run_experimental.sh)
python env          py314
canonical launchers scripts/hipp_parcellation_network/run_mainline.sh
                    scripts/hipp_parcellation_network/run_experimental.sh
low-level entrypoint scripts/hipp_parcellation_network/run_batch.py
```

## Authoritative Runtime Entry Points

- Canonical batch launchers (must use by default):
  - `scripts/hipp_parcellation_network/run_mainline.sh`
  - `scripts/hipp_parcellation_network/run_experimental.sh`
- Low-level batch entrypoint (debugging only):
  - `scripts/hipp_parcellation_network/run_batch.py`
- Subject worker:
  - `scripts/hipp_parcellation_network/run_subject.py`

Important defaults from code:
- `--input-root` defaults to `data/hippunfold_input`
- `--hippunfold-root` defaults to `outputs_migration/dense_corobl_batch`
- `--cortex-root` defaults to `outputs_migration/cortex_pfm`
- `--out-root` defaults to `outputs_migration/hipp_functional_parcellation_network`
- `--present-dir` defaults to `present_network_migration`

Batch launcher scope:
- `run_batch.py` (and fixed launchers that wrap it) now stop after branch outputs + present-copy updates.
- The batch path no longer runs trailing `plot_*tsnr*.py` figure generation as part of success criteria.

## Preflight (Hard Requirements)

Run this first. If any file is missing, do **not** proceed.

```bash
source /opt/miniconda3/bin/activate py314

for s in 102311 102816; do
  # Primary input (authoritative)
  test -f "data/hippunfold_input/sub-${s}/func/sub-${s}_task-rest_run-concat.dtseries.nii" || echo "missing dtseries ${s}"
  test -f "data/hippunfold_input/sub-${s}/func/sub-${s}_task-rest_run-concat_bold.nii.gz" || echo "missing concat_bold ${s}"
  test -f "data/hippunfold_input/sub-${s}/anat/sub-${s}_T1w.nii.gz" || echo "missing T1w ${s}"
  test -f "data/hippunfold_input/sub-${s}/anat/sub-${s}_T2w.nii.gz" || echo "missing T2w ${s}"

  # Dense corobl prerequisite consumed by run_subject.py
  test -f "outputs_migration/dense_corobl_batch/sub-${s}/hippunfold/sub-${s}/surf/sub-${s}_hemi-L_space-corobl_den-512_label-hipp_midthickness.surf.gii" || echo "missing dense L midthickness ${s}"
  test -f "outputs_migration/dense_corobl_batch/sub-${s}/hippunfold/sub-${s}/surf/sub-${s}_hemi-R_space-corobl_den-512_label-hipp_midthickness.surf.gii" || echo "missing dense R midthickness ${s}"
  test -f "outputs_migration/dense_corobl_batch/sub-${s}/hippunfold/sub-${s}/cifti/sub-${s}_space-corobl_den-512_label-hipp_atlas-multihist7_subfields.dlabel.nii" || echo "missing dense structural dlabel ${s}"

  # Cortex ROI prerequisite consumed by run_subject.py reference stage
  test -f "outputs_migration/cortex_pfm/sub-${s}/kong2019/roi_components/hemi_L/PFM_Kong2019priors.components.L.label.gii" || echo "missing cortex L ROI labels ${s}"
  test -f "outputs_migration/cortex_pfm/sub-${s}/kong2019/roi_components/hemi_R/PFM_Kong2019priors.components.R.label.gii" || echo "missing cortex R ROI labels ${s}"
  test -f "outputs_migration/cortex_pfm/sub-${s}/kong2019/roi_components/roi_component_stats.json" || echo "missing cortex ROI stats ${s}"
done
```

## K Selection Modes (Do Not Mix)

`mainline` (default production):
- smallest `K` within `0.02` of best `null_corrected_score`
- requires `min_cluster_size_fraction >= 0.05`

`experimental`:
- local-minimum + 1-SE + non-triviality constraints
- strict screening may skip combinations if no valid `K`

Always use the fixed launcher scripts. Do not hand-write `--k-selection-mode`, `--run-split-mode`, `--out-root`, or `--present-dir` for canonical runs.

## Step 1: Run Network Migration (Current Canonical Command)

```bash
scripts/hipp_parcellation_network/run_mainline.sh \
  --branches network-prob-cluster-nonneg \
  --atlases kong2019 \
  --subjects 102311 102816 \
  --resume-mode force \
  --retain-level render \
  --cleanup-level none \
  --clear-present
```

Experimental mode:

```bash
scripts/hipp_parcellation_network/run_experimental.sh \
  --branches network-prob-cluster-nonneg \
  --atlases kong2019 \
  --subjects 102311 102816 \
  --resume-mode force \
  --retain-level render \
  --cleanup-level none \
  --clear-present
```

## Step 2: Final Acceptance

Required files for both subjects:
- `present_network_migration/sub-102311_kong2019_network-prob-cluster-nonneg_overview.png`
- `present_network_migration/sub-102816_kong2019_network-prob-cluster-nonneg_overview.png`
- `outputs_migration/hipp_functional_parcellation_network/network-prob-cluster-nonneg/kong2019/sub-<subject>/final_selection_summary.json`
- `outputs_migration/hipp_functional_parcellation_network/network-prob-cluster-nonneg/kong2019/sub-<subject>/summary_manifest.json`
- `outputs_migration/hipp_functional_parcellation_network/network-prob-cluster-nonneg/kong2019/sub-<subject>/hipp_functional_parcellation_network_overview.png`

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

## Notes

- `data/hippunfold_input` is the authoritative analysis input tree.
- `run_subject.py` regenerates strict shared raw hippocampal `.func.gii` in
  `outputs_migration/hipp_functional_parcellation_network/_shared/.../surface/raw/`
  from `run-concat_bold` using `trilinear` mapping and `smooth_iters=0`.
- No archive fallback and no `_archive` reuse is part of this runbook.
