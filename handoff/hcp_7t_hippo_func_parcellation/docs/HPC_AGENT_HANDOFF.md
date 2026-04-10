# HPC Agent Handoff

## Scope

This bundle is the minimized network-first HPC package for batch-running the 166-subject cohort with:

- branches: `network-gradient`, `network-prob-cluster-nonneg`
- atlases: `lynch2024`, `kong2019`

The bundle keeps flow code, static templates, Schaefer CIFTI atlas, FASTANS code, and the two prior families. It intentionally excludes any precomputed output trees.

## Directory Layout

```text
network_first_166_bundle/
  AGENTS.md
  README.md
  config/
  data/atlas/schaefer400/
  docs/
  external/FASTANS/
  manifests/
  scripts/
```

## Required Tools

```text
Tool         Requirement
Python       Use the target runtime for nibabel/numpy/scipy/sklearn/Pillow/matplotlib
HippUnfold   Must be runnable as `hippunfold`
Workbench    Must expose `wb_command`
FASTANS      Code/resources are bundled under `external/FASTANS`
```

## Environment Bootstrap

The bundle includes:

- [environment.yml](/Users/jy/Documents/HippoMaps-network-first/handoff/network_first_166_bundle/environment.yml)
- [setup_hpc_env.sh](/Users/jy/Documents/HippoMaps-network-first/handoff/network_first_166_bundle/setup_hpc_env.sh)

Recommended first step on HPC:

```bash
bash setup_hpc_env.sh
source activate_hpc_env.sh
```

By default, the setup script:

- creates conda env `hippo`
- installs bundle Python dependencies
- installs `hippomaps==0.1.17`
- installs `hippunfold` from conda channels `khanlab`, `conda-forge`, `bioconda`
- writes `activate_hpc_env.sh`

## How To Install `hippomaps`

Two supported routes are documented in the setup script.

### Option A: PyPI

This is the default in `setup_hpc_env.sh`.

```bash
pip install hippomaps==0.1.17
```

This version pin is based on the current PyPI release metadata we checked during bundle preparation.

### Option B: Git checkout

If you want a source checkout instead of the PyPI wheel:

```bash
git clone https://github.com/HippAI/hippomaps.git
pip install -e hippomaps
```

The bundle setup script supports this route with:

```bash
HIPPOMAPS_INSTALL_METHOD=git \
HIPPOMAPS_GIT_REF=main \
bash setup_hpc_env.sh
```

## How To Install `hippunfold`

We support two HPC-friendly routes.

### Option A: Conda

This is the default in `setup_hpc_env.sh`, matching the HippUnfold quickstart we checked.

```bash
conda create --name hippunfold-env -c khanlab -c conda-forge -c bioconda hippunfold
conda activate hippunfold-env
hippunfold -h
```

Inside this bundle we instead install it into the shared `hippo` env:

```bash
bash setup_hpc_env.sh
source activate_hpc_env.sh
hippunfold -h
```

### Option B: Apptainer / Singularity

This is often the better fit on HPC when admins prefer containerized neuroimaging tools.

```bash
HIPPUNFOLD_INSTALL_METHOD=apptainer \
HIPPUNFOLD_APPTAINER_IMAGE=/path/to/hippunfold_latest.sif \
bash setup_hpc_env.sh
```

The setup script will pull:

```text
docker://khanlab/hippunfold:latest
```

unless you override `HIPPUNFOLD_APPTAINER_URI`.

## How To Install Connectome Workbench

Workbench is not bundled as a binary because the correct package depends on the HPC OS and local policy.

Supported setup paths:

### Option A: HPC already provides `wb_command`

If your cluster has a module or system install, load it first and then run:

```bash
WB_COMMAND_BIN=$(command -v wb_command) bash setup_hpc_env.sh
```

### Option B: Install from an official Linux archive

The setup script can unpack a pre-downloaded archive:

```bash
INSTALL_WORKBENCH=1 \
WORKBENCH_ARCHIVE=/path/to/workbench-linux.zip \
bash setup_hpc_env.sh
```

or download from a URL you provide:

```bash
INSTALL_WORKBENCH=1 \
WORKBENCH_URL='https://.../workbench-linux.zip' \
bash setup_hpc_env.sh
```

After setup, `activate_hpc_env.sh` exports `WB_COMMAND_BIN`.

If `wb_command` is not at the mac default path, set:

```bash
export WB_COMMAND_BIN=/path/to/wb_command
```

If the fallback helper should use a specific Python binary, set:

```bash
export WB_FALLBACK_PY=/path/to/python
```

## Input Contract

### Preferred mode

If inputs are already staged, point the bundle at a `data/hippunfold_input`-style tree:

```text
data/hippunfold_input/sub-<id>/anat/sub-<id>_T1w.nii.gz
data/hippunfold_input/sub-<id>/anat/sub-<id>_T2w.nii.gz
data/hippunfold_input/sub-<id>/anat/sub-<id>_hemi-L_space-fsLR_den-32k_desc-MSMAll_midthickness.surf.gii
data/hippunfold_input/sub-<id>/anat/sub-<id>_hemi-R_space-fsLR_den-32k_desc-MSMAll_midthickness.surf.gii
data/hippunfold_input/sub-<id>/anat/sub-<id>_hemi-L_space-fsLR_den-32k_desc-MSMAll_inflated.surf.gii
data/hippunfold_input/sub-<id>/anat/sub-<id>_hemi-R_space-fsLR_den-32k_desc-MSMAll_inflated.surf.gii
data/hippunfold_input/sub-<id>/anat/sub-<id>_space-fsLR_den-32k_desc-MSMAll_sulc.dscalar.nii
data/hippunfold_input/sub-<id>/func/sub-<id>_task-rest_run-concat.dtseries.nii
data/hippunfold_input/sub-<id>/func/sub-<id>_task-rest_run-concat_bold.nii.gz
data/hippunfold_input/sub-<id>/func/sub-<id>_task-rest_run-concat_desc-brain_mask.nii.gz
data/hippunfold_input/sub-<id>/func/sub-<id>_task-rest_run-{1..4}.dtseries.nii
data/hippunfold_input/sub-<id>/func/sub-<id>_task-rest_run-{1..4}_bold.nii.gz
```

### Optional staging mode

If you start from per-subject raw source directories, use `--stage-source-dir-template` and let the top-level driver call `stage_hippunfold_inputs.py` first.

## Recommended Commands

### 1. Smoke run from already staged inputs

```bash
python scripts/run_network_first_166_bundle.py \
  --subjects 100610 \
  --skip-stage \
  --input-root /path/to/data/hippunfold_input \
  --hippunfold-root /path/to/outputs/dense_corobl_batch \
  --cortex-root /path/to/outputs/cortex_pfm \
  --parcellation-root /path/to/outputs/hipp_functional_parcellation_network \
  --resume-mode force
```

### 2. Smoke run with raw-source staging

```bash
python scripts/run_network_first_166_bundle.py \
  --subjects 100610 \
  --stage-source-dir-template /path/to/raw/sub-{subject} \
  --input-root /path/to/data/hippunfold_input \
  --hippunfold-root /path/to/outputs/dense_corobl_batch \
  --cortex-root /path/to/outputs/cortex_pfm \
  --parcellation-root /path/to/outputs/hipp_functional_parcellation_network \
  --resume-mode force
```

### 3. Full 166-subject batch

```bash
python scripts/run_network_first_166_bundle.py \
  --skip-stage \
  --subjects-file manifests/hcp_7t_hippocampus_struct_complete_166.txt \
  --input-root /path/to/data/hippunfold_input \
  --hippunfold-root /path/to/outputs/dense_corobl_batch \
  --cortex-root /path/to/outputs/cortex_pfm \
  --parcellation-root /path/to/outputs/hipp_functional_parcellation_network \
  --resume-mode resume
```

## Stage-by-Stage Outputs

```text
Stage                 Root                                            Main outputs
stage_inputs          data/hippunfold_input/                          staged anat/func tree
dense_corobl_batch    outputs/dense_corobl_batch/sub-<id>/           HippUnfold surf + post_dense_corobl products
cortex_pfm            outputs/cortex_pfm/sub-<id>/<atlas>/           PFM labels + roi_components
network_parcellation  outputs/hipp_functional_parcellation_network/  final K summaries + overview + features/clustering
```

## Common Failure Points

1. Missing fsLR cortical anatomy in `input-root/anat`.
2. `hippunfold` unavailable on `PATH` or wrong runtime.
3. `wb_command` missing or incompatible with the runtime host.
4. FASTANS code/resources not found under `external/FASTANS`.
5. `cortex_pfm` exists but `roi_components` were not derived.
6. `run_batch.py` is pointed at the wrong `hippunfold-root` or `cortex-root`.
7. `hippunfold` is installed but its model cache or runtime container path is not writable on the cluster.

## Validation Checklist

After a successful full run:

- the top-level driver resolves all 166 subjects from the manifest
- both atlases finish `cortex_pfm` and `roi_components`
- both branches finish `final_selection_summary.json`
- each `subject x atlas x branch` has an overview PNG and `k_selection_curves.png`
- `features/` and `clustering/` remain present for downstream analysis
