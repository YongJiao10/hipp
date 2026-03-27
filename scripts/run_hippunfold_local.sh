#!/bin/zsh
set -euo pipefail

SUBJECT="${1:-100610}"
INPUT_DIR="${2:-data/hippunfold_input}"
OUT_DIR="${3:-outputs/${SUBJECT}/hippunfold}"
AUTOTOP_LABELS="${HIPPUNFOLD_AUTOTOP_LABELS:-hipp dentate}"

source /opt/miniconda3/etc/profile.d/conda.sh >/dev/null 2>&1
conda activate hippo

export HOME="${HOME:-/tmp/hippo_home}"
if [[ "${HOME}" == "/Users/jy" ]]; then
  export HOME="/tmp/hippo_home"
fi
mkdir -p "${HOME}/Library/Caches/snakemake"
export HIPPUNFOLD_CACHE_DIR="${HIPPUNFOLD_CACHE_DIR:-/tmp/hippunfold_cache}"
export PATH="/Users/jy/Documents/HippoMaps/scripts:${PATH}"
export HIPPUNFOLD_CONDA_PREFIX="${HIPPUNFOLD_CONDA_PREFIX:-/tmp/hippo_snakemake_conda}"
mkdir -p "${HIPPUNFOLD_CONDA_PREFIX}"
export CONDA_SUBDIR="${CONDA_SUBDIR:-osx-64}"
export CONDA_PKGS_DIRS="${CONDA_PKGS_DIRS:-/tmp/hippo_conda_pkgs}"
mkdir -p "${CONDA_PKGS_DIRS}"

if [[ "$(uname -s)" == "Darwin" ]]; then
  export NNUNET_DISABLE_MULTIPROCESSING=1
  AUTOTOP_LABELS="${HIPPUNFOLD_AUTOTOP_LABELS:-hipp}"
  python scripts/patch_nnunet_compat.py --conda-prefix "${HIPPUNFOLD_CONDA_PREFIX}"
  python scripts/patch_laynii_compat.py \
    --hippunfold-site-root /opt/miniconda3/envs/hippo/lib/python3.11/site-packages/hippunfold \
    --runtime-source-cache /tmp/hippo_runtime_source_cache
  python scripts/patch_pyvista_compat.py \
    --hippunfold-site-root /opt/miniconda3/envs/hippo/lib/python3.11/site-packages/hippunfold \
    --runtime-source-cache /tmp/hippo_runtime_source_cache
fi

hippunfold \
  "${INPUT_DIR}" \
  "${OUT_DIR}" \
  participant \
  --modality T2w \
  --output-density 2mm \
  --autotop_labels ${=AUTOTOP_LABELS} \
  --participant-label "${SUBJECT}" \
  --use-conda \
  --conda-prefix "${HIPPUNFOLD_CONDA_PREFIX}" \
  --conda-frontend conda \
  --shared-fs-usage persistence software-deployment sources \
  --runtime-source-cache-path /tmp/hippo_runtime_source_cache \
  --cores 1
