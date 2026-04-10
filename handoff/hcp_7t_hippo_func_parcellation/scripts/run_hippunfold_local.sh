#!/bin/zsh
set -euo pipefail

SUBJECT="${1:-100610}"
INPUT_DIR="${2:-data/hippunfold_input}"
OUT_DIR="${3:-outputs/${SUBJECT}/hippunfold}"
AUTOTOP_LABELS="${HIPPUNFOLD_AUTOTOP_LABELS:-hipp dentate}"
BUNDLE_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
HIPPUNFOLD_ENV_NAME="${HIPPUNFOLD_ENV_NAME:-hippo2}"
HIPPUNFOLD_OUTPUT_DENSITY="${HIPPUNFOLD_OUTPUT_DENSITY:-512}"

source /opt/miniconda3/etc/profile.d/conda.sh >/dev/null 2>&1
conda activate "${HIPPUNFOLD_ENV_NAME}"

export HOME="${HOME:-/tmp/hippo_home}"
mkdir -p "${HOME}/Library/Caches/snakemake"
export HIPPUNFOLD_CACHE_DIR="${HIPPUNFOLD_CACHE_DIR:-/tmp/hippunfold_cache}"
export PATH="${BUNDLE_ROOT}/scripts:${PATH}"
export HIPPUNFOLD_CONDA_PREFIX="${HIPPUNFOLD_CONDA_PREFIX:-/tmp/hippo_snakemake_conda}"
mkdir -p "${HIPPUNFOLD_CONDA_PREFIX}"
export CONDA_SUBDIR="${CONDA_SUBDIR:-osx-64}"
export CONDA_PKGS_DIRS="${CONDA_PKGS_DIRS:-/tmp/hippo_conda_pkgs}"
mkdir -p "${CONDA_PKGS_DIRS}"
export HIPPUNFOLD_RUNTIME_SOURCE_CACHE="${HIPPUNFOLD_RUNTIME_SOURCE_CACHE:-/tmp/hippo_runtime_source_cache}"
mkdir -p "${HIPPUNFOLD_RUNTIME_SOURCE_CACHE}"

HIPPUNFOLD_SITE_ROOT="$(
python - <<'PY'
import inspect
from pathlib import Path
import hippunfold

print(Path(inspect.getfile(hippunfold)).resolve().parent)
PY
)"

if [[ "$(uname -s)" == "Darwin" ]]; then
  export NNUNET_DISABLE_MULTIPROCESSING=1
  AUTOTOP_LABELS="${HIPPUNFOLD_AUTOTOP_LABELS:-hipp}"
  if [[ -f "${BUNDLE_ROOT}/scripts/patch_nnunet_compat.py" ]]; then
    python "${BUNDLE_ROOT}/scripts/patch_nnunet_compat.py" --conda-prefix "${HIPPUNFOLD_CONDA_PREFIX}"
  fi
  if [[ -f "${BUNDLE_ROOT}/scripts/patch_laynii_compat.py" ]]; then
    python "${BUNDLE_ROOT}/scripts/patch_laynii_compat.py" \
      --hippunfold-site-root "${HIPPUNFOLD_SITE_ROOT}" \
      --runtime-source-cache "${HIPPUNFOLD_RUNTIME_SOURCE_CACHE}"
  fi
  if [[ -f "${BUNDLE_ROOT}/scripts/patch_pyvista_compat.py" ]]; then
    python "${BUNDLE_ROOT}/scripts/patch_pyvista_compat.py" \
      --hippunfold-site-root "${HIPPUNFOLD_SITE_ROOT}" \
      --runtime-source-cache "${HIPPUNFOLD_RUNTIME_SOURCE_CACHE}"
  fi
fi

hippunfold \
  "${INPUT_DIR}" \
  "${OUT_DIR}" \
  participant \
  --modality T2w \
  --output-density "${HIPPUNFOLD_OUTPUT_DENSITY}" \
  --output-spaces corobl \
  --autotop_labels ${=AUTOTOP_LABELS} \
  --participant-label "${SUBJECT}" \
  --use-conda \
  --conda-prefix "${HIPPUNFOLD_CONDA_PREFIX}" \
  --conda-frontend conda \
  --runtime-source-cache-path "${HIPPUNFOLD_RUNTIME_SOURCE_CACHE}" \
  --cores 1
