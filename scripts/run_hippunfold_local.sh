#!/bin/zsh
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
SUBJECT="${1:-100610}"
INPUT_DIR="${2:-data/hippunfold_input}"
HIPPOMAPS_OUTPUT_ROOT="${HIPPOMAPS_OUTPUT_ROOT:-outputs_migration}"
OUT_DIR="${3:-${HIPPOMAPS_OUTPUT_ROOT}/${SUBJECT}/hippunfold}"
AUTOTOP_LABELS="${HIPPUNFOLD_AUTOTOP_LABELS:-hipp dentate}"
HIPPUNFOLD_ENV_NAME="${HIPPUNFOLD_ENV_NAME:-hippo2}"
HIPPUNFOLD_OUTPUT_DENSITY="${HIPPUNFOLD_OUTPUT_DENSITY:-512}"

source /opt/miniconda3/etc/profile.d/conda.sh >/dev/null 2>&1
conda activate "${HIPPUNFOLD_ENV_NAME}"

export HOME="${HOME:-/tmp/hippo_home}"
if [[ "${HOME}" == "/Users/jy" ]]; then
  export HOME="/tmp/hippo_home"
fi
mkdir -p "${HOME}/Library/Caches/snakemake"
export HIPPUNFOLD_CACHE_DIR="${HIPPUNFOLD_CACHE_DIR:-/tmp/hippunfold_cache}"
export PATH="${REPO_ROOT}/scripts:${PATH}"
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
  python "${REPO_ROOT}/scripts/patch_nnunet_compat.py" --conda-prefix "${HIPPUNFOLD_CONDA_PREFIX}"
  python "${REPO_ROOT}/scripts/patch_laynii_compat.py" \
    --hippunfold-site-root "${HIPPUNFOLD_SITE_ROOT}" \
    --runtime-source-cache "${HIPPUNFOLD_RUNTIME_SOURCE_CACHE}"
  python "${REPO_ROOT}/scripts/patch_pyvista_compat.py" \
    --hippunfold-site-root "${HIPPUNFOLD_SITE_ROOT}" \
    --runtime-source-cache "${HIPPUNFOLD_RUNTIME_SOURCE_CACHE}"
  python "${REPO_ROOT}/scripts/patch_pyvista_env_compat.py" \
    --hippunfold-site-root "${HIPPUNFOLD_SITE_ROOT}" \
    --runtime-source-cache "${HIPPUNFOLD_RUNTIME_SOURCE_CACHE}"
  python "${REPO_ROOT}/scripts/patch_get_boundary_vertices_compat.py" \
    --hippunfold-site-root "${HIPPUNFOLD_SITE_ROOT}" \
    --runtime-source-cache "${HIPPUNFOLD_RUNTIME_SOURCE_CACHE}"
  python "${REPO_ROOT}/scripts/patch_subfields_shell_compat.py" \
    --hippunfold-site-root "${HIPPUNFOLD_SITE_ROOT}" \
    --runtime-source-cache "${HIPPUNFOLD_RUNTIME_SOURCE_CACHE}"
  python "${REPO_ROOT}/scripts/patch_gen_volume_tsv_compat.py" \
    --hippunfold-site-root "${HIPPUNFOLD_SITE_ROOT}" \
    --runtime-source-cache "${HIPPUNFOLD_RUNTIME_SOURCE_CACHE}"
  python "${REPO_ROOT}/scripts/patch_workbench_compat.py" \
    --hippunfold-site-root "${HIPPUNFOLD_SITE_ROOT}" \
    --runtime-source-cache "${HIPPUNFOLD_RUNTIME_SOURCE_CACHE}"
  python "${REPO_ROOT}/scripts/patch_neurovis_compat.py" \
    --hippunfold-site-root "${HIPPUNFOLD_SITE_ROOT}" \
    --runtime-source-cache "${HIPPUNFOLD_RUNTIME_SOURCE_CACHE}"
  python "${REPO_ROOT}/scripts/patch_pyunfold_compat.py" \
    --hippunfold-site-root "${HIPPUNFOLD_SITE_ROOT}" \
    --runtime-source-cache "${HIPPUNFOLD_RUNTIME_SOURCE_CACHE}"
  python "${REPO_ROOT}/scripts/patch_shape_inject_shell_compat.py" \
    --hippunfold-site-root "${HIPPUNFOLD_SITE_ROOT}" \
    --runtime-source-cache "${HIPPUNFOLD_RUNTIME_SOURCE_CACHE}"
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
