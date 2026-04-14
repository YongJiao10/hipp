#!/bin/zsh
set -euo pipefail

SUBJECT="${1:-100610}"
INPUT_DIR="${2:-data/hippunfold_input}"
OUT_DIR="${3:-outputs_migration/dense_corobl_batch/sub-${SUBJECT}/hippunfold}"
AUTOTOP_LABELS="${HIPPUNFOLD_AUTOTOP_LABELS:-hipp dentate}"
BUNDLE_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
HIPPUNFOLD_ENV_NAME="${HIPPUNFOLD_ENV_NAME:-hippo2}"
HIPPUNFOLD_OUTPUT_DENSITY="${HIPPUNFOLD_OUTPUT_DENSITY:-512}"
HIPPUNFOLD_USE_CONDA="${HIPPUNFOLD_USE_CONDA:-0}"
HIPPUNFOLD_WORKFLOW_PROFILE="${HIPPUNFOLD_WORKFLOW_PROFILE:-none}"
HIPPUNFOLD_SDM="${HIPPUNFOLD_SDM:-env-modules}"
HIPPUNFOLD_EXTERNAL_BIN_DIR="${HIPPUNFOLD_EXTERNAL_BIN_DIR:-/Applications/ITK-SNAP.app/Contents/bin}"
CACHE_ROOT="${HIPPUNFOLD_CACHE_ROOT:-${BUNDLE_ROOT}/runtime/hippunfold_cache}"

source /opt/miniconda3/etc/profile.d/conda.sh >/dev/null 2>&1
conda activate "${HIPPUNFOLD_ENV_NAME}"

cleanup_runtime_dirs() {
  local home_cache="${CACHE_ROOT}/home"
  if [[ -d "${home_cache}" ]]; then
    find "${home_cache}" -type d -empty -delete 2>/dev/null || true
  fi
}
trap cleanup_runtime_dirs EXIT

export XDG_CACHE_HOME="${XDG_CACHE_HOME:-${CACHE_ROOT}/xdg_cache}"
export HIPPUNFOLD_CACHE_DIR="${HIPPUNFOLD_CACHE_DIR:-${CACHE_ROOT}/hippunfold_cache}"
export PATH="${BUNDLE_ROOT}/scripts:${PATH}"
if [[ -d "${HIPPUNFOLD_EXTERNAL_BIN_DIR}" ]]; then
  export PATH="${HIPPUNFOLD_EXTERNAL_BIN_DIR}:${PATH}"
fi
export HIPPUNFOLD_CONDA_PREFIX="${HIPPUNFOLD_CONDA_PREFIX:-${CACHE_ROOT}/snakemake_conda}"
mkdir -p "${HIPPUNFOLD_CONDA_PREFIX}"
export CONDA_SUBDIR="${CONDA_SUBDIR:-osx-64}"
export CONDA_PKGS_DIRS="${CONDA_PKGS_DIRS:-${CACHE_ROOT}/conda_pkgs}"
mkdir -p "${CONDA_PKGS_DIRS}"
export HIPPUNFOLD_RUNTIME_SOURCE_CACHE="${HIPPUNFOLD_RUNTIME_SOURCE_CACHE:-${CACHE_ROOT}/runtime_source_cache}"
laynii_bin="$(find "${HIPPUNFOLD_CONDA_PREFIX}" -maxdepth 3 -type f -name LN2_LAYERS 2>/dev/null | head -n 1 || true)"
if [[ -n "${laynii_bin}" ]]; then
  export HIPPUNFOLD_LN2_LAYERS_BIN="${HIPPUNFOLD_LN2_LAYERS_BIN:-${laynii_bin}}"
  export PATH="${PATH}:$(dirname "${laynii_bin}")"
fi
export KMP_DUPLICATE_LIB_OK="${KMP_DUPLICATE_LIB_OK:-TRUE}"
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD="${TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD:-1}"
export nnUNet_n_proc_DA="${nnUNet_n_proc_DA:-1}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export VECLIB_MAXIMUM_THREADS="${VECLIB_MAXIMUM_THREADS:-1}"
RUNTIME_PY="${BUNDLE_ROOT}/.runtime_py"
mkdir -p "${RUNTIME_PY}"
cat > "${RUNTIME_PY}/sitecustomize.py" <<'PY'
import multiprocessing as mp
try:
    mp.set_start_method("fork")
except RuntimeError:
    pass
try:
    import torch.multiprocessing as tmp
    tmp.set_sharing_strategy("file_system")
except Exception:
    pass
PY
export PYTHONPATH="${RUNTIME_PY}:${PYTHONPATH:-}"
mkdir -p "${XDG_CACHE_HOME}" "${HIPPUNFOLD_CACHE_DIR}" "${HIPPUNFOLD_RUNTIME_SOURCE_CACHE}"

cmd=(
  hippunfold
  "${INPUT_DIR}"
  "${OUT_DIR}"
  participant
  --modality T2w
  --output-density "${HIPPUNFOLD_OUTPUT_DENSITY}"
  --output-spaces corobl
  --autotop_labels ${=AUTOTOP_LABELS}
  --participant-label "${SUBJECT}"
  --runtime-source-cache-path "${HIPPUNFOLD_RUNTIME_SOURCE_CACHE}"
  --workflow-profile "${HIPPUNFOLD_WORKFLOW_PROFILE}"
  --sdm "${HIPPUNFOLD_SDM}"
  --cores 1
  --notemp
)

if [[ "${HIPPUNFOLD_USE_CONDA}" == "1" ]]; then
  cmd+=(
    --use-conda
    --conda-prefix "${HIPPUNFOLD_CONDA_PREFIX}"
    --conda-frontend conda
  )
fi

# Append any caller-supplied extra snakemake args (e.g. --forcerun <rule>).
if [[ -n "${HIPPUNFOLD_EXTRA_ARGS:-}" ]]; then
  cmd+=(${=HIPPUNFOLD_EXTRA_ARGS})
fi

"${cmd[@]}"
