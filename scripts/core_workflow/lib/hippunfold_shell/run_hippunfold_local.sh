#!/bin/zsh
set -euo pipefail

SUBJECT="${1:-100610}"
INPUT_DIR="${2:-data/hippunfold_input}"
OUT_DIR="${3:-outputs_hcp/inputs/hcp/hippunfold/dense_corobl_batch/sub-${SUBJECT}/hippunfold}"
AUTOTOP_LABELS="${HIPPUNFOLD_AUTOTOP_LABELS:-hipp dentate}"
BUNDLE_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
HIPPUNFOLD_ENV_NAME="${HIPPUNFOLD_ENV_NAME:-hippo2}"
HIPPUNFOLD_MODALITY="${HIPPUNFOLD_MODALITY:-T1w}"
HIPPUNFOLD_USE_T1_REG_TEMPLATE="${HIPPUNFOLD_USE_T1_REG_TEMPLATE:-0}"
HIPPUNFOLD_OUTPUT_DENSITY="${HIPPUNFOLD_OUTPUT_DENSITY:-512}"
HIPPUNFOLD_T1W_FILTER="${HIPPUNFOLD_T1W_FILTER:-}"
HIPPUNFOLD_USE_GPU="${HIPPUNFOLD_USE_GPU:-1}"
HIPPUNFOLD_USE_CONDA="${HIPPUNFOLD_USE_CONDA:-0}"
HIPPUNFOLD_CORES="${HIPPUNFOLD_CORES:-1}"
HIPPUNFOLD_WORKFLOW_PROFILE="${HIPPUNFOLD_WORKFLOW_PROFILE:-none}"
HIPPUNFOLD_SDM="${HIPPUNFOLD_SDM:-env-modules}"
HIPPUNFOLD_EXTERNAL_BIN_DIR="${HIPPUNFOLD_EXTERNAL_BIN_DIR:-/Applications/ITK-SNAP.app/Contents/bin}"
CACHE_ROOT="${HIPPUNFOLD_CACHE_ROOT:-${BUNDLE_ROOT}/runtime/hippunfold_cache}"
HIPPUNFOLD_CLEANUP_RUNTIME="${HIPPUNFOLD_CLEANUP_RUNTIME:-0}"
CANONICAL_HIPPUNFOLD_ASSET_ROOT="/home/yojiao/hpchome/hipp/runtime/hippunfold_asset_store_shared"
CANONICAL_HIPPUNFOLD_CACHE_DIR="/home/yojiao/hpchome/hipp/runtime/hippunfold_asset_store_shared/hippunfold_assets"
CANONICAL_HIPPUNFOLD_RUNTIME_SOURCE_CACHE="/home/yojiao/hpchome/hipp/runtime/hippunfold_asset_store_shared/runtime_source_assets"
CANONICAL_HIPPUNFOLD_MODEL_PATH="/home/yojiao/hpchome/hipp/runtime/hippunfold_asset_store_shared/hippunfold_assets/model/trained_model.3d_fullres.Task101_hcp1200_T1w.nnUNetTrainerV2.model_best.tar"
CANONICAL_HIPPO2_PYTHON="/home/yojiao/hpchome/hipp/.cache/conda_envs/hippo2/bin/python"
CANONICAL_HIPPO2_HIPPUNFOLD="/home/yojiao/hpchome/hipp/.cache/conda_envs/hippo2/bin/hippunfold"
PREPARE_LOCAL_MODEL_WORKFLOW="${SCRIPT_DIR}/prepare_local_model_workflow.py"

if [[ -n "${HIPPUNFOLD_CACHE_DIR:-}" && "${HIPPUNFOLD_CACHE_DIR}" != "${CANONICAL_HIPPUNFOLD_CACHE_DIR}" ]]; then
  echo "run_hippunfold_local.sh: Refusing noncanonical HIPPUNFOLD_CACHE_DIR=${HIPPUNFOLD_CACHE_DIR}; expected ${CANONICAL_HIPPUNFOLD_CACHE_DIR}" >&2
  exit 2
fi

if [[ -n "${HIPPUNFOLD_RUNTIME_SOURCE_CACHE:-}" && "${HIPPUNFOLD_RUNTIME_SOURCE_CACHE}" != "${CANONICAL_HIPPUNFOLD_RUNTIME_SOURCE_CACHE}" ]]; then
  echo "run_hippunfold_local.sh: Refusing noncanonical HIPPUNFOLD_RUNTIME_SOURCE_CACHE=${HIPPUNFOLD_RUNTIME_SOURCE_CACHE}; expected ${CANONICAL_HIPPUNFOLD_RUNTIME_SOURCE_CACHE}" >&2
  exit 2
fi

export HIPPUNFOLD_CACHE_DIR="${CANONICAL_HIPPUNFOLD_CACHE_DIR}"
MODEL_PATH="${4:-${HIPPUNFOLD_MODEL_PATH:-${CANONICAL_HIPPUNFOLD_MODEL_PATH}}}"

if [[ "${MODEL_PATH}" != "${CANONICAL_HIPPUNFOLD_MODEL_PATH}" ]]; then
  echo "run_hippunfold_local.sh: Refusing noncanonical HIPPUNFOLD_MODEL_PATH=${MODEL_PATH}; expected ${CANONICAL_HIPPUNFOLD_MODEL_PATH}" >&2
  exit 2
fi

export HIPPUNFOLD_MODEL_PATH="${MODEL_PATH}"
if [[ ! -f "${HIPPUNFOLD_MODEL_PATH}" ]]; then
  echo "run_hippunfold_local.sh: Local HippUnfold model is missing: ${MODEL_PATH}" >&2
  exit 2
fi

echo "HippUnfold model path: ${HIPPUNFOLD_MODEL_PATH}"
echo "HippUnfold model download rule: absent from validated local-model overlay"
export HIPPUNFOLD_RUNTIME_SOURCE_CACHE="${CANONICAL_HIPPUNFOLD_RUNTIME_SOURCE_CACHE}"

CONDA_SH=""
for candidate in /opt/conda/etc/profile.d/conda.sh /opt/miniconda3/etc/profile.d/conda.sh; do
  if [[ -f "${candidate}" ]]; then
    CONDA_SH="${candidate}"
    break
  fi
done
if [[ -z "${CONDA_SH}" ]]; then
  echo "run_hippunfold_local.sh: conda.sh not found under /opt/conda or /opt/miniconda3" >&2
  exit 1
fi
source "${CONDA_SH}" >/dev/null 2>&1
conda activate "${HIPPUNFOLD_ENV_NAME}"

if [[ "$(command -v hippunfold)" != "${CANONICAL_HIPPO2_HIPPUNFOLD}" ]]; then
  echo "run_hippunfold_local.sh: hippo2 gate failed: expected ${CANONICAL_HIPPO2_HIPPUNFOLD}, got $(command -v hippunfold)" >&2
  exit 2
fi
HIPPUNFOLD_CLI_VERSION="$(hippunfold --version 2>&1)"
if [[ "${HIPPUNFOLD_CLI_VERSION}" != "2.0.0" ]]; then
  echo "run_hippunfold_local.sh: hippo2 gate failed: expected version 2.0.0, got ${HIPPUNFOLD_CLI_VERSION}" >&2
  exit 2
fi
if ! hippunfold --modality T1w --output_density 512 --help 2>&1 | grep -q "512"; then
  echo "run_hippunfold_local.sh: hippo2 gate failed: output_density=512 is unavailable" >&2
  exit 2
fi

HIPPUNFOLD_WORKFLOW_OVERLAY="${CACHE_ROOT}/hippunfold_local_model_overlay"
LOCAL_MODEL_SNAKEFILE="$("${CANONICAL_HIPPO2_PYTHON}" "${PREPARE_LOCAL_MODEL_WORKFLOW}" \
  --overlay-root "${HIPPUNFOLD_WORKFLOW_OVERLAY}" \
  --model-path "${HIPPUNFOLD_MODEL_PATH}")"
if [[ ! -f "${LOCAL_MODEL_SNAKEFILE}" ]]; then
  echo "run_hippunfold_local.sh: local-model workflow overlay was not created: ${LOCAL_MODEL_SNAKEFILE}" >&2
  exit 2
fi
export HIPPUNFOLD_WORKFLOW_OVERLAY
echo "HippUnfold local-model workflow: ${LOCAL_MODEL_SNAKEFILE}"

cleanup_runtime_dirs() {
  if [[ "${HIPPUNFOLD_CLEANUP_RUNTIME}" == "1" ]]; then
    local home_cache="${CACHE_ROOT}/home"
    if [[ -d "${home_cache}" ]]; then
      find "${home_cache}" -type d -empty -delete 2>/dev/null || true
    fi
    rm -rf "${CACHE_ROOT}"
  else
    echo "run_hippunfold_local.sh: Retaining runtime cache: ${CACHE_ROOT}" >&2
  fi
}
trap cleanup_runtime_dirs EXIT

export XDG_CACHE_HOME="${XDG_CACHE_HOME:-${CACHE_ROOT}/xdg_cache}"
export PIP_CACHE_DIR="${PIP_CACHE_DIR:-${CACHE_ROOT}/pip_cache}"
export TMPDIR="${TMPDIR:-${CACHE_ROOT}/tmp}"
export PATH="${BUNDLE_ROOT}/scripts:${PATH}"
if [[ -d "${HIPPUNFOLD_EXTERNAL_BIN_DIR}" ]]; then
  export PATH="${HIPPUNFOLD_EXTERNAL_BIN_DIR}:${PATH}"
fi
export HIPPUNFOLD_CONDA_PREFIX="${HIPPUNFOLD_CONDA_PREFIX:-${CACHE_ROOT}/snakemake_conda}"
mkdir -p "${HIPPUNFOLD_CONDA_PREFIX}"
case "$(uname -s)" in
  Darwin) default_conda_subdir="osx-64" ;;
  *) default_conda_subdir="linux-64" ;;
esac
export CONDA_SUBDIR="${CONDA_SUBDIR:-${default_conda_subdir}}"
export CONDA_PKGS_DIRS="${CONDA_PKGS_DIRS:-${CACHE_ROOT}/conda_pkgs}"
mkdir -p "${CONDA_PKGS_DIRS}"
mkdir -p "${PIP_CACHE_DIR}" "${TMPDIR}"
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
  --modality "${HIPPUNFOLD_MODALITY}"
  --output-density "${HIPPUNFOLD_OUTPUT_DENSITY}"
  --output-spaces corobl
  --autotop_labels ${=AUTOTOP_LABELS}
  --participant-label "${SUBJECT}"
  --runtime-source-cache-path "${HIPPUNFOLD_RUNTIME_SOURCE_CACHE}"
  --workflow-profile "${HIPPUNFOLD_WORKFLOW_PROFILE}"
  --sdm "${HIPPUNFOLD_SDM}"
  --cores "${HIPPUNFOLD_CORES}"
  --notemp
)

if [[ -n "${HIPPUNFOLD_T1W_FILTER}" ]]; then
  cmd+=(--filter-T1w "${HIPPUNFOLD_T1W_FILTER}")
fi

if [[ "${HIPPUNFOLD_USE_T1_REG_TEMPLATE}" == "1" ]]; then
  cmd+=(--t1-reg-template)
fi

if [[ "${HIPPUNFOLD_USE_GPU}" == "1" ]]; then
  cmd+=(--use_gpu)
fi


if [[ "${HIPPUNFOLD_USE_CONDA}" == "1" ]]; then
  conda_frontend="${HIPPUNFOLD_CONDA_FRONTEND:-mamba}"
  if [[ "${conda_frontend}" == "mamba" && -z "$(command -v mamba 2>/dev/null || true)" ]]; then
    conda_frontend="conda"
  fi
  cmd+=(
    --use-conda
    --conda-prefix "${HIPPUNFOLD_CONDA_PREFIX}"
    --conda-frontend "${conda_frontend}"
  )
fi

# Only the two non-executing diagnostics used by the fresh-DAG preflight are accepted.
if [[ -n "${HIPPUNFOLD_EXTRA_ARGS:-}" ]]; then
  for extra_arg in ${=HIPPUNFOLD_EXTRA_ARGS}; do
    case "${extra_arg}" in
      --dry-run|--printshellcmds) ;;
      *)
        echo "run_hippunfold_local.sh: unsupported extra workflow argument: ${extra_arg}; only dry-run diagnostics are accepted." >&2
        exit 2
        ;;
    esac
  done
  cmd+=(${=HIPPUNFOLD_EXTRA_ARGS})
fi

# Run the complete participant/all DAG with the validated local-model workflow.
cmd+=(--snakefile "${LOCAL_MODEL_SNAKEFILE}")

"${cmd[@]}"
