#!/usr/bin/env bash
set -euo pipefail

BUNDLE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_NAME="${ENV_NAME:-hippo}"
ENV_FILE="${ENV_FILE:-${BUNDLE_ROOT}/environment.yml}"
CONDA_EXE="${CONDA_EXE:-conda}"
MAMBA_EXE="${MAMBA_EXE:-mamba}"
USE_MAMBA="${USE_MAMBA:-1}"

INSTALL_HIPPOMAPS="${INSTALL_HIPPOMAPS:-1}"
HIPPOMAPS_INSTALL_METHOD="${HIPPOMAPS_INSTALL_METHOD:-pypi}"   # pypi | git
HIPPOMAPS_SPEC="${HIPPOMAPS_SPEC:-hippomaps==0.1.17}"
HIPPOMAPS_GIT_URL="${HIPPOMAPS_GIT_URL:-https://github.com/HippAI/hippomaps.git}"
HIPPOMAPS_GIT_REF="${HIPPOMAPS_GIT_REF:-main}"
HIPPOMAPS_CLONE_DIR="${HIPPOMAPS_CLONE_DIR:-${BUNDLE_ROOT}/external/hippomaps-src}"

INSTALL_HIPPUNFOLD="${INSTALL_HIPPUNFOLD:-1}"
HIPPUNFOLD_INSTALL_METHOD="${HIPPUNFOLD_INSTALL_METHOD:-conda}"   # conda | apptainer
HIPPUNFOLD_CHANNELS=(-c khanlab -c conda-forge -c bioconda)
HIPPUNFOLD_APPTAINER_IMAGE="${HIPPUNFOLD_APPTAINER_IMAGE:-${BUNDLE_ROOT}/external/containers/hippunfold_latest.sif}"
HIPPUNFOLD_APPTAINER_URI="${HIPPUNFOLD_APPTAINER_URI:-docker://khanlab/hippunfold:latest}"

INSTALL_WORKBENCH="${INSTALL_WORKBENCH:-0}"
WORKBENCH_DIR="${WORKBENCH_DIR:-${BUNDLE_ROOT}/external/workbench}"
WORKBENCH_ARCHIVE="${WORKBENCH_ARCHIVE:-}"
WORKBENCH_URL="${WORKBENCH_URL:-}"
WB_COMMAND_BIN="${WB_COMMAND_BIN:-}"

ACTIVATE_SNIPPET="${ACTIVATE_SNIPPET:-${BUNDLE_ROOT}/activate_hpc_env.sh}"

choose_solver() {
  if [[ "${USE_MAMBA}" == "1" ]] && command -v "${MAMBA_EXE}" >/dev/null 2>&1; then
    echo "${MAMBA_EXE}"
  else
    echo "${CONDA_EXE}"
  fi
}

SOLVER="$(choose_solver)"

require_cmd() {
  if ! command -v "$1" >/dev/null 2>&1; then
    echo "Missing required command: $1" >&2
    exit 1
  fi
}

require_cmd "${CONDA_EXE}"

CONDA_BASE="$("${CONDA_EXE}" info --base)"
source "${CONDA_BASE}/etc/profile.d/conda.sh"

echo "[setup] bundle root: ${BUNDLE_ROOT}"
echo "[setup] conda base: ${CONDA_BASE}"
echo "[setup] solver: ${SOLVER}"
echo "[setup] env name: ${ENV_NAME}"

if "${CONDA_EXE}" env list | awk '{print $1}' | grep -Fxq "${ENV_NAME}"; then
  echo "[setup] conda env ${ENV_NAME} already exists"
else
  echo "[setup] creating env from ${ENV_FILE}"
  "${SOLVER}" env create -n "${ENV_NAME}" -f "${ENV_FILE}"
fi

conda activate "${ENV_NAME}"

if [[ "${INSTALL_HIPPOMAPS}" == "1" ]]; then
  echo "[setup] installing hippomaps via ${HIPPOMAPS_INSTALL_METHOD}"
  if [[ "${HIPPOMAPS_INSTALL_METHOD}" == "pypi" ]]; then
    pip install "${HIPPOMAPS_SPEC}"
  elif [[ "${HIPPOMAPS_INSTALL_METHOD}" == "git" ]]; then
    require_cmd git
    mkdir -p "$(dirname "${HIPPOMAPS_CLONE_DIR}")"
    if [[ ! -d "${HIPPOMAPS_CLONE_DIR}/.git" ]]; then
      git clone "${HIPPOMAPS_GIT_URL}" "${HIPPOMAPS_CLONE_DIR}"
    fi
    git -C "${HIPPOMAPS_CLONE_DIR}" fetch --all --tags
    git -C "${HIPPOMAPS_CLONE_DIR}" checkout "${HIPPOMAPS_GIT_REF}"
    pip install -e "${HIPPOMAPS_CLONE_DIR}"
  else
    echo "Unsupported HIPPOMAPS_INSTALL_METHOD=${HIPPOMAPS_INSTALL_METHOD}" >&2
    exit 1
  fi
fi

if [[ "${INSTALL_HIPPUNFOLD}" == "1" ]]; then
  echo "[setup] installing hippunfold via ${HIPPUNFOLD_INSTALL_METHOD}"
  if [[ "${HIPPUNFOLD_INSTALL_METHOD}" == "conda" ]]; then
    "${SOLVER}" install -n "${ENV_NAME}" "${HIPPUNFOLD_CHANNELS[@]}" hippunfold
  elif [[ "${HIPPUNFOLD_INSTALL_METHOD}" == "apptainer" ]]; then
    require_cmd apptainer
    mkdir -p "$(dirname "${HIPPUNFOLD_APPTAINER_IMAGE}")"
    if [[ ! -f "${HIPPUNFOLD_APPTAINER_IMAGE}" ]]; then
      apptainer pull "${HIPPUNFOLD_APPTAINER_IMAGE}" "${HIPPUNFOLD_APPTAINER_URI}"
    fi
  else
    echo "Unsupported HIPPUNFOLD_INSTALL_METHOD=${HIPPUNFOLD_INSTALL_METHOD}" >&2
    exit 1
  fi
fi

if [[ "${INSTALL_WORKBENCH}" == "1" ]]; then
  require_cmd unzip
  mkdir -p "${WORKBENCH_DIR}"
  archive_path=""
  if [[ -n "${WORKBENCH_ARCHIVE}" ]]; then
    archive_path="${WORKBENCH_ARCHIVE}"
  elif [[ -n "${WORKBENCH_URL}" ]]; then
    require_cmd curl
    archive_path="${WORKBENCH_DIR}/workbench_download.zip"
    curl -L "${WORKBENCH_URL}" -o "${archive_path}"
  fi

  if [[ -n "${archive_path}" ]]; then
    unzip -o "${archive_path}" -d "${WORKBENCH_DIR}"
    wb_candidate="$(find "${WORKBENCH_DIR}" -type f -name wb_command | head -n 1 || true)"
    if [[ -n "${wb_candidate}" ]]; then
      WB_COMMAND_BIN="${wb_candidate}"
    fi
  fi
fi

if [[ -z "${WB_COMMAND_BIN}" ]]; then
  wb_found="$(command -v wb_command || true)"
  if [[ -n "${wb_found}" ]]; then
    WB_COMMAND_BIN="${wb_found}"
  else
    wb_candidate="$(find "${WORKBENCH_DIR}" -type f -name wb_command | head -n 1 || true)"
    if [[ -n "${wb_candidate}" ]]; then
      WB_COMMAND_BIN="${wb_candidate}"
    fi
  fi
fi

cat > "${ACTIVATE_SNIPPET}" <<EOF
#!/usr/bin/env bash
set -euo pipefail
source "${CONDA_BASE}/etc/profile.d/conda.sh"
conda activate "${ENV_NAME}"
export WB_COMMAND_BIN="${WB_COMMAND_BIN}"
export HIPPUNFOLD_CACHE_DIR="\${HIPPUNFOLD_CACHE_DIR:-${BUNDLE_ROOT}/.cache/hippunfold}"
export HIPPUNFOLD_CONDA_PREFIX="\${HIPPUNFOLD_CONDA_PREFIX:-${BUNDLE_ROOT}/.cache/hippunfold_conda}"
export WB_FALLBACK_PY="\${WB_FALLBACK_PY:-$(command -v python)}"
EOF
chmod +x "${ACTIVATE_SNIPPET}"

echo "[setup] wrote activation helper: ${ACTIVATE_SNIPPET}"
echo "[setup] summary:"
echo "  env: ${ENV_NAME}"
echo "  python: $(command -v python)"
echo "  hippomaps: $(python - <<'PY'
try:
    import importlib.metadata as m
    print(m.version('hippomaps'))
except Exception:
    print('not installed')
PY
)"
echo "  hippunfold: $(command -v hippunfold || echo not-on-path)"
echo "  wb_command: ${WB_COMMAND_BIN:-not-found}"
if [[ "${HIPPUNFOLD_INSTALL_METHOD}" == "apptainer" ]]; then
  echo "  apptainer image: ${HIPPUNFOLD_APPTAINER_IMAGE}"
fi

echo
echo "Next step:"
echo "  source ${ACTIVATE_SNIPPET}"
