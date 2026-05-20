#!/usr/bin/env bash
set -euo pipefail

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
  echo "source this file instead of executing it: source scripts/activate_hippo2.sh" >&2
  exit 1
fi

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_PREFIX="${REPO_ROOT}/.cache/conda_envs/hippo2"

if [[ ! -d "${ENV_PREFIX}" ]]; then
  echo "activate_hippo2.sh: missing repo-local env at ${ENV_PREFIX}" >&2
  return 1
fi

if [[ -d "${ENV_PREFIX}/conda-meta" ]]; then
  CONDA_SH=""
  for candidate in /opt/conda/etc/profile.d/conda.sh /opt/miniconda3/etc/profile.d/conda.sh; do
    if [[ -f "${candidate}" ]]; then
      CONDA_SH="${candidate}"
      break
    fi
  done

  if [[ -n "${CONDA_SH}" ]]; then
    source "${CONDA_SH}" >/dev/null 2>&1
    conda activate "${ENV_PREFIX}"
    export LD_LIBRARY_PATH="${ENV_PREFIX}/lib:${LD_LIBRARY_PATH:-}"
    return 0
  fi
fi

export LD_LIBRARY_PATH="${ENV_PREFIX}/lib:${LD_LIBRARY_PATH:-}"
export PATH="${ENV_PREFIX}/bin:${PATH}"
export CONDA_PREFIX="${ENV_PREFIX}"
export CONDA_DEFAULT_ENV="hippo2"
export HIPPUNFOLD_ENV_NAME="hippo2"
