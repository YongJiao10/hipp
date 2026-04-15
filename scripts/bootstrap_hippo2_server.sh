#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
ENV_NAME="${1:-hippo2}"
ENV_FILE="${2:-${REPO_ROOT}/environment/hippo2_server.yml}"

source /opt/miniconda3/etc/profile.d/conda.sh

if conda env list | awk '{print $1}' | grep -qx "${ENV_NAME}"; then
  conda env update -n "${ENV_NAME}" -f "${ENV_FILE}"
else
  conda env create -n "${ENV_NAME}" -f "${ENV_FILE}"
fi

conda activate "${ENV_NAME}"

echo "Verified binary sources:"
command -v hippunfold
command -v nnUNet_predict
command -v c3d
command -v greedy
command -v LN2_LAYERS
command -v N4BiasFieldCorrection

echo "HippUnfold version:"
hippunfold --version
