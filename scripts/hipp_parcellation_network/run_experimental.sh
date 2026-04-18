#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
RUN_BATCH="${SCRIPT_DIR}/run_batch.py"

# Locked parameters for canonical experimental runs.
for arg in "$@"; do
  case "${arg}" in
    --k-selection-mode|--k-selection-mode=*|\
    --run-split-mode|--run-split-mode=*|\
    --out-root|--out-root=*|\
    --present-dir|--present-dir=*)
      echo "ERROR: ${arg} is locked in run_experimental.sh and cannot be overridden." >&2
      echo "Use run_batch.py directly only for non-canonical debugging runs." >&2
      exit 2
      ;;
  esac
done

source /opt/miniconda3/etc/profile.d/conda.sh
conda activate py314

python "${RUN_BATCH}" \
  --k-selection-mode experimental \
  --run-split-mode runwise \
  --out-root "${REPO_ROOT}/outputs_experimental/hipp_functional_parcellation_network" \
  --present-dir "${REPO_ROOT}/present_network_experimental" \
  "$@"
