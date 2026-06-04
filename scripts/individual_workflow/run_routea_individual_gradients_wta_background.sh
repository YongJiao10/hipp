#!/usr/bin/env bash
set -euo pipefail

cd /home/yojiao/hpchome/hipp

RUN_ID="${1:-routea_individual_gradients_wta_$(date -u +%Y%m%dT%H%M%SZ)}"
OUT_ROOT="outputs_core/work/individual_atlas"
LOG_DIR="${OUT_ROOT}/gradient_spectral_k3/_runs/logs/${RUN_ID}"
mkdir -p "${LOG_DIR}"

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1

setsid nohup /home/yojiao/.conda/envs/py314/bin/python \
  scripts/individual_workflow/routea_individual_gradients_wta.py \
  --run-id "${RUN_ID}" \
  --jobs 4 \
  --resume \
  > "${LOG_DIR}/supervisor.out" \
  2> "${LOG_DIR}/supervisor.err" \
  < /dev/null &

PID="$!"
printf '%s\n' "${PID}" > "${LOG_DIR}/launcher.pid"
printf 'run_id=%s\npid=%s\nlog_dir=%s\n' "${RUN_ID}" "${PID}" "${LOG_DIR}"
