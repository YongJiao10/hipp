#!/usr/bin/env bash
set -euo pipefail

if [[ "$#" -ne 2 ]]; then
  echo "run_hippunfold_local.sh: expected exactly two arguments: <dataset> <subject>" >&2
  exit 2
fi

exec /home/yojiao/.conda/envs/py314/bin/python \
  /home/yojiao/hpchome/hipp/scripts/core_workflow/01_01_run_hippunfold_t1w_local.py \
  "$2" --dataset "$1"
