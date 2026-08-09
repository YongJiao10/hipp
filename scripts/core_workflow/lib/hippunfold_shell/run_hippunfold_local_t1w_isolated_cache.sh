#!/usr/bin/env bash
set -euo pipefail

SUBJECT="${1:?subject required}"
INPUT_DIR="${2:-data/hippunfold_input}"
OUT_DIR="${3:?output dir required}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

ASSET_ROOT="/home/yojiao/hpchome/hipp/runtime/hippunfold_asset_store_shared"
MODEL_PATH="${4:-${ASSET_ROOT}/hippunfold_assets/model/trained_model.3d_fullres.Task101_hcp1200_T1w.nnUNetTrainerV2.model_best.tar}"
WORK_ROOT="/home/yojiao/hpchome/hipp/runtime/hippunfold_workdirs"
mkdir -p "${WORK_ROOT}" "${ASSET_ROOT}/hippunfold_assets" "${ASSET_ROOT}/runtime_source_assets"

export HIPPUNFOLD_MODALITY="T1w"
export HIPPUNFOLD_USE_T1_REG_TEMPLATE="0"
export HIPPUNFOLD_CACHE_ROOT="${HIPPUNFOLD_CACHE_ROOT:-${WORK_ROOT}/sub-${SUBJECT}_$$}"
export HIPPUNFOLD_CACHE_DIR="/home/yojiao/hpchome/hipp/runtime/hippunfold_asset_store_shared/hippunfold_assets"
export HIPPUNFOLD_RUNTIME_SOURCE_CACHE="/home/yojiao/hpchome/hipp/runtime/hippunfold_asset_store_shared/runtime_source_assets"
export HIPPUNFOLD_MODEL_PATH="${MODEL_PATH}"

exec "${SCRIPT_DIR}/run_hippunfold_local.sh" "${SUBJECT}" "${INPUT_DIR}" "${OUT_DIR}" "${MODEL_PATH}"
