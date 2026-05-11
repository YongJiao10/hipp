#!/usr/bin/env bash
set -euo pipefail

SUBJECT="${1:?subject required}"
INPUT_DIR="${2:-data/hippunfold_input}"
OUT_DIR="${3:?output dir required}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

ASSET_ROOT="${HIPPUNFOLD_ASSET_STORE_ROOT:-${REPO_ROOT}/runtime/hippunfold_asset_store_shared}"
WORK_ROOT="${REPO_ROOT}/runtime/hippunfold_workdirs"
mkdir -p "${WORK_ROOT}" "${ASSET_ROOT}/hippunfold_assets" "${ASSET_ROOT}/runtime_source_assets"

export HIPPUNFOLD_MODALITY="T2w"
export HIPPUNFOLD_USE_T1_REG_TEMPLATE="1"
export HIPPUNFOLD_CACHE_ROOT="${HIPPUNFOLD_CACHE_ROOT:-${WORK_ROOT}/sub-${SUBJECT}_$$}"
export HIPPUNFOLD_CACHE_DIR="${HIPPUNFOLD_CACHE_DIR:-${ASSET_ROOT}/hippunfold_assets}"
export HIPPUNFOLD_RUNTIME_SOURCE_CACHE="${HIPPUNFOLD_RUNTIME_SOURCE_CACHE:-${ASSET_ROOT}/runtime_source_assets}"

exec "${SCRIPT_DIR}/run_hippunfold_local.sh" "${SUBJECT}" "${INPUT_DIR}" "${OUT_DIR}"
