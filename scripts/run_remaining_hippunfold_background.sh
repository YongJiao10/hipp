#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
SELF_PATH="${SCRIPT_DIR}/$(basename "${BASH_SOURCE[0]}")"

INPUT_ROOT="${REPO_ROOT}/data/hippunfold_input"
BATCH_ROOT="${REPO_ROOT}/outputs_migration/dense_corobl_batch"
RUNNER="${REPO_ROOT}/scripts/run_hippunfold_local.sh"
INTEGRITY_CHECKER="${REPO_ROOT}/scripts/check_hippunfold_output_integrity.py"
CONCURRENCY=4
MAX_ATTEMPTS=3
MODE="start"
TS="$(date -u +%Y%m%dT%H%M%SZ)"
MANAGER_LOG=""
NOHUP_LOG=""
MANIFEST_CACHE=""

usage() {
  cat <<'USAGE'
Usage:
  scripts/run_remaining_hippunfold_background.sh [options]

Default behavior:
  Start a detached background manager process that runs remaining (not complete)
  subjects with fixed parallelism.

Options:
  --worker                  Internal mode; run manager loop in foreground.
  --dry-run                 Only print remaining subject list and exit.
  --concurrency N           Parallel subjects (default: 4).
  --max-attempts N          Retry attempts per subject (default: 3).
  --input-root PATH         Default: data/hippunfold_input
  --batch-root PATH         Default: outputs_migration/dense_corobl_batch
  --runner PATH             Default: scripts/run_hippunfold_local.sh
  --integrity-checker PATH  Default: scripts/check_hippunfold_output_integrity.py
  --timestamp TS            Override timestamp label.
  -h, --help                Show this help.

Examples:
  # start detached (manager + per-subject logs under _batch_logs)
  scripts/run_remaining_hippunfold_background.sh --concurrency 4

  # inspect remaining without running
  scripts/run_remaining_hippunfold_background.sh --dry-run
USAGE
}

log_line() {
  local msg="$1"
  printf '[%s] %s\n' "$(date -u '+%Y-%m-%d %H:%M:%S UTC')" "$msg"
}

is_subject_complete() {
  local sid="$1"
  if [[ ! -f "${INTEGRITY_CHECKER}" ]]; then
    log_line "warning: integrity checker not found: ${INTEGRITY_CHECKER}"
    return 1
  fi
  python3 "${INTEGRITY_CHECKER}" \
    --input-root "${INPUT_ROOT}" \
    --batch-root "${BATCH_ROOT}" \
    --manifest-cache "${MANIFEST_CACHE}" \
    --subject "${sid}" \
    --quiet \
    --fail-on-missing
}

collect_remaining_subjects() {
  local all=()
  local sid=""
  mapfile -t all < <(
    find "${INPUT_ROOT}" -maxdepth 1 -type d -name 'sub-*' -printf '%f\n' \
      | sed 's/^sub-//' \
      | sort
  )
  for sid in "${all[@]}"; do
    if ! is_subject_complete "${sid}"; then
      echo "${sid}"
    fi
  done
}

unlock_subject() {
  local sid="$1"
  local out_dir="${BATCH_ROOT}/sub-${sid}/hippunfold"
  HIPPUNFOLD_EXTRA_ARGS="--unlock" \
    "${RUNNER}" "${sid}" "${INPUT_ROOT}" "${out_dir}" >/dev/null 2>&1 || true
}

run_one_subject() {
  local sid="$1"
  local attempt=1
  local rc=0
  local out_dir="${BATCH_ROOT}/sub-${sid}/hippunfold"
  local sub_log="${BATCH_ROOT}/_batch_logs/sub-${sid}_remaining_${TS}.log"

  log_line "launch sub-${sid} attempts<=${MAX_ATTEMPTS}"
  unlock_subject "${sid}"

  while (( attempt <= MAX_ATTEMPTS )); do
    log_line "sub-${sid} attempt=${attempt} start" >>"${sub_log}"
    set +e
    HIPPUNFOLD_EXTRA_ARGS="--rerun-incomplete" \
      "${RUNNER}" "${sid}" "${INPUT_ROOT}" "${out_dir}" >>"${sub_log}" 2>&1
    rc=$?
    set -e
    log_line "sub-${sid} attempt=${attempt} rc=${rc}" >>"${sub_log}"

    if is_subject_complete "${sid}"; then
      log_line "done sub-${sid} rc=0"
      return 0
    fi

    unlock_subject "${sid}"
    attempt=$((attempt + 1))
  done

  log_line "done sub-${sid} rc=1 (not complete after ${MAX_ATTEMPTS} attempts)"
  return 1
}

run_worker() {
  local remaining=()
  local sid=""
  local active=0
  local failures=0

  mapfile -t remaining < <(collect_remaining_subjects)
  if (( ${#remaining[@]} == 0 )); then
    log_line "manager: no remaining subjects"
    return 0
  fi

  printf '%s\n' "${remaining[@]}" > "${BATCH_ROOT}/_batch_logs/remaining_subjects_${TS}.txt"
  log_line "manager start concurrency=${CONCURRENCY} remaining=${#remaining[@]}"

  for sid in "${remaining[@]}"; do
    run_one_subject "${sid}" &
    active=$((active + 1))
    if (( active >= CONCURRENCY )); then
      if ! wait -n; then
        failures=$((failures + 1))
      fi
      active=$((active - 1))
    fi
  done

  while (( active > 0 )); do
    if ! wait -n; then
      failures=$((failures + 1))
    fi
    active=$((active - 1))
  done

  log_line "manager finished failures=${failures}"
  return 0
}

DRY_RUN=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --worker)
      MODE="worker"
      shift
      ;;
    --dry-run)
      DRY_RUN=1
      shift
      ;;
    --concurrency)
      CONCURRENCY="$2"
      shift 2
      ;;
    --max-attempts)
      MAX_ATTEMPTS="$2"
      shift 2
      ;;
    --input-root)
      INPUT_ROOT="$2"
      shift 2
      ;;
    --batch-root)
      BATCH_ROOT="$2"
      shift 2
      ;;
    --runner)
      RUNNER="$2"
      shift 2
      ;;
    --integrity-checker)
      INTEGRITY_CHECKER="$2"
      shift 2
      ;;
    --timestamp)
      TS="$2"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown option: $1" >&2
      usage
      exit 2
      ;;
  esac
done

mkdir -p "${BATCH_ROOT}/_batch_logs"
MANIFEST_CACHE="${BATCH_ROOT}/_batch_logs/output_integrity_manifest.json"

if (( DRY_RUN == 1 )); then
  mapfile -t remaining < <(collect_remaining_subjects)
  log_line "dry-run remaining=${#remaining[@]}"
  printf '%s\n' "${remaining[@]}"
  exit 0
fi

if [[ "${MODE}" == "worker" ]]; then
  MANAGER_LOG="${BATCH_ROOT}/_batch_logs/manager_remaining_${TS}_conc${CONCURRENCY}.log"
  exec > >(tee -a "${MANAGER_LOG}") 2>&1
  run_worker
  exit 0
fi

MANAGER_LOG="${BATCH_ROOT}/_batch_logs/manager_remaining_${TS}_conc${CONCURRENCY}.log"
NOHUP_LOG="${BATCH_ROOT}/_batch_logs/manager_remaining_${TS}_conc${CONCURRENCY}.nohup.log"

nohup "${SELF_PATH}" \
  --worker \
  --timestamp "${TS}" \
  --concurrency "${CONCURRENCY}" \
  --max-attempts "${MAX_ATTEMPTS}" \
  --input-root "${INPUT_ROOT}" \
  --batch-root "${BATCH_ROOT}" \
  --runner "${RUNNER}" \
  --integrity-checker "${INTEGRITY_CHECKER}" \
  > "${NOHUP_LOG}" 2>&1 < /dev/null &

PID="$!"
log_line "started background manager pid=${PID}"
echo "manager_log=${MANAGER_LOG}"
echo "nohup_log=${NOHUP_LOG}"
echo "pid=${PID}"
echo "monitor: tail -f ${MANAGER_LOG}"
