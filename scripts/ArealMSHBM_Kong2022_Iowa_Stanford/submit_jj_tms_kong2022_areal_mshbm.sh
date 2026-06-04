#!/bin/bash
set -euo pipefail
export LANG=C
export LC_ALL=C
export LC_CTYPE=C

repo_root="${REPO_ROOT:-/old_Users/yojiao/hipp}"
script_dir="${SCRIPT_DIR:-${repo_root}/scripts/ArealMSHBM_Kong2022_Iowa_Stanford}"
task_script="${script_dir}/run_jj_tms_kong2022_areal_mshbm_subject_sge.sh"

data_root="${DATA_ROOT:-/Shared/jianglab/3_Data_Working/fmriprep_processed_JJ_TMS/derivatives_surf}"
out_root="${OUT_ROOT:-/Shared/jianglab/3_Data_Working/fmriprep_processed_JJ_TMS/ArealMSHBM_Kong2022}"
python_exe="${PYTHON_EXE:-/Users/yojiao/miniconda3/envs/py313/bin/python}"
cbig_code_dir="${CBIG_CODE_DIR:-${repo_root}/external/CBIG}"
group_prior="${GROUP_PRIOR:-${cbig_code_dir}/stable_projects/brain_parcellation/Kong2022_ArealMSHBM/lib/group_priors/HCP_fs_LR_32k_40sub/400/gMSHBM/beta50/Params_Final.mat}"

dry_run="${DRY_RUN:-0}"
task_dry_run="${TASK_DRY_RUN:-0}"
smoke_n="${SMOKE_N:-0}"
subjects_csv="${SUBJECT_IDS_CSV:-}"
subject_ids_file="${SUBJECT_IDS_FILE:-}"
overwrite="${OVERWRITE:-0}"

usage() {
  cat <<EOF
Usage: $(basename "$0")

Environment controls:
  DRY_RUN=1             Print qsub commands without submitting.
  TASK_DRY_RUN=1        Submitted tasks validate inputs/dependencies and stop before MATLAB.
  SMOKE_N=N             Use only the first N selected subjects.
  SUBJECT_IDS_CSV=...   Explicit comma-separated subject IDs.
  SUBJECT_IDS_FILE=...  Explicit subject list file.
  OVERWRITE=1           Recompute existing outputs.

Queue behavior follows scripts/core_workflow/sge_argon/submit_kong2022_route_a_dual_queues.sh:
  UI tasks use -tc 5.
  PINC tasks use -tc 15 for this JJ_TMS Argon batch.
EOF
}

if [ "${1:-}" = "-h" ] || [ "${1:-}" = "--help" ]; then
  usage
  exit 0
fi

if [ ! -x "${task_script}" ]; then
  echo "ERROR: missing executable task script: ${task_script}" >&2
  exit 2
fi
if [ "${dry_run}" = "0" ] && ! command -v qsub >/dev/null 2>&1; then
  echo "ERROR: qsub command not found in PATH" >&2
  exit 127
fi
if ! [[ "${smoke_n}" =~ ^[0-9]+$ ]]; then
  echo "ERROR: SMOKE_N must be an integer, got: ${smoke_n}" >&2
  exit 2
fi

manifest_dir="${out_root}/_manifests"
log_root="${out_root}/_sge_logs"
mkdir -p "${manifest_dir}" "${log_root}"
timestamp="$(date +%Y%m%d_%H%M%S)"

read_subjects() {
  if [ -n "${subject_ids_file}" ]; then
    if [ ! -f "${subject_ids_file}" ]; then
      echo "ERROR: SUBJECT_IDS_FILE not found: ${subject_ids_file}" >&2
      exit 2
    fi
    awk '{gsub(/^[ \t]+|[ \t]+$/, ""); sub(/^sub-/, ""); if ($0 ~ /^[0-9]+$/) print $0}' "${subject_ids_file}"
  elif [ -n "${subjects_csv}" ]; then
    echo "${subjects_csv}" | tr ',' '\n' | awk '{gsub(/^[ \t]+|[ \t]+$/, ""); sub(/^sub-/, ""); if ($0 ~ /^[0-9]+$/) print $0}'
  else
    find "${data_root}" -maxdepth 1 -type d -name 'sub-*' -printf '%f\n' | sed 's/^sub-//' | sort -V
  fi
}

mapfile -t all_ids < <(read_subjects)
if [ "${#all_ids[@]}" -lt 1 ]; then
  echo "ERROR: no valid subject IDs found" >&2
  exit 2
fi

dup_ids="$(printf '%s\n' "${all_ids[@]}" | sort | uniq -d || true)"
if [ -n "${dup_ids}" ]; then
  echo "ERROR: duplicate subject IDs:" >&2
  echo "${dup_ids}" >&2
  exit 2
fi

if [ "${smoke_n}" -gt 0 ]; then
  if [ "${smoke_n}" -gt "${#all_ids[@]}" ]; then
    echo "ERROR: SMOKE_N ${smoke_n} > subject count ${#all_ids[@]}" >&2
    exit 2
  fi
  mapfile -t all_ids < <(printf '%s\n' "${all_ids[@]}" | head -n "${smoke_n}")
fi

ui_ids=()
pinc_ids=()
for i in "${!all_ids[@]}"; do
  sid="${all_ids[$i]}"
  mod=$(( i % 3 ))
  if [ "${mod}" -eq 0 ]; then
    ui_ids+=("${sid}")
  else
    pinc_ids+=("${sid}")
  fi
done

write_list() {
  local queue_tag="$1"
  shift
  local -a ids=("$@")
  local list_file="${manifest_dir}/jj_tms_${queue_tag}_${timestamp}.txt"
  printf '%s\n' "${ids[@]}" > "${list_file}"
  echo "${list_file}"
}

submit_queue() {
  local queue="$1"
  local tc="$2"
  local subject_list="$3"
  local subject_count="$4"
  local job_name="$5"
  local log_dir="${log_root}/${queue}"
  mkdir -p "${log_dir}"

  local qsub_cmd=(
    qsub
    -N "${job_name}"
    -q "${queue}"
    -t "1-${subject_count}"
    -tc "${tc}"
    -j y
    -o "${log_dir}/${job_name}.\$JOB_ID.\$TASK_ID.log"
    -v "REPO_ROOT=${repo_root},SCRIPT_DIR=${script_dir},DATA_ROOT=${data_root},OUT_ROOT=${out_root},SUBJECT_IDS_FILE=${subject_list},PYTHON_EXE=${python_exe},CBIG_CODE_DIR=${cbig_code_dir},GROUP_PRIOR=${group_prior},TASK_DRY_RUN=${task_dry_run},OVERWRITE=${overwrite}"
    "${task_script}"
  )

  echo "[submit] ${queue}: subjects=${subject_count} tc=${tc} task_dry_run=${task_dry_run} dry_run=${dry_run}" >&2
  echo "[submit] ${queue}: subject_list=${subject_list}" >&2
  echo "[submit] ${queue}: out_root=${out_root}" >&2
  echo "[submit] ${queue}: ${qsub_cmd[*]}" >&2

  if [ "${dry_run}" = "0" ]; then
    "${qsub_cmd[@]}"
  fi
}

echo "[dual-submit] total=${#all_ids[@]} data_root=${data_root}" >&2

if [ "${#ui_ids[@]}" -gt 0 ]; then
  ui_list="$(write_list ui "${ui_ids[@]}")"
  submit_queue "UI" "5" "${ui_list}" "${#ui_ids[@]}" "jj_k22_ui"
fi

if [ "${#pinc_ids[@]}" -gt 0 ]; then
  pinc_list="$(write_list pinc "${pinc_ids[@]}")"
  submit_queue "PINC" "15" "${pinc_list}" "${#pinc_ids[@]}" "jj_k22_pinc"
fi

echo "[done] launcher completed (dry_run=${dry_run}, smoke_n=${smoke_n}, task_dry_run=${task_dry_run})" >&2
