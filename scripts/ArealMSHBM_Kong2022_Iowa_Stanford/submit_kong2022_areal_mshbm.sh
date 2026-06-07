#!/bin/bash
set -euo pipefail
export LANG=C
export LC_ALL=C
export LC_CTYPE=C

repo_root="${REPO_ROOT:-/old_Users/yojiao/hipp}"
script_dir="${SCRIPT_DIR:-${repo_root}/scripts/ArealMSHBM_Kong2022_Iowa_Stanford}"
task_script="${script_dir}/run_kong2022_areal_mshbm_subject_sge.sh"

dataset="$(printf '%s' "${DATASET:-jj_tms}" | tr '[:upper:]-' '[:lower:]_')"
case "${dataset}" in
  jj_tms)
    dataset_label="JJ_TMS"
    default_data_root="/Shared/jianglab/3_Data_Working/fmriprep_processed_JJ_TMS/derivatives_surf"
    default_out_root="/Shared/jianglab/3_Data_Working/fmriprep_processed_JJ_TMS/ArealMSHBM_Kong2022"
    job_prefix="jj_k22"
    manifest_prefix="jj_tms"
    ;;
  stanford)
    dataset_label="Stanford"
    default_data_root="/Shared/jianglab/3_Data_Working/fmriprep_processed_CausalConnectome/rest_fmriprep/DataOrg_PFM_36P"
    default_out_root="/Shared/jianglab/3_Data_Working/fmriprep_processed_CausalConnectome/rest_fmriprep/ArealMSHBM_Kong2022"
    job_prefix="st_k22"
    manifest_prefix="stanford"
    ;;
  *)
    echo "ERROR: unsupported DATASET=${DATASET:-}. Expected jj_tms or stanford." >&2
    exit 2
    ;;
esac

data_root="${DATA_ROOT:-${default_data_root}}"
out_root="${OUT_ROOT:-${default_out_root}}"
python_exe="${PYTHON_EXE:-/Users/yojiao/miniconda3/envs/py313/bin/python}"
cbig_code_dir="${CBIG_CODE_DIR:-${repo_root}/external/CBIG}"
group_prior="${GROUP_PRIOR:-${cbig_code_dir}/stable_projects/brain_parcellation/Kong2022_ArealMSHBM/lib/group_priors/HCP_fs_LR_32k_40sub/400/gMSHBM/beta50/Params_Final.mat}"
wb_command="${WB_COMMAND:-${repo_root}/software/workbench/bin_linux64/wb_command}"

dry_run="${DRY_RUN:-0}"
task_dry_run="${TASK_DRY_RUN:-0}"
smoke_n="${SMOKE_N:-0}"
subjects_csv="${SUBJECT_IDS_CSV:-}"
subject_ids_file="${SUBJECT_IDS_FILE:-}"
overwrite="${OVERWRITE:-0}"
ui_tc="${UI_TC:-5}"
pinc_tc="${PINC_TC:-15}"

usage() {
  cat <<EOF
Usage: $(basename "$0")

Environment controls:
  DATASET=jj_tms|stanford
                        Dataset preset. Default: jj_tms.
  DRY_RUN=1             Print qsub commands without submitting.
  TASK_DRY_RUN=1        Submitted tasks validate inputs/dependencies and stop before MATLAB.
  SMOKE_N=N             Use only the first N selected subjects.
  SUBJECT_IDS_CSV=...   Explicit comma-separated subject IDs.
  SUBJECT_IDS_FILE=...  Explicit subject list file.
  UI_TC=N               UI array throttle. Default: 5.
  PINC_TC=N             PINC array throttle. Default: 15.
  OVERWRITE=1           Recompute existing outputs.

Queue behavior follows scripts/core_workflow/sge_argon/submit_kong2022_route_a_dual_queues.sh:
  UI tasks use -tc ${ui_tc}.
  PINC tasks use -tc ${pinc_tc}.
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
if ! [[ "${ui_tc}" =~ ^[0-9]+$ ]] || [ "${ui_tc}" -lt 1 ]; then
  echo "ERROR: UI_TC must be a positive integer, got: ${ui_tc}" >&2
  exit 2
fi
if ! [[ "${pinc_tc}" =~ ^[0-9]+$ ]] || [ "${pinc_tc}" -lt 1 ]; then
  echo "ERROR: PINC_TC must be a positive integer, got: ${pinc_tc}" >&2
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
  local list_file="${manifest_dir}/${manifest_prefix}_${queue_tag}_${timestamp}.txt"
  printf '%s\n' "${ids[@]}" > "${list_file}"
  echo "${list_file}"
}

shell_quote() {
  printf "%q" "$1"
}

write_job_wrapper() {
  local queue_tag="$1"
  local subject_list="$2"
  local wrapper_file="${manifest_dir}/${manifest_prefix}_${queue_tag}_${timestamp}.job.sh"

  {
    echo "#!/bin/bash"
    echo "set -euo pipefail"
    printf "export DATASET=%s\n" "$(shell_quote "${dataset}")"
    printf "export REPO_ROOT=%s\n" "$(shell_quote "${repo_root}")"
    printf "export SCRIPT_DIR=%s\n" "$(shell_quote "${script_dir}")"
    printf "export DATA_ROOT=%s\n" "$(shell_quote "${data_root}")"
    printf "export OUT_ROOT=%s\n" "$(shell_quote "${out_root}")"
    printf "export SUBJECT_IDS_FILE=%s\n" "$(shell_quote "${subject_list}")"
    printf "export PYTHON_EXE=%s\n" "$(shell_quote "${python_exe}")"
    printf "export CBIG_CODE_DIR=%s\n" "$(shell_quote "${cbig_code_dir}")"
    printf "export GROUP_PRIOR=%s\n" "$(shell_quote "${group_prior}")"
    printf "export WB_COMMAND=%s\n" "$(shell_quote "${wb_command}")"
    printf "export TASK_DRY_RUN=%s\n" "$(shell_quote "${task_dry_run}")"
    printf "export OVERWRITE=%s\n" "$(shell_quote "${overwrite}")"
    printf "exec %s\n" "$(shell_quote "${task_script}")"
  } > "${wrapper_file}"
  chmod +x "${wrapper_file}"
  echo "${wrapper_file}"
}

submit_queue() {
  local queue="$1"
  local tc="$2"
  local subject_list="$3"
  local subject_count="$4"
  local job_name="$5"
  local queue_tag
  queue_tag="$(printf '%s' "${queue}" | tr '[:upper:]' '[:lower:]')"
  local log_dir="${log_root}/${queue}"
  mkdir -p "${log_dir}"
  local job_wrapper
  job_wrapper="$(write_job_wrapper "${queue_tag}" "${subject_list}")"

  local qsub_cmd=(
    qsub
    -N "${job_name}"
    -q "${queue}"
    -t "1-${subject_count}"
    -tc "${tc}"
    -pe smp 3
    -l mem_free=64G
    -l h_rt=24:00:00
    -j y
    -o "${log_dir}/${job_name}.\$JOB_ID.\$TASK_ID.log"
    "${job_wrapper}"
  )

  echo "[submit] ${queue}: subjects=${subject_count} tc=${tc} task_dry_run=${task_dry_run} dry_run=${dry_run}" >&2
  echo "[submit] ${queue}: subject_list=${subject_list}" >&2
  echo "[submit] ${queue}: job_wrapper=${job_wrapper}" >&2
  echo "[submit] ${queue}: out_root=${out_root}" >&2
  echo "[submit] ${queue}: ${qsub_cmd[*]}" >&2

  if [ "${dry_run}" = "0" ]; then
    "${qsub_cmd[@]}"
  fi
}

echo "[dual-submit] dataset=${dataset_label} total=${#all_ids[@]} data_root=${data_root}" >&2

if [ "${#ui_ids[@]}" -gt 0 ]; then
  ui_list="$(write_list ui "${ui_ids[@]}")"
  submit_queue "UI" "${ui_tc}" "${ui_list}" "${#ui_ids[@]}" "${job_prefix}_ui"
fi

if [ "${#pinc_ids[@]}" -gt 0 ]; then
  pinc_list="$(write_list pinc "${pinc_ids[@]}")"
  submit_queue "PINC" "${pinc_tc}" "${pinc_list}" "${#pinc_ids[@]}" "${job_prefix}_pinc"
fi

echo "[done] launcher completed (dry_run=${dry_run}, smoke_n=${smoke_n}, task_dry_run=${task_dry_run})" >&2
