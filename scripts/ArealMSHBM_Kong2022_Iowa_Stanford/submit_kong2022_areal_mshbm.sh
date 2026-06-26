#!/bin/bash
set -euo pipefail
export LANG=C
export LC_ALL=C
export LC_CTYPE=C

repo_root="${REPO_ROOT:-/old_Users/yojiao/hipp}"
script_dir="${SCRIPT_DIR:-${repo_root}/scripts/ArealMSHBM_Kong2022_Iowa_Stanford}"
task_script="${script_dir}/run_kong2022_areal_mshbm_subject_sge.sh"

dataset="$(printf '%s' "${DATASET:-jj_tms}" | tr '[:upper:]-' '[:lower:]_')"
iowa7t_branch="${IOWA7T_BRANCH:-${BRANCH:-}}"
iowa7t_branch="$(printf '%s' "${iowa7t_branch}" | tr '[:upper:]-' '[:lower:]_')"
case "${dataset}" in
  jj_tms)
    dataset_label="JJ_TMS"
    default_data_root="/Shared/jianglab/3_Data_Working/fmriprep_processed_JJ_TMS/xcpd_surf"
    default_out_root="/Shared/jianglab/3_Data_Working/fmriprep_processed_JJ_TMS/ArealMSHBM_Kong2022"
    job_prefix="jj_k22"
    manifest_prefix="jj_tms"
    default_ui_tc=5
    default_pinc_tc=15
    ;;
  stanford)
    dataset_label="Stanford"
    default_data_root="/Shared/jianglab/3_Data_Working/fmriprep_processed_CausalConnectome/rest_fmriprep/DataOrg_PFM_36P"
    default_out_root="/Shared/jianglab/3_Data_Working/fmriprep_processed_CausalConnectome/rest_fmriprep/ArealMSHBM_Kong2022"
    job_prefix="st_k22"
    manifest_prefix="stanford"
    default_ui_tc=5
    default_pinc_tc=15
    ;;
  iowa7t)
    dataset_label="Iowa7T"
    default_data_root="/Shared/jianglab/3_Data_Working/fmriprep_processed_Iowa7T/DataOrg_PFM_36P"
    default_out_root="/Shared/jianglab/3_Data_Working/fmriprep_processed_Iowa7T/DataOrg_PFM_36P/hipp_outputs"
    job_prefix="ia7t_k22"
    manifest_prefix="iowa7t"
    default_ui_tc=5
    default_pinc_tc=15
    ;;
  iowa_hcp500)
    dataset_label="Iowa HCP500"
    default_data_root="/old_Users/yojiao/HCPpipeline500/work/qunex_iowa7t_hcppipeline500/sessions"
    default_out_root="${repo_root}/outputs_iowa_hcp500/inputs/iowa_hcp500/kong2022/cortex_references/routea_areal_mshbm"
    job_prefix="ia500_k22"
    manifest_prefix="iowa_hcp500"
    default_ui_tc=3
    default_pinc_tc=7
    ;;
  *)
    echo "ERROR: unsupported DATASET=${DATASET:-}. Expected jj_tms, stanford, iowa7t, or iowa_hcp500." >&2
    exit 2
    ;;
esac

if [ "${dataset}" = "iowa7t" ]; then
  case "${iowa7t_branch}" in
    multiband)
      job_prefix="ia7ts_k22"
      manifest_prefix="iowa7t_single"
      ;;
    multi_echo)
      job_prefix="ia7tm_k22"
      manifest_prefix="iowa7t_multi"
      ;;
    *)
      echo "ERROR: DATASET=iowa7t requires IOWA7T_BRANCH=multiband or multi_echo." >&2
      exit 2
      ;;
  esac
elif [ -n "${iowa7t_branch}" ]; then
  echo "ERROR: IOWA7T_BRANCH is only valid with DATASET=iowa7t." >&2
  exit 2
fi

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
keep_tmp="${KEEP_TMP:-$([ "${dataset}" = "iowa_hcp500" ] && printf 1 || printf 0)}"
ui_tc="${UI_TC:-${default_ui_tc}}"
pinc_tc="${PINC_TC:-${default_pinc_tc}}"
timestamp="$(date +%Y%m%d_%H%M%S)"
run_id="${RUN_ID:-${manifest_prefix}_${timestamp}}"
if [ "${dataset}" = "iowa7t" ]; then
  manifest_dir="${MANIFEST_DIR:-${out_root}/runs/manifests/${run_id}}"
  log_root="${LOG_ROOT:-${out_root}/runs/logs/${run_id}}"
else
  manifest_dir="${MANIFEST_DIR:-${out_root}/_manifests}"
  log_root="${LOG_ROOT:-${out_root}/_sge_logs}"
fi

usage() {
  cat <<EOF
Usage: $(basename "$0")

Environment controls:
  DATASET=jj_tms|stanford|iowa7t|iowa_hcp500
                        Dataset preset. Default: jj_tms.
  DRY_RUN=1             Print qsub commands without submitting.
  TASK_DRY_RUN=1        Submitted tasks validate inputs/dependencies and stop before MATLAB.
  SMOKE_N=N             Use only the first N selected subjects.
  SUBJECT_IDS_CSV=...   Explicit comma-separated subject IDs.
  SUBJECT_IDS_FILE=...  Explicit subject list file.
  UI_TC=N               UI array throttle. Default: 3 for iowa_hcp500, else 5.
  PINC_TC=N             PINC array throttle. Default: 7 for iowa_hcp500, else 15.
  OVERWRITE=1           Recompute existing outputs.
  KEEP_TMP=1            Retain task temporary files. Default: 1 for iowa_hcp500.
  IOWA7T_BRANCH=...     Required for DATASET=iowa7t: multiband or multi_echo.
  RUN_ID=...            Run identifier. Iowa7T defaults to <branch>_<timestamp>.

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

mkdir -p "${manifest_dir}" "${log_root}"

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
    if [ "${dataset}" = "iowa_hcp500" ]; then
      local session_dir sid rest_path
      for session_dir in "${data_root}"/*; do
        [ -d "${session_dir}" ] || continue
        sid="${session_dir##*/}"
        rest_path="${session_dir}/hcp/${sid}/MNINonLinear/Results/REST/REST_Atlas_MSMAll_hp0_clean_rclean.dtseries.nii"
        [ -s "${rest_path}" ] && printf '%s\n' "${sid}"
      done | sort -V
    else
      local subject_dir
      for subject_dir in "${data_root}"/sub-*; do
        [ -d "${subject_dir}" ] && printf '%s\n' "${subject_dir##*/sub-}"
      done | sort -V
    fi
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
    printf "export IOWA7T_BRANCH=%s\n" "$(shell_quote "${iowa7t_branch}")"
    printf "export RUN_ID=%s\n" "$(shell_quote "${run_id}")"
    printf "export SUBJECT_IDS_FILE=%s\n" "$(shell_quote "${subject_list}")"
    printf "export PYTHON_EXE=%s\n" "$(shell_quote "${python_exe}")"
    printf "export CBIG_CODE_DIR=%s\n" "$(shell_quote "${cbig_code_dir}")"
    printf "export GROUP_PRIOR=%s\n" "$(shell_quote "${group_prior}")"
    printf "export WB_COMMAND=%s\n" "$(shell_quote "${wb_command}")"
    printf "export TASK_DRY_RUN=%s\n" "$(shell_quote "${task_dry_run}")"
    printf "export OVERWRITE=%s\n" "$(shell_quote "${overwrite}")"
    printf "export KEEP_TMP=%s\n" "$(shell_quote "${keep_tmp}")"
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
  if [ "${dataset}" = "iowa7t" ]; then
    log_dir="${log_root}"
  fi
  mkdir -p "${log_dir}"
  local job_wrapper
  job_wrapper="$(write_job_wrapper "${queue_tag}" "${subject_list}")"

  local qsub_cmd=(
    qsub
    -N "${job_name}"
    -q "${queue}"
    -t "1-${subject_count}"
    -tc "${tc}"
    -pe smp 4
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
  echo "[submit] ${queue}: run_id=${run_id} log_root=${log_root}" >&2
  echo "[submit] ${queue}: ${qsub_cmd[*]}" >&2

  if [ "${dry_run}" = "0" ]; then
    "${qsub_cmd[@]}"
  fi
}

echo "[dual-submit] dataset=${dataset_label} branch=${iowa7t_branch:-<none>} total=${#all_ids[@]} data_root=${data_root}" >&2

if [ "${#ui_ids[@]}" -gt 0 ]; then
  ui_list="$(write_list ui "${ui_ids[@]}")"
  submit_queue "UI" "${ui_tc}" "${ui_list}" "${#ui_ids[@]}" "${job_prefix}_ui"
fi

if [ "${#pinc_ids[@]}" -gt 0 ]; then
  pinc_list="$(write_list pinc "${pinc_ids[@]}")"
  submit_queue "PINC" "${pinc_tc}" "${pinc_list}" "${#pinc_ids[@]}" "${job_prefix}_pinc"
fi

echo "[done] launcher completed (dry_run=${dry_run}, smoke_n=${smoke_n}, task_dry_run=${task_dry_run})" >&2
