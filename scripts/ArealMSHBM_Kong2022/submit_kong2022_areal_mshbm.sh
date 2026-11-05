#!/bin/bash
set -euo pipefail
export LANG=C
export LC_ALL=C
export LC_CTYPE=C

repo_root="${REPO_ROOT:-/old_Users/yojiao/hipp}"
script_dir="${SCRIPT_DIR:-${repo_root}/scripts/ArealMSHBM_Kong2022}"

system_name="${SYSTEM_NAME:-${CLUSTER_NAME:-${SGE_CLUSTER_NAME:-}}}"
system_name="$(printf '%s' "${system_name}" | tr '[:upper:]-' '[:lower:]_')"
if [ -z "${system_name}" ]; then
  host_name="$(hostname 2>/dev/null || true)"
  case "$(printf '%s' "${host_name}" | tr '[:upper:]-' '[:lower:]_')" in
    *idas*)
      system_name="idas"
      ;;
    *argon*)
      system_name="argon"
      ;;
  esac
fi
case "${system_name}" in
  idas)
    max_concurrency=10
    default_ui_tc=3
    default_pinc_tc=7
    ;;
  argon)
    max_concurrency=20
    default_ui_tc=5
    default_pinc_tc=15
    ;;
  *)
    echo "ERROR: unable to identify execution system; set SYSTEM_NAME=idas or SYSTEM_NAME=argon (detected: ${system_name:-unknown})." >&2
    exit 2
    ;;
esac
task_script="${script_dir}/run_kong2022_areal_mshbm_subject_sge.sh"

dataset="$(printf '%s' "${DATASET:-jj_tms}" | tr '[:upper:]-' '[:lower:]_')"
stanford_variant="${STANFORD_VARIANT:-}"
stanford_variant="$(printf '%s' "${stanford_variant}" | tr '[:upper:]-' '[:lower:]_')"
estt_variant="${ESTT_VARIANT:-}"
estt_variant="$(printf '%s' "${estt_variant}" | tr '[:upper:]-' '[:lower:]_')"
iowa7t_branch="${IOWA7T_BRANCH:-${BRANCH:-}}"
iowa7t_branch="$(printf '%s' "${iowa7t_branch}" | tr '[:upper:]-' '[:lower:]_')"
iowa7t_variant="${IOWA7T_VARIANT:-}"
iowa7t_variant="$(printf '%s' "${iowa7t_variant}" | tr '[:upper:]-' '[:lower:]_')"
case "${dataset}" in
  jj_tms)
    dataset_label="JJ_TMS"
    default_data_root="/Shared/jianglab/3_Data_Working/fmriprep_processed_JJ_TMS/xcpd_surf"
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
  iowa7t)
    dataset_label="Iowa7T"
    default_data_root="/Shared/jianglab/3_Data_Working/fmriprep_processed_Iowa7T/DataOrg_PFM_36P"
    default_out_root="/Shared/jianglab/3_Data_Working/fmriprep_processed_Iowa7T/DataOrg_PFM_36P/hipp_outputs"
    job_prefix="ia7t_k22"
    manifest_prefix="iowa7t"
    ;;
  estt)
    dataset_label="ESTT"
    default_data_root="/Shared/jianglab/3_Data_Working/fmriprep_processed_esTT/DataOrg_PFM_36P"
    default_out_root="/Shared/jianglab/3_Data_Working/fmriprep_processed_esTT/DataOrg_PFM_36P/hipp_outputs/cortex_kong2022_areal_mshbm"
    job_prefix="estt_k22"
    manifest_prefix="estt"
    ;;
  iowa_hcp500)
    dataset_label="Iowa HCP500"
    default_data_root="/old_Users/yojiao/HCPpipeline500/work/qunex_iowa7t_hcppipeline500/sessions"
    default_out_root="${repo_root}/outputs_iowa_hcp500/inputs/kong2022/cortex_references/routea_areal_mshbm"
    job_prefix="ia500_k22"
    manifest_prefix="iowa_hcp500"
    ;;
  hcp7t)
    dataset_label="HCP7T"
    default_data_root="${repo_root}/data/hippunfold_input"
    default_out_root="${repo_root}/outputs_hcp/inputs/kong2022/cortex_references/pfm_areal_mshbm"
    job_prefix="hcp7t_k22"
    manifest_prefix="hcp7t"
    ;;
  pdc|banda)
    dataset_label="${dataset^^}"
    default_data_root="/Shared/jianglab/3_Data_Working/Depression/${dataset_label}/imaging"
    default_out_root="${repo_root}/outputs_${dataset}/inputs/kong2022/cortex_references/routea_areal_mshbm"
    job_prefix="${dataset}_k22"
    manifest_prefix="${dataset}"
    ;;
  *)
    echo "ERROR: unsupported DATASET=${DATASET:-}. Expected jj_tms, stanford, iowa7t, estt, iowa_hcp500, hcp7t, pdc, or banda." >&2
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
if [ "${dataset}" = "stanford" ]; then
  if [ -n "${stanford_variant}" ] && [ -z "${DATA_ROOT:-}" ]; then
    case "${stanford_variant}" in
      36p)
        data_root="/Shared/jianglab/3_Data_Working/fmriprep_processed_CausalConnectome/rest_fmriprep/DataOrg_PFM_36P"
        ;;
    esac
  fi
  if [ -z "${stanford_variant}" ]; then
    case "${data_root##*/}" in
      DataOrg_PFM_36P)
        stanford_variant="36p"
        ;;
      *)
        echo "ERROR: Stanford DATA_ROOT is not an identifiable DataOrg_PFM_36P root; set STANFORD_VARIANT=36p explicitly." >&2
        exit 2
        ;;
    esac
  fi
  case "${stanford_variant}" in
    36p)
      ;;
    *)
      echo "ERROR: STANFORD_VARIANT must be 36p, got: ${STANFORD_VARIANT:-}" >&2
      exit 2
      ;;
  esac
  case "${data_root##*/}" in
    DataOrg_PFM_36P) ;;
    DataOrg_PFM_*) echo "ERROR: STANFORD_VARIANT=${stanford_variant} conflicts with DATA_ROOT=${data_root}." >&2; exit 2 ;;
  esac
fi
if [ "${dataset}" = "estt" ]; then
  if [ -n "${estt_variant}" ] && [ -z "${DATA_ROOT:-}" ]; then
    case "${estt_variant}" in
      23p)
        data_root="/Shared/jianglab/3_Data_Working/fmriprep_processed_esTT/DataOrg_PFM_23P"
        ;;
      36p)
        data_root="/Shared/jianglab/3_Data_Working/fmriprep_processed_esTT/DataOrg_PFM_36P"
        ;;
    esac
  fi
  if [ -z "${estt_variant}" ]; then
    case "${data_root##*/}" in
      DataOrg_PFM_23P)
        estt_variant="23p"
        ;;
      DataOrg_PFM_36P)
        estt_variant="36p"
        ;;
      *)
        echo "ERROR: ESTT DATA_ROOT is not an identifiable DataOrg_PFM_23P or DataOrg_PFM_36P root; set ESTT_VARIANT=23p|36p explicitly." >&2
        exit 2
        ;;
    esac
  fi
  case "${estt_variant}" in
    23p|36p)
      ;;
    *)
      echo "ERROR: ESTT_VARIANT must be 23p or 36p, got: ${ESTT_VARIANT:-}" >&2
      exit 2
      ;;
  esac
  case "${data_root##*/}:${estt_variant}" in
    DataOrg_PFM_23P:23p|DataOrg_PFM_36P:36p) ;;
    DataOrg_PFM_*:*) echo "ERROR: ESTT_VARIANT=${estt_variant} conflicts with DATA_ROOT=${data_root}." >&2; exit 2 ;;
  esac
fi
if [ "${dataset}" = "iowa7t" ] && [ -n "${iowa7t_variant}" ] && [ -z "${DATA_ROOT:-}" ]; then
  case "${iowa7t_variant}" in
    0p)
      data_root="/Shared/jianglab/3_Data_Working/fmriprep_processed_Iowa7T/DataOrg_PFM_0P"
      ;;
    32p)
      data_root="/Shared/jianglab/3_Data_Working/fmriprep_processed_Iowa7T/DataOrg_PFM_32P"
      ;;
    noxcpd)
      data_root="/Shared/jianglab/3_Data_Working/fmriprep_processed_Iowa7T/DataOrg_PFM_NoXCPD"
      ;;
  esac
fi
if [ "${dataset}" = "iowa7t" ]; then
  if [ -z "${iowa7t_variant}" ]; then
    case "${data_root##*/}" in
      DataOrg_PFM_0P)
        iowa7t_variant="0p"
        ;;
      DataOrg_PFM_32P)
        iowa7t_variant="32p"
        ;;
      DataOrg_PFM_NoXCPD)
        iowa7t_variant="noxcpd"
        ;;
      *)
        iowa7t_variant="36p"
        ;;
    esac
  fi
  case "${iowa7t_variant}" in
    0p|32p|36p|noxcpd)
      ;;
    *)
      echo "ERROR: IOWA7T_VARIANT must be 0p, 32p, 36p, or noxcpd, got: ${IOWA7T_VARIANT:-}" >&2
      exit 2
      ;;
  esac
fi
if [ "${dataset}" = "iowa7t" ]; then
  case "${data_root##*/}:${iowa7t_variant}" in
    DataOrg_PFM_0P:0p|DataOrg_PFM_32P:32p|DataOrg_PFM_36P:36p|DataOrg_PFM_NoXCPD:noxcpd) ;;
    DataOrg_PFM_*:*) echo "ERROR: IOWA7T_VARIANT=${iowa7t_variant} conflicts with DATA_ROOT=${data_root}." >&2; exit 2 ;;
  esac
fi
if [ "${dataset}" = "iowa7t" ] && [ "${iowa7t_variant}" != "36p" ] && [ -z "${OUT_ROOT:-}" ]; then
  out_root="${data_root}/hipp_outputs"
elif [ "${dataset}" = "estt" ] && [ -z "${OUT_ROOT:-}" ]; then
  out_root="${data_root}/hipp_outputs/cortex_kong2022_areal_mshbm"
else
  out_root="${OUT_ROOT:-${default_out_root}}"
fi
python_exe="${PYTHON_EXE:-/Users/yojiao/miniconda3/envs/py313/bin/python}"
if [ "${dataset}" = "estt" ] && [ -z "${PYTHON_EXE:-}" ]; then
  python_exe="/Users/yojiao/miniconda3/envs/hippo_nnunet/bin/python"
fi
cbig_code_dir="${CBIG_CODE_DIR:-${repo_root}/external/CBIG}"
group_prior="${GROUP_PRIOR:-${cbig_code_dir}/stable_projects/brain_parcellation/Kong2022_ArealMSHBM/lib/group_priors/HCP_fs_LR_32k_40sub/400/gMSHBM/beta50/Params_Final.mat}"
wb_command="${WB_COMMAND:-${repo_root}/software/workbench/bin_linux64/wb_command}"

dry_run="${DRY_RUN:-0}"
task_dry_run="${TASK_DRY_RUN:-0}"
smoke_n="${SMOKE_N:-0}"
subjects_csv="${SUBJECT_IDS_CSV:-}"
subject_ids_file="${SUBJECT_IDS_FILE:-}"
overwrite="${OVERWRITE:-0}"
keep_tmp="${KEEP_TMP:-$([[ "${dataset}" = "iowa_hcp500" || "${dataset}" = "hcp7t" || "${dataset}" = "pdc" || "${dataset}" = "banda" ]] && printf 1 || printf 0)}"
hcp_run_metadata_root="${HCP_RUN_METADATA_ROOT:-${repo_root}/outputs_hcp/inputs/qc/run_metadata}"
cohort_manifest="${COHORT_MANIFEST:-}"
hold_jid="${HOLD_JID:-}"
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
  DATASET=jj_tms|stanford|iowa7t|estt|iowa_hcp500|hcp7t|pdc|banda
                        Dataset preset. Default: jj_tms.
  STANFORD_VARIANT=36p
                        Stanford input version. Auto-detected from DATA_ROOT;
                        defaults to 36p only when DATA_ROOT is unset.
  ESTT_VARIANT=23p|36p
                        ESTT input version. Auto-detected from DATA_ROOT;
                        defaults to 36p only when DATA_ROOT is unset.
  SYSTEM_NAME=idas|argon  Scheduler system override. Otherwise detected from
                        SGE_CLUSTER_NAME/CLUSTER_NAME/hostname.
  DRY_RUN=1             Print qsub commands without submitting.
  TASK_DRY_RUN=1        Submitted tasks validate inputs/dependencies and stop before MATLAB.
  SMOKE_N=N             Use only the first N selected subjects.
  SUBJECT_IDS_CSV=...   Explicit comma-separated subject IDs.
  SUBJECT_IDS_FILE=...  Explicit subject list file.
  UI_TC=N               UI array throttle. IDAS default: 3; Argon default: 5.
  PINC_TC=N             PINC array throttle. IDAS default: 7; Argon default: 15.
  OVERWRITE=1           Recompute existing outputs.
  KEEP_TMP=1            Retain task temporary files. Required for pdc/banda.
  HCP_RUN_METADATA_ROOT=... HCP7T physical-run metadata root; defaults to
                        outputs_hcp/inputs/qc/run_metadata.
  COHORT_MANIFEST=...   Exact metadata-derived PDC/BANDA input manifest.
  HOLD_JID=...          Explicit comma-separated prerequisite SGE job IDs.
  IOWA7T_BRANCH=...     Required for DATASET=iowa7t: multiband or multi_echo.
  IOWA7T_VARIANT=...    Optional for DATASET=iowa7t: 0p, 32p, 36p, or noxcpd.
                        Auto-detected from DATA_ROOT basename when possible.
  RUN_ID=...            Run identifier. Iowa7T defaults to <branch>_<timestamp>.

Queue behavior follows the canonical Kong2022 ArealMSHBM launcher in this directory.
The detected system is ${system_name}: IDAS allows at most 10 total tasks;
Argon uses UI -tc 5 and PINC -tc 15 by default, at most 20 total tasks.
Overrides must remain within the detected system cap.
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
if ! bash -n "${task_script}"; then
  echo "ERROR: task script failed Bash syntax validation: ${task_script}" >&2
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
if [ "$((ui_tc + pinc_tc))" -gt "${max_concurrency}" ]; then
  echo "ERROR: UI_TC+PINC_TC=${ui_tc}+${pinc_tc} exceeds ${system_name} concurrency cap ${max_concurrency}." >&2
  exit 2
fi

cohort_ids=()
if [ "${dataset}" = "pdc" ] || [ "${dataset}" = "banda" ]; then
  if [ "${keep_tmp}" != "1" ] || [ "${overwrite}" != "0" ]; then
    echo "ERROR: PDC/BANDA requires KEEP_TMP=1 OVERWRITE=0; existing outputs must remain protected." >&2
    exit 2
  fi
  cohort_manifest="${COHORT_MANIFEST:-${repo_root}/outputs_${dataset}/inputs/kong2022/manifests/${dataset}_cortex_inputs.json}"
  cohort_text="$("${python_exe}" "${script_dir}/prepare_kong2022_areal_mshbm_subject_inputs.py" \
    --action list-cohort --dataset "${dataset}" --data-root "${data_root}" \
    --cohort-manifest "${cohort_manifest}")"
  mapfile -t cohort_ids <<< "${cohort_text}"
fi

mkdir -p "${manifest_dir}" "${log_root}"

read_subjects() {
  if [ "${dataset}" = "pdc" ] || [ "${dataset}" = "banda" ]; then
    if [ -n "${subject_ids_file}" ] && [ -n "${subjects_csv}" ]; then
      echo "ERROR: select SUBJECT_IDS_FILE or SUBJECT_IDS_CSV, not both." >&2
      return 2
    elif [ -n "${subject_ids_file}" ]; then
      awk 'NF {sub(/\r$/, ""); sub(/^sub-/, ""); print}' "${subject_ids_file}"
    elif [ -n "${subjects_csv}" ]; then
      printf '%s\n' "${subjects_csv}" | tr ',' '\n' | sed 's/^sub-//'
    else
      printf '%s\n' "${cohort_ids[@]}"
    fi
    return
  fi
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

selected_ids="$(read_subjects)"
if [ -z "${selected_ids}" ]; then
  echo "ERROR: no valid subject IDs found" >&2
  exit 2
fi
mapfile -t all_ids <<< "${selected_ids}"

if [ "${dataset}" = "pdc" ] || [ "${dataset}" = "banda" ]; then
  for sid in "${all_ids[@]}"; do
    found=0
    for declared_id in "${cohort_ids[@]}"; do
      if [ "${sid}" = "${declared_id}" ]; then found=1; break; fi
    done
    if [ "${found}" != "1" ]; then
      echo "ERROR: ${sid} is not declared in ${cohort_manifest}." >&2
      exit 2
    fi
  done
fi

dup_ids="$(printf '%s\n' "${all_ids[@]}" | sort | uniq -d || true)"
if [ -n "${dup_ids}" ]; then
  echo "ERROR: duplicate subject IDs:" >&2
  echo "${dup_ids}" >&2
  exit 2
fi

if [ "${dataset}" = "hcp7t" ]; then
  for sid in "${all_ids[@]}"; do
    subject="sub-${sid}"
    dtseries="${data_root}/${subject}/func/${subject}_task-rest_run-concat.dtseries.nii"
    run_metadata="${hcp_run_metadata_root}/${subject}.json"
    subject_out="${out_root}/${subject}"
    if [ ! -s "${dtseries}" ] || [ ! -r "${dtseries}" ]; then
      echo "ERROR: missing, empty, or unreadable HCP7T concatenated CIFTI for ${subject}: ${dtseries}" >&2
      exit 2
    fi
    if [ ! -s "${run_metadata}" ] || [ ! -r "${run_metadata}" ]; then
      echo "ERROR: missing, empty, or unreadable HCP7T physical-run metadata for ${subject}: ${run_metadata}" >&2
      exit 2
    fi
    for existing in \
      "${subject_out}/lh_labels.txt" \
      "${subject_out}/rh_labels.txt" \
      "${subject_out}/provenance.json"; do
      if [ -e "${existing}" ]; then
        echo "ERROR: refusing to reuse an existing HCP7T result for ${subject}: ${existing}" >&2
        exit 2
      fi
    done
  done
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
    printf "export STANFORD_VARIANT=%s\n" "$(shell_quote "${stanford_variant}")"
    printf "export ESTT_VARIANT=%s\n" "$(shell_quote "${estt_variant}")"
    printf "export REPO_ROOT=%s\n" "$(shell_quote "${repo_root}")"
    printf "export SYSTEM_NAME=%s\n" "$(shell_quote "${system_name}")"
    printf "export SCRIPT_DIR=%s\n" "$(shell_quote "${script_dir}")"
    printf "export DATA_ROOT=%s\n" "$(shell_quote "${data_root}")"
    printf "export HCP_RUN_METADATA_ROOT=%s\n" "$(shell_quote "${hcp_run_metadata_root}")"
    printf "export OUT_ROOT=%s\n" "$(shell_quote "${out_root}")"
    printf "export IOWA7T_BRANCH=%s\n" "$(shell_quote "${iowa7t_branch}")"
    printf "export IOWA7T_VARIANT=%s\n" "$(shell_quote "${iowa7t_variant}")"
    printf "export RUN_ID=%s\n" "$(shell_quote "${run_id}")"
    printf "export SUBJECT_IDS_FILE=%s\n" "$(shell_quote "${subject_list}")"
    printf "export PYTHON_EXE=%s\n" "$(shell_quote "${python_exe}")"
    printf "export CBIG_CODE_DIR=%s\n" "$(shell_quote "${cbig_code_dir}")"
    printf "export GROUP_PRIOR=%s\n" "$(shell_quote "${group_prior}")"
    printf "export WB_COMMAND=%s\n" "$(shell_quote "${wb_command}")"
    printf "export TASK_DRY_RUN=%s\n" "$(shell_quote "${task_dry_run}")"
    printf "export OVERWRITE=%s\n" "$(shell_quote "${overwrite}")"
    printf "export KEEP_TMP=%s\n" "$(shell_quote "${keep_tmp}")"
    printf "export COHORT_MANIFEST=%s\n" "$(shell_quote "${cohort_manifest}")"
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
  if [ "${dataset}" = "iowa7t" ] || [ "${dataset}" = "estt" ]; then
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
  if [ -n "${hold_jid}" ]; then
    qsub_cmd=("${qsub_cmd[@]:0:1}" -hold_jid "${hold_jid}" "${qsub_cmd[@]:1}")
  fi

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

echo "[dual-submit] system=${system_name} cap=${max_concurrency} dataset=${dataset_label} StanfordVariant=${stanford_variant:-<none>} ESTTVariant=${estt_variant:-<none>} branch=${iowa7t_branch:-<none>} variant=${iowa7t_variant:-<none>} total=${#all_ids[@]} data_root=${data_root}" >&2

if [ "${#ui_ids[@]}" -gt 0 ]; then
  ui_list="$(write_list ui "${ui_ids[@]}")"
  submit_queue "UI" "${ui_tc}" "${ui_list}" "${#ui_ids[@]}" "${job_prefix}_ui"
fi

if [ "${#pinc_ids[@]}" -gt 0 ]; then
  pinc_list="$(write_list pinc "${pinc_ids[@]}")"
  submit_queue "PINC" "${pinc_tc}" "${pinc_list}" "${#pinc_ids[@]}" "${job_prefix}_pinc"
fi

echo "[done] launcher completed (dry_run=${dry_run}, smoke_n=${smoke_n}, task_dry_run=${task_dry_run})" >&2
