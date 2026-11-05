#!/bin/bash

set -euo pipefail
export LANG=C
export LC_ALL=C
export LC_CTYPE=C

repo_root="${REPO_ROOT:-/old_Users/yojiao/hipp}"
script_dir="${SCRIPT_DIR:-${repo_root}/scripts/ArealMSHBM_Kong2022}"
task_script="${RUNWISE_TASK_SCRIPT:-${script_dir}/run_kong2022_areal_mshbm_runwise_task_sge.sh}"

system_name="${SYSTEM_NAME:-${CLUSTER_NAME:-${SGE_CLUSTER_NAME:-}}}"
system_name="$(printf '%s' "${system_name}" | tr '[:upper:]-' '[:lower:]_')"
if [ -z "${system_name}" ]; then
  host_name="$(hostname 2>/dev/null || true)"
  case "$(printf '%s' "${host_name}" | tr '[:upper:]-' '[:lower:]_')" in
    *idas*) system_name="idas" ;;
    *argon*) system_name="argon" ;;
  esac
fi
case "${system_name}" in
  idas) max_concurrency=10 ;;
  argon) max_concurrency=20 ;;
  *)
    echo "ERROR: unable to identify execution system; set SYSTEM_NAME=idas or SYSTEM_NAME=argon (detected: ${system_name:-unknown})." >&2
    exit 2
    ;;
esac

variant="$(printf '%s' "${IOWA7T_VARIANT:-}" | tr '[:upper:]' '[:lower:]')"
data_root="${DATA_ROOT:-}"
if [ -z "${variant}" ] && [ -n "${data_root}" ]; then
  case "${data_root##*/}" in
    DataOrg_PFM_36P) variant="36p" ;;
  esac
fi
if [ -z "${variant}" ]; then variant="36p"; fi

case "${variant}" in
  36p)
    variant_upper="36P"
    default_data_root="/Shared/jianglab/3_Data_Working/fmriprep_processed_Iowa7T/DataOrg_PFM_36P"
    ;;
  *)
    echo "ERROR: IOWA7T_VARIANT must be 36p, got: ${variant}" >&2
    exit 2
    ;;
esac
data_root="${data_root:-${default_data_root}}"
if [ "${data_root##*/}" != "DataOrg_PFM_${variant_upper}" ]; then
  echo "ERROR: IOWA7T_VARIANT=${variant} conflicts with DATA_ROOT=${data_root}." >&2
  exit 2
fi
if [ ! -d "${data_root}" ]; then
  echo "ERROR: DATA_ROOT not found: ${data_root}" >&2
  exit 2
fi

subject_id="${SUBJECT_ID:-${SUBJECT_IDS_CSV:-06112026}}"
subject_id="${subject_id#sub-}"
if [[ "${subject_id}" == *,* ]] || ! [[ "${subject_id}" =~ ^[0-9]+$ ]]; then
  echo "ERROR: runwise launcher accepts exactly one numeric SUBJECT_ID, got: ${subject_id}" >&2
  exit 2
fi
subject="sub-${subject_id}"

run_tag="${RUN_TAG:-${subject_id}_$(date +%Y%m%d)}"
if ! [[ "${run_tag}" =~ ^[A-Za-z0-9._-]+$ ]]; then
  echo "ERROR: RUN_TAG contains unsupported characters: ${run_tag}" >&2
  exit 2
fi
run_id="${RUN_ID:-iowa7t_${variant}_runwise_${run_tag}}"
if ! [[ "${run_id}" =~ ^[A-Za-z0-9._-]+$ ]]; then
  echo "ERROR: RUN_ID contains unsupported characters: ${run_id}" >&2
  exit 2
fi

out_root="${OUT_ROOT:-${data_root}/kong2022_runwise_${run_tag}}"
out_root="${out_root%/}"
old_out_root="${data_root}/hipp_outputs"
if [ "${out_root}" = "${old_out_root}" ] || [[ "${out_root}" == "${old_out_root}/"* ]]; then
  echo "ERROR: runwise OUT_ROOT must not overlap the existing hipp_outputs tree: ${out_root}" >&2
  exit 2
fi
case "${out_root}" in
  "${data_root}/"*) ;;
  *)
    echo "ERROR: runwise OUT_ROOT must be inside DATA_ROOT: ${out_root}" >&2
    exit 2
    ;;
esac

for acquisition in ME01 ME02 MB03 MB04; do
  if [ ! -d "${data_root}/${subject}_${acquisition}" ]; then
    echo "ERROR: missing runwise source directory: ${data_root}/${subject}_${acquisition}" >&2
    exit 3
  fi
done

runwise_sources=(
  "${data_root}/${subject}_ME01/func/${subject}_ME01_task-rest_run-01_space-fsLR_den-91k_desc-MEICA_XCPD${variant_upper}_denoised_bold.dtseries.nii"
  "${data_root}/${subject}_ME02/func/${subject}_ME02_task-rest_run-02_space-fsLR_den-91k_desc-MEICA_XCPD${variant_upper}_denoised_bold.dtseries.nii"
  "${data_root}/${subject}_MB03/func/${subject}_MB03_task-rest_dir-PA_run-03_space-fsLR_den-91k_desc-denoised_bold.dtseries.nii"
  "${data_root}/${subject}_MB04/func/${subject}_MB04_task-rest_dir-AP_run-04_space-fsLR_den-91k_desc-denoised_bold.dtseries.nii"
)
for source_dtseries in "${runwise_sources[@]}"; do
  if [ ! -s "${source_dtseries}" ] || [ ! -r "${source_dtseries}" ]; then
    echo "ERROR: missing, empty, or unreadable runwise source dtseries: ${source_dtseries}" >&2
    exit 3
  fi
done

if [ ! -x "${task_script}" ]; then
  echo "ERROR: missing executable runwise task script: ${task_script}" >&2
  exit 3
fi
if ! bash -n "${task_script}"; then
  echo "ERROR: runwise task script failed Bash syntax validation: ${task_script}" >&2
  exit 3
fi

tc="${TC:-4}"
if ! [[ "${tc}" =~ ^[0-9]+$ ]] || [ "${tc}" -lt 1 ] || [ "${tc}" -gt "${max_concurrency}" ]; then
  echo "ERROR: TC must be an integer from 1 to ${max_concurrency}, got: ${tc}" >&2
  exit 2
fi
dry_run="${DRY_RUN:-0}"
task_dry_run="${TASK_DRY_RUN:-0}"
overwrite="${OVERWRITE:-0}"
matlab_mode="${MATLAB_MODE:-batch}"
wb_command="${WB_COMMAND:-${repo_root}/software/workbench/bin_linux64/wb_command}"
cbig_code_dir="${CBIG_CODE_DIR:-${repo_root}/external/CBIG}"
group_prior="${GROUP_PRIOR:-${cbig_code_dir}/stable_projects/brain_parcellation/Kong2022_ArealMSHBM/lib/group_priors/HCP_fs_LR_32k_40sub/400/gMSHBM/beta50/Params_Final.mat}"
queue="${QUEUE:-UI}"

case "${dry_run}" in
  0|1) ;;
  *)
    echo "ERROR: DRY_RUN must be 0 or 1, got: ${dry_run}" >&2
    exit 2
    ;;
esac
case "${task_dry_run}" in
  0|1) ;;
  *)
    echo "ERROR: TASK_DRY_RUN must be 0 or 1, got: ${task_dry_run}" >&2
    exit 2
    ;;
esac
case "${overwrite}" in
  0|1) ;;
  *)
    echo "ERROR: OVERWRITE must be 0 or 1, got: ${overwrite}" >&2
    exit 2
    ;;
esac

manifest_dir="${out_root}/manifests"
log_dir="${out_root}/logs"
wrapper_file="${manifest_dir}/launcher.job.sh"

shell_quote() {
  printf '%q' "$1"
}

write_wrapper() {
  mkdir -p "${manifest_dir}" "${log_dir}"
  {
    echo '#!/bin/bash'
    echo 'set -euo pipefail'
    printf 'export DATASET=iowa7t\n'
    printf 'export REPO_ROOT=%s\n' "$(shell_quote "${repo_root}")"
    printf 'export SCRIPT_DIR=%s\n' "$(shell_quote "${script_dir}")"
    printf 'export SYSTEM_NAME=%s\n' "$(shell_quote "${system_name}")"
    printf 'export DATA_ROOT=%s\n' "$(shell_quote "${data_root}")"
    printf 'export OUT_ROOT=%s\n' "$(shell_quote "${out_root}")"
    printf 'export SUBJECT_ID=%s\n' "$(shell_quote "${subject_id}")"
    printf 'export IOWA7T_VARIANT=%s\n' "$(shell_quote "${variant}")"
    printf 'export RUN_ID=%s\n' "$(shell_quote "${run_id}")"
    printf 'export WB_COMMAND=%s\n' "$(shell_quote "${wb_command}")"
    printf 'export CBIG_CODE_DIR=%s\n' "$(shell_quote "${cbig_code_dir}")"
    printf 'export GROUP_PRIOR=%s\n' "$(shell_quote "${group_prior}")"
    printf 'export TASK_DRY_RUN=%s\n' "$(shell_quote "${task_dry_run}")"
    printf 'export OVERWRITE=%s\n' "$(shell_quote "${overwrite}")"
    printf 'export MATLAB_MODE=%s\n' "$(shell_quote "${matlab_mode}")"
    printf 'exec %s\n' "$(shell_quote "${task_script}")"
  } > "${wrapper_file}"
  chmod +x "${wrapper_file}"
}

qsub_cmd=(
  qsub
  -N "ia7t_${variant}_rw"
  -q "${queue}"
  -t 1-4
  -tc "${tc}"
  -pe smp 4
  -l mem_free=64G
  -l h_rt=24:00:00
  -j y
  -o "${log_dir}/run-\$TASK_ID.\$JOB_ID.log"
  "${wrapper_file}"
)

echo "[runwise] system=${system_name} cap=${max_concurrency} variant=${variant} subject=${subject}"
echo "[runwise] data_root=${data_root}"
echo "[runwise] out_root=${out_root}"
echo "[runwise] run_id=${run_id}"
echo "[runwise] runs=ME01,ME02,MB03,MB04 -> run-01,run-02,run-03,run-04"
echo "[runwise] task_dry_run=${task_dry_run} overwrite=${overwrite} tc=${tc}"

printf '[runwise] qsub command:'
for arg in "${qsub_cmd[@]}"; do
  printf ' %q' "${arg}"
done
printf '\n'

if [ "${dry_run}" = "1" ]; then
  echo "[runwise] DRY_RUN=1: qsub not submitted; no output directories or wrappers created."
  exit 0
fi
if ! command -v qsub >/dev/null 2>&1; then
  echo "ERROR: qsub command not found in PATH" >&2
  exit 127
fi
if [ -e "${out_root}" ] && [ "${overwrite}" != "1" ]; then
  echo "ERROR: OUT_ROOT already exists; choose a new RUN_TAG/RUN_ID or set OVERWRITE=1 explicitly: ${out_root}" >&2
  exit 2
fi

write_wrapper
qsub_output="$("${qsub_cmd[@]}")"
echo "[runwise] submitted: ${qsub_output}"
echo "[runwise] wrapper=${wrapper_file}"
echo "[runwise] logs=${log_dir}"
