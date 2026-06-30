#!/bin/bash

#$ -N k22_mshbm
#$ -pe smp 3
#$ -l mem_free=64G
#$ -l h_rt=24:00:00
#$ -j y
#$ -o /tmp/$JOB_NAME.$JOB_ID.$TASK_ID.log

set -euo pipefail
export LANG=C
export LC_ALL=C
export LC_CTYPE=C
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1

repo_root="${REPO_ROOT:-/old_Users/yojiao/hipp}"
script_dir="${SCRIPT_DIR:-${repo_root}/scripts/ArealMSHBM_Kong2022_Iowa_Stanford}"

dataset="$(printf '%s' "${DATASET:-jj_tms}" | tr '[:upper:]-' '[:lower:]_')"
iowa7t_branch="${IOWA7T_BRANCH:-${BRANCH:-}}"
iowa7t_branch="$(printf '%s' "${iowa7t_branch}" | tr '[:upper:]-' '[:lower:]_')"
iowa7t_variant="${IOWA7T_VARIANT:-}"
iowa7t_variant="$(printf '%s' "${iowa7t_variant}" | tr '[:upper:]-' '[:lower:]_')"
case "${dataset}" in
  jj_tms)
    dataset_label="JJ_TMS"
    default_data_root="/Shared/jianglab/3_Data_Working/fmriprep_processed_JJ_TMS/xcpd_surf"
    default_out_root="/Shared/jianglab/3_Data_Working/fmriprep_processed_JJ_TMS/ArealMSHBM_Kong2022"
    tmp_prefix="jj_tms_kong2022"
    ;;
  stanford)
    dataset_label="Stanford"
    default_data_root="/Shared/jianglab/3_Data_Working/fmriprep_processed_CausalConnectome/rest_fmriprep/DataOrg_PFM_36P"
    default_out_root="/Shared/jianglab/3_Data_Working/fmriprep_processed_CausalConnectome/rest_fmriprep/ArealMSHBM_Kong2022"
    tmp_prefix="stanford_kong2022"
    ;;
  iowa7t)
    dataset_label="Iowa7T"
    default_data_root="/Shared/jianglab/3_Data_Working/fmriprep_processed_Iowa7T/DataOrg_PFM_36P"
    default_out_root="/Shared/jianglab/3_Data_Working/fmriprep_processed_Iowa7T/DataOrg_PFM_36P/hipp_outputs"
    tmp_prefix="iowa7t_kong2022"
    ;;
  iowa_hcp500)
    dataset_label="Iowa HCP500"
    default_data_root="/old_Users/yojiao/HCPpipeline500/work/qunex_iowa7t_hcppipeline500/sessions"
    default_out_root="${repo_root}/outputs_iowa_hcp500/inputs/iowa_hcp500/kong2022/cortex_references/routea_areal_mshbm"
    tmp_prefix="iowa_hcp500_kong2022"
    ;;
  *)
    echo "ERROR: unsupported DATASET=${DATASET:-}. Expected jj_tms, stanford, iowa7t, or iowa_hcp500." >&2
    exit 2
    ;;
esac

if [ "${dataset}" = "iowa7t" ]; then
  case "${iowa7t_branch}" in
    multiband|multi_echo)
      tmp_prefix="${tmp_prefix}_${iowa7t_branch}"
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
if [ "${dataset}" = "iowa7t" ] && [ "${iowa7t_variant}" = "noxcpd" ] && [ -z "${DATA_ROOT:-}" ]; then
  data_root="/Shared/jianglab/3_Data_Working/fmriprep_processed_Iowa7T/DataOrg_PFM_NoXCPD"
fi
if [ "${dataset}" = "iowa7t" ]; then
  if [ -z "${iowa7t_variant}" ]; then
    case "${data_root##*/}" in
      DataOrg_PFM_NoXCPD)
        iowa7t_variant="noxcpd"
        ;;
      *)
        iowa7t_variant="36p"
        ;;
    esac
  fi
  case "${iowa7t_variant}" in
    36p|noxcpd)
      ;;
    *)
      echo "ERROR: IOWA7T_VARIANT must be 36p or noxcpd, got: ${IOWA7T_VARIANT:-}" >&2
      exit 2
      ;;
  esac
fi
if [ "${dataset}" = "iowa7t" ] && [ "${iowa7t_variant}" = "noxcpd" ] && [ -z "${OUT_ROOT:-}" ]; then
  out_root="${data_root}/hipp_outputs"
else
  out_root="${OUT_ROOT:-${default_out_root}}"
fi
subject_ids_file="${SUBJECT_IDS_FILE:-}"
subject_ids_csv="${SUBJECT_IDS_CSV:-}"
python_exe="${PYTHON_EXE:-/Users/yojiao/miniconda3/envs/py313/bin/python}"
cbig_code_dir="${CBIG_CODE_DIR:-${repo_root}/external/CBIG}"
group_prior="${GROUP_PRIOR:-${cbig_code_dir}/stable_projects/brain_parcellation/Kong2022_ArealMSHBM/lib/group_priors/HCP_fs_LR_32k_40sub/400/gMSHBM/beta50/Params_Final.mat}"
wb_command="${WB_COMMAND:-${repo_root}/software/workbench/bin_linux64/wb_command}"
overwrite="${OVERWRITE:-0}"
keep_tmp="${KEEP_TMP:-$([ "${dataset}" = "iowa_hcp500" ] && printf 1 || printf 0)}"
task_dry_run="${TASK_DRY_RUN:-${DRY_RUN:-0}}"
matlab_mode="${MATLAB_MODE:-batch}"
run_id="${RUN_ID:-manual}"

export PATH="$(dirname "${wb_command}"):${PATH}"

task_id="${SUBJECT_INDEX:-${SGE_TASK_ID:-1}}"
job_id="${JOB_ID:-local}"
host_name="$(hostname)"

read_subjects() {
  if [ -n "${subject_ids_file}" ]; then
    if [ ! -f "${subject_ids_file}" ]; then
      echo "ERROR: SUBJECT_IDS_FILE not found: ${subject_ids_file}" >&2
      exit 2
    fi
    awk 'NF>0 {gsub(/^[ \t]+|[ \t]+$/, ""); sub(/^sub-/, ""); print}' "${subject_ids_file}"
  elif [ -n "${subject_ids_csv}" ]; then
    echo "${subject_ids_csv}" | tr ',' '\n' | awk 'NF>0 {gsub(/^[ \t]+|[ \t]+$/, ""); sub(/^sub-/, ""); print}'
  else
    find "${data_root}" -maxdepth 1 -type d -name 'sub-*' -printf '%f\n' | sed 's/^sub-//' | sort -V
  fi
}

mapfile -t subject_ids < <(read_subjects)
if [ "${#subject_ids[@]}" -lt 1 ]; then
  echo "ERROR: no subjects found. data_root=${data_root}" >&2
  exit 2
fi
if ! [[ "${task_id}" =~ ^[0-9]+$ ]] || [ "${task_id}" -lt 1 ] || [ "${task_id}" -gt "${#subject_ids[@]}" ]; then
  echo "ERROR: invalid task id ${task_id}; subject count=${#subject_ids[@]}" >&2
  exit 2
fi

sid="${subject_ids[$((task_id - 1))]}"
sid="${sid#sub-}"
subject="sub-${sid}"
if [ "${dataset}" = "iowa7t" ]; then
  subject_out="${out_root}/subjects/${subject}/${iowa7t_branch}/cortex/areal_mshbm"
else
  subject_out="${out_root}/${subject}"
fi
tmp_root="/tmp/${tmp_prefix}_${job_id}_${task_id}_${subject}"
project_dir="${tmp_root}/cbig_project"
run_list="${tmp_root}/inputs/task_rest_dtseries.txt"
censor_list="${tmp_root}/inputs/task_rest_censors.txt"
input_manifest="${tmp_root}/inputs/input_manifest.json"
lh_labels="${subject_out}/lh_labels.txt"
rh_labels="${subject_out}/rh_labels.txt"
provenance_path="${subject_out}/provenance.json"
run_start_utc="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
run_start_epoch="$(date -u +%s)"

cleanup_tmp_root() {
  if [ "${keep_tmp}" = "1" ]; then
    echo "KEEP_TMP=1: retaining ${tmp_root}" >&2
    return
  fi
  case "${tmp_root:-}" in
    /tmp/jj_tms_kong2022_*|/tmp/stanford_kong2022_*|/tmp/iowa7t_kong2022_*|/tmp/iowa_hcp500_kong2022_*)
      rm -rf -- "${tmp_root}"
      ;;
    "")
      ;;
    *)
      echo "WARN: refusing to clean unexpected tmp_root=${tmp_root}" >&2
      ;;
  esac
}
trap cleanup_tmp_root EXIT

mkdir -p "${project_dir}"

echo "=================================================================="
echo "${dataset_label} Kong2022 ArealMSHBM"
echo "JobID=${job_id} TaskID=${task_id} Host=${host_name} Subject=${subject}"
echo "Dataset=${dataset}"
echo "Branch=${iowa7t_branch:-<none>}"
echo "Variant=${iowa7t_variant:-<none>}"
echo "Repo=${repo_root}"
echo "DataRoot=${data_root}"
echo "SubjectIDsFile=${subject_ids_file:-<scan-data-root>}"
echo "SubjectCount=${#subject_ids[@]}"
echo "Out=${subject_out}"
echo "Work=${tmp_root}"
echo "WB_COMMAND=${wb_command}"
echo "TASK_DRY_RUN=${task_dry_run} OVERWRITE=${overwrite} KEEP_TMP=${keep_tmp}"
echo "=================================================================="

required=(
  "${script_dir}/run_kong2022_areal_mshbm_subject.m"
  "${wb_command}"
  "${cbig_code_dir}/stable_projects/brain_parcellation/Kong2022_ArealMSHBM/CBIG_ArealMSHBM_parcellation_single_subject.m"
  "${cbig_code_dir}/stable_projects/brain_parcellation/Kong2022_ArealMSHBM/lib/spatial_mask/400/spatial_mask_fs_LR_32k.mat"
  "${cbig_code_dir}/external_packages/python/mapalign-master/mapalign/embed.py"
  "${cbig_code_dir}/utilities/matlab/stats/CBIG_corr.m"
  "${cbig_code_dir}/utilities/matlab/parcellation/CBIG_ComputeConnectedComponentsFromSurface.m"
  "${cbig_code_dir}/data/templates/surface/fs_LR_32k_downsample_900/fslr_downsample_900mesh_parcellation.dlabel.nii"
  "${group_prior}"
)
if [ "${dataset}" != "iowa_hcp500" ] && [ "${dataset}" != "iowa7t" ]; then
  required+=(
    "${python_exe}"
    "${script_dir}/write_kong2022_areal_mshbm_provenance.py"
    "${script_dir}/prepare_kong2022_areal_mshbm_subject_inputs.py"
  )
fi
for path in "${required[@]}"; do
  if [ ! -e "${path}" ]; then
    echo "ERROR: missing required path: ${path}" >&2
    exit 3
  fi
done

json_escape() {
  printf '%s' "$1" | sed 's/\\/\\\\/g; s/"/\\"/g'
}

if [ "${dataset}" = "iowa7t" ]; then
  case "${iowa7t_branch}" in
    multiband)
      case "${iowa7t_variant}" in
        noxcpd)
          dtseries="${data_root}/${subject}/func/${subject}_task-rest_dir-concat_space-fsLR_den-91k_desc-NoXCPD_bold.dtseries.nii"
          ;;
        *)
          dtseries="${data_root}/${subject}/func/${subject}_task-rest_dir-concat_space-fsLR_den-91k_desc-denoised_bold.dtseries.nii"
          ;;
      esac
      ;;
    multi_echo)
      case "${iowa7t_variant}" in
        noxcpd)
          source_dtseries="${data_root}/${subject}/func/${subject}_task-rest_run-concat_space-fsLR_den-91k_desc-MEICA_NoXCPD_bold.dtseries.nii"
          derived_desc="MEICA_NoXCPD"
          ;;
        *)
          source_dtseries="${data_root}/${subject}/func/${subject}_task-rest_run-concat_space-fsLR_den-91k_desc-MEICA_XCPD36P_denoised_bold.dtseries.nii"
          derived_desc="MEICA_XCPD36P_denoised"
          ;;
      esac
      derived_input_dir="${out_root}/runs/derived_inputs/${run_id}/subjects/${subject}/multi_echo/cortex_only"
      dtseries="${derived_input_dir}/${subject}_task-rest_run-concat_space-fsLR_den-32k_desc-${derived_desc}_bold_cortexonly.dtseries.nii"
      lh_metric="${derived_input_dir}/${subject}_task-rest_run-concat_space-fsLR_den-32k_desc-${derived_desc}_bold_hemi-L.func.gii"
      rh_metric="${derived_input_dir}/${subject}_task-rest_run-concat_space-fsLR_den-32k_desc-${derived_desc}_bold_hemi-R.func.gii"
      ;;
  esac
  if [ "${iowa7t_branch}" = "multi_echo" ]; then
    if [ ! -s "${source_dtseries}" ] || [ ! -r "${source_dtseries}" ]; then
      echo "ERROR: missing, empty, or unreadable Iowa7T source dtseries: ${source_dtseries}" >&2
      exit 4
    fi
    if [ "${overwrite}" = "1" ] || [ ! -s "${dtseries}" ]; then
      mkdir -p "${derived_input_dir}"
      "${wb_command}" -cifti-separate "${source_dtseries}" COLUMN \
        -metric CORTEX_LEFT "${lh_metric}" \
        -metric CORTEX_RIGHT "${rh_metric}"
      "${wb_command}" -cifti-create-dense-timeseries "${dtseries}" \
        -left-metric "${lh_metric}" \
        -right-metric "${rh_metric}"
    fi
  fi
  if [ ! -s "${dtseries}" ] || [ ! -r "${dtseries}" ]; then
    echo "ERROR: missing, empty, or unreadable Iowa7T dtseries: ${dtseries}" >&2
    exit 4
  fi
  n_timepoints="$("${wb_command}" -file-information "${dtseries}" -only-number-of-maps)"
  if ! [[ "${n_timepoints}" =~ ^[0-9]+$ ]] || [ "${n_timepoints}" -lt 1 ]; then
    echo "ERROR: invalid timepoint count from Workbench: ${n_timepoints}" >&2
    exit 4
  fi
  inputs_dir="${tmp_root}/inputs"
  censor_dir="${inputs_dir}/censors"
  censor_path="${censor_dir}/${subject}_task-rest_run-01_censor.txt"
  mkdir -p "${censor_dir}"
  printf '%s\n' "${dtseries}" > "${run_list}"
  printf '%s\n' "${censor_path}" > "${censor_list}"
  awk -v count="${n_timepoints}" 'BEGIN { for (i = 0; i < count; i++) print 1 }' > "${censor_path}"
  cat > "${input_manifest}" <<EOF
{
  "branch": "$(json_escape "${iowa7t_branch}")",
  "censor_list": "$(json_escape "${censor_list}")",
  "data_root": "$(json_escape "${data_root}")",
  "dataset": "iowa7t",
  "dtseries_globs": [],
  "run_count": 1,
  "run_list": "$(json_escape "${run_list}")",
  "runs": [
    {
      "censor": "$(json_escape "${censor_path}")",
      "dtseries": "$(json_escape "${dtseries}")",
      "run_index": 1,
      "timepoints": ${n_timepoints}
    }
  ],
  "subject": "$(json_escape "${subject}")",
  "subject_dir": "$(json_escape "${data_root}/${subject}")",
  "variant": "$(json_escape "${iowa7t_variant}")"
}
EOF
  echo "Prepared Iowa7T ${iowa7t_branch} dtseries=${dtseries} timepoints=${n_timepoints}"
elif [ "${dataset}" = "iowa_hcp500" ]; then
  dtseries="${data_root}/${sid}/hcp/${sid}/MNINonLinear/Results/REST/REST_Atlas_MSMAll_hp0_clean_rclean.dtseries.nii"
  if [ ! -s "${dtseries}" ] || [ ! -r "${dtseries}" ]; then
    echo "ERROR: missing, empty, or unreadable Iowa HCP500 dtseries: ${dtseries}" >&2
    exit 4
  fi
  n_timepoints="$("${wb_command}" -file-information "${dtseries}" -only-number-of-maps)"
  if ! [[ "${n_timepoints}" =~ ^[0-9]+$ ]] || [ "${n_timepoints}" -lt 1 ]; then
    echo "ERROR: invalid timepoint count from Workbench: ${n_timepoints}" >&2
    exit 4
  fi
  inputs_dir="${tmp_root}/inputs"
  censor_dir="${inputs_dir}/censors"
  censor_path="${censor_dir}/${subject}_task-rest_run-01_censor.txt"
  mkdir -p "${censor_dir}"
  printf '%s\n' "${dtseries}" > "${run_list}"
  printf '%s\n' "${censor_path}" > "${censor_list}"
  awk -v count="${n_timepoints}" 'BEGIN { for (i = 0; i < count; i++) print 1 }' > "${censor_path}"
  echo "Prepared Iowa HCP500 dtseries=${dtseries} timepoints=${n_timepoints}"
else
  prep_args=(
    --dataset "${dataset}" \
    --subject "${subject}" \
    --data-root "${data_root}" \
    --subject-output-dir "${tmp_root}"
  )
  if [ -n "${iowa7t_branch}" ]; then
    prep_args+=(--branch "${iowa7t_branch}")
  fi
  if [ "${overwrite}" = "1" ]; then
    prep_args+=(--overwrite)
  fi
  "${python_exe}" "${script_dir}/prepare_kong2022_areal_mshbm_subject_inputs.py" "${prep_args[@]}"
fi

if [ ! -f "${run_list}" ] || [ ! -f "${censor_list}" ]; then
  echo "ERROR: prepared run/censor lists missing under ${tmp_root}/inputs" >&2
  exit 4
fi

if [ "${task_dry_run}" = "1" ]; then
  echo "DRY_RUN_OK: prepared inputs and verified required files for ${subject}"
  exit 0
fi

if [ "${overwrite}" != "1" ] && [ -f "${lh_labels}" ] && [ -f "${rh_labels}" ]; then
  echo "SKIP: label text outputs already exist for ${subject}; set OVERWRITE=1 to rerun."
  exit 0
fi

if ! command -v matlab >/dev/null 2>&1; then
  if command -v module >/dev/null 2>&1; then
    module load matlab/R2021a >/dev/null 2>&1 || true
  fi
fi
if ! command -v matlab >/dev/null 2>&1; then
  echo "ERROR: matlab not found in PATH" >&2
  exit 127
fi

overwrite_flag=0
if [ "${overwrite}" = "1" ]; then
  overwrite_flag=1
fi

matlab_expr="addpath('${script_dir}'); run_kong2022_areal_mshbm_subject('${project_dir}', '${subject_out}', '${run_list}', '${censor_list}', '${cbig_code_dir}', '${group_prior}', ${overwrite_flag});"
run_expr="try, ${matlab_expr}, catch ME, disp(getReport(ME,'extended')); exit(1); end; exit(0);"

echo "MATLAB_MODE=${matlab_mode}"
case "${matlab_mode}" in
  batch)
    matlab -batch "${matlab_expr}"
    ;;
  batch_nodisplay)
    matlab -nodisplay -batch "${matlab_expr}"
    ;;
  r_nodesktop)
    matlab -nodisplay -nodesktop -r "${run_expr}"
    ;;
  *)
    echo "ERROR: unsupported MATLAB_MODE=${matlab_mode}" >&2
    exit 5
    ;;
esac

if [ ! -f "${lh_labels}" ] || [ ! -f "${rh_labels}" ]; then
  echo "ERROR: MATLAB completed but label text outputs are missing under ${subject_out}" >&2
  exit 6
fi

if [ -f "${input_manifest}" ]; then
  run_end_utc="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  run_end_epoch="$(date -u +%s)"
  runtime_seconds="$(awk -v start="${run_start_epoch}" -v end="${run_end_epoch}" 'BEGIN { printf "%.3f", end - start }')"
  if [ "${dataset}" = "iowa7t" ]; then
    cat > "${provenance_path}" <<EOF
{
  "branch": "$(json_escape "${iowa7t_branch}")",
  "dataset": "iowa7t",
  "generated_at_utc": "$(date -u +%Y-%m-%dT%H:%M:%SZ)",
  "input_cifti": "$(json_escape "${dtseries}")",
  "input_ciftis": [
    "$(json_escape "${dtseries}")"
  ],
  "source_cifti": "$(json_escape "${source_dtseries:-${dtseries}}")",
  "labels": {
    "lh": "$(json_escape "${lh_labels}")",
    "rh": "$(json_escape "${rh_labels}")"
  },
  "matlab": {
    "mode": "$(json_escape "${matlab_mode}")",
    "script": "$(json_escape "${script_dir}/run_kong2022_areal_mshbm_subject.m")"
  },
  "output_dir": "$(json_escape "${subject_out}")",
  "parameters": {
    "beta": "50",
    "c": "30",
    "cbig_code_dir": "$(json_escape "${cbig_code_dir}")",
    "group_prior": "$(json_escape "${group_prior}")",
    "model": "gMSHBM",
    "num_ROIs": "400",
    "overwrite": "$(json_escape "${overwrite}")",
    "target_mesh": "fs_LR_32k",
    "w": "30",
    "wb_command": "$(json_escape "${wb_command}")"
  },
  "qsub": {
    "host": "$(json_escape "${host_name}")",
    "job_id": "$(json_escape "${job_id}")",
    "task_id": "$(json_escape "${task_id}")"
  },
  "run_list": "$(json_escape "${run_list}")",
  "censor_list": "$(json_escape "${censor_list}")",
  "runs": [
    {
      "censor": "$(json_escape "${censor_path}")",
      "dtseries": "$(json_escape "${dtseries}")",
      "run_index": 1,
      "timepoints": ${n_timepoints}
    }
  ],
  "schema_version": 1,
  "subject": "$(json_escape "${subject}")",
  "timepoints": ${n_timepoints},
  "timing": {
    "run_end_utc": "$(json_escape "${run_end_utc}")",
    "run_start_utc": "$(json_escape "${run_start_utc}")",
    "runtime_seconds": ${runtime_seconds}
  },
  "variant": "$(json_escape "${iowa7t_variant}")"
}
EOF
  else
    "${python_exe}" "${script_dir}/write_kong2022_areal_mshbm_provenance.py" \
      --input-manifest "${input_manifest}" \
      --output-dir "${subject_out}" \
      --provenance "${provenance_path}" \
      --dataset "${dataset}" \
      --branch "${iowa7t_branch}" \
      --subject "${subject}" \
      --job-id "${job_id}" \
      --task-id "${task_id}" \
      --host "${host_name}" \
      --matlab-script "${script_dir}/run_kong2022_areal_mshbm_subject.m" \
      --cbig-code-dir "${cbig_code_dir}" \
      --group-prior "${group_prior}" \
      --wb-command "${wb_command}" \
      --matlab-mode "${matlab_mode}" \
      --overwrite "${overwrite}" \
      --run-start-utc "${run_start_utc}" \
      --run-end-utc "${run_end_utc}" \
      --runtime-seconds "${runtime_seconds}"
  fi
  echo "PROVENANCE: ${provenance_path}"
else
  echo "WARN: input manifest missing; provenance not written: ${input_manifest}" >&2
fi

echo "SUCCESS: ${subject} label text outputs ready under ${subject_out}"
