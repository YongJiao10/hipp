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
export OMP_DYNAMIC=FALSE
export MKL_DYNAMIC=FALSE
export MKL_CBWR=COMPATIBLE

repo_root="${REPO_ROOT:-/old_Users/yojiao/hipp}"
script_dir="${SCRIPT_DIR:-${repo_root}/scripts/ArealMSHBM_Kong2022}"

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
  estt)
    dataset_label="ESTT"
    default_data_root="/Shared/jianglab/3_Data_Working/fmriprep_processed_esTT/DataOrg_PFM_36P"
    default_out_root="/Shared/jianglab/3_Data_Working/fmriprep_processed_esTT/DataOrg_PFM_36P/hipp_outputs/cortex_kong2022_areal_mshbm"
    tmp_prefix="estt_kong2022"
    ;;
  iowa_hcp500)
    dataset_label="Iowa HCP500"
    default_data_root="/old_Users/yojiao/HCPpipeline500/work/qunex_iowa7t_hcppipeline500/sessions"
    default_out_root="${repo_root}/outputs_iowa_hcp500/inputs/kong2022/cortex_references/routea_areal_mshbm"
    tmp_prefix="iowa_hcp500_kong2022"
    ;;
  hcp7t)
    dataset_label="HCP7T"
    default_data_root="${repo_root}/data/hippunfold_input"
    default_out_root="${repo_root}/outputs_hcp/inputs/kong2022/cortex_references/pfm_areal_mshbm"
    tmp_prefix="hcp7t_kong2022"
    ;;
  pdc|banda)
    dataset_label="${dataset^^}"
    default_data_root="/Shared/jianglab/3_Data_Working/Depression/${dataset_label}/imaging"
    default_out_root="${repo_root}/outputs_${dataset}/inputs/kong2022/cortex_references/routea_areal_mshbm"
    tmp_prefix="${dataset}_kong2022"
    ;;
  *)
    echo "ERROR: unsupported DATASET=${DATASET:-}. Expected jj_tms, stanford, iowa7t, estt, iowa_hcp500, hcp7t, pdc, or banda." >&2
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
if [ "${dataset}" = "stanford" ]; then
  if [ -n "${stanford_variant}" ] && [ -z "${DATA_ROOT:-}" ]; then
    case "${stanford_variant}" in
      22p)
        data_root="/Shared/jianglab/3_Data_Working/fmriprep_processed_CausalConnectome/rest_fmriprep/DataOrg_PFM_22P"
        ;;
      36p)
        data_root="/Shared/jianglab/3_Data_Working/fmriprep_processed_CausalConnectome/rest_fmriprep/DataOrg_PFM_36P"
        ;;
    esac
  fi
  if [ -z "${stanford_variant}" ]; then
    case "${data_root##*/}" in
      DataOrg_PFM_22P)
        stanford_variant="22p"
        ;;
      DataOrg_PFM_36P)
        stanford_variant="36p"
        ;;
      *)
        echo "ERROR: Stanford DATA_ROOT is not an identifiable DataOrg_PFM_22P or DataOrg_PFM_36P root; set STANFORD_VARIANT=22p|36p explicitly." >&2
        exit 2
        ;;
    esac
  fi
  case "${stanford_variant}" in
    22p|36p)
      ;;
    *)
      echo "ERROR: STANFORD_VARIANT must be 22p or 36p, got: ${STANFORD_VARIANT:-}" >&2
      exit 2
      ;;
  esac
  case "${data_root##*/}:${stanford_variant}" in
    DataOrg_PFM_22P:36p|DataOrg_PFM_36P:22p)
      echo "ERROR: STANFORD_VARIANT=${stanford_variant} conflicts with DATA_ROOT=${data_root}." >&2
      exit 2
      ;;
  esac
fi
if [ "${dataset}" = "estt" ]; then
  if [ -n "${estt_variant}" ] && [ -z "${DATA_ROOT:-}" ]; then
    case "${estt_variant}" in
      22p)
        data_root="/Shared/jianglab/3_Data_Working/fmriprep_processed_esTT/DataOrg_PFM_22P"
        ;;
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
      DataOrg_PFM_22P)
        estt_variant="22p"
        ;;
      DataOrg_PFM_23P)
        estt_variant="23p"
        ;;
      DataOrg_PFM_36P)
        estt_variant="36p"
        ;;
      *)
        echo "ERROR: ESTT DATA_ROOT is not an identifiable DataOrg_PFM_22P, DataOrg_PFM_23P, or DataOrg_PFM_36P root; set ESTT_VARIANT=22p|23p|36p explicitly." >&2
        exit 2
        ;;
    esac
  fi
  case "${estt_variant}" in
    22p|23p|36p)
      ;;
    *)
      echo "ERROR: ESTT_VARIANT must be 22p, 23p, or 36p, got: ${ESTT_VARIANT:-}" >&2
      exit 2
      ;;
  esac
  case "${data_root##*/}:${estt_variant}" in
    DataOrg_PFM_22P:23p|DataOrg_PFM_22P:36p|DataOrg_PFM_23P:22p|DataOrg_PFM_23P:36p|DataOrg_PFM_36P:22p|DataOrg_PFM_36P:23p)
      echo "ERROR: ESTT_VARIANT=${estt_variant} conflicts with DATA_ROOT=${data_root}." >&2
      exit 2
      ;;
  esac
fi
if [ "${dataset}" = "iowa7t" ] && [ -n "${iowa7t_variant}" ] && [ -z "${DATA_ROOT:-}" ]; then
  case "${iowa7t_variant}" in
    0p)
      data_root="/Shared/jianglab/3_Data_Working/fmriprep_processed_Iowa7T/DataOrg_PFM_0P"
      ;;
    22p)
      data_root="/Shared/jianglab/3_Data_Working/fmriprep_processed_Iowa7T/DataOrg_PFM_22P"
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
      DataOrg_PFM_22P)
        iowa7t_variant="22p"
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
    0p|22p|32p|36p|noxcpd)
      ;;
    *)
      echo "ERROR: IOWA7T_VARIANT must be 0p, 22p, 32p, 36p, or noxcpd, got: ${IOWA7T_VARIANT:-}" >&2
      exit 2
      ;;
  esac
fi
if [ "${dataset}" = "iowa7t" ] && [ "${iowa7t_variant}" != "36p" ] && [ -z "${OUT_ROOT:-}" ]; then
  out_root="${data_root}/hipp_outputs"
elif [ "${dataset}" = "stanford" ] && [ "${stanford_variant}" = "22p" ] && [ -z "${OUT_ROOT:-}" ]; then
  out_root="${data_root%/*}/ArealMSHBM_Kong2022_22P"
elif [ "${dataset}" = "estt" ] && [ -z "${OUT_ROOT:-}" ]; then
  out_root="${data_root}/hipp_outputs/cortex_kong2022_areal_mshbm"
else
  out_root="${OUT_ROOT:-${default_out_root}}"
fi
if [ "${dataset}" = "stanford" ] && [ "${stanford_variant}" = "22p" ]; then
  tmp_prefix="stanford_22p_kong2022"
elif [ "${dataset}" = "estt" ] && [ "${estt_variant}" = "22p" ]; then
  tmp_prefix="estt_22p_kong2022"
fi
subject_ids_file="${SUBJECT_IDS_FILE:-}"
subject_ids_csv="${SUBJECT_IDS_CSV:-}"
python_exe="${PYTHON_EXE:-/Users/yojiao/miniconda3/envs/py313/bin/python}"
if [ "${dataset}" = "estt" ] && [ -z "${PYTHON_EXE:-}" ]; then
  python_exe="/Users/yojiao/miniconda3/envs/hippo_nnunet/bin/python"
fi
cbig_code_dir="${CBIG_CODE_DIR:-${repo_root}/external/CBIG}"
group_prior="${GROUP_PRIOR:-${cbig_code_dir}/stable_projects/brain_parcellation/Kong2022_ArealMSHBM/lib/group_priors/HCP_fs_LR_32k_40sub/400/gMSHBM/beta50/Params_Final.mat}"
wb_command="${WB_COMMAND:-${repo_root}/software/workbench/bin_linux64/wb_command}"
kong2022_dlabel="${KONG2022_DLABEL:-${repo_root}/external/Schaefer2018_Kong2022/Schaefer2018_400Parcels_Kong2022_17Networks_order.dlabel.nii}"
overwrite="${OVERWRITE:-0}"
keep_tmp="${KEEP_TMP:-$([[ "${dataset}" = "iowa_hcp500" || "${dataset}" = "hcp7t" || "${dataset}" = "pdc" || "${dataset}" = "banda" ]] && printf 1 || printf 0)}"
hcp_run_metadata_root="${HCP_RUN_METADATA_ROOT:-${repo_root}/outputs_hcp/inputs/qc/run_metadata}"
cohort_manifest="${COHORT_MANIFEST:-}"
if [ "${dataset}" = "pdc" ] || [ "${dataset}" = "banda" ]; then
  if [ "${keep_tmp}" != "1" ] || [ "${overwrite}" != "0" ]; then
    echo "ERROR: PDC/BANDA requires KEEP_TMP=1 OVERWRITE=0." >&2
    exit 2
  fi
  if [ -z "${subject_ids_file}" ] && [ -z "${subject_ids_csv}" ]; then
    echo "ERROR: PDC/BANDA tasks require an explicit subject list from the shared launcher." >&2
    exit 2
  fi
  cohort_manifest="${COHORT_MANIFEST:-${repo_root}/outputs_${dataset}/inputs/kong2022/manifests/${dataset}_cortex_inputs.json}"
fi
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
if [ "${dataset}" = "estt" ]; then
  # ESTT keeps the same subject-level work/output contract as the completed
  # cohort. Iowa7T remains under its own subjects/<id>/<branch> tree.
  tmp_root="${subject_out}/kong2022_areal_mshbm"
  project_dir="${tmp_root}/project_dir"
else
  if [ "${dataset}" = "hcp7t" ]; then
    tmp_root="${repo_root}/tmp/${tmp_prefix}_${run_id}_${job_id}_${task_id}_${subject}"
  else
    tmp_root="/tmp/${tmp_prefix}_${job_id}_${task_id}_${subject}"
  fi
  if [ "${dataset}" = "pdc" ] || [ "${dataset}" = "banda" ]; then
    tmp_root="$(mktemp -d "/tmp/${tmp_prefix}_${job_id}_${task_id}_${subject}.XXXXXX")"
  fi
  project_dir="${tmp_root}/cbig_project"
fi
run_list="${tmp_root}/inputs/task_rest_dtseries.txt"
censor_list="${tmp_root}/inputs/task_rest_censors.txt"
input_manifest="${tmp_root}/inputs/input_manifest.json"
lh_labels="${subject_out}/lh_labels.txt"
rh_labels="${subject_out}/rh_labels.txt"
provenance_path="${subject_out}/provenance.json"
if [ "${dataset}" = "pdc" ] || [ "${dataset}" = "banda" ]; then
  for existing in "${lh_labels}" "${rh_labels}" "${provenance_path}"; do
    if [ -e "${existing}" ]; then
      echo "ERROR: refusing to reuse an existing PDC/BANDA result: ${existing}; inspect its input provenance before resubmission." >&2
      exit 2
    fi
  done
  run_list="${subject_out}/inputs/task_rest_dtseries.txt"
  censor_list="${subject_out}/inputs/task_rest_censors.txt"
  input_manifest="${subject_out}/inputs/input_manifest.json"
fi
if [ "${dataset}" = "hcp7t" ] && [ "${overwrite}" != "1" ]; then
  for existing in "${lh_labels}" "${rh_labels}" "${provenance_path}"; do
    if [ -e "${existing}" ]; then
      echo "ERROR: refusing to reuse an existing HCP7T result: ${existing}; inspect its input provenance before resubmission." >&2
      exit 2
    fi
  done
fi
run_start_utc="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
run_start_epoch="$(date -u +%s)"

cleanup_tmp_root() {
  if [ "${keep_tmp}" = "1" ]; then
    echo "KEEP_TMP=1: retaining ${tmp_root}" >&2
    return
  fi
  case "${tmp_root:-}" in
    /tmp/jj_tms_kong2022_*|/tmp/stanford_kong2022_*|/tmp/stanford_22p_kong2022_*|/tmp/iowa7t_kong2022_*|/tmp/iowa_hcp500_kong2022_*)
      rm -rf -- "${tmp_root}"
      ;;
    */cortex_kong2022_areal_mshbm/sub-*/kong2022_areal_mshbm)
      echo "ESTT: retaining subject work directory ${tmp_root}" >&2
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
echo "StanfordVariant=${stanford_variant:-<none>}"
echo "ESTTVariant=${estt_variant:-<none>}"
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
if [ "${dataset}" = "pdc" ] || [ "${dataset}" = "banda" ]; then
  required+=("${cohort_manifest}" "${script_dir}/prepare_kong2022_areal_mshbm_subject_inputs.py")
fi
if [ "${dataset}" = "estt" ]; then
  required+=("${kong2022_dlabel}")
fi
if [ "${dataset}" = "hcp7t" ]; then
  required+=("${hcp_run_metadata_root}")
fi
for path in "${required[@]}"; do
  if [ ! -e "${path}" ]; then
    echo "ERROR: missing required path: ${path}" >&2
    exit 3
  fi
done

if [ "${dataset}" = "estt" ] && ! "${python_exe}" -c 'import nibabel, numpy' >/dev/null 2>&1; then
  echo "ERROR: ESTT materialization requires nibabel and numpy in PYTHON_EXE: ${python_exe}" >&2
  exit 127
fi
if [ "${dataset}" = "estt" ]; then
  "${python_exe}" -m py_compile \
    "${script_dir}/write_kong2022_areal_mshbm_provenance.py" \
    "${script_dir}/run_cortex_mshbm_subject.py"
fi

json_escape() {
  printf '%s' "$1" | sed 's/\\/\\\\/g; s/"/\\"/g'
}

left_inflated=""
right_inflated=""
left_midthickness=""
right_midthickness=""
sulc_dscalar=""

if [ "${dataset}" = "pdc" ] || [ "${dataset}" = "banda" ]; then
  "${python_exe}" "${script_dir}/prepare_kong2022_areal_mshbm_subject_inputs.py" \
    --action prepare-cohort \
    --dataset "${dataset}" --data-root "${data_root}" \
    --cohort-manifest "${cohort_manifest}" --subject "${subject}" \
    --subject-output-dir "${subject_out}"
elif [ "${dataset}" = "estt" ]; then
  subject_dir="${data_root}/${subject}"
  dtseries="${subject_dir}/func/${subject}_task-rest_run-concat.dtseries.nii"
  left_midthickness="${subject_dir}/anat/${subject}_hemi-L_space-fsLR_den-32k_midthickness.surf.gii"
  right_midthickness="${subject_dir}/anat/${subject}_hemi-R_space-fsLR_den-32k_midthickness.surf.gii"
  left_inflated="${subject_dir}/anat/${subject}_hemi-L_space-fsLR_den-32k_inflated.surf.gii"
  right_inflated="${subject_dir}/anat/${subject}_hemi-R_space-fsLR_den-32k_inflated.surf.gii"
  sulc_dscalar="${subject_dir}/anat/${subject}_space-fsLR_den-91k_sulc.dscalar.nii"
  for path in "${dtseries}" "${left_midthickness}" "${right_midthickness}" "${left_inflated}" "${right_inflated}" "${sulc_dscalar}"; do
    if [ ! -s "${path}" ] || [ ! -r "${path}" ]; then
      echo "ERROR: missing, empty, or unreadable ESTT input: ${path}" >&2
      exit 4
    fi
  done
  if [ "${overwrite}" != "1" ] && \
     [ -f "${lh_labels}" ] && [ -f "${rh_labels}" ] && \
     [ -f "${provenance_path}" ] && \
     [ -f "${subject_out}/summary.json" ] && [ -f "${subject_out}/stage_manifest.json" ] && \
     [ -f "${subject_out}/sub-${sid}_atlas-kong2022_areal_mshbm.dlabel.nii" ]; then
    echo "SKIP: ESTT subject already has the complete Kong2022 output contract: ${subject_out}" >&2
    exit 0
  fi
  mkdir -p "${tmp_root}/runs" "${tmp_root}/inputs/censors"
  n_timepoints="$(${wb_command} -file-information "${dtseries}" -only-number-of-maps | tr -d '[:space:]')"
  if ! [[ "${n_timepoints}" =~ ^[0-9]+$ ]] || [ "${n_timepoints}" -lt 4 ]; then
    echo "ERROR: invalid ESTT dtseries timepoint count: ${n_timepoints}" >&2
    exit 4
  fi
  base=$((n_timepoints / 4))
  remainder=$((n_timepoints % 4))
  start=1
  run_lengths=()
  : > "${run_list}"
  : > "${censor_list}"
  for session_id in 1 2 3 4; do
    length="${base}"
    if [ "${session_id}" -le "${remainder}" ]; then
      length=$((length + 1))
    fi
    end=$((start + length - 1))
    run_path="${tmp_root}/runs/${subject}_task-rest_run-${session_id}.dtseries.nii"
    censor_path="${tmp_root}/inputs/censors/${subject}_task-rest_run-${session_id}_censor.txt"
    "${wb_command}" -cifti-merge "${run_path}" -direction ROW -cifti "${dtseries}" -index "${start}" -up-to "${end}"
    awk -v count="${length}" 'BEGIN { for (i = 1; i <= count; i++) print 1 }' > "${censor_path}"
    printf '%s\n' "${run_path}" >> "${run_list}"
    printf '%s\n' "${censor_path}" >> "${censor_list}"
    run_lengths+=("${length}")
    start=$((end + 1))
  done
  if [ "${start}" -ne "$((n_timepoints + 1))" ]; then
    echo "ERROR: ESTT run split mismatch: end=${start} expected=$((n_timepoints + 1))" >&2
    exit 4
  fi
  cat > "${input_manifest}" <<EOF
{
  "branch": null,
  "censor_list": "$(json_escape "${censor_list}")",
  "data_root": "$(json_escape "${data_root}")",
  "dataset": "estt",
  "variant": "$(json_escape "${estt_variant}")",
  "dtseries_globs": [],
  "run_count": 4,
  "run_list": "$(json_escape "${run_list}")",
  "source_dtseries": "$(json_escape "${dtseries}")",
  "runs": [
$(for session_id in 1 2 3 4; do
  run_path="${tmp_root}/runs/${subject}_task-rest_run-${session_id}.dtseries.nii"
  censor_path="${tmp_root}/inputs/censors/${subject}_task-rest_run-${session_id}_censor.txt"
  comma=","; [ "${session_id}" -eq 4 ] && comma=""
  printf '    {"censor": "%s", "dtseries": "%s", "run_index": %d, "timepoints": %d}%s\n' \
    "$(json_escape "${censor_path}")" "$(json_escape "${run_path}")" "${session_id}" "${run_lengths[$((session_id - 1))]}" "${comma}"
done)
  ],
  "subject": "$(json_escape "${subject}")",
  "subject_dir": "$(json_escape "${subject_dir}")"
}
EOF
  echo "Prepared ESTT dtseries=${dtseries} timepoints=${n_timepoints} run_lengths=${run_lengths[*]}"
elif [ "${dataset}" = "iowa7t" ]; then
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
        0p)
          source_dtseries="${data_root}/${subject}/func/${subject}_task-rest_run-concat_space-fsLR_den-91k_desc-MEICA_XCPD0P_denoised_bold.dtseries.nii"
          derived_desc="MEICA_XCPD0P_denoised"
          ;;
        32p)
          source_dtseries="${data_root}/${subject}/func/${subject}_task-rest_run-concat_space-fsLR_den-91k_desc-MEICA_XCPD32P_denoised_bold.dtseries.nii"
          derived_desc="MEICA_XCPD32P_denoised"
          ;;
        22p)
          source_dtseries="${data_root}/${subject}/func/${subject}_task-rest_run-concat_space-fsLR_den-91k_desc-MEICA_XCPD22P_denoised_bold.dtseries.nii"
          derived_desc="MEICA_XCPD22P_denoised"
          ;;
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
elif [ "${dataset}" = "hcp7t" ]; then
  dtseries="${data_root}/${subject}/func/${subject}_task-rest_run-concat.dtseries.nii"
  run_metadata="${hcp_run_metadata_root}/${subject}.json"
  if [ ! -s "${dtseries}" ] || [ ! -r "${dtseries}" ]; then
    echo "ERROR: missing, empty, or unreadable HCP7T concatenated CIFTI: ${dtseries}" >&2
    exit 4
  fi
  if [ ! -s "${run_metadata}" ] || [ ! -r "${run_metadata}" ]; then
    echo "ERROR: missing, empty, or unreadable HCP7T physical-run metadata: ${run_metadata}" >&2
    exit 4
  fi
  hcp_metadata_text="$("${python_exe}" - "${run_metadata}" "${subject}" <<'PY'
import json
import sys

path, expected_subject = sys.argv[1:]
with open(path, encoding="utf-8") as stream:
    metadata = json.load(stream)
expected_ids = [1, 2, 3, 4]
lengths = metadata.get("run_lengths")
runs = metadata.get("runs")
if metadata.get("subject") != expected_subject:
    raise SystemExit(f"ERROR: subject mismatch in HCP7T run metadata: {path}")
if metadata.get("run_ids") != expected_ids:
    raise SystemExit(f"ERROR: expected physical run IDs {expected_ids} in {path}")
if not isinstance(lengths, list) or len(lengths) != 4 or any(type(n) is not int or n <= 0 for n in lengths):
    raise SystemExit(f"ERROR: invalid four-run lengths in {path}")
if metadata.get("concat_timepoints") != sum(lengths):
    raise SystemExit(f"ERROR: concatenated frame count disagrees with run lengths in {path}")
if not isinstance(runs, list) or [row.get("run_id") for row in runs] != expected_ids:
    raise SystemExit(f"ERROR: run records do not match the declared order in {path}")
if [row.get("n_timepoints") for row in runs] != lengths:
    raise SystemExit(f"ERROR: run frame counts disagree with run_lengths in {path}")
labels = [row.get("run_label") for row in runs]
if any(not isinstance(label, str) or not label for label in labels):
    raise SystemExit(f"ERROR: missing physical run labels in {path}")
print(metadata["concat_timepoints"])
for length, label in zip(lengths, labels):
    print(length)
    print(label)
PY
)"
  mapfile -t hcp_metadata_values <<< "${hcp_metadata_text}"
  concat_timepoints="${hcp_metadata_values[0]}"
  source_run_lengths=("${hcp_metadata_values[1]}" "${hcp_metadata_values[3]}" "${hcp_metadata_values[5]}" "${hcp_metadata_values[7]}")
  run_labels=("${hcp_metadata_values[2]}" "${hcp_metadata_values[4]}" "${hcp_metadata_values[6]}" "${hcp_metadata_values[8]}")
  n_timepoints="$("${wb_command}" -file-information "${dtseries}" -only-number-of-maps | tr -d '[:space:]')"
  if ! [[ "${n_timepoints}" =~ ^[0-9]+$ ]] || [ "${n_timepoints}" -ne "${concat_timepoints}" ]; then
    echo "ERROR: HCP7T CIFTI/metadata map-count mismatch for ${subject}: ${dtseries} maps=${n_timepoints}, metadata=${concat_timepoints}" >&2
    exit 4
  fi
  inputs_dir="${tmp_root}/inputs"
  run_dir="${inputs_dir}/runs"
  censor_dir="${inputs_dir}/censors"
  mkdir -p "${run_dir}" "${censor_dir}"
  : > "${run_list}"
  : > "${censor_list}"
  run_lengths=()
  start=1
  for run_index in 1 2 3 4; do
    length="${source_run_lengths[$((run_index - 1))]}"
    end=$((start + length - 1))
    run_path="${run_dir}/${subject}_task-rest_run-${run_index}.dtseries.nii"
    censor_path="${censor_dir}/${subject}_task-rest_run-${run_index}_censor.txt"
    "${wb_command}" -cifti-merge "${run_path}" -direction ROW -cifti "${dtseries}" -index "${start}" -up-to "${end}"
    split_timepoints="$("${wb_command}" -file-information "${run_path}" -only-number-of-maps | tr -d '[:space:]')"
    if [ "${split_timepoints}" != "${length}" ]; then
      echo "ERROR: HCP7T run split length mismatch for ${subject} run ${run_index}: expected=${length}, actual=${split_timepoints}" >&2
      exit 4
    fi
    awk -v count="${length}" 'BEGIN { for (i = 1; i <= count; i++) print 1 }' > "${censor_path}"
    printf '%s\n' "${run_path}" >> "${run_list}"
    printf '%s\n' "${censor_path}" >> "${censor_list}"
    run_lengths+=("${length}")
    start=$((end + 1))
  done
  if [ "${start}" -ne "$((concat_timepoints + 1))" ]; then
    echo "ERROR: HCP7T run split boundaries do not cover the concatenated CIFTI for ${subject}" >&2
    exit 4
  fi
  run_lengths_json="$(printf '%s\n' "${run_lengths[@]}" | "${python_exe}" -c 'import json,sys; print(json.dumps([int(line) for line in sys.stdin if line.strip()]))')"
  cat > "${input_manifest}" <<EOF
{
  "branch": null,
  "censor_list": "$(json_escape "${censor_list}")",
  "data_root": "$(json_escape "${data_root}")",
  "dataset": "hcp7t",
  "run_count": 4,
  "run_lengths": ${run_lengths_json},
  "run_list": "$(json_escape "${run_list}")",
  "run_metadata": "$(json_escape "${run_metadata}")",
  "source_dtseries": "$(json_escape "${dtseries}")",
  "runs": [
$(start=1
for run_index in 1 2 3 4; do
  length="${run_lengths[$((run_index - 1))]}"
  end=$((start + length - 1))
  run_path="${run_dir}/${subject}_task-rest_run-${run_index}.dtseries.nii"
  censor_path="${censor_dir}/${subject}_task-rest_run-${run_index}_censor.txt"
  run_label="${run_labels[$((run_index - 1))]}"
  comma=","; [ "${run_index}" -eq 4 ] && comma=""
  printf '    {"censor": "%s", "dtseries": "%s", "run_index": %d, "run_id": %d, "run_label": "%s", "source_end_index": %d, "source_start_index": %d, "timepoints": %d}%s\n' \
    "$(json_escape "${censor_path}")" "$(json_escape "${run_path}")" "${run_index}" "${run_index}" \
    "$(json_escape "${run_label}")" "${end}" "${start}" "${length}" "${comma}"
  start=$((end + 1))
done)
  ],
  "subject": "$(json_escape "${subject}")",
  "subject_dir": "$(json_escape "${data_root}/${subject}")",
  "concat_timepoints": ${concat_timepoints}
}
EOF
  echo "Prepared HCP7T source=${dtseries} run_lengths=${run_lengths[*]} timepoints=${concat_timepoints}"
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

write_estt_provenance() {
  if [ ! -f "${input_manifest}" ]; then
    echo "ERROR: cannot finalize ESTT outputs without input manifest: ${input_manifest}" >&2
    return 1
  fi
  run_end_utc="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  run_end_epoch="$(date -u +%s)"
  runtime_seconds="$(awk -v start="${run_start_epoch}" -v end="${run_end_epoch}" 'BEGIN { printf "%.3f", end - start }')"
  "${python_exe}" "${script_dir}/write_kong2022_areal_mshbm_provenance.py" \
    --input-manifest "${input_manifest}" \
    --output-dir "${subject_out}" \
    --provenance "${provenance_path}" \
    --dataset "${dataset}" \
    --branch "" \
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
    --runtime-seconds "${runtime_seconds}" \
    --materialize-label-assets \
    --kong2022-dlabel "${kong2022_dlabel}" \
    --left-inflated "${left_inflated}" \
    --right-inflated "${right_inflated}" \
    --left-midthickness "${left_midthickness}" \
    --right-midthickness "${right_midthickness}" \
    --sulc-dscalar "${sulc_dscalar}" \
    --run-lengths "${run_lengths[@]}" \
    --matlab-log "${matlab_log}"
  echo "PROVENANCE: ${provenance_path}"
}

if [ "${overwrite}" != "1" ] && [ -f "${lh_labels}" ] && [ -f "${rh_labels}" ]; then
  if [ "${dataset}" = "estt" ] && {
    [ ! -f "${subject_out}/summary.json" ] ||
    [ ! -f "${subject_out}/stage_manifest.json" ] ||
    [ ! -f "${provenance_path}" ] ||
    [ ! -f "${subject_out}/sub-${sid}_atlas-kong2022_areal_mshbm.dlabel.nii" ];
  }; then
    matlab_log="${subject_out}/matlab_wrapper.log"
    echo "RESUME: ESTT labels exist; finalizing missing provenance/materialized assets for ${subject}."
    write_estt_provenance
    exit 0
  fi
  if [ "${dataset}" = "hcp7t" ] && [ ! -s "${provenance_path}" ]; then
    echo "ERROR: HCP7T labels exist without provenance; inspect ${subject_out} before retrying." >&2
    exit 6
  fi
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
wb_dir="$(dirname "${wb_command}")"
matlab_expr="setenv('WB_COMMAND','${wb_command}'); setenv('PATH',['${wb_dir}:' getenv('PATH')]); ${matlab_expr}"
run_expr="try, ${matlab_expr}, catch ME, disp(getReport(ME,'extended')); exit(1); end; exit(0);"

echo "MATLAB_MODE=${matlab_mode}"
run_matlab() {
  case "${matlab_mode}" in
    batch)
      matlab -singleCompThread -batch "${matlab_expr}"
      ;;
    batch_nodisplay)
      matlab -singleCompThread -nodisplay -batch "${matlab_expr}"
      ;;
    r_nodesktop)
      matlab -singleCompThread -nodisplay -nodesktop -r "${run_expr}"
      ;;
    *)
      echo "ERROR: unsupported MATLAB_MODE=${matlab_mode}" >&2
      return 5
      ;;
  esac
}
if [ "${dataset}" = "estt" ]; then
  matlab_log="${subject_out}/matlab_wrapper.log"
  run_matlab > "${matlab_log}" 2>&1
else
  run_matlab
fi

if [ ! -f "${lh_labels}" ] || [ ! -f "${rh_labels}" ]; then
  echo "ERROR: MATLAB completed but label text outputs are missing under ${subject_out}" >&2
  exit 6
fi

if [ -f "${input_manifest}" ]; then
  run_end_utc="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  run_end_epoch="$(date -u +%s)"
  runtime_seconds="$(awk -v start="${run_start_epoch}" -v end="${run_end_epoch}" 'BEGIN { printf "%.3f", end - start }')"
  if [ "${dataset}" = "estt" ]; then
    "${python_exe}" "${script_dir}/write_kong2022_areal_mshbm_provenance.py" \
      --input-manifest "${input_manifest}" \
      --output-dir "${subject_out}" \
      --provenance "${provenance_path}" \
      --dataset "${dataset}" \
      --branch "" \
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
      --runtime-seconds "${runtime_seconds}" \
      --materialize-label-assets \
      --kong2022-dlabel "${kong2022_dlabel}" \
      --left-inflated "${left_inflated}" \
      --right-inflated "${right_inflated}" \
      --left-midthickness "${left_midthickness}" \
      --right-midthickness "${right_midthickness}" \
      --sulc-dscalar "${sulc_dscalar}" \
      --run-lengths "${run_lengths[@]}" \
      --matlab-log "${matlab_log}"
  elif [ "${dataset}" = "iowa7t" ]; then
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
