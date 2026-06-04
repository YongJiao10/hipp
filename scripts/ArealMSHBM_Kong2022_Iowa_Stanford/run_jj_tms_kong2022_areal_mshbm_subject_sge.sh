#!/bin/bash

#$ -N jj_k22_mshbm
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

data_root="${DATA_ROOT:-/Shared/jianglab/3_Data_Working/fmriprep_processed_JJ_TMS/derivatives_surf}"
out_root="${OUT_ROOT:-/Shared/jianglab/3_Data_Working/fmriprep_processed_JJ_TMS/ArealMSHBM_Kong2022}"
subject_ids_file="${SUBJECT_IDS_FILE:-}"
subject_ids_csv="${SUBJECT_IDS_CSV:-}"
python_exe="${PYTHON_EXE:-/Users/yojiao/miniconda3/envs/py313/bin/python}"
cbig_code_dir="${CBIG_CODE_DIR:-${repo_root}/external/CBIG}"
group_prior="${GROUP_PRIOR:-${cbig_code_dir}/stable_projects/brain_parcellation/Kong2022_ArealMSHBM/lib/group_priors/HCP_fs_LR_32k_40sub/400/gMSHBM/beta50/Params_Final.mat}"
overwrite="${OVERWRITE:-0}"
task_dry_run="${TASK_DRY_RUN:-${DRY_RUN:-0}}"
matlab_mode="${MATLAB_MODE:-batch}"

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
subject_out="${out_root}/${subject}"
tmp_root="/tmp/jj_tms_kong2022_${job_id}_${task_id}_${subject}"
project_dir="${tmp_root}/cbig_project"
run_list="${tmp_root}/inputs/task_rest_dtseries.txt"
censor_list="${tmp_root}/inputs/task_rest_censors.txt"
lh_labels="${subject_out}/lh_labels.txt"
rh_labels="${subject_out}/rh_labels.txt"

cleanup_tmp_root() {
  case "${tmp_root:-}" in
    /tmp/jj_tms_kong2022_*)
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

mkdir -p "${subject_out}" "${project_dir}"

echo "=================================================================="
echo "JJ_TMS Kong2022 ArealMSHBM"
echo "JobID=${job_id} TaskID=${task_id} Host=${host_name} Subject=${subject}"
echo "Repo=${repo_root}"
echo "DataRoot=${data_root}"
echo "Out=${subject_out}"
echo "Work=${tmp_root}"
echo "TASK_DRY_RUN=${task_dry_run} OVERWRITE=${overwrite}"
echo "=================================================================="

required=(
  "${python_exe}"
  "${script_dir}/prepare_jj_tms_subject_inputs.py"
  "${script_dir}/matlab/run_kong2022_areal_mshbm_subject.m"
  "${cbig_code_dir}/stable_projects/brain_parcellation/Kong2022_ArealMSHBM/CBIG_ArealMSHBM_parcellation_single_subject.m"
  "${cbig_code_dir}/stable_projects/brain_parcellation/Kong2022_ArealMSHBM/lib/spatial_mask/400/spatial_mask_fs_LR_32k.mat"
  "${cbig_code_dir}/external_packages/python/mapalign-master/mapalign/embed.py"
  "${cbig_code_dir}/utilities/matlab/stats/CBIG_corr.m"
  "${cbig_code_dir}/utilities/matlab/parcellation/CBIG_ComputeConnectedComponentsFromSurface.m"
  "${cbig_code_dir}/data/templates/surface/fs_LR_32k_downsample_900/fslr_downsample_900mesh_parcellation.dlabel.nii"
  "${group_prior}"
)
for path in "${required[@]}"; do
  if [ ! -e "${path}" ]; then
    echo "ERROR: missing required path: ${path}" >&2
    exit 3
  fi
done

"${python_exe}" "${script_dir}/prepare_jj_tms_subject_inputs.py" \
  --subject "${subject}" \
  --data-root "${data_root}" \
  --subject-output-dir "${tmp_root}" \
  $([ "${overwrite}" = "1" ] && printf "%s" "--overwrite")

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

matlab_expr="addpath('${script_dir}/matlab'); run_kong2022_areal_mshbm_subject('${project_dir}', '${subject_out}', '${run_list}', '${censor_list}', '${cbig_code_dir}', '${group_prior}', ${overwrite_flag});"
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

echo "SUCCESS: ${subject} label text outputs ready under ${subject_out}"
