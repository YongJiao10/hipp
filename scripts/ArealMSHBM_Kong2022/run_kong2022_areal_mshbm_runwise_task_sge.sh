#!/bin/bash

#$ -N ia7t_k22_rw
#$ -pe smp 4
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
data_root="${DATA_ROOT:?DATA_ROOT is required}"
out_root="${OUT_ROOT:?OUT_ROOT is required}"
subject_id="${SUBJECT_ID:?SUBJECT_ID is required}"
variant="${IOWA7T_VARIANT:?IOWA7T_VARIANT is required}"
run_id="${RUN_ID:-manual}"
run_index="${RUN_INDEX:-${SGE_TASK_ID:-}}"
task_id="${SGE_TASK_ID:-${run_index:-1}}"
job_id="${JOB_ID:-local}"
host_name="$(hostname)"
overwrite="${OVERWRITE:-0}"
task_dry_run="${TASK_DRY_RUN:-0}"
matlab_mode="${MATLAB_MODE:-batch}"
wb_command="${WB_COMMAND:-${repo_root}/software/workbench/bin_linux64/wb_command}"
cbig_code_dir="${CBIG_CODE_DIR:-${repo_root}/external/CBIG}"
group_prior="${GROUP_PRIOR:-${cbig_code_dir}/stable_projects/brain_parcellation/Kong2022_ArealMSHBM/lib/group_priors/HCP_fs_LR_32k_40sub/400/gMSHBM/beta50/Params_Final.mat}"

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

sid="${subject_id#sub-}"
subject="sub-${sid}"
old_out_root="${data_root}/hipp_outputs"

case "${variant}" in
  22p)
    variant_upper="22P"
    ;;
  36p)
    variant_upper="36P"
    ;;
  *)
    echo "ERROR: IOWA7T_VARIANT must be 22p or 36p, got: ${variant}" >&2
    exit 2
    ;;
esac

if ! [[ "${sid}" =~ ^[0-9]+$ ]]; then
  echo "ERROR: invalid SUBJECT_ID=${subject_id}" >&2
  exit 2
fi
if ! [[ "${run_index}" =~ ^[1-4]$ ]]; then
  echo "ERROR: RUN_INDEX/SGE_TASK_ID must be 1..4, got: ${run_index:-<empty>}" >&2
  exit 2
fi
if [ "${out_root}" = "${old_out_root}" ] || [[ "${out_root}" == "${old_out_root}/"* ]]; then
  echo "ERROR: runwise OUT_ROOT must not overlap the existing hipp_outputs tree: ${out_root}" >&2
  exit 2
fi
case "${out_root}" in
  "${data_root}/"*)
    ;;
  *)
    echo "ERROR: runwise OUT_ROOT must be inside DATA_ROOT: ${out_root}" >&2
    exit 2
    ;;
esac

run_label="$(printf 'run-%02d' "${run_index}")"
source_subject_dir=""
source_dtseries=""
input_kind=""
acquisition=""

case "${run_index}" in
  1)
    input_kind="multi_echo"
    acquisition="ME01"
    source_subject_dir="${data_root}/${subject}_ME01"
    source_dtseries="${source_subject_dir}/func/${subject}_ME01_task-rest_run-01_space-fsLR_den-91k_desc-MEICA_XCPD${variant_upper}_denoised_bold.dtseries.nii"
    ;;
  2)
    input_kind="multi_echo"
    acquisition="ME02"
    source_subject_dir="${data_root}/${subject}_ME02"
    source_dtseries="${source_subject_dir}/func/${subject}_ME02_task-rest_run-02_space-fsLR_den-91k_desc-MEICA_XCPD${variant_upper}_denoised_bold.dtseries.nii"
    ;;
  3)
    input_kind="multiband"
    acquisition="MB03"
    source_subject_dir="${data_root}/${subject}_MB03"
    source_dtseries="${source_subject_dir}/func/${subject}_MB03_task-rest_dir-PA_run-03_space-fsLR_den-91k_desc-denoised_bold.dtseries.nii"
    ;;
  4)
    input_kind="multiband"
    acquisition="MB04"
    source_subject_dir="${data_root}/${subject}_MB04"
    source_dtseries="${source_subject_dir}/func/${subject}_MB04_task-rest_dir-AP_run-04_space-fsLR_den-91k_desc-denoised_bold.dtseries.nii"
    ;;
esac

subject_out="${out_root}/${run_label}"
manifest_dir="${out_root}/manifests/${run_label}"
derived_dir=""
lh_labels="${subject_out}/lh_labels.txt"
rh_labels="${subject_out}/rh_labels.txt"
provenance_path="${subject_out}/provenance.json"
run_start_utc="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
run_start_epoch="$(date -u +%s)"

complete=1
for output_file in "${lh_labels}" "${rh_labels}" "${provenance_path}"; do
  if [ ! -s "${output_file}" ]; then
    complete=0
    break
  fi
done
if [ "${complete}" -eq 1 ] && [ "${overwrite}" != "1" ]; then
  echo "SKIP: ${subject} ${run_label} already has complete runwise Kong2022 outputs under ${subject_out}"
  exit 0
fi
if [ "${overwrite}" != "1" ]; then
  for output_file in "${lh_labels}" "${rh_labels}" "${provenance_path}"; do
    if [ -e "${output_file}" ]; then
      echo "ERROR: partial runwise output exists; set OVERWRITE=1 to replace it: ${subject_out}" >&2
      exit 2
    fi
  done
fi

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
  "${source_dtseries}"
)
for path in "${required[@]}"; do
  if [ ! -s "${path}" ] || [ ! -r "${path}" ]; then
    echo "ERROR: missing, empty, or unreadable runwise input/dependency: ${path}" >&2
    exit 3
  fi
done
if [ ! -x "${wb_command}" ]; then
  echo "ERROR: WB_COMMAND is not executable: ${wb_command}" >&2
  exit 3
fi

json_escape() {
  printf '%s' "$1" | sed 's/\\/\\\\/g; s/"/\\"/g'
}

matlab_quote() {
  printf '%s' "$1" | sed "s/'/''/g"
}

input_dtseries="${source_dtseries}"
if [ "${input_kind}" = "multi_echo" ]; then
  derived_dir="${out_root}/derived_inputs/${run_label}"
  derived_desc="MEICA_XCPD${variant_upper}_denoised"
  lh_metric="${derived_dir}/${subject}_task-rest_${run_label}_space-fsLR_den-32k_desc-${derived_desc}_bold_hemi-L.func.gii"
  rh_metric="${derived_dir}/${subject}_task-rest_${run_label}_space-fsLR_den-32k_desc-${derived_desc}_bold_hemi-R.func.gii"
  input_dtseries="${derived_dir}/${subject}_task-rest_${run_label}_space-fsLR_den-32k_desc-${derived_desc}_bold_cortexonly.dtseries.nii"
  if [ "${overwrite}" = "1" ] || [ ! -s "${lh_metric}" ] || [ ! -s "${rh_metric}" ] || [ ! -s "${input_dtseries}" ]; then
    mkdir -p "${derived_dir}"
    "${wb_command}" -cifti-separate "${source_dtseries}" COLUMN \
      -metric CORTEX_LEFT "${lh_metric}" \
      -metric CORTEX_RIGHT "${rh_metric}"
    "${wb_command}" -cifti-create-dense-timeseries "${input_dtseries}" \
      -left-metric "${lh_metric}" \
      -right-metric "${rh_metric}"
  fi
fi

if [ ! -s "${input_dtseries}" ] || [ ! -r "${input_dtseries}" ]; then
  echo "ERROR: runwise MATLAB input is missing or unreadable: ${input_dtseries}" >&2
  exit 4
fi

mkdir -p "${subject_out}" "${manifest_dir}"
n_timepoints="$("${wb_command}" -file-information "${input_dtseries}" -only-number-of-maps | tr -d '[:space:]')"
if ! [[ "${n_timepoints}" =~ ^[0-9]+$ ]] || [ "${n_timepoints}" -lt 1 ]; then
  echo "ERROR: invalid runwise timepoint count: ${n_timepoints}" >&2
  exit 4
fi

run_list="${manifest_dir}/run_list.txt"
censor_file="${manifest_dir}/censor.txt"
censor_list="${manifest_dir}/censor_list.txt"
input_manifest="${manifest_dir}/input_manifest.json"
printf '%s\n' "${input_dtseries}" > "${run_list}"
printf '%s\n' "${censor_file}" > "${censor_list}"
awk -v count="${n_timepoints}" 'BEGIN { for (i = 1; i <= count; i++) print 1 }' > "${censor_file}"

run_count="$(awk 'NF { count++ } END { print count + 0 }' "${run_list}")"
censor_ref_count="$(awk 'NF { count++ } END { print count + 0 }' "${censor_list}")"
censor_frame_count="$(awk 'NF { count++ } END { print count + 0 }' "${censor_file}")"
if [ "${run_count}" -ne 1 ] || [ "${censor_ref_count}" -ne 1 ] || [ "${censor_frame_count}" -ne "${n_timepoints}" ]; then
  echo "ERROR: runwise input contract mismatch: ${run_count} run entries, ${censor_ref_count} censor entries, ${censor_frame_count} censor frames (expected 1, 1, ${n_timepoints})" >&2
  exit 4
fi
cat > "${input_manifest}" <<EOF
{
  "dataset": "iowa7t",
  "variant": "$(json_escape "${variant}")",
  "subject": "$(json_escape "${subject}")",
  "run_mode": "independent",
  "run_label": "$(json_escape "${run_label}")",
  "run_index": ${run_index},
  "acquisition": "$(json_escape "${acquisition}")",
  "input_kind": "$(json_escape "${input_kind}")",
  "source_subject_dir": "$(json_escape "${source_subject_dir}")",
  "source_dtseries": "$(json_escape "${source_dtseries}")",
  "input_dtseries": "$(json_escape "${input_dtseries}")",
  "run_list": "$(json_escape "${run_list}")",
  "censor_file": "$(json_escape "${censor_file}")",
  "censor_list": "$(json_escape "${censor_list}")",
  "timepoints": ${n_timepoints},
  "output_dir": "$(json_escape "${subject_out}")"
}
EOF

echo "=================================================================="
echo "Iowa7T Kong2022 runwise task"
echo "JobID=${job_id} TaskID=${task_id} Host=${host_name} Subject=${subject} Run=${run_label} Acquisition=${acquisition}"
echo "Dataset=iowa7t Variant=${variant} InputKind=${input_kind}"
echo "DataRoot=${data_root}"
echo "Source=${source_dtseries}"
echo "Input=${input_dtseries}"
echo "Out=${subject_out}"
echo "Manifest=${input_manifest}"
echo "Timepoints=${n_timepoints}"
echo "RunID=${run_id}"
echo "TASK_DRY_RUN=${task_dry_run} OVERWRITE=${overwrite}"
echo "=================================================================="

if [ "${task_dry_run}" = "1" ]; then
  echo "DRY_RUN_OK: prepared and verified ${subject} ${run_label}; MATLAB not started"
  exit 0
fi

project_dir="/tmp/iowa7t_kong2022_runwise_${run_id}_${job_id}_${task_id}_${subject}_${run_label}"
mkdir -p "${project_dir}"
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

script_dir_m="$(matlab_quote "${script_dir}")"
project_dir_m="$(matlab_quote "${project_dir}")"
subject_out_m="$(matlab_quote "${subject_out}")"
run_list_m="$(matlab_quote "${run_list}")"
censor_list_m="$(matlab_quote "${censor_list}")"
cbig_code_dir_m="$(matlab_quote "${cbig_code_dir}")"
group_prior_m="$(matlab_quote "${group_prior}")"
wb_command_m="$(matlab_quote "${wb_command}")"
wb_dir_m="$(matlab_quote "$(dirname "${wb_command}")")"
matlab_expr="setenv('WB_COMMAND','${wb_command_m}'); setenv('PATH',['${wb_dir_m}:' getenv('PATH')]); addpath('${script_dir_m}'); run_kong2022_areal_mshbm_subject('${project_dir_m}', '${subject_out_m}', '${run_list_m}', '${censor_list_m}', '${cbig_code_dir_m}', '${group_prior_m}', ${overwrite_flag});"

case "${matlab_mode}" in
  batch)
    matlab -singleCompThread -batch "${matlab_expr}"
    ;;
  batch_nodisplay)
    matlab -singleCompThread -nodisplay -batch "${matlab_expr}"
    ;;
  r_nodesktop)
    run_expr="try, ${matlab_expr}, catch ME, disp(getReport(ME,'extended')); exit(1); end; exit(0);"
    matlab -singleCompThread -nodisplay -nodesktop -r "${run_expr}"
    ;;
  *)
    echo "ERROR: unsupported MATLAB_MODE=${matlab_mode}" >&2
    exit 5
    ;;
esac

if [ ! -s "${lh_labels}" ] || [ ! -s "${rh_labels}" ]; then
  echo "ERROR: MATLAB completed but runwise label outputs are missing under ${subject_out}" >&2
  exit 6
fi

run_end_utc="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
run_end_epoch="$(date -u +%s)"
runtime_seconds="$(awk -v start="${run_start_epoch}" -v end="${run_end_epoch}" 'BEGIN { printf "%.3f", end - start }')"
cat > "${provenance_path}" <<EOF
{
  "schema_version": 1,
  "dataset": "iowa7t",
  "variant": "$(json_escape "${variant}")",
  "subject": "$(json_escape "${subject}")",
  "run_mode": "independent",
  "run_label": "$(json_escape "${run_label}")",
  "run_index": ${run_index},
  "acquisition": "$(json_escape "${acquisition}")",
  "input_kind": "$(json_escape "${input_kind}")",
  "source_subject_dir": "$(json_escape "${source_subject_dir}")",
  "source_dtseries": "$(json_escape "${source_dtseries}")",
  "input_dtseries": "$(json_escape "${input_dtseries}")",
  "input_ciftis": ["$(json_escape "${input_dtseries}")"],
  "run_list": "$(json_escape "${run_list}")",
  "censor_file": "$(json_escape "${censor_file}")",
  "censor_list": "$(json_escape "${censor_list}")",
  "runs": [
    {
      "run_index": ${run_index},
      "run_label": "$(json_escape "${run_label}")",
      "acquisition": "$(json_escape "${acquisition}")",
      "source_dtseries": "$(json_escape "${source_dtseries}")",
      "dtseries": "$(json_escape "${input_dtseries}")",
      "censor": "$(json_escape "${censor_file}")",
      "timepoints": ${n_timepoints}
    }
  ],
  "labels": {
    "lh": "$(json_escape "${lh_labels}")",
    "rh": "$(json_escape "${rh_labels}")"
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
  "matlab": {
    "mode": "$(json_escape "${matlab_mode}")",
    "script": "$(json_escape "${script_dir}/run_kong2022_areal_mshbm_subject.m")"
  },
  "qsub": {
    "host": "$(json_escape "${host_name}")",
    "job_id": "$(json_escape "${job_id}")",
    "task_id": "$(json_escape "${task_id}")"
  },
  "timepoints": ${n_timepoints},
  "timing": {
    "run_end_utc": "$(json_escape "${run_end_utc}")",
    "run_start_utc": "$(json_escape "${run_start_utc}")",
    "runtime_seconds": ${runtime_seconds}
  },
  "run_id": "$(json_escape "${run_id}")"
}
EOF

echo "PROVENANCE: ${provenance_path}"
echo "SUCCESS: ${subject} ${run_label} label text outputs ready under ${subject_out}"
