#!/bin/bash

#$ -N kong2022_mshbm
#$ -pe smp 3
#$ -l mem_free=16G
#$ -l h_rt=06:00:00
#$ -j y
#$ -o /Users/yojiao/logs/$JOB_NAME.$JOB_ID.out

set -euo pipefail

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

# Single-job mode:
# - default runs subject index 1 when qsub is used without -t.
# - submit subject 2 with:
#   qsub -v SUBJECT_INDEX=2 scripts/core_workflow/sge_argon/matlab_kong2022_mshbm_subject_sge.sh
raw_task_id="${SUBJECT_INDEX:-${SGE_TASK_ID:-1}}"
if [ -z "${raw_task_id}" ] || [ "${raw_task_id}" = "undefined" ]; then
    task_id=1
else
    task_id="${raw_task_id}"
fi
job_id="${JOB_ID:-local}"
host_name="$(hostname)"

repo_root="${REPO_ROOT:-${SGE_O_WORKDIR:-$(pwd)}}"
subject_ids_file="${SUBJECT_IDS_FILE:-}"
subject_ids_csv="${SUBJECT_IDS_CSV:-100610,102311}"
subject_ids=()

if [ -n "${subject_ids_file}" ]; then
    if [ ! -f "${subject_ids_file}" ]; then
        echo "ERROR: SUBJECT_IDS_FILE not found: ${subject_ids_file}"
        exit 2
    fi
    while IFS= read -r line; do
        sid="$(echo "${line}" | sed 's/^[[:space:]]*//;s/[[:space:]]*$//')"
        if [[ "${sid}" =~ ^[0-9]{6}$ ]]; then
            subject_ids+=("${sid}")
        fi
    done < "${subject_ids_file}"
else
    IFS=',' read -r -a subject_ids <<< "${subject_ids_csv}"
fi

if [ "${#subject_ids[@]}" -lt 1 ]; then
    echo "ERROR: empty subject list. SUBJECT_IDS_FILE=${subject_ids_file} SUBJECT_IDS_CSV=${subject_ids_csv}"
    exit 2
fi

if ! [[ "${task_id}" =~ ^[0-9]+$ ]] || [ "${task_id}" -lt 1 ] || [ "${task_id}" -gt "${#subject_ids[@]}" ]; then
    echo "ERROR: invalid subject task index=${task_id}; subject_count=${#subject_ids[@]}"
    exit 2
fi

subject="${subject_ids[$((task_id - 1))]}"
out_root="${repo_root}/outputs_core/inputs/hcp7t_hippocampus/kong2022/cortex_references/routea_areal_mshbm"
subject_dir="${out_root}/sub-${subject}"
runs_dir="${subject_dir}/temp_runs"
project_dir="${subject_dir}/temp_project"
route_out_dir="${project_dir}/route_a_outputs"
run_list_file="${project_dir}/input_run_dtseries.txt"
censor_dir="${subject_dir}/temp_censor_inputs"

data_root="${repo_root}/data/hippunfold_input/sub-${subject}"
dtseries="${data_root}/func/sub-${subject}_task-rest_run-concat.dtseries.nii"
group_prior="${repo_root}/external/CBIG/stable_projects/brain_parcellation/Kong2022_ArealMSHBM/lib/group_priors/HCP_fs_LR_32k_40sub/400/gMSHBM/beta50/Params_Final.mat"
cbig_code_dir="${repo_root}/external/CBIG"
mapalign_embed="${cbig_code_dir}/external_packages/python/mapalign-master/mapalign/embed.py"
cbig_corr="${cbig_code_dir}/utilities/matlab/stats/CBIG_corr.m"
connected_components="${cbig_code_dir}/utilities/matlab/parcellation/CBIG_ComputeConnectedComponentsFromSurface.m"
downsample_900_dlabel="${cbig_code_dir}/data/templates/surface/fs_LR_32k_downsample_900/fslr_downsample_900mesh_parcellation.dlabel.nii"
wb_command="${repo_root}/software/workbench/bin_linux64/wb_command"
matlab_wrapper_log="${subject_dir}/matlab_wrapper.log"
export PATH="${repo_root}/software/workbench/bin_linux64:${PATH}"

mkdir -p "${runs_dir}" "${project_dir}" "${route_out_dir}"
mkdir -p "${censor_dir}"

if [ -f "${subject_dir}/lh_labels.txt" ] && [ -f "${subject_dir}/rh_labels.txt" ]; then
    echo "SKIP: sub-${subject} MATLAB label outputs already exist:"
    echo "  ${subject_dir}/lh_labels.txt"
    echo "  ${subject_dir}/rh_labels.txt"
    exit 0
fi

echo "=================================================================="
echo "Kong2022 ArealMSHBM MATLAB-only"
echo "JobID=${job_id} TaskID=${task_id} Host=${host_name} Subject=${subject}"
echo "Repo=${repo_root}"
echo "Output=${subject_dir}"
echo "=================================================================="

for required in "${dtseries}" "${group_prior}" "${mapalign_embed}" "${cbig_corr}" "${connected_components}" "${downsample_900_dlabel}" "${wb_command}" "${cbig_code_dir}/stable_projects/brain_parcellation/Kong2022_ArealMSHBM/CBIG_ArealMSHBM_parcellation_single_subject.m"; do
    if [ ! -e "${required}" ]; then
        echo "ERROR: missing required file: ${required}"
        exit 3
    fi
done

# Split concat dtseries into 4 runs of 900 timepoints each.
"${wb_command}" -cifti-merge "${runs_dir}/sub-${subject}_task-rest_run-1.dtseries.nii" -direction ROW -cifti "${dtseries}" -index 1 -up-to 900
"${wb_command}" -cifti-merge "${runs_dir}/sub-${subject}_task-rest_run-2.dtseries.nii" -direction ROW -cifti "${dtseries}" -index 901 -up-to 1800
"${wb_command}" -cifti-merge "${runs_dir}/sub-${subject}_task-rest_run-3.dtseries.nii" -direction ROW -cifti "${dtseries}" -index 1801 -up-to 2700
"${wb_command}" -cifti-merge "${runs_dir}/sub-${subject}_task-rest_run-4.dtseries.nii" -direction ROW -cifti "${dtseries}" -index 2701 -up-to 3600

cat > "${run_list_file}" <<EOF
${runs_dir}/sub-${subject}_task-rest_run-1.dtseries.nii
${runs_dir}/sub-${subject}_task-rest_run-2.dtseries.nii
${runs_dir}/sub-${subject}_task-rest_run-3.dtseries.nii
${runs_dir}/sub-${subject}_task-rest_run-4.dtseries.nii
EOF

for rid in 1 2 3 4; do
    censor_file="${censor_dir}/sub-${subject}_task-rest_run-${rid}_censor.txt"
    awk 'BEGIN{for(i=1;i<=900;i++) print 1}' > "${censor_file}"
done

if ! command -v matlab >/dev/null 2>&1; then
    if command -v module >/dev/null 2>&1; then
        module load matlab/R2021a >/dev/null 2>&1 || true
    fi
fi

if ! command -v matlab >/dev/null 2>&1; then
    echo "ERROR: matlab not found in PATH"
    exit 127
fi

run1="${runs_dir}/sub-${subject}_task-rest_run-1.dtseries.nii"
run2="${runs_dir}/sub-${subject}_task-rest_run-2.dtseries.nii"
run3="${runs_dir}/sub-${subject}_task-rest_run-3.dtseries.nii"
run4="${runs_dir}/sub-${subject}_task-rest_run-4.dtseries.nii"
censor1="${censor_dir}/sub-${subject}_task-rest_run-1_censor.txt"
censor2="${censor_dir}/sub-${subject}_task-rest_run-2_censor.txt"
censor3="${censor_dir}/sub-${subject}_task-rest_run-3_censor.txt"
censor4="${censor_dir}/sub-${subject}_task-rest_run-4_censor.txt"

batch_expr="setenv('CBIG_CODE_DIR','${cbig_code_dir}');addpath(fullfile('${cbig_code_dir}','stable_projects','brain_parcellation','Kong2022_ArealMSHBM'));addpath(genpath(fullfile('${cbig_code_dir}','utilities')));addpath(genpath(fullfile('${cbig_code_dir}','external_packages')));params=struct();params.project_dir='${project_dir}';params.target_mesh='fs_LR_32k';params.censor_list={'${censor1}','${censor2}','${censor3}','${censor4}'};params.lh_fMRI_list={'${run1}','${run2}','${run3}','${run4}'};params.model='gMSHBM';params.num_ROIs='400';params.w='30';params.c='30';params.beta='50';params.group_prior='${group_prior}';params.overwrite_flag=0;[lh_labels,rh_labels]=CBIG_ArealMSHBM_parcellation_single_subject(params);out_dir=fullfile('${project_dir}','route_a_outputs');if ~exist(out_dir,'dir');mkdir(out_dir);end;writematrix(round(lh_labels(:)),fullfile(out_dir,'lh_labels.txt'),'FileType','text');writematrix(round(rh_labels(:)),fullfile(out_dir,'rh_labels.txt'),'FileType','text');"
matlab_mode="${MATLAB_MODE:-batch}"
run_expr="try, ${batch_expr}, catch ME, disp(getReport(ME,'extended')); exit(1); end; exit(0);"

echo "MATLAB_MODE=${matlab_mode}" | tee "${matlab_wrapper_log}"
case "${matlab_mode}" in
    batch)
        matlab -batch "${batch_expr}" >> "${matlab_wrapper_log}" 2>&1
        ;;
    batch_nodisplay)
        matlab -nodisplay -batch "${batch_expr}" >> "${matlab_wrapper_log}" 2>&1
        ;;
    r_nodesktop)
        matlab -nodisplay -nodesktop -r "${run_expr}" >> "${matlab_wrapper_log}" 2>&1
        ;;
    r_nodesktop_nojvm)
        matlab -nojvm -nodisplay -nodesktop -r "${run_expr}" >> "${matlab_wrapper_log}" 2>&1
        ;;
    *)
        echo "ERROR: unsupported MATLAB_MODE=${matlab_mode}" | tee -a "${matlab_wrapper_log}"
        exit 5
        ;;
esac

if [ ! -f "${route_out_dir}/lh_labels.txt" ] || [ ! -f "${route_out_dir}/rh_labels.txt" ]; then
    echo "ERROR: MATLAB wrapper completed but lh/rh label files are missing."
    exit 4
fi

# Move outputs to standard flat path directly under subject_dir
mv "${route_out_dir}/lh_labels.txt" "${subject_dir}/lh_labels.txt"
mv "${route_out_dir}/rh_labels.txt" "${subject_dir}/rh_labels.txt"
keep_intermediates="${KEEP_INTERMEDIATES:-0}"
if [ "${keep_intermediates}" != "1" ]; then
    rm -rf "${runs_dir}" "${censor_dir}" "${project_dir}"
    echo "CLEANUP: removed intermediate directories (KEEP_INTERMEDIATES=${keep_intermediates})"
else
    echo "CLEANUP: skipped intermediate cleanup (KEEP_INTERMEDIATES=1)"
fi

echo "SUCCESS: sub-${subject} MATLAB outputs ready in ${subject_dir}"
