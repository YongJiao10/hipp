#!/bin/bash

#$ -N xcpd_36P_nih_rtms
#$ -pe smp 5
#$ -q UI,PINC
#$ -j y
#$ -o /Shared/jianglab/3_Data_Working/Depression/NIH_rTMS/src/logs
#$ -t 1-44
#$ -tc 20

set -euo pipefail

# ==============================================================================
# Resource settings
# ==============================================================================
threads=5
hawk_id=yojiao
export OMP_NUM_THREADS=1

# ==============================================================================
# Paths (Argon Environment Contract: use /Shared/jianglab prefix)
# ==============================================================================
baseDir=/Shared/jianglab/3_Data_Working/Depression/NIH_rTMS
rawDir=${baseDir}/bids_input
inFmriprepDir=${baseDir}/derivatives/restfmri
outXcpdSurfDir=${baseDir}/xcpd_surf/restfmri
outXcpdVolDir=${baseDir}/xcpd/restfmri
logDir=${baseDir}/src/logs
sublist=${baseDir}/src/sublist.txt

singularityDir=/Shared/jianglab/0_scripts/fmriprep_tools
sifFile=${singularityDir}/xcp_d_latest.sif
fsLicense=${singularityDir}/license.txt
runtimeDir=/old_Users/yojiao/hipp/scripts/nih_rtms/runtime

export TEMPLATEFLOW_HOME=${singularityDir}/TemplateFlow
export SINGULARITYENV_TEMPLATEFLOW_HOME=/templateflow

mkdir -p "${logDir}" "${outXcpdSurfDir}" "${outXcpdVolDir}"

# ==============================================================================
# Subject selection (Supports single argument OR SGE_TASK_ID array)
# ==============================================================================
if [[ $# -ge 1 && -n "${1:-}" ]]; then
    if [[ -n "${SGE_TASK_ID:-}" && "${SGE_TASK_ID}" != "undefined" ]]; then
        if [[ "${SGE_TASK_FIRST:-1}" != "1" || "${SGE_TASK_LAST:-1}" != "1" || "${SGE_TASK_ID}" != "1" ]]; then
            echo "ERROR: Single-subject argument '$1' cannot be combined with a multi-task SGE array (SGE_TASK_ID=${SGE_TASK_ID}, range=${SGE_TASK_FIRST:-?}-${SGE_TASK_LAST:-?})."
            echo "Submit single-subject jobs with: qsub -t 1-1 -tc 1 $0 $1"
            exit 1
        fi
    fi
    subject="$1"
    echo "Running in SINGLE-SUBJECT manual mode for: ${subject}"
elif [[ -n "${SGE_TASK_ID:-}" && "${SGE_TASK_ID}" != "undefined" ]]; then
    if [[ ! -f "${sublist}" ]]; then
        echo "ERROR: sublist file not found: ${sublist}"
        exit 1
    fi
    subject=$(sed -n "${SGE_TASK_ID}p" "${sublist}" | tr -d '[:space:]')
    if [[ -z "${subject}" ]]; then
        echo "ERROR: SGE_TASK_ID ${SGE_TASK_ID} resolved to an empty subject in ${sublist}"
        exit 1
    fi
    echo "Running in SGE ARRAY mode: Task ID ${SGE_TASK_ID} -> ${subject}"
else
    echo "ERROR: No subject provided as argument and SGE_TASK_ID is not set."
    echo "Usage (Single subject): qsub -t 1-1 -tc 1 $0 sub-D01"
    echo "Usage (Array job)     : qsub -t 1-44 -tc 20 $0"
    exit 1
fi

subject=${subject#sub-}

# Per-subject isolated scratch directory on Argon compute node (prevents array collisions)
workDir=/nfsscratch/Users/${hawk_id}/xcpd_NIH_rTMS_work/sub-${subject}
mkdir -p "${workDir}/cifti" "${workDir}/nifti"

echo "=============================================================================="
echo "Job Name       : xcpd_36P_nih_rtms"
echo "Subject        : sub-${subject}"
echo "BaseDir        : ${baseDir}"
echo "RawDir         : ${rawDir}"
echo "FmriprepIn     : ${inFmriprepDir}"
echo "XcpdSurfOut    : ${outXcpdSurfDir}"
echo "XcpdVolOut     : ${outXcpdVolDir}"
echo "WorkDir        : ${workDir}"
echo "Threads        : ${threads}"
echo "SIF            : ${sifFile}"
echo "=============================================================================="

# ==============================================================================
# Check input files against authoritative raw BIDS runs (fail-closed)
# ==============================================================================
subRawDir="${rawDir}/sub-${subject}"
subFmriprepDir="${inFmriprepDir}/sub-${subject}"

if [[ ! -d "${subRawDir}" ]]; then
    echo "ERROR: subject raw BIDS folder not found: ${subRawDir}"
    exit 1
fi

if [[ ! -d "${subFmriprepDir}" ]]; then
    echo "ERROR: fMRIPrep derivative folder not found for subject: ${subFmriprepDir}"
    echo "Please ensure fMRIPrep has finished successfully for sub-${subject} before running XCP-D."
    exit 1
fi

raw_bold_listing=$(find "${subRawDir}" -type f -name "sub-${subject}_*_task-rest_bold.nii.gz" | sort)
raw_bold_files=()
if [[ -n "${raw_bold_listing}" ]]; then
    mapfile -t raw_bold_files <<< "${raw_bold_listing}"
fi
if [[ ${#raw_bold_files[@]} -eq 0 ]]; then
    if [[ "${subject}" == "D16" ]]; then
        echo "NOTICE: sub-${subject} is the known anatomical-only subject (0 raw task-rest runs). Skipping XCP-D cleanly."
        exit 0
    fi
    echo "ERROR: No raw task-rest BOLD runs found for sub-${subject} in ${subRawDir} (only sub-D16 is anatomical-only)."
    exit 1
fi

echo "Verifying fMRIPrep volume + CIFTI derivatives for ${#raw_bold_files[@]} expected run(s) of sub-${subject}..."
for raw_bold in "${raw_bold_files[@]}"; do
    rel_path="${raw_bold#"${subRawDir}/"}"
    run_dir=$(dirname "${rel_path}")
    run_base=$(basename "${raw_bold}" _bold.nii.gz)
    expected_vol="${subFmriprepDir}/${run_dir}/${run_base}_space-MNI152NLin6Asym_res-2_desc-preproc_bold.nii.gz"
    expected_cifti="${subFmriprepDir}/${run_dir}/${run_base}_space-fsLR_den-91k_bold.dtseries.nii"
    if [[ ! -f "${expected_vol}" ]]; then
        echo "ERROR: Missing fMRIPrep volumetric derivative for sub-${subject}: ${expected_vol}"
        exit 1
    fi
    if [[ ! -f "${expected_cifti}" ]]; then
        echo "ERROR: Missing fMRIPrep fsLR den-91k CIFTI derivative for sub-${subject}: ${expected_cifti}"
        exit 1
    fi
done

if [[ ! -f "${sifFile}" ]]; then
    echo "ERROR: XCP-D SIF file not found: ${sifFile}"
    exit 1
fi

if [[ ! -f "${fsLicense}" ]]; then
    echo "ERROR: FreeSurfer license not found: ${fsLicense}"
    exit 1
fi

if [[ ! -d "${TEMPLATEFLOW_HOME}" ]]; then
    echo "ERROR: TemplateFlow folder not found: ${TEMPLATEFLOW_HOME}"
    exit 1
fi

if [[ ! -s "${runtimeDir}/sitecustomize.py" ]]; then
    echo "ERROR: Missing required NIH rTMS atomic QC runtime: ${runtimeDir}/sitecustomize.py"
    exit 1
fi
sha256sum "${runtimeDir}/sitecustomize.py"
singularity exec --cleanenv \
    --env PYTHONPATH=/nih_rtms_runtime \
    -B "${runtimeDir}":/nih_rtms_runtime:ro \
    "${sifFile}" /usr/local/miniconda/bin/python -c \
    'from nipype.utils.filemanip import copyfile; assert getattr(copyfile, "_nih_rtms_atomic_qc_copy", False), "Atomic QC runtime not loaded"; print("NIH rTMS atomic QC runtime validated:", copyfile._nih_rtms_source_sha256)'

# ==============================================================================
# Step 1: Run XCP-D CIFTI (fsLR den-91k cortical surface + subcortical dtseries)
# ==============================================================================
echo "Starting XCP-D CIFTI pass (fsLR den-91k) for sub-${subject}..."
singularity run --cleanenv \
    --env PYTHONPATH=/nih_rtms_runtime \
    -B "${runtimeDir}":/nih_rtms_runtime:ro \
    -B "${baseDir}":/data \
    -B "${TEMPLATEFLOW_HOME}":/templateflow \
    -B "${fsLicense}":/license.txt \
    -B "${workDir}":/work \
    "${sifFile}" \
    /data/derivatives/restfmri /data/xcpd_surf/restfmri participant \
    --participant-label "${subject}" \
    --task-id rest \
    --nthreads "${threads}" \
    --omp-nthreads "${threads}" \
    --mem-gb 20 \
    --input-type fmriprep \
    --file-format cifti \
    --mode linc \
    --despike y \
    -p 36P \
    --fd-thresh 0.5 \
    --min-time 0 \
    --lower-bpf 0.01 \
    --upper-bpf 0.10 \
    --smoothing 6 \
    --output-type censored \
    --atlases Glasser Tian \
    --work-dir /work/cifti \
    --bids-database-dir /work/cifti/bids_db \
    --fs-license-file /license.txt \
    --resource-monitor 2>&1 | tee "${workDir}/cifti_execution.log"

if grep -Eq 'NodeExecutionError|Workflow did not execute cleanly|Node .* failed to run' "${workDir}/cifti_execution.log"; then
    echo "ERROR: XCP-D CIFTI pass reported a node failure: ${workDir}/cifti_execution.log"
    exit 1
fi

# ==============================================================================
# Step 2: Run XCP-D NIfTI (MNI152NLin6Asym:res-2 volumetric BOLD for HippUnfold)
# ==============================================================================
echo "Starting XCP-D NIfTI pass (MNI152NLin6Asym:res-2) for sub-${subject}..."
singularity run --cleanenv \
    --env PYTHONPATH=/nih_rtms_runtime \
    -B "${runtimeDir}":/nih_rtms_runtime:ro \
    -B "${baseDir}":/data \
    -B "${TEMPLATEFLOW_HOME}":/templateflow \
    -B "${fsLicense}":/license.txt \
    -B "${workDir}":/work \
    "${sifFile}" \
    /data/derivatives/restfmri /data/xcpd/restfmri participant \
    --participant-label "${subject}" \
    --task-id rest \
    --nthreads "${threads}" \
    --omp-nthreads "${threads}" \
    --mem-gb 20 \
    --input-type fmriprep \
    --file-format nifti \
    --mode linc \
    --despike y \
    -p 36P \
    --fd-thresh 0.5 \
    --min-time 0 \
    --lower-bpf 0.01 \
    --upper-bpf 0.10 \
    --smoothing 6 \
    --output-type censored \
    --atlases Glasser Tian \
    --work-dir /work/nifti \
    --bids-database-dir /work/nifti/bids_db \
    --fs-license-file /license.txt \
    --resource-monitor 2>&1 | tee "${workDir}/nifti_execution.log"

if grep -Eq 'NodeExecutionError|Workflow did not execute cleanly|Node .* failed to run' "${workDir}/nifti_execution.log"; then
    echo "ERROR: XCP-D NIfTI pass reported a node failure: ${workDir}/nifti_execution.log"
    exit 1
fi

# ==============================================================================
# Verify expected XCP-D outputs for every run before setting permissions
# ==============================================================================
for raw_bold in "${raw_bold_files[@]}"; do
    rel_path="${raw_bold#"${subRawDir}/"}"
    run_dir=$(dirname "${rel_path}")
    run_base=$(basename "${raw_bold}" _bold.nii.gz)
    expected_xcpd_cifti="${outXcpdSurfDir}/sub-${subject}/${run_dir}/${run_base}_space-fsLR_den-91k_desc-denoised_bold.dtseries.nii"
    expected_xcpd_vol="${outXcpdVolDir}/sub-${subject}/${run_dir}/${run_base}_space-MNI152NLin6Asym_res-2_desc-denoised_bold.nii.gz"
    if [[ ! -f "${expected_xcpd_cifti}" ]]; then
        echo "ERROR: Missing expected XCP-D CIFTI output: ${expected_xcpd_cifti}"
        exit 1
    fi
    if [[ ! -f "${expected_xcpd_vol}" ]]; then
        echo "ERROR: Missing expected XCP-D volumetric output: ${expected_xcpd_vol}"
        exit 1
    fi
done

# ==============================================================================
# Permissions (Group-shared Research-jianglab, no world-writable 777)
# ==============================================================================
chmod -R ug+rwX,o-w "${outXcpdSurfDir}/sub-${subject}"
if [[ -f "${outXcpdSurfDir}/sub-${subject}.html" ]]; then
    chmod ug+rw,o-w "${outXcpdSurfDir}/sub-${subject}.html"
fi
chmod -R ug+rwX,o-w "${outXcpdVolDir}/sub-${subject}"
if [[ -f "${outXcpdVolDir}/sub-${subject}.html" ]]; then
    chmod ug+rw,o-w "${outXcpdVolDir}/sub-${subject}.html"
fi
if [[ -d "${workDir}" ]]; then
    chmod -R ug+rwX,o-w "${workDir}"
fi

echo "Done XCP-D (CIFTI + NIfTI): sub-${subject}"
