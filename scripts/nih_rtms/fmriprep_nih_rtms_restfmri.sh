#!/bin/bash

#$ -N fmriprep_nih_rtms
#$ -pe smp 8
#$ -q UI,PINC
#$ -j y
#$ -o /Shared/jianglab/3_Data_Working/Depression/NIH_rTMS/src/logs
#$ -t 1-44
#$ -tc 20

set -euo pipefail

# ==============================================================================
# Resource settings
# ==============================================================================
threads=8
hawk_id=yojiao
export OMP_NUM_THREADS=${threads}

# ==============================================================================
# Paths (Argon Environment Contract: use /Shared/jianglab prefix)
# ==============================================================================
baseDir=/Shared/jianglab/3_Data_Working/Depression/NIH_rTMS
rawDir=${baseDir}/bids_input
outDir=${baseDir}/derivatives/restfmri
logDir=${baseDir}/src/logs
sublist=${baseDir}/src/sublist.txt

singularityDir=/Shared/jianglab/0_scripts/fmriprep_tools
sifFile=${singularityDir}/fmriprep_v2314_kw.sif
fsLicense=${singularityDir}/license.txt
runtimeDir=/old_Users/yojiao/hipp/scripts/nih_rtms/fmriprep_overlay

export TEMPLATEFLOW_HOME=${singularityDir}/TemplateFlow
export SINGULARITYENV_TEMPLATEFLOW_HOME=/templateflow

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
fsSubjectsDir=${outDir}/sourcedata/freesurfer_by_subject/sub-${subject}

# Scratch directory on Argon compute node
workDir=/nfsscratch/Users/${hawk_id}/fmriprep_NIH_rTMS_work/sub-${subject}

mkdir -p "${logDir}" "${outDir}" "${workDir}"

echo "=============================================================================="
echo "Job Name   : fmriprep_nih_rtms"
echo "Subject    : sub-${subject}"
echo "BaseDir    : ${baseDir}"
echo "RawDir     : ${rawDir}"
echo "OutDir     : ${outDir}"
echo "WorkDir    : ${workDir}"
echo "Threads    : ${threads}"
echo "SIF        : ${sifFile}"
echo "=============================================================================="

# ==============================================================================
# Check input files
# ==============================================================================
subRawDir="${rawDir}/sub-${subject}"

if [[ ! -d "${subRawDir}" ]]; then
    echo "ERROR: subject raw folder not found: ${subRawDir}"
    exit 1
fi

if [[ ! -f "${sifFile}" ]]; then
    echo "ERROR: SIF file not found: ${sifFile}"
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

# ==============================================================================
# Fail-closed check for unwanted macOS / hidden dot files (no auto-deletion)
# ==============================================================================
dot_files=$(find "${subRawDir}" -type f \( -name "._*" -o -name ".DS_Store" \) -print)
if [[ -n "${dot_files}" ]]; then
    echo "ERROR: Hidden macOS metadata files found in ${subRawDir}:"
    echo "${dot_files}"
    echo "Refusing to auto-delete files in raw BIDS input. Clean approved paths first."
    exit 1
fi

# ==============================================================================
# Check whether subject has functional BOLD data in raw BIDS input
# (Only sub-D16 is expected to be anatomical-only in this cohort)
# ==============================================================================
raw_bold_listing=$(find "${subRawDir}" -type f -name "sub-${subject}_*_task-rest_bold.nii.gz" | sort)
raw_bold_files=()
if [[ -n "${raw_bold_listing}" ]]; then
    mapfile -t raw_bold_files <<< "${raw_bold_listing}"
fi
mode_args=()
if [[ ${#raw_bold_files[@]} -eq 0 ]]; then
    if [[ "${subject}" != "D16" ]]; then
        echo "ERROR: Unexpected missing raw task-rest BOLD files for sub-${subject} in ${subRawDir} (only sub-D16 is anatomical-only)."
        exit 1
    fi
    echo "NOTICE: sub-${subject} is the known anatomical-only subject in ${subRawDir}. Running fMRIPrep with --anat-only."
    mode_args=(--anat-only)
else
    echo "Found ${#raw_bold_files[@]} raw task-rest BOLD run(s) for sub-${subject}:"
    printf '  %s\n' "${raw_bold_files[@]}"
    mode_args=(
        --task-id rest
        --force-bbr
        --cifti-output 91k
        --fd-spike-threshold 0.5
        --dvars-spike-threshold 1.5
    )
fi

# ==============================================================================
# Run fMRIPrep (Singularity)
# Outputs BOTH volumetric (MNI152NLin6Asym:res-2) and cortical surface/CIFTI
# (fsLR den-91k) with FreeSurfer recon-all, slice timing correction (TR=3.0s),
# and longitudinal unbiased T1w template construction.
# ==============================================================================
if [[ ! -s "${runtimeDir}/sitecustomize.py" ]]; then
    echo "ERROR: Missing required exact FreeSurfer table resolver: ${runtimeDir}/sitecustomize.py"
    exit 1
fi
sha256sum "${runtimeDir}/sitecustomize.py"
singularity exec --cleanenv \
    --env PYTHONPATH=/nih_rtms_fmriprep_runtime \
    -B "${runtimeDir}":/nih_rtms_fmriprep_runtime:ro \
    -B "${TEMPLATEFLOW_HOME}":/templateflow:ro \
    "${sifFile}" /opt/conda/envs/fmriprep/bin/python -c \
    'from templateflow import api; assert getattr(api.get, "_nih_rtms_fsaverage_table", None), "Exact FreeSurfer resolver not loaded"; print("NIH rTMS exact FreeSurfer table validated:", api.get._nih_rtms_fsaverage_table, api.get._nih_rtms_table_sha256)'

singularity run --cleanenv \
    --env PYTHONPATH=/nih_rtms_fmriprep_runtime \
    -B "${runtimeDir}":/nih_rtms_fmriprep_runtime:ro \
    -B "${baseDir}":/data \
    -B "${TEMPLATEFLOW_HOME}":/templateflow \
    -B "${fsLicense}":/license.txt \
    -B "${workDir}":/work \
    "${sifFile}" \
    /data/bids_input /data/derivatives/restfmri participant \
    --participant-label "${subject}" \
    --skip-bids-validation \
    --nprocs "${threads}" \
    --omp-nthreads "${threads}" \
    --mem-mb 32000 \
    --ignore fieldmaps \
    --longitudinal \
    --output-spaces MNI152NLin6Asym:res-2 fsLR \
    --fs-subjects-dir "/data/derivatives/restfmri/sourcedata/freesurfer_by_subject/sub-${subject}" \
    --fs-license-file /license.txt \
    --output-layout bids \
    --work-dir /work \
    --resource-monitor \
    --notrack \
    --stop-on-first-crash \
    --clean-workdir \
    "${mode_args[@]}"

# ==============================================================================
# Verify expected fMRIPrep outputs before setting group permissions
# ==============================================================================
if [[ ! -d "${outDir}/sub-${subject}" || ! -f "${outDir}/sub-${subject}.html" ]]; then
    echo "ERROR: fMRIPrep did not produce expected subject output or HTML report for sub-${subject} in ${outDir}."
    exit 1
fi

for raw_bold in "${raw_bold_files[@]}"; do
    rel_path="${raw_bold#"${subRawDir}/"}"
    run_dir=$(dirname "${rel_path}")
    run_base=$(basename "${raw_bold}" _bold.nii.gz)
    expected_vol="${outDir}/sub-${subject}/${run_dir}/${run_base}_space-MNI152NLin6Asym_res-2_desc-preproc_bold.nii.gz"
    expected_cifti="${outDir}/sub-${subject}/${run_dir}/${run_base}_space-fsLR_den-91k_bold.dtseries.nii"
    if [[ ! -f "${expected_vol}" ]]; then
        echo "ERROR: Missing expected fMRIPrep volumetric output: ${expected_vol}"
        exit 1
    fi
    if [[ ! -f "${expected_cifti}" ]]; then
        echo "ERROR: Missing expected fMRIPrep CIFTI output: ${expected_cifti}"
        exit 1
    fi
done

# ==============================================================================
# Permissions (Group-shared Research-jianglab, no world-writable 777)
# ==============================================================================
chmod -R ug+rwX,o-w "${outDir}/sub-${subject}"
chmod ug+rw,o-w "${outDir}/sub-${subject}.html"
if [[ -d "${fsSubjectsDir}/sub-${subject}" ]]; then
    chmod -R ug+rwX,o-w "${fsSubjectsDir}"
fi
if [[ -d "${workDir}" ]]; then
    chmod -R ug+rwX,o-w "${workDir}"
fi

echo "Done: sub-${subject}"
