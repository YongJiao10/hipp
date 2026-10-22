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
export OMP_NUM_THREADS=1

# ==============================================================================
# Paths (Argon Environment Contract: use /Shared/jianglab prefix)
# ==============================================================================
baseDir=/Shared/jianglab/3_Data_Working/Depression/NIH_rTMS
inFmriprepDir=${baseDir}/derivatives/restfmri
outXcpdDir=${baseDir}/xcpd/restfmri
logDir=${baseDir}/src/logs
sublist=${baseDir}/src/sublist.txt

singularityDir=/Shared/jianglab/0_scripts/fmriprep_tools
sifFile=${singularityDir}/xcp_d_latest.sif
fsLicense=${singularityDir}/license.txt

mkdir -p "${logDir}" "${outXcpdDir}"

# ==============================================================================
# Subject selection (Supports single argument OR SGE_TASK_ID array)
# ==============================================================================
if [[ $# -ge 1 && -n "${1:-}" ]]; then
    subject="$1"
    echo "Running in SINGLE-SUBJECT manual mode for: ${subject}"
elif [[ -n "${SGE_TASK_ID:-}" && "${SGE_TASK_ID}" != "undefined" ]]; then
    if [[ ! -f "${sublist}" ]]; then
        echo "ERROR: sublist file not found: ${sublist}"
        exit 1
    fi
    subject=$(sed -n "${SGE_TASK_ID}p" "${sublist}" | tr -d '[:space:]')
    echo "Running in SGE ARRAY mode: Task ID ${SGE_TASK_ID} -> ${subject}"
else
    echo "ERROR: No subject provided as argument and SGE_TASK_ID is not set."
    echo "Usage (Single subject): qsub $0 sub-D01"
    echo "Usage (Array job)     : qsub -t 1-44 -tc 20 $0"
    exit 1
fi

subject=${subject#sub-}

echo "=============================================================================="
echo "Job Name   : xcpd_36P_nih_rtms"
echo "Subject    : sub-${subject}"
echo "BaseDir    : ${baseDir}"
echo "FmriprepIn : ${inFmriprepDir}"
echo "XcpdOut    : ${outXcpdDir}"
echo "SIF        : ${sifFile}"
echo "=============================================================================="

# ==============================================================================
# Check input files
# ==============================================================================
subFmriprepDir="${inFmriprepDir}/sub-${subject}"

if [[ ! -d "${subFmriprepDir}" ]]; then
    echo "ERROR: fMRIPrep derivative folder not found for subject: ${subFmriprepDir}"
    echo "Please ensure fMRIPrep has finished successfully for sub-${subject} before running XCP-D."
    exit 1
fi

# Check if subject has functional BOLD preprocessed data (e.g. sub-D16 is anatomical-only)
has_func=$(find "${subFmriprepDir}" -type f -name "*_desc-preproc_bold.nii.gz" 2>/dev/null | head -n 1 || true)
if [[ -z "${has_func}" ]]; then
    echo "NOTICE: sub-${subject} has no functional preprocessed BOLD data (anatomical-only). Skipping XCP-D cleanly."
    exit 0
fi

if [[ ! -f "${sifFile}" ]]; then
    echo "ERROR: XCP-D SIF file not found: ${sifFile}"
    exit 1
fi

if [[ ! -f "${fsLicense}" ]]; then
    echo "ERROR: FreeSurfer license not found: ${fsLicense}"
    exit 1
fi

# ==============================================================================
# Run XCP-D (Singularity)
# ==============================================================================
singularity run --cleanenv \
    -B "${baseDir}":/data \
    "${sifFile}" \
    /data/derivatives/restfmri /data/xcpd/restfmri participant \
    --participant-label "${subject}" \
    --nthreads 5 \
    --omp-nthreads 5 \
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
    --clean-workdir \
    --fs-license-file "${fsLicense}" \
    --resource-monitor

# ==============================================================================
# Permissions
# ==============================================================================
chmod -R 777 "${outXcpdDir}/sub-${subject}" 2>/dev/null || true

echo "Done XCP-D: sub-${subject}"
