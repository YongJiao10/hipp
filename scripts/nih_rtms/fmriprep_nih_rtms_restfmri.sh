#!/bin/bash

#$ -N fmriprep_nih_rtms
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

export TEMPLATEFLOW_HOME=${singularityDir}/TemplateFlow
export SINGULARITYENV_TEMPLATEFLOW_HOME=/templateflow

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
# Clean hidden macOS / dot files if any
# ==============================================================================
find "${subRawDir}" \
    -type f \( -name "._*" -o -name ".DS_Store" \) \
    -print -delete || true

# ==============================================================================
# Run fMRIPrep (Singularity)
# ==============================================================================
singularity run --cleanenv \
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
    --mem-mb 24000 \
    --ignore fieldmaps slicetiming \
    --output-spaces MNI152NLin6Asym:res-2 \
    --force-bbr \
    --fs-license-file /license.txt \
    --output-layout bids \
    --work-dir /work \
    --resource-monitor \
    --notrack \
    --stop-on-first-crash \
    --clean-workdir \
    --fs-no-reconall \
    --task-id rest

# ==============================================================================
# Permissions
# ==============================================================================
chmod -R 777 "${outDir}/sub-${subject}" 2>/dev/null || true
chmod 777 "${outDir}/sub-${subject}.html" 2>/dev/null || true
chmod -R 777 "${outDir}/sub-${subject}/log" 2>/dev/null || true
chmod -R 777 "${workDir}" 2>/dev/null || true

echo "Done: sub-${subject}"
