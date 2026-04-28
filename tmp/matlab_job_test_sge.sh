#!/bin/bash

#$ -N matlab_test
#$ -pe smp 1
#$ -l mem_free=8G
#$ -l h_rt=00:20:00
#$ -j y
#$ -q UI
#$ -o /Users/yojiao/logs/$JOB_NAME.$JOB_ID.$TASK_ID.out
#$ -t 1-2
#$ -tc 2

# -------------------------------------------------------------------------
# 0. Shell safety
# -------------------------------------------------------------------------
set -euo pipefail

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

# -------------------------------------------------------------------------
# 1. Runtime context
# -------------------------------------------------------------------------
task_id="${SGE_TASK_ID:-0}"
job_id="${JOB_ID:-local}"
host_name="$(hostname)"

work_root="${SGE_O_WORKDIR:-$(pwd)}"
test_root="${work_root}/tmp/matlab_job_test"
result_dir="${test_root}/results"
mkdir -p "${result_dir}"

run_tag="job_${job_id}_task_${task_id}"
job_log="${result_dir}/${run_tag}.log"
ok_flag="${result_dir}/${run_tag}.ok"

echo "==================================================================" | tee "${job_log}"
echo "MATLAB submit test (2 jobs per submission)" | tee -a "${job_log}"
echo "JobID: ${job_id}  TaskID: ${task_id}  Host: ${host_name}" | tee -a "${job_log}"
echo "Output dir: ${result_dir}" | tee -a "${job_log}"
echo "==================================================================" | tee -a "${job_log}"

# -------------------------------------------------------------------------
# 2. MATLAB smoke test
# -------------------------------------------------------------------------
if ! command -v matlab >/dev/null 2>&1; then
    if command -v module >/dev/null 2>&1; then
        module load matlab/R2021a >/dev/null 2>&1 || true
    fi
fi

if ! command -v matlab >/dev/null 2>&1; then
    echo "ERROR: matlab command not found in PATH (module load matlab/R2021a also failed)" | tee -a "${job_log}"
    exit 127
fi

export MATLAB_TEST_TASK_ID="${task_id}"
export MATLAB_TEST_JOB_ID="${job_id}"
export MATLAB_TEST_HOST="${host_name}"

matlab -batch "fprintf('MATLAB_OK job=%s task=%s host=%s time=%s\n', getenv('MATLAB_TEST_JOB_ID'), getenv('MATLAB_TEST_TASK_ID'), getenv('MATLAB_TEST_HOST'), char(datetime('now')));" >> "${job_log}" 2>&1

date +"%F %T %Z" > "${ok_flag}"
echo "SUCCESS: MATLAB ran successfully (${run_tag})" | tee -a "${job_log}"
