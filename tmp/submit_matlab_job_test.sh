#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "${script_dir}/.." && pwd)"
job_script="${script_dir}/matlab_job_test_sge.sh"

if [ ! -f "${job_script}" ]; then
    echo "ERROR: job script not found: ${job_script}"
    exit 1
fi

if ! command -v qsub >/dev/null 2>&1; then
    echo "ERROR: qsub command not found in PATH"
    exit 127
fi

# Avoid invalid locale warnings that can be treated as JSV stderr failures.
export LANG=C
export LC_ALL=C

echo "Submitting MATLAB smoke test (2 jobs): ${job_script}"
cd "${repo_root}"
qsub "${job_script}"
