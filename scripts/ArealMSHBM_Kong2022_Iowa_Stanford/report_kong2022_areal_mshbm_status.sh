#!/bin/bash
set -euo pipefail
export LANG=C
export LC_ALL=C
export LC_CTYPE=C

dataset="$(printf '%s' "${DATASET:-jj_tms}" | tr '[:upper:]-' '[:lower:]_')"
case "${dataset}" in
  jj_tms)
    out_root="${OUT_ROOT:-/Shared/jianglab/3_Data_Working/fmriprep_processed_JJ_TMS/ArealMSHBM_Kong2022}"
    ;;
  stanford)
    out_root="${OUT_ROOT:-/Shared/jianglab/3_Data_Working/fmriprep_processed_CausalConnectome/rest_fmriprep/ArealMSHBM_Kong2022}"
    ;;
  iowa7t)
    out_root="${OUT_ROOT:-/Shared/jianglab/3_Data_Working/fmriprep_processed_Iowa7T/ArealMSHBM_Kong2022}"
    ;;
  iowa_hcp500)
    out_root="${OUT_ROOT:-/old_Users/yojiao/hipp/outputs_iowa_hcp500/inputs/iowa_hcp500/kong2022/cortex_references/routea_areal_mshbm}"
    ;;
  *)
    echo "ERROR: unsupported DATASET=${DATASET:-}. Expected jj_tms, stanford, iowa7t, or iowa_hcp500." >&2
    exit 2
    ;;
esac
job_id="${1:-}"
subject="${2:-sub-1006}"
subject="sub-${subject#sub-}"
subject_out="${out_root}/${subject}"

if [ -z "${job_id}" ]; then
  echo "Usage: $(basename "$0") <job_id> [sub-1006]" >&2
  exit 2
fi

echo "Job: ${job_id}"
echo "Dataset: ${dataset}"
echo "Subject: ${subject}"
echo "Output: ${subject_out}"
echo

echo "qstat:"
qstat -j "${job_id}" 2>/dev/null || qstat -u yojiao 2>/dev/null || true
echo

echo "qacct:"
qacct -j "${job_id}" 2>/dev/null || true
echo

echo "logs:"
shopt -s nullglob
logs=("${out_root}"/_sge_logs/*/*."${job_id}".*.log)
printf '%s\n' "${logs[@]}" | sort
echo

echo "expected label text outputs:"
for path in \
  "${subject_out}/lh_labels.txt" \
  "${subject_out}/rh_labels.txt"; do
  if [ -e "${path}" ]; then
    echo "OK      ${path}"
  else
    echo "MISSING ${path}"
  fi
done
