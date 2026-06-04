#!/bin/bash
set -euo pipefail
export LANG=C
export LC_ALL=C
export LC_CTYPE=C

manifest_default="/old_Users/yojiao/hipp/outputs_experiments/hippunfold_recommended_t1w_20260423/kong2022_route_b/manifests/hcp_7t_hippocampus_fmri_qc_pass_t1only_151.txt"
manifest="${1:-$manifest_default}"

if [ ! -f "$manifest" ]; then
  echo "ERROR: manifest not found: $manifest" >&2
  exit 2
fi

repo_root="/old_Users/yojiao/hipp"
job_script="$repo_root/scripts/core_workflow/sge_argon/matlab_kong2022_mshbm_subject_sge.sh"
if [ ! -f "$job_script" ]; then
  echo "ERROR: missing job script: $job_script" >&2
  exit 2
fi

if ! command -v qsub >/dev/null 2>&1; then
  echo "ERROR: qsub command not found in PATH" >&2
  exit 127
fi

# Read IDs: keep only 6-digit numeric IDs.
mapfile -t all_ids < <(awk '{gsub(/^[ \t]+|[ \t]+$/, ""); if ($0 ~ /^[0-9]{6}$/) print $0}' "$manifest")

if [ "${#all_ids[@]}" -lt 1 ]; then
  echo "ERROR: no valid subject IDs found in manifest: $manifest" >&2
  exit 2
fi

# Duplicate check inside manifest.
dup_ids="$(printf '%s\n' "${all_ids[@]}" | sort | uniq -d || true)"
if [ -n "$dup_ids" ]; then
  echo "ERROR: duplicate subject IDs in manifest:" >&2
  echo "$dup_ids" >&2
  exit 2
fi

ui_ids=()
pinc_ids=()

# Deterministic partition by index ratio 1:2 (UI:PINC):
# index 0,3,6,... -> UI ; others -> PINC
for i in "${!all_ids[@]}"; do
  sid="${all_ids[$i]}"
  mod=$(( i % 3 ))
  if [ "$mod" -eq 0 ]; then
    ui_ids+=("$sid")
  else
    pinc_ids+=("$sid")
  fi
done

if [ "${#ui_ids[@]}" -lt 1 ] || [ "${#pinc_ids[@]}" -lt 1 ]; then
  echo "ERROR: partition produced empty queue list (ui=${#ui_ids[@]} pinc=${#pinc_ids[@]})" >&2
  exit 2
fi

# Cross-check no overlap.
intersection="$(comm -12 <(printf '%s\n' "${ui_ids[@]}" | sort) <(printf '%s\n' "${pinc_ids[@]}" | sort) || true)"
if [ -n "$intersection" ]; then
  echo "ERROR: UI/PINC split has overlap:" >&2
  echo "$intersection" >&2
  exit 2
fi

ui_n="${#ui_ids[@]}"
pinc_n="${#pinc_ids[@]}"
timestamp="$(date +%Y%m%d_%H%M%S)"
subject_list_dir="${repo_root}/tmp/subject_lists"
mkdir -p "${subject_list_dir}"
ui_list_file="${subject_list_dir}/k2a_ui_${timestamp}.txt"
pinc_list_file="${subject_list_dir}/k2a_pinc_${timestamp}.txt"
printf '%s\n' "${ui_ids[@]}" > "${ui_list_file}"
printf '%s\n' "${pinc_ids[@]}" > "${pinc_list_file}"

ui_name="k2a_ui_prod"
pinc_name="k2a_pinc_prod"

echo "[dual-submit] manifest=$manifest"
echo "[dual-submit] total=${#all_ids[@]} UI=$ui_n PINC=$pinc_n"
echo "[dual-submit] UI first5=$(printf '%s ' "${ui_ids[@]:0:5}")"
echo "[dual-submit] PINC first5=$(printf '%s ' "${pinc_ids[@]:0:5}")"
echo "[dual-submit] UI list file=${ui_list_file}"
echo "[dual-submit] PINC list file=${pinc_list_file}"

qsub -N "$ui_name" \
  -q UI \
  -t "1-${ui_n}" \
  -tc 10 \
  -v "REPO_ROOT=${repo_root},MATLAB_MODE=batch,SUBJECT_IDS_FILE=${ui_list_file}" \
  "$job_script"

qsub -N "$pinc_name" \
  -q PINC \
  -t "1-${pinc_n}" \
  -tc 5 \
  -v "REPO_ROOT=${repo_root},MATLAB_MODE=batch,SUBJECT_IDS_FILE=${pinc_list_file}" \
  "$job_script"
