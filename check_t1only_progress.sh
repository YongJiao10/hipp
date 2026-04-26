#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BATCH_ROOT_DEFAULT="${REPO_ROOT}/outputs_experiments/hippunfold_recommended_t1w_20260423/dense_corobl_batch"
BATCH_ROOT="${1:-$BATCH_ROOT_DEFAULT}"
BATCH_TAG="${BATCH_ROOT#${REPO_ROOT}/}"

if [[ ! -d "$BATCH_ROOT" ]]; then
  echo "ERROR: batch root not found: $BATCH_ROOT"
  exit 1
fi

now_utc() { date -u '+%Y-%m-%d %H:%M:%S UTC'; }

# Count finished subjects by final den-512 outputs.
TOTAL=$(find "$BATCH_ROOT" -maxdepth 1 -type d -name 'sub-*' | wc -l | tr -d ' ')
DONE=0
while IFS= read -r subdir; do
  sub="$(basename "$subdir")"
  base="$subdir/hippunfold/$sub"
  spec="$base/surf/${sub}_den-512_surfaces.spec"
  [[ -f "$spec" ]] || continue
  hipp=$(find "$base/surf" -type f -name "${sub}_hemi-*_space-corobl_den-512_label-hipp_midthickness.surf.gii" 2>/dev/null | wc -l | tr -d ' ')
  dent=$(find "$base/surf" -type f -name "${sub}_hemi-*_space-corobl_den-512_label-dentate_midthickness.surf.gii" 2>/dev/null | wc -l | tr -d ' ')
  if [[ "$hipp" -ge 2 && "$dent" -ge 2 ]]; then
    DONE=$((DONE + 1))
  fi
done < <(find "$BATCH_ROOT" -maxdepth 1 -type d -name 'sub-*' | sort)

# Running subjects in THIS batch only.
RUN_SUBS="$(pgrep -af "run_hippunfold_local.sh" | awk -v br="$BATCH_ROOT" -v tag="$BATCH_TAG" '
  index($0, br) > 0 || index($0, tag) > 0 {
    for (i = 1; i <= NF; i++) {
      if ($i ~ /^[0-9]{6}$/) {
        printf "sub-%s\n", $i;
        break;
      }
    }
  }
' | sort -u)"
RUNNING=$(printf "%s\n" "$RUN_SUBS" | sed '/^$/d' | wc -l | tr -d ' ')
INCOMPLETE=$((TOTAL - DONE - RUNNING))

# Manager/worker processes for this batch.
MGR_LINES="$(pgrep -af "run_remaining_hippunfold_background.sh" | awk -v br="$BATCH_ROOT" -v tag="$BATCH_TAG" 'index($0, br) > 0 || index($0, tag) > 0 {print}')"

# Latest manager log.
LATEST_MANAGER_LOG="$(ls -1t "$BATCH_ROOT"/_batch_logs/manager_remaining_*_conc*.log 2>/dev/null | head -n 1 || true)"

echo "TIME: $(now_utc)"
echo "BATCH_ROOT: $BATCH_ROOT"
echo "SUMMARY: total=$TOTAL done=$DONE running=$RUNNING incomplete=$INCOMPLETE"

echo "RUNNING_SUBJECTS:"
if [[ -n "${RUN_SUBS// /}" ]]; then
  printf "%s\n" "$RUN_SUBS"
else
  echo "(none)"
fi

echo "MANAGER_WORKERS:"
if [[ -n "${MGR_LINES// /}" ]]; then
  printf "%s\n" "$MGR_LINES"
else
  echo "(none)"
fi

if [[ -n "$LATEST_MANAGER_LOG" ]]; then
  echo "LATEST_MANAGER_LOG: $LATEST_MANAGER_LOG"
  echo "LAST_LOG_LINES:"
  tail -n 12 "$LATEST_MANAGER_LOG" || true
else
  echo "LATEST_MANAGER_LOG: (none)"
fi
