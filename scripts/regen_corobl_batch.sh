#!/bin/zsh
# regen_corobl_batch.sh
#
# Regenerates space-corobl_den-512 midthickness surfaces for the 3 HCP 7T subjects.
#
# WHY THIS IS NEEDED:
#   hippunfold marks space-corobl surfaces as snakemake temp() outputs.
#   After the initial run they are auto-deleted.  run_hippunfold_local.sh now
#   passes --notemp so future runs keep them, but existing outputs were already
#   cleaned.  This script forces a re-run of the specific snakemake rule that
#   produces corobl surfaces and keeps them.
#
# WHAT IT PRODUCES (per subject, per hemi in {L,R}, per label in {hipp,dentate}):
#   outputs_migration/dense_corobl_batch/sub-{SUB}/hippunfold/sub-{SUB}/surf/
#     sub-{SUB}_hemi-{H}_space-corobl_den-512_label-{label}_midthickness.surf.gii
#
# REQUIREMENTS:
#   - conda env hippo2  (hippunfold 2.0.0 from khanlab channel)
#   - /Applications/ITK-SNAP.app/  (provides greedy + c3d_affine_tool)
#   - Template cache at .cache/hippunfold/hippunfold_cache/  (already present;
#     set HIPPUNFOLD_CACHE_ROOT to that dir to avoid re-downloading ~500 MB)
#
# USAGE (from any directory):
#   bash /path/to/repo/scripts/regen_corobl_batch.sh
#
set -euo pipefail

BUNDLE_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
INPUT_DIR="${BUNDLE_ROOT}/data/hippunfold_input"
BATCH_DIR="${BUNDLE_ROOT}/outputs_migration/dense_corobl_batch"

# Point to the existing template cache so hippunfold does not re-download.
export HIPPUNFOLD_CACHE_ROOT="${BUNDLE_ROOT}/.cache/hippunfold"

# --forcerun: corobl surf files are gone (deleted as temp()); without this flag
#   snakemake sees downstream unfold surfaces as up-to-date and skips recreation.
# --notemp: keep corobl surfaces after the run (also hardcoded in run_hippunfold_local.sh).
export HIPPUNFOLD_EXTRA_ARGS="--forcerun resample_native_surf_to_atlas_density --notemp"

for SUB in 100610 102311 102816; do
  echo ""
  echo "══════════════════════════════════════════════════════"
  echo "  Regenerating corobl surfaces: sub-${SUB}"
  echo "══════════════════════════════════════════════════════"
  "${BUNDLE_ROOT}/scripts/run_hippunfold_local.sh" \
    "${SUB}" \
    "${INPUT_DIR}" \
    "${BATCH_DIR}/sub-${SUB}/hippunfold"
done

echo ""
echo "Done.  Corobl surfaces present:"
find "${BATCH_DIR}" -name "*space-corobl*midthickness*.surf.gii" | sort
