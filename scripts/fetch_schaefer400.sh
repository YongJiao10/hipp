#!/bin/zsh
set -euo pipefail

OUTDIR="${1:-data/atlas/schaefer400}"
mkdir -p "$OUTDIR"

BASE="https://raw.githubusercontent.com/ThomasYeoLab/CBIG/master/stable_projects/brain_parcellation/Schaefer2018_LocalGlobal/Parcellations"

curl -L \
  -o "$OUTDIR/Schaefer2018_400Parcels_7Networks_order.dlabel.nii" \
  "$BASE/HCP/fslr32k/cifti/Schaefer2018_400Parcels_7Networks_order.dlabel.nii"

curl -L \
  -o "$OUTDIR/Schaefer2018_400Parcels_7Networks_order_FSLMNI152_2mm.nii.gz" \
  "$BASE/MNI/Schaefer2018_400Parcels_7Networks_order_FSLMNI152_2mm.nii.gz"

echo "Downloaded Schaefer400 7-network atlas to $OUTDIR"
