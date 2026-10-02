#!/usr/bin/env python3
"""Export 2 mm vertex-level template-matching labels to LSS template_matched_profiles."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import sys

import numpy as np

REPO_ROOT = Path("/home/yojiao/hpchome/hipp")
SRC_ROOT = REPO_ROOT / "outputs_stanford" / "work" / "hcp151_fixed_template_matching" / "subjects"
DST_ROOT = Path("/home/yojiao/LSS/jianglab/3_Data_Working/fmriprep_processed_CausalConnectome/template_matched_profiles")

HEMIS = ("L", "R")
EXPECTED_SHAPE = (512,)
EXPECTED_LABELS = {0, 1, 2, 3}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    if not SRC_ROOT.is_dir():
        print(f"Error: Source directory missing: {SRC_ROOT}", file=sys.stderr)
        return 1
    if not DST_ROOT.is_dir():
        print(f"Error: Destination directory missing: {DST_ROOT}", file=sys.stderr)
        return 1

    subject_dirs = sorted([d for d in DST_ROOT.iterdir() if d.is_dir() and d.name.startswith("sub-")])
    if not subject_dirs:
        print("Error: No subject directories found in destination", file=sys.stderr)
        return 1

    print(f"Found {len(subject_dirs)} subjects to process.")

    copied_count = 0
    records = []

    for s_dir in subject_dirs:
        sub_id = s_dir.name  # e.g., "sub-1001"
        for hemi in HEMIS:
            src_file = SRC_ROOT / sub_id / f"matching/2mm/hemi-{hemi}/labels.npy"
            if not src_file.is_file():
                print(f"Error: Missing source file {src_file}", file=sys.stderr)
                return 1

            arr = np.load(src_file, allow_pickle=False)
            if arr.shape != EXPECTED_SHAPE:
                print(f"Error: Unexpected shape {arr.shape} in {src_file}", file=sys.stderr)
                return 1
            unique_labels = set(np.unique(arr))
            if not unique_labels.issubset(EXPECTED_LABELS):
                print(f"Error: Unexpected labels {unique_labels} in {src_file}", file=sys.stderr)
                return 1

            dst_file = s_dir / f"{sub_id}_2mm_hemi-{hemi}_labels.npy"
            temp_file = s_dir / f".{dst_file.name}.tmp"

            with open(temp_file, "wb") as f:
                np.save(f, arr, allow_pickle=False)

            src_hash = sha256(src_file)
            tmp_hash = sha256(temp_file)
            if src_hash != tmp_hash:
                temp_file.unlink(missing_ok=True)
                print(f"Error: Hash mismatch for {src_file} -> {temp_file}", file=sys.stderr)
                return 1

            os.replace(temp_file, dst_file)
            copied_count += 1
            records.append({
                "subject": sub_id,
                "hemi": hemi,
                "path": str(dst_file),
                "sha256": src_hash,
            })

    print(f"Successfully exported {copied_count} label files ({copied_count // 2} subjects x 2 hemispheres).")

    # Update destination README.md
    readme_path = DST_ROOT / "README.md"
    readme_content = """# Stanford template-matched cluster FC and vertex-level labels

Each subject directory contains the two hemispheres from the Stanford 2 mm HCP151 template-matching result.

## Files per subject (`sub-<id>/`):

1. `sub-<id>_2mm_hemi-<L|R>_labels.npy`:
   - Shape: `(512,)`, dtype: `int32`
   - Vertex-level subregion assignment for each of the 512 HippUnfold `den-512` vertices:
     - `0`: Unassigned / Invalid mask (excluded due to low tSNR / non-finite signal)
     - `1`: Anterior / VAN
     - `2`: Body / SMN
     - `3`: Posterior / DMN

2. `sub-<id>_2mm_hemi-<L|R>_cluster_mean_timeseries_fc_pearson.npy` and `*_fisherz.npy`:
   - Shape: `(3, 400)`, dtype: `float32`
   - Rows: 3 subject-specific template-match clusters (1=Anterior/VAN, 2=Body/SMN, 3=Posterior/DMN)
   - Columns: Kong2022 cortical ROI IDs 1 through 400 in the subject registry order

3. `sub-<id>_2mm_cluster_mean_timeseries_fc_manifest.json`:
   - Provenance manifest for cluster-mean timeseries calculation.
"""
    readme_path.write_text(readme_content, encoding="utf-8")
    print(f"Updated {readme_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
