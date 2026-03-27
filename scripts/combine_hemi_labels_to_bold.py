#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import nibabel as nib
import numpy as np
from nilearn.image import resample_to_img


def main() -> int:
    parser = argparse.ArgumentParser(description="Merge left/right hippocampal label volumes and resample to BOLD grid")
    parser.add_argument("--left-labels", required=True)
    parser.add_argument("--right-labels", required=True)
    parser.add_argument("--bold-ref", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    left_img = nib.load(args.left_labels)
    right_img = nib.load(args.right_labels)
    left = left_img.get_fdata().astype(np.int16)
    right = right_img.get_fdata().astype(np.int16)

    overlap = (left > 0) & (right > 0)
    if np.any(overlap):
        raise RuntimeError(f"Left/right hippocampal labels overlap in {int(overlap.sum())} voxels")

    merged = left.copy()
    merged[right > 0] = right[right > 0]
    merged_img = nib.Nifti1Image(merged.astype(np.int16), left_img.affine, left_img.header)

    bold_ref = nib.load(args.bold_ref)
    merged_bold = resample_to_img(
        merged_img,
        bold_ref,
        interpolation="nearest",
        force_resample=True,
    )
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    nib.save(merged_bold, out_path)

    summary = {
        "left_voxels": int((left > 0).sum()),
        "right_voxels": int((right > 0).sum()),
        "merged_t1w_voxels": int((merged > 0).sum()),
        "merged_bold_voxels": int((merged_bold.get_fdata() > 0).sum()),
        "output": str(out_path),
    }
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
