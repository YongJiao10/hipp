#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import nibabel as nib
import numpy as np


def zscore_cols(x: np.ndarray) -> np.ndarray:
    x = x - x.mean(axis=0, keepdims=True)
    std = x.std(axis=0, keepdims=True)
    return x / np.clip(std, 1e-12, None)


def main() -> int:
    parser = argparse.ArgumentParser(description="Compute seed-based whole-brain FC from voxel labels")
    parser.add_argument("--bold", required=True)
    parser.add_argument("--brain-mask", required=True)
    parser.add_argument("--seed-labels", required=True)
    parser.add_argument("--outdir", required=True)
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    bold_img = nib.load(args.bold)
    bold = np.asarray(bold_img.dataobj, dtype=np.float32)
    brain_mask = nib.load(args.brain_mask).get_fdata() > 0
    seed_labels = nib.load(args.seed_labels).get_fdata().astype(np.int16)

    vox_ts = bold[brain_mask]
    vox_z = zscore_cols(vox_ts.T)
    labels = sorted(int(x) for x in np.unique(seed_labels) if x > 0)
    summary = []
    for label in labels:
        seed_mask = (seed_labels == label)
        if not np.any(seed_mask):
            continue
        seed_ts = bold[seed_mask].mean(axis=0)
        seed_z = (seed_ts - seed_ts.mean()) / max(seed_ts.std(), 1e-12)
        fc = vox_z.T @ seed_z / seed_z.shape[0]
        fc_map = np.zeros(brain_mask.shape, dtype=np.float32)
        fc_map[brain_mask] = fc
        out_path = outdir / f"seed-{label:02d}_fc.nii.gz"
        nib.save(nib.Nifti1Image(fc_map, bold_img.affine, bold_img.header), out_path)
        summary.append({"label": label, "n_seed_voxels": int(seed_mask.sum()), "output": str(out_path)})

    (outdir / "seed_fc_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
