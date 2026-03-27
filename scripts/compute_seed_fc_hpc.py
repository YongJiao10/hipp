#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import warnings
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import nibabel as nib
import numpy as np

warnings.filterwarnings("ignore", r"Mean of empty slice")


def parse_label_names(spec: str | None) -> dict[int, str]:
    if spec is None:
        return {}
    path = Path(spec)
    if path.exists():
        if path.suffix.lower() == ".json":
            data = json.loads(path.read_text(encoding="utf-8"))
            return {int(k): str(v) for k, v in data.items()}
        mapping: dict[int, str] = {}
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split(None, 1)
            if len(parts) != 2:
                raise ValueError(f"Invalid label mapping line: {line}")
            mapping[int(parts[0])] = parts[1].strip()
        return mapping
    mapping = {}
    for item in spec.split(","):
        key, value = item.split(":", 1)
        mapping[int(key)] = value
    return mapping


def sanitize_name(name: str) -> str:
    safe = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in name.strip())
    while "__" in safe:
        safe = safe.replace("__", "_")
    return safe.strip("_") or "label"


def zscore_series(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float64, copy=False)
    mean = np.nanmean(x)
    std = np.nanstd(x)
    if not np.isfinite(std) or std < 1e-12:
        return np.zeros_like(x, dtype=np.float32)
    return ((x - mean) / std).astype(np.float32)


def main() -> int:
    parser = argparse.ArgumentParser(description="Chunked seed-based FC for HPC using a discrete label volume")
    parser.add_argument("--bold", required=True, help="4D BOLD NIfTI(.gz)")
    parser.add_argument("--brain-mask", required=True, help="3D brain mask in BOLD space")
    parser.add_argument("--seed-labels", required=True, help="3D discrete label image in BOLD space")
    parser.add_argument("--outdir", required=True)
    parser.add_argument("--chunk-size", type=int, default=100)
    parser.add_argument("--max-nan-ratio", type=float, default=0.10)
    parser.add_argument("--label-names", default=None, help="Optional JSON/TSV or inline '1:Vis,2:SomMot'")
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    bold_img = nib.load(args.bold)
    mask_img = nib.load(args.brain_mask)
    labels_img = nib.load(args.seed_labels)

    if bold_img.shape[:3] != mask_img.shape[:3] or bold_img.shape[:3] != labels_img.shape[:3]:
        raise ValueError(
            f"Shape mismatch: bold={bold_img.shape[:3]}, mask={mask_img.shape[:3]}, labels={labels_img.shape[:3]}"
        )
    if not np.allclose(bold_img.affine, mask_img.affine, atol=1e-4):
        raise ValueError("BOLD/mask affine mismatch")
    if not np.allclose(bold_img.affine, labels_img.affine, atol=1e-4):
        raise ValueError("BOLD/seed-label affine mismatch")

    mask_data = np.asanyarray(mask_img.dataobj).squeeze() > 0
    label_data = np.rint(np.asanyarray(labels_img.dataobj).squeeze()).astype(np.int16)
    label_ids = sorted(int(x) for x in np.unique(label_data) if x > 0)
    if not label_ids:
        raise ValueError("No positive labels found in seed-label image")

    label_names = parse_label_names(args.label_names)
    seed_masks = []
    for label in label_ids:
        seed_mask = label_data == label
        if not np.any(seed_mask):
            continue
        outside = int(np.count_nonzero(seed_mask & ~mask_data))
        if outside:
            raise ValueError(f"Label {label} contains {outside} voxels outside the brain mask")
        seed_masks.append(seed_mask)

    n_frames = int(bold_img.shape[-1])
    n_vox = int(np.count_nonzero(mask_data))
    n_seeds = len(seed_masks)
    dataobj = bold_img.dataobj

    sum_x = np.zeros(n_vox, dtype=np.float64)
    count_x = np.zeros(n_vox, dtype=np.int32)
    seed_ts = np.zeros((n_frames, n_seeds), dtype=np.float32)

    for start_t in range(0, n_frames, args.chunk_size):
        end_t = min(start_t + args.chunk_size, n_frames)
        chunk_data = np.asanyarray(dataobj[..., start_t:end_t])
        for i, t in enumerate(range(start_t, end_t)):
            vol = np.asarray(chunk_data[..., i], dtype=np.float32)
            v_brain = vol[mask_data]
            valid_mask = ~np.isnan(v_brain)
            sum_x[valid_mask] += v_brain[valid_mask]
            count_x[valid_mask] += 1
            for s_idx, seed_mask in enumerate(seed_masks):
                seed_vals = vol[seed_mask]
                seed_ts[t, s_idx] = np.nanmean(seed_vals) if seed_vals.size > 0 else np.nan

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        voxel_means = (sum_x / count_x).astype(np.float32)

    bad_voxel_mask = count_x < (n_frames * (1.0 - args.max_nan_ratio))
    voxel_means[bad_voxel_mask] = np.nan

    valid_frames = ~np.any(np.isnan(seed_ts), axis=1)
    n_valid_frames = int(np.count_nonzero(valid_frames))
    if n_valid_frames < 10:
        raise ValueError(f"Too few valid frames after censoring: {n_valid_frames}")

    valid_seed_ts = seed_ts[valid_frames, :]
    seed_means = np.mean(valid_seed_ts, axis=0)
    sum_x2 = np.zeros(n_vox, dtype=np.float64)
    sum_y2 = np.zeros(n_seeds, dtype=np.float64)
    sum_xy = np.zeros((n_vox, n_seeds), dtype=np.float64)

    for start_t in range(0, n_frames, args.chunk_size):
        end_t = min(start_t + args.chunk_size, n_frames)
        cur_len = end_t - start_t
        chunk_full = np.zeros((cur_len, n_vox), dtype=np.float32)
        chunk_seeds = np.zeros((cur_len, n_seeds), dtype=np.float32)
        frame_mask = np.zeros(cur_len, dtype=bool)
        chunk_data = np.asanyarray(dataobj[..., start_t:end_t])

        for i, t in enumerate(range(start_t, end_t)):
            if not valid_frames[t]:
                continue
            frame_mask[i] = True
            vol = np.asarray(chunk_data[..., i], dtype=np.float32)
            v_brain = vol[mask_data]
            nan_mask = np.isnan(v_brain)
            v_brain[nan_mask] = voxel_means[nan_mask]
            chunk_full[i, :] = v_brain - voxel_means
            chunk_seeds[i, :] = seed_ts[t, :] - seed_means

        valid_chunk_full = chunk_full[frame_mask]
        valid_chunk_seeds = chunk_seeds[frame_mask]
        if valid_chunk_full.shape[0] == 0:
            continue
        chunk_full64 = valid_chunk_full.astype(np.float64, copy=False)
        chunk_seeds64 = valid_chunk_seeds.astype(np.float64, copy=False)
        sum_x2 += np.sum(chunk_full64**2, axis=0)
        sum_y2 += np.sum(chunk_seeds64**2, axis=0)
        sum_xy += chunk_full64.T @ chunk_seeds64

    denom_std = max(1, n_valid_frames - 1)
    std_x = np.sqrt(np.maximum(sum_x2 / denom_std, 1e-12)).astype(np.float32)
    std_y = np.sqrt(np.maximum(sum_y2 / denom_std, 1e-12)).astype(np.float32)

    out_header = bold_img.header.copy()
    out_header.set_slope_inter(1.0, 0.0)
    out_header.set_data_dtype(np.float32)

    summary: list[dict[str, object]] = []
    for j, label in enumerate(label_ids):
        cov_xy = sum_xy[:, j] / denom_std
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            r = cov_xy / (std_x * std_y[j])
        r = np.clip(r, -0.999999, 0.999999)
        z = np.arctanh(r).astype(np.float32)
        z[bad_voxel_mask] = np.nan

        out_vol = np.full(mask_img.shape, np.nan, dtype=np.float32)
        out_vol[~mask_data] = 0.0
        out_vol[mask_data] = z

        label_name = label_names.get(label, f"label-{label:02d}")
        out_name = f"seed-{label:02d}_{sanitize_name(label_name)}_fc_z.nii.gz"
        out_path = outdir / out_name
        nib.save(nib.Nifti1Image(out_vol, bold_img.affine, out_header), out_path)
        summary.append(
            {
                "label": label,
                "label_name": label_name,
                "n_seed_voxels": int(np.count_nonzero(label_data == label)),
                "valid_frames": n_valid_frames,
                "output": str(out_path),
            }
        )

    (outdir / "seed_fc_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
