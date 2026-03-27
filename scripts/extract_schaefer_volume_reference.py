#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path

import nibabel as nib
import numpy as np
from nilearn.image import resample_to_img


NETWORK_ORDER = ["Vis", "SomMot", "DorsAttn", "SalVentAttn", "Limbic", "Cont", "Default"]


def load_label_map(dlabel_path: Path) -> dict[int, str]:
    img = nib.load(str(dlabel_path))
    axis = img.header.get_axis(0)
    label_table = axis.label[0]
    out = {}
    for idx, (name, _rgba) in label_table.items():
        if idx == 0:
            continue
        out[int(idx)] = name
    return out


def schaefer_network_from_name(label_name: str) -> str:
    m = re.match(r"7Networks_[LR]H_([^_]+)", label_name)
    if not m:
        raise ValueError(f"Cannot parse network from label name: {label_name}")
    return m.group(1)


def corrcoef_rows(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a = a - a.mean(axis=1, keepdims=True)
    b = b - b.mean(axis=1, keepdims=True)
    a_std = np.linalg.norm(a, axis=1, keepdims=True)
    b_std = np.linalg.norm(b, axis=1, keepdims=True)
    return (a @ b.T) / np.clip(a_std * b_std.T, 1e-12, None)


def main() -> int:
    parser = argparse.ArgumentParser(description="Extract Schaefer400 parcel and 7-network reference timeseries from volume BOLD")
    parser.add_argument("--bold", required=True)
    parser.add_argument("--brain-mask", required=True)
    parser.add_argument("--atlas-volume", required=True)
    parser.add_argument("--atlas-dlabel", required=True)
    parser.add_argument("--outdir", required=True)
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    bold_img = nib.load(args.bold)
    bold = np.asarray(bold_img.dataobj, dtype=np.float32)
    mask = nib.load(args.brain_mask).get_fdata() > 0

    atlas_resampled = resample_to_img(
        nib.load(args.atlas_volume),
        bold_img,
        interpolation="nearest",
        force_resample=True,
    )
    atlas = atlas_resampled.get_fdata().astype(np.int32)
    nib.save(atlas_resampled, outdir / "Schaefer400_2mm_in_bold_space.nii.gz")

    label_map = load_label_map(Path(args.atlas_dlabel))
    parcel_ids = sorted(int(x) for x in np.unique(atlas) if x > 0)
    parcel_ts = []
    parcel_rows: list[dict[str, object]] = []
    for pid in parcel_ids:
        vox = (atlas == pid) & mask
        if not np.any(vox):
            continue
        ts = np.nanmean(bold[vox], axis=0)
        name = label_map.get(pid, f"parcel_{pid}")
        net = schaefer_network_from_name(name)
        parcel_ts.append(ts)
        parcel_rows.append(
            {
                "parcel_id": pid,
                "parcel_name": name,
                "network": net,
                "n_voxels": int(vox.sum()),
            }
        )

    if not parcel_ts:
        raise RuntimeError("No Schaefer parcels overlapped the BOLD brain mask after resampling")

    parcel_ts_arr = np.asarray(parcel_ts, dtype=np.float32)
    np.save(outdir / "schaefer400_parcel_timeseries.npy", parcel_ts_arr)

    with (outdir / "schaefer400_parcels.tsv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["parcel_id", "parcel_name", "network", "n_voxels"], delimiter="\t")
        writer.writeheader()
        writer.writerows(parcel_rows)

    network_ts = []
    network_rows = []
    for net in NETWORK_ORDER:
        inds = [i for i, row in enumerate(parcel_rows) if row["network"] == net]
        if not inds:
            continue
        ts = np.nanmean(parcel_ts_arr[inds], axis=0)
        network_ts.append(ts)
        network_rows.append({"network": net, "n_parcels": len(inds)})
    network_ts_arr = np.asarray(network_ts, dtype=np.float32)
    np.save(outdir / "schaefer7_network_timeseries.npy", network_ts_arr)
    (outdir / "schaefer7_networks.json").write_text(json.dumps(network_rows, indent=2), encoding="utf-8")

    print(json.dumps({"n_parcels": len(parcel_rows), "n_networks": len(network_rows)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
