#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path

import nibabel as nib
import numpy as np


NETWORK_ORDER = ["Vis", "SomMot", "DorsAttn", "SalVentAttn", "Limbic", "Cont", "Default"]


def load_label_map(dlabel_path: Path) -> tuple[np.ndarray, dict[int, str], nib.cifti2.cifti2_axes.BrainModelAxis]:
    img = nib.load(str(dlabel_path))
    label_axis = img.header.get_axis(0)
    brain_axis = img.header.get_axis(1)
    label_table = label_axis.label[0]
    label_map = {}
    for idx, (name, _rgba) in label_table.items():
        if idx == 0:
            continue
        label_map[int(idx)] = name
    labels = np.asarray(img.dataobj[0], dtype=np.int32)
    return labels, label_map, brain_axis


def schaefer_network_from_name(label_name: str) -> str:
    match = re.match(r"7Networks_[LR]H_([^_]+)", label_name)
    if not match:
        raise ValueError(f"Cannot parse network from label name: {label_name}")
    return match.group(1)


def align_labels_to_dtseries(
    dt_axis: nib.cifti2.cifti2_axes.BrainModelAxis,
    dlabel_axis: nib.cifti2.cifti2_axes.BrainModelAxis,
    dlabel_values: np.ndarray,
) -> np.ndarray:
    aligned = np.zeros(dt_axis.size, dtype=np.int32)
    dlabel_structs: dict[str, tuple[nib.cifti2.cifti2_axes.BrainModelAxis, np.ndarray]] = {}
    for name, slc, subaxis in dlabel_axis.iter_structures():
        dlabel_structs[name] = (subaxis, dlabel_values[slc])

    for name, slc, subaxis in dt_axis.iter_structures():
        if name not in dlabel_structs:
            continue
        d_subaxis, d_values = dlabel_structs[name]
        if getattr(subaxis, "vertex", None) is not None and getattr(d_subaxis, "vertex", None) is not None:
            if np.array_equal(subaxis.vertex, d_subaxis.vertex):
                aligned[slc] = d_values
                continue
            value_by_vertex = {int(vertex): int(value) for vertex, value in zip(d_subaxis.vertex, d_values, strict=True)}
            aligned[slc] = np.asarray([value_by_vertex.get(int(vertex), 0) for vertex in subaxis.vertex], dtype=np.int32)
            continue
        raise RuntimeError(f"Unsupported brain model alignment for structure {name}")
    return aligned


def main() -> int:
    parser = argparse.ArgumentParser(description="Extract Schaefer400 parcel and 7-network reference timeseries from CIFTI dtseries")
    parser.add_argument("--dtseries", required=True)
    parser.add_argument("--atlas-dlabel", required=True)
    parser.add_argument("--outdir", required=True)
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    label_values, label_map, dlabel_axis = load_label_map(Path(args.atlas_dlabel))
    dt_img = nib.load(args.dtseries)
    dt_axis = dt_img.header.get_axis(1)
    aligned_labels = align_labels_to_dtseries(dt_axis, dlabel_axis, label_values)
    parcel_ids = sorted(int(x) for x in np.unique(aligned_labels) if x > 0)
    if not parcel_ids:
        raise RuntimeError("No Schaefer parcels were aligned onto the dtseries grayordinates")

    dt_data = np.asarray(dt_img.dataobj, dtype=np.float32)
    parcel_ts = []
    parcel_rows: list[dict[str, object]] = []
    for pid in parcel_ids:
        grayordinates = aligned_labels == pid
        if not np.any(grayordinates):
            continue
        ts = np.nanmean(dt_data[:, grayordinates], axis=1)
        name = label_map.get(pid, f"parcel_{pid}")
        network = schaefer_network_from_name(name)
        parcel_ts.append(ts)
        parcel_rows.append(
            {
                "parcel_id": pid,
                "parcel_name": name,
                "network": network,
                "n_grayordinates": int(grayordinates.sum()),
            }
        )

    parcel_ts_arr = np.asarray(parcel_ts, dtype=np.float32)
    np.save(outdir / "schaefer400_parcel_timeseries.npy", parcel_ts_arr)

    with (outdir / "schaefer400_parcels.tsv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["parcel_id", "parcel_name", "network", "n_grayordinates"],
            delimiter="\t",
        )
        writer.writeheader()
        writer.writerows(parcel_rows)

    network_ts = []
    network_rows = []
    for network in NETWORK_ORDER:
        inds = [idx for idx, row in enumerate(parcel_rows) if row["network"] == network]
        if not inds:
            continue
        ts = np.nanmean(parcel_ts_arr[inds], axis=0)
        network_ts.append(ts)
        network_rows.append({"network": network, "n_parcels": len(inds)})
    np.save(outdir / "schaefer7_network_timeseries.npy", np.asarray(network_ts, dtype=np.float32))
    (outdir / "schaefer7_networks.json").write_text(json.dumps(network_rows, indent=2), encoding="utf-8")
    (outdir / "reference_input.json").write_text(
        json.dumps(
            {
                "mode": "cifti",
                "dtseries": str(Path(args.dtseries)),
                "atlas_dlabel": str(Path(args.atlas_dlabel)),
                "n_parcels": len(parcel_rows),
                "n_timepoints": int(parcel_ts_arr.shape[1]),
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    print(json.dumps({"n_parcels": len(parcel_rows), "n_networks": len(network_rows), "mode": "cifti"}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
