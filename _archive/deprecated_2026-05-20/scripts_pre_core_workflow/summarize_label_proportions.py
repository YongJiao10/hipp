#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import nibabel as nib
import numpy as np


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


def main() -> int:
    parser = argparse.ArgumentParser(description="Summarize voxel counts and proportions for a discrete label image")
    parser.add_argument("--labels", required=True, help="3D label image")
    parser.add_argument("--label-names", default=None, help="Optional JSON/TSV or inline '1:Vis,2:SomMot'")
    parser.add_argument("--mask", default=None, help="Optional analysis mask to intersect before summarizing")
    parser.add_argument("--out", default=None, help="Optional JSON output path")
    args = parser.parse_args()

    labels_img = nib.load(args.labels)
    labels = np.rint(np.asanyarray(labels_img.dataobj).squeeze()).astype(np.int16)
    include = np.ones(labels.shape, dtype=bool)
    if args.mask:
        mask = np.asanyarray(nib.load(args.mask).dataobj).squeeze() > 0
        if mask.shape != labels.shape:
            raise ValueError(f"Mask shape mismatch: {mask.shape} vs {labels.shape}")
        include &= mask

    label_names = parse_label_names(args.label_names)
    label_ids = sorted(int(x) for x in np.unique(labels[include]) if x > 0)
    total = int(np.count_nonzero((labels > 0) & include))
    if total == 0:
        raise ValueError("No positive labels found in the requested analysis mask")

    rows = []
    for label in label_ids:
        count = int(np.count_nonzero((labels == label) & include))
        rows.append(
            {
                "label": label,
                "label_name": label_names.get(label, f"label-{label:02d}"),
                "n_voxels": count,
                "proportion": count / total,
                "percent": (count / total) * 100.0,
            }
        )

    payload = {"total_labeled_voxels": total, "labels": rows}
    if args.out:
        Path(args.out).write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
