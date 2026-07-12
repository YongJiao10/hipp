#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import adjusted_rand_score


SMOOTHINGS = ("2mm", "4mm")
HEMIS = ("L", "R")
SEMANTICS = {1: "anterior", 2: "body", 3: "posterior"}


def compare_label_arrays(old: np.ndarray, new: np.ndarray, *, k: int = 3) -> dict[str, Any]:
    old_labels = np.asarray(old, dtype=np.int32).reshape(-1)
    new_labels = np.asarray(new, dtype=np.int32).reshape(-1)
    if old_labels.shape != new_labels.shape:
        raise ValueError(
            f"Old/new label shape mismatch: old={old_labels.shape} new={new_labels.shape}"
        )
    common = (old_labels > 0) & (new_labels > 0)
    if not np.any(common):
        raise ValueError("Old/new labels have no common non-null support")
    first = old_labels[common]
    second = new_labels[common]
    confusion = np.asarray(
        [
            [np.sum((first == source) & (second == target)) for target in range(1, k + 1)]
            for source in range(1, k + 1)
        ],
        dtype=np.int64,
    )
    rows, columns = linear_sum_assignment(-confusion)
    mapping = {int(columns[index] + 1): int(rows[index] + 1) for index in range(len(rows))}
    dice: dict[int, float] = {}
    for label in range(1, int(k) + 1):
        old_members = first == label
        new_members = second == label
        denominator = int(old_members.sum() + new_members.sum())
        dice[label] = 1.0 if denominator == 0 else float(
            2 * np.sum(old_members & new_members) / denominator
        )
    row: dict[str, Any] = {
        "n_common_support": int(common.sum()),
        "changed_vertex_count": int(np.sum(first != second)),
        "changed_vertex_fraction": float(np.mean(first != second)),
        "adjusted_rand_index": float(adjusted_rand_score(first, second)),
        "mean_semantic_dice": float(np.mean(list(dice.values()))),
        "min_semantic_dice": float(np.min(list(dice.values()))),
        "hungarian_mapping": {str(key): value for key, value in sorted(mapping.items())},
        "hungarian_mapping_identity": mapping == {label: label for label in range(1, k + 1)},
    }
    for label, semantic in SEMANTICS.items():
        row[f"dice_{semantic}"] = dice[label]
        row[f"old_size_{semantic}"] = int(np.sum(first == label))
        row[f"new_size_{semantic}"] = int(np.sum(second == label))
    return row


def _run_dir(root: Path, smoothing: str, hemi: str) -> Path:
    return (
        Path(root)
        / "gradient-spectral"
        / "kong2022"
        / smoothing
        / "cov_0p80"
        / "g1g2g3"
        / f"hemi_{hemi}"
    )


def _network_assignments(path: Path) -> dict[str, str | None]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    result: dict[str, str | None] = {}
    for item in payload.get("items", []):
        cluster = str(int(item["cluster_id"]))
        result[cluster] = item.get("top1_network") or item.get("assigned_network")
    return result


def compare_roots(name: str, old_root: Path, new_root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for smoothing in SMOOTHINGS:
        for hemi in HEMIS:
            old_dir = _run_dir(old_root, smoothing, hemi)
            new_dir = _run_dir(new_root, smoothing, hemi)
            old_labels_path = old_dir / "group_labels_final.npy"
            new_labels_path = new_dir / "group_labels_final.npy"
            for path in (old_labels_path, new_labels_path):
                if not path.exists():
                    raise FileNotFoundError(f"Missing comparison input: {path}")
            row = {
                "comparison": str(name),
                "smoothing": smoothing,
                "hemi": hemi,
                "old_root": str(Path(old_root).resolve()),
                "new_root": str(Path(new_root).resolve()),
                **compare_label_arrays(np.load(old_labels_path), np.load(new_labels_path)),
            }
            for variant in ("fisherz", "count"):
                old_status = old_dir / f"cluster_assignment_status_{variant}.json"
                new_status = new_dir / f"cluster_assignment_status_{variant}.json"
                old_networks = _network_assignments(old_status)
                new_networks = _network_assignments(new_status)
                row[f"old_{variant}_networks"] = json.dumps(old_networks, sort_keys=True)
                row[f"new_{variant}_networks"] = json.dumps(new_networks, sort_keys=True)
                row[f"{variant}_network_assignments_changed"] = int(old_networks != new_networks)
            rows.append(row)
    return rows


def _write_outputs(rows: list[dict[str, Any]], output_dir: Path) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    tsv = output_dir / "gradient_spectral_affinity_difference_summary.tsv"
    manifest = output_dir / "gradient_spectral_affinity_difference_manifest.json"
    markdown = output_dir / "gradient_spectral_affinity_difference_summary.md"
    fields = list(rows[0])
    with tsv.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    manifest.write_text(
        json.dumps({"status": "ok", "rows": rows}, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    lines = [
        "# Gradient-Spectral Affinity Difference Summary",
        "",
        "Old results use legacy ReLU-cosine; new results use shifted-cosine.",
        "",
        "| Dataset | Smooth | Hemi | ARI | Mean Dice | Min Dice | Changed | Identity AP map |",
        "|---|---:|:---:|---:|---:|---:|---:|:---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['comparison']} | {row['smoothing']} | {row['hemi']} | "
            f"{row['adjusted_rand_index']:.3f} | {row['mean_semantic_dice']:.3f} | "
            f"{row['min_semantic_dice']:.3f} | {100 * row['changed_vertex_fraction']:.1f}% | "
            f"{row['hungarian_mapping_identity']} |"
        )
    markdown.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {"tsv": tsv.resolve(), "manifest": manifest.resolve(), "markdown": markdown.resolve()}


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Compare legacy and shifted-cosine group atlases.")
    parser.add_argument(
        "--comparison",
        nargs=3,
        action="append",
        metavar=("NAME", "OLD_ROOT", "NEW_ROOT"),
        required=True,
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_arg_parser().parse_args(list(argv) if argv is not None else None)
    rows: list[dict[str, Any]] = []
    for name, old_root, new_root in args.comparison:
        rows.extend(compare_roots(name, Path(old_root), Path(new_root)))
    outputs = _write_outputs(rows, args.output_dir)
    print(json.dumps({key: str(value) for key, value in outputs.items()}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
