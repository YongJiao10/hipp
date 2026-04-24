#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_GROUP_ROOT = REPO_ROOT / "outputs_migration" / "hipp_group_prior_fastpfm"
DEFAULT_SUBJECT_ROOT = REPO_ROOT / "outputs_migration" / "hipp_functional_parcellation_network"


def reorder_rows(rows: np.ndarray, mapping: dict[str, int], k: int) -> np.ndarray:
    if rows.shape[0] != k:
        raise RuntimeError(f"row count mismatch: got {rows.shape[0]} rows, expected K={k}")
    out = np.zeros((k, rows.shape[1]), dtype=np.float64)
    for old_label_str, new_label in mapping.items():
        old_label = int(old_label_str)
        if not (1 <= old_label <= k and 1 <= int(new_label) <= k):
            raise RuntimeError(f"invalid mapping old={old_label} new={new_label} for K={k}")
        out[int(new_label) - 1, :] = rows[old_label - 1, :]
    return out


def process_prior(
    *,
    prior_path: Path,
    subject_root: Path,
) -> dict[str, object]:
    prior = pickle.loads(prior_path.read_bytes())
    branch = str(prior["branch"])
    atlas = str(prior["atlas"])
    smoothing = str(prior["smoothing"])
    hemi = str(prior["hemi"])
    k_final = int(prior["k_final"])
    subjects = [str(x) for x in prior["subjects"]]
    networks = [str(x) for x in prior["profile_networks"]]
    mappings = prior["label_mappings_to_reference"]

    aligned_raw: list[np.ndarray] = []
    for sid in subjects:
        ann_path = (
            subject_root
            / branch
            / atlas
            / f"sub-{sid}"
            / "clustering"
            / smoothing
            / f"hemi_{hemi}"
            / f"k_{k_final}"
            / "cluster_annotation.json"
        )
        if not ann_path.exists():
            raise FileNotFoundError(f"Missing cluster annotation: {ann_path}")
        ann = json.loads(ann_path.read_text(encoding="utf-8"))
        ann_networks = [str(x) for x in ann.get("networks", [])]
        if ann_networks != networks:
            raise RuntimeError(
                f"Network order mismatch in {ann_path}: expected {networks}, got {ann_networks}"
            )
        raw = np.asarray(ann.get("raw_profile_rows", []), dtype=np.float64)
        if raw.ndim != 2:
            raise RuntimeError(f"raw_profile_rows must be 2D in {ann_path}")
        reordered = reorder_rows(raw, mappings[sid], k_final)
        aligned_raw.append(reordered)

    cluster_network_raw_corr = np.mean(np.stack(aligned_raw, axis=0), axis=0).astype(np.float32)
    prior["cluster_network_raw_corr"] = cluster_network_raw_corr

    row_max = np.max(cluster_network_raw_corr, axis=1)
    n_cluster_rows_negative = int(np.sum(row_max < 0))

    return {
        "prior": prior,
        "summary": {
            "prior_pickle": str(prior_path),
            "branch": branch,
            "atlas": atlas,
            "smoothing": smoothing,
            "hemi": hemi,
            "k_final": k_final,
            "n_subjects": len(subjects),
            "n_networks": len(networks),
            "cluster_network_raw_corr_shape": [
                int(cluster_network_raw_corr.shape[0]),
                int(cluster_network_raw_corr.shape[1]),
            ],
            "n_cluster_rows_with_negative_max_corr": n_cluster_rows_negative,
            "cluster_rows_with_negative_max_corr_rate": float(
                n_cluster_rows_negative / max(1, cluster_network_raw_corr.shape[0])
            ),
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Backfill group-level signed cluster-network correlation matrix into group prior pickles."
    )
    parser.add_argument("--group-root", default=str(DEFAULT_GROUP_ROOT))
    parser.add_argument("--subject-root", default=str(DEFAULT_SUBJECT_ROOT))
    parser.add_argument("--write", action="store_true", help="Overwrite prior pickles with new key.")
    args = parser.parse_args()

    group_root = Path(args.group_root).resolve()
    subject_root = Path(args.subject_root).resolve()

    manifests = sorted(group_root.glob("*/*/*/group_prior_manifest.json"))
    if not manifests:
        raise RuntimeError(f"No group manifests under {group_root}")

    summary_rows: list[dict[str, object]] = []
    for man_path in manifests:
        man = json.loads(man_path.read_text(encoding="utf-8"))
        for hemi in ["L", "R"]:
            prior_path = Path(str(man["hemi_priors"][hemi]["prior_pickle"]))
            result = process_prior(prior_path=prior_path, subject_root=subject_root)
            if args.write:
                prior_path.write_bytes(pickle.dumps(result["prior"]))
            summary_rows.append(result["summary"])

    out_json = group_root / "group_cluster_raw_corr_backfill_summary.json"
    out_json.write_text(json.dumps({"items": summary_rows}, indent=2), encoding="utf-8")

    # Write TSV for quick spreadsheet review.
    tsv_path = group_root / "group_cluster_raw_corr_backfill_summary.tsv"
    headers = [
        "branch",
        "atlas",
        "smoothing",
        "hemi",
        "k_final",
        "n_subjects",
        "n_networks",
        "cluster_network_raw_corr_shape",
        "n_cluster_rows_with_negative_max_corr",
        "cluster_rows_with_negative_max_corr_rate",
        "prior_pickle",
    ]
    lines = ["\t".join(headers)]
    for row in summary_rows:
        lines.append(
            "\t".join(
                [
                    str(row.get("branch", "")),
                    str(row.get("atlas", "")),
                    str(row.get("smoothing", "")),
                    str(row.get("hemi", "")),
                    str(row.get("k_final", "")),
                    str(row.get("n_subjects", "")),
                    str(row.get("n_networks", "")),
                    str(row.get("cluster_network_raw_corr_shape", "")),
                    str(row.get("n_cluster_rows_with_negative_max_corr", "")),
                    str(row.get("cluster_rows_with_negative_max_corr_rate", "")),
                    str(row.get("prior_pickle", "")),
                ]
            )
        )
    tsv_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(json.dumps({"group_root": str(group_root), "count": len(summary_rows), "write": bool(args.write)}, indent=2))
    print(f"Wrote {out_json}")
    print(f"Wrote {tsv_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

