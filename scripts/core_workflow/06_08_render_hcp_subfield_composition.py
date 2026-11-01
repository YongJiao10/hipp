#!/usr/bin/env python3
# pyright: reportMissingImports=false, reportAttributeAccessIssue=false
from __future__ import annotations

"""
Render HCP 151-subject weighted_fisherz gradient cluster subfield composition.

The figure overlaps RouteA weighted_fisherz K=3 gradient-spectral group labels
with subject-level HippUnfold multihist7 subfield labels. Composition is
computed within each subject first, using den-512 T1w midthickness vertex
area weights, then averaged across subjects.
"""

import argparse
import json
import math
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np

from lib.io_utils import load_json
from lib.workflow_utils import write_json


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_GROUP_ROOT = REPO_ROOT / "outputs_hcp" / "work" / "group_atlas" / "pfm"
DEFAULT_HIPPUNFOLD_ROOT = REPO_ROOT / "outputs_hcp" / "inputs" / "hippunfold" / "dense_corobl_batch"
DEFAULT_OUT_DIR = REPO_ROOT / "outputs_hcp" / "present" / "subfield_composition"
METHOD = "gradient-spectral"
ATLAS = "kong2022"
COVERAGE_TAG = "cov_0p80"
COMPONENT_TAG = "g1g2g3"
SMOOTHINGS = ("2mm", "4mm")
HEMIS = ("L", "R")
CLUSTER_NAMES = {1: "Anterior", 2: "Middle", 3: "Posterior"}
SUBFIELD_ORDER = (1, 2, 3, 4, 5, 6, 7)
SUBFIELD_NAMES = {
    1: "Sub",
    2: "CA1",
    3: "CA2",
    4: "CA3",
    5: "CA4",
    6: "DG",
    7: "SRLM",
}
SUBFIELD_FULL_NAMES = {
    1: "subiculum",
    2: "CA1",
    3: "CA2",
    4: "CA3",
    5: "CA4",
    6: "dentate_gyrus",
    7: "SRLM",
    8: "cysts",
}
SUBFIELD_COLORS = {
    1: "#292BFC",
    2: "#66D1D8",
    3: "#00B33C",
    4: "#FFB600",
    5: "#FF2B17",
    6: "#F5EC00",
    7: "#77369B",
}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render HCP weighted_fisherz K=3 gradient cluster subfield composition bars."
    )
    parser.add_argument("--group-root", default=str(DEFAULT_GROUP_ROOT))
    parser.add_argument("--hippunfold-root", default=str(DEFAULT_HIPPUNFOLD_ROOT))
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    parser.add_argument("--smoothings", nargs="+", default=list(SMOOTHINGS))
    parser.add_argument("--hemis", nargs="+", default=list(HEMIS))
    parser.add_argument("--method", default=METHOD)
    parser.add_argument("--atlas", default=ATLAS)
    parser.add_argument("--coverage-tag", default=COVERAGE_TAG)
    parser.add_argument("--gradient-components", default=COMPONENT_TAG)
    return parser.parse_args(argv)


def write_tsv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    lines = ["\t".join(fields)]
    for row in rows:
        lines.append("\t".join(_format_tsv_value(row.get(field, "")) for field in fields))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _format_tsv_value(value: Any) -> str:
    if isinstance(value, float):
        if math.isnan(value):
            return "nan"
        return f"{value:.10g}"
    return str(value)


def group_run_dir(
    *,
    group_root: Path,
    method: str,
    atlas: str,
    smoothing: str,
    coverage_tag: str,
    gradient_components: str,
    hemi: str,
) -> Path:
    return group_root / method / atlas / smoothing / coverage_tag / gradient_components / f"hemi_{hemi}"


def validate_group_manifest(manifest: dict[str, Any], manifest_path: Path) -> list[str]:
    if int(manifest.get("n_subjects", -1)) != 151:
        raise ValueError(f"Expected n_subjects=151 in {manifest_path}, got {manifest.get('n_subjects')}")
    if int(manifest.get("chosen_k_final", -1)) != 3:
        raise ValueError(f"Expected chosen_k_final=3 in {manifest_path}, got {manifest.get('chosen_k_final')}")
    if manifest.get("cluster_label_ordering") != "long_axis_head_to_tail":
        raise ValueError(
            f"Expected cluster_label_ordering=long_axis_head_to_tail in {manifest_path}, "
            f"got {manifest.get('cluster_label_ordering')}"
        )
    raw_subjects = manifest.get("cluster_roi_profile_subjects_used")
    if not raw_subjects:
        raw_subjects = [f"sub-{key}" for key in sorted(manifest.get("actual_consumed_input_paths", {}))]
    subjects = [str(subject).removeprefix("sub-") for subject in raw_subjects]
    if len(subjects) != 151:
        raise ValueError(f"Expected 151 subjects in {manifest_path}, got {len(subjects)}")
    return subjects


def load_vector(path: Path, *, dtype: Any) -> np.ndarray:
    arr = np.load(path).astype(dtype, copy=False).reshape(-1)
    if arr.shape[0] != 512:
        raise ValueError(f"Expected 512 values in {path}, got {arr.shape[0]}")
    return arr


def load_label_gifti(path: Path) -> np.ndarray:
    img = nib.load(str(path))
    if not img.darrays:
        raise ValueError(f"GIFTI has no data arrays: {path}")
    arr = np.asarray(img.darrays[0].data).astype(np.int32, copy=False).reshape(-1)
    if arr.shape[0] != 512:
        raise ValueError(f"Expected 512 label values in {path}, got {arr.shape[0]}")
    return arr


def vertex_area_weights(surface_path: Path) -> np.ndarray:
    img = nib.load(str(surface_path))
    coords = np.asarray(img.agg_data("pointset"), dtype=np.float64)
    faces = np.asarray(img.agg_data("triangle"), dtype=np.int64)
    if coords.ndim != 2 or coords.shape[1] != 3:
        raise ValueError(f"Expected Nx3 coordinates in {surface_path}, got {coords.shape}")
    if faces.ndim != 2 or faces.shape[1] != 3:
        raise ValueError(f"Expected triangle faces in {surface_path}, got {faces.shape}")
    tri = coords[faces]
    areas = 0.5 * np.linalg.norm(np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]), axis=1)
    weights = np.zeros(coords.shape[0], dtype=np.float64)
    for corner in range(3):
        np.add.at(weights, faces[:, corner], areas / 3.0)
    if not np.all(np.isfinite(weights)) or float(np.sum(weights)) <= 0.0:
        raise ValueError(f"Invalid vertex area weights computed from {surface_path}")
    return weights


def _load_512_vertex_area_weights(surface_path: Path) -> np.ndarray:
    weights = vertex_area_weights(surface_path)
    if weights.shape != (512,):
        raise ValueError(f"Expected 512 vertex area weights in {surface_path}, got {weights.shape}")
    return weights


def subfield_label_path(hippunfold_root: Path, subject: str, hemi: str) -> Path:
    return (
        hippunfold_root
        / f"sub-{subject}"
        / "hippunfold"
        / f"sub-{subject}"
        / "metric"
        / f"sub-{subject}_hemi-{hemi}_den-512_label-hipp_atlas-multihist7_subfields.label.gii"
    )


def midthickness_surface_path(hippunfold_root: Path, subject: str, hemi: str) -> Path:
    return (
        hippunfold_root
        / f"sub-{subject}"
        / "hippunfold"
        / f"sub-{subject}"
        / "surf"
        / f"sub-{subject}_hemi-{hemi}_space-T1w_den-512_label-hipp_midthickness.surf.gii"
    )


def subject_cluster_subfield_rows(
    *,
    subject: str,
    smoothing: str,
    hemi: str,
    cluster_labels: np.ndarray,
    support_mask: np.ndarray,
    subfield_labels: np.ndarray,
    vertex_weights: np.ndarray,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    labels = np.asarray(cluster_labels, dtype=np.int32).reshape(-1)
    support = np.asarray(support_mask, dtype=bool).reshape(-1)
    subfields = np.asarray(subfield_labels, dtype=np.int32).reshape(-1)
    weights = np.asarray(vertex_weights, dtype=np.float64).reshape(-1)
    if not (labels.shape == support.shape == subfields.shape == weights.shape):
        raise ValueError(
            "Expected matching labels/support/subfields/weights, got "
            f"{labels.shape}, {support.shape}, {subfields.shape}, {weights.shape}"
        )

    rows: list[dict[str, Any]] = []
    qc = {
        "subject": subject,
        "smoothing": smoothing,
        "hemi": hemi,
        "excluded_zero_area": float(np.sum(weights[support & (labels > 0) & (subfields == 0)])),
        "excluded_cyst_area": float(np.sum(weights[support & (labels > 0) & (subfields == 8)])),
        "cluster_valid_area": {},
    }
    valid_anatomic = support & (labels > 0) & np.isin(subfields, np.asarray(SUBFIELD_ORDER, dtype=np.int32))
    for cluster in (1, 2, 3):
        cluster_mask = valid_anatomic & (labels == int(cluster))
        denom = float(np.sum(weights[cluster_mask]))
        qc["cluster_valid_area"][int(cluster)] = denom
        for subfield_index in SUBFIELD_ORDER:
            sf_mask = cluster_mask & (subfields == int(subfield_index))
            area = float(np.sum(weights[sf_mask]))
            prop = area / denom if denom > 0.0 else 0.0
            rows.append(
                {
                    "subject": subject,
                    "smoothing": smoothing,
                    "hemi": hemi,
                    "cluster": int(cluster),
                    "cluster_name": CLUSTER_NAMES[int(cluster)],
                    "subfield_index": int(subfield_index),
                    "subfield": SUBFIELD_NAMES[int(subfield_index)],
                    "subfield_full_name": SUBFIELD_FULL_NAMES[int(subfield_index)],
                    "area": area,
                    "cluster_valid_area": denom,
                    "proportion": float(prop),
                }
            )
    return rows, qc


def collect_subject_rows_for_combo(
    *,
    group_root: Path,
    hippunfold_root: Path,
    method: str,
    atlas: str,
    smoothing: str,
    coverage_tag: str,
    gradient_components: str,
    hemi: str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    run_dir = group_run_dir(
        group_root=group_root,
        method=method,
        atlas=atlas,
        smoothing=smoothing,
        coverage_tag=coverage_tag,
        gradient_components=gradient_components,
        hemi=hemi,
    )
    manifest_path = run_dir / "group_atlas_manifest.json"
    manifest = load_json(manifest_path)
    subjects = validate_group_manifest(manifest, manifest_path)
    cluster_labels = load_vector(Path(manifest["group_labels_final"]), dtype=np.int32)
    support_mask = load_vector(Path(manifest["group_support_mask"]), dtype=bool)

    rows: list[dict[str, Any]] = []
    qc_rows: list[dict[str, Any]] = []
    for subject in subjects:
        label_path = subfield_label_path(hippunfold_root, subject, hemi)
        surface_path = midthickness_surface_path(hippunfold_root, subject, hemi)
        if not label_path.exists():
            raise FileNotFoundError(f"Missing subfield label: {label_path}")
        if not surface_path.exists():
            raise FileNotFoundError(f"Missing midthickness surface: {surface_path}")
        subject_rows, qc = subject_cluster_subfield_rows(
            subject=subject,
            smoothing=smoothing,
            hemi=hemi,
            cluster_labels=cluster_labels,
            support_mask=support_mask,
            subfield_labels=load_label_gifti(label_path),
            vertex_weights=_load_512_vertex_area_weights(surface_path),
        )
        rows.extend(subject_rows)
        qc_rows.append(qc)

    return rows, {
        "run_dir": str(run_dir),
        "manifest": str(manifest_path),
        "n_subjects": len(subjects),
        "subjects": [f"sub-{subject}" for subject in subjects],
        "qc": qc_rows,
    }


def summarize_subject_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, int, str], list[float]] = defaultdict(list)
    meta: dict[tuple[str, str, int, str], dict[str, Any]] = {}
    for row in rows:
        key = (str(row["smoothing"]), str(row["hemi"]), int(row["cluster"]), str(row["subfield"]))
        grouped[key].append(float(row["proportion"]))
        meta[key] = row

    out: list[dict[str, Any]] = []
    for key in sorted(grouped, key=lambda k: (k[0], k[1], k[2], SUBFIELD_ORDER.index(_subfield_index(k[3])))):
        vals = np.asarray(grouped[key], dtype=np.float64)
        mean = float(np.mean(vals)) if vals.size else float("nan")
        sem = float(np.std(vals, ddof=1) / math.sqrt(vals.size)) if vals.size > 1 else 0.0
        row_meta = meta[key]
        subfield_index = int(row_meta.get("subfield_index", _subfield_index(key[3])))
        out.append(
            {
                "smoothing": key[0],
                "hemi": key[1],
                "cluster": key[2],
                "cluster_name": CLUSTER_NAMES[key[2]],
                "subfield_index": subfield_index,
                "subfield": key[3],
                "subfield_full_name": row_meta.get("subfield_full_name", SUBFIELD_FULL_NAMES[subfield_index]),
                "mean_proportion": mean,
                "sem_proportion": sem,
                "mean_percent": mean * 100.0,
                "sem_percent": sem * 100.0,
                "n_subjects": int(vals.size),
            }
        )
    return out


def _subfield_index(name: str) -> int:
    for idx, label in SUBFIELD_NAMES.items():
        if label == name:
            return int(idx)
    raise KeyError(f"Unknown subfield name: {name}")


def active_subfield_indices(summary_rows: list[dict[str, Any]], *, epsilon: float = 1e-12) -> list[int]:
    active: set[int] = set()
    for row in summary_rows:
        value = float(row.get("mean_proportion", 0.0))
        if math.isfinite(value) and value > float(epsilon):
            active.add(int(row["subfield_index"]))
    return [idx for idx in SUBFIELD_ORDER if idx in active]


def plot_stacked_bars(
    *,
    summary_rows: list[dict[str, Any]],
    smoothings: list[str],
    hemis: list[str],
    output_png: Path,
) -> None:
    nrows = len(smoothings)
    ncols = len(hemis)
    fig, axes = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        figsize=(5.8 * ncols, 2.6 * nrows + 0.8),
        squeeze=False,
        sharex=True,
    )
    by_key = {
        (row["smoothing"], row["hemi"], int(row["cluster"]), int(row["subfield_index"])): float(row["mean_percent"])
        for row in summary_rows
    }
    plotted_subfields = active_subfield_indices(summary_rows)
    if not plotted_subfields:
        raise ValueError("No nonzero subfield proportions available for plotting")
    y_positions = np.arange(3)
    cluster_order = (1, 2, 3)
    for row_idx, smoothing in enumerate(smoothings):
        for col_idx, hemi in enumerate(hemis):
            ax = axes[row_idx][col_idx]
            left = np.zeros(3, dtype=np.float64)
            for subfield_idx in plotted_subfields:
                vals = np.asarray(
                    [
                        by_key.get((smoothing, hemi, cluster, int(subfield_idx)), 0.0)
                        for cluster in cluster_order
                    ],
                    dtype=np.float64,
                )
                bars = ax.barh(
                    y_positions,
                    vals,
                    left=left,
                    height=0.62,
                    color=SUBFIELD_COLORS[int(subfield_idx)],
                    edgecolor="#2f3437",
                    linewidth=0.35,
                    label=SUBFIELD_NAMES[int(subfield_idx)] if row_idx == 0 and col_idx == 0 else None,
                )
                for bar, value in zip(bars, vals):
                    if value >= 8.0:
                        ax.text(
                            bar.get_x() + bar.get_width() / 2.0,
                            bar.get_y() + bar.get_height() / 2.0,
                            f"{value:.0f}",
                            ha="center",
                            va="center",
                            fontsize=7,
                            color="#111111",
                        )
                left += vals
            ax.set_title(f"{smoothing} hemi-{hemi}", fontsize=11)
            ax.set_yticks(y_positions)
            ax.set_yticklabels([CLUSTER_NAMES[c] for c in cluster_order], fontsize=9)
            ax.invert_yaxis()
            ax.set_xlim(0, 100)
            ax.set_xlabel("Subject-mean subfield composition (%)", fontsize=9)
            ax.grid(axis="x", color="#d8dde2", linewidth=0.6)
            ax.set_axisbelow(True)
            for spine in ("top", "right"):
                ax.spines[spine].set_visible(False)

    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=len(plotted_subfields), frameon=False, fontsize=9)
    fig.suptitle(
        "HCP 151 weighted_fisherz gradient clusters: HippUnfold subfield composition",
        fontsize=13,
        y=0.98,
    )
    fig.tight_layout(rect=(0.02, 0.08, 1.0, 0.94))
    output_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_png, dpi=220)
    plt.close(fig)


def output_stem() -> str:
    return "hcp_pfm_weighted-fisherz_gradient-spectral_k3_g1g2g3_subfield_composition"


def run(args: argparse.Namespace) -> dict[str, Path]:
    group_root = Path(args.group_root).resolve()
    hippunfold_root = Path(args.hippunfold_root).resolve()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    subject_rows: list[dict[str, Any]] = []
    combo_manifests: list[dict[str, Any]] = []
    for smoothing in args.smoothings:
        for hemi in args.hemis:
            rows, combo = collect_subject_rows_for_combo(
                group_root=group_root,
                hippunfold_root=hippunfold_root,
                method=args.method,
                atlas=args.atlas,
                smoothing=str(smoothing),
                coverage_tag=args.coverage_tag,
                gradient_components=args.gradient_components,
                hemi=str(hemi).upper(),
            )
            subject_rows.extend(rows)
            combo_manifests.append(combo)

    summary_rows = summarize_subject_rows(subject_rows)
    stem = output_stem()
    png = out_dir / f"{stem}_2mm4mm_LR_stacked_bar.png"
    subject_tsv = out_dir / f"{stem}_subject_long.tsv"
    summary_tsv = out_dir / f"{stem}_summary.tsv"
    manifest_json = out_dir / f"{stem}_manifest.json"
    plot_stacked_bars(summary_rows=summary_rows, smoothings=list(args.smoothings), hemis=[h.upper() for h in args.hemis], output_png=png)
    write_tsv(
        subject_tsv,
        subject_rows,
        [
            "subject",
            "smoothing",
            "hemi",
            "cluster",
            "cluster_name",
            "subfield_index",
            "subfield",
            "subfield_full_name",
            "area",
            "cluster_valid_area",
            "proportion",
        ],
    )
    write_tsv(
        summary_tsv,
        summary_rows,
        [
            "smoothing",
            "hemi",
            "cluster",
            "cluster_name",
            "subfield_index",
            "subfield",
            "subfield_full_name",
            "mean_proportion",
            "sem_proportion",
            "mean_percent",
            "sem_percent",
            "n_subjects",
        ],
    )
    write_json(
        manifest_json,
        {
            "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "group_root": str(group_root),
            "hippunfold_root": str(hippunfold_root),
            "method": args.method,
            "atlas": args.atlas,
            "coverage_tag": args.coverage_tag,
            "gradient_components": args.gradient_components,
            "gradient_default_policy": "unqualified gradient uses pfm",
            "smoothings": list(args.smoothings),
            "hemis": [h.upper() for h in args.hemis],
            "cluster_display_names": CLUSTER_NAMES,
            "subfield_order": [SUBFIELD_NAMES[idx] for idx in SUBFIELD_ORDER],
            "excluded_labels": {"0": "non-label", "8": "cysts"},
            "weighting": "subject_level_space-T1w_den-512_label-hipp_midthickness_vertex_area",
            "aggregation": "subject_normalized_proportions_then_mean_and_sem",
            "outputs": {
                "png": str(png),
                "subject_long_tsv": str(subject_tsv),
                "summary_tsv": str(summary_tsv),
            },
            "combo_manifests": combo_manifests,
        },
        sort_keys=True, trailing_newline=True,
    )
    return {"png": png, "subject_tsv": subject_tsv, "summary_tsv": summary_tsv, "manifest": manifest_json}


def main() -> int:
    outputs = run(parse_args())
    for key, path in outputs.items():
        print(f"{key}: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
