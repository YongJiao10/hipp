#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import pickle
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import nibabel as nib
import numpy as np
from nibabel.gifti import GiftiDataArray, GiftiImage, GiftiLabel, GiftiLabelTable
from scipy.optimize import linear_sum_assignment


REPO_ROOT = Path(__file__).resolve().parents[2]
PYTHON_EXE = sys.executable or "/opt/miniconda3/envs/py314/bin/python"
WB_COMMAND = str((REPO_ROOT / "scripts" / "wb_command").resolve())
RENDER_SCENE_BATCH = REPO_ROOT / "scripts" / "workbench" / "render_wb_scene_batch.py"
COMPOSE_WB_GRID = REPO_ROOT / "scripts" / "workbench" / "compose_wb_grid_with_legend.py"
NETWORK_STYLE_JSON = REPO_ROOT / "config" / "hipp_network_style.json"
DEFAULT_SCENE = REPO_ROOT / "config" / "wb_locked_native_view_lateral_medial.scene"
DEFAULT_OUT_ROOT = REPO_ROOT / "outputs_migration" / "hipp_functional_parcellation_network"
DEFAULT_GROUP_OUT_ROOT = REPO_ROOT / "outputs_migration" / "hipp_group_prior_fastpfm"

SPECTRAL_BRANCHES = [
    "network-spectral",
    "network-spectral-nonneg",
    "intrinsic-spectral",
    "intrinsic-spectral-nonneg",
]
DEFAULT_ATLASES = ["lynch2024", "kong2019"]
DEFAULT_SUBJECTS = ["100610", "102311", "102816"]
DEFAULT_SMOOTHINGS = ["2mm", "4mm"]
HEMIS = ["L", "R"]
VERSION = "v1"


@dataclass(frozen=True)
class SubjectInputs:
    subject: str
    per_k_summary_tsv: Path
    final_selection_summary_json: Path
    timeseries_npy: Path
    valid_mask_npy: Path


@dataclass(frozen=True)
class RenderAssets:
    left_surface: Path
    right_surface: Path
    spec_path: Path


@dataclass(frozen=True)
class Combo:
    branch: str
    atlas: str
    smoothing: str


def run(cmd: list[str]) -> None:
    proc = subprocess.run(cmd, text=True, capture_output=True)
    if proc.returncode != 0:
        raise RuntimeError(
            f"Command failed ({proc.returncode}): {' '.join(cmd)}\\nSTDOUT:\\n{proc.stdout}\\nSTDERR:\\n{proc.stderr}"
        )


def save_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def read_tsv_rows(path: Path) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            rows.append({str(k): str(v) for k, v in row.items()})
    return rows


def write_tsv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter="\t", extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build hippocampal group priors from spectral outputs and run fast-PFM style individual soft maps."
    )
    parser.add_argument("--branches", nargs="+", default=SPECTRAL_BRANCHES, choices=SPECTRAL_BRANCHES)
    parser.add_argument("--atlases", nargs="+", default=DEFAULT_ATLASES)
    parser.add_argument("--subjects", nargs="+", default=DEFAULT_SUBJECTS)
    parser.add_argument("--smoothings", nargs="+", default=DEFAULT_SMOOTHINGS)
    parser.add_argument("--group-k-rule", default="mean-instability-1se", choices=["mean-instability-1se"])
    parser.add_argument("--min-parcel-pass-rate", type=float, default=0.67)
    parser.add_argument("--inference-subjects", nargs="+", default=None)
    parser.add_argument("--out-root", default=str(DEFAULT_OUT_ROOT))
    parser.add_argument("--group-out-root", default=str(DEFAULT_GROUP_OUT_ROOT))
    parser.add_argument("--scene", default=str(DEFAULT_SCENE))
    parser.add_argument("--views", default="ventral,dorsal")
    parser.add_argument("--layout", choices=["1x2", "2x2"], default="1x2")
    return parser.parse_args()


def load_network_colors(path: Path) -> dict[str, tuple[int, int, int, int]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    out: dict[str, tuple[int, int, int, int]] = {}
    for _, row in payload.items():
        name = str(row["name"])
        rgba_raw = row["rgba"]
        out[name] = tuple(int(v) for v in rgba_raw)
    return out


def extract_network_name(cluster_label: str) -> str:
    if "_" not in cluster_label:
        raise ValueError(f"Cluster label must include '_<network>' suffix, got: {cluster_label}")
    return cluster_label.split("_", 1)[1]


def make_label_gifti(
    labels: np.ndarray,
    key_to_name: dict[int, str],
    network_colors: dict[str, tuple[int, int, int, int]],
) -> GiftiImage:
    table = GiftiLabelTable()
    null_label = GiftiLabel(key=0, red=0.62, green=0.62, blue=0.62, alpha=1.0)
    null_label.label = "Null"
    table.labels.append(null_label)

    for key in sorted(key_to_name):
        cluster_name = key_to_name[key]
        net_name = extract_network_name(cluster_name)
        rgba = network_colors.get(net_name, (128, 128, 128, 255))
        label = GiftiLabel(
            key=int(key),
            red=float(rgba[0]) / 255.0,
            green=float(rgba[1]) / 255.0,
            blue=float(rgba[2]) / 255.0,
            alpha=float(rgba[3]) / 255.0,
        )
        label.label = cluster_name
        table.labels.append(label)

    arr = GiftiDataArray(data=labels.astype(np.int32), intent="NIFTI_INTENT_LABEL", datatype="NIFTI_TYPE_INT32")
    return GiftiImage(darrays=[arr], labeltable=table)


def save_combined_label_assets(
    *,
    subject: str,
    density: str,
    left_labels: np.ndarray,
    right_labels: np.ndarray,
    output_dir: Path,
    left_surface: Path,
    right_surface: Path,
    left_key_to_name: dict[int, str],
    right_key_to_name: dict[int, str],
    stem: str,
    network_colors: dict[str, tuple[int, int, int, int]],
) -> dict[str, str]:
    output_dir.mkdir(parents=True, exist_ok=True)

    right_offset = max(left_key_to_name.keys(), default=0)
    right_labels_shifted = right_labels.astype(np.int32, copy=True)
    right_labels_shifted[right_labels_shifted > 0] += right_offset
    right_key_shifted = {key + right_offset: value for key, value in right_key_to_name.items()}

    left_path = output_dir / f"sub-{subject}_hemi-L_space-corobl_den-{density}_label-{stem}.label.gii"
    right_path = output_dir / f"sub-{subject}_hemi-R_space-corobl_den-{density}_label-{stem}.label.gii"
    dlabel_path = output_dir / f"sub-{subject}_space-corobl_den-{density}_label-{stem}.dlabel.nii"

    nib.save(make_label_gifti(left_labels, left_key_to_name, network_colors), str(left_path))
    nib.save(make_label_gifti(right_labels_shifted, right_key_shifted, network_colors), str(right_path))

    run(
        [
            WB_COMMAND,
            "-cifti-create-label",
            str(dlabel_path),
            "-left-label",
            str(left_path),
            "-right-label",
            str(right_path),
        ]
    )

    return {
        "left_label": str(left_path),
        "right_label": str(right_path),
        "dlabel": str(dlabel_path),
        "left_surface": str(left_surface),
        "right_surface": str(right_surface),
    }


def render_locked_native_view(
    *,
    subject: str,
    scene: Path,
    outdir: Path,
    name: str,
    left_labels: Path,
    right_labels: Path,
    left_surface: Path,
    right_surface: Path,
    spec_path: Path,
) -> Path:
    render_root = outdir / "batch"
    render_root.mkdir(parents=True, exist_ok=True)
    cmd = [
        PYTHON_EXE,
        str(RENDER_SCENE_BATCH),
        "--scene",
        str(scene),
        "--subjects",
        subject,
        "--outdir",
        str(render_root),
        "--scene-index",
        "1",
        "--width",
        "1600",
        "--height",
        "1024",
        "--renderer",
        "OSMesa",
        "--name",
        name,
        "--no-template-scene",
        "--left-label-template",
        str(left_labels),
        "--right-label-template",
        str(right_labels),
        "--left-surface-template",
        str(left_surface),
        "--right-surface-template",
        str(right_surface),
        "--spec-template",
        str(spec_path),
    ]
    run(cmd)
    return render_root / f"sub-{subject}" / f"sub-{subject}_wb_{name}_native.png"


def render_locked_grid_png(
    *,
    subject: str,
    scene: Path,
    outdir: Path,
    name: str,
    title: str,
    left_labels: Path,
    right_labels: Path,
    left_surface: Path,
    right_surface: Path,
    spec_path: Path,
    layout: str,
    views: list[str],
) -> dict[str, str]:
    outdir.mkdir(parents=True, exist_ok=True)

    if "ventral" not in views:
        raise ValueError("views must include ventral for current locked-scene renderer")
    if layout != "1x2":
        raise ValueError(
            "Current locked-scene renderer supports only --layout 1x2 "
            "(single native scene capture, ventral panels only)"
        )

    native_png = render_locked_native_view(
        subject=subject,
        scene=scene,
        outdir=outdir / "native",
        name=f"{name}_native",
        left_labels=left_labels,
        right_labels=right_labels,
        left_surface=left_surface,
        right_surface=right_surface,
        spec_path=spec_path,
    )

    final_png = outdir / f"sub-{subject}_wb_{name}_biglegend.png"
    compose_cmd = [
        PYTHON_EXE,
        str(COMPOSE_WB_GRID),
        "--ventral-image",
        str(native_png),
        "--layout",
        "1x2",
        "--left-labels",
        str(left_labels),
        "--right-labels",
        str(right_labels),
        "--legend-group",
        "label",
        "--title",
        title,
        "--out",
        str(final_png),
    ]
    run(compose_cmd)
    return {
        "biglegend_png": str(final_png),
        "native_scene_png": str(native_png),
    }


def _safe_float(value: str, path: Path, key: str) -> float:
    try:
        return float(value)
    except Exception as exc:
        raise ValueError(f"Failed to parse float key='{key}' from {path}: {value}") from exc


def _safe_int(value: str, path: Path, key: str) -> int:
    try:
        return int(float(value))
    except Exception as exc:
        raise ValueError(f"Failed to parse int key='{key}' from {path}: {value}") from exc


def compute_local_minima_flags(series: list[float]) -> list[bool]:
    if not series:
        return []
    all_equal = len(set(float(x) for x in series)) == 1
    flags: list[bool] = []
    for idx, current in enumerate(series):
        left = series[idx - 1] if idx > 0 else math.inf
        right = series[idx + 1] if idx < len(series) - 1 else math.inf
        is_local = all_equal or (current <= left and current <= right and (current < left or current < right or len(series) == 1))
        flags.append(bool(is_local))
    return flags


def choose_group_k(
    *,
    subjects: list[str],
    per_subject_tsv: dict[str, Path],
    min_parcel_pass_rate: float,
) -> tuple[int, dict[str, object], list[dict[str, object]]]:
    by_k: dict[int, dict[str, list[float]]] = {}
    for subject in subjects:
        tsv_path = per_subject_tsv[subject]
        rows = read_tsv_rows(tsv_path)
        for row in rows:
            k = _safe_int(row["k"], tsv_path, "k")
            slot = by_k.setdefault(
                k,
                {
                    "instability_mean": [],
                    "instability_se": [],
                    "min_parcel_ok": [],
                    "connectivity_ok": [],
                    "null_corrected_score": [],
                    "silhouette": [],
                },
            )
            slot["instability_mean"].append(_safe_float(row["instability_mean"], tsv_path, "instability_mean"))
            slot["instability_se"].append(_safe_float(row["instability_se"], tsv_path, "instability_se"))
            slot["min_parcel_ok"].append(_safe_float(row["min_parcel_ok"], tsv_path, "min_parcel_ok"))
            slot["connectivity_ok"].append(_safe_float(row["connectivity_ok"], tsv_path, "connectivity_ok"))
            slot["null_corrected_score"].append(_safe_float(row["null_corrected_score"], tsv_path, "null_corrected_score"))
            slot["silhouette"].append(_safe_float(row["silhouette"], tsv_path, "silhouette"))

    if not by_k:
        raise RuntimeError("No K rows found while computing group-level K selection")

    ordered_k = sorted(by_k)
    aggregate_rows: list[dict[str, object]] = []
    for k in ordered_k:
        slot = by_k[k]
        instability = np.asarray(slot["instability_mean"], dtype=np.float64)
        inst_mean = float(np.mean(instability))
        inst_se = float(np.std(instability, ddof=1) / np.sqrt(instability.size)) if instability.size > 1 else 0.0
        aggregate_rows.append(
            {
                "k": int(k),
                "n_subjects": int(instability.size),
                "group_instability_mean": inst_mean,
                "group_instability_se": inst_se,
                "group_null_corrected_score_mean": float(np.mean(np.asarray(slot["null_corrected_score"], dtype=np.float64))),
                "group_silhouette_mean": float(np.mean(np.asarray(slot["silhouette"], dtype=np.float64))),
                "min_parcel_pass_rate": float(np.mean(np.asarray(slot["min_parcel_ok"], dtype=np.float64))),
                "connectivity_pass_rate": float(np.mean(np.asarray(slot["connectivity_ok"], dtype=np.float64))),
            }
        )

    inst_series = [float(row["group_instability_mean"]) for row in aggregate_rows]
    local_flags = compute_local_minima_flags(inst_series)
    best_idx = int(np.argmin(inst_series))
    best_mean = float(aggregate_rows[best_idx]["group_instability_mean"])
    best_se = float(aggregate_rows[best_idx]["group_instability_se"])
    cutoff = best_mean + best_se

    for row, is_local in zip(aggregate_rows, local_flags, strict=True):
        parcel_rate = float(row["min_parcel_pass_rate"])
        row["local_min"] = int(is_local)
        row["within_1se_best"] = int(float(row["group_instability_mean"]) <= cutoff + 1e-12)
        # Keep threshold semantics stable for small-N groups where 2/3 can be represented
        # as 0.666..., but users commonly configure 0.67 to mean "about two thirds".
        row["parcel_rate_ok"] = int(round(parcel_rate, 2) >= round(float(min_parcel_pass_rate), 2))

    one_se_candidates = [
        row
        for row in aggregate_rows
        if int(row["local_min"]) == 1 and int(row["within_1se_best"]) == 1
    ]
    if not one_se_candidates:
        raise RuntimeError("No group-level K survived local-minimum and 1-SE screening")

    eligible = [row for row in one_se_candidates if int(row["parcel_rate_ok"]) == 1]
    if not eligible:
        raise RuntimeError(
            "No group-level K survived local-minimum, 1-SE, and min-parcel-pass-rate constraints"
        )

    eligible_sorted = sorted(eligible, key=lambda x: int(x["k"]))
    selected = eligible_sorted[0]
    k_final = int(selected["k"])

    decision = {
        "rule": "mean-instability-1se",
        "subjects": subjects,
        "best_by_instability": int(aggregate_rows[best_idx]["k"]),
        "candidate_local_minima": [int(row["k"]) for row in aggregate_rows if int(row["local_min"]) == 1],
        "one_se_selected": int(min(int(row["k"]) for row in one_se_candidates)),
        "post_constraint_selected": k_final,
        "main_analysis_k": k_final,
        "group_instability_best_mean": best_mean,
        "group_instability_best_se": best_se,
        "one_se_cutoff": cutoff,
        "min_parcel_pass_rate_threshold": float(min_parcel_pass_rate),
    }
    return k_final, decision, aggregate_rows


def build_hungarian_mapping(ref_labels: np.ndarray, tgt_labels: np.ndarray, k: int) -> dict[int, int]:
    overlap = (ref_labels > 0) & (tgt_labels > 0)
    if int(overlap.sum()) == 0:
        raise RuntimeError("No overlapping labeled vertices between reference and target for Hungarian matching")

    contingency = np.zeros((k, k), dtype=np.int64)
    ref_vec = ref_labels[overlap].astype(np.int32)
    tgt_vec = tgt_labels[overlap].astype(np.int32)

    for r, t in zip(ref_vec, tgt_vec, strict=False):
        if 1 <= int(r) <= k and 1 <= int(t) <= k:
            contingency[int(r) - 1, int(t) - 1] += 1

    if int(contingency.sum()) == 0:
        raise RuntimeError("All overlap contingency counts are zero; cannot align labels")

    max_val = int(contingency.max())
    cost = max_val - contingency
    row_ind, col_ind = linear_sum_assignment(cost)

    mapping: dict[int, int] = {}
    used_ref: set[int] = set()
    for row, col in zip(row_ind, col_ind, strict=True):
        src = int(col + 1)
        dst = int(row + 1)
        mapping[src] = dst
        used_ref.add(dst)

    free_ref = [idx for idx in range(1, k + 1) if idx not in used_ref]
    for src in range(1, k + 1):
        if src not in mapping:
            if free_ref:
                mapping[src] = int(free_ref.pop(0))
            else:
                mapping[src] = int(src)
    return mapping


def remap_labels(labels: np.ndarray, mapping: dict[int, int]) -> np.ndarray:
    out = np.zeros_like(labels, dtype=np.int32)
    for src, dst in mapping.items():
        out[labels == int(src)] = int(dst)
    return out


def probability_rows_reordered(
    *,
    probability_rows: np.ndarray,
    mapping: dict[int, int],
    k: int,
) -> np.ndarray:
    if probability_rows.ndim != 2:
        raise ValueError(f"probability_rows must be 2D, got shape={probability_rows.shape}")
    n_network = int(probability_rows.shape[1])
    out = np.zeros((k, n_network), dtype=np.float64)
    for src in range(1, k + 1):
        if src > probability_rows.shape[0]:
            continue
        dst = int(mapping[src])
        out[dst - 1, :] = probability_rows[src - 1, :]
    return out


def infer_density_from_surface(surface_path: Path) -> str:
    match = re.search(r"_den-([^_]+)_", surface_path.name)
    if not match:
        raise RuntimeError(f"Could not infer density from surface filename: {surface_path}")
    return str(match.group(1))


def infer_spec_path(subject: str, density: str, left_surface: Path) -> Path:
    surf_dir = left_surface.parent
    spec_path = surf_dir / f"sub-{subject}_den-{density}_label-hipp_surfaces.spec"
    if not spec_path.exists():
        raise FileNotFoundError(
            f"Missing canonical hippocampal spec file: {spec_path} "
            f"(subject={subject}, density={density})"
        )
    return spec_path


def subject_inputs_for_combo(
    *,
    out_root: Path,
    branch: str,
    atlas: str,
    subject: str,
    smoothing: str,
    hemi: str,
) -> SubjectInputs:
    subject_root = out_root / branch / atlas / f"sub-{subject}"
    clustering_root = subject_root / "clustering" / smoothing / f"hemi_{hemi}"
    per_k = clustering_root / "per_k_summary.tsv"
    final_summary = subject_root / "final_selection_summary.json"

    if not per_k.exists():
        raise FileNotFoundError(f"Missing per_k_summary.tsv: {per_k}")
    if not final_summary.exists():
        raise FileNotFoundError(f"Missing final_selection_summary.json: {final_summary}")

    shared_root = out_root / "_shared" / f"sub-{subject}" / "surface"
    ts_path = shared_root / smoothing / f"sub-{subject}_hemi-{hemi}_timeseries.npy"
    valid_mask_path = shared_root / "tsnr_gate" / f"sub-{subject}_hemi-{hemi}_valid_mask.npy"

    if not ts_path.exists():
        raise FileNotFoundError(f"Missing hippocampal timeseries: {ts_path}")
    if not valid_mask_path.exists():
        raise FileNotFoundError(f"Missing hippocampal valid mask: {valid_mask_path}")

    return SubjectInputs(
        subject=subject,
        per_k_summary_tsv=per_k,
        final_selection_summary_json=final_summary,
        timeseries_npy=ts_path,
        valid_mask_npy=valid_mask_path,
    )


def gather_subject_render_assets(
    *,
    summary_path: Path,
    subject: str,
    smoothing: str,
) -> RenderAssets:
    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    try:
        final_assets = payload["per_smooth"][smoothing]["final_assets"]
    except KeyError as exc:
        raise KeyError(f"Missing final_assets for smooth='{smoothing}' in {summary_path}") from exc

    left_surface = Path(str(final_assets["left_surface"]))
    right_surface = Path(str(final_assets["right_surface"]))
    if not left_surface.exists() or not right_surface.exists():
        raise FileNotFoundError(
            f"Missing render surfaces for subject={subject}, smoothing={smoothing}: "
            f"left={left_surface.exists()} right={right_surface.exists()}"
        )
    density = infer_density_from_surface(left_surface)
    spec_path = infer_spec_path(subject=subject, density=density, left_surface=left_surface)

    return RenderAssets(left_surface=left_surface, right_surface=right_surface, spec_path=spec_path)


def one_hot_from_labels(labels: np.ndarray, k: int) -> np.ndarray:
    out = np.zeros((k, labels.size), dtype=np.float32)
    for idx in range(1, k + 1):
        out[idx - 1, labels == idx] = 1.0
    return out


def zscore_time_axis(mat: np.ndarray) -> np.ndarray:
    # mat: T x V
    mat = np.asarray(mat, dtype=np.float32)
    mat = np.nan_to_num(mat, nan=0.0, posinf=0.0, neginf=0.0)
    mean = np.mean(mat, axis=0, keepdims=True)
    std = np.std(mat, axis=0, keepdims=True)
    std_safe = np.where(std < 1e-6, 1.0, std)
    out = (mat - mean) / std_safe
    out[:, std.ravel() < 1e-6] = 0.0
    return out.astype(np.float32, copy=False)


def row_min_shift_to_prob(scores: np.ndarray, valid_mask: np.ndarray | None) -> np.ndarray:
    # scores: K x V
    probs = np.zeros_like(scores, dtype=np.float32)
    if valid_mask is None:
        valid = np.ones(scores.shape[1], dtype=bool)
    else:
        valid = valid_mask.astype(bool, copy=False)
    if not np.any(valid):
        return probs

    active = scores[:, valid].astype(np.float64, copy=False)
    active = np.nan_to_num(active, nan=0.0, posinf=0.0, neginf=0.0)
    active = active - np.min(active, axis=0, keepdims=True) + 1e-6
    denom = np.sum(active, axis=0, keepdims=True)
    active_prob = active / np.clip(denom, 1e-12, None)
    probs[:, valid] = active_prob.astype(np.float32)
    return probs


def inference_soft_map(
    *,
    prior_matrix: np.ndarray,
    timeseries_vt: np.ndarray,
    valid_mask: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    # timeseries_vt: V x T
    if timeseries_vt.ndim != 2:
        raise ValueError(f"timeseries must be 2D VxT, got shape={timeseries_vt.shape}")

    v, t = int(timeseries_vt.shape[0]), int(timeseries_vt.shape[1])
    if int(prior_matrix.shape[1]) != v:
        raise ValueError(
            f"prior vertex count mismatch: prior V={prior_matrix.shape[1]} vs subject V={v}"
        )
    if t < 2:
        raise ValueError(f"Need at least 2 timepoints, got T={t}")

    x_tv = zscore_time_axis(timeseries_vt.T.astype(np.float32, copy=False))
    prior_ts = (prior_matrix.astype(np.float32, copy=False) @ x_tv.T).astype(np.float32)

    prior_mean = np.mean(prior_ts, axis=1, keepdims=True)
    prior_std = np.std(prior_ts, axis=1, keepdims=True)
    prior_std_safe = np.where(prior_std < 1e-6, 1.0, prior_std)
    prior_z = (prior_ts - prior_mean) / prior_std_safe
    prior_z[prior_std.ravel() < 1e-6, :] = 0.0

    scores_raw = (prior_z @ x_tv) / float(t - 1)
    scores_raw = scores_raw.astype(np.float32)

    if valid_mask is not None:
        valid = valid_mask.astype(bool, copy=False)
        if valid.size != v:
            raise ValueError(f"valid_mask shape mismatch: {valid.size} vs V={v}")
        x_tv[:, ~valid] = 0.0
        scores_raw[:, ~valid] = 0.0
    else:
        valid = np.ones(v, dtype=bool)

    scores_raw = np.nan_to_num(scores_raw, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32, copy=False)
    scores_prob = row_min_shift_to_prob(scores_raw, valid)

    wta_labels = np.zeros(v, dtype=np.int32)
    confidence_margin = np.zeros(v, dtype=np.float32)
    if np.any(valid):
        active_prob = scores_prob[:, valid]
        order = np.argsort(active_prob, axis=0)
        best_idx = order[-1, :]
        second_idx = order[-2, :] if active_prob.shape[0] > 1 else order[-1, :]
        best_prob = active_prob[best_idx, np.arange(active_prob.shape[1])]
        second_prob = active_prob[second_idx, np.arange(active_prob.shape[1])]
        wta_labels[valid] = best_idx.astype(np.int32) + 1
        confidence_margin[valid] = (best_prob - second_prob).astype(np.float32)

    return scores_raw, scores_prob, wta_labels, confidence_margin


def collect_preflight_missing_paths(
    *,
    combo: Combo,
    subjects: list[str],
    inference_subjects: list[str],
    out_root: Path,
) -> list[Path]:
    missing: list[Path] = []

    for subject in subjects:
        subject_root = out_root / combo.branch / combo.atlas / f"sub-{subject}"
        summary_path = subject_root / "final_selection_summary.json"
        if not summary_path.exists():
            missing.append(summary_path)
        for hemi in HEMIS:
            per_k = subject_root / "clustering" / combo.smoothing / f"hemi_{hemi}" / "per_k_summary.tsv"
            if not per_k.exists():
                missing.append(per_k)

    for subject in inference_subjects:
        subject_root = out_root / combo.branch / combo.atlas / f"sub-{subject}"
        summary_path = subject_root / "final_selection_summary.json"
        if not summary_path.exists():
            missing.append(summary_path)
        for hemi in HEMIS:
            ts = (
                out_root
                / "_shared"
                / f"sub-{subject}"
                / "surface"
                / combo.smoothing
                / f"sub-{subject}_hemi-{hemi}_timeseries.npy"
            )
            mask = (
                out_root
                / "_shared"
                / f"sub-{subject}"
                / "surface"
                / "tsnr_gate"
                / f"sub-{subject}_hemi-{hemi}_valid_mask.npy"
            )
            if not ts.exists():
                missing.append(ts)
            if not mask.exists():
                missing.append(mask)
    return missing


def process_combo(
    *,
    combo: Combo,
    subjects: list[str],
    inference_subjects: list[str],
    out_root: Path,
    group_out_root: Path,
    scene: Path,
    views: list[str],
    layout: str,
    min_parcel_pass_rate: float,
    network_colors: dict[str, tuple[int, int, int, int]],
) -> dict[str, object]:
    combo_root = group_out_root / combo.branch / combo.atlas / combo.smoothing
    combo_root.mkdir(parents=True, exist_ok=True)

    group_k_rows: list[dict[str, object]] = []
    group_selection_by_hemi: dict[str, dict[str, object]] = {}
    group_prior_pickles: dict[str, str] = {}
    prior_meta_by_hemi: dict[str, dict[str, object]] = {}
    key_to_name_by_hemi: dict[str, dict[int, str]] = {}
    wta_labels_by_hemi: dict[str, np.ndarray] = {}

    # choose template subject for group rendering from training list
    template_subject = subjects[0]
    template_summary_path = (
        out_root
        / combo.branch
        / combo.atlas
        / f"sub-{template_subject}"
        / "final_selection_summary.json"
    )
    template_assets = gather_subject_render_assets(
        summary_path=template_summary_path,
        subject=template_subject,
        smoothing=combo.smoothing,
    )
    density = infer_density_from_surface(template_assets.left_surface)

    for hemi in HEMIS:
        per_subject_inputs = {
            subject: subject_inputs_for_combo(
                out_root=out_root,
                branch=combo.branch,
                atlas=combo.atlas,
                subject=subject,
                smoothing=combo.smoothing,
                hemi=hemi,
            )
            for subject in subjects
        }
        per_k_tsv = {sub: item.per_k_summary_tsv for sub, item in per_subject_inputs.items()}

        k_final, selection_log, agg_rows = choose_group_k(
            subjects=subjects,
            per_subject_tsv=per_k_tsv,
            min_parcel_pass_rate=min_parcel_pass_rate,
        )

        # enrich rows with combo keys
        for row in agg_rows:
            row["branch"] = combo.branch
            row["atlas"] = combo.atlas
            row["smoothing"] = combo.smoothing
            row["hemi"] = hemi
        group_k_rows.extend(agg_rows)

        # load labels and annotations, then align to first subject as reference
        labels_by_subject: dict[str, np.ndarray] = {}
        probs_by_subject: dict[str, np.ndarray] = {}
        profile_networks: list[str] | None = None

        for subject in subjects:
            cluster_root = (
                out_root
                / combo.branch
                / combo.atlas
                / f"sub-{subject}"
                / "clustering"
                / combo.smoothing
                / f"hemi_{hemi}"
                / f"k_{k_final}"
            )
            labels_path = cluster_root / "cluster_labels_full.npy"
            annotation_path = cluster_root / "cluster_annotation.json"
            if not labels_path.exists():
                raise FileNotFoundError(f"Missing cluster labels for K={k_final}: {labels_path}")
            if not annotation_path.exists():
                raise FileNotFoundError(f"Missing cluster annotation for K={k_final}: {annotation_path}")

            labels = np.load(labels_path).astype(np.int32)
            labels_by_subject[subject] = labels

            annotation = json.loads(annotation_path.read_text(encoding="utf-8"))
            prob_rows = np.asarray(annotation.get("probability_rows", []), dtype=np.float64)
            if prob_rows.shape[0] != k_final:
                raise RuntimeError(
                    f"Unexpected probability_rows row count in {annotation_path}: "
                    f"{prob_rows.shape[0]} != K={k_final}"
                )
            probs_by_subject[subject] = prob_rows

            networks = [str(x) for x in annotation.get("networks", [])]
            if not networks:
                raise RuntimeError(f"Missing networks in {annotation_path}")
            if profile_networks is None:
                profile_networks = networks
            elif profile_networks != networks:
                raise RuntimeError(
                    f"Inconsistent profile_networks across subjects for {combo}/{combo.smoothing}/{hemi}"
                )

        assert profile_networks is not None

        ref_subject = subjects[0]
        ref_labels = labels_by_subject[ref_subject]
        aligned_onehot: list[np.ndarray] = []
        aligned_probs: list[np.ndarray] = []
        mappings: dict[str, dict[str, int]] = {}

        for subject in subjects:
            labels = labels_by_subject[subject]
            if labels.shape != ref_labels.shape:
                raise RuntimeError(
                    f"Label shape mismatch for {subject}: {labels.shape} vs reference {ref_labels.shape}"
                )
            if subject == ref_subject:
                mapping = {idx: idx for idx in range(1, k_final + 1)}
                remapped = labels
            else:
                mapping = build_hungarian_mapping(ref_labels, labels, k_final)
                remapped = remap_labels(labels, mapping)

            mappings[subject] = {str(k): int(v) for k, v in mapping.items()}
            aligned_onehot.append(one_hot_from_labels(remapped, k_final))
            aligned_probs.append(
                probability_rows_reordered(
                    probability_rows=probs_by_subject[subject],
                    mapping=mapping,
                    k=k_final,
                )
            )

        prior_matrix = np.mean(np.stack(aligned_onehot, axis=0), axis=0).astype(np.float32)
        cluster_network_probs = np.mean(np.stack(aligned_probs, axis=0), axis=0).astype(np.float32)

        dominant_idx = np.argmax(cluster_network_probs, axis=1)
        dominant_network = [profile_networks[int(idx)] for idx in dominant_idx]
        valid_vertex_mask = (prior_matrix.sum(axis=0) > 0).astype(np.uint8)

        hemi_key_to_name: dict[int, str] = {
            int(cluster_id): f"{hemi}C{cluster_id}_{dominant_network[cluster_id - 1]}"
            for cluster_id in range(1, k_final + 1)
        }
        key_to_name_by_hemi[hemi] = hemi_key_to_name

        wta = np.zeros(prior_matrix.shape[1], dtype=np.int32)
        nz = prior_matrix.sum(axis=0) > 0
        wta[nz] = np.argmax(prior_matrix[:, nz], axis=0).astype(np.int32) + 1
        wta_labels_by_hemi[hemi] = wta

        prior_payload: dict[str, object] = {
            "prior_matrix": prior_matrix,
            "k_final": int(k_final),
            "subjects": subjects,
            "branch": combo.branch,
            "atlas": combo.atlas,
            "smoothing": combo.smoothing,
            "hemi": hemi,
            "profile_networks": profile_networks,
            "cluster_network_probs": cluster_network_probs,
            "cluster_dominant_network": dominant_network,
            "valid_vertex_mask": valid_vertex_mask,
            "selection_log": selection_log,
            "label_mappings_to_reference": mappings,
            "version": VERSION,
        }

        prior_path = combo_root / "priors" / f"group_prior_{combo.branch}_{combo.atlas}_{combo.smoothing}_hemi-{hemi}.pickle"
        prior_path.parent.mkdir(parents=True, exist_ok=True)
        prior_path.write_bytes(pickle.dumps(prior_payload))

        group_prior_pickles[hemi] = str(prior_path)
        selection_with_hemi = dict(selection_log)
        selection_with_hemi["hemi"] = hemi
        selection_with_hemi["k_final"] = int(k_final)
        group_selection_by_hemi[hemi] = selection_with_hemi

        prior_meta_by_hemi[hemi] = {
            "k_final": int(k_final),
            "prior_shape": [int(prior_matrix.shape[0]), int(prior_matrix.shape[1])],
            "n_valid_vertices": int(valid_vertex_mask.sum()),
            "profile_networks": profile_networks,
            "cluster_dominant_network": dominant_network,
            "prior_pickle": str(prior_path),
        }

    # write group-K table + json
    group_k_tsv = combo_root / "group_k_selection.tsv"
    write_tsv(
        group_k_tsv,
        group_k_rows,
        [
            "branch",
            "atlas",
            "smoothing",
            "hemi",
            "k",
            "n_subjects",
            "group_instability_mean",
            "group_instability_se",
            "group_null_corrected_score_mean",
            "group_silhouette_mean",
            "min_parcel_pass_rate",
            "connectivity_pass_rate",
            "local_min",
            "within_1se_best",
            "parcel_rate_ok",
        ],
    )
    group_k_json = combo_root / "group_k_selection.json"
    save_json(
        group_k_json,
        {
            "branch": combo.branch,
            "atlas": combo.atlas,
            "smoothing": combo.smoothing,
            "rule": "mean-instability-1se",
            "min_parcel_pass_rate_threshold": float(min_parcel_pass_rate),
            "hemi_selection": group_selection_by_hemi,
            "rows": group_k_rows,
            "version": VERSION,
        },
    )

    # render group template using template subject geometry
    group_assets = save_combined_label_assets(
        subject=template_subject,
        density=density,
        left_labels=wta_labels_by_hemi["L"],
        right_labels=wta_labels_by_hemi["R"],
        output_dir=combo_root / "template" / "workbench_assets",
        left_surface=template_assets.left_surface,
        right_surface=template_assets.right_surface,
        left_key_to_name=key_to_name_by_hemi["L"],
        right_key_to_name=key_to_name_by_hemi["R"],
        stem=f"group_prior_{combo.branch}_{combo.atlas}_{combo.smoothing}",
        network_colors=network_colors,
    )
    group_render = render_locked_grid_png(
        subject=template_subject,
        scene=scene,
        outdir=combo_root / "template" / "renders",
        name=f"group_prior_{combo.branch}_{combo.atlas}_{combo.smoothing}",
        title=(
            f"Group prior {combo.branch} {combo.atlas} {combo.smoothing} "
            f"(L K={group_selection_by_hemi['L']['k_final']}, R K={group_selection_by_hemi['R']['k_final']})"
        ),
        left_labels=Path(group_assets["left_label"]),
        right_labels=Path(group_assets["right_label"]),
        left_surface=template_assets.left_surface,
        right_surface=template_assets.right_surface,
        spec_path=template_assets.spec_path,
        layout=layout,
        views=views,
    )

    # subject inference + rendering
    subject_manifest: list[dict[str, object]] = []
    for subject in inference_subjects:
        summary_path = out_root / combo.branch / combo.atlas / f"sub-{subject}" / "final_selection_summary.json"
        render_assets = gather_subject_render_assets(
            summary_path=summary_path,
            subject=subject,
            smoothing=combo.smoothing,
        )

        soft_pickles_by_hemi: dict[str, str] = {}
        hemi_wta: dict[str, np.ndarray] = {}
        hemi_key_to_name = key_to_name_by_hemi

        for hemi in HEMIS:
            prior_path = Path(group_prior_pickles[hemi])
            prior_obj = pickle.loads(prior_path.read_bytes())
            prior_matrix = np.asarray(prior_obj["prior_matrix"], dtype=np.float32)
            valid_prior_mask = np.asarray(prior_obj["valid_vertex_mask"]).astype(bool)
            k_final = int(prior_obj["k_final"])

            ts_path = (
                out_root
                / "_shared"
                / f"sub-{subject}"
                / "surface"
                / combo.smoothing
                / f"sub-{subject}_hemi-{hemi}_timeseries.npy"
            )
            if not ts_path.exists():
                raise FileNotFoundError(f"Missing inference timeseries: {ts_path}")
            timeseries = np.load(ts_path).astype(np.float32)

            valid_mask_path = (
                out_root
                / "_shared"
                / f"sub-{subject}"
                / "surface"
                / "tsnr_gate"
                / f"sub-{subject}_hemi-{hemi}_valid_mask.npy"
            )
            if not valid_mask_path.exists():
                raise FileNotFoundError(f"Missing inference valid mask: {valid_mask_path}")
            valid_mask = np.load(valid_mask_path).astype(bool)
            if valid_mask.shape[0] != prior_matrix.shape[1]:
                raise ValueError(
                    f"valid_mask length mismatch for sub-{subject} hemi-{hemi}: "
                    f"{valid_mask.shape[0]} vs prior V={prior_matrix.shape[1]}"
                )
            inference_valid_mask = valid_mask & valid_prior_mask

            scores_raw, scores_prob, wta_labels, confidence_margin = inference_soft_map(
                prior_matrix=prior_matrix,
                timeseries_vt=timeseries,
                valid_mask=inference_valid_mask,
            )

            inactive = ~inference_valid_mask
            scores_raw[:, inactive] = 0.0
            scores_prob[:, inactive] = 0.0
            wta_labels[inactive] = 0
            confidence_margin[inactive] = 0.0

            soft_payload: dict[str, object] = {
                "scores_raw": scores_raw,
                "scores_prob": scores_prob,
                "wta_labels": wta_labels,
                "confidence_margin": confidence_margin,
                "k_final": int(k_final),
                "hemi": hemi,
                "smoothing": combo.smoothing,
                "branch": combo.branch,
                "atlas": combo.atlas,
                "subject": subject,
                "prior_pickle": str(prior_path),
                "valid_vertex_mask": inference_valid_mask.astype(np.uint8),
                "version": VERSION,
            }
            soft_path = (
                combo_root
                / "individual_soft_maps"
                / f"sub-{subject}"
                / f"sub-{subject}_{combo.branch}_{combo.atlas}_{combo.smoothing}_hemi-{hemi}_soft_functional_map.pickle"
            )
            soft_path.parent.mkdir(parents=True, exist_ok=True)
            soft_path.write_bytes(pickle.dumps(soft_payload))
            soft_pickles_by_hemi[hemi] = str(soft_path)
            hemi_wta[hemi] = wta_labels.astype(np.int32)

        subject_assets = save_combined_label_assets(
            subject=subject,
            density=density,
            left_labels=hemi_wta["L"],
            right_labels=hemi_wta["R"],
            output_dir=combo_root / "individual_soft_maps" / f"sub-{subject}" / "workbench_assets",
            left_surface=render_assets.left_surface,
            right_surface=render_assets.right_surface,
            left_key_to_name=hemi_key_to_name["L"],
            right_key_to_name=hemi_key_to_name["R"],
            stem=f"soft_map_{combo.branch}_{combo.atlas}_{combo.smoothing}",
            network_colors=network_colors,
        )
        render_payload = render_locked_grid_png(
            subject=subject,
            scene=scene,
            outdir=combo_root / "individual_soft_maps" / f"sub-{subject}" / "renders",
            name=f"soft_map_{combo.branch}_{combo.atlas}_{combo.smoothing}",
            title=(
                f"Soft map {combo.branch} {combo.atlas} {combo.smoothing} sub-{subject} "
                f"(L K={group_selection_by_hemi['L']['k_final']}, R K={group_selection_by_hemi['R']['k_final']})"
            ),
            left_labels=Path(subject_assets["left_label"]),
            right_labels=Path(subject_assets["right_label"]),
            left_surface=render_assets.left_surface,
            right_surface=render_assets.right_surface,
            spec_path=render_assets.spec_path,
            layout=layout,
            views=views,
        )

        subject_manifest.append(
            {
                "subject": subject,
                "soft_pickles": soft_pickles_by_hemi,
                "workbench_assets": subject_assets,
                "renders": render_payload,
            }
        )

    group_prior_manifest = {
        "branch": combo.branch,
        "atlas": combo.atlas,
        "smoothing": combo.smoothing,
        "group_k_selection_json": str(group_k_json),
        "group_k_selection_tsv": str(group_k_tsv),
        "hemi_priors": prior_meta_by_hemi,
        "template_subject": template_subject,
        "template_workbench_assets": group_assets,
        "template_renders": group_render,
        "version": VERSION,
    }
    individual_manifest = {
        "branch": combo.branch,
        "atlas": combo.atlas,
        "smoothing": combo.smoothing,
        "subjects": subject_manifest,
        "version": VERSION,
    }

    save_json(combo_root / "group_prior_manifest.json", group_prior_manifest)
    save_json(combo_root / "individual_softmap_manifest.json", individual_manifest)

    return {
        "combo": {
            "branch": combo.branch,
            "atlas": combo.atlas,
            "smoothing": combo.smoothing,
        },
        "group_prior_manifest": str((combo_root / "group_prior_manifest.json").resolve()),
        "individual_softmap_manifest": str((combo_root / "individual_softmap_manifest.json").resolve()),
    }


def main() -> int:
    args = parse_args()

    out_root = Path(args.out_root).resolve()
    group_out_root = Path(args.group_out_root).resolve()
    scene = Path(args.scene).resolve()
    views = [v.strip() for v in str(args.views).split(",") if v.strip()]
    layout = str(args.layout)

    if not out_root.exists():
        raise FileNotFoundError(f"Input out-root does not exist: {out_root}")
    if not scene.exists():
        raise FileNotFoundError(f"Scene file does not exist: {scene}")
    if not (0.0 < float(args.min_parcel_pass_rate) <= 1.0):
        raise ValueError(f"--min-parcel-pass-rate must be in (0,1], got {args.min_parcel_pass_rate}")
    if str(args.layout) != "1x2":
        raise ValueError(
            "This pipeline currently supports only --layout 1x2 with "
            "wb_locked_native_view_lateral_medial.scene"
        )

    network_colors = load_network_colors(NETWORK_STYLE_JSON)

    subjects = [str(x) for x in args.subjects]
    inference_subjects = [str(x) for x in (args.inference_subjects if args.inference_subjects else args.subjects)]

    combo_summaries: list[dict[str, object]] = []
    all_missing: list[Path] = []

    combos = [
        Combo(branch=str(branch), atlas=str(atlas), smoothing=str(smoothing))
        for branch in args.branches
        for atlas in args.atlases
        for smoothing in args.smoothings
    ]
    for combo in combos:
        all_missing.extend(
            collect_preflight_missing_paths(
                combo=combo,
                subjects=subjects,
                inference_subjects=inference_subjects,
                out_root=out_root,
            )
        )
    if all_missing:
        uniq_missing = sorted({path.resolve() for path in all_missing}, key=str)
        missing_text = "\n".join(str(path) for path in uniq_missing)
        raise FileNotFoundError(
            f"Missing required inputs for group-prior fast-PFM run ({len(uniq_missing)} paths):\n{missing_text}"
        )

    for combo in combos:
        combo_summary = process_combo(
            combo=combo,
            subjects=subjects,
            inference_subjects=inference_subjects,
            out_root=out_root,
            group_out_root=group_out_root,
            scene=scene,
            views=views,
            layout=layout,
            min_parcel_pass_rate=float(args.min_parcel_pass_rate),
            network_colors=network_colors,
        )
        combo_summaries.append(combo_summary)

    run_manifest = {
        "branches": [str(x) for x in args.branches],
        "atlases": [str(x) for x in args.atlases],
        "subjects": subjects,
        "inference_subjects": inference_subjects,
        "smoothings": [str(x) for x in args.smoothings],
        "group_k_rule": str(args.group_k_rule),
        "min_parcel_pass_rate": float(args.min_parcel_pass_rate),
        "out_root": str(out_root),
        "group_out_root": str(group_out_root),
        "scene": str(scene),
        "views": views,
        "layout": layout,
        "combos": combo_summaries,
        "version": VERSION,
    }
    save_json(group_out_root / "run_manifest.json", run_manifest)
    print(json.dumps(run_manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
