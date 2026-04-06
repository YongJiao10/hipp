#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import nibabel as nib
import numpy as np
from nibabel.gifti import GiftiDataArray, GiftiImage, GiftiLabel, GiftiLabelTable
from scipy import sparse
from scipy.sparse import csgraph
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import (
    adjusted_rand_score,
    calinski_harabasz_score,
    davies_bouldin_score,
    silhouette_score,
)

REPO_ROOT = Path(__file__).resolve().parents[3]
LEGACY_SOURCE_ROOT = REPO_ROOT.parent / "HippoMaps"


def default_source_root() -> Path:
    local_input = REPO_ROOT / "data" / "hippunfold_input"
    legacy_input = LEGACY_SOURCE_ROOT / "data" / "hippunfold_input"
    if local_input.exists() or not LEGACY_SOURCE_ROOT.exists() or not legacy_input.exists():
        return REPO_ROOT
    return LEGACY_SOURCE_ROOT


SOURCE_ROOT = default_source_root()


def resolve_local_or_legacy_path(relative_path: str) -> Path:
    local_path = REPO_ROOT / relative_path
    if local_path.exists():
        return local_path
    legacy_path = LEGACY_SOURCE_ROOT / relative_path
    return legacy_path


COMMON_DIR = REPO_ROOT / "scripts" / "common"
if str(COMMON_DIR) not in sys.path:
    sys.path.insert(0, str(COMMON_DIR))

from compute_fc_gradients import build_sparse_affinity, corrcoef_rows, diffusion_map_embedding


WB_COMMAND = str((REPO_ROOT / "scripts" / "wb_command").resolve())
PYTHON_EXE = sys.executable or "/opt/miniconda3/envs/py314/bin/python"
NETWORK_STYLE_JSON = resolve_local_or_legacy_path("config/hipp_network_style.json")
CROSS_ATLAS_NETWORK_MERGE_JSON = resolve_local_or_legacy_path("config/cross_atlas_network_merge.json")
DEFAULT_SCENE = resolve_local_or_legacy_path("config/wb_locked_native_view_lateral_medial.scene")
EVAL_K = list(range(3, 9))
SMOOTH_ORDER = ["2mm", "4mm"]
HEMIS = ["L", "R"]
BRANCHES = [
    "network-gradient",
    "network-prob-cluster",
    "network-prob-cluster-nonneg",
    "network-prob-soft",
    "network-prob-soft-nonneg",
    "network-wta",
]
ATLAS_CONFIG = {
    "lynch2024": {
        "label_prefix": "PFM_Lynch2024priors.components",
        "display_name": "Lynch2024",
    },
    "hermosillo2024": {
        "label_prefix": "PFM_Hermosillo2024priors.components",
        "display_name": "Hermosillo2024",
    },
    "kong2019": {
        "label_prefix": "PFM_Kong2019priors.components",
        "display_name": "Kong2019",
    },
}


def is_soft_branch(branch_slug: str) -> bool:
    return branch_slug in {"network-prob-soft", "network-prob-soft-nonneg"}


def is_wta_branch(branch_slug: str) -> bool:
    return branch_slug == "network-wta"


def is_gradient_branch(branch_slug: str) -> bool:
    return branch_slug == "network-gradient"


def is_probability_cluster_branch(branch_slug: str) -> bool:
    return branch_slug in {"network-prob-cluster", "network-prob-cluster-nonneg"}


def uses_nonnegative_probabilities(branch_slug: str) -> bool:
    return branch_slug in {"network-prob-cluster-nonneg", "network-prob-soft-nonneg"}


def run(cmd: list[str]) -> None:
    proc = subprocess.run(cmd, text=True, capture_output=True)
    if proc.returncode != 0:
        raise RuntimeError(
            f"Command failed ({proc.returncode}): {' '.join(cmd)}\nSTDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}"
        )


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def file_stamp(path: Path) -> dict[str, object]:
    stat = path.stat()
    return {
        "path": str(path),
        "size": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
    }


def collect_input_stamps(paths: list[Path]) -> list[dict[str, object]]:
    unique = sorted({str(path.resolve()) for path in paths})
    return [file_stamp(Path(path)) for path in unique]


def stage_manifest_path(stage_dir: Path) -> Path:
    return stage_dir / "stage_manifest.json"


def stage_is_up_to_date(
    *,
    stage_dir: Path,
    resume_mode: str,
    stage_name: str,
    params: dict[str, object],
    inputs: list[Path],
    outputs: list[Path],
) -> bool:
    if resume_mode == "force":
        return False
    manifest_path = stage_manifest_path(stage_dir)
    if not manifest_path.exists():
        return False
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    if str(payload.get("status", "")) != "done":
        return False
    if str(payload.get("stage", "")) != stage_name:
        return False
    if payload.get("params", {}) != params:
        return False
    try:
        expected_inputs = collect_input_stamps(inputs)
    except FileNotFoundError:
        return False
    if payload.get("inputs", []) != expected_inputs:
        return False
    for output in outputs:
        if not output.exists():
            return False
    return True


def write_stage_manifest(
    *,
    stage_dir: Path,
    stage_name: str,
    params: dict[str, object],
    inputs: list[Path],
    outputs: list[Path],
) -> None:
    stage_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "stage": stage_name,
        "status": "done",
        "timestamp_utc": utc_now_iso(),
        "params": params,
        "inputs": collect_input_stamps(inputs),
        "outputs": [str(path.resolve()) for path in outputs],
    }
    stage_manifest_path(stage_dir).write_text(json.dumps(payload, indent=2), encoding="utf-8")


def write_surface_store_pointer(
    *,
    pointer_dir: Path,
    shared_surface_store_dir: Path,
    subject: str,
    two_mm_left_path: Path,
    two_mm_right_path: Path,
    fwhm_left_path: Path,
    fwhm_right_path: Path,
) -> None:
    pointer_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "kind": "shared_surface_store_pointer",
        "timestamp_utc": utc_now_iso(),
        "subject": subject,
        "shared_surface_store_dir": str(shared_surface_store_dir.resolve()),
        "timeseries": {
            "2mm": {
                "left": str(two_mm_left_path.resolve()),
                "right": str(two_mm_right_path.resolve()),
            },
            "4mm": {
                "left": str(fwhm_left_path.resolve()),
                "right": str(fwhm_right_path.resolve()),
            },
        },
    }
    (pointer_dir / "shared_surface_store.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")


def write_reference_store_pointer(
    *,
    pointer_dir: Path,
    shared_reference_store_dir: Path,
    subject: str,
    atlas_slug: str,
    reference_summary_path: Path,
    canonical_network_table_path: Path,
    canonical_network_timeseries_path: Path,
) -> None:
    pointer_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "kind": "shared_reference_store_pointer",
        "timestamp_utc": utc_now_iso(),
        "subject": subject,
        "atlas_slug": atlas_slug,
        "shared_reference_store_dir": str(shared_reference_store_dir.resolve()),
        "artifacts": {
            "reference_summary": str(reference_summary_path.resolve()),
            "cortex_canonical_networks_tsv": str(canonical_network_table_path.resolve()),
            "cortex_canonical_network_timeseries_npy": str(canonical_network_timeseries_path.resolve()),
        },
    }
    (pointer_dir / "shared_reference_store.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")


def apply_retain_level(out_root: Path, retain_level: str) -> None:
    if retain_level in {"feature", "all"}:
        return
    if retain_level == "render":
        archive_names = {"fc", "features"}
    elif retain_level == "label":
        archive_names = {"fc", "features", "clustering", "soft_outputs"}
    else:
        raise ValueError(f"Unsupported retain_level: {retain_level}")

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    outputs_root = (REPO_ROOT / "outputs" / "hipp_functional_parcellation_network").resolve()
    out_root_resolved = out_root.resolve()
    try:
        out_rel = out_root_resolved.relative_to(outputs_root)
    except ValueError:
        out_rel = Path(out_root_resolved.name)
    archive_root = REPO_ROOT / "_archive" / "hipp_functional_parcellation_network" / "retain" / stamp / out_rel / retain_level
    manifest: list[dict[str, str]] = []
    for child in sorted(out_root.iterdir()):
        if child.name.startswith("."):
            continue
        if child.name in {
            "_archive_retain",
            "_archive",
            "reference",
            "workbench_assets",
            "hipp_functional_parcellation_network_overview.png",
            "k_selection_curves.png",
            "network_probability_heatmaps.png",
            "final_selection_core.json",
            "final_selection_summary.json",
            "summary_manifest.json",
            "summary",
            "overview_probability_summary.json",
        }:
            continue
        if child.name not in archive_names:
            continue
        dst = archive_root / child.name
        dst.parent.mkdir(parents=True, exist_ok=True)
        if dst.exists():
            raise RuntimeError(f"Retain archive destination already exists: {dst}")
        child.rename(dst)
        manifest.append({"src": str(child), "dst": str(dst)})
    if manifest:
        archive_root.mkdir(parents=True, exist_ok=True)
        (archive_root / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def load_metric_array(path: Path, expected_n_vertices: int | None = None) -> np.ndarray:
    metric_img = nib.load(str(path))
    metric = np.asarray(metric_img.agg_data(), dtype=np.float32)
    if metric.ndim == 1:
        metric = metric[:, None]
    if expected_n_vertices is not None and metric.shape[0] != expected_n_vertices and metric.shape[1] == expected_n_vertices:
        metric = metric.T
    coords = None
    try:
        coords = np.asarray(metric_img.agg_data("pointset"))
    except Exception:
        coords = None
    if coords is not None and metric.shape[0] != coords.shape[0] and metric.shape[1] == coords.shape[0]:
        metric = metric.T
    return metric.astype(np.float32, copy=False)


def load_surface(path: Path) -> tuple[np.ndarray, np.ndarray]:
    img = nib.load(str(path))
    coords = np.asarray(img.agg_data("pointset"), dtype=np.float32)
    faces = np.asarray(img.agg_data("triangle"), dtype=np.int32)
    return coords, faces


def build_surface_adjacency(faces: np.ndarray, n_vertices: int) -> sparse.csr_matrix:
    row_parts = []
    col_parts = []
    for tri in faces:
        a, b, c = (int(tri[0]), int(tri[1]), int(tri[2]))
        row_parts.extend([a, b, b, c, c, a])
        col_parts.extend([b, a, c, b, a, c])
    data = np.ones(len(row_parts), dtype=np.float32)
    graph = sparse.csr_matrix((data, (row_parts, col_parts)), shape=(n_vertices, n_vertices), dtype=np.float32)
    graph = graph.maximum(graph.T)
    graph.setdiag(0)
    graph.eliminate_zeros()
    return graph


def zscore_columns(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float32, copy=False)
    mean = np.nanmean(x, axis=0, keepdims=True)
    std = np.nanstd(x, axis=0, keepdims=True)
    out = (x - mean) / np.clip(std, 1e-12, None)
    out[~np.isfinite(out)] = 0.0
    return out


def sanitize_timeseries(metric: np.ndarray) -> np.ndarray:
    metric = metric.astype(np.float32, copy=True)
    if metric.ndim != 2:
        raise ValueError(f"Expected 2D timeseries array, got shape {metric.shape}")
    finite = np.isfinite(metric)
    counts = finite.sum(axis=1, keepdims=True).astype(np.float32)
    sums = np.where(finite, metric, 0.0).sum(axis=1, keepdims=True, dtype=np.float32)
    row_means = sums / np.clip(counts, 1.0, None)
    mask = ~np.isfinite(metric)
    if np.any(mask):
        metric[mask] = np.broadcast_to(row_means, metric.shape)[mask]
    metric[~np.isfinite(metric)] = 0.0
    return metric


def connected_component_count(labels: np.ndarray, connectivity: sparse.csr_matrix) -> tuple[int, dict[int, int]]:
    per_cluster: dict[int, int] = {}
    total = 0
    for key in sorted(int(x) for x in np.unique(labels)):
        mask = labels == key
        subgraph = connectivity[mask][:, mask]
        n_comp, _ = csgraph.connected_components(subgraph, directed=False, return_labels=True)
        per_cluster[key] = int(n_comp)
        total += int(n_comp)
    return total, per_cluster


def compute_silhouette(features: np.ndarray, labels: np.ndarray) -> float:
    sample_size = min(4096, int(features.shape[0]))
    kwargs = {"sample_size": sample_size, "random_state": 0} if sample_size < int(features.shape[0]) else {}
    return float(silhouette_score(features, labels, **kwargs))


def compute_wcss(features: np.ndarray, labels: np.ndarray) -> float:
    total = 0.0
    for key in sorted(int(x) for x in np.unique(labels)):
        cluster = features[labels == key]
        center = np.mean(cluster, axis=0, keepdims=True)
        total += float(np.sum((cluster - center) ** 2))
    return total


def compute_balance_entropy(labels: np.ndarray) -> float:
    counts = np.asarray([np.count_nonzero(labels == key) for key in sorted(np.unique(labels))], dtype=np.float64)
    probs = counts / np.clip(counts.sum(), 1.0, None)
    probs = probs[probs > 0]
    return float(-(probs * np.log(probs)).sum())


def reorder_cluster_labels(raw_labels: np.ndarray) -> np.ndarray:
    unique = sorted(int(x) for x in np.unique(raw_labels))
    size_pairs = sorted(
        ((int(np.count_nonzero(raw_labels == key)), key) for key in unique),
        key=lambda item: (-item[0], item[1]),
    )
    mapping = {old_key: new_idx + 1 for new_idx, (_size, old_key) in enumerate(size_pairs)}
    return np.asarray([mapping[int(key)] for key in raw_labels], dtype=np.int32)


def cluster_embedding(features: np.ndarray, connectivity: sparse.csr_matrix, k: int) -> np.ndarray:
    model = AgglomerativeClustering(n_clusters=k, linkage="ward", connectivity=connectivity)
    raw_labels = model.fit_predict(features)
    return reorder_cluster_labels(raw_labels)


def load_cross_atlas_network_merge(
    path: Path,
) -> tuple[list[str], set[str], dict[str, dict[str, str]], dict[str, tuple[int, int, int, int]]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    canonical_order = [str(item) for item in payload["canonical_network_order"]]
    exclude_labels = {str(item) for item in payload.get("exclude_labels", [])}
    atlas_mapping = {
        str(atlas_slug): {str(key): str(value) for key, value in atlas_spec["mapping"].items()}
        for atlas_slug, atlas_spec in payload["atlases"].items()
    }
    shared_colors = {
        str(name): tuple(int(round(float(v))) for v in rgba)
        for name, rgba in payload.get("shared_colors_rgba", {}).items()
    }
    return canonical_order, exclude_labels, atlas_mapping, shared_colors


_MERGED_NETWORK_ORDER, _EXCLUDED_MERGED_NETWORKS, _ATLAS_NETWORK_MERGE, MERGED_NETWORK_COLORS = (
    load_cross_atlas_network_merge(CROSS_ATLAS_NETWORK_MERGE_JSON)
)


def grouped_fc_to_probabilities(grouped_fc: np.ndarray, *, zero_negative: bool = False) -> np.ndarray:
    fisher = np.arctanh(np.clip(grouped_fc, -0.999999, 0.999999)).astype(np.float32)
    if zero_negative:
        fisher = np.clip(fisher, 0.0, None)
    else:
        fisher = fisher - np.nanmin(fisher, axis=1, keepdims=True)
        fisher = fisher + 1e-6
    total = np.nansum(fisher, axis=1, keepdims=True)
    out = fisher / np.clip(total, 1e-12, None)
    out[~np.isfinite(out)] = 0.0
    bad_rows = ~np.isfinite(out).all(axis=1) | (np.nansum(out, axis=1) <= 0)
    if np.any(bad_rows):
        out[bad_rows, :] = 1.0 / max(1, out.shape[1])
    return out.astype(np.float32, copy=False)


def normalize_probability_rows(probabilities: np.ndarray) -> np.ndarray:
    out = np.clip(probabilities.astype(np.float32, copy=True), 0.0, None)
    total = np.nansum(out, axis=1, keepdims=True)
    out = out / np.clip(total, 1e-12, None)
    out[~np.isfinite(out)] = 0.0
    bad_rows = ~np.isfinite(out).all(axis=1) | (np.nansum(out, axis=1) <= 0)
    if np.any(bad_rows):
        out[bad_rows, :] = 1.0 / max(1, out.shape[1])
    return out.astype(np.float32, copy=False)


def compute_long_axis_order(surface_coords: np.ndarray | None) -> np.ndarray | None:
    if surface_coords is None or surface_coords.ndim != 2 or surface_coords.shape[0] < 3:
        return None
    centered = surface_coords.astype(np.float32, copy=False) - np.mean(surface_coords, axis=0, keepdims=True)
    _u, _s, vt = np.linalg.svd(centered, full_matrices=False)
    axis_scores = centered @ vt[0].astype(np.float32)
    return np.argsort(axis_scores, kind="stable").astype(np.int32)


def smooth_probabilities_along_axis(probabilities: np.ndarray, vertex_order: np.ndarray | None) -> np.ndarray:
    if vertex_order is None or probabilities.shape[0] < 3:
        return probabilities.astype(np.float32, copy=True)
    ordered = probabilities[vertex_order, :].astype(np.float32, copy=False)
    prev_rows = np.vstack([ordered[:1, :], ordered[:-1, :]])
    next_rows = np.vstack([ordered[1:, :], ordered[-1:, :]])
    smoothed = 0.25 * prev_rows + 0.50 * ordered + 0.25 * next_rows
    out = np.empty_like(smoothed)
    out[vertex_order, :] = smoothed
    return out.astype(np.float32, copy=False)


def regularize_probability_profiles(
    probabilities: np.ndarray,
    connectivity: sparse.csr_matrix,
    *,
    long_axis_order: np.ndarray | None = None,
    n_iter: int = 3,
    mesh_mix: float = 0.35,
    axis_mix: float = 0.20,
) -> np.ndarray:
    degree = np.asarray(connectivity.sum(axis=1)).ravel().astype(np.float32)
    inv_degree = 1.0 / np.clip(degree, 1e-12, None)
    norm_graph = sparse.diags(inv_degree) @ connectivity
    out = normalize_probability_rows(probabilities)
    for _ in range(n_iter):
        axis_term = smooth_probabilities_along_axis(out, long_axis_order)
        total_mix = mesh_mix + (axis_mix if long_axis_order is not None else 0.0)
        keep_mix = max(0.0, 1.0 - total_mix)
        updated = keep_mix * out + mesh_mix * (norm_graph @ out)
        if long_axis_order is not None:
            updated = updated + axis_mix * axis_term
        out = normalize_probability_rows(updated)
    return out.astype(np.float32, copy=False)


def regularize_argmax_labels(
    probabilities: np.ndarray,
    connectivity: sparse.csr_matrix,
    *,
    long_axis_order: np.ndarray | None = None,
    min_component_fraction: float = 0.02,
    n_iter: int = 3,
) -> tuple[np.ndarray, np.ndarray]:
    smoothed = regularize_probability_profiles(
        probabilities,
        connectivity,
        long_axis_order=long_axis_order,
        n_iter=n_iter,
    )
    labels = np.argmax(smoothed, axis=1).astype(np.int32) + 1
    min_component_size = max(8, int(round(labels.size * min_component_fraction)))
    for _pass in range(4):
        changed = False
        for key in sorted(int(x) for x in np.unique(labels)):
            mask = labels == key
            if not np.any(mask):
                continue
            subgraph = connectivity[mask][:, mask]
            n_comp, comp_labels = csgraph.connected_components(subgraph, directed=False, return_labels=True)
            if n_comp <= 1:
                continue
            vertices = np.flatnonzero(mask)
            for comp_id in range(n_comp):
                comp_vertices = vertices[comp_labels == comp_id]
                if comp_vertices.size >= min_component_size:
                    continue
                neighbor_vertices = connectivity[comp_vertices].indices
                neighbor_vertices = neighbor_vertices[~np.isin(neighbor_vertices, comp_vertices)]
                if neighbor_vertices.size > 0:
                    adjacent_labels = sorted(int(x) for x in np.unique(labels[neighbor_vertices]) if int(x) != key)
                else:
                    adjacent_labels = []
                alt_scores = np.nanmean(smoothed[comp_vertices, :], axis=0)
                alt_scores[key - 1] = -np.inf
                if adjacent_labels:
                    allowed = np.full_like(alt_scores, -np.inf)
                    for label_id in adjacent_labels:
                        allowed[label_id - 1] = alt_scores[label_id - 1]
                    alt_scores = allowed
                labels[comp_vertices] = int(np.argmax(alt_scores)) + 1
                changed = True
        if not changed:
            break
    return labels.astype(np.int32), smoothed.astype(np.float32, copy=False)


def summarize_argmax_occupancy(labels: np.ndarray, n_networks: int) -> np.ndarray:
    counts = np.bincount(labels.astype(np.int32), minlength=n_networks + 1)[1:].astype(np.float32)
    total = float(counts.sum())
    if total <= 0:
        return np.full(n_networks, 1.0 / max(1, n_networks), dtype=np.float32)
    return (counts / total).astype(np.float32)


def load_network_colors(path: Path) -> dict[str, tuple[int, int, int, int]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    colors: dict[str, tuple[int, int, int, int]] = {}
    for spec in payload.values():
        name = str(spec["name"])
        rgba = tuple(int(round(float(v))) for v in spec["rgba"])
        if len(rgba) != 4:
            raise ValueError(f"Expected RGBA length 4 for network '{name}', got {rgba}")
        colors[name] = rgba
    colors.update(MERGED_NETWORK_COLORS)
    return colors


NETWORK_COLORS = load_network_colors(NETWORK_STYLE_JSON)


def extract_network_name(cluster_label: str) -> str:
    if "_" not in cluster_label:
        raise ValueError(f"Cluster label must include '_<network>' suffix, got: {cluster_label}")
    # Cluster labels follow "{hemi}C{k}_{network_name}", where network_name
    # may itself contain underscores. Split only once at the first underscore.
    return cluster_label.split("_", 1)[1]


def make_label_gifti(labels: np.ndarray, key_to_name: dict[int, str]) -> GiftiImage:
    table = GiftiLabelTable()
    unknown = GiftiLabel(key=0, red=0.0, green=0.0, blue=0.0, alpha=0.0)
    unknown.label = "???"
    table.labels.append(unknown)
    for key in sorted(key_to_name):
        cluster_name = key_to_name[key]
        network_name = extract_network_name(cluster_name)
        if network_name not in NETWORK_COLORS:
            raise KeyError(f"Missing network color for '{network_name}' in {NETWORK_STYLE_JSON}")
        rgba = NETWORK_COLORS[network_name]
        label = GiftiLabel(
            key=int(key),
            red=float(rgba[0]) / 255.0,
            green=float(rgba[1]) / 255.0,
            blue=float(rgba[2]) / 255.0,
            alpha=float(rgba[3]) / 255.0,
        )
        label.label = key_to_name[key]
        table.labels.append(label)
    arr = GiftiDataArray(data=labels.astype(np.int32), intent="NIFTI_INTENT_LABEL", datatype="NIFTI_TYPE_INT32")
    return GiftiImage(darrays=[arr], labeltable=table)


def save_combined_label_assets(
    *,
    subject: str,
    left_labels: np.ndarray,
    right_labels: np.ndarray,
    output_dir: Path,
    left_surface: Path,
    right_surface: Path,
    left_key_to_name: dict[int, str],
    right_key_to_name: dict[int, str],
    stem: str,
) -> dict[str, str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    right_offset = max(left_key_to_name.keys(), default=0)
    right_labels_shifted = right_labels.astype(np.int32, copy=True)
    right_labels_shifted[right_labels_shifted > 0] += right_offset
    right_key_to_name_shifted = {key + right_offset: value for key, value in right_key_to_name.items()}

    left_path = output_dir / f"sub-{subject}_hemi-L_space-corobl_den-2mm_label-{stem}.label.gii"
    right_path = output_dir / f"sub-{subject}_hemi-R_space-corobl_den-2mm_label-{stem}.label.gii"
    nib.save(make_label_gifti(left_labels, left_key_to_name), str(left_path))
    nib.save(make_label_gifti(right_labels_shifted, right_key_to_name_shifted), str(right_path))

    dlabel_path = output_dir / f"sub-{subject}_space-corobl_den-2mm_label-{stem}.dlabel.nii"
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
) -> Path:
    render_root = outdir / "batch"
    render_root.mkdir(parents=True, exist_ok=True)
    cmd = [
        PYTHON_EXE,
        str(REPO_ROOT / "scripts" / "workbench" / "render_wb_scene_batch.py"),
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
    ]
    cmd.extend(["--left-label-template", str(left_labels)])
    cmd.extend(["--right-label-template", str(right_labels)])
    run(cmd)
    return render_root / f"sub-{subject}" / f"sub-{subject}_wb_{name}_native.png"


def render_locked_grid_png(
    *,
    subject: str,
    scene: Path,
    views: list[str],
    layout: str,
    outdir: Path,
    name: str,
    title: str,
    left_labels: Path,
    right_labels: Path,
    legend_group: str = "label",
) -> dict[str, str]:
    outdir.mkdir(parents=True, exist_ok=True)

    if "ventral" not in views:
        raise ValueError("views must include ventral")

    native_png = render_locked_native_view(
        subject=subject,
        scene=scene,
        outdir=outdir / "native",
        name=f"{name}_native",
        left_labels=left_labels,
        right_labels=right_labels,
    )

    final_png = outdir / f"sub-{subject}_wb_{name}_biglegend.png"
    compose_cmd = [
        PYTHON_EXE,
        str(REPO_ROOT / "scripts" / "workbench" / "compose_wb_grid_with_legend.py"),
        "--ventral-image",
        str(native_png),
        "--layout",
        "1x2",
        "--left-labels",
        str(left_labels),
        "--right-labels",
        str(right_labels),
        "--legend-group",
        legend_group,
        "--title",
        title,
        "--out",
        str(final_png),
    ]
    run(compose_cmd)
    payload: dict[str, str] = {
        "biglegend_png": str(final_png),
        "native_scene_png": str(native_png),
    }
    return payload


def save_json(path: Path, data: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def load_canonical_network_rows(path: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            rows.append(
                {
                    "canonical_network": row["canonical_network"],
                    "n_parcels_merged": int(row["n_parcels_merged"]),
                    "original_parent_networks": [
                        token for token in str(row["original_parent_networks"]).split(",") if token
                    ],
                }
            )
    return rows


def select_final_k(k_metrics: list[dict[str, object]]) -> int:
    best_row = max(k_metrics, key=lambda row: float(row["ari_odd_even"]))
    target_ari = float(best_row["ari_odd_even"]) - 0.02
    eligible = [
        row
        for row in sorted(k_metrics, key=lambda row: int(row["k"]))
        if float(row["ari_odd_even"]) >= target_ari and float(row["min_cluster_size_fraction"]) >= 0.05
    ]
    return int(eligible[0]["k"] if eligible else best_row["k"])


def build_cluster_annotations(
    *,
    labels: np.ndarray,
    profile_source: np.ndarray,
    profile_networks: list[str],
    hemi: str,
    profile_mode: str,
) -> tuple[list[dict[str, object]], dict[int, str], np.ndarray]:
    annotations: list[dict[str, object]] = []
    key_to_name: dict[int, str] = {}
    probability_rows: list[np.ndarray] = []
    for cluster_id in sorted(int(x) for x in np.unique(labels)):
        mask = labels == cluster_id
        profile = np.nanmean(profile_source[mask, :], axis=0).astype(np.float32)
        order = np.argsort(profile)[::-1]
        top1 = int(order[0])
        top2 = int(order[1]) if len(order) > 1 else int(order[0])
        dominant = profile_networks[top1]
        margin = float(profile[top1] - profile[top2])
        cluster_name = f"{hemi}C{cluster_id}_{dominant}"
        key_to_name[cluster_id] = cluster_name
        if profile_mode == "fc":
            row = grouped_fc_to_probabilities(profile[None, :])[0]
        elif profile_mode == "probability":
            row = normalize_probability_rows(profile[None, :])[0]
        else:
            raise ValueError(f"Unsupported profile_mode: {profile_mode}")
        probability_rows.append(row)
        annotations.append(
            {
                "cluster_id": int(cluster_id),
                "cluster_name": cluster_name,
                "cluster_fraction": float(mask.mean()),
                "dominant_network": dominant,
                "top1_minus_top2_margin": margin,
            }
        )
    return annotations, key_to_name, np.asarray(probability_rows, dtype=np.float32)


def evaluate_k_range(
    *,
    features_full: np.ndarray,
    features_odd: np.ndarray,
    features_even: np.ndarray,
    profile_source: np.ndarray,
    profile_networks: list[str],
    connectivity: sparse.csr_matrix,
    outdir: Path,
    hemi: str,
    profile_mode: str,
) -> dict[str, object]:
    k_metrics: list[dict[str, object]] = []
    k_to_annotations: dict[int, list[dict[str, object]]] = {}
    k_to_key_names: dict[int, dict[int, str]] = {}
    k_to_probability_rows: dict[int, np.ndarray] = {}
    tss = float(np.sum((features_full - np.mean(features_full, axis=0, keepdims=True)) ** 2))

    previous_wcss: float | None = None
    for k in EVAL_K:
        labels_full = cluster_embedding(features_full, connectivity, k)
        labels_odd = cluster_embedding(features_odd, connectivity, k)
        labels_even = cluster_embedding(features_even, connectivity, k)
        sizes = [int(np.count_nonzero(labels_full == label)) for label in sorted(np.unique(labels_full))]
        min_frac = min(size / labels_full.size for size in sizes)
        total_cc, per_cluster_cc = connected_component_count(labels_full, connectivity)
        ari = adjusted_rand_score(labels_odd, labels_even)
        sil = compute_silhouette(features_full, labels_full)
        ch = float(calinski_harabasz_score(features_full, labels_full))
        db = float(davies_bouldin_score(features_full, labels_full))
        wcss = compute_wcss(features_full, labels_full)
        bss_ratio = float(1.0 - (wcss / max(tss, 1e-12)))
        entropy = compute_balance_entropy(labels_full)
        delta_wcss = None if previous_wcss is None else float(previous_wcss - wcss)
        previous_wcss = wcss

        annotations, key_to_name, probability_rows = build_cluster_annotations(
            labels=labels_full,
            profile_source=profile_source,
            profile_networks=profile_networks,
            hemi=hemi,
            profile_mode=profile_mode,
        )
        k_to_annotations[k] = annotations
        k_to_key_names[k] = key_to_name
        k_to_probability_rows[k] = probability_rows

        k_dir = outdir / f"k_{k}"
        k_dir.mkdir(parents=True, exist_ok=True)
        np.save(k_dir / "cluster_labels.npy", labels_full.astype(np.int32))
        save_json(
            k_dir / "cluster_annotation.json",
            {
                "hemi": hemi,
                "k": int(k),
                "clusters": annotations,
                "probability_rows": probability_rows.tolist(),
                "networks": profile_networks,
            },
        )
        metric_row = {
            "k": int(k),
            "ari_odd_even": float(ari),
            "silhouette": float(sil),
            "calinski_harabasz": ch,
            "davies_bouldin": db,
            "wcss": float(wcss),
            "delta_wcss": delta_wcss,
            "min_cluster_size_fraction": float(min_frac),
            "bss_tss_ratio": bss_ratio,
            "cluster_balance_entropy": entropy,
            "connected_component_count": int(total_cc),
            "per_cluster_connected_components": {str(key): int(value) for key, value in per_cluster_cc.items()},
        }
        k_metrics.append(metric_row)

    k_final = select_final_k(k_metrics)
    labels_final = np.load(outdir / f"k_{k_final}" / "cluster_labels.npy").astype(np.int32)
    save_json(
        outdir / "selection_summary.json",
        {
            "hemi": hemi,
            "k_metrics": k_metrics,
            "k_final": int(k_final),
            "clusters": k_to_annotations[k_final],
            "probability_rows": k_to_probability_rows[k_final].tolist(),
            "networks": profile_networks,
        },
    )
    return {
        "k_metrics": k_metrics,
        "k_final": int(k_final),
        "labels_final": labels_final,
        "cluster_annotations": k_to_annotations[k_final],
        "probability_rows": k_to_probability_rows[k_final],
        "profile_networks": profile_networks,
        "key_to_name": k_to_key_names[k_final],
    }


def run_gradient_branch(
    *,
    grouped_fc: np.ndarray,
    grouped_fc_odd: np.ndarray,
    grouped_fc_even: np.ndarray,
    networks: list[str],
    connectivity: sparse.csr_matrix,
    feature_dir: Path,
    clustering_dir: Path,
    hemi: str,
) -> dict[str, object]:
    gradients, eigvals = diffusion_map_embedding(build_sparse_affinity(grouped_fc, 0.1), n_components=5)
    gradients_odd, eigvals_odd = diffusion_map_embedding(build_sparse_affinity(grouped_fc_odd, 0.1), n_components=5)
    gradients_even, eigvals_even = diffusion_map_embedding(build_sparse_affinity(grouped_fc_even, 0.1), n_components=5)
    features_full = zscore_columns(gradients[:, :3])
    features_odd = zscore_columns(gradients_odd[:, :3])
    features_even = zscore_columns(gradients_even[:, :3])

    feature_dir.mkdir(parents=True, exist_ok=True)
    np.save(feature_dir / "hipp_network_fc_gradients.npy", gradients.astype(np.float32))
    np.save(feature_dir / "hipp_network_fc_gradient_eigenvalues.npy", eigvals.astype(np.float32))
    np.save(feature_dir / "hipp_network_fc_gradients_odd.npy", gradients_odd.astype(np.float32))
    np.save(feature_dir / "hipp_network_fc_gradients_even.npy", gradients_even.astype(np.float32))
    save_json(
        feature_dir / "feature_summary.json",
        {
            "hemi": hemi,
            "feature_kind": "network-gradient",
            "feature_shape": [int(features_full.shape[0]), int(features_full.shape[1])],
            "source_shape": [int(grouped_fc.shape[0]), int(grouped_fc.shape[1])],
            "source_networks": networks,
            "eigenvalues": [float(x) for x in eigvals.tolist()],
            "eigenvalues_odd": [float(x) for x in eigvals_odd.tolist()],
            "eigenvalues_even": [float(x) for x in eigvals_even.tolist()],
        },
    )

    cluster = evaluate_k_range(
        features_full=features_full,
        features_odd=features_odd,
        features_even=features_even,
        profile_source=grouped_fc,
        profile_networks=networks,
        connectivity=connectivity,
        outdir=clustering_dir,
        hemi=hemi,
        profile_mode="fc",
    )
    cluster["feature_summary"] = {
        "feature_kind": "network-gradient",
        "feature_shape": [int(features_full.shape[0]), int(features_full.shape[1])],
        "source_shape": [int(grouped_fc.shape[0]), int(grouped_fc.shape[1])],
        "source_networks": networks,
        "eigenvalues": [float(x) for x in eigvals.tolist()],
    }
    return cluster


def run_probability_branch(
    *,
    grouped_fc: np.ndarray,
    grouped_fc_odd: np.ndarray,
    grouped_fc_even: np.ndarray,
    networks: list[str],
    connectivity: sparse.csr_matrix,
    surface_coords: np.ndarray | None,
    feature_dir: Path,
    clustering_dir: Path,
    soft_dir: Path,
    hemi: str,
    save_soft_extras: bool,
    strict_soft_route: bool,
    zero_negative: bool,
) -> dict[str, object]:
    probabilities = grouped_fc_to_probabilities(grouped_fc, zero_negative=zero_negative)
    probabilities_odd = grouped_fc_to_probabilities(grouped_fc_odd, zero_negative=zero_negative)
    probabilities_even = grouped_fc_to_probabilities(grouped_fc_even, zero_negative=zero_negative)
    long_axis_order = compute_long_axis_order(surface_coords) if strict_soft_route else None
    if strict_soft_route:
        regularized_probabilities = regularize_probability_profiles(
            probabilities,
            connectivity,
            long_axis_order=long_axis_order,
        )
        regularized_probabilities_odd = regularize_probability_profiles(
            probabilities_odd,
            connectivity,
            long_axis_order=long_axis_order,
        )
        regularized_probabilities_even = regularize_probability_profiles(
            probabilities_even,
            connectivity,
            long_axis_order=long_axis_order,
        )
        features_full = zscore_columns(regularized_probabilities)
        features_odd = zscore_columns(regularized_probabilities_odd)
        features_even = zscore_columns(regularized_probabilities_even)
    else:
        regularized_probabilities = None
        regularized_probabilities_odd = None
        regularized_probabilities_even = None
        features_full = zscore_columns(probabilities)
        features_odd = zscore_columns(probabilities_odd)
        features_even = zscore_columns(probabilities_even)

    feature_dir.mkdir(parents=True, exist_ok=True)
    np.save(feature_dir / "network_probabilities.npy", probabilities.astype(np.float32))
    np.save(feature_dir / "network_probabilities_odd.npy", probabilities_odd.astype(np.float32))
    np.save(feature_dir / "network_probabilities_even.npy", probabilities_even.astype(np.float32))
    np.save(feature_dir / "grouped_fc.npy", grouped_fc.astype(np.float32))
    save_json(
        feature_dir / "feature_summary.json",
        {
            "hemi": hemi,
            "feature_kind": "probability-regularized" if strict_soft_route else "probability",
            "feature_shape": [int(features_full.shape[0]), int(features_full.shape[1])],
            "networks": networks,
            "negative_fc_policy": "clip-to-zero" if zero_negative else "row-min-shift",
            "regularization": (
                {
                    "kind": "mesh-plus-long-axis",
                    "mesh_mix": 0.35,
                    "axis_mix": 0.20,
                    "n_iter": 3,
                }
                if strict_soft_route
                else None
            ),
        },
    )

    soft_outputs: dict[str, object] = {
        "networks": networks,
        "mean_probabilities": probabilities.mean(axis=0).tolist(),
        "max_probability_mean": float(probabilities.max(axis=1).mean()),
        "max_probability_median": float(np.median(probabilities.max(axis=1))),
        "entropy_mean": float(np.mean(-(probabilities * np.log(np.clip(probabilities, 1e-12, None))).sum(axis=1))),
    }

    if save_soft_extras:
        argmax_labels, regularized_for_argmax = regularize_argmax_labels(
            probabilities,
            connectivity,
            long_axis_order=long_axis_order,
        )
        soft_dir.mkdir(parents=True, exist_ok=True)
        if strict_soft_route:
            np.save(
                soft_dir / "network_probabilities_regularized.npy",
                regularized_probabilities.astype(np.float32),
            )
        np.save(soft_dir / "network_probabilities_argmax_basis.npy", regularized_for_argmax.astype(np.float32))
        np.save(soft_dir / "regularized_argmax_labels.npy", argmax_labels.astype(np.int32))
        soft_summary = {
            "hemi": hemi,
            "networks": networks,
            "mean_probabilities": probabilities.mean(axis=0).tolist(),
            "argmax_occupancy": summarize_argmax_occupancy(argmax_labels, len(networks)).tolist(),
            "max_probability_mean": float(probabilities.max(axis=1).mean()),
            "max_probability_median": float(np.median(probabilities.max(axis=1))),
            "regularized_argmax_label_count": int(np.unique(argmax_labels).size),
        }
        if strict_soft_route:
            soft_summary["mean_regularized_probabilities"] = regularized_probabilities.mean(axis=0).tolist()
            soft_summary["regularization"] = {
                "kind": "mesh-plus-long-axis",
                "mesh_mix": 0.35,
                "axis_mix": 0.20,
                "n_iter": 3,
            }
        save_json(soft_dir / "soft_output_summary.json", soft_summary)
        soft_outputs["soft_output_summary"] = str(soft_dir / "soft_output_summary.json")
        soft_outputs["regularized_argmax_path"] = str(soft_dir / "regularized_argmax_labels.npy")
        soft_outputs["argmax_occupancy"] = summarize_argmax_occupancy(argmax_labels, len(networks)).tolist()
        if strict_soft_route:
            soft_outputs["mean_regularized_probabilities"] = regularized_probabilities.mean(axis=0).tolist()
            soft_outputs["regularization"] = {
                "kind": "mesh-plus-long-axis",
                "mesh_mix": 0.35,
                "axis_mix": 0.20,
                "n_iter": 3,
            }

    cluster = evaluate_k_range(
        features_full=features_full,
        features_odd=features_odd,
        features_even=features_even,
        profile_source=regularized_probabilities if strict_soft_route else probabilities,
        profile_networks=networks,
        connectivity=connectivity,
        outdir=clustering_dir,
        hemi=hemi,
        profile_mode="probability",
    )
    cluster["feature_summary"] = {
        "feature_kind": "probability-regularized" if strict_soft_route else "probability",
        "feature_shape": [int(features_full.shape[0]), int(features_full.shape[1])],
        "networks": networks,
        "negative_fc_policy": "clip-to-zero" if zero_negative else "row-min-shift",
    }
    if strict_soft_route:
        cluster["feature_summary"]["regularization"] = {
            "kind": "mesh-plus-long-axis",
            "mesh_mix": 0.35,
            "axis_mix": 0.20,
            "n_iter": 3,
        }
    cluster["soft_outputs"] = soft_outputs
    return cluster


def run_wta_branch(
    *,
    grouped_fc: np.ndarray,
    networks: list[str],
    soft_dir: Path,
) -> dict[str, object]:
    soft_dir.mkdir(parents=True, exist_ok=True)

    order = np.argsort(grouped_fc, axis=1)
    best = order[:, -1]
    second = order[:, -2] if grouped_fc.shape[1] > 1 else order[:, -1]
    labels_final = best + 1
    confidence = grouped_fc[np.arange(grouped_fc.shape[0]), best] - grouped_fc[np.arange(grouped_fc.shape[0]), second]

    np.save(soft_dir / "hipp_wta_labels.npy", labels_final.astype(np.int32))
    np.save(soft_dir / "hipp_wta_confidence.npy", confidence.astype(np.float32))
    np.save(soft_dir / "hipp_to_network_correlations.npy", grouped_fc.astype(np.float32))

    key_to_name = {idx + 1: f"WTA_{net}" for idx, net in enumerate(networks)}

    cluster_annotations = []
    occupancy = summarize_argmax_occupancy(labels_final, len(networks)).tolist()
    for k, v in key_to_name.items():
        net = networks[k - 1]
        rgb = NETWORK_COLORS.get(net, (128, 128, 128, 255))[:3]
        mask = labels_final == k
        cluster_annotations.append(
            {
                "label_id": k,
                "network": net,
                "rgb": list(rgb),
                "cluster_name": v,
                "cluster_fraction": float(mask.mean()),
                "dominant_network": net,
                "top1_minus_top2_margin": float(confidence[mask].mean()) if np.any(mask) else 0.0,
            }
        )

    probability_rows = np.eye(len(networks), dtype=np.float32)

    return {
        "feature_summary": {
            "feature_kind": "network-wta",
            "n_vertices": int(grouped_fc.shape[0]),
            "n_networks": int(grouped_fc.shape[1]),
        },
        "k_metrics": [],
        "k_final": len(networks),
        "labels_final": labels_final.astype(np.int32),
        "cluster_annotations": cluster_annotations,
        "probability_rows": probability_rows,
        "profile_networks": networks,
        "key_to_name": key_to_name,
        "soft_outputs": {
            "networks": networks,
            "mean_grouped_fc": grouped_fc.mean(axis=0).astype(np.float32).tolist(),
            "network_occupancy": occupancy,
            "mean_confidence": float(np.mean(confidence)),
            "median_confidence": float(np.median(confidence)),
        },
    }


def build_panel_titles(branch_slug: str, smooth_name: str, hemi: str, k_final: int) -> str:
    smooth_label = smooth_name
    if is_soft_branch(branch_slug):
        return f"{smooth_label} {hemi} network soft-profile subregions (K={k_final})"
    if is_wta_branch(branch_slug):
        return f"{smooth_label} {hemi} network winner-takes-all"
    if is_gradient_branch(branch_slug):
        return f"{smooth_label} {hemi} network-gradient final K={k_final}"
    return f"{smooth_label} {hemi} network-cluster final K={k_final}"


def main() -> int:
    parser = argparse.ArgumentParser(description="Run single-subject network-first hippocampal functional parcellation")
    parser.add_argument("--subject", default="100610")
    parser.add_argument("--branch", default="network-gradient", choices=BRANCHES)
    parser.add_argument("--atlas-slug", default="lynch2024", choices=sorted(ATLAS_CONFIG))
    parser.add_argument("--input-root", default=str(SOURCE_ROOT / "data" / "hippunfold_input"))
    parser.add_argument("--hippunfold-root", default=str(SOURCE_ROOT / "outputs" / "dense_corobl_batch"))
    parser.add_argument("--cortex-root", default=str(SOURCE_ROOT / "outputs" / "cortex_pfm"))
    parser.add_argument("--out-root", default=str(REPO_ROOT / "outputs" / "hipp_functional_parcellation_network"))
    parser.add_argument(
        "--shared-surface-store-root",
        default=None,
        help="Optional shared surface store root for subject-level timeseries (defaults to <out-root>/_shared)",
    )
    parser.add_argument("--scene", default=str(DEFAULT_SCENE))
    parser.add_argument("--views", default="ventral,dorsal")
    parser.add_argument("--layout", choices=["1x2", "2x2"], default="2x2")
    parser.add_argument("--resume-mode", choices=["resume", "force"], default="resume")
    parser.add_argument("--retain-level", choices=["label", "render", "feature", "all"], default="render")
    parser.add_argument(
        "--surface-source-dir",
        default=None,
        help="Existing corobl hippocampal surface sampling directory containing raw .func.gii/.npy outputs",
    )
    args = parser.parse_args()

    views = [token.strip() for token in args.views.split(",") if token.strip()]
    valid_views = {"ventral", "dorsal"}
    if not views or any(token not in valid_views for token in views):
        raise ValueError(f"Invalid --views: {args.views}")
    subject = args.subject
    branch_slug = args.branch
    atlas_slug = args.atlas_slug
    resume_mode = args.resume_mode
    atlas_cfg = ATLAS_CONFIG[atlas_slug]
    input_root = Path(args.input_root).resolve()
    hipp_root = Path(args.hippunfold_root).resolve() / f"sub-{subject}"
    cortex_root = Path(args.cortex_root).resolve() / f"sub-{subject}" / atlas_slug
    parcellation_root = Path(args.out_root).resolve()
    out_root = parcellation_root / branch_slug / atlas_slug / f"sub-{subject}"
    shared_store_root = (
        Path(args.shared_surface_store_root).resolve()
        if args.shared_surface_store_root
        else parcellation_root / "_shared"
    )
    shared_reference_store_dir = shared_store_root / f"sub-{subject}" / "reference" / atlas_slug
    shared_surface_store_dir = shared_store_root / f"sub-{subject}" / "surface"
    scene = Path(args.scene).resolve()
    surface_source_dir = (
        Path(args.surface_source_dir).resolve()
        if args.surface_source_dir
        else (
            Path(args.hippunfold_root).resolve()
            / "_archived_volume_functional"
            / f"sub-{subject}"
            / "post_dense_corobl"
            / "surface"
        )
    )

    dtseries = input_root / f"sub-{subject}" / "func" / f"sub-{subject}_task-rest_run-concat.dtseries.nii"
    surf_dir = hipp_root / "hippunfold" / f"sub-{subject}" / "surf"
    if not dtseries.exists():
        raise FileNotFoundError(f"Missing dtseries: {dtseries}")

    left_surface = surf_dir / f"sub-{subject}_hemi-L_space-corobl_label-hipp_midthickness.surf.gii"
    right_surface = surf_dir / f"sub-{subject}_hemi-R_space-corobl_label-hipp_midthickness.surf.gii"
    left_struct_labels = surf_dir / f"sub-{subject}_hemi-L_space-corobl_label-hipp_atlas-multihist7_subfields.label.gii"
    right_struct_labels = surf_dir / f"sub-{subject}_hemi-R_space-corobl_label-hipp_atlas-multihist7_subfields.label.gii"

    reference_dir = out_root / "reference"
    surface_dir = out_root / "surface"
    fc_dir = out_root / "fc"
    feature_root = out_root / "features"
    clustering_root = out_root / "clustering"
    soft_root = out_root / "soft_outputs"
    workbench_dir = out_root / "workbench_assets"
    renders_dir = out_root / "renders"
    core_selection_path = out_root / "final_selection_core.json"
    summary_selection_path = out_root / "final_selection_summary.json"
    for path in [reference_dir, surface_dir, fc_dir, feature_root, clustering_root, soft_root, workbench_dir, renders_dir]:
        path.mkdir(parents=True, exist_ok=True)

    left_cortex_labels = cortex_root / "roi_components" / "hemi_L" / f"{atlas_cfg['label_prefix']}.L.label.gii"
    right_cortex_labels = cortex_root / "roi_components" / "hemi_R" / f"{atlas_cfg['label_prefix']}.R.label.gii"
    roi_summary_path = cortex_root / "roi_components" / "roi_component_stats.json"

    reference_summary_path = shared_reference_store_dir / "reference_summary.json"
    canonical_network_table_path = shared_reference_store_dir / "cortex_canonical_networks.tsv"
    canonical_network_timeseries_path = shared_reference_store_dir / "cortex_canonical_network_timeseries.npy"
    reference_params = {
        "subject": subject,
        "atlas_slug": atlas_slug,
        "label_prefix": str(atlas_cfg["label_prefix"]),
    }
    reference_inputs = [dtseries, left_cortex_labels, right_cortex_labels, roi_summary_path]
    reference_outputs = [
        reference_summary_path,
        canonical_network_table_path,
        canonical_network_timeseries_path,
    ]
    if not stage_is_up_to_date(
        stage_dir=shared_reference_store_dir,
        resume_mode=resume_mode,
        stage_name="reference",
        params=reference_params,
        inputs=reference_inputs,
        outputs=reference_outputs,
    ):
        run(
            [
                PYTHON_EXE,
                str(REPO_ROOT / "scripts" / "cortex" / "extract_cortex_roi_component_timeseries.py"),
                "--subject",
                subject,
                "--dtseries",
                str(dtseries),
                "--left-labels",
                str(left_cortex_labels),
                "--right-labels",
                str(right_cortex_labels),
                "--roi-summary",
                str(roi_summary_path),
                "--atlas-slug",
                atlas_slug,
                "--outdir",
                str(shared_reference_store_dir),
            ]
        )
        write_stage_manifest(
            stage_dir=shared_reference_store_dir,
            stage_name="reference",
            params=reference_params,
            inputs=reference_inputs,
            outputs=reference_outputs,
        )

    write_reference_store_pointer(
        pointer_dir=reference_dir,
        shared_reference_store_dir=shared_reference_store_dir,
        subject=subject,
        atlas_slug=atlas_slug,
        reference_summary_path=reference_summary_path,
        canonical_network_table_path=canonical_network_table_path,
        canonical_network_timeseries_path=canonical_network_timeseries_path,
    )

    reference_summary = json.loads(reference_summary_path.read_text(encoding="utf-8"))
    canonical_network_rows = load_canonical_network_rows(canonical_network_table_path)
    network_ts = np.load(canonical_network_timeseries_path).astype(np.float32)
    networks = [str(row["canonical_network"]) for row in canonical_network_rows]

    left_raw_metric = surface_source_dir / f"sub-{subject}_hemi-L_space-corobl_den-2mm_label-hipp_bold.func.gii"
    right_raw_metric = surface_source_dir / f"sub-{subject}_hemi-R_space-corobl_den-2mm_label-hipp_bold.func.gii"
    if not left_raw_metric.exists() or not right_raw_metric.exists():
        raise FileNotFoundError(
            f"Missing archived surface sampling inputs under {surface_source_dir}: "
            f"{left_raw_metric.name}, {right_raw_metric.name}"
        )

    left_coords, left_faces = load_surface(left_surface)
    right_coords, right_faces = load_surface(right_surface)
    left_adj = build_surface_adjacency(left_faces, int(left_coords.shape[0]))
    right_adj = build_surface_adjacency(right_faces, int(right_coords.shape[0]))

    two_mm_left_func = shared_surface_store_dir / "2mm" / f"sub-{subject}_hemi-L_space-corobl_den-2mm_label-hipp_bold.func.gii"
    two_mm_right_func = shared_surface_store_dir / "2mm" / f"sub-{subject}_hemi-R_space-corobl_den-2mm_label-hipp_bold.func.gii"
    two_mm_left_path = shared_surface_store_dir / "2mm" / f"sub-{subject}_hemi-L_timeseries.npy"
    two_mm_right_path = shared_surface_store_dir / "2mm" / f"sub-{subject}_hemi-R_timeseries.npy"
    fwhm_left_func = shared_surface_store_dir / "4mm" / f"sub-{subject}_hemi-L_space-corobl_den-2mm_label-hipp_bold.func.gii"
    fwhm_right_func = shared_surface_store_dir / "4mm" / f"sub-{subject}_hemi-R_space-corobl_den-2mm_label-hipp_bold.func.gii"
    fwhm_left_path = shared_surface_store_dir / "4mm" / f"sub-{subject}_hemi-L_timeseries.npy"
    fwhm_right_path = shared_surface_store_dir / "4mm" / f"sub-{subject}_hemi-R_timeseries.npy"
    surface_params = {"subject": subject, "smoothings": SMOOTH_ORDER}
    surface_inputs = [left_raw_metric, right_raw_metric, left_surface, right_surface]
    surface_outputs = [
        two_mm_left_func,
        two_mm_right_func,
        two_mm_left_path,
        two_mm_right_path,
        fwhm_left_func,
        fwhm_right_func,
        fwhm_left_path,
        fwhm_right_path,
    ]
    if not stage_is_up_to_date(
        stage_dir=shared_surface_store_dir,
        resume_mode=resume_mode,
        stage_name="surface",
        params=surface_params,
        inputs=surface_inputs,
        outputs=surface_outputs,
    ):
        for hemi, surface_path, metric_path, smooth_mm, out_metric in [
            ("L", left_surface, left_raw_metric, "2", two_mm_left_func),
            ("R", right_surface, right_raw_metric, "2", two_mm_right_func),
            ("L", left_surface, left_raw_metric, "4", fwhm_left_func),
            ("R", right_surface, right_raw_metric, "4", fwhm_right_func),
        ]:
            smooth_dir = out_metric.parent
            smooth_dir.mkdir(parents=True, exist_ok=True)
            run(
                [
                    WB_COMMAND,
                    "-metric-smoothing",
                    str(surface_path),
                    str(metric_path),
                    smooth_mm,
                    str(out_metric),
                    "-fwhm",
                ]
            )

        np.save(
            two_mm_left_path,
            sanitize_timeseries(load_metric_array(two_mm_left_func, expected_n_vertices=int(left_coords.shape[0]))),
        )
        np.save(
            two_mm_right_path,
            sanitize_timeseries(load_metric_array(two_mm_right_func, expected_n_vertices=int(right_coords.shape[0]))),
        )
        np.save(
            fwhm_left_path,
            sanitize_timeseries(load_metric_array(fwhm_left_func, expected_n_vertices=int(left_coords.shape[0]))),
        )
        np.save(
            fwhm_right_path,
            sanitize_timeseries(load_metric_array(fwhm_right_func, expected_n_vertices=int(right_coords.shape[0]))),
        )
        write_stage_manifest(
            stage_dir=shared_surface_store_dir,
            stage_name="surface",
            params=surface_params,
            inputs=surface_inputs,
            outputs=surface_outputs,
        )

    write_surface_store_pointer(
        pointer_dir=surface_dir,
        shared_surface_store_dir=shared_surface_store_dir,
        subject=subject,
        two_mm_left_path=two_mm_left_path,
        two_mm_right_path=two_mm_right_path,
        fwhm_left_path=fwhm_left_path,
        fwhm_right_path=fwhm_right_path,
    )

    smooth_specs: dict[str, dict[str, np.ndarray | str]] = {
        "2mm": {
            "left": np.load(two_mm_left_path).astype(np.float32),
            "right": np.load(two_mm_right_path).astype(np.float32),
            "description": "Workbench metric smoothing, 2 mm FWHM",
        },
        "4mm": {
            "left": np.load(fwhm_left_path).astype(np.float32),
            "right": np.load(fwhm_right_path).astype(np.float32),
            "description": "Workbench metric smoothing, 4 mm FWHM",
        },
    }

    final_selection_core: dict[str, object] = {
        "subject": subject,
        "branch_slug": branch_slug,
        "atlas_slug": atlas_slug,
        "atlas_display_name": atlas_cfg["display_name"],
        "smoothings": SMOOTH_ORDER,
        "hemisphere_policy": "per-hemi",
        "k_policy": "independent_3_to_8_per_hemi",
        "reference_summary": reference_summary,
        "per_smooth": {},
    }

    branch_tag = branch_slug.replace("-", "_")
    compute_params = {
        "subject": subject,
        "branch_slug": branch_slug,
        "atlas_slug": atlas_slug,
        "smoothings": SMOOTH_ORDER,
        "eval_k": EVAL_K,
        "strict_soft_route": bool(is_soft_branch(branch_slug)),
        "negative_fc_policy": "clip-to-zero" if uses_nonnegative_probabilities(branch_slug) else "row-min-shift",
    }
    compute_inputs = [
        canonical_network_table_path,
        canonical_network_timeseries_path,
        two_mm_left_path,
        two_mm_right_path,
        fwhm_left_path,
        fwhm_right_path,
        left_surface,
        right_surface,
    ]
    compute_outputs = [core_selection_path]
    for smooth_name in SMOOTH_ORDER:
        if not is_wta_branch(branch_slug):
            compute_outputs.extend(
                [
                    clustering_root / smooth_name / "hemi_L" / "selection_summary.json",
                    clustering_root / smooth_name / "hemi_R" / "selection_summary.json",
                ]
            )
        compute_outputs.extend(
            [
                workbench_dir
                / smooth_name
                / "final"
                / f"sub-{subject}_hemi-L_space-corobl_den-2mm_label-hipp_network_cluster_{branch_tag}.label.gii",
                workbench_dir
                / smooth_name
                / "final"
                / f"sub-{subject}_hemi-R_space-corobl_den-2mm_label-hipp_network_cluster_{branch_tag}.label.gii",
                workbench_dir
                / smooth_name
                / "final"
                / f"sub-{subject}_space-corobl_den-2mm_label-hipp_network_cluster_{branch_tag}.dlabel.nii",
            ]
        )
    if not stage_is_up_to_date(
        stage_dir=clustering_root,
        resume_mode=resume_mode,
        stage_name="compute",
        params=compute_params,
        inputs=compute_inputs,
        outputs=compute_outputs,
    ):
        for smooth_name in SMOOTH_ORDER:
            spec = smooth_specs[smooth_name]
            left_clean = np.asarray(spec["left"], dtype=np.float32)
            right_clean = np.asarray(spec["right"], dtype=np.float32)

            hemi_results: dict[str, dict[str, object]] = {}
            for hemi, ts, adj in [("L", left_clean, left_adj), ("R", right_clean, right_adj)]:
                hemi_fc_dir = fc_dir / smooth_name / f"hemi_{hemi}"
                hemi_fc_dir.mkdir(parents=True, exist_ok=True)
                fc = corrcoef_rows(ts, network_ts)
                np.save(hemi_fc_dir / "hipp_vertex_to_network_fc.npy", fc.astype(np.float32))
                save_json(
                    hemi_fc_dir / "fc_summary.json",
                    {
                        "smooth": smooth_name,
                        "hemi": hemi,
                        "fc_shape": [int(fc.shape[0]), int(fc.shape[1])],
                        "n_vertices_total": int(ts.shape[0]),
                        "n_timepoints": int(ts.shape[1]),
                        "n_networks_used": int(network_ts.shape[0]),
                        "networks": networks,
                    },
                )

                odd_idx = np.arange(ts.shape[1]) % 2 == 0
                even_idx = ~odd_idx
                fc_odd = corrcoef_rows(ts[:, odd_idx], network_ts[:, odd_idx])
                fc_even = corrcoef_rows(ts[:, even_idx], network_ts[:, even_idx])

                feature_dir = feature_root / smooth_name / f"hemi_{hemi}"
                clustering_dir = clustering_root / smooth_name / f"hemi_{hemi}"
                soft_dir = soft_root / smooth_name / f"hemi_{hemi}"

                if is_gradient_branch(branch_slug):
                    cluster = run_gradient_branch(
                        grouped_fc=fc,
                        grouped_fc_odd=fc_odd,
                        grouped_fc_even=fc_even,
                        networks=networks,
                        connectivity=adj,
                        feature_dir=feature_dir,
                        clustering_dir=clustering_dir,
                        hemi=hemi,
                    )
                elif is_wta_branch(branch_slug):
                    cluster = run_wta_branch(
                        grouped_fc=fc,
                        networks=networks,
                        soft_dir=soft_dir,
                    )
                else:
                    cluster = run_probability_branch(
                        grouped_fc=fc,
                        grouped_fc_odd=fc_odd,
                        grouped_fc_even=fc_even,
                        networks=networks,
                        connectivity=adj,
                        surface_coords=left_coords if hemi == "L" else right_coords,
                        feature_dir=feature_dir,
                        clustering_dir=clustering_dir,
                        soft_dir=soft_dir,
                        hemi=hemi,
                        save_soft_extras=is_soft_branch(branch_slug),
                        strict_soft_route=is_soft_branch(branch_slug),
                        zero_negative=uses_nonnegative_probabilities(branch_slug),
                    )
                hemi_results[hemi] = cluster

            left_result = hemi_results["L"]
            right_result = hemi_results["R"]
            final_assets = save_combined_label_assets(
                subject=subject,
                left_labels=left_result["labels_final"],  # type: ignore[arg-type]
                right_labels=right_result["labels_final"],  # type: ignore[arg-type]
                output_dir=workbench_dir / smooth_name / "final",
                left_surface=left_surface,
                right_surface=right_surface,
                left_key_to_name=left_result["key_to_name"],  # type: ignore[arg-type]
                right_key_to_name=right_result["key_to_name"],  # type: ignore[arg-type]
                stem=f"hipp_network_cluster_{branch_tag}",
            )

            hemi_nodes = {}
            for hemi in HEMIS:
                node = hemi_results[hemi]
                hemi_nodes[hemi] = {
                    "feature_summary": node["feature_summary"],
                    "k_metrics": node["k_metrics"],
                    "k_final": node["k_final"],
                    "cluster_annotations": node["cluster_annotations"],
                    "profile_networks": node["profile_networks"],
                    "probability_rows": node["probability_rows"].tolist(),
                    "panel_title": build_panel_titles(
                        branch_slug,
                        smooth_name,
                        hemi,
                        int(node["k_final"]),
                    ),
                    "soft_outputs": node.get("soft_outputs"),
                }

            final_selection_core["per_smooth"][smooth_name] = {
                "description": spec["description"],
                "hemis": hemi_nodes,
                "final_assets": final_assets,
            }

        save_json(core_selection_path, final_selection_core)
        write_stage_manifest(
            stage_dir=clustering_root,
            stage_name="compute",
            params=compute_params,
            inputs=compute_inputs,
            outputs=compute_outputs,
        )
    else:
        final_selection_core = json.loads(core_selection_path.read_text(encoding="utf-8"))

    render_params = {
        "subject": subject,
        "branch_slug": branch_slug,
        "layout": args.layout,
        "views": views,
        "scene": str(scene),
        "legend_group_structural": "label",
        "legend_group_functional": "network",
    }
    render_outputs = [
        summary_selection_path,
        renders_dir / "structural" / f"sub-{subject}_wb_structural_biglegend.png",
    ]
    for smooth_name in SMOOTH_ORDER:
        render_outputs.append(
            renders_dir
            / "functional"
            / smooth_name
            / "final"
            / f"sub-{subject}_wb_{branch_tag}_{smooth_name}_final_biglegend.png"
        )
    render_inputs = [core_selection_path, left_struct_labels, right_struct_labels, scene]
    for smooth_name in SMOOTH_ORDER:
        assets = final_selection_core["per_smooth"][smooth_name]["final_assets"]
        render_inputs.extend([Path(assets["left_label"]), Path(assets["right_label"])])

    if not stage_is_up_to_date(
        stage_dir=renders_dir,
        resume_mode=resume_mode,
        stage_name="render",
        params=render_params,
        inputs=render_inputs,
        outputs=render_outputs,
    ):
        final_selection = dict(final_selection_core)
        structural_render = render_locked_grid_png(
            subject=subject,
            scene=scene,
            views=views,
            layout=args.layout,
            outdir=renders_dir / "structural",
            name="structural",
            title=f"sub-{subject} Structural",
            left_labels=left_struct_labels,
            right_labels=right_struct_labels,
            legend_group="label",
        )
        final_selection["structural_png"] = str(structural_render["biglegend_png"])
        final_selection["structural_native_scene_png"] = str(structural_render["native_scene_png"])

        for smooth_name in SMOOTH_ORDER:
            assets = final_selection["per_smooth"][smooth_name]["final_assets"]
            final_render = render_locked_grid_png(
                subject=subject,
                scene=scene,
                views=views,
                layout=args.layout,
                outdir=renders_dir / "functional" / smooth_name / "final",
                name=f"{branch_tag}_{smooth_name}_final",
                title=f"sub-{subject} {branch_slug} ({smooth_name})",
                left_labels=Path(assets["left_label"]),
                right_labels=Path(assets["right_label"]),
                legend_group="network",
            )
            final_selection["per_smooth"][smooth_name]["final_png"] = str(final_render["biglegend_png"])
            final_selection["per_smooth"][smooth_name]["native_scene_png"] = str(final_render["native_scene_png"])

        final_selection["render_config"] = {
            "layout": args.layout,
            "views": views,
            "scene": str(scene),
        }
        save_json(summary_selection_path, final_selection)
        write_stage_manifest(
            stage_dir=renders_dir,
            stage_name="render",
            params=render_params,
            inputs=render_inputs,
            outputs=render_outputs,
        )

    final_selection = json.loads(summary_selection_path.read_text(encoding="utf-8"))
    apply_retain_level(out_root, args.retain_level)
    print(
        json.dumps(
            {
                "subject": subject,
                "branch_slug": branch_slug,
                "atlas_slug": atlas_slug,
                "source_root": str(SOURCE_ROOT),
                "out_root": str(out_root),
                "shared_reference_store_dir": str(shared_reference_store_dir),
                "shared_surface_store_dir": str(shared_surface_store_dir),
                "resume_mode": resume_mode,
                "retain_level": args.retain_level,
                "layout": args.layout,
                "views": views,
                "final_selection_summary": str(summary_selection_path),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
