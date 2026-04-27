#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path

import nibabel as nib
import numpy as np
from nibabel.gifti import GiftiDataArray, GiftiImage, GiftiLabel, GiftiLabelTable
from scipy import sparse
from scipy import stats
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import SpectralClustering
from sklearn.metrics import adjusted_rand_score

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT / "scripts" / "common") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "scripts" / "common"))
if str(REPO_ROOT / "scripts" / "cortex") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "scripts" / "cortex"))

from surface_utils import infer_spec_path, long_axis_order
from kong2022_label_utils import load_kong2022_bundle


DEFAULT_SHARED_ROOT = REPO_ROOT / "outputs_migration" / "hipp_group_inputs_shared"
DEFAULT_OUT_ROOT = REPO_ROOT / "outputs_migration" / "hipp_group_merged_canonical_atlas"
DEFAULT_SUBJECTS = REPO_ROOT / "manifests" / "hcp_7t_hippocampus_fmri_qc_pass_153.txt"
DEFAULT_HIPPUNFOLD_BATCH_ROOT = REPO_ROOT / "outputs_migration" / "dense_corobl_batch"
DEFAULT_NETWORK_STYLE = REPO_ROOT / "config" / "hipp_network_style.json"
DEFAULT_CROSS_ATLAS_MERGE = REPO_ROOT / "config" / "cross_atlas_network_merge.json"
DEFAULT_GROUP_AVG_SURFACES_DIR = DEFAULT_OUT_ROOT / "group_average_surfaces"

METHOD_CHOICES = [
    "group-network-spectral",
    "group-network-partial-spectral",
    "group-intrinsic-spectral",
    "group-intrinsic-partial-spectral",
    "group-network-top",
    "group-network-all",
]
METHODS_DEFAULT = [
    "group-intrinsic-spectral",
    "group-network-top",
    "group-network-all",
]
ATLASES = ["kong2019", "kong2019_8net", "kong2022"]
SMOOTHINGS = ["2mm", "4mm"]
HEMIS = ["L", "R"]
K_RANGE = list(range(2, 11))
KONG2022_ONLY_METHODS = {"group-network-top", "group-network-all"}
CLUSTERLESS_METHODS = {"group-network-top", "group-network-all"}


@dataclass(frozen=True)
class Combo:
    method: str
    atlas: str
    smoothing: str
    hemi: str
    coverage_threshold: float


def save_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def write_tsv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, delimiter="\t", extrasaction="ignore")
        w.writeheader()
        for row in rows:
            w.writerow(row)


def fisher_z_transform_fc(x: np.ndarray) -> np.ndarray:
    """Unconditional Fisher-z transform for FC caches (raw r in [-1, 1] on disk).

    NaN is preserved for invalid vertices so downstream `np.nanmean` can skip them.
    """
    arr = np.asarray(x, dtype=np.float32)
    clipped = np.clip(arr, -0.999999, 0.999999)
    out = np.arctanh(clipped).astype(np.float32)
    out[~np.isfinite(arr)] = np.nan
    return out


def zscore_cols(x: np.ndarray) -> np.ndarray:
    arr = np.asarray(x, dtype=np.float32)
    mu = np.nanmean(arr, axis=0, keepdims=True)
    sd = np.nanstd(arr, axis=0, keepdims=True)
    out = (arr - mu) / np.clip(sd, 1e-12, None)
    out[~np.isfinite(out)] = 0.0
    return out.astype(np.float32)


def cosine_affinity(features: np.ndarray) -> np.ndarray:
    x = np.asarray(features, dtype=np.float32)
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    x = x / np.clip(norms, 1e-12, None)
    sim = x @ x.T
    sim = np.clip(sim, -1.0, 1.0)
    aff = np.maximum(sim, 0.0)
    np.fill_diagonal(aff, 1.0)
    return aff.astype(np.float32)


def reorder_labels_by_size(labels_zero: np.ndarray) -> np.ndarray:
    labels_zero = labels_zero.astype(np.int32, copy=False)
    keys = sorted(int(k) for k in np.unique(labels_zero))
    pairs = sorted(((int(np.sum(labels_zero == k)), k) for k in keys), key=lambda x: (-x[0], x[1]))
    mapping = {old: idx + 1 for idx, (_sz, old) in enumerate(pairs)}
    return np.asarray([mapping[int(k)] for k in labels_zero], dtype=np.int32)


def reorder_labels_by_long_axis(labels_full: np.ndarray, axis_order: np.ndarray) -> np.ndarray:
    labels = np.asarray(labels_full, dtype=np.int32)
    order = np.asarray(axis_order, dtype=np.int64)
    if labels.ndim != 1:
        raise ValueError(f"Expected 1D labels, got shape={labels.shape}")
    if order.ndim != 1 or int(order.size) != int(labels.size):
        raise ValueError(f"Long-axis order size mismatch: order={order.shape}, labels={labels.shape}")

    x = labels[order]
    rows: list[tuple[float, int]] = []
    for cid in sorted(int(v) for v in np.unique(labels) if int(v) > 0):
        idx = np.flatnonzero(x == int(cid))
        pos = float(np.mean(idx)) if idx.size else float("inf")
        rows.append((pos, int(cid)))
    rows.sort(key=lambda t: (t[0], t[1]))
    mapping = {old: new_idx + 1 for new_idx, (_pos, old) in enumerate(rows)}
    out = np.zeros_like(labels, dtype=np.int32)
    for old, new in mapping.items():
        out[labels == int(old)] = int(new)
    return out


def cluster_ids_by_long_axis(labels_full: np.ndarray, axis_order: np.ndarray) -> list[int]:
    labels = np.asarray(labels_full, dtype=np.int32)
    order = np.asarray(axis_order, dtype=np.int64)
    x = labels[order]
    rows: list[tuple[float, int]] = []
    for cid in sorted(int(v) for v in np.unique(labels) if int(v) > 0):
        idx = np.flatnonzero(x == int(cid))
        pos = float(np.mean(idx)) if idx.size else float("inf")
        rows.append((pos, int(cid)))
    rows.sort(key=lambda t: (t[0], t[1]))
    return [int(cid) for _pos, cid in rows]


def spectral_cluster_from_affinity(aff: np.ndarray, k: int, seed: int) -> np.ndarray:
    model = SpectralClustering(
        n_clusters=int(k),
        affinity="precomputed",
        assign_labels="kmeans",
        random_state=int(seed),
        n_init=50,
        eigen_solver="arpack",
    )
    labels0 = model.fit_predict(aff)
    return reorder_labels_by_size(labels0)


def load_subjects(path: Path) -> list[str]:
    out: list[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        s = line.strip()
        if not s:
            continue
        out.append(s.replace("sub-", ""))
    if not out:
        raise RuntimeError(f"No subjects in manifest: {path}")
    return out


def parse_network_names(tsv: Path) -> list[str]:
    with tsv.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        out = []
        for row in reader:
            if "canonical_network" in row:
                out.append(str(row["canonical_network"]))
            else:
                out.append(str(next(iter(row.values()))))
    if not out:
        raise RuntimeError(f"No canonical network names found: {tsv}")
    return out


def is_network_method(method: str) -> bool:
    return method in {"group-network-spectral", "group-network-partial-spectral"}


def is_clusterless_method(method: str) -> bool:
    return method in CLUSTERLESS_METHODS


def network_feature_filename(method: str) -> str:
    if method == "group-network-spectral":
        return "hipp_vertex_to_network_fc.npy"
    if method == "group-network-partial-spectral":
        return "hipp_vertex_to_network_partial_fc.npy"
    if method in {"group-network-top", "group-network-all"}:
        return "hipp_vertex_to_network_fc.npy"
    raise ValueError(f"Not a network feature method: {method}")


def runwise_network_feature_filename(method: str) -> str:
    if method == "group-network-spectral":
        return "hipp_vertex_to_network_fc_active.npy"
    if method == "group-network-partial-spectral":
        return "hipp_vertex_to_network_partial_fc_active.npy"
    raise ValueError(f"Not a network feature method: {method}")


def bh_fdr(p_values: np.ndarray) -> np.ndarray:
    p = np.asarray(p_values, dtype=np.float64)
    q = np.full_like(p, np.nan, dtype=np.float64)
    finite = np.isfinite(p)
    m = int(np.sum(finite))
    if m == 0:
        return q
    p_f = p[finite]
    order = np.argsort(p_f)
    sorted_p = p_f[order]
    ranks = np.arange(1, m + 1, dtype=np.float64)
    adjusted = sorted_p * float(m) / ranks
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    adjusted = np.clip(adjusted, 0.0, 1.0)
    q_f = np.empty_like(sorted_p)
    q_f[order] = adjusted
    q[finite] = q_f
    return q


def one_sample_one_sided_vector(values: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    arr = np.asarray(values, dtype=np.float64)
    if arr.ndim != 2:
        raise ValueError(f"Expected 2D values, got shape={arr.shape}")
    n = np.sum(np.isfinite(arr), axis=0).astype(np.int32)
    mean = np.nanmean(arr, axis=0).astype(np.float64)
    sd = np.nanstd(arr, axis=0, ddof=1).astype(np.float64)
    t_stat = np.full_like(mean, np.nan, dtype=np.float64)
    p_pos = np.full_like(mean, np.nan, dtype=np.float64)
    p_neg = np.full_like(mean, np.nan, dtype=np.float64)

    valid = n >= 2
    stable = valid & np.isfinite(sd) & (sd > 1e-12)
    if np.any(stable):
        nn = n[stable].astype(np.float64)
        t_stat[stable] = mean[stable] / (sd[stable] / np.sqrt(nn))
        df = np.clip(n[stable] - 1, 1, None)
        p_pos[stable] = stats.t.sf(t_stat[stable], df=df)
        p_neg[stable] = stats.t.cdf(t_stat[stable], df=df)

    degenerate = valid & ~stable
    if np.any(degenerate):
        mdeg = mean[degenerate]
        t_fill = np.zeros_like(mdeg)
        ppos_fill = np.full_like(mdeg, 0.5)
        pneg_fill = np.full_like(mdeg, 0.5)
        pos = mdeg > 0.0
        neg = mdeg < 0.0
        t_fill[pos] = np.inf
        t_fill[neg] = -np.inf
        ppos_fill[pos] = 0.0
        ppos_fill[neg] = 1.0
        pneg_fill[pos] = 1.0
        pneg_fill[neg] = 0.0
        t_stat[degenerate] = t_fill
        p_pos[degenerate] = ppos_fill
        p_neg[degenerate] = pneg_fill

    return mean, t_stat, p_pos, p_neg, n.astype(np.float64)


def split_kong2022_columns_by_network(roi_registry: list[dict[str, object]]) -> tuple[dict[str, list[int]], list[str]]:
    cols_by_network: dict[str, list[int]] = {}
    for col, row in enumerate(roi_registry):
        net = str(row["canonical8"])
        cols_by_network.setdefault(net, []).append(int(col))
    network_names = sorted(cols_by_network.keys())
    return cols_by_network, network_names


def top_two_from_counts(
    count_by_network: dict[str, int],
    score_sum_by_network: dict[str, float],
) -> tuple[str, int, str, int]:
    rows = [
        (int(count_by_network.get(name, 0)), float(score_sum_by_network.get(name, 0.0)), str(name))
        for name in count_by_network
    ]
    rows.sort(key=lambda x: (-x[0], -x[1], x[2]))
    if not rows:
        return "unassigned", 0, "none", 0
    top1 = rows[0]
    if len(rows) > 1:
        top2 = rows[1]
        return str(top1[2]), int(top1[0]), str(top2[2]), int(top2[0])
    return str(top1[2]), int(top1[0]), "none", 0


def top_two_from_scores(score_by_network: dict[str, float]) -> tuple[str, float, str, float]:
    rows: list[tuple[float, str, float]] = []
    for k, v in score_by_network.items():
        score = float(v)
        sort_score = score if np.isfinite(score) else -math.inf
        rows.append((sort_score, str(k), score))
    rows.sort(key=lambda x: (-x[0], x[1]))
    top1 = rows[0] if rows else (-math.inf, "unassigned", float("nan"))
    top2 = rows[1] if len(rows) > 1 else top1
    return str(top1[1]), float(top1[2]), str(top2[1]), float(top2[2])


def verify_canonical_network_order_consistency(shared_root: Path, subjects: list[str], atlas: str) -> list[str]:
    ref_subject = subjects[0]
    ref_path = shared_root / f"sub-{ref_subject}" / "reference" / atlas / "cortex_canonical_networks.tsv"
    ref_names = parse_network_names(ref_path)
    mismatched: list[str] = []
    for subject in subjects[1:]:
        path = shared_root / f"sub-{subject}" / "reference" / atlas / "cortex_canonical_networks.tsv"
        names = parse_network_names(path)
        if names != ref_names:
            mismatched.append(f"sub-{subject}")
    if mismatched:
        raise RuntimeError(
            "Canonical network ordering mismatch across subjects "
            f"for atlas={atlas}; template=sub-{ref_subject}; mismatched={mismatched[:10]}"
        )
    return ref_names


def load_surface(path: Path) -> tuple[np.ndarray, np.ndarray]:
    img = nib.load(str(path))
    coords = np.asarray(img.agg_data("pointset"), dtype=np.float32)
    faces = np.asarray(img.agg_data("triangle"), dtype=np.int32)
    return coords, faces


def build_surface_adjacency(faces: np.ndarray, n_vertices: int) -> sparse.csr_matrix:
    rows: list[int] = []
    cols: list[int] = []
    for tri in faces:
        a, b, c = int(tri[0]), int(tri[1]), int(tri[2])
        rows.extend([a, b, b, c, c, a])
        cols.extend([b, a, c, b, a, c])
    data = np.ones(len(rows), dtype=np.float32)
    g = sparse.csr_matrix((data, (rows, cols)), shape=(n_vertices, n_vertices), dtype=np.float32)
    g = g.maximum(g.T)
    g.setdiag(0.0)
    g.eliminate_zeros()
    return g


def connected_components_per_label(labels: np.ndarray, adjacency: sparse.csr_matrix) -> dict[int, int]:
    out: dict[int, int] = {}
    for k in sorted(int(x) for x in np.unique(labels) if int(x) > 0):
        m = labels == k
        if int(np.sum(m)) == 0:
            out[k] = 0
            continue
        sub = adjacency[m][:, m]
        n_comp, _ = sparse.csgraph.connected_components(sub, directed=False, return_labels=True)
        out[k] = int(n_comp)
    return out


def cluster_profile_coherence(features: np.ndarray, labels: np.ndarray, k: int) -> tuple[float, float]:
    # Mean cosine similarity between vertex profiles and cluster centroid.
    x = np.asarray(features, dtype=np.float32)
    out: list[float] = []
    for c in range(1, int(k) + 1):
        m = labels == c
        n = int(np.sum(m))
        if n == 0:
            out.append(0.0)
            continue
        if n == 1:
            out.append(1.0)
            continue
        xc = x[m, :]
        ctr = np.mean(xc, axis=0, keepdims=True)
        ctr_norm = np.linalg.norm(ctr, axis=1, keepdims=True)
        xc_norm = np.linalg.norm(xc, axis=1, keepdims=True)
        sims = (xc @ ctr.T) / np.clip(xc_norm * ctr_norm, 1e-12, None)
        sims = np.clip(sims.ravel(), -1.0, 1.0)
        out.append(float(np.mean(sims)))
    if not out:
        return 0.0, 0.0
    return float(min(out)), float(np.mean(out))


def build_hungarian_mapping(ref_labels: np.ndarray, tgt_labels: np.ndarray, k: int) -> dict[int, int]:
    contingency = np.zeros((k, k), dtype=np.int64)
    for a, b in zip(ref_labels, tgt_labels, strict=False):
        if 1 <= int(a) <= k and 1 <= int(b) <= k:
            contingency[int(a) - 1, int(b) - 1] += 1
    if int(contingency.sum()) == 0:
        return {i: i for i in range(1, k + 1)}
    cost = int(contingency.max()) - contingency
    ridx, cidx = linear_sum_assignment(cost)
    mapping: dict[int, int] = {int(c + 1): int(r + 1) for r, c in zip(ridx, cidx, strict=True)}
    for src in range(1, k + 1):
        mapping.setdefault(src, src)
    return mapping


def remap_labels(labels: np.ndarray, mapping: dict[int, int]) -> np.ndarray:
    out = np.zeros_like(labels, dtype=np.int32)
    for src, dst in mapping.items():
        out[labels == int(src)] = int(dst)
    return out


def local_maxima_flags(values: list[float]) -> list[bool]:
    if not values:
        return []
    all_equal = len(set(float(x) for x in values)) == 1
    flags: list[bool] = []
    for i, v in enumerate(values):
        left = values[i - 1] if i > 0 else -math.inf
        right = values[i + 1] if i < len(values) - 1 else -math.inf
        is_local = all_equal or (v >= left and v >= right and (v > left or v > right or len(values) == 1))
        flags.append(bool(is_local))
    return flags


def load_cluster_colors() -> list[tuple[int, int, int, int]]:
    payload = json.loads(DEFAULT_NETWORK_STYLE.read_text(encoding="utf-8"))
    colors = [(128, 128, 128, 255)]
    for row in payload.values():
        rgba = tuple(int(v) for v in row["rgba"])
        colors.append(rgba)
    return colors


def load_canonical_network_color_map() -> dict[str, tuple[int, int, int, int]]:
    out: dict[str, tuple[int, int, int, int]] = {}
    if DEFAULT_CROSS_ATLAS_MERGE.exists():
        payload = json.loads(DEFAULT_CROSS_ATLAS_MERGE.read_text(encoding="utf-8"))
        for name, rgba in payload.get("shared_colors_rgba", {}).items():
            if not isinstance(rgba, list) or len(rgba) != 4:
                continue
            out[str(name)] = tuple(int(v) for v in rgba)
    if DEFAULT_NETWORK_STYLE.exists():
        payload = json.loads(DEFAULT_NETWORK_STYLE.read_text(encoding="utf-8"))
        for row in payload.values():
            name = row.get("name")
            rgba = row.get("rgba", [])
            if not name or not isinstance(rgba, list) or len(rgba) != 4:
                continue
            out.setdefault(str(name), tuple(int(v) for v in rgba))
    return out


def make_cluster_label_gifti(labels: np.ndarray, k: int, colors: list[tuple[int, int, int, int]]) -> GiftiImage:
    tab = GiftiLabelTable()
    null = GiftiLabel(key=0, red=0.62, green=0.62, blue=0.62, alpha=1.0)
    null.label = "Null"
    tab.labels.append(null)
    for idx in range(1, int(k) + 1):
        rgba = colors[idx % len(colors)]
        g = GiftiLabel(
            key=int(idx),
            red=float(rgba[0]) / 255.0,
            green=float(rgba[1]) / 255.0,
            blue=float(rgba[2]) / 255.0,
            alpha=float(rgba[3]) / 255.0,
        )
        g.label = f"Cluster{idx}"
        tab.labels.append(g)
    da = GiftiDataArray(data=labels.astype(np.int32), intent="NIFTI_INTENT_LABEL", datatype="NIFTI_TYPE_INT32")
    return GiftiImage(darrays=[da], labeltable=tab)


def make_named_label_gifti(labels: np.ndarray, label_table_items: list[tuple[int, str, tuple[int, int, int, int]]]) -> GiftiImage:
    tab = GiftiLabelTable()
    for key, name, rgba in label_table_items:
        g = GiftiLabel(
            key=int(key),
            red=float(rgba[0]) / 255.0,
            green=float(rgba[1]) / 255.0,
            blue=float(rgba[2]) / 255.0,
            alpha=float(rgba[3]) / 255.0,
        )
        g.label = str(name)
        tab.labels.append(g)
    da = GiftiDataArray(data=labels.astype(np.int32), intent="NIFTI_INTENT_LABEL", datatype="NIFTI_TYPE_INT32")
    return GiftiImage(darrays=[da], labeltable=tab)


def infer_surface_path(subject: str, hemi: str, batch_root: Path | None = None) -> Path:
    root = Path(batch_root).resolve() if batch_root is not None else DEFAULT_HIPPUNFOLD_BATCH_ROOT.resolve()
    p = (
        root
        / f"sub-{subject}"
        / "hippunfold"
        / f"sub-{subject}"
        / "surf"
        / f"sub-{subject}_hemi-{hemi}_space-corobl_den-512_label-hipp_midthickness.surf.gii"
    )
    if not p.exists():
        raise FileNotFoundError(f"Missing midthickness surface: {p}")
    return p


def subject_valid_mask(shared_root: Path, subject: str, hemi: str) -> np.ndarray:
    p = shared_root / f"sub-{subject}" / "surface" / "tsnr_gate" / f"sub-{subject}_hemi-{hemi}_valid_mask.npy"
    if not p.exists():
        raise FileNotFoundError(f"Missing valid mask: {p}")
    return np.load(p).astype(bool)


def subject_network_feature(shared_root: Path, subject: str, atlas: str, smoothing: str, hemi: str, method: str) -> np.ndarray:
    p = shared_root / f"sub-{subject}" / "fc" / atlas / smoothing / f"hemi_{hemi}" / network_feature_filename(method)
    if not p.exists():
        raise FileNotFoundError(f"Missing network FC cache: {p}")
    arr = np.load(p).astype(np.float32)
    if arr.ndim != 2:
        raise ValueError(f"Expected 2D network feature matrix: {p} shape={arr.shape}")
    arr = fisher_z_transform_fc(arr)
    return arr


def intrinsic_feature_candidates(
    shared_root: Path,
    subject: str,
    smoothing: str,
    hemi: str,
    run_id: int | None = None,
) -> list[Path]:
    file_rel = Path("hipp_vertex_to_vertex_fc_active.npy")
    run_rel = Path("runs") / f"run-{run_id}" / file_rel if run_id is not None else file_rel
    return [shared_root / f"sub-{subject}" / "fc" / "intrinsic" / smoothing / f"hemi_{hemi}" / run_rel]


def resolve_intrinsic_feature_path(
    shared_root: Path,
    subject: str,
    smoothing: str,
    hemi: str,
    run_id: int | None = None,
) -> Path:
    candidates = intrinsic_feature_candidates(shared_root, subject, smoothing, hemi, run_id)
    for p in candidates:
        if p.exists():
            return p
    raise FileNotFoundError(
        "Missing intrinsic FC cache. Tried:\n" + "\n".join(str(p) for p in candidates)
    )


def subject_intrinsic_feature(shared_root: Path, subject: str, atlas: str, smoothing: str, hemi: str, valid_mask: np.ndarray) -> np.ndarray:
    p = resolve_intrinsic_feature_path(shared_root, subject, smoothing, hemi)
    arr = np.load(p).astype(np.float32)
    n_v = int(valid_mask.size)
    n_valid = int(np.sum(valid_mask))
    if arr.shape == (n_v, n_v):
        full = arr
    elif arr.shape == (n_valid, n_valid):
        full = np.full((n_v, n_v), np.nan, dtype=np.float32)
        idx = np.flatnonzero(valid_mask)
        full[np.ix_(idx, idx)] = arr
    else:
        raise ValueError(
            f"Unexpected intrinsic FC shape for sub-{subject} {atlas} {smoothing} hemi-{hemi}: {arr.shape}; "
            f"expected ({n_v},{n_v}) or ({n_valid},{n_valid})"
        )
    full = fisher_z_transform_fc(full)
    np.fill_diagonal(full, 0.0)
    return full


def verify_intrinsic_atlas_invariance(shared_root: Path, subjects: list[str], smoothing: str, hemi: str, atlases: list[str]) -> None:
    for subject in subjects:
        p_intrinsic = shared_root / f"sub-{subject}" / "fc" / "intrinsic" / smoothing / f"hemi_{hemi}" / "hipp_vertex_to_vertex_fc_active.npy"
        ref = np.load(p_intrinsic) if p_intrinsic.exists() else None
        ref_path = p_intrinsic if p_intrinsic.exists() else None
        for atlas in atlases:
            p_cmp = shared_root / f"sub-{subject}" / "fc" / atlas / smoothing / f"hemi_{hemi}" / "hipp_vertex_to_vertex_fc_active.npy"
            if not p_cmp.exists():
                continue
            cmp = np.load(p_cmp)
            if ref is None:
                ref = cmp
                ref_path = p_cmp
                continue
            if ref.shape != cmp.shape:
                raise RuntimeError(
                    f"Intrinsic cache atlas-invariance violation for sub-{subject} {smoothing} hemi-{hemi}: "
                    f"{ref_path} vs {p_cmp}"
                )
            if not np.allclose(ref, cmp, rtol=1e-6, atol=1e-7, equal_nan=True):
                max_abs = float(np.nanmax(np.abs(ref - cmp)))
                raise RuntimeError(
                    f"Intrinsic cache atlas-invariance violation for sub-{subject} {smoothing} hemi-{hemi}: "
                    f"{ref_path} vs {p_cmp}; max_abs_diff={max_abs:.3e}"
                )


def group_feature_matrix(
    *,
    method: str,
    feature_bank: dict[str, np.ndarray],
    subjects_subset: list[str],
    support_mask: np.ndarray,
) -> np.ndarray:
    idx = np.flatnonzero(support_mask)
    if idx.size == 0:
        raise RuntimeError("Support mask has zero vertices")

    if is_network_method(method):
        mats = []
        for s in subjects_subset:
            m = feature_bank[s][idx, :]
            mats.append(m)
        stack = np.stack(mats, axis=0).astype(np.float32)
        group = np.nanmean(stack, axis=0)
        group[~np.isfinite(group)] = 0.0
        return zscore_cols(group)

    if method in {"group-intrinsic-spectral", "group-intrinsic-partial-spectral"}:
        mats = []
        for s in subjects_subset:
            m = feature_bank[s][np.ix_(idx, idx)]
            mats.append(m)
        stack = np.stack(mats, axis=0).astype(np.float32)
        group = np.nanmean(stack, axis=0)
        group[~np.isfinite(group)] = 0.0
        np.fill_diagonal(group, 0.0)
        return group.astype(np.float32)

    raise ValueError(f"Unknown method: {method}")


def cluster_for_subset(
    *,
    method: str,
    feature_bank: dict[str, np.ndarray],
    subjects_subset: list[str],
    support_mask: np.ndarray,
    adjacency_full: sparse.csr_matrix,
    k: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    features = group_feature_matrix(
        method=method,
        feature_bank=feature_bank,
        subjects_subset=subjects_subset,
        support_mask=support_mask,
    )
    idx = np.flatnonzero(support_mask)
    adj = adjacency_full[idx][:, idx]
    aff = cosine_affinity(features)
    fused = sparse.csr_matrix(aff).multiply(adj).tocsr()
    fused = fused.maximum(fused.T)
    labels = spectral_cluster_from_affinity(fused.toarray(), int(k), int(seed))
    full = np.zeros((support_mask.size,), dtype=np.int32)
    full[idx] = labels
    return labels.astype(np.int32), full


def subject_split_ari(
    *,
    method: str,
    feature_bank: dict[str, np.ndarray],
    valid_masks_by_subject: dict[str, np.ndarray],
    subjects: list[str],
    tau: float,
    adjacency_full: sparse.csr_matrix,
    k: int,
    seeds: list[int],
) -> list[float]:
    vals: list[float] = []
    n_skipped_overlap = 0
    n_skipped_empty = 0
    n = len(subjects)
    if n < 4:
        raise RuntimeError("Need at least four subjects for subject-split instability")
    half = n // 2
    for seed in seeds:
        rng = np.random.RandomState(int(seed))
        perm = np.asarray(subjects, dtype=object).copy()
        rng.shuffle(perm)
        sub_a = [str(x) for x in perm[:half]]
        sub_b = [str(x) for x in perm[half:]]

        cov_a = np.mean(np.stack([valid_masks_by_subject[s] for s in sub_a], axis=0), axis=0)
        cov_b = np.mean(np.stack([valid_masks_by_subject[s] for s in sub_b], axis=0), axis=0)
        mask_a = cov_a >= float(tau)
        mask_b = cov_b >= float(tau)
        overlap = mask_a & mask_b
        if int(np.sum(overlap)) <= int(k):
            n_skipped_overlap += 1
            continue

        _, full_a = cluster_for_subset(
            method=method,
            feature_bank=feature_bank,
            subjects_subset=sub_a,
            support_mask=mask_a,
            adjacency_full=adjacency_full,
            k=k,
            seed=seed + 1001,
        )
        _, full_b = cluster_for_subset(
            method=method,
            feature_bank=feature_bank,
            subjects_subset=sub_b,
            support_mask=mask_b,
            adjacency_full=adjacency_full,
            k=k,
            seed=seed + 2001,
        )
        ref = full_a[overlap]
        tgt = full_b[overlap]
        if ref.size == 0:
            n_skipped_empty += 1
            continue
        mapping = build_hungarian_mapping(ref, tgt, int(k))
        tgt_r = remap_labels(tgt, mapping)
        ari = float(adjusted_rand_score(ref, tgt_r))
        vals.append(float(ari))
    return {
        "values": vals,
        "n_requested": int(len(seeds)),
        "n_used": int(len(vals)),
        "n_skipped_overlap": int(n_skipped_overlap),
        "n_skipped_empty": int(n_skipped_empty),
    }


def strong_null_ari(
    *,
    method: str,
    feature_bank: dict[str, np.ndarray],
    valid_masks_by_subject: dict[str, np.ndarray],
    subjects: list[str],
    tau: float,
    adjacency_full: sparse.csr_matrix,
    k: int,
    seeds: list[int],
) -> dict[str, object]:
    null_bank: dict[str, np.ndarray] = {}
    for s in subjects:
        arr = feature_bank[s]
        rng = np.random.RandomState(int(seeds[0]) + int(abs(hash(s)) % 100000))
        perm = rng.permutation(arr.shape[0])
        if is_network_method(method):
            null_bank[s] = arr[perm, :].astype(np.float32)
        else:
            null_bank[s] = arr[np.ix_(perm, perm)].astype(np.float32)
    out = subject_split_ari(
        method=method,
        feature_bank=null_bank,
        valid_masks_by_subject=valid_masks_by_subject,
        subjects=subjects,
        tau=tau,
        adjacency_full=adjacency_full,
        k=k,
        seeds=seeds,
    )
    out["permutation_scope"] = "subject_specific"
    return out


def weak_null_ari(k: int, n_vertices: int, seeds: list[int]) -> dict[str, object]:
    vals: list[float] = []
    for seed in seeds:
        rng = np.random.RandomState(int(seed))
        a = rng.randint(1, int(k) + 1, size=(int(n_vertices),), dtype=np.int32)
        b = rng.randint(1, int(k) + 1, size=(int(n_vertices),), dtype=np.int32)
        ari = float(adjusted_rand_score(a, b))
        vals.append(float(ari))
    return {
        "values": vals,
        "n_requested": int(len(seeds)),
        "n_used": int(len(vals)),
    }


def runwise_feature_bank(
    *,
    method: str,
    shared_root: Path,
    subjects: list[str],
    atlas: str,
    smoothing: str,
    hemi: str,
    valid_masks: dict[str, np.ndarray],
) -> dict[int, dict[str, np.ndarray]]:
    run_ids = [1, 2, 3, 4]
    out: dict[int, dict[str, np.ndarray]] = {rid: {} for rid in run_ids}
    for rid in run_ids:
        for s in subjects:
            if is_network_method(method):
                p = (
                    shared_root
                    / f"sub-{s}"
                    / "fc"
                    / atlas
                    / smoothing
                    / f"hemi_{hemi}"
                    / "runs"
                    / f"run-{rid}"
                    / runwise_network_feature_filename(method)
                )
                if not p.exists():
                    raise FileNotFoundError(f"Missing runwise network feature: {p}")
                active = fisher_z_transform_fc(np.load(p).astype(np.float32))
                full = np.full((valid_masks[s].size, active.shape[1]), np.nan, dtype=np.float32)
                idx = np.flatnonzero(valid_masks[s])
                if active.shape[0] == idx.size:
                    full[idx, :] = active
                elif active.shape[0] == valid_masks[s].size:
                    full = active
                else:
                    raise ValueError(f"Unexpected runwise network feature shape for {p}: {active.shape}")
                out[rid][s] = full
            else:
                p = resolve_intrinsic_feature_path(shared_root, s, smoothing, hemi, rid)
                arr = np.load(p).astype(np.float32)
                arr = fisher_z_transform_fc(arr)
                np.fill_diagonal(arr, 0.0)
                idx = np.flatnonzero(valid_masks[s])
                full = np.full((valid_masks[s].size, valid_masks[s].size), np.nan, dtype=np.float32)
                if arr.shape == (idx.size, idx.size):
                    full[np.ix_(idx, idx)] = arr
                elif arr.shape == (valid_masks[s].size, valid_masks[s].size):
                    full = arr
                else:
                    raise ValueError(f"Unexpected runwise intrinsic feature shape for {p}: {arr.shape}")
                out[rid][s] = full
    return out


def run_aware_ari(
    *,
    method: str,
    run_feature_banks: dict[int, dict[str, np.ndarray]],
    subjects: list[str],
    support_mask: np.ndarray,
    adjacency_full: sparse.csr_matrix,
    k: int,
    seed_base: int,
) -> dict[str, object]:
    runs = sorted(run_feature_banks.keys())
    labels_by_run: dict[int, np.ndarray] = {}
    for rid in runs:
        _, full = cluster_for_subset(
            method=method,
            feature_bank=run_feature_banks[rid],
            subjects_subset=subjects,
            support_mask=support_mask,
            adjacency_full=adjacency_full,
            k=int(k),
            seed=int(seed_base + rid * 1000 + k),
        )
        labels_by_run[rid] = full[support_mask]
    vals: list[float] = []
    n_pairs = 0
    for i in range(len(runs)):
        for j in range(i + 1, len(runs)):
            n_pairs += 1
            a = labels_by_run[runs[i]]
            b = labels_by_run[runs[j]]
            mapping = build_hungarian_mapping(a, b, int(k))
            b_r = remap_labels(b, mapping)
            ari = float(adjusted_rand_score(a, b_r))
            vals.append(float(ari))
    return {
        "values": vals,
        "n_pairs": int(n_pairs),
        "n_used": int(len(vals)),
    }


def verify_surface_topology_consistency(
    subjects: list[str],
    hemis: list[str],
    template_surfaces: dict[str, Path],
    batch_root: Path,
) -> dict[str, object]:
    out: dict[str, object] = {}
    for hemi in hemis:
        t_coords, t_faces = load_surface(template_surfaces[hemi])
        n_vertices = int(t_coords.shape[0])
        n_faces = int(t_faces.shape[0])
        mismatched: list[str] = []
        for s in subjects:
            p = infer_surface_path(s, hemi, batch_root)
            coords, faces = load_surface(p)
            if int(coords.shape[0]) != n_vertices or int(faces.shape[0]) != n_faces:
                mismatched.append(f"sub-{s}")
                continue
            if faces.shape != t_faces.shape or not np.array_equal(faces, t_faces):
                mismatched.append(f"sub-{s}")
        if mismatched:
            raise RuntimeError(
                f"Surface topology mismatch for hemi-{hemi}: template={template_surfaces[hemi]} mismatched={mismatched[:10]}"
            )
        out[hemi] = {
            "n_vertices": n_vertices,
            "n_faces": n_faces,
            "topology_consistent_subjects": int(len(subjects)),
        }
    return out


def corrcoef_rows(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    x = np.asarray(a, dtype=np.float32)
    y = np.asarray(b, dtype=np.float32)
    x_mu = np.nanmean(x, axis=1, keepdims=True)
    y_mu = np.nanmean(y, axis=1, keepdims=True)
    x_sd = np.nanstd(x, axis=1, keepdims=True, ddof=1)
    y_sd = np.nanstd(y, axis=1, keepdims=True, ddof=1)
    x_z = (x - x_mu) / np.clip(x_sd, 1e-12, None)
    y_z = (y - y_mu) / np.clip(y_sd, 1e-12, None)
    x_z[~np.isfinite(x_z)] = 0.0
    y_z[~np.isfinite(y_z)] = 0.0
    corr = (x_z @ y_z.T) / float(max(1, x.shape[1] - 1))
    corr = np.clip(corr, -0.999999, 0.999999)
    corr[~np.isfinite(corr)] = 0.0
    return corr.astype(np.float32)


def one_sided_network_partial_corr_rows(rows: np.ndarray, network_ts: np.ndarray) -> np.ndarray:
    """Residualize rows against other networks, then correlate with each raw target network."""
    x = np.asarray(rows, dtype=np.float32)
    y = np.asarray(network_ts, dtype=np.float32)
    if x.ndim != 2 or y.ndim != 2:
        raise ValueError(f"Expected 2D rows and network_ts, got {x.shape} and {y.shape}")
    if x.shape[1] != y.shape[1]:
        raise ValueError(f"Time dimension mismatch for partial FC: {x.shape} vs {y.shape}")
    out = np.zeros((x.shape[0], y.shape[0]), dtype=np.float32)
    x_center = x - np.nanmean(x, axis=1, keepdims=True)
    x_center[~np.isfinite(x_center)] = 0.0
    y_center = y - np.nanmean(y, axis=1, keepdims=True)
    y_center[~np.isfinite(y_center)] = 0.0
    for j in range(y.shape[0]):
        if y.shape[0] <= 1:
            resid = x_center
        else:
            keep = [i for i in range(y.shape[0]) if i != j]
            controls = y_center[np.asarray(keep, dtype=np.int32), :]
            gram = controls @ controls.T
            beta = (x_center @ controls.T) @ np.linalg.pinv(gram, rcond=1e-6)
            resid = x_center - beta @ controls
        out[:, j] = corrcoef_rows(resid, y[[j], :])[:, 0]
    return out.astype(np.float32)


def cluster_network_profile(
    *,
    group_labels_final: np.ndarray,
    support_mask: np.ndarray,
    subjects: list[str],
    shared_root: Path,
    atlas: str,
    smoothing: str,
    hemi: str,
    valid_masks_by_subject: dict[str, np.ndarray],
    profile_fc_mode: str,
) -> np.ndarray:
    k_final = int(np.max(group_labels_final))
    profiles: list[np.ndarray] = []
    for subject in subjects:
        ts_path = shared_root / f"sub-{subject}" / "surface" / smoothing / f"sub-{subject}_hemi-{hemi}_timeseries.npy"
        net_path = shared_root / f"sub-{subject}" / "reference" / atlas / "cortex_canonical_network_timeseries.npy"
        if not (ts_path.exists() and net_path.exists()):
            continue
        ts = np.load(ts_path).astype(np.float32)
        net = np.load(net_path).astype(np.float32)
        if ts.ndim != 2 or net.ndim != 2:
            continue
        if ts.shape[1] != net.shape[1]:
            continue
        valid = support_mask & valid_masks_by_subject[subject]
        cluster_ts = np.full((k_final, ts.shape[1]), np.nan, dtype=np.float32)
        for c in range(1, k_final + 1):
            m = (group_labels_final == c) & valid
            if int(np.sum(m)) == 0:
                continue
            with np.errstate(invalid="ignore"):
                cluster_ts[c - 1, :] = np.nanmean(ts[m, :], axis=0).astype(np.float32)
        if profile_fc_mode == "pearson":
            corr = corrcoef_rows(cluster_ts, net)
        elif profile_fc_mode == "partial_networks_only_one_sided":
            corr = one_sided_network_partial_corr_rows(cluster_ts, net)
        else:
            raise ValueError(f"Unknown profile_fc_mode: {profile_fc_mode}")
        corr = np.clip(corr, -0.999999, 0.999999)
        fisher = np.arctanh(corr).astype(np.float32)
        fisher[~np.isfinite(fisher)] = np.nan
        profiles.append(fisher)
    if not profiles:
        raise RuntimeError("No subject-level profiles available for cluster-network annotation")
    out = np.nanmean(np.stack(profiles, axis=0), axis=0).astype(np.float32)
    out[~np.isfinite(out)] = 0.0
    return out


def load_kong2022_roi_registry() -> list[dict[str, object]]:
    bundle = load_kong2022_bundle()
    rows: list[dict[str, object]] = []
    for row in bundle["roi_registry"]:
        rows.append(
            {
                "roi_id": int(row["roi_id"]),
                "roi_name": str(row["roi_name"]),
                "hemi": str(row["hemi"]),
                "family17": str(row["family17"]),
                "subregion": str(row["subregion"]),
                "rank": int(row["rank"]),
                "canonical8": str(row["canonical8"]),
            }
        )
    if len(rows) != 400:
        raise RuntimeError(f"Expected 400 Kong2022 registry rows, got {len(rows)}")
    return rows


def read_kong2022_roi_tsv(path: Path) -> list[dict[str, object]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        out = []
        for row in csv.DictReader(f, delimiter="\t"):
            out.append(
                {
                    "parcel_id": int(row["parcel_id"]),
                    "parcel_name": str(row["parcel_name"]),
                    "family17": str(row["family17"]),
                    "canonical8": str(row["canonical8"]),
                    "hemisphere": str(row["hemisphere"]),
                    "subregion": str(row.get("subregion", "")),
                    "rank": int(row.get("rank", "0")),
                }
            )
    return out


def cluster_roi_profile(
    *,
    group_labels_final: np.ndarray,
    support_mask: np.ndarray,
    subjects: list[str],
    shared_root: Path,
    smoothing: str,
    hemi: str,
    valid_masks_by_subject: dict[str, np.ndarray],
) -> tuple[np.ndarray, list[dict[str, object]]]:
    k_final = int(np.max(group_labels_final))
    full_registry = load_kong2022_roi_registry()
    reg_by_id = {int(r["roi_id"]): r for r in full_registry}
    col_by_id = {int(r["roi_id"]): i for i, r in enumerate(full_registry)}

    profiles: list[np.ndarray] = []
    for subject in subjects:
        ts_path = shared_root / f"sub-{subject}" / "surface" / smoothing / f"sub-{subject}_hemi-{hemi}_timeseries.npy"
        roi_path = shared_root / f"sub-{subject}" / "reference" / "kong2022" / "cortex_roi_parcel_timeseries.npy"
        roi_tsv = shared_root / f"sub-{subject}" / "reference" / "kong2022" / "cortex_roi_parcels.tsv"
        if not (ts_path.exists() and roi_path.exists() and roi_tsv.exists()):
            continue
        ts = np.load(ts_path).astype(np.float32)
        roi = np.load(roi_path).astype(np.float32)
        rows = read_kong2022_roi_tsv(roi_tsv)
        if ts.ndim != 2 or roi.ndim != 2:
            continue
        if ts.shape[1] != roi.shape[1]:
            continue
        if roi.shape[0] != len(rows):
            raise RuntimeError(
                f"ROI TSV rows mismatch for sub-{subject}: roi_shape={roi.shape} rows={len(rows)}"
            )

        valid = support_mask & valid_masks_by_subject[subject]
        cluster_ts = np.full((k_final, ts.shape[1]), np.nan, dtype=np.float32)
        for c in range(1, k_final + 1):
            m = (group_labels_final == c) & valid
            if int(np.sum(m)) == 0:
                continue
            with np.errstate(invalid="ignore"):
                cluster_ts[c - 1, :] = np.nanmean(ts[m, :], axis=0).astype(np.float32)

        corr = corrcoef_rows(cluster_ts, roi)
        corr = np.clip(corr, -0.999999, 0.999999)
        fisher = np.arctanh(corr).astype(np.float32)
        fisher[~np.isfinite(fisher)] = np.nan

        scatter = np.full((k_final, 400), np.nan, dtype=np.float32)
        cols: list[int] = []
        for row in rows:
            roi_id = int(row["parcel_id"])
            reg = reg_by_id.get(roi_id)
            if reg is None:
                raise KeyError(f"Unexpected ROI id in sub-{subject}: {roi_id}")
            if str(row["parcel_name"]) != str(reg["roi_name"]):
                raise RuntimeError(
                    f"ROI name mismatch in sub-{subject} for id={roi_id}: "
                    f"{row['parcel_name']} vs {reg['roi_name']}"
                )
            if str(row["family17"]) != str(reg["family17"]) or str(row["canonical8"]) != str(reg["canonical8"]):
                raise RuntimeError(
                    f"ROI metadata mismatch in sub-{subject} for id={roi_id}: "
                    f"family17/canonical8 inconsistent with registry"
                )
            cols.append(int(col_by_id[roi_id]))
        scatter[:, np.asarray(cols, dtype=np.int32)] = fisher
        profiles.append(scatter)

    if not profiles:
        raise RuntimeError("No subject-level ROI profiles available for kong2022 cluster annotation")
    out = np.nanmean(np.stack(profiles, axis=0), axis=0).astype(np.float32)
    return out, full_registry


def cluster_roi_profile_subjectwise(
    *,
    group_labels_final: np.ndarray,
    support_mask: np.ndarray,
    subjects: list[str],
    shared_root: Path,
    smoothing: str,
    hemi: str,
    valid_masks_by_subject: dict[str, np.ndarray],
) -> tuple[np.ndarray, np.ndarray, list[dict[str, object]], list[str]]:
    k_final = int(np.max(group_labels_final))
    full_registry = load_kong2022_roi_registry()
    reg_by_id = {int(r["roi_id"]): r for r in full_registry}
    col_by_id = {int(r["roi_id"]): i for i, r in enumerate(full_registry)}

    profiles: list[np.ndarray] = []
    used_subjects: list[str] = []
    for subject in subjects:
        ts_path = shared_root / f"sub-{subject}" / "surface" / smoothing / f"sub-{subject}_hemi-{hemi}_timeseries.npy"
        roi_path = shared_root / f"sub-{subject}" / "reference" / "kong2022" / "cortex_roi_parcel_timeseries.npy"
        roi_tsv = shared_root / f"sub-{subject}" / "reference" / "kong2022" / "cortex_roi_parcels.tsv"
        if not (ts_path.exists() and roi_path.exists() and roi_tsv.exists()):
            continue
        ts = np.load(ts_path).astype(np.float32)
        roi = np.load(roi_path).astype(np.float32)
        rows = read_kong2022_roi_tsv(roi_tsv)
        if ts.ndim != 2 or roi.ndim != 2:
            continue
        if ts.shape[1] != roi.shape[1]:
            continue
        if roi.shape[0] != len(rows):
            raise RuntimeError(
                f"ROI TSV rows mismatch for sub-{subject}: roi_shape={roi.shape} rows={len(rows)}"
            )

        valid = support_mask & valid_masks_by_subject[subject]
        cluster_ts = np.full((k_final, ts.shape[1]), np.nan, dtype=np.float32)
        for c in range(1, k_final + 1):
            m = (group_labels_final == c) & valid
            if int(np.sum(m)) == 0:
                continue
            with np.errstate(invalid="ignore"):
                cluster_ts[c - 1, :] = np.nanmean(ts[m, :], axis=0).astype(np.float32)

        corr = corrcoef_rows(cluster_ts, roi)
        corr = np.clip(corr, -0.999999, 0.999999)
        fisher = np.arctanh(corr).astype(np.float32)
        fisher[~np.isfinite(fisher)] = np.nan

        scatter = np.full((k_final, 400), np.nan, dtype=np.float32)
        cols: list[int] = []
        for row in rows:
            roi_id = int(row["parcel_id"])
            reg = reg_by_id.get(roi_id)
            if reg is None:
                raise KeyError(f"Unexpected ROI id in sub-{subject}: {roi_id}")
            if str(row["parcel_name"]) != str(reg["roi_name"]):
                raise RuntimeError(
                    f"ROI name mismatch in sub-{subject} for id={roi_id}: "
                    f"{row['parcel_name']} vs {reg['roi_name']}"
                )
            if str(row["family17"]) != str(reg["family17"]) or str(row["canonical8"]) != str(reg["canonical8"]):
                raise RuntimeError(
                    f"ROI metadata mismatch in sub-{subject} for id={roi_id}: "
                    f"family17/canonical8 inconsistent with registry"
                )
            cols.append(int(col_by_id[roi_id]))
        scatter[:, np.asarray(cols, dtype=np.int32)] = fisher
        profiles.append(scatter)
        used_subjects.append(subject)

    if not profiles:
        raise RuntimeError("No subject-level ROI profiles available for kong2022 cluster annotation")
    stack = np.stack(profiles, axis=0).astype(np.float32)
    out_mean = np.nanmean(stack, axis=0).astype(np.float32)
    return stack, out_mean, full_registry, used_subjects


def cluster_roi_top_k(
    profile_400: np.ndarray,
    roi_registry: list[dict[str, object]],
    k_top: int = 20,
) -> dict[str, object]:
    out: dict[str, object] = {}
    for cidx in range(profile_400.shape[0]):
        row = profile_400[cidx, :]
        valid = np.flatnonzero(np.isfinite(row))
        if valid.size == 0:
            out[str(cidx + 1)] = {"top_positive": [], "top_negative": []}
            continue
        row_valid = row[valid]
        pos_order = valid[np.argsort(-row_valid)]
        neg_order = valid[np.argsort(row_valid)]
        n_take = int(min(k_top, valid.size))

        def build_items(indices: np.ndarray) -> list[dict[str, object]]:
            items: list[dict[str, object]] = []
            for col in indices[:n_take]:
                reg = roi_registry[int(col)]
                items.append(
                    {
                        "roi_idx": int(reg["roi_id"]),
                        "roi_name": str(reg["roi_name"]),
                        "family17": str(reg["family17"]),
                        "canonical8": str(reg["canonical8"]),
                        "fisher_z": float(row[int(col)]),
                    }
                )
            return items

        out[str(cidx + 1)] = {
            "top_positive": build_items(pos_order),
            "top_negative": build_items(neg_order),
        }
    return {"cluster_id_to_top20": out, "roi_registry": roi_registry}


def assignment_status_from_profile(profile: np.ndarray, network_names: list[str]) -> tuple[list[dict[str, object]], list[str | None]]:
    rows: list[dict[str, object]] = []
    dominant: list[str | None] = []
    for cidx in range(profile.shape[0]):
        row = profile[cidx, :]
        order = np.argsort(row)
        top1_i = int(order[-1])
        top2_i = int(order[-2]) if row.size > 1 else int(order[-1])
        top1 = float(row[top1_i])
        top2 = float(row[top2_i])
        margin = float(top1 - top2)
        status = "dominant"
        dom = str(network_names[top1_i])
        rows.append(
            {
                "cluster_id": int(cidx + 1),
                "status": status,
                "top1_network": str(network_names[top1_i]),
                "top1_fisher_z": top1,
                "top2_network": str(network_names[top2_i]),
                "top2_fisher_z": top2,
                "margin": margin,
                "low_margin_lt_0p05": int(margin < 0.05),
            }
        )
        dominant.append(dom)
    return rows, dominant


def assignment_status_from_cluster_roi_topn(
    *,
    profile_3d: np.ndarray,
    roi_registry: list[dict[str, object]],
    q_thresh: float,
    top_n_cap: int,
) -> tuple[list[dict[str, object]], list[str | None], np.ndarray]:
    k_final = int(profile_3d.shape[1])
    mean_profile = np.nanmean(profile_3d, axis=0).astype(np.float32)
    rows: list[dict[str, object]] = []
    dominant: list[str | None] = []
    for cidx in range(k_final):
        values = profile_3d[:, cidx, :]
        mean_z, _t_stat, p_pos, p_neg, n_vec = one_sample_one_sided_vector(values)
        q_pos = bh_fdr(p_pos)
        q_neg = bh_fdr(p_neg)
        sig_pos = np.flatnonzero(np.isfinite(q_pos) & (q_pos < float(q_thresh)) & np.isfinite(mean_z) & (mean_z > 0.0))
        sig_neg = np.flatnonzero(np.isfinite(q_neg) & (q_neg < float(q_thresh)) & np.isfinite(mean_z) & (mean_z < 0.0))
        pos_idx = np.flatnonzero(np.isfinite(mean_z) & (mean_z > 0.0))
        sig_order = sig_pos[np.argsort(-mean_z[sig_pos])] if sig_pos.size > 0 else np.asarray([], dtype=np.int32)
        sig_take = sig_order[: int(min(int(top_n_cap), int(sig_order.size)))]
        top_items: list[dict[str, object]] = []
        for col in sig_take:
            reg = roi_registry[int(col)]
            top_items.append(
                {
                    "roi_idx": int(reg["roi_id"]),
                    "roi_name": str(reg["roi_name"]),
                    "family17": str(reg["family17"]),
                    "canonical8": str(reg["canonical8"]),
                    "fisher_z": float(mean_z[int(col)]),
                    "q_positive": float(q_pos[int(col)]),
                }
            )
        if pos_idx.size == 0:
            status = "unassigned"
            dom = None
            top1_network = "unassigned"
            top2_network = "unassigned"
            top1_count = 0
            top2_count = 0
            margin = 0
            n_top = 0
            unassigned_reason: str | None = "no_positive_roi"
        else:
            order = pos_idx[np.argsort(-mean_z[pos_idx])]
            n_top = int(min(int(top_n_cap), int(order.size)))
            top_idx = order[:n_top]
            count_by_network: dict[str, int] = {}
            score_sum_by_network: dict[str, float] = {}
            for col in top_idx:
                reg = roi_registry[int(col)]
                net = str(reg["canonical8"])
                count_by_network[net] = int(count_by_network.get(net, 0) + 1)
                score_sum_by_network[net] = float(score_sum_by_network.get(net, 0.0) + float(mean_z[int(col)]))
            top1_network, top1_count, top2_network, top2_count = top_two_from_counts(count_by_network, score_sum_by_network)
            margin = int(top1_count - top2_count)
            status = "dominant"
            dom = str(top1_network)
            unassigned_reason = None
        row_out: dict[str, object] = {
            "cluster_id": int(cidx + 1),
            "status": status,
            "top1_network": str(top1_network),
            "top2_network": str(top2_network),
            "top1_count": int(top1_count),
            "top2_count": int(top2_count),
            "margin_count": int(margin),
            "n_significant_positive": int(sig_pos.size),
            "n_significant_negative": int(sig_neg.size),
            "n_significant_any": int(sig_pos.size + sig_neg.size),
            "n_top": int(n_top),
            "top1_fraction": float(top1_count / n_top) if n_top > 0 else 0.0,
            "top2_fraction": float(top2_count / n_top) if n_top > 0 else 0.0,
            "top_positive_selected": top_items,
            "n_subjects_non_nan_median": int(np.nanmedian(n_vec)) if np.isfinite(np.nanmedian(n_vec)) else 0,
        }
        if unassigned_reason is not None:
            row_out["unassigned_reason"] = str(unassigned_reason)
        rows.append(row_out)
        dominant.append(dom)
    return rows, dominant, mean_profile


def vertex_roi_profile_subjectwise(
    *,
    feature_bank: dict[str, np.ndarray],
    subjects: list[str],
    support_mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    idx = np.flatnonzero(support_mask).astype(np.int32)
    stack = np.stack([feature_bank[s][idx, :] for s in subjects], axis=0).astype(np.float32)
    mean_profile = np.nanmean(stack, axis=0).astype(np.float32)
    mean_profile[~np.isfinite(mean_profile)] = np.nan
    return stack, mean_profile, idx


def build_global_top_connections(
    *,
    profile_3d: np.ndarray,
    support_indices: np.ndarray,
    roi_registry: list[dict[str, object]],
    q_thresh: float,
    top_n_cap: int,
) -> dict[str, object]:
    pos_rows: list[dict[str, object]] = []
    neg_rows: list[dict[str, object]] = []
    for local_vidx, global_vidx in enumerate(support_indices):
        values = profile_3d[:, local_vidx, :]
        mean_z, _t_stat, p_pos, p_neg, _n = one_sample_one_sided_vector(values)
        q_pos = bh_fdr(p_pos)
        q_neg = bh_fdr(p_neg)
        sig_pos = np.flatnonzero(np.isfinite(q_pos) & (q_pos < float(q_thresh)) & np.isfinite(mean_z) & (mean_z > 0.0))
        sig_neg = np.flatnonzero(np.isfinite(q_neg) & (q_neg < float(q_thresh)) & np.isfinite(mean_z) & (mean_z < 0.0))
        for col in sig_pos:
            reg = roi_registry[int(col)]
            pos_rows.append(
                {
                    "vertex_index": int(global_vidx),
                    "roi_idx": int(reg["roi_id"]),
                    "roi_name": str(reg["roi_name"]),
                    "canonical8": str(reg["canonical8"]),
                    "sign": "positive",
                    "fisher_z": float(mean_z[int(col)]),
                    "q_value": float(q_pos[int(col)]),
                }
            )
        for col in sig_neg:
            reg = roi_registry[int(col)]
            neg_rows.append(
                {
                    "vertex_index": int(global_vidx),
                    "roi_idx": int(reg["roi_id"]),
                    "roi_name": str(reg["roi_name"]),
                    "canonical8": str(reg["canonical8"]),
                    "sign": "negative",
                    "fisher_z": float(mean_z[int(col)]),
                    "q_value": float(q_neg[int(col)]),
                }
            )
    pos_rows.sort(key=lambda r: float(r["fisher_z"]), reverse=True)
    neg_rows.sort(key=lambda r: float(r["fisher_z"]))
    pos_take = pos_rows[: int(min(int(top_n_cap), len(pos_rows)))]
    neg_take = neg_rows[: int(min(int(top_n_cap), len(neg_rows)))]
    return {
        "top_n_cap": int(top_n_cap),
        "q_threshold": float(q_thresh),
        "n_candidates_positive": int(len(pos_rows)),
        "n_candidates_negative": int(len(neg_rows)),
        "top_positive": pos_take,
        "top_negative": neg_take,
    }


def assignment_status_for_vertices_top(
    *,
    profile_3d: np.ndarray,
    support_indices: np.ndarray,
    roi_registry: list[dict[str, object]],
    q_thresh: float,
    top_n_cap: int,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for local_vidx, global_vidx in enumerate(support_indices):
        values = profile_3d[:, local_vidx, :]
        mean_z, _t_stat, p_pos, p_neg, n_vec = one_sample_one_sided_vector(values)
        q_pos = bh_fdr(p_pos)
        q_neg = bh_fdr(p_neg)
        sig_pos = np.flatnonzero(np.isfinite(q_pos) & (q_pos < float(q_thresh)) & np.isfinite(mean_z) & (mean_z > 0.0))
        sig_neg = np.flatnonzero(np.isfinite(q_neg) & (q_neg < float(q_thresh)) & np.isfinite(mean_z) & (mean_z < 0.0))
        pos_idx = np.flatnonzero(np.isfinite(mean_z) & (mean_z > 0.0))
        if pos_idx.size == 0:
            rows.append(
                {
                    "vertex_index": int(global_vidx),
                    "status": "unassigned",
                    "unassigned_reason": "no_positive_roi",
                    "top1_network": "unassigned",
                    "top2_network": "unassigned",
                    "top1_count": 0,
                    "top2_count": 0,
                    "margin_count": 0,
                    "n_significant_positive": 0,
                    "n_significant_negative": int(sig_neg.size),
                    "n_significant_any": int(sig_neg.size),
                    "n_top": 0,
                    "n_subjects_non_nan_median": int(np.nanmedian(n_vec)) if np.isfinite(np.nanmedian(n_vec)) else 0,
                }
            )
            continue
        order = pos_idx[np.argsort(-mean_z[pos_idx])]
        n_top = int(min(int(top_n_cap), int(order.size)))
        top_idx = order[:n_top]
        count_by_network: dict[str, int] = {}
        score_sum_by_network: dict[str, float] = {}
        for col in top_idx:
            net = str(roi_registry[int(col)]["canonical8"])
            count_by_network[net] = int(count_by_network.get(net, 0) + 1)
            score_sum_by_network[net] = float(score_sum_by_network.get(net, 0.0) + float(mean_z[int(col)]))
        top1_network, top1_count, top2_network, top2_count = top_two_from_counts(count_by_network, score_sum_by_network)
        rows.append(
            {
                "vertex_index": int(global_vidx),
                "status": "dominant",
                "top1_network": str(top1_network),
                "top2_network": str(top2_network),
                "top1_count": int(top1_count),
                "top2_count": int(top2_count),
                "margin_count": int(top1_count - top2_count),
                "n_significant_positive": int(sig_pos.size),
                "n_significant_negative": int(sig_neg.size),
                "n_significant_any": int(sig_pos.size + sig_neg.size),
                "n_top": int(n_top),
                "top1_fraction": float(top1_count / n_top) if n_top > 0 else 0.0,
                "top2_fraction": float(top2_count / n_top) if n_top > 0 else 0.0,
                "n_subjects_non_nan_median": int(np.nanmedian(n_vec)) if np.isfinite(np.nanmedian(n_vec)) else 0,
            }
        )
    return rows


def assignment_status_for_vertices_all(
    *,
    profile_3d: np.ndarray,
    support_indices: np.ndarray,
    roi_registry: list[dict[str, object]],
    q_thresh: float,
) -> list[dict[str, object]]:
    cols_by_network, _network_names = split_kong2022_columns_by_network(roi_registry)
    rows: list[dict[str, object]] = []
    for local_vidx, global_vidx in enumerate(support_indices):
        values = profile_3d[:, local_vidx, :]
        mean_z, _t_stat, p_pos, p_neg, n_vec = one_sample_one_sided_vector(values)
        q_pos = bh_fdr(p_pos)
        q_neg = bh_fdr(p_neg)
        sig_pos = np.flatnonzero(np.isfinite(q_pos) & (q_pos < float(q_thresh)) & np.isfinite(mean_z) & (mean_z > 0.0))
        sig_neg = np.flatnonzero(np.isfinite(q_neg) & (q_neg < float(q_thresh)) & np.isfinite(mean_z) & (mean_z < 0.0))
        n_sig_any = int(sig_pos.size + sig_neg.size)
        score_by_network: dict[str, float] = {}
        for net, cols in cols_by_network.items():
            vals = mean_z[np.asarray(cols, dtype=np.int32)]
            score_by_network[str(net)] = float(np.nanmean(vals)) if np.isfinite(vals).any() else float("nan")
        top1_network, top1_score, top2_network, top2_score = top_two_from_scores(score_by_network)
        rows.append(
            {
                "vertex_index": int(global_vidx),
                "status": "dominant",
                "top1_network": str(top1_network),
                "top2_network": str(top2_network),
                "top1_score": float(top1_score),
                "top2_score": float(top2_score),
                "margin_score": float(top1_score - top2_score),
                "n_significant_positive": int(sig_pos.size),
                "n_significant_negative": int(sig_neg.size),
                "n_significant_any": int(n_sig_any),
                "n_subjects_non_nan_median": int(np.nanmedian(n_vec)) if np.isfinite(np.nanmedian(n_vec)) else 0,
            }
        )
    return rows


def functional_atlas_from_clusters(
    *,
    group_labels_final: np.ndarray,
    status_rows: list[dict[str, object]],
    network_names: list[str],
    color_map: dict[str, tuple[int, int, int, int]],
) -> tuple[np.ndarray, dict[str, object], list[tuple[int, str, tuple[int, int, int, int]]]]:
    net_to_key = {str(n): int(i + 1) for i, n in enumerate(network_names)}
    key_unassigned = int(len(network_names) + 1)
    cluster_to_key: dict[int, int] = {}

    for row in status_rows:
        cid = int(row["cluster_id"])
        status = str(row["status"])
        top1 = str(row["top1_network"])
        if status == "dominant" or status == "mixed":
            if top1 not in net_to_key:
                raise KeyError(f"Unknown network in assignment status: {top1}")
            cluster_to_key[cid] = int(net_to_key[top1])
        elif status == "unassigned":
            cluster_to_key[cid] = key_unassigned
        else:
            raise ValueError(f"Unknown status in assignment status: {status}")

    out = np.zeros_like(group_labels_final, dtype=np.int32)
    for cid, key in cluster_to_key.items():
        out[group_labels_final == int(cid)] = int(key)

    label_items: list[tuple[int, str, tuple[int, int, int, int]]] = [(0, "Null", (158, 158, 158, 255))]
    for i, n in enumerate(network_names, start=1):
        label_items.append((int(i), str(n), color_map.get(str(n), (80, 80, 80, 255))))
    label_items.append((key_unassigned, "unassigned", (20, 20, 20, 255)))

    meta = {
        "network_names": [str(x) for x in network_names],
        "network_label_keys": {str(n): int(i + 1) for i, n in enumerate(network_names)},
        "unassigned_label_key": int(key_unassigned),
        "cluster_to_functional_label_key": {str(k): int(v) for k, v in sorted(cluster_to_key.items())},
    }
    return out, meta, label_items


def functional_atlas_from_vertices(
    *,
    n_vertices: int,
    status_rows: list[dict[str, object]],
    network_names: list[str],
    color_map: dict[str, tuple[int, int, int, int]],
) -> tuple[np.ndarray, dict[str, object], list[tuple[int, str, tuple[int, int, int, int]]]]:
    net_to_key = {str(n): int(i + 1) for i, n in enumerate(network_names)}
    key_unassigned = int(len(network_names) + 1)
    out = np.zeros((int(n_vertices),), dtype=np.int32)
    for row in status_rows:
        vidx = int(row["vertex_index"])
        status = str(row.get("status", "unassigned"))
        top1 = str(row.get("top1_network", "unassigned"))
        if status == "dominant" and top1 in net_to_key:
            out[vidx] = int(net_to_key[top1])
        else:
            out[vidx] = int(key_unassigned)

    label_items: list[tuple[int, str, tuple[int, int, int, int]]] = [(0, "Null", (158, 158, 158, 255))]
    for i, n in enumerate(network_names, start=1):
        label_items.append((int(i), str(n), color_map.get(str(n), (80, 80, 80, 255))))
    label_items.append((key_unassigned, "unassigned", (20, 20, 20, 255)))

    meta = {
        "assignment_mode": "vertex_level",
        "network_names": [str(x) for x in network_names],
        "network_label_keys": {str(n): int(i + 1) for i, n in enumerate(network_names)},
        "unassigned_label_key": int(key_unassigned),
    }
    return out, meta, label_items


def main() -> int:
    parser = argparse.ArgumentParser(description="Run merged canonical group hippocampal atlas (v3 plan).")
    parser.add_argument("--shared-root", default=str(DEFAULT_SHARED_ROOT))
    parser.add_argument("--out-root", default=str(DEFAULT_OUT_ROOT))
    parser.add_argument("--subjects-manifest", default=str(DEFAULT_SUBJECTS))
    parser.add_argument("--hippunfold-batch-root", default=str(DEFAULT_HIPPUNFOLD_BATCH_ROOT))
    parser.add_argument("--methods", nargs="+", default=METHODS_DEFAULT, choices=METHOD_CHOICES)
    parser.add_argument("--atlases", nargs="+", default=ATLASES)
    parser.add_argument("--smoothings", nargs="+", default=SMOOTHINGS)
    parser.add_argument("--hemis", nargs="+", default=HEMIS)
    parser.add_argument("--coverage-thresholds", nargs="+", type=float, default=[0.70, 0.80, 0.90])
    parser.add_argument("--k-min", type=int, default=2)
    parser.add_argument("--k-max", type=int, default=10)
    parser.add_argument("--instability-resamples", type=int, default=100)
    parser.add_argument("--null-resamples", type=int, default=100)
    parser.add_argument("--seed", type=int, default=20260420)
    parser.add_argument("--v-min-fraction", type=float, default=0.03)
    parser.add_argument("--max-components-threshold", type=int, default=3)
    parser.add_argument("--min-profile-coherence", type=float, default=0.0)
    parser.add_argument("--resume", action="store_true", help="Skip combos that already have group_atlas_manifest.json.")
    parser.add_argument("--k-selection-mode", default="experimental", choices=["experimental"])
    parser.add_argument("--overview-threshold", type=float, default=0.80)
    parser.add_argument(
        "--jobs",
        type=int,
        default=1,
        help="Reserved for pipeline compatibility; current implementation executes combos in-process.",
    )
    parser.add_argument(
        "--group-average-surfaces-dir",
        default=str(DEFAULT_GROUP_AVG_SURFACES_DIR),
        help="Directory containing group-average midthickness surf.gii files named "
             "group-average_hemi-{L|R}_space-corobl_den-512_label-hipp_midthickness.surf.gii. "
             "When present, these are used as the template geometry instead of subjects[0].",
    )
    args = parser.parse_args()

    shared_root = Path(args.shared_root).resolve()
    out_root = Path(args.out_root).resolve()
    hippunfold_batch_root = Path(args.hippunfold_batch_root).resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    subjects = load_subjects(Path(args.subjects_manifest).resolve())
    coverage_thresholds = [float(x) for x in args.coverage_thresholds]
    k_values = list(range(int(args.k_min), int(args.k_max) + 1))
    rng = np.random.RandomState(int(args.seed))
    split_seeds = [int(x) for x in rng.randint(1, 2_000_000_000, size=int(args.instability_resamples))]
    null_seeds = [int(x) for x in rng.randint(1, 2_000_000_000, size=int(args.null_resamples))]

    avg_surf_dir = Path(args.group_average_surfaces_dir).resolve()
    allowed_roots = [(REPO_ROOT / "outputs_migration").resolve(), (REPO_ROOT / "outputs_experiments").resolve()]
    if "present_group" in avg_surf_dir.parts:
        raise ValueError(
            f"Invalid --group-average-surfaces-dir={avg_surf_dir}: present_group paths are non-canonical. "
            f"Use a canonical outputs_migration or outputs_experiments group-average surface directory."
        )
    if not any(avg_surf_dir.is_relative_to(root) for root in allowed_roots):
        raise ValueError(
            f"Invalid --group-average-surfaces-dir={avg_surf_dir}: expected a path under one of {allowed_roots}."
        )
    avg_surf_paths = {
        hemi: avg_surf_dir / f"group-average_hemi-{hemi}_space-corobl_den-512_label-hipp_midthickness.surf.gii"
        for hemi in args.hemis
    }
    missing_avg = [str(p) for p in avg_surf_paths.values() if not p.exists()]
    if missing_avg:
        raise FileNotFoundError(
            "Missing required group-average surfaces. "
            "Run from outputs_migration/hipp_group_merged_canonical_atlas/group_average_surfaces only.\n"
            + "\n".join(missing_avg)
        )
    template_label = "group-average"
    surfaces = {hemi: avg_surf_paths[hemi] for hemi in args.hemis}
    topology_audit = verify_surface_topology_consistency(subjects, args.hemis, surfaces, hippunfold_batch_root)
    surface_meta: dict[str, dict[str, object]] = {}
    adjacency_by_hemi: dict[str, sparse.csr_matrix] = {}
    long_axis_order_by_hemi: dict[str, np.ndarray] = {}
    for hemi in args.hemis:
        coords, faces = load_surface(surfaces[hemi])
        adjacency_by_hemi[hemi] = build_surface_adjacency(faces, int(coords.shape[0]))
        long_axis_order_by_hemi[hemi] = long_axis_order(surfaces[hemi])
        surface_meta[hemi] = {
            "surface_path": str(surfaces[hemi]),
            "n_vertices": int(coords.shape[0]),
            "n_faces": int(faces.shape[0]),
        }

    if any(m in args.methods for m in ("group-intrinsic-spectral", "group-intrinsic-partial-spectral")):
        for smoothing in args.smoothings:
            for hemi in args.hemis:
                print(f"[check] intrinsic atlas-invariance smoothing={smoothing} hemi={hemi}", flush=True)
                verify_intrinsic_atlas_invariance(shared_root, subjects, smoothing, hemi, list(args.atlases))

    colors = load_cluster_colors()
    canonical_color_map = load_canonical_network_color_map()
    canonical_network_order_by_atlas: dict[str, list[str]] = {}

    for atlas in args.atlases:
        print(f"[check] canonical-network ordering atlas={atlas}", flush=True)
        canonical_network_order_by_atlas[atlas] = verify_canonical_network_order_consistency(shared_root, subjects, atlas)

    for method in args.methods:
        for atlas in args.atlases:
            if method in KONG2022_ONLY_METHODS and atlas != "kong2022":
                print(
                    f"[warn] skipping unsupported combo atlas={atlas} method={method} "
                    "(method is defined for kong2022 only)",
                    flush=True,
                )
                continue
            if atlas == "kong2022" and method in {"group-network-partial-spectral", "group-intrinsic-partial-spectral"}:
                print(
                    f"[warn] skipping unsupported combo atlas={atlas} method={method} "
                    "(partial methods are not defined for Kong2022 Route B)",
                    flush=True,
                )
                continue
            network_names = canonical_network_order_by_atlas[atlas]
            for smoothing in args.smoothings:
                for hemi in args.hemis:
                    n_vertices = int(surface_meta[hemi]["n_vertices"])
                    valid_masks = {s: subject_valid_mask(shared_root, s, hemi) for s in subjects}
                    coverage_count = np.sum(np.stack([valid_masks[s] for s in subjects], axis=0), axis=0).astype(np.int32)
                    coverage_fraction = np.mean(np.stack([valid_masks[s] for s in subjects], axis=0), axis=0).astype(np.float32)

                    feature_bank: dict[str, np.ndarray] = {}
                    actual_input_paths: dict[str, str] = {}
                    for s in subjects:
                        if is_network_method(method) or method in {"group-network-top", "group-network-all"}:
                            p = shared_root / f"sub-{s}" / "fc" / atlas / smoothing / f"hemi_{hemi}" / network_feature_filename(method)
                            actual_input_paths[s] = str(p)
                            feature_bank[s] = subject_network_feature(shared_root, s, atlas, smoothing, hemi, method)
                        else:
                            p = resolve_intrinsic_feature_path(shared_root, s, smoothing, hemi)
                            actual_input_paths[s] = str(p)
                            feature_bank[s] = subject_intrinsic_feature(shared_root, s, atlas, smoothing, hemi, valid_masks[s])

                    run_feature_banks = (
                        runwise_feature_bank(
                            method=method,
                            shared_root=shared_root,
                            subjects=subjects,
                            atlas=atlas,
                            smoothing=smoothing,
                            hemi=hemi,
                            valid_masks=valid_masks,
                        )
                        if not is_clusterless_method(method)
                        else {}
                    )

                    for tau in coverage_thresholds:
                        combo = Combo(method=method, atlas=atlas, smoothing=smoothing, hemi=hemi, coverage_threshold=float(tau))
                        print(
                            f"[combo] method={method} atlas={atlas} smoothing={smoothing} hemi={hemi} tau={tau:.2f}",
                            flush=True,
                        )
                        tau_tag = str(f"{tau:.2f}").replace(".", "p")
                        run_dir = out_root / method / atlas / smoothing / f"cov_{tau_tag}" / f"hemi_{hemi}"
                        run_dir.mkdir(parents=True, exist_ok=True)
                        manifest_path = run_dir / "group_atlas_manifest.json"
                        if args.resume and manifest_path.exists():
                            print(f"  [skip] existing manifest: {manifest_path}", flush=True)
                            continue

                        support = coverage_fraction >= float(tau)
                        n_support = int(np.sum(support))
                        v_min = int(math.ceil(float(args.v_min_fraction) * max(1, n_support)))
                        k_max_eff = min(max(k_values), n_support // max(1, v_min))
                        k_eff = [k for k in k_values if k <= k_max_eff and k <= n_support - 1]
                        k_selection_fields = [
                            "method",
                            "atlas",
                            "smoothing",
                            "hemi",
                            "coverage_threshold",
                            "k",
                            "n_subjects",
                            "n_support_vertices",
                            "V_min_effective",
                            "group_ari_mean",
                            "group_ari_se",
                            "subject_split_requested",
                            "subject_split_used",
                            "subject_split_skipped_overlap",
                            "subject_split_skipped_empty",
                            "run_aware_ari_mean",
                            "run_aware_ari_se",
                            "run_aware_n_pairs",
                            "run_aware_n_used",
                            "ARI_null_mean",
                            "ARI_null_5pct",
                            "ARI_weak_null_mean",
                            "B_null_requested",
                            "B_null_used_strong",
                            "B_null_used_weak",
                            "min_cluster_size",
                            "max_components_per_cluster",
                            "min_profile_coherence",
                            "mean_profile_coherence",
                            "nontrivial_min_size_ok",
                            "nontrivial_connectedness_ok",
                            "nontrivial_coherence_ok",
                            "local_max",
                            "within_1se_best",
                            "selected_nontrivial",
                        ]

                        np.save(run_dir / "group_support_mask.npy", support.astype(np.uint8))
                        np.save(run_dir / "subject_coverage_count.npy", coverage_count.astype(np.int32))
                        np.save(run_dir / "subject_coverage_fraction.npy", coverage_fraction.astype(np.float32))

                        if is_clusterless_method(method):
                            if atlas != "kong2022":
                                raise RuntimeError(f"Internal error: clusterless method requires kong2022, got atlas={atlas}")
                            roi_registry = load_kong2022_roi_registry()
                            vertex_profile_3d, vertex_profile_mean, support_indices = vertex_roi_profile_subjectwise(
                                feature_bank=feature_bank,
                                subjects=subjects,
                                support_mask=support,
                            )
                            if method == "group-network-top":
                                vertex_status_rows = assignment_status_for_vertices_top(
                                    profile_3d=vertex_profile_3d,
                                    support_indices=support_indices,
                                    roi_registry=roi_registry,
                                    q_thresh=0.05,
                                    top_n_cap=20,
                                )
                                confidence_metric = "topn_count"
                            elif method == "group-network-all":
                                vertex_status_rows = assignment_status_for_vertices_all(
                                    profile_3d=vertex_profile_3d,
                                    support_indices=support_indices,
                                    roi_registry=roi_registry,
                                    q_thresh=0.05,
                                )
                                confidence_metric = "network_mean_score"
                            else:
                                raise ValueError(f"Unknown clusterless method: {method}")

                            global_top = build_global_top_connections(
                                profile_3d=vertex_profile_3d,
                                support_indices=support_indices,
                                roi_registry=roi_registry,
                                q_thresh=0.05,
                                top_n_cap=20,
                            )
                            functional_labels, functional_meta, functional_label_items = functional_atlas_from_vertices(
                                n_vertices=n_vertices,
                                status_rows=vertex_status_rows,
                                network_names=network_names,
                                color_map=canonical_color_map,
                            )
                            np.save(run_dir / "group_functional_atlas_labels.npy", functional_labels.astype(np.int32))
                            save_json(run_dir / "group_functional_atlas_label_map.json", functional_meta)
                            np.save(run_dir / "vertex_roi_profile.npy", vertex_profile_mean.astype(np.float32))
                            np.save(run_dir / "vertex_support_indices.npy", support_indices.astype(np.int32))
                            save_json(run_dir / "vertex_assignment_status.json", {"items": vertex_status_rows})
                            save_json(run_dir / "global_top_connections.json", global_top)
                            save_json(run_dir / "k_selection_summary.json", {"rows": [], "k_final": None, "mode": "clusterless"})
                            write_tsv(run_dir / "k_selection_summary.tsv", [], k_selection_fields)

                            func_lbl = make_named_label_gifti(functional_labels.astype(np.int32), functional_label_items)
                            func_lbl_path = (
                                run_dir
                                / "template"
                                / "workbench_assets"
                                / f"{template_label}_hemi-{hemi}_space-corobl_den-512_label-group_functional_atlas_{method}_{atlas}_{smoothing}_cov-{tau_tag}.label.gii"
                            )
                            func_lbl_path.parent.mkdir(parents=True, exist_ok=True)
                            nib.save(func_lbl, str(func_lbl_path))

                            manifest = {
                                "status": "ok",
                                "method": method,
                                "atlas": atlas,
                                "smoothing": smoothing,
                                "hemi": hemi,
                                "assignment_mode": "vertex_level",
                                "confidence_metric": confidence_metric,
                                "coverage_threshold": float(tau),
                                "chosen_k_final": None,
                                "q_threshold": 0.05,
                                "top_n_cap": 20,
                                "actual_consumed_input_paths": actual_input_paths,
                                "feature_family": "vertex_to_kong2022_400roi_fc_fisherz",
                                "fc_mode": "pearson_vertex_to_roi",
                                "n_subjects": int(len(subjects)),
                                "subjects": [f"sub-{s}" for s in subjects],
                                "n_support_vertices": int(n_support),
                                "n_excluded_vertices": int(n_vertices - n_support),
                                "group_support_mask": str((run_dir / "group_support_mask.npy").resolve()),
                                "subject_coverage_count": str((run_dir / "subject_coverage_count.npy").resolve()),
                                "subject_coverage_fraction": str((run_dir / "subject_coverage_fraction.npy").resolve()),
                                "group_functional_atlas_labels": str((run_dir / "group_functional_atlas_labels.npy").resolve()),
                                "group_functional_atlas_label_map": str((run_dir / "group_functional_atlas_label_map.json").resolve()),
                                "vertex_roi_profile": str((run_dir / "vertex_roi_profile.npy").resolve()),
                                "vertex_support_indices": str((run_dir / "vertex_support_indices.npy").resolve()),
                                "vertex_assignment_status": str((run_dir / "vertex_assignment_status.json").resolve()),
                                "global_top_connections": str((run_dir / "global_top_connections.json").resolve()),
                                "cluster_network_profile": None,
                                "cluster_dominant_network": None,
                                "cluster_assignment_status": None,
                                "cluster_roi_profile": None,
                                "cluster_roi_top20": None,
                                "cluster_roi_registry": str((run_dir / "cluster_roi_registry.tsv").resolve()),
                                "k_selection_summary_tsv": str((run_dir / "k_selection_summary.tsv").resolve()),
                                "k_selection_summary_json": str((run_dir / "k_selection_summary.json").resolve()),
                                "template_subject": template_label,
                                "template_left_or_right_surface": str(surfaces[hemi]),
                                "template_spec_path": str(
                                    infer_spec_path(subjects[0], infer_surface_path(subjects[0], hemi, hippunfold_batch_root))
                                ),
                                "template_cluster_label": None,
                                "template_functional_atlas_label": str(func_lbl_path.resolve()),
                            }
                            write_tsv(
                                run_dir / "cluster_roi_registry.tsv",
                                roi_registry,
                                ["roi_id", "roi_name", "hemi", "family17", "subregion", "rank", "canonical8"],
                            )
                            save_json(run_dir / "group_feature_summary.json", {"network_names": network_names, "N_ref": int(len(network_names))})
                            save_json(manifest_path, manifest)
                            continue

                        if not k_eff:
                            skip_reason = "no_valid_k_for_support"
                            skip_detail = (
                                f"No valid K for combo={combo} n_support={n_support} "
                                f"v_min={v_min} k_max_eff={k_max_eff}"
                            )
                            write_tsv(run_dir / "k_selection_summary.tsv", [], k_selection_fields)
                            save_json(
                                run_dir / "k_selection_summary.json",
                                {
                                    "rows": [],
                                    "k_final": None,
                                    "skip_reason": skip_reason,
                                    "skip_detail": skip_detail,
                                },
                            )
                            manifest = {
                                "status": "skipped_no_valid_k",
                                "skip_reason": skip_reason,
                                "skip_detail": skip_detail,
                                "method": method,
                                "atlas": atlas,
                                "smoothing": smoothing,
                                "hemi": hemi,
                                "k_selection_mode": str(args.k_selection_mode),
                                "coverage_threshold": float(tau),
                                "V_min_fraction": float(args.v_min_fraction),
                                "V_min_effective": int(v_min),
                                "candidate_k_range_requested": [int(min(k_values)), int(max(k_values))],
                                "candidate_k_range_effective": [],
                                "chosen_k_final": None,
                                "subject_split_seeds": split_seeds,
                                "null_seeds": null_seeds,
                                "n_subjects": int(len(subjects)),
                                "subjects": [f"sub-{s}" for s in subjects],
                                "n_support_vertices": int(n_support),
                                "n_excluded_vertices": int(n_vertices - n_support),
                                "actual_consumed_input_paths": actual_input_paths,
                                "nontriviality_thresholds": {
                                    "V_min_fraction": float(args.v_min_fraction),
                                    "max_components_threshold": int(args.max_components_threshold),
                                    "min_profile_coherence": float(args.min_profile_coherence),
                                },
                                "density_evidence": {
                                    "surface_path": str(surfaces[hemi]),
                                    "n_vertices": int(surface_meta[hemi]["n_vertices"]),
                                    "n_faces": int(surface_meta[hemi]["n_faces"]),
                                },
                                "surface_topology_audit": topology_audit[hemi],
                                "group_support_mask": str((run_dir / "group_support_mask.npy").resolve()),
                                "subject_coverage_count": str((run_dir / "subject_coverage_count.npy").resolve()),
                                "subject_coverage_fraction": str((run_dir / "subject_coverage_fraction.npy").resolve()),
                                "k_selection_summary_tsv": str((run_dir / "k_selection_summary.tsv").resolve()),
                                "k_selection_summary_json": str((run_dir / "k_selection_summary.json").resolve()),
                            }
                            save_json(manifest_path, manifest)
                            print(f"  [skip] {skip_detail}", flush=True)
                            continue

                        rows: list[dict[str, object]] = []
                        labels_by_k: dict[int, np.ndarray] = {}

                        for k in k_eff:
                            print(f"  [k] {k}", flush=True)
                            labels_active, labels_full = cluster_for_subset(
                                method=method,
                                feature_bank=feature_bank,
                                subjects_subset=subjects,
                                support_mask=support,
                                adjacency_full=adjacency_by_hemi[hemi],
                                k=k,
                                seed=int(args.seed + 11 * k),
                            )
                            labels_by_k[int(k)] = labels_full
                            np.save(run_dir / f"group_labels_k{k}.npy", labels_full.astype(np.int32))

                            sizes = [int(np.sum(labels_active == c)) for c in range(1, k + 1)]
                            min_size = int(min(sizes)) if sizes else 0
                            comp = connected_components_per_label(labels_active, adjacency_by_hemi[hemi][np.flatnonzero(support)][:, np.flatnonzero(support)])
                            max_components = int(max(comp.values())) if comp else 0
                            group_features_active = group_feature_matrix(
                                method=method,
                                feature_bank=feature_bank,
                                subjects_subset=subjects,
                                support_mask=support,
                            )
                            min_coh, mean_coh = cluster_profile_coherence(group_features_active, labels_active, int(k))

                            ari_obj = subject_split_ari(
                                method=method,
                                feature_bank=feature_bank,
                                valid_masks_by_subject=valid_masks,
                                subjects=subjects,
                                tau=float(tau),
                                adjacency_full=adjacency_by_hemi[hemi],
                                k=int(k),
                                seeds=split_seeds,
                            )
                            ari_vals = list(ari_obj["values"])
                            if not ari_vals:
                                raise RuntimeError(f"No valid ARI resamples for combo={combo} k={k}")
                            ari_arr = np.asarray(ari_vals, dtype=np.float64)
                            ari_mean = float(np.mean(ari_arr))
                            ari_se = float(np.std(ari_arr, ddof=1) / np.sqrt(ari_arr.size)) if ari_arr.size > 1 else 0.0

                            weak_obj = weak_null_ari(int(k), int(n_support), null_seeds)
                            weak = list(weak_obj["values"])
                            strong_obj = strong_null_ari(
                                method=method,
                                feature_bank=feature_bank,
                                valid_masks_by_subject=valid_masks,
                                subjects=subjects,
                                tau=float(tau),
                                adjacency_full=adjacency_by_hemi[hemi],
                                k=int(k),
                                seeds=null_seeds,
                            )
                            strong = list(strong_obj["values"])
                            strong_arr = np.asarray(strong, dtype=np.float64) if strong else np.array([0.0], dtype=np.float64)
                            run_obj = run_aware_ari(
                                method=method,
                                run_feature_banks=run_feature_banks,
                                subjects=subjects,
                                support_mask=support,
                                adjacency_full=adjacency_by_hemi[hemi],
                                k=int(k),
                                seed_base=int(args.seed),
                            )
                            run_vals = list(run_obj["values"])
                            run_arr = np.asarray(run_vals, dtype=np.float64) if run_vals else np.array([np.nan], dtype=np.float64)
                            run_ari_mean = float(np.nanmean(run_arr))
                            run_ari_se = (
                                float(np.nanstd(run_arr, ddof=1) / np.sqrt(np.sum(np.isfinite(run_arr))))
                                if np.sum(np.isfinite(run_arr)) > 1
                                else 0.0
                            )

                            rows.append(
                                {
                                    "method": method,
                                    "atlas": atlas,
                                    "smoothing": smoothing,
                                    "hemi": hemi,
                                    "coverage_threshold": float(tau),
                                    "k": int(k),
                                    "n_subjects": int(len(subjects)),
                                    "n_support_vertices": int(n_support),
                                    "V_min_effective": int(v_min),
                                    "group_ari_mean": ari_mean,
                                    "group_ari_se": ari_se,
                                    "subject_split_requested": int(ari_obj["n_requested"]),
                                    "subject_split_used": int(ari_obj["n_used"]),
                                    "subject_split_skipped_overlap": int(ari_obj["n_skipped_overlap"]),
                                    "subject_split_skipped_empty": int(ari_obj["n_skipped_empty"]),
                                    "run_aware_ari_mean": run_ari_mean,
                                    "run_aware_ari_se": run_ari_se,
                                    "run_aware_n_pairs": int(run_obj["n_pairs"]),
                                    "run_aware_n_used": int(run_obj["n_used"]),
                                    "ARI_null_mean": float(np.mean(strong_arr)),
                                    "ARI_null_5pct": float(np.percentile(strong_arr, 5.0)),
                                    "ARI_weak_null_mean": float(np.mean(np.asarray(weak, dtype=np.float64))),
                                    "B_null_requested": int(args.null_resamples),
                                    "B_null_used_strong": int(strong_obj["n_used"]),
                                    "B_null_used_weak": int(weak_obj["n_used"]),
                                    "min_cluster_size": int(min_size),
                                    "max_components_per_cluster": int(max_components),
                                    "min_profile_coherence": float(min_coh),
                                    "mean_profile_coherence": float(mean_coh),
                                    "nontrivial_min_size_ok": int(min_size >= v_min),
                                    "nontrivial_connectedness_ok": int(max_components <= int(args.max_components_threshold)),
                                    "nontrivial_coherence_ok": int(min_coh >= float(args.min_profile_coherence)),
                                }
                            )

                        rows = sorted(rows, key=lambda r: int(r["k"]))
                        means = [float(r["group_ari_mean"]) for r in rows]
                        ses = [float(r["group_ari_se"]) for r in rows]
                        locals_ = local_maxima_flags(means)
                        best_idx = int(np.argmax(np.asarray(means, dtype=np.float64)))
                        cutoff = float(means[best_idx] - ses[best_idx])
                        for i, row in enumerate(rows):
                            row["local_max"] = int(locals_[i])
                            row["within_1se_best"] = int(float(row["group_ari_mean"]) >= cutoff - 1e-12)
                            row["selected_nontrivial"] = int(
                                int(row["nontrivial_min_size_ok"]) == 1
                                and int(row["nontrivial_connectedness_ok"]) == 1
                                and int(row["nontrivial_coherence_ok"]) == 1
                            )

                        survivors = [
                            r
                            for r in rows
                            if int(r["local_max"]) == 1
                            and int(r["within_1se_best"]) == 1
                            and int(r["selected_nontrivial"]) == 1
                        ]
                        if not survivors:
                            skip_reason = "no_k_survived_selection_constraints"
                            skip_detail = f"No K survived selection constraints for combo={combo}"
                            write_tsv(run_dir / "k_selection_summary.tsv", rows, k_selection_fields)
                            save_json(
                                run_dir / "k_selection_summary.json",
                                {
                                    "rows": rows,
                                    "k_final": None,
                                    "skip_reason": skip_reason,
                                    "skip_detail": skip_detail,
                                },
                            )
                            manifest = {
                                "status": "skipped_no_k_survived_constraints",
                                "skip_reason": skip_reason,
                                "skip_detail": skip_detail,
                                "method": method,
                                "atlas": atlas,
                                "smoothing": smoothing,
                                "hemi": hemi,
                                "k_selection_mode": str(args.k_selection_mode),
                                "coverage_threshold": float(tau),
                                "V_min_fraction": float(args.v_min_fraction),
                                "V_min_effective": int(v_min),
                                "candidate_k_range_requested": [int(min(k_values)), int(max(k_values))],
                                "candidate_k_range_effective": [int(min(k_eff)), int(max(k_eff))],
                                "chosen_k_final": None,
                                "subject_split_seeds": split_seeds,
                                "null_seeds": null_seeds,
                                "n_subjects": int(len(subjects)),
                                "subjects": [f"sub-{s}" for s in subjects],
                                "n_support_vertices": int(n_support),
                                "n_excluded_vertices": int(n_vertices - n_support),
                                "actual_consumed_input_paths": actual_input_paths,
                                "nontriviality_thresholds": {
                                    "V_min_fraction": float(args.v_min_fraction),
                                    "max_components_threshold": int(args.max_components_threshold),
                                    "min_profile_coherence": float(args.min_profile_coherence),
                                },
                                "density_evidence": {
                                    "surface_path": str(surfaces[hemi]),
                                    "n_vertices": int(surface_meta[hemi]["n_vertices"]),
                                    "n_faces": int(surface_meta[hemi]["n_faces"]),
                                },
                                "surface_topology_audit": topology_audit[hemi],
                                "group_support_mask": str((run_dir / "group_support_mask.npy").resolve()),
                                "subject_coverage_count": str((run_dir / "subject_coverage_count.npy").resolve()),
                                "subject_coverage_fraction": str((run_dir / "subject_coverage_fraction.npy").resolve()),
                                "k_selection_summary_tsv": str((run_dir / "k_selection_summary.tsv").resolve()),
                                "k_selection_summary_json": str((run_dir / "k_selection_summary.json").resolve()),
                            }
                            save_json(manifest_path, manifest)
                            print(f"  [skip] {skip_detail}", flush=True)
                            continue
                        k_final = int(sorted(survivors, key=lambda x: int(x["k"]))[0]["k"])
                        labels_final = reorder_labels_by_long_axis(
                            labels_by_k[k_final],
                            long_axis_order_by_hemi[hemi],
                        )
                        fixed_cluster_order = cluster_ids_by_long_axis(
                            labels_final,
                            long_axis_order_by_hemi[hemi],
                        )

                        profile: np.ndarray
                        status_rows: list[dict[str, object]]
                        dominant: list[str | None]
                        roi_profile: np.ndarray | None = None
                        roi_profile_3d: np.ndarray | None = None
                        roi_profile_subjects_used: list[str] | None = None
                        roi_registry: list[dict[str, object]] | None = None
                        roi_top20: dict[str, object] | None = None
                        if atlas == "kong2022" and method == "group-intrinsic-spectral":
                            roi_profile_3d, roi_profile, roi_registry, roi_profile_subjects_used = cluster_roi_profile_subjectwise(
                                group_labels_final=labels_final,
                                support_mask=support,
                                subjects=subjects,
                                shared_root=shared_root,
                                smoothing=smoothing,
                                hemi=hemi,
                                valid_masks_by_subject=valid_masks,
                            )
                            status_rows, dominant, _profile400_mean = assignment_status_from_cluster_roi_topn(
                                profile_3d=roi_profile_3d,
                                roi_registry=roi_registry,
                                q_thresh=0.05,
                                top_n_cap=20,
                            )
                            cols_by_network, _network_names_from_registry = split_kong2022_columns_by_network(roi_registry)
                            profile = np.full((int(roi_profile.shape[0]), int(len(network_names))), np.nan, dtype=np.float32)
                            for net_idx, net_name in enumerate(network_names):
                                cols = cols_by_network.get(str(net_name), [])
                                if not cols:
                                    continue
                                profile[:, net_idx] = np.nanmean(roi_profile[:, np.asarray(cols, dtype=np.int32)], axis=1).astype(np.float32)
                        else:
                            profile = cluster_network_profile(
                                group_labels_final=labels_final,
                                support_mask=support,
                                subjects=subjects,
                                shared_root=shared_root,
                                atlas=atlas,
                                smoothing=smoothing,
                                hemi=hemi,
                                valid_masks_by_subject=valid_masks,
                                profile_fc_mode=(
                                    "partial_networks_only_one_sided"
                                    if method in {"group-network-partial-spectral", "group-intrinsic-partial-spectral"}
                                    else "pearson"
                                ),
                            )
                            status_rows, dominant = assignment_status_from_profile(profile, network_names)
                        if atlas == "kong2022" and roi_profile is None:
                            roi_profile, roi_registry = cluster_roi_profile(
                                group_labels_final=labels_final,
                                support_mask=support,
                                subjects=subjects,
                                shared_root=shared_root,
                                smoothing=smoothing,
                                hemi=hemi,
                                valid_masks_by_subject=valid_masks,
                            )
                        if atlas == "kong2022":
                            assert roi_profile is not None and roi_registry is not None
                            roi_top20 = cluster_roi_top_k(roi_profile, roi_registry, k_top=20)

                        np.save(run_dir / "group_labels_final.npy", labels_final.astype(np.int32))
                        np.save(run_dir / "cluster_network_profile.npy", profile.astype(np.float32))
                        save_json(
                            run_dir / "cluster_assignment_status.json",
                            {
                                "items": status_rows,
                                "assignment_metric": (
                                    "topn_count_positive_roi"
                                    if (atlas == "kong2022" and method == "group-intrinsic-spectral")
                                    else "top1_fisher_z"
                                ),
                                "q_threshold": 0.05 if (atlas == "kong2022" and method == "group-intrinsic-spectral") else None,
                                "top_n_cap": 20 if (atlas == "kong2022" and method == "group-intrinsic-spectral") else None,
                            },
                        )
                        save_json(run_dir / "cluster_dominant_network.json", {"dominant": dominant, "network_names": network_names})
                        if atlas == "kong2022":
                            assert roi_profile is not None and roi_registry is not None and roi_top20 is not None
                            np.save(run_dir / "cluster_roi_profile.npy", roi_profile.astype(np.float32))
                            save_json(run_dir / "cluster_roi_top20.json", roi_top20)
                            write_tsv(
                                run_dir / "cluster_roi_registry.tsv",
                                roi_registry,
                                ["roi_id", "roi_name", "hemi", "family17", "subregion", "rank", "canonical8"],
                            )
                        functional_labels, functional_meta, functional_label_items = functional_atlas_from_clusters(
                            group_labels_final=labels_final,
                            status_rows=status_rows,
                            network_names=network_names,
                            color_map=canonical_color_map,
                        )
                        np.save(run_dir / "group_functional_atlas_labels.npy", functional_labels.astype(np.int32))
                        save_json(run_dir / "group_functional_atlas_label_map.json", functional_meta)

                        write_tsv(
                            run_dir / "k_selection_summary.tsv",
                            rows,
                            k_selection_fields,
                        )
                        save_json(run_dir / "k_selection_summary.json", {"rows": rows, "k_final": int(k_final)})

                        # Template label for downstream overview rendering.
                        lbl = make_cluster_label_gifti(labels_final.astype(np.int32), int(k_final), colors)
                        lbl_path = (
                            run_dir
                            / "template"
                            / "workbench_assets"
                            / f"{template_label}_hemi-{hemi}_space-corobl_den-512_label-group_cluster_{method}_{atlas}_{smoothing}_cov-{tau_tag}.label.gii"
                        )
                        lbl_path.parent.mkdir(parents=True, exist_ok=True)
                        nib.save(lbl, str(lbl_path))
                        func_lbl = make_named_label_gifti(functional_labels.astype(np.int32), functional_label_items)
                        func_lbl_path = (
                            run_dir
                            / "template"
                            / "workbench_assets"
                            / f"{template_label}_hemi-{hemi}_space-corobl_den-512_label-group_functional_atlas_{method}_{atlas}_{smoothing}_cov-{tau_tag}.label.gii"
                        )
                        func_lbl_path.parent.mkdir(parents=True, exist_ok=True)
                        nib.save(func_lbl, str(func_lbl_path))

                        manifest = {
                            "status": "ok",
                            "method": method,
                            "atlas": atlas,
                            "smoothing": smoothing,
                            "hemi": hemi,
                            "assignment_mode": "cluster_level",
                            "cluster_assignment_metric": (
                                "topn_count_positive_roi"
                                if (atlas == "kong2022" and method == "group-intrinsic-spectral")
                                else "top1_fisher_z"
                            ),
                            "k_selection_mode": str(args.k_selection_mode),
                            "coverage_threshold": float(tau),
                            "V_min_fraction": float(args.v_min_fraction),
                            "V_min_effective": int(v_min),
                            "feature_family": (
                                "vertex_to_kong2022_400roi_fc_fisherz_colzscore"
                                if method == "group-network-spectral" and atlas == "kong2022"
                                else (
                                    "vertex_to_canonical_network_fc_fisherz_colzscore"
                                    if method == "group-network-spectral"
                                    else (
                                        "vertex_to_canonical_network_partial_fc_networks_only_one_sided_fisherz_colzscore"
                                        if method == "group-network-partial-spectral"
                                        else (
                                            "vertex_to_vertex_intrinsic_fc_fisherz_diag0"
                                            if method == "group-intrinsic-spectral"
                                            else "vertex_to_vertex_intrinsic_fc_fisherz_diag0_plus_partial_profile"
                                        )
                                    )
                                )
                            ),
                            "fc_mode": (
                                "partial_networks_only_one_sided"
                                if method in {"group-network-partial-spectral", "group-intrinsic-partial-spectral"}
                                else (
                                    "pearson"
                                    if method == "group-network-spectral"
                                    else (
                                        "intrinsic_vertex_to_vertex_pearson"
                                        if method == "group-intrinsic-spectral"
                                        else "intrinsic_vertex_to_vertex_partial_profile"
                                    )
                                )
                            ),
                            "partial_fc_definition": (
                                {
                                    "controls": "other_canonical_network_timeseries_only",
                                    "target_network_residualized": False,
                                    "hippocampal_vertex_covariates": False,
                                    "profile_annotation_uses_same_one_sided_partial_fc": True,
                                }
                                if method in {"group-network-partial-spectral", "group-intrinsic-partial-spectral"}
                                else None
                            ),
                            "actual_consumed_input_paths": actual_input_paths,
                            "clustering_algorithm": "spectral_clustering_precomputed_affinity_relu_cosine_diag0_kmeans_n_init_50",
                            "affinity_description": "max(cosine_similarity,0)_hadamard_surface_adjacency_diag0",
                            "candidate_k_range_requested": [int(min(k_values)), int(max(k_values))],
                            "candidate_k_range_effective": [int(min(k_eff)), int(max(k_eff))],
                            "chosen_k_final": int(k_final),
                            "subject_split_seeds": split_seeds,
                            "null_seeds": null_seeds,
                            "strong_null_network_permutation_scope": "subject_specific",
                            "run_aware_diagnostic": "independent_runwise_group_representations",
                            "nontriviality_thresholds": {
                                "V_min_fraction": float(args.v_min_fraction),
                                "max_components_threshold": int(args.max_components_threshold),
                                "min_profile_coherence": float(args.min_profile_coherence),
                            },
                            "density_evidence": {
                                "surface_path": str(surfaces[hemi]),
                                "n_vertices": int(surface_meta[hemi]["n_vertices"]),
                                "n_faces": int(surface_meta[hemi]["n_faces"]),
                            },
                            "surface_topology_audit": topology_audit[hemi],
                            "canonical_network_order_consistency_checked": True,
                            "canonical_network_order_template_subject": f"sub-{subjects[0]}",
                            "n_subjects": int(len(subjects)),
                            "subjects": [f"sub-{s}" for s in subjects],
                            "n_support_vertices": int(n_support),
                            "n_excluded_vertices": int(n_vertices - n_support),
                            "group_support_mask": str((run_dir / "group_support_mask.npy").resolve()),
                            "subject_coverage_count": str((run_dir / "subject_coverage_count.npy").resolve()),
                            "subject_coverage_fraction": str((run_dir / "subject_coverage_fraction.npy").resolve()),
                            "group_labels_final": str((run_dir / "group_labels_final.npy").resolve()),
                            "group_functional_atlas_labels": str((run_dir / "group_functional_atlas_labels.npy").resolve()),
                            "cluster_network_profile": str((run_dir / "cluster_network_profile.npy").resolve()),
                            "cluster_dominant_network": str((run_dir / "cluster_dominant_network.json").resolve()),
                            "cluster_assignment_status": str((run_dir / "cluster_assignment_status.json").resolve()),
                            "group_functional_atlas_label_map": str((run_dir / "group_functional_atlas_label_map.json").resolve()),
                            "cluster_roi_profile": (
                                str((run_dir / "cluster_roi_profile.npy").resolve()) if atlas == "kong2022" else None
                            ),
                            "cluster_roi_top20": (
                                str((run_dir / "cluster_roi_top20.json").resolve()) if atlas == "kong2022" else None
                            ),
                            "cluster_roi_registry": (
                                str((run_dir / "cluster_roi_registry.tsv").resolve()) if atlas == "kong2022" else None
                            ),
                            "cluster_roi_profile_subjects_used": (
                                [f"sub-{s}" for s in roi_profile_subjects_used]
                                if roi_profile_subjects_used is not None
                                else None
                            ),
                            "template_subject": template_label,
                            "cluster_label_ordering": "long_axis_head_to_tail",
                            "cluster_order_head_to_tail": fixed_cluster_order,
                            "template_left_or_right_surface": str(surfaces[hemi]),
                            "template_spec_path": str(
                                infer_spec_path(subjects[0], infer_surface_path(subjects[0], hemi, hippunfold_batch_root))
                            ),
                            "template_cluster_label": str(lbl_path.resolve()),
                            "template_functional_atlas_label": str(func_lbl_path.resolve()),
                        }
                        save_json(run_dir / "group_feature_summary.json", {"network_names": network_names, "N_ref": int(len(network_names))})
                        save_json(manifest_path, manifest)

    print(json.dumps({"out_root": str(out_root), "status": "ok"}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
