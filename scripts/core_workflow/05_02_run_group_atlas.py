#!/usr/bin/env python3
from __future__ import annotations

"""
Process Step Description:
Run group-level atlas clustering and network mapping. Performs clustering analysis on group-averaged functional connectivity or gradients to generate hippocampal functional parcellation atlases and maps them to known cortical networks.

Main Flow Path:
- 8th step in the numbered pipeline sequence (the core clustering and mapping execution step):
  01_01_run_hippunfold_t1w_local.py -> 01_02_check_hippunfold_outputs.py -> 02_01_prepare_group_inputs.py -> 02_02_build_qc_manifest.py -> 03_01/03_02 reference building -> 04_01/04_02/04_03 FC materialization -> 05_01 gradient computation -> 05_02_run_group_atlas.py (this step) -> 06_xx presentation rendering -> 07_01 group matrix overview.

Inputs:
- Command-line arguments:
  - --shared-root: Path to shared processed files (default: `outputs_core/inputs/hcp7t_hippocampus/kong2022/shared_inputs/routeb_151`).
  - --surface-root: Path to BOLD surfaces root directory (default: `outputs_core/inputs/hcp7t_hippocampus/kong2022/shared_inputs/surface_151`).
  - --out-root: Directory where group-atlas intermediate files will be saved (default: `outputs_core/work/group_atlas`).
  - --gradient-root: Directory containing computed continuous gradients (default: `outputs_core/work/hippomaps_gradients`).
  - --hippomaps-root: Root path to external HippoMaps codebase (default: `external/hippomaps`).
  - --route-tag: Expected workflow branch tag, e.g., "routea", "routeb", "auto".
  - --subjects-manifest: Path to QC-passed subject list manifest.
  - --hippunfold-batch-root: Batch output root containing sub-*/hippunfold outputs.
  - --kong2022-dlabel: Path to standard Kong2022 dlabel.nii cortical atlas file.
  - --methods: Parcellation methods to evaluate: "intrinsic-spectral", "network-spectral", "gradient-spectral", "wta-top".
  - --atlases: Cortical atlases to map: "kong2022", "kong2019", etc.
  - --smoothings: Smoothing levels to process: "2mm", "4mm".
  - --hemis: Hemispheres to process: "L", "R".
  - --coverage-thresholds: Minimum coverage fraction values to evaluate (default: [0.70, 0.80, 0.90]).
  - --k-min: Minimum number of clusters to evaluate (default: 2).
  - --k-max: Maximum number of clusters to evaluate (default: 10).
  - --force-final-k: Fixed K value for final presentation labels (default: 3).
  - --instability-resamples: Number of split-half resamples for cluster instability (default: 100).
  - --null-resamples: Number of random spin tests for cortical mapping null distribution (default: 100).
  - --seed: Random seed for repeatability (default: 20260420).
  - --v-min-fraction: Minimum cluster vertex size fraction (default: 0.03).
  - --max-components-threshold: Max components for gradient selection.
  - --min-profile-coherence: Minimum profile coherence score threshold.
  - --subject-fc-mode: FC preprocessing sparsification mode: "full", "top-percent".
  - --subject-fc-top-percent: Sparsification top percentage integer.
  - --gradient-components: Selected gradient components for gradient-spectral (default: "1,2,3").
  - --resume: Resume from existing runs.
  - --k-selection-mode: K selection model: "redundancy-aware", "experimental".
  - --redundancy-early-stop: Stop evaluation of larger K values once first redundant K is found (default: False).
  - --overview-threshold: Selected coverage threshold for overview outputs (default: 0.80).
  - --jobs: Parallel worker pool size (default: 1).
  - --group-average-surfaces-dir: Directory containing group average template surfaces.
- Physical inputs:
  - Subject functional connectivity caches: e.g. `{shared_root}/sub-<subject>/fc/.../hipp_vertex_to_network_fc_active.npy`.
  - Group average template surfaces: e.g. `{group_average_surfaces_dir}/group-average_hemi-{L|R}_space-corobl_den-512_label-hipp_midthickness.surf.gii`.
  - Pre-computed continuous gradients under `{gradient_root}/` (for gradient-spectral method).

Outputs:
- Command-line output / file outputs:
  - Detailed group-atlas results directories under `{out_root}/{route_tag}/{method}/smooth-{smoothing}_hemi-{hemi}_cov-{threshold}/`:
    - `group_atlas_manifest.json`: Step-level execution trace and results.
    - `group_functional_atlas_labels.npy`: Parcellation label array for chosen final K.
    - `cluster_assignment_status.json`: Network mapping diagnostic summary.
    - Suffixed label / profile files (`*_fisherz.npy`, `*_count.npy`, etc.).
"""
import argparse
import csv
import hashlib
import importlib.metadata
import json
import math
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

import nibabel as nib
import numpy as np
from nibabel.gifti import GiftiImage
from scipy import sparse
from scipy import stats
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import SpectralClustering
from sklearn.metrics import adjusted_rand_score

REPO_ROOT = Path(__file__).resolve().parents[2]
from lib.hippomaps_adapter import (
    DEFAULT_HIPPOMAPS_ROOT,
    ensure_hippomaps_on_path,
    fit_gradientmaps_pearson,
    hippomaps_commit,
    hippomaps_import_path,
)
from lib.surface_utils import infer_spec_path, long_axis_order
from lib.kong2022_label_utils import load_kong2022_bundle
from lib.output_layout import (
    DEFAULT_GROUP_ATLAS_ROOT,
    DEFAULT_HIPPOMAPS_GRADIENT_ROOT,
    DEFAULT_HIPPUNFOLD_BATCH_ROOT as CORE_HIPPUNFOLD_BATCH_ROOT,
    DEFAULT_KONG2022_MANIFEST_ROOT,
    DEFAULT_KONG2022_ROOT,
    DEFAULT_KONG2022_ROUTEB_ROOT,
    DEFAULT_KONG2022_SURFACE_ROOT,
    DEFAULT_KONG2022_TEMPLATE_ROOT,
)
from lib.fc_math import corrcoef_rows, fisher_z_transform_fc
from lib.gifti_labels import make_cluster_label_gifti, make_named_label_gifti
from lib.io_utils import load_subject_ids, write_tsv
from lib.workflow_utils import write_json


DEFAULT_T1ONLY_ROOT = DEFAULT_KONG2022_ROOT
DEFAULT_SHARED_ROOT = DEFAULT_KONG2022_ROUTEB_ROOT
DEFAULT_SURFACE_ROOT = DEFAULT_KONG2022_SURFACE_ROOT
DEFAULT_OUT_ROOT = DEFAULT_GROUP_ATLAS_ROOT / "routeb_auto"
DEFAULT_GRADIENT_ROOT = DEFAULT_HIPPOMAPS_GRADIENT_ROOT
DEFAULT_SUBJECTS = DEFAULT_KONG2022_MANIFEST_ROOT / "hcp_7t_hippocampus_fmri_qc_pass_t1only_latest.txt"
DEFAULT_HIPPUNFOLD_BATCH_ROOT = CORE_HIPPUNFOLD_BATCH_ROOT
DEFAULT_NETWORK_STYLE = REPO_ROOT / "config" / "hipp_network_style.json"
DEFAULT_CROSS_ATLAS_MERGE = REPO_ROOT / "config" / "cross_atlas_network_merge.json"
DEFAULT_GROUP_AVG_SURFACES_DIR = DEFAULT_KONG2022_TEMPLATE_ROOT / "group_average_surfaces"
KONG2022_DLABEL_OVERRIDE: Path | None = None
HIPPOMAPS_ROOT_OVERRIDE: Path = DEFAULT_HIPPOMAPS_ROOT

METHOD_CHOICES = [
    "group-roi-sim-spectral",
    "network-spectral",
    "gradient-spectral",
    "intrinsic-spectral",
    "wta-top",
]
METHODS_DEFAULT = [
    "intrinsic-spectral",
    "network-spectral",
    "wta-top",
]
ATLASES = ["kong2019", "kong2019_8net", "kong2022"]
SMOOTHINGS = ["2mm", "4mm"]
HEMIS = ["L", "R"]
K_RANGE = list(range(2, 11))
KONG2022_ONLY_METHODS = {"group-roi-sim-spectral", "gradient-spectral", "wta-top"}
CLUSTERLESS_METHODS = {"wta-top"}
PROFILE_REDUNDANCY_MAX_SIM = 0.80
SUBJECT_FC_PREPROCESS_METHODS = {"group-roi-sim-spectral", "network-spectral", "intrinsic-spectral"}
SUBJECT_FC_VALUE_RULE = "largest_raw_values"
SUBJECT_FC_MODE_CHOICES = ["full", "top-percent"]
DEFAULT_GRADIENT_COMPONENTS = (1, 2, 3)
GRADIENT_N_COMPONENTS_COMPUTED = 5
GRADIENT_BRAINSPACE_KERNEL = "pearson"
ROUTE_TAG_CHOICES = ("routea", "routeb")
MAX_K_EVAL_JOBS = 10
RESUME_SIGNATURE_VERSION = 1


@dataclass(frozen=True)
class Combo:
    method: str
    atlas: str
    smoothing: str
    hemi: str
    coverage_threshold: float


@dataclass(frozen=True)
class SubjectFCMode:
    mode: str
    mode_tag: str
    top_percent: int | None
    top_fraction: float
    binarize: bool


@dataclass(frozen=True)
class GroupAtlasConfig:
    args: argparse.Namespace
    shared_root: Path
    surface_root: Path
    out_root: Path
    gradient_root: Path
    route_tag: str
    hippunfold_batch_root: Path
    subjects_manifest_path: Path
    subjects: list[str]
    subject_fc_mode: SubjectFCMode
    gradient_components: tuple[int, ...]
    coverage_thresholds: list[float]
    k_values: list[int]
    split_seeds: list[int]
    null_seeds: list[int]


@dataclass(frozen=True)
class PreparedComboInputs:
    n_vertices: int
    valid_masks: dict[str, np.ndarray]
    coverage_count: np.ndarray
    coverage_fraction: np.ndarray
    feature_bank: dict[str, np.ndarray]
    actual_input_paths: dict[str, str]
    run_feature_banks: dict[str, object]
    roi_fc_bank_for_redundancy: dict[str, np.ndarray] | None
    subject_fc_preprocessing_for_resume: dict[str, object]
    resume_input_category_hashes: dict[str, str]


def canonical_json_sha256(payload: object) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def file_state(path: Path) -> dict[str, object]:
    p = Path(path).resolve()
    out: dict[str, object] = {"path": str(p), "exists": bool(p.exists())}
    if bool(out["exists"]):
        st = p.stat()
        out["size_bytes"] = int(st.st_size)
        out["mtime_ns"] = int(st.st_mtime_ns)
    return out


def file_state_list(paths: list[Path]) -> list[dict[str, object]]:
    uniq: dict[str, Path] = {}
    for p in paths:
        rp = Path(p).resolve()
        uniq[str(rp)] = rp
    return [file_state(uniq[k]) for k in sorted(uniq.keys())]


def resume_manifest_matches(manifest_path: Path, expected_signature: str) -> tuple[bool, str]:
    if not manifest_path.exists():
        return False, "missing_manifest"
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except Exception as exc:  # pragma: no cover
        return False, f"invalid_manifest:{exc.__class__.__name__}"
    if int(payload.get("resume_input_signature_version", -1)) != int(RESUME_SIGNATURE_VERSION):
        return False, "missing_or_legacy_signature"
    got = payload.get("resume_input_signature")
    if not isinstance(got, str):
        return False, "missing_signature"
    if got != expected_signature:
        return False, "upstream_signature_changed"
    return True, "signature_match"


def zscore_cols(x: np.ndarray) -> np.ndarray:
    arr = np.asarray(x, dtype=np.float32)
    mu = np.nanmean(arr, axis=0, keepdims=True)
    sd = np.nanstd(arr, axis=0, keepdims=True)
    out = (arr - mu) / np.clip(sd, 1e-12, None)
    out[~np.isfinite(out)] = 0.0
    return out.astype(np.float32)


def centered_cosine_similarity(a: np.ndarray, b: np.ndarray | None = None) -> np.ndarray:
    x = np.asarray(a, dtype=np.float32)
    y = x if b is None else np.asarray(b, dtype=np.float32)
    x = x - np.nanmean(x, axis=1, keepdims=True)
    y = y - np.nanmean(y, axis=1, keepdims=True)
    x[~np.isfinite(x)] = 0.0
    y[~np.isfinite(y)] = 0.0
    x = x / np.clip(np.linalg.norm(x, axis=1, keepdims=True), 1e-12, None)
    y = y / np.clip(np.linalg.norm(y, axis=1, keepdims=True), 1e-12, None)
    sim = x @ y.T
    sim = np.clip(sim, -1.0, 1.0)
    sim[~np.isfinite(sim)] = 0.0
    return sim.astype(np.float32)


def profile_redundancy_metrics(profile: np.ndarray) -> tuple[float, int]:
    p = np.asarray(profile, dtype=np.float32)
    if p.ndim != 2 or int(p.shape[0]) <= 1:
        return 0.0, 0
    sim = centered_cosine_similarity(p)
    upper = np.triu_indices(int(p.shape[0]), k=1)
    vals = sim[upper]
    if vals.size == 0:
        return 0.0, 0
    max_sim = float(np.max(vals))
    redundant_pairs = int(np.sum(vals > float(PROFILE_REDUNDANCY_MAX_SIM)))
    return max_sim, redundant_pairs


def cosine_affinity(features: np.ndarray) -> np.ndarray:
    x = np.asarray(features, dtype=np.float32)
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    x = x / np.clip(norms, 1e-12, None)
    sim = x @ x.T
    sim = np.clip(sim, -1.0, 1.0)
    aff = np.maximum(sim, 0.0)
    np.fill_diagonal(aff, 1.0)
    return aff.astype(np.float32)


def suppress_known_runtime_warnings() -> None:
    warnings.filterwarnings(
        "ignore",
        message="Graph is not fully connected, spectral embedding may not work as expected",
        category=UserWarning,
    )


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
    return method == "network-spectral"


def is_gradient_method(method: str) -> bool:
    return method == "gradient-spectral"


def is_roi_similarity_method(method: str) -> bool:
    return method == "group-roi-sim-spectral"


def method_uses_network_fc_input(method: str) -> bool:
    return is_network_method(method) or method in {"group-roi-sim-spectral", "gradient-spectral", "wta-top"}


def is_clusterless_method(method: str) -> bool:
    return method in CLUSTERLESS_METHODS


def method_uses_subject_fc_preprocessing(method: str) -> bool:
    return method in SUBJECT_FC_PREPROCESS_METHODS


def feature_family_for_method(method: str, atlas: str, gradient_components: tuple[int, ...]) -> str:
    if method == "group-roi-sim-spectral":
        return "vertex_to_kong2022_400roi_fc_to_vertex_similarity_centered_cosine"
    if method == "gradient-spectral":
        comp_tag = "".join(f"g{int(c)}" for c in gradient_components)
        return f"group_vertex_to_kong2022_400roi_fc_diffusion_gradients_{comp_tag}"
    if method == "network-spectral" and atlas == "kong2022":
        return "vertex_to_kong2022_400roi_fc_fisherz_colzscore"
    if method == "network-spectral":
        return "vertex_to_canonical_network_fc_fisherz_colzscore"
    if method == "intrinsic-spectral":
        return "vertex_to_vertex_intrinsic_fc_fisherz_diag0"
    return "vertex_to_vertex_intrinsic_fc_fisherz_diag0_plus_partial_profile"


def fc_mode_for_method(method: str) -> str:
    if method == "group-roi-sim-spectral":
        return "pearson_vertex_to_roi_then_centered_cosine_vertex_to_vertex"
    if method == "gradient-spectral":
        return "pearson_vertex_to_roi_group_average_then_diffusion_gradients"
    if method == "network-spectral":
        return "pearson"
    if method == "intrinsic-spectral":
        return "intrinsic_vertex_to_vertex_pearson"
    return "intrinsic_vertex_to_vertex_partial_profile"


def resolve_subject_fc_mode(subject_fc_mode: str, subject_fc_top_percent: int | None) -> SubjectFCMode:
    mode = str(subject_fc_mode)
    if mode not in SUBJECT_FC_MODE_CHOICES:
        raise ValueError(f"--subject-fc-mode must be one of {SUBJECT_FC_MODE_CHOICES}, got {subject_fc_mode}")
    if mode == "full":
        if subject_fc_top_percent is not None:
            raise ValueError("--subject-fc-top-percent is not allowed when --subject-fc-mode=full")
        return SubjectFCMode(mode="full", mode_tag="full", top_percent=None, top_fraction=1.0, binarize=False)
    if subject_fc_top_percent is None:
        raise ValueError("--subject-fc-top-percent is required when --subject-fc-mode=top-percent")
    pct = int(subject_fc_top_percent)
    if pct <= 0 or pct > 100:
        raise ValueError(f"--subject-fc-top-percent must be within (0, 100], got {pct}")
    return SubjectFCMode(
        mode="top-percent",
        mode_tag=f"{pct}perc",
        top_percent=pct,
        top_fraction=float(pct) / 100.0,
        binarize=True,
    )


def parse_gradient_components(raw: str | None) -> tuple[int, ...]:
    if raw is None:
        return DEFAULT_GRADIENT_COMPONENTS
    text = str(raw).strip()
    if not text:
        raise ValueError("--gradient-components must not be empty")
    parts = text.split(",")
    try:
        comps = tuple(int(p.strip()) for p in parts)
    except ValueError as exc:
        raise ValueError(f"--gradient-components must be comma-separated integers from 1..3, got {raw!r}") from exc
    if any(c not in DEFAULT_GRADIENT_COMPONENTS for c in comps):
        raise ValueError(f"--gradient-components must only contain 1, 2, and/or 3, got {raw!r}")
    if len(set(comps)) != len(comps):
        raise ValueError(f"--gradient-components must not contain duplicates, got {raw!r}")
    if tuple(sorted(comps)) != comps:
        raise ValueError(f"--gradient-components must be in ascending order, got {raw!r}")
    return comps


def subject_fc_preprocessing_manifest_block(method: str, subject_fc_mode: SubjectFCMode) -> dict[str, object]:
    enabled = bool(method_uses_subject_fc_preprocessing(method))
    if not enabled:
        return {
            "enabled": False,
            "mode": None,
            "mode_tag": None,
            "top_percent": None,
            "applied_methods": sorted(SUBJECT_FC_PREPROCESS_METHODS),
            "value_rule": SUBJECT_FC_VALUE_RULE,
        }
    return {
        "enabled": True,
        "mode": str(subject_fc_mode.mode),
        "mode_tag": str(subject_fc_mode.mode_tag),
        "top_percent": (
            None if subject_fc_mode.top_percent is None else int(subject_fc_mode.top_percent)
        ),
        "applied_methods": sorted(SUBJECT_FC_PREPROCESS_METHODS),
        "value_rule": SUBJECT_FC_VALUE_RULE,
    }


def _n_keep_from_fraction(n_total: int, top_fraction: float) -> int:
    if int(n_total) <= 0:
        return 0
    frac = float(np.clip(float(top_fraction), 0.0, 1.0))
    if frac <= 0.0:
        return 0
    if frac >= 1.0:
        return int(n_total)
    return int(max(1, math.ceil(frac * float(n_total))))


def _select_top_mask_1d(values: np.ndarray, top_fraction: float) -> np.ndarray:
    x = np.asarray(values, dtype=np.float32).reshape(-1)
    finite = np.isfinite(x)
    keep = np.zeros((x.size,), dtype=bool)
    n_finite = int(np.sum(finite))
    if n_finite <= 0:
        return keep
    n_keep = _n_keep_from_fraction(n_finite, top_fraction)
    if n_keep <= 0:
        return keep
    if n_keep >= n_finite:
        keep[finite] = True
        return keep
    finite_idx = np.flatnonzero(finite)
    finite_vals = x[finite_idx]
    top_local = np.argpartition(finite_vals, n_finite - n_keep)[n_finite - n_keep:]
    keep[finite_idx[top_local]] = True
    return keep


def preprocess_network_fc_map(fc: np.ndarray, top_fraction: float, binarize: bool) -> np.ndarray:
    arr = np.asarray(fc, dtype=np.float32)
    if arr.ndim != 2:
        raise ValueError(f"Expected 2D FC matrix for network preprocessing, got shape={arr.shape}")
    out = arr.copy()
    flat = arr.reshape(-1)
    keep_flat = _select_top_mask_1d(flat, top_fraction)
    finite_flat = np.isfinite(flat)
    out_flat = out.reshape(-1)
    out_flat[finite_flat & ~keep_flat] = 0.0
    if bool(binarize):
        out_flat[keep_flat] = 1.0
    out_flat[~finite_flat] = np.nan
    return out.astype(np.float32)


def preprocess_intrinsic_fc_map(fc: np.ndarray, top_fraction: float, binarize: bool) -> np.ndarray:
    arr = np.asarray(fc, dtype=np.float32)
    if arr.ndim != 2 or int(arr.shape[0]) != int(arr.shape[1]):
        raise ValueError(f"Expected square FC matrix for intrinsic preprocessing, got shape={arr.shape}")
    out = arr.copy()
    tri = np.triu_indices(int(arr.shape[0]), k=1)
    tri_vals = arr[tri]
    keep_tri = _select_top_mask_1d(tri_vals, top_fraction)
    finite_tri = np.isfinite(tri_vals)
    zero_tri = finite_tri & ~keep_tri
    if np.any(zero_tri):
        zi = tri[0][zero_tri]
        zj = tri[1][zero_tri]
        out[zi, zj] = 0.0
        out[zj, zi] = 0.0
    if np.any(keep_tri):
        ki = tri[0][keep_tri]
        kj = tri[1][keep_tri]
        if bool(binarize):
            kv = np.ones((ki.size,), dtype=np.float32)
        else:
            kv = arr[ki, kj].astype(np.float32, copy=False)
        out[ki, kj] = kv
        out[kj, ki] = kv
    out[~np.isfinite(arr)] = np.nan
    np.fill_diagonal(out, 0.0)
    return out.astype(np.float32)


def preprocess_subject_fc_matrix(method: str, fc: np.ndarray, subject_fc_mode: SubjectFCMode) -> np.ndarray:
    arr = np.asarray(fc, dtype=np.float32)
    if not method_uses_subject_fc_preprocessing(method):
        return arr
    if str(subject_fc_mode.mode) == "full":
        return arr
    if str(subject_fc_mode.mode) != "top-percent":
        raise ValueError(f"Unsupported subject FC mode: {subject_fc_mode.mode}")
    if method in {"group-roi-sim-spectral", "network-spectral"}:
        return preprocess_network_fc_map(arr, subject_fc_mode.top_fraction, bool(subject_fc_mode.binarize))
    if method == "intrinsic-spectral":
        return preprocess_intrinsic_fc_map(arr, subject_fc_mode.top_fraction, bool(subject_fc_mode.binarize))
    return arr


def roi_similarity_from_network_fc(fc: np.ndarray) -> np.ndarray:
    arr = np.asarray(fc, dtype=np.float32)
    if arr.ndim != 2:
        raise ValueError(f"Expected 2D vertex-to-ROI FC matrix, got shape={arr.shape}")
    sim = centered_cosine_similarity(arr)
    sim = 0.5 * (sim + sim.T)
    sim = np.clip(sim, -1.0, 1.0)
    sim[~np.isfinite(sim)] = 0.0
    np.fill_diagonal(sim, 0.0)
    return sim.astype(np.float32)


def brainspace_version() -> str:
    try:
        return str(importlib.metadata.version("brainspace"))
    except importlib.metadata.PackageNotFoundError:
        return "missing"


def _extract_brainspace_array(model: object, primary_name: str, fallback_name: str, label: str) -> np.ndarray:
    raw = getattr(model, primary_name, None)
    if raw is None:
        raw = getattr(model, fallback_name, None)
    if raw is None:
        raise RuntimeError(
            f"BrainSpace fit output missing `{primary_name}`/`{fallback_name}` for {label}"
        )
    if isinstance(raw, list):
        if len(raw) != 1:
            raise RuntimeError(f"Expected a single dataset in BrainSpace {label}, got {len(raw)} entries")
        raw = raw[0]
    arr = np.asarray(raw, dtype=np.float32)
    if arr.size == 0:
        raise RuntimeError(f"BrainSpace {label} is empty")
    return arr


def brainspace_diffusion_gradients(
    features: np.ndarray,
    *,
    n_components: int = GRADIENT_N_COMPONENTS_COMPUTED,
    hippomaps_root: Path = DEFAULT_HIPPOMAPS_ROOT,
    gradientmaps_fit_fn: object | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    group_fc = np.asarray(features, dtype=np.float32)
    if group_fc.ndim != 2:
        raise ValueError(f"Expected 2D group FC feature matrix, got shape={group_fc.shape}")
    n_comp = int(n_components)
    if n_comp <= 0:
        raise ValueError(f"n_components must be positive, got {n_components}")
    if int(group_fc.shape[0]) <= n_comp:
        raise RuntimeError(
            f"Not enough supported vertices to compute G1..G{n_comp}: n_vertices={group_fc.shape[0]}"
        )
    ensure_hippomaps_on_path(hippomaps_root)
    fit_fn = fit_gradientmaps_pearson if gradientmaps_fit_fn is None else gradientmaps_fit_fn
    gradients, lambdas, ev_fractions = fit_fn(group_fc, n_components=n_comp)
    gradients = np.asarray(gradients, dtype=np.float32)
    lambdas = np.asarray(lambdas, dtype=np.float64).reshape(-1)
    ev_fractions = np.asarray(ev_fractions, dtype=np.float64).reshape(-1)
    if gradients.ndim != 2:
        raise RuntimeError(f"Expected 2D GradientMaps gradients, got shape={gradients.shape}")
    if gradients.shape[1] < n_comp:
        raise RuntimeError(
            f"GradientMaps returned {gradients.shape[1]} gradients, expected at least {n_comp}"
        )
    if lambdas.size < n_comp:
        raise RuntimeError(f"GradientMaps returned {lambdas.size} lambdas, expected at least {n_comp}")
    if ev_fractions.size < n_comp:
        raise RuntimeError(f"GradientMaps returned {ev_fractions.size} EV fractions, expected at least {n_comp}")
    gradients = gradients[:, :n_comp]
    gradients[~np.isfinite(gradients)] = 0.0
    lambdas = lambdas[:n_comp]
    ev_fractions = ev_fractions[:n_comp]
    if not np.all(np.isfinite(lambdas)):
        raise RuntimeError(f"GradientMaps lambdas contain non-finite values: {lambdas.tolist()}")
    if not np.all(np.isfinite(ev_fractions)):
        raise RuntimeError(f"GradientMaps EV fractions contain non-finite values: {ev_fractions.tolist()}")
    return gradients.astype(np.float32), lambdas.astype(np.float64), ev_fractions.astype(np.float64)


def build_group_gradient_features(
    *,
    feature_bank: dict[str, np.ndarray],
    subjects_subset: list[str],
    support_mask: np.ndarray,
    gradient_components: tuple[int, ...] = DEFAULT_GRADIENT_COMPONENTS,
    hippomaps_root: Path = DEFAULT_HIPPOMAPS_ROOT,
    gradientmaps_fit_fn: object | None = None,
) -> tuple[np.ndarray, dict[str, object]]:
    idx = np.flatnonzero(support_mask)
    if idx.size == 0:
        raise RuntimeError("Support mask has zero vertices")
    comps = parse_gradient_components(",".join(str(c) for c in gradient_components))
    mats = []
    for s in subjects_subset:
        m = np.asarray(feature_bank[s], dtype=np.float32)[idx, :]
        mats.append(m)
    stack = np.stack(mats, axis=0).astype(np.float32)
    group_fc = np.nanmean(stack, axis=0)
    group_fc[~np.isfinite(group_fc)] = 0.0
    gradients, lambdas, ev_fractions = brainspace_diffusion_gradients(
        group_fc,
        n_components=GRADIENT_N_COMPONENTS_COMPUTED,
        hippomaps_root=hippomaps_root,
        gradientmaps_fit_fn=gradientmaps_fit_fn,
    )
    selected = gradients[:, [int(c) - 1 for c in comps]]
    selected[~np.isfinite(selected)] = 0.0
    selected_labels = [f"G{int(c)}" for c in comps]
    hippomaps_root_resolved = ensure_hippomaps_on_path(hippomaps_root)
    meta = {
        "gradient_components": [int(c) for c in comps],
        "gradient_component_labels": selected_labels,
        "selected_components": selected_labels,
        "gradient_embedding_engine": "brainspace_gradientmaps_via_hippomaps_adapter",
        "brainspace_version": brainspace_version(),
        "gradientmaps_call": "brainspace.gradient.GradientMaps(n_components=N, kernel='pearson').fit(column_zscored_group_mean_fisherz_fc)",
        "feature_zscore_pre_gradientmaps": "column-wise z-score across vertices after group Fisher-z averaging",
        "kernel": GRADIENT_BRAINSPACE_KERNEL,
        "brainspace_sparsity": "default(0.9)",
        "n_components_computed": int(GRADIENT_N_COMPONENTS_COMPUTED),
        "components_displayed": [int(c) for c in DEFAULT_GRADIENT_COMPONENTS],
        "group_vertex_to_roi_fc_shape": [int(v) for v in group_fc.shape],
        "gradient_eigenvalues": [float(v) for v in lambdas],
        "ev_fractions": [float(v) for v in ev_fractions],
        "hippomaps_repo": str(Path(hippomaps_root_resolved).resolve()),
        "hippomaps_import_path": hippomaps_import_path(hippomaps_root_resolved),
        "hippomaps_commit": hippomaps_commit(hippomaps_root_resolved),
    }
    return selected.astype(np.float32), meta


def load_hippomaps_gradient_features(
    *,
    gradient_root: Path,
    route: str,
    smoothing: str,
    hemi: str,
    support_mask: np.ndarray,
    gradient_components: tuple[int, ...] = DEFAULT_GRADIENT_COMPONENTS,
) -> tuple[np.ndarray, dict[str, object]]:
    comps = parse_gradient_components(",".join(str(c) for c in gradient_components))
    run_dir = Path(gradient_root).resolve() / route / smoothing / f"hemi_{hemi}"
    manifest_path = run_dir / "hippomaps_gradient_manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"Missing HippoMaps gradient manifest for gradient-spectral clustering: {manifest_path}. "
            "Run 05_01_compute_hippomaps_gradients.py before 05_02_run_group_atlas.py."
        )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if str(manifest.get("status", "ok")) != "ok":
        raise RuntimeError(f"HippoMaps gradient manifest is not ok: {manifest_path} status={manifest.get('status')}")
    local_gradients_path = run_dir / "hippomaps_gradients_full.npy"
    outputs = manifest.get("outputs", {})
    gradients_path_raw = outputs.get("hippomaps_gradients_full") if isinstance(outputs, dict) else None
    manifest_gradients_path = Path(str(gradients_path_raw)).resolve() if gradients_path_raw else None
    gradients_path = local_gradients_path if local_gradients_path.exists() else manifest_gradients_path
    if gradients_path is None or not gradients_path.exists():
        raise FileNotFoundError(
            f"Missing HippoMaps full gradient array. Checked local={local_gradients_path} "
            f"and manifest={manifest_gradients_path}"
        )
    gradients_full = np.load(gradients_path).astype(np.float32)
    support = np.asarray(support_mask, dtype=bool)
    if gradients_full.ndim != 2:
        raise ValueError(f"Expected 2D HippoMaps gradient array: {gradients_path} shape={gradients_full.shape}")
    if gradients_full.shape[0] != support.size:
        raise ValueError(
            f"Gradient/support vertex count mismatch: gradients={gradients_full.shape[0]} support={support.size} "
            f"path={gradients_path}"
        )
    max_comp = max(comps)
    if gradients_full.shape[1] < max_comp:
        raise ValueError(
            f"HippoMaps gradient array has {gradients_full.shape[1]} components; requested G{max_comp}: {gradients_path}"
        )
    selected = gradients_full[np.flatnonzero(support)][:, [int(c) - 1 for c in comps]].astype(np.float32)
    selected[~np.isfinite(selected)] = 0.0
    labels = [f"G{int(c)}" for c in comps]
    meta = {
        "gradient_source": "hippomaps_gradient_artifacts",
        "hippomaps_gradient_manifest": str(manifest_path.resolve()),
        "hippomaps_gradients_full": str(gradients_path.resolve()),
        "route": route,
        "smoothing": smoothing,
        "hemi": hemi,
        "gradient_components": [int(c) for c in comps],
        "gradient_component_labels": labels,
        "selected_components": labels,
        "n_components_computed": int(manifest.get("n_components_computed", gradients_full.shape[1])),
        "gradientmaps_call": manifest.get("gradientmaps_call"),
        "gradient_shape_full": [int(x) for x in gradients_full.shape],
    }
    for key in ("ev_fractions", "lambdas", "hippomaps_repo", "hippomaps_import_path", "hippomaps_commit"):
        if key in manifest:
            meta[key] = manifest[key]
    return selected, meta


def network_feature_filename(method: str) -> str:
    if method in {"group-roi-sim-spectral", "network-spectral", "gradient-spectral", "wta-top"}:
        return "hipp_vertex_to_network_fc.npy"
    raise ValueError(f"Not a network feature method: {method}")


def runwise_network_feature_filename(method: str) -> str:
    if method in {"group-roi-sim-spectral", "network-spectral", "gradient-spectral"}:
        return "hipp_vertex_to_network_fc_active.npy"
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


def select_k_final_from_survivors(survivors: list[dict]) -> tuple[int, str]:
    """Three-tier K selection from the survivor pool.

    Tier 1 — 1-SE rule anchored on the survivor pool's best mean ARI (not the
    global argmax, which may include rejected K values).
    Tier 2 — among within-1SE survivors, prefer local-maximum K values (true
    peaks of the ARI-vs-K curve rather than noise plateaus).
    Tier 3 — parsimony tiebreak: smallest K wins when quality is genuinely tied.

    Returns (k_final, decision_path) where decision_path is one of:
      "local_max_within_1se"              — normal path
      "within_1se_smallest"               — within-1SE but no local max
      "smallest_among_survivors_fallback" — degenerate fallback (should not occur)
    """
    surv_means = np.asarray([float(r["group_ari_mean"]) for r in survivors], dtype=np.float64)
    best_idx = int(np.argmax(surv_means))
    cutoff = float(surv_means[best_idx]) - float(survivors[best_idx]["group_ari_se"])
    within_1se = [s for s in survivors if float(s["group_ari_mean"]) >= cutoff - 1e-12]
    local_max_in_1se = [s for s in within_1se if int(s.get("local_max", 0)) == 1]
    if local_max_in_1se:
        pool, path = local_max_in_1se, "local_max_within_1se"
    elif within_1se:
        pool, path = within_1se, "within_1se_smallest"
    else:
        pool, path = survivors, "smallest_among_survivors_fallback"
    return int(min(int(s["k"]) for s in pool)), path


def select_k_final_after_redundancy_filter(candidate_pool: list[dict]) -> tuple[int, str, set[int], int]:
    """Select final K after removing redundant candidates.

    This re-applies the standard 1-SE/local-max/parsimony rule on the
    remaining nonredundant, nontrivial pool instead of comparing against
    redundant K values that are no longer eligible.

    Returns:
      k_final
      decision_path
      within_1se_k_set
      pool_argmax_k
    """
    ordered = sorted(candidate_pool, key=lambda r: int(r["k"]))
    means = [float(r["group_ari_mean"]) for r in ordered]
    ses = [float(r["group_ari_se"]) for r in ordered]
    local_flags = local_maxima_flags(means)
    best_idx = int(np.argmax(np.asarray(means, dtype=np.float64)))
    cutoff = float(means[best_idx] - ses[best_idx])
    within_1se = [
        row
        for row in ordered
        if float(row["group_ari_mean"]) >= cutoff - 1e-12
    ]
    within_1se_k_set = {int(row["k"]) for row in within_1se}
    local_max_in_1se = [
        row
        for row, is_local in zip(ordered, local_flags, strict=True)
        if is_local and int(row["k"]) in within_1se_k_set
    ]
    if local_max_in_1se:
        pool, path = local_max_in_1se, "local_max_within_1se_after_redundancy_filter"
    elif within_1se:
        pool, path = within_1se, "within_1se_smallest_after_redundancy_filter"
    else:
        pool, path = ordered, "smallest_among_nonredundant_fallback"
    return (
        int(min(int(row["k"]) for row in pool)),
        path,
        within_1se_k_set,
        int(ordered[best_idx]["k"]),
    )


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


def infer_valid_mask_from_fc(shared_root: Path, subject: str, smoothing: str, hemi: str) -> np.ndarray:
    p = (
        shared_root
        / f"sub-{subject}"
        / "fc"
        / "kong2022"
        / smoothing
        / f"hemi_{hemi}"
        / "hipp_vertex_to_network_fc.npy"
    )
    if not p.exists():
        raise FileNotFoundError(
            "Missing both surface tsnr mask and FC-based fallback source. "
            f"Expected FC file for fallback: {p}"
        )
    arr = np.load(p).astype(np.float32)
    if arr.ndim != 2:
        raise ValueError(f"Expected 2D FC array for fallback mask, got {arr.shape} from {p}")
    mask = np.any(np.isfinite(arr), axis=1)
    if int(np.sum(mask)) <= 0:
        raise RuntimeError(f"FC-based fallback valid mask is empty: {p}")
    return mask.astype(bool)


def subject_valid_mask(surface_root: Path, shared_root: Path, subject: str, smoothing: str, hemi: str) -> np.ndarray:
    p = surface_root / f"sub-{subject}" / "surface" / "tsnr_gate" / f"sub-{subject}_hemi-{hemi}_valid_mask.npy"
    if p.exists():
        return np.load(p).astype(bool)
    print(
        f"[warn] missing tsnr_gate mask for sub-{subject} {smoothing} hemi-{hemi}; "
        "inferring valid mask from precomputed FC file",
        flush=True,
    )
    return infer_valid_mask_from_fc(shared_root, subject, smoothing, hemi)


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
    gradient_components: tuple[int, ...] = DEFAULT_GRADIENT_COMPONENTS,
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

    if is_gradient_method(method):
        features, _meta = build_group_gradient_features(
            feature_bank=feature_bank,
            subjects_subset=subjects_subset,
            support_mask=support_mask,
            gradient_components=gradient_components,
            hippomaps_root=HIPPOMAPS_ROOT_OVERRIDE,
        )
        return features

    if method in {"group-roi-sim-spectral", "intrinsic-spectral"}:
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
    gradient_components: tuple[int, ...] = DEFAULT_GRADIENT_COMPONENTS,
) -> tuple[np.ndarray, np.ndarray]:
    features = group_feature_matrix(
        method=method,
        feature_bank=feature_bank,
        subjects_subset=subjects_subset,
        support_mask=support_mask,
        gradient_components=gradient_components,
    )
    return cluster_from_precomputed_group_features(
        group_features=features,
        support_mask=support_mask,
        adjacency_full=adjacency_full,
        k=int(k),
        seed=int(seed),
    )


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
    gradient_components: tuple[int, ...] = DEFAULT_GRADIENT_COMPONENTS,
) -> dict[str, object]:
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
            gradient_components=gradient_components,
        )
        _, full_b = cluster_for_subset(
            method=method,
            feature_bank=feature_bank,
            subjects_subset=sub_b,
            support_mask=mask_b,
            adjacency_full=adjacency_full,
            k=k,
            seed=seed + 2001,
            gradient_components=gradient_components,
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
    gradient_components: tuple[int, ...] = DEFAULT_GRADIENT_COMPONENTS,
) -> dict[str, object]:
    null_bank: dict[str, np.ndarray] = {}
    for s in subjects:
        arr = feature_bank[s]
        rng = np.random.RandomState(int(seeds[0]) + int(abs(hash(s)) % 100000))
        perm = rng.permutation(arr.shape[0])
        if method_uses_network_fc_input(method):
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
        gradient_components=gradient_components,
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


def resolve_k_eval_jobs(requested_jobs: int, n_k: int) -> int:
    if int(n_k) <= 0:
        return 1
    return max(1, min(int(requested_jobs), int(n_k), int(MAX_K_EVAL_JOBS)))


def redundancy_early_stop_triggered(row: dict[str, object], enabled: bool) -> bool:
    return bool(enabled) and int(row.get("nonredundant_ok", 1)) == 0


def cluster_from_precomputed_group_features(
    *,
    group_features: np.ndarray,
    support_mask: np.ndarray,
    adjacency_full: sparse.csr_matrix,
    k: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    idx = np.flatnonzero(support_mask)
    if idx.size == 0:
        raise RuntimeError("Support mask has zero vertices")
    adj = adjacency_full[idx][:, idx]
    aff = cosine_affinity(group_features)
    fused = sparse.csr_matrix(aff).multiply(adj).tocsr()
    fused = fused.maximum(fused.T)
    labels = spectral_cluster_from_affinity(fused.toarray(), int(k), int(seed))
    full = np.zeros((support_mask.size,), dtype=np.int32)
    full[idx] = labels
    return labels.astype(np.int32), full


def evaluate_k_candidate(
    *,
    combo: Combo,
    method: str,
    k: int,
    seed: int,
    n_support: int,
    v_min: int,
    group_features_active: np.ndarray,
    support_mask: np.ndarray,
    adjacency_full: sparse.csr_matrix,
    adjacency_support: sparse.csr_matrix,
    feature_bank: dict[str, np.ndarray],
    valid_masks: dict[str, np.ndarray],
    subjects: list[str],
    tau: float,
    split_seeds: list[int],
    null_seeds: list[int],
    null_resamples: int,
    run_feature_banks: dict[int, dict[str, np.ndarray]],
    seed_base: int,
    max_components_threshold: int,
    min_profile_coherence_threshold: float,
    k_selection_mode: str,
    roi_fc_bank_for_redundancy: dict[str, np.ndarray] | None,
    gradient_components: tuple[int, ...] = DEFAULT_GRADIENT_COMPONENTS,
) -> tuple[int, np.ndarray, dict[str, object]]:
    labels_active, labels_full = cluster_from_precomputed_group_features(
        group_features=group_features_active,
        support_mask=support_mask,
        adjacency_full=adjacency_full,
        k=int(k),
        seed=int(seed),
    )

    sizes = [int(np.sum(labels_active == c)) for c in range(1, int(k) + 1)]
    min_size = int(min(sizes)) if sizes else 0
    comp = connected_components_per_label(labels_active, adjacency_support)
    max_components = int(max(comp.values())) if comp else 0
    min_coh, mean_coh = cluster_profile_coherence(group_features_active, labels_active, int(k))

    ari_obj = subject_split_ari(
        method=method,
        feature_bank=feature_bank,
        valid_masks_by_subject=valid_masks,
        subjects=subjects,
        tau=float(tau),
        adjacency_full=adjacency_full,
        k=int(k),
        seeds=split_seeds,
        gradient_components=gradient_components,
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
        adjacency_full=adjacency_full,
        k=int(k),
        seeds=null_seeds,
        gradient_components=gradient_components,
    )
    strong = list(strong_obj["values"])
    strong_arr = np.asarray(strong, dtype=np.float64) if strong else np.array([0.0], dtype=np.float64)

    run_obj = run_aware_ari(
        method=method,
        run_feature_banks=run_feature_banks,
        subjects=subjects,
        support_mask=support_mask,
        adjacency_full=adjacency_full,
        k=int(k),
        seed_base=int(seed_base),
        gradient_components=gradient_components,
    )
    run_vals = list(run_obj["values"])
    run_arr = np.asarray(run_vals, dtype=np.float64) if run_vals else np.array([np.nan], dtype=np.float64)
    run_ari_mean = float(np.nanmean(run_arr))
    run_ari_se = (
        float(np.nanstd(run_arr, ddof=1) / np.sqrt(np.sum(np.isfinite(run_arr))))
        if np.sum(np.isfinite(run_arr)) > 1
        else 0.0
    )

    max_profile_similarity = float("nan")
    redundant_pair_count = -1
    if k_selection_mode == "redundancy-aware" and roi_fc_bank_for_redundancy is not None:
        roi_profile_k = cluster_roi_profile_from_fc_bank(
            labels_full=labels_full,
            support_mask=support_mask,
            subjects=subjects,
            roi_fc_bank=roi_fc_bank_for_redundancy,
            valid_masks_by_subject=valid_masks,
        )
        max_profile_similarity, redundant_pair_count = profile_redundancy_metrics(roi_profile_k)

    row = {
        "method": method,
        "atlas": combo.atlas,
        "smoothing": combo.smoothing,
        "hemi": combo.hemi,
        "coverage_threshold": float(combo.coverage_threshold),
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
        "B_null_requested": int(null_resamples),
        "B_null_used_strong": int(strong_obj["n_used"]),
        "B_null_used_weak": int(weak_obj["n_used"]),
        "min_cluster_size": int(min_size),
        "max_components_per_cluster": int(max_components),
        "min_profile_coherence": float(min_coh),
        "mean_profile_coherence": float(mean_coh),
        "nontrivial_min_size_ok": int(min_size >= v_min),
        "nontrivial_connectedness_ok": int(max_components <= int(max_components_threshold)),
        "nontrivial_coherence_ok": int(min_coh >= float(min_profile_coherence_threshold)),
        "max_profile_similarity": float(max_profile_similarity),
        "redundant_pair_count": int(redundant_pair_count),
        "nonredundant_ok": int(
            k_selection_mode != "redundancy-aware"
            or (
                np.isfinite(max_profile_similarity)
                and max_profile_similarity <= float(PROFILE_REDUNDANCY_MAX_SIM)
                and int(redundant_pair_count) == 0
            )
        ),
        "reject_nontrivial_fail": 0,
        "reject_redundant": 0,
        "reject_reason_primary": "",
        "selected_final": 0,
    }
    return int(k), labels_full.astype(np.int32), row


def runwise_feature_bank(
    *,
    method: str,
    shared_root: Path,
    subjects: list[str],
    atlas: str,
    smoothing: str,
    hemi: str,
    valid_masks: dict[str, np.ndarray],
    subject_fc_mode: SubjectFCMode,
) -> dict[int, dict[str, np.ndarray]]:
    run_ids = [1, 2, 3, 4]
    out: dict[int, dict[str, np.ndarray]] = {rid: {} for rid in run_ids}
    for rid in run_ids:
        for s in subjects:
            if method_uses_network_fc_input(method):
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
                run_feature = preprocess_subject_fc_matrix(
                    method=method,
                    fc=full,
                    subject_fc_mode=subject_fc_mode,
                )
                out[rid][s] = (
                    roi_similarity_from_network_fc(run_feature)
                    if is_roi_similarity_method(method)
                    else run_feature
                )
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
                out[rid][s] = preprocess_subject_fc_matrix(
                    method=method,
                    fc=full,
                    subject_fc_mode=subject_fc_mode,
                )
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
    gradient_components: tuple[int, ...] = DEFAULT_GRADIENT_COMPONENTS,
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
            gradient_components=gradient_components,
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


def build_subject_surface_consensus(
    subjects: list[str],
    hemis: list[str],
    batch_root: Path,
) -> tuple[dict[str, Path], dict[str, dict[str, object]], dict[str, np.ndarray]]:
    """Build subject-consensus surface metadata from all subjects in the manifest.

    Consensus policy is strict: any topology mismatch OR long-axis order mismatch
    causes a hard failure and lists mismatched subjects.
    """
    consensus_surfaces: dict[str, Path] = {}
    topology_audit: dict[str, dict[str, object]] = {}
    long_axis_order_by_hemi: dict[str, np.ndarray] = {}
    for hemi in hemis:
        ref_subject = subjects[0]
        ref_surface = infer_surface_path(ref_subject, hemi, batch_root)
        ref_coords, ref_faces = load_surface(ref_surface)
        ref_order = long_axis_order(ref_surface)
        topology_mismatched: list[str] = []
        axis_mismatched: list[str] = []
        for subject in subjects[1:]:
            surf_path = infer_surface_path(subject, hemi, batch_root)
            coords, faces = load_surface(surf_path)
            topo_ok = (
                int(coords.shape[0]) == int(ref_coords.shape[0])
                and int(faces.shape[0]) == int(ref_faces.shape[0])
                and faces.shape == ref_faces.shape
                and np.array_equal(faces, ref_faces)
            )
            if not topo_ok:
                topology_mismatched.append(f"sub-{subject}")
                continue
            this_order = long_axis_order(surf_path)
            if this_order.shape != ref_order.shape or not np.array_equal(this_order, ref_order):
                axis_mismatched.append(f"sub-{subject}")
        if topology_mismatched:
            raise RuntimeError(
                "Subject-consensus surface topology mismatch "
                f"for hemi-{hemi}; reference=sub-{ref_subject}; mismatched={topology_mismatched[:10]}"
            )
        if axis_mismatched:
            raise RuntimeError(
                "Subject-consensus long-axis order mismatch "
                f"for hemi-{hemi}; reference=sub-{ref_subject}; mismatched={axis_mismatched[:10]}"
            )
        consensus_surfaces[hemi] = ref_surface
        long_axis_order_by_hemi[hemi] = ref_order
        topology_audit[hemi] = {
            "template_source_mode": "subject_consensus",
            "reference_subject_for_comparison": f"sub-{ref_subject}",
            "n_vertices": int(ref_coords.shape[0]),
            "n_faces": int(ref_faces.shape[0]),
            "topology_consistent_subjects": int(len(subjects)),
            "long_axis_order_consistent_subjects": int(len(subjects)),
        }
    return consensus_surfaces, topology_audit, long_axis_order_by_hemi


def verify_subject_topology_against_template(
    subjects: list[str],
    hemis: list[str],
    template_surfaces: dict[str, Path],
    batch_root: Path,
) -> dict[str, dict[str, object]]:
    out: dict[str, dict[str, object]] = {}
    for hemi in hemis:
        t_coords, t_faces = load_surface(template_surfaces[hemi])
        n_vertices = int(t_coords.shape[0])
        n_faces = int(t_faces.shape[0])
        mismatched: list[str] = []
        for subject in subjects:
            p = infer_surface_path(subject, hemi, batch_root)
            coords, faces = load_surface(p)
            topo_ok = (
                int(coords.shape[0]) == n_vertices
                and int(faces.shape[0]) == n_faces
                and faces.shape == t_faces.shape
                and np.array_equal(faces, t_faces)
            )
            if not topo_ok:
                mismatched.append(f"sub-{subject}")
        if mismatched:
            raise RuntimeError(
                "Template topology mismatch against manifest subjects "
                f"for hemi-{hemi}; template={template_surfaces[hemi]}; mismatched={mismatched[:10]}"
            )
        out[hemi] = {
            "template_source_mode": "group_average_template",
            "n_vertices": n_vertices,
            "n_faces": n_faces,
            "topology_consistent_subjects": int(len(subjects)),
        }
    return out


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
    surface_root: Path,
    atlas: str,
    smoothing: str,
    hemi: str,
    valid_masks_by_subject: dict[str, np.ndarray],
    profile_fc_mode: str,
) -> np.ndarray:
    k_final = int(np.max(group_labels_final))
    profiles: list[np.ndarray] = []
    for subject in subjects:
        ts_path = surface_root / f"sub-{subject}" / "surface" / smoothing / f"sub-{subject}_hemi-{hemi}_timeseries.npy"
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
    bundle = load_kong2022_bundle(dlabel_path=KONG2022_DLABEL_OVERRIDE)
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


def cluster_roi_profile_from_fc_bank(
    *,
    labels_full: np.ndarray,
    support_mask: np.ndarray,
    subjects: list[str],
    roi_fc_bank: dict[str, np.ndarray],
    valid_masks_by_subject: dict[str, np.ndarray],
) -> np.ndarray:
    labels = np.asarray(labels_full, dtype=np.int32)
    if labels.ndim != 1:
        raise ValueError(f"Expected 1D labels, got shape={labels.shape}")
    k = int(np.max(labels))
    if k <= 0:
        raise ValueError("No positive clusters in labels_full")
    support = np.asarray(support_mask, dtype=bool)
    profiles: list[np.ndarray] = []
    for subject in subjects:
        fc = np.asarray(roi_fc_bank[subject], dtype=np.float32)
        if fc.ndim != 2:
            raise ValueError(f"Expected 2D ROI-FC matrix for sub-{subject}, got shape={fc.shape}")
        if fc.shape[0] != labels.size:
            raise ValueError(
                f"ROI-FC vertex mismatch for sub-{subject}: fc={fc.shape[0]} labels={labels.size}"
            )
        valid = support & np.asarray(valid_masks_by_subject[subject], dtype=bool)
        cluster_profile = np.full((k, fc.shape[1]), np.nan, dtype=np.float32)
        for c in range(1, k + 1):
            m = valid & (labels == c)
            if int(np.sum(m)) == 0:
                continue
            with np.errstate(invalid="ignore"):
                cluster_profile[c - 1, :] = np.nanmean(fc[m, :], axis=0).astype(np.float32)
        profiles.append(cluster_profile)
    if not profiles:
        raise RuntimeError("No ROI profiles available from roi_fc_bank")
    out = np.nanmean(np.stack(profiles, axis=0), axis=0).astype(np.float32)
    out[~np.isfinite(out)] = 0.0
    return out


def cluster_roi_profile(
    *,
    group_labels_final: np.ndarray,
    support_mask: np.ndarray,
    subjects: list[str],
    shared_root: Path,
    surface_root: Path,
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
        ts_path = surface_root / f"sub-{subject}" / "surface" / smoothing / f"sub-{subject}_hemi-{hemi}_timeseries.npy"
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
    surface_root: Path,
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
        ts_path = surface_root / f"sub-{subject}" / "surface" / smoothing / f"sub-{subject}_hemi-{hemi}_timeseries.npy"
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


def method_uses_kong2022_topn_cluster_annotation(method: str, atlas: str) -> bool:
    return str(atlas) == "kong2022" and str(method) in {
        "group-roi-sim-spectral",
        "gradient-spectral",
        "intrinsic-spectral",
        "network-spectral",
    }


def network_mean_profile_from_kong2022_roi_profile(
    roi_profile: np.ndarray,
    roi_registry: list[dict[str, object]],
    network_names: list[str],
) -> np.ndarray:
    cols_by_network, _network_names_from_registry = split_kong2022_columns_by_network(roi_registry)
    out = np.full((int(roi_profile.shape[0]), int(len(network_names))), np.nan, dtype=np.float32)
    for net_idx, net_name in enumerate(network_names):
        cols = cols_by_network.get(str(net_name), [])
        if not cols:
            continue
        out[:, net_idx] = np.nanmean(roi_profile[:, np.asarray(cols, dtype=np.int32)], axis=1).astype(np.float32)
    return out


def assignment_status_from_kong2022_roi_profile(
    roi_profile: np.ndarray,
    roi_registry: list[dict[str, object]],
    network_names: list[str],
) -> tuple[np.ndarray, list[dict[str, object]], list[str | None]]:
    profile = network_mean_profile_from_kong2022_roi_profile(roi_profile, roi_registry, network_names)
    status_rows, dominant = assignment_status_from_profile(profile, network_names)
    return profile, status_rows, dominant


def cluster_assignment_variants_from_kong2022_roi_profile(
    *,
    roi_profile: np.ndarray,
    roi_profile_3d: np.ndarray,
    roi_registry: list[dict[str, object]],
    network_names: list[str],
    q_thresh: float,
    top_n_cap: int,
) -> dict[str, dict[str, object]]:
    profile, fisherz_rows, fisherz_dominant = assignment_status_from_kong2022_roi_profile(
        roi_profile,
        roi_registry,
        network_names,
    )
    count_rows, count_dominant, _profile400_mean = assignment_status_from_cluster_roi_topn(
        profile_3d=roi_profile_3d,
        roi_registry=roi_registry,
        q_thresh=float(q_thresh),
        top_n_cap=int(top_n_cap),
    )
    return {
        "fisherz": {
            "profile": profile,
            "status_rows": fisherz_rows,
            "dominant": fisherz_dominant,
            "assignment_metric": "top1_fisher_z",
            "q_threshold": None,
            "top_n_cap": None,
        },
        "count": {
            "profile": profile,
            "status_rows": count_rows,
            "dominant": count_dominant,
            "assignment_metric": "topn_count_positive_roi",
            "q_threshold": float(q_thresh),
            "top_n_cap": int(top_n_cap),
        },
    }


def annotate_kong2022_clusters_topn(
    *,
    group_labels_final: np.ndarray,
    support_mask: np.ndarray,
    subjects: list[str],
    shared_root: Path,
    surface_root: Path,
    smoothing: str,
    hemi: str,
    valid_masks_by_subject: dict[str, np.ndarray],
    network_names: list[str],
    q_thresh: float,
    top_n_cap: int,
) -> tuple[
    np.ndarray,
    list[dict[str, object]],
    list[str | None],
    np.ndarray,
    list[dict[str, object]],
    list[str],
    dict[str, object],
    list[dict[str, object]],
]:
    roi_profile_3d, roi_profile, roi_registry, roi_profile_subjects_used = cluster_roi_profile_subjectwise(
        group_labels_final=group_labels_final,
        support_mask=support_mask,
        subjects=subjects,
        shared_root=shared_root,
        surface_root=surface_root,
        smoothing=smoothing,
        hemi=hemi,
        valid_masks_by_subject=valid_masks_by_subject,
    )
    assignment_variants = cluster_assignment_variants_from_kong2022_roi_profile(
        roi_profile=roi_profile,
        roi_profile_3d=roi_profile_3d,
        roi_registry=roi_registry,
        network_names=network_names,
        q_thresh=float(q_thresh),
        top_n_cap=int(top_n_cap),
    )
    fisherz_variant = assignment_variants["fisherz"]
    count_variant = assignment_variants["count"]
    profile = np.asarray(fisherz_variant["profile"], dtype=np.float32)
    status_rows = [dict(x) for x in fisherz_variant["status_rows"]]
    dominant = list(fisherz_variant["dominant"])
    roi_topn_status_rows = [dict(x) for x in count_variant["status_rows"]]
    roi_top20 = cluster_roi_top_k(roi_profile, roi_registry, k_top=int(top_n_cap))
    return profile, status_rows, dominant, roi_profile, roi_registry, roi_profile_subjects_used, roi_top20, roi_topn_status_rows


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



def prepare_combo_inputs(
    *,
    method: str,
    atlas: str,
    smoothing: str,
    hemi: str,
    shared_root: Path,
    surface_root: Path,
    surfaces: dict[str, Path],
    subjects: list[str],
    subject_fc_mode: SubjectFCMode,
    k_selection_mode: str,
) -> PreparedComboInputs:
    valid_masks = {s: subject_valid_mask(surface_root, shared_root, s, smoothing, hemi) for s in subjects}
    n_vertices = int(next(iter(valid_masks.values())).size)
    coverage_count = np.sum(np.stack([valid_masks[s] for s in subjects], axis=0), axis=0).astype(np.int32)
    coverage_fraction = np.mean(np.stack([valid_masks[s] for s in subjects], axis=0), axis=0).astype(np.float32)

    feature_bank: dict[str, np.ndarray] = {}
    actual_input_paths: dict[str, str] = {}
    for s in subjects:
        if method_uses_network_fc_input(method):
            p = shared_root / f"sub-{s}" / "fc" / atlas / smoothing / f"hemi_{hemi}" / network_feature_filename(method)
            actual_input_paths[s] = str(p)
            raw_feature = subject_network_feature(shared_root, s, atlas, smoothing, hemi, method)
            pre_feature = preprocess_subject_fc_matrix(
                method=method,
                fc=raw_feature,
                subject_fc_mode=subject_fc_mode,
            )
            feature_bank[s] = roi_similarity_from_network_fc(pre_feature) if is_roi_similarity_method(method) else pre_feature
        else:
            p = resolve_intrinsic_feature_path(shared_root, s, smoothing, hemi)
            actual_input_paths[s] = str(p)
            raw_feature = subject_intrinsic_feature(shared_root, s, atlas, smoothing, hemi, valid_masks[s])
            feature_bank[s] = preprocess_subject_fc_matrix(
                method=method,
                fc=raw_feature,
                subject_fc_mode=subject_fc_mode,
            )

    run_feature_banks = (
        runwise_feature_bank(
            method=method,
            shared_root=shared_root,
            subjects=subjects,
            atlas=atlas,
            smoothing=smoothing,
            hemi=hemi,
            valid_masks=valid_masks,
            subject_fc_mode=subject_fc_mode,
        )
        if not is_clusterless_method(method)
        else {}
    )
    roi_fc_bank_for_redundancy: dict[str, np.ndarray] | None = None
    if k_selection_mode == "redundancy-aware" and not is_clusterless_method(method):
        roi_fc_bank_for_redundancy = {}
        for s in subjects:
            if atlas == "kong2022" and method == "network-spectral":
                roi_fc_bank_for_redundancy[s] = feature_bank[s]
            else:
                roi_raw = subject_network_feature(
                    shared_root,
                    s,
                    "kong2022",
                    smoothing,
                    hemi,
                    "network-spectral",
                )
                if method_uses_subject_fc_preprocessing(method):
                    roi_fc_bank_for_redundancy[s] = preprocess_subject_fc_matrix(
                        method="network-spectral",
                        fc=roi_raw,
                        subject_fc_mode=subject_fc_mode,
                    )
                else:
                    roi_fc_bank_for_redundancy[s] = roi_raw

    subject_fc_preprocessing_for_resume = subject_fc_preprocessing_manifest_block(method, subject_fc_mode)
    valid_mask_paths = [
        surface_root / f"sub-{s}" / "surface" / "tsnr_gate" / f"sub-{s}_hemi-{hemi}_valid_mask.npy"
        for s in subjects
    ]
    runwise_feature_paths: list[Path] = []
    if not is_clusterless_method(method):
        for rid in [1, 2, 3, 4]:
            for s in subjects:
                if method_uses_network_fc_input(method):
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
                else:
                    p = resolve_intrinsic_feature_path(shared_root, s, smoothing, hemi, rid)
                runwise_feature_paths.append(p)
    reference_network_order_paths = [
        shared_root / f"sub-{s}" / "reference" / atlas / "cortex_canonical_networks.tsv"
        for s in subjects
    ]
    reference_network_ts_paths = [
        shared_root / f"sub-{s}" / "reference" / atlas / "cortex_canonical_network_timeseries.npy"
        for s in subjects
    ]
    surface_timeseries_paths = [
        surface_root / f"sub-{s}" / "surface" / smoothing / f"sub-{s}_hemi-{hemi}_timeseries.npy"
        for s in subjects
    ]
    kong2022_roi_reference_paths: list[Path] = []
    if atlas == "kong2022":
        for s in subjects:
            kong2022_roi_reference_paths.extend(
                [
                    shared_root / f"sub-{s}" / "reference" / "kong2022" / "cortex_roi_parcel_timeseries.npy",
                    shared_root / f"sub-{s}" / "reference" / "kong2022" / "cortex_roi_parcels.tsv",
                ]
            )
    resume_file_state_payload = {
        "primary_feature_inputs": file_state_list([Path(p) for p in actual_input_paths.values()]),
        "valid_masks": file_state_list(valid_mask_paths),
        "runwise_feature_inputs": file_state_list(runwise_feature_paths),
        "reference_network_order_tsv": file_state_list(reference_network_order_paths),
        "reference_network_timeseries": file_state_list(reference_network_ts_paths),
        "surface_timeseries": file_state_list(surface_timeseries_paths),
        "kong2022_roi_reference_inputs": file_state_list(kong2022_roi_reference_paths),
        "template_surface": file_state_list([surfaces[hemi]]),
    }
    resume_input_category_hashes = {
        k: canonical_json_sha256(v) for k, v in resume_file_state_payload.items()
    }
    return PreparedComboInputs(
        n_vertices=n_vertices,
        valid_masks=valid_masks,
        coverage_count=coverage_count,
        coverage_fraction=coverage_fraction,
        feature_bank=feature_bank,
        actual_input_paths=actual_input_paths,
        run_feature_banks=run_feature_banks,
        roi_fc_bank_for_redundancy=roi_fc_bank_for_redundancy,
        subject_fc_preprocessing_for_resume=subject_fc_preprocessing_for_resume,
        resume_input_category_hashes=resume_input_category_hashes,
    )


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run merged canonical group hippocampal atlas (v3 plan).")
    parser.add_argument("--shared-root", default=str(DEFAULT_SHARED_ROOT))
    parser.add_argument("--surface-root", default=str(DEFAULT_SURFACE_ROOT))
    parser.add_argument("--out-root", default=str(DEFAULT_OUT_ROOT))
    parser.add_argument("--gradient-root", default=str(DEFAULT_GRADIENT_ROOT))
    parser.add_argument("--hippomaps-root", default=str(DEFAULT_HIPPOMAPS_ROOT))
    parser.add_argument("--route-tag", choices=["auto", *ROUTE_TAG_CHOICES], default="auto")
    parser.add_argument("--subjects-manifest", default=str(DEFAULT_SUBJECTS))
    parser.add_argument("--hippunfold-batch-root", default=str(DEFAULT_HIPPUNFOLD_BATCH_ROOT))
    parser.add_argument("--kong2022-dlabel", default=None)
    parser.add_argument("--methods", nargs="+", default=METHODS_DEFAULT, choices=METHOD_CHOICES)
    parser.add_argument("--atlases", nargs="+", default=ATLASES)
    parser.add_argument("--smoothings", nargs="+", default=SMOOTHINGS)
    parser.add_argument("--hemis", nargs="+", default=HEMIS)
    parser.add_argument("--coverage-thresholds", nargs="+", type=float, default=[0.70, 0.80, 0.90])
    parser.add_argument("--k-min", type=int, default=2)
    parser.add_argument("--k-max", type=int, default=10)
    parser.add_argument(
        "--force-final-k",
        type=int,
        default=3,
        help="Force chosen_k_final to this K after evaluating candidate K rows; default is the maintained fixed K=3 display policy.",
    )
    parser.add_argument(
        "--auto-k",
        dest="force_final_k",
        action="store_const",
        const=None,
        help="Re-enable legacy automatic K selection; fixed K=3 remains the default.",
    )
    parser.add_argument("--instability-resamples", type=int, default=100)
    parser.add_argument("--null-resamples", type=int, default=100)
    parser.add_argument("--seed", type=int, default=20260420)
    parser.add_argument("--v-min-fraction", type=float, default=0.03)
    parser.add_argument("--max-components-threshold", type=int, default=3)
    parser.add_argument("--min-profile-coherence", type=float, default=0.0)
    parser.add_argument(
        "--subject-fc-mode",
        default="full",
        choices=SUBJECT_FC_MODE_CHOICES,
        help="Per-subject FC preprocessing mode for spectral methods: full (no sparsification) or top-percent (Top integer %% + retained entries binarized to 1).",
    )
    parser.add_argument(
        "--subject-fc-top-percent",
        type=int,
        default=None,
        help="Required only when --subject-fc-mode=top-percent. Integer percent in (0,100].",
    )
    parser.add_argument(
        "--gradient-components",
        default="1,2,3",
        help="Comma-separated gradient components for gradient-spectral clustering. "
             "Supported examples: 1, 2, 3, 1,2, 1,2,3. Default: 1,2,3. "
             "05_03 computes G1-G5 via BrainSpace/HippoMaps; this option selects clustering columns.",
    )
    parser.add_argument("--resume", action="store_true", help="Skip combos that already have group_atlas_manifest.json.")
    parser.add_argument("--k-selection-mode", default="redundancy-aware", choices=["experimental", "redundancy-aware"])
    parser.add_argument(
        "--redundancy-early-stop",
        action="store_true",
        help="Heuristic speed mode for redundancy-aware runs: evaluate K in ascending order and stop once the first redundant K is observed; larger K are skipped.",
    )
    parser.add_argument("--overview-threshold", type=float, default=0.80)
    parser.add_argument(
        "--jobs",
        type=int,
        default=1,
        help="Maximum parallel K evaluations per (method, atlas, smoothing, hemi, coverage) combo; capped at 10.",
    )
    parser.add_argument(
        "--group-average-surfaces-dir",
        default=str(DEFAULT_GROUP_AVG_SURFACES_DIR),
        help="Directory containing group-average template midthickness surfaces named "
             "group-average_hemi-{L|R}_space-corobl_den-512_label-hipp_midthickness.surf.gii.",
    )
    return parser


def infer_route_tag(shared_root: Path, explicit: str = "auto") -> str:
    if explicit != "auto":
        return explicit
    parts = {p.lower() for p in Path(shared_root).parts}
    name = Path(shared_root).name.lower()
    if "routea" in name or "routea_151" in parts:
        return "routea"
    if "routeb" in name or "routeb_151" in parts:
        return "routeb"
    raise ValueError(
        f"Could not infer --route-tag from --shared-root={shared_root}; pass --route-tag routea or routeb."
    )


def main() -> int:
    parser = build_arg_parser()
    args = parser.parse_args()
    suppress_known_runtime_warnings()

    global KONG2022_DLABEL_OVERRIDE, HIPPOMAPS_ROOT_OVERRIDE
    KONG2022_DLABEL_OVERRIDE = Path(args.kong2022_dlabel).resolve() if args.kong2022_dlabel else None
    HIPPOMAPS_ROOT_OVERRIDE = Path(args.hippomaps_root).resolve()

    shared_root = Path(args.shared_root).resolve()
    surface_root = Path(args.surface_root).resolve()
    out_root = Path(args.out_root).resolve()
    gradient_root = Path(args.gradient_root).resolve()
    route_tag = infer_route_tag(shared_root, str(args.route_tag))
    hippunfold_batch_root = Path(args.hippunfold_batch_root).resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    subjects_manifest_path = Path(args.subjects_manifest).resolve()
    subjects = load_subject_ids(subjects_manifest_path)
    subject_fc_mode = resolve_subject_fc_mode(args.subject_fc_mode, args.subject_fc_top_percent)
    gradient_components = parse_gradient_components(args.gradient_components)
    coverage_thresholds = [float(x) for x in args.coverage_thresholds]
    k_values = list(range(int(args.k_min), int(args.k_max) + 1))
    if not k_values:
        raise ValueError(f"Empty K range: --k-min={args.k_min} --k-max={args.k_max}")
    if args.force_final_k is not None and int(args.force_final_k) not in k_values:
        raise ValueError(
            f"--force-final-k must be within requested K range [{min(k_values)}, {max(k_values)}], "
            f"got {args.force_final_k}"
        )
    rng = np.random.RandomState(int(args.seed))
    split_seeds = [int(x) for x in rng.randint(1, 2_000_000_000, size=int(args.instability_resamples))]
    null_seeds = [int(x) for x in rng.randint(1, 2_000_000_000, size=int(args.null_resamples))]
    config = GroupAtlasConfig(
        args=args,
        shared_root=shared_root,
        surface_root=surface_root,
        out_root=out_root,
        gradient_root=gradient_root,
        route_tag=route_tag,
        hippunfold_batch_root=hippunfold_batch_root,
        subjects_manifest_path=subjects_manifest_path,
        subjects=subjects,
        subject_fc_mode=subject_fc_mode,
        gradient_components=gradient_components,
        coverage_thresholds=coverage_thresholds,
        k_values=k_values,
        split_seeds=split_seeds,
        null_seeds=null_seeds,
    )

    group_avg_dir = Path(args.group_average_surfaces_dir).resolve()
    surfaces = {
        hemi: group_avg_dir / f"group-average_hemi-{hemi}_space-corobl_den-512_label-hipp_midthickness.surf.gii"
        for hemi in args.hemis
    }
    missing_template = [str(p) for p in surfaces.values() if not p.exists()]
    if missing_template:
        raise FileNotFoundError(
            "Missing required group-average template surfaces. "
            "Build template first (prepare_t1only_group_inputs.py --stage group-average) and point "
            "--group-average-surfaces-dir to the shared template directory.\n"
            + "\n".join(missing_template)
        )
    template_label = "group-average"
    template_source_mode = "group_average_template"
    template_source_note = (
        "Template-first mode: clustering topology/order is derived from shared group-average template surfaces; "
        "manifest subjects are validated for topology compatibility."
    )
    template_render_subject = f"sub-{subjects[0]}"
    template_spec_path = infer_spec_path(
        subject=subjects[0],
        surface_path=infer_surface_path(subjects[0], args.hemis[0], hippunfold_batch_root),
    )
    topology_audit = verify_subject_topology_against_template(
        subjects,
        list(args.hemis),
        surfaces,
        hippunfold_batch_root,
    )
    long_axis_order_by_hemi: dict[str, np.ndarray] = {
        hemi: long_axis_order(surfaces[hemi]) for hemi in args.hemis
    }
    surface_meta: dict[str, dict[str, object]] = {}
    adjacency_by_hemi: dict[str, sparse.csr_matrix] = {}
    for hemi in config.args.hemis:
        coords, faces = load_surface(surfaces[hemi])
        adjacency_by_hemi[hemi] = build_surface_adjacency(faces, int(coords.shape[0]))
        surface_meta[hemi] = {
            "surface_path": str(surfaces[hemi]),
            "n_vertices": int(coords.shape[0]),
            "n_faces": int(faces.shape[0]),
        }

    if "intrinsic-spectral" in args.methods:
        for smoothing in config.args.smoothings:
            for hemi in config.args.hemis:
                print(f"[check] intrinsic atlas-invariance smoothing={smoothing} hemi={hemi}", flush=True)
                verify_intrinsic_atlas_invariance(shared_root, subjects, smoothing, hemi, list(args.atlases))

    colors = load_cluster_colors()
    canonical_color_map = load_canonical_network_color_map()
    canonical_network_order_by_atlas: dict[str, list[str]] = {}

    for atlas in config.args.atlases:
        print(f"[check] canonical-network ordering atlas={atlas}", flush=True)
        canonical_network_order_by_atlas[atlas] = verify_canonical_network_order_consistency(shared_root, subjects, atlas)

    for method in config.args.methods:
        for atlas in config.args.atlases:
            if method in KONG2022_ONLY_METHODS and atlas != "kong2022":
                print(
                    f"[warn] skipping unsupported combo atlas={atlas} method={method} "
                    "(method is defined for kong2022 only)",
                    flush=True,
                )
                continue
            network_names = canonical_network_order_by_atlas[atlas]
            for smoothing in config.args.smoothings:
                for hemi in config.args.hemis:
                    prepared = prepare_combo_inputs(
                        method=method,
                        atlas=atlas,
                        smoothing=smoothing,
                        hemi=hemi,
                        shared_root=shared_root,
                        surface_root=surface_root,
                        surfaces=surfaces,
                        subjects=subjects,
                        subject_fc_mode=subject_fc_mode,
                        k_selection_mode=str(args.k_selection_mode),
                    )
                    n_vertices = int(prepared.n_vertices)
                    valid_masks = prepared.valid_masks
                    coverage_count = prepared.coverage_count
                    coverage_fraction = prepared.coverage_fraction
                    feature_bank = prepared.feature_bank
                    actual_input_paths = prepared.actual_input_paths
                    run_feature_banks = prepared.run_feature_banks
                    roi_fc_bank_for_redundancy = prepared.roi_fc_bank_for_redundancy
                    subject_fc_preprocessing_for_resume = prepared.subject_fc_preprocessing_for_resume
                    resume_input_category_hashes = prepared.resume_input_category_hashes

                    for tau in config.coverage_thresholds:
                        combo = Combo(method=method, atlas=atlas, smoothing=smoothing, hemi=hemi, coverage_threshold=float(tau))
                        print(
                            f"[combo] method={method} atlas={atlas} smoothing={smoothing} hemi={hemi} tau={tau:.2f}",
                            flush=True,
                        )
                        tau_tag = str(f"{tau:.2f}").replace(".", "p")
                        run_dir = out_root / method / atlas / smoothing / f"cov_{tau_tag}" / f"hemi_{hemi}"
                        run_dir.mkdir(parents=True, exist_ok=True)
                        manifest_path = run_dir / "group_atlas_manifest.json"
                        resume_signature_payload = {
                            "resume_input_signature_version": int(RESUME_SIGNATURE_VERSION),
                            "combo": {
                                "method": method,
                                "atlas": atlas,
                                "smoothing": smoothing,
                                "hemi": hemi,
                                "coverage_threshold": float(tau),
                            },
                            "subjects_manifest": str(subjects_manifest_path),
                            "subjects": [f"sub-{s}" for s in subjects],
                            "k_selection_mode": str(args.k_selection_mode),
                            "force_final_k": int(args.force_final_k) if args.force_final_k is not None else None,
                            "candidate_k_values": [int(k) for k in k_values],
                            "seed": int(args.seed),
                            "instability_resamples": int(args.instability_resamples),
                            "null_resamples": int(args.null_resamples),
                            "v_min_fraction": float(args.v_min_fraction),
                            "max_components_threshold": int(args.max_components_threshold),
                            "min_profile_coherence": float(args.min_profile_coherence),
                            "subject_fc_preprocessing": subject_fc_preprocessing_for_resume,
                            "gradient_components": [int(c) for c in gradient_components],
                            "gradient_root": str(gradient_root) if method == "gradient-spectral" else None,
                            "route_tag": route_tag if method == "gradient-spectral" else None,
                            "redundancy_early_stop": bool(args.redundancy_early_stop),
                            "input_category_hashes": resume_input_category_hashes,
                        }
                        resume_input_signature = canonical_json_sha256(resume_signature_payload)
                        if args.resume and manifest_path.exists():
                            skip_ok, resume_reason = resume_manifest_matches(manifest_path, resume_input_signature)
                            if skip_ok:
                                print(
                                    f"  [skip] existing manifest (resume-signature matched): {manifest_path}",
                                    flush=True,
                                )
                                continue
                            print(
                                f"  [resume-invalidate] {manifest_path} reason={resume_reason}; recomputing",
                                flush=True,
                            )

                        support = coverage_fraction >= float(tau)
                        subject_fc_preprocessing = subject_fc_preprocessing_for_resume
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
                            "max_profile_similarity",
                            "redundant_pair_count",
                            "nonredundant_ok",
                            "reject_nontrivial_fail",
                            "reject_redundant",
                            "reject_reason_primary",
                            "selected_final",
                            "within_1se_survivor_pool",
                            "survivor_pool_argmax_k",
                            "k_final_decision_path",
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
                            if method != "wta-top":
                                raise ValueError(f"Unknown clusterless method: {method}")
                            vertex_status_rows = assignment_status_for_vertices_top(
                                profile_3d=vertex_profile_3d,
                                support_indices=support_indices,
                                roi_registry=roi_registry,
                                q_thresh=0.05,
                                top_n_cap=20,
                            )
                            confidence_metric = "topn_count"

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
                            write_json(run_dir / "group_functional_atlas_label_map.json", functional_meta)
                            np.save(run_dir / "vertex_roi_profile.npy", vertex_profile_mean.astype(np.float32))
                            np.save(run_dir / "vertex_support_indices.npy", support_indices.astype(np.int32))
                            write_json(run_dir / "vertex_assignment_status.json", {"items": vertex_status_rows})
                            write_json(run_dir / "global_top_connections.json", global_top)
                            write_json(run_dir / "k_selection_summary.json", {"rows": [], "k_final": None, "mode": "clusterless"})
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
                                "resume_input_signature_version": int(RESUME_SIGNATURE_VERSION),
                                "resume_input_signature": resume_input_signature,
                                "resume_input_signature_payload": resume_signature_payload,
                                "feature_family": "vertex_to_kong2022_400roi_fc_fisherz",
                                "fc_mode": "pearson_vertex_to_roi",
                                "subject_fc_preprocessing": subject_fc_preprocessing,
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
                                "template_subject": template_render_subject,
                                "template_source_mode": template_source_mode,
                                "template_source_note": template_source_note,
                                "template_left_or_right_surface": str(surfaces[hemi]),
                                "template_spec_path": str(template_spec_path),
                                "template_cluster_label": None,
                                "template_functional_atlas_label": str(func_lbl_path.resolve()),
                            }
                            write_tsv(
                                run_dir / "cluster_roi_registry.tsv",
                                roi_registry,
                                ["roi_id", "roi_name", "hemi", "family17", "subregion", "rank", "canonical8"],
                            )
                            write_json(run_dir / "group_feature_summary.json", {"network_names": network_names, "N_ref": int(len(network_names))})
                            write_json(manifest_path, manifest)
                            continue

                        if not k_eff:
                            skip_reason = "no_valid_k_for_support"
                            skip_detail = (
                                f"No valid K for combo={combo} n_support={n_support} "
                                f"v_min={v_min} k_max_eff={k_max_eff}"
                            )
                            write_tsv(run_dir / "k_selection_summary.tsv", [], k_selection_fields)
                            write_json(
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
                                "force_final_k": int(args.force_final_k) if args.force_final_k is not None else None,
                                "coverage_threshold": float(tau),
                                "V_min_fraction": float(args.v_min_fraction),
                                "V_min_effective": int(v_min),
                                "gradient_components": [int(c) for c in gradient_components] if is_gradient_method(method) else None,
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
                                "resume_input_signature_version": int(RESUME_SIGNATURE_VERSION),
                                "resume_input_signature": resume_input_signature,
                                "resume_input_signature_payload": resume_signature_payload,
                                "subject_fc_preprocessing": subject_fc_preprocessing,
                                "group_gradient_features_active": (
                                    str((run_dir / "group_gradient_features_active.npy").resolve())
                                    if is_gradient_method(method)
                                    else None
                                ),
                                "group_gradient_feature_summary": (
                                    str((run_dir / "group_gradient_feature_summary.json").resolve())
                                    if is_gradient_method(method)
                                    else None
                                ),
                                "nontriviality_thresholds": {
                                    "V_min_fraction": float(args.v_min_fraction),
                                    "max_components_threshold": int(args.max_components_threshold),
                                    "min_profile_coherence": float(args.min_profile_coherence),
                                },
                                "redundancy_aware_thresholds": {
                                    "max_profile_similarity_lte": float(PROFILE_REDUNDANCY_MAX_SIM),
                                    "redundant_pair_count_eq": 0,
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
                            write_json(manifest_path, manifest)
                            print(f"  [skip] {skip_detail}", flush=True)
                            continue

                        rows: list[dict[str, object]] = []
                        labels_by_k: dict[int, np.ndarray] = {}
                        support_idx = np.flatnonzero(support)
                        adjacency_support = adjacency_by_hemi[hemi][support_idx][:, support_idx]
                        group_gradient_meta: dict[str, object] | None = None
                        if is_gradient_method(method):
                            group_features_active, group_gradient_meta = load_hippomaps_gradient_features(
                                gradient_root=gradient_root,
                                route=route_tag,
                                smoothing=smoothing,
                                hemi=hemi,
                                support_mask=support,
                                gradient_components=gradient_components,
                            )
                            np.save(run_dir / "group_gradient_features_active.npy", group_features_active.astype(np.float32))
                            write_json(run_dir / "group_gradient_feature_summary.json", group_gradient_meta)
                        else:
                            group_features_active = group_feature_matrix(
                                method=method,
                                feature_bank=feature_bank,
                                subjects_subset=subjects,
                                support_mask=support,
                                gradient_components=gradient_components,
                            )
                        redundancy_early_stop_enabled = bool(
                            args.redundancy_early_stop and args.k_selection_mode == "redundancy-aware"
                        )
                        if redundancy_early_stop_enabled:
                            print(
                                f"  [k-seq-heuristic] n_k={len(k_eff)} stop_on_first_redundant=1",
                                flush=True,
                            )
                            for k in k_eff:
                                print(f"  [k-start] {k}", flush=True)
                                k_result, labels_full, row = evaluate_k_candidate(
                                    combo=combo,
                                    method=method,
                                    k=int(k),
                                    seed=int(args.seed + 11 * k),
                                    n_support=int(n_support),
                                    v_min=int(v_min),
                                    group_features_active=group_features_active,
                                    support_mask=support,
                                    adjacency_full=adjacency_by_hemi[hemi],
                                    adjacency_support=adjacency_support,
                                    feature_bank=feature_bank,
                                    valid_masks=valid_masks,
                                    subjects=subjects,
                                    tau=float(tau),
                                    split_seeds=split_seeds,
                                    null_seeds=null_seeds,
                                    null_resamples=int(args.null_resamples),
                                    run_feature_banks=run_feature_banks,
                                    seed_base=int(args.seed),
                                    max_components_threshold=int(args.max_components_threshold),
                                    min_profile_coherence_threshold=float(args.min_profile_coherence),
                                    k_selection_mode=str(args.k_selection_mode),
                                    roi_fc_bank_for_redundancy=roi_fc_bank_for_redundancy,
                                    gradient_components=gradient_components,
                                )
                                print(f"  [k-done] {k_result}", flush=True)
                                labels_by_k[int(k_result)] = labels_full
                                np.save(run_dir / f"group_labels_k{k_result}.npy", labels_full.astype(np.int32))
                                rows.append(row)
                                if redundancy_early_stop_triggered(row, redundancy_early_stop_enabled):
                                    print(
                                        f"  [k-early-stop] first_redundant_k={k_result}; skipping larger K",
                                        flush=True,
                                    )
                                    break
                        else:
                            k_eval_jobs = resolve_k_eval_jobs(int(args.jobs), len(k_eff))
                            print(
                                f"  [k-parallel] n_k={len(k_eff)} jobs={k_eval_jobs}",
                                flush=True,
                            )
                            with ThreadPoolExecutor(max_workers=k_eval_jobs) as executor:
                                future_to_k = {
                                    executor.submit(
                                        evaluate_k_candidate,
                                        combo=combo,
                                        method=method,
                                        k=int(k),
                                        seed=int(args.seed + 11 * k),
                                        n_support=int(n_support),
                                        v_min=int(v_min),
                                        group_features_active=group_features_active,
                                        support_mask=support,
                                        adjacency_full=adjacency_by_hemi[hemi],
                                        adjacency_support=adjacency_support,
                                        feature_bank=feature_bank,
                                        valid_masks=valid_masks,
                                        subjects=subjects,
                                        tau=float(tau),
                                        split_seeds=split_seeds,
                                        null_seeds=null_seeds,
                                        null_resamples=int(args.null_resamples),
                                        run_feature_banks=run_feature_banks,
                                        seed_base=int(args.seed),
                                        max_components_threshold=int(args.max_components_threshold),
                                        min_profile_coherence_threshold=float(args.min_profile_coherence),
                                        k_selection_mode=str(args.k_selection_mode),
                                        roi_fc_bank_for_redundancy=roi_fc_bank_for_redundancy,
                                        gradient_components=gradient_components,
                                    ): int(k)
                                    for k in k_eff
                                }
                                for future in as_completed(future_to_k):
                                    k_done = int(future_to_k[future])
                                    print(f"  [k-done] {k_done}", flush=True)
                                    k_result, labels_full, row = future.result()
                                    labels_by_k[int(k_result)] = labels_full
                                    np.save(run_dir / f"group_labels_k{k_result}.npy", labels_full.astype(np.int32))
                                    rows.append(row)

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
                            # Survivor-pool audit columns; overwritten after k_final is chosen.
                            row["within_1se_survivor_pool"] = 0
                            row["survivor_pool_argmax_k"] = -1
                            row["k_final_decision_path"] = "none"
                        forced_fixed_k_target = (
                            int(args.force_final_k)
                            if args.force_final_k is not None
                            else (int(k_values[0]) if len(k_values) == 1 else None)
                        )
                        if forced_fixed_k_target is not None and not any(
                            int(row["k"]) == int(forced_fixed_k_target) for row in rows
                        ):
                            raise RuntimeError(
                                "Forced final K was not evaluated for this combo: "
                                f"k={forced_fixed_k_target}, evaluated={[int(r['k']) for r in rows]}, combo={combo}"
                            )
                        forced_fixed_k = forced_fixed_k_target is not None
                        if forced_fixed_k:
                            k_forced = int(forced_fixed_k_target)
                            for row in rows:
                                is_target = int(row["k"]) == k_forced
                                row["selected_final"] = int(is_target)
                                row["within_1se_survivor_pool"] = int(is_target)
                                row["survivor_pool_argmax_k"] = int(k_forced)
                                row["k_final_decision_path"] = "forced_fixed_k"
                                row["reject_nontrivial_fail"] = 0
                                row["reject_redundant"] = 0
                                row["reject_reason_primary"] = ""
                            survivors = [row for row in rows if int(row["selected_final"]) == 1]
                        elif args.k_selection_mode == "redundancy-aware":
                            for row in rows:
                                row["nonredundant_ok"] = int(
                                    np.isfinite(float(row["max_profile_similarity"]))
                                    and float(row["max_profile_similarity"]) <= float(PROFILE_REDUNDANCY_MAX_SIM)
                                    and int(row["redundant_pair_count"]) == 0
                                )
                                row["reject_nontrivial_fail"] = int(int(row["selected_nontrivial"]) != 1)
                                row["reject_redundant"] = int(int(row["nonredundant_ok"]) != 1)
                                row["selected_final"] = 0
                                if int(row["reject_nontrivial_fail"]) == 1:
                                    row["reject_reason_primary"] = "nontrivial_fail"
                                elif int(row["reject_redundant"]) == 1:
                                    row["reject_reason_primary"] = "redundant"
                                else:
                                    row["reject_reason_primary"] = "outside_selection_rule"
                            candidate_pool = [
                                row
                                for row in rows
                                if int(row["selected_nontrivial"]) == 1 and int(row["nonredundant_ok"]) == 1
                            ]
                            if not candidate_pool:
                                survivors = []
                            else:
                                k_final, decision_path, within_1se_pool, survivor_argmax_k = (
                                    select_k_final_after_redundancy_filter(candidate_pool)
                                )
                                for row in rows:
                                    row["survivor_pool_argmax_k"] = int(survivor_argmax_k)
                                    row["within_1se_survivor_pool"] = int(int(row["k"]) in within_1se_pool)
                                    row["k_final_decision_path"] = decision_path
                                    if int(row["k"]) == int(k_final):
                                        row["selected_final"] = 1
                                        row["reject_reason_primary"] = ""
                                survivors = [r for r in rows if int(r["selected_final"]) == 1]
                        else:
                            for row in rows:
                                row["selected_final"] = int(
                                    int(row["local_max"]) == 1
                                    and int(row["within_1se_best"]) == 1
                                    and int(row["selected_nontrivial"]) == 1
                                )
                                row["reject_nontrivial_fail"] = int(int(row["selected_nontrivial"]) != 1)
                                row["reject_redundant"] = 0
                                row["nonredundant_ok"] = 1
                                if int(row["selected_final"]) == 1:
                                    row["reject_reason_primary"] = ""
                                elif int(row["reject_nontrivial_fail"]) == 1:
                                    row["reject_reason_primary"] = "nontrivial_fail"
                                else:
                                    row["reject_reason_primary"] = "outside_selection_rule"
                            survivors = [r for r in rows if int(r["selected_final"]) == 1]
                        if not survivors:
                            skip_reason = "no_k_survived_selection_constraints"
                            skip_detail = f"No K survived selection constraints for combo={combo}"
                            write_tsv(run_dir / "k_selection_summary.tsv", rows, k_selection_fields)
                            write_json(
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
                                "force_final_k": int(args.force_final_k) if args.force_final_k is not None else None,
                                "coverage_threshold": float(tau),
                                "V_min_fraction": float(args.v_min_fraction),
                                "V_min_effective": int(v_min),
                                "feature_family": feature_family_for_method(method, atlas, gradient_components),
                                "fc_mode": fc_mode_for_method(method),
                                "gradient_components": [int(c) for c in gradient_components] if is_gradient_method(method) else None,
                                "candidate_k_range_requested": [int(min(k_values)), int(max(k_values))],
                                "candidate_k_range_effective": [int(min(k_eff)), int(max(k_eff))],
                                "candidate_k_values_evaluated": [int(r["k"]) for r in rows],
                                "redundancy_early_stop_enabled": bool(redundancy_early_stop_enabled),
                                "chosen_k_final": None,
                                "subject_split_seeds": split_seeds,
                                "null_seeds": null_seeds,
                                "n_subjects": int(len(subjects)),
                                "subjects": [f"sub-{s}" for s in subjects],
                                "n_support_vertices": int(n_support),
                                "n_excluded_vertices": int(n_vertices - n_support),
                                "actual_consumed_input_paths": actual_input_paths,
                                "resume_input_signature_version": int(RESUME_SIGNATURE_VERSION),
                                "resume_input_signature": resume_input_signature,
                                "resume_input_signature_payload": resume_signature_payload,
                                "subject_fc_preprocessing": subject_fc_preprocessing,
                                "group_gradient_features_active": (
                                    str((run_dir / "group_gradient_features_active.npy").resolve())
                                    if is_gradient_method(method)
                                    else None
                                ),
                                "group_gradient_feature_summary": (
                                    str((run_dir / "group_gradient_feature_summary.json").resolve())
                                    if is_gradient_method(method)
                                    else None
                                ),
                                "nontriviality_thresholds": {
                                    "V_min_fraction": float(args.v_min_fraction),
                                    "max_components_threshold": int(args.max_components_threshold),
                                    "min_profile_coherence": float(args.min_profile_coherence),
                                },
                                "redundancy_aware_thresholds": {
                                    "max_profile_similarity_lte": float(PROFILE_REDUNDANCY_MAX_SIM),
                                    "redundant_pair_count_eq": 0,
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
                            write_json(manifest_path, manifest)
                            print(f"  [skip] {skip_detail}", flush=True)
                            continue
                        if args.k_selection_mode != "redundancy-aware":
                            # Three-tier K selection: 1-SE on survivor pool → local-max
                            # preference → parsimony tiebreak. Annotate audit columns.
                            surv_means = np.asarray(
                                [float(s["group_ari_mean"]) for s in survivors], dtype=np.float64
                            )
                            surv_best_idx = int(np.argmax(surv_means))
                            surv_cutoff = float(surv_means[surv_best_idx]) - float(
                                survivors[surv_best_idx]["group_ari_se"]
                            )
                            surv_argmax_k = int(survivors[surv_best_idx]["k"])
                            k_final, decision_path = select_k_final_from_survivors(survivors)
                            for row in rows:
                                row["survivor_pool_argmax_k"] = surv_argmax_k
                                row["within_1se_survivor_pool"] = int(
                                    int(row["selected_final"]) == 1
                                    and float(row["group_ari_mean"]) >= surv_cutoff - 1e-12
                                )
                                row["k_final_decision_path"] = decision_path
                        else:
                            selected_rows = [row for row in rows if int(row["selected_final"]) == 1]
                            if len(selected_rows) != 1:
                                raise RuntimeError(
                                    f"Expected exactly one selected_final row for combo={combo}, got {len(selected_rows)}"
                                )
                            k_final = int(selected_rows[0]["k"])
                            decision_path = str(selected_rows[0]["k_final_decision_path"])
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
                        roi_topn_status_rows: list[dict[str, object]] | None = None
                        if method_uses_kong2022_topn_cluster_annotation(method, atlas):
                            (
                                profile,
                                status_rows,
                                dominant,
                                roi_profile,
                                roi_registry,
                                roi_profile_subjects_used,
                                roi_top20,
                                roi_topn_status_rows,
                            ) = annotate_kong2022_clusters_topn(
                                group_labels_final=labels_final,
                                support_mask=support,
                                subjects=subjects,
                                shared_root=shared_root,
                                surface_root=surface_root,
                                smoothing=smoothing,
                                hemi=hemi,
                                valid_masks_by_subject=valid_masks,
                                network_names=network_names,
                                q_thresh=0.05,
                                top_n_cap=20,
                            )
                        else:
                            profile = cluster_network_profile(
                                group_labels_final=labels_final,
                                support_mask=support,
                                subjects=subjects,
                                shared_root=shared_root,
                                surface_root=surface_root,
                                atlas=atlas,
                                smoothing=smoothing,
                                hemi=hemi,
                                valid_masks_by_subject=valid_masks,
                                profile_fc_mode="pearson",
                            )
                            status_rows, dominant = assignment_status_from_profile(profile, network_names)
                        if atlas == "kong2022" and roi_profile is None:
                            roi_profile, roi_registry = cluster_roi_profile(
                                group_labels_final=labels_final,
                                support_mask=support,
                                subjects=subjects,
                                shared_root=shared_root,
                                surface_root=surface_root,
                                smoothing=smoothing,
                                hemi=hemi,
                                valid_masks_by_subject=valid_masks,
                            )
                        if atlas == "kong2022" and roi_top20 is None:
                            assert roi_profile is not None and roi_registry is not None
                            roi_top20 = cluster_roi_top_k(roi_profile, roi_registry, k_top=20)

                        np.save(run_dir / "group_labels_final.npy", labels_final.astype(np.int32))
                        np.save(run_dir / "cluster_network_profile.npy", profile.astype(np.float32))

                        assignment_variant_specs: dict[str, dict[str, object]] = {
                            "fisherz": {
                                "status_rows": status_rows,
                                "dominant": dominant,
                                "assignment_metric": "top1_fisher_z",
                                "q_threshold": None,
                                "top_n_cap": None,
                            }
                        }
                        if roi_topn_status_rows is not None:
                            assignment_variant_specs["count"] = {
                                "status_rows": roi_topn_status_rows,
                                "dominant": [str(row.get("top1_network", "unassigned")) for row in roi_topn_status_rows],
                                "assignment_metric": "topn_count_positive_roi",
                                "q_threshold": 0.05,
                                "top_n_cap": 20,
                            }

                        for variant_name, variant_spec in assignment_variant_specs.items():
                            payload = {
                                "items": [dict(x) for x in variant_spec["status_rows"]],
                                "assignment_metric": str(variant_spec["assignment_metric"]),
                                "q_threshold": variant_spec["q_threshold"],
                                "top_n_cap": variant_spec["top_n_cap"],
                            }
                            write_json(run_dir / f"cluster_assignment_status_{variant_name}.json", payload)
                            if variant_name == "fisherz":
                                write_json(run_dir / "cluster_assignment_status.json", payload)
                            if variant_name == "count":
                                write_json(run_dir / "cluster_roi_topn_assignment_status.json", payload)
                        write_json(
                            run_dir / "cluster_dominant_network.json",
                            {"dominant": dominant, "network_names": network_names, "assignment_variant": "fisherz"},
                        )
                        if "count" in assignment_variant_specs:
                            write_json(
                                run_dir / "cluster_dominant_network_count.json",
                                {
                                    "dominant": assignment_variant_specs["count"]["dominant"],
                                    "network_names": network_names,
                                    "assignment_variant": "count",
                                },
                            )
                        if atlas == "kong2022":
                            assert roi_profile is not None and roi_registry is not None and roi_top20 is not None
                            np.save(run_dir / "cluster_roi_profile.npy", roi_profile.astype(np.float32))
                            write_json(run_dir / "cluster_roi_top20.json", roi_top20)
                            write_tsv(
                                run_dir / "cluster_roi_registry.tsv",
                                roi_registry,
                                ["roi_id", "roi_name", "hemi", "family17", "subregion", "rank", "canonical8"],
                            )

                        functional_variant_payloads: dict[str, dict[str, object]] = {}
                        functional_labels = np.zeros_like(labels_final, dtype=np.int32)
                        functional_label_items: list[tuple[int, str, tuple[int, int, int, int]]] = []
                        for variant_name, variant_spec in assignment_variant_specs.items():
                            labels_variant, meta_variant, items_variant = functional_atlas_from_clusters(
                                group_labels_final=labels_final,
                                status_rows=[dict(x) for x in variant_spec["status_rows"]],
                                network_names=network_names,
                                color_map=canonical_color_map,
                            )
                            labels_variant = labels_variant.astype(np.int32)
                            np.save(run_dir / f"group_functional_atlas_labels_{variant_name}.npy", labels_variant)
                            write_json(run_dir / f"group_functional_atlas_label_map_{variant_name}.json", meta_variant)
                            functional_variant_payloads[variant_name] = {
                                "assignment_metric": str(variant_spec["assignment_metric"]),
                                "cluster_assignment_status": str((run_dir / f"cluster_assignment_status_{variant_name}.json").resolve()),
                                "group_functional_atlas_labels": str((run_dir / f"group_functional_atlas_labels_{variant_name}.npy").resolve()),
                                "group_functional_atlas_label_map": str((run_dir / f"group_functional_atlas_label_map_{variant_name}.json").resolve()),
                            }
                            if variant_name == "fisherz":
                                functional_labels = labels_variant
                                functional_label_items = items_variant
                                np.save(run_dir / "group_functional_atlas_labels.npy", labels_variant)
                                write_json(run_dir / "group_functional_atlas_label_map.json", meta_variant)

                        write_tsv(
                            run_dir / "k_selection_summary.tsv",
                            rows,
                            k_selection_fields,
                        )
                        write_json(run_dir / "k_selection_summary.json", {"rows": rows, "k_final": int(k_final)})

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
                        func_lbl_path = (
                            run_dir
                            / "template"
                            / "workbench_assets"
                            / f"{template_label}_hemi-{hemi}_space-corobl_den-512_label-group_functional_atlas_{method}_{atlas}_{smoothing}_cov-{tau_tag}.label.gii"
                        )
                        func_lbl_path.parent.mkdir(parents=True, exist_ok=True)
                        for variant_name, variant_spec in assignment_variant_specs.items():
                            labels_path = run_dir / f"group_functional_atlas_labels_{variant_name}.npy"
                            map_path = run_dir / f"group_functional_atlas_label_map_{variant_name}.json"
                            labels_variant = np.load(labels_path).astype(np.int32)
                            _labels_tmp, _meta_tmp, items_variant = functional_atlas_from_clusters(
                                group_labels_final=labels_final,
                                status_rows=[dict(x) for x in variant_spec["status_rows"]],
                                network_names=network_names,
                                color_map=canonical_color_map,
                            )
                            func_lbl_variant = make_named_label_gifti(labels_variant, items_variant)
                            if variant_name == "fisherz":
                                variant_lbl_path = func_lbl_path
                            else:
                                variant_lbl_path = func_lbl_path.with_name(f"{func_lbl_path.stem}_{variant_name}{func_lbl_path.suffix}")
                            nib.save(func_lbl_variant, str(variant_lbl_path))
                            suffixed_lbl_path = func_lbl_path.with_name(f"{func_lbl_path.stem}_{variant_name}{func_lbl_path.suffix}")
                            if suffixed_lbl_path != variant_lbl_path:
                                nib.save(func_lbl_variant, str(suffixed_lbl_path))
                            functional_variant_payloads[variant_name].update(
                                {
                                    "group_functional_atlas_labels": str(labels_path.resolve()),
                                    "group_functional_atlas_label_map": str(map_path.resolve()),
                                    "template_functional_atlas_label": str(variant_lbl_path.resolve()),
                                    "template_functional_atlas_label_variant": str(suffixed_lbl_path.resolve()),
                                }
                            )

                        manifest = {
                            "status": "ok",
                            "method": method,
                            "atlas": atlas,
                            "smoothing": smoothing,
                            "hemi": hemi,
                            "assignment_mode": "cluster_level",
                            "cluster_assignment_metric": "top1_fisher_z",
                            "default_assignment_variant": "fisherz",
                            "assignment_variants": functional_variant_payloads,
                            "cluster_roi_topn_assignment_metric": (
                                "topn_count_positive_roi" if "count" in assignment_variant_specs else None
                            ),
                            "k_selection_mode": str(args.k_selection_mode),
                            "force_final_k": int(args.force_final_k) if args.force_final_k is not None else None,
                            "coverage_threshold": float(tau),
                            "V_min_fraction": float(args.v_min_fraction),
                            "V_min_effective": int(v_min),
                            "feature_family": feature_family_for_method(method, atlas, gradient_components),
                            "fc_mode": fc_mode_for_method(method),
                            "gradient_components": [int(c) for c in gradient_components] if is_gradient_method(method) else None,
                            "actual_consumed_input_paths": actual_input_paths,
                            "resume_input_signature_version": int(RESUME_SIGNATURE_VERSION),
                            "resume_input_signature": resume_input_signature,
                            "resume_input_signature_payload": resume_signature_payload,
                            "subject_fc_preprocessing": subject_fc_preprocessing,
                            "group_gradient_features_active": (
                                str((run_dir / "group_gradient_features_active.npy").resolve())
                                if is_gradient_method(method)
                                else None
                            ),
                            "group_gradient_feature_summary": (
                                str((run_dir / "group_gradient_feature_summary.json").resolve())
                                if is_gradient_method(method)
                                else None
                            ),
                            "clustering_algorithm": "spectral_clustering_precomputed_affinity_relu_cosine_diag0_kmeans_n_init_50",
                            "affinity_description": "max(cosine_similarity,0)_hadamard_surface_adjacency_diag0",
                            "candidate_k_range_requested": [int(min(k_values)), int(max(k_values))],
                            "candidate_k_range_effective": [int(min(k_eff)), int(max(k_eff))],
                            "candidate_k_values_evaluated": [int(r["k"]) for r in rows],
                            "redundancy_early_stop_enabled": bool(redundancy_early_stop_enabled),
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
                            "redundancy_aware_thresholds": {
                                "max_profile_similarity_lte": float(PROFILE_REDUNDANCY_MAX_SIM),
                                "redundant_pair_count_eq": 0,
                            },
                            "density_evidence": {
                                "surface_path": str(surfaces[hemi]),
                                "n_vertices": int(surface_meta[hemi]["n_vertices"]),
                                "n_faces": int(surface_meta[hemi]["n_faces"]),
                            },
                            "surface_topology_audit": topology_audit[hemi],
                            "canonical_network_order_consistency_checked": True,
                            "canonical_network_order_consensus_subject_count": int(len(subjects)),
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
                            "cluster_assignment_status_fisherz": str((run_dir / "cluster_assignment_status_fisherz.json").resolve()),
                            "cluster_assignment_status_count": (
                                str((run_dir / "cluster_assignment_status_count.json").resolve())
                                if "count" in assignment_variant_specs
                                else None
                            ),
                            "cluster_roi_topn_assignment_status": (
                                str((run_dir / "cluster_roi_topn_assignment_status.json").resolve())
                                if "count" in assignment_variant_specs
                                else None
                            ),
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
                            "template_subject": template_render_subject,
                            "template_source_mode": template_source_mode,
                            "template_source_note": template_source_note,
                            "cluster_label_ordering": "long_axis_head_to_tail",
                            "cluster_order_head_to_tail": fixed_cluster_order,
                            "template_left_or_right_surface": str(surfaces[hemi]),
                            "template_spec_path": str(template_spec_path),
                            "template_cluster_label": str(lbl_path.resolve()),
                            "template_functional_atlas_label": str(func_lbl_path.resolve()),
                        }
                        feature_summary: dict[str, object] = {"network_names": network_names, "N_ref": int(len(network_names))}
                        if group_gradient_meta is not None:
                            feature_summary["gradient"] = group_gradient_meta
                        write_json(run_dir / "group_feature_summary.json", feature_summary)
                        write_json(manifest_path, manifest)

    print(json.dumps({"out_root": str(out_root), "status": "ok"}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
