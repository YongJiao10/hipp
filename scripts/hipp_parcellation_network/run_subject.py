#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import nibabel as nib
import numpy as np
from nibabel.gifti import GiftiDataArray, GiftiImage, GiftiLabel, GiftiLabelTable
from scipy import sparse
from scipy.optimize import linear_sum_assignment
from scipy.sparse import csgraph
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import (
    adjusted_rand_score,
    calinski_harabasz_score,
    davies_bouldin_score,
    silhouette_score,
)

REPO_ROOT = Path(__file__).resolve().parents[2]


COMMON_DIR = REPO_ROOT / "scripts" / "common"
if str(COMMON_DIR) not in sys.path:
    sys.path.insert(0, str(COMMON_DIR))

from compute_fc_gradients import build_sparse_affinity, corrcoef_rows, diffusion_map_embedding
from spectral_clustering import (
    fisher_z_transform_fc,
    prepare_intrinsic_spectral_features,
    spectral_cluster_from_features,
)
from hipp_density_assets import (
    find_cifti_asset_strict,
    find_surface_asset_strict,
    load_surface_density_from_pipeline_config,
)


WB_COMMAND = str((REPO_ROOT / "scripts" / "wb_command").resolve())
PYTHON_EXE = sys.executable or "/opt/miniconda3/envs/py314/bin/python"
NETWORK_STYLE_JSON = REPO_ROOT / "config" / "hipp_network_style.json"
CROSS_ATLAS_NETWORK_MERGE_JSON = REPO_ROOT / "config" / "cross_atlas_network_merge.json"
DEFAULT_SCENE = REPO_ROOT / "config" / "wb_locked_native_view_lateral_medial.scene"
EVAL_K = list(range(2, 11))
SMOOTH_ORDER = ["2mm", "4mm"]
HEMIS = ["L", "R"]
RUN_SPECS = [
    {"run_id": "1", "label": "REST1_PA"},
    {"run_id": "2", "label": "REST2_AP"},
    {"run_id": "3", "label": "REST3_PA"},
    {"run_id": "4", "label": "REST4_AP"},
]
DEFAULT_INSTABILITY_RESAMPLES = 6
DEFAULT_V_MIN_FRACTION = 0.05
DEFAULT_HIPP_DENSITY = load_surface_density_from_pipeline_config(REPO_ROOT / "config" / "hippo_pipeline.toml")
HCP_MODE = 10000.0
TSNR_THRESHOLD = 25.0
BRANCHES = [
    "network-gradient",
    "network-prob-cluster",
    "network-prob-cluster-nonneg",
    "network-prob-soft",
    "network-prob-soft-nonneg",
    "network-wta",
    "network-spectral",
    "network-spectral-nonneg",
    "intrinsic-spectral",
    "intrinsic-spectral-nonneg",
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


def uses_nonnegative_spectral_features(branch_slug: str) -> bool:
    return branch_slug == "network-spectral-nonneg"


def uses_nonnegative_intrinsic_spectral_features(branch_slug: str) -> bool:
    return branch_slug == "intrinsic-spectral-nonneg"


def is_intrinsic_spectral_branch(branch_slug: str) -> bool:
    return branch_slug in {"intrinsic-spectral", "intrinsic-spectral-nonneg"}


def is_spectral_branch(branch_slug: str) -> bool:
    return branch_slug in {
        "network-spectral",
        "network-spectral-nonneg",
        "intrinsic-spectral",
        "intrinsic-spectral-nonneg",
    }


def run(cmd: list[str]) -> None:
    proc = subprocess.run(cmd, text=True, capture_output=True)
    if proc.returncode != 0:
        raise RuntimeError(
            f"Command failed ({proc.returncode}): {' '.join(cmd)}\nSTDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}"
        )


def find_hippunfold_surface_asset(
    *,
    surf_dir: Path,
    subject: str,
    hemi: str,
    space: str,
    density: str,
    suffix: str,
) -> Path:
    return find_surface_asset_strict(
        surf_dir=surf_dir,
        subject=subject,
        hemi=hemi,
        space=space,
        density=density,
        suffix=suffix,
    )

def find_hippunfold_cifti_asset(*, cifti_dir: Path, subject: str, density: str, suffix: str) -> Path:
    from hipp_density_assets import DensityAssetError
    try:
        return find_cifti_asset_strict(cifti_dir=cifti_dir, subject=subject, density=density, suffix=suffix)
    except DensityAssetError:
        surf_dir = cifti_dir.parent / "surf"
        if surf_dir.exists():
            try:
                return find_cifti_asset_strict(cifti_dir=surf_dir, subject=subject, density=density, suffix=suffix)
            except DensityAssetError:
                pass
        raise


def separate_hippunfold_structural_dlabel(
    *,
    dlabel_path: Path,
    output_dir: Path,
    subject: str,
    density: str,
    resume_mode: str,
) -> tuple[Path, Path]:
    left_out = output_dir / f"sub-{subject}_hemi-L_space-corobl_den-{density}_label-hipp_atlas-multihist7_subfields.label.gii"
    right_out = output_dir / f"sub-{subject}_hemi-R_space-corobl_den-{density}_label-hipp_atlas-multihist7_subfields.label.gii"
    if resume_mode == "force" or not (left_out.exists() and right_out.exists()):
        output_dir.mkdir(parents=True, exist_ok=True)
        run(
            [
                WB_COMMAND,
                "-cifti-separate",
                str(dlabel_path),
                "COLUMN",
                "-label",
                "HIPPOCAMPUS_LEFT",
                str(left_out),
                "-label",
                "HIPPOCAMPUS_RIGHT",
                str(right_out),
            ]
        )
    return left_out, right_out


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def resolve_v_min_count(
    *,
    n_vertices: int,
    v_min_fraction: float | None,
    v_min_count: int | None,
) -> tuple[int, str]:
    if n_vertices <= 0:
        raise ValueError(f"n_vertices must be positive, got {n_vertices}")
    if v_min_count is not None:
        if v_min_count <= 0:
            raise ValueError(f"--v-min-count must be positive, got {v_min_count}")
        if v_min_count > n_vertices:
            raise ValueError(f"--v-min-count={v_min_count} exceeds n_vertices={n_vertices}")
        return int(v_min_count), "count"
    if v_min_fraction is None:
        raise ValueError("Either --v-min-count or --v-min-fraction must be provided")
    if not (0.0 < v_min_fraction <= 1.0):
        raise ValueError(f"--v-min-fraction must be in (0, 1], got {v_min_fraction}")
    return int(math.ceil(n_vertices * float(v_min_fraction))), "fraction"


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


def split_run_bounds(n_timepoints: int, n_runs: int) -> list[tuple[int, int]]:
    if n_runs <= 0:
        raise ValueError(f"n_runs must be positive, got {n_runs}")
    if n_timepoints <= 0:
        raise ValueError(f"n_timepoints must be positive, got {n_timepoints}")
    if n_timepoints % n_runs != 0:
        raise ValueError(
            f"Concat timeseries length {n_timepoints} is not evenly divisible by n_runs={n_runs}"
        )
    run_length = n_timepoints // n_runs
    return [(idx * run_length, (idx + 1) * run_length) for idx in range(n_runs)]


def split_dtseries_concat_to_runwise(concat_path: Path, out_paths: list[Path]) -> list[int]:
    img = nib.load(str(concat_path))
    series_axis = img.header.get_axis(0)
    brain_axis = img.header.get_axis(1)
    shape = img.shape
    if len(shape) != 2:
        raise ValueError(f"Expected 2D dtseries data, got shape {shape} for {concat_path}")
    bounds = split_run_bounds(int(shape[0]), len(out_paths))
    run_lengths: list[int] = []
    for (start, stop), out_path in zip(bounds, out_paths, strict=True):
        out_path.parent.mkdir(parents=True, exist_ok=True)
        block = np.asarray(img.dataobj[start:stop, :], dtype=np.float32)
        run_axis = series_axis[start:stop]
        header = nib.cifti2.Cifti2Header.from_axes((run_axis, brain_axis))
        out_img = nib.Cifti2Image(block, header=header, nifti_header=img.nifti_header.copy())
        out_img.update_headers()
        nib.save(out_img, str(out_path))
        run_lengths.append(int(stop - start))
    return run_lengths


def split_bold_concat_to_runwise(concat_path: Path, out_paths: list[Path]) -> list[int]:
    img = nib.load(str(concat_path))
    shape = img.shape
    if len(shape) != 4:
        raise ValueError(f"Expected 4D BOLD image, got shape {shape} for {concat_path}")
    bounds = split_run_bounds(int(shape[3]), len(out_paths))
    run_lengths: list[int] = []
    qform, qform_code = img.get_qform(coded=True)
    sform, sform_code = img.get_sform(coded=True)
    for (start, stop), out_path in zip(bounds, out_paths, strict=True):
        out_path.parent.mkdir(parents=True, exist_ok=True)
        block = np.asarray(img.dataobj[..., start:stop], dtype=np.float32)
        out_img = nib.Nifti1Image(block, affine=img.affine, header=img.header.copy())
        if qform is not None:
            out_img.set_qform(qform, int(qform_code))
        if sform is not None:
            out_img.set_sform(sform, int(sform_code))
        nib.save(out_img, str(out_path))
        run_lengths.append(int(stop - start))
    return run_lengths


def split_surface_timeseries_to_runs(
    concat_path: Path,
    out_paths: list[Path],
    run_lengths: list[int],
) -> None:
    arr = np.load(concat_path).astype(np.float32)
    if arr.ndim != 2:
        raise ValueError(f"Expected 2D surface timeseries, got shape {arr.shape} for {concat_path}")
    expected = int(sum(int(x) for x in run_lengths))
    if int(arr.shape[1]) != expected:
        raise ValueError(
            "Concat surface timeseries length mismatch: "
            f"{concat_path} has {arr.shape[1]} frames but run lengths sum to {expected}"
        )
    offset = 0
    for out_path, run_len in zip(out_paths, run_lengths, strict=True):
        out_path.parent.mkdir(parents=True, exist_ok=True)
        next_offset = offset + int(run_len)
        np.save(out_path, arr[:, offset:next_offset].astype(np.float32, copy=False))
        offset = next_offset


def resolve_runwise_dtseries(
    *,
    subject: str,
    concat_dtseries: Path,
    runwise_dtseries: list[Path],
    stage_root: Path,
    resume_mode: str,
) -> tuple[list[Path], str, list[int]]:
    if all(path.exists() for path in runwise_dtseries):
        lengths = [int(nib.load(str(path)).shape[0]) for path in runwise_dtseries]
        return runwise_dtseries, "provided", lengths
    if not concat_dtseries.exists():
        raise FileNotFoundError(f"Missing concat dtseries required to split runs: {concat_dtseries}")
    stage_dir = stage_root / "dtseries"
    staged_paths = [stage_dir / path.name for path in runwise_dtseries]
    params = {"subject": subject, "source": str(concat_dtseries.resolve()), "n_runs": len(staged_paths)}
    if not stage_is_up_to_date(
        stage_dir=stage_dir,
        resume_mode=resume_mode,
        stage_name="runwise_dtseries_from_concat",
        params=params,
        inputs=[concat_dtseries],
        outputs=staged_paths,
    ):
        lengths = split_dtseries_concat_to_runwise(concat_dtseries, staged_paths)
        write_stage_manifest(
            stage_dir=stage_dir,
            stage_name="runwise_dtseries_from_concat",
            params={**params, "run_lengths": lengths},
            inputs=[concat_dtseries],
            outputs=staged_paths,
        )
    payload = json.loads(stage_manifest_path(stage_dir).read_text(encoding="utf-8"))
    lengths = [int(x) for x in payload.get("params", {}).get("run_lengths", [])]
    if len(lengths) != len(staged_paths):
        lengths = [int(nib.load(str(path)).shape[0]) for path in staged_paths]
    return staged_paths, "split_from_concat", lengths


def resolve_runwise_bold(
    *,
    subject: str,
    concat_bold: Path,
    runwise_bold: list[Path],
    stage_root: Path,
    resume_mode: str,
) -> tuple[list[Path], str, list[int]]:
    if all(path.exists() for path in runwise_bold):
        lengths = [int(nib.load(str(path)).shape[3]) for path in runwise_bold]
        return runwise_bold, "provided", lengths
    if not concat_bold.exists():
        raise FileNotFoundError(
            "Missing run-wise volume BOLD inputs and concat BOLD is unavailable for splitting: "
            f"{concat_bold}"
        )
    stage_dir = stage_root / "bold"
    staged_paths = [stage_dir / path.name for path in runwise_bold]
    params = {"subject": subject, "source": str(concat_bold.resolve()), "n_runs": len(staged_paths)}
    if not stage_is_up_to_date(
        stage_dir=stage_dir,
        resume_mode=resume_mode,
        stage_name="runwise_bold_from_concat",
        params=params,
        inputs=[concat_bold],
        outputs=staged_paths,
    ):
        lengths = split_bold_concat_to_runwise(concat_bold, staged_paths)
        write_stage_manifest(
            stage_dir=stage_dir,
            stage_name="runwise_bold_from_concat",
            params={**params, "run_lengths": lengths},
            inputs=[concat_bold],
            outputs=staged_paths,
        )
    payload = json.loads(stage_manifest_path(stage_dir).read_text(encoding="utf-8"))
    lengths = [int(x) for x in payload.get("params", {}).get("run_lengths", [])]
    if len(lengths) != len(staged_paths):
        lengths = [int(nib.load(str(path)).shape[3]) for path in staged_paths]
    return staged_paths, "split_from_concat", lengths


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


def write_fc_store_pointer(
    *,
    pointer_dir: Path,
    shared_fc_store_dir: Path,
    subject: str,
    atlas_slug: str,
) -> None:
    pointer_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "kind": "shared_fc_store_pointer",
        "timestamp_utc": utc_now_iso(),
        "subject": subject,
        "atlas_slug": atlas_slug,
        "shared_fc_store_dir": str(shared_fc_store_dir.resolve()),
    }
    (pointer_dir / "shared_fc_store.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")


def apply_retain_level(out_root: Path, retain_level: str) -> None:
    if retain_level in {"feature", "all"}:
        return
    if retain_level == "render":
        archive_names = {"features"}
    elif retain_level == "label":
        archive_names = {"features", "clustering", "soft_outputs"}
    else:
        raise ValueError(f"Unsupported retain_level: {retain_level}")

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    outputs_root = (REPO_ROOT / "outputs_migration" / "hipp_functional_parcellation_network").resolve()
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


def load_surface_timeseries_for_tsnr(*, metric_path: Path, expected_n_vertices: int) -> tuple[np.ndarray, str]:
    return load_metric_array(metric_path, expected_n_vertices=expected_n_vertices), str(metric_path.resolve())


def load_surface(path: Path) -> tuple[np.ndarray, np.ndarray]:
    img = nib.load(str(path))
    coords = np.asarray(img.agg_data("pointset"), dtype=np.float32)
    faces = np.asarray(img.agg_data("triangle"), dtype=np.int32)
    return coords, faces


def save_shape_gii(values: np.ndarray, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    arr = GiftiDataArray(
        data=np.asarray(values, dtype=np.float32),
        intent=nib.nifti1.intent_codes["NIFTI_INTENT_NONE"],
        datatype="NIFTI_TYPE_FLOAT32",
    )
    nib.save(GiftiImage(darrays=[arr]), str(out_path))


def compute_tsnr(metric: np.ndarray) -> np.ndarray:
    if metric.ndim != 2:
        raise ValueError(f"Expected 2D timeseries array for tSNR, got {metric.shape}")
    sd = np.nanstd(metric.astype(np.float32, copy=False), axis=1, ddof=1)
    tsnr = np.where(sd > 0, HCP_MODE / sd, np.nan)
    return tsnr.astype(np.float32)


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


def build_boundary_vertex_mask(faces: np.ndarray, n_vertices: int) -> np.ndarray:
    edge_counts: dict[tuple[int, int], int] = {}
    for tri in faces:
        a, b, c = (int(tri[0]), int(tri[1]), int(tri[2]))
        for u, v in ((a, b), (b, c), (c, a)):
            key = (u, v) if u < v else (v, u)
            edge_counts[key] = edge_counts.get(key, 0) + 1
    boundary = np.zeros(n_vertices, dtype=bool)
    for (u, v), count in edge_counts.items():
        if count == 1:
            boundary[u] = True
            boundary[v] = True
    return boundary


def induced_subgraph(connectivity: sparse.csr_matrix, vertex_indices: np.ndarray) -> sparse.csr_matrix:
    if vertex_indices.ndim != 1:
        raise ValueError(f"Expected 1D vertex indices, got {vertex_indices.shape}")
    return connectivity[vertex_indices][:, vertex_indices]


def component_graph_diameter(subgraph: sparse.csr_matrix) -> int:
    n_vertices = int(subgraph.shape[0])
    if n_vertices <= 1:
        return 0
    distances = csgraph.shortest_path(subgraph, directed=False, unweighted=True)
    finite = distances[np.isfinite(distances)]
    if finite.size == 0:
        return 0
    return int(np.max(finite))


def sanitize_timeseries_with_mask(metric: np.ndarray, valid_mask: np.ndarray) -> np.ndarray:
    metric = sanitize_timeseries(metric)
    masked = metric.astype(np.float32, copy=True)
    masked[~valid_mask, :] = np.nan
    return masked


def save_masked_metric(metric: np.ndarray, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    darrays = [
        GiftiDataArray(data=np.asarray(metric[:, idx], dtype=np.float32), datatype="NIFTI_TYPE_FLOAT32")
        for idx in range(metric.shape[1])
    ]
    nib.save(GiftiImage(darrays=darrays), str(out_path))


def smooth_metric_with_roi(
    *,
    surface_path: Path,
    metric_path: Path,
    smooth_mm: str,
    out_metric: Path,
    roi_path: Path,
) -> None:
    out_metric.parent.mkdir(parents=True, exist_ok=True)
    run(
        [
            WB_COMMAND,
            "-metric-smoothing",
            str(surface_path),
            str(metric_path),
            smooth_mm,
            str(out_metric),
            "-fwhm",
            "-roi",
            str(roi_path),
        ]
    )


def extract_structure_data(
    dt_axis: nib.cifti2.cifti2_axes.BrainModelAxis,
    dt_data_t: np.ndarray,
    structure_name: str,
) -> tuple[np.ndarray, np.ndarray]:
    for name, slc, subaxis in dt_axis.iter_structures():
        if name == structure_name or name.endswith(structure_name):
            if getattr(subaxis, "vertex", None) is None:
                raise RuntimeError(f"Structure {structure_name} does not expose vertices")
            return np.asarray(subaxis.vertex, dtype=np.int32), dt_data_t[slc, :]
    raise RuntimeError(f"Could not find structure {structure_name} in dtseries")


def compute_cortex_tsnr_gate(
    *,
    dtseries_path: Path,
    output_dir: Path,
) -> dict[str, object]:
    img = nib.load(str(dtseries_path))
    dt_data = np.asarray(img.dataobj, dtype=np.float32)
    if dt_data.ndim != 2:
        raise ValueError(f"Expected 2D dtseries, got {dt_data.shape} for {dtseries_path}")
    dt_axis = img.header.get_axis(1)
    dt_data_t = dt_data.T
    left_vertices, left_dt = extract_structure_data(dt_axis, dt_data_t, "CORTEX_LEFT")
    right_vertices, right_dt = extract_structure_data(dt_axis, dt_data_t, "CORTEX_RIGHT")
    left_tsnr = compute_tsnr(left_dt)
    right_tsnr = compute_tsnr(right_dt)
    left_valid = np.isfinite(left_tsnr) & (left_tsnr >= TSNR_THRESHOLD)
    right_valid = np.isfinite(right_tsnr) & (right_tsnr >= TSNR_THRESHOLD)
    output_dir.mkdir(parents=True, exist_ok=True)
    left_tsnr_path = output_dir / "cortex_left_tsnr.npy"
    right_tsnr_path = output_dir / "cortex_right_tsnr.npy"
    left_mask_path = output_dir / "cortex_left_valid_mask.npy"
    right_mask_path = output_dir / "cortex_right_valid_mask.npy"
    np.save(left_tsnr_path, left_tsnr.astype(np.float32))
    np.save(right_tsnr_path, right_tsnr.astype(np.float32))
    np.save(left_mask_path, left_valid.astype(bool))
    np.save(right_mask_path, right_valid.astype(bool))
    summary = {
        "dtseries": str(dtseries_path.resolve()),
        "tsnr_definition": "10000/std",
        "threshold": float(TSNR_THRESHOLD),
        "left": {
            "n_grayordinates_total": int(left_dt.shape[0]),
            "n_grayordinates_used": int(left_valid.sum()),
            "n_grayordinates_masked": int((~left_valid).sum()),
            "tsnr_path": str(left_tsnr_path.resolve()),
            "mask_path": str(left_mask_path.resolve()),
            "vertex_index_min": int(left_vertices.min(initial=0)),
            "vertex_index_max": int(left_vertices.max(initial=0)),
        },
        "right": {
            "n_grayordinates_total": int(right_dt.shape[0]),
            "n_grayordinates_used": int(right_valid.sum()),
            "n_grayordinates_masked": int((~right_valid).sum()),
            "tsnr_path": str(right_tsnr_path.resolve()),
            "mask_path": str(right_mask_path.resolve()),
            "vertex_index_min": int(right_vertices.min(initial=0)),
            "vertex_index_max": int(right_vertices.max(initial=0)),
        },
        "combined": {
            "n_grayordinates_total": int(left_dt.shape[0] + right_dt.shape[0]),
            "n_grayordinates_used": int(left_valid.sum() + right_valid.sum()),
            "n_grayordinates_masked": int((~left_valid).sum() + (~right_valid).sum()),
        },
    }
    summary_path = output_dir / "cortex_tsnr_gate_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return {
        "summary_path": summary_path,
        "left_tsnr_path": left_tsnr_path,
        "right_tsnr_path": right_tsnr_path,
        "left_mask_path": left_mask_path,
        "right_mask_path": right_mask_path,
        "summary": summary,
    }


def compute_hipp_tsnr_gate(
    *,
    subject: str,
    hemi: str,
    raw_metric: np.ndarray,
    raw_metric_source: str,
    connectivity: sparse.csr_matrix,
    faces: np.ndarray,
    output_dir: Path,
) -> dict[str, object]:
    tsnr = compute_tsnr(raw_metric)
    invalid_mask = ~np.isfinite(tsnr) | (tsnr < TSNR_THRESHOLD)
    valid_mask = ~invalid_mask
    output_dir.mkdir(parents=True, exist_ok=True)
    tsnr_path = output_dir / f"sub-{subject}_hemi-{hemi}_tsnr.npy"
    valid_mask_path = output_dir / f"sub-{subject}_hemi-{hemi}_valid_mask.npy"
    invalid_mask_path = output_dir / f"sub-{subject}_hemi-{hemi}_invalid_initial_mask.npy"
    np.save(tsnr_path, tsnr.astype(np.float32))
    np.save(valid_mask_path, valid_mask.astype(bool))
    np.save(invalid_mask_path, invalid_mask.astype(bool))
    summary = {
        "subject": subject,
        "hemi": hemi,
        "threshold": float(TSNR_THRESHOLD),
        "tsnr_definition": "10000/std",
        "raw_tsnr_input_source": raw_metric_source,
        "n_vertices_total": int(raw_metric.shape[0]),
        "n_vertices_valid_high_tsnr": int(valid_mask.sum()),
        "n_vertices_invalid_initial": int(invalid_mask.sum()),
        "paths": {
            "tsnr": str(tsnr_path.resolve()),
            "valid_mask": str(valid_mask_path.resolve()),
            "invalid_initial_mask": str(invalid_mask_path.resolve()),
        },
    }
    summary_path = output_dir / f"sub-{subject}_hemi-{hemi}_tsnr_gate_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return {
        "summary_path": summary_path,
        "tsnr_path": tsnr_path,
        "valid_mask_path": valid_mask_path,
        "invalid_mask_path": invalid_mask_path,
        "tsnr": tsnr,
        "valid_mask": valid_mask,
        "invalid_initial_mask": invalid_mask,
        "summary": summary,
    }


def compact_active_vertices(
    data_full: np.ndarray,
    active_mask: np.ndarray,
) -> np.ndarray:
    return np.asarray(data_full[active_mask, :], dtype=np.float32)


def expand_cluster_labels(labels_active: np.ndarray, active_mask: np.ndarray, total_vertices: int) -> np.ndarray:
    labels_full = np.zeros(total_vertices, dtype=np.int32)
    labels_full[np.flatnonzero(active_mask)] = labels_active.astype(np.int32)
    return labels_full


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
    n_samples = int(features.shape[0])
    n_labels = int(np.unique(labels).size)
    if n_samples <= 2 or n_labels < 2 or n_labels >= n_samples:
        return float("nan")
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


def allocate_component_cluster_counts(component_sizes: list[int], k_total: int) -> list[int]:
    n_components = len(component_sizes)
    if n_components <= 0:
        raise ValueError("component_sizes must be non-empty")
    if k_total < n_components:
        raise ValueError(f"k_total={k_total} is smaller than number of connected components={n_components}")
    total_size = float(sum(component_sizes))
    raw = [size / total_size * k_total for size in component_sizes]
    assigned = [max(1, int(math.floor(value))) for value in raw]
    while sum(assigned) > k_total:
        idx = int(np.argmax([count for count in assigned]))
        if assigned[idx] <= 1:
            break
        assigned[idx] -= 1
    remainders = [value - math.floor(value) for value in raw]
    while sum(assigned) < k_total:
        candidates = [idx for idx, size in enumerate(component_sizes)]
        idx = max(candidates, key=lambda i: (remainders[i], component_sizes[i], -i))
        assigned[idx] += 1
        remainders[idx] = 0.0
    return assigned


def cluster_embedding(features: np.ndarray, connectivity: sparse.csr_matrix, k: int) -> np.ndarray:
    n_components, component_labels = csgraph.connected_components(connectivity, directed=False, return_labels=True)
    if int(n_components) <= 1:
        model = AgglomerativeClustering(n_clusters=k, linkage="ward", connectivity=connectivity)
        raw_labels = model.fit_predict(features)
        return reorder_cluster_labels(raw_labels)
    if k < int(n_components):
        model = AgglomerativeClustering(n_clusters=k, linkage="ward", connectivity=connectivity)
        raw_labels = model.fit_predict(features)
        return reorder_cluster_labels(raw_labels)

    component_indices = [np.flatnonzero(component_labels == comp_id) for comp_id in range(int(n_components))]
    cluster_counts = allocate_component_cluster_counts([int(idx.size) for idx in component_indices], k)
    labels = np.zeros(features.shape[0], dtype=np.int32)
    next_label = 1
    for indices, k_component in zip(component_indices, cluster_counts, strict=True):
        if int(k_component) <= 1 or indices.size <= 1:
            labels[indices] = next_label
            next_label += 1
            continue
        subgraph = induced_subgraph(connectivity, indices)
        model = AgglomerativeClustering(n_clusters=int(k_component), linkage="ward", connectivity=subgraph)
        raw_sub = model.fit_predict(features[indices, :])
        ordered_sub = reorder_cluster_labels(raw_sub)
        for local_label in sorted(np.unique(ordered_sub)):
            labels[indices[ordered_sub == local_label]] = next_label
            next_label += 1
    return reorder_cluster_labels(labels)


def write_tsv_rows(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter="\t", extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def match_labels_hungarian(labels_a: np.ndarray, labels_b: np.ndarray) -> np.ndarray:
    labels_a = labels_a.astype(np.int32, copy=False)
    labels_b = labels_b.astype(np.int32, copy=False)
    a_keys = sorted(int(x) for x in np.unique(labels_a))
    b_keys = sorted(int(x) for x in np.unique(labels_b))
    contingency = np.zeros((len(a_keys), len(b_keys)), dtype=np.int32)
    for i, a_key in enumerate(a_keys):
        mask_a = labels_a == a_key
        for j, b_key in enumerate(b_keys):
            contingency[i, j] = int(np.count_nonzero(mask_a & (labels_b == b_key)))
    row_ind, col_ind = linear_sum_assignment(-contingency)
    mapping = {b_keys[col]: a_keys[row] for row, col in zip(row_ind.tolist(), col_ind.tolist(), strict=True)}
    next_key = max(a_keys, default=0) + 1
    for b_key in b_keys:
        if b_key not in mapping:
            mapping[b_key] = next_key
            next_key += 1
    return np.asarray([mapping[int(key)] for key in labels_b], dtype=np.int32)


def compute_homogeneity(features: np.ndarray, labels: np.ndarray) -> float:
    tss = float(np.sum((features - np.mean(features, axis=0, keepdims=True)) ** 2))
    if tss <= 1e-12:
        return 0.0
    wcss = compute_wcss(features, labels)
    return float(max(0.0, 1.0 - (wcss / tss)))


def build_run_pair_resamples(n_runs: int, max_resamples: int) -> list[tuple[int, int]]:
    if n_runs < 2:
        return []
    pairs = [(i, j) for i in range(n_runs) for j in range(i + 1, n_runs)]
    if max_resamples > 0:
        return pairs[: min(len(pairs), max_resamples)]
    return pairs


def mark_instability_decisions(
    rows: list[dict[str, object]],
) -> tuple[list[dict[str, object]], dict[str, object]]:
    if not rows:
        raise ValueError("No K rows available for instability selection")
    ordered = sorted(rows, key=lambda row: int(row["k"]))
    instabilities = [float(row["instability_mean"]) for row in ordered]
    best_idx = int(np.argmin(instabilities))
    best_row = ordered[best_idx]
    best_instability = float(best_row["instability_mean"])
    best_se = float(best_row["instability_se"])

    all_instability_equal = len(set(instabilities)) == 1

    local_minima: list[int] = []
    for idx, row in enumerate(ordered):
        current = float(row["instability_mean"])
        left = float(ordered[idx - 1]["instability_mean"]) if idx > 0 else math.inf
        right = float(ordered[idx + 1]["instability_mean"]) if idx < len(ordered) - 1 else math.inf
        is_local = all_instability_equal or (
            current <= left and current <= right and (current < left or current < right or len(ordered) == 1)
        )
        row["local_min"] = int(is_local)
        if is_local:
            local_minima.append(int(row["k"]))

    one_se_cutoff = best_instability + best_se
    for row in ordered:
        row["within_1se_best"] = int(float(row["instability_mean"]) <= one_se_cutoff + 1e-12)

    one_se_candidates = [
        row
        for row in ordered
        if int(row["local_min"]) == 1
        and int(row["within_1se_best"]) == 1
    ]
    if not one_se_candidates:
        raise RuntimeError("No K survived local-minimum and 1-SE screening")
    eligible = [
        row
        for row in one_se_candidates
        if int(row["min_parcel_ok"]) == 1
    ]
    if not eligible:
        raise RuntimeError("No K survived local-minimum, 1-SE, and min-parcel constraints")
    selected = eligible[0]
    k_star = int(selected["k"])
    sensitivity = []
    for candidate in (k_star - 1, k_star + 1):
        if any(int(row["k"]) == candidate for row in ordered):
            sensitivity.append(candidate)
    decision = {
        "best_by_instability": int(best_row["k"]),
        "candidate_local_minima": local_minima,
        "one_se_selected": int(min(int(row["k"]) for row in one_se_candidates)),
        "post_constraint_selected": k_star,
        "main_analysis_k": k_star,
        "sensitivity_k": sensitivity,
        "degenerate_instability": bool(all_instability_equal),
    }
    return ordered, decision


def select_final_k_mainline(rows: list[dict[str, object]]) -> tuple[int, dict[str, object]]:
    if not rows:
        raise ValueError("No K rows available for mainline selection")
    ordered = sorted(rows, key=lambda row: int(row["k"]))
    best_row = max(ordered, key=lambda row: float(row["null_corrected_score"]))
    target_ari = float(best_row["null_corrected_score"]) - 0.02
    eligible = [
        row
        for row in ordered
        if float(row["null_corrected_score"]) >= target_ari
        and float(row["min_cluster_size_fraction"]) >= 0.05
    ]
    selected = eligible[0] if eligible else best_row
    k_final = int(selected["k"])
    for row in ordered:
        row["local_min"] = 0
        row["within_1se_best"] = int(float(row["null_corrected_score"]) >= target_ari)
    decision = {
        "best_by_instability": int(best_row["k"]),
        "candidate_local_minima": [],
        "one_se_selected": int(k_final),
        "post_constraint_selected": int(k_final),
        "main_analysis_k": int(k_final),
        "sensitivity_k": [],
    }
    return k_final, decision


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
    fisher = fisher_z_transform_fc(grouped_fc)
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
    null_label = GiftiLabel(key=0, red=0.62, green=0.62, blue=0.62, alpha=1.0)
    null_label.label = "Null"
    table.labels.append(null_label)
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
    density: str,
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

    left_path = output_dir / f"sub-{subject}_hemi-L_space-corobl_den-{density}_label-{stem}.label.gii"
    right_path = output_dir / f"sub-{subject}_hemi-R_space-corobl_den-{density}_label-{stem}.label.gii"
    nib.save(make_label_gifti(left_labels, left_key_to_name), str(left_path))
    nib.save(make_label_gifti(right_labels_shifted, right_key_to_name_shifted), str(right_path))

    dlabel_path = output_dir / f"sub-{subject}_space-corobl_den-{density}_label-{stem}.dlabel.nii"
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
    cmd.extend(["--left-surface-template", str(left_surface)])
    cmd.extend(["--right-surface-template", str(right_surface)])
    cmd.extend(["--spec-template", str(spec_path)])
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
    left_surface: Path,
    right_surface: Path,
    spec_path: Path,
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
        left_surface=left_surface,
        right_surface=right_surface,
        spec_path=spec_path,
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
    run_features: list[np.ndarray],
    run_labels: list[str],
    profile_source: np.ndarray,
    profile_networks: list[str],
    connectivity: sparse.csr_matrix,
    outdir: Path,
    hemi: str,
    profile_mode: str,
    split_strategy: str,
    instability_resamples: int,
    v_min_fraction: float | None,
    v_min_count: int | None,
    k_selection_mode: str,
    clustering_method: str = "AgglomerativeClustering(linkage=ward)",
    distance_metric: str = "euclidean",
    cluster_fn=None,
) -> dict[str, object]:
    k_metrics: list[dict[str, object]] = []
    k_to_annotations: dict[int, list[dict[str, object]]] = {}
    k_to_key_names: dict[int, dict[int, str]] = {}
    k_to_probability_rows: dict[int, np.ndarray] = {}
    tss = float(np.sum((features_full - np.mean(features_full, axis=0, keepdims=True)) ** 2))
    resample_pairs = build_run_pair_resamples(len(run_features), instability_resamples)
    if not resample_pairs:
        raise RuntimeError(f"Need at least two run-wise feature blocks for instability, got {len(run_features)}")
    resolved_v_min_count, v_min_mode = resolve_v_min_count(
        n_vertices=int(features_full.shape[0]),
        v_min_fraction=v_min_fraction,
        v_min_count=v_min_count,
    )
    valid_eval_k = [int(k) for k in EVAL_K if int(k) <= int(features_full.shape[0])]
    if not valid_eval_k:
        raise RuntimeError(
            f"Not enough active hippocampal vertices to evaluate any K in {EVAL_K}: n_active={features_full.shape[0]}"
        )

    previous_wcss: float | None = None
    for k in valid_eval_k:
        _cluster = cluster_fn if cluster_fn is not None else cluster_embedding
        labels_full = _cluster(features_full, connectivity, k)
        labels_by_run = [_cluster(features_run, connectivity, k) for features_run in run_features]
        ari_values: list[float] = []
        resample_rows: list[dict[str, object]] = []
        for pair_idx, (run_idx_a, run_idx_b) in enumerate(resample_pairs, start=1):
            labels_a = labels_by_run[run_idx_a]
            labels_b = match_labels_hungarian(labels_a, labels_by_run[run_idx_b])
            ari = float(adjusted_rand_score(labels_a, labels_b))
            ari_values.append(ari)
            resample_rows.append(
                {
                    "resample_id": pair_idx,
                    "run_a": run_labels[run_idx_a],
                    "run_b": run_labels[run_idx_b],
                    "ari": ari,
                    "instability": float(1.0 - ari),
                }
            )
        sizes = [int(np.count_nonzero(labels_full == label)) for label in sorted(np.unique(labels_full))]
        min_size_vertices = min(sizes)
        min_frac = min_size_vertices / labels_full.size
        total_cc, per_cluster_cc = connected_component_count(labels_full, connectivity)
        sil = compute_silhouette(features_full, labels_full)
        try:
            ch = float(calinski_harabasz_score(features_full, labels_full))
        except ValueError:
            ch = float("nan")
        try:
            db = float(davies_bouldin_score(features_full, labels_full))
        except ValueError:
            db = float("nan")
        wcss = compute_wcss(features_full, labels_full)
        bss_ratio = float(1.0 - (wcss / max(tss, 1e-12)))
        entropy = compute_balance_entropy(labels_full)
        delta_wcss = None if previous_wcss is None else float(previous_wcss - wcss)
        previous_wcss = wcss
        instability = 1.0 - np.asarray(ari_values, dtype=np.float32)
        instability_mean = float(np.mean(instability))
        instability_se = float(np.std(instability, ddof=1) / np.sqrt(len(instability))) if len(instability) > 1 else 0.0
        ari_mean = float(np.mean(ari_values))
        homogeneity = compute_homogeneity(features_full, labels_full)
        min_parcel_ok = int(min_size_vertices >= resolved_v_min_count)
        connectivity_ok = int(total_cc == k)

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
        save_json(k_dir / "instability_resamples.json", {"hemi": hemi, "k": int(k), "resamples": resample_rows})
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
            "n_valid_resamples": int(len(resample_rows)),
            "instability_mean": instability_mean,
            "instability_se": instability_se,
            "null_corrected_score": ari_mean,
            "silhouette": float(sil),
            "calinski_harabasz": ch,
            "davies_bouldin": db,
            "wcss": float(wcss),
            "delta_wcss": delta_wcss,
            "homogeneity": homogeneity,
            "min_cluster_size_vertices": int(min_size_vertices),
            "min_cluster_size_fraction": float(min_frac),
            "v_min_vertices": int(resolved_v_min_count),
            "min_parcel_ok": min_parcel_ok,
            "bss_tss_ratio": bss_ratio,
            "cluster_balance_entropy": entropy,
            "connected_component_count": int(total_cc),
            "connectivity_ok": connectivity_ok,
            "per_cluster_connected_components": {str(key): int(value) for key, value in per_cluster_cc.items()},
        }
        k_metrics.append(metric_row)

    if k_selection_mode == "mainline":
        k_final, decision = select_final_k_mainline(k_metrics)
        primary_reason = (
            "Mainline selection: choose smallest K whose null-corrected score is within 0.02 of best and min cluster fraction >= 0.05."
        )
        secondary_reason = "If no K satisfies the threshold, use the K with maximum null-corrected score."
        deviations = "using current mainline rule"
    elif k_selection_mode == "experimental":
        k_metrics, decision = mark_instability_decisions(k_metrics)
        k_final = int(decision["main_analysis_k"])
        primary_reason = "Selected the smallest local instability minimum within 1-SE that passed the parcel-size constraint."
        secondary_reason = "Lower-complexity solution retained unless a larger K showed clearly better stability within protocol rules."
        deviations = "none"
    else:
        raise ValueError(f"Unsupported --k-selection-mode: {k_selection_mode}")
    labels_final = np.load(outdir / f"k_{k_final}" / "cluster_labels.npy").astype(np.int32)
    run_metadata = {
        "project_id": "hipp_functional_parcellation_network",
        "analysis_date": datetime.now().date().isoformat(),
        "operator": "codex",
        "code_commit": None,
        "feature_definition": profile_mode,
        "hemisphere": hemi,
        "subject_set": run_labels,
        "split_strategy": split_strategy,
        "clustering_method": clustering_method,
        "distance_metric": distance_metric,
        "spatial_constraints": str(outdir),
        "random_seed_policy": "deterministic run-pair ordering",
        "B_resamples": int(len(resample_pairs)),
        "K_min": int(min(valid_eval_k)),
        "K_max": int(max(valid_eval_k)),
        "k_selection_mode": k_selection_mode,
        "V_min": int(resolved_v_min_count),
        "V_min_mode": v_min_mode,
    }
    save_json(outdir / "run_metadata.json", run_metadata)
    save_json(
        outdir / "final_selection_log.json",
        {
            "best_by_instability": decision["best_by_instability"],
            "candidate_local_minima": decision["candidate_local_minima"],
            "1SE_selected": decision["one_se_selected"],
            "post_constraint_selected": decision["post_constraint_selected"],
            "main_analysis_K": decision["main_analysis_k"],
            "sensitivity_K": decision["sensitivity_k"],
            "primary_reason": primary_reason,
            "secondary_reason": secondary_reason,
            "deviations_from_protocol": deviations,
        },
    )
    write_tsv_rows(
        outdir / "per_k_summary.tsv",
        k_metrics,
        [
            "k",
            "n_valid_resamples",
            "instability_mean",
            "instability_se",
            "local_min",
            "within_1se_best",
            "homogeneity",
            "min_parcel_ok",
            "connectivity_ok",
            "null_corrected_score",
            "silhouette",
            "min_cluster_size_vertices",
            "min_cluster_size_fraction",
            "v_min_vertices",
            "connected_component_count",
        ],
    )
    save_json(
        outdir / "selection_summary.json",
        {
            "hemi": hemi,
            "k_metrics": k_metrics,
            "k_final": int(k_final),
            "run_metadata": run_metadata,
            "selection_log": {
                "best_by_instability": decision["best_by_instability"],
                "candidate_local_minima": decision["candidate_local_minima"],
                "1SE_selected": decision["one_se_selected"],
                "post_constraint_selected": decision["post_constraint_selected"],
                "main_analysis_K": decision["main_analysis_k"],
                "sensitivity_K": decision["sensitivity_k"],
            },
            "clusters": k_to_annotations[k_final],
            "probability_rows": k_to_probability_rows[k_final].tolist(),
            "networks": profile_networks,
        },
    )
    return {
        "k_metrics": k_metrics,
        "k_final": int(k_final),
        "selection_log": {
            "best_by_instability": decision["best_by_instability"],
            "candidate_local_minima": decision["candidate_local_minima"],
            "1SE_selected": decision["one_se_selected"],
            "post_constraint_selected": decision["post_constraint_selected"],
            "main_analysis_K": decision["main_analysis_k"],
            "sensitivity_K": decision["sensitivity_k"],
        },
        "run_metadata": run_metadata,
        "labels_final": labels_final,
        "cluster_annotations": k_to_annotations[k_final],
        "probability_rows": k_to_probability_rows[k_final],
        "profile_networks": profile_networks,
        "key_to_name": k_to_key_names[k_final],
    }


def compute_gradient_state(grouped_fc: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    fisher_fc = fisher_z_transform_fc(grouped_fc)
    gradients, eigvals = diffusion_map_embedding(build_sparse_affinity(fisher_fc, 0.1), n_components=5)
    features = zscore_columns(gradients[:, :3])
    return features, gradients.astype(np.float32), eigvals.astype(np.float32)


def run_gradient_branch(
    *,
    grouped_fc: np.ndarray,
    run_grouped_fcs: list[np.ndarray],
    run_labels: list[str],
    networks: list[str],
    connectivity: sparse.csr_matrix,
    feature_dir: Path,
    clustering_dir: Path,
    hemi: str,
    split_strategy: str,
    instability_resamples: int,
    v_min_fraction: float | None,
    v_min_count: int | None,
    k_selection_mode: str,
) -> dict[str, object]:
    features_full, gradients, eigvals = compute_gradient_state(grouped_fc)
    run_features = [compute_gradient_state(fc_run)[0] for fc_run in run_grouped_fcs]

    feature_dir.mkdir(parents=True, exist_ok=True)
    np.save(feature_dir / "hipp_network_fc_gradients.npy", gradients.astype(np.float32))
    np.save(feature_dir / "hipp_network_fc_gradient_eigenvalues.npy", eigvals.astype(np.float32))
    save_json(
        feature_dir / "feature_summary.json",
        {
            "hemi": hemi,
            "feature_kind": "network-gradient",
            "feature_shape": [int(features_full.shape[0]), int(features_full.shape[1])],
            "source_shape": [int(grouped_fc.shape[0]), int(grouped_fc.shape[1])],
            "source_networks": networks,
            "fisher_z_transform": True,
            "eigenvalues": [float(x) for x in eigvals.tolist()],
            "run_labels": run_labels,
            "split_strategy": split_strategy,
        },
    )

    cluster = evaluate_k_range(
        features_full=features_full,
        run_features=run_features,
        run_labels=run_labels,
        profile_source=grouped_fc,
        profile_networks=networks,
        connectivity=connectivity,
        outdir=clustering_dir,
        hemi=hemi,
        profile_mode="fc",
        split_strategy=split_strategy,
        instability_resamples=instability_resamples,
        v_min_fraction=v_min_fraction,
        v_min_count=v_min_count,
        k_selection_mode=k_selection_mode,
        clustering_method="AgglomerativeClustering(linkage=ward)",
        distance_metric="euclidean",
    )
    cluster["feature_summary"] = {
        "feature_kind": "network-gradient",
        "feature_shape": [int(features_full.shape[0]), int(features_full.shape[1])],
        "source_shape": [int(grouped_fc.shape[0]), int(grouped_fc.shape[1])],
        "source_networks": networks,
        "fisher_z_transform": True,
        "eigenvalues": [float(x) for x in eigvals.tolist()],
        "run_labels": run_labels,
        "split_strategy": split_strategy,
    }
    return cluster


def run_probability_branch(
    *,
    grouped_fc: np.ndarray,
    run_grouped_fcs: list[np.ndarray],
    run_labels: list[str],
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
    split_strategy: str,
    instability_resamples: int,
    v_min_fraction: float | None,
    v_min_count: int | None,
    k_selection_mode: str,
) -> dict[str, object]:
    probabilities = grouped_fc_to_probabilities(grouped_fc, zero_negative=zero_negative)
    run_probabilities = [grouped_fc_to_probabilities(fc_run, zero_negative=zero_negative) for fc_run in run_grouped_fcs]
    long_axis_order = compute_long_axis_order(surface_coords) if strict_soft_route else None
    if strict_soft_route:
        regularized_probabilities = regularize_probability_profiles(
            probabilities,
            connectivity,
            long_axis_order=long_axis_order,
        )
        features_full = zscore_columns(regularized_probabilities)
        run_features = [
            zscore_columns(
                regularize_probability_profiles(prob_run, connectivity, long_axis_order=long_axis_order)
            )
            for prob_run in run_probabilities
        ]
    else:
        regularized_probabilities = None
        features_full = zscore_columns(probabilities)
        run_features = [zscore_columns(prob_run) for prob_run in run_probabilities]

    feature_dir.mkdir(parents=True, exist_ok=True)
    np.save(feature_dir / "network_probabilities.npy", probabilities.astype(np.float32))
    np.save(feature_dir / "grouped_fc.npy", grouped_fc.astype(np.float32))
    save_json(
        feature_dir / "feature_summary.json",
        {
            "hemi": hemi,
            "feature_kind": "probability-regularized" if strict_soft_route else "probability",
            "feature_shape": [int(features_full.shape[0]), int(features_full.shape[1])],
            "networks": networks,
            "fisher_z_transform": True,
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
            "run_labels": run_labels,
            "split_strategy": split_strategy,
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
        run_features=run_features,
        run_labels=run_labels,
        profile_source=regularized_probabilities if strict_soft_route else probabilities,
        profile_networks=networks,
        connectivity=connectivity,
        outdir=clustering_dir,
        hemi=hemi,
        profile_mode="probability",
        split_strategy=split_strategy,
        instability_resamples=instability_resamples,
        v_min_fraction=v_min_fraction,
        v_min_count=v_min_count,
        k_selection_mode=k_selection_mode,
        clustering_method="AgglomerativeClustering(linkage=ward)",
        distance_metric="euclidean",
    )
    cluster["feature_summary"] = {
        "feature_kind": "probability-regularized" if strict_soft_route else "probability",
        "feature_shape": [int(features_full.shape[0]), int(features_full.shape[1])],
        "networks": networks,
        "fisher_z_transform": True,
        "negative_fc_policy": "clip-to-zero" if zero_negative else "row-min-shift",
        "run_labels": run_labels,
        "split_strategy": split_strategy,
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

    fisher_fc = fisher_z_transform_fc(grouped_fc)
    order = np.argsort(fisher_fc, axis=1)
    best = order[:, -1]
    second = order[:, -2] if grouped_fc.shape[1] > 1 else order[:, -1]
    labels_final = best + 1
    confidence = fisher_fc[np.arange(fisher_fc.shape[0]), best] - fisher_fc[np.arange(fisher_fc.shape[0]), second]

    np.save(soft_dir / "hipp_wta_labels.npy", labels_final.astype(np.int32))
    np.save(soft_dir / "hipp_wta_confidence.npy", confidence.astype(np.float32))
    np.save(soft_dir / "hipp_to_network_correlations.npy", grouped_fc.astype(np.float32))
    np.save(soft_dir / "hipp_to_network_correlations_fisherz.npy", fisher_fc.astype(np.float32))

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
            "fisher_z_transform": True,
            "winner_selection_basis": "fisher-z-fc",
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
            "mean_grouped_fc_fisherz": fisher_fc.mean(axis=0).astype(np.float32).tolist(),
            "network_occupancy": occupancy,
            "mean_confidence": float(np.mean(confidence)),
            "median_confidence": float(np.median(confidence)),
        },
    }


def run_spectral_branch(
    *,
    grouped_fc: np.ndarray,
    run_grouped_fcs: list[np.ndarray],
    run_labels: list[str],
    networks: list[str],
    connectivity: sparse.csr_matrix,
    feature_dir: Path,
    clustering_dir: Path,
    hemi: str,
    split_strategy: str,
    instability_resamples: int,
    v_min_fraction: float | None,
    v_min_count: int | None,
    k_selection_mode: str,
    zero_negative: bool,
) -> dict[str, object]:
    grouped_fc = fisher_z_transform_fc(grouped_fc)
    run_grouped_fcs = [fisher_z_transform_fc(fc_run) for fc_run in run_grouped_fcs]
    if zero_negative:
        grouped_fc = np.clip(grouped_fc, 0.0, None)
        run_grouped_fcs = [np.clip(fc_run, 0.0, None) for fc_run in run_grouped_fcs]

    features_full = zscore_columns(grouped_fc)
    run_features = [zscore_columns(fc_run) for fc_run in run_grouped_fcs]

    feature_dir.mkdir(parents=True, exist_ok=True)
    np.save(feature_dir / "grouped_fc.npy", grouped_fc.astype(np.float32))
    save_json(
        feature_dir / "feature_summary.json",
        {
            "hemi": hemi,
            "feature_kind": "network-spectral",
            "feature_shape": [int(features_full.shape[0]), int(features_full.shape[1])],
            "source_networks": networks,
            "fisher_z_transform": True,
            "negative_fc_policy": "clip-to-zero" if zero_negative else "preserve-signed-fc",
            "spatial_constraint": "surface_mesh_adjacency",
            "run_labels": run_labels,
            "split_strategy": split_strategy,
        },
    )

    def _spectral_fn(features: np.ndarray, adjacency: sparse.csr_matrix, k: int) -> np.ndarray:
        return spectral_cluster_from_features(features, adjacency, k)

    cluster = evaluate_k_range(
        features_full=features_full,
        run_features=run_features,
        run_labels=run_labels,
        profile_source=grouped_fc,
        profile_networks=networks,
        connectivity=connectivity,
        outdir=clustering_dir,
        hemi=hemi,
        profile_mode="fc",
        split_strategy=split_strategy,
        instability_resamples=instability_resamples,
        v_min_fraction=v_min_fraction,
        v_min_count=v_min_count,
        k_selection_mode=k_selection_mode,
        clustering_method="SpectralClustering(affinity=precomputed, assign_labels=kmeans)",
        distance_metric="cosine-similarity-affinity",
        cluster_fn=_spectral_fn,
    )
    cluster["feature_summary"] = {
        "feature_kind": "network-spectral",
        "feature_shape": [int(features_full.shape[0]), int(features_full.shape[1])],
        "source_networks": networks,
        "fisher_z_transform": True,
        "negative_fc_policy": "clip-to-zero" if zero_negative else "preserve-signed-fc",
        "spatial_constraint": "surface_mesh_adjacency",
        "run_labels": run_labels,
        "split_strategy": split_strategy,
    }
    return cluster


def run_intrinsic_spectral_branch(
    *,
    intrinsic_fc: np.ndarray,
    run_intrinsic_fcs: list[np.ndarray],
    grouped_fc_for_annotation: np.ndarray,
    run_labels: list[str],
    networks: list[str],
    connectivity: sparse.csr_matrix,
    feature_dir: Path,
    clustering_dir: Path,
    hemi: str,
    split_strategy: str,
    instability_resamples: int,
    v_min_fraction: float | None,
    v_min_count: int | None,
    k_selection_mode: str,
    zero_negative: bool,
) -> dict[str, object]:
    transformed_full = prepare_intrinsic_spectral_features(intrinsic_fc, zero_negative=zero_negative)
    transformed_runs = [
        prepare_intrinsic_spectral_features(run_fc, zero_negative=zero_negative)
        for run_fc in run_intrinsic_fcs
    ]

    features_full = zscore_columns(transformed_full)
    run_features = [zscore_columns(run_fc) for run_fc in transformed_runs]

    feature_dir.mkdir(parents=True, exist_ok=True)
    np.save(feature_dir / "intrinsic_vertex_to_vertex_fc.npy", intrinsic_fc.astype(np.float32))
    np.save(feature_dir / "intrinsic_vertex_to_vertex_fc_fisherz.npy", transformed_full.astype(np.float32))
    save_json(
        feature_dir / "feature_summary.json",
        {
            "hemi": hemi,
            "feature_kind": "intrinsic-spectral",
            "feature_shape": [int(features_full.shape[0]), int(features_full.shape[1])],
            "source_type": "vertex-to-vertex-fc",
            "annotation_source_type": "vertex-to-network-fc",
            "source_networks": networks,
            "fisher_z_transform": True,
            "diagonal_policy": "set-to-zero",
            "negative_fc_policy": "clip-to-zero" if zero_negative else "preserve-signed-fc",
            "spatial_constraint": "surface_mesh_adjacency",
            "run_labels": run_labels,
            "split_strategy": split_strategy,
        },
    )

    def _spectral_fn(features: np.ndarray, adjacency: sparse.csr_matrix, k: int) -> np.ndarray:
        return spectral_cluster_from_features(features, adjacency, k)

    cluster = evaluate_k_range(
        features_full=features_full,
        run_features=run_features,
        run_labels=run_labels,
        profile_source=grouped_fc_for_annotation,
        profile_networks=networks,
        connectivity=connectivity,
        outdir=clustering_dir,
        hemi=hemi,
        profile_mode="fc",
        split_strategy=split_strategy,
        instability_resamples=instability_resamples,
        v_min_fraction=v_min_fraction,
        v_min_count=v_min_count,
        k_selection_mode=k_selection_mode,
        clustering_method="SpectralClustering(affinity=precomputed, assign_labels=kmeans)",
        distance_metric="cosine-similarity-affinity",
        cluster_fn=_spectral_fn,
    )
    cluster["feature_summary"] = {
        "feature_kind": "intrinsic-spectral",
        "feature_shape": [int(features_full.shape[0]), int(features_full.shape[1])],
        "source_type": "vertex-to-vertex-fc",
        "annotation_source_type": "vertex-to-network-fc",
        "source_networks": networks,
        "fisher_z_transform": True,
        "diagonal_policy": "set-to-zero",
        "negative_fc_policy": "clip-to-zero" if zero_negative else "preserve-signed-fc",
        "spatial_constraint": "surface_mesh_adjacency",
        "run_labels": run_labels,
        "split_strategy": split_strategy,
    }
    return cluster


def build_panel_titles(branch_slug: str, smooth_name: str, hemi: str, k_final: int) -> str:
    smooth_label = smooth_name
    if is_soft_branch(branch_slug):
        return f"{smooth_label} {hemi} network soft-profile subregions (K={k_final})"
    if is_wta_branch(branch_slug):
        return f"{smooth_label} {hemi} network winner-takes-all"
    if is_gradient_branch(branch_slug):
        return f"{smooth_label} {hemi} network-gradient final K={k_final}"
    if is_intrinsic_spectral_branch(branch_slug):
        return f"{smooth_label} {hemi} {branch_slug} final K={k_final}"
    return f"{smooth_label} {hemi} network-cluster final K={k_final}"


def main() -> int:
    parser = argparse.ArgumentParser(description="Run single-subject network-first hippocampal functional parcellation")
    parser.add_argument("--subject", default="100610")
    parser.add_argument("--branch", default="network-gradient", choices=BRANCHES)
    parser.add_argument("--atlas-slug", default="lynch2024", choices=sorted(ATLAS_CONFIG))
    parser.add_argument("--input-root", default=str(REPO_ROOT / "data" / "hippunfold_input"))
    parser.add_argument("--hippunfold-root", default=str(REPO_ROOT / "outputs_migration" / "dense_corobl_batch"))
    parser.add_argument("--cortex-root", default=str(REPO_ROOT / "outputs_migration" / "cortex_pfm"))
    parser.add_argument("--out-root", default=str(REPO_ROOT / "outputs_migration" / "hipp_functional_parcellation_network"))
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
    parser.add_argument("--instability-resamples", type=int, default=DEFAULT_INSTABILITY_RESAMPLES)
    parser.add_argument("--v-min-fraction", type=float, default=DEFAULT_V_MIN_FRACTION)
    parser.add_argument("--v-min-count", type=int, default=None)
    parser.add_argument("--k-selection-mode", choices=["mainline", "experimental"], default="mainline")
    parser.add_argument("--run-split-mode", choices=["none", "runwise"], default="none")
    parser.add_argument("--hipp-density", default=DEFAULT_HIPP_DENSITY)
    args = parser.parse_args()

    views = [token.strip() for token in args.views.split(",") if token.strip()]
    valid_views = {"ventral", "dorsal"}
    if not views or any(token not in valid_views for token in views):
        raise ValueError(f"Invalid --views: {args.views}")
    subject = args.subject
    branch_slug = args.branch
    atlas_slug = args.atlas_slug
    hipp_density = str(args.hipp_density)
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
    shared_fc_store_dir = shared_store_root / f"sub-{subject}" / "fc" / atlas_slug
    scene = Path(args.scene).resolve()
    func_input_dir = input_root / f"sub-{subject}" / "func"

    roi_summary_probe = cortex_root / "roi_components" / "roi_component_stats.json"

    dtseries = func_input_dir / f"sub-{subject}_task-rest_run-concat.dtseries.nii"
    concat_bold = func_input_dir / f"sub-{subject}_task-rest_run-concat_bold.nii.gz"
    runwise_dtseries = [
        func_input_dir / f"sub-{subject}_task-rest_run-{spec['run_id']}.dtseries.nii"
        for spec in RUN_SPECS
    ]
    runwise_bold = [
        func_input_dir / f"sub-{subject}_task-rest_run-{spec['run_id']}_bold.nii.gz"
        for spec in RUN_SPECS
    ]
    surf_dir = hipp_root / "hippunfold" / f"sub-{subject}" / "surf"
    if not dtseries.exists():
        raise FileNotFoundError(f"Missing dtseries: {dtseries}")
    if not concat_bold.exists():
        raise FileNotFoundError(
            "Missing concat BOLD required for strict hippocampal raw surface sampling: "
            f"{concat_bold}"
        )
    shared_runwise_input_dir = shared_store_root / f"sub-{subject}" / "runwise_inputs"
    if args.run_split_mode == "runwise":
        resolved_runwise_dtseries, dtseries_run_input_mode, dtseries_run_lengths = resolve_runwise_dtseries(
            subject=subject,
            concat_dtseries=dtseries,
            runwise_dtseries=runwise_dtseries,
            stage_root=shared_runwise_input_dir,
            resume_mode=resume_mode,
        )
        bold_assets_available = concat_bold.exists() or all(path.exists() for path in runwise_bold)
        resolved_runwise_bold: list[Path] = []
        bold_run_input_mode = "not_available_split_surface_from_concat_timeseries"
        if bold_assets_available:
            resolved_runwise_bold, bold_run_input_mode, bold_run_lengths = resolve_runwise_bold(
                subject=subject,
                concat_bold=concat_bold,
                runwise_bold=runwise_bold,
                stage_root=shared_runwise_input_dir,
                resume_mode=resume_mode,
            )
            if dtseries_run_lengths != bold_run_lengths:
                raise ValueError(
                    "Run-wise dtseries and BOLD lengths do not match after input resolution: "
                    f"dtseries={dtseries_run_lengths}, bold={bold_run_lengths}"
                )
    else:
        total_tp = int(nib.load(str(dtseries)).shape[0])
        resolved_runwise_dtseries = [dtseries for _ in RUN_SPECS]
        dtseries_run_input_mode = "concat_no_split"
        dtseries_run_lengths = [total_tp for _ in RUN_SPECS]
        resolved_runwise_bold = []
        bold_run_input_mode = "not_used_concat_no_split"

    left_surface = find_hippunfold_surface_asset(
        surf_dir=surf_dir,
        subject=subject,
        hemi="L",
        space="corobl",
        density=hipp_density,
        suffix="midthickness.surf.gii",
    )
    right_surface = find_hippunfold_surface_asset(
        surf_dir=surf_dir,
        subject=subject,
        hemi="R",
        space="corobl",
        density=hipp_density,
        suffix="midthickness.surf.gii",
    )
    structural_dlabel = find_hippunfold_cifti_asset(
        cifti_dir=hipp_root / "hippunfold" / f"sub-{subject}" / "cifti",
        subject=subject,
        density=hipp_density,
        suffix="atlas-multihist7_subfields.dlabel.nii",
    )
    structural_spec = find_hippunfold_surface_asset(
        surf_dir=surf_dir,
        subject=subject,
        hemi="L",
        space="corobl",
        density=hipp_density,
        suffix="midthickness.surf.gii",
    ).parent / f"sub-{subject}_den-{hipp_density}_surfaces.spec"

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

    left_struct_labels, right_struct_labels = separate_hippunfold_structural_dlabel(
        dlabel_path=structural_dlabel,
        output_dir=workbench_dir / "structural_input",
        subject=subject,
        density=hipp_density,
        resume_mode=resume_mode,
    )

    left_cortex_labels = cortex_root / "roi_components" / "hemi_L" / f"{atlas_cfg['label_prefix']}.L.label.gii"
    right_cortex_labels = cortex_root / "roi_components" / "hemi_R" / f"{atlas_cfg['label_prefix']}.R.label.gii"
    roi_summary_path = cortex_root / "roi_components" / "roi_component_stats.json"

    cortex_tsnr_dir = shared_reference_store_dir / "tsnr_gate"
    cortex_tsnr_summary_path = cortex_tsnr_dir / "cortex_tsnr_gate_summary.json"
    cortex_tsnr_outputs = [
        cortex_tsnr_summary_path,
        cortex_tsnr_dir / "cortex_left_tsnr.npy",
        cortex_tsnr_dir / "cortex_right_tsnr.npy",
        cortex_tsnr_dir / "cortex_left_valid_mask.npy",
        cortex_tsnr_dir / "cortex_right_valid_mask.npy",
    ]
    cortex_tsnr_params = {
        "subject": subject,
        "threshold": float(TSNR_THRESHOLD),
        "tsnr_definition": "10000/std",
    }
    if not stage_is_up_to_date(
        stage_dir=cortex_tsnr_dir,
        resume_mode=resume_mode,
        stage_name="cortex_tsnr_gate",
        params=cortex_tsnr_params,
        inputs=[dtseries],
        outputs=cortex_tsnr_outputs,
    ):
        cortex_tsnr_result = compute_cortex_tsnr_gate(
            dtseries_path=dtseries,
            output_dir=cortex_tsnr_dir,
        )
        write_stage_manifest(
            stage_dir=cortex_tsnr_dir,
            stage_name="cortex_tsnr_gate",
            params=cortex_tsnr_params,
            inputs=[dtseries],
            outputs=[
                cortex_tsnr_result["summary_path"],
                cortex_tsnr_result["left_tsnr_path"],
                cortex_tsnr_result["right_tsnr_path"],
                cortex_tsnr_result["left_mask_path"],
                cortex_tsnr_result["right_mask_path"],
            ],
        )

    reference_summary_path = shared_reference_store_dir / "reference_summary.json"
    canonical_network_table_path = shared_reference_store_dir / "cortex_canonical_networks.tsv"
    canonical_network_timeseries_path = shared_reference_store_dir / "cortex_canonical_network_timeseries.npy"
    reference_params = {
        "subject": subject,
        "atlas_slug": atlas_slug,
        "label_prefix": str(atlas_cfg["label_prefix"]),
        "tsnr_threshold": float(TSNR_THRESHOLD),
    }
    reference_inputs = [
        dtseries,
        left_cortex_labels,
        right_cortex_labels,
        roi_summary_path,
        cortex_tsnr_dir / "cortex_left_valid_mask.npy",
        cortex_tsnr_dir / "cortex_right_valid_mask.npy",
    ]
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
                "--left-tsnr-mask",
                str(cortex_tsnr_dir / "cortex_left_valid_mask.npy"),
                "--right-tsnr-mask",
                str(cortex_tsnr_dir / "cortex_right_valid_mask.npy"),
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
    write_fc_store_pointer(
        pointer_dir=fc_dir,
        shared_fc_store_dir=shared_fc_store_dir,
        subject=subject,
        atlas_slug=atlas_slug,
    )

    reference_summary = json.loads(reference_summary_path.read_text(encoding="utf-8"))
    canonical_network_rows = load_canonical_network_rows(canonical_network_table_path)
    network_ts = np.load(canonical_network_timeseries_path).astype(np.float32)
    if args.run_split_mode == "runwise":
        run_reference_paths: list[Path] = []
        for spec, run_dtseries in zip(RUN_SPECS, resolved_runwise_dtseries, strict=True):
            run_reference_dir = shared_reference_store_dir / "runs" / f"run-{spec['run_id']}"
            run_reference_summary = run_reference_dir / "reference_summary.json"
            run_canonical_network_timeseries = run_reference_dir / "cortex_canonical_network_timeseries.npy"
            run_reference_paths.append(run_canonical_network_timeseries)
            run_reference_params = {
                "subject": subject,
                "atlas_slug": atlas_slug,
                "label_prefix": str(atlas_cfg["label_prefix"]),
                "run_id": spec["run_id"],
                "tsnr_threshold": float(TSNR_THRESHOLD),
            }
            if not stage_is_up_to_date(
                stage_dir=run_reference_dir,
                resume_mode=resume_mode,
                stage_name="reference_run",
                params=run_reference_params,
                inputs=[
                    run_dtseries,
                    left_cortex_labels,
                    right_cortex_labels,
                    roi_summary_path,
                    cortex_tsnr_dir / "cortex_left_valid_mask.npy",
                    cortex_tsnr_dir / "cortex_right_valid_mask.npy",
                ],
                outputs=[run_reference_summary, run_canonical_network_timeseries],
            ):
                run(
                    [
                        PYTHON_EXE,
                        str(REPO_ROOT / "scripts" / "cortex" / "extract_cortex_roi_component_timeseries.py"),
                        "--subject",
                        subject,
                        "--dtseries",
                        str(run_dtseries),
                        "--left-labels",
                        str(left_cortex_labels),
                        "--right-labels",
                        str(right_cortex_labels),
                        "--roi-summary",
                        str(roi_summary_path),
                        "--atlas-slug",
                        atlas_slug,
                        "--left-tsnr-mask",
                        str(cortex_tsnr_dir / "cortex_left_valid_mask.npy"),
                        "--right-tsnr-mask",
                        str(cortex_tsnr_dir / "cortex_right_valid_mask.npy"),
                        "--outdir",
                        str(run_reference_dir),
                    ]
                )
                write_stage_manifest(
                    stage_dir=run_reference_dir,
                    stage_name="reference_run",
                    params=run_reference_params,
                    inputs=[
                        run_dtseries,
                        left_cortex_labels,
                        right_cortex_labels,
                        roi_summary_path,
                        cortex_tsnr_dir / "cortex_left_valid_mask.npy",
                        cortex_tsnr_dir / "cortex_right_valid_mask.npy",
                    ],
                    outputs=[run_reference_summary, run_canonical_network_timeseries],
                )
        run_network_ts = [np.load(path).astype(np.float32) for path in run_reference_paths]
    else:
        run_reference_paths = []
        run_network_ts = [network_ts.astype(np.float32, copy=False) for _ in RUN_SPECS]
    networks = [str(row["canonical_network"]) for row in canonical_network_rows]

    shared_raw_surface_dir = shared_surface_store_dir / "raw"
    left_raw_metric = shared_raw_surface_dir / f"sub-{subject}_hemi-L_space-corobl_den-{hipp_density}_label-hipp_bold.func.gii"
    right_raw_metric = shared_raw_surface_dir / f"sub-{subject}_hemi-R_space-corobl_den-{hipp_density}_label-hipp_bold.func.gii"

    left_coords, left_faces = load_surface(left_surface)
    right_coords, right_faces = load_surface(right_surface)
    left_adj = build_surface_adjacency(left_faces, int(left_coords.shape[0]))
    right_adj = build_surface_adjacency(right_faces, int(right_coords.shape[0]))

    hipp_tsnr_dir = shared_surface_store_dir / "tsnr_gate"
    left_roi_path = hipp_tsnr_dir / f"sub-{subject}_hemi-L_valid_roi.shape.gii"
    right_roi_path = hipp_tsnr_dir / f"sub-{subject}_hemi-R_valid_roi.shape.gii"
    left_hipp_tsnr_outputs = [
        hipp_tsnr_dir / f"sub-{subject}_hemi-L_tsnr_gate_summary.json",
        hipp_tsnr_dir / f"sub-{subject}_hemi-L_tsnr.npy",
        hipp_tsnr_dir / f"sub-{subject}_hemi-L_valid_mask.npy",
        hipp_tsnr_dir / f"sub-{subject}_hemi-L_invalid_initial_mask.npy",
        left_roi_path,
    ]
    right_hipp_tsnr_outputs = [
        hipp_tsnr_dir / f"sub-{subject}_hemi-R_tsnr_gate_summary.json",
        hipp_tsnr_dir / f"sub-{subject}_hemi-R_tsnr.npy",
        hipp_tsnr_dir / f"sub-{subject}_hemi-R_valid_mask.npy",
        hipp_tsnr_dir / f"sub-{subject}_hemi-R_invalid_initial_mask.npy",
        right_roi_path,
    ]
    shared_raw_surface_params = {
        "subject": subject,
        "hipp_density": hipp_density,
        "mapping_method": "trilinear",
        "smooth_iters": 0,
        "source_bold": str(concat_bold.resolve()),
        "raw_source_policy": "strict_shared_pipeline_func_gii_only",
    }
    shared_raw_surface_outputs = [left_raw_metric, right_raw_metric, shared_raw_surface_dir / "surface_sampling_summary.json"]
    if not stage_is_up_to_date(
        stage_dir=shared_raw_surface_dir,
        resume_mode=resume_mode,
        stage_name="surface_raw",
        params=shared_raw_surface_params,
        inputs=[concat_bold, left_surface, right_surface],
        outputs=shared_raw_surface_outputs,
    ):
        run(
            [
                PYTHON_EXE,
                str(REPO_ROOT / "scripts" / "common" / "sample_hipp_surface_timeseries.py"),
                "--bold",
                str(concat_bold),
                "--hippunfold-dir",
                str(hipp_root / "hippunfold"),
                "--subject",
                subject,
                "--density",
                hipp_density,
                "--space",
                "corobl",
                "--mapping-method",
                "trilinear",
                "--smooth-iters",
                "0",
                "--outdir",
                str(shared_raw_surface_dir),
            ]
        )
        write_stage_manifest(
            stage_dir=shared_raw_surface_dir,
            stage_name="surface_raw",
            params=shared_raw_surface_params,
            inputs=[concat_bold, left_surface, right_surface],
            outputs=shared_raw_surface_outputs,
        )

    hipp_tsnr_params = {
        "subject": subject,
        "threshold": float(TSNR_THRESHOLD),
        "tsnr_definition": "10000/std",
        "hipp_density": hipp_density,
        "raw_source_policy": "strict_shared_pipeline_func_gii_only",
        "null_policy": "strict_invalid_initial_only",
    }
    if not stage_is_up_to_date(
        stage_dir=hipp_tsnr_dir / "hemi-L",
        resume_mode=resume_mode,
        stage_name="hipp_tsnr_gate",
        params={**hipp_tsnr_params, "hemi": "L"},
        inputs=[left_raw_metric, left_surface, concat_bold],
        outputs=left_hipp_tsnr_outputs,
    ):
        left_raw_timeseries, left_raw_tsnr_source = load_surface_timeseries_for_tsnr(
            metric_path=left_raw_metric,
            expected_n_vertices=int(left_coords.shape[0]),
        )
        left_raw_timeseries = sanitize_timeseries(left_raw_timeseries)
        left_hipp_tsnr = compute_hipp_tsnr_gate(
            subject=subject,
            hemi="L",
            raw_metric=left_raw_timeseries,
            raw_metric_source=left_raw_tsnr_source,
            connectivity=left_adj,
            faces=left_faces,
            output_dir=hipp_tsnr_dir,
        )
        save_shape_gii(left_hipp_tsnr["valid_mask"].astype(np.float32), left_roi_path)
        write_stage_manifest(
            stage_dir=hipp_tsnr_dir / "hemi-L",
            stage_name="hipp_tsnr_gate",
            params={**hipp_tsnr_params, "hemi": "L"},
                inputs=[left_raw_metric, left_surface, concat_bold],
                outputs=[
                left_hipp_tsnr["summary_path"],
                left_hipp_tsnr["tsnr_path"],
                left_hipp_tsnr["valid_mask_path"],
                left_hipp_tsnr["invalid_mask_path"],
                left_roi_path,
            ],
        )
    if not stage_is_up_to_date(
        stage_dir=hipp_tsnr_dir / "hemi-R",
        resume_mode=resume_mode,
        stage_name="hipp_tsnr_gate",
        params={**hipp_tsnr_params, "hemi": "R"},
        inputs=[right_raw_metric, right_surface, concat_bold],
        outputs=right_hipp_tsnr_outputs,
    ):
        right_raw_timeseries, right_raw_tsnr_source = load_surface_timeseries_for_tsnr(
            metric_path=right_raw_metric,
            expected_n_vertices=int(right_coords.shape[0]),
        )
        right_raw_timeseries = sanitize_timeseries(right_raw_timeseries)
        right_hipp_tsnr = compute_hipp_tsnr_gate(
            subject=subject,
            hemi="R",
            raw_metric=right_raw_timeseries,
            raw_metric_source=right_raw_tsnr_source,
            connectivity=right_adj,
            faces=right_faces,
            output_dir=hipp_tsnr_dir,
        )
        save_shape_gii(right_hipp_tsnr["valid_mask"].astype(np.float32), right_roi_path)
        write_stage_manifest(
            stage_dir=hipp_tsnr_dir / "hemi-R",
            stage_name="hipp_tsnr_gate",
            params={**hipp_tsnr_params, "hemi": "R"},
                inputs=[right_raw_metric, right_surface, concat_bold],
                outputs=[
                right_hipp_tsnr["summary_path"],
                right_hipp_tsnr["tsnr_path"],
                right_hipp_tsnr["valid_mask_path"],
                right_hipp_tsnr["invalid_mask_path"],
                right_roi_path,
            ],
        )

    left_valid_mask = np.load(hipp_tsnr_dir / f"sub-{subject}_hemi-L_valid_mask.npy").astype(bool)
    right_valid_mask = np.load(hipp_tsnr_dir / f"sub-{subject}_hemi-R_valid_mask.npy").astype(bool)
    left_invalid_initial_mask = np.load(hipp_tsnr_dir / f"sub-{subject}_hemi-L_invalid_initial_mask.npy").astype(bool)
    right_invalid_initial_mask = np.load(hipp_tsnr_dir / f"sub-{subject}_hemi-R_invalid_initial_mask.npy").astype(bool)
    left_hipp_tsnr_summary = json.loads((hipp_tsnr_dir / f"sub-{subject}_hemi-L_tsnr_gate_summary.json").read_text(encoding="utf-8"))
    right_hipp_tsnr_summary = json.loads((hipp_tsnr_dir / f"sub-{subject}_hemi-R_tsnr_gate_summary.json").read_text(encoding="utf-8"))

    two_mm_left_func = shared_surface_store_dir / "2mm" / f"sub-{subject}_hemi-L_space-corobl_den-{hipp_density}_label-hipp_bold.func.gii"
    two_mm_right_func = shared_surface_store_dir / "2mm" / f"sub-{subject}_hemi-R_space-corobl_den-{hipp_density}_label-hipp_bold.func.gii"
    two_mm_left_path = shared_surface_store_dir / "2mm" / f"sub-{subject}_hemi-L_timeseries.npy"
    two_mm_right_path = shared_surface_store_dir / "2mm" / f"sub-{subject}_hemi-R_timeseries.npy"
    fwhm_left_func = shared_surface_store_dir / "4mm" / f"sub-{subject}_hemi-L_space-corobl_den-{hipp_density}_label-hipp_bold.func.gii"
    fwhm_right_func = shared_surface_store_dir / "4mm" / f"sub-{subject}_hemi-R_space-corobl_den-{hipp_density}_label-hipp_bold.func.gii"
    fwhm_left_path = shared_surface_store_dir / "4mm" / f"sub-{subject}_hemi-L_timeseries.npy"
    fwhm_right_path = shared_surface_store_dir / "4mm" / f"sub-{subject}_hemi-R_timeseries.npy"
    surface_params = {
        "subject": subject,
        "smoothings": SMOOTH_ORDER,
        "tsnr_threshold": float(TSNR_THRESHOLD),
        "smoothing_roi_policy": "high_tsnr_vertices_only",
        "raw_source_policy": "strict_shared_pipeline_func_gii_only",
        "null_policy": "strict_invalid_initial_only",
    }
    surface_inputs = [left_raw_metric, right_raw_metric, left_surface, right_surface, left_roi_path, right_roi_path, concat_bold]
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
        for _hemi, surface_path, metric_path, smooth_mm, out_metric, roi_path in [
            ("L", left_surface, left_raw_metric, "2", two_mm_left_func, left_roi_path),
            ("R", right_surface, right_raw_metric, "2", two_mm_right_func, right_roi_path),
            ("L", left_surface, left_raw_metric, "4", fwhm_left_func, left_roi_path),
            ("R", right_surface, right_raw_metric, "4", fwhm_right_func, right_roi_path),
        ]:
            smooth_metric_with_roi(
                surface_path=surface_path,
                metric_path=metric_path,
                smooth_mm=smooth_mm,
                out_metric=out_metric,
                roi_path=roi_path,
            )
        np.save(
            two_mm_left_path,
            sanitize_timeseries_with_mask(
                load_metric_array(two_mm_left_func, expected_n_vertices=int(left_coords.shape[0])),
                left_valid_mask,
            ),
        )
        np.save(
            two_mm_right_path,
            sanitize_timeseries_with_mask(
                load_metric_array(two_mm_right_func, expected_n_vertices=int(right_coords.shape[0])),
                right_valid_mask,
            ),
        )
        np.save(
            fwhm_left_path,
            sanitize_timeseries_with_mask(
                load_metric_array(fwhm_left_func, expected_n_vertices=int(left_coords.shape[0])),
                left_valid_mask,
            ),
        )
        np.save(
            fwhm_right_path,
            sanitize_timeseries_with_mask(
                load_metric_array(fwhm_right_func, expected_n_vertices=int(right_coords.shape[0])),
                right_valid_mask,
            ),
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

    run_surface_specs: list[dict[str, object]] = []
    if args.run_split_mode == "none":
        for spec in RUN_SPECS:
            run_surface_specs.append(
                {
                    "run_id": spec["run_id"],
                    "label": spec["label"],
                    "2mm_left": two_mm_left_path,
                    "2mm_right": two_mm_right_path,
                    "4mm_left": fwhm_left_path,
                    "4mm_right": fwhm_right_path,
                }
            )
    elif resolved_runwise_bold:
        for spec, run_bold in zip(RUN_SPECS, resolved_runwise_bold, strict=True):
            run_surface_dir = shared_surface_store_dir / "runs" / f"run-{spec['run_id']}"
            raw_dir = run_surface_dir / "raw"
            run_raw_left_metric = raw_dir / f"sub-{subject}_hemi-L_space-corobl_den-{hipp_density}_label-hipp_bold.func.gii"
            run_raw_right_metric = raw_dir / f"sub-{subject}_hemi-R_space-corobl_den-{hipp_density}_label-hipp_bold.func.gii"
            run_two_mm_left_func = run_surface_dir / "2mm" / f"sub-{subject}_hemi-L_space-corobl_den-{hipp_density}_label-hipp_bold.func.gii"
            run_two_mm_right_func = run_surface_dir / "2mm" / f"sub-{subject}_hemi-R_space-corobl_den-{hipp_density}_label-hipp_bold.func.gii"
            run_two_mm_left_path = run_surface_dir / "2mm" / f"sub-{subject}_hemi-L_timeseries.npy"
            run_two_mm_right_path = run_surface_dir / "2mm" / f"sub-{subject}_hemi-R_timeseries.npy"
            run_four_mm_left_func = run_surface_dir / "4mm" / f"sub-{subject}_hemi-L_space-corobl_den-{hipp_density}_label-hipp_bold.func.gii"
            run_four_mm_right_func = run_surface_dir / "4mm" / f"sub-{subject}_hemi-R_space-corobl_den-{hipp_density}_label-hipp_bold.func.gii"
            run_four_mm_left_path = run_surface_dir / "4mm" / f"sub-{subject}_hemi-L_timeseries.npy"
            run_four_mm_right_path = run_surface_dir / "4mm" / f"sub-{subject}_hemi-R_timeseries.npy"
            run_surface_params = {
                "subject": subject,
                "run_id": spec["run_id"],
                "smoothings": SMOOTH_ORDER,
                "tsnr_threshold": float(TSNR_THRESHOLD),
                "smoothing_roi_policy": "high_tsnr_vertices_only",
                "raw_source_policy": "strict_runwise_pipeline_func_gii_only",
                "null_policy": "strict_invalid_initial_only",
            }
            run_surface_outputs = [
                run_raw_left_metric,
                run_raw_right_metric,
                run_two_mm_left_func,
                run_two_mm_right_func,
                run_two_mm_left_path,
                run_two_mm_right_path,
                run_four_mm_left_func,
                run_four_mm_right_func,
                run_four_mm_left_path,
                run_four_mm_right_path,
            ]
            if not stage_is_up_to_date(
                stage_dir=run_surface_dir,
                resume_mode=resume_mode,
                stage_name="surface_run",
                params=run_surface_params,
                inputs=[run_bold, left_surface, right_surface],
                outputs=run_surface_outputs,
            ):
                run(
                    [
                        PYTHON_EXE,
                        str(REPO_ROOT / "scripts" / "common" / "sample_hipp_surface_timeseries.py"),
                        "--bold",
                        str(run_bold),
                        "--hippunfold-dir",
                        str(hipp_root / "hippunfold"),
                        "--subject",
                        subject,
                        "--density",
                        hipp_density,
                        "--space",
                        "corobl",
                        "--mapping-method",
                        "trilinear",
                        "--smooth-iters",
                        "0",
                        "--outdir",
                        str(raw_dir),
                    ]
                )
                for _hemi, surface_path, metric_path, smooth_mm, out_metric, roi_path in [
                    ("L", left_surface, run_raw_left_metric, "2", run_two_mm_left_func, left_roi_path),
                    ("R", right_surface, run_raw_right_metric, "2", run_two_mm_right_func, right_roi_path),
                    ("L", left_surface, run_raw_left_metric, "4", run_four_mm_left_func, left_roi_path),
                    ("R", right_surface, run_raw_right_metric, "4", run_four_mm_right_func, right_roi_path),
                ]:
                    smooth_metric_with_roi(
                        surface_path=surface_path,
                        metric_path=metric_path,
                        smooth_mm=smooth_mm,
                        out_metric=out_metric,
                        roi_path=roi_path,
                    )
                np.save(
                    run_two_mm_left_path,
                    sanitize_timeseries_with_mask(
                        load_metric_array(run_two_mm_left_func, expected_n_vertices=int(left_coords.shape[0])),
                        left_valid_mask,
                    ),
                )
                np.save(
                    run_two_mm_right_path,
                    sanitize_timeseries_with_mask(
                        load_metric_array(run_two_mm_right_func, expected_n_vertices=int(right_coords.shape[0])),
                        right_valid_mask,
                    ),
                )
                np.save(
                    run_four_mm_left_path,
                    sanitize_timeseries_with_mask(
                        load_metric_array(run_four_mm_left_func, expected_n_vertices=int(left_coords.shape[0])),
                        left_valid_mask,
                    ),
                )
                np.save(
                    run_four_mm_right_path,
                    sanitize_timeseries_with_mask(
                        load_metric_array(run_four_mm_right_func, expected_n_vertices=int(right_coords.shape[0])),
                        right_valid_mask,
                    ),
                )
                write_stage_manifest(
                    stage_dir=run_surface_dir,
                    stage_name="surface_run",
                    params=run_surface_params,
                    inputs=[run_bold, left_surface, right_surface],
                    outputs=run_surface_outputs,
                )
            run_surface_specs.append(
                {
                    "run_id": spec["run_id"],
                    "label": spec["label"],
                    "2mm_left": run_two_mm_left_path,
                    "2mm_right": run_two_mm_right_path,
                    "4mm_left": run_four_mm_left_path,
                    "4mm_right": run_four_mm_right_path,
                }
            )
    else:
        run_surface_base_dir = shared_surface_store_dir / "runs_from_concat"
        for spec, run_len in zip(RUN_SPECS, dtseries_run_lengths, strict=True):
            run_surface_dir = run_surface_base_dir / f"run-{spec['run_id']}"
            run_two_mm_left_path = run_surface_dir / "2mm" / f"sub-{subject}_hemi-L_timeseries.npy"
            run_two_mm_right_path = run_surface_dir / "2mm" / f"sub-{subject}_hemi-R_timeseries.npy"
            run_four_mm_left_path = run_surface_dir / "4mm" / f"sub-{subject}_hemi-L_timeseries.npy"
            run_four_mm_right_path = run_surface_dir / "4mm" / f"sub-{subject}_hemi-R_timeseries.npy"
            run_surface_params = {
                "subject": subject,
                "run_id": spec["run_id"],
                "source_mode": "concat_surface_split",
                "run_length": int(run_len),
                "run_lengths": [int(x) for x in dtseries_run_lengths],
                "null_policy": "strict_invalid_initial_only",
            }
            run_surface_outputs = [
                run_two_mm_left_path,
                run_two_mm_right_path,
                run_four_mm_left_path,
                run_four_mm_right_path,
            ]
            if not stage_is_up_to_date(
                stage_dir=run_surface_dir,
                resume_mode=resume_mode,
                stage_name="surface_run_from_concat_surface",
                params=run_surface_params,
                inputs=[two_mm_left_path, two_mm_right_path, fwhm_left_path, fwhm_right_path],
                outputs=run_surface_outputs,
            ):
                split_surface_timeseries_to_runs(
                    two_mm_left_path,
                    [
                        run_surface_base_dir / f"run-{s['run_id']}" / "2mm" / f"sub-{subject}_hemi-L_timeseries.npy"
                        for s in RUN_SPECS
                    ],
                    dtseries_run_lengths,
                )
                split_surface_timeseries_to_runs(
                    two_mm_right_path,
                    [
                        run_surface_base_dir / f"run-{s['run_id']}" / "2mm" / f"sub-{subject}_hemi-R_timeseries.npy"
                        for s in RUN_SPECS
                    ],
                    dtseries_run_lengths,
                )
                split_surface_timeseries_to_runs(
                    fwhm_left_path,
                    [
                        run_surface_base_dir / f"run-{s['run_id']}" / "4mm" / f"sub-{subject}_hemi-L_timeseries.npy"
                        for s in RUN_SPECS
                    ],
                    dtseries_run_lengths,
                )
                split_surface_timeseries_to_runs(
                    fwhm_right_path,
                    [
                        run_surface_base_dir / f"run-{s['run_id']}" / "4mm" / f"sub-{subject}_hemi-R_timeseries.npy"
                        for s in RUN_SPECS
                    ],
                    dtseries_run_lengths,
                )
                write_stage_manifest(
                    stage_dir=run_surface_dir,
                    stage_name="surface_run_from_concat_surface",
                    params=run_surface_params,
                    inputs=[two_mm_left_path, two_mm_right_path, fwhm_left_path, fwhm_right_path],
                    outputs=run_surface_outputs,
                )
            run_surface_specs.append(
                {
                    "run_id": spec["run_id"],
                    "label": spec["label"],
                    "2mm_left": run_two_mm_left_path,
                    "2mm_right": run_two_mm_right_path,
                    "4mm_left": run_four_mm_left_path,
                    "4mm_right": run_four_mm_right_path,
                }
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

    shared_fc_stage_inputs = [
        canonical_network_timeseries_path,
        *run_reference_paths,
        two_mm_left_path,
        two_mm_right_path,
        fwhm_left_path,
        fwhm_right_path,
        *[Path(item["2mm_left"]) for item in run_surface_specs],
        *[Path(item["2mm_right"]) for item in run_surface_specs],
        *[Path(item["4mm_left"]) for item in run_surface_specs],
        *[Path(item["4mm_right"]) for item in run_surface_specs],
        hipp_tsnr_dir / f"sub-{subject}_hemi-L_valid_mask.npy",
        hipp_tsnr_dir / f"sub-{subject}_hemi-R_valid_mask.npy",
        hipp_tsnr_dir / f"sub-{subject}_hemi-L_invalid_initial_mask.npy",
        hipp_tsnr_dir / f"sub-{subject}_hemi-R_invalid_initial_mask.npy",
    ]
    need_intrinsic_fc = is_intrinsic_spectral_branch(branch_slug)
    shared_fc_params = {
        "subject": subject,
        "atlas_slug": atlas_slug,
        "smoothings": SMOOTH_ORDER,
        "run_labels": [f"run-{spec['run_id']}" for spec in RUN_SPECS],
        "dtseries_run_input_mode": dtseries_run_input_mode,
        "bold_run_input_mode": bold_run_input_mode,
        "run_lengths": dtseries_run_lengths,
        "hipp_density": hipp_density,
        "tsnr_threshold": float(TSNR_THRESHOLD),
        "fc_definition": (
            "pearson_vertex_to_canonical_network_timeseries_and_vertex_to_vertex_timeseries"
            if need_intrinsic_fc
            else "pearson_vertex_to_canonical_network_timeseries"
        ),
    }
    for smooth_name in SMOOTH_ORDER:
        spec = smooth_specs[smooth_name]
        for hemi in HEMIS:
            ts = np.asarray(spec["left" if hemi == "L" else "right"], dtype=np.float32)
            valid_mask = left_valid_mask if hemi == "L" else right_valid_mask
            invalid_initial_mask = left_invalid_initial_mask if hemi == "L" else right_invalid_initial_mask
            active_mask = valid_mask
            run_ts_paths = [Path(item[f"{smooth_name}_{'left' if hemi == 'L' else 'right'}"]) for item in run_surface_specs]
            shared_hemi_fc_dir = shared_fc_store_dir / smooth_name / f"hemi_{hemi}"
            shared_hemi_fc_outputs = [
                shared_hemi_fc_dir / "hipp_vertex_to_network_fc.npy",
                shared_hemi_fc_dir / "hipp_vertex_to_network_fc_active.npy",
                shared_hemi_fc_dir / "fc_summary.json",
                *[shared_hemi_fc_dir / "runs" / f"run-{item['run_id']}" / "hipp_vertex_to_network_fc_active.npy" for item in RUN_SPECS],
                *(
                    [
                        shared_hemi_fc_dir / "hipp_vertex_to_vertex_fc_active.npy",
                        *[
                            shared_hemi_fc_dir / "runs" / f"run-{item['run_id']}" / "hipp_vertex_to_vertex_fc_active.npy"
                            for item in RUN_SPECS
                        ],
                    ]
                    if need_intrinsic_fc
                    else []
                ),
            ]
            shared_hemi_fc_params = {
                **shared_fc_params,
                "smooth": smooth_name,
                "hemi": hemi,
            }
            if not stage_is_up_to_date(
                stage_dir=shared_hemi_fc_dir,
                resume_mode=resume_mode,
                stage_name="shared_fc",
                params=shared_hemi_fc_params,
                inputs=[*shared_fc_stage_inputs, *run_ts_paths],
                outputs=shared_hemi_fc_outputs,
            ):
                shared_hemi_fc_dir.mkdir(parents=True, exist_ok=True)
                fc_valid = corrcoef_rows(ts[valid_mask, :], network_ts)
                fc_full = np.full((ts.shape[0], network_ts.shape[0]), np.nan, dtype=np.float32)
                fc_full[valid_mask, :] = fc_valid.astype(np.float32)
                fc_active = compact_active_vertices(fc_full, active_mask).astype(np.float32)
                np.save(shared_hemi_fc_dir / "hipp_vertex_to_network_fc.npy", fc_full.astype(np.float32))
                np.save(shared_hemi_fc_dir / "hipp_vertex_to_network_fc_active.npy", fc_active)
                intrinsic_fc_active = None
                if need_intrinsic_fc:
                    intrinsic_fc_active = corrcoef_rows(ts[active_mask, :], ts[active_mask, :]).astype(np.float32)
                    np.save(shared_hemi_fc_dir / "hipp_vertex_to_vertex_fc_active.npy", intrinsic_fc_active)
                summary_payload: dict[str, object] = {
                    "subject": subject,
                    "atlas_slug": atlas_slug,
                    "smooth": smooth_name,
                    "hemi": hemi,
                    "fc_shape": [int(fc_full.shape[0]), int(fc_full.shape[1])],
                    "active_fc_shape": [int(fc_active.shape[0]), int(fc_active.shape[1])],
                    "n_vertices_total": int(ts.shape[0]),
                    "n_vertices_valid_high_tsnr": int(valid_mask.sum()),
                    "n_vertices_invalid_initial": int(invalid_initial_mask.sum()),
                    "n_vertices_clustered": int(active_mask.sum()),
                    "n_timepoints": int(ts.shape[1]),
                    "n_networks_used": int(network_ts.shape[0]),
                    "networks": networks,
                    "tsnr_threshold": float(TSNR_THRESHOLD),
                    "run_labels": [f"run-{item['run_id']}" for item in RUN_SPECS],
                    "dtseries_run_input_mode": dtseries_run_input_mode,
                    "bold_run_input_mode": bold_run_input_mode,
                }
                if intrinsic_fc_active is not None:
                    summary_payload["vertex_to_vertex_fc_shape"] = [
                        int(intrinsic_fc_active.shape[0]),
                        int(intrinsic_fc_active.shape[1]),
                    ]
                save_json(
                    shared_hemi_fc_dir / "fc_summary.json",
                    summary_payload,
                )
                for run_spec, run_ts_path, run_network in zip(RUN_SPECS, run_ts_paths, run_network_ts, strict=True):
                    run_ts_full = np.load(run_ts_path).astype(np.float32)
                    run_fc_valid = corrcoef_rows(run_ts_full[valid_mask, :], run_network)
                    run_fc_full = np.full((run_ts_full.shape[0], run_network.shape[0]), np.nan, dtype=np.float32)
                    run_fc_full[valid_mask, :] = run_fc_valid.astype(np.float32)
                    run_fc_active = compact_active_vertices(run_fc_full, active_mask).astype(np.float32)
                    run_fc_dir = shared_hemi_fc_dir / "runs" / f"run-{run_spec['run_id']}"
                    run_fc_dir.mkdir(parents=True, exist_ok=True)
                    np.save(run_fc_dir / "hipp_vertex_to_network_fc_active.npy", run_fc_active)
                    if need_intrinsic_fc:
                        run_intrinsic_fc_active = corrcoef_rows(
                            run_ts_full[active_mask, :],
                            run_ts_full[active_mask, :],
                        ).astype(np.float32)
                        np.save(run_fc_dir / "hipp_vertex_to_vertex_fc_active.npy", run_intrinsic_fc_active)
                write_stage_manifest(
                    stage_dir=shared_hemi_fc_dir,
                    stage_name="shared_fc",
                    params=shared_hemi_fc_params,
                    inputs=[*shared_fc_stage_inputs, *run_ts_paths],
                    outputs=shared_hemi_fc_outputs,
                )

    final_selection_core: dict[str, object] = {
        "subject": subject,
        "branch_slug": branch_slug,
        "atlas_slug": atlas_slug,
        "atlas_display_name": atlas_cfg["display_name"],
        "hipp_density": hipp_density,
        "shared_fc_store_dir": str(shared_fc_store_dir),
        "smoothings": SMOOTH_ORDER,
        "hemisphere_policy": "per-hemi",
        "k_policy": "run-aware-instability_2_to_10_per_hemi",
        "k_selection_mode": str(args.k_selection_mode),
        "run_input_sources": {
            "dtseries": dtseries_run_input_mode,
            "bold": bold_run_input_mode,
            "run_lengths": dtseries_run_lengths,
        },
        "reference_summary": reference_summary,
        "cortex_tsnr_gate": json.loads(cortex_tsnr_summary_path.read_text(encoding="utf-8")),
        "hipp_tsnr_gate": {
            "left": left_hipp_tsnr_summary,
            "right": right_hipp_tsnr_summary,
        },
        "per_smooth": {},
    }

    branch_tag = branch_slug.replace("-", "_")
    split_strategy = "run-pair"
    compute_params = {
        "subject": subject,
        "branch_slug": branch_slug,
        "atlas_slug": atlas_slug,
        "smoothings": SMOOTH_ORDER,
        "eval_k": EVAL_K,
        "split_strategy": split_strategy,
        "instability_resamples": int(args.instability_resamples),
        "k_selection_mode": str(args.k_selection_mode),
        "dtseries_run_input_mode": dtseries_run_input_mode,
        "bold_run_input_mode": bold_run_input_mode,
        "run_lengths": dtseries_run_lengths,
        "hipp_density": hipp_density,
        "tsnr_threshold": float(TSNR_THRESHOLD),
        "v_min_count": None if args.v_min_count is None else int(args.v_min_count),
        "v_min_fraction": float(args.v_min_fraction),
        "strict_soft_route": bool(is_soft_branch(branch_slug)),
        "negative_fc_policy": (
            "clip-to-zero"
            if (
                uses_nonnegative_probabilities(branch_slug)
                or uses_nonnegative_spectral_features(branch_slug)
                or uses_nonnegative_intrinsic_spectral_features(branch_slug)
            )
            else ("preserve-signed-fc" if is_spectral_branch(branch_slug) else "row-min-shift")
        ),
    }
    compute_inputs = [
        canonical_network_table_path,
        canonical_network_timeseries_path,
        *run_reference_paths,
        two_mm_left_path,
        two_mm_right_path,
        fwhm_left_path,
        fwhm_right_path,
        *[Path(spec["2mm_left"]) for spec in run_surface_specs],
        *[Path(spec["2mm_right"]) for spec in run_surface_specs],
        *[Path(spec["4mm_left"]) for spec in run_surface_specs],
        *[Path(spec["4mm_right"]) for spec in run_surface_specs],
        left_surface,
        right_surface,
        cortex_tsnr_summary_path,
        hipp_tsnr_dir / f"sub-{subject}_hemi-L_tsnr_gate_summary.json",
        hipp_tsnr_dir / f"sub-{subject}_hemi-R_tsnr_gate_summary.json",
    ]
    for smooth_name in SMOOTH_ORDER:
        for hemi in HEMIS:
            shared_hemi_fc_dir = shared_fc_store_dir / smooth_name / f"hemi_{hemi}"
            compute_inputs.extend(
                [
                    shared_hemi_fc_dir / "hipp_vertex_to_network_fc.npy",
                    shared_hemi_fc_dir / "hipp_vertex_to_network_fc_active.npy",
                    shared_hemi_fc_dir / "fc_summary.json",
                    *[
                        shared_hemi_fc_dir / "runs" / f"run-{spec['run_id']}" / "hipp_vertex_to_network_fc_active.npy"
                        for spec in RUN_SPECS
                    ],
                    *(
                        [
                            shared_hemi_fc_dir / "hipp_vertex_to_vertex_fc_active.npy",
                            *[
                                shared_hemi_fc_dir / "runs" / f"run-{spec['run_id']}" / "hipp_vertex_to_vertex_fc_active.npy"
                                for spec in RUN_SPECS
                            ],
                        ]
                        if need_intrinsic_fc
                        else []
                    ),
                ]
            )
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
                / f"sub-{subject}_hemi-L_space-corobl_den-{hipp_density}_label-hipp_network_cluster_{branch_tag}.label.gii",
                workbench_dir
                / smooth_name
                / "final"
                / f"sub-{subject}_hemi-R_space-corobl_den-{hipp_density}_label-hipp_network_cluster_{branch_tag}.label.gii",
                workbench_dir
                / smooth_name
                / "final"
                / f"sub-{subject}_space-corobl_den-{hipp_density}_label-hipp_network_cluster_{branch_tag}.dlabel.nii",
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
                valid_mask = left_valid_mask if hemi == "L" else right_valid_mask
                invalid_initial_mask = left_invalid_initial_mask if hemi == "L" else right_invalid_initial_mask
                active_mask = valid_mask
                adj_active = induced_subgraph(adj, np.flatnonzero(active_mask))
                shared_hemi_fc_dir = shared_fc_store_dir / smooth_name / f"hemi_{hemi}"
                network_fc_active = np.load(shared_hemi_fc_dir / "hipp_vertex_to_network_fc_active.npy").astype(np.float32)
                fc_summary = json.loads((shared_hemi_fc_dir / "fc_summary.json").read_text(encoding="utf-8"))
                network_fc_runs = [
                    np.load(shared_hemi_fc_dir / "runs" / f"run-{spec['run_id']}" / "hipp_vertex_to_network_fc_active.npy").astype(np.float32)
                    for spec in RUN_SPECS
                ]
                intrinsic_fc_active = None
                intrinsic_fc_runs = None
                if is_intrinsic_spectral_branch(branch_slug):
                    intrinsic_fc_active = np.load(
                        shared_hemi_fc_dir / "hipp_vertex_to_vertex_fc_active.npy"
                    ).astype(np.float32)
                    intrinsic_fc_runs = [
                        np.load(
                            shared_hemi_fc_dir
                            / "runs"
                            / f"run-{spec['run_id']}"
                            / "hipp_vertex_to_vertex_fc_active.npy"
                        ).astype(np.float32)
                        for spec in RUN_SPECS
                    ]
                run_labels = [f"run-{spec['run_id']}" for spec in RUN_SPECS]

                feature_dir = feature_root / smooth_name / f"hemi_{hemi}"
                clustering_dir = clustering_root / smooth_name / f"hemi_{hemi}"
                soft_dir = soft_root / smooth_name / f"hemi_{hemi}"

                if is_gradient_branch(branch_slug):
                    cluster = run_gradient_branch(
                        grouped_fc=network_fc_active,
                        run_grouped_fcs=network_fc_runs,
                        run_labels=run_labels,
                        networks=networks,
                        connectivity=adj_active,
                        feature_dir=feature_dir,
                        clustering_dir=clustering_dir,
                        hemi=hemi,
                        split_strategy=split_strategy,
                        instability_resamples=int(args.instability_resamples),
                        v_min_fraction=float(args.v_min_fraction),
                        v_min_count=None if args.v_min_count is None else int(args.v_min_count),
                        k_selection_mode=str(args.k_selection_mode),
                    )
                elif is_wta_branch(branch_slug):
                    cluster = run_wta_branch(
                        grouped_fc=network_fc_active,
                        networks=networks,
                        soft_dir=soft_dir,
                    )
                elif is_intrinsic_spectral_branch(branch_slug):
                    if intrinsic_fc_active is None or intrinsic_fc_runs is None:
                        raise RuntimeError(
                            f"Missing intrinsic FC features for {branch_slug} at {shared_hemi_fc_dir}"
                        )
                    cluster = run_intrinsic_spectral_branch(
                        intrinsic_fc=intrinsic_fc_active,
                        run_intrinsic_fcs=intrinsic_fc_runs,
                        grouped_fc_for_annotation=network_fc_active,
                        run_labels=run_labels,
                        networks=networks,
                        connectivity=adj_active,
                        feature_dir=feature_dir,
                        clustering_dir=clustering_dir,
                        hemi=hemi,
                        split_strategy=split_strategy,
                        instability_resamples=int(args.instability_resamples),
                        v_min_fraction=float(args.v_min_fraction),
                        v_min_count=None if args.v_min_count is None else int(args.v_min_count),
                        k_selection_mode=str(args.k_selection_mode),
                        zero_negative=uses_nonnegative_intrinsic_spectral_features(branch_slug),
                    )
                elif is_spectral_branch(branch_slug):
                    cluster = run_spectral_branch(
                        grouped_fc=network_fc_active,
                        run_grouped_fcs=network_fc_runs,
                        run_labels=run_labels,
                        networks=networks,
                        connectivity=adj_active,
                        feature_dir=feature_dir,
                        clustering_dir=clustering_dir,
                        hemi=hemi,
                        split_strategy=split_strategy,
                        instability_resamples=int(args.instability_resamples),
                        v_min_fraction=float(args.v_min_fraction),
                        v_min_count=None if args.v_min_count is None else int(args.v_min_count),
                        k_selection_mode=str(args.k_selection_mode),
                        zero_negative=uses_nonnegative_spectral_features(branch_slug),
                    )
                else:
                    cluster = run_probability_branch(
                        grouped_fc=network_fc_active,
                        run_grouped_fcs=network_fc_runs,
                        run_labels=run_labels,
                        networks=networks,
                        connectivity=adj_active,
                        surface_coords=(left_coords if hemi == "L" else right_coords)[active_mask],
                        feature_dir=feature_dir,
                        clustering_dir=clustering_dir,
                        soft_dir=soft_dir,
                        hemi=hemi,
                        save_soft_extras=is_soft_branch(branch_slug),
                        strict_soft_route=is_soft_branch(branch_slug),
                        zero_negative=uses_nonnegative_probabilities(branch_slug),
                        split_strategy=split_strategy,
                        instability_resamples=int(args.instability_resamples),
                        v_min_fraction=float(args.v_min_fraction),
                        v_min_count=None if args.v_min_count is None else int(args.v_min_count),
                        k_selection_mode=str(args.k_selection_mode),
                    )
                cluster["labels_final"] = expand_cluster_labels(cluster["labels_final"], active_mask, int(ts.shape[0]))
                for row in cluster["k_metrics"]:
                    k_value = int(row["k"])
                    k_dir = clustering_dir / f"k_{k_value}"
                    k_active_path = k_dir / "cluster_labels.npy"
                    if k_active_path.exists():
                        k_active_labels = np.load(k_active_path).astype(np.int32)
                        k_full_labels = expand_cluster_labels(k_active_labels, active_mask, int(ts.shape[0]))
                        np.save(k_dir / "cluster_labels_full.npy", k_full_labels.astype(np.int32))
                cluster["tsnr_gate"] = {
                    "n_vertices_total": int(ts.shape[0]),
                    "n_vertices_valid_high_tsnr": int(valid_mask.sum()),
                    "n_vertices_invalid_initial": int(invalid_initial_mask.sum()),
                    "n_vertices_clustered": int(active_mask.sum()),
                    "tsnr_threshold": float(TSNR_THRESHOLD),
                }
                cluster["fc_summary"] = fc_summary
                hemi_results[hemi] = cluster

            left_result = hemi_results["L"]
            right_result = hemi_results["R"]
            final_assets = save_combined_label_assets(
                subject=subject,
                density=hipp_density,
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
                    "fc_summary": node.get("fc_summary"),
                    "feature_summary": node["feature_summary"],
                    "k_metrics": node["k_metrics"],
                    "k_final": node["k_final"],
                    "selection_log": node.get("selection_log"),
                    "run_metadata": node.get("run_metadata"),
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
            left_surface=left_surface,
            right_surface=right_surface,
            spec_path=structural_spec,
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
                left_surface=Path(assets["left_surface"]),
                right_surface=Path(assets["right_surface"]),
                spec_path=Path(assets["left_surface"]).parent / f"sub-{subject}_den-{hipp_density}_surfaces.spec",
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
                "source_root": str(REPO_ROOT),
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
