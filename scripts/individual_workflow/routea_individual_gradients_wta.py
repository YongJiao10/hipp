#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import os
import sys
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, NamedTuple

import nibabel as nib
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
CORE_WORKFLOW_DIR = REPO_ROOT / "scripts" / "core_workflow"
if str(CORE_WORKFLOW_DIR) not in sys.path:
    sys.path.insert(0, str(CORE_WORKFLOW_DIR))

from lib.hippomaps_adapter import (  # noqa: E402
    DEFAULT_GRADIENT_RANDOM_STATE,
    DEFAULT_HIPPOMAPS_ROOT,
    ensure_hippomaps_on_path,
    fit_gradientmaps_pearson,
    hippomaps_commit,
    hippomaps_import_path,
    reconstruct_full_surface,
)
from lib.output_layout import (  # noqa: E402
    DEFAULT_DATASET,
    resolve_hippunfold_batch_root,
    resolve_kong2022_manifest_root,
    resolve_kong2022_routea_root,
    resolve_kong2022_surface_root,
    resolve_overviews_dir,
    resolve_work_root,
)
from lib.kong2022_label_utils import canonical_families_8_order, canonical_families_17_order  # noqa: E402
from lib.weighted_fisherz import (  # noqa: E402
    assign_weighted_network_profile_rows,
    assignment_status_from_weighted_network_profile,
    weighted_cluster_network_fisherz_profile_from_timeseries,
    weighted_network_fisherz_profile_from_timeseries,
)
from lib.workflow_utils import git_commit, utc_now_iso, utc_timestamp_compact_z  # noqa: E402


DEFAULT_SUBJECTS_MANIFEST = (
    resolve_kong2022_manifest_root() / "hcp_7t_hippocampus_fmri_qc_pass_t1only_latest.txt"
)
DEFAULT_KONG2022_ROUTEA_ROOT = resolve_kong2022_routea_root()
DEFAULT_KONG2022_SURFACE_ROOT = resolve_kong2022_surface_root()
DEFAULT_HIPPUNFOLD_BATCH_ROOT = resolve_hippunfold_batch_root()
DEFAULT_OUT_ROOT = resolve_work_root() / "individual_atlas"
DEFAULT_INDIVIDUAL_OVERVIEW_ROOT = resolve_overviews_dir() / "individual"
DEFAULT_SMOOTHINGS = ("2mm", "4mm")
DEFAULT_HEMIS = ("L", "R")
DEFAULT_JOBS = 4
DEFAULT_TOP_N_CAP = 20
DEFAULT_K = 3
THREAD_ENV_OVERRIDES = {
    "OMP_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1",
    "VECLIB_MAXIMUM_THREADS": "1",
}


class SubjectCombo(NamedTuple):
    subject: str
    smoothing: str
    hemi: str


def write_json(path: Path, payload: object, *, sort_keys: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=sort_keys) + "\n", encoding="utf-8")


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def normalize_subject(raw: str) -> str:
    text = str(raw).strip()
    if not text:
        raise ValueError("Empty subject id")
    return text.replace("sub-", "")


def load_subject_ids(path: Path) -> list[str]:
    subjects = [normalize_subject(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]
    if not subjects:
        raise RuntimeError(f"No subjects in manifest: {path}")
    return subjects


def iter_subject_combos(
    *,
    subjects: list[str],
    smoothings: list[str] | tuple[str, ...],
    hemis: list[str] | tuple[str, ...],
) -> list[SubjectCombo]:
    return [
        SubjectCombo(normalize_subject(subject), str(smoothing), str(hemi))
        for subject in subjects
        for smoothing in smoothings
        for hemi in hemis
    ]


def method_combo_output_dir(out_root: Path, method: str, combo: SubjectCombo) -> Path:
    method_dirs = {
        "gradient_spectral_k3": "gradient",
        "gradient_weighted_fisherz": "gradient",
        "wta_top": "wta",
        "wta_weighted_fisherz": "wta",
        "wta_normalized_top20_count": "wta",
        "wta_family17_normalized_top20_count": "wta",
    }
    if method not in method_dirs:
        raise ValueError(f"Unknown individual atlas method: {method!r}")
    return (
        Path(out_root)
        / f"sub-{combo.subject}"
        / method_dirs[method]
        / str(combo.smoothing)
        / str(combo.hemi)
    )


def gradient_combo_output_dir(out_root: Path, combo: SubjectCombo) -> Path:
    return method_combo_output_dir(out_root, "gradient_spectral_k3", combo)


def wta_combo_output_dir(out_root: Path, combo: SubjectCombo) -> Path:
    return method_combo_output_dir(out_root, "wta_top", combo)


OVERVIEW_METHODS: dict[str, dict[str, str]] = {
    "gradient_spectral_k3": {
        "task_dir": "gradient",
        "filename_stem": "gradient_fisherz",
        "label_filename": "cluster_labels_k3.npy",
        "title": "Individual gradient spectral K=3 fisher-z",
    },
    "gradient_count": {
        "task_dir": "gradient",
        "filename_stem": "gradient_count",
        "label_filename": "cluster_labels_k3.npy",
        "title": "Individual gradient spectral K=3 count",
    },
    "gradient_weighted_fisherz": {
        "task_dir": "gradient",
        "filename_stem": "gradient_weighted_fisherz",
        "label_filename": "group_functional_atlas_labels_weighted_fisherz.npy",
        "title": "Individual gradient spectral K=3 weighted fisher-z",
    },
    "wta_top": {
        "task_dir": "wta",
        "filename_stem": "wta",
        "label_filename": "wta_top_labels.npy",
        "title": "Individual WTA top",
    },
    "wta_weighted_fisherz": {
        "task_dir": "wta",
        "filename_stem": "wta_weighted_fisherz",
        "label_filename": "wta_top_labels_weighted_fisherz.npy",
        "title": "Individual WTA weighted fisher-z",
    },
    "wta_normalized_top20_count": {
        "task_dir": "wta",
        "filename_stem": "wta_normalized_top20_count",
        "label_filename": "wta_top_labels_normalized_top20_count.npy",
        "title": "Individual WTA normalized top20 count",
    },
    "wta_family17_normalized_top20_count": {
        "task_dir": "wta",
        "filename_stem": "wta_family17_normalized_top20_count",
        "label_filename": "wta_top_labels_family17_normalized_top20_count.npy",
        "title": "Individual WTA family17 normalized top20 count",
    },
}
OVERVIEW_METHOD_ALIASES = {
    "gradient": "gradient_spectral_k3",
    "gradient_spectral_k3": "gradient_spectral_k3",
    "gradient_count": "gradient_count",
    "count": "gradient_count",
    "gradient_weighted_fisherz": "gradient_weighted_fisherz",
    "weighted_fisherz": "gradient_weighted_fisherz",
    "wta": "wta_top",
    "wta_top": "wta_top",
    "wta_weighted_fisherz": "wta_weighted_fisherz",
    "wta_normalized_top20_count": "wta_normalized_top20_count",
    "normalized_top20_count": "wta_normalized_top20_count",
    "wta_family17_normalized_top20_count": "wta_family17_normalized_top20_count",
    "family17_normalized_top20_count": "wta_family17_normalized_top20_count",
}


def normalize_overview_method(method: str) -> str:
    key = str(method).strip()
    if key not in OVERVIEW_METHOD_ALIASES:
        raise ValueError(f"Unknown overview method: {method!r}")
    return OVERVIEW_METHOD_ALIASES[key]


def overview_output_path(overview_root: Path, method: str, subject: str, *, network_column: str = "canonical8") -> Path:
    method_key = normalize_overview_method(method)
    spec = OVERVIEW_METHODS[method_key]
    subject_id = normalize_subject(subject)
    stem = str(spec["filename_stem"])
    column = str(network_column)
    if column == "family17" and method_key in {"gradient_spectral_k3", "gradient_count", "gradient_weighted_fisherz"}:
        stem = stem.replace("gradient_", "gradient_family17_", 1)
    elif column != "canonical8":
        raise ValueError(f"Unsupported network column: {network_column!r}")
    return Path(overview_root) / spec["task_dir"] / f"sub-{subject_id}_{stem}_overview.png"


def overview_label_path(work_root: Path, method: str, combo: SubjectCombo) -> Path:
    method_key = normalize_overview_method(method)
    spec = OVERVIEW_METHODS[method_key]
    return method_combo_output_dir(Path(work_root), method_key, combo) / spec["label_filename"]


def combo_output_dir(out_root: Path, combo: SubjectCombo) -> Path:
    return gradient_combo_output_dir(out_root, combo)


def subject_surface_path(hippunfold_batch_root: Path, subject: str, hemi: str) -> Path:
    subject = normalize_subject(subject)
    return (
        Path(hippunfold_batch_root)
        / f"sub-{subject}"
        / "hippunfold"
        / f"sub-{subject}"
        / "surf"
        / f"sub-{subject}_hemi-{hemi}_space-corobl_den-512_label-hipp_midthickness.surf.gii"
    )


def subject_spec_path(surface_path: Path, subject: str) -> Path:
    subject = normalize_subject(subject)
    spec = Path(surface_path).parent / f"sub-{subject}_den-512_label-hipp_surfaces.spec"
    if spec.exists():
        return spec
    hemi_spec = Path(surface_path).parent / f"sub-{subject}_hemi-{Path(surface_path).name.split('_hemi-')[1][0]}_den-512_label-hipp_surfaces.spec"
    if hemi_spec.exists():
        return hemi_spec
    raise FileNotFoundError(f"Missing subject surface spec for sub-{subject}: {spec}")


def valid_mask_path(surface_root: Path, subject: str, hemi: str) -> Path:
    subject = normalize_subject(subject)
    return Path(surface_root) / f"sub-{subject}" / "surface" / "tsnr_gate" / f"sub-{subject}_hemi-{hemi}_valid_mask.npy"


def roi_fc_path(shared_root: Path, subject: str, smoothing: str, hemi: str) -> Path:
    subject = normalize_subject(subject)
    return (
        Path(shared_root)
        / f"sub-{subject}"
        / "fc"
        / "kong2022"
        / str(smoothing)
        / f"hemi_{hemi}"
        / "hipp_vertex_to_network_fc.npy"
    )


def load_core_group_atlas_module():
    module_name = "_individual_workflow_core_group_atlas"
    if module_name in sys.modules:
        return sys.modules[module_name]
    path = CORE_WORKFLOW_DIR / "05_02_run_group_atlas.py"
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load core group-atlas module: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def load_surface(path: Path) -> tuple[np.ndarray, np.ndarray]:
    core = load_core_group_atlas_module()
    return core.load_surface(Path(path))


def load_kong2022_roi_registry() -> list[dict[str, object]]:
    core = load_core_group_atlas_module()
    return [dict(row) for row in core.load_kong2022_roi_registry()]


def load_network_names_and_keys(roi_registry: list[dict[str, object]], network_column: str) -> tuple[list[str], dict[str, int]]:
    if network_column == "canonical8":
        expected = canonical_families_8_order()
    elif network_column == "family17":
        expected = canonical_families_17_order()
    else:
        raise ValueError(f"Unsupported network column: {network_column!r}")
    present = {str(row[network_column]) for row in roi_registry}
    missing = [name for name in expected if name not in present]
    extra = sorted(present - set(expected))
    if missing or extra:
        raise ValueError(
            f"Kong2022 {network_column} network set mismatch: missing={missing} extra={extra}"
        )
    names = list(expected)
    return names, {name: idx + 1 for idx, name in enumerate(names)}


def finite_fisher_z(fc: np.ndarray) -> np.ndarray:
    arr = np.asarray(fc, dtype=np.float32)
    clipped = np.clip(arr, -0.999999, 0.999999)
    out = np.arctanh(clipped).astype(np.float32)
    out[~np.isfinite(out)] = np.nan
    return out


def subject_wta_from_fc(
    *,
    fc_full: np.ndarray,
    valid_mask: np.ndarray,
    roi_registry: list[dict[str, object]],
    network_names: list[str],
    network_to_key: dict[str, int],
    network_column: str,
    top_n_cap: int,
) -> tuple[np.ndarray, list[dict[str, object]], dict[str, object]]:
    fc = np.asarray(fc_full, dtype=np.float32)
    valid = np.asarray(valid_mask, dtype=bool).reshape(-1)
    if fc.ndim != 2:
        raise ValueError(f"Expected 2D FC matrix, got shape={fc.shape}")
    if fc.shape[0] != valid.size:
        raise ValueError(f"FC/mask vertex mismatch: fc_rows={fc.shape[0]} mask={valid.size}")
    if fc.shape[1] != len(roi_registry):
        raise ValueError(f"FC/ROI registry mismatch: fc_cols={fc.shape[1]} registry={len(roi_registry)}")

    unassigned_key = int(len(network_names) + 1)
    labels = np.zeros((valid.size,), dtype=np.int32)
    rows: list[dict[str, object]] = []
    for vertex_index in np.flatnonzero(valid):
        values = fc[int(vertex_index), :]
        pos_idx = np.flatnonzero(np.isfinite(values) & (values > 0.0))
        if pos_idx.size == 0:
            labels[int(vertex_index)] = unassigned_key
            rows.append(
                {
                    "vertex_index": int(vertex_index),
                    "status": "unassigned",
                    "unassigned_reason": "no_positive_roi",
                    "top1_network": "unassigned",
                    "top2_network": "unassigned",
                    "top1_count": 0,
                    "top2_count": 0,
                    "margin_count": 0,
                    "n_top": 0,
                }
            )
            continue

        ordered = pos_idx[np.argsort(-values[pos_idx])]
        top_idx = ordered[: int(min(int(top_n_cap), int(ordered.size)))]
        count_by_network: dict[str, int] = {}
        score_by_network: dict[str, float] = {}
        for col in top_idx:
            net = str(roi_registry[int(col)][network_column])
            count_by_network[net] = int(count_by_network.get(net, 0) + 1)
            score_by_network[net] = float(score_by_network.get(net, 0.0) + float(values[int(col)]))
        ranked = sorted(
            (
                (int(count_by_network[name]), float(score_by_network[name]), str(name))
                for name in count_by_network
            ),
            key=lambda item: (-item[0], -item[1], item[2]),
        )
        top1_count, top1_score, top1_network = ranked[0]
        if len(ranked) > 1:
            top2_count, top2_score, top2_network = ranked[1]
        else:
            top2_count, top2_score, top2_network = 0, 0.0, "none"
        labels[int(vertex_index)] = int(network_to_key.get(str(top1_network), unassigned_key))
        rows.append(
            {
                "vertex_index": int(vertex_index),
                "status": "dominant",
                "top1_network": str(top1_network),
                "top2_network": str(top2_network),
                "top1_count": int(top1_count),
                "top2_count": int(top2_count),
                "margin_count": int(top1_count - top2_count),
                "top1_score_sum": float(top1_score),
                "top2_score_sum": float(top2_score),
                "n_top": int(top_idx.size),
                "top1_fraction": float(top1_count / top_idx.size) if top_idx.size else 0.0,
                "top2_fraction": float(top2_count / top_idx.size) if top_idx.size else 0.0,
            }
        )
    meta = {
        "assignment_mode": "vertex_level",
        "assignment_definition": "single_subject_top_positive_roi_network_vote",
        "statistical_inference": False,
        "network_column": str(network_column),
        "top_n_cap": int(top_n_cap),
        "network_names": [str(x) for x in network_names],
        "network_label_keys": {str(k): int(v) for k, v in network_to_key.items()},
        "unassigned_label_key": int(unassigned_key),
    }
    return labels.astype(np.int32), rows, meta


def label_items_from_networks(network_names: list[str], color_map: dict[str, tuple[int, int, int, int]]) -> list[tuple[int, str, tuple[int, int, int, int]]]:
    items: list[tuple[int, str, tuple[int, int, int, int]]] = [(0, "Null", (158, 158, 158, 255))]
    for idx, name in enumerate(network_names, start=1):
        if name not in color_map:
            raise KeyError(f"Missing network color for {name!r}")
        items.append((int(idx), str(name), color_map[name]))
    items.append((len(network_names) + 1, "unassigned", (20, 20, 20, 255)))
    return items


def write_label_gifti(path: Path, labels: np.ndarray, label_items: list[tuple[int, str, tuple[int, int, int, int]]]) -> None:
    core = load_core_group_atlas_module()
    path.parent.mkdir(parents=True, exist_ok=True)
    nib.save(core.make_named_label_gifti(labels.astype(np.int32), label_items), str(path))


def read_kong2022_roi_weight_rows(path: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    with Path(path).open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f, delimiter="	")
        for row in reader:
            rows.append(
                {
                    "parcel_id": int(row["parcel_id"]),
                    "parcel_name": str(row["parcel_name"]),
                    "family17": str(row["family17"]),
                    "canonical8": str(row["canonical8"]),
                    "hemisphere": str(row.get("hemisphere", "")),
                    "subregion": str(row.get("subregion", "")),
                    "rank": int(row.get("rank", "0") or 0),
                    "n_grayordinates_used": int(row.get("n_grayordinates_used", "0") or 0),
                }
            )
    if not rows:
        raise RuntimeError(f"No ROI rows found: {path}")
    return rows


def write_weighted_fisherz_outputs(
    *,
    gradient_dir: Path,
    wta_dir: Path,
    subject: str,
    smoothing: str,
    hemi: str,
    labels_k3: np.ndarray,
    valid_mask: np.ndarray,
    hippocampal_timeseries: np.ndarray,
    roi_timeseries: np.ndarray,
    roi_rows: list[dict[str, object]],
    network_names: list[str],
    network_column: str,
    color_map: dict[str, tuple[int, int, int, int]],
) -> dict[str, object]:
    core = load_core_group_atlas_module()
    valid = np.asarray(valid_mask, dtype=bool).reshape(-1)
    labels = np.asarray(labels_k3, dtype=np.int32).reshape(-1)

    cluster_profile = weighted_cluster_network_fisherz_profile_from_timeseries(
        group_labels_final=labels,
        support_mask=valid,
        hippocampal_timeseries=np.asarray(hippocampal_timeseries, dtype=np.float32),
        roi_timeseries=np.asarray(roi_timeseries, dtype=np.float32),
        roi_rows=roi_rows,
        network_names=network_names,
        network_column=network_column,
    )
    cluster_rows, dominant = assignment_status_from_weighted_network_profile(cluster_profile, network_names)
    functional_labels, functional_meta, label_items = core.functional_atlas_from_clusters(
        group_labels_final=labels,
        status_rows=cluster_rows,
        network_names=network_names,
        color_map=color_map,
    )
    functional_meta["assignment_variant"] = "weighted_fisherz"
    functional_meta["network_column"] = network_column
    np.save(gradient_dir / "cluster_network_profile_weighted_fisherz.npy", cluster_profile.astype(np.float32))
    np.save(gradient_dir / "group_functional_atlas_labels_weighted_fisherz.npy", functional_labels.astype(np.int32))
    write_json(gradient_dir / "cluster_assignment_status_weighted_fisherz.json", {"items": cluster_rows, "assignment_metric": "top1_weighted_network_fisher_z", "network_column": network_column})
    write_json(gradient_dir / "cluster_dominant_network_weighted_fisherz.json", {"dominant": dominant, "network_names": network_names, "assignment_variant": "weighted_fisherz", "network_column": network_column})
    write_json(gradient_dir / "group_functional_atlas_label_map_weighted_fisherz.json", functional_meta)
    write_label_gifti(
        gradient_dir / "workbench_assets" / f"sub-{subject}_hemi-{hemi}_space-corobl_den-512_label-individual_gradient_spectral_k3_{smoothing}_weighted_fisherz.label.gii",
        functional_labels,
        label_items,
    )

    vertex_profile = weighted_network_fisherz_profile_from_timeseries(
        hippocampal_timeseries=np.asarray(hippocampal_timeseries, dtype=np.float32),
        roi_timeseries=np.asarray(roi_timeseries, dtype=np.float32),
        roi_rows=roi_rows,
        network_names=network_names,
        network_column=network_column,
    )
    support_indices = np.flatnonzero(valid).astype(np.int32)
    vertex_rows = assign_weighted_network_profile_rows(
        profile=vertex_profile[support_indices, :],
        support_indices=support_indices,
        network_names=network_names,
    )
    wta_labels, wta_meta, wta_label_items = core.functional_atlas_from_vertices(
        n_vertices=int(valid.size),
        status_rows=vertex_rows,
        network_names=network_names,
        color_map=color_map,
    )
    wta_meta["assignment_variant"] = "weighted_fisherz"
    wta_meta["network_column"] = network_column
    np.save(wta_dir / "vertex_network_profile_weighted_fisherz.npy", vertex_profile.astype(np.float32))
    np.save(wta_dir / "wta_top_labels_weighted_fisherz.npy", wta_labels.astype(np.int32))
    write_json(wta_dir / "vertex_assignment_status_weighted_fisherz.json", {"items": vertex_rows, "assignment_metric": "top1_weighted_network_fisher_z", "network_column": network_column})
    write_json(wta_dir / "wta_top_label_map_weighted_fisherz.json", wta_meta)
    write_label_gifti(
        wta_dir / "workbench_assets" / f"sub-{subject}_hemi-{hemi}_space-corobl_den-512_label-individual_wta_top_{smoothing}_weighted_fisherz.label.gii",
        wta_labels,
        wta_label_items,
    )
    return {
        "gradient_cluster_network_profile": str((gradient_dir / "cluster_network_profile_weighted_fisherz.npy").resolve()),
        "gradient_labels": str((gradient_dir / "group_functional_atlas_labels_weighted_fisherz.npy").resolve()),
        "wta_vertex_network_profile": str((wta_dir / "vertex_network_profile_weighted_fisherz.npy").resolve()),
        "wta_labels": str((wta_dir / "wta_top_labels_weighted_fisherz.npy").resolve()),
    }


def write_normalized_wta_outputs(
    *,
    wta_dir: Path,
    subject: str,
    smoothing: str,
    hemi: str,
    fc_full_fisherz: np.ndarray,
    valid_mask: np.ndarray,
    roi_registry: list[dict[str, object]],
    network_column: str,
    filename_suffix: str,
    top_n_cap: int,
    color_map: dict[str, tuple[int, int, int, int]],
) -> dict[str, object]:
    core = load_core_group_atlas_module()
    column = core.resolve_kong2022_network_registry_column(network_column)
    support_indices = np.flatnonzero(np.asarray(valid_mask, dtype=bool)).astype(np.int32)
    profile = np.asarray(fc_full_fisherz, dtype=np.float32)[support_indices, :]
    rows = core.build_normalized_topn_count_vertex_rows(
        mean_profile=profile,
        support_indices=support_indices,
        roi_registry=roi_registry,
        top_n_cap=int(top_n_cap),
        network_column=column,
    )
    network_names, _network_to_key = load_network_names_and_keys(roi_registry, column)
    labels, label_map, label_items = core.functional_atlas_from_vertices(
        n_vertices=int(np.asarray(valid_mask).size),
        status_rows=rows,
        network_names=network_names,
        color_map=color_map,
    )
    np.save(wta_dir / f"wta_top_labels_{filename_suffix}.npy", labels.astype(np.int32))
    write_json(
        wta_dir / f"vertex_assignment_status_{filename_suffix}.json",
        {
            "items": rows,
            "assignment_metric": "top20_count_normalized_by_network_roi_count",
            "normalization": "top20_count_divided_by_finite_available_roi_count",
            "network_column": column,
            "top_n_cap": int(top_n_cap),
        },
        sort_keys=True,
    )
    write_json(wta_dir / f"wta_top_label_map_{filename_suffix}.json", label_map, sort_keys=True)
    label_path = (
        wta_dir
        / "workbench_assets"
        / f"sub-{subject}_hemi-{hemi}_space-corobl_den-512_label-individual_wta_top_{smoothing}_{filename_suffix}.label.gii"
    )
    write_label_gifti(label_path, labels, label_items)
    return {
        "labels": str((wta_dir / f"wta_top_labels_{filename_suffix}.npy").resolve()),
        "status": str((wta_dir / f"vertex_assignment_status_{filename_suffix}.json").resolve()),
        "label_map": str((wta_dir / f"wta_top_label_map_{filename_suffix}.json").resolve()),
        "label_gifti": str(label_path.resolve()),
        "network_column": column,
        "assignment_metric": "top20_count_normalized_by_network_roi_count",
    }


def compute_individual_gradients(
    *,
    fc_full_fisherz: np.ndarray,
    valid_mask: np.ndarray,
    n_components: int,
    gradient_random_state: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    valid = np.asarray(valid_mask, dtype=bool).reshape(-1)
    active_fc = np.asarray(fc_full_fisherz, dtype=np.float32)[valid, :]
    active_fc[~np.isfinite(active_fc)] = 0.0
    gradients_active, lambdas, ev_fractions = fit_gradientmaps_pearson(
        active_fc,
        n_components=int(n_components),
        random_state=int(gradient_random_state),
    )
    gradients_full = reconstruct_full_surface(gradients_active, valid, valid.size)
    return (
        gradients_active.astype(np.float32),
        gradients_full.astype(np.float32),
        lambdas.astype(np.float64),
        ev_fractions.astype(np.float64),
    )


def cluster_individual_k3(
    *,
    gradients_active: np.ndarray,
    valid_mask: np.ndarray,
    surface_path: Path,
    seed: int,
) -> np.ndarray:
    core = load_core_group_atlas_module()
    coords, faces = core.load_surface(Path(surface_path))
    adjacency = core.build_surface_adjacency(faces, int(coords.shape[0]))
    labels_active, labels_full = core.cluster_from_precomputed_group_features(
        group_features=np.asarray(gradients_active[:, :3], dtype=np.float32),
        support_mask=np.asarray(valid_mask, dtype=bool),
        adjacency_full=adjacency,
        k=DEFAULT_K,
        seed=int(seed),
    )
    axis_order = core.long_axis_order(Path(surface_path))
    return core.reorder_labels_by_long_axis(labels_full.astype(np.int32), axis_order).astype(np.int32)


def process_combo(config: dict[str, object], combo_dict: dict[str, str]) -> dict[str, object]:
    for key, value in THREAD_ENV_OVERRIDES.items():
        os.environ[key] = value

    combo = SubjectCombo(
        subject=normalize_subject(combo_dict["subject"]),
        smoothing=str(combo_dict["smoothing"]),
        hemi=str(combo_dict["hemi"]),
    )
    out_root = Path(str(config["out_root"]))
    gradient_run_dir = gradient_combo_output_dir(out_root, combo)
    wta_run_dir = wta_combo_output_dir(out_root, combo)
    manifest_path = gradient_run_dir / "combo_manifest.json"
    wta_manifest_path = wta_run_dir / "combo_manifest.json"
    if bool(config.get("resume", False)) and manifest_path.exists() and wta_manifest_path.exists():
        old = read_json(manifest_path)
        old_wta = read_json(wta_manifest_path)
        if str(old.get("status")) == "ok" and str(old_wta.get("status")) == "ok":
            return {"status": "ok", "subject": combo.subject, "smoothing": combo.smoothing, "hemi": combo.hemi, "resumed": True}

    started = time.time()
    try:
        hippomaps_root = ensure_hippomaps_on_path(Path(str(config["hippomaps_root"])))
        shared_root = Path(str(config["shared_root"]))
        surface_root = Path(str(config["surface_root"]))
        hippunfold_batch_root = Path(str(config["hippunfold_batch_root"]))
        roi_registry = load_kong2022_roi_registry()
        network_column = str(config.get("network_column", "canonical8"))
        network_names, network_to_key = load_network_names_and_keys(roi_registry, network_column)
        core = load_core_group_atlas_module()
        color_map = core.load_canonical_network_color_map()

        fc_path = roi_fc_path(shared_root, combo.subject, combo.smoothing, combo.hemi)
        mask_path = valid_mask_path(surface_root, combo.subject, combo.hemi)
        surf_path = subject_surface_path(hippunfold_batch_root, combo.subject, combo.hemi)
        spec_path = subject_spec_path(surf_path, combo.subject)
        for required in (fc_path, mask_path, surf_path, spec_path):
            if not required.exists():
                raise FileNotFoundError(f"Missing required input: {required}")

        fc_full_raw = np.load(fc_path).astype(np.float32)
        valid_mask = np.load(mask_path).astype(bool)
        if fc_full_raw.shape[0] != valid_mask.size:
            raise ValueError(f"FC/mask vertex mismatch for {combo}: {fc_full_raw.shape[0]} vs {valid_mask.size}")
        fc_full_fisherz = finite_fisher_z(fc_full_raw)

        gradients_active, gradients_full, lambdas, ev_fractions = compute_individual_gradients(
            fc_full_fisherz=fc_full_fisherz,
            valid_mask=valid_mask,
            n_components=int(config["n_components"]),
            gradient_random_state=int(config["gradient_random_state"]),
        )
        gradient_dir = gradient_run_dir / "gradients"
        gradient_dir.mkdir(parents=True, exist_ok=True)
        np.save(gradient_dir / "support_mask.npy", valid_mask.astype(np.uint8))
        np.save(gradient_dir / "hippomaps_gradients_active.npy", gradients_active.astype(np.float32))
        np.save(gradient_dir / "hippomaps_gradients_full.npy", gradients_full.astype(np.float32))
        np.save(gradient_dir / "hippomaps_lambdas.npy", lambdas.astype(np.float64))
        np.save(gradient_dir / "hippomaps_ev_fractions.npy", ev_fractions.astype(np.float64))

        labels_k3 = cluster_individual_k3(
            gradients_active=gradients_active,
            valid_mask=valid_mask,
            surface_path=surf_path,
            seed=int(config["seed"]),
        )
        k3_dir = gradient_run_dir
        k3_dir.mkdir(parents=True, exist_ok=True)
        np.save(k3_dir / "cluster_labels_k3.npy", labels_k3.astype(np.int32))
        cluster_lbl_path = (
            k3_dir
            / "workbench_assets"
            / f"sub-{combo.subject}_hemi-{combo.hemi}_space-corobl_den-512_label-individual_gradient_spectral_k3_{combo.smoothing}.label.gii"
        )
        cluster_lbl_path.parent.mkdir(parents=True, exist_ok=True)
        nib.save(
            core.make_cluster_label_gifti(labels_k3.astype(np.int32), DEFAULT_K, core.load_cluster_palette()),
            str(cluster_lbl_path),
        )

        wta_labels, wta_rows, wta_meta = subject_wta_from_fc(
            fc_full=fc_full_fisherz,
            valid_mask=valid_mask,
            roi_registry=roi_registry,
            network_names=network_names,
            network_to_key=network_to_key,
            network_column=network_column,
            top_n_cap=int(config["top_n_cap"]),
        )
        wta_dir = wta_run_dir
        wta_dir.mkdir(parents=True, exist_ok=True)
        np.save(wta_dir / "wta_top_labels.npy", wta_labels.astype(np.int32))
        write_json(wta_dir / "vertex_assignment_status.json", {"items": wta_rows})
        write_json(wta_dir / "wta_top_label_map.json", wta_meta)
        wta_lbl_path = (
            wta_dir
            / "workbench_assets"
            / f"sub-{combo.subject}_hemi-{combo.hemi}_space-corobl_den-512_label-individual_wta_top_{combo.smoothing}.label.gii"
        )
        write_label_gifti(wta_lbl_path, wta_labels, label_items_from_networks(network_names, color_map))
        normalized_wta_outputs = {
            "normalized_top20_count": write_normalized_wta_outputs(
                wta_dir=wta_dir,
                subject=combo.subject,
                smoothing=combo.smoothing,
                hemi=combo.hemi,
                fc_full_fisherz=fc_full_fisherz,
                valid_mask=valid_mask,
                roi_registry=roi_registry,
                network_column="canonical8",
                filename_suffix="normalized_top20_count",
                top_n_cap=int(config["top_n_cap"]),
                color_map=color_map,
            ),
            "family17_normalized_top20_count": write_normalized_wta_outputs(
                wta_dir=wta_dir,
                subject=combo.subject,
                smoothing=combo.smoothing,
                hemi=combo.hemi,
                fc_full_fisherz=fc_full_fisherz,
                valid_mask=valid_mask,
                roi_registry=roi_registry,
                network_column="family17",
                filename_suffix="family17_normalized_top20_count",
                top_n_cap=int(config["top_n_cap"]),
                color_map=color_map,
            ),
        }
        reference_root = Path(str(config["shared_root"])) / f"sub-{combo.subject}" / "reference" / "kong2022"
        weighted_outputs = write_weighted_fisherz_outputs(
            gradient_dir=k3_dir,
            wta_dir=wta_dir,
            subject=combo.subject,
            smoothing=combo.smoothing,
            hemi=combo.hemi,
            labels_k3=labels_k3,
            valid_mask=valid_mask,
            hippocampal_timeseries=np.load(Path(str(config["surface_root"])) / f"sub-{combo.subject}" / "surface" / combo.smoothing / f"sub-{combo.subject}_hemi-{combo.hemi}_timeseries.npy").astype(np.float32),
            roi_timeseries=np.load(reference_root / "cortex_roi_parcel_timeseries.npy").astype(np.float32),
            roi_rows=read_kong2022_roi_weight_rows(reference_root / "cortex_roi_parcels.tsv"),
            network_names=network_names,
            network_column=network_column,
            color_map=color_map,
        )

        elapsed = float(time.time() - started)
        manifest = {
            "status": "ok",
            "workflow": "routea_individual_gradients_and_wta",
            "route": "routea",
            "subject": f"sub-{combo.subject}",
            "smoothing": combo.smoothing,
            "hemi": combo.hemi,
            "method_outputs": ["gradient", "gradient-spectral-k3", "wta-top", "wta-normalized-top20-count", "wta-family17-normalized-top20-count"],
            "individual_level": True,
            "n_subjects": 1,
            "n_vertices": int(valid_mask.size),
            "n_support_vertices": int(np.sum(valid_mask)),
            "gradient_components": ["G1", "G2", "G3"],
            "gradient_random_state": int(config["gradient_random_state"]),
            "k_final": DEFAULT_K,
            "wta_assignment_definition": wta_meta["assignment_definition"],
            "wta_statistical_inference": False,
            "inputs": {
                "fc": str(fc_path.resolve()),
                "valid_mask": str(mask_path.resolve()),
                "surface": str(surf_path.resolve()),
                "spec": str(spec_path.resolve()),
            },
            "outputs": {
                "gradient_dir": str(gradient_dir.resolve()),
                "gradient_spectral_k3_dir": str(k3_dir.resolve()),
                "wta_top_dir": str(wta_dir.resolve()),
                "cluster_label_gifti": str(cluster_lbl_path.resolve()),
                "wta_label_gifti": str(wta_lbl_path.resolve()),
                "wta_normalized_top20_count": normalized_wta_outputs["normalized_top20_count"],
                "wta_family17_normalized_top20_count": normalized_wta_outputs["family17_normalized_top20_count"],
                "weighted_fisherz": weighted_outputs,
            },
            "hippomaps_repo": str(Path(hippomaps_root).resolve()),
            "hippomaps_import_path": hippomaps_import_path(hippomaps_root),
            "hippomaps_commit": hippomaps_commit(hippomaps_root),
            "git_commit": git_commit(short=True),
            "started_at": utc_now_iso(zulu=True),
            "elapsed_seconds": elapsed,
        }
        write_json(manifest_path, manifest, sort_keys=True)
        wta_manifest = dict(manifest)
        wta_manifest["method_outputs"] = ["wta-top"]
        wta_manifest["outputs"] = {
            "wta_top_dir": str(wta_dir.resolve()),
            "wta_label_gifti": str(wta_lbl_path.resolve()),
            "wta_normalized_top20_count": normalized_wta_outputs["normalized_top20_count"],
            "wta_family17_normalized_top20_count": normalized_wta_outputs["family17_normalized_top20_count"],
            "weighted_fisherz": weighted_outputs,
        }
        write_json(wta_manifest_path, wta_manifest, sort_keys=True)
        return {
            "status": "ok",
            "subject": combo.subject,
            "smoothing": combo.smoothing,
            "hemi": combo.hemi,
            "elapsed_seconds": elapsed,
            "manifest": str(manifest_path),
        }
    except Exception as exc:
        elapsed = float(time.time() - started)
        failure = {
            "status": "failed",
            "workflow": "routea_individual_gradients_and_wta",
            "subject": f"sub-{combo.subject}",
            "smoothing": combo.smoothing,
            "hemi": combo.hemi,
            "error_type": exc.__class__.__name__,
            "error": str(exc),
            "traceback": traceback.format_exc(),
            "elapsed_seconds": elapsed,
        }
        write_json(manifest_path, failure, sort_keys=True)
        write_json(wta_manifest_path, failure, sort_keys=True)
        return {
            "status": "failed",
            "subject": combo.subject,
            "smoothing": combo.smoothing,
            "hemi": combo.hemi,
            "error": str(exc),
            "manifest": str(manifest_path),
        }


def manifest_rows(out_root: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for path in sorted(Path(out_root).glob("sub-*/gradient/*/*/combo_manifest.json")):
        try:
            payload = read_json(path)
        except Exception as exc:
            payload = {"status": "unreadable", "error": str(exc)}
        payload["manifest_path"] = str(path)
        rows.append(payload)
    return rows


def write_status_tsv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = ["status", "subject", "smoothing", "hemi", "elapsed_seconds", "error_type", "error", "manifest_path"]
    lines = ["\t".join(fields)]
    for row in rows:
        subject = str(row.get("subject", "")).replace("sub-", "")
        values = {
            "status": row.get("status", ""),
            "subject": subject,
            "smoothing": row.get("smoothing", ""),
            "hemi": row.get("hemi", ""),
            "elapsed_seconds": row.get("elapsed_seconds", ""),
            "error_type": row.get("error_type", ""),
            "error": row.get("error", ""),
            "manifest_path": row.get("manifest_path", ""),
        }
        lines.append("\t".join(str(values[f]).replace("\t", " ").replace("\n", " ") for f in fields))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_monitor_summary(
    *,
    out_root: Path,
    heartbeat_path: Path,
    stale_seconds: int = 120,
    now_epoch: float | None = None,
) -> dict[str, object]:
    rows = manifest_rows(out_root)
    counts: dict[str, int] = {}
    for row in rows:
        status = str(row.get("status", "unknown"))
        counts[status] = int(counts.get(status, 0) + 1)
    heartbeat: dict[str, object] = {}
    if Path(heartbeat_path).exists():
        heartbeat = read_json(Path(heartbeat_path))
    now = float(time.time() if now_epoch is None else now_epoch)
    updated = float(heartbeat.get("updated_at_epoch", 0.0) or 0.0)
    heartbeat_status = str(heartbeat.get("status", "missing_heartbeat"))
    active_heartbeat_states = {"starting", "running"}
    heartbeat_stale = bool(
        heartbeat_status in active_heartbeat_states
        and (updated <= 0.0 or (now - updated) > float(stale_seconds))
    )
    return {
        "status": heartbeat_status,
        "total_manifests": int(len(rows)),
        "ok": int(counts.get("ok", 0)),
        "failed": int(counts.get("failed", 0)),
        "skipped": int(counts.get("skipped", 0)),
        "unreadable": int(counts.get("unreadable", 0)),
        "running": int(heartbeat.get("active", 0) or 0),
        "heartbeat_path": str(heartbeat_path),
        "heartbeat_stale": heartbeat_stale,
        "seconds_since_heartbeat": float(now - updated) if updated > 0 else None,
        "pid": heartbeat.get("pid"),
    }


def write_handoff(
    *,
    out_root: Path,
    run_id: str,
    command: list[str],
    heartbeat_path: Path,
    total_combos: int,
    jobs: int,
) -> Path:
    handoff = Path(out_root) / "_runs" / "handoff" / f"{run_id}_handoff.md"
    summary = build_monitor_summary(out_root=out_root, heartbeat_path=heartbeat_path, stale_seconds=120)
    body = f"""# RouteA Individual Gradients + WTA Handoff

Run ID: `{run_id}`

Command:

```bash
{" ".join(command)}
```

Runtime:
- PID: `{os.getpid()}`
- Jobs: `{jobs}`
- Total planned combos: `{total_combos}`
- Output root: `{Path(out_root).resolve()}`
- Heartbeat: `{heartbeat_path.resolve()}`
- Status TSV: `{(Path(out_root) / "_runs" / "summaries" / "combo_status.tsv").resolve()}`

Current summary:

```json
{json.dumps(summary, indent=2)}
```

Monitor:

```bash
/home/yojiao/.conda/envs/py314/bin/python scripts/individual_workflow/01_monitor_individual_run.py --out-root {Path(out_root)}
```

Resume:

```bash
/home/yojiao/.conda/envs/py314/bin/python scripts/individual_workflow/routea_individual_gradients_wta.py --resume --jobs {jobs}
```
"""
    handoff.parent.mkdir(parents=True, exist_ok=True)
    handoff.write_text(body, encoding="utf-8")
    return handoff


def run_supervisor(args: argparse.Namespace) -> int:
    for key, value in THREAD_ENV_OVERRIDES.items():
        os.environ[key] = value
    out_root = Path(args.out_root).resolve()
    run_id = str(args.run_id or f"routea_individual_gradients_wta_{utc_timestamp_compact_z()}")
    log_dir = out_root / "_runs" / "logs" / run_id
    summary_dir = out_root / "_runs" / "summaries"
    log_dir.mkdir(parents=True, exist_ok=True)
    summary_dir.mkdir(parents=True, exist_ok=True)
    heartbeat_path = log_dir / "heartbeat.json"
    pid_path = log_dir / "supervisor.pid"
    pid_path.write_text(str(os.getpid()) + "\n", encoding="utf-8")

    subjects = [normalize_subject(s) for s in args.subjects] if args.subjects else load_subject_ids(Path(args.subjects_manifest))
    if args.max_subjects is not None:
        subjects = subjects[: int(args.max_subjects)]
    combos = iter_subject_combos(subjects=subjects, smoothings=list(args.smoothings), hemis=list(args.hemis))
    jobs = max(1, min(int(args.jobs), 10, len(combos)))
    config: dict[str, object] = {
        "out_root": str(out_root),
        "shared_root": str(Path(args.shared_root).resolve()),
        "surface_root": str(Path(args.surface_root).resolve()),
        "hippunfold_batch_root": str(Path(args.hippunfold_batch_root).resolve()),
        "hippomaps_root": str(Path(args.hippomaps_root).resolve()),
        "n_components": int(args.n_components),
        "gradient_random_state": int(args.gradient_random_state),
        "seed": int(args.seed),
        "top_n_cap": int(args.top_n_cap),
        "network_column": str(args.network_column),
        "resume": bool(args.resume),
    }
    command = [sys.executable, str(Path(__file__).resolve()), *sys.argv[1:]]
    write_json(
        heartbeat_path,
        {
            "status": "starting",
            "run_id": run_id,
            "pid": os.getpid(),
            "updated_at": utc_now_iso(zulu=True),
            "updated_at_epoch": time.time(),
            "active": 0,
            "done": 0,
            "failed": 0,
            "total": len(combos),
        },
    )
    write_handoff(
        out_root=out_root,
        run_id=run_id,
        command=command,
        heartbeat_path=heartbeat_path,
        total_combos=len(combos),
        jobs=jobs,
    )

    if args.dry_run:
        write_json(
            summary_dir / "run_manifest.json",
            {
                "status": "dry_run",
                "run_id": run_id,
                "jobs": jobs,
                "n_subjects": len(subjects),
                "n_combos": len(combos),
                "combos_preview": [combo._asdict() for combo in combos[:10]],
            },
        )
        return 0

    done = 0
    failed = 0
    active = 0
    with ProcessPoolExecutor(max_workers=jobs) as executor:
        future_to_combo = {
            executor.submit(process_combo, config, combo._asdict()): combo
            for combo in combos
        }
        active = len(future_to_combo)
        for future in as_completed(future_to_combo):
            combo = future_to_combo[future]
            result: dict[str, object]
            try:
                result = future.result()
            except Exception as exc:
                result = {
                    "status": "failed",
                    "subject": combo.subject,
                    "smoothing": combo.smoothing,
                    "hemi": combo.hemi,
                    "error": str(exc),
                }
            done += 1
            if str(result.get("status")) == "failed":
                failed += 1
            active = max(0, len(future_to_combo) - done)
            rows = manifest_rows(out_root)
            write_status_tsv(summary_dir / "combo_status.tsv", rows)
            write_json(
                heartbeat_path,
                {
                    "status": "running" if done < len(combos) else ("failed" if failed else "ok"),
                    "run_id": run_id,
                    "pid": os.getpid(),
                    "updated_at": utc_now_iso(zulu=True),
                    "updated_at_epoch": time.time(),
                    "active": active,
                    "done": done,
                    "failed": failed,
                    "total": len(combos),
                    "last_result": result,
                },
            )
            if failed and bool(args.stop_on_first_failure):
                break

    rows = manifest_rows(out_root)
    write_status_tsv(summary_dir / "combo_status.tsv", rows)
    final_status = "failed" if failed else "ok"
    run_manifest = {
        "status": final_status,
        "run_id": run_id,
        "jobs": jobs,
        "n_subjects": len(subjects),
        "n_combos": len(combos),
        "done": done,
        "failed": failed,
        "out_root": str(out_root),
        "heartbeat": str(heartbeat_path),
        "pid_file": str(pid_path),
        "git_commit": git_commit(short=True),
    }
    write_json(summary_dir / "run_manifest.json", run_manifest, sort_keys=True)
    write_json(
        heartbeat_path,
        {
            **run_manifest,
            "updated_at": utc_now_iso(zulu=True),
            "updated_at_epoch": time.time(),
            "active": 0,
        },
        sort_keys=True,
    )
    write_handoff(
        out_root=out_root,
        run_id=run_id,
        command=command,
        heartbeat_path=heartbeat_path,
        total_combos=len(combos),
        jobs=jobs,
    )
    return 1 if failed else 0


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run RouteA individual-level G1-G3 gradient-spectral K=3 and WTA-top.")
    parser.add_argument("--subjects-manifest", default=str(DEFAULT_SUBJECTS_MANIFEST))
    parser.add_argument("--subjects", nargs="*", default=None)
    parser.add_argument("--max-subjects", type=int, default=None)
    parser.add_argument("--shared-root", default=str(DEFAULT_KONG2022_ROUTEA_ROOT))
    parser.add_argument("--surface-root", default=str(DEFAULT_KONG2022_SURFACE_ROOT))
    parser.add_argument("--hippunfold-batch-root", default=str(DEFAULT_HIPPUNFOLD_BATCH_ROOT))
    parser.add_argument("--hippomaps-root", default=str(DEFAULT_HIPPOMAPS_ROOT))
    parser.add_argument("--out-root", default=str(DEFAULT_OUT_ROOT))
    parser.add_argument("--smoothings", nargs="+", default=list(DEFAULT_SMOOTHINGS), choices=list(DEFAULT_SMOOTHINGS))
    parser.add_argument("--hemis", nargs="+", default=list(DEFAULT_HEMIS), choices=list(DEFAULT_HEMIS))
    parser.add_argument("--jobs", type=int, default=DEFAULT_JOBS)
    parser.add_argument("--n-components", type=int, default=3)
    parser.add_argument("--gradient-random-state", type=int, default=DEFAULT_GRADIENT_RANDOM_STATE)
    parser.add_argument("--seed", type=int, default=20260420)
    parser.add_argument("--top-n-cap", type=int, default=DEFAULT_TOP_N_CAP)
    parser.add_argument("--network-column", default="canonical8", choices=["canonical8", "family17"])
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--stop-on-first-failure", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    return run_supervisor(args)


if __name__ == "__main__":
    raise SystemExit(main())
