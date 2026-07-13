#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import importlib.util
import os
import shutil
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import nibabel as nib
import numpy as np

from routea_individual_gradients_wta import (
    CORE_WORKFLOW_DIR,
    DEFAULT_HEMIS,
    DEFAULT_INDIVIDUAL_OVERVIEW_ROOT,
    DEFAULT_JOBS,
    DEFAULT_K,
    DEFAULT_OUT_ROOT,
    DEFAULT_SMOOTHINGS,
    DEFAULT_SUBJECTS_MANIFEST,
    DEFAULT_KONG2022_ROUTEA_ROOT,
    DEFAULT_KONG2022_SURFACE_ROOT,
    DEFAULT_HIPPUNFOLD_BATCH_ROOT,
    OVERVIEW_METHODS,
    OVERVIEW_METHOD_ALIASES,
    THREAD_ENV_OVERRIDES,
    SubjectCombo,
    finite_fisher_z,
    git_commit,
    gradient_combo_output_dir,
    load_kong2022_roi_registry,
    load_network_names_and_keys,
    load_subject_ids,
    normalize_overview_method,
    normalize_subject,
    overview_output_path,
    read_json,
    roi_fc_path,
    subject_spec_path,
    subject_surface_path,
    utc_now_iso,
    valid_mask_path,
    write_json,
    wta_combo_output_dir,
)
from lib.weighted_fisherz import (
    assignment_status_from_weighted_network_profile,
    weighted_cluster_network_fisherz_profile_from_timeseries,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ADAPTER_ROOT = DEFAULT_OUT_ROOT / "_overview_core_adapter"
DEFAULT_ADAPTER_PRESENT_ROOT = DEFAULT_ADAPTER_ROOT / "present_tmp"
DEFAULT_SCENE = REPO_ROOT / "config" / "wb_locked_native_view_lateral_medial.scene"
COVERAGE_TAG = "cov_0p80"
COVERAGE_THRESHOLD = "0.80"
ATLAS = "kong2022"
ROUTE_TAG = "routea"
GRADIENT_CORE_METHOD = "gradient-spectral"
WTA_CORE_METHOD = "wta-top"
GRADIENT_COMP_TAG = "g1g2g3"


def require_label_map_network_order(label_map_payload: dict[str, Any], expected_network_names: list[str], *, context: str) -> None:
    raw_names = label_map_payload.get("network_names")
    if not isinstance(raw_names, list):
        raise RuntimeError(f"Missing network_names in {context} label map")
    got = [str(x) for x in raw_names]
    expected = [str(x) for x in expected_network_names]
    if got != expected:
        raise RuntimeError(
            f"Network order mismatch in {context}: label_map={got} expected={expected}. "
            "Run scripts/iowa7t/audit_repair_weighted_network_order.py --repair before rendering."
        )


K_SELECTION_FIELDS = [
    "method", "atlas", "smoothing", "hemi", "coverage_threshold", "k", "n_subjects",
    "n_support_vertices", "V_min_effective", "group_ari_mean", "group_ari_se",
    "subject_split_requested", "subject_split_used", "subject_split_skipped_overlap",
    "subject_split_skipped_empty", "run_aware_ari_mean", "run_aware_ari_se", "run_aware_n_pairs",
    "run_aware_n_used", "ARI_null_mean", "ARI_null_5pct", "ARI_weak_null_mean",
    "B_null_requested", "B_null_used_strong", "B_null_used_weak", "min_cluster_size",
    "max_components_per_cluster", "min_profile_coherence", "mean_profile_coherence",
    "nontrivial_min_size_ok", "nontrivial_connectedness_ok", "nontrivial_coherence_ok",
    "local_max", "within_1se_best", "selected_nontrivial", "max_profile_similarity",
    "redundant_pair_count", "nonredundant_ok", "reject_nontrivial_fail", "reject_redundant",
    "reject_reason_primary", "selected_final", "within_1se_survivor_pool", "survivor_pool_argmax_k",
    "k_final_decision_path",
]


def load_module(path: Path, module_name: str):
    if str(CORE_WORKFLOW_DIR) not in sys.path:
        sys.path.insert(0, str(CORE_WORKFLOW_DIR))
    if module_name in sys.modules:
        return sys.modules[module_name]
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load module: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def core_overview_module():
    return load_module(CORE_WORKFLOW_DIR / "06_01_build_group_overview.py", "_individual_adapter_core_overview")


def core_atlas_module():
    return load_module(CORE_WORKFLOW_DIR / "05_02_run_group_atlas.py", "_individual_adapter_core_atlas")


def write_tsv(path: Path, rows: list[dict[str, object]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, delimiter="\t", extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def fixed_k_selection_row(*, method: str, smoothing: str, hemi: str, n_support: int) -> dict[str, object]:
    return fixed_k_selection_rows(
        method=method,
        smoothing=smoothing,
        hemi=hemi,
        n_support=n_support,
        k_values=(DEFAULT_K,),
    )[0]


def fixed_k_selection_rows(
    *,
    method: str,
    smoothing: str,
    hemi: str,
    n_support: int,
    k_values: tuple[int, ...] | range = range(2, 9),
    selected_k: int = DEFAULT_K,
    redundancy_by_k: dict[int, tuple[float, int]] | None = None,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    redundancy_by_k = redundancy_by_k or {}
    for k_raw in k_values:
        k = int(k_raw)
        max_sim, redundant_pairs = redundancy_by_k.get(k, (0.0, 0))
        row = {field: "" for field in K_SELECTION_FIELDS}
        selected = int(k == int(selected_k))
        row.update(
            {
                "method": method,
                "atlas": ATLAS,
                "smoothing": smoothing,
                "hemi": hemi,
                "coverage_threshold": COVERAGE_THRESHOLD,
                "k": k,
                "n_subjects": 1,
                "n_support_vertices": int(n_support),
                "V_min_effective": 1,
                "group_ari_mean": 1.0,
                "group_ari_se": 0.0,
                "subject_split_requested": 0,
                "subject_split_used": 0,
                "subject_split_skipped_overlap": 0,
                "subject_split_skipped_empty": 0,
                "run_aware_ari_mean": "",
                "run_aware_ari_se": "",
                "run_aware_n_pairs": 0,
                "run_aware_n_used": 0,
                "ARI_null_mean": 0.0,
                "ARI_null_5pct": 0.0,
                "ARI_weak_null_mean": 0.0,
                "B_null_requested": 0,
                "B_null_used_strong": 0,
                "B_null_used_weak": 0,
                "min_cluster_size": int(n_support),
                "max_components_per_cluster": 1,
                "min_profile_coherence": 1.0,
                "mean_profile_coherence": 1.0,
                "nontrivial_min_size_ok": 1,
                "nontrivial_connectedness_ok": 1,
                "nontrivial_coherence_ok": 1,
                "local_max": selected,
                "within_1se_best": selected,
                "selected_nontrivial": selected,
                "max_profile_similarity": float(max_sim),
                "redundant_pair_count": int(redundant_pairs),
                "nonredundant_ok": int(float(max_sim) <= 0.80 and int(redundant_pairs) == 0),
                "reject_nontrivial_fail": 0,
                "reject_redundant": int(float(max_sim) > 0.80 or int(redundant_pairs) > 0),
                "reject_reason_primary": "fixed_individual_k3" if selected else "individual_diagnostic_only",
                "selected_final": selected,
                "within_1se_survivor_pool": 1,
                "survivor_pool_argmax_k": int(selected_k),
                "k_final_decision_path": "individual_fixed_k3_full_k_diagnostic",
            }
        )
        rows.append(row)
    return rows


def k_selection_summary_has_full_range(path: Path, *, k_values: range = range(2, 9)) -> bool:
    if not Path(path).exists():
        return False
    lines = Path(path).read_text(encoding="utf-8").splitlines()
    if len(lines) < 2:
        return False
    header = lines[0].split("	")
    if "k" not in header:
        return False
    k_idx = header.index("k")
    got: set[int] = set()
    for line in lines[1:]:
        parts = line.split("	")
        if k_idx >= len(parts):
            continue
        try:
            got.add(int(parts[k_idx]))
        except ValueError:
            continue
    return set(int(k) for k in k_values).issubset(got)


def individual_gradient_redundancy_by_k(
    *,
    core,
    gradients_active: np.ndarray,
    valid_mask: np.ndarray,
    surface_path: Path,
    subject: str,
    shared_root: Path,
    surface_root: Path,
    smoothing: str,
    hemi: str,
    k_values: range = range(2, 9),
) -> dict[int, tuple[float, int]]:
    coords, faces = core.load_surface(Path(surface_path))
    adjacency = core.build_surface_adjacency(faces, int(coords.shape[0]))
    axis_order = core.long_axis_order(Path(surface_path))
    valid = np.asarray(valid_mask, dtype=bool)
    gradients = np.asarray(gradients_active, dtype=np.float32)[:, :3]
    out: dict[int, tuple[float, int]] = {}
    for k_raw in k_values:
        k = int(k_raw)
        _labels_active, labels_full = core.cluster_from_precomputed_group_features(
            method="gradient-spectral",
            group_features=gradients,
            support_mask=valid,
            adjacency_full=adjacency,
            k=k,
            seed=0,
        )
        labels_full = core.reorder_labels_by_long_axis(labels_full.astype(np.int32), axis_order).astype(np.int32)
        roi_profile, _roi_registry = core.cluster_roi_profile(
            group_labels_final=labels_full,
            support_mask=valid,
            subjects=[subject],
            shared_root=shared_root,
            surface_root=surface_root,
            reference_root=shared_root,
            smoothing=smoothing,
            hemi=hemi,
            valid_masks_by_subject={subject: valid},
        )
        out[k] = core.profile_redundancy_metrics(roi_profile)
    return out


def adapter_run_dir(adapter_root: Path, subject: str, method: str, smoothing: str, hemi: str) -> Path:
    if method == GRADIENT_CORE_METHOD:
        return adapter_root / f"sub-{subject}" / method / ATLAS / smoothing / COVERAGE_TAG / GRADIENT_COMP_TAG / f"hemi_{hemi}"
    return adapter_root / f"sub-{subject}" / method / ATLAS / smoothing / COVERAGE_TAG / f"hemi_{hemi}"



def count_profile_from_roi_profile(
    *,
    roi_profile: np.ndarray,
    roi_registry: list[dict[str, object]],
    network_names: list[str],
    network_column: str,
    top_n_cap: int,
) -> np.ndarray:
    arr = np.asarray(roi_profile, dtype=np.float32)
    if arr.ndim != 2:
        raise ValueError(f"Expected 2D ROI profile, got shape={arr.shape}")
    if arr.shape[1] != len(roi_registry):
        raise ValueError(f"ROI profile/registry mismatch: profile_cols={arr.shape[1]} registry={len(roi_registry)}")
    out = np.zeros((int(arr.shape[0]), int(len(network_names))), dtype=np.float32)
    col = str(network_column)
    net_to_idx = {str(name): idx for idx, name in enumerate(network_names)}
    for row_idx in range(arr.shape[0]):
        values = arr[row_idx, :]
        pos_idx = np.flatnonzero(np.isfinite(values) & (values > 0.0))
        if pos_idx.size == 0:
            continue
        ordered = pos_idx[np.argsort(-values[pos_idx])]
        top_idx = ordered[: int(min(int(top_n_cap), int(ordered.size)))]
        for roi_col in top_idx:
            net = str(roi_registry[int(roi_col)][col])
            if net in net_to_idx:
                out[row_idx, int(net_to_idx[net])] += 1.0
    return out.astype(np.float32)


def read_kong2022_roi_weight_rows(path: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    with Path(path).open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
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


def weighted_gradient_profile_from_reference(
    *,
    core,
    labels: np.ndarray,
    valid_mask: np.ndarray,
    subject: str,
    shared_root: Path,
    surface_root: Path,
    smoothing: str,
    hemi: str,
    network_names: list[str],
    network_column: str,
) -> np.ndarray:
    ts_path = Path(surface_root) / f"sub-{subject}" / "surface" / smoothing / f"sub-{subject}_hemi-{hemi}_timeseries.npy"
    roi_path = Path(shared_root) / f"sub-{subject}" / "reference" / ATLAS / "cortex_roi_parcel_timeseries.npy"
    roi_tsv = Path(shared_root) / f"sub-{subject}" / "reference" / ATLAS / "cortex_roi_parcels.tsv"
    if not ts_path.exists():
        raise FileNotFoundError(f"Missing hippocampal timeseries for weighted profile: {ts_path}")
    if not roi_path.exists():
        raise FileNotFoundError(f"Missing ROI timeseries for weighted profile: {roi_path}")
    if not roi_tsv.exists():
        raise FileNotFoundError(f"Missing ROI metadata for weighted profile: {roi_tsv}")
    return weighted_cluster_network_fisherz_profile_from_timeseries(
        group_labels_final=np.asarray(labels, dtype=np.int32),
        support_mask=np.asarray(valid_mask, dtype=bool),
        hippocampal_timeseries=np.load(ts_path).astype(np.float32),
        roi_timeseries=np.load(roi_path).astype(np.float32),
        roi_rows=read_kong2022_roi_weight_rows(roi_tsv),
        network_names=network_names,
        network_column=network_column,
    )


def gradient_profile_for_render(
    *,
    assignment_variant: str,
    direct_profile: np.ndarray,
    assignment_variants: dict[str, dict[str, object]],
    weighted_profile_path: Path,
    count_profile: np.ndarray | None = None,
    weighted_profile: np.ndarray | None = None,
) -> np.ndarray:
    if assignment_variant == "fisherz":
        profile = assignment_variants.get("fisherz", {}).get("profile")
        if profile is None:
            raise RuntimeError("Missing fisherz assignment profile for gradient overview render")
        return np.asarray(profile, dtype=np.float32)
    if assignment_variant == "count":
        if count_profile is None:
            raise RuntimeError("Missing count assignment profile for gradient overview render")
        return np.asarray(count_profile, dtype=np.float32)
    if assignment_variant == "weighted_fisherz":
        if weighted_profile is not None:
            return np.asarray(weighted_profile, dtype=np.float32)
        if weighted_profile_path.exists():
            return np.load(weighted_profile_path).astype(np.float32)
    return np.asarray(direct_profile, dtype=np.float32)

def materialize_gradient_core_inputs(*, subject: str, work_root: Path, adapter_root: Path, shared_root: Path, surface_root: Path, hippunfold_batch_root: Path, assignment_variant: str = "fisherz", network_column: str = "canonical8") -> None:
    core = core_atlas_module()
    network_names = core.verify_canonical_network_order_consistency(shared_root, [subject], ATLAS, network_taxonomy=network_column)
    color_map = core.load_canonical_network_color_map()
    for smoothing in DEFAULT_SMOOTHINGS:
        for hemi in DEFAULT_HEMIS:
            combo = SubjectCombo(subject, smoothing, hemi)
            src = gradient_combo_output_dir(work_root, combo)
            dst = adapter_run_dir(adapter_root, subject, GRADIENT_CORE_METHOD, smoothing, hemi)
            dst.mkdir(parents=True, exist_ok=True)
            labels = np.load(src / "cluster_labels_k3.npy").astype(np.int32)
            valid_mask = np.load(valid_mask_path(surface_root, subject, hemi)).astype(bool)
            profile = core.cluster_network_profile(
                group_labels_final=labels,
                support_mask=valid_mask,
                subjects=[subject],
                shared_root=shared_root,
                surface_root=surface_root,
                reference_root=shared_root,
                atlas=ATLAS,
                smoothing=smoothing,
                hemi=hemi,
                valid_masks_by_subject={subject: valid_mask},
                profile_fc_mode="pearson",
                network_taxonomy=network_column,
            )
            roi_profile, roi_registry = core.cluster_roi_profile(
                group_labels_final=labels,
                support_mask=valid_mask,
                subjects=[subject],
                shared_root=shared_root,
                surface_root=surface_root,
                reference_root=shared_root,
                smoothing=smoothing,
                hemi=hemi,
                valid_masks_by_subject={subject: valid_mask},
            )
            assignment_variants = core.cluster_assignment_variants_from_kong2022_roi_profile(
                roi_profile=roi_profile,
                roi_profile_3d=roi_profile[np.newaxis, :, :],
                roi_registry=roi_registry,
                network_names=network_names,
                q_thresh=0.05,
                top_n_cap=20,
                network_column=network_column,
            )
            functional_variant_payloads: dict[str, dict[str, object]] = {}
            count_profile = count_profile_from_roi_profile(
                roi_profile=roi_profile,
                roi_registry=roi_registry,
                network_names=network_names,
                network_column=network_column,
                top_n_cap=20,
            )
            computed_weighted_profile = None
            if assignment_variant == "weighted_fisherz":
                computed_weighted_profile = weighted_gradient_profile_from_reference(
                    core=core,
                    labels=labels,
                    valid_mask=valid_mask,
                    subject=subject,
                    shared_root=shared_root,
                    surface_root=surface_root,
                    smoothing=smoothing,
                    hemi=hemi,
                    network_names=network_names,
                    network_column=network_column,
                )
            profile_for_render = gradient_profile_for_render(
                assignment_variant=assignment_variant,
                direct_profile=profile,
                assignment_variants=assignment_variants,
                weighted_profile_path=src / "cluster_network_profile_weighted_fisherz.npy",
                count_profile=count_profile,
                weighted_profile=computed_weighted_profile,
            )
            np.save(dst / "group_labels_final.npy", labels.astype(np.int32))
            np.save(dst / "cluster_network_profile.npy", profile_for_render.astype(np.float32))
            np.save(dst / "cluster_roi_profile.npy", roi_profile.astype(np.float32))
            write_json(dst / "group_feature_summary.json", {"network_names": network_names, "N_ref": int(len(network_names)), "network_column": network_column})
            for variant_name, variant_spec in assignment_variants.items():
                variant_rows = [dict(x) for x in variant_spec["status_rows"]]
                functional_labels, functional_meta, label_items = core.functional_atlas_from_clusters(
                    group_labels_final=labels,
                    status_rows=variant_rows,
                    network_names=network_names,
                    color_map=color_map,
                )
                func_label = dst / "template" / "workbench_assets" / f"sub-{subject}_hemi-{hemi}_space-corobl_den-512_label-individual_functional_atlas_gradient-spectral_{smoothing}_{variant_name}.label.gii"
                func_label.parent.mkdir(parents=True, exist_ok=True)
                nib.save(core.make_named_label_gifti(functional_labels.astype(np.int32), label_items), str(func_label))
                np.save(dst / f"group_functional_atlas_labels_{variant_name}.npy", functional_labels.astype(np.int32))
                status_payload = {
                    "items": variant_rows,
                    "assignment_metric": str(variant_spec["assignment_metric"]),
                    "q_threshold": variant_spec["q_threshold"],
                    "top_n_cap": variant_spec["top_n_cap"],
                    "individual_level": True,
                    "network_column": network_column,
                }
                write_json(dst / f"cluster_assignment_status_{variant_name}.json", status_payload)
                write_json(dst / f"group_functional_atlas_label_map_{variant_name}.json", functional_meta)
                functional_variant_payloads[str(variant_name)] = {
                    "cluster_assignment_status": str((dst / f"cluster_assignment_status_{variant_name}.json").resolve()),
                    "group_functional_atlas_labels": str((dst / f"group_functional_atlas_labels_{variant_name}.npy").resolve()),
                    "group_functional_atlas_label_map": str((dst / f"group_functional_atlas_label_map_{variant_name}.json").resolve()),
                    "template_functional_atlas_label": str(func_label.resolve()),
                }
            if assignment_variant == "weighted_fisherz" and computed_weighted_profile is not None:
                weighted_rows, weighted_dominant = assignment_status_from_weighted_network_profile(computed_weighted_profile, network_names)
                functional_labels, functional_meta, label_items = core.functional_atlas_from_clusters(
                    group_labels_final=labels,
                    status_rows=weighted_rows,
                    network_names=network_names,
                    color_map=color_map,
                )
                functional_meta["assignment_variant"] = "weighted_fisherz"
                functional_meta["network_column"] = network_column
                local_weighted_label = dst / "template" / "workbench_assets" / f"sub-{subject}_hemi-{hemi}_space-corobl_den-512_label-individual_functional_atlas_gradient-spectral_{smoothing}_weighted_fisherz.label.gii"
                local_weighted_label.parent.mkdir(parents=True, exist_ok=True)
                nib.save(core.make_named_label_gifti(functional_labels.astype(np.int32), label_items), str(local_weighted_label))
                np.save(dst / "cluster_network_profile_weighted_fisherz.npy", computed_weighted_profile.astype(np.float32))
                np.save(dst / "group_functional_atlas_labels_weighted_fisherz.npy", functional_labels.astype(np.int32))
                write_json(dst / "cluster_assignment_status_weighted_fisherz.json", {"items": weighted_rows, "assignment_metric": "top1_weighted_network_fisher_z", "network_column": network_column})
                write_json(dst / "cluster_dominant_network_weighted_fisherz.json", {"dominant": weighted_dominant, "network_names": network_names, "assignment_variant": "weighted_fisherz", "network_column": network_column})
                write_json(dst / "group_functional_atlas_label_map_weighted_fisherz.json", functional_meta)
                functional_variant_payloads["weighted_fisherz"] = {
                    "cluster_assignment_status": str((dst / "cluster_assignment_status_weighted_fisherz.json").resolve()),
                    "group_functional_atlas_labels": str((dst / "group_functional_atlas_labels_weighted_fisherz.npy").resolve()),
                    "group_functional_atlas_label_map": str((dst / "group_functional_atlas_label_map_weighted_fisherz.json").resolve()),
                    "template_functional_atlas_label": str(local_weighted_label.resolve()),
                }
            dominant = [str(row.get("top1_network", "unassigned")) for row in assignment_variants["fisherz"]["status_rows"]]
            write_json(dst / "cluster_dominant_network.json", {"dominant": dominant, "network_names": network_names, "assignment_variant": "fisherz", "network_column": network_column})
            write_json(dst / "cluster_dominant_network_count.json", {"dominant": [str(row.get("top1_network", "unassigned")) for row in assignment_variants["count"]["status_rows"]], "network_names": network_names, "assignment_variant": "count", "network_column": network_column})
            surf = subject_surface_path(hippunfold_batch_root, subject, hemi)
            k_summary_path = dst / "k_selection_summary.tsv"
            if not k_selection_summary_has_full_range(k_summary_path):
                gradients_active = np.load(src / "gradients" / "hippomaps_gradients_active.npy").astype(np.float32)
                redundancy_by_k = individual_gradient_redundancy_by_k(
                    core=core,
                    gradients_active=gradients_active,
                    valid_mask=valid_mask,
                    surface_path=surf,
                    subject=subject,
                    shared_root=shared_root,
                    surface_root=surface_root,
                    smoothing=smoothing,
                    hemi=hemi,
                )
                write_tsv(
                    k_summary_path,
                    fixed_k_selection_rows(
                        method=GRADIENT_CORE_METHOD,
                        smoothing=smoothing,
                        hemi=hemi,
                        n_support=int(np.sum(valid_mask)),
                        redundancy_by_k=redundancy_by_k,
                    ),
                    K_SELECTION_FIELDS,
                )
            default_variant = assignment_variant if assignment_variant in functional_variant_payloads else "fisherz"
            manifest = {
                "status": "ok",
                "method": GRADIENT_CORE_METHOD,
                "atlas": ATLAS,
                "smoothing": smoothing,
                "hemi": hemi,
                "chosen_k_final": 3,
                "n_subjects": 1,
                "template_subject": subject,
                "template_left_or_right_surface": str(surf.resolve()),
                "template_spec_path": str(subject_spec_path(surf, subject).resolve()),
                "assignment_mode": "cluster_level",
                "network_column": network_column,
                "default_assignment_variant": default_variant,
                "group_functional_atlas_labels": functional_variant_payloads[default_variant]["group_functional_atlas_labels"],
                "group_functional_atlas_label_map": functional_variant_payloads[default_variant]["group_functional_atlas_label_map"],
                "template_functional_atlas_label": functional_variant_payloads[default_variant]["template_functional_atlas_label"],
                "assignment_variants": functional_variant_payloads,
            }
            write_json(dst / "group_atlas_manifest.json", manifest, sort_keys=True)


def wta_variant_suffix(assignment_variant: str) -> str:
    if assignment_variant == "weighted_fisherz":
        return "_weighted_fisherz"
    if assignment_variant == "normalized_top20_count":
        return "_normalized_top20_count"
    if assignment_variant == "family17_normalized_top20_count":
        return "_family17_normalized_top20_count"
    return ""


def resolve_wta_label_gifti(*, src: Path, subject: str, smoothing: str, hemi: str, suffix: str) -> Path:
    expected = (
        Path(src)
        / "workbench_assets"
        / f"sub-{subject}_hemi-{hemi}_space-corobl_den-512_label-individual_wta_top_{smoothing}{suffix}.label.gii"
    )
    if expected.exists():
        return expected
    if suffix:
        raise FileNotFoundError(f"Missing WTA label GIFTI for suffix {suffix!r}: {expected}")
    candidates = sorted((Path(src) / "workbench_assets").glob("*.label.gii"))
    if not candidates:
        raise FileNotFoundError(f"Missing WTA label GIFTI under {Path(src) / 'workbench_assets'}")
    return candidates[0]


def materialize_wta_core_inputs(*, subject: str, work_root: Path, adapter_root: Path, hippunfold_batch_root: Path, assignment_variant: str = "fisherz") -> None:
    for smoothing in DEFAULT_SMOOTHINGS:
        for hemi in DEFAULT_HEMIS:
            combo = SubjectCombo(subject, smoothing, hemi)
            src = wta_combo_output_dir(work_root, combo)
            dst = adapter_run_dir(adapter_root, subject, WTA_CORE_METHOD, smoothing, hemi)
            dst.mkdir(parents=True, exist_ok=True)
            suffix = wta_variant_suffix(assignment_variant)
            labels = np.load(src / f"wta_top_labels{suffix}.npy").astype(np.int32)
            label_map = read_json(src / f"wta_top_label_map{suffix}.json")
            if assignment_variant == "weighted_fisherz":
                roi_registry = load_kong2022_roi_registry()
                expected_network_names, _network_to_key = load_network_names_and_keys(roi_registry, "canonical8")
                require_label_map_network_order(
                    label_map,
                    expected_network_names,
                    context=f"sub-{subject} wta {smoothing} hemi-{hemi} weighted_fisherz",
                )
            status = read_json(src / f"vertex_assignment_status{suffix}.json")
            src_label = resolve_wta_label_gifti(src=src, subject=subject, smoothing=smoothing, hemi=hemi, suffix=suffix)
            local_label = dst / "template" / "workbench_assets" / src_label.name
            local_label.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src_label, local_label)
            np.save(dst / "group_functional_atlas_labels.npy", labels)
            write_json(dst / "group_functional_atlas_label_map.json", label_map)
            write_json(dst / "vertex_assignment_status.json", status)
            surf = subject_surface_path(hippunfold_batch_root, subject, hemi)
            write_json(
                dst / "group_atlas_manifest.json",
                {
                    "status": "ok",
                    "method": WTA_CORE_METHOD,
                    "atlas": ATLAS,
                    "smoothing": smoothing,
                    "hemi": hemi,
                    "n_subjects": 1,
                    "template_subject": subject,
                    "template_left_or_right_surface": str(surf.resolve()),
                    "template_spec_path": str(subject_spec_path(surf, subject).resolve()),
                    "template_functional_atlas_label": str(local_label.resolve()),
                    "group_functional_atlas_labels": str((dst / "group_functional_atlas_labels.npy").resolve()),
                    "group_functional_atlas_label_map": str((dst / "group_functional_atlas_label_map.json").resolve()),
                    "assignment_variant": assignment_variant,
                    "confidence_metric": assignment_variant if assignment_variant == "weighted_fisherz" else "topn_count",
                },
                sort_keys=True,
            )


def render_via_core(*, subject: str, method: str, work_root: Path, overview_root: Path, adapter_root: Path, adapter_present_root: Path, shared_root: Path, surface_root: Path, hippunfold_batch_root: Path, scene: Path, network_column: str = "canonical8") -> Path:
    core_overview = core_overview_module()
    method_key = normalize_overview_method(method)
    method_adapter_root = adapter_root / method_key
    method_adapter_present_root = adapter_present_root / method_key
    subject_root = method_adapter_root / f"sub-{subject}"
    if method_key in {"gradient_spectral_k3", "gradient_count", "gradient_weighted_fisherz"}:
        gradient_variant = "weighted_fisherz" if method_key == "gradient_weighted_fisherz" else ("count" if method_key == "gradient_count" else "fisherz")
        materialize_gradient_core_inputs(subject=subject, work_root=work_root, adapter_root=method_adapter_root, shared_root=shared_root, surface_root=surface_root, hippunfold_batch_root=hippunfold_batch_root, assignment_variant=gradient_variant, network_column=network_column)
        meta = core_overview.render_combo(
            group_root=subject_root,
            present_dir=method_adapter_present_root / f"sub-{subject}" / "gradient",
            method=GRADIENT_CORE_METHOD,
            atlas=ATLAS,
            coverage_tag=COVERAGE_TAG,
            scene=scene,
            route_tag=ROUTE_TAG,
            mode_tag=("weighted_fisherz" if method_key == "gradient_weighted_fisherz" else None),
            assignment_variant=("weighted_fisherz" if method_key == "gradient_weighted_fisherz" else ("count" if method_key == "gradient_count" else "fisherz")),
            gradient_comp_tag=GRADIENT_COMP_TAG,
        )
    else:
        wta_variant_by_method = {
            "wta_weighted_fisherz": "weighted_fisherz",
            "wta_normalized_top20_count": "normalized_top20_count",
            "wta_family17_normalized_top20_count": "family17_normalized_top20_count",
        }
        wta_variant = wta_variant_by_method.get(method_key, "fisherz")
        materialize_wta_core_inputs(subject=subject, work_root=work_root, adapter_root=method_adapter_root, hippunfold_batch_root=hippunfold_batch_root, assignment_variant=wta_variant)
        mode_tag = None if wta_variant == "fisherz" else wta_variant
        meta = core_overview.render_combo(
            group_root=subject_root,
            present_dir=method_adapter_present_root / f"sub-{subject}" / "wta",
            method=WTA_CORE_METHOD,
            atlas=ATLAS,
            coverage_tag=COVERAGE_TAG,
            scene=scene,
            route_tag=ROUTE_TAG,
            mode_tag=mode_tag,
            assignment_variant=("weighted_fisherz" if mode_tag else "fisherz"),
            gradient_comp_tag="",
        )
    final = overview_output_path(overview_root, method_key, subject, network_column=network_column)
    final.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(Path(str(meta["overview_png"])), final)
    return final


def overview_worker(payload: dict[str, object]) -> dict[str, object]:
    for key, value in THREAD_ENV_OVERRIDES.items():
        os.environ[key] = value
    started = time.time()
    subject = normalize_subject(str(payload["subject"]))
    method = normalize_overview_method(str(payload["method"]))
    try:
        out_path = render_via_core(
            subject=subject,
            method=method,
            work_root=Path(str(payload["work_root"])),
            overview_root=Path(str(payload["overview_root"])),
            adapter_root=Path(str(payload["adapter_root"])),
            adapter_present_root=Path(str(payload["adapter_present_root"])),
            shared_root=Path(str(payload["shared_root"])),
            surface_root=Path(str(payload["surface_root"])),
            hippunfold_batch_root=Path(str(payload["hippunfold_batch_root"])),
            scene=Path(str(payload["scene"])),
            network_column=str(payload.get("network_column", "canonical8")),
        )
        return {"status": "ok", "subject": f"sub-{subject}", "method": method, "task_dir": OVERVIEW_METHODS[method]["task_dir"], "output": str(out_path), "elapsed_seconds": time.time() - started}
    except Exception as exc:
        return {"status": "failed", "subject": f"sub-{subject}", "method": method, "task_dir": OVERVIEW_METHODS[method]["task_dir"], "error_type": exc.__class__.__name__, "error": str(exc), "elapsed_seconds": time.time() - started}


def write_status_tsv(path: Path, rows: list[dict[str, object]]) -> None:
    fields = ["status", "subject", "method", "task_dir", "elapsed_seconds", "output", "error_type", "error"]
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = ["\t".join(fields)]
    for row in rows:
        lines.append("\t".join(str(row.get(field, "")).replace("\t", " ").replace("\n", " ") for field in fields))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build individual overviews by adapting subject-level products to core_workflow/06_01_build_group_overview.py inputs.")
    parser.add_argument("--work-root", default=str(DEFAULT_OUT_ROOT))
    parser.add_argument("--overview-root", default=str(DEFAULT_INDIVIDUAL_OVERVIEW_ROOT))
    parser.add_argument("--adapter-root", default=str(DEFAULT_ADAPTER_ROOT))
    parser.add_argument("--adapter-present-root", default=str(DEFAULT_ADAPTER_PRESENT_ROOT))
    parser.add_argument("--shared-root", default=str(DEFAULT_KONG2022_ROUTEA_ROOT))
    parser.add_argument("--surface-root", default=str(DEFAULT_KONG2022_SURFACE_ROOT))
    parser.add_argument("--hippunfold-batch-root", default=str(DEFAULT_HIPPUNFOLD_BATCH_ROOT))
    parser.add_argument("--scene", default=str(DEFAULT_SCENE))
    parser.add_argument("--subjects-manifest", default=str(DEFAULT_SUBJECTS_MANIFEST))
    parser.add_argument("--subjects", nargs="*", default=None)
    parser.add_argument("--max-subjects", type=int, default=None)
    parser.add_argument("--methods", nargs="+", default=["gradient_spectral_k3", "wta_top"], choices=["gradient", "count", "wta", "gradient_spectral_k3", "gradient_count", "gradient_weighted_fisherz", "wta_top", "wta_weighted_fisherz", "wta_normalized_top20_count", "wta_family17_normalized_top20_count"])
    parser.add_argument("--jobs", type=int, default=DEFAULT_JOBS)
    parser.add_argument("--network-column", default="canonical8", choices=["canonical8", "family17"])
    parser.add_argument("--resume", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    work_root = Path(args.work_root).resolve()
    overview_root = Path(args.overview_root).resolve()
    adapter_root = Path(args.adapter_root).resolve()
    adapter_present_root = Path(args.adapter_present_root).resolve()
    subjects = [normalize_subject(s) for s in args.subjects] if args.subjects else load_subject_ids(Path(args.subjects_manifest))
    if args.max_subjects is not None:
        subjects = subjects[: int(args.max_subjects)]
    methods = [normalize_overview_method(method) for method in args.methods]

    tasks: list[dict[str, object]] = []
    rows: list[dict[str, object]] = []
    for subject in subjects:
        for method in methods:
            expected = overview_output_path(overview_root, method, subject, network_column=str(args.network_column))
            if args.resume and expected.exists():
                rows.append({"status": "ok", "subject": f"sub-{subject}", "method": method, "task_dir": OVERVIEW_METHODS[method]["task_dir"], "output": str(expected), "elapsed_seconds": 0.0, "resumed": True})
                continue
            tasks.append({"work_root": str(work_root), "overview_root": str(overview_root), "adapter_root": str(adapter_root), "adapter_present_root": str(adapter_present_root), "shared_root": str(Path(args.shared_root).resolve()), "surface_root": str(Path(args.surface_root).resolve()), "hippunfold_batch_root": str(Path(args.hippunfold_batch_root).resolve()), "scene": str(Path(args.scene).resolve()), "subject": subject, "method": method, "network_column": str(args.network_column)})

    jobs = max(1, min(int(args.jobs), 10, len(tasks) if tasks else 1))
    started = time.time()
    if tasks:
        with ProcessPoolExecutor(max_workers=jobs) as executor:
            future_to_task = {executor.submit(overview_worker, task): task for task in tasks}
            for future in as_completed(future_to_task):
                rows.append(future.result())
                write_status_tsv(overview_root / "overview_status.tsv", sorted(rows, key=lambda x: (str(x.get("method", "")), str(x.get("subject", "")))))
    rows = sorted(rows, key=lambda x: (str(x.get("method", "")), str(x.get("subject", ""))))
    write_status_tsv(overview_root / "overview_status.tsv", rows)
    summary = {"status": "failed" if any(str(row.get("status")) == "failed" for row in rows) else "ok", "workflow": "individual_core_06_01_overview_adapter", "work_root": str(work_root), "adapter_root": str(adapter_root), "overview_root": str(overview_root), "subjects": len(subjects), "methods": methods, "total": len(rows), "ok": sum(1 for row in rows if str(row.get("status")) == "ok"), "failed": sum(1 for row in rows if str(row.get("status")) == "failed"), "jobs": jobs, "elapsed_seconds": time.time() - started, "git_commit": git_commit(short=True), "updated_at": utc_now_iso(zulu=True), "status_tsv": str((overview_root / "overview_status.tsv").resolve())}
    write_json(overview_root / "overview_manifest.json", {"summary": summary, "items": rows}, sort_keys=True)
    print(f"status={summary['status']} ok={summary['ok']} failed={summary['failed']} total={summary['total']}")
    print(f"overview_root={overview_root}")
    print(f"adapter_root={adapter_root}")
    print(f"status_tsv={overview_root / 'overview_status.tsv'}")
    print(f"manifest={overview_root / 'overview_manifest.json'}")
    return 1 if summary["failed"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
