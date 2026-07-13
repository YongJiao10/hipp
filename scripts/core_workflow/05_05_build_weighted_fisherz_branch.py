#!/usr/bin/env python3
from __future__ import annotations

"""
Build a post-hoc RouteA weighted_fisherz comparison branch.

This does not modify the maintained fisherz outputs. It copies selected RouteA
K=3 group-atlas runs into an isolated output root, then recomputes:
- gradient-spectral cluster-level weighted_fisherz assignment labels
- wta-top vertex-level weighted_fisherz assignment labels

The weighted target network profiles are computed by first reconstructing
network time series from Kong2022 ROI mean time series and ROI valid-grayordinate
counts, then recomputing hippocampus-to-network Fisher-z FC.
"""

import argparse
import csv
import json
import shutil
from pathlib import Path

import nibabel as nib
import numpy as np

from lib.gifti_labels import make_named_label_gifti, materialize_combo
from lib.io_utils import load_json
from lib.network_colors import load_network_colors
from lib.output_layout import (
    DEFAULT_KONG2022_ROUTEA_ROOT,
    DEFAULT_KONG2022_SURFACE_ROOT,
    DEFAULT_KONG2022_TEMPLATE_ROOT,
    DEFAULT_OVERVIEWS_DIR,
    DEFAULT_ROUTEA_GROUP_ATLAS_ROOT,
)
from lib.workflow_utils import write_json
from lib.weighted_fisherz import (
    assign_weighted_network_profile_rows,
    assignment_status_from_weighted_network_profile,
    weighted_cluster_network_fisherz_profile_from_timeseries,
    weighted_network_fisherz_profile_from_timeseries,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SOURCE_ROOT = DEFAULT_ROUTEA_GROUP_ATLAS_ROOT
DEFAULT_OUT_ROOT = REPO_ROOT / "outputs_hcp" / "work" / "group_atlas" / "routea_weighted_fisherz_full_k3"
DEFAULT_ROUTEA_ROOT = REPO_ROOT / "outputs_hcp" / "inputs" / "hcp" / "kong2022" / "shared_inputs" / "routea"
DEFAULT_REFERENCE_ROOT = DEFAULT_ROUTEA_ROOT if DEFAULT_ROUTEA_ROOT.exists() else DEFAULT_KONG2022_ROUTEA_ROOT
DEFAULT_ROUTEA_SURFACE_TIMESERIES_ROOT = REPO_ROOT / "outputs_hcp" / "inputs" / "hcp" / "kong2019_fast_pfm_t1only" / "shared_inputs" / "pfm_151"
DEFAULT_SURFACE_ROOT = DEFAULT_ROUTEA_SURFACE_TIMESERIES_ROOT if DEFAULT_ROUTEA_SURFACE_TIMESERIES_ROOT.exists() else DEFAULT_KONG2022_SURFACE_ROOT
DEFAULT_PRESENT_DIR = DEFAULT_OVERVIEWS_DIR / "weighted_fisherz"
DEFAULT_STYLE = REPO_ROOT / "config" / "hipp_network_style.json"
DEFAULT_TEMPLATE_SURFACE_DIR = DEFAULT_KONG2022_TEMPLATE_ROOT / "group_average_surfaces"
DEFAULT_TEMPLATE_SPEC = REPO_ROOT / "outputs_hcp" / "inputs" / "hcp" / "hippunfold" / "dense_corobl_batch" / "sub-100610" / "hippunfold" / "sub-100610" / "surf" / "sub-100610_den-512_label-hipp_surfaces.spec"
METHODS = ("gradient-spectral", "wta-top")
SMOOTHINGS = ("2mm", "4mm")
HEMIS = ("L", "R")
ATLAS = "kong2022"
COVERAGE_TAG = "cov_0p80"
WEIGHTED_VARIANT = "weighted_fisherz"


def read_tsv(path: Path) -> list[dict[str, object]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return [dict(row) for row in csv.DictReader(f, delimiter="\t")]


def write_tsv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter="\t", extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def copy_run_tree(src_run: Path, dst_run: Path) -> None:
    if not src_run.exists():
        raise FileNotFoundError(f"Missing source run directory: {src_run}")
    dst_run.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(src_run, dst_run, dirs_exist_ok=True)


def reference_paths(reference_root: Path, subject: str) -> tuple[Path, Path]:
    base = reference_root / f"sub-{subject}" / "reference" / ATLAS
    return base / "cortex_roi_parcel_timeseries.npy", base / "cortex_roi_parcels.tsv"


def surface_timeseries_path(surface_root: Path, subject: str, smoothing: str, hemi: str) -> Path:
    return surface_root / f"sub-{subject}" / "surface" / smoothing / f"sub-{subject}_hemi-{hemi}_timeseries.npy"


def load_subject_weighted_vertex_profile(
    surface_root: Path,
    reference_root: Path,
    subject: str,
    smoothing: str,
    hemi: str,
    network_names: list[str],
) -> np.ndarray:
    ts_path = surface_timeseries_path(surface_root, subject, smoothing, hemi)
    roi_path, rows_path = reference_paths(reference_root, subject)
    if not ts_path.exists():
        raise FileNotFoundError(f"Missing hippocampal surface timeseries for strict weighted_fisherz: {ts_path}")
    ts = np.load(ts_path).astype(np.float32)
    roi_ts = np.load(roi_path).astype(np.float32)
    rows = read_tsv(rows_path)
    if roi_ts.shape[0] != len(rows):
        raise RuntimeError(f"ROI rows mismatch for sub-{subject}: {roi_ts.shape} vs {len(rows)} rows")
    return weighted_network_fisherz_profile_from_timeseries(ts, roi_ts, rows, network_names)


def load_subject_weighted_cluster_profile(
    surface_root: Path,
    reference_root: Path,
    subject: str,
    smoothing: str,
    hemi: str,
    labels: np.ndarray,
    support: np.ndarray,
    network_names: list[str],
) -> np.ndarray:
    ts_path = surface_timeseries_path(surface_root, subject, smoothing, hemi)
    roi_path, rows_path = reference_paths(reference_root, subject)
    if not ts_path.exists():
        raise FileNotFoundError(f"Missing hippocampal surface timeseries for strict weighted_fisherz: {ts_path}")
    ts = np.load(ts_path).astype(np.float32)
    roi_ts = np.load(roi_path).astype(np.float32)
    rows = read_tsv(rows_path)
    if roi_ts.shape[0] != len(rows):
        raise RuntimeError(f"ROI rows mismatch for sub-{subject}: {roi_ts.shape} vs {len(rows)} rows")
    return weighted_cluster_network_fisherz_profile_from_timeseries(labels, support, ts, roi_ts, rows, network_names)


def label_items(network_names: list[str], color_map: dict[str, tuple[int, int, int, int]]) -> list[tuple[int, str, tuple[int, int, int, int]]]:
    items: list[tuple[int, str, tuple[int, int, int, int]]] = [(0, "Null", (158, 158, 158, 255))]
    for idx, name in enumerate(network_names, start=1):
        if name not in color_map:
            raise KeyError(f"Missing network color for {name!r}")
        items.append((idx, name, color_map[name]))
    items.append((len(network_names) + 1, "unassigned", (20, 20, 20, 255)))
    return items


def functional_from_vertex_rows(n_vertices: int, rows: list[dict[str, object]], network_names: list[str]) -> tuple[np.ndarray, dict[str, object]]:
    net_to_key = {str(name): idx for idx, name in enumerate(network_names, start=1)}
    unassigned_key = len(network_names) + 1
    labels = np.zeros((n_vertices,), dtype=np.int32)
    for row in rows:
        vidx = int(row["vertex_index"])
        top1 = str(row.get("top1_network", "unassigned"))
        if str(row.get("status", "unassigned")) == "dominant" and top1 in net_to_key:
            labels[vidx] = int(net_to_key[top1])
        else:
            labels[vidx] = int(unassigned_key)
    label_map = {
        "assignment_mode": "vertex_level",
        "network_names": [str(x) for x in network_names],
        "network_label_keys": {str(name): idx for idx, name in enumerate(network_names, start=1)},
        "unassigned_label_key": int(unassigned_key),
    }
    return labels, label_map


def template_subject_from_spec(template_spec: Path) -> str:
    name = Path(template_spec).name
    token = name.split("_", 1)[0]
    return token if token.startswith("sub-") else "sub-100610"


def normalize_template_paths(
    manifest: dict,
    *,
    template_surface_dir: Path = DEFAULT_TEMPLATE_SURFACE_DIR,
    template_spec: Path = DEFAULT_TEMPLATE_SPEC,
) -> dict:
    hemi = str(manifest.get("hemi", ""))
    surface = Path(template_surface_dir) / f"group-average_hemi-{hemi}_space-corobl_den-512_label-hipp_midthickness.surf.gii"
    if surface.exists():
        surface_resolved = str(surface.resolve())
        manifest["template_left_or_right_surface"] = surface_resolved
        density_evidence = manifest.get("density_evidence")
        if isinstance(density_evidence, dict):
            density_evidence["surface_path"] = surface_resolved
    spec = Path(template_spec)
    if spec.exists():
        manifest["template_subject"] = template_subject_from_spec(spec)
        manifest["template_spec_path"] = str(spec.resolve())
    return manifest


def normalize_copied_manifest_paths(
    run_dir: Path,
    manifest: dict,
    *,
    template_surface_dir: Path = DEFAULT_TEMPLATE_SURFACE_DIR,
    template_spec: Path = DEFAULT_TEMPLATE_SPEC,
) -> dict:
    variants = manifest.get("assignment_variants", {})
    if isinstance(variants, dict):
        for variant, spec in variants.items():
            if not isinstance(spec, dict):
                continue
            spec["cluster_assignment_status"] = str((run_dir / f"cluster_assignment_status_{variant}.json").resolve())
            spec["group_functional_atlas_labels"] = str((run_dir / f"group_functional_atlas_labels_{variant}.npy").resolve())
            spec["group_functional_atlas_label_map"] = str((run_dir / f"group_functional_atlas_label_map_{variant}.json").resolve())
            raw_label = spec.get("template_functional_atlas_label") or spec.get("template_functional_atlas_label_variant")
            if raw_label:
                spec["template_functional_atlas_label"] = str((run_dir / "template" / "workbench_assets" / Path(str(raw_label)).name).resolve())
                spec["template_functional_atlas_label_variant"] = spec["template_functional_atlas_label"]
    raw_cluster = manifest.get("template_cluster_label")
    if raw_cluster:
        manifest["template_cluster_label"] = str((run_dir / "template" / "workbench_assets" / Path(str(raw_cluster)).name).resolve())
    return normalize_template_paths(manifest, template_surface_dir=template_surface_dir, template_spec=template_spec)


def recompute_gradient_weighted_run(
    *,
    dst_run: Path,
    surface_root: Path,
    reference_root: Path,
    color_map: dict[str, tuple[int, int, int, int]],
    template_surface_dir: Path,
    template_spec: Path,
) -> dict[str, object]:
    manifest_path = dst_run / "group_atlas_manifest.json"
    manifest = normalize_copied_manifest_paths(
        dst_run,
        load_json(manifest_path),
        template_surface_dir=template_surface_dir,
        template_spec=template_spec,
    )
    if str(manifest.get("status", "ok")) != "ok":
        raise RuntimeError(f"Cannot recompute non-ok run: {manifest_path}")
    labels = np.load(dst_run / "group_labels_final.npy").astype(np.int32)
    support = np.load(dst_run / "group_support_mask.npy").astype(bool)
    feature_summary = load_json(dst_run / "group_feature_summary.json")
    network_names = [str(x) for x in feature_summary["network_names"]]
    subjects = [str(x).replace("sub-", "") for x in manifest["subjects"]]
    smoothing = str(manifest["smoothing"])
    hemi = str(manifest["hemi"])
    k_final = int(np.max(labels))

    profiles: list[np.ndarray] = []
    for subject in subjects:
        cluster_profile = load_subject_weighted_cluster_profile(
            surface_root,
            reference_root,
            subject,
            smoothing,
            hemi,
            labels,
            support,
            network_names,
        )
        if cluster_profile.shape != (k_final, len(network_names)):
            raise RuntimeError(f"Weighted cluster profile mismatch for sub-{subject}: {cluster_profile.shape}")
        profiles.append(cluster_profile)

    profile = np.nanmean(np.stack(profiles, axis=0), axis=0).astype(np.float32)
    profile[~np.isfinite(profile)] = 0.0
    old_profile = dst_run / "cluster_network_profile.npy"
    if old_profile.exists():
        shutil.copy2(old_profile, dst_run / "cluster_network_profile_roi_equal_fisherz_baseline.npy")
    np.save(dst_run / "cluster_network_profile.npy", profile)
    np.save(dst_run / "cluster_network_profile_weighted_fisherz.npy", profile)

    status_rows, dominant = assignment_status_from_weighted_network_profile(profile, network_names)
    status_payload = {
        "items": status_rows,
        "assignment_metric": "top1_weighted_network_fisher_z",
        "q_threshold": None,
        "top_n_cap": None,
        "network_profile_weighting": "network_timeseries_weighted_by_roi_n_grayordinates_used_then_fisherz_fc",
    }
    write_json(dst_run / "cluster_assignment_status_weighted_fisherz.json", status_payload)
    write_json(
        dst_run / "cluster_dominant_network_weighted_fisherz.json",
        {"dominant": dominant, "network_names": network_names, "assignment_variant": WEIGHTED_VARIANT},
    )

    variants = manifest.setdefault("assignment_variants", {})
    variants[WEIGHTED_VARIANT] = {
        "assignment_metric": "top1_weighted_network_fisher_z",
        "cluster_assignment_status": str((dst_run / "cluster_assignment_status_weighted_fisherz.json").resolve()),
        "group_functional_atlas_labels": str((dst_run / "group_functional_atlas_labels_weighted_fisherz.npy").resolve()),
        "group_functional_atlas_label_map": str((dst_run / "group_functional_atlas_label_map_weighted_fisherz.json").resolve()),
        "template_functional_atlas_label": str((dst_run / "template" / "workbench_assets" / f"weighted_fisherz_{smoothing}_hemi-{hemi}.label.gii").resolve()),
    }
    manifest["default_assignment_variant"] = WEIGHTED_VARIANT
    manifest["cluster_assignment_metric"] = "top1_weighted_network_fisher_z"
    manifest["cluster_assignment_status"] = str((dst_run / "cluster_assignment_status_weighted_fisherz.json").resolve())
    manifest["cluster_assignment_status_weighted_fisherz"] = str((dst_run / "cluster_assignment_status_weighted_fisherz.json").resolve())
    manifest["cluster_network_profile"] = str((dst_run / "cluster_network_profile.npy").resolve())
    manifest["weighted_fisherz_branch"] = {
        "status": "posthoc_recomputed",
        "source_assignment_variant_preserved": "fisherz",
        "network_profile_weighting": "network_timeseries_weighted_by_roi_n_grayordinates_used_then_fisherz_fc",
    }
    write_json(manifest_path, manifest)
    if not materialize_combo(manifest_path, color_map):
        raise RuntimeError(f"Failed to materialize weighted_fisherz labels: {manifest_path}")
    return {"run_dir": str(dst_run), "status": "ok", "method": "gradient-spectral"}


def recompute_wta_weighted_run(
    *,
    dst_run: Path,
    surface_root: Path,
    reference_root: Path,
    color_map: dict[str, tuple[int, int, int, int]],
    template_surface_dir: Path,
    template_spec: Path,
) -> dict[str, object]:
    manifest_path = dst_run / "group_atlas_manifest.json"
    manifest = load_json(manifest_path)
    if str(manifest.get("status", "ok")) != "ok":
        raise RuntimeError(f"Cannot recompute non-ok run: {manifest_path}")
    support = np.load(dst_run / "group_support_mask.npy").astype(bool)
    support_indices = np.flatnonzero(support).astype(np.int32)
    feature_summary = load_json(dst_run / "group_feature_summary.json")
    network_names = [str(x) for x in feature_summary["network_names"]]
    subjects = [str(x).replace("sub-", "") for x in manifest["subjects"]]
    smoothing = str(manifest["smoothing"])
    hemi = str(manifest["hemi"])

    profiles: list[np.ndarray] = []
    for subject in subjects:
        weighted_fc = load_subject_weighted_vertex_profile(surface_root, reference_root, subject, smoothing, hemi, network_names)
        if weighted_fc.shape[0] != support.size:
            raise RuntimeError(f"Weighted FC vertex mismatch for sub-{subject}: {weighted_fc.shape} support={support.size}")
        profiles.append(weighted_fc[support_indices, :].astype(np.float32))

    profile_3d = np.stack(profiles, axis=0).astype(np.float32)
    profile_mean = np.nanmean(profile_3d, axis=0).astype(np.float32)
    rows = assign_weighted_network_profile_rows(
        profile=profile_mean,
        support_indices=support_indices,
        network_names=network_names,
    )
    n_vertices = int(support.size)
    functional_labels, label_map = functional_from_vertex_rows(n_vertices, rows, network_names)
    np.save(dst_run / "group_functional_atlas_labels.npy", functional_labels.astype(np.int32))
    write_json(dst_run / "group_functional_atlas_label_map.json", label_map)
    np.save(dst_run / "vertex_network_profile_weighted_fisherz.npy", profile_mean.astype(np.float32))
    np.save(dst_run / "vertex_network_profile_weighted_fisherz_subjectwise.npy", profile_3d.astype(np.float32))
    np.save(dst_run / "vertex_support_indices.npy", support_indices.astype(np.int32))
    write_json(
        dst_run / "vertex_assignment_status.json",
        {
            "items": rows,
            "assignment_metric": "top1_weighted_network_fisher_z",
            "network_profile_weighting": "network_timeseries_weighted_by_roi_n_grayordinates_used_then_fisherz_fc",
        },
    )

    label_path = Path(str(manifest.get("template_functional_atlas_label", ""))).resolve()
    if not label_path.name:
        label_path = dst_run / "template" / "workbench_assets" / f"group-average_hemi-{hemi}_space-corobl_den-512_label-group_functional_atlas_wta-top_kong2022_{smoothing}_cov-0p80_weighted_fisherz.label.gii"
    else:
        label_path = dst_run / "template" / "workbench_assets" / label_path.name.replace(".label.gii", "_weighted_fisherz.label.gii")
    label_path.parent.mkdir(parents=True, exist_ok=True)
    nib.save(make_named_label_gifti(functional_labels.astype(np.int32), label_items(network_names, color_map)), str(label_path))

    manifest = normalize_template_paths(manifest, template_surface_dir=template_surface_dir, template_spec=template_spec)
    manifest.update(
        {
            "assignment_mode": "vertex_level",
            "confidence_metric": "weighted_fisherz",
            "assignment_metric": "top1_weighted_network_fisher_z",
            "feature_family": "vertex_to_weighted_kong2022_network_timeseries_fc_fisherz",
            "fc_mode": "network_timeseries_weighted_by_roi_n_grayordinates_used_then_fisherz_fc",
            "group_functional_atlas_labels": str((dst_run / "group_functional_atlas_labels.npy").resolve()),
            "group_functional_atlas_label_map": str((dst_run / "group_functional_atlas_label_map.json").resolve()),
            "vertex_assignment_status": str((dst_run / "vertex_assignment_status.json").resolve()),
            "vertex_network_profile": str((dst_run / "vertex_network_profile_weighted_fisherz.npy").resolve()),
            "template_functional_atlas_label": str(label_path.resolve()),
            "weighted_fisherz_branch": {
                "status": "posthoc_recomputed",
                "source_assignment_policy": "topn_count_positive_roi",
                "network_profile_weighting": "network_timeseries_weighted_by_roi_n_grayordinates_used_then_fisherz_fc",
            },
        }
    )
    write_json(manifest_path, manifest)
    return {"run_dir": str(dst_run), "status": "ok", "method": "wta-top"}


def run_dirs_for_method(root: Path, method: str) -> list[Path]:
    if method == "gradient-spectral":
        base = root / method / ATLAS
        out: list[Path] = []
        for smoothing in SMOOTHINGS:
            cov_dir = base / smoothing / COVERAGE_TAG
            if not cov_dir.exists():
                continue
            for comp_dir in sorted(p for p in cov_dir.iterdir() if p.is_dir() and p.name.startswith("g")):
                for hemi in HEMIS:
                    out.append(comp_dir / f"hemi_{hemi}")
        return out
    if method == "wta-top":
        return [root / method / ATLAS / smoothing / COVERAGE_TAG / f"hemi_{hemi}" for smoothing in SMOOTHINGS for hemi in HEMIS]
    raise ValueError(f"Unsupported method: {method}")


def write_comparison_summary(source_root: Path, out_root: Path, present_dir: Path) -> None:
    rows: list[dict[str, object]] = []
    for method in METHODS:
        for new_run in run_dirs_for_method(out_root, method):
            rel = new_run.relative_to(out_root)
            old_run = source_root / rel
            if not old_run.exists() or not (new_run / "group_atlas_manifest.json").exists():
                continue
            new_manifest = load_json(new_run / "group_atlas_manifest.json")
            smoothing = str(new_manifest.get("smoothing", ""))
            hemi = str(new_manifest.get("hemi", ""))
            component = new_run.parent.name if method == "gradient-spectral" else ""
            if method == "gradient-spectral":
                old_labels = np.load(old_run / "group_functional_atlas_labels_fisherz.npy").astype(np.int32)
                new_labels = np.load(new_run / "group_functional_atlas_labels_weighted_fisherz.npy").astype(np.int32)
                old_profile = np.load(new_run / "cluster_network_profile_roi_equal_fisherz_baseline.npy").astype(np.float32)
                new_profile = np.load(new_run / "cluster_network_profile_weighted_fisherz.npy").astype(np.float32)
                changed = (old_labels != new_labels) & (old_labels > 0)
                n_old = int(np.sum(old_labels > 0))
                rows.append(
                    {
                        "method": method,
                        "component": component,
                        "smoothing": smoothing,
                        "hemi": hemi,
                        "n_vertices_compared": n_old,
                        "n_vertices_changed": int(np.sum(changed)),
                        "fraction_changed": float(np.sum(changed) / max(1, n_old)),
                        "mean_abs_profile_delta": float(np.nanmean(np.abs(new_profile - old_profile))),
                        "max_abs_profile_delta": float(np.nanmax(np.abs(new_profile - old_profile))),
                    }
                )
            else:
                old_labels = np.load(old_run / "group_functional_atlas_labels.npy").astype(np.int32)
                new_labels = np.load(new_run / "group_functional_atlas_labels.npy").astype(np.int32)
                mask = old_labels > 0
                changed = (old_labels != new_labels) & mask
                rows.append(
                    {
                        "method": method,
                        "component": component,
                        "smoothing": smoothing,
                        "hemi": hemi,
                        "n_vertices_compared": int(np.sum(mask)),
                        "n_vertices_changed": int(np.sum(changed)),
                        "fraction_changed": float(np.sum(changed) / max(1, int(np.sum(mask)))),
                        "mean_abs_profile_delta": "",
                        "max_abs_profile_delta": "",
                    }
                )
    present_dir.mkdir(parents=True, exist_ok=True)
    fields = ["method", "component", "smoothing", "hemi", "n_vertices_compared", "n_vertices_changed", "fraction_changed", "mean_abs_profile_delta", "max_abs_profile_delta"]
    write_tsv(present_dir / "weighted_fisherz_difference_summary.tsv", rows, fields)
    lines = [
        "# Weighted Fisherz Branch Difference Summary",
        "",
        f"Baseline: `{source_root}`",
        f"Weighted branch: `{out_root}`",
        "",
        "This branch preserves the original `fisherz` outputs and adds/recomputes `weighted_fisherz` comparison labels from vertex-count weighted network-timeseries Fisher-z FC profiles.",
        "",
        "| method | component | smoothing | hemi | vertices | changed | fraction changed | mean abs profile delta |",
        "|---|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        delta = row["mean_abs_profile_delta"]
        delta_text = "" if delta == "" else f"{float(delta):.6f}"
        lines.append(
            f"| {row['method']} | {row.get('component', '')} | {row['smoothing']} | {row['hemi']} | {row['n_vertices_compared']} | "
            f"{row['n_vertices_changed']} | {float(row['fraction_changed']):.6f} | {delta_text} |"
        )
    lines.extend(
        [
            "",
            "Rendered overview products are listed in `group_overview_weighted_fisherz_manifest.json` after running `06_01_build_group_overview.py`.",
        ]
    )
    (present_dir / "weighted_fisherz_difference_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_branch(args: argparse.Namespace) -> dict[str, object]:
    source_root = Path(args.source_root).resolve()
    out_root = Path(args.out_root).resolve()
    surface_root = Path(args.surface_root).resolve()
    reference_root = Path(args.reference_root).resolve()
    present_dir = Path(args.present_dir).resolve()
    template_surface_dir = Path(args.template_surface_dir).resolve()
    template_spec = Path(args.template_spec).resolve()
    color_map = load_network_colors(Path(args.network_style).resolve())
    methods = tuple(str(method) for method in args.methods)

    results: list[dict[str, object]] = []
    for method in methods:
        for src_run in run_dirs_for_method(source_root, method):
            if not src_run.exists():
                continue
            dst_run = out_root / src_run.relative_to(source_root)
            copy_run_tree(src_run, dst_run)
            if method == "gradient-spectral":
                results.append(
                    recompute_gradient_weighted_run(
                        dst_run=dst_run,
                        surface_root=surface_root,
                        reference_root=reference_root,
                        color_map=color_map,
                        template_surface_dir=template_surface_dir,
                        template_spec=template_spec,
                    )
                )
            else:
                results.append(
                    recompute_wta_weighted_run(
                        dst_run=dst_run,
                        surface_root=surface_root,
                        reference_root=reference_root,
                        color_map=color_map,
                        template_surface_dir=template_surface_dir,
                        template_spec=template_spec,
                    )
                )
    write_comparison_summary(source_root, out_root, present_dir)
    manifest = {
        "status": "ok",
        "branch": "weighted_fisherz",
        "source_root": str(source_root),
        "out_root": str(out_root),
        "surface_root": str(surface_root),
        "surface_root_note": "strict weighted_fisherz uses hippocampal surface timeseries plus ROI timeseries weighted by n_grayordinates_used",
        "reference_root": str(reference_root),
        "present_dir": str(present_dir),
        "template_surface_dir": str(template_surface_dir),
        "template_spec": str(template_spec),
        "methods": list(methods),
        "runs": results,
    }
    write_json(out_root / "weighted_fisherz_branch_manifest.json", manifest)
    return manifest


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build RouteA weighted_fisherz comparison branch from existing HCP group-atlas outputs.")
    parser.add_argument("--source-root", default=str(DEFAULT_SOURCE_ROOT))
    parser.add_argument("--out-root", default=str(DEFAULT_OUT_ROOT))
    parser.add_argument("--surface-root", default=str(DEFAULT_SURFACE_ROOT), help="Subject shared-input root containing sub-*/surface/<smoothing> hippocampal timeseries for strict weighted_fisherz recomputation.")
    parser.add_argument("--reference-root", default=str(DEFAULT_REFERENCE_ROOT))
    parser.add_argument("--present-dir", default=str(DEFAULT_PRESENT_DIR))
    parser.add_argument("--network-style", default=str(DEFAULT_STYLE))
    parser.add_argument(
        "--methods",
        nargs="+",
        choices=METHODS,
        default=list(METHODS),
        help=(
            "Weighted branches to derive. Defaults to both gradient-spectral "
            "and wta-top for backward compatibility."
        ),
    )
    parser.add_argument("--template-surface-dir", default=str(DEFAULT_TEMPLATE_SURFACE_DIR), help="Directory containing group-average hemi-L/R midthickness surfaces used in copied manifests.")
    parser.add_argument("--template-spec", default=str(DEFAULT_TEMPLATE_SPEC), help="Template subject HippUnfold spec path used in copied manifests.")
    return parser


def main() -> int:
    args = build_arg_parser().parse_args()
    manifest = build_branch(args)
    print(json.dumps({"status": manifest["status"], "out_root": manifest["out_root"], "runs": len(manifest["runs"])}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
