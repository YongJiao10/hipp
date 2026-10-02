"""Unified Stanford Kember article-metric feature extraction.

Functional metrics remain on their consumed 512-source time-series surface.
System area is independently aggregated from aligned native labels and the
native HippUnfold surfareacorobl metric.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

try:  # Imports work both as a package and from ``python src/...py``.
    from .inputs import (
        DEFAULT_CONFIG_PATH,
        WORKFLOW_ROOT,
        REPO_ROOT,
        _artifact_paths,
        _load_config,
        build_factorial_roster,
        validate_rest_only_path,
    )
    from .surface_area import compute_system_surface_area
    from .native_label_alignment import (
        _prepare_branch_input,
        _resolve_workbench_command,
        align_labels_to_native,
    )
except ImportError:  # pragma: no cover - direct source-directory execution
    from inputs import (
        DEFAULT_CONFIG_PATH,
        WORKFLOW_ROOT,
        REPO_ROOT,
        _artifact_paths,
        _load_config,
        build_factorial_roster,
        validate_rest_only_path,
    )
    from surface_area import compute_system_surface_area
    from native_label_alignment import (
        _prepare_branch_input,
        _resolve_workbench_command,
        align_labels_to_native,
    )

try:
    from .surface_geometry import load_midthickness, surface_adjacency
except ImportError:  # pragma: no cover - direct source-directory execution
    from surface_geometry import load_midthickness, surface_adjacency


SYSTEM_LABELS = (1, 2, 3)
SYSTEM_NAMES = {1: "anterior", 2: "body", 3: "posterior"}
METHODS = ("hcp151_template_matching", "hcp151_fixed")
BOUNDARY_RADIUS_MM = 4.0
PRIMARY_METRIC_FIELDS = (
    "surface_area_mm2",
    "functional_segregation_d",
    "boundary_sharpness_d",
)
KEMBER_FEATURE_COLUMNS = [
    "subject",
    "trauma",
    "symptomatic",
    "four_group",
    "method",
    "smoothing_mm",
    "hemisphere",
    "system",
    "system_label",
    "status",
    "reason",
    "endpoint_missing_fields",
    "valid_vertex_count",
    "source_vertex_count",
    "native_vertex_count",
    "total_surface_area_mm2",
    "total_valid_surface_area_mm2",
    "total_labeled_area_mm2",
    "unassigned_surface_area_mm2",
    "surface_area_mm2",
    "surface_area_fraction",
    "vertex_area_source",
    "vertex_area_metric_path",
    "vertex_area_convention",
    "functional_segregation_d",
    "boundary_sharpness_d",
    "boundary_vertex_count",
    "boundary_vertex_valid_count",
    "boundary_radius_mm",
    "boundary_distance_convention",
    "source_timeseries_path",
    "source_label_path",
    "source_surface_path",
    "native_surface_path",
    "native_labels_path",
    "source_density_evidence",
    "native_density_evidence",
    "label_transfer_method",
    "label_transfer_provenance",
    "native_area_metric_sha256",
    "input_hashes",
    "metric_provenance",
]
KEMBER_STATUS_COLUMNS = [
    "subject",
    "trauma",
    "symptomatic",
    "four_group",
    "method",
    "smoothing_mm",
    "hemisphere",
    "feature_status",
    "missing_components",
    "reason",
    "endpoint_missing_fields",
    "source_timeseries_path",
    "source_label_path",
    "source_surface_path",
    "native_surface_path",
    "vertex_area_metric_path",
    "label_transfer_method",
]


def _finite_pooled_cohens_d(inside: np.ndarray, outside: np.ndarray) -> float:
    """Return signed pooled-SD Cohen's *d* for two finite value sets."""

    left = np.asarray(inside, dtype=np.float64)
    right = np.asarray(outside, dtype=np.float64)
    left = left[np.isfinite(left)]
    right = right[np.isfinite(right)]
    if left.size < 2 or right.size < 2:
        return float("nan")
    left_variance = float(np.var(left, ddof=1))
    right_variance = float(np.var(right, ddof=1))
    denominator = int(left.size + right.size - 2)
    if denominator <= 0:
        return float("nan")
    pooled_variance = (
        ((left.size - 1) * left_variance + (right.size - 1) * right_variance)
        / denominator
    )
    if not np.isfinite(pooled_variance) or pooled_variance <= np.finfo(float).eps:
        return float("nan")
    return float((np.mean(left) - np.mean(right)) / np.sqrt(pooled_variance))


def _vertex_correlations(
    time_series: np.ndarray,
    valid_mask: np.ndarray,
    target_mask: np.ndarray,
) -> np.ndarray:
    """Correlate each valid vertex BOLD series with a target-system mean BOLD."""

    values = np.asarray(time_series, dtype=np.float64)
    if values.ndim != 2:
        raise ValueError(f"time_series must be two-dimensional, got {values.shape}")
    if values.shape[0] != valid_mask.size:
        raise ValueError("time_series must be vertex by timepoint")
    if values.shape[1] < 2:
        raise ValueError("time_series needs at least two timepoints")
    eligible_target = valid_mask & target_mask
    if not np.any(eligible_target):
        return np.full(valid_mask.size, np.nan, dtype=np.float64)
    target_series = np.mean(values[eligible_target], axis=0)
    target_centered = target_series - np.mean(target_series)
    target_norm = float(np.linalg.norm(target_centered))
    correlations = np.full(valid_mask.size, np.nan, dtype=np.float64)
    if not np.isfinite(target_norm) or target_norm <= np.finfo(float).eps:
        return correlations
    eligible_values = values[valid_mask]
    centered = eligible_values - np.mean(eligible_values, axis=1, keepdims=True)
    norms = np.linalg.norm(centered, axis=1)
    usable = np.isfinite(norms) & (norms > np.finfo(float).eps)
    values_out = np.full(eligible_values.shape[0], np.nan, dtype=np.float64)
    values_out[usable] = (
        centered[usable] @ target_centered / (norms[usable] * target_norm)
    )
    correlations[valid_mask] = np.clip(values_out, -1.0, 1.0)
    return correlations


def _target_boundary_vertices(
    labels: np.ndarray,
    valid_mask: np.ndarray,
    faces: np.ndarray,
    target_label: int,
) -> np.ndarray:
    """Return target vertices with a mesh neighbor belonging to another system."""

    adjacency = surface_adjacency(faces, labels.size)
    target = valid_mask & labels.eq(target_label) if isinstance(labels, pd.Series) else valid_mask & (labels == target_label)
    vertices: list[int] = []
    for vertex in np.flatnonzero(target):
        neighbors = adjacency.indices[adjacency.indptr[vertex] : adjacency.indptr[vertex + 1]]
        if np.any(valid_mask[neighbors] & (labels[neighbors] != target_label)):
            vertices.append(int(vertex))
    return np.asarray(vertices, dtype=np.int64)


def compute_kember_metrics(
    *,
    labels: np.ndarray,
    time_series: np.ndarray,
    native_vertex_area_mm2: np.ndarray,
    native_valid_mask: np.ndarray,
    coordinates: np.ndarray,
    faces: np.ndarray,
    vertex_area_metric_path: str | Path,
    native_labels: np.ndarray | None = None,
    boundary_radius_mm: float = BOUNDARY_RADIUS_MM,
) -> dict[str, dict[str, Any]]:
    """Compute Kember system area, segregation, and boundary sharpness.

    The article describes per-system area as the sum of HippUnfold vertex
    areas, segregation as target-versus-rest vertexwise BOLD-correlation
    Cohen's *d*, and sharpness as that same contrast locally averaged over
    target-system boundary vertices within a 4-mm physical neighborhood.
    Each hemisphere is intentionally computed separately.
    """

    system_labels = np.asarray(labels, dtype=np.int32).reshape(-1)
    values = np.asarray(time_series, dtype=np.float64)
    points = np.asarray(coordinates, dtype=np.float64)
    triangles = np.asarray(faces, dtype=np.int64)
    if boundary_radius_mm <= 0:
        raise ValueError("boundary_radius_mm must be positive")
    if values.ndim != 2 or values.shape[0] != system_labels.size:
        raise ValueError("time_series must have one row per label vertex")
    if points.shape != (system_labels.size, 3):
        raise ValueError("coordinates must have shape (vertices, 3)")
    if not np.isfinite(points).all():
        raise ValueError("coordinates must be finite")

    native_system_labels = (
        system_labels if native_labels is None else np.asarray(native_labels)
    )
    area_summary = compute_system_surface_area(
        np.asarray(native_vertex_area_mm2),
        native_system_labels,
        valid_mask=np.asarray(native_valid_mask),
    )

    label_valid = np.isin(system_labels, SYSTEM_LABELS)
    finite_series = np.isfinite(values).all(axis=1)
    functional_valid = label_valid & finite_series
    tree = cKDTree(points)
    result: dict[str, dict[str, Any]] = {}
    for label in SYSTEM_LABELS:
        system_name = SYSTEM_NAMES[label]
        target = label_valid & (system_labels == label)
        area_record = area_summary["systems"][system_name]
        correlations = _vertex_correlations(values, functional_valid, target)
        inside = correlations[functional_valid & (system_labels == label)]
        outside = correlations[functional_valid & (system_labels != label)]
        segregation = _finite_pooled_cohens_d(inside, outside)
        border_vertices = _target_boundary_vertices(
            system_labels, label_valid, triangles, label
        )
        local_effects: list[float] = []
        for vertex in border_vertices:
            nearby = np.asarray(
                tree.query_ball_point(points[vertex], r=float(boundary_radius_mm)),
                dtype=np.int64,
            )
            nearby = nearby[functional_valid[nearby]]
            local_inside = correlations[nearby[system_labels[nearby] == label]]
            local_outside = correlations[nearby[system_labels[nearby] != label]]
            local_d = _finite_pooled_cohens_d(local_inside, local_outside)
            if np.isfinite(local_d):
                local_effects.append(float(local_d))
        result[system_name] = {
            "valid_vertex_count": int(np.sum(label_valid)),
            "source_vertex_count": int(system_labels.size),
            "native_vertex_count": int(np.asarray(native_vertex_area_mm2).size),
            **area_record,
            "vertex_area_metric_path": str(vertex_area_metric_path),
            "functional_segregation_d": segregation,
            "boundary_sharpness_d": (
                float(np.mean(local_effects)) if local_effects else float("nan")
            ),
            "boundary_vertex_count": int(border_vertices.size),
            "boundary_vertex_valid_count": int(len(local_effects)),
        }
    return result


def _coerce_roster(
    roster: pd.DataFrame | Iterable[Mapping[str, Any]] | None,
    config: Mapping[str, Any],
) -> pd.DataFrame:
    result = build_factorial_roster(dict(config)) if roster is None else pd.DataFrame(roster).copy()
    required = {"subject", "trauma", "symptomatic", "four_group"}
    missing = sorted(required - set(result.columns))
    if missing:
        raise ValueError(f"roster missing required columns: {missing}")
    result = result.loc[:, ["subject", "trauma", "symptomatic", "four_group"]].copy()
    result["subject"] = result["subject"].astype(str)
    if result["subject"].duplicated().any():
        raise ValueError("roster contains duplicate subjects")
    result["trauma"] = result["trauma"].astype(int)
    result["symptomatic"] = result["symptomatic"].astype(int)
    return result.sort_values(
        "subject", key=lambda values: values.str.extract(r"(\d+)$")[0].astype(int)
    ).reset_index(drop=True)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _metadata(paths: Mapping[str, Path], method: str) -> dict[str, str]:
    label_key = "matching_labels" if method == "hcp151_template_matching" else "fixed_labels"
    return {
        "source_timeseries_path": str(paths["surface_timeseries"]),
        "source_label_path": str(paths[label_key]),
        "source_surface_path": str(paths["surface_gifti"]),
        "native_surface_path": "",
        "native_labels_path": "",
        "vertex_area_metric_path": "",
        "source_density_evidence": "",
        "native_density_evidence": "",
        "label_transfer_method": "ADAP_BARY_AREA",
        "label_transfer_provenance": "",
        "native_area_metric_sha256": "",
        "input_hashes": "",
        "metric_provenance": (
            "native HippUnfold surfareacorobl area plus 512-source functional "
            "segregation and boundary sharpness"
        ),
    }


def _missing_fields(row: Mapping[str, Any]) -> str:
    return ",".join(
        field
        for field in PRIMARY_METRIC_FIELDS
        if not np.isfinite(pd.to_numeric(pd.Series([row.get(field)]), errors="coerce").iloc[0])
    )


def _empty_rows(
    record: Mapping[str, Any],
    method: str,
    smoothing_mm: int,
    hemisphere: str,
    *,
    status: str,
    reason: str,
    metadata: Mapping[str, str],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for label in SYSTEM_LABELS:
        rows.append(
            {
                "subject": str(record["subject"]),
                "trauma": int(record["trauma"]),
                "symptomatic": int(record["symptomatic"]),
                "four_group": str(record["four_group"]),
                "method": method,
                "smoothing_mm": int(smoothing_mm),
                "hemisphere": hemisphere,
                "system": SYSTEM_NAMES[label],
                "system_label": label,
                "status": status,
                "reason": reason,
                "endpoint_missing_fields": ",".join(PRIMARY_METRIC_FIELDS),
                "valid_vertex_count": np.nan,
                "source_vertex_count": np.nan,
                "native_vertex_count": np.nan,
                "total_surface_area_mm2": np.nan,
                "total_valid_surface_area_mm2": np.nan,
                "total_labeled_area_mm2": np.nan,
                "unassigned_surface_area_mm2": np.nan,
                "surface_area_mm2": np.nan,
                "surface_area_fraction": np.nan,
                "vertex_area_source": "hippunfold_native_surfareacorobl",
                "vertex_area_convention": (
                    "HippUnfold native per-vertex surface-area values in mm^2 "
                    "summed over vertices selected by the native system label"
                ),
                "functional_segregation_d": np.nan,
                "boundary_sharpness_d": np.nan,
                "boundary_vertex_count": np.nan,
                "boundary_vertex_valid_count": np.nan,
                "boundary_radius_mm": BOUNDARY_RADIUS_MM,
                "boundary_distance_convention": "source_512_midthickness_euclidean_radius_mm",
                **metadata,
            }
        )
    return rows


def _compute_combo(
    config: Mapping[str, Any],
    record: Mapping[str, Any],
    method: str,
    smoothing_mm: int,
    hemisphere: str,
    *,
    alignment_output_dir: Path,
    workbench_command: Path,
) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
    subject = str(record["subject"])
    paths = _artifact_paths(dict(config), subject, int(smoothing_mm), hemisphere)
    metadata = _metadata(paths, method)
    required = ("surface_timeseries",)
    missing = [key for key in required if not paths[key].is_file()]
    if missing:
        reason = "missing_components=" + ",".join(missing)
        return _empty_rows(record, method, smoothing_mm, hemisphere, status="missing", reason=reason, metadata=metadata), {
            "subject": subject,
            "trauma": int(record["trauma"]),
            "symptomatic": int(record["symptomatic"]),
            "four_group": str(record["four_group"]),
            "method": method,
            "smoothing_mm": int(smoothing_mm),
            "hemisphere": hemisphere,
            "feature_status": "missing",
            "missing_components": ",".join(missing),
            "reason": reason,
            "endpoint_missing_fields": ",".join(PRIMARY_METRIC_FIELDS),
            **metadata,
        }, {}
    try:
        entry = _prepare_branch_input(
            root=REPO_ROOT,
            config=config,
            branch=method,
            subject=subject,
            smoothing_mm=int(smoothing_mm),
            hemisphere=hemisphere,
        )
        coordinates, faces = load_midthickness(Path(entry["source_surface_path"]))
        labels = np.asarray(entry["labels"], dtype=np.int32)
        source_valid = np.asarray(entry["valid_mask"], dtype=bool)
        time_series = np.asarray(
            np.load(paths["surface_timeseries"], allow_pickle=False), dtype=np.float64
        )
        if labels.size != coordinates.shape[0] or source_valid.size != labels.size:
            raise ValueError("source labels/valid mask do not match the consumed 512 surface")
        labels = np.where(source_valid & np.isin(labels, SYSTEM_LABELS), labels, 0)
        native_labels, native_valid, transfer = align_labels_to_native(
            entry,
            output_dir=alignment_output_dir,
            workbench_command=workbench_command,
        )
        metrics = compute_kember_metrics(
            labels=labels,
            time_series=time_series,
            native_vertex_area_mm2=np.asarray(entry["vertex_area_mm2"]),
            native_labels=native_labels,
            native_valid_mask=native_valid,
            coordinates=coordinates,
            faces=faces,
            vertex_area_metric_path=Path(entry["native_area_metric_path"]),
            boundary_radius_mm=BOUNDARY_RADIUS_MM,
        )
        consumed = entry["consumed_input"]
        label_metadata = consumed["labels"]
        valid_metadata = (
            consumed["valid_mask"]
            if method == "hcp151_template_matching"
            else consumed["support"]
        )
        native_metric = consumed["native_area_metric"]
        source_surface = consumed["label_source_surface"]
        native_surface = consumed["native_area_surface"]
        input_hashes = {
            "source_timeseries": _sha256(paths["surface_timeseries"]),
            "source_labels": str(label_metadata["sha256"]),
            "source_valid_mask": str(valid_metadata["sha256"]),
            "source_surface": str(source_surface["surface_fingerprint"]["sha256"]),
            "native_surface": str(native_surface["surface_fingerprint"]["sha256"]),
            "native_area_metric": str(native_metric["sha256"]),
        }
        metadata.update(
            {
                "source_label_path": str(label_metadata["path"]),
                "source_surface_path": str(entry["source_surface_path"]),
                "native_surface_path": str(entry["surface_path"]),
                "native_labels_path": str(transfer["native_labels"]["path"]),
                "vertex_area_metric_path": str(entry["native_area_metric_path"]),
                "source_density_evidence": json.dumps(
                    entry["source_density_evidence"], sort_keys=True
                ),
                "native_density_evidence": json.dumps(
                    entry["density_evidence"], sort_keys=True
                ),
                "label_transfer_provenance": json.dumps(transfer, sort_keys=True),
                "native_area_metric_sha256": str(native_metric["sha256"]),
                "input_hashes": json.dumps(input_hashes, sort_keys=True),
            }
        )
        rows: list[dict[str, Any]] = []
        for label in SYSTEM_LABELS:
            system = SYSTEM_NAMES[label]
            row: dict[str, Any] = {
                "subject": subject,
                "trauma": int(record["trauma"]),
                "symptomatic": int(record["symptomatic"]),
                "four_group": str(record["four_group"]),
                "method": method,
                "smoothing_mm": int(smoothing_mm),
                "hemisphere": hemisphere,
                "system": system,
                "system_label": label,
                "status": "success",
                "reason": "",
                **metrics[system],
                "boundary_radius_mm": BOUNDARY_RADIUS_MM,
                "boundary_distance_convention": "source_512_midthickness_euclidean_radius_mm",
                **metadata,
            }
            row["endpoint_missing_fields"] = _missing_fields(row)
            rows.append(row)
        missing_fields = ",".join(
            sorted(
                {
                    field
                    for row in rows
                    for field in str(row["endpoint_missing_fields"]).split(",")
                    if field
                }
            )
        )
        return rows, {
            "subject": subject,
            "trauma": int(record["trauma"]),
            "symptomatic": int(record["symptomatic"]),
            "four_group": str(record["four_group"]),
            "method": method,
            "smoothing_mm": int(smoothing_mm),
            "hemisphere": hemisphere,
            "feature_status": "success",
            "missing_components": "",
            "reason": "",
            "endpoint_missing_fields": missing_fields,
            **metadata,
        }, {
            "method": method,
            "subject": subject,
            "smoothing_mm": int(smoothing_mm),
            "hemisphere": hemisphere,
            "native_area_metric": native_metric,
            "consumed_native_surface": native_surface,
            "native_labels": transfer["native_labels"],
            "label_transfer": transfer,
            "density_evidence": entry["density_evidence"],
            "source_functional_inputs": {
                "timeseries_path": str(paths["surface_timeseries"]),
                "labels": label_metadata,
                "valid_mask": valid_metadata,
                "surface": source_surface,
                "source_vertex_count": int(labels.size),
            },
            "input_hashes": input_hashes,
        }
    except Exception as exc:  # Record a per-combination failure without abandoning the run.
        reason = f"{type(exc).__name__}: {exc}"
        status_name = "missing" if isinstance(exc, FileNotFoundError) else "failed"
        return _empty_rows(record, method, smoothing_mm, hemisphere, status=status_name, reason=reason, metadata=metadata), {
            "subject": subject,
            "trauma": int(record["trauma"]),
            "symptomatic": int(record["symptomatic"]),
            "four_group": str(record["four_group"]),
            "method": method,
            "smoothing_mm": int(smoothing_mm),
            "hemisphere": hemisphere,
            "feature_status": "failed",
            "missing_components": "",
            "reason": reason,
            "endpoint_missing_fields": ",".join(PRIMARY_METRIC_FIELDS),
            **metadata,
        }, {}


def _build_feature_outputs(
    config: Mapping[str, Any] | str | Path,
    *,
    alignment_output_dir: Path,
    workbench_command: str | Path | None,
    roster: pd.DataFrame | Iterable[Mapping[str, Any]] | None = None,
    workers: int = 1,
) -> tuple[pd.DataFrame, pd.DataFrame, list[dict[str, Any]]]:
    """Build combined features, statuses, and native provenance bundles."""

    config_dict = _load_config(Path(config)) if isinstance(config, (str, Path)) else dict(config)
    for value in config_dict.get("paths", {}).values():
        validate_rest_only_path(Path(value).expanduser())
    roster_table = _coerce_roster(roster, config_dict)
    methods = tuple(str(value) for value in config_dict.get("methods", ()))
    if methods != METHODS:
        raise ValueError(f"active Kember methods must be exactly {METHODS}")
    smoothings = tuple(int(value) for value in config_dict.get("smoothings_mm", (2, 4)))
    hemispheres = tuple(str(value) for value in config_dict.get("hemispheres", ("L", "R")))
    resolved_workbench = _resolve_workbench_command(workbench_command)
    jobs = [
        (record, method, smoothing, hemisphere)
        for record in roster_table.to_dict("records")
        for method in methods
        for smoothing in smoothings
        for hemisphere in hemispheres
    ]
    worker_count = max(1, min(int(workers), 4))
    if worker_count == 1:
        completed = [
            _compute_combo(
                config_dict,
                *job,
                alignment_output_dir=alignment_output_dir,
                workbench_command=resolved_workbench,
            )
            for job in jobs
        ]
    else:
        with ThreadPoolExecutor(max_workers=worker_count) as executor:
            completed = list(
                executor.map(
                    lambda job: _compute_combo(
                        config_dict,
                        *job,
                        alignment_output_dir=alignment_output_dir,
                        workbench_command=resolved_workbench,
                    ),
                    jobs,
                )
            )
    rows: list[dict[str, Any]] = []
    statuses: list[dict[str, Any]] = []
    provenance: list[dict[str, Any]] = []
    for combo_rows, status, bundle in completed:
        rows.extend(combo_rows)
        statuses.append(status)
        if bundle:
            provenance.append(bundle)
    return (
        pd.DataFrame(rows).reindex(columns=KEMBER_FEATURE_COLUMNS),
        pd.DataFrame(statuses).reindex(columns=KEMBER_STATUS_COLUMNS),
        provenance,
    )


def build_kember_feature_table(
    config: Mapping[str, Any] | str | Path,
    *,
    output_dir: str | Path,
    workbench_command: str | Path | None = None,
    roster: pd.DataFrame | Iterable[Mapping[str, Any]] | None = None,
    workers: int = 1,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return one combined two-method feature table and status table."""

    table, status, _ = _build_feature_outputs(
        config,
        alignment_output_dir=Path(output_dir),
        workbench_command=workbench_command,
        roster=roster,
        workers=workers,
    )
    return table, status


def run_feature_build(
    config_path: Mapping[str, Any] | str | Path,
    output_dir: str | Path,
    *,
    workbench_command: str | Path | None = None,
    roster: pd.DataFrame | Iterable[Mapping[str, Any]] | None = None,
    workers: int = 4,
) -> Path:
    """Write the stable combined Kember feature artifacts."""

    config_file = Path(config_path) if isinstance(config_path, (str, Path)) else None
    config_dict = _load_config(config_file) if config_file is not None else dict(config_path)
    destination = Path(output_dir).expanduser().resolve()
    validate_rest_only_path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    table, status, provenance = _build_feature_outputs(
        config_dict,
        alignment_output_dir=destination,
        workbench_command=workbench_command,
        roster=roster,
        workers=max(1, min(int(workers), 4)),
    )
    feature_path = destination / "subject_features.tsv"
    status_path = destination / "feature_status.tsv"
    table.to_csv(feature_path, sep="\t", index=False, na_rep="")
    status.to_csv(status_path, sep="\t", index=False, na_rep="")
    manifest = {
        "task": "Stanford unified Kember article-metric feature extraction",
        "scope": "Stanford rest-fMRI only",
        "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "workers": max(1, min(int(workers), 4)),
        "methods": list(METHODS),
        "smoothings_mm": [int(value) for value in config_dict.get("smoothings_mm", (2, 4))],
        "hemispheres": [str(value) for value in config_dict.get("hemispheres", ("L", "R"))],
        "metrics": {
            "surface_area_mm2": "sum of native HippUnfold surfareacorobl values selected by aligned native system labels",
            "functional_segregation_d": "signed pooled-SD Cohen's d of target-system versus other-system vertexwise BOLD correlations",
            "boundary_sharpness_d": "mean signed local pooled-SD Cohen's d within a 4-mm source-surface neighborhood of target-system boundary vertices",
        },
        "surface_area_fraction": "contextual normalization only; not an additional direct-comparison endpoint",
        "boundary_radius_mm": BOUNDARY_RADIUS_MM,
        "boundary_distance_convention": "source_512_midthickness_euclidean_radius_mm",
        "area_source": "hippunfold_native_surfareacorobl",
        "source_functional_density": 512,
        "native_provenance": provenance,
        "feature_table_schema": KEMBER_FEATURE_COLUMNS,
        "feature_status_schema": KEMBER_STATUS_COLUMNS,
        "feature_rows": int(len(table)),
        "status_rows": int(len(status)),
        "status_counts": {
            str(name): int(count) for name, count in table["status"].value_counts(dropna=False).items()
        },
        "outputs": {"subject_features": str(feature_path), "feature_status": str(status_path)},
    }
    if config_file is not None and config_file.is_file():
        manifest["config_path"] = str(config_file.resolve())
        manifest["config_sha256"] = _sha256(config_file)
    (destination / "feature_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return destination


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--output-dir", type=Path, default=WORKFLOW_ROOT / "work" / "features")
    parser.add_argument("--workbench-command", type=Path, default=None)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args(argv)
    print(
        run_feature_build(
            args.config,
            args.output_dir,
            workbench_command=args.workbench_command,
            workers=args.workers,
        )
    )
    return 0


if __name__ == "__main__":  # pragma: no cover - command-line path
    raise SystemExit(main())
