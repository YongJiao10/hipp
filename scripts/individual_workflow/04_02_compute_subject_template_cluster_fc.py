#!/usr/bin/env python3
"""Compute subject-specific template-matched hippocampal cluster FC.

The template-matching labels are individual outputs.  For each subject and
hemisphere this script therefore averages that subject's 2 mm hippocampal
surface time series within labels 1, 2, and 3, and only then correlates the
three cluster time series with the subject's 400 Kong2022 ROI time series.
Existing vertex-to-ROI FC matrices are never used as a substitute input.
"""

from __future__ import annotations

import argparse
import csv
from copy import deepcopy
import hashlib
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


NATIVE_THREAD_ENV_KEYS = (
    "CORE_WORKFLOW_WB_THREADS",
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS",
)
for _thread_env_key in NATIVE_THREAD_ENV_KEYS:
    os.environ[_thread_env_key] = "1"
del _thread_env_key

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
CORE_WORKFLOW_DIR = REPO_ROOT / "scripts" / "core_workflow"
if str(CORE_WORKFLOW_DIR) not in sys.path:
    sys.path.insert(0, str(CORE_WORKFLOW_DIR))

from lib.fc_math import corrcoef_rows, fisher_z_transform_fc


N_VERTICES = 512
N_ROIS = 400
CLUSTER_IDS = (1, 2, 3)
HEMIS = ("L", "R")
SMOOTHING = "2mm"
LABEL_SEMANTICS = {
    "1": "Anterior/VAN",
    "2": "Body/SMN",
    "3": "Posterior/DMN",
}
WORKFLOW = "stanford_hcp151_template_matched_cluster_mean_fc"
ARTIFACT_CONTRACT_SCHEMA_VERSION = 1

DEFAULT_MATCH_ROOT = (
    REPO_ROOT / "outputs_stanford" / "work" / "hcp151_fixed_template_matching" / "subjects"
)
DEFAULT_SURFACE_ROOT = (
    REPO_ROOT / "outputs_stanford" / "inputs" / "stanford" / "kong2022" / "shared_inputs" / "surface"
)
DEFAULT_ROI_ROOT = (
    REPO_ROOT / "outputs_stanford" / "inputs" / "stanford" / "kong2022" / "shared_inputs" / "routea"
)
DEFAULT_EXTERNAL_ROOT = Path(
    "/home/yojiao/LSS/jianglab/3_Data_Working/fmriprep_processed_CausalConnectome/template_matched_profiles"
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def fingerprint(path: Path) -> dict[str, Any]:
    """Return a content fingerprint and logical metadata for one artifact."""

    resolved = Path(path).resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"Missing required artifact: {resolved}")
    result: dict[str, Any] = {
        "path": str(resolved),
        "size_bytes": int(resolved.stat().st_size),
        "sha256": _sha256(resolved),
    }
    if resolved.suffix == ".npy":
        array = np.load(resolved, mmap_mode="r", allow_pickle=False)
        result["shape"] = [int(value) for value in array.shape]
        result["dtype"] = str(array.dtype)
    return result


def _read_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Malformed JSON manifest: {path}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object in manifest: {path}")
    return payload


def _write_json_exclusive(path: Path, payload: dict[str, Any]) -> None:
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite existing manifest: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    serialized = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    with path.open("x", encoding="utf-8") as handle:
        handle.write(serialized)


def _replace_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    """Replace one existing manifest only from an explicit provenance refresh."""

    if not path.is_file():
        raise FileNotFoundError(f"Cannot refresh missing manifest: {path}")
    partial = path.with_name(f".{path.name}.provenance-refresh.partial")
    if partial.exists():
        raise FileExistsError(f"Stale provenance-refresh partial exists: {partial}")
    serialized = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    with partial.open("x", encoding="utf-8") as handle:
        handle.write(serialized)
    os.replace(partial, path)


def _write_text_exclusive(path: Path, content: str) -> None:
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite existing text file: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        handle.write(content)


def _save_npy_exclusive(path: Path, array: np.ndarray) -> None:
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite existing array: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as handle:
        np.save(handle, np.asarray(array), allow_pickle=False)


def _copy_exclusive(source: Path, destination: Path) -> None:
    """Copy one generated artifact without replacing an existing colleague file."""

    if destination.exists():
        raise FileExistsError(f"Refusing to overwrite existing copied artifact: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with source.open("rb") as source_handle, destination.open("xb") as destination_handle:
        for block in iter(lambda: source_handle.read(1024 * 1024), b""):
            destination_handle.write(block)
    source_hash = _sha256(source)
    destination_hash = _sha256(destination)
    if source_hash != destination_hash:
        raise IOError(
            f"Copied artifact hash mismatch: source={source} destination={destination}"
        )


def _artifact_record(path: Path) -> dict[str, Any]:
    return fingerprint(path)


def _normalize_subject(raw: str) -> str:
    subject = str(raw).strip()
    if subject.startswith("sub-"):
        subject = subject[4:]
    if not subject or "/" in subject or "\\" in subject:
        raise ValueError(f"Invalid subject identifier: {raw!r}")
    return subject


def discover_subjects(
    match_root: Path,
    *,
    subjects_manifest: Path | None = None,
    explicit_subjects: list[str] | None = None,
) -> list[str]:
    if explicit_subjects:
        raw_subjects = explicit_subjects
    elif subjects_manifest is not None:
        manifest_path = Path(subjects_manifest).resolve()
        if not manifest_path.is_file():
            raise FileNotFoundError(f"Missing subject manifest: {manifest_path}")
        raw_subjects = [
            line.strip()
            for line in manifest_path.read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.lstrip().startswith("#")
        ]
    else:
        raw_subjects = [
            path.name[4:]
            for path in match_root.glob("sub-*")
            if path.is_dir() and (path / "subject_manifest.json").is_file()
        ]
    subjects = sorted(_normalize_subject(subject) for subject in raw_subjects)
    if not subjects:
        raise ValueError(f"No subjects found under the explicit matching root: {match_root}")
    if len(set(subjects)) != len(subjects):
        raise ValueError("Subject roster contains duplicate identifiers")
    return subjects


def _input_paths(
    *,
    match_root: Path,
    surface_root: Path,
    roi_root: Path,
    subject: str,
    hemi: str,
) -> dict[str, Path]:
    match_subject = match_root / f"sub-{subject}"
    surface_subject = surface_root / f"sub-{subject}"
    roi_subject = roi_root / f"sub-{subject}"
    combo_dir = match_subject / "matching" / SMOOTHING / f"hemi-{hemi}"
    return {
        "subject_manifest": match_subject / "subject_manifest.json",
        "surface_timeseries": (
            surface_subject
            / "surface"
            / SMOOTHING
            / f"sub-{subject}_hemi-{hemi}_timeseries.npy"
        ),
        "roi_timeseries": (
            roi_subject / "reference" / "kong2022" / "cortex_roi_parcel_timeseries.npy"
        ),
        "roi_registry": roi_subject / "reference" / "kong2022" / "cortex_roi_parcels.tsv",
        "labels": combo_dir / "labels.npy",
        "valid_mask": combo_dir / "valid_mask.npy",
        "combo_manifest": combo_dir / "combo_manifest.json",
    }


def _local_output_paths(match_root: Path, subject: str, hemi: str) -> dict[str, Path]:
    combo_dir = match_root / f"sub-{subject}" / "matching" / SMOOTHING / f"hemi-{hemi}"
    return {
        "pearson_fc": combo_dir / "cluster_mean_timeseries_fc_pearson.npy",
        "fisherz_fc": combo_dir / "cluster_mean_timeseries_fc_fisherz.npy",
        "manifest": combo_dir / "cluster_mean_timeseries_fc_manifest.json",
    }


def _external_output_paths(external_root: Path, subject: str, hemi: str) -> dict[str, Path]:
    subject_dir = external_root / f"sub-{subject}"
    prefix = f"sub-{subject}_{SMOOTHING}_hemi-{hemi}_cluster_mean_timeseries_fc"
    return {
        "pearson_fc": subject_dir / f"{prefix}_pearson.npy",
        "fisherz_fc": subject_dir / f"{prefix}_fisherz.npy",
    }


def _external_subject_manifest_path(external_root: Path, subject: str) -> Path:
    return external_root / f"sub-{subject}" / f"sub-{subject}_{SMOOTHING}_cluster_mean_timeseries_fc_manifest.json"


def _read_roi_registry(path: Path) -> tuple[np.ndarray, list[dict[str, str]]]:
    required = {
        "parcel_id",
        "n_grayordinates_used",
    }
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    if len(rows) != N_ROIS:
        raise ValueError(f"Expected {N_ROIS} ROI rows in {path}, got {len(rows)}")
    if not rows or not required.issubset(rows[0]):
        raise ValueError(f"ROI registry is missing required columns: {path}")
    roi_ids = [int(row["parcel_id"]) for row in rows]
    expected_ids = list(range(1, N_ROIS + 1))
    if roi_ids != expected_ids:
        raise ValueError(f"ROI registry must be ordered 1..{N_ROIS}: {path}")
    counts = np.asarray([int(row["n_grayordinates_used"]) for row in rows], dtype=np.int32)
    if np.any(counts < 0):
        raise ValueError(f"ROI registry contains negative vertex counts: {path}")
    return counts, rows


def _validate_subject_manifest(path: Path, subject: str) -> None:
    payload = _read_json(path)
    if payload.get("status") != "ok":
        raise ValueError(f"Subject manifest is not complete: {path}")
    if payload.get("subject") not in {subject, f"sub-{subject}"}:
        raise ValueError(f"Subject identity mismatch in manifest: {path}")


def _validate_combo_manifest(path: Path, subject: str, hemi: str) -> None:
    payload = _read_json(path)
    if payload.get("status") != "ok":
        raise ValueError(f"Template-matching combo is not complete: {path}")
    if payload.get("subject") != f"sub-{subject}":
        raise ValueError(f"Subject identity mismatch in combo manifest: {path}")
    if payload.get("smoothing") != SMOOTHING:
        raise ValueError(f"Expected {SMOOTHING} combo manifest: {path}")
    if payload.get("hemi") != hemi:
        raise ValueError(f"Hemisphere mismatch in combo manifest: {path}")
    if payload.get("label_semantics") != LABEL_SEMANTICS:
        raise ValueError(f"Unexpected template-match label semantics: {path}")


def _validate_input_arrays(
    *,
    paths: dict[str, Path],
    subject: str,
    hemi: str,
    roi_used_mask: np.ndarray,
) -> dict[str, Any]:
    surface = np.load(paths["surface_timeseries"], mmap_mode="r", allow_pickle=False)
    roi = np.load(paths["roi_timeseries"], mmap_mode="r", allow_pickle=False)
    labels = np.load(paths["labels"], mmap_mode="r", allow_pickle=False)
    valid_mask = np.load(paths["valid_mask"], mmap_mode="r", allow_pickle=False)

    if surface.ndim != 2 or surface.shape[0] != N_VERTICES:
        raise ValueError(
            f"Expected surface time series shape ({N_VERTICES}, T) for {subject} hemi-{hemi}; got {surface.shape}"
        )
    if roi.ndim != 2 or roi.shape[0] != N_ROIS:
        raise ValueError(
            f"Expected ROI time series shape ({N_ROIS}, T) for {subject}; got {roi.shape}"
        )
    if surface.shape[1] != roi.shape[1] or surface.shape[1] < 2:
        raise ValueError(
            f"Surface/ROI time-point mismatch for {subject} hemi-{hemi}: surface={surface.shape} roi={roi.shape}"
        )
    if labels.shape != (N_VERTICES,) or not np.issubdtype(labels.dtype, np.integer):
        raise ValueError(f"Expected integer labels shape ({N_VERTICES},) for {paths['labels']}; got {labels.shape}")
    if valid_mask.shape != (N_VERTICES,) or valid_mask.dtype != np.dtype(bool):
        raise ValueError(
            f"Expected boolean valid mask shape ({N_VERTICES},) for {paths['valid_mask']}; got {valid_mask.shape} {valid_mask.dtype}"
        )

    valid_labels = np.asarray(labels[valid_mask])
    unsupported = sorted(set(valid_labels.tolist()) - set(CLUSTER_IDS))
    if unsupported:
        raise ValueError(
            f"Unexpected valid template-match labels for {subject} hemi-{hemi}: {unsupported}"
        )
    cluster_counts = {
        str(cluster): int(np.sum(valid_mask & (labels == cluster)))
        for cluster in CLUSTER_IDS
    }
    missing_clusters = [cluster for cluster, count in cluster_counts.items() if count == 0]
    if missing_clusters:
        raise ValueError(
            f"Subject-specific template match has no valid cluster {missing_clusters[0]} for {subject} hemi-{hemi}"
        )

    selected_surface = surface[valid_mask]
    if not np.isfinite(selected_surface).all():
        raise ValueError(
            f"Non-finite values in valid 2mm hippocampal surface time series: {paths['surface_timeseries']}"
        )
    used_roi = np.flatnonzero(roi_used_mask)
    if used_roi.size and not np.isfinite(roi[used_roi]).all():
        raise ValueError(
            f"Non-finite values in ROI time series for non-empty ROIs: {paths['roi_timeseries']}"
        )

    return {
        "timepoints": int(surface.shape[1]),
        "cluster_vertex_counts": cluster_counts,
        "valid_vertices": int(np.sum(valid_mask)),
        "roi_used": int(np.sum(roi_used_mask)),
    }


def _preflight_subjects(
    *,
    subjects: list[str],
    match_root: Path,
    surface_root: Path,
    roi_root: Path,
) -> dict[str, dict[str, Any]]:
    records: dict[str, dict[str, Any]] = {}
    for subject in subjects:
        subject_manifest = match_root / f"sub-{subject}" / "subject_manifest.json"
        _validate_subject_manifest(subject_manifest, subject)
        common_roi_paths = _input_paths(
            match_root=match_root,
            surface_root=surface_root,
            roi_root=roi_root,
            subject=subject,
            hemi="L",
        )
        roi_counts, _ = _read_roi_registry(common_roi_paths["roi_registry"])
        roi_used_mask = roi_counts > 0
        subject_record: dict[str, Any] = {
            "subject": f"sub-{subject}",
            "roi_used_mask": roi_used_mask,
            "roi_counts": roi_counts,
            "hemis": {},
        }
        for hemi in HEMIS:
            paths = _input_paths(
                match_root=match_root,
                surface_root=surface_root,
                roi_root=roi_root,
                subject=subject,
                hemi=hemi,
            )
            for path in paths.values():
                if not path.is_file():
                    raise FileNotFoundError(f"Missing required {subject} hemi-{hemi} input: {path}")
            _validate_combo_manifest(paths["combo_manifest"], subject, hemi)
            array_summary = _validate_input_arrays(
                paths=paths,
                subject=subject,
                hemi=hemi,
                roi_used_mask=roi_used_mask,
            )
            subject_record["hemis"][hemi] = {
                "paths": paths,
                "summary": array_summary,
                "inputs": {name: fingerprint(path) for name, path in paths.items()},
            }
        records[subject] = subject_record
    return records


def compute_cluster_fc(
    *,
    surface_timeseries: np.ndarray,
    roi_timeseries: np.ndarray,
    labels: np.ndarray,
    valid_mask: np.ndarray,
    roi_used_mask: np.ndarray | None = None,
) -> dict[str, np.ndarray]:
    """Compute cluster-mean time series and its 3×N ROI FC matrices.

    ``surface_timeseries`` and ``roi_timeseries`` are both stored as
    feature-by-time arrays.  The cluster means are formed before calling the
    repository's established row-wise correlation implementation.
    """

    surface = np.asarray(surface_timeseries, dtype=np.float32)
    roi = np.asarray(roi_timeseries, dtype=np.float32)
    labels_array = np.asarray(labels)
    valid = np.asarray(valid_mask, dtype=bool)
    if surface.ndim != 2:
        raise ValueError(f"surface_timeseries must be 2D, got {surface.shape}")
    if roi.ndim != 2:
        raise ValueError(f"roi_timeseries must be 2D, got {roi.shape}")
    if surface.shape[0] != labels_array.size or valid.shape != labels_array.shape:
        raise ValueError(
            f"Surface/label shape mismatch: surface={surface.shape}, labels={labels_array.shape}, valid={valid.shape}"
        )
    if surface.shape[1] != roi.shape[1]:
        raise ValueError(f"Surface/ROI time-point mismatch: surface={surface.shape}, roi={roi.shape}")
    if not np.issubdtype(labels_array.dtype, np.integer):
        raise ValueError(f"labels must have an integer dtype, got {labels_array.dtype}")

    if roi_used_mask is None:
        roi_used = np.ones(roi.shape[0], dtype=bool)
    else:
        roi_used = np.asarray(roi_used_mask, dtype=bool)
        if roi_used.shape != (roi.shape[0],):
            raise ValueError(
                f"roi_used_mask shape mismatch: expected {(roi.shape[0],)}, got {roi_used.shape}"
            )

    cluster_means: list[np.ndarray] = []
    cluster_vertex_counts: list[int] = []
    for cluster in CLUSTER_IDS:
        selected = valid & (labels_array == cluster)
        count = int(np.sum(selected))
        if count == 0:
            raise ValueError(f"Subject-specific template match has no valid cluster {cluster}")
        selected_timeseries = surface[selected]
        if not np.isfinite(selected_timeseries).all():
            raise ValueError(f"Non-finite values in valid cluster {cluster} time series")
        cluster_means.append(
            np.mean(selected_timeseries, axis=0, dtype=np.float32).astype(np.float32, copy=False)
        )
        cluster_vertex_counts.append(count)

    cluster_mean_timeseries = np.vstack(cluster_means).astype(np.float32, copy=False)
    pearson_fc = corrcoef_rows(cluster_mean_timeseries, roi)
    pearson_fc[:, ~roi_used] = np.nan
    fisherz_fc = fisher_z_transform_fc(pearson_fc)
    return {
        "cluster_mean_timeseries": cluster_mean_timeseries,
        "pearson_fc": np.asarray(pearson_fc, dtype=np.float32),
        "fisherz_fc": np.asarray(fisherz_fc, dtype=np.float32),
        "cluster_vertex_counts": np.asarray(cluster_vertex_counts, dtype=np.int32),
    }


def _local_manifest(
    *,
    subject: str,
    hemi: str,
    input_fingerprints: dict[str, dict[str, Any]],
    local_paths: dict[str, Path],
    external_paths: dict[str, Path],
    result: dict[str, np.ndarray],
) -> dict[str, Any]:
    return {
        "status": "ok",
        "artifact_contract_schema_version": ARTIFACT_CONTRACT_SCHEMA_VERSION,
        "workflow": WORKFLOW,
        "subject": f"sub-{subject}",
        "smoothing": SMOOTHING,
        "hemi": hemi,
        "density": N_VERTICES,
        "n_clusters": len(CLUSTER_IDS),
        "n_rois": N_ROIS,
        "matrix_shape": [len(CLUSTER_IDS), N_ROIS],
        "cluster_label_order": LABEL_SEMANTICS,
        "cluster_vertex_counts": {
            str(cluster): int(count)
            for cluster, count in zip(CLUSTER_IDS, result["cluster_vertex_counts"].tolist())
        },
        "roi_columns": "Kong2022 ROI IDs 1..400 in the stored cortex_roi_parcels.tsv order",
        "definition": {
            "cluster_timeseries": (
                "arithmetic mean of the subject's 2mm hippocampal surface time series over "
                "vertices where valid_mask is true and template-match labels equal the cluster"
            ),
            "pearson_fc": "repository lib.fc_math.corrcoef_rows(cluster_mean_timeseries, roi_timeseries)",
            "fisherz_fc": "arctanh clipped to +/-0.999999 after Pearson FC",
            "vertex_fc_reuse": False,
        },
        "inputs": input_fingerprints,
        "outputs": {
            "pearson_fc": _artifact_record(local_paths["pearson_fc"]),
            "fisherz_fc": _artifact_record(local_paths["fisherz_fc"]),
        },
        "external_copies": {
            "pearson_fc": _artifact_record(external_paths["pearson_fc"]),
            "fisherz_fc": _artifact_record(external_paths["fisherz_fc"]),
        },
    }


PROVENANCE_REFRESH_REASON = (
    "upstream provenance manifests changed after FC calculation; output arrays were retained after hash validation"
)


def refresh_local_manifest_payload(
    payload: dict[str, Any],
    *,
    current_inputs: dict[str, dict[str, Any]],
    refreshed_utc: str | None = None,
) -> dict[str, Any]:
    """Refresh input fingerprints while preserving all generated output records."""

    if payload.get("status") != "ok":
        raise ValueError("Only complete local manifests can be provenance-refreshed")
    old_inputs = payload.get("inputs")
    if not isinstance(old_inputs, dict):
        raise ValueError("Local manifest has no structured input fingerprints")
    changed_inputs = sorted(
        key
        for key in set(old_inputs) | set(current_inputs)
        if old_inputs.get(key) != current_inputs.get(key)
    )
    refreshed = deepcopy(payload)
    if not changed_inputs:
        return refreshed
    history = refreshed.get("provenance_refresh_history", [])
    if not isinstance(history, list):
        raise ValueError("Local manifest has malformed provenance refresh history")
    refreshed["inputs"] = deepcopy(current_inputs)
    history.append(
        {
            "changed_inputs": changed_inputs,
            "reason": PROVENANCE_REFRESH_REASON,
            "refreshed_utc": refreshed_utc or datetime.now(timezone.utc).isoformat(),
        }
    )
    refreshed["provenance_refresh_history"] = history
    return refreshed


def _external_subject_manifest(
    *,
    subject: str,
    match_root: Path,
    surface_root: Path,
    roi_root: Path,
    hemi_records: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    return {
        "status": "ok",
        "artifact_contract_schema_version": ARTIFACT_CONTRACT_SCHEMA_VERSION,
        "workflow": WORKFLOW,
        "subject": f"sub-{subject}",
        "dataset": "stanford",
        "smoothing": SMOOTHING,
        "density": N_VERTICES,
        "n_clusters": len(CLUSTER_IDS),
        "n_rois": N_ROIS,
        "matrix_shape": [len(CLUSTER_IDS), N_ROIS],
        "hemisphere_order": list(HEMIS),
        "cluster_label_order": LABEL_SEMANTICS,
        "definition": (
            "Each matrix is computed from this subject's template-match labels by averaging "
            "the subject's 2mm hippocampal surface time series within each cluster, then "
            "correlating the three cluster time series with the subject's 400 Kong2022 ROI time series."
        ),
        "source_roots": {
            "matching": str(match_root.resolve()),
            "surface_timeseries": str(surface_root.resolve()),
            "roi_timeseries": str(roi_root.resolve()),
        },
        "outputs": hemi_records,
    }


def _expected_subject_outputs(
    *,
    match_root: Path,
    external_root: Path,
    subject: str,
) -> list[Path]:
    paths: list[Path] = []
    for hemi in HEMIS:
        local = _local_output_paths(match_root, subject, hemi)
        external = _external_output_paths(external_root, subject, hemi)
        paths.extend(local.values())
        paths.extend(external.values())
    paths.append(_external_subject_manifest_path(external_root, subject))
    return paths


def _validate_existing_subject(
    *,
    subject: str,
    record: dict[str, Any],
    match_root: Path,
    external_root: Path,
    require_input_match: bool = True,
) -> dict[str, Any]:
    """Validate a complete resumable subject and return its external record."""

    external_manifest_path = _external_subject_manifest_path(external_root, subject)
    external_manifest = _read_json(external_manifest_path)
    if external_manifest.get("status") != "ok" or external_manifest.get("subject") != f"sub-{subject}":
        raise ValueError(f"Invalid resumable external manifest: {external_manifest_path}")
    for hemi in HEMIS:
        local_paths = _local_output_paths(match_root, subject, hemi)
        external_paths = _external_output_paths(external_root, subject, hemi)
        local_manifest = _read_json(local_paths["manifest"])
        if (
            local_manifest.get("status") != "ok"
            or local_manifest.get("subject") != f"sub-{subject}"
            or local_manifest.get("hemi") != hemi
            or local_manifest.get("smoothing") != SMOOTHING
        ):
            raise ValueError(f"Invalid resumable local manifest: {local_paths['manifest']}")
        expected_inputs = record["hemis"][hemi]["inputs"]
        if require_input_match and local_manifest.get("inputs") != expected_inputs:
            raise ValueError(f"Input provenance drift prevents resume: {local_paths['manifest']}")
        for key in ("pearson_fc", "fisherz_fc"):
            local_artifact = _artifact_record(local_paths[key])
            external_artifact = _artifact_record(external_paths[key])
            if local_artifact["shape"] != [3, 400] or external_artifact["shape"] != [3, 400]:
                raise ValueError(f"Invalid resumable matrix shape for {subject} hemi-{hemi}")
            if local_artifact["sha256"] != external_artifact["sha256"]:
                raise ValueError(f"Local/external hash mismatch prevents resume for {subject} hemi-{hemi}")
            if local_manifest.get("outputs", {}).get(key, {}).get("sha256") != local_artifact["sha256"]:
                raise ValueError(f"Local output fingerprint drift prevents resume: {local_paths[key]}")
            if local_manifest.get("external_copies", {}).get(key, {}).get("sha256") != external_artifact["sha256"]:
                raise ValueError(f"External output fingerprint drift prevents resume: {external_paths[key]}")
    return {
        "subject": f"sub-{subject}",
        "manifest": str(external_manifest_path.resolve()),
        "outputs": external_manifest.get("outputs", {}),
    }


def _check_collisions(
    *,
    subjects: list[str],
    match_root: Path,
    external_root: Path,
    resume: bool,
    records: dict[str, dict[str, Any]],
    allow_input_drift: bool = False,
) -> set[str]:
    complete: set[str] = set()
    for subject in subjects:
        expected = _expected_subject_outputs(
            match_root=match_root,
            external_root=external_root,
            subject=subject,
        )
        existing = [path for path in expected if path.exists()]
        if not existing:
            continue
        if not resume:
            raise FileExistsError(
                "Output collision; rerun with --resume only for a complete matching run: "
                + ", ".join(str(path) for path in existing[:5])
            )
        if len(existing) != len(expected):
            raise ValueError(
                f"Partial output exists for {subject}; refusing to overwrite or guess how to repair it"
            )
        _validate_existing_subject(
            subject=subject,
            record=records[subject],
            match_root=match_root,
            external_root=external_root,
            require_input_match=not allow_input_drift,
        )
        complete.add(subject)
    return complete


def _refresh_subject_provenance(
    *,
    subject: str,
    record: dict[str, Any],
    match_root: Path,
    external_root: Path,
    refreshed_utc: str,
) -> dict[str, Any]:
    """Refresh only input metadata after output hashes have been validated."""

    external_manifest_path = _external_subject_manifest_path(external_root, subject)
    external_manifest = _read_json(external_manifest_path)
    changed_by_hemi: dict[str, list[str]] = {}
    for hemi in HEMIS:
        local_paths = _local_output_paths(match_root, subject, hemi)
        current_inputs = record["hemis"][hemi]["inputs"]
        original = _read_json(local_paths["manifest"])
        old_inputs = original.get("inputs", {})
        changed = sorted(
            key
            for key in set(old_inputs) | set(current_inputs)
            if old_inputs.get(key) != current_inputs.get(key)
        )
        refreshed = refresh_local_manifest_payload(
            original,
            current_inputs=current_inputs,
            refreshed_utc=refreshed_utc,
        )
        if refreshed != original:
            _replace_json_atomic(local_paths["manifest"], refreshed)
        if changed:
            changed_by_hemi[hemi] = changed
        output_record = external_manifest.get("outputs", {}).get(hemi)
        if not isinstance(output_record, dict):
            raise ValueError(f"External manifest is missing output records for {subject} hemi-{hemi}")
        output_record["local_manifest"] = str(local_paths["manifest"].resolve())
        output_record["local_manifest_fingerprint"] = _artifact_record(local_paths["manifest"])

    if changed_by_hemi:
        history = external_manifest.get("provenance_refresh_history", [])
        if not isinstance(history, list):
            raise ValueError(f"External manifest has malformed refresh history: {external_manifest_path}")
        history.append(
            {
                "changed_inputs_by_hemi": changed_by_hemi,
                "reason": PROVENANCE_REFRESH_REASON,
                "refreshed_utc": refreshed_utc,
            }
        )
        external_manifest["provenance_refresh_history"] = history
        _replace_json_atomic(external_manifest_path, external_manifest)
    return {
        "subject": f"sub-{subject}",
        "manifest": str(external_manifest_path.resolve()),
        "outputs": external_manifest.get("outputs", {}),
    }


def _process_subject(
    *,
    subject: str,
    record: dict[str, Any],
    match_root: Path,
    surface_root: Path,
    roi_root: Path,
    external_root: Path,
) -> dict[str, Any]:
    roi_path = record["hemis"]["L"]["paths"]["roi_timeseries"]
    roi_timeseries = np.load(roi_path, allow_pickle=False).astype(np.float32, copy=False)
    roi_used_mask = np.asarray(record["roi_used_mask"], dtype=bool)
    external_hemi_records: dict[str, dict[str, Any]] = {}
    local_manifests: dict[str, Path] = {}

    for hemi in HEMIS:
        hemi_record = record["hemis"][hemi]
        paths = hemi_record["paths"]
        surface_timeseries = np.load(paths["surface_timeseries"], allow_pickle=False)
        labels = np.load(paths["labels"], allow_pickle=False)
        valid_mask = np.load(paths["valid_mask"], allow_pickle=False)
        result = compute_cluster_fc(
            surface_timeseries=surface_timeseries,
            roi_timeseries=roi_timeseries,
            labels=labels,
            valid_mask=valid_mask,
            roi_used_mask=roi_used_mask,
        )
        local_paths = _local_output_paths(match_root, subject, hemi)
        external_paths = _external_output_paths(external_root, subject, hemi)
        _save_npy_exclusive(local_paths["pearson_fc"], result["pearson_fc"])
        _save_npy_exclusive(local_paths["fisherz_fc"], result["fisherz_fc"])
        _copy_exclusive(local_paths["pearson_fc"], external_paths["pearson_fc"])
        _copy_exclusive(local_paths["fisherz_fc"], external_paths["fisherz_fc"])
        local_manifest = _local_manifest(
            subject=subject,
            hemi=hemi,
            input_fingerprints=hemi_record["inputs"],
            local_paths=local_paths,
            external_paths=external_paths,
            result=result,
        )
        _write_json_exclusive(local_paths["manifest"], local_manifest)
        local_manifests[hemi] = local_paths["manifest"]
        external_hemi_records[hemi] = {
            "pearson_fc": _artifact_record(external_paths["pearson_fc"]),
            "fisherz_fc": _artifact_record(external_paths["fisherz_fc"]),
            "local_manifest": str(local_paths["manifest"].resolve()),
        }

    external_manifest_path = _external_subject_manifest_path(external_root, subject)
    external_manifest = _external_subject_manifest(
        subject=subject,
        match_root=match_root,
        surface_root=surface_root,
        roi_root=roi_root,
        hemi_records=external_hemi_records,
    )
    _write_json_exclusive(external_manifest_path, external_manifest)
    return {
        "subject": f"sub-{subject}",
        "manifest": str(external_manifest_path.resolve()),
        "outputs": external_hemi_records,
    }


def _root_readme() -> str:
    return """# Stanford template-matched cluster FC\n\nEach subject directory contains the two hemispheres from the Stanford 2 mm HCP151 template-matching result. Every `*_pearson.npy` and `*_fisherz.npy` file has shape `(3, 400)`.\n\nRows are the subject-specific template-match clusters in this fixed order:\n\n1. Anterior / VAN\n2. Body / SMN\n3. Posterior / DMN\n\nColumns are Kong2022 ROI IDs 1 through 400 in the order recorded by the subject's `cortex_roi_parcels.tsv`. The cluster time series are averaged from that subject's hippocampal surface time series before FC is calculated. The Fisher-z files are the usual template-matching analysis scale; Pearson files are retained alongside them.\n\nEach subject directory has a JSON provenance manifest. The repository-local per-hemisphere manifests remain beside the original template-matching labels under `outputs_stanford/work/hcp151_fixed_template_matching/subjects/`.\n"""


def _root_manifest(
    *,
    external_root: Path,
    subjects: list[str],
    subject_records: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "status": "complete",
        "artifact_contract_schema_version": ARTIFACT_CONTRACT_SCHEMA_VERSION,
        "workflow": WORKFLOW,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "dataset": "stanford",
        "smoothing": SMOOTHING,
        "density": N_VERTICES,
        "n_subjects": len(subjects),
        "subjects": [f"sub-{subject}" for subject in subjects],
        "hemis": list(HEMIS),
        "matrix_shape": [len(CLUSTER_IDS), N_ROIS],
        "cluster_label_order": LABEL_SEMANTICS,
        "output_root": str(external_root.resolve()),
        "subjects_outputs": subject_records,
    }


def _write_or_validate_root_artifacts(
    *,
    external_root: Path,
    subjects: list[str],
    subject_records: list[dict[str, Any]],
    resume: bool,
) -> None:
    readme_path = external_root / "README.md"
    if readme_path.exists():
        if not resume or readme_path.read_text(encoding="utf-8") != _root_readme():
            raise ValueError(f"Existing destination README is not the expected workflow README: {readme_path}")
    else:
        _write_text_exclusive(readme_path, _root_readme())

    manifest_path = external_root / "template_matched_profiles_manifest.json"
    manifest = _root_manifest(
        external_root=external_root,
        subjects=subjects,
        subject_records=subject_records,
    )
    if manifest_path.exists():
        if not resume:
            raise FileExistsError(f"Refusing to overwrite existing root manifest: {manifest_path}")
        existing = _read_json(manifest_path)
        if (
            existing.get("status") != "complete"
            or existing.get("subjects") != manifest["subjects"]
            or existing.get("matrix_shape") != manifest["matrix_shape"]
        ):
            raise ValueError(f"Existing root manifest does not match this complete run: {manifest_path}")
    else:
        _write_json_exclusive(manifest_path, manifest)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--match-root", type=Path, default=DEFAULT_MATCH_ROOT)
    parser.add_argument("--surface-root", type=Path, default=DEFAULT_SURFACE_ROOT)
    parser.add_argument("--roi-root", type=Path, default=DEFAULT_ROI_ROOT)
    parser.add_argument("--external-root", type=Path, default=DEFAULT_EXTERNAL_ROOT)
    parser.add_argument("--subjects-manifest", type=Path, default=None)
    parser.add_argument("--subjects", nargs="+", default=None)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate all 2 mm inputs and report collisions without writing files.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Reuse only complete, fingerprint-matching subject outputs.",
    )
    parser.add_argument(
        "--refresh-provenance",
        action="store_true",
        help="Refresh changed upstream input fingerprints after validating output hashes; never recomputes FC.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    match_root = args.match_root.resolve()
    surface_root = args.surface_root.resolve()
    roi_root = args.roi_root.resolve()
    external_root = args.external_root.resolve()
    subjects = discover_subjects(
        match_root,
        subjects_manifest=args.subjects_manifest,
        explicit_subjects=args.subjects,
    )
    records = _preflight_subjects(
        subjects=subjects,
        match_root=match_root,
        surface_root=surface_root,
        roi_root=roi_root,
    )
    complete = _check_collisions(
        subjects=subjects,
        match_root=match_root,
        external_root=external_root,
        resume=args.resume or args.refresh_provenance,
        records=records,
        allow_input_drift=args.refresh_provenance,
    )
    if args.refresh_provenance:
        if complete != set(subjects):
            raise ValueError(
                "Provenance refresh requires complete local and external outputs for every selected subject"
            )
        if args.dry_run:
            print(
                json.dumps(
                    {
                        "status": "provenance_refresh_dry_run_ok",
                        "workflow": WORKFLOW,
                        "subjects": len(subjects),
                        "hemis": len(HEMIS),
                        "matrix_shape": [len(CLUSTER_IDS), N_ROIS],
                        "subjects_ready_for_refresh": len(complete),
                        "external_root": str(external_root),
                    },
                    indent=2,
                )
            )
            return 0
        external_root.mkdir(parents=True, exist_ok=True)
        refreshed_utc = datetime.now(timezone.utc).isoformat()
        subject_records = []
        for index, subject in enumerate(subjects, start=1):
            subject_records.append(
                _refresh_subject_provenance(
                    subject=subject,
                    record=records[subject],
                    match_root=match_root,
                    external_root=external_root,
                    refreshed_utc=refreshed_utc,
                )
            )
            print(f"[{index}/{len(subjects)}] {subject} provenance refreshed")
        _write_or_validate_root_artifacts(
            external_root=external_root,
            subjects=subjects,
            subject_records=subject_records,
            resume=True,
        )
        print(f"provenance refresh complete: {len(subjects)} subjects × {len(HEMIS)} hemispheres")
        return 0

    if args.dry_run:
        print(
            json.dumps(
                {
                    "status": "dry_run_ok",
                    "workflow": WORKFLOW,
                    "subjects": len(subjects),
                    "hemis": len(HEMIS),
                    "matrix_shape": [len(CLUSTER_IDS), N_ROIS],
                    "complete_subjects_to_resume": len(complete),
                    "subjects_to_compute": len(subjects) - len(complete),
                    "external_root": str(external_root),
                },
                indent=2,
            )
        )
        return 0

    external_root.mkdir(parents=True, exist_ok=True)
    subject_records: list[dict[str, Any]] = []
    for index, subject in enumerate(subjects, start=1):
        if subject in complete:
            external_manifest_path = _external_subject_manifest_path(external_root, subject)
            external_manifest = _read_json(external_manifest_path)
            subject_records.append(
                {
                    "subject": f"sub-{subject}",
                    "manifest": str(external_manifest_path.resolve()),
                    "outputs": external_manifest.get("outputs", {}),
                }
            )
            print(f"[{index}/{len(subjects)}] {subject} already complete")
            continue
        subject_records.append(
            _process_subject(
                subject=subject,
                record=records[subject],
                match_root=match_root,
                surface_root=surface_root,
                roi_root=roi_root,
                external_root=external_root,
            )
        )
        print(f"[{index}/{len(subjects)}] {subject} complete")

    _write_or_validate_root_artifacts(
        external_root=external_root,
        subjects=subjects,
        subject_records=subject_records,
        resume=args.resume,
    )
    print(f"complete: {len(subjects)} subjects × {len(HEMIS)} hemispheres")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
