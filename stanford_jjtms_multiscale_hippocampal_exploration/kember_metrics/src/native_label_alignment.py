"""Auditable runner primitives for Kember-method system surface area.

This module keeps the two retained subject-level analysis branches separate.
It does not define a boundary-area endpoint: it only implements per-system
midthickness surface area and the bilateral arithmetic mean described for the
Kember-method area fields.
"""

from __future__ import annotations

import csv
import hashlib
import json
import re
import shutil
import subprocess
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import nibabel as nib
from nibabel.gifti import GiftiDataArray, GiftiImage, GiftiLabel, GiftiLabelTable

if __package__:  # Normal unified-package import.
    from .surface_geometry import load_midthickness
    from .input_contract import audit_stanford_input_contract
    from .surface_area import (
        SYSTEM_NAMES,
        combine_hemisphere_means,
        compute_system_surface_area,
    )
else:  # pragma: no cover - direct source-directory execution
    from surface_geometry import load_midthickness
    from input_contract import audit_stanford_input_contract
    from surface_area import (
        SYSTEM_NAMES,
        combine_hemisphere_means,
        compute_system_surface_area,
    )


BRANCHES = frozenset(
    {
        "hcp151_fixed",
        "hcp151_template_matching",
    }
)
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_HEMISPHERES = frozenset({"L", "R"})


def _require_asset_backed_density_evidence(
    density_evidence: Mapping[str, object], *, n_vertices: int
) -> dict[str, object]:
    """Validate and normalize density evidence against consumed geometry."""

    if not isinstance(density_evidence, Mapping):
        raise ValueError("asset-backed density evidence must be an object")
    consumed_asset = density_evidence.get("consumed_asset")
    declared_vertices = density_evidence.get("n_vertices")
    fingerprint = density_evidence.get("surface_fingerprint")
    sha256 = fingerprint.get("sha256") if isinstance(fingerprint, Mapping) else None
    if (
        not isinstance(consumed_asset, str)
        or not consumed_asset
        or not isinstance(declared_vertices, int)
        or declared_vertices != n_vertices
        or not isinstance(sha256, str)
        or not _SHA256.fullmatch(sha256)
    ):
        raise ValueError(
            "asset-backed density evidence must name the consumed surface, "
            "record its SHA-256, and match the consumed vertex count"
        )
    return json.loads(json.dumps(density_evidence, sort_keys=True))


def evaluate_hemisphere_area(
    *,
    branch: str,
    subject: str,
    smoothing_mm: int,
    hemisphere: str,
    surface_path: str | Path,
    vertex_area_mm2: np.ndarray,
    labels: np.ndarray,
    valid_mask: np.ndarray | None,
    density_evidence: Mapping[str, object],
) -> dict[str, object]:
    """Evaluate one frozen subject/hemisphere/branch area record.

    The caller must supply density evidence from the exact native surface/area
    assets before invoking the area calculation.  This makes a density token
    in a filename non-authoritative: the recorded consumed vertex count is
    checked against the area vector.
    """

    if branch not in BRANCHES:
        raise ValueError(f"unsupported Kember-method branch: {branch!r}")
    if not isinstance(subject, str) or not subject:
        raise ValueError("subject must be a non-empty string")
    if int(smoothing_mm) <= 0:
        raise ValueError("smoothing_mm must be positive")
    if hemisphere not in _HEMISPHERES:
        raise ValueError("hemisphere must be L or R")

    area_values = np.asarray(vertex_area_mm2, dtype=np.float64)
    evidence = _require_asset_backed_density_evidence(
        density_evidence, n_vertices=int(area_values.size)
    )
    area = compute_system_surface_area(
        area_values,
        np.asarray(labels),
        valid_mask=valid_mask,
    )
    return {
        "branch": branch,
        "subject": subject,
        "smoothing_mm": int(smoothing_mm),
        "hemisphere": hemisphere,
        "surface_path": str(surface_path),
        "density_evidence": evidence,
        **area,
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while True:
            block = stream.read(1024 * 1024)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def _flatten_hemisphere_record(record: Mapping[str, object]) -> dict[str, object]:
    area = record.get("system_area_mm2")
    fraction = record.get("system_area_fraction")
    density = record.get("density_evidence")
    if not isinstance(area, Mapping) or not isinstance(fraction, Mapping):
        raise ValueError("area record lacks per-system area and fraction mappings")
    if not isinstance(density, Mapping):
        raise ValueError("area record lacks density evidence")
    fingerprint = density.get("surface_fingerprint")
    sha256 = fingerprint.get("sha256") if isinstance(fingerprint, Mapping) else ""
    return {
        "branch": record["branch"],
        "subject": record["subject"],
        "smoothing_mm": record["smoothing_mm"],
        "hemisphere": record["hemisphere"],
        "anterior_mm2": area["anterior"],
        "body_mm2": area["body"],
        "posterior_mm2": area["posterior"],
        "anterior_fraction": fraction["anterior"],
        "body_fraction": fraction["body"],
        "posterior_fraction": fraction["posterior"],
        "total_surface_area_mm2": record["total_surface_area_mm2"],
        "total_valid_surface_area_mm2": record["total_valid_surface_area_mm2"],
        "total_labeled_area_mm2": record["total_labeled_area_mm2"],
        "unassigned_surface_area_mm2": record["unassigned_surface_area_mm2"],
        "valid_labeled_vertex_count": record["valid_labeled_vertex_count"],
        "surface_path": record["surface_path"],
        "density_consumed_asset": density.get("consumed_asset", ""),
        "density_n_vertices": density.get("n_vertices", ""),
        "density_surface_sha256": sha256,
        "density_evidence_source": density.get("evidence_source", ""),
        "vertex_area_convention": record["vertex_area_convention"],
    }


def _bilateral_rows(records: Sequence[Mapping[str, object]]) -> list[dict[str, object]]:
    grouped: dict[tuple[str, str, int], dict[str, Mapping[str, object]]] = defaultdict(dict)
    for record in records:
        branch = record.get("branch")
        subject = record.get("subject")
        smoothing = record.get("smoothing_mm")
        hemisphere = record.get("hemisphere")
        if not isinstance(branch, str) or not isinstance(subject, str):
            raise ValueError("area records require branch and subject strings")
        if not isinstance(smoothing, int) or hemisphere not in _HEMISPHERES:
            raise ValueError("area records require integer smoothing and L/R hemisphere")
        key = (branch, subject, smoothing)
        if hemisphere in grouped[key]:
            raise ValueError(f"duplicate hemisphere area record for {key} hemi-{hemisphere}")
        grouped[key][hemisphere] = record

    rows: list[dict[str, object]] = []
    for (branch, subject, smoothing), hemispheres in sorted(grouped.items()):
        if set(hemispheres) != _HEMISPHERES:
            raise ValueError(
                "bilateral Kember-method output requires both L and R records "
                f"for {(branch, subject, smoothing)}"
            )
        left = hemispheres["L"]
        right = hemispheres["R"]
        left_area = left.get("system_area_mm2")
        right_area = right.get("system_area_mm2")
        left_fraction = left.get("system_area_fraction")
        right_fraction = right.get("system_area_fraction")
        if not all(
            isinstance(value, Mapping)
            for value in (left_area, right_area, left_fraction, right_fraction)
        ):
            raise ValueError("bilateral area records lack a system mapping")
        area_mean = combine_hemisphere_means(left_area, right_area)
        fraction_mean = combine_hemisphere_means(left_fraction, right_fraction)
        rows.append(
            {
                "branch": branch,
                "subject": subject,
                "smoothing_mm": smoothing,
                "anterior_mm2": area_mean["anterior"],
                "body_mm2": area_mean["body"],
                "posterior_mm2": area_mean["posterior"],
                "anterior_fraction": fraction_mean["anterior"],
                "body_fraction": fraction_mean["body"],
                "posterior_fraction": fraction_mean["posterior"],
                "hemisphere_aggregation": "arithmetic mean of left and right system values",
            }
        )
    return rows


def _write_csv(path: Path, rows: Sequence[Mapping[str, object]], fieldnames: Sequence[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(fieldnames), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _source_lock_excerpt(source_lock: Mapping[str, object]) -> dict[str, str]:
    tag = source_lock.get("tag")
    commit = source_lock.get("commit")
    if not isinstance(tag, str) or not tag or not isinstance(commit, str) or not commit:
        raise ValueError("source lock must provide a non-empty tag and commit")
    return {"tag": tag, "commit": commit}


def write_area_run(
    output_dir: Path,
    *,
    records: Sequence[Mapping[str, object]],
    input_contract: Mapping[str, object],
    source_lock: Mapping[str, object],
    runtime_provenance: Mapping[str, object],
) -> dict[str, object]:
    """Write current per-hemisphere and bilateral area results.

    Output hashes cover the tabular result files.  The JSON manifest preserves
    source-lock, frozen-input, and branch provenance without attempting an
    impossible self-hash of the manifest itself.
    """

    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    if not records:
        raise ValueError("at least one bilateral area record is required")
    if not isinstance(input_contract, Mapping):
        raise ValueError("input_contract must be an object")
    if not isinstance(runtime_provenance, Mapping):
        raise ValueError("runtime_provenance must be an object")

    flattened = [_flatten_hemisphere_record(record) for record in records]
    flattened.sort(
        key=lambda row: (
            str(row["branch"]),
            str(row["subject"]),
            int(row["smoothing_mm"]),
            str(row["hemisphere"]),
        )
    )
    bilateral = _bilateral_rows(records)
    per_hemisphere_path = destination / "system_surface_area_per_hemisphere.csv"
    bilateral_path = destination / "system_surface_area_bilateral.csv"
    manifest_path = destination / "run_manifest.json"
    hemisphere_fields = (
        "branch",
        "subject",
        "smoothing_mm",
        "hemisphere",
        "anterior_mm2",
        "body_mm2",
        "posterior_mm2",
        "anterior_fraction",
        "body_fraction",
        "posterior_fraction",
        "total_surface_area_mm2",
        "total_valid_surface_area_mm2",
        "total_labeled_area_mm2",
        "unassigned_surface_area_mm2",
        "valid_labeled_vertex_count",
        "surface_path",
        "density_consumed_asset",
        "density_n_vertices",
        "density_surface_sha256",
        "density_evidence_source",
        "vertex_area_convention",
    )
    bilateral_fields = (
        "branch",
        "subject",
        "smoothing_mm",
        "anterior_mm2",
        "body_mm2",
        "posterior_mm2",
        "anterior_fraction",
        "body_fraction",
        "posterior_fraction",
        "hemisphere_aggregation",
    )
    _write_csv(per_hemisphere_path, flattened, hemisphere_fields)
    _write_csv(bilateral_path, bilateral, bilateral_fields)
    manifest = {
        "schema_version": 1,
        "source_lock": _source_lock_excerpt(source_lock),
        "input_contract": json.loads(json.dumps(input_contract, sort_keys=True)),
        "runtime_provenance": json.loads(json.dumps(runtime_provenance, sort_keys=True)),
        "branches": sorted({str(record["branch"]) for record in records}),
        "hemisphere_aggregation": "arithmetic mean of left and right system values",
        "vertex_area_convention": "HippUnfold native per-vertex surface-area metric in mm^2",
        "area_metric_roles": {
            "primary": "absolute system surface area in mm^2",
            "secondary": "system surface-area fraction",
            "secondary_denominator": "total native surface area within the valid native ROI",
            "secondary_aggregation": "arithmetic mean of left and right hemisphere fractions",
        },
        "outputs": {
            per_hemisphere_path.name: {"sha256": _sha256(per_hemisphere_path)},
            bilateral_path.name: {"sha256": _sha256(bilateral_path)},
        },
    }
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return {
        "output_dir": str(destination),
        "per_hemisphere_path": str(per_hemisphere_path),
        "bilateral_path": str(bilateral_path),
        "manifest_path": str(manifest_path),
    }
def freeze_input_manifest(
    output_dir: Path,
    *,
    source_lock: Mapping[str, object],
    input_contract: Mapping[str, object],
    consumed_inputs: Mapping[str, object],
) -> dict[str, object]:
    """Record every consumed input before calculation."""

    destination = Path(output_dir)
    if not isinstance(input_contract, Mapping):
        raise ValueError("input_contract must be an object")
    if not isinstance(consumed_inputs, Mapping):
        raise ValueError("consumed_inputs must be an object")
    try:
        manifest = {
            "schema_version": 1,
            "status": "frozen_precalculation",
            "source_lock": _source_lock_excerpt(source_lock),
            "input_contract": json.loads(json.dumps(input_contract, sort_keys=True)),
            "consumed_inputs": json.loads(json.dumps(consumed_inputs, sort_keys=True)),
        }
    except (TypeError, ValueError) as exc:
        raise ValueError("input manifest must be JSON serializable") from exc
    destination.mkdir(parents=True, exist_ok=True)
    manifest_path = destination / "input_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return {
        "output_dir": str(destination),
        "input_manifest_path": str(manifest_path),
    }


def _read_json_mapping(path: Path, *, label: str) -> Mapping[str, object]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid {label}: {path}") from exc
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be a JSON object: {path}")
    return value


def _configured_path(
    root: Path,
    config: Mapping[str, object],
    key: str,
    *,
    required: bool = True,
) -> Path | None:
    paths = config.get("paths")
    if not isinstance(paths, Mapping):
        raise ValueError("input configuration needs a paths object")
    value = paths.get(key)
    if value is None:
        if required:
            raise ValueError(f"input configuration lacks paths.{key}")
        return None
    candidate = Path(str(value))
    return candidate if candidate.is_absolute() else root / candidate


def _file_provenance(path: Path, *, array: np.ndarray | None = None) -> dict[str, object]:
    if not path.is_file():
        raise FileNotFoundError(f"required Kember-method input is missing: {path}")
    result: dict[str, object] = {
        "path": str(path.resolve()),
        "sha256": _sha256(path),
        "size_bytes": int(path.stat().st_size),
    }
    if array is not None:
        result["shape"] = [int(value) for value in array.shape]
        result["dtype"] = str(array.dtype)
    return result


def _surface_density_evidence(path: Path, coordinates: np.ndarray) -> dict[str, object]:
    file_record = _file_provenance(path)
    return {
        "consumed_asset": file_record["path"],
        "n_vertices": int(coordinates.shape[0]),
        "surface_fingerprint": {
            "sha256": file_record["sha256"],
            "size_bytes": file_record["size_bytes"],
        },
        "evidence_source": "consumed_surface_path_and_gifti_point_count",
    }


def _surface_path(
    surface_root: Path,
    subject: str,
    hemisphere: str,
    *,
    space: str,
    density: str,
) -> Path:
    return (
        surface_root
        / subject
        / "hippunfold"
        / subject
        / "surf"
        / f"{subject}_hemi-{hemisphere}_space-{space}_den-{density}_label-hipp_midthickness.surf.gii"
    )


def _native_area_metric_path(
    surface_root: Path, subject: str, hemisphere: str
) -> Path:
    return (
        surface_root
        / subject
        / "hippunfold"
        / subject
        / "metric"
        / f"{subject}_hemi-{hemisphere}_den-native_label-hipp_surfareacorobl.shape.gii"
    )


def _load_gifti_vector(path: Path, *, dtype: Any, label: str) -> np.ndarray:
    if not path.is_file():
        raise FileNotFoundError(f"required Kember-method input is missing: {path}")
    image = nib.load(str(path))
    if not isinstance(image, GiftiImage) or len(image.darrays) != 1:
        raise ValueError(f"{label} must be a single-array GIFTI file: {path}")
    values = np.asarray(image.darrays[0].data, dtype=dtype).reshape(-1)
    if not values.size:
        raise ValueError(f"{label} is empty: {path}")
    return values


def _load_vector(path: Path, *, dtype: Any) -> np.ndarray:
    if not path.is_file():
        raise FileNotFoundError(f"required Kember-method input is missing: {path}")
    return np.asarray(np.load(path, allow_pickle=False), dtype=dtype).reshape(-1)


def _template_density_evidence(
    combo_manifest_path: Path,
    *,
    actual_surface_evidence: Mapping[str, object],
    n_vertices: int,
) -> dict[str, object]:
    combo_manifest = _read_json_mapping(
        combo_manifest_path, label="template combo manifest"
    )
    source_density = combo_manifest.get("density_evidence")
    source_evidence = _require_asset_backed_density_evidence(
        source_density if isinstance(source_density, Mapping) else {},
        n_vertices=n_vertices,
    )
    source_fingerprint = source_evidence.get("surface_fingerprint")
    actual_fingerprint = actual_surface_evidence.get("surface_fingerprint")
    source_sha256 = (
        source_fingerprint.get("sha256")
        if isinstance(source_fingerprint, Mapping)
        else None
    )
    actual_sha256 = (
        actual_fingerprint.get("sha256")
        if isinstance(actual_fingerprint, Mapping)
        else None
    )
    if source_sha256 != actual_sha256:
        raise ValueError(
            "template combo manifest surface fingerprint does not match the "
            f"consumed midthickness surface: {combo_manifest_path}"
        )
    evidence = dict(actual_surface_evidence)
    evidence["template_combo_manifest"] = str(combo_manifest_path.resolve())
    evidence["template_manifest_density_evidence"] = source_evidence
    return evidence


def _require_authorized_source_lock(source_lock: Mapping[str, object]) -> None:
    implementation = source_lock.get("kember_method_area_implementation")
    if not isinstance(implementation, Mapping):
        raise ValueError("source lock lacks kember_method_area_implementation")
    if implementation.get("next_action_requires_user_authorization") is not False:
        raise PermissionError(
            "source lock still requires explicit user authorization before "
            "Kember-method area calculation"
        )


def _resolve_workbench_command(value: str | Path | None) -> Path:
    if value is None:
        discovered = shutil.which("wb_command")
        if discovered is None:
            raise FileNotFoundError(
                "Connectome Workbench is required for native-label resampling; "
                "pass workbench_command explicitly when wb_command is not on PATH"
            )
        candidate = Path(discovered)
    else:
        candidate = Path(value)
    if not candidate.is_file() or not candidate.stat().st_mode & 0o111:
        raise FileNotFoundError(
            f"Connectome Workbench command is not an executable file: {candidate}"
        )
    return candidate.resolve()


def _write_label_gifti(path: Path, labels: np.ndarray) -> None:
    values = np.asarray(labels, dtype=np.int32).reshape(-1)
    unknown = sorted(set(np.unique(values)) - {0, 1, 2, 3})
    if unknown:
        raise ValueError(f"native-label source has unsupported label(s): {unknown}")
    table = GiftiLabelTable()
    definitions = (
        (0, "Unassigned", (0.0, 0.0, 0.0, 0.0)),
        (1, "Anterior", (0.85, 0.25, 0.25, 1.0)),
        (2, "Body", (0.25, 0.65, 0.25, 1.0)),
        (3, "Posterior", (0.25, 0.35, 0.85, 1.0)),
    )
    for key, name, color in definitions:
        label = GiftiLabel(key=key)
        label.label = name
        label.red, label.green, label.blue, label.alpha = color
        table.labels.append(label)
    image = GiftiImage(
        darrays=[
            GiftiDataArray(data=values, intent="NIFTI_INTENT_LABEL"),
        ]
    )
    image.labeltable = table
    nib.save(image, str(path))


def _write_roi_gifti(path: Path, valid_mask: np.ndarray) -> None:
    values = np.asarray(valid_mask, dtype=np.float32).reshape(-1)
    image = GiftiImage(
        darrays=[
            GiftiDataArray(data=values, intent="NIFTI_INTENT_SHAPE"),
        ]
    )
    nib.save(image, str(path))


def _normalise_resampled_labels(
    labels: np.ndarray, native_valid_roi: np.ndarray
) -> np.ndarray:
    """Set Workbench fill values outside the native valid ROI to unassigned."""

    values = np.asarray(labels, dtype=np.int32).reshape(-1)
    valid = np.asarray(native_valid_roi, dtype=bool).reshape(-1)
    if values.size != valid.size:
        raise ValueError("resampled native labels and valid ROI have different lengths")
    normalised = np.where(valid, values, 0).astype(np.int32)
    unknown = sorted(set(np.unique(normalised)) - {0, 1, 2, 3})
    if unknown:
        raise ValueError(f"resampled native labels have unsupported label(s): {unknown}")
    return normalised


def align_labels_to_native(
    entry: Mapping[str, object],
    *,
    output_dir: Path,
    workbench_command: Path,
) -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
    """Map fixed 512 labels to native vertices without generating new labels."""

    branch = str(entry["branch"])
    subject = str(entry["subject"])
    smoothing_mm = int(entry["smoothing_mm"])
    hemisphere = str(entry["hemisphere"])
    source_labels = np.asarray(entry["labels"], dtype=np.int32).reshape(-1)
    source_valid = np.asarray(entry["valid_mask"], dtype=bool).reshape(-1)
    if source_labels.size != source_valid.size:
        raise ValueError("512 labels and valid mask have different lengths")
    source_labels = np.where(source_valid, source_labels, 0).astype(np.int32)
    artifact_dir = (
        Path(output_dir)
        / "label_alignment"
        / branch
        / subject
        / f"{smoothing_mm}mm"
        / f"hemi-{hemisphere}"
    )
    artifact_dir.mkdir(parents=True, exist_ok=True)
    source_label_path = artifact_dir / "source_512_labels.label.gii"
    source_roi_path = artifact_dir / "source_512_valid_roi.shape.gii"
    native_label_path = artifact_dir / "native_labels.label.gii"
    native_valid_roi_path = artifact_dir / "native_valid_roi.shape.gii"
    _write_label_gifti(source_label_path, source_labels)
    _write_roi_gifti(source_roi_path, source_valid)

    source_unfold_surface = Path(entry["source_unfold_surface_path"])
    native_unfold_surface = Path(entry["native_unfold_surface_path"])
    source_area_surface = Path(entry["source_surface_path"])
    native_area_surface = Path(entry["surface_path"])
    command = [
        str(workbench_command),
        "-label-resample",
        str(source_label_path),
        str(source_unfold_surface),
        str(native_unfold_surface),
        "ADAP_BARY_AREA",
        str(native_label_path),
        "-area-surfs",
        str(source_area_surface),
        str(native_area_surface),
        "-current-roi",
        str(source_roi_path),
        "-valid-roi-out",
        str(native_valid_roi_path),
        "-bypass-sphere-check",
    ]
    completed = subprocess.run(
        command,
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            "Connectome Workbench native-label resampling failed: "
            f"{completed.stderr.strip() or completed.stdout.strip()}"
        )
    native_labels = _load_gifti_vector(
        native_label_path,
        dtype=np.int32,
        label="resampled native labels",
    )
    native_valid = _load_gifti_vector(
        native_valid_roi_path,
        dtype=np.float64,
        label="resampled native valid ROI",
    ) > 0
    native_area = np.asarray(entry["vertex_area_mm2"], dtype=np.float64).reshape(-1)
    if native_labels.size != native_area.size or native_valid.size != native_area.size:
        raise ValueError(
            "resampled native labels/ROI do not match the native per-vertex area metric"
        )
    native_labels = _normalise_resampled_labels(native_labels, native_valid)
    return native_labels, native_valid, {
        "method": "ADAP_BARY_AREA",
        "registration_space": "HippUnfold unfolded midthickness",
        "bypass_sphere_check": True,
        "workbench_command": str(workbench_command),
        "source_registration_surface": _file_provenance(source_unfold_surface),
        "native_registration_surface": _file_provenance(native_unfold_surface),
        "source_area_surface": _file_provenance(source_area_surface),
        "native_area_surface": _file_provenance(native_area_surface),
        "source_labels": _file_provenance(source_label_path, array=source_labels),
        "source_valid_roi": _file_provenance(source_roi_path, array=source_valid),
        "native_labels": _file_provenance(native_label_path, array=native_labels),
        "native_valid_roi": _file_provenance(
            native_valid_roi_path, array=native_valid
        ),
    }


def _prepare_branch_input(
    *,
    root: Path,
    config: Mapping[str, object],
    branch: str,
    subject: str,
    smoothing_mm: int,
    hemisphere: str,
) -> dict[str, object]:
    surface_root = _configured_path(root, config, "hippunfold_root")
    fixed_root = _configured_path(root, config, "hcp151_template_root")
    matching_root = _configured_path(root, config, "template_matching_root")
    assert surface_root is not None
    assert fixed_root is not None
    assert matching_root is not None
    source_surface_path = _surface_path(
        surface_root,
        subject,
        hemisphere,
        space="corobl",
        density="512",
    )
    native_surface_path = _surface_path(
        surface_root,
        subject,
        hemisphere,
        space="corobl",
        density="native",
    )
    source_unfold_surface_path = _surface_path(
        surface_root,
        subject,
        hemisphere,
        space="unfold",
        density="512",
    )
    native_unfold_surface_path = _surface_path(
        surface_root,
        subject,
        hemisphere,
        space="unfold",
        density="native",
    )
    native_area_metric_path = _native_area_metric_path(
        surface_root, subject, hemisphere
    )
    source_coordinates, _ = load_midthickness(source_surface_path)
    native_coordinates, _ = load_midthickness(native_surface_path)
    source_surface_evidence = _surface_density_evidence(
        source_surface_path, source_coordinates
    )
    native_surface_evidence = _surface_density_evidence(
        native_surface_path, native_coordinates
    )
    source_n_vertices = int(source_coordinates.shape[0])
    native_n_vertices = int(native_coordinates.shape[0])
    vertex_area_mm2 = _load_gifti_vector(
        native_area_metric_path,
        dtype=np.float64,
        label="native HippUnfold per-vertex area metric",
    )
    if vertex_area_mm2.size != native_n_vertices:
        raise ValueError(
            "native HippUnfold per-vertex area metric does not match the "
            "consumed native midthickness surface"
        )
    if not np.isfinite(vertex_area_mm2).all() or np.any(vertex_area_mm2 < 0):
        raise ValueError("native HippUnfold per-vertex area metric is invalid")
    resampling_surfaces = {
        "source_unfold_surface": _file_provenance(source_unfold_surface_path),
        "native_unfold_surface": _file_provenance(native_unfold_surface_path),
        "method": "ADAP_BARY_AREA",
        "bypass_sphere_check": True,
    }
    native_area_inputs = {
        "native_area_surface": native_surface_evidence,
        "native_area_metric": _file_provenance(
            native_area_metric_path, array=vertex_area_mm2
        ),
        "label_resampling": resampling_surfaces,
    }
    entry: dict[str, object] = {
        "branch": branch,
        "subject": subject,
        "smoothing_mm": smoothing_mm,
        "hemisphere": hemisphere,
        "source_surface_path": source_surface_path,
        "surface_path": native_surface_path,
        "source_unfold_surface_path": source_unfold_surface_path,
        "native_unfold_surface_path": native_unfold_surface_path,
        "vertex_area_mm2": vertex_area_mm2,
        "density_evidence": native_surface_evidence,
        "source_density_evidence": source_surface_evidence,
        "native_area_metric_path": native_area_metric_path,
    }

    if branch == "hcp151_fixed":
        labels_path = fixed_root / f"hcp151_{smoothing_mm}mm_hemi-{hemisphere}_labels.npy"
        support_path = fixed_root / f"hcp151_{smoothing_mm}mm_hemi-{hemisphere}_support.npy"
        raw_labels = _load_vector(labels_path, dtype=np.int32)
        support = _load_vector(support_path, dtype=bool)
        if raw_labels.size != source_n_vertices or support.size != source_n_vertices:
            raise ValueError("fixed labels/support do not match the 512 label surface")
        labels = np.where(support & np.isin(raw_labels, (1, 2, 3)), raw_labels, 0)
        entry.update(
            {
                "labels": labels,
                "valid_mask": support,
                "consumed_input": {
                    "hemisphere": hemisphere,
                    "label_source_surface": source_surface_evidence,
                    "labels": _file_provenance(labels_path, array=raw_labels),
                    "support": _file_provenance(support_path, array=support),
                    **native_area_inputs,
                },
            }
        )
        return entry

    if branch == "hcp151_template_matching":
        matching = (
            matching_root
            / "subjects"
            / subject
            / "matching"
            / f"{smoothing_mm}mm"
            / f"hemi-{hemisphere}"
        )
        labels_path = matching / "labels.npy"
        valid_mask_path = matching / "valid_mask.npy"
        combo_manifest_path = matching / "combo_manifest.json"
        raw_labels = _load_vector(labels_path, dtype=np.int32)
        valid_mask = _load_vector(valid_mask_path, dtype=bool)
        if raw_labels.size != source_n_vertices or valid_mask.size != source_n_vertices:
            raise ValueError("template-matching arrays do not match the 512 label surface")
        evidence = _template_density_evidence(
            combo_manifest_path,
            actual_surface_evidence=source_surface_evidence,
            n_vertices=source_n_vertices,
        )
        labels = np.where(
            valid_mask & np.isin(raw_labels, (1, 2, 3)), raw_labels, 0
        )
        entry.update(
            {
                "labels": labels,
                "valid_mask": valid_mask,
                "consumed_input": {
                    "hemisphere": hemisphere,
                    "label_source_surface": evidence,
                    "labels": _file_provenance(labels_path, array=raw_labels),
                    "valid_mask": _file_provenance(valid_mask_path, array=valid_mask),
                    "combo_manifest": _file_provenance(combo_manifest_path),
                    **native_area_inputs,
                },
            }
        )
        return entry

    raise ValueError(f"unsupported Kember-method branch: {branch!r}")


def run_kember_method_area(
    *,
    root: Path,
    config_path: Path,
    source_lock_path: Path,
    output_dir: Path,
    subject: str,
    smoothing_mm: int,
    workbench_command: str | Path | None = None,
    branches: Sequence[str] = (
        "hcp151_fixed",
        "hcp151_template_matching",
    ),
) -> dict[str, object]:
    """Run one dual-hemisphere, all-inputs-frozen Kember-method area pilot."""

    root_path = Path(root).resolve()
    config = _read_json_mapping(Path(config_path), label="input configuration")
    source_lock = _read_json_mapping(Path(source_lock_path), label="source lock")
    _require_authorized_source_lock(source_lock)
    selected_branches = tuple(str(branch) for branch in branches)
    if not selected_branches or len(selected_branches) != len(set(selected_branches)):
        raise ValueError("branches must be a non-empty, duplicate-free sequence")
    unsupported = sorted(set(selected_branches) - BRANCHES)
    if unsupported:
        raise ValueError(f"unsupported Kember-method branches: {unsupported}")
    configured_smoothings = config.get("smoothings_mm")
    if not isinstance(configured_smoothings, list) or int(smoothing_mm) not in {
        int(value) for value in configured_smoothings
    }:
        raise ValueError(f"smoothing {smoothing_mm}mm is not configured")
    input_contract = audit_stanford_input_contract(Path(config_path), root=root_path)
    if subject not in input_contract["common_subjects"]:
        raise ValueError(f"subject is not in the frozen common cohort: {subject}")
    resolved_workbench_command = _resolve_workbench_command(workbench_command)

    prepared: list[dict[str, object]] = []
    for branch in selected_branches:
        for hemisphere in ("L", "R"):
            prepared.append(
                _prepare_branch_input(
                    root=root_path,
                    config=config,
                    branch=branch,
                    subject=subject,
                    smoothing_mm=int(smoothing_mm),
                    hemisphere=hemisphere,
                )
            )
    consumed_inputs: dict[str, list[object]] = {
        branch: [
            entry["consumed_input"]
            for entry in prepared
            if entry["branch"] == branch
        ]
        for branch in selected_branches
    }
    freeze_result = freeze_input_manifest(
        Path(output_dir),
        source_lock=source_lock,
        input_contract=input_contract,
        consumed_inputs=consumed_inputs,
    )
    frozen_manifest_path = Path(str(freeze_result["input_manifest_path"]))

    records: list[dict[str, object]] = []
    label_resampling: list[dict[str, object]] = []
    for entry in prepared:
        branch = str(entry["branch"])
        labels, valid_mask, resampling = align_labels_to_native(
            entry,
            output_dir=Path(output_dir),
            workbench_command=resolved_workbench_command,
        )
        label_resampling.append(
            {
                "branch": branch,
                "subject": entry["subject"],
                "smoothing_mm": entry["smoothing_mm"],
                "hemisphere": entry["hemisphere"],
                **resampling,
            }
        )
        record = evaluate_hemisphere_area(
            branch=branch,
            subject=str(entry["subject"]),
            smoothing_mm=int(entry["smoothing_mm"]),
            hemisphere=str(entry["hemisphere"]),
            surface_path=Path(entry["surface_path"]),
            vertex_area_mm2=np.asarray(entry["vertex_area_mm2"], dtype=np.float64),
            labels=labels,
            valid_mask=valid_mask,
            density_evidence=entry["density_evidence"],
        )
        records.append(record)

    runtime_provenance = {
        "mode": "small_dual_hemi_pilot",
        "subject": subject,
        "smoothing_mm": int(smoothing_mm),
        "branches": list(selected_branches),
        "input_manifest_sha256": _sha256(frozen_manifest_path),
        "native_label_resampling": label_resampling,
    }
    return write_area_run(
        Path(output_dir),
        records=records,
        input_contract=input_contract,
        source_lock=source_lock,
        runtime_provenance=runtime_provenance,
    )
