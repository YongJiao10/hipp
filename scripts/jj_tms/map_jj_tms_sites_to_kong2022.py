#!/usr/bin/env python3
"""Map JJ_TMS volume-space targets to individualized Kong2022 surface labels."""

from __future__ import annotations

import argparse
import csv
import json
import posixpath
import re
import sys
import tempfile
import zipfile
from collections import Counter
from pathlib import Path, PurePosixPath
from typing import Any
from xml.etree import ElementTree as ET

import nibabel as nib
import numpy as np
from scipy.spatial import cKDTree

REPO_ROOT = Path(__file__).resolve().parents[2]
CORE_WORKFLOW_DIR = REPO_ROOT / "scripts" / "core_workflow"
if str(CORE_WORKFLOW_DIR) not in sys.path:
    sys.path.insert(0, str(CORE_WORKFLOW_DIR))

from lib.kong2022_label_utils import (
    KONG2022_DLABEL_PATH,
    KONG2022_MERGE_CONFIG,
    load_kong2022_bundle,
)


SHEET_NS = "http://schemas.openxmlformats.org/spreadsheetml/2006/main"
OFFICE_REL_NS = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"
PACKAGE_REL_NS = "http://schemas.openxmlformats.org/package/2006/relationships"
CELL_REF_RE = re.compile(r"^(?P<column>[A-Z]+)(?P<row>[1-9][0-9]*)$")

DEFAULT_WORKBOOK = REPO_ROOT / "outputs_iowa_jj_tms" / "TMSfMRI_DataSharing.xlsx"
DEFAULT_SUBJECTS_FILE = REPO_ROOT / "manifests" / "jj_tms_kong2022_70_subjects.txt"
DEFAULT_JJ_TMS_ROOT = Path(
    "/home/yojiao/LSS/jianglab/3_Data_Working/fmriprep_processed_JJ_TMS"
)
DEFAULT_SURFACE_ROOT = DEFAULT_JJ_TMS_ROOT / "derivatives_surf"
DEFAULT_LABEL_ROOT = DEFAULT_JJ_TMS_ROOT / "ArealMSHBM_Kong2022"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "outputs_iowa_jj_tms" / "present" / "tms_site_networks"
CSV_NAME = "jj_tms_kong2022_tms_site_networks.csv"
SUMMARY_NAME = "jj_tms_kong2022_tms_site_networks_summary.json"
COMPARISON_CSV_NAME = "jj_tms_kong2022_individual_vs_group_atlas.csv"
COMPARISON_SUMMARY_NAME = (
    "jj_tms_kong2022_individual_vs_group_atlas_summary.json"
)
FSLR_32K_VERTICES_PER_HEMISPHERE = 32492
REQUIRED_WORKBOOK_HEADERS = frozenset(
    {
        "record_subject_id",
        "visit2_target",
        "native_x",
        "native_y",
        "native_z",
        "mni_x",
        "mni_y",
        "mni_z",
        "visit2_target_notes",
    }
)


def _column_index(cell_reference: str) -> int:
    match = CELL_REF_RE.fullmatch(cell_reference)
    if match is None:
        raise ValueError(f"Invalid XLSX cell reference: {cell_reference!r}")
    result = 0
    for character in match.group("column"):
        result = result * 26 + ord(character) - ord("A") + 1
    return result - 1


def _shared_strings(archive: zipfile.ZipFile) -> list[str]:
    try:
        root = ET.fromstring(archive.read("xl/sharedStrings.xml"))
    except KeyError:
        return []
    return [
        "".join(node.text or "" for node in item.iter(f"{{{SHEET_NS}}}t"))
        for item in root.findall(f"{{{SHEET_NS}}}si")
    ]


def _worksheet_archive_path(archive: zipfile.ZipFile, sheet_name: str) -> str:
    workbook = ET.fromstring(archive.read("xl/workbook.xml"))
    relationship_id: str | None = None
    sheets = workbook.find(f"{{{SHEET_NS}}}sheets")
    if sheets is None:
        raise ValueError("XLSX workbook has no sheets")
    for sheet in sheets:
        if sheet.attrib.get("name") == sheet_name:
            relationship_id = sheet.attrib.get(f"{{{OFFICE_REL_NS}}}id")
            break
    if relationship_id is None:
        available = [sheet.attrib.get("name", "") for sheet in sheets]
        raise KeyError(f"XLSX sheet {sheet_name!r} not found; available={available}")

    relationships = ET.fromstring(archive.read("xl/_rels/workbook.xml.rels"))
    for relationship in relationships.findall(f"{{{PACKAGE_REL_NS}}}Relationship"):
        if relationship.attrib.get("Id") != relationship_id:
            continue
        target = relationship.attrib.get("Target")
        if not target:
            break
        if target.startswith("/"):
            normalized = posixpath.normpath(target.lstrip("/"))
        else:
            normalized = posixpath.normpath(str(PurePosixPath("xl") / target))
        if normalized == ".." or normalized.startswith("../"):
            raise ValueError(
                f"Worksheet relationship escapes the XLSX package root: {target!r}"
            )
        return normalized
    raise ValueError(f"No worksheet relationship target for {sheet_name!r}")


def _cell_value(cell: ET.Element, shared_strings: list[str]) -> str | None:
    cell_type = cell.attrib.get("t")
    if cell_type == "inlineStr":
        return "".join(node.text or "" for node in cell.iter(f"{{{SHEET_NS}}}t"))
    value_node = cell.find(f"{{{SHEET_NS}}}v")
    if value_node is None or value_node.text is None:
        return None
    value = value_node.text
    if cell_type == "s":
        return shared_strings[int(value)]
    return value


def read_xlsx_rows(
    path: Path,
    sheet_name: str,
    *,
    required_headers: set[str] | frozenset[str] | None = None,
) -> list[dict[str, str | None]]:
    """Read an XLSX worksheet as string-valued records without optional engines."""

    with zipfile.ZipFile(path) as archive:
        shared_strings = _shared_strings(archive)
        worksheet_path = _worksheet_archive_path(archive, sheet_name)
        worksheet = ET.fromstring(archive.read(worksheet_path))

    xml_rows = worksheet.findall(f".//{{{SHEET_NS}}}sheetData/{{{SHEET_NS}}}row")
    if not xml_rows:
        missing_headers = sorted(set(required_headers or ()))
        if missing_headers:
            raise KeyError(
                f"XLSX sheet {sheet_name!r} is missing required columns: "
                + ",".join(missing_headers)
            )
        return []

    sparse_rows: list[dict[int, str | None]] = []
    for xml_row in xml_rows:
        sparse: dict[int, str | None] = {}
        for cell in xml_row.findall(f"{{{SHEET_NS}}}c"):
            reference = cell.attrib.get("r")
            if reference is None:
                raise ValueError("XLSX cell is missing its reference")
            sparse[_column_index(reference)] = _cell_value(cell, shared_strings)
        sparse_rows.append(sparse)

    headers = sparse_rows[0]
    header_names = [value for value in headers.values() if value not in (None, "")]
    header_name_set = set(header_names)
    if len(header_name_set) != len(header_names):
        raise ValueError("XLSX worksheet contains duplicate column headers")
    if required_headers is not None:
        missing_headers = sorted(set(required_headers) - header_name_set)
        if missing_headers:
            raise KeyError(
                f"XLSX sheet {sheet_name!r} is missing required columns: "
                + ",".join(missing_headers)
            )

    records: list[dict[str, str | None]] = []
    for sparse in sparse_rows[1:]:
        record = {
            str(header): sparse.get(column)
            for column, header in headers.items()
            if header not in (None, "")
        }
        records.append(record)
    return records


def project_target_to_surface(
    *,
    target_xyz: np.ndarray,
    surface_xyz: np.ndarray,
    parcel_labels: np.ndarray,
    roi_by_id: dict[int, dict[str, Any]],
    neighborhood_radius_mm: float,
    expected_hemi: str | None = None,
) -> dict[str, object]:
    """Assign a world-coordinate target to the closest labeled surface vertex."""

    target = np.asarray(target_xyz, dtype=np.float64)
    surface = np.asarray(surface_xyz, dtype=np.float64)
    labels = np.asarray(parcel_labels, dtype=np.int32).reshape(-1)
    if target.shape != (3,) or not np.all(np.isfinite(target)):
        raise ValueError("target_xyz must contain three finite world coordinates")
    if surface.ndim != 2 or surface.shape[1] != 3 or not np.all(np.isfinite(surface)):
        raise ValueError("surface_xyz must be a finite n-by-3 array")
    if surface.shape[0] != labels.shape[0]:
        raise ValueError(
            f"surface vertices ({surface.shape[0]}) and parcel labels ({labels.shape[0]}) differ"
        )
    if surface.shape[0] == 0:
        raise ValueError("surface_xyz is empty")
    if not np.isfinite(neighborhood_radius_mm) or neighborhood_radius_mm <= 0:
        raise ValueError("neighborhood_radius_mm must be finite and positive")
    if expected_hemi not in (None, "L", "R"):
        raise ValueError(f"expected_hemi must be L or R, got {expected_hemi!r}")

    tree = cKDTree(surface)
    distance, vertex_index = tree.query(target, k=1)
    vertex_index = int(vertex_index)
    parcel_id = int(labels[vertex_index])
    if parcel_id == 0:
        raise ValueError("Nearest surface vertex has medial-wall parcel 0")
    if parcel_id not in roi_by_id:
        raise ValueError(f"Nearest surface vertex has unknown Kong2022 parcel {parcel_id}")
    roi = roi_by_id[parcel_id]
    parcel_hemi = str(roi["hemi"])
    if expected_hemi is not None and parcel_hemi != expected_hemi:
        raise ValueError(
            f"Nearest Kong2022 parcel {parcel_id} belongs to hemisphere {parcel_hemi}, "
            f"expected {expected_hemi}"
        )

    neighborhood_indices = sorted(tree.query_ball_point(target, r=neighborhood_radius_mm))
    valid_neighborhood: list[int] = []
    for index in neighborhood_indices:
        neighborhood_parcel = int(labels[index])
        if neighborhood_parcel == 0:
            continue
        if neighborhood_parcel not in roi_by_id:
            raise ValueError(
                f"Neighborhood contains unknown Kong2022 parcel {neighborhood_parcel}"
            )
        neighborhood_hemi = str(roi_by_id[neighborhood_parcel]["hemi"])
        if expected_hemi is not None and neighborhood_hemi != expected_hemi:
            raise ValueError(
                f"Neighborhood parcel {neighborhood_parcel} belongs to hemisphere "
                f"{neighborhood_hemi}, expected {expected_hemi}"
            )
        valid_neighborhood.append(index)
    if not valid_neighborhood:
        valid_neighborhood = [vertex_index]
    neighborhood_families = [
        str(roi_by_id[int(labels[index])]["family17"]) for index in valid_neighborhood
    ]
    family_counts = Counter(neighborhood_families)
    dominant_family, dominant_count = sorted(
        family_counts.items(), key=lambda item: (-item[1], item[0])
    )[0]
    primary_family = str(roi["family17"])
    neighborhood_count = len(valid_neighborhood)

    return {
        "vertex_index": vertex_index,
        "vertex_x": float(surface[vertex_index, 0]),
        "vertex_y": float(surface[vertex_index, 1]),
        "vertex_z": float(surface[vertex_index, 2]),
        "target_to_surface_distance_mm": float(distance),
        "parcel_id": parcel_id,
        "parcel_name": str(roi["roi_name"]),
        "hemi": parcel_hemi,
        "family17": primary_family,
        "canonical8": str(roi["canonical8"]),
        "neighborhood_radius_mm": float(neighborhood_radius_mm),
        "neighborhood_vertex_count": neighborhood_count,
        "neighborhood_primary_family17_fraction": (
            family_counts[primary_family] / neighborhood_count
        ),
        "neighborhood_dominant_family17": dominant_family,
        "neighborhood_dominant_family17_fraction": dominant_count / neighborhood_count,
    }


def normalize_subject_id(value: str) -> str:
    text = str(value).strip()
    match = re.fullmatch(r"(?:(?:JJ_TMS_|sub-))?(\d+)", text, flags=re.IGNORECASE)
    if match is None:
        raise ValueError(f"Invalid JJ_TMS subject identifier: {value!r}")
    return match.group(1)


def _required_float(record: dict[str, str | None], field: str, subject: str) -> float:
    value = record.get(field)
    if value in (None, ""):
        raise ValueError(f"sub-{subject} is missing required workbook field {field}")
    try:
        result = float(str(value))
    except ValueError as exc:
        raise ValueError(f"sub-{subject} has non-numeric {field}={value!r}") from exc
    if not np.isfinite(result):
        raise ValueError(f"sub-{subject} has non-finite {field}={value!r}")
    return result


def _optional_float(record: dict[str, str | None], field: str, subject: str) -> float | None:
    value = record.get(field)
    if value in (None, ""):
        return None
    return _required_float(record, field, subject)


def _target_hemisphere(target: str | None, native_x: float, subject: str) -> str:
    normalized = str(target or "").strip().upper()
    target_to_hemi = {"LDLPFC": "L", "RDLPFC": "R"}
    if normalized not in target_to_hemi:
        raise ValueError(f"sub-{subject} has unsupported visit2_target={target!r}")
    hemi = target_to_hemi[normalized]
    if (hemi == "L" and native_x >= 0) or (hemi == "R" and native_x <= 0):
        raise ValueError(
            f"sub-{subject}: visit2_target={normalized} disagrees with native_x={native_x}"
        )
    return hemi


def _load_surface_xyz(path: Path, *, expected_hemi: str) -> np.ndarray:
    if expected_hemi not in {"L", "R"}:
        raise ValueError(f"expected_hemi must be L or R, got {expected_hemi!r}")
    if not path.is_file():
        raise FileNotFoundError(f"Missing subject surface: {path}")
    image = nib.load(str(path))
    pointset_code = nib.nifti1.intent_codes.code["NIFTI_INTENT_POINTSET"]
    expected_structure = "CortexLeft" if expected_hemi == "L" else "CortexRight"
    for data_array in image.darrays:
        if int(data_array.intent) != int(pointset_code):
            continue
        metadata = dict(data_array.meta)
        if metadata.get("AnatomicalStructurePrimary") != expected_structure:
            raise ValueError(
                f"Surface anatomical structure is not {expected_structure}: {path}"
            )
        if metadata.get("AnatomicalStructureSecondary") != "MidThickness":
            raise ValueError(f"Surface is not marked as MidThickness: {path}")
        if metadata.get("GeometricType") != "Anatomical":
            raise ValueError(f"Surface is not marked as Anatomical geometry: {path}")
        if data_array.coordsys is None:
            raise ValueError(f"Surface pointset has no coordinate-system metadata: {path}")
        transform = np.asarray(data_array.coordsys.xform, dtype=np.float64)
        if transform.shape != (4, 4) or not np.allclose(
            transform,
            np.eye(4, dtype=np.float64),
            rtol=0.0,
            atol=1e-6,
        ):
            raise ValueError(f"Surface pointset has non-identity coordinate transform: {path}")
        coordinates = np.asarray(data_array.data, dtype=np.float64)
        vertex_count = int(coordinates.shape[0]) if coordinates.ndim >= 1 else 0
        if vertex_count != FSLR_32K_VERTICES_PER_HEMISPHERE:
            raise ValueError(
                f"fsLR den-32k surface expected "
                f"{FSLR_32K_VERTICES_PER_HEMISPHERE} vertices, "
                f"found {vertex_count}: {path}"
            )
        return coordinates
    raise ValueError(f"GIFTI surface has no pointset coordinates: {path}")


def _load_parcel_labels(path: Path) -> np.ndarray:
    if not path.is_file():
        raise FileNotFoundError(f"Missing individualized Kong2022 labels: {path}")
    return np.asarray(np.loadtxt(path, dtype=np.int32), dtype=np.int32).reshape(-1)


def build_dense_group_parcel_labels(
    vertex_arrays: dict[str, tuple[np.ndarray, np.ndarray]],
    *,
    vertex_count: int = FSLR_32K_VERTICES_PER_HEMISPHERE,
) -> dict[str, np.ndarray]:
    """Restore group dlabel cortex data to full fsLR vertex indexing."""

    if vertex_count <= 0:
        raise ValueError("vertex_count must be positive")
    result: dict[str, np.ndarray] = {}
    for hemi in ("L", "R"):
        if hemi not in vertex_arrays:
            raise KeyError(f"Group Kong2022 dlabel is missing hemisphere {hemi}")
        raw_indices, raw_parcels = vertex_arrays[hemi]
        indices = np.asarray(raw_indices, dtype=np.int64).reshape(-1)
        parcels = np.asarray(raw_parcels, dtype=np.int32).reshape(-1)
        if indices.shape[0] != parcels.shape[0]:
            raise ValueError(
                f"Group {hemi} vertex indices ({indices.shape[0]}) and "
                f"parcel labels ({parcels.shape[0]}) differ"
            )
        if np.any(indices < 0) or np.any(indices >= vertex_count):
            raise ValueError(
                f"Group {hemi} dlabel contains a vertex outside 0..{vertex_count - 1}"
            )
        if np.unique(indices).shape[0] != indices.shape[0]:
            raise ValueError(f"Group {hemi} dlabel contains duplicate vertex indices")
        if np.any(parcels < 0):
            raise ValueError(f"Group {hemi} dlabel contains negative parcel IDs")
        dense = np.zeros(vertex_count, dtype=np.int32)
        dense[indices] = parcels
        result[hemi] = dense
    return result


def _agreement_summary(
    rows: list[dict[str, object]],
    field: str,
) -> dict[str, object]:
    different_subjects = [
        str(row["subject"])
        for row in rows
        if not bool(row[field])
    ]
    n_compared = len(rows)
    n_different = len(different_subjects)
    n_same = n_compared - n_different
    return {
        "n_same": n_same,
        "n_different": n_different,
        "agreement_fraction": n_same / n_compared if n_compared else None,
        "different_subjects": different_subjects,
    }


def compare_with_group_atlas(
    rows: list[dict[str, object]],
    *,
    group_parcel_labels: dict[str, np.ndarray],
    roi_registry: list[dict[str, object]],
) -> tuple[list[dict[str, object]], dict[str, object]]:
    """Annotate mapped rows from the common atlas at the identical fsLR vertex."""

    roi_by_id = {int(row["roi_id"]): row for row in roi_registry}
    if len(roi_by_id) != len(roi_registry):
        raise ValueError("Kong2022 ROI registry contains duplicate parcel IDs")

    compared_rows: list[dict[str, object]] = []
    successful_rows: list[dict[str, object]] = []
    for source_row in rows:
        row = dict(source_row)
        if row.get("mapping_status") != "mapped":
            compared_rows.append(row)
            continue
        hemi = str(row["hemi"])
        if hemi not in group_parcel_labels:
            raise KeyError(f"Group Kong2022 labels are missing hemisphere {hemi}")
        labels = np.asarray(group_parcel_labels[hemi], dtype=np.int32).reshape(-1)
        vertex_index = int(row["vertex_index"])
        if vertex_index < 0 or vertex_index >= labels.shape[0]:
            raise IndexError(
                f"{row['subject']} vertex {vertex_index} is outside group {hemi} labels"
            )
        group_parcel_id = int(labels[vertex_index])
        if group_parcel_id == 0:
            raise ValueError(
                f"{row['subject']} group atlas vertex {vertex_index} is medial wall"
            )
        if group_parcel_id not in roi_by_id:
            raise ValueError(
                f"{row['subject']} group atlas has unknown Kong2022 parcel "
                f"{group_parcel_id}"
            )
        group_roi = roi_by_id[group_parcel_id]
        group_hemi = str(group_roi["hemi"])
        if group_hemi != hemi:
            raise ValueError(
                f"{row['subject']} group parcel {group_parcel_id} belongs to "
                f"hemisphere {group_hemi}, expected {hemi}"
            )

        same_parcel = int(row["parcel_id"]) == group_parcel_id
        same_family17 = str(row["family17"]) == str(group_roi["family17"])
        same_canonical8 = str(row["canonical8"]) == str(group_roi["canonical8"])
        row.update(
            {
                "group_parcel_id": group_parcel_id,
                "group_parcel_name": str(group_roi["roi_name"]),
                "group_family17": str(group_roi["family17"]),
                "group_canonical8": str(group_roi["canonical8"]),
                "individual_vs_group_same_parcel": same_parcel,
                "individual_vs_group_same_family17": same_family17,
                "individual_vs_group_same_canonical8": same_canonical8,
            }
        )
        compared_rows.append(row)
        successful_rows.append(row)

    comparison = {
        "n_compared_subjects": len(successful_rows),
        "parcel": _agreement_summary(
            successful_rows,
            "individual_vs_group_same_parcel",
        ),
        "family17": _agreement_summary(
            successful_rows,
            "individual_vs_group_same_family17",
        ),
        "canonical8": _agreement_summary(
            successful_rows,
            "individual_vs_group_same_canonical8",
        ),
    }
    return compared_rows, comparison


def _count_strings(rows: list[dict[str, object]], field: str) -> dict[str, int]:
    counts = Counter(str(row[field]) for row in rows)
    return dict(sorted(counts.items()))


def _count_transitions(
    rows: list[dict[str, object]],
    individual_field: str,
    group_field: str,
) -> dict[str, int]:
    counts = Counter(
        f"{row[individual_field]} -> {row[group_field]}" for row in rows
    )
    return dict(sorted(counts.items()))


def map_cohort(
    *,
    workbook_path: Path,
    subjects: list[str],
    surface_root: Path,
    label_root: Path,
    roi_registry: list[dict[str, object]],
    neighborhood_radius_mm: float = 5.0,
) -> tuple[list[dict[str, object]], dict[str, object]]:
    workbook_records = read_xlsx_rows(
        workbook_path,
        "data",
        required_headers=REQUIRED_WORKBOOK_HEADERS,
    )
    records_by_subject: dict[str, dict[str, str | None]] = {}
    for record in workbook_records:
        raw_subject = record.get("record_subject_id")
        if raw_subject in (None, ""):
            continue
        subject = normalize_subject_id(str(raw_subject))
        if subject in records_by_subject:
            raise ValueError(f"Duplicate workbook record for sub-{subject}")
        records_by_subject[subject] = record

    normalized_subjects = [normalize_subject_id(subject) for subject in subjects]
    if len(set(normalized_subjects)) != len(normalized_subjects):
        raise ValueError("Requested subject roster contains duplicate identifiers")
    roi_by_id = {int(row["roi_id"]): row for row in roi_registry}
    if len(roi_by_id) != len(roi_registry):
        raise ValueError("Kong2022 ROI registry contains duplicate parcel IDs")

    mapped_rows: list[dict[str, object]] = []
    for subject in normalized_subjects:
        if subject not in records_by_subject:
            mapped_rows.append(
                {
                    "subject": f"sub-{subject}",
                    "mapping_status": "missing_workbook_record",
                    "mapping_error": "No record_subject_id match in workbook",
                }
            )
            continue
        record = records_by_subject[subject]
        missing_native = [
            field
            for field in ("native_x", "native_y", "native_z")
            if record.get(field) in (None, "")
        ]
        if missing_native:
            mapped_rows.append(
                {
                    "subject": f"sub-{subject}",
                    "mapping_status": "missing_native_coordinates",
                    "mapping_error": (
                        "Missing native coordinate fields: " + ",".join(missing_native)
                    ),
                    "visit2_target": str(
                        record.get("visit2_target") or ""
                    ).strip().upper(),
                    "mni_x": _optional_float(record, "mni_x", subject),
                    "mni_y": _optional_float(record, "mni_y", subject),
                    "mni_z": _optional_float(record, "mni_z", subject),
                    "target_notes": record.get("visit2_target_notes"),
                }
            )
            continue
        native = np.array(
            [
                _required_float(record, "native_x", subject),
                _required_float(record, "native_y", subject),
                _required_float(record, "native_z", subject),
            ],
            dtype=np.float64,
        )
        target = str(record.get("visit2_target") or "").strip().upper()
        hemi = _target_hemisphere(target, float(native[0]), subject)
        surface_path = (
            surface_root
            / f"sub-{subject}"
            / "anat"
            / f"sub-{subject}_hemi-{hemi}_space-fsLR_den-32k_midthickness.surf.gii"
        )
        if not surface_path.is_file():
            fallback_path = (
                REPO_ROOT
                / "outputs_iowa_jj_tms"
                / "inputs"
                / "surfaces_32k"
                / f"sub-{subject}"
                / "anat"
                / f"sub-{subject}_hemi-{hemi}_space-fsLR_den-32k_midthickness.surf.gii"
            )
            if fallback_path.is_file():
                surface_path = fallback_path
        label_path = (
            label_root
            / f"sub-{subject}"
            / ("lh_labels.txt" if hemi == "L" else "rh_labels.txt")
        )
        projection = project_target_to_surface(
            target_xyz=native,
            surface_xyz=_load_surface_xyz(surface_path, expected_hemi=hemi),
            parcel_labels=_load_parcel_labels(label_path),
            roi_by_id=roi_by_id,
            neighborhood_radius_mm=neighborhood_radius_mm,
            expected_hemi=hemi,
        )
        primary_fraction = float(projection["neighborhood_primary_family17_fraction"])
        dominant_disagrees = (
            projection["neighborhood_dominant_family17"] != projection["family17"]
        )
        mapped_rows.append(
            {
                "subject": f"sub-{subject}",
                "mapping_status": "mapped",
                "mapping_error": None,
                "visit2_target": target,
                "native_x": float(native[0]),
                "native_y": float(native[1]),
                "native_z": float(native[2]),
                "mni_x": _optional_float(record, "mni_x", subject),
                "mni_y": _optional_float(record, "mni_y", subject),
                "mni_z": _optional_float(record, "mni_z", subject),
                "target_notes": record.get("visit2_target_notes"),
                **projection,
                "qc_distance_gt_5mm": (
                    float(projection["target_to_surface_distance_mm"]) > 5.0
                ),
                "qc_neighborhood_primary_fraction_lt_0p5": primary_fraction < 0.5,
                "qc_neighborhood_dominant_disagrees": dominant_disagrees,
                "surface_path": str(surface_path.resolve()),
                "label_path": str(label_path.resolve()),
            }
        )

    successful_rows = [row for row in mapped_rows if row["mapping_status"] == "mapped"]
    unmapped_subjects: dict[str, list[str]] = {}
    for row in mapped_rows:
        status = str(row["mapping_status"])
        if status == "mapped":
            continue
        unmapped_subjects.setdefault(status, []).append(str(row["subject"]))

    if successful_rows:
        distances = np.asarray(
            [
                float(row["target_to_surface_distance_mm"])
                for row in successful_rows
            ],
            dtype=np.float64,
        )
        distance_summary: dict[str, float | None] = {
            "min": float(np.min(distances)),
            "median": float(np.median(distances)),
            "mean": float(np.mean(distances)),
            "p95": float(np.percentile(distances, 95)),
            "max": float(np.max(distances)),
        }
    else:
        distance_summary = {
            "min": None,
            "median": None,
            "mean": None,
            "p95": None,
            "max": None,
        }
    summary: dict[str, object] = {
        "schema_version": "1.0",
        "coordinate_source": "native_x/native_y/native_z",
        "coordinate_interpretation": "subject T1w world coordinates (RAS+, millimetres)",
        "projection_surface": "subject fsLR den-32k midthickness in native T1w geometry",
        "assignment_rule": "nearest surface vertex",
        "neighborhood_qc": (
            f"Euclidean vertices within {float(neighborhood_radius_mm):g} mm of target"
        ),
        "workbook_path": str(workbook_path.resolve()),
        "surface_root": str(surface_root.resolve()),
        "label_root": str(label_root.resolve()),
        "n_requested_subjects": len(normalized_subjects),
        "n_output_rows": len(mapped_rows),
        "n_mapped_subjects": len(successful_rows),
        "n_unmapped_subjects": len(mapped_rows) - len(successful_rows),
        "unmapped_subjects": dict(sorted(unmapped_subjects.items())),
        "neighborhood_radius_mm": float(neighborhood_radius_mm),
        "family17_counts": _count_strings(successful_rows, "family17"),
        "canonical8_counts": _count_strings(successful_rows, "canonical8"),
        "distance_mm": distance_summary,
        "qc_subjects": {
            "distance_gt_5mm": [
                str(row["subject"])
                for row in successful_rows
                if row["qc_distance_gt_5mm"]
            ],
            "neighborhood_primary_fraction_lt_0p5": [
                str(row["subject"])
                for row in successful_rows
                if row["qc_neighborhood_primary_fraction_lt_0p5"]
            ],
            "neighborhood_dominant_disagrees": [
                str(row["subject"])
                for row in successful_rows
                if row["qc_neighborhood_dominant_disagrees"]
            ],
        },
    }
    return mapped_rows, summary


def compare_existing_mapping_results(
    *,
    individual_results_csv: Path,
    group_vertex_arrays: dict[str, tuple[np.ndarray, np.ndarray]],
    roi_registry: list[dict[str, object]],
    group_atlas_path: Path,
    vertex_count: int = FSLR_32K_VERTICES_PER_HEMISPHERE,
) -> tuple[list[dict[str, object]], dict[str, object]]:
    """Compare saved individual assignments with the group atlas at the same vertex."""

    required_headers = {
        "subject",
        "mapping_status",
        "hemi",
        "vertex_index",
        "parcel_id",
        "family17",
        "canonical8",
    }
    with individual_results_csv.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        headers = set(reader.fieldnames or [])
        missing_headers = sorted(required_headers - headers)
        if missing_headers:
            raise KeyError(
                "Individual mapping CSV is missing required columns: "
                + ",".join(missing_headers)
            )
        source_rows: list[dict[str, object]] = [dict(row) for row in reader]
    if not source_rows:
        raise ValueError(f"Individual mapping CSV is empty: {individual_results_csv}")

    dense_group_labels = build_dense_group_parcel_labels(
        group_vertex_arrays,
        vertex_count=vertex_count,
    )
    compared_rows, agreement = compare_with_group_atlas(
        source_rows,
        group_parcel_labels=dense_group_labels,
        roi_registry=roi_registry,
    )
    successful_rows = [
        row for row in compared_rows if row["mapping_status"] == "mapped"
    ]
    summary: dict[str, object] = {
        "schema_version": "1.0",
        "comparison_method": (
            "reuse saved individualized fsLR-32k vertex_index; no target reprojection"
        ),
        "individual_results_csv": str(individual_results_csv.resolve()),
        "group_atlas_dlabel_path": str(group_atlas_path.resolve()),
        "n_output_rows": len(compared_rows),
        "n_compared_subjects": len(successful_rows),
        "n_unmapped_subjects": len(compared_rows) - len(successful_rows),
        "group_family17_counts": _count_strings(successful_rows, "group_family17"),
        "group_canonical8_counts": _count_strings(successful_rows, "group_canonical8"),
        "family17_transition_counts": _count_transitions(
            successful_rows,
            "family17",
            "group_family17",
        ),
        "canonical8_transition_counts": _count_transitions(
            successful_rows,
            "canonical8",
            "group_canonical8",
        ),
        "agreement": agreement,
    }
    return compared_rows, summary


def _validate_output_filename(name: str, label: str) -> None:
    if not name or Path(name).name != name or name in {".", ".."}:
        raise ValueError(f"{label} must be a plain filename, got {name!r}")


def write_outputs(
    rows: list[dict[str, object]],
    summary: dict[str, object],
    output_dir: Path,
    *,
    csv_name: str = CSV_NAME,
    summary_name: str = SUMMARY_NAME,
) -> tuple[Path, Path]:
    if not rows:
        raise ValueError("Cannot write an empty site-to-network result table")
    _validate_output_filename(csv_name, "csv_name")
    _validate_output_filename(summary_name, "summary_name")
    csv_path = output_dir / csv_name
    json_path = output_dir / summary_name
    existing = [path for path in (csv_path, json_path) if path.exists()]
    if existing:
        raise FileExistsError(f"Output already exists: {existing[0]}")
    output_dir.mkdir(parents=True, exist_ok=True)

    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        newline="",
        prefix=f".{csv_name}.",
        suffix=".tmp",
        dir=output_dir,
        delete=False,
    ) as stream:
        fieldnames = list(
            dict.fromkeys(key for row in rows for key in row)
        )
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
        csv_temp = Path(stream.name)
    csv_temp.replace(csv_path)

    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        prefix=f".{summary_name}.",
        suffix=".tmp",
        dir=output_dir,
        delete=False,
    ) as stream:
        json.dump(summary, stream, ensure_ascii=False, indent=2, sort_keys=True)
        stream.write("\n")
        json_temp = Path(stream.name)
    json_temp.replace(json_path)
    return csv_path, json_path


def read_subjects_file(path: Path) -> list[str]:
    subjects = [
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    if not subjects:
        raise ValueError(f"Subject roster is empty: {path}")
    return subjects


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Assign JJ_TMS native-volume target coordinates to individualized "
            "Kong2022 cortical parcels and networks."
        )
    )
    parser.add_argument("--workbook", type=Path, default=DEFAULT_WORKBOOK)
    parser.add_argument("--subjects-file", type=Path, default=DEFAULT_SUBJECTS_FILE)
    parser.add_argument("--surface-root", type=Path, default=DEFAULT_SURFACE_ROOT)
    parser.add_argument("--label-root", type=Path, default=DEFAULT_LABEL_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--neighborhood-radius-mm", type=float, default=5.0)
    parser.add_argument(
        "--compare-existing-individual-results",
        action="store_true",
        help=(
            "Reuse saved hemi/vertex_index assignments and compare the common "
            "Kong2022 atlas without repeating target-to-surface projection."
        ),
    )
    parser.add_argument(
        "--individual-results-csv",
        type=Path,
        default=DEFAULT_OUTPUT_DIR / CSV_NAME,
    )
    return parser.parse_args(argv)


def completion_lines(
    *,
    rows: list[dict[str, object]],
    summary: dict[str, object],
    csv_path: Path,
    json_path: Path,
) -> list[str]:
    return [
        f"Output rows: {len(rows)}",
        f"Mapped subjects: {int(summary['n_mapped_subjects'])}",
        f"CSV: {csv_path}",
        f"Summary: {json_path}",
    ]


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    bundle = load_kong2022_bundle()
    if args.compare_existing_individual_results:
        rows, summary = compare_existing_mapping_results(
            individual_results_csv=args.individual_results_csv,
            group_vertex_arrays=bundle["vertex_arrays"],
            roi_registry=bundle["roi_registry"],
            group_atlas_path=KONG2022_DLABEL_PATH,
        )
        summary["kong2022_merge_config_path"] = str(KONG2022_MERGE_CONFIG.resolve())
        csv_path, json_path = write_outputs(
            rows,
            summary,
            args.output_dir,
            csv_name=COMPARISON_CSV_NAME,
            summary_name=COMPARISON_SUMMARY_NAME,
        )
        print(f"Output rows: {len(rows)}")
        print(f"Compared subjects: {int(summary['n_compared_subjects'])}")
        print(f"CSV: {csv_path}")
        print(f"Summary: {json_path}")
        return 0

    rows, summary = map_cohort(
        workbook_path=args.workbook,
        subjects=read_subjects_file(args.subjects_file),
        surface_root=args.surface_root,
        label_root=args.label_root,
        roi_registry=bundle["roi_registry"],
        neighborhood_radius_mm=args.neighborhood_radius_mm,
    )
    summary["subjects_file"] = str(args.subjects_file.resolve())
    summary["kong2022_dlabel_path"] = str(KONG2022_DLABEL_PATH.resolve())
    summary["kong2022_merge_config_path"] = str(KONG2022_MERGE_CONFIG.resolve())
    csv_path, json_path = write_outputs(rows, summary, args.output_dir)
    for line in completion_lines(
        rows=rows,
        summary=summary,
        csv_path=csv_path,
        json_path=json_path,
    ):
        print(line)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
