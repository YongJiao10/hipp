#!/usr/bin/env python3
from __future__ import annotations

import importlib.util
import sys
import zipfile
from pathlib import Path

import numpy as np
import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
MODULE_PATH = REPO_ROOT / "scripts" / "core_workflow" / "map_jj_tms_sites_to_kong2022.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("jj_tms_target_network_mapping", MODULE_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load mapping module: {MODULE_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


module = _load_module()


def _write_minimal_xlsx(
    path: Path,
    *,
    worksheet_override: str | None = None,
) -> None:
    content_types = """<?xml version="1.0" encoding="UTF-8"?>
<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">
  <Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>
  <Default Extension="xml" ContentType="application/xml"/>
</Types>"""
    workbook = """<?xml version="1.0" encoding="UTF-8"?>
<workbook xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main"
 xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">
  <sheets><sheet name="data" sheetId="1" r:id="rId1"/></sheets>
</workbook>"""
    relationships = """<?xml version="1.0" encoding="UTF-8"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
  <Relationship Id="rId1"
    Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet"
    Target="worksheets/sheet1.xml"/>
  <Relationship Id="rId2"
    Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/sharedStrings"
    Target="sharedStrings.xml"/>
</Relationships>"""
    shared_strings = """<?xml version="1.0" encoding="UTF-8"?>
<sst xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">
  <si><t>record_subject_id</t></si>
  <si><t>native_x</t></si>
  <si><t>JJ_TMS_1001</t></si>
</sst>"""
    worksheet = """<?xml version="1.0" encoding="UTF-8"?>
<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">
  <sheetData>
    <row r="1"><c r="A1" t="s"><v>0</v></c><c r="B1" t="s"><v>1</v></c><c r="C1" t="inlineStr"><is><t>visit2_target</t></is></c></row>
    <row r="2"><c r="A2" t="s"><v>2</v></c><c r="B2"><v>-43.5</v></c><c r="C2" t="inlineStr"><is><t>LDLPFC</t></is></c></row>
  </sheetData>
</worksheet>"""
    if worksheet_override is not None:
        worksheet = worksheet_override
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("[Content_Types].xml", content_types)
        archive.writestr("xl/workbook.xml", workbook)
        archive.writestr("xl/_rels/workbook.xml.rels", relationships)
        archive.writestr("xl/sharedStrings.xml", shared_strings)
        archive.writestr("xl/worksheets/sheet1.xml", worksheet)


def _registry() -> dict[int, dict[str, object]]:
    return {
        11: {
            "roi_id": 11,
            "roi_name": "left-default",
            "hemi": "L",
            "family17": "DefaultA",
            "canonical8": "Default",
        },
        12: {
            "roi_id": 12,
            "roi_name": "left-control",
            "hemi": "L",
            "family17": "ContA",
            "canonical8": "Control",
        },
    }


def test_read_xlsx_rows_handles_shared_and_inline_strings(tmp_path: Path) -> None:
    workbook = tmp_path / "targets.xlsx"
    _write_minimal_xlsx(workbook)

    rows = module.read_xlsx_rows(workbook, "data")

    assert rows == [
        {
            "record_subject_id": "JJ_TMS_1001",
            "native_x": "-43.5",
            "visit2_target": "LDLPFC",
        }
    ]


def test_projection_chooses_nearest_vertex_and_reports_network_qc() -> None:
    surface = np.array([[0, 0, 0], [2, 0, 0], [2, 1, 0]], dtype=float)
    labels = np.array([11, 12, 12], dtype=np.int32)

    result = module.project_target_to_surface(
        target_xyz=np.array([1.8, 0.1, 0]),
        surface_xyz=surface,
        parcel_labels=labels,
        roi_by_id=_registry(),
        neighborhood_radius_mm=1.5,
        expected_hemi="L",
    )

    assert result["vertex_index"] == 1
    assert result["parcel_id"] == 12
    assert result["family17"] == "ContA"
    assert result["canonical8"] == "Control"
    assert result["target_to_surface_distance_mm"] == pytest.approx(0.2236068)
    assert result["neighborhood_vertex_count"] == 2
    assert result["neighborhood_primary_family17_fraction"] == 1.0
    assert result["neighborhood_dominant_family17"] == "ContA"


def test_projection_falls_back_to_nearest_vertex_when_radius_is_empty() -> None:
    result = module.project_target_to_surface(
        target_xyz=np.array([10, 0, 0], dtype=float),
        surface_xyz=np.array([[0, 0, 0], [2, 0, 0]], dtype=float),
        parcel_labels=np.array([11, 12], dtype=np.int32),
        roi_by_id=_registry(),
        neighborhood_radius_mm=0.5,
        expected_hemi="L",
    )

    assert result["vertex_index"] == 1
    assert result["neighborhood_vertex_count"] == 1
    assert result["neighborhood_primary_family17_fraction"] == 1.0


def test_projection_rejects_surface_label_length_mismatch() -> None:
    with pytest.raises(ValueError, match="surface vertices.*parcel labels"):
        module.project_target_to_surface(
            target_xyz=np.array([0, 0, 0], dtype=float),
            surface_xyz=np.array([[0, 0, 0], [1, 0, 0]], dtype=float),
            parcel_labels=np.array([11], dtype=np.int32),
            roi_by_id=_registry(),
            neighborhood_radius_mm=5.0,
            expected_hemi="L",
        )


@pytest.mark.parametrize(
    ("parcel_labels", "message"),
    [
        (np.array([0], dtype=np.int32), "medial-wall parcel 0"),
        (np.array([99], dtype=np.int32), "unknown Kong2022 parcel 99"),
    ],
)
def test_projection_rejects_invalid_nearest_parcel(
    parcel_labels: np.ndarray,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        module.project_target_to_surface(
            target_xyz=np.array([0, 0, 0], dtype=float),
            surface_xyz=np.array([[0, 0, 0]], dtype=float),
            parcel_labels=parcel_labels,
            roi_by_id=_registry(),
            neighborhood_radius_mm=5.0,
            expected_hemi="L",
        )


def test_projection_rejects_parcel_from_wrong_hemisphere() -> None:
    registry = _registry()
    registry[12] = {**registry[12], "hemi": "R"}

    with pytest.raises(ValueError, match="belongs to hemisphere R"):
        module.project_target_to_surface(
            target_xyz=np.array([2, 0, 0], dtype=float),
            surface_xyz=np.array([[0, 0, 0], [2, 0, 0]], dtype=float),
            parcel_labels=np.array([11, 12], dtype=np.int32),
            roi_by_id=registry,
            neighborhood_radius_mm=1.0,
            expected_hemi="L",
        )


def test_read_xlsx_rows_rejects_missing_required_schema_columns(tmp_path: Path) -> None:
    workbook = tmp_path / "targets.xlsx"
    _write_minimal_xlsx(workbook)

    with pytest.raises(KeyError, match="missing required columns.*native_z"):
        module.read_xlsx_rows(
            workbook,
            "data",
            required_headers={
                "record_subject_id",
                "visit2_target",
                "native_x",
                "native_y",
                "native_z",
            },
        )


def test_read_xlsx_rows_rejects_empty_sheet_when_schema_is_required(
    tmp_path: Path,
) -> None:
    workbook = tmp_path / "targets.xlsx"
    empty_worksheet = """<?xml version="1.0" encoding="UTF-8"?>
<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">
  <sheetData/>
</worksheet>"""
    _write_minimal_xlsx(workbook, worksheet_override=empty_worksheet)

    with pytest.raises(KeyError, match="missing required columns.*native_x"):
        module.read_xlsx_rows(
            workbook,
            "data",
            required_headers={"record_subject_id", "native_x"},
        )


def test_projection_rejects_unknown_nonzero_neighborhood_parcel() -> None:
    with pytest.raises(ValueError, match="Neighborhood.*unknown Kong2022 parcel 99"):
        module.project_target_to_surface(
            target_xyz=np.array([0, 0, 0], dtype=float),
            surface_xyz=np.array([[0, 0, 0], [1, 0, 0]], dtype=float),
            parcel_labels=np.array([12, 99], dtype=np.int32),
            roi_by_id=_registry(),
            neighborhood_radius_mm=2.0,
            expected_hemi="L",
        )


def test_projection_rejects_opposite_hemisphere_neighborhood_parcel() -> None:
    registry = _registry()
    registry[13] = {
        "roi_id": 13,
        "roi_name": "right-control",
        "hemi": "R",
        "family17": "ContA",
        "canonical8": "Control",
    }

    with pytest.raises(ValueError, match="Neighborhood parcel 13 belongs to hemisphere R"):
        module.project_target_to_surface(
            target_xyz=np.array([0, 0, 0], dtype=float),
            surface_xyz=np.array([[0, 0, 0], [1, 0, 0]], dtype=float),
            parcel_labels=np.array([12, 13], dtype=np.int32),
            roi_by_id=registry,
            neighborhood_radius_mm=2.0,
            expected_hemi="L",
        )


def test_load_surface_rejects_nonidentity_coordinate_transform(tmp_path: Path) -> None:
    import nibabel as nib

    transform = np.eye(4, dtype=np.float32)
    transform[0, 3] = 10.0
    pointset = nib.gifti.GiftiDataArray(
        data=np.array([[-2, 0, 0], [-2, 1, 0]], dtype=np.float32),
        intent="NIFTI_INTENT_POINTSET",
        coordsys=nib.gifti.GiftiCoordSystem(
            dataspace=3,
            xformspace=3,
            xform=transform,
        ),
        meta={
            "AnatomicalStructurePrimary": "CortexLeft",
            "AnatomicalStructureSecondary": "MidThickness",
            "GeometricType": "Anatomical",
        },
    )
    path = tmp_path / "sub-1001_hemi-L_space-fsLR_den-32k_midthickness.surf.gii"
    nib.save(nib.gifti.GiftiImage(darrays=[pointset]), path)

    with pytest.raises(ValueError, match="non-identity coordinate transform"):
        module._load_surface_xyz(path, expected_hemi="L")


def test_load_surface_rejects_non_32k_vertex_count(tmp_path: Path) -> None:
    import nibabel as nib

    pointset = nib.gifti.GiftiDataArray(
        data=np.array([[-2, 0, 0], [-2, 1, 0]], dtype=np.float32),
        intent="NIFTI_INTENT_POINTSET",
        coordsys=nib.gifti.GiftiCoordSystem(
            dataspace=3,
            xformspace=3,
            xform=np.eye(4, dtype=np.float32),
        ),
        meta={
            "AnatomicalStructurePrimary": "CortexLeft",
            "AnatomicalStructureSecondary": "MidThickness",
            "GeometricType": "Anatomical",
        },
    )
    path = tmp_path / "sub-1001_hemi-L_space-fsLR_den-32k_midthickness.surf.gii"
    nib.save(nib.gifti.GiftiImage(darrays=[pointset]), path)

    with pytest.raises(ValueError, match="expected 32492 vertices, found 2"):
        module._load_surface_xyz(path, expected_hemi="L")


def test_build_dense_group_labels_restores_full_surface_vertex_indexing() -> None:
    labels = module.build_dense_group_parcel_labels(
        {
            "L": (np.array([0, 2]), np.array([11, 12])),
            "R": (np.array([1, 3]), np.array([211, 212])),
        },
        vertex_count=4,
    )

    assert labels["L"].tolist() == [11, 0, 12, 0]
    assert labels["R"].tolist() == [0, 211, 0, 212]


def test_build_dense_group_labels_preserves_explicit_medial_wall_zero() -> None:
    labels = module.build_dense_group_parcel_labels(
        {
            "L": (np.array([0, 1]), np.array([11, 0])),
            "R": (np.array([0, 1]), np.array([0, 211])),
        },
        vertex_count=2,
    )

    assert labels["L"].tolist() == [11, 0]
    assert labels["R"].tolist() == [0, 211]


def test_compare_group_atlas_reports_parcel_and_network_agreement() -> None:
    registry = _registry()
    registry[13] = {
        "roi_id": 13,
        "roi_name": "left-control-b",
        "hemi": "L",
        "family17": "ContB",
        "canonical8": "Control",
    }
    rows = [
        {
            "subject": "sub-1001",
            "mapping_status": "mapped",
            "hemi": "L",
            "vertex_index": 0,
            "parcel_id": 11,
            "family17": "DefaultA",
            "canonical8": "Default",
        },
        {
            "subject": "sub-1002",
            "mapping_status": "mapped",
            "hemi": "L",
            "vertex_index": 1,
            "parcel_id": 12,
            "family17": "ContA",
            "canonical8": "Control",
        },
        {
            "subject": "sub-1003",
            "mapping_status": "mapped",
            "hemi": "L",
            "vertex_index": 2,
            "parcel_id": 12,
            "family17": "ContA",
            "canonical8": "Control",
        },
        {
            "subject": "sub-3003",
            "mapping_status": "missing_workbook_record",
        },
    ]

    compared_rows, comparison = module.compare_with_group_atlas(
        rows,
        group_parcel_labels={"L": np.array([11, 13, 11])},
        roi_registry=list(registry.values()),
    )

    assert compared_rows[0]["group_parcel_id"] == 11
    assert compared_rows[0]["individual_vs_group_same_family17"] is True
    assert compared_rows[1]["group_family17"] == "ContB"
    assert compared_rows[1]["individual_vs_group_same_parcel"] is False
    assert compared_rows[1]["individual_vs_group_same_family17"] is False
    assert compared_rows[1]["individual_vs_group_same_canonical8"] is True
    assert compared_rows[2]["group_canonical8"] == "Default"
    assert compared_rows[2]["individual_vs_group_same_canonical8"] is False
    assert compared_rows[3] == rows[3]
    assert comparison == {
        "n_compared_subjects": 3,
        "parcel": {
            "n_same": 1,
            "n_different": 2,
            "agreement_fraction": pytest.approx(1 / 3),
            "different_subjects": ["sub-1002", "sub-1003"],
        },
        "family17": {
            "n_same": 1,
            "n_different": 2,
            "agreement_fraction": pytest.approx(1 / 3),
            "different_subjects": ["sub-1002", "sub-1003"],
        },
        "canonical8": {
            "n_same": 2,
            "n_different": 1,
            "agreement_fraction": pytest.approx(2 / 3),
            "different_subjects": ["sub-1003"],
        },
    }
