#!/usr/bin/env python3
from __future__ import annotations

import csv
import importlib.util
import json
import sys
import zipfile
from pathlib import Path

import nibabel as nib
import numpy as np
import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
MODULE_PATH = REPO_ROOT / "scripts/jj_tms/map_jj_tms_sites_to_kong2022.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("jj_tms_target_network_mapping_cohort", MODULE_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load mapping module: {MODULE_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


module = _load_module()


def _write_cohort_xlsx(path: Path, rows: list[dict[str, object]]) -> None:
    headers = [
        "record_subject_id",
        "visit2_target",
        "native_x",
        "native_y",
        "native_z",
        "mni_x",
        "mni_y",
        "mni_z",
        "visit2_target_notes",
    ]
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
</Relationships>"""
    xml_rows = []
    all_rows = [dict(zip(headers, headers, strict=True)), *rows]
    for row_number, row in enumerate(all_rows, start=1):
        cells = []
        for column_number, header in enumerate(headers, start=1):
            reference = f"{chr(64 + column_number)}{row_number}"
            value = row.get(header)
            if value is None:
                cells.append(f'<c r="{reference}"/>')
            elif row_number == 1 or header in {
                "record_subject_id",
                "visit2_target",
                "visit2_target_notes",
            }:
                cells.append(f'<c r="{reference}" t="inlineStr"><is><t>{value}</t></is></c>')
            else:
                cells.append(f'<c r="{reference}"><v>{value}</v></c>')
        xml_rows.append(f'<row r="{row_number}">{"".join(cells)}</row>')
    worksheet = (
        '<?xml version="1.0" encoding="UTF-8"?>'
        f'<worksheet xmlns="{module.SHEET_NS}"><sheetData>'
        f'{"".join(xml_rows)}</sheetData></worksheet>'
    )
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("xl/workbook.xml", workbook)
        archive.writestr("xl/_rels/workbook.xml.rels", relationships)
        archive.writestr("xl/worksheets/sheet1.xml", worksheet)


def _write_surface_and_labels(
    surface_root: Path,
    label_root: Path,
    subject: str,
    hemi: str,
    vertices: np.ndarray,
    labels: np.ndarray,
) -> None:
    anat = surface_root / f"sub-{subject}" / "anat"
    anat.mkdir(parents=True)
    surface_path = (
        anat / f"sub-{subject}_hemi-{hemi}_space-fsLR_den-32k_midthickness.surf.gii"
    )
    image = nib.gifti.GiftiImage(
        darrays=[
            nib.gifti.GiftiDataArray(
                data=np.asarray(vertices, dtype=np.float32),
                intent="NIFTI_INTENT_POINTSET",
                coordsys=nib.gifti.GiftiCoordSystem(
                    dataspace=3,
                    xformspace=3,
                    xform=np.eye(4, dtype=np.float32),
                ),
                meta={
                    "AnatomicalStructurePrimary": (
                        "CortexLeft" if hemi == "L" else "CortexRight"
                    ),
                    "AnatomicalStructureSecondary": "MidThickness",
                    "GeometricType": "Anatomical",
                },
            )
        ]
    )
    nib.save(image, surface_path)
    labels_dir = label_root / f"sub-{subject}"
    labels_dir.mkdir(parents=True)
    label_name = "lh_labels.txt" if hemi == "L" else "rh_labels.txt"
    np.savetxt(labels_dir / label_name, labels, fmt="%d")


def _registry() -> list[dict[str, object]]:
    return [
        {
            "roi_id": 11,
            "roi_name": "17networks_LH_DefaultA_1",
            "hemi": "L",
            "family17": "DefaultA",
            "canonical8": "Default",
        },
        {
            "roi_id": 211,
            "roi_name": "17networks_RH_ContA_1",
            "hemi": "R",
            "family17": "ContA",
            "canonical8": "Control",
        },
    ]


def test_map_cohort_uses_manifest_order_and_real_surface_files(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(module, "FSLR_32K_VERTICES_PER_HEMISPHERE", 2)
    workbook = tmp_path / "targets.xlsx"
    _write_cohort_xlsx(
        workbook,
        [
            {
                "record_subject_id": "JJ_TMS_1001",
                "visit2_target": "LDLPFC",
                "native_x": -1.8,
                "native_y": 0.1,
                "native_z": 0,
                "mni_x": -43,
                "mni_y": 25,
                "mni_z": 43,
                "visit2_target_notes": "left target",
            },
            {
                "record_subject_id": "JJ_TMS_1002",
                "visit2_target": "RDLPFC",
                "native_x": 1.9,
                "native_y": 0,
                "native_z": 0,
                "mni_x": 44,
                "mni_y": 25,
                "mni_z": 43,
                "visit2_target_notes": None,
            },
        ],
    )
    surface_root = tmp_path / "surfaces"
    label_root = tmp_path / "labels"
    _write_surface_and_labels(
        surface_root,
        label_root,
        "1001",
        "L",
        np.array([[-2, 0, 0], [-2, 1, 0]]),
        np.array([11, 11]),
    )
    _write_surface_and_labels(
        surface_root,
        label_root,
        "1002",
        "R",
        np.array([[2, 0, 0], [2, 1, 0]]),
        np.array([211, 211]),
    )

    rows, summary = module.map_cohort(
        workbook_path=workbook,
        subjects=["sub-1002", "1001"],
        surface_root=surface_root,
        label_root=label_root,
        roi_registry=_registry(),
        neighborhood_radius_mm=1.5,
    )

    assert [row["subject"] for row in rows] == ["sub-1002", "sub-1001"]
    assert rows[0]["hemi"] == "R"
    assert rows[0]["parcel_id"] == 211
    assert rows[0]["family17"] == "ContA"
    assert rows[1]["target_notes"] == "left target"
    assert rows[1]["native_x"] == -1.8
    assert rows[1]["target_to_surface_distance_mm"] == pytest.approx(0.2236068)
    assert summary["n_requested_subjects"] == 2
    assert summary["n_mapped_subjects"] == 2
    assert summary["family17_counts"] == {"ContA": 1, "DefaultA": 1}
    assert summary["coordinate_source"] == "native_x/native_y/native_z"


def test_map_cohort_marks_manifest_subject_missing_from_workbook(tmp_path: Path) -> None:
    workbook = tmp_path / "targets.xlsx"
    _write_cohort_xlsx(workbook, [])

    rows, summary = module.map_cohort(
        workbook_path=workbook,
        subjects=["1001"],
        surface_root=tmp_path / "surfaces",
        label_root=tmp_path / "labels",
        roi_registry=_registry(),
        neighborhood_radius_mm=5.0,
    )

    assert rows == [
        {
            "subject": "sub-1001",
            "mapping_status": "missing_workbook_record",
            "mapping_error": "No record_subject_id match in workbook",
        }
    ]
    assert summary["n_requested_subjects"] == 1
    assert summary["n_mapped_subjects"] == 0
    assert summary["n_unmapped_subjects"] == 1
    assert summary["unmapped_subjects"] == {
        "missing_workbook_record": ["sub-1001"],
    }


def test_map_cohort_rejects_target_hemisphere_and_native_x_disagreement(
    tmp_path: Path,
) -> None:
    workbook = tmp_path / "targets.xlsx"
    _write_cohort_xlsx(
        workbook,
        [
            {
                "record_subject_id": "JJ_TMS_1001",
                "visit2_target": "LDLPFC",
                "native_x": 10,
                "native_y": 20,
                "native_z": 30,
                "mni_x": -43,
                "mni_y": 25,
                "mni_z": 43,
                "visit2_target_notes": None,
            }
        ],
    )

    with pytest.raises(ValueError, match="LDLPFC.*native_x=10.0"):
        module.map_cohort(
            workbook_path=workbook,
            subjects=["1001"],
            surface_root=tmp_path / "surfaces",
            label_root=tmp_path / "labels",
            roi_registry=_registry(),
            neighborhood_radius_mm=5.0,
        )


def test_write_outputs_creates_csv_and_json_without_overwriting(tmp_path: Path) -> None:
    output_dir = tmp_path / "results"
    rows = [
        {
            "subject": "sub-1001",
            "hemi": "L",
            "parcel_id": 11,
            "family17": "DefaultA",
        }
    ]
    summary = {"n_mapped_subjects": 1}

    csv_path, json_path = module.write_outputs(rows, summary, output_dir)

    with csv_path.open(newline="", encoding="utf-8") as stream:
        assert list(csv.DictReader(stream)) == [
            {
                "subject": "sub-1001",
                "hemi": "L",
                "parcel_id": "11",
                "family17": "DefaultA",
            }
        ]
    assert json.loads(json_path.read_text(encoding="utf-8")) == summary
    with pytest.raises(FileExistsError, match="already exists"):
        module.write_outputs(rows, summary, output_dir)


def test_map_cohort_marks_subject_with_missing_native_coordinates(tmp_path: Path) -> None:
    workbook = tmp_path / "targets.xlsx"
    _write_cohort_xlsx(
        workbook,
        [
            {
                "record_subject_id": "JJ_TMS_3026",
                "visit2_target": "RDLPFC",
                "native_x": None,
                "native_y": None,
                "native_z": None,
                "mni_x": None,
                "mni_y": None,
                "mni_z": None,
                "visit2_target_notes": None,
            }
        ],
    )

    rows, summary = module.map_cohort(
        workbook_path=workbook,
        subjects=["3026"],
        surface_root=tmp_path / "surfaces",
        label_root=tmp_path / "labels",
        roi_registry=_registry(),
        neighborhood_radius_mm=5.0,
    )

    assert rows == [
        {
            "subject": "sub-3026",
            "mapping_status": "missing_native_coordinates",
            "mapping_error": "Missing native coordinate fields: native_x,native_y,native_z",
            "visit2_target": "RDLPFC",
            "mni_x": None,
            "mni_y": None,
            "mni_z": None,
            "target_notes": None,
        }
    ]
    assert summary["n_mapped_subjects"] == 0
    assert summary["unmapped_subjects"] == {
        "missing_native_coordinates": ["sub-3026"],
    }


def test_completion_lines_distinguish_output_rows_from_mapped_subjects() -> None:
    lines = module.completion_lines(
        rows=[{"subject": "sub-1001"}, {"subject": "sub-1002"}],
        summary={"n_mapped_subjects": 1},
        csv_path=Path("/tmp/result.csv"),
        json_path=Path("/tmp/result.json"),
    )

    assert lines == [
        "Output rows: 2",
        "Mapped subjects: 1",
        "CSV: /tmp/result.csv",
        "Summary: /tmp/result.json",
    ]


def test_write_outputs_supports_unmapped_first_row_with_mapped_fields(
    tmp_path: Path,
) -> None:
    rows = [
        {
            "subject": "sub-3003",
            "mapping_status": "missing_workbook_record",
            "mapping_error": "No record_subject_id match in workbook",
        },
        {
            "subject": "sub-1001",
            "mapping_status": "mapped",
            "mapping_error": None,
            "parcel_id": 11,
            "family17": "DefaultA",
        },
    ]

    csv_path, _ = module.write_outputs(
        rows,
        {"n_mapped_subjects": 1},
        tmp_path / "results",
    )

    with csv_path.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        output_rows = list(reader)
    assert reader.fieldnames == [
        "subject",
        "mapping_status",
        "mapping_error",
        "parcel_id",
        "family17",
    ]
    assert output_rows[0]["parcel_id"] == ""
    assert output_rows[1]["parcel_id"] == "11"

def test_compare_existing_mapping_results_reuses_saved_vertex_indices(
    tmp_path: Path,
) -> None:
    individual_csv = tmp_path / "individual.csv"
    with individual_csv.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=[
                "subject",
                "mapping_status",
                "hemi",
                "vertex_index",
                "parcel_id",
                "family17",
                "canonical8",
            ],
        )
        writer.writeheader()
        writer.writerows(
            [
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
                    "subject": "sub-3003",
                    "mapping_status": "missing_workbook_record",
                    "hemi": "",
                    "vertex_index": "",
                    "parcel_id": "",
                    "family17": "",
                    "canonical8": "",
                },
            ]
        )

    rows, summary = module.compare_existing_mapping_results(
        individual_results_csv=individual_csv,
        group_vertex_arrays={
            "L": (np.array([0]), np.array([11])),
            "R": (np.array([1]), np.array([211])),
        },
        roi_registry=_registry(),
        group_atlas_path=tmp_path / "group_atlas.dlabel.nii",
        vertex_count=2,
    )

    assert len(rows) == 2
    assert rows[0]["group_parcel_id"] == 11
    assert rows[0]["individual_vs_group_same_family17"] is True
    assert rows[1]["mapping_status"] == "missing_workbook_record"
    assert "group_parcel_id" not in rows[1]
    assert summary["comparison_method"] == (
        "reuse saved individualized fsLR-32k vertex_index; no target reprojection"
    )
    assert summary["n_output_rows"] == 2
    assert summary["n_compared_subjects"] == 1
    assert summary["n_unmapped_subjects"] == 1
    assert summary["group_family17_counts"] == {"DefaultA": 1}
    assert summary["group_canonical8_counts"] == {"Default": 1}
    assert summary["family17_transition_counts"] == {
        "DefaultA -> DefaultA": 1,
    }
    assert summary["canonical8_transition_counts"] == {
        "Default -> Default": 1,
    }
    assert summary["agreement"]["family17"]["n_same"] == 1
    assert summary["individual_results_csv"] == str(individual_csv.resolve())


def test_write_outputs_accepts_separate_group_comparison_names(
    tmp_path: Path,
) -> None:
    csv_path, json_path = module.write_outputs(
        [{"subject": "sub-1001", "mapping_status": "mapped"}],
        {"n_compared_subjects": 1},
        tmp_path,
        csv_name="individual_vs_group.csv",
        summary_name="individual_vs_group_summary.json",
    )

    assert csv_path.name == "individual_vs_group.csv"
    assert json_path.name == "individual_vs_group_summary.json"
    assert csv_path.is_file()
    assert json_path.is_file()

def test_main_comparison_mode_does_not_reproject_targets(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    individual_csv = tmp_path / "individual.csv"
    with individual_csv.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=[
                "subject",
                "mapping_status",
                "hemi",
                "vertex_index",
                "parcel_id",
                "family17",
                "canonical8",
            ],
        )
        writer.writeheader()
        writer.writerow(
            {
                "subject": "sub-1001",
                "mapping_status": "mapped",
                "hemi": "L",
                "vertex_index": 0,
                "parcel_id": 11,
                "family17": "DefaultA",
                "canonical8": "Default",
            }
        )

    monkeypatch.setattr(
        module,
        "load_kong2022_bundle",
        lambda: {
            "vertex_arrays": {
                "L": (np.array([0]), np.array([11])),
                "R": (np.array([1]), np.array([211])),
            },
            "roi_registry": _registry(),
        },
    )

    def fail_if_reprojected(**kwargs):
        raise AssertionError("comparison mode must not call map_cohort")

    monkeypatch.setattr(module, "map_cohort", fail_if_reprojected)

    exit_code = module.main(
        [
            "--compare-existing-individual-results",
            "--individual-results-csv",
            str(individual_csv),
            "--output-dir",
            str(tmp_path / "results"),
        ]
    )

    assert exit_code == 0
    assert (tmp_path / "results" / module.COMPARISON_CSV_NAME).is_file()
    summary_path = tmp_path / "results" / module.COMPARISON_SUMMARY_NAME
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert summary["n_compared_subjects"] == 1
    assert summary["comparison_method"].endswith("no target reprojection")
