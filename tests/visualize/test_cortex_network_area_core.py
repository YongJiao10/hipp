from __future__ import annotations

from pathlib import Path

import nibabel as nib
import numpy as np
import pytest

from scripts.visualize.cortex_network_area.core import (
    AlignedLabels,
    InputLayout,
    SurfaceMesh,
    SubjectRecord,
    audit_subject_records,
    calculate_area_metrics,
    calculate_subject_record,
    compute_vertex_areas,
    load_cifti_labels,
    load_formal_rosters,
    load_gifti_labels,
    load_taxonomies,
    resolve_subject_record,
)


REPO_ROOT = Path(__file__).resolve().parents[2]


def _square_mesh(hemisphere: str = "L") -> SurfaceMesh:
    return SurfaceMesh(
        coordinates=np.asarray(
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [0.0, 1.0, 0.0]],
            dtype=np.float64,
        ),
        triangles=np.asarray([[0, 1, 2], [0, 2, 3]], dtype=np.int32),
        hemisphere=hemisphere,
        source=Path(f"{hemisphere}.surf.gii"),
    )


def _write_surface(path: Path, mesh: SurfaceMesh) -> None:
    image = nib.gifti.GiftiImage(
        darrays=[
            nib.gifti.GiftiDataArray(
                data=mesh.coordinates.astype(np.float32),
                intent="NIFTI_INTENT_POINTSET",
                meta={
                    "AnatomicalStructurePrimary": (
                        "CortexLeft" if mesh.hemisphere == "L" else "CortexRight"
                    ),
                    "AnatomicalStructureSecondary": "MidThickness",
                },
            ),
            nib.gifti.GiftiDataArray(
                data=mesh.triangles.astype(np.int32),
                intent="NIFTI_INTENT_TRIANGLE",
            ),
        ]
    )
    nib.save(image, path)


def _write_dlabel(
    path: Path,
    *,
    left_mask: np.ndarray,
    right_mask: np.ndarray,
    data: np.ndarray,
    label_names: dict[int, str],
) -> None:
    left_axis = nib.cifti2.BrainModelAxis.from_mask(left_mask, name="CortexLeft")
    right_axis = nib.cifti2.BrainModelAxis.from_mask(right_mask, name="CortexRight")
    brain_axis = left_axis + right_axis
    table = {
        int(key): (str(value), (0.2, 0.3, 0.4, 1.0))
        for key, value in label_names.items()
    }
    table.setdefault(0, ("Unassigned", (0.0, 0.0, 0.0, 0.0)))
    label_axis = nib.cifti2.LabelAxis(["network"], [table])
    image = nib.Cifti2Image(
        dataobj=np.asarray(data, dtype=np.int32).reshape(1, -1),
        header=nib.Cifti2Header.from_axes((label_axis, brain_axis)),
    )
    nib.save(image, path)


def test_vertex_areas_apportion_each_triangle_and_conserve_mesh_area() -> None:
    mesh = _square_mesh()

    areas = compute_vertex_areas(mesh.coordinates, mesh.triangles)

    assert areas.tolist() == pytest.approx([1 / 3, 1 / 6, 1 / 3, 1 / 6])
    assert float(areas.sum()) == pytest.approx(1.0)


def test_vertex_areas_reject_nonfinite_coordinates() -> None:
    mesh = _square_mesh()
    coordinates = mesh.coordinates.copy()
    coordinates[2, 1] = np.nan

    with pytest.raises(ValueError, match="non-finite"):
        compute_vertex_areas(coordinates, mesh.triangles)


def test_cifti_alignment_uses_brainmodel_vertex_indices_and_declared_surface_size(
    tmp_path: Path,
) -> None:
    dlabel = tmp_path / "network.dlabel.nii"
    _write_dlabel(
        dlabel,
        left_mask=np.asarray([True, False, True, True]),
        right_mask=np.asarray([False, True, True]),
        data=np.asarray([1, 0, 2, 1, 2]),
        label_names={1: "DefaultA", 2: "Noise"},
    )

    aligned = load_cifti_labels(dlabel, {"L": 4, "R": 3})

    assert aligned["L"].vertex_indices.tolist() == [0, 2, 3]
    assert aligned["L"].valid_anatomical_mask.tolist() == [True, False, True, True]
    assert aligned["L"].values.tolist() == [1, 0, 0, 2]
    assert aligned["R"].vertex_indices.tolist() == [1, 2]
    with pytest.raises(ValueError, match="BrainModel.*4.*surface.*3"):
        load_cifti_labels(dlabel, {"L": 3, "R": 3})


def test_gifti_labels_require_explicit_label_and_anatomical_vertex_indices(
    tmp_path: Path,
) -> None:
    label_path = tmp_path / "left.label.gii"
    image = nib.gifti.GiftiImage(
        darrays=[
            nib.gifti.GiftiDataArray(
                data=np.asarray([1, 2], dtype=np.int32),
                intent="NIFTI_INTENT_LABEL",
                meta={"AnatomicalStructurePrimary": "CortexLeft"},
            )
        ]
    )
    label_default = nib.gifti.GiftiLabel(key=1)
    label_default.label = "DefaultA"
    label_noise = nib.gifti.GiftiLabel(key=2)
    label_noise.label = "Noise"
    image.labeltable.labels = [label_default, label_noise]
    nib.save(image, label_path)

    with pytest.raises(ValueError, match="explicit vertex_indices"):
        load_gifti_labels(label_path, hemisphere="L", surface_vertex_count=4)

    aligned = load_gifti_labels(
        label_path,
        hemisphere="L",
        surface_vertex_count=4,
        vertex_indices=np.asarray([0, 3]),
        anatomical_vertex_indices=np.asarray([0, 1, 3]),
    )

    assert aligned.values.tolist() == [1, 0, 0, 2]
    assert aligned.valid_anatomical_mask.tolist() == [True, True, False, True]


def test_canonical7_merge_conserves_network_area_and_keeps_noise_unassigned() -> None:
    mesh = _square_mesh()
    labels = AlignedLabels(
        values=np.asarray([1, 2, 0, 3], dtype=np.int32),
        label_names={1: "DefaultA", 2: "Aud", 3: "Noise"},
        valid_anatomical_mask=np.ones(4, dtype=bool),
        vertex_indices=np.arange(4, dtype=np.int32),
        hemisphere="L",
        source=Path("subject.dlabel.nii"),
        alignment_source="CIFTI BrainModel",
    )
    taxonomies = load_taxonomies(REPO_ROOT)

    native = calculate_area_metrics(
        meshes={"L": mesh}, labels={"L": labels}, taxonomy=taxonomies["kong_native17"]
    )
    canonical = calculate_area_metrics(
        meshes={"L": mesh}, labels={"L": labels}, taxonomy=taxonomies["kong_canonical7"]
    )

    assert native.total_cortical_area_mm2 == pytest.approx(1.0)
    assert native.network_area_mm2["DefaultA"] == pytest.approx(1 / 3)
    assert native.network_area_mm2["Aud"] == pytest.approx(1 / 6)
    assert canonical.network_area_mm2["Default"] == pytest.approx(1 / 3)
    assert canonical.network_area_mm2["Somatomotor"] == pytest.approx(1 / 6)
    assert sum(native.network_area_mm2.values()) == pytest.approx(
        sum(canonical.network_area_mm2.values())
    )
    assert canonical.noise_area_mm2 == pytest.approx(1 / 6)
    assert canonical.unassigned_area_mm2 == pytest.approx(1 / 3)
    assert canonical.area_percent["Default"] == pytest.approx(100 / 3)
    canonical.assert_conserved()


def test_formal_rosters_have_fixed_counts_and_exclude_3003() -> None:
    rosters = load_formal_rosters(REPO_ROOT)

    assert {key: len(value.subjects) for key, value in rosters.items()} == {
        "HCP HC": 151,
        "Iowa HC": 67,
        "Patient": 26,
        "Pure MDD": 7,
        "Comorbid MDD": 19,
    }
    assert "sub-3003" not in rosters["Patient"].subjects


def test_resolver_reuses_one_hcp_input_and_selects_exact_iowa_variant_branch(
    tmp_path: Path,
) -> None:
    layout = InputLayout(
        hcp_surface_template=str(tmp_path / "hcp/{sub_subject}/{hemi}.surf.gii"),
        hcp_functional_manifest_template=str(tmp_path / "hcp/{sub_subject}/ica_fix.json"),
        hcp_label_templates={
            "kong2022": str(tmp_path / "hcp/{sub_subject}/kong.dlabel.nii"),
            "lynch2024": str(tmp_path / "hcp/{sub_subject}/lynch.dlabel.nii"),
        },
        iowa_surface_templates={
            "36p": str(tmp_path / "iowa36/{sub_subject}/{hemi}.surf.gii"),
        },
        iowa_functional_manifest_templates={
            "36p": str(tmp_path / "iowa36/{sub_subject}/provenance.json"),
        },
        iowa_label_templates={
            "36p": {"kong2022": str(tmp_path / "iowa36/{sub_subject}/kong.dlabel.nii")},
        },
    )

    hcp36 = resolve_subject_record("100610", "HCP HC", "36p", "kong2022", layout)
    iowa36 = resolve_subject_record("1001", "Iowa HC", "36p", "kong2022", layout)

    assert "iowa36" in str(iowa36.label_path)


def test_audit_reports_missing_and_incompatible_inputs_explicitly(tmp_path: Path) -> None:
    missing = SubjectRecord(
        subject="sub-1001",
        cohort="Iowa HC",
        dataset="iowa",
        variant="36p",
        atlas="kong2022",
        surface_paths={"L": tmp_path / "missing.L.surf.gii", "R": tmp_path / "missing.R.surf.gii"},
        label_path=tmp_path / "missing.dlabel.nii",
        functional_manifest=tmp_path / "missing.provenance.json",
    )

    missing_row = audit_subject_records([missing])[0]

    assert missing_row["status"] == "missing"
    assert set(missing_row["missing_inputs"]) == {
        "surface_L",
        "surface_R",
        "label",
        "functional_manifest",
    }

    left = tmp_path / "left.surf.gii"
    right = tmp_path / "right.surf.gii"
    _write_surface(left, _square_mesh("L"))
    _write_surface(right, _square_mesh("R"))
    dlabel = tmp_path / "bad.dlabel.nii"
    _write_dlabel(
        dlabel,
        left_mask=np.asarray([True, True, True]),
        right_mask=np.asarray([True, True, True, True]),
        data=np.asarray([1, 1, 1, 1, 1, 1, 1]),
        label_names={1: "DefaultA"},
    )
    provenance = tmp_path / "provenance.json"
    provenance.write_text("{}\n", encoding="utf-8")
    incompatible = SubjectRecord(
        subject="sub-1001",
        cohort="Iowa HC",
        dataset="iowa",
        variant="36p",
        atlas="kong2022",
        surface_paths={"L": left, "R": right},
        label_path=dlabel,
        functional_manifest=provenance,
    )

    incompatible_row = audit_subject_records([incompatible])[0]

    assert incompatible_row["status"] == "incompatible"
    assert "BrainModel" in incompatible_row["error"]


def test_subject_record_loader_calculates_native_and_canonical_metrics(tmp_path: Path) -> None:
    left = tmp_path / "left.surf.gii"
    right = tmp_path / "right.surf.gii"
    _write_surface(left, _square_mesh("L"))
    _write_surface(right, _square_mesh("R"))
    dlabel = tmp_path / "network.dlabel.nii"
    _write_dlabel(
        dlabel,
        left_mask=np.ones(4, dtype=bool),
        right_mask=np.ones(4, dtype=bool),
        data=np.asarray([1, 2, 0, 3, 1, 2, 0, 3]),
        label_names={1: "DefaultA", 2: "Aud", 3: "Noise"},
    )
    provenance = tmp_path / "provenance.json"
    provenance.write_text("{}\n", encoding="utf-8")
    record = SubjectRecord(
        subject="sub-1001",
        cohort="Iowa HC",
        dataset="iowa",
        variant="36p",
        atlas="kong2022",
        surface_paths={"L": left, "R": right},
        label_path=dlabel,
        functional_manifest=provenance,
    )

    metrics = calculate_subject_record(record, taxonomies=load_taxonomies(REPO_ROOT))

    assert set(metrics) == {"kong_native17", "kong_canonical7"}
    assert metrics["kong_native17"].total_cortical_area_mm2 == pytest.approx(2.0)
    assert metrics["kong_canonical7"].network_area_mm2["Somatomotor"] == pytest.approx(1 / 3)
