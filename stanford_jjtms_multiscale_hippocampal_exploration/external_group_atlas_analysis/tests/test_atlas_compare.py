"""Contract tests for the unified Stanford group-atlas geometry family."""

from __future__ import annotations

import json
import inspect
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))

from atlas_compare import (  # noqa: E402
    FACTORIAL_GROUP_ORDER,
    GROUP_ORDER,
    HC_COMPARISON_ORDER,
    NESTED_GROUP_ORDER,
    PAIR_ORDER,
    LoadedAtlas,
    _area_rows,
    _atlas_directory,
    _boundary_position_by_map,
    build_descriptive_contrasts,
    build_matched_group_pairs,
    build_spec_stability,
    compare_label_maps,
    compute_atlas_region_area,
    discover_atlases,
    load_validated_atlas,
    mesh_geodesic_matrix,
    validate_manifest,
)
import atlas_compare as atlas_compare_module  # noqa: E402


EXPECTED_COHORTS = (
    "HC",
    "NTHC",
    "TEHC",
    "NTS",
    "NTS Any MDD",
    "NTS Pure MDD",
    "TIS",
    "TIS PTSD",
)


def _manifest(surface: str) -> dict[str, object]:
    return {
        "status": "ok",
        "chosen_k_final": 3,
        "cluster_label_ordering": "long_axis_head_to_tail",
        "cluster_order_head_to_tail": [1, 2, 3],
        "template_left_or_right_surface": surface,
        "group_labels_final": str(Path(surface).parent / "group_labels_final.npy"),
        "density_evidence": {"surface_path": surface, "n_vertices": 4, "n_faces": 2},
    }


def test_atlas_registry_covers_all_configured_stanford_cohorts() -> None:
    assert GROUP_ORDER == EXPECTED_COHORTS
    assert len(HC_COMPARISON_ORDER) == 7
    assert HC_COMPARISON_ORDER == tuple((group, "HC") for group in EXPECTED_COHORTS if group != "HC")
    assert PAIR_ORDER == HC_COMPARISON_ORDER
    assert _atlas_directory(Path("/tmp/group_atlas"), "NTS Any MDD", "plain", 2, "L").name == "hemi_L"
    assert "routea_nts_any_mdd_full_k3_fisherz" in str(
        _atlas_directory(Path("/tmp/group_atlas"), "NTS Any MDD", "plain", 2, "L")
    )
    assert "routea_nts_pure_mdd_weighted_fisherz_full_k3" in str(
        _atlas_directory(Path("/tmp/group_atlas"), "NTS Pure MDD", "weighted", 4, "R")
    )
    assert "routea_tis_ptsd_full_k3_fisherz" in str(
        _atlas_directory(Path("/tmp/group_atlas"), "TIS PTSD", "plain", 2, "L")
    )


def test_validate_manifest_requires_k3_order_and_consumed_surface(tmp_path: Path) -> None:
    surface = tmp_path / "surface.surf.gii"
    surface.write_bytes(b"placeholder")
    good = _manifest(str(surface))
    checked = validate_manifest(good, manifest_path=tmp_path / "group_atlas_manifest.json")
    assert checked["consumed_geometry_path"] == str(surface.resolve())
    assert checked["n_vertices"] == 4

    bad = dict(good, chosen_k_final=4)
    with pytest.raises(ValueError, match="chosen_k_final"):
        validate_manifest(bad, manifest_path=tmp_path / "group_atlas_manifest.json")


def test_compute_atlas_region_area_uses_triangle_area_divided_across_vertices() -> None:
    vertices = np.asarray(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
    )
    faces = np.asarray([[0, 1, 2], [0, 1, 3]])
    labels = np.asarray([1, 1, 2, 3])
    result = compute_atlas_region_area(vertices, faces, labels)

    # Two right triangles contribute 0.5 mm² each, one third to every vertex.
    assert result["atlas_total_surface_area_mm2"] == pytest.approx(1.0)
    assert result["atlas_region_area_mm2"][1] == pytest.approx(2.0 / 3.0)
    assert result["atlas_region_area_mm2"][2] == pytest.approx(1.0 / 6.0)
    assert result["atlas_region_area_fraction"][3] == pytest.approx(1.0 / 6.0)


def test_compare_label_maps_reports_ordered_dice_and_mesh_boundary_distance() -> None:
    vertices = np.asarray(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0], [3.0, 0.0, 0.0]]
    )
    faces = np.asarray([[0, 1, 2], [1, 2, 3]])
    first = np.asarray([1, 1, 2, 3])
    second = np.asarray([1, 2, 2, 3])
    result = compare_label_maps(first, second, vertices, faces)

    assert result["raw_label_agreement"] == pytest.approx(0.75)
    assert result["ordered_cluster_dice"][1] == pytest.approx(2.0 / 3.0)
    assert result["boundary_dice"]["1-2"] == pytest.approx(1.0)
    assert result["boundary_distance_mm"]["2-3"]["mean"] == pytest.approx(0.2)
    assert result["boundary_distance_mm"]["2-3"]["hausdorff"] == pytest.approx(1.0)


def test_load_validated_atlas_reads_labels_and_consumed_geometry(tmp_path: Path) -> None:
    nib = pytest.importorskip("nibabel")
    vertices = np.asarray(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0], [3.0, 0.0, 0.0]],
        dtype=np.float32,
    )
    faces = np.asarray([[0, 1, 2], [1, 2, 3]], dtype=np.int32)
    surface = tmp_path / "surface.surf.gii"
    nib.save(nib.GiftiImage(darrays=[nib.gifti.GiftiDataArray(vertices), nib.gifti.GiftiDataArray(faces)]), surface)
    labels = np.asarray([1, 1, 2, 3], dtype=np.int32)
    np.save(tmp_path / "group_labels_final.npy", labels)
    manifest = _manifest(str(surface))
    (tmp_path / "group_atlas_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    loaded = load_validated_atlas(tmp_path)
    assert loaded.labels.tolist() == [1, 1, 2, 3]
    assert loaded.vertices.shape == (4, 3)
    assert loaded.faces.shape == (2, 3)

def test_boundary_positions_use_the_matching_hemisphere_reference_axis() -> None:
    """Removing the hemisphere filter must change the left boundary result."""

    vertices = np.asarray(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0], [3.0, 0.0, 0.0]]
    )
    faces = np.asarray([[0, 1, 2], [1, 2, 3]])
    records = []
    for hemisphere, labels in (
        ("L", np.asarray([1, 1, 2, 3])),
        ("R", np.asarray([1, 2, 3, 3])),
    ):
        for variant in ("plain", "weighted"):
            records.append(
                LoadedAtlas(
                    group="HC",
                    variant=variant,
                    smoothing_mm=2,
                    hemisphere=hemisphere,
                    directory=Path("."),
                    manifest_path=Path("group_atlas_manifest.json"),
                    labels_path=Path("group_labels_final.npy"),
                    labels=labels,
                    vertices=vertices,
                    faces=faces,
                    manifest={
                        "template_left_or_right_surface": (
                            f"/tmp/{variant}_{hemisphere}.surf.gii"
                        )
                    },
                    density_evidence={},
                )
            )

    geodesic = mesh_geodesic_matrix(vertices, faces)
    summaries = {record.key: {"geodesic": geodesic} for record in records}
    positions = _boundary_position_by_map(records, summaries)

    assert positions[("HC", "plain", 2, "L", "1-2")] == pytest.approx(1.0 / 6.0)
    assert positions[("HC", "plain", 2, "R", "1-2")] == pytest.approx(0.25)

def _synthetic_record(
    *,
    group: str,
    variant: str,
    smoothing_mm: int,
    hemisphere: str,
    labels: np.ndarray,
    vertices: np.ndarray,
    faces: np.ndarray,
) -> LoadedAtlas:
    surface = f"/tmp/{variant}_{hemisphere}.surf.gii"
    return LoadedAtlas(
        group=group,
        variant=variant,
        smoothing_mm=smoothing_mm,
        hemisphere=hemisphere,
        directory=Path("."),
        manifest_path=Path("group_atlas_manifest.json"),
        labels_path=Path("group_labels_final.npy"),
        labels=labels,
        vertices=vertices,
        faces=faces,
        manifest={"template_left_or_right_surface": surface},
        density_evidence={
            "n_vertices": int(vertices.shape[0]),
            "n_faces": int(faces.shape[0]),
        },
    )


def test_boundary_positions_use_same_variant_reference_axis() -> None:
    """Using the plain axis for weighted labels must change this literal result."""

    vertices = np.asarray(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0], [3.0, 0.0, 0.0]]
    )
    faces = np.asarray([[0, 1, 2], [1, 2, 3]])
    records = []
    for hemisphere in ("L", "R"):
        records.append(
            _synthetic_record(
                group="HC",
                variant="plain",
                smoothing_mm=2,
                hemisphere=hemisphere,
                labels=np.asarray([1, 1, 2, 3]),
                vertices=vertices,
                faces=faces,
            )
        )
        records.append(
            _synthetic_record(
                group="HC",
                variant="weighted",
                smoothing_mm=2,
                hemisphere=hemisphere,
                labels=np.asarray([1, 2, 3, 3]),
                vertices=vertices,
                faces=faces,
            )
        )
    geodesic = mesh_geodesic_matrix(vertices, faces)
    summaries = {record.key: {"geodesic": geodesic} for record in records}

    positions = _boundary_position_by_map(records, summaries)

    assert positions[("HC", "weighted", 2, "L", "1-2")] == pytest.approx(0.25)


def test_cross_geometry_spec_rows_do_not_report_physical_boundary_distance() -> None:
    """Equal vertex labels on different surfaces are label-only, not zero millimetres."""

    faces = np.asarray([[0, 1, 2], [1, 2, 3]])
    labels = np.asarray([1, 1, 2, 3])
    records = []
    summaries = {}
    for group in EXPECTED_COHORTS:
        for variant, scale in (("plain", 1.0), ("weighted", 2.0)):
            vertices = np.asarray(
                [[0.0, 0.0, 0.0], [scale, 0.0, 0.0], [2 * scale, 0.0, 0.0], [3 * scale, 0.0, 0.0]]
            )
            for smoothing_mm in (2, 4):
                for hemisphere in ("L", "R"):
                    record = _synthetic_record(
                        group=group,
                        variant=variant,
                        smoothing_mm=smoothing_mm,
                        hemisphere=hemisphere,
                        labels=labels,
                        vertices=vertices,
                        faces=faces,
                    )
                    records.append(record)
                    summaries[record.key] = {
                        "geodesic": mesh_geodesic_matrix(vertices, faces)
                    }

    stability = build_spec_stability(records, summaries)
    assignment = stability.loc[stability["spec_axis"] == "assignment_variant"]

    assert len(assignment) == 32
    assert assignment["geometry_match"].eq(False).all()
    assert assignment["ari"].eq(1.0).all()
    assert assignment["boundary_1-2_dice"].eq(1.0).all()
    for column in (
        "boundary_1-2_distance_mean_mm",
        "boundary_1-2_distance_p95_mm",
        "boundary_1-2_distance_hausdorff_mm",
        "boundary_1-2_position_a",
        "boundary_1-2_position_b",
        "boundary_1-2_position_delta_b_minus_a",
        "boundary_2-3_distance_mean_mm",
        "boundary_2-3_distance_p95_mm",
        "boundary_2-3_distance_hausdorff_mm",
        "boundary_2-3_position_a",
        "boundary_2-3_position_b",
        "boundary_2-3_position_delta_b_minus_a",
    ):
        assert assignment[column].isna().all(), column

def test_load_validated_atlas_uses_manifest_label_path_and_checks_local_copy(
    tmp_path: Path,
) -> None:
    nib = pytest.importorskip("nibabel")
    atlas_dir = tmp_path / "weighted_atlas"
    source_dir = tmp_path / "plain_source"
    atlas_dir.mkdir()
    source_dir.mkdir()
    vertices = np.asarray(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0], [3.0, 0.0, 0.0]],
        dtype=np.float32,
    )
    faces = np.asarray([[0, 1, 2], [1, 2, 3]], dtype=np.int32)
    surface = tmp_path / "surface.surf.gii"
    nib.save(
        nib.GiftiImage(
            darrays=[
                nib.gifti.GiftiDataArray(vertices),
                nib.gifti.GiftiDataArray(faces),
            ]
        ),
        surface,
    )
    source_labels = source_dir / "group_labels_final.npy"
    labels = np.asarray([1, 1, 2, 3], dtype=np.int32)
    np.save(source_labels, labels)
    np.save(atlas_dir / "group_labels_final.npy", labels)
    manifest = _manifest(str(surface))
    manifest["group_labels_final"] = str(source_labels)
    (atlas_dir / "group_atlas_manifest.json").write_text(
        json.dumps(manifest), encoding="utf-8"
    )

    loaded = load_validated_atlas(atlas_dir)
    assert loaded.labels_path == source_labels.resolve()
    assert loaded.labels.tolist() == [1, 1, 2, 3]

    np.save(atlas_dir / "group_labels_final.npy", np.asarray([1, 2, 2, 3]))
    with pytest.raises(ValueError, match="local label copy disagrees"):
        load_validated_atlas(atlas_dir)

def test_current_atlas_run_has_stable_id_and_overwrites_existing_outputs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    assert inspect.signature(atlas_compare_module.run_analysis).parameters["run_id"].default == (
        "atlas_descriptive"
    )

    analysis_root = tmp_path / "analysis"
    result_dir = analysis_root / "results" / "atlas_descriptive"
    result_dir.mkdir(parents=True)
    (analysis_root / "figures" / "atlas_descriptive").mkdir(parents=True)
    (analysis_root / "reports" / "atlas_descriptive").mkdir(parents=True)
    (result_dir / "atlas_inventory.csv").write_text("old\n", encoding="utf-8")

    monkeypatch.setattr(atlas_compare_module, "discover_atlases", lambda _: [])
    monkeypatch.setattr(
        atlas_compare_module,
        "build_inventory",
        lambda _: pd.DataFrame({"status": ["current"]}),
    )
    monkeypatch.setattr(
        atlas_compare_module,
        "_area_rows",
        lambda _: (pd.DataFrame(), {}),
    )
    monkeypatch.setattr(
        atlas_compare_module,
        "build_matched_group_pairs",
        lambda *_: pd.DataFrame(),
    )
    monkeypatch.setattr(
        atlas_compare_module,
        "build_spec_stability",
        lambda *_: pd.DataFrame(),
    )
    monkeypatch.setattr(
        atlas_compare_module,
        "build_descriptive_contrasts",
        lambda *_: pd.DataFrame(),
    )
    for name in (
        "plot_reference_area_profiles",
        "plot_pairwise_ari",
        "plot_boundary_spec_comparison",
    ):
        monkeypatch.setattr(atlas_compare_module, name, lambda *_: None)
    monkeypatch.setattr(atlas_compare_module, "_technical_summary", lambda *_: "current\n")

    paths = atlas_compare_module.run_analysis(
        input_root=tmp_path / "inputs",
        analysis_root=analysis_root,
    )

    assert paths["atlas_inventory"].read_text(encoding="utf-8") == "status\ncurrent\n"
    manifest = json.loads(paths["analysis_manifest"].read_text(encoding="utf-8"))
    assert manifest["run_id"] == "atlas_descriptive"
def test_discover_atlases_reads_all_seven_cohorts(tmp_path: Path) -> None:
    nib = pytest.importorskip("nibabel")
    vertices = np.asarray(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0], [3.0, 0.0, 0.0]],
        dtype=np.float32,
    )
    faces = np.asarray([[0, 1, 2], [1, 2, 3]], dtype=np.int32)
    surface = tmp_path / "surface.surf.gii"
    nib.save(
        nib.GiftiImage(
            darrays=[
                nib.gifti.GiftiDataArray(vertices),
                nib.gifti.GiftiDataArray(faces),
            ]
        ),
        surface,
    )
    labels_path = tmp_path / "group_labels_final.npy"
    np.save(labels_path, np.asarray([1, 1, 2, 3], dtype=np.int32))
    input_root = tmp_path / "group_atlas"
    for group in EXPECTED_COHORTS:
        for variant in ("plain", "weighted"):
            for smoothing_mm in (2, 4):
                for hemisphere in ("L", "R"):
                    directory = _atlas_directory(
                        input_root, group, variant, smoothing_mm, hemisphere
                    )
                    directory.mkdir(parents=True, exist_ok=True)
                    manifest = _manifest(str(surface))
                    manifest["group"] = group
                    manifest["hemi"] = hemisphere
                    manifest["group_labels_final"] = str(labels_path)
                    (directory / "group_atlas_manifest.json").write_text(
                        json.dumps(manifest), encoding="utf-8"
                    )

    records = discover_atlases(input_root, expected_n_vertices=4)

    assert len(records) == 64
    assert {record.group for record in records} == set(EXPECTED_COHORTS)


def test_descriptive_contrasts_include_nested_clinical_cohorts() -> None:
    vertices = np.asarray(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0], [3.0, 0.0, 0.0]]
    )
    faces = np.asarray([[0, 1, 2], [1, 2, 3]])
    records = []
    for group in EXPECTED_COHORTS:
        for variant in ("plain", "weighted"):
            for smoothing_mm in (2, 4):
                for hemisphere in ("L", "R"):
                    records.append(
                        _synthetic_record(
                            group=group,
                            variant=variant,
                            smoothing_mm=smoothing_mm,
                            hemisphere=hemisphere,
                            labels=np.asarray([1, 1, 2, 3]),
                            vertices=vertices,
                            faces=faces,
                        )
                    )

    area_df, summaries = _area_rows(records)
    pair_df = build_matched_group_pairs(records, summaries)
    contrast_df = build_descriptive_contrasts(records, area_df, pair_df, summaries)

    expected_contrasts = {
        "NTS Any MDD-HC",
        "NTS Pure MDD-HC",
        "TIS PTSD-HC",
    }
    assert expected_contrasts.issubset(set(contrast_df["contrast"]))
    assert all(
        (first == "HC" or second == "HC")
        for first, second in zip(pair_df["group_a"], pair_df["group_b"])
    )
    assert not any(
        first != "HC" and second != "HC"
        for first, second in zip(pair_df["group_a"], pair_df["group_b"])
    )

def test_reference_area_plot_accepts_all_configured_cohorts(tmp_path: Path) -> None:
    pytest.importorskip("matplotlib")
    rows = []
    for group in EXPECTED_COHORTS:
        for hemisphere in ("L", "R"):
            for cluster in (1, 2, 3):
                rows.append(
                    {
                        "group": group,
                        "variant": "plain",
                        "smoothing_mm": 2,
                        "hemisphere": hemisphere,
                        "cluster": cluster,
                        "atlas_region_area_fraction": 1.0 / 3.0,
                    }
                )

    atlas_compare_module.plot_reference_area_profiles(pd.DataFrame(rows), tmp_path)

    assert (tmp_path / "reference_2mm_plain_area_profiles.png").is_file()
    assert not (tmp_path / "reference_2mm_plain_area_profiles.pdf").exists()
