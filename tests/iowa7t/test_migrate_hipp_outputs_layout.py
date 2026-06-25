"""Tests for Iowa7T hippocampal output layout migration."""

from pathlib import Path

import pytest

from scripts.iowa7t import migrate_hipp_outputs_layout as mig


def write_file(path: Path, text: str = "x") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def test_build_mapping_places_04172026_single_echo_outputs_without_variants(tmp_path):
    legacy = tmp_path / "hipp_outputs"
    write_file(
        legacy
        / "inputs/hippunfold/dense_corobl_batch/sub-04172026/hippunfold/sub-04172026/surf/a.func.gii"
    )
    write_file(
        legacy
        / "inputs/kong2022/shared_inputs/surface/sub-04172026/surface/raw/b.func.gii"
    )
    write_file(legacy / "inputs/kong2022/shared_inputs/routea/sub-04172026/fc/c.npy")
    write_file(legacy / "work/individual_atlas/sub-04172026/gradient/d.json")
    write_file(legacy / "work/individual_atlas/sub-04172026/wta/e.json")
    write_file(legacy / "present/overviews/individual/gradient/f.png")

    mappings = mig.build_mappings(legacy, subjects=("04172026",))
    destinations = {m.destination.relative_to(legacy) for m in mappings}

    assert Path("subjects/sub-04172026/hippunfold/sub-04172026/surf/a.func.gii") in destinations
    assert Path("subjects/sub-04172026/single_echo/surface/raw/b.func.gii") in destinations
    assert Path("subjects/sub-04172026/single_echo/fc/c.npy") in destinations
    assert Path("subjects/sub-04172026/single_echo/atlas/gradient/d.json") in destinations
    assert Path("subjects/sub-04172026/single_echo/atlas/wta/e.json") in destinations
    assert Path("subjects/sub-04172026/single_echo/figures/gradient/f.png") in destinations
    assert all("variants" not in path.parts for path in destinations)


def test_empty_group_sources_are_not_mapped_and_do_not_create_group(tmp_path):
    legacy = tmp_path / "hipp_outputs"
    (legacy / "inputs/kong2022/templates/hipp_group_average_t1only_151").mkdir(parents=True)
    (legacy / "inputs/kong2022/qc").mkdir(parents=True)
    (legacy / "inputs/kong2022/manifests").mkdir(parents=True)

    mappings = mig.build_mappings(legacy, subjects=("04172026",))
    mig.apply_mappings(mappings, mode="copy", write_manifest=False)

    assert mappings == []
    assert not (legacy / "group").exists()


def test_nonempty_group_sources_are_mapped(tmp_path):
    legacy = tmp_path / "hipp_outputs"
    write_file(legacy / "inputs/kong2022/manifests/status.tsv")

    mappings = mig.build_mappings(legacy, subjects=("04172026",))

    assert [m.destination.relative_to(legacy) for m in mappings] == [
        Path("group/manifests/status.tsv")
    ]


def test_collision_detection_refuses_different_destination_content(tmp_path):
    legacy = tmp_path / "hipp_outputs"
    write_file(legacy / "inputs/kong2022/shared_inputs/routea/sub-04172026/fc/a.txt", "old")
    write_file(legacy / "subjects/sub-04172026/single_echo/fc/a.txt", "new")

    mappings = mig.build_mappings(legacy, subjects=("04172026",))

    with pytest.raises(FileExistsError, match="Destination exists with different content"):
        mig.validate_mappings(mappings)
