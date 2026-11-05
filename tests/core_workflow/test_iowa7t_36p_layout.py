"""Tests for Iowa7T 36P canonical output layout."""

from pathlib import Path

import pytest

from scripts.core_workflow.lib import output_layout as layout


def test_iowa7t_36p_root_uses_local_jianglab_prefix(monkeypatch):
    monkeypatch.setattr(
        layout.Path,
        "exists",
        lambda self: str(self) == "/home/yojiao/LSS/jianglab",
    )

    root = layout.resolve_iowa7t_36p_root()

    assert root == Path(
        "/home/yojiao/LSS/jianglab/3_Data_Working/"
        "fmriprep_processed_Iowa7T/DataOrg_PFM_36P"
    )


def test_iowa7t_36p_root_uses_argon_prefix_when_local_missing(monkeypatch):
    monkeypatch.setattr(
        layout.Path,
        "exists",
        lambda self: str(self) == "/Shared/jianglab",
    )

    root = layout.resolve_iowa7t_36p_root()

    assert root == Path(
        "/Shared/jianglab/3_Data_Working/"
        "fmriprep_processed_Iowa7T/DataOrg_PFM_36P"
    )


def test_iowa7t_subject_layout_has_shared_hippunfold_and_direct_branches():
    root = Path("/tmp/hipp_outputs")

    subject = layout.iowa7t_subject_layout("06112026", hipp_outputs_root=root)

    assert subject.subject_root == root / "subjects" / "sub-06112026"
    assert subject.hippunfold_root == root / "subjects" / "sub-06112026" / "hippunfold"
    assert subject.branch_root("multiband") == root / "subjects" / "sub-06112026" / "multiband"
    assert subject.branch_root("multi_echo") == root / "subjects" / "sub-06112026" / "multi_echo"
    assert "variants" not in subject.branch_root("multi_echo").parts


def test_iowa7t_branch_layout_paths_are_branch_scoped():
    branch = layout.iowa7t_branch_layout(
        "06112026",
        "multi_echo",
        hipp_outputs_root=Path("/tmp/hipp_outputs"),
    )

    assert branch.root == Path("/tmp/hipp_outputs/subjects/sub-06112026/multi_echo")
    assert branch.cortex == branch.root / "cortex"
    assert branch.surface == branch.root / "surface"
    assert branch.fc == branch.root / "fc"
    assert branch.atlas == branch.root / "atlas"
    assert branch.figures == branch.root / "figures"
    assert branch.provenance == branch.root / "provenance.json"


def test_iowa7t_layout_rejects_bad_subject_and_branch():
    with pytest.raises(ValueError, match="Invalid Iowa7T subject"):
        layout.iowa7t_subject_layout("sub-abc", hipp_outputs_root=Path("/tmp/hipp_outputs"))

    with pytest.raises(ValueError, match="Invalid Iowa7T branch"):
        layout.iowa7t_branch_layout(
            "06112026",
            "variants",
            hipp_outputs_root=Path("/tmp/hipp_outputs"),
        )


def test_iowa7t_layout_never_resolves_to_repo_outputs_iowa():
    root = layout.resolve_iowa7t_hipp_outputs_root(jianglab_root=Path("/tmp/jianglab"))

    assert root == Path(
        "/tmp/jianglab/3_Data_Working/fmriprep_processed_Iowa7T/"
        "DataOrg_PFM_36P/hipp_outputs"
    )
    assert "outputs_iowa" not in root.parts


def test_iowa7t_noxcpd_root_uses_variant_specific_directory():
    root = layout.resolve_iowa7t_data_root(
        variant="noxcpd",
        jianglab_root=Path("/tmp/jianglab"),
    )

    assert root == Path(
        "/tmp/jianglab/3_Data_Working/fmriprep_processed_Iowa7T/"
        "DataOrg_PFM_NoXCPD"
    )


def test_iowa7t_hipp_outputs_root_defaults_to_36p_variant():
    root = layout.resolve_iowa7t_hipp_outputs_root(jianglab_root=Path("/tmp/jianglab"))

    assert root == Path(
        "/tmp/jianglab/3_Data_Working/fmriprep_processed_Iowa7T/"
        "DataOrg_PFM_36P/hipp_outputs"
    )


def test_iowa7t_hipp_outputs_root_accepts_noxcpd_variant():
    root = layout.resolve_iowa7t_hipp_outputs_root(
        variant="noxcpd",
        jianglab_root=Path("/tmp/jianglab"),
    )

    assert root == Path(
        "/tmp/jianglab/3_Data_Working/fmriprep_processed_Iowa7T/"
        "DataOrg_PFM_NoXCPD/hipp_outputs"
    )


def test_iowa7t_hipp_outputs_root_accepts_0p_32p_variants():
    cases = {
        "0p": "DataOrg_PFM_0P/hipp_outputs",
        "32p": "DataOrg_PFM_32P/hipp_outputs",
    }
    for variant, suffix in cases.items():
        root = layout.resolve_iowa7t_hipp_outputs_root(
            variant=variant,
            jianglab_root=Path("/tmp/jianglab"),
        )
        assert root == Path("/tmp/jianglab/3_Data_Working/fmriprep_processed_Iowa7T") / suffix


def test_iowa7t_layout_rejects_bad_variant():
    with pytest.raises(ValueError, match="Invalid Iowa7T variant"):
        layout.resolve_iowa7t_data_root(
            variant="xcpd-lite",
            jianglab_root=Path("/tmp/jianglab"),
        )


def test_iowa7t_migration_preserves_fc_without_restoring_individual_clustering(tmp_path):
    from scripts.iowa7t.migrate_hipp_outputs_layout import build_mappings

    subject = "06112026"
    current_fc = (
        tmp_path / "inputs/kong2022/shared_inputs/routea"
        / f"sub-{subject}" / "fc/kong2022/2mm/L/fc.npy"
    )
    retired = [
        tmp_path / f"work/individual_atlas/sub-{subject}/gradient/2mm/L/labels.npy",
        tmp_path / f"work/individual_atlas/sub-{subject}/wta/2mm/L/labels.npy",
        tmp_path / f"present/overviews/individual/gradient/sub-{subject}_overview.png",
        tmp_path / f"present/overviews/individual/wta/sub-{subject}_overview.png",
        tmp_path / "work/individual_atlas/_overview_core_adapter/manifest.json",
        tmp_path / "work/individual_atlas/_runs/log.json",
    ]
    for path in [current_fc, *retired]:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"fixture")

    mappings = build_mappings(tmp_path, [subject])

    assert [(item.source, item.destination) for item in mappings] == [
        (
            current_fc,
            tmp_path / f"subjects/sub-{subject}/multiband/fc/kong2022/2mm/L/fc.npy",
        )
    ]
    assert all(path.is_file() for path in retired)
