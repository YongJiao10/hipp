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
