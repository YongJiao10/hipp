"""Contracts keeping active Kember workflow documentation internally aligned."""

from __future__ import annotations

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
ACTIVE_DOCUMENTS = (
    PROJECT_ROOT / "README.md",
    PROJECT_ROOT / "literature_and_kember_method_notes.md",
    PROJECT_ROOT / "docs" / "analysis_map.md",
    PROJECT_ROOT / "study_spec.md",
)
WORKFLOWS = (
    "cross_dataset_rest_fc",
    "kember_metrics",
    "external_group_atlas_analysis",
)
METHODS = ("hcp_template_matching", "hcp_fixed")


def test_active_docs_share_the_frozen_three_workflow_kember_contract() -> None:
    for path in ACTIVE_DOCUMENTS:
        text = path.read_text(encoding="utf-8")
        assert all(workflow in text for workflow in WORKFLOWS), path
        assert all(method in text for method in METHODS), path
        assert "512" in text and "surfareacorobl" in text, path
        assert "六个" in text and "432" in text, path


def test_active_docs_remove_stale_third_subject_branch_and_endpoint_claims() -> None:
    combined = "\n".join(path.read_text(encoding="utf-8") for path in ACTIVE_DOCUMENTS)

    assert "group-specific clustering：第三分支" not in combined
    assert "三条分支都用于个体差异分析" not in combined
    assert "合计 180 个 endpoint units" not in combined
    assert "native_midthickness_euclidean_radius_mm" not in combined
