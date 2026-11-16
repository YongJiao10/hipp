"""Documentation-status contracts for the unified Kember workflow."""

from __future__ import annotations

import json
from pathlib import Path


ANALYSIS_ROOT = Path(__file__).resolve().parents[2]


def test_prior_transition_area_is_not_a_kember_endpoint() -> None:
    text = (ANALYSIS_ROOT / "docs" / "kember_area_method.md").read_text(encoding="utf-8")
    assert "transition_area" in text
    assert "not a Kember endpoint" in text


def test_group_atlas_geometry_is_not_a_kember_method_result() -> None:
    text = (ANALYSIS_ROOT / "external_group_atlas_analysis" / "README.md").read_text(encoding="utf-8")
    assert "KEMBER_REPLICATION_STATUS=group_geometry_only" in text


def test_new_area_branch_documents_the_native_area_workflow() -> None:
    text = (ANALYSIS_ROOT / "kember_metrics" / "README.md").read_text(encoding="utf-8")
    assert "KEMBER_METHOD_NATIVE_AREA_WORKFLOW_READY" in text
    assert "HippUnfold native" in text
    assert "per-vertex surface-area" in text
    assert "ADAP_BARY_AREA" in text
    assert "禁止个体聚类" in text
    assert "绝对面积是 Kember 对齐的主指标" in text
    assert "相对面积是辅助指标" in text
    assert "total_valid_surface_area_mm2" in text


def test_source_lock_freezes_the_current_native_area_contract() -> None:
    lock = json.loads(
        (ANALYSIS_ROOT / "kember_metrics" / "source_lock.json").read_text(
            encoding="utf-8"
        )
    )
    implementation = lock["kember_method_area_implementation"]

    assert implementation["status"] == "KEMBER_METHOD_NATIVE_AREA_WORKFLOW_READY"
    assert "HippUnfold native per-vertex" in implementation["evidence"]
    assert implementation["label_transfer"]["method"] == "ADAP_BARY_AREA"
    assert implementation["metric_roles"]["primary"] == "absolute system surface area in mm^2"
    assert implementation["metric_roles"]["secondary_denominator"] == (
        "total native surface area within the valid native ROI"
    )
    assert implementation["implementation_scope"] == (
        "The unified subject-level workflow has exactly two active methods: "
        "hcp_template_matching and hcp_fixed."
    )
    assert implementation["no_per_subject_clustering"] == (
        "Group-specific clustering belongs only to "
        "external_group_atlas_analysis; kember_metrics has no clustering method."
    )


def test_canonical_docs_define_three_workflows_and_two_kember_methods() -> None:
    text = (ANALYSIS_ROOT / "docs" / "analysis_map.md").read_text(encoding="utf-8")
    study_spec = (ANALYSIS_ROOT / "study_spec.md").read_text(encoding="utf-8")

    for document in (text, study_spec):
        assert "cross_dataset_rest_fc" in document
        assert "kember_metrics" in document
        assert "external_group_atlas_analysis" in document
        assert document.index("hcp_template_matching") < document.index("hcp_fixed")
        assert "432" in document
        assert "512" in document
        assert "surfareacorobl" in document
    assert "禁止个体聚类" in text
