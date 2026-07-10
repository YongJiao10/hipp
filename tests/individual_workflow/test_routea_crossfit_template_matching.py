from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
CORE_WORKFLOW_DIR = REPO_ROOT / "scripts" / "core_workflow"
MODULE_PATH = (
    REPO_ROOT
    / "scripts"
    / "individual_workflow"
    / "routea_crossfit_template_matching.py"
)
if str(CORE_WORKFLOW_DIR) not in sys.path:
    sys.path.insert(0, str(CORE_WORKFLOW_DIR))

from lib import output_layout  # noqa: E402


def load_module():
    spec = importlib.util.spec_from_file_location(
        "routea_crossfit_template_matching",
        MODULE_PATH,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_grouped_folds_are_deterministic_balanced_and_keep_families_together():
    workflow = load_module()
    subjects = ["01", "02", "03", "04", "05", "06"]
    families = {"01": "a", "02": "a", "03": "b", "04": "c", "05": "d", "06": "e"}
    first = workflow.assign_grouped_folds(subjects, families, n_folds=3, seed=17)
    second = workflow.assign_grouped_folds(subjects, families, n_folds=3, seed=17)
    assert first == second
    assert first["01"] == first["02"]
    counts = np.bincount(list(first.values()), minlength=3)
    assert int(counts.max() - counts.min()) <= 1


def test_grouped_folds_normalize_subject_prefixes_and_reject_missing_families():
    workflow = load_module()

    got = workflow.assign_grouped_folds(
        ["sub-01", "02"],
        {"01": "a", "sub-02": "b"},
        n_folds=2,
        seed=17,
    )

    assert set(got) == {"01", "02"}
    with pytest.raises(ValueError, match="family"):
        workflow.assign_grouped_folds(["01", "02"], {"01": "a"}, n_folds=2, seed=17)


def test_validate_fold_assignment_rejects_families_split_across_folds():
    workflow = load_module()
    subjects = ["sub-01", "02", "03"]
    families = {"01": "a", "sub-02": "a", "03": "b"}

    workflow.validate_fold_assignment(subjects, families, {"01": 0, "02": 0, "03": 1}, n_folds=2)

    with pytest.raises(ValueError, match="Family"):
        workflow.validate_fold_assignment(
            subjects,
            families,
            {"01": 0, "02": 1, "03": 0},
            n_folds=2,
        )


def test_crossfit_output_roots_are_isolated_from_existing_individual_atlas():
    assert output_layout.resolve_crossfit_template_matching_root().name == "crossfit_template_matching"
    assert output_layout.resolve_crossfit_template_matching_overview_root().name == "crossfit_template_matching"


def test_cluster_prototypes_average_only_supported_ordered_cluster_vertices():
    workflow = load_module()
    fc = np.asarray(
        [[1.0, 0.0], [3.0, 2.0], [9.0, 9.0], [0.0, 4.0]],
        dtype=np.float32,
    )
    labels = np.asarray([1, 1, 0, 2], dtype=np.int32)
    support = np.asarray([1, 1, 0, 1], dtype=bool)

    got = workflow.build_cluster_prototypes(fc, labels, support, k=2)

    assert np.allclose(got, [[2.0, 1.0], [0.0, 4.0]])


def test_cluster_prototypes_reject_missing_cluster():
    workflow = load_module()

    with pytest.raises(ValueError, match="cluster 2"):
        workflow.build_cluster_prototypes(
            np.asarray([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32),
            np.asarray([1, 1], dtype=np.int32),
            np.asarray([1, 1], dtype=bool),
            k=2,
        )


def test_vertex_matching_returns_labels_scores_and_top1_top2_margin():
    workflow = load_module()
    fc = np.asarray(
        [[1.0, 0.0, -1.0], [-1.0, 0.0, 1.0], [5.0, 5.0, 5.0]],
        dtype=np.float32,
    )
    valid = np.asarray([1, 1, 1], dtype=bool)
    prototypes = np.asarray(
        [[1.0, 0.0, -1.0], [-1.0, 0.0, 1.0]],
        dtype=np.float32,
    )

    labels, scores, margin = workflow.match_vertices_to_prototypes(
        fc,
        valid,
        prototypes,
    )

    assert labels.tolist() == [1, 2, 0]
    assert scores.shape == (3, 2)
    assert np.allclose(margin[:2], [2.0, 2.0])
    assert np.isnan(margin[2])


def test_vertex_matching_uses_jointly_finite_features_and_keeps_invalid_vertices_null():
    workflow = load_module()
    fc = np.asarray(
        [[1.0, np.nan, -1.0, 0.0], [1.0, 0.0, -1.0, 0.0]],
        dtype=np.float32,
    )
    prototypes = np.asarray(
        [[1.0, 9.0, -1.0, 0.0], [-1.0, 9.0, 1.0, 0.0]],
        dtype=np.float32,
    )

    labels, scores, margin = workflow.match_vertices_to_prototypes(
        fc,
        np.asarray([1, 0], dtype=bool),
        prototypes,
    )

    assert labels.tolist() == [1, 0]
    assert scores[0, 0] == pytest.approx(1.0)
    assert scores[0, 1] == pytest.approx(-1.0)
    assert np.all(np.isnan(scores[1]))
    assert margin[0] == pytest.approx(2.0)
    assert np.isnan(margin[1])
