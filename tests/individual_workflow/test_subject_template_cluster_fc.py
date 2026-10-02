from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
MODULE_PATH = (
    REPO_ROOT
    / "scripts"
    / "individual_workflow"
    / "04_02_compute_subject_template_cluster_fc.py"
)


def load_module():
    assert MODULE_PATH.is_file(), "Subject template-cluster FC script is missing"
    spec = importlib.util.spec_from_file_location(
        "compute_subject_template_cluster_fc",
        MODULE_PATH,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_cluster_fc_averages_each_subject_cluster_timeseries_before_fc():
    workflow = load_module()
    surface = np.array(
        [
            [0, 1, 2, 3, 4, 5],
            [0, 0, 1, 2, 3, 5],
            [5, 4, 3, 2, 1, 0],
            [5, 5, 4, 3, 2, 0],
            [0, 2, 1, 3, 5, 4],
            [1, 3, 0, 4, 2, 5],
        ],
        dtype=np.float32,
    )
    labels = np.array([1, 1, 2, 2, 3, 3], dtype=np.int32)
    valid_mask = np.ones(6, dtype=bool)
    roi = np.array(
        [
            [0, 1, 2, 3, 4, 5],
            [5, 4, 3, 2, 1, 0],
            [2, 1, 3, 0, 4, 5],
            [1, 4, 0, 5, 2, 3],
        ],
        dtype=np.float32,
    )

    result = workflow.compute_cluster_fc(
        surface_timeseries=surface,
        roi_timeseries=roi,
        labels=labels,
        valid_mask=valid_mask,
    )

    expected_cluster_ts = np.vstack(
        [surface[labels == cluster].mean(axis=0) for cluster in (1, 2, 3)]
    )
    expected_pearson = workflow.corrcoef_rows(expected_cluster_ts, roi)
    expected_fisherz = workflow.fisher_z_transform_fc(expected_pearson)
    np.testing.assert_allclose(
        result["cluster_mean_timeseries"], expected_cluster_ts, rtol=0, atol=1e-6
    )
    np.testing.assert_allclose(
        result["pearson_fc"], expected_pearson, rtol=0, atol=1e-6
    )
    np.testing.assert_allclose(
        result["fisherz_fc"], expected_fisherz, rtol=0, atol=1e-6
    )
    assert result["pearson_fc"].shape == (3, 4)
    assert result["cluster_vertex_counts"].tolist() == [2, 2, 2]


def test_cluster_fc_applies_roi_availability_mask_without_changing_cluster_means():
    workflow = load_module()
    surface = np.arange(18, dtype=np.float32).reshape(3, 6)
    roi = np.arange(24, dtype=np.float32).reshape(4, 6)
    labels = np.array([1, 2, 3], dtype=np.int32)
    valid_mask = np.ones(3, dtype=bool)
    roi_used_mask = np.array([True, False, True, True], dtype=bool)

    result = workflow.compute_cluster_fc(
        surface_timeseries=surface,
        roi_timeseries=roi,
        labels=labels,
        valid_mask=valid_mask,
        roi_used_mask=roi_used_mask,
    )

    assert np.isnan(result["pearson_fc"][:, 1]).all()
    assert np.isnan(result["fisherz_fc"][:, 1]).all()
    assert np.isfinite(result["pearson_fc"][:, [0, 2, 3]]).all()


def test_cluster_fc_rejects_missing_subject_specific_cluster():
    workflow = load_module()
    with pytest.raises(ValueError, match="cluster 3"):
        workflow.compute_cluster_fc(
            surface_timeseries=np.ones((3, 4), dtype=np.float32),
            roi_timeseries=np.ones((2, 4), dtype=np.float32),
            labels=np.array([1, 2, 2], dtype=np.int32),
            valid_mask=np.ones(3, dtype=bool),
        )

def test_refresh_manifest_updates_inputs_without_changing_output_records():
    workflow = load_module()
    old_inputs = {
        "combo_manifest": {"path": "/old/combo.json", "sha256": "old"},
        "labels": {"path": "/old/labels.npy", "sha256": "same"},
    }
    current_inputs = {
        "combo_manifest": {"path": "/new/combo.json", "sha256": "new"},
        "labels": {"path": "/old/labels.npy", "sha256": "same"},
    }
    original = {
        "status": "ok",
        "inputs": old_inputs,
        "outputs": {"pearson_fc": {"sha256": "matrix"}},
        "external_copies": {"pearson_fc": {"sha256": "matrix"}},
    }

    refreshed = workflow.refresh_local_manifest_payload(
        original,
        current_inputs=current_inputs,
        refreshed_utc="2026-09-14T00:00:00+00:00",
    )

    assert refreshed["inputs"] == current_inputs
    assert refreshed["outputs"] == original["outputs"]
    assert refreshed["external_copies"] == original["external_copies"]
    assert refreshed["provenance_refresh_history"][-1] == {
        "changed_inputs": ["combo_manifest"],
        "reason": "upstream provenance manifests changed after FC calculation; output arrays were retained after hash validation",
        "refreshed_utc": "2026-09-14T00:00:00+00:00",
    }
    assert original["inputs"] == old_inputs

