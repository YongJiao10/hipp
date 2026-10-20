"""Cluster and WTA cortex profiles pool within-run signal with variance weights."""
from __future__ import annotations

import importlib
import json

import numpy as np
import pytest


panels = importlib.import_module("scripts.core_workflow.06_03_build_cluster_cortex_panels")
templates = importlib.import_module("lib.cortex_template_profiles")
CONSUMERS = [panels.compute_subject_cluster_roi_profiles, templates.compute_subject_label_roi_profiles]


def _write_inputs(root, *, missing=np.nan, write_summary=True):
    subject = root / "sub-01"
    surface_dir = subject / "surface/2mm"
    surface_dir.mkdir(parents=True)
    # The runs have unequal hippocampal variances. Per-run normalization would
    # cancel these opposing correlations instead of retaining the pooled FC.
    hippocampal = np.array([-2, 0, 2, -1, 0, 1], dtype=np.float32)
    cortex = np.array([-1, 0, 1, 1, 0, -1], dtype=np.float32)
    np.save(surface_dir / "sub-01_hemi-L_timeseries.npy", np.array([
        hippocampal + [100, 100, 100, -200, -200, -200],
        [missing, 5, 2, 0, 1, 8],
        -cortex * 100,
        [20, 20, 20, 40, 40, 40],
    ], dtype=np.float32))
    gate_dir = subject / "surface/tsnr_gate"
    gate_dir.mkdir()
    np.save(gate_dir / "sub-01_hemi-L_valid_mask.npy", np.array([True, True, False, True]))
    reference_dir = subject / "reference/kong2022"
    reference_dir.mkdir(parents=True)
    np.save(reference_dir / "cortex_roi_parcel_timeseries.npy", (cortex + [-300, -300, -300, 500, 500, 500])[None, :])
    (reference_dir / "cortex_roi_parcels.tsv").write_text(
        "parcel_id\tparcel_name\tfamily17\tcanonical7\tcanonical8\themisphere\n"
        "1\troi1\tDefaultA\tDefault\tDefault\tL\n", encoding="utf-8",
    )
    summary = reference_dir / "reference_summary.json"
    if write_summary:
        summary.write_text(json.dumps({"run_ids": [1, 2], "run_lengths": [3, 3]}), encoding="utf-8")
    return summary, hippocampal, cortex


def _profiles(root, consumer):
    return consumer(
        subjects=["01"], shared_root=root, reference_root=root, mask_root=root,
        smoothing="2mm", hemi="L",
        group_labels=np.array([1, 1, 1, 2]), support_mask=np.ones(4, bool),
        roi_registry=[{"roi_id": 1, "roi_name": "roi1"}],
    )


@pytest.mark.parametrize("missing", [np.nan, np.inf])
@pytest.mark.parametrize("consumer", CONSUMERS, ids=["cluster_panels", "wta_template"])
def test_cortex_comparison_centers_before_pooling_and_preserves_invalid_rows(tmp_path, missing, consumer):
    _, hippocampal, cortex = _write_inputs(tmp_path, missing=missing)
    profiles, subjects = _profiles(tmp_path, consumer)
    expected = np.arctanh(np.corrcoef(hippocampal, cortex)[0, 1])
    assert subjects == ["01"]
    np.testing.assert_allclose(profiles[0, 0, 0], expected, atol=2e-7)
    assert np.isnan(profiles[0, 1]).all(), "Run-offset-only cluster is undefined"
    assert np.isnan(profiles[:, :, 1:]).all(), "Absent ROI columns remain missing"


@pytest.mark.parametrize("consumer", CONSUMERS, ids=["cluster_panels", "wta_template"])
def test_cortex_comparison_requires_exact_reference_run_summary(tmp_path, consumer):
    summary, _, _ = _write_inputs(tmp_path, write_summary=False)
    with pytest.raises(FileNotFoundError, match=str(summary)):
        _profiles(tmp_path, consumer)
