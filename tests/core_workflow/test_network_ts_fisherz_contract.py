"""Network-time-series cluster FC respects subject masks, physical runs, and missing signal."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest


CORE_DIR = Path(__file__).resolve().parents[2] / "scripts" / "core_workflow"
if str(CORE_DIR) not in sys.path:
    sys.path.insert(0, str(CORE_DIR))

from lib.network_ts_fisherz import (  # noqa: E402
    assignment_status_from_weighted_network_profile,
    subjectwise_cluster_network_fisherz_profile,
    weighted_cluster_network_fisherz_profile_from_timeseries,
    weighted_network_fisherz_profile_from_timeseries,
)


HIPP = np.array([0, 1, 4, 2, 5, 3], dtype=np.float32)
CORTEX = np.array([3, 0, 2, 5, 1, 4], dtype=np.float32)
ROI_ROWS = [{"canonical7": "A", "n_grayordinates_used": 1}]


def _cluster_profile(hipp, *, support=None, run_lengths=None):
    values = np.asarray(hipp, dtype=np.float32)
    kwargs = {} if run_lengths is None else {"run_lengths": run_lengths}
    return weighted_cluster_network_fisherz_profile_from_timeseries(
        np.ones(values.shape[0], dtype=np.int32),
        np.ones(values.shape[0], dtype=bool) if support is None else support,
        values, CORTEX[None, :], ROI_ROWS, ["A"], **kwargs,
    )


@pytest.mark.parametrize("missing", [np.nan, np.inf])
def test_incomplete_vertex_cannot_be_reanimated_by_cluster_nanmean(missing):
    incomplete = np.array([missing, 5, 2, 0, 1, 8], dtype=np.float32)
    actual = _cluster_profile([HIPP, incomplete])
    expected = np.arctanh(np.corrcoef(HIPP, CORTEX)[0, 1])
    np.testing.assert_allclose(actual, [[expected]], atol=2e-7)


@pytest.mark.parametrize("padding", [0.0, 7.0])
def test_zero_and_constant_padding_preserve_finite_cluster_correlation(padding):
    actual = _cluster_profile([HIPP, np.full(6, padding, dtype=np.float32)])
    expected = np.arctanh(np.corrcoef(HIPP, CORTEX)[0, 1])
    np.testing.assert_allclose(actual, [[expected]], atol=2e-7)


@pytest.mark.parametrize("row", [np.zeros(6), np.full(6, 7), np.full(6, np.nan)])
def test_cluster_with_no_eligible_vertex_stays_missing(row):
    assert np.isnan(_cluster_profile([row])).all()


def test_run_offsets_are_removed_before_cluster_row_eligibility():
    signal = np.array([-1, 1, 0, -1, 0, 1], dtype=np.float32)
    # This vertex varies only between physical runs and must not contribute.
    offset_only = np.array([5, 5, 5, 9, 9, 9], dtype=np.float32)
    actual = _cluster_profile([signal + [20, 20, 20, 40, 40, 40], offset_only], run_lengths=[3, 3])
    centered_cortex = np.array([4/3, -5/3, 1/3, 5/3, -7/3, 2/3])
    expected = np.arctanh(np.corrcoef(signal, centered_cortex)[0, 1])
    np.testing.assert_allclose(actual, [[expected]], atol=2e-7)
    assert np.isnan(_cluster_profile([offset_only], run_lengths=[3, 3])).all()


def test_weighted_network_fc_removes_both_sides_run_offsets():
    signal = np.array([-1, 1, 0, -1, 0, 1], dtype=np.float32)
    actual = weighted_network_fisherz_profile_from_timeseries(
        (signal + [20, 20, 20, 40, 40, 40])[None, :],
        CORTEX[None, :] + [30, 30, 30, -40, -40, -40],
        ROI_ROWS, ["A"], run_lengths=[3, 3],
    )
    centered_cortex = np.array([4/3, -5/3, 1/3, 5/3, -7/3, 2/3])
    expected = np.arctanh(np.corrcoef(signal, centered_cortex)[0, 1])
    np.testing.assert_allclose(actual, [[expected]], atol=2e-7)


def _write_subject(root, *, hipp=None, mask=(True, False), write_mask=True, write_summary=True):
    surface_root = root / "surface_inputs"
    reference_root = root / "reference_inputs"
    ts_path = surface_root / "sub-01/surface/2mm/sub-01_hemi-L_timeseries.npy"
    ts_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(ts_path, np.array([HIPP, CORTEX]) if hipp is None else hipp)
    mask_path = surface_root / "sub-01/surface/validity/sub-01_hemi-L_valid_mask.npy"
    if write_mask:
        mask_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(mask_path, np.asarray(mask, dtype=bool))
    reference_dir = reference_root / "sub-01/reference/kong2022"
    reference_dir.mkdir(parents=True, exist_ok=True)
    np.save(reference_dir / "cortex_roi_parcel_timeseries.npy", CORTEX[None, :])
    (reference_dir / "cortex_roi_parcels.tsv").write_text(
        "parcel_id\tparcel_name\tfamily17\tcanonical8\tcanonical7\themisphere\tsubregion\trank\tn_grayordinates_used\n"
        "1\troi1\tDefaultA\tA\tA\tL\t\t1\t1\n", encoding="utf-8"
    )
    summary = reference_dir / "reference_summary.json"
    if write_summary:
        summary.write_text(json.dumps({"run_ids": [1], "run_lengths": [6]}), encoding="utf-8")
    return surface_root, reference_root, mask_path, summary


def _load_subject(surface_root, reference_root, masks=None):
    if masks is None:
        mask_path = surface_root / "sub-01/surface/validity/sub-01_hemi-L_valid_mask.npy"
        masks = {"01": np.load(mask_path)}
    _, mean, used = subjectwise_cluster_network_fisherz_profile(
        group_labels_final=np.array([1, 1]), support_mask=np.ones(2, bool), subjects=["01"],
        surface_root=surface_root, reference_root=reference_root, smoothing="2mm", hemi="L",
        valid_masks_by_subject=masks, network_names=["A"], network_column="canonical7",
    )
    assert used == ["01"]
    return mean


def test_subject_loader_excludes_masked_nonconstant_vertex(tmp_path):
    surface_root, reference_root, _, _ = _write_subject(tmp_path)
    actual = _load_subject(surface_root, reference_root)
    expected = np.arctanh(np.corrcoef(HIPP, CORTEX)[0, 1])
    np.testing.assert_allclose(actual, [[expected]], atol=2e-7)


def test_subject_loader_requires_the_subject_mask(tmp_path):
    surface_root, reference_root, _, _ = _write_subject(tmp_path)
    with pytest.raises(ValueError, match="Missing valid mask for sub-01"):
        _load_subject(surface_root, reference_root, masks={})


def test_subject_loader_rejects_mask_vertex_mismatch(tmp_path):
    surface_root, reference_root, _, _ = _write_subject(tmp_path)
    with pytest.raises(ValueError, match="valid mask vertex mismatch for sub-01"):
        _load_subject(surface_root, reference_root, masks={"01": np.array([True])})


def test_subject_loader_consumes_declared_physical_run_lengths(tmp_path):
    signal = np.array([-1, 1, 0, -1, 0, 1], dtype=np.float32)
    surface_root, reference_root, _, summary = _write_subject(
        tmp_path, hipp=np.array([signal + [20, 20, 20, 40, 40, 40], CORTEX])
    )
    summary.write_text(json.dumps({"run_ids": [1, 2], "run_lengths": [3, 3]}), encoding="utf-8")
    actual = _load_subject(surface_root, reference_root)
    centered_cortex = np.array([4/3, -5/3, 1/3, 5/3, -7/3, 2/3])
    expected = np.arctanh(np.corrcoef(signal, centered_cortex)[0, 1])
    np.testing.assert_allclose(actual, [[expected]], atol=2e-7)


def test_subject_loader_fails_for_missing_run_metadata(tmp_path):
    surface_root, reference_root, _, summary = _write_subject(tmp_path, write_summary=False)
    with pytest.raises(FileNotFoundError, match=str(summary)):
        _load_subject(surface_root, reference_root)


def test_missing_profile_cells_stay_unassigned():
    profile = np.array([[-0.4, np.nan], [np.nan, np.nan]], dtype=np.float32)
    statuses, _ = assignment_status_from_weighted_network_profile(profile, ["A", "B"])
    assert statuses[0]["top1_network"] == "A"
    assert statuses[1]["status"] == "unassigned"
