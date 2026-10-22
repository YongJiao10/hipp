"""Weighted cluster FC respects subject masks, physical runs, and missing signal."""
from __future__ import annotations

import importlib
import json

import numpy as np
import pytest


branch = importlib.import_module("scripts.core_workflow.05_05_build_weighted_fisherz_branch")
from lib.weighted_fisherz import (  # noqa: E402
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
    mask_path = surface_root / "sub-01/surface/tsnr_gate/sub-01_hemi-L_valid_mask.npy"
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


def _load_subject(surface_root, reference_root):
    return branch.load_subject_weighted_cluster_profile(
        surface_root, reference_root, "01", "2mm", "L",
        np.array([1, 1]), np.ones(2, bool), ["A"],
        network_column="canonical7",
    )


def test_subject_loader_excludes_masked_nonconstant_vertex(tmp_path):
    surface_root, reference_root, _, _ = _write_subject(tmp_path)
    actual = _load_subject(surface_root, reference_root)
    expected = np.arctanh(np.corrcoef(HIPP, CORTEX)[0, 1])
    np.testing.assert_allclose(actual, [[expected]], atol=2e-7)


def test_subject_loader_requires_exact_subject_mask(tmp_path):
    surface_root, reference_root, mask_path, _ = _write_subject(tmp_path, write_mask=False)
    with pytest.raises(FileNotFoundError, match=str(mask_path)):
        _load_subject(surface_root, reference_root)


def test_subject_loader_rejects_mask_vertex_mismatch(tmp_path):
    surface_root, reference_root, mask_path, _ = _write_subject(tmp_path, mask=[True])
    with pytest.raises(ValueError, match=str(mask_path)):
        _load_subject(surface_root, reference_root)


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


def test_group_writer_preserves_missing_cells_for_assignment(tmp_path, monkeypatch):
    monkeypatch.setattr(branch, "ATLAS", "kong2022")
    monkeypatch.setattr(branch, "NETWORK_COLUMN", "canonical7")
    surface_root, reference_root, _, _ = _write_subject(tmp_path)
    reference_dir = reference_root / "sub-01/reference/kong2022"
    np.save(reference_dir / "cortex_roi_parcel_timeseries.npy", np.array([CORTEX, np.full(6, np.nan)]))
    (reference_dir / "cortex_roi_parcels.tsv").write_text(
        "canonical7\tn_grayordinates_used\nA\t1\nB\t1\n", encoding="utf-8"
    )
    dst = tmp_path / "outputs_hcp/work/group_atlas/weighted/gradient-spectral/kong2022/2mm/hemi_L"
    dst.mkdir(parents=True)
    np.save(dst / "group_labels_final.npy", np.array([1, 2]))
    np.save(dst / "group_support_mask.npy", np.ones(2, bool))
    np.save(dst / "cluster_network_profile.npy", np.zeros((2, 2), dtype=np.float32))
    (dst / "group_feature_summary.json").write_text(json.dumps({"network_names": ["A", "B"]}), encoding="utf-8")
    manifest_path = dst / "group_atlas_manifest.json"
    manifest_path.write_text(json.dumps({
        "subjects": ["01"], "n_subjects": 1, "smoothing": "2mm", "hemi": "L",
        "surface_affinity": "shifted-cosine",
        "affinity_description": "((1+cosine_similarity)/2)_hadamard_surface_adjacency_diag0",
        "resume_input_signature_payload": {"subjects_manifest": str(tmp_path / "subjects.txt")},
    }), encoding="utf-8")
    # Lineage validation and GIFTI/template rendering concern unrelated assets;
    # profile construction, aggregation, saving and assignment remain real.
    monkeypatch.setattr(branch, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(branch, "validate_group_atlas_lineage", lambda path: None)
    monkeypatch.setattr(branch, "normalize_copied_manifest_paths", lambda **kwargs: kwargs["manifest"])
    monkeypatch.setattr(branch, "materialize_combo", lambda *args: True)
    branch.recompute_gradient_weighted_run(
        dst_run=dst, surface_root=surface_root, reference_root=reference_root,
        color_map={}, template_surface_dir=tmp_path, template_spec=tmp_path / "unused.spec",
        source_manifest=manifest_path,
    )
    profile = np.load(dst / "cluster_network_profile_weighted_fisherz.npy")
    assert profile[0, 0] < 0
    assert np.isnan(profile[:, 1]).all()
    assert np.isnan(profile[1]).all()
    statuses = json.loads((dst / "cluster_assignment_status_weighted_fisherz.json").read_text())["items"]
    assert statuses[0]["top1_network"] == "A"
    assert statuses[1]["status"] == "unassigned"
