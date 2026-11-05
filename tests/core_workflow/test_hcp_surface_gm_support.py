"""NIfTI geometry and per-run sampled signals define hippocampal coverage."""
from __future__ import annotations

import importlib
import json
import sys

import nibabel as nib
import numpy as np
import pytest


sampling = importlib.import_module("lib.surface_sampling")
prep = importlib.import_module("scripts.core_workflow.02_01_prepare_group_inputs")


def write_surface(path, points):
    nib.save(nib.gifti.GiftiImage(darrays=[nib.gifti.GiftiDataArray(
        np.asarray(points, dtype=np.float32), intent="NIFTI_INTENT_POINTSET",
    )]), path)


def write_bold(path, frames=2, *, affine=None):
    nib.save(nib.Nifti1Image(np.zeros((3, 3, 3, frames), dtype=np.float32),
                            np.eye(4) if affine is None else affine), path)


def test_nifti_support_intersects_declared_volume_grids(tmp_path):
    first, second = tmp_path / 'first.nii.gz', tmp_path / 'second.nii.gz'
    write_bold(first)
    affine = np.eye(4); affine[0, 3] = 2
    write_bold(second, affine=affine)
    surfaces = {h: tmp_path / f'{h}.surf.gii' for h in ('L', 'R')}
    write_surface(surfaces['L'], [[.25, 0, 0], [1.5, 0, 0]])
    write_surface(surfaces['R'], [[2.5, 1, 0], [2, 1, 0]])
    weights, summary = sampling.sample_hippocampal_nifti_support(
        bold_inputs=[first, second], run_ids=[1, 2], run_lengths=[2, 2], sampling_surfaces=surfaces)
    np.testing.assert_allclose(weights['L'], [0, .5])
    np.testing.assert_allclose(weights['R'], [.5, 1])
    assert summary['L']['n_vertices_without_grid_support'] == 1


def test_nifti_support_uses_volume_affine_at_registered_vertices(tmp_path):
    affine = np.eye(4); affine[:3, :3] *= 2; affine[:3, 3] = [10, 20, 30]
    path = tmp_path / 'bold.nii.gz'
    write_bold(path, affine=affine)
    surfaces = {h: tmp_path / f'{h}.surf.gii' for h in ('L', 'R')}
    write_surface(surfaces['L'], [[11, 20, 30], [15, 20, 30]])
    write_surface(surfaces['R'], [[17, 22, 30]])
    weights, _ = sampling.sample_hippocampal_nifti_support(
        bold_inputs=[path], run_ids=[1], run_lengths=[2], sampling_surfaces=surfaces)
    np.testing.assert_allclose(weights['L'], [1, .5])
    np.testing.assert_array_equal(weights['R'], [0])


def test_nifti_support_rejects_inconsistent_physical_run_frames(tmp_path):
    path = tmp_path / 'bold.nii.gz'; write_bold(path)
    surfaces = {h: tmp_path / f'{h}.surf.gii' for h in ('L', 'R')}
    for surface in surfaces.values(): write_surface(surface, [[0, 0, 0]])
    with pytest.raises(ValueError, match='run|frame|timepoint'):
        sampling.sample_hippocampal_nifti_support(bold_inputs=[path], run_ids=[1, 2],
            run_lengths=[2, 2], sampling_surfaces=surfaces)


def test_hcp_vertex_mask_combines_nifti_support_with_consumed_signal_validity(tmp_path):
    shared = tmp_path / "shared"
    raw = shared / "sub-01/surface/raw"
    raw.mkdir(parents=True)
    data = np.asarray([[0, 1, 2], [0, 1, 2], [1, 1, 1], [0, np.nan, 2]], dtype=np.float32)
    records = {}
    for hemi in ("L", "R"):
        nib.save(prep.make_func_gifti(data), raw / f"sub-01_hemi-{hemi}_space-T1w_den-512_label-hipp_bold.func.gii")
        weight = raw / f"sub-01_hemi-{hemi}_nifti_support.npy"
        np.save(weight, np.asarray([0, .25, 1, 1], dtype=np.float32))
        records[hemi] = {"n_vertices_total": 4, "n_vertices_unsupported": 1,
                         "nifti_support_path": str(weight), "n_vertices_valid": 1}
        surface = prep.surface_path(tmp_path / "batch", "01", hemi)
        surface.parent.mkdir(parents=True, exist_ok=True)
        write_surface(surface, np.zeros((4, 3)))
    (raw / "sub-01_coverage_summary.json").write_text(json.dumps({
        "subject": "01", "coverage_policy": "bold_nifti_runwise_trilinear_validity",
        "sampling_space": "MNI152NLin6Asym", "run_ids": [1], "run_lengths": [3],
        "hemispheres": records,
    }))
    (raw / "surface_sampling_summary.json").write_text(json.dumps({"run_ids": [1], "run_lengths": [3]}))

    result = prep.run_validity("01", str(tmp_path / "batch"), str(shared))

    assert result["ok"]
    gate = shared / "sub-01/surface/validity"
    np.testing.assert_array_equal(np.load(gate / "sub-01_hemi-L_valid_mask.npy"), [False, True, False, False])
    summary = json.loads((gate / "sub-01_hemi-L_validity_summary.json").read_text())
    assert summary["coverage_policy"] == "bold_nifti_runwise_trilinear_validity"


def test_hcp_concat_sampling_records_nifti_support_and_physical_runs(tmp_path, monkeypatch):
    bold = tmp_path / "bold.nii.gz"
    nib.save(nib.Nifti1Image(np.zeros((3, 3, 3, 4), dtype=np.float32), np.eye(4)), bold)
    metadata = tmp_path / "runs.json"
    metadata.write_text(json.dumps({"subject": "sub-01", "run_ids": [1, 2], "run_lengths": [2, 2],
        "runs": [{"run_id": 1, "run_label": "first", "n_timepoints": 2},
                 {"run_id": 2, "run_label": "second", "n_timepoints": 2}]}))
    logs = tmp_path / "logs"
    logs.mkdir()
    surfaces = {hemi: tmp_path / f"hemi-{hemi}_space-T1w_midthickness.surf.gii" for hemi in ("L", "R")}
    write_surface(surfaces["L"], [[.5, 0, 0], [2, 0, 0]])
    write_surface(surfaces["R"], [[.5, 1, 0], [2, 1, 0]])

    def sample(**kwargs):
        records = []
        for hemi in ("L", "R"):
            metric = kwargs["outdir"] / f"sub-01_hemi-{hemi}_space-T1w_den-512_label-hipp_bold.func.gii"
            nib.save(prep.make_func_gifti(np.asarray([[0, 1, 2, 3], [0, 1, 2, 3]])), metric)
            records.append({"hemi": hemi, "surface": str(surfaces[hemi]), "source_surface": str(surfaces[hemi]),
                            "space": "T1w", "sampling_space": "MNI152NLin6Asym", "mapping_method": "trilinear"})
        return records

    monkeypatch.setattr(prep, "sample_hipp_surface_timeseries", sample)
    result = prep.run_raw_sampling(
        "01", str(tmp_path / "batch"), str(tmp_path / "shared"), str(tmp_path),
        str(bold), str(logs), run_metadata_template=str(metadata),
    )

    assert result["ok"], result
    raw = tmp_path / "shared/sub-01/surface/raw"
    coverage = json.loads((raw / "sub-01_coverage_summary.json").read_text())
    assert coverage["run_ids"] == [1, 2]
    assert coverage["run_lengths"] == [2, 2]
    assert coverage["run_labels"] == ["first", "second"]
    assert coverage["density_evidence"]["n_vertices"] == {"L": 2, "R": 2}
    assert coverage["hemispheres"]["L"]["n_vertices_unsupported"] == 0
    assert coverage["hemispheres"]["R"]["n_vertices_unsupported"] == 0
    assert prep.raw_done(tmp_path / "shared", "01")


def test_hcp_concat_checks_declared_run_frames_before_sampling(tmp_path, monkeypatch):
    bold = tmp_path / "bold.nii.gz"
    nib.save(nib.Nifti1Image(np.zeros((3, 3, 3, 4), dtype=np.float32), np.eye(4)), bold)
    metadata = tmp_path / "runs.json"
    metadata.write_text(json.dumps({"subject": "01", "run_ids": [1], "run_lengths": [2],
                                   "runs": [{"run_id": 1, "run_label": "first", "n_timepoints": 2}]}))
    logs = tmp_path / "logs"
    logs.mkdir()

    def cannot_sample(**kwargs):
        raise AssertionError("sampling ran before the physical-run contract was checked")

    monkeypatch.setattr(prep, "sample_hipp_surface_timeseries", cannot_sample)
    result = prep.run_raw_sampling(
        "01", str(tmp_path / "batch"), str(tmp_path / "shared"), str(tmp_path),
        str(bold), str(logs), run_metadata_template=str(metadata),
    )

    assert not result["ok"]
    assert "timepoints" in result["error"]


def test_explicit_log_directory_keeps_run_logs_outside_formal_inputs(tmp_path, monkeypatch):
    route = tmp_path / "inputs/kong2022"
    logs = tmp_path / "runs/current/prepare"
    monkeypatch.setattr(sys, "argv", ["prepare", "--stage", "validity", "--subjects", "01",
        "--hippunfold-batch-root", str(tmp_path / "anatomy"), "--route-root", str(route),
        "--shared-root", str(tmp_path / "surface"), "--log-dir", str(logs)])

    assert prep.main() == 0
    assert logs.is_dir()
    assert not (route / "logs").exists()

