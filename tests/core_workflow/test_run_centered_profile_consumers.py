from __future__ import annotations

import importlib
import json
from pathlib import Path

import numpy as np
import pytest

atlas = importlib.import_module("scripts.core_workflow.05_02_run_group_atlas")
fc = importlib.import_module("scripts.core_workflow.04_01_materialize_kong2022_roi_fc")


def _inputs(tmp_path: Path) -> dict:
    surface = tmp_path / "surface" / "sub-01" / "surface" / "2mm"
    reference = tmp_path / "reference" / "sub-01" / "reference" / "fixture"
    surface.mkdir(parents=True)
    reference.mkdir(parents=True)
    x = np.array([[-1, 1, -1, 1, 99, 101, 99, 101]], np.float32)
    y = np.array([[1, -1, 1, -1, 101, 99, 101, 99]], np.float32)
    np.save(surface / "sub-01_hemi-L_timeseries.npy", x)
    np.save(reference / "cortex_family17_network_timeseries.npy", y)
    (reference / "cortex_family17_networks.tsv").write_text("family17_network\nDefault\n")
    (reference / "reference_summary.json").write_text(json.dumps({"run_ids": [1, 2], "run_lengths": [4, 4]}))
    return dict(group_labels_final=np.array([1]), support_mask=np.ones(1, bool),
                subjects=["01"], shared_root=tmp_path, surface_root=tmp_path / "surface",
                reference_root=tmp_path / "reference", atlas="fixture", smoothing="2mm", hemi="L",
                valid_masks_by_subject={"01": np.ones(1, bool)}, profile_fc_mode="pearson",
                network_taxonomy="family17")


def test_group_network_profile_removes_declared_run_offsets(tmp_path):
    result = atlas.cluster_network_profile(**_inputs(tmp_path))
    assert result.item() < -5


def test_group_network_profile_preserves_unobserved_cluster(tmp_path):
    kwargs = _inputs(tmp_path)
    kwargs["support_mask"][:] = False
    result = atlas.cluster_network_profile(**kwargs)
    assert np.isnan(result).all()


def test_missing_network_never_wins_over_negative_observed_fc():
    rows, dominant = atlas.assignment_status_from_profile(np.array([[-.5, np.nan], [np.nan, np.nan]]), ["A", "B"])
    assert dominant == ["A", None]
    assert rows[1]["status"] == "unassigned"


def test_ctxdrop_default_methods_only_request_materialized_inputs():
    args = atlas.build_arg_parser().parse_args(["--route-tag", "routea_ctxdrop"])
    atlas._resolve_methods_and_taxonomy(args)
    assert set(args.methods) == {"gradient-spectral", "wta-top"}


def test_ctxdrop_rejects_explicit_unsupported_method():
    args = atlas.build_arg_parser().parse_args(["--route-tag", "routea_ctxdrop", "--methods", "group-roi-sim-spectral"])
    with pytest.raises(ValueError, match="ctxdrop"):
        atlas._resolve_methods_and_taxonomy(args)


def test_fc_run_boundaries_do_not_invent_missing_run_ids(tmp_path):
    ref = tmp_path / "reference" / "kong2022"
    ref.mkdir(parents=True)
    (ref / "reference_summary.json").write_text(json.dumps({"run_lengths": [4, 4]}))
    with pytest.raises(ValueError, match="run IDs"):
        fc.load_run_segments(tmp_path)


@pytest.mark.parametrize("subjectwise", [False, True])
def test_roi_cluster_profiles_center_runs_before_averaging(tmp_path, monkeypatch, subjectwise):
    kwargs = _inputs(tmp_path)
    ref = kwargs["reference_root"] / "sub-01" / "reference" / "fixture"
    network = np.load(ref / "cortex_family17_network_timeseries.npy")
    np.save(ref / "cortex_roi_parcel_timeseries.npy", np.repeat(network, 400, axis=0))
    (ref / "cortex_roi_parcels.tsv").write_text("fixture registry supplied below\n")
    registry = [dict(roi_id=i, roi_name=f"roi{i}", family17="Default", canonical8="Default") for i in range(1, 401)]
    rows = [dict(parcel_id=r["roi_id"], parcel_name=r["roi_name"], family17="Default", canonical8="Default") for r in registry]
    monkeypatch.setattr(atlas, "load_kong2022_roi_registry", lambda: registry)
    monkeypatch.setattr(atlas, "read_kong2022_roi_tsv", lambda path: rows)
    original = np.load(kwargs["surface_root"] / "sub-01/surface/2mm/sub-01_hemi-L_timeseries.npy")
    stepped = np.array([[50] * 4 + [1000] * 4], np.float32)
    np.save(kwargs["surface_root"] / "sub-01/surface/2mm/sub-01_hemi-L_timeseries.npy", np.vstack([original, stepped, -original]))
    kwargs.update(group_labels_final=np.array([1, 1, 2]), support_mask=np.ones(3, bool),
                  valid_masks_by_subject={"01": np.array([True, True, False])}, reference_atlas="fixture")
    for key in ("atlas", "profile_fc_mode", "network_taxonomy"):
        kwargs.pop(key)
    if subjectwise:
        stack, result, _, used = atlas.cluster_roi_profile_subjectwise(**kwargs)
        assert used == ["01"]
        assert np.isnan(stack[0, 1]).all()
    else:
        result, _ = atlas.cluster_roi_profile(**kwargs)
    assert np.all(result[0] < -5)
    assert np.isnan(result[1]).all()
