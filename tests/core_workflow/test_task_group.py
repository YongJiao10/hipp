#!/usr/bin/env python3
from __future__ import annotations

import json
import importlib
import sys
from pathlib import Path

import numpy as np
import pytest
from scipy.stats import ttest_1samp

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
CORE_DIR = REPO_ROOT / "scripts" / "core_workflow"
if str(CORE_DIR) not in sys.path:
    sys.path.insert(0, str(CORE_DIR))

from scripts.core_workflow.lib.task_group import (
    anatomical_cluster_names,
    group_summary_stats,
    load_subject_stack,
    subject_cluster_rows,
)


def test_bin_statistics_use_subject_means_and_effective_sample_size() -> None:
    from scripts.core_workflow.lib import task_group

    stack = np.array([
        [1., 3., 8., 10., np.nan, np.nan],
        [2., 4., 9., 11., 7., np.nan],
        [3., 5., 10., 12., np.nan, np.nan],
        [4., 6., 11., 13., np.nan, np.nan],
    ])
    bins = [np.array([0, 1]), np.array([2, 3]), np.array([4, 5])]
    result = task_group.group_bin_stats(stack, bins)
    expected = ttest_1samp(np.array([[2., 9.], [3., 10.], [4., 11.], [5., 12.]]), 0, axis=0)
    np.testing.assert_array_equal(result['n'], [4, 4, 1])
    np.testing.assert_allclose(result['mean'], [3.5, 10.5, 7.])
    np.testing.assert_allclose(result['tstat'][:2], expected.statistic)
    np.testing.assert_allclose(result['p'][:2], expected.pvalue)
    assert np.isnan(result['tstat'][2]) and np.isnan(result['p'][2])


def test_bin_statistics_do_not_count_vertices_as_independent_subjects() -> None:
    from scripts.core_workflow.lib import task_group

    stack = np.array([[1., 2.], [2., 3.], [3., 4.], [4., 5.]])
    original = task_group.group_bin_stats(stack, [np.arange(2)])
    repeated = task_group.group_bin_stats(np.tile(stack, (1, 20)), [np.arange(40)])
    np.testing.assert_array_equal(repeated['n'], [4])
    np.testing.assert_allclose(repeated['tstat'], original['tstat'])
    np.testing.assert_allclose(repeated['p'], original['p'])


def _write_contrast(shared: Path, subject: str, values: np.ndarray) -> None:
    out_dir = shared / f"sub-{subject}" / "task" / "emotion" / "2mm" / "hemi_L"
    out_dir.mkdir(parents=True)
    np.save(str(out_dir / "contrast_faces_gt_shapes.npy"), values.astype(np.float32))


def test_group_cli_requires_an_explicit_complete_roster(tmp_path, monkeypatch):
    group = importlib.import_module("scripts.core_workflow.05_06_build_task_activation_group")
    output = tmp_path / "group"
    monkeypatch.setattr(sys, "argv", [
        "group", "--task", "emotion", "--shared-root", str(tmp_path / "shared"),
        "--out-root", str(output),
    ])
    with pytest.raises(SystemExit) as exc:
        group.main()
    assert exc.value.code == 2
    assert not output.exists()


def test_group_cli_requires_all_maintained_contrasts_for_the_roster(tmp_path, monkeypatch):
    group = importlib.import_module("scripts.core_workflow.05_06_build_task_activation_group")
    shared, output = tmp_path / "shared", tmp_path / "group"
    for subject in ["001", "002"]:
        directory = shared / f"sub-{subject}/task/wm/2mm/hemi_L"
        directory.mkdir(parents=True)
        np.save(directory / "task_betas.npy", np.zeros((8, 8), dtype=np.float32))
        np.save(directory / "contrast_twobk_gt_zerobk.npy", np.zeros(8, dtype=np.float32))
    monkeypatch.setattr(group, "load_group_labels", lambda *a, **k: (
        np.ones(8, dtype=np.int32), np.ones(8, dtype=bool), {},
    ))
    monkeypatch.setattr(group, "group_average_surface", lambda *a: tmp_path / "surface.gii")
    monkeypatch.setattr(group, "anatomical_cluster_names", lambda **k: {1: "Anterior", 2: "Middle", 3: "Posterior"})
    monkeypatch.setattr(sys, "argv", [
        "group", "--task", "wm", "--shared-root", str(shared),
        "--subjects", "001", "002", "--smoothings", "2mm", "--hemis", "L",
        "--out-root", str(output),
    ])
    with pytest.raises(ValueError, match="roster is incomplete"):
        group.main()
    assert not output.exists()


def test_load_stack_fails_closed_on_shape_drift(tmp_path: Path) -> None:
    shared = tmp_path / "shared"
    _write_contrast(shared, "001", np.zeros(8))
    _write_contrast(shared, "002", np.zeros(7))
    with pytest.raises(ValueError):
        load_subject_stack(
            shared,
            subjects=["001", "002"],
            task="emotion",
            smoothing="2mm",
            hemi="L",
            contrast="faces_gt_shapes",
        )
    with pytest.raises(FileNotFoundError):
        load_subject_stack(
            shared,
            subjects=["001", "ghost"],
            task="emotion",
            smoothing="2mm",
            hemi="L",
            contrast="faces_gt_shapes",
        )


def test_group_stats_recover_known_effect() -> None:
    rng = np.random.default_rng(3)
    truth = np.linspace(-1.0, 1.0, 16)
    stack = truth[None, :] + 0.05 * rng.normal(size=(12, 16))
    stats = group_summary_stats(stack)
    np.testing.assert_allclose(stats["mean"], truth, atol=0.1)
    assert np.nanmean(np.abs(stats["tstat"])) > 5.0
    with pytest.raises(ValueError):
        group_summary_stats(stack[:1])


def test_subject_cluster_rows_and_anatomical_names(tmp_path: Path) -> None:
    labels = np.array([0, 1, 1, 2, 2, 3, 3, 1])
    mask = np.array([False, True, True, True, True, True, True, True])
    manifest = {"cluster_order_head_to_tail": [1, 2, 3]}
    surf = tmp_path / "surf.surf.gii"
    surf.write_bytes(b"placeholder")
    import scripts.core_workflow.lib.task_group as task_group

    calls: dict[str, Path] = {}

    def fake_axis(path: Path) -> np.ndarray:
        calls["path"] = path
        return np.arange(8, dtype=np.int32)

    task_group.long_axis_order = fake_axis  # type: ignore[assignment]
    try:
        names = anatomical_cluster_names(
            manifest=manifest, group_labels=labels, surface_path=surf
        )
    finally:
        import importlib

        importlib.reload(task_group)
    assert names == {1: "Anterior", 2: "Middle", 3: "Posterior"}
    assert calls["path"] == surf
    stack = np.arange(16, dtype=np.float64).reshape(2, 8)
    areas = {
        "a": np.array([1.0, 2.0, 1.0, 1.0, 1.0, 1.0, 1.0, 10.0]),
        "b": np.ones(8),
    }
    rows = subject_cluster_rows(
        stack,
        ["a", "b"],
        group_labels=labels,
        support_mask=mask,
        names_by_id=names,
        vertex_areas=areas,
    )
    assert len(rows) == 6
    anterior = [r for r in rows if r["cluster_name"] == "Anterior" and r["subject"] == "sub-a"]
    # Anterior vertices (idx 1,2,7) carry areas (2,1,10): weighted mean only.
    assert anterior[0]["mean"] == pytest.approx((1.0 * 2 + 2.0 * 1 + 7.0 * 10) / 13.0)
    assert anterior[0]["mean"] != pytest.approx((1.0 + 2.0 + 7.0) / 3.0)
    assert anterior[0]["n_vertices"] == 3
    assert anterior[0]["area_mm2"] == pytest.approx(13.0)
    assert np.isfinite(anterior[0]["sd"])
    uniform = [r for r in rows if r["cluster_name"] == "Anterior" and r["subject"] == "sub-b"]
    assert uniform[0]["mean"] == pytest.approx((9.0 + 10.0 + 15.0) / 3.0)


def test_subject_cluster_rows_require_matching_areas() -> None:
    labels = np.array([1, 1, 2, 2])
    mask = np.array([True, True, True, True])
    with pytest.raises(ValueError):
        subject_cluster_rows(
            np.zeros((1, 4)),
            ["a"],
            group_labels=labels,
            support_mask=mask,
            names_by_id={1: "Anterior", 2: "Middle"},
            vertex_areas={"a": np.ones(5)},
        )


def test_parcellation_zone_names() -> None:
    from scripts.core_workflow.lib.task_group import (
        PARCELLATION_CHOICES,
        PARCELLATION_HCP151_ATLAS,
        PARCELLATION_TEMPLATE_MATCHING,
    )

    assert PARCELLATION_CHOICES == (
        PARCELLATION_HCP151_ATLAS,
        PARCELLATION_TEMPLATE_MATCHING,
    )
    assert PARCELLATION_HCP151_ATLAS == "hcp151_atlas"
    assert PARCELLATION_TEMPLATE_MATCHING == "template_matching"


def test_resolve_zone_subjects_excludes_missing_only(tmp_path: Path) -> None:
    from scripts.core_workflow.lib.task_group import resolve_zone_subjects

    crossfit = tmp_path / "crossfit"
    labels_dir = crossfit / "sub-a" / "2mm" / "hemi_L"
    labels_dir.mkdir(parents=True)
    np.save(str(labels_dir / "individual_labels_k3.npy"), np.array([0, 1, 2, 3, 1, 2]))
    kept, notes = resolve_zone_subjects(
        zone="template_matching",
        used=["a", "b"],
        smoothing="2mm",
        hemi="L",
        crossfit_root=crossfit,
    )
    # No spatial validation: a present file is always kept.
    assert kept == ["a"]
    assert notes["a"] == "individual K=3 labels"
    assert "excluded" in notes["b"]
    kept_all, _ = resolve_zone_subjects(
        zone="hcp151_atlas",
        used=["a", "b"],
        smoothing="2mm",
        hemi="L",
        crossfit_root=crossfit,
    )
    assert kept_all == ["a", "b"]
    with pytest.raises(ValueError):
        resolve_zone_subjects(
            zone="bogus", used=["a"], smoothing="2mm", hemi="L", crossfit_root=crossfit
        )


def test_anatomical_names_reject_inconsistent_order(tmp_path: Path) -> None:
    labels = np.array([1, 1, 2, 2])
    manifest = {"cluster_order_head_to_tail": [2, 1]}
    surf = tmp_path / "surf.surf.gii"
    surf.write_bytes(b"x")
    import scripts.core_workflow.lib.task_group as task_group

    def fake_axis(path: Path) -> np.ndarray:
        return np.arange(4, dtype=np.int32)

    task_group.long_axis_order = fake_axis  # type: ignore[assignment]
    try:
        with pytest.raises(ValueError):
            anatomical_cluster_names(
                manifest=manifest, group_labels=labels, surface_path=surf
            )
    finally:
        import importlib

        importlib.reload(task_group)


def test_load_subject_betas_validates_regressor_order(tmp_path: Path) -> None:
    from scripts.core_workflow.lib.task_group import load_subject_betas

    shared = tmp_path / "shared"
    for subject, names in (("001", ["fear", "neut"]), ("002", ["fear", "neut"])):
        task_dir = shared / f"sub-{subject}" / "task" / "emotion" / "2mm" / "hemi_L"
        task_dir.mkdir(parents=True)
        np.save(str(task_dir / "task_betas.npy"), np.zeros((6, 2), dtype=np.float32))
        (task_dir / "task_activation_summary.json").write_text(
            json.dumps({"regressor_names": names}), encoding="utf-8"
        )
    stacks, stems = load_subject_betas(
        shared, subjects=["001", "002"], task="emotion", smoothing="2mm", hemi="L"
    )
    assert stems == ["fear", "neut"]
    assert stacks["fear"].shape == (2, 6)
    bad_dir = shared / "sub-003" / "task" / "emotion" / "2mm" / "hemi_L"
    bad_dir.mkdir(parents=True)
    np.save(str(bad_dir / "task_betas.npy"), np.zeros((6, 2), dtype=np.float32))
    (bad_dir / "task_activation_summary.json").write_text(
        json.dumps({"regressor_names": ["neut", "fear"]}), encoding="utf-8"
    )
    with pytest.raises(ValueError):
        load_subject_betas(
            shared, subjects=["001", "003"], task="emotion", smoothing="2mm", hemi="L"
        )


def test_two_tailed_p_and_sig_marker() -> None:
    from scripts.core_workflow.lib.task_group import sig_marker, two_tailed_p

    assert two_tailed_p(4.5, 25) == pytest.approx(0.00015, rel=0.05)
    assert np.isnan(two_tailed_p(float("nan"), 25))
    assert np.isnan(two_tailed_p(2.0, 1))
    assert sig_marker(0.03) == "+"
    assert sig_marker(0.005) == "*"
    assert sig_marker(0.0005) == "**"
    assert sig_marker(0.00001) == "***"
    assert sig_marker(0.5) == ""
    assert sig_marker(float("nan")) == ""


def test_fdr_adjust() -> None:
    from scripts.core_workflow.lib.task_group import fdr_adjust

    assert fdr_adjust([0.001, 0.5, 0.02]) == pytest.approx(
        [0.003, 0.5, 0.03], rel=0.01
    )


def test_corrcoef_rows_nan_propagates_missing() -> None:
    from scripts.core_workflow.lib.task_glm import corrcoef_rows_nan

    rng = np.random.default_rng(0)
    x = rng.normal(size=(4, 50))
    y = rng.normal(size=(3, 50))
    full = corrcoef_rows_nan(x, y)
    assert full.shape == (4, 3)
    assert np.all(np.isfinite(full))
    ref = np.corrcoef(x[0], y[0])[0, 1]
    assert abs(full[0, 0] - ref) < 1e-6
    x_nan = x.copy()
    x_nan[2, :] = np.nan
    partial = corrcoef_rows_nan(x_nan, y)
    assert np.all(np.isnan(partial[2, :]))
    assert np.all(np.isfinite(np.delete(partial, 2, axis=0)))


def test_task_runs_from_manifest(tmp_path: Path) -> None:
    from scripts.core_workflow.lib.task_manifests import (
        build_task_run_artifact_manifest,
        task_runs_from_manifest,
    )

    data_root = tmp_path / "HCP_tast_3T"
    for label in ("tfMRI_EMOTION_LR", "tfMRI_EMOTION_RL"):
        run_dir = data_root / "sub-115017" / "func" / label
        run_dir.mkdir(parents=True)
        (run_dir / f"{label}_hp0_clean_rclean_tclean.nii.gz").write_bytes(b"v")
        (run_dir / f"{label}_Atlas_MSMAll_hp0_clean_rclean_tclean.dtseries.nii").write_bytes(b"d")
    manifest_root = tmp_path / "manifests"
    out = build_task_run_artifact_manifest(
        subject="115017", task="emotion", task_data_root=data_root, manifest_root=manifest_root,
    )
    runs = task_runs_from_manifest(out, subject="115017")
    assert [(r["run_id"], r["run_label"]) for r in runs] == [(1, "tfMRI_EMOTION_LR"), (2, "tfMRI_EMOTION_RL")]


def test_subject_cluster_rows_coverage_gate() -> None:
    labels = np.array([1, 1, 1, 1])
    mask = np.array([True, True, True, True])
    names = {1: "Anterior"}
    areas = {"a": np.ones(4)}
    # Only one of four vertices finite -> valid fraction 0.25.
    stack = np.array([[5.0, np.nan, np.nan, np.nan]])
    gated = subject_cluster_rows(
        stack, ["a"], group_labels=labels, support_mask=mask,
        names_by_id=names, vertex_areas=areas, min_valid_area_fraction=0.5,
    )
    assert gated[0]["excluded_low_coverage"] is True
    assert not np.isfinite(gated[0]["mean"])
    assert gated[0]["valid_area_fraction"] == pytest.approx(0.25)
    kept = subject_cluster_rows(
        stack, ["a"], group_labels=labels, support_mask=mask,
        names_by_id=names, vertex_areas=areas, min_valid_area_fraction=0.25,
    )
    assert kept[0]["excluded_low_coverage"] is False
    assert kept[0]["mean"] == pytest.approx(5.0)
    assert kept[0]["area_total_mm2"] == pytest.approx(4.0)
    with pytest.raises(ValueError):
        subject_cluster_rows(
            stack, ["a"], group_labels=labels, support_mask=mask,
            names_by_id=names, vertex_areas=areas, min_valid_area_fraction=0.0,
        )
    with pytest.raises(ValueError):
        subject_cluster_rows(
            stack, ["a"], group_labels=labels, support_mask=mask,
            names_by_id=names, vertex_areas=areas, min_valid_area_fraction=1.5,
        )
