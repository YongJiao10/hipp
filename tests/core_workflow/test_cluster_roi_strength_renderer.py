# pyright: reportAttributeAccessIssue=false, reportArgumentType=false
from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import nibabel as nib
import numpy as np
import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
MODULE_PATH = REPO_ROOT / "scripts" / "core_workflow" / "06_10_render_cluster_roi_strength_panels.py"
SPEC = importlib.util.spec_from_file_location("cluster_roi_strength_renderer", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
renderer = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(renderer)


def test_global_symmetric_scale_uses_all_finite_template_values() -> None:
    profiles = [
        np.asarray([[0.1, -0.8], [np.nan, 0.3]], dtype=np.float32),
        np.asarray([[1.25, -0.2], [0.4, np.nan]], dtype=np.float32),
    ]

    assert renderer.compute_global_symmetric_scale(profiles) == pytest.approx(1.25)


def test_strength_colors_are_blue_white_red_and_nan_gray() -> None:
    values = np.asarray([-1.0, 0.0, 1.0, np.nan], dtype=np.float32)

    colors = renderer.strength_colors(values, vmax=1.0)

    assert colors.shape == (4, 4)
    assert colors[0, 2] > colors[0, 0]
    assert np.allclose(colors[1, :3], [1.0, 1.0, 1.0], atol=0.02)
    assert colors[2, 0] > colors[2, 2]
    assert np.allclose(colors[3, :3], [0.65, 0.65, 0.65], atol=0.02)


def test_cluster_names_follow_verified_head_to_tail_order() -> None:
    labels = np.asarray([1, 1, 2, 2, 3, 3], dtype=np.int32)
    long_axis_order = np.arange(labels.size, dtype=np.int32)
    manifest = {"cluster_order_head_to_tail": [1, 2, 3]}

    names = renderer.resolve_cluster_names(
        manifest=manifest,
        group_labels=labels,
        long_axis_order=long_axis_order,
    )

    assert names == {1: "Anterior", 2: "Body", 3: "Posterior"}


def test_cluster_names_reject_manifest_order_that_disagrees_with_long_axis() -> None:
    labels = np.asarray([1, 1, 2, 2, 3, 3], dtype=np.int32)
    long_axis_order = np.arange(labels.size, dtype=np.int32)
    manifest = {"cluster_order_head_to_tail": [3, 2, 1]}

    with pytest.raises(ValueError, match="head-to-tail"):
        renderer.resolve_cluster_names(
            manifest=manifest,
            group_labels=labels,
            long_axis_order=long_axis_order,
        )


def test_panel_order_is_smoothing_then_hemi_then_anatomical_cluster() -> None:
    order = renderer.panel_order(
        smoothings=("2mm", "4mm"),
        hemis=("L", "R"),
        cluster_names=("Anterior", "Body", "Posterior"),
    )

    assert order == [
        ("2mm", "L", "Anterior"),
        ("2mm", "L", "Body"),
        ("2mm", "L", "Posterior"),
        ("2mm", "R", "Anterior"),
        ("2mm", "R", "Body"),
        ("2mm", "R", "Posterior"),
        ("4mm", "L", "Anterior"),
        ("4mm", "L", "Body"),
        ("4mm", "L", "Posterior"),
        ("4mm", "R", "Anterior"),
        ("4mm", "R", "Body"),
        ("4mm", "R", "Posterior"),
    ]


def test_validate_template_profile_requires_400_parcels() -> None:
    with pytest.raises(ValueError, match="K x 400"):
        renderer.validate_template_profile(np.zeros((3, 399), dtype=np.float32), Path("profile.npy"))


def test_blank_opposite_hemi_label_keeps_surface_vertex_count(tmp_path: Path) -> None:
    registry = [
        {
            "roi_id": index,
            "roi_name": f"ROI-{index:03d}",
            "hemi": "L" if index <= 200 else "R",
        }
        for index in range(1, 401)
    ]
    vertex_arrays = {
        "L": (np.asarray([0, 1], dtype=np.int32), np.asarray([1, 2], dtype=np.int32)),
        "R": (np.asarray([0, 1, 2], dtype=np.int32), np.asarray([3, 4, 5], dtype=np.int32)),
    }
    output = tmp_path / "blank.R.label.gii"

    renderer.write_strength_label(
        output_path=output,
        hemi="R",
        include_values=False,
        values=np.zeros(400, dtype=np.float32),
        vmax=1.0,
        roi_registry=registry,
        vertex_arrays=vertex_arrays,
    )

    data = nib.load(str(output)).darrays[0].data
    assert data.shape == (3,)
    assert np.all(data == 0)


def test_template_surface_path_rejects_legacy_fallback_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    canonical_root = tmp_path / "hipp_group_average"
    canonical_surface = (
        canonical_root
        / "group_average_surfaces"
        / "group-average_hemi-L_space-corobl_den-512_label-hipp_midthickness.surf.gii"
    )
    canonical_surface.parent.mkdir(parents=True)
    canonical_surface.write_bytes(b"canonical")
    monkeypatch.setattr(
        renderer.validate_template_surface_path.__globals__["resolve_kong2022_template_root"],
        "__code__",
        (lambda dataset="hcp", variant="36p": canonical_root).__code__,
    ) if False else monkeypatch.setitem(
        renderer.validate_template_surface_path.__globals__,
        "resolve_kong2022_template_root",
        lambda dataset="hcp", variant="36p": canonical_root,
    )
    with pytest.raises(ValueError, match="not canonical"):
        renderer.validate_template_surface_path(
            Path("/old/hcp7t_hippocampus/group-average_hemi-L_space-corobl_den-512_label-hipp_midthickness.surf.gii"),
            dataset="hcp",
            hemi="L",
        )


def test_ordered_cluster_names_preserve_verified_anatomical_rank() -> None:
    names_by_id = {3: "Anterior", 1: "Body", 2: "Posterior"}

    assert renderer.ordered_cluster_names(names_by_id) == (
        "Anterior",
        "Body",
        "Posterior",
    )


def test_render_panel_colors_both_cortical_hemispheres_for_each_source_hemi(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    registry = [
        {
            "roi_id": index,
            "roi_name": f"ROI-{index:03d}",
            "hemi": "L" if index <= 200 else "R",
        }
        for index in range(1, 401)
    ]
    vertex_arrays = {
        "L": (np.asarray([0, 1], dtype=np.int32), np.asarray([1, 2], dtype=np.int32)),
        "R": (np.asarray([0, 1], dtype=np.int32), np.asarray([201, 202], dtype=np.int32)),
    }

    def fake_render(**kwargs):
        out_png = kwargs["out_png"]
        out_png.parent.mkdir(parents=True, exist_ok=True)
        out_png.write_bytes(b"png")
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(renderer, "render_cortex_pfm_scene", fake_render)
    panel_png = renderer.render_panel(
        runtime_root=tmp_path,
        panel_key=("2mm", "L", "Anterior"),
        cluster_id=1,
        values=np.zeros(400, dtype=np.float32),
        vmax=1.0,
        roi_registry=registry,
        vertex_arrays=vertex_arrays,
        cortex_assets={key: tmp_path / f"{key}.asset" for key in (
            "scene",
            "left_inflated",
            "right_inflated",
            "left_sulc",
            "right_sulc",
            "sulc_dscalar",
            "dlabel",
        )},
    )

    assert panel_png.exists()
    panel_dir = tmp_path / "panels" / "2mm_L_anterior"
    left_values = nib.load(str(panel_dir / "strength_2mm_L_anterior.L.label.gii")).darrays[0].data
    right_values = nib.load(str(panel_dir / "strength_2mm_L_anterior.R.label.gii")).darrays[0].data
    assert np.count_nonzero(left_values) == 2
    assert np.count_nonzero(right_values) == 2


def test_overview_text_is_compact_and_uses_mean_fisher_z_title() -> None:
    assert renderer.overview_title() == "HCP 7T | Kong2022 400 ROIs | gradient-spectral"
    assert renderer.overview_title("wta-top", "canonical7") == "HCP 7T | Kong2022 400 ROIs | WTA normalized | 7-network"
    for taxonomy in ("canonical8", "legacy"):
        with pytest.raises(ValueError, match="canonical7"):
            renderer.overview_title("wta-top", taxonomy)
    assert renderer.overview_row_label("2mm", "L") == "2mm | L"
    assert "source hippocampus" not in renderer.overview_row_label("4mm", "R")


def test_wta_top_network_selection_is_cohort_wide_and_deterministic() -> None:
    runs = [
        {"status_items": [{"top1_network": "Default"}] * 3 + [{"top1_network": "Control"}]},
        {"status_items": [{"top1_network": "Default"}] * 2 + [{"top1_network": "Control"}] * 2},
        {"status_items": [{"top1_network": "Auditory"}] * 2 + [{"top1_network": "Somatomotor"}]},
    ]

    names, counts = renderer.select_top_wta_networks(runs, n_networks=3)

    assert names == ("Default", "Control", "Auditory")
    assert counts == {"Default": 5, "Control": 3, "Auditory": 2}


def test_wta_pseudocluster_labels_keep_only_selected_networks() -> None:
    run = {
        "support_mask": np.asarray([1, 1, 1, 0, 1], dtype=bool),
        "network_by_vertex": {
            0: "Default",
            1: "Control",
            2: "Other",
            3: "Default",
            4: "Control",
        },
    }

    labels = renderer.build_wta_pseudocluster_labels(run, ("Default", "Control"))

    assert labels.tolist() == [1, 2, 0, 0, 2]


def test_render_panel_invalidates_stale_png_when_label_content_changes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    registry = [
        {
            "roi_id": index,
            "roi_name": f"ROI-{index:03d}",
            "hemi": "L" if index <= 200 else "R",
        }
        for index in range(1, 401)
    ]
    vertex_arrays = {
        "L": (np.asarray([0, 1], dtype=np.int32), np.asarray([1, 2], dtype=np.int32)),
        "R": (np.asarray([0, 1], dtype=np.int32), np.asarray([201, 202], dtype=np.int32)),
    }
    calls: list[dict[str, object]] = []

    def fake_render(**kwargs):
        calls.append(kwargs)
        out_png = kwargs["out_png"]
        out_png.parent.mkdir(parents=True, exist_ok=True)
        out_png.write_bytes(b"png")
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(renderer, "render_cortex_pfm_scene", fake_render)
    assets = {key: tmp_path / f"{key}.asset" for key in (
        "scene",
        "left_inflated",
        "right_inflated",
        "left_sulc",
        "right_sulc",
        "sulc_dscalar",
        "dlabel",
    )}
    renderer.render_panel(
        runtime_root=tmp_path,
        panel_key=("2mm", "L", "Anterior"),
        cluster_id=1,
        values=np.zeros(400, dtype=np.float32),
        vmax=1.0,
        roi_registry=registry,
        vertex_arrays=vertex_arrays,
        cortex_assets=assets,
    )
    renderer.render_panel(
        runtime_root=tmp_path,
        panel_key=("2mm", "L", "Anterior"),
        cluster_id=1,
        values=np.concatenate(([1.0], np.zeros(399, dtype=np.float32))),
        vmax=1.0,
        roi_registry=registry,
        vertex_arrays=vertex_arrays,
        cortex_assets=assets,
    )

    assert len(calls) == 2
