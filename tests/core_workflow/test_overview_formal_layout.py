#!/usr/bin/env python3
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
CORE_DIR = REPO_ROOT / "scripts" / "core_workflow"
if str(CORE_DIR) not in sys.path:
    sys.path.insert(0, str(CORE_DIR))


def _load_overview_module():
    path = CORE_DIR / "06_01_build_group_overview.py"
    spec = importlib.util.spec_from_file_location("group_overview_formal_layout", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load module: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


overview_mod = _load_overview_module()
plot_ovr_association_panel = overview_mod.plot_ovr_association_panel
plot_k_gate_panel = overview_mod.plot_k_gate_panel
plot_ari_summary_curve = overview_mod.plot_ari_summary_curve


def test_plot_ovr_association_panel_two_one_zero_betas():
    fig, ax = plt.subplots(figsize=(6, 4))
    # 3 clusters:
    # Cluster 0: two positive betas (0.4, 0.2)
    # Cluster 1: one positive beta (0.3, -0.1)
    # Cluster 2: zero positive betas (-0.2, -0.4)
    profile = np.array([
        [0.4, 0.2, -0.1],
        [0.3, -0.1, -0.2],
        [-0.2, -0.4, -0.3],
    ])
    network_names = ["Visual", "Somatomotor", "DorsalAttention"]
    row_positions = [0, 1, 2]
    row_labels = ["L-C1", "L-C2", "L-C3"]

    plot_ovr_association_panel(
        ax,
        profile=profile,
        row_positions=row_positions,
        row_labels=row_labels,
        smoothing="2mm",
        network_names=network_names,
    )

    # 3 y-ticks corresponding to the 3 clusters (none dropped)
    assert len(ax.get_yticks()) == 3
    labels = [t.get_text() for t in ax.get_yticklabels()]
    assert "L-C1 VN | SMN" in labels[0]
    assert "L-C2 VN" in labels[1]
    assert "L-C3" in labels[2]

    # Check for text in ax: "No positive β" should be present for L-C3
    texts = [t.get_text() for t in ax.texts]
    assert any("No positive β" in t for t in texts)
    assert any("Δβ=" in t for t in texts)
    plt.close(fig)


def test_plot_k_gate_panel_axis_and_threshold():
    fig, ax = plt.subplots(figsize=(6, 4))
    rows_l = [{"k": str(k), "max_profile_similarity": f"{0.6 + 0.05 * k:.2f}"} for k in range(2, 9)]
    rows_r = [{"k": str(k), "max_profile_similarity": f"{0.62 + 0.04 * k:.2f}"} for k in range(2, 9)]

    plot_k_gate_panel(
        ax,
        rows_l=rows_l,
        rows_r=rows_r,
        k_l=3,
        k_r=3,
        smoothing="2mm",
        redundancy_threshold=0.90,
    )

    ylim = ax.get_ylim()
    assert ylim[0] == pytest.approx(0.50)
    assert ylim[1] == pytest.approx(1.02)

    # Check ticks
    yticks = ax.get_yticks()
    assert 0.50 in yticks
    assert 0.90 in yticks
    assert 1.00 in yticks

    xticks = ax.get_xticks()
    assert list(xticks) == list(range(2, 9))
    plt.close(fig)


def test_plot_ari_summary_curve_k_range():
    fig, ax = plt.subplots(figsize=(6, 4))
    rows = [{"k": str(k), "group_ari_mean": "0.45", "group_ari_se": "0.03", "ARI_null_95pct": "0.05"} for k in range(2, 9)]

    ok = plot_ari_summary_curve(ax, rows, hemi="L", color="#1f77b4")
    assert ok is True
    xticks = ax.get_xticks()
    assert list(xticks) == list(range(2, 9))
    assert [line.get_label() for line in ax.lines] == ["hemi-L (mean±SE)", "hemi-L (null 95th%)"]
    plt.close(fig)


def test_plot_ari_summary_curve_requires_null_95th_percentile():
    fig, ax = plt.subplots(figsize=(6, 4))
    rows = [{"k": str(k), "group_ari_mean": "0.45", "group_ari_se": "0.03", "ARI_null_5pct": "0.05"} for k in range(2, 9)]
    with pytest.raises(ValueError, match="ARI_null_95pct.*rerun 05_02"):
        plot_ari_summary_curve(ax, rows, hemi="L", color="#1f77b4")
    plt.close(fig)


def test_directory_manifest_keeps_other_figures_rendered_by_earlier_calls(tmp_path):
    import json

    first, second, gone = (tmp_path / name for name in ("roi-fisherz.png", "top20.png", "removed.png"))
    first.write_bytes(b"a")
    second.write_bytes(b"b")
    manifest_path = tmp_path / "group_overview_manifest.json"
    manifest_path.write_text(json.dumps({"items": [
        {"overview_png": str(first), "assignment_variant": "roi-fisherz", "note": "old"},
        {"overview_png": str(gone), "assignment_variant": "stale"},
    ]}), encoding="utf-8")

    merged = overview_mod.merge_overview_manifest_items(
        manifest_path, [{"overview_png": str(second), "assignment_variant": "top20"}],
    )
    assert [item["assignment_variant"] for item in merged] == ["roi-fisherz", "top20"]

    replaced = overview_mod.merge_overview_manifest_items(
        manifest_path, [{"overview_png": str(first), "assignment_variant": "roi-fisherz", "note": "new"}],
    )
    assert [item.get("note") for item in replaced] == ["new"]


def _write_profile_run(root, hemi, profile, *, k=3, declare=True):
    run_dir = root / f"hemi_{hemi}"
    run_dir.mkdir(parents=True)
    profile_path = run_dir / "cluster_roi_profile.npy"
    np.save(profile_path, profile.astype(np.float32))
    manifest = {
        "status": "ok",
        "chosen_k_final": k,
        "cluster_roi_profile": str(profile_path.resolve()) if declare else None,
        "cluster_roi_profile_contract": "roi_ap",
        "cluster_label_ordering": "long_axis_head_to_tail",
        "cluster_order_head_to_tail": [1, 2, 3],
    }
    return run_dir / "group_atlas_manifest.json", manifest


def test_bilateral_cluster_roi_profile_is_the_left_right_mean(tmp_path):
    left = np.full((3, 400), 0.2)
    right = np.full((3, 400), 0.6)
    paths, manifests = zip(_write_profile_run(tmp_path, "L", left), _write_profile_run(tmp_path, "R", right))
    profile = overview_mod.bilateral_cluster_roi_profile(list(paths), list(manifests))
    assert profile.shape == (3, 400)
    np.testing.assert_allclose(profile, 0.4, rtol=1e-6)


@pytest.mark.parametrize("override", [{"k": 4}, {"declare": False}])
def test_bilateral_cluster_roi_profile_is_absent_without_a_declared_three_cluster_profile(tmp_path, override):
    paths, manifests = zip(
        _write_profile_run(tmp_path, "L", np.zeros((3, 400))),
        _write_profile_run(tmp_path, "R", np.zeros((3, 400)), **override),
    )
    assert overview_mod.bilateral_cluster_roi_profile(list(paths), list(manifests)) is None


def test_bilateral_cluster_roi_profile_rejects_a_profile_outside_the_contract(tmp_path):
    paths, manifests = zip(
        _write_profile_run(tmp_path, "L", np.zeros((3, 400))),
        _write_profile_run(tmp_path, "R", np.zeros((3, 399))),
    )
    with pytest.raises(ValueError, match="shape"):
        overview_mod.bilateral_cluster_roi_profile(list(paths), list(manifests))


@pytest.mark.parametrize(("bars", "size"), [("below", (40, 68)), ("right", (88, 30))])
def test_cortex_tile_places_the_bars_below_or_beside_the_surface(tmp_path, bars, size):
    from PIL import Image

    from lib.cortex_fc_tiles import join_surface_and_bars

    Image.new("RGBA", (40, 30), "red").save(tmp_path / "surface.png")
    Image.new("RGBA", (40, 30), "blue").save(tmp_path / "bars.png")
    tile = join_surface_and_bars(tmp_path / "surface.png", tmp_path / "bars.png", tmp_path / "tile.png", bars=bars)
    with Image.open(tile) as image:
        assert image.size == size
        assert image.getpixel((0, 0))[:3] == (255, 0, 0)
        assert image.getpixel((size[0] - 1, size[1] - 1))[:3] == (0, 0, 255)
