#!/usr/bin/env python3
from __future__ import annotations

import importlib
import csv
import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
CORE_DIR = REPO_ROOT / "scripts" / "core_workflow"
if str(CORE_DIR) not in sys.path:
    sys.path.insert(0, str(CORE_DIR))

render = importlib.import_module(
    "scripts.core_workflow.06_11_render_task_activation_products"
)

LOAD_X_PLACES_WEIGHTS = {
    "0bk_faces": 1.0 / 6.0,
    "0bk_places": -0.5,
    "0bk_tools": 1.0 / 6.0,
    "0bk_body": 1.0 / 6.0,
    "2bk_faces": -1.0 / 6.0,
    "2bk_places": 0.5,
    "2bk_tools": -1.0 / 6.0,
    "2bk_body": -1.0 / 6.0,
}


@pytest.fixture
def bin_summary() -> dict:
    return {
        ('L', effect): {'mean': np.ones(20), 'q': np.ones(20), 'n': np.full(20, 6)}
        for effect in ('faces_gt_shapes', 'beta_fear', 'beta_neut')
    }


def test_bin_band_aligns_bins_merges_equal_levels_and_keeps_curve_scale() -> None:
    fig, ax = render.plt.subplots()
    try:
        ax.set_xlim(.5, 8.5)
        ax.set_ylim(-50., 50.)
        corrected = np.array([.2, .049, .04, .005, .0005, .00005, .2, np.nan])
        render.draw_bin_band(ax, corrected)
        assert not ax.texts
        assert ax.get_ylim() == (-50., 50.)
        np.testing.assert_allclose([p.get_x() for p in ax.patches], [.5, 1.5, 3.5, 4.5, 5.5, 6.5])
        np.testing.assert_allclose([p.get_width() for p in ax.patches], [1, 2, 1, 1, 1, 2])
        brightness = [np.mean(p.get_facecolor()[:3]) for p in ax.patches]
        assert all(a > b for a, b in zip(brightness[:5], brightness[1:5]))
        assert brightness[0] == brightness[-1]
        fig.canvas.draw()
        height = ax.patches[1].get_window_extent().height
        ax.set_ylim(-500., 500.)
        fig.canvas.draw()
        assert ax.patches[1].get_window_extent().height == pytest.approx(height)
    finally:
        render.plt.close(fig)


def test_bin_summary_requires_exact_twenty_bins(tmp_path: Path) -> None:
    path = tmp_path / 'group_bin_summary.tsv'
    fields = ['smoothing', 'hemi', 'contrast', 'bin', 'n_subjects', 'group_mean', 'group_p_fdr']
    rows = [dict(smoothing='2mm', hemi='L', contrast='faces_gt_shapes', bin=i,
                 n_subjects=4, group_mean=float(i), group_p_fdr=.04) for i in range(20, 0, -1)]
    with path.open('w') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, delimiter='\t')
        writer.writeheader()
        writer.writerows(rows)
    profiles = render.read_bin_summary(path)
    np.testing.assert_allclose(profiles['L', 'faces_gt_shapes']['mean'], np.arange(1, 21))
    with path.open('a') as stream:
        csv.DictWriter(stream, fieldnames=fields, delimiter='\t').writerow(rows[-1])
    with pytest.raises(ValueError, match='bins'):
        render.read_bin_summary(path)


def test_contrast_side_weights_sum_to_one() -> None:
    for contrast in ("faces_gt_shapes", "twobk_gt_zerobk", "load_x_places"):
        pos, neg = render.contrast_side_weights("wm" if "bk" in contrast or "places" in contrast else "emotion", contrast)
        assert sum(pos.values()) == pytest.approx(1.0)
        assert sum(neg.values()) == pytest.approx(1.0)


def test_weighted_sides_reconstruct_load_x_places(tmp_path: Path) -> None:
    group_root = tmp_path / "group"
    combo = group_root / "2mm" / "hemi_L"
    combo.mkdir(parents=True)
    rng = np.random.default_rng(0)
    betas = {stem: rng.normal(size=8) for stem in LOAD_X_PLACES_WEIGHTS}
    for stem, values in betas.items():
        np.save(str(combo / f"group_beta_{stem}_mean.npy"), values)
    pos_w, neg_w = render.contrast_side_weights("wm", "load_x_places")
    pos_map = render.weighted_side_map(group_root, weights=pos_w, hemi="L")
    neg_map = render.weighted_side_map(group_root, weights=neg_w, hemi="L")
    truth = sum(w * betas[s] for s, w in LOAD_X_PLACES_WEIGHTS.items())
    np.testing.assert_allclose(pos_map - neg_map, truth, rtol=1e-6)
    # The reviewer's counterexample: only 2bk_places = 1 -> contrast 0.5.
    only = dict.fromkeys(LOAD_X_PLACES_WEIGHTS, 0.0)
    only["2bk_places"] = 1.0
    for stem, value in only.items():
        np.save(str(combo / f"group_beta_{stem}_mean.npy"), np.full(8, value))
    pos_map = render.weighted_side_map(group_root, weights=pos_w, hemi="L")
    neg_map = render.weighted_side_map(group_root, weights=neg_w, hemi="L")
    np.testing.assert_allclose(pos_map - neg_map, np.full(8, 0.5), rtol=1e-6)


def test_parse_bh_fdr_p_requires_adjusted_value() -> None:
    row = {"parcellation": "hcp_atlas", "cluster_name": "Middle",
           "contrast": "c", "hemi": "L", "group_p_fdr": "0.004",
           "group_p": "0.001"}
    assert render.parse_bh_fdr_p(row, contrast="c", hemi="L") == pytest.approx(0.004)
    for bad in ({"group_p": "0.001"}, {"group_p_fdr": ""}, {"group_p_fdr": "  "}):
        tagged = dict(bad)
        tagged.update({"parcellation": "hcp_atlas", "cluster_name": "Middle"})
        with pytest.raises(ValueError, match="[Ff][Dd][Rr]"):
            render.parse_bh_fdr_p(tagged, contrast="c", hemi="L")


def test_weighted_side_preserves_missing_conditions(tmp_path: Path) -> None:
    combo = tmp_path / "2mm" / "hemi_L"
    combo.mkdir(parents=True)
    np.save(combo / "group_beta_a_mean.npy", [1., 2., np.nan, 4.])
    np.save(combo / "group_beta_b_mean.npy", [np.nan, 6., np.nan, np.inf])
    actual = render.weighted_side_map(tmp_path, weights={"a": .5, "b": .5}, hemi="L")
    np.testing.assert_allclose(actual, [np.nan, 4., np.nan, np.nan], equal_nan=True)


def test_cluster_plot_labels_effective_counts(monkeypatch: pytest.MonkeyPatch, bin_summary: dict) -> None:
    monkeypatch.setattr(render, "group_roster", lambda *a, **kw: ["a", "b", "c"])
    monkeypatch.setattr(render, "long_axis_order", lambda *a: np.arange(40, dtype=np.int32))
    monkeypatch.setattr(render, "equal_vertex_long_axis_bins",
                        lambda *a: [np.array([2 * i, 2 * i + 1]) for i in range(20)])
    monkeypatch.setattr(render, "group_beta_mean", lambda *a, **kw: np.ones(40))
    # Surface/map IO is unrelated to the effective-N behavior under test.
    monkeypatch.setattr(render.np, "load", lambda *a, **kw: np.ones(40))
    summary, subjects = [], []
    for zone in render.ZONE_ORDER:
        for cluster in render.CLUSTER_ORDER:
            n = 2 if cluster == "Middle" else 3
            summary.append(dict(parcellation=zone, contrast="faces_gt_shapes", hemi="L",
                                cluster_name=cluster, n_subjects=str(n), group_mean="1",
                                group_sd="1", group_p_fdr="1"))
            for i, subject in enumerate(["a", "b", "c"]):
                subjects.append(dict(parcellation=zone, contrast="faces_gt_shapes", hemi="L",
                                     cluster_name=cluster, subject=subject,
                                     mean=str(i) if i < n else "nan", n_finite="2", n_vertices="4"))
    fig, axes = render.plt.subplots(2, 1)
    try:
        render.render_task_figure(task="emotion", contrast="faces_gt_shapes", hemis=["L"],
                                 group_root=Path("unused"), surfaces_dir=Path("unused"),
                                 summary_rows=summary, subject_table=subjects, n_subjects=3,
                                 bin_summary=bin_summary,
                                 profile_axes=[axes[0]], bars_axes=[axes[1]], bracket_tests=[])
        labels = [t.get_text() for t in axes[1].texts]
        assert labels.count("n=2") == 2
        assert labels.count("n=3") == 4
        assert "N=3" not in axes[1].get_title()
    finally:
        render.plt.close(fig)


def test_region_brackets_drawn_with_legend(monkeypatch: pytest.MonkeyPatch, bin_summary: dict) -> None:
    roster = ["a", "b", "c", "d", "e", "f"]
    monkeypatch.setattr(render, "group_roster", lambda *a, **kw: list(roster))
    monkeypatch.setattr(render, "long_axis_order", lambda *a: np.arange(40, dtype=np.int32))
    monkeypatch.setattr(render, "equal_vertex_long_axis_bins",
                        lambda *a: [np.array([2 * i, 2 * i + 1]) for i in range(20)])
    monkeypatch.setattr(render, "group_beta_mean", lambda *a, **kw: np.ones(40))
    monkeypatch.setattr(render.np, "load", lambda *a, **kw: np.ones(40))
    summary, subjects = [], []
    offsets = {"Anterior": 10.0, "Middle": 0.0, "Posterior": 0.0}
    for zone in render.ZONE_ORDER:
        for cluster in render.CLUSTER_ORDER:
            summary.append(dict(parcellation=zone, contrast="faces_gt_shapes", hemi="L",
                                cluster_name=cluster, n_subjects=str(len(roster)),
                                group_mean=str(offsets[cluster]),
                                group_sd="1", group_p_fdr="1"))
            for i, subject in enumerate(roster):
                value = (offsets[cluster] + i * 0.29
                         + ((i * 7 + len(cluster)) % 5) * 0.11
                         + ((i * 3 + len(zone)) % 4) * 0.13)
                subjects.append(dict(parcellation=zone, contrast="faces_gt_shapes", hemi="L",
                                      cluster_name=cluster, subject=subject,
                                      mean=str(value),
                                      n_finite="2", n_vertices="4"))
    fig, axes = render.plt.subplots(2, 1)
    try:
        bracket_tests: list = []
        render.render_task_figure(task="emotion", contrast="faces_gt_shapes", hemis=["L"],
                                 group_root=Path("unused"), surfaces_dir=Path("unused"),
                                 summary_rows=summary, subject_table=subjects, n_subjects=6,
                                 bin_summary=bin_summary,
                                 profile_axes=[axes[0]], bars_axes=[axes[1]],
                                 bracket_tests=bracket_tests)
        assert len(bracket_tests) == 6  # 2 zones x 3 subregion pairs, no method tests
        render.draw_brackets(bracket_tests)
        labels = [t.get_text() for t in axes[1].texts]
        assert "***" in labels
        legend_labels = [t.get_text() for t in axes[1].get_legend().get_texts()]
        assert legend_labels == ["hcp", "tm"]
        lo, hi = axes[1].get_ylim()
        assert hi >= max(
            t.get_position()[1] for t in axes[1].texts if t.get_text() in ("+", "*", "**", "***")
        )
    finally:
        render.plt.close(fig)


def test_contrast_physiological_sides_wm() -> None:
    categories = ("faces", "places", "tools", "body")
    # 1. Load contrast
    side1, side2, l1, l2 = render.contrast_physiological_sides("wm", "twobk_gt_zerobk")
    assert l1 == "2-back (mean)" and l2 == "0-back (mean)"
    assert sum(side1.values()) == pytest.approx(1.0)
    assert sum(side2.values()) == pytest.approx(1.0)
    for c in categories:
        assert side1[f"2bk_{c}"] == 0.25
        assert side2[f"0bk_{c}"] == 0.25

    # 2. Category contrasts
    for cat in categories:
        contrast = f"{cat}_gt_others"
        side1, side2, l1, l2 = render.contrast_physiological_sides("wm", contrast)
        assert l1 == f"{cat.capitalize()} (mean)"
        assert l2 == "Others (mean)"
        assert sum(side1.values()) == pytest.approx(1.0)
        assert sum(side2.values()) == pytest.approx(1.0)
        expected_weights = render.TASK_CONTRASTS["wm"][contrast]
        for stem in expected_weights:
            diff = side1.get(stem, 0.0) - side2.get(stem, 0.0)
            assert diff == pytest.approx(expected_weights[stem])

    # 3. Load x Category interactions
    for cat in categories:
        contrast = f"load_x_{cat}"
        side1, side2, l1, l2 = render.contrast_physiological_sides("wm", contrast)
        assert l1 == f"Δ {cat.capitalize()} (2bk-0bk)"
        assert l2 == "Δ Others (2bk-0bk)"
        expected_weights = render.TASK_CONTRASTS["wm"][contrast]
        for stem in expected_weights:
            diff = side1.get(stem, 0.0) - side2.get(stem, 0.0)
            assert diff == pytest.approx(expected_weights[stem])

    # 4. Category-specific load contrasts (2bk - 0bk per category)
    for cat in categories:
        contrast = f"twobk_gt_zerobk_{cat}"
        side1, side2, l1, l2 = render.contrast_physiological_sides("wm", contrast)
        assert l1 == "2-back" and l2 == "0-back"
        assert sum(side1.values()) == pytest.approx(1.0)
        assert sum(side2.values()) == pytest.approx(1.0)
        assert side1[f"2bk_{cat}"] == 1.0
        assert side2[f"0bk_{cat}"] == 1.0
        expected_weights = render.TASK_CONTRASTS["wm"][contrast]
        for stem in expected_weights:
            diff = side1.get(stem, 0.0) - side2.get(stem, 0.0)
            assert diff == pytest.approx(expected_weights[stem])


def test_draw_brackets_collision_free_and_headroom(monkeypatch: pytest.MonkeyPatch, bin_summary: dict) -> None:
    """Verify that multiple overlapping significance brackets never collide or overflow the top frame."""
    roster = ["s1", "s2", "s3", "s4", "s5", "s6"]
    monkeypatch.setattr(render, "group_roster", lambda *a, **kw: list(roster))
    monkeypatch.setattr(render, "long_axis_order", lambda *a: np.arange(40, dtype=np.int32))
    monkeypatch.setattr(render, "equal_vertex_long_axis_bins",
                        lambda *a: [np.array([2 * i, 2 * i + 1]) for i in range(20)])
    monkeypatch.setattr(render, "group_beta_mean", lambda *a, **kw: np.ones(40))
    monkeypatch.setattr(render.np, "load", lambda *a, **kw: np.ones(40))
    summary, subjects = [], []
    offsets = {"Anterior": 30.0, "Middle": 15.0, "Posterior": 0.0}
    for zone in render.ZONE_ORDER:
        for cluster in render.CLUSTER_ORDER:
            summary.append(dict(parcellation=zone, contrast="faces_gt_shapes", hemi="L",
                                cluster_name=cluster, n_subjects=str(len(roster)),
                                group_mean=str(offsets[cluster]),
                                group_sd="1", group_p_fdr="1"))
            for i, subject in enumerate(roster):
                value = offsets[cluster] + (i * 0.1)
                subjects.append(dict(parcellation=zone, contrast="faces_gt_shapes", hemi="L",
                                      cluster_name=cluster, subject=subject,
                                      mean=str(value), n_finite="2", n_vertices="4"))
    fig, axes = render.plt.subplots(2, 1)
    try:
        bracket_tests: list = []
        render.render_task_figure(task="emotion", contrast="faces_gt_shapes", hemis=["L"],
                                 group_root=Path("unused"), surfaces_dir=Path("unused"),
                                 summary_rows=summary, subject_table=subjects, n_subjects=len(roster),
                                 bin_summary=bin_summary,
                                 profile_axes=[axes[0]], bars_axes=[axes[1]],
                                 bracket_tests=bracket_tests)
        render.draw_brackets(bracket_tests)
        sig_texts = [t for t in axes[1].texts if t.get_text() in ("+", "*", "**", "***")]
        assert len(sig_texts) >= 2
        lo, hi = axes[1].get_ylim()
        span = hi - lo
        max_marker_y = max(t.get_position()[1] for t in sig_texts)
        # Verify generous headroom between top marker and top frame: at least 8% of span
        assert (hi - max_marker_y) >= 0.08 * span
    finally:
        render.plt.close(fig)


