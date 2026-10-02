import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'scripts/exploratory_beta'))
import render_jj_tms_beta_surface_bar as renderer


def fixture_rows():
    return [dict(subject=s, parcellation='hcp151', hippocampal_hemi=h,
                 cluster=str(c), beta_mean=str(c), n_sites='1', roi_policy='nonzero')
            for s in renderer.SUBJECTS for h in ('L', 'R') for c in (1, 2, 3)]


def test_selection_excludes_3015_and_rejects_duplicate_or_missing_cells():
    rows = fixture_rows()
    extra = dict(rows[0], subject='3015')
    selected = renderer.select_rows(rows + [extra])
    assert len(selected) == 96
    assert {r['subject'] for r in selected} == {
        '1006', '1007', '1017', '1020', '1023', '1035', '1041', '1044', '1052', '1054',
        '3010', '3011', '3017', '3018', '3020', '3023',
    }
    assert '3015' not in {r['subject'] for r in selected}
    assert sum(r['figure_group'] == 'HC' for r in selected) == 60
    assert {r['tms_site_network'] for r in selected if r['subject'] == '3011'} == {'CON'}
    with pytest.raises(ValueError):
        renderer.select_rows(rows + [rows[0]])
    with pytest.raises(ValueError):
        renderer.select_rows(rows[:-1])
    with pytest.raises(ValueError):
        renderer.select_rows([dict(rows[0], beta_mean='nan'), *rows[1:]])


def test_mask_and_weighted_validation_use_vertices_not_region_fill():
    values = np.ones(512)
    values[:4] = [1, 3, 100, np.nan]
    bad = np.zeros(512)
    bad[2] = 1
    labels = np.full(512, 3)
    labels[:4] = 1
    labels[4:8] = 2
    areas = np.ones(512)
    areas[1] = 3
    valid, means = renderer.validate_metric(values, bad, labels, areas, [2.5, 1, 1])
    assert means == pytest.approx([2.5, 1, 1])
    assert not valid[2] and not valid[3]
    with pytest.raises(ValueError):
        renderer.validate_metric(values, bad, labels, areas, [2, 1, 1])
    with pytest.raises(ValueError):
        renderer.validate_metric(values[:-1], bad, labels, areas, [2.5, 1, 1])


def test_summary_keeps_hemispheres_and_individuals():
    rows = fixture_rows()
    for row in rows:
        if row['hippocampal_hemi'] == 'R':
            row['beta_mean'] = str(float(row['beta_mean']) + 10)
    summaries = renderer.summarize(renderer.select_rows(rows))
    assert len(summaries) == 12
    assert {r['n'] for r in summaries} == {6, 10}
    assert next(r['mean'] for r in summaries if r['hemi'] == 'R' and r['cluster'] == 1) == 11


def test_custom_renderer_has_no_hc_mdd_comparison_pipeline():
    assert not hasattr(renderer, 'compare_groups')
    assert not hasattr(renderer, 'annotate_group_comparisons')


def test_cards_use_two_hc_rows_then_one_pure_mdd_row():
    assert renderer.card_rows() == [
        renderer.CONTROLS[:5],
        renderer.CONTROLS[5:],
        list(renderer.PATIENTS),
    ]
    assert all(subject in renderer.CONTROLS for row in renderer.card_rows()[:2] for subject in row)
    assert all(subject in renderer.PATIENTS for subject in renderer.card_rows()[2])
    assert renderer.card_title('1006') == 'sub-1006  ·  HC  ·  VAN'
    assert renderer.card_title('3010') == 'sub-3010  ·  MDD  ·  DMN'


def test_bilateral_card_panels_are_combined_before_composition():
    from PIL import Image

    combined = renderer.combine_card_panels({
        'L': Image.new('RGB', (80, 120), '#a08080'),
        'R': Image.new('RGB', (80, 120), '#8080a0'),
    })
    assert combined.size == (170, 120)


def test_mdd_van_subjects_share_one_marker_distinct_from_other_mdd():
    assert renderer.marker_for_subject('3017') == renderer.marker_for_subject('3020') == 'D'
    assert renderer.marker_for_subject('3017') != renderer.marker_for_subject('3010')
    assert renderer.marker_for_subject('3020') != renderer.marker_for_subject('3011')
    assert renderer.marker_for_subject('3010') == renderer.marker_for_subject('3011') == 'o'


def test_display_encodes_individual_raw_vertices_and_masks_invalid_values():
    values = np.linspace(-2, 2, 512)
    values[0] = np.nan
    image = renderer.vertex_colors(values, 2)
    keys = image.darrays[0].data
    assert keys[0] == 0
    assert len(set(keys[1:])) == 511
    colors = {label.key: label for label in image.labeltable.labels}
    assert colors[2].blue > colors[2].red
    assert colors[512].red > colors[512].blue
    assert colors[0].red == colors[0].blue


def test_compositor_has_96_subject_points_and_png_only_by_default(tmp_path, monkeypatch):
    from PIL import Image
    rows = renderer.select_rows(fixture_rows())
    cards = {s: {h: Image.new('RGB', (80, 120), '#a08080') for h in ('L','R')}
             for s in renderer.SUBJECTS}
    figures = []
    real_close = renderer.plt.close
    monkeypatch.setattr(renderer.plt, 'close', lambda fig: figures.append(fig))
    renderer.compose(rows, renderer.summarize(rows), cards, 3, tmp_path)
    fig = next(f for f in figures if hasattr(f, 'axes'))
    try:
        panels = fig.axes[-2:]
        scatters = [c for ax in panels for c in ax.collections
                    if isinstance(c, renderer.matplotlib.collections.PathCollection)]
        assert len(scatters) == 18
        assert sum(len(c.get_offsets()) for c in scatters) == 96
        assert len(fig.axes) == 19  # 16 bilateral cards, one colorbar, L/R bars
        assert fig._suptitle.get_text().startswith('Iowa 3T')
        assert all(np.min(c.get_sizes()) >= 180 for c in scatters)
        annotation_texts = [t.get_text() for ax in panels for t in ax.texts]
        assert annotation_texts == []
        assert fig._suptitle.get_fontsize() >= 30
        assert panels[0].get_title().endswith('hippocampus')
        assert renderer.SCATTER_TEXT_SCALE == pytest.approx(1.5)
        assert panels[0].title.get_fontsize() == pytest.approx(33)
        assert panels[0].yaxis.label.get_fontsize() == pytest.approx(25.5)
        assert all(label.get_fontsize() == pytest.approx(22.5) for label in panels[0].get_xticklabels())
        assert all(label.get_fontsize() == pytest.approx(22.5) for label in panels[0].get_yticklabels())
        assert all(text.get_fontsize() == pytest.approx(25.5) for ax in panels for text in ax.texts if text.get_text() == 'ns')
        assert all(text.get_fontsize() == pytest.approx(24) for text in fig.legends[0].get_texts())
        fig.canvas.draw()
        canvas_renderer = fig.canvas.get_renderer()
        legend_box = fig.legends[0].get_window_extent(canvas_renderer)
        assert all(not legend_box.overlaps(label.get_window_extent(canvas_renderer))
                   for label in panels[0].get_xticklabels())
        left_panel_box = panels[0].get_window_extent(canvas_renderer)
        right_ylabel_box = panels[1].yaxis.label.get_window_extent(canvas_renderer)
        assert not right_ylabel_box.overlaps(left_panel_box)
        assert fig.axes[0].images[0].get_array().shape[:2] == (120, 170)
        colorbar_axes = [ax for ax in fig.axes if 'Native surface beta' in ax.get_xlabel()]
        assert len(colorbar_axes) == 1
        assert colorbar_axes[0].get_xlabel() == 'Native surface beta (shared scale)'
        assert not any('gray = masked' in text.get_text().lower() for text in fig.texts)
        assert not any('2mm_36p' in text.get_text() for text in fig.texts)
        assert colorbar_axes[0].get_position().width < 0.25
        assert (tmp_path / f'{renderer.STEM}.png').is_file()
        assert not (tmp_path / f'{renderer.STEM}.pdf').exists()
    finally:
        real_close(fig)
