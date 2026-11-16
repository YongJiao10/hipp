from pathlib import Path
import sys

import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.figure import Figure
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'scripts/core_workflow'))
from lib.cortex_fc_tiles import cortex_fc_norm, render_cortex_top10_bars, style_cortex_fc_colorbar


def test_cortex_bar_value_labels_have_two_decimals(tmp_path, monkeypatch):
    figures = []
    original_close = plt.close
    monkeypatch.setattr(Figure, 'savefig', lambda fig, *args, **kwargs: figures.append(fig))
    monkeypatch.setattr(plt, 'close', lambda _figure: None)
    registry = [
        {'hemi': 'L', 'family17': 'DefaultA', 'subregion': 'A', 'rank': 1, 'canonical7': 'Default'},
        {'hemi': 'R', 'family17': 'VisualA', 'subregion': 'B', 'rank': 2, 'canonical7': 'Visual'},
    ]
    render_cortex_top10_bars(panel_tag='test', values=np.array([0.1234, -0.2345]),
                             roi_registry=registry, panels_dir=tmp_path)
    labels = [text.get_text() for text in figures[0].axes[0].texts]
    assert any('(+0.12)' in label for label in labels)
    assert any('(-0.23)' in label for label in labels)
    original_close(figures[0])


def test_cortex_colorbar_keeps_ticks_without_min_max_annotations():
    norm = cortex_fc_norm(np.array([-0.24, 0.58]))
    fig, ax = plt.subplots()
    cbar = fig.colorbar(ScalarMappable(norm=norm, cmap='RdBu_r'), ax=ax, orientation='horizontal')
    style_cortex_fc_colorbar(cbar, norm, fontsize=8)
    assert cbar.get_ticks().size > 0
    assert not cbar.ax.texts
    plt.close(fig)


def test_cortex_tiles_report_reused_and_rendered_surfaces(tmp_path, monkeypatch):
    import importlib.util
    from PIL import Image

    module_path = Path(__file__).resolve().parents[2] / 'scripts/core_workflow/06_01_build_group_overview.py'
    spec = importlib.util.spec_from_file_location('overview_cortex_cache_records_test', module_path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    image = tmp_path / 'tile.png'
    Image.new('RGB', (1000, 700), 'white').save(image)
    stored = tmp_path / 'stored.png'
    monkeypatch.setattr(mod, 'REPO_ROOT', tmp_path)
    (tmp_path / 'tmp').mkdir()
    monkeypatch.setattr(mod, 'load_kong2022_bundle', lambda: {'roi_registry': [], 'vertex_arrays': {}})
    monkeypatch.setattr(mod, 'cortex_asset_paths', lambda *_args: {})
    monkeypatch.setattr(mod, 'resolve_wb_command', lambda: image)
    monkeypatch.setattr(mod, 'resolve_cortex_assets', lambda *_args, **_kwargs: {})
    monkeypatch.setattr(mod, 'cortex_surface_cache_paths', lambda **_kwargs: [image, None, image])
    monkeypatch.setattr(mod, 'store_cortex_surface', lambda **_kwargs: stored)
    monkeypatch.setattr(mod, 'render_cortex_fc_tile', lambda **_kwargs: image)
    fig = plt.figure(figsize=(12, 8))
    records = mod.draw_cluster_cortex_fc(fig, fig.add_gridspec(1, 1)[0], np.array([[-0.2, 0.3]] * 3),
                                         stacked=True, label='', profile_paths=[])
    plt.close(fig)
    assert [(r['cluster'], r['status'], r['surface_png']) for r in records] == [
        (1, 'reused', str(image)), (2, 'rendered', str(stored)), (3, 'reused', str(image))]
