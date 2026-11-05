"""Content-based cache validation must never conceal changed or missing inputs."""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'scripts/visualize/lib'))
from surface_panel_cache import panel_inputs, cache_matches, publish_cache


def test_input_changes_and_output_corruption_invalidate_cache(tmp_path):
    source = tmp_path / 'label.gii'
    png = tmp_path / 'panel.png'
    source.write_bytes(b'old labels')
    png.write_bytes(b'original image')
    inputs = panel_inputs([source], smoothing='2mm', variant='36p')
    assert not cache_matches(png, inputs)
    publish_cache(png, inputs)
    assert cache_matches(png, inputs)
    source.write_bytes(b'new labels')
    assert not cache_matches(png, panel_inputs([source], smoothing='2mm', variant='36p'))
    assert not cache_matches(png, panel_inputs([source], smoothing='4mm', variant='36p'))
    png.write_bytes(b'wrong image')
    assert not cache_matches(png, inputs)


def test_missing_source_fails_even_with_cached_png(tmp_path):
    with pytest.raises(FileNotFoundError, match='missing.gii'):
        panel_inputs([tmp_path / 'missing.gii'], smoothing='2mm', variant='36p')


@pytest.mark.parametrize('kind', ['confidence', 'subfield'])
def test_real_loader_does_not_hide_missing_manifest_behind_old_png(tmp_path, kind):
    import runpy
    import numpy as np
    from PIL import Image
    root = Path(__file__).resolve().parents[2]
    panels = tmp_path / 'panels'
    panels.mkdir()
    if kind == 'confidence':
        module = runpy.run_path(str(root / 'scripts/visualize/multidataset_cluster_confidence_stacked/plot_multidataset_cluster_positive_stacked.py'))
        png = panels / 'wb_cohort_2mm_36p_tight4view.png'
    else:
        module = runpy.run_path(str(root / 'scripts/visualize/multidataset_subfield_composition_2mm_2col/render_multidataset_subfield_composition_2mm_2col.py'))
        png = panels / 'wb_subfield_Test_Cohort_2mm_36p.png'
    Image.fromarray(np.random.default_rng(0).integers(0, 256, (100, 100, 3), dtype=np.uint8)).save(png)
    assert png.stat().st_size > 5000
    with pytest.raises(FileNotFoundError, match='group_atlas_manifest.json'):
        if kind == 'confidence':
            module['load_cohort_wb_overlay'](tmp_path / 'cohort', tmp_path, tmp_path / 'scene', panels_dir=panels)
        else:
            module['render_cohort_surface_img'](
                {'dataset_name': 'Test', 'cohort_name': 'Cohort', 'group_root': tmp_path / 'cohort'},
                tmp_path, panels_dir=panels)
