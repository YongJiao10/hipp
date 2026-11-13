from pathlib import Path
import os
import sys

import nibabel as nib
import numpy as np
from nibabel.gifti import GiftiDataArray, GiftiImage, GiftiLabel, GiftiLabelTable
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'scripts/core_workflow'))
from lib.surface_utils import render_hippocampal_grid
from lib import surface_utils


def test_reuse_surface_when_only_overview_text_changes(tmp_path, monkeypatch):
    import brainspace.mesh.mesh_io
    import brainspace.plotting

    class Mesh:
        n_points = 4
        def append_array(self, *_args, **_kwargs):
            pass

    calls = []
    class Plotter:
        def screenshot(self, filename, **_kwargs):
            Image.new('RGB', (3200, 2048), 'white').save(filename)
        def close(self):
            pass

    def fake_plot(*_args, **_kwargs):
        calls.append('render')
        return Plotter()

    monkeypatch.setattr(brainspace.mesh.mesh_io, 'read_surface', lambda _path: Mesh())
    monkeypatch.setattr(brainspace.plotting, 'plot_surf', fake_plot)
    monkeypatch.setattr(surface_utils, 'compose_wb_grid_with_legend',
                        lambda **kwargs: Path(kwargs['out']).write_text(kwargs['title']))
    surfaces = []
    labels = []
    for hemi in ('L', 'R'):
        surface = tmp_path / f'{hemi}.surf.gii'
        surface.write_text('surface')
        table = GiftiLabelTable()
        label = GiftiLabel(key=1, red=.8, green=.3, blue=.3, alpha=1.)
        label.label = 'Default'
        table.labels.append(label)
        gifti = GiftiImage(darrays=[GiftiDataArray(data=np.ones(4, dtype=np.int32), intent='NIFTI_INTENT_LABEL')], labeltable=table)
        label_path = tmp_path / f'{hemi}.label.gii'
        nib.save(gifti, label_path)
        surfaces.append(surface)
        labels.append(label_path)

    args = dict(subject='1001', name='test', left_labels=labels[0], right_labels=labels[1],
                left_surface=surfaces[0], right_surface=surfaces[1],
                outdir=tmp_path / 'panel', surface_cache_root=tmp_path / 'cache')
    first = render_hippocampal_grid(**args, title='First title')
    first_bytes = Path(first['biglegend_png']).read_bytes()
    cache = Path(first['surface_png'])
    second = render_hippocampal_grid(**args, title='Updated title')
    assert len(calls) == 1
    assert first_bytes != Path(second['biglegend_png']).read_bytes()

    original = labels[0].stat()
    changed = nib.load(labels[0])
    changed.labeltable.labels[0].red = .25
    nib.save(changed, labels[0])
    os.utime(labels[0], ns=(original.st_atime_ns, original.st_mtime_ns))
    render_hippocampal_grid(**args, title='Updated title')
    assert len(calls) == 2

    changed_time = cache.stat().st_mtime_ns + 1_000_000_000
    os.utime(labels[0], ns=(changed_time, changed_time))
    render_hippocampal_grid(**args, title='Updated title')
    assert len(calls) == 3

    alternate = tmp_path / 'alternate.label.gii'
    alternate.write_bytes(labels[0].read_bytes())
    other_args = dict(args, left_labels=alternate)
    other = render_hippocampal_grid(**other_args, title='Updated title')
    assert len(calls) == 4
    assert Path(other['surface_png']) != cache

    other_cache = Path(other['surface_png'])
    other_cache.write_bytes(b'corrupted image')
    render_hippocampal_grid(**other_args, title='Updated title')
    assert len(calls) == 5

    alternate_surface = tmp_path / 'alternate.surf.gii'
    alternate_surface.write_text('another surface')
    changed_surface = render_hippocampal_grid(
        **dict(args, left_surface=alternate_surface), title='Updated title'
    )
    assert len(calls) == 6
    assert Path(changed_surface['surface_png']) != cache

    monkeypatch.setattr(surface_utils, 'version', lambda _package: 'other-version')
    changed_runtime = render_hippocampal_grid(**args, title='Updated title')
    assert len(calls) == 7
    assert Path(changed_runtime['surface_png']) != cache

    uncached = dict(args, surface_cache_root=None)
    render_hippocampal_grid(**uncached, title='Updated title')
    render_hippocampal_grid(**uncached, title='Updated title')
    assert len(calls) == 9
