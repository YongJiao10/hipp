from pathlib import Path
import os
import sys

from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'scripts/core_workflow'))
from lib.cortex_overview_cache import cortex_surface_cache_paths, store_cortex_surface


def test_cortex_cache_checks_inputs_code_and_image(tmp_path):
    inputs = [tmp_path / 'left.npy', tmp_path / 'right.npy', tmp_path / 'scene']
    code = [tmp_path / 'renderer.py']
    for path in [*inputs, *code]:
        path.write_text('original')
    kwargs = dict(cache_root=tmp_path / 'cache', inputs=inputs, code=code, size=(1000, 700))
    assert cortex_surface_cache_paths(**kwargs) == [None, None, None]
    source = tmp_path / 'surface.png'
    Image.new('RGB', (1000, 700), 'white').save(source)
    cached = [store_cortex_surface(cache_root=kwargs['cache_root'], inputs=inputs, code=code, index=i, source=source)
              for i in range(3)]
    assert cortex_surface_cache_paths(**kwargs) == cached

    cached[1].write_text('damaged')
    assert cortex_surface_cache_paths(**kwargs) == [cached[0], None, cached[2]]
    store_cortex_surface(cache_root=kwargs['cache_root'], inputs=inputs, code=code, index=1, source=source)
    assert cortex_surface_cache_paths(**kwargs) == cached

    old = inputs[0].stat()
    inputs[0].write_text('updated')
    os.utime(inputs[0], ns=(old.st_atime_ns, old.st_mtime_ns))
    assert cortex_surface_cache_paths(**kwargs) == [None, None, None]
    for i in range(3):
        store_cortex_surface(cache_root=kwargs['cache_root'], inputs=inputs, code=code, index=i, source=source)
    assert cortex_surface_cache_paths(**kwargs) == cached
    code[0].write_text('changed renderer')
    assert cortex_surface_cache_paths(**kwargs) == [None, None, None]

    alternate = tmp_path / 'alternate.npy'
    alternate.write_text('updated')
    assert cortex_surface_cache_paths(**dict(kwargs, inputs=[alternate, *inputs[1:]])) == [None, None, None]


def test_cortex_cache_is_separate_for_each_rendering_library_version(tmp_path, monkeypatch):
    import lib.cortex_overview_cache as cache

    inputs = [tmp_path / 'profile.npy']
    code = [tmp_path / 'renderer.py']
    for path in [*inputs, *code]:
        path.write_text('original')
    source = tmp_path / 'surface.png'
    Image.new('RGB', (1000, 700), 'white').save(source)
    kwargs = dict(cache_root=tmp_path / 'cache', inputs=inputs, code=code, size=(1000, 700))
    for index in range(3):
        store_cortex_surface(cache_root=kwargs['cache_root'], inputs=inputs, code=code, index=index, source=source)
    assert None not in cortex_surface_cache_paths(**kwargs)
    real_version = cache.version
    monkeypatch.setattr(cache, 'version', lambda name: '0.0' if name == 'matplotlib' else real_version(name))
    assert cortex_surface_cache_paths(**kwargs) == [None, None, None]
