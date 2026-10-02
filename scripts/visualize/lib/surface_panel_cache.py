"""Validate derived surface PNGs against exact canonical inputs and render code."""
import hashlib
import json
from pathlib import Path


def fingerprint(path):
    path = Path(path).resolve(strict=True)
    return {'path': str(path), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}


def panel_inputs(paths, **selection):
    # Missing source files must raise before considering a cached image.
    root = Path(__file__).resolve().parents[3]
    renderer_dependencies = [
        Path(__file__),
        root / 'scripts/core_workflow/lib/workbench_grid.py',
        root / 'scripts/core_workflow/lib/workbench_scene_batch.py',
        root / 'config/hipp_network_style.json',
    ]
    return {'schema': 2, 'selection': selection,
            'files': [fingerprint(p) for p in [*paths, *renderer_dependencies]]}


def cache_matches(png, inputs):
    if png is None:
        return False
    png = Path(png)
    sidecar = png.with_suffix('.sources.json')
    if not png.is_file() or not sidecar.is_file():
        return False
    try:
        recorded = json.loads(sidecar.read_text())
        return recorded == {'inputs': inputs, 'output': fingerprint(png)}
    except (ValueError, OSError):
        # A derived cache may be rebuilt from the already validated sources.
        return False


def publish_cache(png, inputs):
    png = Path(png)
    png.with_suffix('.sources.json').write_text(
        json.dumps({'inputs': inputs, 'output': fingerprint(png)}, indent=2) + '\n')
