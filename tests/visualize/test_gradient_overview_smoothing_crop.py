"""Regression coverage for selecting the requested row of a two-smoothing overview."""
import importlib.util
from io import BytesIO
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

PATH = Path(__file__).resolve().parents[2] / 'scripts/visualize/multidataset_gradient_subregion_top20_stacked_2col/render_multidataset_gradient_subregion_top20_stacked_2col.py'
SPEC = importlib.util.spec_from_file_location('smoothing_crop_renderer', PATH)
renderer = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(renderer)


def overview(rows=2):
    im = Image.new('RGB', (4620, 3660), 'white')
    for y, color in [(179, 'red'), (2001, 'blue')][:rows]:
        im.paste('black', (2700, y, 4400, y + 1646))
        im.paste(color, (2800, y + 300, 3100, y + 1300))
    stream = BytesIO()
    im.save(stream, format='PNG')
    stream.seek(0)
    return stream


def single_row_overview():
    im = Image.new('RGB', (4620, 1830), 'white')
    im.paste('black', (2700, 179, 4400, 1825))
    im.paste('red', (2800, 479, 3100, 1479))
    stream = BytesIO()
    im.save(stream, format='PNG')
    stream.seek(0)
    return stream


def white_background_overview():
    im = Image.new('RGB', (4620, 3660), 'white')
    im.paste('red', (2800, 479, 3100, 1479))
    im.paste('blue', (2800, 2301, 3100, 3301))
    stream = BytesIO()
    im.save(stream, format='PNG')
    stream.seek(0)
    return stream


@pytest.mark.parametrize('smoothing,channel', [('2mm', 0), ('4mm', 2)])
def test_crop_selects_requested_smoothing(smoothing, channel):
    crop = np.asarray(renderer.crop_gradient_surface_from_overview(overview(), smoothing=smoothing))
    assert crop.shape == (1644, 860, 3)
    assert crop[500, 200, channel] == 255
    assert crop[500, 200].sum() == 255
    assert np.array_equal(crop[0, 0], [255, 255, 255])
    assert np.array_equal(crop[-1, -1], [255, 255, 255])


def test_single_row_overview_is_explicitly_2mm_only():
    crop = np.asarray(
        renderer.crop_gradient_surface_from_overview(
            single_row_overview(), smoothing='2mm'
        )
    )
    assert crop.shape == (1644, 860, 3)
    assert np.array_equal(crop[0, 0], [255, 255, 255])
    assert np.array_equal(crop[500, 220], [255, 0, 0])


def test_single_row_overview_rejects_4mm():
    with pytest.raises(ValueError, match='single 2mm row'):
        renderer.crop_gradient_surface_from_overview(
            single_row_overview(), smoothing='4mm'
        )


@pytest.mark.parametrize('smoothing,channel', [('2mm', 0), ('4mm', 2)])
def test_white_background_overview_uses_canonical_row_crop(smoothing, channel):
    crop = np.asarray(
        renderer.crop_gradient_surface_from_overview(
            white_background_overview(), smoothing=smoothing
        )
    )
    assert crop.shape == (1644, 860, 3)
    assert np.array_equal(crop[0, 0], [255, 255, 255])
    assert crop[500, 220, channel] == 255


def test_missing_second_row_fails_closed():
    with pytest.raises(ValueError, match='two smoothing rows'):
        renderer.crop_gradient_surface_from_overview(overview(rows=1), smoothing='4mm')


def test_unknown_smoothing_rejected():
    with pytest.raises(ValueError, match='Unsupported smoothing'):
        renderer.crop_gradient_surface_from_overview(overview(), smoothing='8mm')


@pytest.mark.parametrize(
    ('hemi', 'metric_mode', 'expected'),
    [
        ('L', renderer.SUBREGION_METRIC_NORMALIZED_TOP20, 'Left Hipp.'),
        ('R', renderer.SUBREGION_METRIC_NORMALIZED_TOP20, 'Right Hipp.'),
        ('L', renderer.SUBREGION_METRIC_FISHERZ_PROFILE, 'Left Hipp.'),
        ('R', renderer.SUBREGION_METRIC_FISHERZ_PROFILE, 'Right Hipp.'),
    ],
)
def test_subregion_column_headers_use_compact_hemisphere_labels(hemi, metric_mode, expected):
    label = renderer.subregion_bar_column_label(hemi, metric_mode)

    assert label == expected
    assert "Hippocampus" not in label
    assert "hemi-" not in label


@pytest.mark.parametrize(
    ('metric_mode', 'expected'),
    [
        (renderer.SUBREGION_METRIC_NORMALIZED_TOP20, 'Normalized Top-20 Count'),
        (renderer.SUBREGION_METRIC_FISHERZ_PROFILE, 'Positive Fisher-z Profile'),
    ],
)
def test_atlas_column_headers_drop_redundant_gradient_atlas_phrase(metric_mode, expected):
    label = renderer.gradient_atlas_column_label(metric_mode)

    assert label == expected
    assert 'Gradient Atlas' not in label
