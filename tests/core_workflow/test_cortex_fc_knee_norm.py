from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts" / "core_workflow"))

from lib.cortex_fc_tiles import COLORBAR_KNEE, KneeNorm, cortex_fc_norm, strength_colors  # noqa: E402


def test_knee_norm_keeps_inner_contrast_and_reaches_true_extremes() -> None:
    norm = cortex_fc_norm(np.array([[-0.25, 0.0, np.nan], [0.1, 0.62, 0.05]]))
    assert (norm.vmin, norm.vmax) == pytest.approx((-0.25, 0.62))
    mapped = np.asarray(norm(np.array([-0.25, -COLORBAR_KNEE, 0.0, COLORBAR_KNEE, 0.62])))
    assert mapped == pytest.approx([0.0, 0.1, 0.5, 0.9, 1.0])
    assert norm.inverse(mapped) == pytest.approx([-0.25, -COLORBAR_KNEE, 0.0, COLORBAR_KNEE, 0.62])


def test_knee_norm_side_within_knee_does_not_stretch() -> None:
    norm = cortex_fc_norm(np.array([-0.05, 0.4]))
    assert norm.vmin == pytest.approx(-COLORBAR_KNEE)
    assert np.asarray(norm(np.array([-0.08, 0.08]))) == pytest.approx([0.5 - 0.4 * 0.08 / 0.15, 0.5 + 0.4 * 0.08 / 0.15])


def test_strength_colors_with_knee_norm_distinguishes_values_beyond_knee() -> None:
    norm = KneeNorm(vmin=-0.2, vmax=0.6)
    colors = strength_colors(np.array([0.2, 0.6, 0.0, np.nan]), norm=norm)
    assert not np.allclose(colors[0], colors[1])
    assert colors[2, :3] == pytest.approx([1.0, 1.0, 1.0])
    assert colors[3] == pytest.approx([0.65, 0.65, 0.65, 1.0])


def test_cortex_fc_norm_rejects_all_nan() -> None:
    with pytest.raises(ValueError, match="finite"):
        cortex_fc_norm(np.array([np.nan, np.nan]))
