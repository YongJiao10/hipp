from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pytest
from matplotlib.cm import ScalarMappable

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts" / "core_workflow"))

from lib.cortex_fc_tiles import (  # noqa: E402
    COLOR_CAP_PERCENTILE,
    DEFAULT_CMAP,
    PercentileCapNorm,
    cortex_fc_colorbar_kwargs,
    cortex_fc_colorbar_label,
    cortex_fc_norm,
    strength_colors,
    style_cortex_fc_colorbar,
)


def test_cap_is_the_percentile_of_absolute_values_and_mapping_is_linear() -> None:
    values = np.linspace(-0.2, 0.6, 81)
    norm = cortex_fc_norm(values)
    cap = np.percentile(np.abs(values), COLOR_CAP_PERCENTILE)
    assert norm.cap == pytest.approx(cap)
    mapped = np.asarray(norm(np.array([-cap, -cap / 2, 0.0, cap / 2, cap])))
    assert mapped == pytest.approx([0.0, 0.25, 0.5, 0.75, 1.0])


def test_values_beyond_cap_saturate_and_bar_spans_unsaturated_range() -> None:
    norm = PercentileCapNorm(data_min=-0.25, data_max=0.62, cap=0.15)
    assert np.asarray(norm(np.array([0.3, 0.62, -0.25]))) == pytest.approx([1.0, 1.0, 0.0])
    assert (norm.vmin, norm.vmax) == pytest.approx((-0.15, 0.15))
    assert norm.extend == "both"
    assert PercentileCapNorm(data_min=-0.1, data_max=0.62, cap=0.15).extend == "max"
    unsaturated = PercentileCapNorm(data_min=-0.1, data_max=0.12, cap=0.15)
    assert unsaturated.extend == "neither"
    assert (unsaturated.vmin, unsaturated.vmax) == pytest.approx((-0.1, 0.12))


def test_colorbar_matches_map_colours_and_marks_saturated_ends() -> None:
    values = np.concatenate([np.linspace(-0.12, 0.12, 50), [-0.25, 0.62]])
    norm = cortex_fc_norm(values)
    fig, ax = plt.subplots()
    cbar = fig.colorbar(ScalarMappable(norm=norm, cmap=DEFAULT_CMAP), ax=ax, orientation="horizontal",
                        **cortex_fc_colorbar_kwargs(norm))
    style_cortex_fc_colorbar(cbar, norm, fontsize=8)
    assert cbar.extend == "both"
    assert cbar.ax.get_xlim() == pytest.approx((norm.vmin, norm.vmax))
    bar = cbar.cmap(cbar.norm(np.asarray(cbar._values, dtype=np.float64)))
    ends = strength_colors(np.array([norm.vmin, norm.vmax]), norm=norm)
    assert bar[0] == pytest.approx(ends[0], abs=0.02)
    assert bar[-1] == pytest.approx(ends[1], abs=0.02)
    assert not any(label.get_text().startswith("+") for label in cbar.ax.get_xticklabels())
    plt.close(fig)


def test_colorbar_label_states_true_range_and_saturation_point() -> None:
    label = cortex_fc_colorbar_label(PercentileCapNorm(data_min=-0.25, data_max=0.62, cap=0.15))
    assert "range -0.25 to 0.62" in label and "±0.15" in label
    assert "saturated" not in cortex_fc_colorbar_label(PercentileCapNorm(data_min=-0.1, data_max=0.12, cap=0.15))


def test_strength_colors_keep_zero_white_and_nan_grey() -> None:
    colors = strength_colors(np.array([0.1, 0.6, 0.0, np.nan]), norm=PercentileCapNorm(data_min=-0.2, data_max=0.6, cap=0.2))
    assert not np.allclose(colors[0], colors[1])
    assert colors[2, :3] == pytest.approx([1.0, 1.0, 1.0])
    assert colors[3] == pytest.approx([0.65, 0.65, 0.65, 1.0])


def test_cortex_fc_norm_rejects_invalid_input() -> None:
    with pytest.raises(ValueError, match="finite"):
        cortex_fc_norm(np.array([np.nan, np.nan]))
    with pytest.raises(ValueError, match="non-zero"):
        cortex_fc_norm(np.array([0.0, 0.0]))
    with pytest.raises(ValueError, match="percentile"):
        cortex_fc_norm(np.array([-0.1, 0.2]), cap_percentile=0.0)
