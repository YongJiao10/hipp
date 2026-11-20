"""Cortical FC tile: a four-view inflated cortex coloured by one value per Kong2022 ROI,
joined with the bar chart of its ten most positive and ten most negative ROIs.

The group overview and the multi-dataset hippocampus/cortex figure draw the same tile.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import nibabel as nib
import numpy as np
from matplotlib.colors import Normalize
from PIL import Image

from lib.gifti_labels import make_named_label_gifti
from lib.render_cortex_pfm_scene import render_cortex_pfm_scene

DEFAULT_NAN_RGBA = (0.65, 0.65, 0.65, 1.0)
DEFAULT_CMAP = "RdBu_r"
COLORBAR_VMAX = 0.15
COLORBAR_TICKS = (-0.15, -0.08, 0.0, 0.08, 0.15)
COLORBAR_LABEL = "Mean Fisher-z Hippo-Cortex FC"
# Overview colour scale: one linear, zero-centred mapping per figure. L is the
# COLOR_CAP_PERCENTILE-th percentile of |Fisher-z| over the figure's cortex maps; z maps to
# 0.5 + z / (2L), so |z| = L reaches the colormap ends and |z| > L saturates there. The colour
# bar spans the unsaturated range, marks saturated ends with triangles, and its label states the
# true data range and the saturation point. The percentile is fixed across all overviews.
COLOR_CAP_PERCENTILE = 90.0
SURFACE_SIZE = (1000, 700)
COMPOSITE_DIVIDER_PX = 8


class PercentileCapNorm(Normalize):
    """Linear, zero-centred norm saturating beyond +/-cap: z -> clip(0.5 + z / (2 * cap), 0, 1).

    vmin/vmax are the unsaturated part of the data range (the colour bar span); data_min/data_max
    keep the true extremes and `extend` names the saturated ends for the colour bar.
    """

    def __init__(self, *, data_min: float, data_max: float, cap: float) -> None:
        data_min, data_max, cap = float(data_min), float(data_max), float(cap)
        if not (np.isfinite(data_min) and np.isfinite(data_max)) or data_min > data_max:
            raise ValueError(f"PercentileCapNorm data range must be finite and ordered, got {data_min}..{data_max}")
        if not (np.isfinite(cap) and cap > 0.0):
            raise ValueError(f"PercentileCapNorm cap must be positive and finite, got {cap}")
        super().__init__(vmin=max(data_min, -cap), vmax=min(data_max, cap), clip=False)
        self.cap = cap
        self.data_min = data_min
        self.data_max = data_max
        below, above = data_min < -cap, data_max > cap
        self.extend = "both" if below and above else "min" if below else "max" if above else "neither"

    def __call__(self, value, clip=None):
        arr = np.ma.asarray(value, dtype=np.float64)
        out = np.clip(0.5 + 0.5 * arr.filled(np.nan) / self.cap, 0.0, 1.0)
        return np.ma.masked_array(out, mask=np.ma.getmaskarray(arr) | ~np.isfinite(out))

    def inverse(self, value):
        """Inverse inside the unsaturated range only (saturated colours have no unique value)."""
        return (np.asarray(value, dtype=np.float64) - 0.5) * 2.0 * self.cap


def cortex_fc_norm(values: np.ndarray, *, cap_percentile: float = COLOR_CAP_PERCENTILE) -> PercentileCapNorm:
    """Build the colour scale for one figure from all of its finite cortex FC values."""
    arr = np.asarray(values, dtype=np.float64)
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        raise ValueError("Cortex FC colour scale requires at least one finite value")
    if not 0.0 < float(cap_percentile) <= 100.0:
        raise ValueError(f"Cortex FC cap percentile must be in (0, 100], got {cap_percentile}")
    cap = float(np.percentile(np.abs(finite), float(cap_percentile)))
    if not cap > 0.0:
        raise ValueError("Cortex FC colour scale requires non-zero values below the cap percentile")
    return PercentileCapNorm(data_min=float(finite.min()), data_max=float(finite.max()), cap=cap)


def cortex_fc_colorbar_boundaries(norm: PercentileCapNorm, n: int = 256) -> np.ndarray:
    """Colour bar cells spanning exactly the unsaturated range; each cell is coloured by `norm` like the maps."""
    return np.linspace(norm.vmin, norm.vmax, int(n) + 1)


def cortex_fc_colorbar_kwargs(norm: PercentileCapNorm) -> dict[str, Any]:
    """Keyword arguments for `Figure.colorbar` so the bar matches the maps and marks saturated ends."""
    return {"boundaries": cortex_fc_colorbar_boundaries(norm), "extend": norm.extend, "extendfrac": 0.04}


def cortex_fc_colorbar_label(norm: PercentileCapNorm) -> str:
    """Colour bar label stating the true data range and, when values saturate, the saturation point."""
    text = f"{COLORBAR_LABEL}\nrange {norm.data_min:.2f} to {norm.data_max:.2f}"
    if norm.extend != "neither":
        text += f"; colour saturated beyond \u00b1{norm.cap:.2f} (|z| percentile {COLOR_CAP_PERCENTILE:g})"
    return text


def style_cortex_fc_colorbar(cbar, norm: PercentileCapNorm, *, fontsize: float) -> None:
    """Draw a horizontal colour bar on a linear Fisher-z axis spanning the unsaturated range.

    The norm is linear there, so the bar shows exactly the colours used on the cortex maps.
    """
    cbar.ax.set_xscale("linear")
    cbar.ax.set_xlim(norm.vmin, norm.vmax)
    span = norm.vmax - norm.vmin
    ticks = ticker.MaxNLocator(nbins=6, steps=[1, 2, 2.5, 5, 10]).tick_values(norm.vmin, norm.vmax)
    ticks = [float(t) for t in ticks if norm.vmin + 0.04 * span <= t <= norm.vmax - 0.04 * span]
    cbar.set_ticks(ticks, labels=[f"{t:.2f}" if abs(t) >= 0.005 else "0" for t in ticks])
    cbar.ax.tick_params(labelsize=fontsize)


def strength_colors(values: np.ndarray, vmax: float = COLORBAR_VMAX, *, norm: Normalize | None = None) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64)
    if norm is None:
        norm = Normalize(vmin=-float(vmax), vmax=float(vmax), clip=True)
    cmap = plt.get_cmap(DEFAULT_CMAP)
    colors = np.asarray(cmap(norm(np.nan_to_num(arr, nan=0.0))), dtype=np.float64)
    colors[np.isfinite(arr) & (arr == 0.0), :3] = 1.0
    colors[~np.isfinite(arr), :] = np.asarray(DEFAULT_NAN_RGBA, dtype=np.float64)
    return colors


NETWORK_BAR_COLORS = {
    "Default": "#de616c",
    "Somatomotor": "#5297c8",
    "Auditory": "#18b7c9",
    "VentralAttention": "#da42f5",
    "DorsalAttention": "#33a02c",
    "Control": "#f39c12",
    "Language": "#8b4513",
    "Visual": "#6a3d9a",
}


def shorten_roi_label(roi: dict[str, Any]) -> str:
    hemi = roi.get("hemi", "")
    fam = str(roi.get("family17", ""))
    fam_short = (
        fam.replace("SalVenAttn", "VAN")
        .replace("Default", "DMN")
        .replace("Cont", "CON")
        .replace("SomMot", "SMN")
        .replace("DorsAttn", "DAN")
        .replace("Visual", "VN")
        .replace("Language", "LAN")
        .replace("Aud", "AUD")
    )
    sub = str(roi.get("subregion", ""))
    rank = str(roi.get("rank", ""))
    if sub:
        return f"{hemi}_{fam_short}_{sub}_{rank}"
    return f"{hemi}_{fam_short}_{rank}"


def render_cortex_top10_bars(
    *,
    panel_tag: str,
    values: np.ndarray,
    roi_registry: list[dict[str, Any]],
    panels_dir: Path,
) -> Path:
    panel_dir = panels_dir / panel_tag
    panel_dir.mkdir(parents=True, exist_ok=True)
    out_png = panel_dir / f"{panel_tag}_top10_bars_v3.png"

    arr = np.asarray(values, dtype=np.float64)
    pos_idx = np.where(np.isfinite(arr) & (arr > 0))[0]
    neg_idx = np.where(np.isfinite(arr) & (arr < 0))[0]

    top_pos = pos_idx[np.argsort(arr[pos_idx])[::-1][:10]]
    top_neg = neg_idx[np.argsort(arr[neg_idx])[:10]]

    pos_vals = arr[top_pos]
    neg_vals = arr[top_neg]

    pos_colors = [
        NETWORK_BAR_COLORS.get(roi_registry[i].get("canonical7", "Visual"), "#de616c")
        for i in top_pos
    ]
    neg_colors = [
        NETWORK_BAR_COLORS.get(roi_registry[i].get("canonical7", "Visual"), "#33a02c")
        for i in top_neg
    ]

    # Target 1000 x 700 px -> figsize=(5.0, 3.5) @ 200 DPI (matching cortical map height exactly)
    fig, ax = plt.subplots(figsize=(5.0, 3.5), dpi=200)
    fig.patch.set_facecolor("#ffffff")
    ax.set_facecolor("#f8fafc")

    y_pos = np.arange(10)
    y_rev = 9 - y_pos

    max_mag = max(
        float(np.max(np.abs(pos_vals))) if len(pos_vals) > 0 else 0.10,
        float(np.max(np.abs(neg_vals))) if len(neg_vals) > 0 else 0.10,
        0.12,
    )
    x_lim = max_mag * 2.50

    if len(pos_vals) > 0:
        ax.barh(
            y_rev[: len(pos_vals)],
            pos_vals,
            height=0.62,
            color=pos_colors,
            alpha=0.90,
            edgecolor="#1e293b",
            lw=0.8,
            zorder=3,
        )
    if len(neg_vals) > 0:
        ax.barh(
            y_rev[: len(neg_vals)],
            neg_vals,
            height=0.62,
            color=neg_colors,
            alpha=0.90,
            edgecolor="#1e293b",
            lw=0.8,
            zorder=3,
        )

    ax.axvline(0, color="#0f172a", lw=1.4, ls="-", zorder=4)
    ax.set_xlim(-x_lim, x_lim)
    ax.set_ylim(-0.8, 10.3)

    for i in range(len(pos_vals)):
        y = y_rev[i]
        lbl = shorten_roi_label(roi_registry[top_pos[i]])
        val = pos_vals[i]
        ax.text(
            val + x_lim * 0.020,
            y,
            f"{lbl} (+{val:.2f})",
            va="center",
            ha="left",
            fontsize=8.2,
            fontweight="bold",
            color="#0f172a",
            zorder=5,
        )

    for i in range(len(neg_vals)):
        y = y_rev[i]
        lbl = shorten_roi_label(roi_registry[top_neg[i]])
        val = neg_vals[i]
        ax.text(
            val - x_lim * 0.020,
            y,
            f"({val:.2f}) {lbl}",
            va="center",
            ha="right",
            fontsize=8.2,
            fontweight="bold",
            color="#0f172a",
            zorder=5,
        )

    ax.set_yticks([])

    locator = ticker.MaxNLocator(nbins=5, symmetric=True)
    ticks = np.asarray(locator.tick_values(-x_lim, x_lim), dtype=float)
    ticks = ticks[(ticks >= -x_lim * 0.95) & (ticks <= x_lim * 0.95)]
    ax.set_xticks(ticks)
    ax.set_xticklabels(
        [f"{t:+.2f}" if abs(t) >= 0.005 else "0" for t in ticks],
        fontsize=7.6,
        color="#334155",
        fontweight="semibold",
    )
    ax.tick_params(axis="x", length=3.5, pad=2)
    ax.grid(axis="x", alpha=0.45, ls=":", zorder=1)

    ax.text(
        -x_lim * 0.50,
        9.85,
        "◀ Top 10 Neg (Anticorr)",
        ha="center",
        va="bottom",
        fontsize=9.5,
        fontweight="bold",
        color="#1e40af",
    )
    ax.text(
        x_lim * 0.50,
        9.85,
        "Top 10 Pos (Corr) ▶",
        ha="center",
        va="bottom",
        fontsize=9.5,
        fontweight="bold",
        color="#991b1b",
    )

    for s in ["top", "right", "left"]:
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color("#64748b")
    ax.spines["bottom"].set_linewidth(1.0)

    plt.tight_layout(pad=0.20)
    fig.savefig(out_png, dpi=200, facecolor="#ffffff")
    plt.close(fig)
    return out_png


def render_cortex_surface(
    *,
    panel_tag: str,
    colors: np.ndarray,
    roi_registry: list[dict[str, Any]],
    vertex_arrays: dict[str, tuple[np.ndarray, np.ndarray]],
    cortex_assets: dict[str, Path],
    panel_dir: Path,
    out_png: Path,
) -> Path:
    """Render the same 1000x700 four-view geometry for FC and t statistics."""
    label_items = [(0, "Null", (166, 166, 166, 255))]
    for idx, row in enumerate(roi_registry):
        r, g, b, a = [int(round(float(v) * 255.0)) for v in colors[idx]]
        label_items.append((int(row["roi_id"]), str(row["roi_name"]), (r, g, b, a)))

    for hemi in ["L", "R"]:
        vertex_idx, parcel_ids = vertex_arrays[hemi]
        n_verts = int(np.max(vertex_idx)) + 1
        hemi_values = np.zeros((n_verts,), dtype=np.int32)
        hemi_values[vertex_idx.astype(np.int32)] = parcel_ids.astype(np.int32)
        img = make_named_label_gifti(hemi_values, label_items)
        nib.save(img, str(panel_dir / f"{panel_tag}.{hemi}.label.gii"))

    left_label = panel_dir / f"{panel_tag}.L.label.gii"
    right_label = panel_dir / f"{panel_tag}.R.label.gii"
    width, height = SURFACE_SIZE
    render_cortex_pfm_scene(
        source_scene=cortex_assets["scene"],
        left_surface=cortex_assets["left_inflated"], right_surface=cortex_assets["right_inflated"],
        left_label=left_label, right_label=right_label,
        left_sulc=cortex_assets["left_sulc"], right_sulc=cortex_assets["right_sulc"],
        sulc_dscalar=cortex_assets["sulc_dscalar"], dlabel=cortex_assets["dlabel"],
        out_scene=panel_dir / f"{panel_tag}.scene", out_png=out_png,
        width=width, height=height, renderer="OSMesa",
    )
    return out_png


def join_surface_and_bars(surface_png: Path, bar_png: Path, out_png: Path, *, bars: str = "below") -> Path:
    """Paste the bar chart below or to the right of the cortex render, with a white divider."""
    if bars not in {"below", "right"}:
        raise ValueError(f"bars must be 'below' or 'right', got {bars!r}")
    with Image.open(surface_png) as surface_src, Image.open(bar_png) as bar_src:
        surface = surface_src.convert("RGBA")
        bar_chart = bar_src.convert("RGBA")
    if bars == "below":
        size = (surface.width, surface.height + COMPOSITE_DIVIDER_PX + bar_chart.height)
        offset = (0, surface.height + COMPOSITE_DIVIDER_PX)
    else:
        size = (surface.width + COMPOSITE_DIVIDER_PX + bar_chart.width, surface.height)
        offset = (surface.width + COMPOSITE_DIVIDER_PX, 0)
    tile = Image.new("RGBA", size, "#ffffff")
    tile.paste(surface, (0, 0))
    tile.paste(bar_chart, offset)
    tile.save(out_png)
    return out_png


def render_cortex_fc_tile(
    *,
    panel_tag: str,
    values: np.ndarray,
    roi_registry: list[dict[str, Any]],
    vertex_arrays: dict[str, tuple[np.ndarray, np.ndarray]],
    cortex_assets: dict[str, Path],
    panels_dir: Path,
    norm: Normalize,
    bars: str = "below",
    cached_surface: Path | None = None,
) -> Path:
    """Render one tile under `panels_dir/<panel_tag>/` and return the joined image."""
    panel_dir = Path(panels_dir) / panel_tag
    panel_dir.mkdir(parents=True, exist_ok=True)
    surface_png = cached_surface
    if surface_png is None:
        surface_png = render_cortex_surface(
            panel_tag=panel_tag, colors=strength_colors(values, norm=norm), roi_registry=roi_registry,
            vertex_arrays=vertex_arrays, cortex_assets=cortex_assets, panel_dir=panel_dir,
            out_png=panel_dir / f"{panel_tag}_surface.png",
        )
    bar_png = render_cortex_top10_bars(panel_tag=panel_tag, values=values, roi_registry=roi_registry, panels_dir=panels_dir)
    return join_surface_and_bars(surface_png, bar_png, panel_dir / f"{panel_tag}_tile.png", bars=bars)
