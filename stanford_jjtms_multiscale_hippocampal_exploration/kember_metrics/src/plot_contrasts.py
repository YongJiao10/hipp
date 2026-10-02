"""Readable, result-table-driven figures for Kember template-matching effects."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


FAMILY_ORDER = ("surface_area", "functional_segregation", "boundary_sharpness")
METRIC_BY_FAMILY = {
    "surface_area": "surface_area_mm2",
    "functional_segregation": "functional_segregation_d",
    "boundary_sharpness": "boundary_sharpness_d",
}
TITLE_BY_FAMILY = {
    "surface_area": "System surface area (mm²)",
    "functional_segregation": "Functional segregation (Cohen d)",
    "boundary_sharpness": "Boundary sharpness (Cohen d)",
}
SYSTEMS = ("anterior", "body", "posterior")
INK = "#25313b"
MUTED = "#55616c"
WHITE = "#ffffff"
FIGURE_DPI = 300
VALUE_FONT_SIZE = 14.0
TICK_FONT_SIZE = 12.0
TITLE_FONT_SIZE = 18.0


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _table(effects: Any, *, require_row_id: bool) -> pd.DataFrame:
    result = effects.copy() if isinstance(effects, pd.DataFrame) else pd.DataFrame(effects).copy()
    required = {
        "contrast",
        "method",
        "metric_family",
        "metric",
        "smoothing_mm",
        "hemisphere",
        "system",
        "cohens_d",
        "p_value",
    }
    if require_row_id:
        required.add("row_id")
    missing = sorted(required - set(result.columns))
    if missing:
        raise ValueError(f"direct effects table missing visualization columns: {missing}")
    result["smoothing_mm"] = pd.to_numeric(result["smoothing_mm"], errors="raise").astype(int)
    result["cohens_d"] = pd.to_numeric(result["cohens_d"], errors="coerce")
    result["p_value"] = pd.to_numeric(result["p_value"], errors="coerce")
    if require_row_id:
        result["row_id"] = result["row_id"].astype(str)
        if result["row_id"].duplicated().any():
            raise ValueError("direct effects table contains duplicate row_id values")
    return result


def select_figure_smoothings(effects: Any) -> tuple[int, ...]:
    """Select 2 mm alone only when every 2/4-mm conclusion agrees.

    A conclusion agrees when the two effect sizes have the same nonzero sign
    and their unadjusted two-sided Welch p values fall on the same side of
    0.05.  Any missing or unmatched pair retains both smoothing levels.
    """

    table = _table(effects, require_row_id=False)
    available = tuple(sorted(set(table["smoothing_mm"])))
    if not {2, 4}.issubset(available):
        return available
    key_columns = ["method", "contrast", "metric_family", "metric", "system", "hemisphere"]
    saw_pair = False
    for _key, group in table.groupby(key_columns, dropna=False, sort=False):
        pair = group.loc[group["smoothing_mm"].isin((2, 4))]
        if pair["smoothing_mm"].value_counts().to_dict() != {2: 1, 4: 1}:
            return (2, 4)
        effect_2 = pair.loc[pair["smoothing_mm"].eq(2)].iloc[0]
        effect_4 = pair.loc[pair["smoothing_mm"].eq(4)].iloc[0]
        values = (effect_2["cohens_d"], effect_4["cohens_d"], effect_2["p_value"], effect_4["p_value"])
        if not all(np.isfinite(float(value)) for value in values):
            return (2, 4)
        if float(effect_2["cohens_d"]) * float(effect_4["cohens_d"]) <= 0:
            return (2, 4)
        if bool(float(effect_2["p_value"]) < 0.05) != bool(float(effect_4["p_value"]) < 0.05):
            return (2, 4)
        saw_pair = True
    return (2,) if saw_pair else (2, 4)


def _axes(table: pd.DataFrame) -> list[tuple[str, str, str]]:
    pairs = table.loc[:, ["method", "contrast", "hemisphere"]].drop_duplicates()
    pairs = pairs.sort_values(["method", "contrast", "hemisphere"], kind="mergesort")
    return [
        (str(row.method), str(row.contrast), str(row.hemisphere))
        for row in pairs.itertuples()
    ]


def _format_p(value: float) -> str:
    if not np.isfinite(value):
        return "NA"
    if value < 0.001:
        return "<.001"
    return f"{value:.3f}".lstrip("0")


def _limit(values: pd.Series) -> float:
    finite = pd.to_numeric(values, errors="coerce").to_numpy(dtype=float)
    finite = finite[np.isfinite(finite)]
    return max(0.25, float(np.max(np.abs(finite))) * 1.12) if finite.size else 1.0


def _metric_matrices(
    table: pd.DataFrame,
    metric: str,
    smoothing_mm: int,
    axes: list[tuple[str, str, str]],
) -> tuple[np.ndarray, np.ndarray]:
    selected = table.loc[
        table["metric"].astype(str).eq(metric) & table["smoothing_mm"].eq(smoothing_mm)
    ]
    d_values = np.full((len(SYSTEMS), len(axes)), np.nan, dtype=float)
    p_values = np.full((len(SYSTEMS), len(axes)), np.nan, dtype=float)
    row_index = {system: index for index, system in enumerate(SYSTEMS)}
    axis_index = {axis: index for index, axis in enumerate(axes)}
    for row in selected.to_dict("records"):
        system = str(row["system"])
        axis = (str(row["method"]), str(row["contrast"]), str(row["hemisphere"]))
        if system not in row_index or axis not in axis_index:
            continue
        target = (row_index[system], axis_index[axis])
        d_values[target] = float(row["cohens_d"]) if np.isfinite(row["cohens_d"]) else np.nan
        p_values[target] = float(row["p_value"]) if np.isfinite(row["p_value"]) else np.nan
    return d_values, p_values


def _draw_panel(
    axis: plt.Axes,
    d_values: np.ndarray,
    p_values: np.ndarray,
    labels: list[str],
    *,
    title: str,
    limit: float,
) -> Any:
    masked = np.ma.masked_invalid(d_values)
    cmap = plt.get_cmap("RdBu_r").copy()
    cmap.set_bad("#edf0f2")
    image = axis.imshow(masked, aspect="auto", cmap=cmap, vmin=-limit, vmax=limit)
    axis.set_xticks(np.arange(len(labels)), labels, rotation=0, fontsize=TICK_FONT_SIZE)
    axis.set_yticks(np.arange(len(SYSTEMS)), SYSTEMS, fontsize=TICK_FONT_SIZE)
    axis.set_title(title, loc="left", fontsize=TITLE_FONT_SIZE, fontweight="bold", color=INK, pad=12)
    axis.tick_params(length=0, colors=INK)
    for spine in axis.spines.values():
        spine.set_visible(False)
    for row_index in range(d_values.shape[0]):
        for column_index in range(d_values.shape[1]):
            d_value = d_values[row_index, column_index]
            if np.isfinite(d_value):
                text_color = WHITE if abs(d_value) > limit * 0.58 else INK
                axis.text(
                    column_index,
                    row_index,
                    f"d={d_value:.2f}\np={_format_p(p_values[row_index, column_index])}",
                    ha="center",
                    va="center",
                    fontsize=VALUE_FONT_SIZE,
                    fontweight="semibold",
                    color=text_color,
                    linespacing=1.25,
                )
    axis.set_ylabel("system", fontsize=TICK_FONT_SIZE, color=MUTED)
    return image


def _figure_entry(
    figure_id: str,
    png_path: Path,
    source_table: str,
    row_ids: list[str],
) -> dict[str, Any]:
    return {
        "figure_id": figure_id,
        "png_path": str(png_path.resolve()),
        "source_table": source_table,
        "source_row_ids": row_ids,
        "source_row_count": len(row_ids),
        "png_sha256": _sha256(png_path),
    }


def _render_smoothing(
    table: pd.DataFrame,
    output_dir: Path,
    source_table: str,
    smoothing_mm: int,
) -> dict[str, Any]:
    subset = table.loc[table["smoothing_mm"].eq(smoothing_mm)].copy()
    axes = _axes(subset)
    labels = [
        f"{method.replace('hcp151_', '')}\n{contrast}\n{hemisphere}"
        for method, contrast, hemisphere in axes
    ]
    figure, panels = plt.subplots(3, 1, figsize=(28, 15), squeeze=False, constrained_layout=True)
    limit = _limit(subset["cohens_d"])
    image = None
    for axis, family in zip(panels.flat, FAMILY_ORDER):
        d_values, p_values = _metric_matrices(subset, METRIC_BY_FAMILY[family], smoothing_mm, axes)
        image = _draw_panel(
            axis,
            d_values,
            p_values,
            labels,
            title=TITLE_BY_FAMILY[family],
            limit=limit,
        )
    if image is not None:
        figure.colorbar(
            image,
            ax=panels.ravel().tolist(),
            shrink=0.74,
            label="Cohen d (case − NTHC)",
        )
    figure.suptitle(
        f"Stanford unified Kember metrics — {smoothing_mm} mm",
        x=0.02,
        ha="left",
        fontsize=23,
        fontweight="bold",
        color=INK,
    )
    figure.text(
        0.02,
        0.003,
        "Each cell is one direct result-table row. Values are Cohen d and raw two-sided Welch p; gray means endpoint-local missing.",
        fontsize=12,
        color=MUTED,
    )
    figure_id = f"kember_effects_{smoothing_mm}mm"
    png_path = output_dir / f"{figure_id}.png"
    figure.savefig(png_path, dpi=FIGURE_DPI, facecolor=WHITE, bbox_inches="tight")
    plt.close(figure)
    return _figure_entry(
        figure_id,
        png_path,
        source_table,
        subset["row_id"].astype(str).tolist(),
    )


def render_direct_contrast_figures(
    effects: pd.DataFrame,
    output_dir: str | Path,
    *,
    source_table: str = "direct_contrast_effects.tsv",
) -> dict[str, Any]:
    """Render Kember effects while omitting redundant 4-mm figures."""

    table = _table(effects, require_row_id=True)
    output_path = Path(output_dir).expanduser().resolve()
    output_path.mkdir(parents=True, exist_ok=True)
    smoothings = select_figure_smoothings(table)
    entries = [_render_smoothing(table, output_path, source_table, smoothing) for smoothing in smoothings]
    known_ids = set(table["row_id"].astype(str))
    if any(not set(item["source_row_ids"]) <= known_ids for item in entries):
        raise AssertionError("figure provenance references a row outside the result table")
    return {
        "figure_sources": entries,
        "source_table": source_table,
        "source_table_row_count": int(len(table)),
        "source_row_id_integrity": True,
        "displayed_smoothings_mm": list(smoothings),
        "smoothing_display_rule": "2mm only when every 2/4-mm pair agrees in d direction and raw p<0.05 conclusion",
    }
