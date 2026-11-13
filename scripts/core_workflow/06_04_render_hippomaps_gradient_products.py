#!/usr/bin/env python3
"""
Process Step Description:
Render HippoMaps gradient products. Renders computed continuous functional gradient data into standard visualization products (including unfolded/folded surface renderings and explained variance plots).

Main Flow Path:
- 9th step in the numbered pipeline sequence (first stage of presentation output generation, run in parallel with 06_01, 06_02, 06_03, 06_05, 06_06, 06_07):
  01_01_run_hippunfold_t1w_local.py -> 01_02_check_hippunfold_outputs.py -> 02_01_prepare_group_inputs.py -> 02_02_build_qc_manifest.py -> 03_01/03_02 reference building -> 04_01/04_02 FC materialization -> 05_01 gradient computation -> 05_02 group atlas parcellation -> 06_04_render_hippomaps_gradient_products.py (this step) -> 07_01 group matrix overview.

Inputs:
- Command-line arguments:
  - --gradient-root: Root directory containing gradient data outputs (default: `outputs_hcp/work/hippomaps_gradients`).
  - --cohort / --variant: Select the same cohort and 36p gradient bank as 05_01 and 05_02.
  - --present-root: Root directory for saving presentation figures (default: `outputs_hcp/present/gradients`).
  - --hippomaps-root: Root directory for the HippoMaps repository package (default: `external/HippoMaps`).
  - --routes: Routes to process (choices: "pfm", "pfm_ctxreg", "pfm_ctxdrop", "fixed"; default: all).
  - --smoothings: Smoothing kernel values to process (default: ["2mm", "4mm"]).
  - --den: Output density resolution (default: "512").
  - --components: Comma-separated list of gradient components to plot (default: "1,2,3").
  - --hemis: Hemispheres to process (default: ["L", "R"]).
  - --cross-route-display-reference: Route used as the display-only sign reference (default: "pfm").
  - --no-cross-route-display-alignment: Disable cross-route display sign alignment and shared color ranges.
  - --display-color-trim-fraction: Quantile trim for shared per-component color ranges (default: 0.05).
- Physical inputs:
  - Gradient output arrays and manifests: `{gradient_root}/{route}/[<cohort>/]{smoothing}/hemi_{hemi}/hippomaps_gradient_manifest.json` and matching `*.npy`/`*.gii` files.

Outputs:
- File outputs:
  - Folded/Unfolded gradient surface rendering PNG: `{present_root}/{route}/[<cohort>/]{smoothing}/hippomaps_gradients_{route}_{smoothing}_{out_tag}_g1g2g3.png`
  - Explained variance fraction plot PNG: `{present_root}/{route}/[<cohort>/]{smoothing}/hippomaps_ev_{route}_{smoothing}_{out_tag}.png`
  - Manifest JSON: `{present_root}/{route}/[<cohort>/]{smoothing}/hippomaps_gradient_products_{route}_{smoothing}_{out_tag}_manifest.json`
"""
from __future__ import annotations

import argparse
import json
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import matplotlib  # type: ignore[import-untyped]
import numpy as np
from lib.fc_math import finite_pearson_1d as finite_pearson_correlation
from lib.hippomaps_adapter import (
    DEFAULT_HIPPOMAPS_ROOT,
    capture_hippomaps_plot,
    enable_den512_plotting_v2,
    ensure_hippomaps_on_path,
    hippomaps_commit,
    load_hippomaps_plotter,
    normalize_hemis,
    wait_for_stable_file,
)
from lib.hippomaps_adapter import (
    stitch_hemi_images as stitch_hemi_gradient_images,
)
from lib.io_utils import load_json
from lib.output_layout import DEFAULT_GRADIENTS_DIR, DEFAULT_HIPPOMAPS_GRADIENT_ROOT, resolve_cohort_gradient_root
from lib.traceability import StageSpec, run_stage
from lib.gradient_profiles import GRADIENT_PROFILE_CHOICES
from lib.workflow_utils import write_json
from PIL import Image, ImageDraw, ImageFont

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # type: ignore[import-untyped]  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_GRADIENT_ROOT = DEFAULT_HIPPOMAPS_GRADIENT_ROOT
DEFAULT_PRESENT_ROOT = DEFAULT_GRADIENTS_DIR
ROUTES = ("pfm", "pfm_ctxreg", "pfm_ctxdrop", "fixed")
SMOOTHINGS = ("2mm", "4mm")
HEMIS = ("L", "R")


def compute_row_ev_percentages(manifests: dict[str, dict[str, Any]], selected_components: list[int]) -> list[float]:
    ev_by_hemi: dict[str, np.ndarray] = {}
    for hemi, manifest in manifests.items():
        ev = np.asarray(manifest["ev_fractions"], dtype=np.float64)
        if ev.size < max(selected_components):
            raise ValueError(f"Manifest for hemi {hemi} has too few EV fractions: {ev.size}")
        ev_by_hemi[hemi] = ev
    out: list[float] = []
    for comp in selected_components:
        vals = [float(ev_by_hemi[hemi][int(comp) - 1]) for hemi in sorted(ev_by_hemi)]
        out.append(float(np.mean(vals) * 100.0))
    return out


def compute_hemi_ev_percentages(manifest: dict[str, Any], selected_components: list[int]) -> list[float]:
    ev = np.asarray(manifest["ev_fractions"], dtype=np.float64)
    if ev.size < max(selected_components):
        raise ValueError(f"Manifest has too few EV fractions: {ev.size}")
    return [float(ev[int(comp) - 1]) * 100.0 for comp in selected_components]


def hemi_output_tag(hemis: list[str]) -> str:
    return "bilateral" if tuple(hemis) == HEMIS else "hemi-" + "".join(hemis)


def align_display_gradients_to_reference(
    *,
    route: str,
    smoothing: str,
    gradients_by_hemi: dict[str, np.ndarray],
    reference_by_hemi: dict[str, np.ndarray],
    selected_components: list[int],
    reference_route: str = "pfm",
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    display = {hemi: np.asarray(arr, dtype=np.float32).copy() for hemi, arr in gradients_by_hemi.items()}
    decisions: list[dict[str, Any]] = []
    for hemi in sorted(display):
        ref = reference_by_hemi.get(hemi)
        if ref is None:
            raise FileNotFoundError(f"Missing reference gradients for route={reference_route} smoothing={smoothing} hemi={hemi}")
        ref_arr = np.asarray(ref, dtype=np.float32)
        src_arr = np.asarray(gradients_by_hemi[hemi], dtype=np.float32)
        for comp in selected_components:
            comp_idx = int(comp) - 1
            if comp_idx < 0 or comp_idx >= src_arr.shape[1] or comp_idx >= ref_arr.shape[1]:
                raise ValueError(
                    f"Component G{comp} is unavailable for route={route} smoothing={smoothing} hemi={hemi}: "
                    f"source_shape={src_arr.shape} reference_shape={ref_arr.shape}"
                )
            corr = finite_pearson_correlation(src_arr[:, comp_idx], ref_arr[:, comp_idx])
            sign = -1 if np.isfinite(corr) and corr < 0.0 else 1
            if sign == -1:
                display[hemi][:, comp_idx] *= -1.0
            decisions.append(
                {
                    "hemi": hemi,
                    "component": f"G{int(comp)}",
                    "correlation_to_reference": None if not np.isfinite(corr) else float(corr),
                    "display_sign": int(sign),
                    "flipped": bool(sign == -1),
                }
            )
    return display, {
        "enabled": True,
        "mode": "sign_only_per_hemi_component",
        "reference_route": reference_route,
        "route": route,
        "smoothing": smoothing,
        "array_scope": "display_only_saved_gradient_arrays_unchanged",
        "decisions": decisions,
    }


def compute_shared_display_color_ranges(
    display_gradient_sets: list[dict[str, np.ndarray]],
    *,
    selected_components: list[int],
    trim_fraction: float = 0.05,
) -> dict[int, tuple[float, float]]:
    trim = float(trim_fraction)
    if trim < 0.0 or trim >= 0.5:
        raise ValueError(f"trim_fraction must be in [0, 0.5), got {trim_fraction}")
    ranges: dict[int, tuple[float, float]] = {}
    for comp in selected_components:
        comp_idx = int(comp) - 1
        chunks: list[np.ndarray] = []
        for by_hemi in display_gradient_sets:
            for arr in by_hemi.values():
                mat = np.asarray(arr, dtype=np.float32)
                if comp_idx < 0 or comp_idx >= mat.shape[1]:
                    raise ValueError(f"Component G{comp} is unavailable for display color range; shape={mat.shape}")
                vals = mat[:, comp_idx]
                vals = vals[np.isfinite(vals)]
                if vals.size:
                    chunks.append(vals.astype(np.float64, copy=False))
        if not chunks:
            ranges[int(comp)] = (-1.0, 1.0)
            continue
        all_vals = np.concatenate(chunks)
        lo = float(np.quantile(all_vals, trim))
        hi = float(np.quantile(all_vals, 1.0 - trim))
        if not np.isfinite(lo) or not np.isfinite(hi):
            lo, hi = -1.0, 1.0
        if lo == hi:
            pad = max(abs(lo) * 0.05, 1.0)
            lo -= pad
            hi += pad
        ranges[int(comp)] = (float(lo), float(hi))
    return ranges


def format_color_ranges_for_components(
    color_ranges_by_component: dict[int, tuple[float, float]], selected_components: list[int]
) -> list[list[tuple[float, float]]]:
    return [[(float(color_ranges_by_component[int(comp)][0]), float(color_ranges_by_component[int(comp)][1]))] for comp in selected_components]


def load_gradient_arrays_for_display(
    *, gradient_root: Path, route: str, smoothing: str, hemis: list[str],
    cohort: str | None = None, variant: str = "36p", profile: str | None = None,
) -> dict[str, np.ndarray]:
    branch_root = resolve_cohort_gradient_root(
        route=route, cohort=cohort, variant=variant, base_root=gradient_root, profile=profile,
    )
    return {
        hemi: np.load(branch_root / smoothing / f"hemi_{hemi}" / "hippomaps_gradients_full.npy").astype(np.float32)
        for hemi in hemis
    }


def build_cross_route_display_specs(
    *,
    gradient_root: Path,
    routes: list[str],
    smoothings: list[str],
    hemis: list[str],
    selected_components: list[int],
    reference_route: str,
    trim_fraction: float = 0.05,
    cohort: str | None = None,
    variant: str = "36p",
    profile: str | None = None,
) -> dict[tuple[str, str], dict[str, Any]]:
    specs: dict[tuple[str, str], dict[str, Any]] = {}
    for smoothing in smoothings:
        reference_by_hemi = load_gradient_arrays_for_display(
            gradient_root=gradient_root,
            route=reference_route,
            smoothing=smoothing,
            hemis=hemis,
            cohort=cohort,
            variant=variant,
            profile=profile,
        )
        aligned_sets: list[dict[str, np.ndarray]] = []
        pending: dict[str, tuple[dict[str, np.ndarray], dict[str, Any]]] = {}
        for route in routes:
            gradients_by_hemi = load_gradient_arrays_for_display(
                gradient_root=gradient_root,
                route=route,
                smoothing=smoothing,
                hemis=hemis,
                cohort=cohort,
                variant=variant,
                profile=profile,
            )
            display_by_hemi, alignment_meta = align_display_gradients_to_reference(
                route=route,
                smoothing=smoothing,
                gradients_by_hemi=gradients_by_hemi,
                reference_by_hemi=reference_by_hemi,
                selected_components=selected_components,
                reference_route=reference_route,
            )
            pending[route] = (display_by_hemi, alignment_meta)
            aligned_sets.append(display_by_hemi)
        shared_ranges = compute_shared_display_color_ranges(
            aligned_sets,
            selected_components=selected_components,
            trim_fraction=trim_fraction,
        )
        for route, (display_by_hemi, alignment_meta) in pending.items():
            specs[(route, smoothing)] = {
                "display_gradients_by_hemi": display_by_hemi,
                "cross_route_display_alignment": alignment_meta,
                "display_color_ranges_by_component": shared_ranges,
                "display_color_trim_fraction": float(trim_fraction),
            }
    return specs


def annotate_gradient_image(
    image_path: Path,
    *,
    selected_components: list[int],
    ev_percentages: list[float],
    title: str = "Gradient",
) -> Path:
    img = Image.open(image_path).convert("RGB")
    src_w, src_h = img.size
    margin_w = 230
    padding_h = 30
    out = Image.new("RGB", (src_w + margin_w, src_h + 2 * padding_h), (255, 255, 255))
    out.paste(img, (margin_w, padding_h))
    draw = ImageDraw.Draw(out)
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 22)
        small_font = ImageFont.truetype("DejaVuSans.ttf", 16)
    except Exception:
        font = ImageFont.load_default()
        small_font = font

    draw.text((18, 14 + padding_h), title, fill=(0, 0, 0), font=font)
    n_rows = max(1, len(selected_components))
    row_h = src_h / float(n_rows)
    for idx, comp in enumerate(selected_components):
        y = int(round((idx + 0.5) * row_h)) + padding_h
        label = f"G{int(comp)}"
        ev_text = f"EV={float(ev_percentages[idx]):.1f}%"
        draw.text((18, y - 18), label, fill=(0, 0, 0), font=font)
        draw.text((18, y + 8), ev_text, fill=(0, 0, 0), font=small_font)
    out.save(image_path)
    return image_path


def ensure_bilateral_gradient_annotations(
    *,
    side_paths: dict[str, Path],
    output_png: Path,
    selected_components: list[int],
    ev_by_hemi: dict[str, list[float]],
    min_annotated_width: int = 550,
    settle_seconds: float = 1.0,
) -> Path:
    # VTK can finish writing the screenshot after the plotter call returns. Do one
    # final late pass so the maintained artifact keeps the EV label margin.
    time.sleep(float(settle_seconds))
    changed = False
    for hemi, title in [("L", "L Gradient"), ("R", "R Gradient")]:
        side = side_paths[hemi]
        wait_for_stable_file(side)
        width, _height = Image.open(side).size
        if int(width) < int(min_annotated_width):
            annotate_gradient_image(
                side,
                selected_components=selected_components,
                ev_percentages=ev_by_hemi[hemi],
                title=title,
            )
            changed = True
    if changed or not output_png.exists():
        stitch_hemi_gradient_images(side_paths["L"], side_paths["R"], output_png)
    wait_for_stable_file(output_png)
    return output_png


def render_gradient_figure(
    *,
    left: np.ndarray,
    right: np.ndarray | None = None,
    output_png: Path,
    den: str = "512",
    hemis: list[str] | tuple[str, ...] = HEMIS,
    plotter: Callable[..., object] | None = None,
    color_ranges: list[list[tuple[float, float]]] | None = None,
    tighten_cwindow: bool = True,
) -> Path:
    hemis = normalize_hemis(hemis)
    arrays: list[np.ndarray] = []
    if "L" in hemis:
        arrays.append(np.asarray(left, dtype=np.float32))
    if "R" in hemis:
        if right is None:
            raise ValueError("right gradient array is required when rendering hemisphere R")
        arrays.append(np.asarray(right, dtype=np.float32))
    if len(arrays) == 2 and arrays[0].shape != arrays[1].shape:
        raise ValueError(f"Left/right gradient shape mismatch: {arrays[0].shape} vs {arrays[1].shape}")
    cdata = np.stack(arrays, axis=1)
    if plotter is None:
        plotter = load_hippomaps_plotter()
    unfold_ap_rescale = str(den) != "512"
    plotter_kwargs = {
        "hemis": hemis,
        "labels": ["hipp"],
        "den": str(den),
        "unfoldAPrescale": unfold_ap_rescale,
        "tighten_cwindow": bool(tighten_cwindow),
        "cmap": "plasma",
        "share": "row",
        "color_bar": "right",
        "background": (1, 1, 1),
        "transparent_bg": False,
        "interactive": False,
        "screenshot": True,
        "suppress_warnings": True,
        "size": [450, 300],
        "cb__width": 0.95,
        "cb__position": [0.02, 0.25],
        "cb__barRatio": 0.18,
        "cb__labelTextProperty": {"fontSize": 13},
        "zoom": 1.45,
    }
    if color_ranges is not None:
        plotter_kwargs["color_range"] = color_ranges
    return capture_hippomaps_plot(plotter, cdata, output_png=output_png, **plotter_kwargs)


def render_bilateral_gradient_figure(
    *,
    left: np.ndarray,
    right: np.ndarray,
    output_png: Path,
    den: str = "512",
    selected_components: list[int],
    ev_by_hemi: dict[str, list[float]],
    plotter: Callable[..., object] | None = None,
    color_ranges: list[list[tuple[float, float]]] | None = None,
    tighten_cwindow: bool = True,
) -> Path:
    output_png.parent.mkdir(parents=True, exist_ok=True)
    if plotter is None:
        plotter = load_hippomaps_plotter()
    side_paths = {
        "L": output_png.with_name(f"{output_png.stem}_hemi-L{output_png.suffix}"),
        "R": output_png.with_name(f"{output_png.stem}_hemi-R{output_png.suffix}"),
    }
    render_gradient_figure(
        left=left,
        output_png=side_paths["L"],
        den=den,
        hemis=["L"],
        plotter=plotter,
        color_ranges=color_ranges,
        tighten_cwindow=tighten_cwindow,
    )
    render_gradient_figure(
        left=np.empty((0, np.asarray(right).shape[1]), dtype=np.float32),
        right=right,
        output_png=side_paths["R"],
        den=den,
        hemis=["R"],
        plotter=plotter,
        color_ranges=color_ranges,
        tighten_cwindow=tighten_cwindow,
    )
    for hemi, title in [("L", "L Gradient"), ("R", "R Gradient")]:
        wait_for_stable_file(side_paths[hemi])
        annotate_gradient_image(
            side_paths[hemi],
            selected_components=selected_components,
            ev_percentages=ev_by_hemi[hemi],
            title=title,
        )
    return stitch_hemi_gradient_images(side_paths["L"], side_paths["R"], output_png)


EV_FONT_DELTA_PT = 4
EV_TITLE_FS = 12 + EV_FONT_DELTA_PT  # 16
EV_LABEL_FS = 10 + EV_FONT_DELTA_PT  # 14
EV_TICK_FS = 10 + EV_FONT_DELTA_PT   # 14
EV_LEGEND_FS = 10 + EV_FONT_DELTA_PT # 14


def render_ev_plot(
    manifests: dict[str, dict[str, Any]],
    output_png: Path,
    title: str = "Variance Explained by Gradients",
) -> Path:
    fig, ax = plt.subplots(figsize=(7.0, 3.8))
    max_components = 0
    for hemi, manifest in sorted(manifests.items()):
        ev = np.asarray(manifest["ev_fractions"], dtype=np.float64) * 100.0
        max_components = max(max_components, int(ev.size))
        ax.plot(np.arange(1, ev.size + 1), ev, marker="o", linewidth=1.5, label=f"hemi {hemi}")
    ax.tick_params(axis="both", labelsize=EV_TICK_FS)
    if max_components > 0:
        ticks = np.arange(1, max_components + 1)
        ax.set_xticks(ticks)
        ax.set_xticklabels([str(t) for t in ticks], fontsize=EV_TICK_FS)
    ax.set_xlabel("Gradient", fontsize=EV_LABEL_FS)
    ax.set_ylabel("EV (%)", fontsize=EV_LABEL_FS)
    if title:
        ax.set_title(title, fontsize=EV_TITLE_FS)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, fontsize=EV_LEGEND_FS)
    output_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_png, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return output_png


def render_combo(
    *,
    route: str,
    smoothing: str,
    gradient_root: Path,
    present_root: Path,
    hippomaps_root: Path,
    den: str,
    selected_components: list[int],
    hemis: list[str],
    display_spec: dict[str, Any] | None = None,
    cohort: str | None = None,
    variant: str = "36p",
    profile: str | None = None,
) -> Path:
    hemis = normalize_hemis(hemis)
    branch_root = resolve_cohort_gradient_root(
        route=route, cohort=cohort, variant=variant, base_root=gradient_root, profile=profile,
    )
    combo_root = resolve_cohort_gradient_root(
        route=route, cohort=cohort, variant=variant, base_root=present_root, profile=profile,
    ) / smoothing
    manifests: dict[str, dict[str, Any]] = {}
    gradients_by_hemi: dict[str, np.ndarray] = {}
    support_by_hemi: dict[str, np.ndarray] = {}
    for hemi in hemis:
        run_dir = branch_root / smoothing / f"hemi_{hemi}"
        manifest_path = run_dir / "hippomaps_gradient_manifest.json"
        if not manifest_path.exists():
            raise FileNotFoundError(f"Missing HippoMaps gradient manifest: {manifest_path}")
        manifests[hemi] = load_json(manifest_path)
        gradients_by_hemi[hemi] = np.load(run_dir / "hippomaps_gradients_full.npy").astype(np.float32)
        support_by_hemi[hemi] = np.load(run_dir / "group_support_mask.npy").astype(bool)

    comp_indices = [int(c) - 1 for c in selected_components]
    display_gradients_by_hemi = gradients_by_hemi
    cross_route_display_alignment = {"enabled": False}
    display_color_ranges_by_component: dict[int, tuple[float, float]] | None = None
    display_color_trim_fraction: float | None = None
    if display_spec is not None:
        display_gradients_by_hemi = display_spec["display_gradients_by_hemi"]
        cross_route_display_alignment = display_spec["cross_route_display_alignment"]
        display_color_ranges_by_component = display_spec["display_color_ranges_by_component"]
        display_color_trim_fraction = float(display_spec["display_color_trim_fraction"])
    plot_color_ranges = (
        format_color_ranges_for_components(display_color_ranges_by_component, selected_components)
        if display_color_ranges_by_component is not None
        else None
    )
    plot_tighten_cwindow = plot_color_ranges is None
    left = display_gradients_by_hemi["L"][:, comp_indices] if "L" in hemis else np.empty((0, len(comp_indices)), dtype=np.float32)
    right = display_gradients_by_hemi["R"][:, comp_indices] if "R" in hemis else None
    out_tag = hemi_output_tag(hemis)
    gradient_png = combo_root / f"hippomaps_gradients_{route}_{smoothing}_{out_tag}_g1g2g3.png"
    den512_plotting_v2_added = enable_den512_plotting_v2(hippomaps_root) if str(den) == "512" else False
    unfold_ap_rescale = str(den) != "512"
    ev_by_hemi = {hemi: compute_hemi_ev_percentages(manifests[hemi], selected_components) for hemi in hemis}
    plotter = load_hippomaps_plotter(hippomaps_root)
    side_gradient_outputs: dict[str, str] = {}
    if tuple(hemis) == HEMIS:
        render_bilateral_gradient_figure(
            left=left,
            right=np.asarray(right, dtype=np.float32),
            output_png=gradient_png,
            den=den,
            selected_components=selected_components,
            ev_by_hemi=ev_by_hemi,
            plotter=plotter,
            color_ranges=plot_color_ranges,
            tighten_cwindow=plot_tighten_cwindow,
        )
        side_gradient_outputs = {
            "L": str(gradient_png.with_name(f"{gradient_png.stem}_hemi-L{gradient_png.suffix}").resolve()),
            "R": str(gradient_png.with_name(f"{gradient_png.stem}_hemi-R{gradient_png.suffix}").resolve()),
        }
    else:
        render_gradient_figure(
            left=left,
            right=right,
            output_png=gradient_png,
            den=den,
            hemis=hemis,
            plotter=plotter,
            color_ranges=plot_color_ranges,
            tighten_cwindow=plot_tighten_cwindow,
        )
        hemi = hemis[0]
        annotate_gradient_image(
            gradient_png,
            selected_components=selected_components,
            ev_percentages=ev_by_hemi[hemi],
            title=f"{hemi} Gradient",
        )

    ev_png = render_ev_plot(manifests, combo_root / f"hippomaps_ev_{route}_{smoothing}_{out_tag}.png")
    if tuple(hemis) == HEMIS:
        ensure_bilateral_gradient_annotations(
            side_paths={
                "L": gradient_png.with_name(f"{gradient_png.stem}_hemi-L{gradient_png.suffix}"),
                "R": gradient_png.with_name(f"{gradient_png.stem}_hemi-R{gradient_png.suffix}"),
            },
            output_png=gradient_png,
            selected_components=selected_components,
            ev_by_hemi=ev_by_hemi,
        )
    manifest_out = combo_root / f"hippomaps_gradient_products_{route}_{smoothing}_{out_tag}_manifest.json"
    write_json(
        manifest_out,
        {
            "status": "ok",
            "route": route,
            "smoothing": smoothing,
            "cohort": cohort,
            "variant": variant,
            "profile": profile,
            "source_manifests": {hemi: str((branch_root / smoothing / f"hemi_{hemi}" / "hippomaps_gradient_manifest.json").resolve()) for hemi in hemis},
            "hippomaps_repo": str(Path(hippomaps_root).resolve()),
            "hippomaps_commit": hippomaps_commit(hippomaps_root),
            "plotter": "hippomaps.plotting.surfplot_canonical_foldunfold",
            "plotter_kwargs": {
                "cmap": "plasma",
                "share": "row",
                "color_bar": "right",
                "labels": ["hipp"],
                "hemis": hemis,
                "den": den,
                "unfoldAPrescale": unfold_ap_rescale,
                "tighten_cwindow": bool(plot_tighten_cwindow),
                "color_range": plot_color_ranges,
            },
            "bilateral_rendering": {
                "hemi_specific_ev_labels": True,
                "hemi_specific_colorbars": tuple(hemis) == HEMIS,
                "side_gradient_figures": side_gradient_outputs,
            },
            "plotter_support_assets": {
                "den512_plotting_v2_enabled": bool(str(den) == "512"),
                "den512_plotting_v2_added_this_run": bool(den512_plotting_v2_added),
            },
            "selected_components": [f"G{c}" for c in selected_components],
            "gradient_array_semantics": {
                hemi: manifests[hemi].get("gradient_array_semantics", "legacy_unaligned_or_unspecified") for hemi in hemis
            },
            "gradient_alignment": {
                hemi: manifests[hemi].get("gradient_alignment", {"enabled": False}) for hemi in hemis
            },
            "cross_route_display_alignment": cross_route_display_alignment,
            "display_color_ranges_by_component": (
                {f"G{comp}": list(map(float, rng)) for comp, rng in sorted(display_color_ranges_by_component.items())}
                if display_color_ranges_by_component is not None
                else None
            ),
            "display_color_trim_fraction": display_color_trim_fraction,
            "outputs": {
                "gradient_figure": str(gradient_png.resolve()),
                "ev_plot": str(ev_png.resolve()),
            },
        },
    )
    print(json.dumps({"status": "ok", "manifest": str(manifest_out)}, indent=2), flush=True)
    return manifest_out


def parse_components(raw: str) -> list[int]:
    comps = [int(x.strip().replace("G", "").replace("g", "")) for x in str(raw).split(",") if x.strip()]
    if not comps:
        raise ValueError("--components must not be empty")
    if any(c < 1 for c in comps):
        raise ValueError(f"Components must be positive: {raw!r}")
    return comps


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render HippoMaps-style continuous gradient products.")
    parser.add_argument("--gradient-root", default=str(DEFAULT_GRADIENT_ROOT))
    parser.add_argument("--present-root", default=str(DEFAULT_PRESENT_ROOT))
    parser.add_argument("--cohort", default=None, help="Cohort directory below each route; omit for HCP.")
    parser.add_argument("--variant", choices=("36p",), default="36p")
    parser.add_argument(
        "--profile", choices=GRADIENT_PROFILE_CHOICES, default=None,
        help="Formal gradient profile below each route (for example topsim); omit for unprofiled gradient banks.",
    )
    parser.add_argument("--hippomaps-root", default=str(DEFAULT_HIPPOMAPS_ROOT))
    parser.add_argument("--routes", nargs="+", choices=ROUTES, default=list(ROUTES))
    parser.add_argument("--smoothings", nargs="+", choices=SMOOTHINGS, default=list(SMOOTHINGS))
    parser.add_argument("--den", default="512")
    parser.add_argument("--components", default="1,2,3")
    parser.add_argument("--hemis", nargs="+", choices=HEMIS, default=list(HEMIS))
    parser.add_argument(
        "--cross-route-display-reference",
        choices=ROUTES,
        default="pfm",
        help="Reference route used for display-only sign alignment across branches.",
    )
    parser.add_argument(
        "--no-cross-route-display-alignment",
        action="store_true",
        help="Disable display-only cross-route sign alignment and shared color ranges.",
    )
    parser.add_argument(
        "--display-color-trim-fraction",
        type=float,
        default=0.05,
        help="Per-component quantile trim used to define shared cross-route display color ranges.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    hippomaps_root = ensure_hippomaps_on_path(Path(args.hippomaps_root))
    components = parse_components(args.components)
    hemis = normalize_hemis(args.hemis)
    gradient_root = Path(args.gradient_root).resolve()
    present_root = Path(args.present_root).resolve()
    display_specs: dict[tuple[str, str], dict[str, Any]] = {}
    if not bool(args.no_cross_route_display_alignment):
        display_specs = build_cross_route_display_specs(
            gradient_root=gradient_root,
            routes=list(args.routes),
            smoothings=list(args.smoothings),
            hemis=hemis,
            selected_components=components,
            reference_route=str(args.cross_route_display_reference),
            trim_fraction=float(args.display_color_trim_fraction),
            cohort=args.cohort,
            variant=args.variant,
            profile=args.profile,
        )
    for route in args.routes:
        for smoothing in args.smoothings:
            render_combo(
                route=route,
                smoothing=smoothing,
                gradient_root=gradient_root,
                present_root=present_root,
                hippomaps_root=hippomaps_root,
                den=str(args.den),
                selected_components=components,
                hemis=hemis,
                display_spec=display_specs.get((route, smoothing)),
                cohort=args.cohort,
                variant=args.variant,
                profile=args.profile,
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(
        run_stage(
            StageSpec(
                step_id="06_04_render_hippomaps_gradient_products",
                purpose="Render HippoMaps folded/unfolded gradient figures and EV plots.",
                stage_main=main,
                upstream_defaults=("05_01_compute_hippomaps_gradients",),
            )
        )
    )
