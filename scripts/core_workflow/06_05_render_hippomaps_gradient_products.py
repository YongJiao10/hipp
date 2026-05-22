#!/usr/bin/env python3
from __future__ import annotations

"""
Process Step Description:
Render HippoMaps gradient products. Renders computed continuous functional gradient data into standard visualization products (including unfolded/folded surface renderings, explained variance plots, and endpoint functional connectivity contrasts).

Main Flow Path:
- 9th step in the numbered pipeline sequence (first stage of presentation output generation, run in parallel with 06_01, 06_02, 06_03, 06_04, 06_06):
  01_01_run_hippunfold_t1w_local.py -> 01_02_check_hippunfold_outputs.py -> 02_01_prepare_group_inputs.py -> 02_02_build_qc_manifest.py -> 03_01/03_02 reference building -> 04_01/04_02/04_03 FC materialization -> 05_01 gradient computation -> 05_02 group atlas parcellation -> 06_05_render_hippomaps_gradient_products.py (this step) -> 07_01 group matrix overview.

Inputs:
- Command-line arguments:
  - --gradient-root: Root directory containing gradient data outputs (default: `outputs_core/work/hippomaps_gradients`).
  - --present-root: Root directory for saving presentation figures (default: `outputs_core/present/gradients`).
  - --hippomaps-root: Root directory for the HippoMaps repository package (default: `external/HippoMaps`).
  - --routes: Routes to process (choices: "routea", "routeb", default: ["routea", "routeb"]).
  - --smoothings: Smoothing kernel values to process (default: ["2mm", "4mm"]).
  - --den: Output density resolution (default: "512").
  - --components: Comma-separated list of gradient components to plot (default: "1,2,3").
  - --hemis: Hemispheres to process (default: ["L", "R"]).
- Physical inputs:
  - Gradient output arrays and manifests: `{gradient_root}/{route}/{smoothing}/hemi_{hemi}/hippomaps_gradient_manifest.json` and matching `*.npy`/`*.gii` files.
  - Group-level FC matrix: `{routes_root}/intrinsic_group_fc_smooth-{smoothing}_hemi-{hemi}.npy` or matching path.

Outputs:
- File outputs:
  - Folded/Unfolded gradient surface rendering PNG: `{present_root}/combos/{route}_{smoothing}_{out_tag}/hippomaps_gradients_{route}_{smoothing}_{out_tag}.png`
  - Explained variance fraction plot PNG: `{present_root}/combos/{route}_{smoothing}_{out_tag}/hippomaps_ev_plot_{route}_{smoothing}_{out_tag}.png`
  - Endpoint FC contrast plot PNG: `{present_root}/combos/{route}_{smoothing}_{out_tag}/hippomaps_endpoint_fc_contrast_{route}_{smoothing}_{out_tag}.png`
  - Manifest JSON: `{present_root}/combos/{route}_{smoothing}_{out_tag}/hippomaps_gradient_products_{route}_{smoothing}_{out_tag}_manifest.json`
"""
import argparse
import json
import os
import tempfile
import time
from pathlib import Path
from typing import Any, Callable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from lib.hippomaps_adapter import DEFAULT_HIPPOMAPS_ROOT, ensure_hippomaps_on_path, hippomaps_commit
from lib.output_layout import DEFAULT_GRADIENTS_DIR, DEFAULT_HIPPOMAPS_GRADIENT_ROOT
from lib.workflow_utils import write_json
from lib.traceability import StageSpec, run_stage


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_GRADIENT_ROOT = DEFAULT_HIPPOMAPS_GRADIENT_ROOT
DEFAULT_PRESENT_ROOT = DEFAULT_GRADIENTS_DIR
ROUTES = ("routea", "routeb")
SMOOTHINGS = ("2mm", "4mm")
HEMIS = ("L", "R")




def wait_for_stable_file(path: Path, *, checks: int = 3, interval: float = 0.15) -> None:
    last: tuple[int, int] | None = None
    stable = 0
    for _ in range(max(1, checks * 10)):
        if path.exists():
            st = path.stat()
            current = (int(st.st_size), int(st.st_mtime_ns))
            if current == last and current[0] > 0:
                stable += 1
                if stable >= checks:
                    return
            else:
                stable = 0
                last = current
        time.sleep(float(interval))
    if not path.exists() or path.stat().st_size <= 0:
        raise TypeError(f"Render target did not stabilize: {path}")

def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


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


def normalize_hemis(hemis: list[str] | tuple[str, ...]) -> list[str]:
    out = [str(hemi).upper() for hemi in hemis]
    if not out:
        raise ValueError("At least one hemisphere is required")
    invalid = [hemi for hemi in out if hemi not in HEMIS]
    if invalid:
        raise ValueError(f"Invalid hemisphere(s): {invalid}; expected one or more of {HEMIS}")
    return out


def hemi_output_tag(hemis: list[str]) -> str:
    return "bilateral" if tuple(hemis) == HEMIS else "hemi-" + "".join(hemis)


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
    out = Image.new("RGB", (src_w + margin_w, src_h), (255, 255, 255))
    out.paste(img, (margin_w, 0))
    draw = ImageDraw.Draw(out)
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 22)
        small_font = ImageFont.truetype("DejaVuSans.ttf", 16)
    except Exception:
        font = ImageFont.load_default()
        small_font = font

    draw.text((18, 14), title, fill=(0, 0, 0), font=font)
    n_rows = max(1, len(selected_components))
    row_h = src_h / float(n_rows)
    for idx, comp in enumerate(selected_components):
        y = int(round((idx + 0.5) * row_h))
        label = f"G{int(comp)}"
        ev_text = f"EV={float(ev_percentages[idx]):.1f}%"
        draw.text((18, y - 18), label, fill=(0, 0, 0), font=font)
        draw.text((18, y + 8), ev_text, fill=(0, 0, 0), font=small_font)
    out.save(image_path)
    return image_path


def stitch_hemi_gradient_images(left_png: Path, right_png: Path, output_png: Path) -> Path:
    left_img = Image.open(left_png).convert("RGB")
    right_img = Image.open(right_png).convert("RGB")
    out = Image.new(
        "RGB",
        (left_img.size[0] + right_img.size[0], max(left_img.size[1], right_img.size[1])),
        (255, 255, 255),
    )
    out.paste(left_img, (0, 0))
    out.paste(right_img, (left_img.size[0], 0))
    output_png.parent.mkdir(parents=True, exist_ok=True)
    out.save(output_png)
    return output_png


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


def enable_den512_plotting_v2(hippomaps_root: Path) -> bool:
    ensure_hippomaps_on_path(hippomaps_root)
    import hippomaps.plotting as hm_plotting

    v2_densities = getattr(hm_plotting, "_V2_DENS", None)
    if not isinstance(v2_densities, set):
        raise TypeError("Expected hippomaps.plotting._V2_DENS to be a set")
    if "512" in v2_densities:
        return False
    v2_densities.add("512")
    return True


def load_hippomaps_plotter(hippomaps_root: Path = DEFAULT_HIPPOMAPS_ROOT) -> Callable[..., object]:
    ensure_hippomaps_on_path(hippomaps_root)
    enable_den512_plotting_v2(hippomaps_root)
    from hippomaps.plotting import surfplot_canonical_foldunfold

    return surfplot_canonical_foldunfold


def render_gradient_figure(
    *,
    left: np.ndarray,
    right: np.ndarray | None = None,
    output_png: Path,
    den: str = "512",
    hemis: list[str] | tuple[str, ...] = HEMIS,
    plotter: Callable[..., object] | None = None,
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
    output_png.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        prefix=f".{output_png.stem}.", suffix=output_png.suffix, dir=output_png.parent, delete=False
    ) as tmp:
        render_png = Path(tmp.name)
    fig = plotter(
        cdata,
        hemis=hemis,
        labels=["hipp"],
        den=str(den),
        unfoldAPrescale=unfold_ap_rescale,
        tighten_cwindow=True,
        cmap="plasma",
        share="row",
        color_bar="right",
        background=(1, 1, 1),
        transparent_bg=False,
        interactive=False,
        screenshot=True,
        filename=str(render_png),
        suppress_warnings=True,
        size=[450, 300],
        cb__width=0.95,
        cb__position=[0.02, 0.25],
        cb__barRatio=0.18,
        cb__labelTextProperty={"fontSize": 13},
    )
    final_render = render_png
    if fig is None:
        if not render_png.exists() or render_png.stat().st_size <= 0:
            raise TypeError("HippoMaps plotter returned None and did not create the requested screenshot")
    elif isinstance(fig, str):
        got = Path(fig)
        if got.exists() and got.stat().st_size > 0:
            final_render = got
        elif not render_png.exists() or render_png.stat().st_size <= 0:
            raise TypeError(f"HippoMaps plotter returned screenshot path but file is missing: {fig}")
    elif hasattr(fig, "savefig"):
        fig.savefig(render_png, dpi=220, bbox_inches="tight")
        try:
            plt.close(fig)
        except TypeError:
            pass
    elif hasattr(fig, "screenshot"):
        fig.screenshot(str(render_png))
    else:
        raise TypeError(f"Unsupported HippoMaps plotter return type: {type(fig)!r}")
    if not final_render.exists() or final_render.stat().st_size <= 0:
        raise TypeError(f"HippoMaps render target was not created: {final_render}")
    os.replace(final_render, output_png)
    wait_for_stable_file(output_png)
    return output_png


def render_bilateral_gradient_figure(
    *,
    left: np.ndarray,
    right: np.ndarray,
    output_png: Path,
    den: str = "512",
    selected_components: list[int],
    ev_by_hemi: dict[str, list[float]],
    plotter: Callable[..., object] | None = None,
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
    )
    render_gradient_figure(
        left=np.empty((0, np.asarray(right).shape[1]), dtype=np.float32),
        right=right,
        output_png=side_paths["R"],
        den=den,
        hemis=["R"],
        plotter=plotter,
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


def endpoint_fc_contrast(
    *,
    gradient_full: np.ndarray,
    group_fc_full: np.ndarray,
    support_mask: np.ndarray,
    endpoint_fraction: float = 0.25,
) -> np.ndarray:
    grad = np.asarray(gradient_full, dtype=np.float32).reshape(-1)
    fc = np.asarray(group_fc_full, dtype=np.float32)
    support = np.asarray(support_mask, dtype=bool).reshape(-1)
    if fc.ndim != 2:
        raise ValueError(f"Expected 2D group FC full matrix, got shape={fc.shape}")
    if grad.size != support.size or fc.shape[0] != support.size:
        raise ValueError(f"Shape mismatch: gradient={grad.shape} fc={fc.shape} support={support.shape}")
    finite = support & np.isfinite(grad) & np.all(np.isfinite(fc), axis=1)
    vals = grad[finite]
    if vals.size == 0:
        raise RuntimeError("No finite supported vertices available for endpoint contrast")
    frac = float(endpoint_fraction)
    if frac <= 0.0 or frac >= 0.5:
        raise ValueError(f"endpoint_fraction must be within (0, 0.5), got {endpoint_fraction}")
    lo = float(np.nanquantile(vals, frac))
    hi = float(np.nanquantile(vals, 1.0 - frac))
    finite_idx = np.flatnonzero(finite)
    bottom_idx = finite_idx[grad[finite] <= lo]
    top_idx = finite_idx[grad[finite] >= hi]
    if bottom_idx.size == 0 or top_idx.size == 0:
        raise RuntimeError(f"Endpoint selection is empty: bottom={bottom_idx.size} top={top_idx.size}")
    return (np.nanmean(fc[top_idx, :], axis=0) - np.nanmean(fc[bottom_idx, :], axis=0)).astype(np.float32)


def render_ev_plot(manifests: dict[str, dict[str, Any]], output_png: Path) -> Path:
    fig, ax = plt.subplots(figsize=(7.0, 3.8))
    for hemi, manifest in sorted(manifests.items()):
        ev = np.asarray(manifest["ev_fractions"], dtype=np.float64) * 100.0
        ax.plot(np.arange(1, ev.size + 1), ev, marker="o", linewidth=1.5, label=f"hemi {hemi}")
    ax.set_xlabel("Gradient")
    ax.set_ylabel("EV (%)")
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False)
    output_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_png, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return output_png


def render_endpoint_contrast_plot(contrasts: dict[tuple[str, str], np.ndarray], output_png: Path) -> Path:
    keys = sorted(contrasts.keys())
    max_len = max(int(contrasts[k].size) for k in keys)
    mat = np.full((len(keys), max_len), np.nan, dtype=np.float32)
    for row, key in enumerate(keys):
        vec = contrasts[key]
        mat[row, : vec.size] = vec
    vmax = float(np.nanmax(np.abs(mat))) if np.any(np.isfinite(mat)) else 1.0
    if not np.isfinite(vmax) or vmax <= 0.0:
        vmax = 1.0
    fig, ax = plt.subplots(figsize=(8.0, 0.65 * len(keys) + 2.2))
    im = ax.imshow(mat, aspect="auto", cmap="coolwarm", vmin=-vmax, vmax=vmax)
    ax.set_yticks(np.arange(len(keys)))
    ax.set_yticklabels([f"{hemi} {comp}" for hemi, comp in keys])
    ax.set_xlabel("Kong2022 ROI")
    ax.set_ylabel("Endpoint")
    ax.spines[["top", "right"]].set_visible(False)
    fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02, label="Top 25% - bottom 25% Fisher-z FC")
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
) -> Path:
    hemis = normalize_hemis(hemis)
    combo_root = present_root / route / smoothing
    manifests: dict[str, dict[str, Any]] = {}
    gradients_by_hemi: dict[str, np.ndarray] = {}
    support_by_hemi: dict[str, np.ndarray] = {}
    group_fc_by_hemi: dict[str, np.ndarray] = {}
    for hemi in hemis:
        run_dir = gradient_root / route / smoothing / f"hemi_{hemi}"
        manifest_path = run_dir / "hippomaps_gradient_manifest.json"
        if not manifest_path.exists():
            raise FileNotFoundError(f"Missing HippoMaps gradient manifest: {manifest_path}")
        manifests[hemi] = load_json(manifest_path)
        gradients_by_hemi[hemi] = np.load(run_dir / "hippomaps_gradients_full.npy").astype(np.float32)
        support_by_hemi[hemi] = np.load(run_dir / "group_support_mask.npy").astype(bool)
        group_fc_by_hemi[hemi] = np.load(run_dir / "group_mean_fisherz_fc_full.npy").astype(np.float32)

    comp_indices = [int(c) - 1 for c in selected_components]
    left = gradients_by_hemi["L"][:, comp_indices] if "L" in hemis else np.empty((0, len(comp_indices)), dtype=np.float32)
    right = gradients_by_hemi["R"][:, comp_indices] if "R" in hemis else None
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
        )
        hemi = hemis[0]
        annotate_gradient_image(
            gradient_png,
            selected_components=selected_components,
            ev_percentages=ev_by_hemi[hemi],
            title=f"{hemi} Gradient",
        )

    ev_png = render_ev_plot(manifests, combo_root / f"hippomaps_ev_{route}_{smoothing}_{out_tag}.png")
    contrasts: dict[tuple[str, str], np.ndarray] = {}
    for hemi in hemis:
        for comp in selected_components:
            contrasts[(hemi, f"G{comp}")] = endpoint_fc_contrast(
                gradient_full=gradients_by_hemi[hemi][:, int(comp) - 1],
                group_fc_full=group_fc_by_hemi[hemi],
                support_mask=support_by_hemi[hemi],
                endpoint_fraction=0.25,
            )
    contrast_png = render_endpoint_contrast_plot(
        contrasts,
        combo_root / f"hippomaps_endpoint_fc_contrast_{route}_{smoothing}_{out_tag}.png",
    )
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
            "source_manifests": {hemi: str((gradient_root / route / smoothing / f"hemi_{hemi}" / "hippomaps_gradient_manifest.json").resolve()) for hemi in hemis},
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
                "tighten_cwindow": True,
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
            "endpoint_contrast": "top_25pct_minus_bottom_25pct_gradient_vertices_by_kong2022_roi",
            "outputs": {
                "gradient_figure": str(gradient_png.resolve()),
                "ev_plot": str(ev_png.resolve()),
                "endpoint_fc_contrast_plot": str(contrast_png.resolve()),
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
    parser.add_argument("--hippomaps-root", default=str(DEFAULT_HIPPOMAPS_ROOT))
    parser.add_argument("--routes", nargs="+", choices=ROUTES, default=list(ROUTES))
    parser.add_argument("--smoothings", nargs="+", choices=SMOOTHINGS, default=list(SMOOTHINGS))
    parser.add_argument("--den", default="512")
    parser.add_argument("--components", default="1,2,3")
    parser.add_argument("--hemis", nargs="+", choices=HEMIS, default=list(HEMIS))
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    hippomaps_root = ensure_hippomaps_on_path(Path(args.hippomaps_root))
    components = parse_components(args.components)
    hemis = normalize_hemis(args.hemis)
    for route in args.routes:
        for smoothing in args.smoothings:
            render_combo(
                route=route,
                smoothing=smoothing,
                gradient_root=Path(args.gradient_root).resolve(),
                present_root=Path(args.present_root).resolve(),
                hippomaps_root=hippomaps_root,
                den=str(args.den),
                selected_components=components,
                hemis=hemis,
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(
        run_stage(
            StageSpec(
                step_id="06_05_render_hippomaps_gradient_products",
                purpose="Render HippoMaps folded/unfolded gradient figures, EV plots, and endpoint FC contrasts.",
                stage_main=main,
                upstream_defaults=("05_01_compute_hippomaps_gradients",),
            )
        )
    )
