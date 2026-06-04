#!/usr/bin/env python3
from __future__ import annotations

"""
Render gradient-spectral confidence maps on folded and unfolded hippocampal surfaces.
Raw silhouette arrays remain signed; rendered figures use a Yeo/Buckner-style
nonnegative display window from 0 to an automatically selected upper bound.
"""

import argparse
import os
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from lib.confidence_map import publication_display_data, publication_display_upper
from lib.hippomaps_adapter import DEFAULT_HIPPOMAPS_ROOT, ensure_hippomaps_on_path
from lib.output_layout import DEFAULT_GRADIENTS_DIR, DEFAULT_ROUTEA_GROUP_ATLAS_ROOT
from lib.route_tags import ROUTE_TAG_CHOICES, infer_route_tag_from_group_root, validate_group_atlas_root_for_route
from lib.traceability import StageSpec, run_stage
from lib.io_utils import load_json

HEMIS = ("L", "R")
SMOOTHINGS = ("2mm", "4mm")


@dataclass(frozen=True)
class ConfidenceProduct:
    name: str
    title: str
    array_name: str
    summary_name: str


CONFIDENCE_PRODUCTS = (
    ConfidenceProduct(
        name="gradient_feature",
        title="Gradient-feature Silhouette",
        array_name="confidence_silhouette_gradient_feature.npy",
        summary_name="confidence_silhouette_gradient_feature_summary.json",
    ),
)


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


def normalize_hemis(hemis: list[str] | tuple[str, ...]) -> list[str]:
    out = [str(hemi).upper() for hemi in hemis]
    if not out:
        raise ValueError("At least one hemisphere is required")
    invalid = [hemi for hemi in out if hemi not in HEMIS]
    if invalid:
        raise ValueError(f"Invalid hemisphere(s): {invalid}; expected one or more of {HEMIS}")
    return out


def annotate_confidence_image(
    image_path: Path,
    *,
    mean_value: float,
    display_upper: float,
    title: str,
) -> Path:
    img = Image.open(image_path).convert("RGB")
    src_w, src_h = img.size
    margin_w = 250
    padding_h = 30
    out = Image.new("RGB", (src_w + margin_w, src_h + 2 * padding_h), (255, 255, 255))
    out.paste(img, (margin_w, padding_h))
    draw = ImageDraw.Draw(out)
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 21)
        small_font = ImageFont.truetype("DejaVuSans.ttf", 15)
    except Exception:
        font = ImageFont.load_default()
        small_font = font

    draw.text((18, 14 + padding_h), title, fill=(0, 0, 0), font=font)
    draw.text((18, 50 + padding_h), f"Mean = {mean_value:.3f}", fill=(0, 0, 0), font=small_font)
    draw.text((18, 73 + padding_h), f"Display 0-{display_upper:g}", fill=(0, 0, 0), font=small_font)
    out.save(image_path)
    return image_path


def stitch_hemi_images(left_png: Path, right_png: Path, output_png: Path) -> Path:
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


def render_confidence_figure(
    *,
    left: np.ndarray,
    right: np.ndarray | None = None,
    output_png: Path,
    color_range: tuple[float, float],
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
            raise ValueError("right array is required when rendering hemisphere R")
        arrays.append(np.asarray(right, dtype=np.float32))
    if len(arrays) == 2 and arrays[0].shape != arrays[1].shape:
        raise ValueError(f"Left/right shape mismatch: {arrays[0].shape} vs {arrays[1].shape}")
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
        tighten_cwindow=False,
        cmap="viridis",
        color_range=color_range,
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
        zoom=1.45,
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


def render_bilateral_confidence_figure(
    *,
    left: np.ndarray,
    right: np.ndarray,
    output_png: Path,
    den: str = "512",
    means: dict[str, float],
    title: str,
    display_vmax: float | None,
    plotter: Callable[..., object] | None = None,
) -> Path:
    output_png.parent.mkdir(parents=True, exist_ok=True)
    if plotter is None:
        plotter = load_hippomaps_plotter()
    display_upper = publication_display_upper(np.concatenate([left.ravel(), right.ravel()]), vmax=display_vmax)
    left_display = publication_display_data(left, vmax=display_upper)
    right_display = publication_display_data(right, vmax=display_upper)
    color_range = (0.0, float(display_upper))
    side_paths = {
        "L": output_png.with_name(f"{output_png.stem}_hemi-L{output_png.suffix}"),
        "R": output_png.with_name(f"{output_png.stem}_hemi-R{output_png.suffix}"),
    }
    render_confidence_figure(
        left=left_display,
        output_png=side_paths["L"],
        den=den,
        hemis=["L"],
        color_range=color_range,
        plotter=plotter,
    )
    render_confidence_figure(
        left=np.empty((0,), dtype=np.float32),
        right=right_display,
        output_png=side_paths["R"],
        den=den,
        hemis=["R"],
        color_range=color_range,
        plotter=plotter,
    )
    for hemi in HEMIS:
        wait_for_stable_file(side_paths[hemi])
        annotate_confidence_image(
            side_paths[hemi],
            mean_value=means[hemi],
            display_upper=display_upper,
            title=f"{hemi} {title}",
        )
    return stitch_hemi_images(side_paths["L"], side_paths["R"], output_png)


def _product_file(run_dir: Path, product: ConfidenceProduct, *, summary: bool) -> Path:
    return run_dir / (product.summary_name if summary else product.array_name)


def render_product(
    *,
    product: ConfidenceProduct,
    route_tag: str,
    smoothing: str,
    out_root: Path,
    output_dir: Path,
    plotter: Callable[..., object],
    den: str,
    comp_dir_name: str,
    thresh_str: str,
    display_vmax: float | None,
) -> Path:
    arrays: dict[str, np.ndarray] = {}
    means: dict[str, float] = {}
    for hemi in HEMIS:
        run_dir = (
            out_root
            / "gradient-spectral"
            / "kong2022"
            / smoothing
            / thresh_str
            / comp_dir_name
            / f"hemi_{hemi}"
        )
        sil_file = _product_file(run_dir, product, summary=False)
        sum_file = _product_file(run_dir, product, summary=True)
        if not sil_file.exists():
            raise FileNotFoundError(f"Missing silhouette array for {product.name}: {sil_file}")
        if not sum_file.exists():
            raise FileNotFoundError(f"Missing silhouette summary for {product.name}: {sum_file}")
        arrays[hemi] = np.load(sil_file)
        summary = load_json(sum_file)
        means[hemi] = float(summary["global_mean"])

    output_png = output_dir / f"confidence_silhouette_{product.name}_{route_tag}_{smoothing}_bilateral.png"
    render_bilateral_confidence_figure(
        left=arrays["L"],
        right=arrays["R"],
        output_png=output_png,
        den=den,
        means=means,
        title=product.title,
        display_vmax=display_vmax,
        plotter=plotter,
    )
    print(f"Saved {product.name} rendering to: {output_png}")
    return output_png


def render_combo(
    *,
    smoothing: str,
    route_tag: str,
    out_root: Path,
    present_root: Path,
    hippomaps_root: Path,
    den: str,
    comp_dir_name: str,
    thresh_str: str,
    display_vmax: float | None,
) -> None:
    print(f"Rendering confidence map for smoothing={smoothing}")
    output_dir = present_root / route_tag / smoothing
    output_dir.mkdir(parents=True, exist_ok=True)
    plotter = load_hippomaps_plotter(hippomaps_root)
    for product in CONFIDENCE_PRODUCTS:
        render_product(
            product=product,
            route_tag=route_tag,
            smoothing=smoothing,
            out_root=out_root,
            output_dir=output_dir,
            plotter=plotter,
            den=den,
            comp_dir_name=comp_dir_name,
            thresh_str=thresh_str,
            display_vmax=display_vmax,
        )


def main(args: argparse.Namespace) -> int:
    out_root = Path(args.out_root).resolve()
    if args.route_tag == "auto":
        args.route_tag = infer_route_tag_from_group_root(out_root)
    else:
        validate_group_atlas_root_for_route(out_root, str(args.route_tag))
    present_root = Path(args.present_root).resolve()
    hippomaps_root = Path(args.hippomaps_root).resolve()
    comp_str = args.gradient_components.strip()
    comp_dir_name = "g" + comp_str.replace(",", "g")

    for smoothing in args.smoothings:
        for thresh in args.coverage_thresholds:
            thresh_str = f"cov_{thresh:.2f}".replace(".", "p")
            try:
                render_combo(
                    smoothing=smoothing,
                    route_tag=str(args.route_tag),
                    out_root=out_root,
                    present_root=present_root,
                    hippomaps_root=hippomaps_root,
                    den=str(args.den),
                    comp_dir_name=comp_dir_name,
                    thresh_str=thresh_str,
                    display_vmax=args.display_vmax,
                )
            except Exception as exc:
                print(f"Failed rendering combo for smoothing={smoothing}: {exc}")
                import traceback
                traceback.print_exc()

    return 0


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Render functional parcellation confidence maps.")
    parser.add_argument("--out-root", default=str(DEFAULT_ROUTEA_GROUP_ATLAS_ROOT))
    parser.add_argument("--route-tag", default="auto", choices=["auto", *ROUTE_TAG_CHOICES])
    parser.add_argument("--present-root", default=str(DEFAULT_GRADIENTS_DIR))
    parser.add_argument("--hippomaps-root", default=str(DEFAULT_HIPPOMAPS_ROOT))
    parser.add_argument("--smoothings", nargs="+", default=list(SMOOTHINGS))
    parser.add_argument("--coverage-thresholds", nargs="+", type=float, default=[0.80])
    parser.add_argument(
        "--gradient-components",
        default="1,2,3",
        help="Comma-separated gradient components to process.",
    )
    parser.add_argument("--den", default="512", help="Surface density resolution.")
    parser.add_argument("--display-vmax", type=float, default=None, help="Override positive display upper bound; default auto.")
    return parser


if __name__ == "__main__":
    parser = build_arg_parser()
    args = parser.parse_args()
    raise SystemExit(
        run_stage(
            StageSpec(
                step_id="06_06_render_confidence_maps",
                purpose="Render functional parcellation confidence maps on folded and unfolded surfaces.",
                stage_main=lambda: main(args),
                upstream_defaults=("05_04_compute_confidence_maps",),
            )
        )
    )
