#!/usr/bin/env python3
from __future__ import annotations

"""
Render group-average local cortex-neighbor counts.

This diagnostic summarizes, for each hippocampal surface vertex, how many valid
cortical vertices were inside the 20 mm local-control radius used by the RouteA
ctxreg and ctxdrop branches. Inputs can come from either branch's per-subject
neighbor-count diagnostics.
"""

import argparse
import csv
import json
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

from lib.hippomaps_adapter import DEFAULT_HIPPOMAPS_ROOT, ensure_hippomaps_on_path
from lib.io_utils import load_subject_ids
from lib.output_layout import (
    DEFAULT_KONG2022_MANIFEST_ROOT,
    DEFAULT_KONG400_CTXDROP_SHARED_ROOT,
    DEFAULT_KONG400_CTXREG_SHARED_ROOT,
    DEFAULT_LOCAL_CORTEX_NEIGHBORS_DIR,
)
from lib.workflow_utils import utc_now_iso, write_json


DEFAULT_SHARED_ROOT = DEFAULT_KONG400_CTXDROP_SHARED_ROOT
DEFAULT_SUBJECTS = DEFAULT_KONG2022_MANIFEST_ROOT / "hcp_7t_hippocampus_fmri_qc_pass_t1only_latest.txt"
DEFAULT_OUT_DIR = DEFAULT_LOCAL_CORTEX_NEIGHBORS_DIR
HEMIS = ("L", "R")
SOURCE_BRANCHES = ("routea_ctxdrop", "routea_ctxreg")


def resolve_neighbor_shared_root(
    shared_root: str | Path | None,
    *,
    source_branch: str,
) -> Path:
    if shared_root is not None:
        return Path(shared_root).resolve()
    if source_branch == "routea_ctxreg":
        return DEFAULT_KONG400_CTXREG_SHARED_ROOT.resolve()
    if source_branch == "routea_ctxdrop":
        return DEFAULT_KONG400_CTXDROP_SHARED_ROOT.resolve()
    raise ValueError(f"Unsupported source_branch: {source_branch!r}")


@dataclass(frozen=True)
class HemiSummary:
    hemi: str
    count_paths: list[Path]
    mean: np.ndarray
    sd: np.ndarray
    n_subjects: np.ndarray


def count_path(shared_root: Path, subject: str, hemi: str, source_branch: str = "routea_ctxdrop") -> Path:
    subject = str(subject).replace("sub-", "")
    if source_branch not in SOURCE_BRANCHES:
        raise ValueError(f"Unsupported source_branch: {source_branch}")
    return shared_root / f"sub-{subject}" / "surface" / "local_cortex_neighbors" / f"hemi_{hemi}" / "neighbor_counts.npy"


def load_hemi_summary(shared_root: Path, subjects: list[str], hemi: str, source_branch: str) -> HemiSummary:
    arrays: list[np.ndarray] = []
    paths: list[Path] = []
    for subject in subjects:
        path = count_path(shared_root, subject, hemi, source_branch=source_branch)
        if not path.exists():
            continue
        arr = np.load(path).astype(np.float32).reshape(-1)
        arrays.append(arr)
        paths.append(path.resolve())
    if not arrays:
        raise FileNotFoundError(f"No local cortex neighbor count arrays found for source_branch={source_branch} hemi={hemi} under {shared_root}")
    n_vertices = int(arrays[0].size)
    for path, arr in zip(paths, arrays, strict=True):
        if int(arr.size) != n_vertices:
            raise ValueError(f"Vertex count mismatch: {path} size={arr.size} expected={n_vertices}")
    stack = np.stack(arrays, axis=0).astype(np.float32)
    return HemiSummary(
        hemi=hemi,
        count_paths=paths,
        mean=np.nanmean(stack, axis=0).astype(np.float32),
        sd=np.nanstd(stack, axis=0).astype(np.float32),
        n_subjects=np.sum(np.isfinite(stack), axis=0).astype(np.int32),
    )


def output_paths(out_dir: Path, *, distance_mm: float) -> dict[str, Path]:
    distance_tag = int(float(distance_mm))
    return {
        "mean_array": out_dir / "group_mean_local_cortex_neighbors_hemi-{hemi}.npy",
        "sd_array": out_dir / "group_sd_local_cortex_neighbors_hemi-{hemi}.npy",
        "n_array": out_dir / "group_n_local_cortex_neighbors_hemi-{hemi}.npy",
        "tsv": out_dir / "group_mean_local_cortex_neighbors.tsv",
        "png": out_dir / f"kong2022_routea_local_cortex_neighbors_mean{distance_tag}mm.png",
        "manifest": out_dir / f"kong2022_routea_local_cortex_neighbors_mean{distance_tag}mm_manifest.json",
    }


def write_long_tsv(path: Path, summaries: list[HemiSummary]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["hemi", "vertex_index", "mean_neighbor_count", "sd_neighbor_count", "n_subjects"], delimiter="\t")
        writer.writeheader()
        for summary in summaries:
            for idx, value in enumerate(summary.mean.tolist()):
                writer.writerow(
                    {
                        "hemi": summary.hemi,
                        "vertex_index": int(idx),
                        "mean_neighbor_count": float(value),
                        "sd_neighbor_count": float(summary.sd[idx]),
                        "n_subjects": int(summary.n_subjects[idx]),
                    }
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


def annotate_overlay_image(
    image_path: Path,
    *,
    mean_value: float,
    display_lower: float,
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
    draw.text((18, 50 + padding_h), f"Mean = {mean_value:.1f}", fill=(0, 0, 0), font=small_font)
    draw.text((18, 73 + padding_h), f"Display {display_lower:g}-{display_upper:g}", fill=(0, 0, 0), font=small_font)
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


def render_hemi_overlay(
    *,
    values: np.ndarray,
    hemi: str,
    output_png: Path,
    color_range: tuple[float, float],
    den: str,
    plotter: Callable[..., object],
) -> Path:
    cdata = np.asarray(values, dtype=np.float32).reshape(-1, 1)
    output_png.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        prefix=f".{output_png.stem}.", suffix=output_png.suffix, dir=output_png.parent, delete=False
    ) as tmp:
        render_png = Path(tmp.name)
    fig = plotter(
        cdata,
        hemis=[hemi],
        labels=["hipp"],
        den=str(den),
        unfoldAPrescale=str(den) != "512",
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
        cb__labelFormat="%.0f",
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


def render_overlay_png(
    path: Path,
    summaries: list[HemiSummary],
    *,
    title: str,
    den: str = "512",
    display_vmin: float | None = None,
    display_vmax: float | None = None,
    plotter: Callable[..., object] | None = None,
) -> Path:
    by_hemi = {summary.hemi: summary for summary in summaries}
    missing = [hemi for hemi in HEMIS if hemi not in by_hemi]
    if missing:
        raise ValueError(f"Overlay rendering requires both hemispheres; missing {missing}")
    if plotter is None:
        plotter = load_hippomaps_plotter()
    finite_values = np.concatenate([summary.mean[np.isfinite(summary.mean)] for summary in summaries])
    if finite_values.size:
        data_min = float(np.nanmin(finite_values))
        data_max = float(np.nanmax(finite_values))
    else:
        data_min = 0.0
        data_max = 1.0
    display_lower = float(display_vmin) if display_vmin is not None else data_min
    display_upper = float(display_vmax) if display_vmax is not None else data_max
    if not np.isfinite(display_lower):
        display_lower = 0.0
    if not np.isfinite(display_upper):
        display_upper = display_lower + 1.0
    if display_upper <= display_lower:
        display_upper = display_lower + 1.0
    color_range = (float(display_lower), float(display_upper))
    side_paths = {
        "L": path.with_name(f"{path.stem}_hemi-L{path.suffix}"),
        "R": path.with_name(f"{path.stem}_hemi-R{path.suffix}"),
    }
    for hemi in HEMIS:
        summary = by_hemi[hemi]
        render_hemi_overlay(
            values=summary.mean,
            hemi=hemi,
            output_png=side_paths[hemi],
            color_range=color_range,
            den=den,
            plotter=plotter,
        )
        annotate_overlay_image(
            side_paths[hemi],
            mean_value=float(np.nanmean(summary.mean)),
            display_lower=display_lower,
            display_upper=display_upper,
            title=f"{hemi} Local Cortex Neighbors",
        )
    return stitch_hemi_images(side_paths["L"], side_paths["R"], path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render group-average RouteA 20 mm local cortex-neighbor count summaries.")
    parser.add_argument("--shared-root", default=None, help="Neighbor-count input root; omitted paths follow --source-branch's dated 400-ROI control root.")
    parser.add_argument("--subjects-manifest", default=str(DEFAULT_SUBJECTS))
    parser.add_argument("--subjects", nargs="*", default=None)
    parser.add_argument("--hemis", nargs="+", default=list(HEMIS), choices=HEMIS)
    parser.add_argument("--distance-mm", type=float, default=20.0)
    parser.add_argument("--source-branch", default="routea_ctxdrop", choices=SOURCE_BRANCHES)
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    parser.add_argument("--hippomaps-root", default=str(DEFAULT_HIPPOMAPS_ROOT))
    parser.add_argument("--den", default="512")
    parser.add_argument("--display-vmin", type=float, default=None)
    parser.add_argument("--display-vmax", type=float, default=None)
    return parser.parse_args()


def parse_args_for_test(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render group-average RouteA 20 mm local cortex-neighbor count summaries.")
    parser.add_argument("--shared-root", default=None, help="Neighbor-count input root; omitted paths follow --source-branch's dated 400-ROI control root.")
    parser.add_argument("--subjects-manifest", default=str(DEFAULT_SUBJECTS))
    parser.add_argument("--subjects", nargs="*", default=None)
    parser.add_argument("--hemis", nargs="+", default=list(HEMIS), choices=HEMIS)
    parser.add_argument("--distance-mm", type=float, default=20.0)
    parser.add_argument("--source-branch", default="routea_ctxdrop", choices=SOURCE_BRANCHES)
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    parser.add_argument("--hippomaps-root", default=str(DEFAULT_HIPPOMAPS_ROOT))
    parser.add_argument("--den", default="512")
    parser.add_argument("--display-vmin", type=float, default=None)
    parser.add_argument("--display-vmax", type=float, default=None)
    return parser.parse_args(argv)


def manifest_payload(
    *,
    summaries: list[HemiSummary],
    subjects: list[str],
    subjects_manifest: Path,
    source_branch: str,
    distance_mm: float,
    display_vmin: float,
    display_vmax: float,
    hippomaps_root: Path,
    outputs: dict[str, object],
) -> dict[str, object]:
    return {
        "stage": "06_07_render_local_cortex_neighbors",
        "status": "done",
        "timestamp_utc": utc_now_iso(),
        "route_tag": "routea",
        "source_branch": str(source_branch),
        "distance_mm": float(distance_mm),
        "plot_type": "hippocampal_surface_overlay",
        "display_vmin": float(display_vmin),
        "display_vmax": float(display_vmax),
        "display_range_semantics": "finite_data_min_to_finite_data_max_unless_overridden",
        "hippomaps_root": str(Path(hippomaps_root).resolve()),
        "subjects_manifest": str(Path(subjects_manifest).resolve()),
        "n_subjects_requested": int(len(subjects)),
        "hemis": [summary.hemi for summary in summaries],
        "inputs": {summary.hemi: [str(path) for path in summary.count_paths] for summary in summaries},
        "outputs": outputs,
    }


def main() -> int:
    args = parse_args()
    shared_root = resolve_neighbor_shared_root(args.shared_root, source_branch=str(args.source_branch))
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    subjects = [s.replace("sub-", "") for s in args.subjects] if args.subjects else load_subject_ids(Path(args.subjects_manifest).resolve())
    summaries = [load_hemi_summary(shared_root, subjects, str(hemi), str(args.source_branch)) for hemi in args.hemis]

    paths = output_paths(out_dir, distance_mm=float(args.distance_mm))
    for summary in summaries:
        np.save(Path(str(paths["mean_array"]).format(hemi=summary.hemi)), summary.mean)
        np.save(Path(str(paths["sd_array"]).format(hemi=summary.hemi)), summary.sd)
        np.save(Path(str(paths["n_array"]).format(hemi=summary.hemi)), summary.n_subjects)
    tsv_path = paths["tsv"]
    png_path = paths["png"]
    manifest_path = paths["manifest"]
    write_long_tsv(tsv_path, summaries)
    plotter = load_hippomaps_plotter(Path(args.hippomaps_root).resolve())
    render_overlay_png(
        png_path,
        summaries,
        title=f"RouteA group mean cortex vertices within {float(args.distance_mm):.0f} mm",
        den=str(args.den),
        display_vmin=args.display_vmin,
        display_vmax=args.display_vmax,
        plotter=plotter,
    )
    write_json(
        manifest_path,
        manifest_payload(
            summaries=summaries,
            subjects=subjects,
            subjects_manifest=Path(args.subjects_manifest),
            source_branch=str(args.source_branch),
            distance_mm=float(args.distance_mm),
            display_vmin=float(min(float(np.nanmin(summary.mean)) for summary in summaries)) if args.display_vmin is None else float(args.display_vmin),
            display_vmax=float(max(float(np.nanmax(summary.mean)) for summary in summaries)) if args.display_vmax is None else float(args.display_vmax),
            hippomaps_root=Path(args.hippomaps_root),
            outputs={
                "png": str(png_path),
                "side_pngs": {
                    "L": str(png_path.with_name(f"{png_path.stem}_hemi-L{png_path.suffix}")),
                    "R": str(png_path.with_name(f"{png_path.stem}_hemi-R{png_path.suffix}")),
                },
                "tsv": str(tsv_path),
                "manifest": str(manifest_path),
                "arrays": {
                    summary.hemi: {
                        "mean": str(Path(str(paths["mean_array"]).format(hemi=summary.hemi))),
                        "sd": str(Path(str(paths["sd_array"]).format(hemi=summary.hemi))),
                        "n": str(Path(str(paths["n_array"]).format(hemi=summary.hemi))),
                    }
                    for summary in summaries
                },
            },
        ),
        trailing_newline=True,
    )
    print(json.dumps({"status": "ok", "png": str(png_path), "manifest": str(manifest_path)}, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
