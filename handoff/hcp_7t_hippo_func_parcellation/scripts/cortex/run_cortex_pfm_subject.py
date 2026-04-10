#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib
import json
import os
from pathlib import Path
import subprocess
import sys

import nibabel as nib
import numpy as np
from PIL import Image, ImageDraw, ImageFont


REPO_ROOT = Path(__file__).resolve().parents[2]
WB_COMMAND = str((REPO_ROOT / "scripts" / "wb_command").resolve())
DEFAULT_FASTANS_ROOT = REPO_ROOT / "external" / "FASTANS"
DEFAULT_SCENE = REPO_ROOT / "config" / "manual_wb_scenes" / "cortex_manual.scene"


METHODS = {
    "Lynch2024": {
        "slug": "lynch2024",
        "display_name": "Lynch2024",
        "expected_count": 21,
        "labels_file": "Lynch2024_LabelList.txt",
    },
    "Kong2019": {
        "slug": "kong2019",
        "display_name": "Kong2019",
        "expected_count": 17,
        "labels_file": "Kong2019_LabelList.txt",
    },
}

VIEW_SPECS = [
    {"name": "lateral", "axis": None, "deg": 0.0},
    {"name": "medial", "axis": "y", "deg": -90.0},
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run FASTANS cortex PFM for one HCP subject and render Workbench screenshots."
    )
    parser.add_argument("--subject", required=True, help="Subject ID without sub- prefix")
    parser.add_argument("--dtseries", required=True)
    parser.add_argument("--left-midthickness", required=True)
    parser.add_argument("--right-midthickness", required=True)
    parser.add_argument("--left-inflated", required=True)
    parser.add_argument("--right-inflated", required=True)
    parser.add_argument("--sulc-dscalar", required=True)
    parser.add_argument("--fastans-root", default=str(DEFAULT_FASTANS_ROOT))
    parser.add_argument("--source-scene", default=str(DEFAULT_SCENE))
    parser.add_argument("--out-root", default=str(REPO_ROOT / "outputs" / "cortex_pfm"))
    parser.add_argument("--methods", nargs="+", default=["Lynch2024", "Kong2019"], choices=sorted(METHODS))
    return parser.parse_args()


def run(cmd: list[str]) -> None:
    proc = subprocess.run(cmd, text=True, capture_output=True)
    if proc.returncode != 0:
        raise RuntimeError(f"Command failed: {' '.join(cmd)}\nSTDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}")


def import_fastans(fastans_root: Path):
    sys.path.insert(0, str((fastans_root / "code").resolve()))
    fastans = importlib.import_module("FASTANS")
    fastans.FASTANS_installation_folderpath = str(fastans_root.resolve())
    fastans.resources_folderpath = str((fastans_root / "resources").resolve())
    return fastans


def parse_label_list(label_file: Path) -> list[str]:
    lines = [line.strip() for line in label_file.read_text(encoding="utf-8").splitlines() if line.strip()]
    return lines[0::2]


def parse_label_legend(label_file: Path) -> list[dict[str, object]]:
    lines = [line.strip() for line in label_file.read_text(encoding="utf-8").splitlines() if line.strip()]
    legend: list[dict[str, object]] = []
    for idx in range(0, len(lines), 2):
        name = lines[idx]
        parts = lines[idx + 1].split()
        if len(parts) != 5:
            raise ValueError(f"Unexpected label row in {label_file}: {lines[idx + 1]}")
        rgba = tuple(int(value) for value in parts[1:])
        legend.append({"name": name, "rgba": rgba})
    return legend


def load_label_counts(dlabel_path: Path) -> dict[str, int]:
    img = nib.load(str(dlabel_path))
    values = np.asarray(img.get_fdata()).astype(int).ravel()
    label_axis = img.header.get_axis(0)
    label_table = label_axis.label[0]
    counts: dict[str, int] = {}
    for value in sorted(np.unique(values)):
        if value == 0:
            continue
        label = label_table.get(int(value))
        name = label[0] if label else str(value)
        counts[name] = int(np.sum(values == value))
    return counts


def separate_labels(dlabel_path: Path, method_dir: Path) -> tuple[Path, Path]:
    left_label = method_dir / f"{dlabel_path.stem.replace('.dlabel', '')}.L.label.gii"
    right_label = method_dir / f"{dlabel_path.stem.replace('.dlabel', '')}.R.label.gii"
    run(
        [
            WB_COMMAND,
            "-cifti-separate",
            str(dlabel_path),
            "COLUMN",
            "-label",
            "CORTEX_LEFT",
            str(left_label),
            "-label",
            "CORTEX_RIGHT",
            str(right_label),
        ]
    )
    return left_label, right_label


def separate_sulc(sulc_dscalar: Path, assets_dir: Path, subject: str) -> tuple[Path, Path]:
    left_metric = assets_dir / f"sub-{subject}_hemi-L_space-fsLR_den-32k_desc-MSMAll_sulc.func.gii"
    right_metric = assets_dir / f"sub-{subject}_hemi-R_space-fsLR_den-32k_desc-MSMAll_sulc.func.gii"
    if left_metric.exists() and right_metric.exists():
        return left_metric, right_metric
    run(
        [
            WB_COMMAND,
            "-cifti-separate",
            str(sulc_dscalar),
            "COLUMN",
            "-metric",
            "CORTEX_LEFT",
            str(left_metric),
            "-metric",
            "CORTEX_RIGHT",
            str(right_metric),
        ]
    )
    return left_metric, right_metric


def write_summary(
    method: str,
    method_dir: Path,
    fastans_root: Path,
    dlabel_path: Path,
    left_label: Path,
    right_label: Path,
    dtseries: Path,
    left_midthickness: Path,
    right_midthickness: Path,
) -> Path:
    labels_rel = METHODS[method]["labels_file"]
    label_file = fastans_root / "resources" / "PFM" / "priors" / method / labels_rel
    label_names = parse_label_list(label_file)
    observed_counts = load_label_counts(dlabel_path)
    summary = {
        "method": method,
        "display_name": METHODS[method]["display_name"],
        "subject": method_dir.parents[0].name.replace("sub-", ""),
        "expected_network_count": METHODS[method]["expected_count"],
        "expected_network_names": label_names,
        "observed_network_count": len(observed_counts),
        "observed_network_counts": observed_counts,
        "input_dtseries": str(dtseries),
        "input_left_midthickness": str(left_midthickness),
        "input_right_midthickness": str(right_midthickness),
        "output_dlabel": str(dlabel_path),
        "output_left_label": str(left_label),
        "output_right_label": str(right_label),
    }
    summary_path = method_dir / "pfm_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary_path


def render_view(
    source_scene: Path,
    left_inflated: Path,
    right_inflated: Path,
    left_sulc: Path,
    right_sulc: Path,
    sulc_dscalar: Path,
    dlabel: Path,
    left_label: Path,
    right_label: Path,
    out_scene: Path,
    out_png: Path,
    rotation_axis: str | None,
    rotation_deg: float,
) -> None:
    cmd = [
        str((REPO_ROOT / "scripts" / "cortex" / "render_cortex_pfm_scene.py").resolve()),
        "--source-scene",
        str(source_scene),
        "--left-surface",
        str(left_inflated),
        "--right-surface",
        str(right_inflated),
        "--left-label",
        str(left_label),
        "--right-label",
        str(right_label),
        "--left-sulc",
        str(left_sulc),
        "--right-sulc",
        str(right_sulc),
        "--sulc-dscalar",
        str(sulc_dscalar),
        "--dlabel",
        str(dlabel),
        "--out-scene",
        str(out_scene),
        "--out-png",
        str(out_png),
        "--width",
        "1000",
        "--height",
        "700",
    ]
    if rotation_axis:
        cmd.extend(["--rotation-axis", rotation_axis, "--rotation-deg", str(rotation_deg)])
    run(cmd)


def compose_multiview(
    subject: str,
    title: str,
    legend_items: list[dict[str, object]],
    view_pngs: list[tuple[str, Path]],
    out_png: Path,
    subtitle: str | None = None,
    font_scale: float = 1.0,
) -> None:
    def load_font(size: int) -> ImageFont.ImageFont:
        for font_name in ["DejaVuSans.ttf", "Arial.ttf", "Helvetica.ttc"]:
            try:
                return ImageFont.truetype(font_name, size=size)
            except OSError:
                continue
        return ImageFont.load_default()

    images = [(label, Image.open(path).convert("RGB")) for label, path in view_pngs]
    tile_w = images[0][1].width
    tile_h = images[0][1].height
    cols = len(images)
    rows = 1
    scale = max(float(font_scale), 1.0)
    margin = int(round(24 * scale))
    title_h = int(round((96 if subtitle else 70) * scale))
    label_h = int(round(34 * scale))
    legend_w = int(round(240 * scale))
    canvas_w = cols * tile_w + legend_w + margin * 4
    canvas_h = title_h + rows * (tile_h + label_h) + margin * 3
    canvas = Image.new("RGB", (canvas_w, canvas_h), (255, 255, 255))
    draw = ImageDraw.Draw(canvas)
    title_font = load_font(int(round(22 * scale)))
    label_font = load_font(int(round(18 * scale)))
    legend_font = load_font(int(round(20 * scale)))
    legend_title_font = load_font(int(round(22 * scale)))
    title_prefix = f"sub-{subject}  " if subject and str(subject).lower() != "group" else ""
    draw.text((margin, int(round(18 * scale))), f"{title_prefix}{title}", fill=(0, 0, 0), font=title_font)
    if subtitle:
        subtitle_font = load_font(int(round(18 * scale)))
        draw.text((margin, int(round(50 * scale))), subtitle, fill=(40, 40, 40), font=subtitle_font)
    for idx, (label, img) in enumerate(images):
        row = idx // cols
        col = idx % cols
        x = margin + col * tile_w
        y = title_h + margin + row * (tile_h + label_h)
        canvas.paste(img, (x, y))
        draw.text((x + int(round(12 * scale)), y + tile_h + int(round(6 * scale))), label.capitalize(), fill=(0, 0, 0), font=label_font)

    legend_x = cols * tile_w + margin * 3
    legend_y = title_h + margin
    draw.text((legend_x, legend_y), "Legend", fill=(0, 0, 0), font=legend_title_font)
    row_gap = int(round(28 * scale))
    swatch = int(round(16 * scale))
    legend_start_y = legend_y + int(round(28 * scale))
    for idx, item in enumerate(legend_items):
        col = 0
        row = idx
        x = legend_x + int(round(col * 190 * scale))
        y = legend_start_y + row * row_gap
        rgba = item["rgba"]
        rgb = tuple(rgba[:3])
        draw.rectangle((x, y + int(round(4 * scale)), x + swatch, y + int(round(4 * scale)) + swatch), fill=rgb, outline=(0, 0, 0))
        draw.text((x + swatch + int(round(10 * scale)), y), str(item["name"]), fill=(0, 0, 0), font=legend_font)

    out_png.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_png)


def render_method(
    subject: str,
    method: str,
    source_scene: Path,
    left_inflated: Path,
    right_inflated: Path,
    left_sulc: Path,
    right_sulc: Path,
    sulc_dscalar: Path,
    dlabel: Path,
    left_label: Path,
    right_label: Path,
    method_dir: Path,
    stem: str,
    fastans_root: Path,
) -> Path:
    label_file = fastans_root / "resources" / "PFM" / "priors" / method / METHODS[method]["labels_file"]
    legend_items = parse_label_legend(label_file)
    views_dir = method_dir / "views"
    views_dir.mkdir(parents=True, exist_ok=True)
    rendered_views: list[tuple[str, Path]] = []
    for index, view in enumerate(VIEW_SPECS):
        scene_name = f"wb_{stem}_{view['name']}.scene"
        png_name = f"wb_{stem}_{view['name']}.png"
        out_scene = method_dir / scene_name if index == 0 else views_dir / scene_name
        out_png = views_dir / png_name
        render_view(
            source_scene=source_scene,
            left_inflated=left_inflated,
            right_inflated=right_inflated,
            left_sulc=left_sulc,
            right_sulc=right_sulc,
            sulc_dscalar=sulc_dscalar,
            dlabel=dlabel,
            left_label=left_label,
            right_label=right_label,
            out_scene=out_scene,
            out_png=out_png,
            rotation_axis=view["axis"],
            rotation_deg=view["deg"],
        )
        rendered_views.append((view["name"], out_png))

    montage_png = method_dir / f"wb_{stem}_inflated.png"
    compose_multiview(
        subject=subject,
        title=METHODS[method]["display_name"],
        legend_items=legend_items,
        view_pngs=rendered_views,
        out_png=montage_png,
    )
    return montage_png


def build_comparison(subject: str, top_png: Path, bottom_png: Path, out_png: Path) -> None:
    top_image = Image.open(top_png).convert("RGB")
    bottom_image = Image.open(bottom_png).convert("RGB")
    canvas_w = max(top_image.width, bottom_image.width)
    canvas_h = top_image.height + bottom_image.height + 80
    canvas = Image.new("RGB", (canvas_w, canvas_h), (255, 255, 255))
    draw = ImageDraw.Draw(canvas)
    draw.text((20, 18), f"sub-{subject} cortex individualized functional parcellation comparison", fill=(0, 0, 0))
    canvas.paste(top_image, (0, 60))
    canvas.paste(bottom_image, (0, 60 + top_image.height))
    out_png.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_png)


def main() -> int:
    args = parse_args()
    subject = args.subject
    dtseries = Path(args.dtseries).resolve()
    left_midthickness = Path(args.left_midthickness).resolve()
    right_midthickness = Path(args.right_midthickness).resolve()
    left_inflated = Path(args.left_inflated).resolve()
    right_inflated = Path(args.right_inflated).resolve()
    sulc_dscalar = Path(args.sulc_dscalar).resolve()
    fastans_root = Path(args.fastans_root).resolve()
    source_scene = Path(args.source_scene).resolve()
    out_root = Path(args.out_root).resolve()

    for path in [dtseries, left_midthickness, right_midthickness, left_inflated, right_inflated, sulc_dscalar, fastans_root, source_scene]:
        if not path.exists():
            raise FileNotFoundError(path)

    repo_scripts = str((REPO_ROOT / "scripts").resolve())
    os.environ["PATH"] = repo_scripts + os.pathsep + os.environ["PATH"]

    fastans = import_fastans(fastans_root)
    subject_root = out_root / f"sub-{subject}"
    assets_dir = subject_root / "assets"
    assets_dir.mkdir(parents=True, exist_ok=True)
    left_sulc, right_sulc = separate_sulc(sulc_dscalar, assets_dir, subject)

    rendered_pngs: dict[str, Path] = {}
    for method in args.methods:
        if method not in METHODS:
            raise ValueError(f"Unsupported method: {method}")
        method_dir = subject_root / METHODS[method]["slug"]
        method_dir.mkdir(parents=True, exist_ok=True)
        fastans.fast_pfm(
            str(dtseries),
            method,
            str(method_dir),
            str(left_midthickness),
            str(right_midthickness),
        )
        dlabel_path = method_dir / f"PFM_{method}priors.dlabel.nii"
        if not dlabel_path.exists():
            raise FileNotFoundError(dlabel_path)
        file_info = subprocess.run([WB_COMMAND, "-file-information", str(dlabel_path)], text=True, capture_output=True)
        if file_info.returncode != 0:
            raise RuntimeError(file_info.stderr)
        left_label, right_label = separate_labels(dlabel_path, method_dir)
        write_summary(
            method,
            method_dir,
            fastans_root,
            dlabel_path,
            left_label,
            right_label,
            dtseries,
            left_midthickness,
            right_midthickness,
        )
        rendered_pngs[method] = render_method(
            subject=subject,
            method=method,
            source_scene=source_scene,
            left_inflated=left_inflated,
            right_inflated=right_inflated,
            left_sulc=left_sulc,
            right_sulc=right_sulc,
            sulc_dscalar=sulc_dscalar,
            dlabel=dlabel_path,
            left_label=left_label,
            right_label=right_label,
            method_dir=method_dir,
            stem=METHODS[method]["slug"],
            fastans_root=fastans_root,
        )

    if "Lynch2024" in rendered_pngs and "Kong2019" in rendered_pngs:
        comparison_dir = subject_root / "comparison"
        build_comparison(
            subject,
            rendered_pngs["Lynch2024"],
            rendered_pngs["Kong2019"],
            comparison_dir / "wb_lynch2024_vs_kong2019.png",
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
