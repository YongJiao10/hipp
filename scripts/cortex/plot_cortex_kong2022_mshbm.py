#!/usr/bin/env python3
from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


REPO_ROOT = Path(__file__).resolve().parents[2]
WB_COMMAND = str((REPO_ROOT / "scripts" / "wb_command").resolve())
RENDER_SCENE_SCRIPT = REPO_ROOT / "scripts" / "cortex" / "render_cortex_pfm_scene.py"
DEFAULT_SCENE = REPO_ROOT / "config" / "manual_wb_scenes" / "cortex_manual.scene"
ATLAS_SLUG = "kong2022_areal_mshbm"

VIEW_SPECS = [
    {"name": "lateral", "axis": None, "deg": 0.0},
    {"name": "medial", "axis": "y", "deg": -90.0},
    {"name": "posterior", "axis": "y", "deg": 180.0},
    {"name": "dorsal", "axis": "x", "deg": -90.0},
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Render one overview PNG for subject-specific Kong2022 MSHBM cortex dlabel.")
    p.add_argument("--subject", required=True, help="Subject ID without sub- prefix.")
    p.add_argument("--dlabel", required=True)
    p.add_argument("--left-inflated", required=True)
    p.add_argument("--right-inflated", required=True)
    p.add_argument("--sulc-dscalar", required=True)
    p.add_argument("--source-scene", default=str(DEFAULT_SCENE))
    p.add_argument("--atlas-dir", default=None, help="Override atlas output directory. Defaults to dlabel parent.")
    p.add_argument("--out-png", default=None)
    p.add_argument("--scene-index", type=int, default=1)
    p.add_argument("--renderer", default="OSMesa")
    p.add_argument("--width", type=int, default=1200)
    p.add_argument("--height", type=int, default=900)
    return p.parse_args()


def run(cmd: list[str]) -> None:
    proc = subprocess.run(cmd, text=True, capture_output=True)
    if proc.returncode != 0:
        raise RuntimeError(
            "Command failed: "
            + " ".join(cmd)
            + "\nSTDOUT:\n"
            + (proc.stdout or "")
            + "\nSTDERR:\n"
            + (proc.stderr or "")
        )


def ensure_exists(paths: list[Path]) -> None:
    for path in paths:
        if not path.exists():
            raise FileNotFoundError(path)


def split_subject_assets(
    *,
    dlabel: Path,
    sulc_dscalar: Path,
    assets_dir: Path,
    subject: str,
) -> dict[str, Path]:
    assets_dir.mkdir(parents=True, exist_ok=True)
    left_label = assets_dir / f"sub-{subject}_atlas-{ATLAS_SLUG}.L.label.gii"
    right_label = assets_dir / f"sub-{subject}_atlas-{ATLAS_SLUG}.R.label.gii"
    left_sulc = assets_dir / f"sub-{subject}_hemi-L_desc-MSMAll_sulc.func.gii"
    right_sulc = assets_dir / f"sub-{subject}_hemi-R_desc-MSMAll_sulc.func.gii"

    if not left_label.exists() or not right_label.exists():
        run(
            [
                WB_COMMAND,
                "-cifti-separate",
                str(dlabel),
                "COLUMN",
                "-label",
                "CORTEX_LEFT",
                str(left_label),
                "-label",
                "CORTEX_RIGHT",
                str(right_label),
            ]
        )

    if not left_sulc.exists() or not right_sulc.exists():
        run(
            [
                WB_COMMAND,
                "-cifti-separate",
                str(sulc_dscalar),
                "COLUMN",
                "-metric",
                "CORTEX_LEFT",
                str(left_sulc),
                "-metric",
                "CORTEX_RIGHT",
                str(right_sulc),
            ]
        )

    return {
        "left_label": left_label,
        "right_label": right_label,
        "left_sulc": left_sulc,
        "right_sulc": right_sulc,
    }


def render_views(
    *,
    subject: str,
    source_scene: Path,
    left_inflated: Path,
    right_inflated: Path,
    left_label: Path,
    right_label: Path,
    left_sulc: Path,
    right_sulc: Path,
    sulc_dscalar: Path,
    dlabel: Path,
    figures_dir: Path,
    scene_index: int,
    renderer: str,
    width: int,
    height: int,
) -> list[tuple[str, Path]]:
    view_dir = figures_dir / "views"
    scene_dir = figures_dir / "scenes"
    view_dir.mkdir(parents=True, exist_ok=True)
    scene_dir.mkdir(parents=True, exist_ok=True)

    outputs: list[tuple[str, Path]] = []
    for idx, spec in enumerate(VIEW_SPECS):
        out_scene = scene_dir / f"sub-{subject}_atlas-{ATLAS_SLUG}_{spec['name']}.scene"
        out_png = view_dir / f"sub-{subject}_atlas-{ATLAS_SLUG}_{spec['name']}.png"
        cmd = [
            str(RENDER_SCENE_SCRIPT),
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
            "--scene-index",
            str(scene_index),
            "--renderer",
            renderer,
            "--width",
            str(width),
            "--height",
            str(height),
        ]
        if spec["axis"] is not None:
            cmd.extend(["--rotation-axis", str(spec["axis"]), "--rotation-deg", str(spec["deg"])])
        run(cmd)
        outputs.append((str(spec["name"]), out_png))

    return outputs


def load_font(size: int) -> ImageFont.ImageFont:
    for font_name in ["DejaVuSans.ttf", "Arial.ttf", "Helvetica.ttc"]:
        try:
            return ImageFont.truetype(font_name, size=size)
        except OSError:
            continue
    return ImageFont.load_default()


def compose_overview(
    *,
    subject: str,
    view_pngs: list[tuple[str, Path]],
    out_png: Path,
) -> None:
    images = [(label, Image.open(path).convert("RGB")) for label, path in view_pngs]
    if len(images) != 4:
        raise RuntimeError(f"Expected 4 rendered views, got {len(images)}")

    tile_w = images[0][1].width
    tile_h = images[0][1].height
    margin = 28
    title_h = 86
    label_h = 36

    canvas_w = tile_w * 2 + margin * 3
    canvas_h = title_h + (tile_h + label_h) * 2 + margin * 3
    canvas = Image.new("RGB", (canvas_w, canvas_h), (255, 255, 255))
    draw = ImageDraw.Draw(canvas)

    title_font = load_font(28)
    subtitle_font = load_font(18)
    label_font = load_font(18)

    draw.text(
        (margin, 20),
        f"sub-{subject} Kong2022 MSHBM Cortex Overview",
        fill=(0, 0, 0),
        font=title_font,
    )
    draw.text(
        (margin, 56),
        "Views: lateral, medial, posterior, dorsal",
        fill=(35, 35, 35),
        font=subtitle_font,
    )

    for idx, (label, image) in enumerate(images):
        row = idx // 2
        col = idx % 2
        x = margin + col * (tile_w + margin)
        y = title_h + margin + row * (tile_h + label_h)
        canvas.paste(image, (x, y))
        draw.text((x + 8, y + tile_h + 8), label.capitalize(), fill=(0, 0, 0), font=label_font)

    out_png.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_png)


def main() -> int:
    args = parse_args()

    subject = str(args.subject).replace("sub-", "")
    dlabel = Path(args.dlabel).resolve()
    left_inflated = Path(args.left_inflated).resolve()
    right_inflated = Path(args.right_inflated).resolve()
    sulc_dscalar = Path(args.sulc_dscalar).resolve()
    source_scene = Path(args.source_scene).resolve()

    ensure_exists([dlabel, left_inflated, right_inflated, sulc_dscalar, source_scene, Path(WB_COMMAND), RENDER_SCENE_SCRIPT])

    atlas_dir = Path(args.atlas_dir).resolve() if args.atlas_dir else dlabel.parent.resolve()
    figures_dir = atlas_dir / "figures"
    assets_dir = atlas_dir / "assets"

    split_assets = split_subject_assets(
        dlabel=dlabel,
        sulc_dscalar=sulc_dscalar,
        assets_dir=assets_dir,
        subject=subject,
    )
    view_pngs = render_views(
        subject=subject,
        source_scene=source_scene,
        left_inflated=left_inflated,
        right_inflated=right_inflated,
        left_label=split_assets["left_label"],
        right_label=split_assets["right_label"],
        left_sulc=split_assets["left_sulc"],
        right_sulc=split_assets["right_sulc"],
        sulc_dscalar=sulc_dscalar,
        dlabel=dlabel,
        figures_dir=figures_dir,
        scene_index=int(args.scene_index),
        renderer=str(args.renderer),
        width=int(args.width),
        height=int(args.height),
    )

    out_png = (
        Path(args.out_png).resolve()
        if args.out_png
        else (figures_dir / f"sub-{subject}_{ATLAS_SLUG}_overview.png").resolve()
    )
    compose_overview(subject=subject, view_pngs=view_pngs, out_png=out_png)
    print(str(out_png))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
