#!/usr/bin/env python3
"""
Plot tSNR surface (masked version): vertices with tSNR < 25 rendered as gray.

Produces:
  outputs_migration/tsnr_surface_masked.png

Pipeline:
  1. Compute tSNR (10000/σ) from BOLD func.gii for each subject × hemi.
  2. NaN-mask vertices below GRAY_THRESH, save as *_tsnr_masked.shape.gii
     into the surf/ dir alongside the corobl midthickness surface.
  3. Clone config/wb_locked_native_view_lateral_medial.scene per subject,
     swap gyrification → tsnr_masked overlay, disable label overlays,
     fix ROY-BIG-BL palette to global tSNR range.
  4. Render via wb_command -scene-capture-image (OSMesa, 1600×1200).
  5. Split each render at midline → L/R panels, tight-crop, compose 3×2 grid.
  6. Draw unified colorbar with gray band below threshold.
  7. Annotate each panel with "tSNR<25: X.X%".

Requirements:
  - conda env py314  (nibabel, numpy, matplotlib, Pillow)
  - /Applications/wb_view.app  (wb_command with OSMesa renderer)
  - outputs_migration/dense_corobl_batch/sub-{100610,102311,102816}/hippunfold/
      sub-X/surf/sub-X_hemi-{L,R}_space-corobl_den-512_label-hipp_midthickness.surf.gii
    (regenerate with: bash scripts/regen_corobl_batch.sh if missing)
  - outputs_migration/hipp_functional_parcellation_network/_shared/
      sub-X/surface/raw/sub-X_hemi-{L,R}_space-corobl_den-512_label-hipp_bold.func.gii

Usage (from repo root):
  conda run -n py314 python scripts/plot_tsnr_surface.py
"""
from __future__ import annotations

import re
import subprocess
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path

import nibabel as nib
import numpy as np
from matplotlib import colormaps, colors
from PIL import Image, ImageDraw, ImageFont

# ── Hardcoded paths (all relative to repo root) ───────────────────────────────
REPO_ROOT  = Path(__file__).resolve().parents[1]
SCENE_SRC  = REPO_ROOT / "config" / "wb_locked_native_view_lateral_medial.scene"
BATCH_DIR  = REPO_ROOT / "outputs_migration" / "dense_corobl_batch"
BOLD_DIR   = REPO_ROOT / "outputs_migration" / "hipp_functional_parcellation_network" / "_shared"
OUT_PATH   = REPO_ROOT / "outputs_migration" / "tsnr_surface_masked.png"
WB_CMD     = 'arch -x86_64 "/Applications/wb_view.app/Contents/usr/bin/wb_command"'

# ── Constants ─────────────────────────────────────────────────────────────────
SUBJECTS    = ["sub-100610", "sub-102311", "sub-102816"]
HEMIS       = ["L", "R"]
HEMI_LABEL  = {"L": "Left", "R": "Right"}
HCP_MODE    = 10000.0   # tSNR = 10000 / σ  (HCP convention)
GRAY_THRESH = 25.0      # vertices below this are NaN → rendered gray

GYRI_STEM         = "_gyrification.shape.gii"
TSNR_MASKED_STEM  = "_tsnr_masked.shape.gii"
LABEL_STEM        = "_atlas-multihist7_subfields.label.gii"
DLABEL_STEM       = "_atlas-multihist7_subfields.dlabel.nii"

# Canvas layout
LABEL_H  = 52
BAR_W    = 140
GUTTER   = 14
MARGIN   = 28
BG_COLOR = (12, 12, 12, 255)
WHITE    = (255, 255, 255, 255)
GREY     = (170, 170, 170, 255)
ANN_CLR  = (200, 200, 200, 255)


# ── Helpers ───────────────────────────────────────────────────────────────────
def load_font(size: int) -> ImageFont.FreeTypeFont:
    for candidate in [
        "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/System/Library/Fonts/Supplemental/Helvetica.ttc",
        "/System/Library/Fonts/SFNS.ttf",
    ]:
        p = Path(candidate)
        if p.exists():
            try:
                return ImageFont.truetype(str(p), size=size)
            except Exception:
                continue
    return ImageFont.load_default()


def tight_crop(rgba: np.ndarray, bg_thresh: int = 8, margin: int = 10) -> np.ndarray:
    mask = rgba[:, :, :3].max(axis=2) > bg_thresh
    rows = np.where(mask.any(axis=1))[0]
    cols = np.where(mask.any(axis=0))[0]
    if rows.size == 0 or cols.size == 0:
        return rgba
    r0 = max(0, rows[0] - margin)
    r1 = min(rgba.shape[0], rows[-1] + margin + 1)
    c0 = max(0, cols[0] - margin)
    c1 = min(rgba.shape[1], cols[-1] + margin + 1)
    return rgba[r0:r1, c0:c1]


def save_shape_gii(values: np.ndarray, out_path: Path) -> None:
    darray = nib.gifti.GiftiDataArray(
        data=values.astype(np.float32),
        intent=nib.nifti1.intent_codes["NIFTI_INTENT_NONE"],
        datatype="NIFTI_TYPE_FLOAT32",
    )
    nib.save(nib.gifti.GiftiImage(darrays=[darray]), str(out_path))


def absolutize(root: ET.Element, scene_path: Path) -> None:
    """Resolve relative scene pathName entries → absolute, remapped to BATCH_DIR."""
    base = scene_path.parent
    old_batch = Path.home() / "Documents" / "HippoMaps" / "outputs" / "dense_corobl_batch"
    for obj in root.iter("Object"):
        if obj.attrib.get("Type") == "pathName" and obj.text:
            text = obj.text.strip()
            if not text:
                continue
            p = Path(text)
            resolved = (base / p).resolve() if not p.is_absolute() else p
            try:
                resolved = BATCH_DIR / resolved.relative_to(old_batch)
            except ValueError:
                pass
            obj.text = str(resolved)


def detect_template_subject(root: ET.Element) -> str:
    for elem in root.iter():
        if elem.text:
            m = re.search(r"sub-(\d+)", elem.text)
            if m:
                return m.group(1)
    raise RuntimeError("Template subject not found in scene XML")


def replace_text_global(root: ET.Element, old: str, new: str) -> None:
    for elem in root.iter():
        if elem.text and old in elem.text:
            elem.text = elem.text.replace(old, new)


def disable_label_overlays(root: ET.Element) -> None:
    parent_map: dict[int, ET.Element] = {}
    for elem in root.iter():
        for child in elem:
            parent_map[id(child)] = elem
    for obj in root.iter("Object"):
        if obj.attrib.get("Name") != "m_enabled":
            continue
        parent = parent_map.get(id(obj))
        if parent is None:
            continue
        for sib in list(parent):
            txt = (sib.text or "").strip()
            if LABEL_STEM in txt or DLABEL_STEM in txt:
                obj.text = "false"
                break


def inject_fixed_palette(root: ET.Element, vmin: float, vmax: float) -> None:
    for obj in root.iter("Object"):
        name = obj.attrib.get("Name", "")
        if name == "m_paletteNormalizationMode":
            obj.text = "NORMALIZATION_SPECIFIED_VALUES"
        elif name == "m_selectedPaletteName":
            obj.text = "ROY-BIG-BL"
        elif name == "m_userScalePercentageMinimum":
            obj.text = str(vmin)
        elif name == "m_userScalePercentageMaximum":
            obj.text = str(vmax)


# ── Preflight: verify all required input files exist ─────────────────────────
missing = []
for sub in SUBJECTS:
    for hemi in HEMIS:
        bold = BOLD_DIR / sub / "surface" / "raw" / \
            f"{sub}_hemi-{hemi}_space-corobl_den-512_label-hipp_bold.func.gii"
        surf = BATCH_DIR / sub / "hippunfold" / sub / "surf" / \
            f"{sub}_hemi-{hemi}_space-corobl_den-512_label-hipp_midthickness.surf.gii"
        if not bold.exists():
            missing.append(str(bold))
        if not surf.exists():
            missing.append(str(surf))
if not SCENE_SRC.exists():
    missing.append(str(SCENE_SRC))
if missing:
    raise FileNotFoundError(
        "Missing required input files (re-run regen_corobl_batch.sh if surf files are absent):\n"
        + "\n".join(f"  {p}" for p in missing)
    )

# ── Step 1: Compute tSNR and build masked shape.gii ───────────────────────────
print("Step 1: Computing tSNR and saving masked shape.gii …")
tsnr_data: dict[tuple[str, str], np.ndarray] = {}
pct_below: dict[tuple[str, str], float] = {}

for sub in SUBJECTS:
    for hemi in HEMIS:
        bold_path = BOLD_DIR / sub / "surface" / "raw" / \
            f"{sub}_hemi-{hemi}_space-corobl_den-512_label-hipp_bold.func.gii"
        img = nib.load(str(bold_path))
        bold = np.stack([d.data for d in img.darrays], axis=1).astype(np.float64)
        sd = bold.std(axis=1, ddof=1)
        tsnr = np.where(sd > 0, HCP_MODE / sd, np.nan).astype(np.float32)
        tsnr_data[(sub, hemi)] = tsnr

        masked = tsnr.copy()
        masked[masked < GRAY_THRESH] = np.nan
        n_masked = int(np.isnan(masked).sum())
        pct_below[(sub, hemi)] = n_masked / len(masked) * 100

        surf_dir = BATCH_DIR / sub / "hippunfold" / sub / "surf"
        out_gii  = surf_dir / f"{sub}_hemi-{hemi}_space-corobl_den-512_label-hipp_tsnr_masked.shape.gii"
        save_shape_gii(masked, out_gii)
        print(f"  {sub} {hemi}: mean={np.nanmean(tsnr):.1f}  "
              f"below<{GRAY_THRESH:.0f}: {n_masked}/{len(masked)} ({pct_below[(sub,hemi)]:.1f}%)")

all_vals = np.concatenate([v[np.isfinite(v)] for v in tsnr_data.values()])
vmin = float(all_vals.min())
vmax = float(all_vals.max())
print(f"\nGlobal tSNR range: {vmin:.2f} – {vmax:.2f}")

# ── Step 2: Build per-subject scenes ──────────────────────────────────────────
print("\nStep 2: Building scenes …")
base_root = ET.parse(str(SCENE_SRC)).getroot()
absolutize(base_root, SCENE_SRC)
replace_text_global(base_root, GYRI_STEM, TSNR_MASKED_STEM)
disable_label_overlays(base_root)
inject_fixed_palette(base_root, vmin, vmax)
template_sub = detect_template_subject(base_root)

# ── Step 3: Render ────────────────────────────────────────────────────────────
print("\nStep 3: Rendering via wb_command …")
tmpdir = Path(tempfile.mkdtemp(prefix="tsnr_masked_"))
render_pngs: dict[str, Path] = {}

for sub in SUBJECTS:
    root = ET.fromstring(ET.tostring(base_root, encoding="unicode"))
    for elem in root.iter():
        if elem.text and f"sub-{template_sub}" in elem.text:
            elem.text = elem.text.replace(f"sub-{template_sub}", sub)

    scene_path = tmpdir / f"{sub}_tsnr_masked.scene"
    tree = ET.ElementTree(root)
    ET.indent(tree, space="    ")
    tree.write(str(scene_path), encoding="unicode", xml_declaration=False)

    out_png = tmpdir / f"{sub}_tsnr_masked.png"
    cmd = (f'{WB_CMD} -scene-capture-image "{scene_path}" 1 "{out_png}"'
           f' -size-width-height 1600 1200 -renderer OSMesa')
    proc = subprocess.run(cmd, shell=True, text=True, capture_output=True)
    if proc.returncode != 0:
        raise RuntimeError(f"wb_command failed for {sub}:\n{proc.stderr}")
    render_pngs[sub] = out_png
    print(f"  {sub} → {out_png}")

# ── Step 4: Split renders into L/R panels ─────────────────────────────────────
print("\nStep 4: Splitting L/R panels …")
panels: dict[tuple[str, str], np.ndarray] = {}
for sub in SUBJECTS:
    img = np.asarray(Image.open(render_pngs[sub]).convert("RGBA"))
    mid = img.shape[1] // 2
    panels[(sub, "L")] = img[:, :mid, :]
    panels[(sub, "R")] = img[:, mid:, :]

# ── Step 5: Compose figure ────────────────────────────────────────────────────
print("\nStep 5: Composing figure …")
cropped = {k: tight_crop(v) for k, v in panels.items()}
cell_h = max(v.shape[0] for v in cropped.values())
cell_w = max(v.shape[1] for v in cropped.values())

n_rows, n_cols = 3, 2
canvas_w = MARGIN * 2 + n_cols * cell_w + (n_cols - 1) * GUTTER + GUTTER + BAR_W
canvas_h = MARGIN * 2 + n_rows * (LABEL_H + cell_h) + (n_rows - 1) * GUTTER + 36
canvas   = Image.new("RGBA", (canvas_w, canvas_h), BG_COLOR)

title_font = load_font(28)
tick_font  = load_font(20)
ann_font   = load_font(18)

for row, sub in enumerate(SUBJECTS):
    for col, hemi in enumerate(HEMIS):
        panel = cropped[(sub, hemi)]
        ph, pw = panel.shape[:2]
        if pw > cell_w or ph > cell_h:
            scale = min(cell_w / pw, cell_h / ph)
            panel = np.asarray(
                Image.fromarray(panel, "RGBA").resize(
                    (int(pw * scale), int(ph * scale)), Image.Resampling.LANCZOS)
            )
            ph, pw = panel.shape[:2]

        cell_x = MARGIN + col * (cell_w + GUTTER)
        cell_y = MARGIN + 36 + row * (LABEL_H + cell_h + GUTTER)
        ox = cell_x + (cell_w - pw) // 2
        oy = cell_y + LABEL_H + (cell_h - ph) // 2

        draw = ImageDraw.Draw(canvas)
        draw.text((ox, cell_y + 10),
                  f"{sub.replace('sub-', '')}  ·  {HEMI_LABEL[hemi]} Hipp.",
                  fill=WHITE, font=title_font)
        canvas.alpha_composite(Image.fromarray(panel, "RGBA"), (ox, oy))

        pct = pct_below[(sub, hemi)]
        draw.text((ox, oy + ph - 24),
                  f"tSNR<{GRAY_THRESH:.0f}: {pct:.1f}%",
                  fill=ANN_CLR, font=ann_font)

# ── Colorbar ──────────────────────────────────────────────────────────────────
cmap_obj  = colormaps["hot"]
norm      = colors.Normalize(vmin=vmin, vmax=vmax)
bar_x     = canvas_w - BAR_W - MARGIN // 2
bar_top   = MARGIN + 36 + LABEL_H
bar_h     = canvas_h - MARGIN - bar_top
bar_iw    = 26

vals     = np.linspace(vmax, vmin, bar_h, dtype=np.float32)
bar_rgba = np.round(cmap_obj(norm(vals)) * 255).astype(np.uint8).reshape(bar_h, 1, 4)
bar_rgba = np.repeat(bar_rgba, bar_iw, axis=1)
canvas.alpha_composite(Image.fromarray(bar_rgba, "RGBA"), (bar_x + 26, bar_top))

# Gray band for sub-threshold region
gray_frac = (GRAY_THRESH - vmin) / (vmax - vmin)
gray_px   = int(bar_h * (1 - gray_frac))
gray_band = np.full((bar_h - gray_px, bar_iw, 4), [120, 120, 120, 255], dtype=np.uint8)
canvas.alpha_composite(Image.fromarray(gray_band, "RGBA"), (bar_x + 26, bar_top + gray_px))

draw = ImageDraw.Draw(canvas)
draw.rectangle((bar_x + 26, bar_top, bar_x + 26 + bar_iw, bar_top + bar_h), outline=GREY, width=1)

# Dashed threshold line
thresh_y = bar_top + gray_px
for dash_x in range(bar_x + 18, bar_x + 26 + bar_iw + 40, 6):
    draw.line([(dash_x, thresh_y), (min(dash_x + 3, bar_x + 26 + bar_iw + 40), thresh_y)],
              fill=WHITE, width=1)
draw.text((bar_x + 26 + bar_iw + 14, thresh_y - 10),
          f"{GRAY_THRESH:.0f}", fill=WHITE, font=tick_font)

for tick_val, tick_y in [(vmax, bar_top), (vmin, bar_top + bar_h)]:
    tx = bar_x + 26 + bar_iw + 6
    draw.line((tx, tick_y, tx + 8, tick_y), fill=GREY, width=1)
    draw.text((tx + 12, tick_y - 10), f"{tick_val:.1f}", fill=GREY, font=tick_font)

# Rotated colorbar label
tmp = Image.new("RGBA", (bar_h, 32), (0, 0, 0, 0))
ImageDraw.Draw(tmp).text((0, 4), "tSNR  (10000 / σt)", fill=WHITE, font=tick_font)
tmp = tmp.rotate(90, expand=True)
canvas.alpha_composite(tmp, (bar_x + 2, bar_top + (bar_h - tmp.height) // 2))

# Figure title
draw.text(
    (MARGIN, 6),
    f"Hippocampal tSNR  |  HCP 7T · corobl surface  [gray: tSNR < {GRAY_THRESH:.0f}]",
    fill=WHITE, font=load_font(30),
)

OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
canvas.save(str(OUT_PATH))
print(f"\nSaved → {OUT_PATH}")
