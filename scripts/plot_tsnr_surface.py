#!/usr/bin/env python3
"""
tSNR surface overlay using the existing wb_command + scene pipeline.

Steps:
  1. Compute tSNR (10000/std) per vertex, save as .shape.gii
  2. Clone wb_locked_native_view.scene, swap gyrification → tsnr,
     disable label overlay, set fixed HOT palette
  3. Render 3 subjects via render_wb_scene_batch.py
  4. Split each rendered image at midline → 6 hemi panels
  5. Compose 3×2 grid + unified colorbar
"""
from __future__ import annotations

import copy
import subprocess
import sys
import tempfile
import xml.etree.ElementTree as ET
import argparse
from pathlib import Path

import nibabel as nib
import numpy as np
from matplotlib import cm, colors
from PIL import Image, ImageDraw, ImageFont

# ── Paths ─────────────────────────────────────────────────────────────────────
REPO_ROOT   = Path(__file__).resolve().parents[1]
SCENE_SRC   = REPO_ROOT / "config" / "wb_locked_native_view_lateral_medial.scene"
RENDER_SCRIPT = REPO_ROOT / "scripts" / "workbench" / "render_wb_scene_batch.py"

_p = argparse.ArgumentParser()
_p.add_argument("--batch-dir", default=str(REPO_ROOT / "outputs_migration" / "dense_corobl_batch"))
_p.add_argument("--out", default=str(REPO_ROOT / "outputs_migration" / "tsnr_surface.png"))
_p.add_argument("--out-masked", default=str(REPO_ROOT / "outputs_migration" / "tsnr_surface_masked.png"))
_args = _p.parse_args()
BATCH_DIR = Path(_args.batch_dir)
OUT_FIG   = Path(_args.out)

SUBJECTS    = ["sub-100610", "sub-102311", "sub-102816"]
HEMIS       = ["L", "R"]
HEMI_LABEL  = {"L": "Left", "R": "Right"}
HCP_MODE    = 10000.0
PYTHON_EXE  = sys.executable

# ── Step 1: Compute tSNR and save as shape.gii ────────────────────────────────
def load_bold(sub: str, hemi: str) -> np.ndarray:
    surface_dir = REPO_ROOT / "outputs_migration" / "hipp_functional_parcellation_network" / "_shared" / sub / "surface" / "raw"
    npy = surface_dir / f"{sub}_hemi-{hemi}_space-corobl_den-512_label-hipp_bold.npy"
    if npy.exists():
        return np.load(npy).astype(np.float64)
    gii = surface_dir / f"{sub}_hemi-{hemi}_space-corobl_den-512_label-hipp_bold.func.gii"
    img = nib.load(str(gii))
    return np.stack([d.data for d in img.darrays], axis=1).astype(np.float64)


def compute_tsnr(bold: np.ndarray) -> np.ndarray:
    sd = bold.std(axis=1, ddof=1)
    return np.where(sd > 0, HCP_MODE / sd, np.nan).astype(np.float32)


def save_shape_gii(values: np.ndarray, ref_surf_path: Path, out_path: Path) -> None:
    """Write a scalar metric as a shape.gii, borrowing structure intent from ref surface."""
    darray = nib.gifti.GiftiDataArray(
        data=values.astype(np.float32),
        intent=nib.nifti1.intent_codes["NIFTI_INTENT_NONE"],
        datatype="NIFTI_TYPE_FLOAT32",
    )
    img = nib.gifti.GiftiImage(darrays=[darray])
    nib.save(img, str(out_path))


print("Step 1: Computing tSNR …")
tsnr_data: dict[tuple[str, str], np.ndarray] = {}
tsnr_paths: dict[tuple[str, str], Path] = {}

for sub in SUBJECTS:
    for hemi in HEMIS:
        bold = load_bold(sub, hemi)
        tsnr = compute_tsnr(bold)
        tsnr_data[(sub, hemi)] = tsnr

        surf_dir = BATCH_DIR / sub / "hippunfold" / sub / "surf"
        ref_surf  = surf_dir / f"{sub}_hemi-{hemi}_space-corobl_den-512_label-hipp_midthickness.surf.gii"
        out_shape = surf_dir / f"{sub}_hemi-{hemi}_space-corobl_den-512_label-hipp_tsnr.shape.gii"
        save_shape_gii(tsnr, ref_surf, out_shape)
        tsnr_paths[(sub, hemi)] = out_shape
        print(f"  {sub} {hemi}: mean={np.nanmean(tsnr):.1f}  min={np.nanmin(tsnr):.1f}  max={np.nanmax(tsnr):.1f}")

# Global range
all_vals = np.concatenate([v[np.isfinite(v)] for v in tsnr_data.values()])
vmin = float(all_vals.min())
vmax = float(all_vals.max())
print(f"\nGlobal tSNR range: {vmin:.2f} – {vmax:.2f}")


# ── Step 2: Build modified scene with tSNR overlay ───────────────────────────
GYRI_STEM = "_gyrification.shape.gii"
TSNR_STEM = "_tsnr.shape.gii"
LABEL_STEM = "_atlas-multihist7_subfields.label.gii"
DLABEL_STEM = "_atlas-multihist7_subfields.dlabel.nii"

# Workbench palette name (hot sequential, available in wb_view)
WB_PALETTE  = "ROY-BIG-BL"   # will be replaced by custom hot below


def absolutize(root: ET.Element, scene_path: Path) -> None:
    """Resolve relative pathName entries to absolute paths and remap to current BATCH_DIR.

    The scene was saved with paths relative to config/ that resolve into the old
    HippoMaps/outputs/dense_corobl_batch tree.  We remap those to the current
    BATCH_DIR so wb_command can find the files regardless of where the data lives.
    """
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
                rel = resolved.relative_to(old_batch)
                resolved = BATCH_DIR / rel
            except ValueError:
                pass
            obj.text = str(resolved)


def detect_template_subject(root: ET.Element) -> str:
    import re
    pattern = re.compile(r"sub-(\d+)")
    for elem in root.iter():
        if elem.text:
            m = pattern.search(elem.text)
            if m:
                return m.group(1)
    raise RuntimeError("Template subject not found")


def replace_text_global(root: ET.Element, old: str, new: str) -> None:
    for elem in root.iter():
        if elem.text and old in elem.text:
            elem.text = elem.text.replace(old, new)


def disable_label_overlays(root: ET.Element) -> None:
    """Disable label/dlabel overlays so only tSNR scalar overlay shows.

    The scene overlay XML structure nests m_enabled and selectedMapFile as
    siblings inside the same parent Object element.  We walk the full parent
    map, find every m_enabled sibling of a label/dlabel file reference, and
    set it to false.
    """
    # Build parent map once
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
        # Check if any sibling element's text mentions a label/dlabel file
        for sib in list(parent):
            txt = (sib.text or "").strip()
            if LABEL_STEM in txt or DLABEL_STEM in txt:
                obj.text = "false"
                break


def inject_fixed_palette(root: ET.Element, vmin: float, vmax: float) -> None:
    """
    For each shape.gii that now contains tSNR, set:
      m_paletteNormalizationMode → NORMALIZATION_SPECIFIED_VALUES
      m_paletteScaleModeMapped   → MAP_DATA_NORMAL
      m_selectedPaletteName      → ROY-BIG-BL (built-in hot-like in wb)
      userScalePercentageMinimum → vmin
      userScalePercentageMaximum → vmax
    We inject these values into the XML structure that wb_view saves.
    """
    for obj in root.iter("Object"):
        if obj.attrib.get("Name") == "m_paletteNormalizationMode":
            # Check if this is inside a tsnr shape.gii context
            # Walk up to check parent names
            obj.text = "NORMALIZATION_SPECIFIED_VALUES"
        if obj.attrib.get("Name") == "m_selectedPaletteName":
            obj.text = "ROY-BIG-BL"
    # Also inject scale objects if they exist
    for obj in root.iter("Object"):
        name = obj.attrib.get("Name", "")
        if name == "m_userScalePercentageMinimum":
            obj.text = str(vmin)
        elif name == "m_userScalePercentageMaximum":
            obj.text = str(vmax)


print("\nStep 2: Building modified scene …")
base_tree = ET.parse(str(SCENE_SRC))
base_root = base_tree.getroot()
absolutize(base_root, SCENE_SRC)
template_sub = detect_template_subject(base_root)

# Replace gyrification → tsnr everywhere in the XML
replace_text_global(base_root, GYRI_STEM, TSNR_STEM)

# Disable label/dlabel overlays so tSNR scalar is the only visible layer
disable_label_overlays(base_root)

# Set palette to fixed range
inject_fixed_palette(base_root, vmin, vmax)

# ── Step 3: Render each subject ───────────────────────────────────────────────
print("\nStep 3: Rendering subjects …")
tmpdir = Path(tempfile.mkdtemp(prefix="tsnr_render_"))
render_pngs: dict[str, Path] = {}

for sub in SUBJECTS:
    # Deep copy base tree for this subject
    root = ET.fromstring(ET.tostring(base_root, encoding="unicode"))

    # Replace template subject → target subject
    import re as _re
    template_label = f"sub-{template_sub}"
    target_label   = f"sub-{sub.replace('sub-', '')}"
    for elem in root.iter():
        if elem.text and template_label in elem.text:
            elem.text = elem.text.replace(template_label, target_label)

    # Write per-subject scene
    scene_path = tmpdir / f"{sub}_tsnr.scene"
    tree = ET.ElementTree(root)
    ET.indent(tree, space="    ")
    tree.write(str(scene_path), encoding="unicode", xml_declaration=False)

    out_png = tmpdir / f"{sub}_tsnr_native.png"
    cmd = (
        f'arch -x86_64 "/Applications/wb_view.app/Contents/usr/bin/wb_command"'
        f' -scene-capture-image "{scene_path}" 1 "{out_png}"'
        f' -size-width-height 1600 1200 -renderer OSMesa'
    )
    proc = subprocess.run(cmd, shell=True, text=True, capture_output=True)
    if proc.returncode != 0:
        raise RuntimeError(
            f"wb_command failed for {sub}:\nSTDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}"
        )
    render_pngs[sub] = out_png
    print(f"  rendered {sub} → {out_png}")


# ── Step 4: Split each render into L and R panels ────────────────────────────
print("\nStep 4: Splitting into L/R panels …")
panels: dict[tuple[str, str], np.ndarray] = {}

for sub in SUBJECTS:
    img = np.asarray(Image.open(render_pngs[sub]).convert("RGBA"))
    h, w = img.shape[:2]
    mid = w // 2
    panels[(sub, "L")] = img[:, :mid, :]
    panels[(sub, "R")] = img[:, mid:, :]


# ── Step 5: Compose 3×2 grid with unified colorbar ───────────────────────────
print("\nStep 5: Composing final figure …")

def load_font(size: int) -> ImageFont.ImageFont:
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


cropped = {k: tight_crop(v) for k, v in panels.items()}
cell_h = max(v.shape[0] for v in cropped.values())
cell_w = max(v.shape[1] for v in cropped.values())

LABEL_H  = 52
BAR_W    = 140
GUTTER   = 14
MARGIN   = 28
BG_COLOR = (12, 12, 12, 255)
WHITE    = (255, 255, 255, 255)
GREY     = (170, 170, 170, 255)

n_rows, n_cols = 3, 2
canvas_w = MARGIN * 2 + n_cols * cell_w + (n_cols - 1) * GUTTER + GUTTER + BAR_W
canvas_h = MARGIN * 2 + n_rows * (LABEL_H + cell_h) + (n_rows - 1) * GUTTER + 36

canvas = Image.new("RGBA", (canvas_w, canvas_h), BG_COLOR)

title_font = load_font(28)
tick_font  = load_font(20)

for row, sub in enumerate(SUBJECTS):
    for col, hemi in enumerate(HEMIS):
        panel = cropped[(sub, hemi)]
        ph, pw = panel.shape[:2]
        if pw > cell_w or ph > cell_h:
            scale = min(cell_w / pw, cell_h / ph)
            nw, nh = int(pw * scale), int(ph * scale)
            panel = np.asarray(
                Image.fromarray(panel, "RGBA").resize((nw, nh), Image.Resampling.LANCZOS)
            )
            ph, pw = panel.shape[:2]

        cell_x = MARGIN + col * (cell_w + GUTTER)
        cell_y = MARGIN + 36 + row * (LABEL_H + cell_h + GUTTER)

        draw = ImageDraw.Draw(canvas)
        label = f"{sub.replace('sub-', '')}  ·  {HEMI_LABEL[hemi]} Hipp."
        lx = cell_x + (cell_w - pw) // 2
        draw.text((lx, cell_y + 10), label, fill=WHITE, font=title_font)

        ox = cell_x + (cell_w - pw) // 2
        oy = cell_y + LABEL_H + (cell_h - ph) // 2
        canvas.alpha_composite(Image.fromarray(panel, "RGBA"), (ox, oy))

# Colorbar
cmap_obj = cm.get_cmap("hot")  # matches wb_view ROY-BIG-BL positive range: black→red→orange→yellow
norm     = colors.Normalize(vmin=vmin, vmax=vmax)
bar_x    = canvas_w - BAR_W - MARGIN // 2
bar_top  = MARGIN + 36 + LABEL_H
bar_bot  = canvas_h - MARGIN
bar_h    = bar_bot - bar_top
bar_iw   = 26

vals = np.linspace(vmax, vmin, bar_h, dtype=np.float32)
bar_rgba = np.round(cmap_obj(norm(vals)) * 255).astype(np.uint8).reshape(bar_h, 1, 4)
bar_rgba = np.repeat(bar_rgba, bar_iw, axis=1)
canvas.alpha_composite(Image.fromarray(bar_rgba, "RGBA"), (bar_x + 26, bar_top))

draw = ImageDraw.Draw(canvas)
draw.rectangle((bar_x + 26, bar_top, bar_x + 26 + bar_iw, bar_top + bar_h), outline=GREY, width=1)

for tick_val, tick_y in [(vmax, bar_top), ((vmax + vmin) / 2, bar_top + bar_h // 2), (vmin, bar_top + bar_h)]:
    tx = bar_x + 26 + bar_iw + 6
    draw.line((tx, tick_y, tx + 8, tick_y), fill=GREY, width=1)
    draw.text((tx + 12, tick_y - 10), f"{tick_val:.1f}", fill=GREY, font=tick_font)

# Rotated label
tmp = Image.new("RGBA", (bar_h, 32), (0, 0, 0, 0))
ImageDraw.Draw(tmp).text((0, 4), "tSNR  (10000 / σt)", fill=WHITE, font=tick_font)
tmp = tmp.rotate(90, expand=True)
canvas.alpha_composite(tmp, (bar_x + 2, bar_top + (bar_h - tmp.height) // 2))

# Title
draw.text(
    (MARGIN, 6),
    "Hippocampal tSNR  |  HCP 7T · corobl surface · 10000 / σt",
    fill=WHITE, font=load_font(30),
)

OUT_FIG.parent.mkdir(parents=True, exist_ok=True)
canvas.save(str(OUT_FIG))
print(f"\nSaved → {OUT_FIG}")


# ═══════════════════════════════════════════════════════════════════════════════
# Figure 2: tSNR < 25 shown as gray (NaN → transparent → curvature shows through)
# ═══════════════════════════════════════════════════════════════════════════════
GRAY_THRESH = 25.0
OUT_FIG2    = Path(_args.out_masked)
TSNR_MASKED_STEM = "_tsnr_masked.shape.gii"

print(f"\n{'─'*60}")
print(f"Figure 2: tSNR < {GRAY_THRESH:.0f} masked as gray")
print(f"{'─'*60}")

# ── Save masked shape.gii (NaN where tSNR < threshold) ───────────────────────
print("\nStep 1b: Saving masked tSNR shape.gii …")
for sub in SUBJECTS:
    for hemi in HEMIS:
        tsnr = tsnr_data[(sub, hemi)].copy()
        tsnr[tsnr < GRAY_THRESH] = np.nan
        surf_dir = BATCH_DIR / sub / "hippunfold" / sub / "surf"
        ref_surf = surf_dir / f"{sub}_hemi-{hemi}_space-corobl_den-512_label-hipp_midthickness.surf.gii"
        out_path = surf_dir / f"{sub}_hemi-{hemi}_space-corobl_den-512_label-hipp_tsnr_masked.shape.gii"
        save_shape_gii(tsnr, ref_surf, out_path)
        n_masked = np.isnan(tsnr).sum()
        print(f"  {sub} {hemi}: {n_masked}/{len(tsnr)} vertices masked (<{GRAY_THRESH:.0f})")

# ── Build scene with masked files ─────────────────────────────────────────────
print("\nStep 2b: Building masked scene …")
base_tree2 = ET.parse(str(SCENE_SRC))
base_root2 = base_tree2.getroot()
absolutize(base_root2, SCENE_SRC)
template_sub2 = detect_template_subject(base_root2)

replace_text_global(base_root2, GYRI_STEM, TSNR_MASKED_STEM)
disable_label_overlays(base_root2)
inject_fixed_palette(base_root2, vmin, vmax)

# ── Render ────────────────────────────────────────────────────────────────────
print("\nStep 3b: Rendering subjects …")
tmpdir2 = Path(tempfile.mkdtemp(prefix="tsnr_masked_render_"))
render_pngs2: dict[str, Path] = {}

for sub in SUBJECTS:
    root2 = ET.fromstring(ET.tostring(base_root2, encoding="unicode"))
    import re as _re
    for elem in root2.iter():
        if elem.text and f"sub-{template_sub2}" in elem.text:
            elem.text = elem.text.replace(f"sub-{template_sub2}", sub)

    scene_path2 = tmpdir2 / f"{sub}_tsnr_masked.scene"
    tree2 = ET.ElementTree(root2)
    ET.indent(tree2, space="    ")
    tree2.write(str(scene_path2), encoding="unicode", xml_declaration=False)

    out_png2 = tmpdir2 / f"{sub}_tsnr_masked_native.png"
    cmd2 = (
        f'arch -x86_64 "/Applications/wb_view.app/Contents/usr/bin/wb_command"'
        f' -scene-capture-image "{scene_path2}" 1 "{out_png2}"'
        f' -size-width-height 1600 1200 -renderer OSMesa'
    )
    proc2 = subprocess.run(cmd2, shell=True, text=True, capture_output=True)
    if proc2.returncode != 0:
        raise RuntimeError(f"wb_command failed for {sub}:\n{proc2.stderr}")
    render_pngs2[sub] = out_png2
    print(f"  rendered {sub} → {out_png2}")

# ── Split and compose ─────────────────────────────────────────────────────────
print("\nStep 4b–5b: Splitting and composing …")
panels2: dict[tuple[str, str], np.ndarray] = {}
for sub in SUBJECTS:
    img = np.asarray(Image.open(render_pngs2[sub]).convert("RGBA"))
    mid = img.shape[1] // 2
    panels2[(sub, "L")] = img[:, :mid, :]
    panels2[(sub, "R")] = img[:, mid:, :]

cropped2 = {k: tight_crop(v) for k, v in panels2.items()}
cell_h2 = max(v.shape[0] for v in cropped2.values())
cell_w2 = max(v.shape[1] for v in cropped2.values())

canvas2 = Image.new("RGBA", (canvas_w, canvas_h), BG_COLOR)

for row, sub in enumerate(SUBJECTS):
    for col, hemi in enumerate(HEMIS):
        panel = cropped2[(sub, hemi)]
        ph, pw = panel.shape[:2]
        if pw > cell_w2 or ph > cell_h2:
            scale = min(cell_w2 / pw, cell_h2 / ph)
            panel = np.asarray(
                Image.fromarray(panel, "RGBA").resize((int(pw * scale), int(ph * scale)), Image.Resampling.LANCZOS)
            )
            ph, pw = panel.shape[:2]

        cell_x = MARGIN + col * (cell_w + GUTTER)
        cell_y = MARGIN + 36 + row * (LABEL_H + cell_h + GUTTER)

        draw2 = ImageDraw.Draw(canvas2)
        label = f"{sub.replace('sub-', '')}  ·  {HEMI_LABEL[hemi]} Hipp."
        draw2.text((cell_x + (cell_w - pw) // 2, cell_y + 10), label, fill=WHITE, font=title_font)

        ox = cell_x + (cell_w - pw) // 2
        oy = cell_y + LABEL_H + (cell_h - ph) // 2
        canvas2.alpha_composite(Image.fromarray(panel, "RGBA"), (ox, oy))

# Colorbar (same hot colormap, add gray band annotation)
bar_x2   = canvas_w - BAR_W - MARGIN // 2
bar_top2 = MARGIN + 36 + LABEL_H
bar_bot2 = canvas_h - MARGIN
bar_h2   = bar_bot2 - bar_top2

# Draw full hot colorbar
vals2 = np.linspace(vmax, vmin, bar_h2, dtype=np.float32)
bar_rgba2 = np.round(cmap_obj(norm(vals2)) * 255).astype(np.uint8).reshape(bar_h2, 1, 4)
bar_rgba2 = np.repeat(bar_rgba2, bar_iw, axis=1)
canvas2.alpha_composite(Image.fromarray(bar_rgba2, "RGBA"), (bar_x2 + 26, bar_top2))

draw2 = ImageDraw.Draw(canvas2)

# Gray band overlay on colorbar for < 25
gray_frac = (GRAY_THRESH - vmin) / (vmax - vmin)  # fraction from bottom
gray_px   = int(bar_h2 * (1 - gray_frac))          # pixel from top where gray ends
gray_band = np.full((bar_h2 - gray_px, bar_iw, 4), [120, 120, 120, 255], dtype=np.uint8)
canvas2.alpha_composite(Image.fromarray(gray_band, "RGBA"), (bar_x2 + 26, bar_top2 + gray_px))

draw2.rectangle((bar_x2 + 26, bar_top2, bar_x2 + 26 + bar_iw, bar_top2 + bar_h2), outline=GREY, width=1)

# Dashed threshold line
thresh_y = bar_top2 + gray_px
for dash_x in range(bar_x2 + 18, bar_x2 + 26 + bar_iw + 40, 6):
    draw2.line([(dash_x, thresh_y), (min(dash_x + 3, bar_x2 + 26 + bar_iw + 40), thresh_y)],
               fill=WHITE, width=1)
draw2.text((bar_x2 + 26 + bar_iw + 14, thresh_y - 10),
           f"{GRAY_THRESH:.0f}", fill=WHITE, font=tick_font)

for tick_val, tick_y in [(vmax, bar_top2), (vmin, bar_top2 + bar_h2)]:
    tx = bar_x2 + 26 + bar_iw + 6
    draw2.line((tx, tick_y, tx + 8, tick_y), fill=GREY, width=1)
    draw2.text((tx + 12, tick_y - 10), f"{tick_val:.1f}", fill=GREY, font=tick_font)

tmp2 = Image.new("RGBA", (bar_h2, 32), (0, 0, 0, 0))
ImageDraw.Draw(tmp2).text((0, 4), "tSNR  (10000 / σt)", fill=WHITE, font=tick_font)
tmp2 = tmp2.rotate(90, expand=True)
canvas2.alpha_composite(tmp2, (bar_x2 + 2, bar_top2 + (bar_h2 - tmp2.height) // 2))

draw2.text(
    (MARGIN, 6),
    f"Hippocampal tSNR  |  HCP 7T · corobl surface  [gray: tSNR < {GRAY_THRESH:.0f}]",
    fill=WHITE, font=load_font(30),
)

canvas2.save(str(OUT_FIG2))
print(f"\nSaved → {OUT_FIG2}")
