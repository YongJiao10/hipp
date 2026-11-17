#!/usr/bin/env python3
"""Update Cross-Dataset Hippocampal Parcellation.pptx with all grid combinations.

Supports canonical7 sections by default; canonical8 is an explicit historical option.

Canonical 8-Network Historical Sections:
  - 2mm 36p (Section 1, 7 slides including Surface Area)
  - 4mm 36p (Section 2, 6 slides)

Canonical7 Sections:
  - 2mm_36p (6 slides)
  - 4mm_36p (6 slides)

Pages in each 7net section:
  Page 1: Hippocampal Functional Network Profiles (Gradient)
  Page 2: Subregion Vertices Composition (gradient)
  Page 3: WTA Atlas & Long-Axis Network Composition (WTA)
  Page 4: Cortical Functional Projections (gradient & WTA)
  Page 5: Network Composition of Hippocampal Gradient Atlas
  Page 6: Hippocampal Subfield Composition

Final section:
  - cortical network area (2 slides: Kong2022 and Lynch2024)
"""

from __future__ import annotations

import argparse
import copy
import os
import shutil
import sys
import tempfile
import uuid
from pathlib import Path
from PIL import Image
import pptx
from pptx.oxml import parse_xml
from pptx.util import Pt

# Handle high-resolution cortical and hippocampal renderings without warning
Image.MAX_IMAGE_PIXELS = None

REPO_ROOT = Path("/home/yojiao/hpchome/hipp")
DEFAULT_PPTX_PATH = REPO_ROOT / "Hippocampal Parcellation 7net.pptx"
if not DEFAULT_PPTX_PATH.exists() and (REPO_ROOT / "Cross-Dataset Hippocampal Parcellation.pptx").exists():
    DEFAULT_PPTX_PATH = REPO_ROOT / "Cross-Dataset Hippocampal Parcellation.pptx"
DEFAULT_BACKUP_PATH = DEFAULT_PPTX_PATH.with_suffix(".pptx.bak")

SW, SH = 12192000, 6858000  # 13.333" x 7.500" widescreen

FIGURE_CONFIGS_8NET = [
    {
        "page": 1,
        "title": "Hippocampal Functional Network Profiles (Gradient)",
        "template": str(REPO_ROOT / "scripts/visualize/multidataset_cluster_confidence_stacked/multidataset_cluster_confidence_stacked_8net_{grid}.png"),
    },
    {
        "page": 2,
        "title": "Subregion Vertices Composition (gradient)",
        "template": str(REPO_ROOT / "scripts/visualize/multidataset_workbench_hippocampus_overlay/multidataset_workbench_hippocampus_overlay_8net_{grid}.png"),
    },
    {
        "page": 3,
        "title": "WTA Atlas & Long-Axis Network Composition (WTA)",
        "template": str(REPO_ROOT / "scripts/visualize/multidataset_wta_normalized_longaxis_stacked_2col/multidataset_wta_normalized_longaxis_stacked_2col_{grid}.png"),
    },
    {
        "page": 4,
        "title": "Cortical Functional Projections (gradient & WTA)",
        "template": str(REPO_ROOT / "scripts/visualize/multidataset_hipp_cortex_gradient_wta/multidataset_hipp_cortex_gradient_wta_8net_{grid}.png"),
    },
    {
        "page": 5,
        "title": "Network Composition of Hippocampal Gradient Atlas",
        "template": str(REPO_ROOT / "scripts/visualize/multidataset_gradient_subregion_top20_stacked_2col/multidataset_gradient_subregion_top20_stacked_2col_8net_{grid}.png"),
    },
    {
        "page": 6,
        "title": "Hippocampal Subfield Composition",
        "template": str(REPO_ROOT / "scripts/visualize/multidataset_subfield_composition_2mm_2col/multidataset_subfield_composition_2mm_2col_8net_{grid}.png"),
    },
    {
        "page": 7,
        "title": "Hippocampal Subfield statistical analysis (corrected within each group)",
        "template": str(REPO_ROOT / "scripts/visualize/multidataset_subfield_composition_2mm_2col/stanford_subfield_group_differences_figure_8net_{grid}.png"),
    },
]

FIGURE_CONFIGS_7NET = [
    {
        "page": 1,
        "title": "Hippocampal Functional Network Profiles (Gradient)",
        "template": str(REPO_ROOT / "scripts/visualize/multidataset_cluster_confidence_stacked/multidataset_cluster_confidence_stacked_{grid}.png"),
    },
    {
        "page": 2,
        "title": "Subregion Vertices Composition (gradient)",
        "template": str(REPO_ROOT / "scripts/visualize/multidataset_workbench_hippocampus_overlay/multidataset_workbench_hippocampus_overlay_{grid}.png"),
    },
    {
        "page": 3,
        "title": "WTA Atlas & Long-Axis Network Composition (WTA)",
        "template": str(REPO_ROOT / "scripts/visualize/multidataset_wta_normalized_longaxis_stacked_2col/multidataset_wta_normalized_longaxis_stacked_2col_{grid}.png"),
    },
    {
        "page": 4,
        "title": "Cortical Functional Projections (gradient & WTA)",
        "template": str(REPO_ROOT / "scripts/visualize/multidataset_hipp_cortex_gradient_wta/multidataset_hipp_cortex_gradient_wta_{grid}.png"),
    },
    {
        "page": 5,
        "title": "Network Composition of Hippocampal Gradient Atlas",
        "template": str(REPO_ROOT / "scripts/visualize/multidataset_gradient_subregion_top20_stacked_2col/multidataset_gradient_subregion_top20_stacked_2col_{grid}.png"),
    },
    {
        "page": 6,
        "title": "Hippocampal Subfield Composition",
        "template": str(REPO_ROOT / "scripts/visualize/multidataset_subfield_composition_2mm_2col/multidataset_subfield_composition_2mm_2col_{grid}.png"),
    },
    {
        "page": 7,
        "title": "HCP 7T Local Cortex Regression (Standard vs CtxReg 10 mm vs 20 mm)",
        "template": str(REPO_ROOT / "scripts/visualize/hcp_cortex_control_gradient_wta/hcp_cortex_control_gradient_wta.png"),
    },
    {
        "page": 8,
        "title": "Iowa 3T Template Matching Dominant Subregions & Heatmaps",
        "template": str(REPO_ROOT / "scripts/visualize/template_matching_dominant_subregions/template_matching_dominant_subregions_{grid}.png"),
    },
]

FIGURE_CONFIGS_HCP_JJTMS_7NET = [
    {
        "page": 1,
        "title": "Hippocampal Functional Network Profiles (Gradient)",
        "template": str(REPO_ROOT / "scripts/visualize/multidataset_cluster_confidence_stacked/multidataset_cluster_confidence_stacked_hcp_jjtms_7net_{grid}.png"),
    },
    {
        "page": 2,
        "title": "Subregion Vertices Composition (gradient)",
        "template": str(REPO_ROOT / "scripts/visualize/multidataset_workbench_hippocampus_overlay/multidataset_workbench_hippocampus_overlay_hcp_jjtms_7net_{grid}.png"),
    },
    {
        "page": 3,
        "title": "WTA Atlas & Long-Axis Network Composition (WTA)",
        "template": str(REPO_ROOT / "scripts/visualize/multidataset_wta_normalized_longaxis_stacked_2col/multidataset_wta_normalized_longaxis_stacked_2col_hcp_jjtms_7net_{grid}.png"),
    },
    {
        "page": 4,
        "title": "Cortical Functional Projections (gradient & WTA)",
        "template": str(REPO_ROOT / "scripts/visualize/multidataset_hipp_cortex_gradient_wta/multidataset_hipp_cortex_gradient_wta_{grid}.png"),
    },
    {
        "page": 5,
        "title": "Network Composition of Hippocampal Gradient Atlas",
        "template": str(REPO_ROOT / "scripts/visualize/multidataset_gradient_subregion_top20_stacked_2col/multidataset_gradient_subregion_top20_stacked_2col_hcp_jjtms_7net_{grid}.png"),
    },
    {
        "page": 6,
        "title": "Hippocampal Subfield Composition",
        "template": str(REPO_ROOT / "scripts/visualize/multidataset_subfield_composition_2mm_2col/multidataset_subfield_composition_2mm_2col_hcp_jjtms_7net_{grid}.png"),
    },
    {
        "page": 7,
        "title": "HCP 7T Local Cortex Regression (Standard vs CtxReg 10 mm vs 20 mm)",
        "template": str(REPO_ROOT / "scripts/visualize/hcp_cortex_control_gradient_wta/hcp_cortex_control_gradient_wta.png"),
    },
    {
        "page": 8,
        "title": "Iowa 3T Template Matching Dominant Subregions & Heatmaps",
        "template": str(REPO_ROOT / "scripts/visualize/template_matching_dominant_subregions/template_matching_dominant_subregions_{grid}.png"),
    },
]

SECTIONS_GRID_8NET = [
    ("8net_2mm_36p", "2mm_36p"),
    ("8net_4mm_36p", "4mm_36p"),
]

SECTIONS_GRID_7NET = [
    ("2mm 36p", "2mm_36p"),
    ("4mm 36p", "4mm_36p"),
]

# The checked-in deck predates the current canonical section labels and uses
# ``7net_<grid>``.  This is an explicit section-name migration map, not a
# figure/data fallback: a matching historical section is updated in place and
# ambiguous matches fail closed.
SECTION_ALIASES_7NET = {
    "2mm 36p": ("7net_2mm_36p",),
    "4mm 36p": ("7net_4mm_36p",),
}

CORTEX_NETWORK_AREA_SECTION = "cortical network area"
CORTEX_NETWORK_AREA_PAGES = [
    {
        "title": "Cortical network area — Kong2022",
        "figures": (
            REPO_ROOT / "scripts/visualize/cortex_network_area/figures/cortex_network_area_patient_kong_canonical7_forest.png",
            REPO_ROOT / "scripts/visualize/cortex_network_area/figures/cortex_network_area_patient_kong_native17_forest.png",
        ),
    },
    {
        "title": "Cortical network area — Lynch2024",
        "figures": (
            REPO_ROOT / "scripts/visualize/cortex_network_area/figures/cortex_network_area_patient_lynch_canonical7_forest.png",
            REPO_ROOT / "scripts/visualize/cortex_network_area/figures/cortex_network_area_patient_lynch_native20_forest.png",
        ),
    },
    {
        "title": "Cortical network area — HC baseline",
        "figures": (
            REPO_ROOT / "scripts/visualize/cortex_network_area/figures/cortex_network_area_hc_baseline_forest.png",
        ),
    },
]


def compute_picture_geometry(
    fig_idx: int, img_path: str
) -> tuple[int, int, int, int]:
    """Compute (left, top, width, height) in EMU preserving aspect ratio and maximizing page coverage."""
    with Image.open(img_path) as img:
        w_px, h_px = img.size
    aspect = w_px / h_px

    if fig_idx == 0:  # Page 1: cluster confidence (aspect ~2.98)
        # Full width of slide, vertically centered below title
        pic_w = SW
        pic_h = round(pic_w / aspect)
        pic_left = 0
        pic_top = 1508673 + (5349327 - pic_h) // 2
    elif fig_idx == 1:  # Page 2: subregion vertices overlay (aspect ~1.71)
        # Below the top title (t_bottom = 680724), maximized width and height
        pic_top = 700000
        pic_h = SH - pic_top - 50000  # 6,108,000 EMU
        pic_w = round(pic_h * aspect)  # ~10,420,000 EMU
        pic_left = (SW - pic_w) // 2
    elif fig_idx == 2:  # Page 3: WTA long-axis (aspect ~1.70)
        avail_w = SW - 50000 - 1750000
        pic_w = avail_w
        pic_h = round(pic_w / aspect)
        pic_left = 1750000
        pic_top = (SH - pic_h) // 2
    elif fig_idx == 3:  # Page 4: cortical functional projections (aspect ~1.32)
        # Right of title (t_right = 2158409), maximized height
        pic_h = SH - 2 * 79000  # 6,700,000 EMU
        pic_w = round(pic_h * aspect)  # ~8,831,800 EMU
        avail_w = SW - 50000 - 2250000  # 9,892,000 EMU
        pic_left = 2250000 + (avail_w - pic_w) // 2
        pic_top = 79000
    elif fig_idx == 4:  # Page 5: network composition top20 (aspect ~1.71)
        # Right of title (t_right = 1670858), maximized to slide boundaries
        avail_w = SW - 50000 - 1750000  # 10,392,000 EMU
        pic_w = avail_w
        pic_h = round(pic_w / aspect)  # ~6,063,000 EMU
        pic_left = 1750000
        pic_top = (SH - pic_h) // 2
    elif fig_idx == 5:  # Page 6: subfield composition (aspect ~1.41)
        # Right of title (t_right = 2158409), maximized height
        pic_h = SH - 2 * 79000  # 6,700,000 EMU
        pic_w = round(pic_h * aspect)  # ~9,429,580 EMU
        avail_w = SW - 50000 - 2250000  # 9,892,000 EMU
        pic_left = 2250000 + (avail_w - pic_w) // 2
        pic_top = 79000
    elif fig_idx == 6:  # Page 7: HCP cortex proximity control (aspect ~2.72) OR 8net subfield statistical analysis
        if aspect > 2.0:
            pic_w = SW
            pic_h = round(pic_w / aspect)
            pic_left = 0
            pic_top = 1508673 + (5349327 - pic_h) // 2
        else:
            pic_w = 10592000
            pic_h = round(pic_w / aspect)
            pic_left = SW - pic_w
            pic_top = (SH - pic_h) // 2
    elif fig_idx == 7:  # Page 8: Iowa 3T template matching dominant subregions (aspect ~0.75)
        pic_h = SH - 2 * 79000  # 6,700,000 EMU
        pic_w = round(pic_h * aspect)  # ~5,038,000 EMU
        avail_w = SW - 50000 - 2250000  # 9,892,000 EMU
        pic_left = 2250000 + (avail_w - pic_w) // 2
        pic_top = 79000
    else:
        pic_h = 5500000
        pic_top = (SH - pic_h) // 2
        pic_w = round(pic_h * aspect)
        pic_left = SW - pic_w - 50000

    return pic_left, pic_top, pic_w, pic_h


def resolve_existing_section_name(
    section_map: dict[str, object], canonical_name: str, aliases: tuple[str, ...] = ()
) -> str | None:
    """Resolve one explicitly supported section name, failing on ambiguity."""
    candidates = (canonical_name, *aliases)
    matches = [name for name in candidates if name in section_map]
    if len(matches) > 1:
        raise RuntimeError(
            f"Ambiguous section names for {canonical_name!r}: {matches}. "
            "Keep exactly one supported section name before synchronizing."
        )
    return matches[0] if matches else None


def compute_two_picture_geometry(
    image_paths: tuple[Path, Path],
) -> tuple[tuple[int, int, int, int], tuple[int, int, int, int]]:
    """Return two side-by-side contain boxes for a method page."""
    if len(image_paths) != 2:
        raise ValueError(f"Expected exactly two images, got {len(image_paths)}")

    margin_x = 120000
    gutter = 90000
    content_top = 650000
    content_bottom = 80000
    column_width = (SW - 2 * margin_x - gutter) // 2
    content_height = SH - content_top - content_bottom
    boxes = []

    for column, image_path in enumerate(image_paths):
        with Image.open(image_path) as image:
            image_width, image_height = image.size
        if image_width <= 0 or image_height <= 0:
            raise ValueError(f"Invalid image dimensions: {image_path}")

        aspect = image_width / image_height
        picture_width = column_width
        picture_height = round(picture_width / aspect)
        if picture_height > content_height:
            picture_height = content_height
            picture_width = round(picture_height * aspect)

        column_left = margin_x + column * (column_width + gutter)
        picture_left = column_left + (column_width - picture_width) // 2
        picture_top = content_top + (content_height - picture_height) // 2
        boxes.append((picture_left, picture_top, picture_width, picture_height))

    return boxes[0], boxes[1]


def _remove_picture_shapes(slide: pptx.slide.Slide) -> None:
    """Remove only picture shapes and their image relationships from a slide."""
    old_pics = [
        shape
        for shape in slide.shapes
        if shape.shape_type == pptx.enum.shapes.MSO_SHAPE_TYPE.PICTURE
    ]
    for picture in old_pics:
        for blip in picture._element.xpath('.//*[local-name()="blip"]'):
            for key, value in blip.attrib.items():
                if key.endswith("embed") and value in slide.part.rels:
                    slide.part.drop_rel(value)
        slide.shapes._spTree.remove(picture._element)


def _set_area_title(slide: pptx.slide.Slide, title: str) -> None:
    """Set or create the single title box owned by an area slide."""
    title_shapes = [shape for shape in slide.shapes if shape.has_text_frame]
    if title_shapes:
        title_shape = title_shapes[0]
        title_shape.left = 260000
        title_shape.top = 120000
        title_shape.width = SW - 520000
        title_shape.height = 420000
    else:
        title_shape = slide.shapes.add_textbox(260000, 120000, SW - 520000, 420000)

    text_frame = title_shape.text_frame
    text_frame.clear()
    text_frame.margin_left = 0
    text_frame.margin_right = 0
    text_frame.margin_top = 0
    text_frame.margin_bottom = 0
    paragraph = text_frame.paragraphs[0]
    paragraph.text = title
    paragraph.alignment = pptx.enum.text.PP_ALIGN.LEFT
    run = paragraph.runs[0]
    run.font.name = "Arial"
    run.font.size = Pt(24)
    run.font.bold = True
    run.font.color.rgb = pptx.dml.color.RGBColor(31, 52, 73)

def format_slide_title(slide: pptx.slide.Slide, fig_idx: int, title: str) -> None:
    """Format title and optional scientific notes on slide text frame."""
    tbs = [s for s in slide.shapes if s.has_text_frame]
    if not tbs:
        return
    tb = tbs[0]
    tf = tb.text_frame

    if fig_idx == 6:  # Page 7: HCP 7T Cortical Proximity Control
        # Position top title and explanatory notes cleanly above picture (pic_top ~ 1943990)
        tb.left = 250000
        tb.top = 150000
        tb.width = 11692000
        tb.height = 1650000

        tf.word_wrap = True
        tf.clear()

        p0 = tf.paragraphs[0]
        p0.text = title
        p0.font.name = "Arial"
        p0.font.size = Pt(20)
        p0.font.bold = True
        p0.font.color.rgb = pptx.dml.color.RGBColor(31, 52, 73)
        p0.space_after = Pt(6)

        p1 = tf.add_paragraph()
        r1_b = p1.add_run()
        r1_b.text = "• CtxReg (Local Cortex Regression): "
        r1_b.font.name = "Arial"
        r1_b.font.size = Pt(13)
        r1_b.font.bold = True
        r1_b.font.color.rgb = pptx.dml.color.RGBColor(180, 50, 50)

        r1_t = p1.add_run()
        r1_t.text = "Regresses the mean of cortex within 10 mm or 20 mm out of each hippocampal time series per run; a conservative sensitivity analysis that also removes signal shared at the hippocampal boundary. The figure shows 2mm (top) and 4mm (bottom) smoothing."
        r1_t.font.name = "Arial"
        r1_t.font.size = Pt(12)
        r1_t.font.color.rgb = pptx.dml.color.RGBColor(70, 70, 70)
        p1.space_after = Pt(3)

        p2 = tf.add_paragraph()
        r2_b = p2.add_run()
        r2_b.text = "• Local cortex diagnostics: "
        r2_b.font.name = "Arial"
        r2_b.font.size = Pt(13)
        r2_b.font.bold = True
        r2_b.font.color.rgb = pptx.dml.color.RGBColor(30, 100, 160)

        r2_t = p2.add_run()
        r2_t.text = "Excluding local cortex from the ROI targets barely changes FC or WTA labels; CtxReg changes are driven by signal shared within about 3 mm of the hippocampus (06_07 diagnostics)."
        r2_t.font.name = "Arial"
        r2_t.font.size = Pt(12)
        r2_t.font.color.rgb = pptx.dml.color.RGBColor(70, 70, 70)
    else:
        tb.text = title


def _register_section(
    section_list, section_name: str, slide_ids: list[str]
):
    """Append a PowerPoint section and return its XML element."""
    guid = f"{{{str(uuid.uuid4()).upper()}}}"
    section = parse_xml(
        f'<p14:section xmlns:p14="http://schemas.microsoft.com/office/powerpoint/2010/main" '
        f'name="{section_name}" id="{guid}"><p14:sldIdLst/></p14:section>'
    )
    slide_id_list = section.xpath(".//*[local-name()='sldIdLst']")[0]
    for slide_id in slide_ids:
        slide_id_list.append(
            parse_xml(
                f'<p14:sldId xmlns:p14="http://schemas.microsoft.com/office/powerpoint/2010/main" '
                f'id="{slide_id}"/>'
            )
        )
    section_list.append(section)
    return section


def update_cortical_network_area_section(
    prs: pptx.Presentation,
    section_list,
    section_map: dict[str, object],
    slide_by_id: dict[str, pptx.slide.Slide],
) -> None:
    """Add or refresh the two-page Kong/Lynch cortical-area section."""
    for page in CORTEX_NETWORK_AREA_PAGES:
        for image_path in page["figures"]:
            if not image_path.exists():
                raise FileNotFoundError(
                    f"Missing cortical network area figure required by the PPTX: {image_path}"
                )

    existing_name = resolve_existing_section_name(section_map, CORTEX_NETWORK_AREA_SECTION)
    if existing_name is not None:
        section = section_map[existing_name]
        slide_ids = [
            child.get("id") for child in section.xpath(".//*[local-name()='sldId']")
        ]
        if len(slide_ids) != len(CORTEX_NETWORK_AREA_PAGES):
            raise RuntimeError(
                f"Section {existing_name!r} has {len(slide_ids)} slides; "
                f"expected {len(CORTEX_NETWORK_AREA_PAGES)}. Refusing partial update."
            )
    else:
        section = None
        slide_ids = []

    if section is None:
        blank_layout = prs.slide_layouts[6]
        for page in CORTEX_NETWORK_AREA_PAGES:
            slide = prs.slides.add_slide(blank_layout)
            slide_id = str(slide.slide_id)
            slide_ids.append(slide_id)
            slide_by_id[slide_id] = slide
        section = _register_section(
            section_list, CORTEX_NETWORK_AREA_SECTION, slide_ids
        )
        section_map[CORTEX_NETWORK_AREA_SECTION] = section

    for page, slide_id in zip(CORTEX_NETWORK_AREA_PAGES, slide_ids):
        slide = slide_by_id.get(slide_id)
        if slide is None:
            raise RuntimeError(
                f"Section {CORTEX_NETWORK_AREA_SECTION!r} references missing slide id {slide_id}"
            )
        _remove_picture_shapes(slide)
        _set_area_title(slide, page["title"])
        if len(page["figures"]) == 2:
            geom = compute_two_picture_geometry(page["figures"])
        else:
            geom = [(2471000, 720000, 7300000, 6051000)]
        for image_path, (left, top, width, height) in zip(page["figures"], geom):
            slide.shapes.add_picture(str(image_path), left, top, width, height)

    print(
        f"  Updated section '{CORTEX_NETWORK_AREA_SECTION}' with "
        f"{len(CORTEX_NETWORK_AREA_PAGES)} Kong/Lynch slides."
    )


def ensure_comparison_first_slide(prs: pptx.Presentation, img_path: Path) -> None:
    """Ensure kong_lynch comparison image is on Slide 1, not belonging to any section."""
    if not img_path.exists():
        print(f"[Warning] Comparison first slide image not found at: {img_path}")
        return

    SW, SH = prs.slide_width, prs.slide_height
    with Image.open(img_path) as im:
        aspect = im.size[0] / im.size[1]

    # Calculate maximized geometry filling slide height
    pic_h = SH
    pic_w = round(pic_h * aspect)
    pic_top = 0
    pic_left = (SW - pic_w) // 2

    secLst_nodes = prs._element.xpath(".//*[local-name()='sectionLst']")
    secLst = secLst_nodes[0] if secLst_nodes else None
    all_sec_sld_ids = set()
    if secLst is not None:
        for c in secLst.xpath(".//*[local-name()='sldId']"):
            all_sec_sld_ids.add(c.get("id"))

    # Check if first slide is already a standalone comparison slide (outside any section)
    if len(prs.slides) > 0:
        first_slide = prs.slides[0]
        fs_id = str(first_slide.slide_id)
        if fs_id not in all_sec_sld_ids:
            # First slide is already not in any section! Update its picture
            old_pics = [s for s in first_slide.shapes if s.shape_type == pptx.enum.shapes.MSO_SHAPE_TYPE.PICTURE]
            for op in old_pics:
                for b in op._element.xpath('.//*[local-name()="blip"]'):
                    for k, v in b.attrib.items():
                        if k.endswith("embed") and v in first_slide.part.rels:
                            first_slide.part.drop_rel(v)
                first_slide.shapes._spTree.remove(op._element)
            first_slide.shapes.add_picture(str(img_path), pic_left, pic_top, pic_w, pic_h)
            # Remove any black background element so Slide 1 uses the clean default layout background
            for bg in first_slide._element.xpath("./p:bg"):
                first_slide._element.remove(bg)
            print(f"  Slide 1 (id={fs_id}) already outside sections: updated comparison image ({os.path.basename(img_path)})")
            return

    # Otherwise, create a new blank slide and insert at index 0 of sldIdLst
    blank_layout = prs.slide_layouts[6]
    new_slide = prs.slides.add_slide(blank_layout)
    new_slide.shapes.add_picture(str(img_path), pic_left, pic_top, pic_w, pic_h)
    for bg in new_slide._element.xpath("./p:bg"):
        new_slide._element.remove(bg)

    # Move new_slide to index 0 in sldIdLst
    sldIdLst = prs._element.xpath("//p:sldIdLst")[0]
    new_sldId_elem = sldIdLst[-1]
    sldIdLst.remove(new_sldId_elem)
    sldIdLst.insert(0, new_sldId_elem)
    print(f"  Inserted Slide 1 (id={new_slide.slide_id}) outside sections: {os.path.basename(img_path)}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Update or add sections into Cross-Dataset Hippocampal Parcellation.pptx."
    )
    parser.add_argument(
        "--version",
        choices=["legacy", "hcp_jjtms_7net"],
        default="hcp_jjtms_7net",
        help="Comparison version: 'hcp_jjtms_7net' (HCP 7T + Iowa JJ_TMS 7net, default) or 'legacy' (12-cohort Stanford suite).",
    )
    parser.add_argument(
        "--atlas",
        choices=["all", "kong2022", "8net", "7net"],
        default="7net",
        help="Atlas suite to process: 7net/canonical7 is the default and uses unsuffixed outputs; use '8net'/'kong2022' only for explicit historical output, or 'all' for both.",
    )
    parser.add_argument(
        "--pptx-path",
        type=Path,
        default=DEFAULT_PPTX_PATH,
        help=f"Path to master PPTX file (default: {DEFAULT_PPTX_PATH})",
    )
    parser.add_argument(
        "--backup-path",
        type=Path,
        default=DEFAULT_BACKUP_PATH,
        help=f"Path to backup PPTX file (default: {DEFAULT_BACKUP_PATH})",
    )
    parser.add_argument(
        "--page",
        type=int,
        choices=range(1, 10),
        default=None,
        help="Update only a specific page/figure number (1-based, e.g. 6 for subfield composition). Default: update all pages in the section.",
    )
    parser.add_argument(
        "--comparison-img",
        type=Path,
        default=REPO_ROOT / "scripts/visualize/kong2022_standard_lateral/kong_lynch_native_canonical7_comparison.png",
        help="Path to comparison figure to place on Slide 1 outside sections.",
    )
    parser.add_argument(
        "--skip-comparison",
        action="store_true",
        help="Skip ensuring comparison image is on Slide 1.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Check figure paths and print actions without modifying the presentation file.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    pptx_path: Path = args.pptx_path
    backup_path: Path = args.backup_path

    if not pptx_path.exists():
        raise FileNotFoundError(f"Presentation file not found at: {pptx_path}")

    # Determine which suites to process
    is_hcp_jjtms = args.version == "hcp_jjtms_7net"
    fig_configs_7net = FIGURE_CONFIGS_HCP_JJTMS_7NET if is_hcp_jjtms else FIGURE_CONFIGS_7NET
    do_8net = (not is_hcp_jjtms) and args.atlas in ("all", "kong2022", "8net")
    do_7net = args.atlas in ("all", "7net")

    print("=" * 78)
    print("CROSS-DATASET PRESENTATION SLIDES SYNCHRONIZER")
    print(f"Target PPTX: {pptx_path}")
    print(f"Version: {args.version}")
    print(f"Process 8net: {do_8net} | Process 7net: {do_7net} | Filter Page: {args.page}")
    print("=" * 78)

    # Preflight check all required images
    missing_files = []
    if do_8net:
        for sec_name, grid in SECTIONS_GRID_8NET:
            for cfg in FIGURE_CONFIGS_8NET:
                if args.page is not None and cfg["page"] != args.page:
                    continue
                p = Path(cfg["template"].format(grid=grid))
                if not p.exists():
                    missing_files.append(str(p))
    if do_7net:
        for sec_name, grid in SECTIONS_GRID_7NET:
            for cfg in fig_configs_7net:
                if args.page is not None and cfg["page"] != args.page:
                    continue
                p = Path(cfg["template"].format(grid=grid))
                if not p.exists():
                    missing_files.append(str(p))

    if args.page is None:
        for area_page in CORTEX_NETWORK_AREA_PAGES:
            for p in area_page["figures"]:
                if not p.exists():
                    missing_files.append(str(p))

    if not args.skip_comparison and args.comparison_img:
        if not args.comparison_img.exists():
            missing_files.append(str(args.comparison_img))

    if missing_files:
        print("\nERROR: The following required figure files were not found:")
        for mf in missing_files:
            print(f"  - {mf}")
        sys.exit(1)
    print("\nPreflight check: All required figure images verified on disk.")

    if args.dry_run:
        print("\n[DRY RUN] Verification successful. Exiting without modifying presentation.")
        return

    # Create initial backup if none exists
    if not backup_path.exists():
        print(f"\nCreating initial backup: {backup_path}")
        shutil.copyfile(pptx_path, backup_path)
    else:
        print(f"\nExisting backup found: {backup_path}")

    prs = pptx.Presentation(str(pptx_path))
    blank_layout = prs.slide_layouts[6]

    # Ensure Slide 1 is comparison figure outside any section
    if not args.skip_comparison and args.comparison_img:
        print("\n--- Ensuring Slide 1 Comparison Image (Outside Sections) ---")
        ensure_comparison_first_slide(prs, args.comparison_img)

    # Map existing slides by slide_id
    slide_by_id = {str(s.slide_id): s for s in prs.slides}

    # Inspect XML sections
    sectionLst_nodes = prs._element.xpath(".//*[local-name()='sectionLst']")
    if not sectionLst_nodes:
        raise RuntimeError("No <sectionLst> found in presentation XML!")
    sectionLst = sectionLst_nodes[0]

    sec_map = {s.get("name"): s for s in sectionLst}
    print("Existing XML sections in presentation:", list(sec_map.keys()))

    sections_to_process: list[tuple[str, str, list[dict], str]] = []
    if do_8net:
        for sec_name, grid in SECTIONS_GRID_8NET:
            sections_to_process.append((sec_name, grid, FIGURE_CONFIGS_8NET, "8net"))
    if do_7net:
        for sec_name, grid in SECTIONS_GRID_7NET:
            sections_to_process.append((sec_name, grid, fig_configs_7net, "7net"))

    for sec_name, grid, fig_configs, atlas_tag in sections_to_process:
        print(f"\n--- Processing Section: '{sec_name}' (grid: {grid}, {atlas_tag}) ---")
        aliases = SECTION_ALIASES_7NET.get(sec_name, ()) if atlas_tag == "7net" else ()
        resolved_name = resolve_existing_section_name(sec_map, sec_name, aliases)
        if resolved_name is not None:
            # Section already exists: update existing slides in-place
            sec_elem = sec_map[resolved_name]
            sld_id_elems = sec_elem.xpath(".//*[local-name()='sldId']")
            existing_sld_ids = [c.get("id") for c in sld_id_elems]
            print(f"  Section '{resolved_name}' already exists ({len(existing_sld_ids)} slides). Updating in-place...")

            for fig_idx, cfg in enumerate(fig_configs):
                if args.page is not None and cfg["page"] != args.page:
                    continue
                if fig_idx < len(existing_sld_ids):
                    sid = existing_sld_ids[fig_idx]
                    slide = slide_by_id.get(sid)
                    if slide is None:
                        print(f"  [Warning] Slide id {sid} not found in presentation slides!")
                        continue

                    img_path = cfg["template"].format(grid=grid)
                    # Update title text and explanatory notes if present
                    if "title" in cfg:
                        format_slide_title(slide, fig_idx, cfg["title"])

                    # Remove existing picture shape(s) and drop their relationships
                    old_pics = [s for s in slide.shapes if s.shape_type == pptx.enum.shapes.MSO_SHAPE_TYPE.PICTURE]
                    left, top, w, h = compute_picture_geometry(fig_idx, img_path)
                    for op in old_pics:
                        for b in op._element.xpath('.//*[local-name()="blip"]'):
                            for k, v in b.attrib.items():
                                if k.endswith("embed") and v in slide.part.rels:
                                    slide.part.drop_rel(v)
                        slide.shapes._spTree.remove(op._element)

                    # Add new picture shape
                    slide.shapes.add_picture(img_path, left, top, w, h)
                    print(f"  Updated Slide {sid}: {cfg['title'][:40]} -> {os.path.basename(img_path)}")
                else:
                    # New slide for existing section (e.g. Page 7 or Page 8)
                    new_slide = prs.slides.add_slide(blank_layout)
                    sid_str = str(new_slide.slide_id)
                    slide_by_id[sid_str] = new_slide

                    # Choose appropriate template slide for title text box
                    if fig_idx == 6:  # Page 7: top title
                        tmpl_sid = existing_sld_ids[0]
                    else:  # Page 8: left title
                        tmpl_sid = existing_sld_ids[min(5, len(existing_sld_ids) - 1)]
                    tmpl_slide = slide_by_id.get(tmpl_sid) or prs.slides[0]
                    tb_shapes = [s for s in tmpl_slide.shapes if s.has_text_frame]
                    if tb_shapes:
                        new_tb = copy.deepcopy(tb_shapes[0]._element)
                        new_slide.shapes._spTree.append(new_tb)
                        if "title" in cfg:
                            format_slide_title(new_slide, fig_idx, cfg["title"])

                    img_path = cfg["template"].format(grid=grid)
                    left, top, w, h = compute_picture_geometry(fig_idx, img_path)
                    new_slide.shapes.add_picture(img_path, left, top, w, h)

                    # Move new_slide in presentation's sldIdLst right after the previous slide of this section
                    prev_sid = existing_sld_ids[-1]
                    presentation_sldIdLst = prs._element.xpath("//p:sldIdLst")[0]
                    new_sld_elem = next(el for el in presentation_sldIdLst if el.get("id") == sid_str)
                    presentation_sldIdLst.remove(new_sld_elem)
                    prev_elem = next(el for el in presentation_sldIdLst if el.get("id") == prev_sid)
                    prev_idx = list(presentation_sldIdLst).index(prev_elem)
                    presentation_sldIdLst.insert(prev_idx + 1, new_sld_elem)

                    # Register in section's sldIdLst
                    sec_sldIdLst = sec_elem.xpath(".//*[local-name()='sldIdLst']")[0]
                    sec_sldIdLst.append(
                        parse_xml(
                            f'<p14:sldId xmlns:p14="http://schemas.microsoft.com/office/powerpoint/2010/main" id="{sid_str}"/>'
                        )
                    )
                    existing_sld_ids.append(sid_str)
                    print(f"  Added Slide {sid_str} to section '{resolved_name}': {cfg['title'][:40]} -> {os.path.basename(img_path)}")

        else:
            # Section does not exist: create new slides and append new section
            print(f"  Section '{sec_name}' does not exist. Creating new section and {len(fig_configs)} slides...")
            new_sld_ids: list[str] = []

            for fig_idx, cfg in enumerate(fig_configs):
                # Pick a template slide that has a title text frame shape
                template_slide = None
                if fig_idx == 6:
                    template_slide = prs.slides[1] if len(prs.slides) > 1 else prs.slides[0]
                elif fig_idx == 7:
                    template_slide = prs.slides[6] if len(prs.slides) > 6 else prs.slides[0]
                else:
                    template_slide = prs.slides[fig_idx] if fig_idx < len(prs.slides) else prs.slides[0]
                tb_shapes = [s for s in template_slide.shapes if s.has_text_frame]
                if not tb_shapes:
                    tb_shapes = [s for s in prs.slides[0].shapes if s.has_text_frame]
                if not tb_shapes:
                    raise RuntimeError("Template slide has no text frame shape!")
                tb_shape = tb_shapes[0]

                new_slide = prs.slides.add_slide(blank_layout)
                sid_str = str(new_slide.slide_id)
                new_sld_ids.append(sid_str)
                slide_by_id[sid_str] = new_slide

                # Deep-copy title text box from template slide
                new_tb = copy.deepcopy(tb_shape._element)
                new_slide.shapes._spTree.append(new_tb)
                if "title" in cfg:
                    format_slide_title(new_slide, fig_idx, cfg["title"])

                # Add picture shape
                img_path = cfg["template"].format(grid=grid)
                left, top, w, h = compute_picture_geometry(fig_idx, img_path)
                new_slide.shapes.add_picture(img_path, left, top, w, h)
                print(f"  Added Slide {sid_str}: {cfg['title'][:40]} -> {os.path.basename(img_path)}")

            # Create XML <p14:section> and append to <sectionLst>
            guid = f"{{{str(uuid.uuid4()).upper()}}}"
            new_sec_elem = parse_xml(
                f'<p14:section xmlns:p14="http://schemas.microsoft.com/office/powerpoint/2010/main" '
                f'name="{sec_name}" id="{guid}">'
                f'<p14:sldIdLst/>'
                f'</p14:section>'
            )
            sldIdLst = new_sec_elem.xpath(".//*[local-name()='sldIdLst']")[0]
            for sid in new_sld_ids:
                sldIdLst.append(
                    parse_xml(
                        f'<p14:sldId xmlns:p14="http://schemas.microsoft.com/office/powerpoint/2010/main" id="{sid}"/>'
                    )
                )
            sectionLst.append(new_sec_elem)
            sec_map[sec_name] = new_sec_elem
            print(f"  Section '{sec_name}' registered in sectionLst with {len(new_sld_ids)} slides.")

    if args.page is None:
        print("\n--- Processing Section: 'cortical network area' ---")
        update_cortical_network_area_section(prs, sectionLst, sec_map, slide_by_id)

    # Sweep and purge all unreferenced image relationships across all slides
    total_purged = 0
    for s in prs.slides:
        used_rIds = set()
        for shp in s.shapes:
            for b in shp._element.xpath('.//*[local-name()="blip"]'):
                for k, v in b.attrib.items():
                    if k.endswith("embed"):
                        used_rIds.add(v)
        for rId, rel in list(s.part.rels.items()):
            if "image" in rel.reltype and rId not in used_rIds:
                s.part.drop_rel(rId)
                total_purged += 1
    if total_purged > 0:
        print(f"Purged {total_purged} orphaned image relationships from presentation.")

    print("\n--- Saving Presentation ---")
    with tempfile.NamedTemporaryFile(suffix=".pptx", delete=False) as tmp:
        tmp_path = tmp.name

    try:
        prs.save(tmp_path)
        # Validate readability
        prs_check = pptx.Presentation(tmp_path)
        reloaded_sections = [s.get("name") for s in prs_check._element.xpath(".//*[local-name()='sectionLst']")[0]]
        print(f"Validation success: {len(prs_check.slides)} total slides loaded.")
        print("Presentation sections:", reloaded_sections)
        shutil.move(tmp_path, str(pptx_path))
        print(f"Successfully updated presentation: {pptx_path}")

        # Keep companion presentation file in lockstep
        companion_path = (
            REPO_ROOT / "Cross-Dataset Hippocampal Parcellation.pptx"
            if pptx_path.name == "Hippocampal Parcellation 7net.pptx"
            else REPO_ROOT / "Hippocampal Parcellation 7net.pptx"
        )
        shutil.copyfile(str(pptx_path), str(companion_path))
        print(f"Successfully synchronized companion deck: {companion_path}")
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


if __name__ == "__main__":
    main()
