"""Tests for Cross-Dataset Hippocampal Parcellation presentation contract.

Verifies:
1. Slide 1 contains the standalone comparison figure outside all sections.
2. Slide 1 has clean default master background (no black fill) and centered, aspect-preserved geometry.
3. Section slides strictly preserve image aspect ratios without title overlap or boundary overflow.
4. Exactly 4 7net sections are registered with 8 slides each.
5. The final cortical network area section has three method pages.
6. Page 7 explains CtxReg and the local-cortex diagnostics.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
import pptx
from pptx.enum.shapes import MSO_SHAPE_TYPE
import pytest

REPO_ROOT = Path("/home/yojiao/hpchome/hipp")
PPTX_PATH = (
    REPO_ROOT / "Hippocampal Parcellation 7net.pptx"
    if (REPO_ROOT / "Hippocampal Parcellation 7net.pptx").exists()
    else REPO_ROOT / "Cross-Dataset Hippocampal Parcellation.pptx"
)
PRESENTATION_UPDATER_PATH = REPO_ROOT / "scripts/visualize/update_cross_dataset_presentation.py"


@pytest.fixture(scope="module")
def presentation():
    if not PPTX_PATH.exists():
        pytest.skip(f"{PPTX_PATH} does not exist.")
    return pptx.Presentation(str(PPTX_PATH))


def test_canonical7_wta_template_uses_unsuffixed_grid_output():
    spec = importlib.util.spec_from_file_location(
        "cross_dataset_presentation_updater", PRESENTATION_UPDATER_PATH
    )
    assert spec is not None and spec.loader is not None
    updater = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(updater)

    wta_config = next(config for config in updater.FIGURE_CONFIGS_7NET if config["page"] == 3)
    expected_template = str(
        REPO_ROOT
        / "scripts/visualize/multidataset_wta_normalized_longaxis_stacked_2col"
        / "multidataset_wta_normalized_longaxis_stacked_2col_{grid}.png"
    )

    assert wta_config["template"] == expected_template

def test_presentation_total_slide_count(presentation):
    assert len(presentation.slides) == 36, f"Expected 36 slides, got {len(presentation.slides)}"


def test_slide_1_outside_all_sections(presentation):
    secLst_nodes = presentation._element.xpath(".//*[local-name()='sectionLst']")
    assert secLst_nodes, "No sectionLst found in presentation XML"
    secLst = secLst_nodes[0]

    all_sec_sids = set()
    for s in secLst:
        for c in s.xpath(".//*[local-name()='sldId']"):
            all_sec_sids.add(c.get("id"))

    first_slide = presentation.slides[0]
    first_sid = str(first_slide.slide_id)
    assert first_sid not in all_sec_sids, f"Slide 1 (id={first_sid}) must NOT belong to any section!"


def test_slide_1_comparison_image_geometry(presentation):
    first_slide = presentation.slides[0]
    pics = [s for s in first_slide.shapes if s.shape_type == MSO_SHAPE_TYPE.PICTURE]
    assert len(pics) == 1, "Slide 1 must have exactly 1 picture shape"
    pic = pics[0]

    SW, SH = presentation.slide_width, presentation.slide_height
    assert pic.height == SH, f"Picture height {pic.height} should match slide height {SH}"
    assert pic.top == 0, f"Picture top {pic.top} should be 0"
    assert pic.left > 0, "Picture should be horizontally centered"
    assert pic.left + pic.width <= SW, "Picture should not overflow slide width"


def test_slide_1_no_black_background(presentation):
    first_slide = presentation.slides[0]
    bg_nodes = first_slide._element.xpath("./p:bg")
    if bg_nodes:
        solid_fill = bg_nodes[0].xpath(".//a:solidFill/a:srgbClr")
        if solid_fill:
            assert solid_fill[0].get("val").upper() != "000000", "Slide 1 should not have black background"


def test_cortical_network_area_section_has_method_pages(presentation):
    section_nodes = presentation._element.xpath(".//*[local-name()='sectionLst']")[0]
    area_nodes = [s for s in section_nodes if s.get("name") == "cortical network area"]
    assert len(area_nodes) == 1, "Exactly one cortical network area section is required"

    area_ids = [c.get("id") for c in area_nodes[0].xpath(".//*[local-name()='sldId']")]
    assert len(area_ids) == 3, "The cortical network area section must contain three slides"
    slides_by_id = {str(slide.slide_id): slide for slide in presentation.slides}

    expected_sizes = [
        {(3210, 1863), (3210, 3108)},
        {(3210, 1863), (3210, 3504)},
        {(3118, 2584)},
    ]
    SW, SH = presentation.slide_width, presentation.slide_height

    for sid, sizes in zip(area_ids, expected_sizes):
        slide = slides_by_id[sid]
        pics = [s for s in slide.shapes if s.shape_type == MSO_SHAPE_TYPE.PICTURE]
        assert len(pics) == len(sizes), f"Each method page must contain {len(sizes)} forest figures"
        assert {pic.image.size for pic in pics} == sizes
        for pic in pics:
            assert pic.left >= 0 and pic.top >= 0
            assert pic.left + pic.width <= SW
            assert pic.top + pic.height <= SH

        if len(pics) == 2:
            first, second = sorted(pics, key=lambda pic: pic.left)
            assert first.left + first.width <= second.left, "The two figures must not overlap"


def test_section_slides_aspect_and_no_overlap(presentation):
    SW, SH = presentation.slide_width, presentation.slide_height

    section_nodes = presentation._element.xpath(".//*[local-name()='sectionLst']")[0]
    area_nodes = [s for s in section_nodes if s.get("name") == "cortical network area"]
    area_ids = {
        c.get("id")
        for c in area_nodes[0].xpath(".//*[local-name()='sldId']")
    } if area_nodes else set()

    for idx in range(1, len(presentation.slides)):
        slide = presentation.slides[idx]
        if str(slide.slide_id) in area_ids:
            continue
        tbs = [s for s in slide.shapes if s.has_text_frame]
        pics = [s for s in slide.shapes if s.shape_type == MSO_SHAPE_TYPE.PICTURE]

        assert len(pics) == 1, f"Slide {idx + 1} must contain exactly 1 picture"
        pic = pics[0]

        # No boundary overflow
        assert pic.left >= 0, f"Slide {idx + 1} picture left {pic.left} < 0"
        assert pic.top >= 0, f"Slide {idx + 1} picture top {pic.top} < 0"
        assert pic.left + pic.width <= SW, f"Slide {idx + 1} picture right exceeds slide width"
        assert pic.top + pic.height <= SH, f"Slide {idx + 1} picture bottom exceeds slide height"

        # Check disjoint bounding box with title text box
        if tbs:
            tb = tbs[0]
            tb_left, tb_top, tb_right, tb_bottom = tb.left, tb.top, tb.left + tb.width, tb.top + tb.height
            pic_left, pic_top, pic_right, pic_bottom = pic.left, pic.top, pic.left + pic.width, pic.top + pic.height

            overlap_x = not (pic_right <= tb_left or pic_left >= tb_right)
            overlap_y = not (pic_bottom <= tb_top or pic_top >= tb_bottom)
            assert not (overlap_x and overlap_y), (
                f"Slide {idx + 1} picture ({pic_left}, {pic_top}, {pic_right}, {pic_bottom}) "
                f"overlaps title box ({tb_left}, tb_top, tb_right, tb_bottom)"
            )


def test_each_7net_section_has_eight_slides(presentation):
    section_nodes = presentation._element.xpath(".//*[local-name()='sectionLst']")[0]
    sec_dict = {s.get("name"): s for s in section_nodes}
    expected_sections = ["7net_2mm_36p", "7net_4mm_36p"]
    for sec_name in expected_sections:
        assert sec_name in sec_dict, f"Expected section '{sec_name}' not found in presentation"
        sec = sec_dict[sec_name]
        sids = [c.get("id") for c in sec.xpath(".//*[local-name()='sldId']")]
        assert len(sids) == 8, f"Section '{sec_name}' expected 8 slides, got {len(sids)}"


def test_page_7_has_cortex_control_explanations(presentation):
    section_nodes = presentation._element.xpath(".//*[local-name()='sectionLst']")[0]
    sec_dict = {s.get("name"): s for s in section_nodes}
    slides_by_id = {str(s.slide_id): s for s in presentation.slides}
    for sec_name in ["7net_2mm_36p", "7net_4mm_36p"]:
        sec = sec_dict[sec_name]
        sids = [c.get("id") for c in sec.xpath(".//*[local-name()='sldId']")]
        page7_slide = slides_by_id[sids[6]]
        tbs = [s for s in page7_slide.shapes if s.has_text_frame]
        assert tbs, f"Page 7 in {sec_name} must have a text frame"
        full_text = tbs[0].text_frame.text
        assert "CtxReg" in full_text, f"Page 7 in {sec_name} missing CtxReg explanation"
        assert "Local cortex diagnostics" in full_text, f"Page 7 in {sec_name} missing local-cortex diagnostics note"
        assert "CtxDrop" not in full_text, f"Page 7 in {sec_name} still describes the retired CtxDrop branch"


