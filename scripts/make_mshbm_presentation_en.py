from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor
from pathlib import Path

OUT = Path('/home/yojiao/hpchome/hipp/present_group/MSHBM_Kong_Adaptation_Overview_EN.pptx')
IMG_KONG = Path('/home/yojiao/hpchome/hipp/present_group/kong2019_network-mshbm_group_overview.png')
IMG_LYNCH = Path('/home/yojiao/hpchome/hipp/present_group/lynch2024_network-mshbm_group_overview.png')

prs = Presentation()
prs.slide_width = Inches(13.333)
prs.slide_height = Inches(7.5)

C_BG_DARK = RGBColor(0x1E, 0x27, 0x61)
C_BG_LIGHT = RGBColor(0xF2, 0xF5, 0xFA)
C_PANEL = RGBColor(0xCA, 0xDC, 0xFC)
C_ACCENT = RGBColor(0x06, 0x5A, 0x82)
C_TEXT_DARK = RGBColor(0x21, 0x29, 0x5C)
C_TEXT_LIGHT = RGBColor(0xFF, 0xFF, 0xFF)
C_MUTED = RGBColor(0x4A, 0x59, 0x7A)


def add_bg(slide, color):
    fill = slide.background.fill
    fill.solid()
    fill.fore_color.rgb = color


def add_title(slide, text, y=0.45, color=C_TEXT_DARK):
    tb = slide.shapes.add_textbox(Inches(0.7), Inches(y), Inches(11.9), Inches(0.8))
    p = tb.text_frame.paragraphs[0]
    p.text = text
    p.font.name = 'Georgia'
    p.font.bold = True
    p.font.size = Pt(40)
    p.font.color.rgb = color


def add_subtitle(slide, text, y=1.25, color=C_MUTED):
    tb = slide.shapes.add_textbox(Inches(0.72), Inches(y), Inches(11.6), Inches(0.6))
    p = tb.text_frame.paragraphs[0]
    p.text = text
    p.font.name = 'Calibri'
    p.font.size = Pt(18)
    p.font.color.rgb = color


def add_card(slide, x, y, w, h, title, body, color=C_PANEL, title_color=C_TEXT_DARK, body_color=C_TEXT_DARK):
    sh = slide.shapes.add_shape(1, Inches(x), Inches(y), Inches(w), Inches(h))
    sh.fill.solid()
    sh.fill.fore_color.rgb = color
    sh.line.color.rgb = RGBColor(0xBD, 0xCD, 0xE8)

    t1 = slide.shapes.add_textbox(Inches(x+0.25), Inches(y+0.18), Inches(w-0.5), Inches(0.45))
    p1 = t1.text_frame.paragraphs[0]
    p1.text = title
    p1.font.name = 'Georgia'
    p1.font.bold = True
    p1.font.size = Pt(20)
    p1.font.color.rgb = title_color

    t2 = slide.shapes.add_textbox(Inches(x+0.25), Inches(y+0.68), Inches(w-0.5), Inches(h-0.85))
    tf2 = t2.text_frame
    tf2.word_wrap = True
    lines = body.split('\n')
    for i, ln in enumerate(lines):
        p = tf2.paragraphs[0] if i == 0 else tf2.add_paragraph()
        p.text = ln
        p.font.name = 'Calibri'
        p.font.size = Pt(14)
        p.font.color.rgb = body_color


def add_bullets(slide, x, y, w, h, bullets, color=C_TEXT_DARK, size=18):
    tb = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    tf = tb.text_frame
    tf.clear()
    tf.word_wrap = True
    for i, b in enumerate(bullets):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.text = b
        p.level = 0
        p.font.name = 'Calibri'
        p.font.size = Pt(size)
        p.font.color.rgb = color
        p.space_after = Pt(8)


# 1
slide = prs.slides.add_slide(prs.slide_layouts[6])
add_bg(slide, C_BG_DARK)
add_title(slide, 'MSHBM for Hippocampal Network Parcellation', y=0.9, color=C_TEXT_LIGHT)
add_subtitle(slide, 'Kong-style adaptation: practical workflow, interpretable outputs, and reporting logic', y=1.9, color=RGBColor(0xCA,0xDC,0xFC))
add_card(slide, 0.8, 3.1, 3.9, 2.6, 'Goal', 'Build robust group priors\nfor hippocampal vertices\nthen infer individualized labels.', color=RGBColor(0x2A,0x37,0x7C), title_color=C_TEXT_LIGHT, body_color=C_TEXT_LIGHT)
add_card(slide, 4.95, 3.1, 3.9, 2.6, 'Constraint', 'Keep canonical atlas alignment\n(Kong2019: 7 networks,\nLynch2024: 8 networks).', color=RGBColor(0x2A,0x37,0x7C), title_color=C_TEXT_LIGHT, body_color=C_TEXT_LIGHT)
add_card(slide, 9.1, 3.1, 3.4, 2.6, 'Output', 'Group prior map theta\n+ per-subject posterior\nand hard labels.', color=RGBColor(0x2A,0x37,0x7C), title_color=C_TEXT_LIGHT, body_color=C_TEXT_LIGHT)

# 2
slide = prs.slides.add_slide(prs.slide_layouts[6])
add_bg(slide, C_BG_LIGHT)
add_title(slide, 'Why Not a Naive Pipeline?')
add_subtitle(slide, 'MSHBM addresses stability and interpretability at group and individual levels')
add_card(slide, 0.7, 2.0, 4.0, 4.9, 'Common Failure Modes', '• Fragmented labels\n• Unstable network mapping\n• Weak transfer to individuals\n• Hard to explain metrics')
add_card(slide, 4.95, 2.0, 3.9, 4.9, 'What MSHBM Adds', '• Hierarchical sharing across runs/subjects\n• Explicit group prior theta\n• Tunable prior-vs-smoothness balance\n• Better reporting structure')
add_card(slide, 9.1, 2.0, 3.5, 4.9, 'What We Present', '• Initialization logic\n• Group EM training\n• Meaning of theta / w / c\n• How to read overview panels')

# 3
slide = prs.slides.add_slide(prs.slide_layouts[6])
add_bg(slide, RGBColor(0xEC,0xF2,0xF7))
add_title(slide, 'Pipeline in 4 Stages')
steps = [
    ('Step 1', 'Build shared feature basis\nand initialize group clusters'),
    ('Step 2', 'Run group-level EM\nto estimate mu and theta'),
    ('Tune', 'Search (w, c) on\nvalidation subjects'),
    ('Step 3', 'Infer individual posterior\nand hard labels')
]
x0 = 0.8
for i, (t, b) in enumerate(steps):
    x = x0 + i*3.15
    add_card(slide, x, 2.2, 2.85, 2.4, t, b, color=RGBColor(0xD8,0xE6,0xF9))
    if i < 3:
        arr = slide.shapes.add_shape(13, Inches(x+2.9), Inches(3.15), Inches(0.2), Inches(0.35))
        arr.fill.solid(); arr.fill.fore_color.rgb = C_ACCENT
        arr.line.color.rgb = C_ACCENT
add_bullets(slide, 0.9, 5.2, 12.0, 1.8, [
    'Input feature per vertex: hippocampal time series projected to shared cortical reference basis',
    'Core output for downstream use: theta (group prior over networks at each vertex)'
], size=16)

# 4
slide = prs.slides.add_slide(prs.slide_layouts[6])
add_bg(slide, C_BG_LIGHT)
add_title(slide, 'How Group Priors Are Learned')
add_card(slide, 0.7, 1.8, 6.15, 5.3, 'Step 1: Initialization', '1) Build ROI intersection across subjects\n2) Compute run-wise vertex profiles\n3) Group-average profiles\n4) Spherical k-means init\n5) Match clusters to canonical networks')
add_card(slide, 6.95, 1.8, 5.65, 5.3, 'Step 2: Group EM', 'E-step:\nUpdate lambda using data-fit + log(theta).\n\nM-step:\nUpdate session/subject/group prototypes\nand concentration parameters.\n\nRecompute theta from lambda and iterate\nuntil relative cost change converges.')

# 5
slide = prs.slides.add_slide(prs.slide_layouts[6])
add_bg(slide, RGBColor(0xF5,0xF7,0xFB))
add_title(slide, 'What theta, w, and c Mean in Practice')
add_card(slide, 0.75, 2.0, 3.9, 4.8, 'theta', 'Group prior probability map\n(vertex x network).\n\nThis is the main group product\nused for individual inference.', color=RGBColor(0xD2,0xE4,0xFA))
add_card(slide, 4.85, 2.0, 3.9, 4.8, 'w', 'Weight of group prior term\nin individual posterior scoring.\n\nHigher w => stronger pull\ntoward group prior.', color=RGBColor(0xD2,0xE4,0xFA))
add_card(slide, 8.95, 2.0, 3.9, 4.8, 'c', 'Weight of spatial regularization\n(MRF smoothness).\n\nHigher c => less fragmented\nlabels on surface.', color=RGBColor(0xD2,0xE4,0xFA))

# 6
slide = prs.slides.add_slide(prs.slide_layouts[6])
add_bg(slide, C_BG_LIGHT)
add_title(slide, 'How to Read the network-mshbm Overview Figure')
add_bullets(slide, 0.8, 1.6, 6.3, 5.6, [
    'Top-left: group prototypes (network-to-network similarity)',
    'Top-right: theta (network x vertex), ordered by long axis',
    'Bottom-left: inter vs intra similarity by network',
    'Bottom-right: WTA occupancy (vertex counts per network)',
    'Right panel: surface rendering of theta WTA labels'
], size=17)
if IMG_KONG.exists():
    slide.shapes.add_picture(str(IMG_KONG), Inches(7.0), Inches(1.45), width=Inches(5.9), height=Inches(5.45))

# 7
slide = prs.slides.add_slide(prs.slide_layouts[6])
add_bg(slide, RGBColor(0xEC,0xF3,0xF9))
add_title(slide, 'Current Output Snapshot')
add_subtitle(slide, 'Same workflow, atlas-specific canonical network constraints')
if IMG_KONG.exists():
    slide.shapes.add_picture(str(IMG_KONG), Inches(0.7), Inches(1.55), width=Inches(6.0), height=Inches(3.95))
if IMG_LYNCH.exists():
    slide.shapes.add_picture(str(IMG_LYNCH), Inches(6.9), Inches(1.55), width=Inches(5.75), height=Inches(3.95))
add_card(slide, 0.7, 5.7, 12.0, 1.3, 'Key Message', 'Kong2019 is constrained to 7 canonical networks; Lynch2024 is constrained to 8.\nInterpret theta first, then judge occupancy and inter/intra patterns.')

# 8
slide = prs.slides.add_slide(prs.slide_layouts[6])
add_bg(slide, C_BG_DARK)
add_title(slide, 'Takeaways for Presentation and Decision-Making', y=0.65, color=C_TEXT_LIGHT)
add_bullets(slide, 0.9, 1.8, 11.7, 4.6, [
    'Use one narrative: init -> group EM -> tune (w,c) -> individual inference.',
    'Define theta as the final group prior; avoid overloading with low-level implementation details.',
    'Read inter/intra jointly: cross-subject consistency vs within-subject stability.',
    'For reporting, prioritize overview panels over equations unless asked.'
], color=RGBColor(0xE8,0xF0,0xFF), size=20)
add_subtitle(slide, 'Appendix can carry equations and code-level details if technical questions come up.', y=6.5, color=RGBColor(0xCA,0xDC,0xFC))

OUT.parent.mkdir(parents=True, exist_ok=True)
prs.save(str(OUT))
print(str(OUT))
