# Hippocampal Surface Rendering

The maintained hippocampal label panels use BrainSpace and VTK via
`scripts/core_workflow/lib/surface_utils.py::render_hippocampal_grid`.
The renderer reads the exact left and right folded surface GIFTIs and label
GIFTIs selected by the calling workflow. It renders the two complementary
views in a mirrored L/R layout, with white background and the source GIFTI label
colors. The native surface image is rendered at 3200 × 2048 pixels before
composition. The right-hand legends show per-hemisphere vertex percentages
and the project's network abbreviations; when both sides have at most three
labels, the label text and color swatches are enlarged.

For HCP group overviews, `scripts/core_workflow/06_01_build_group_overview.py`
consumes the manifest-declared group surfaces and exact assignment variants.
The group overview does not take a hippocampal Workbench scene argument.
Subject overviews use each subject's declared surface and label files. Surface
panels and legends are rendered from the same vertex labels; the rendering
does not change the underlying data or network assignment.

The existing `*_wb_*_biglegend.png` intermediate filenames remain part of
callers' current selection contracts; they contain BrainSpace renders, not
Workbench scenes. Existing hippocampal panel caches include the changed
renderer inputs or are regenerated so prior Workbench panels are not reused.
HCP cross-fit individual overviews created by the old renderer require
`--resume --refresh-visuals` for an explicit redraw. Output manifests identify
the renderer in each regenerated individual record. Some existing individual
and mixed cortex/hippocampus workflows retain their scene arguments and
scene-related input records for their existing workflow contracts; those scene
files do not control the BrainSpace hippocampal camera.

`06_04_render_hippomaps_gradient_products.py` continues to render continuous
folded/unfolded gradients through the HippoMaps plotting API. Connectome
Workbench remains in use for cortex scene rendering, volume-to-surface mapping,
registration and other workflow operations; changing the hippocampal label
renderer does not replace those uses.

On a headless machine, run the figure command under `xvfb-run -a` if a display
is not available. The rendering runtime uses the repository's py314 environment
with its installed BrainSpace and VTK packages. `LP_NUM_THREADS=1` and
`VTK_SMP_MAX_THREADS=1` limit the rendering backend's worker threads.
