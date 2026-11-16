# Compact editable workflow revision

Previous approved PNG/PDF files remain under `final/`. New editable outputs are
under `compact/`. No previous output is deleted.

`revise_assets.py` uses actual sub-100610 T1w MRI, native HippUnfold 512-vertex
surface, canonical display mesh, Kong labels and cached 2 mm FC. It selects a
visible body vertex within 35–65% of the native posterior-to-anterior extent
and 40–62% of the rendered image height, maximizing its distance from the
rendered silhouette (at least 10 pixels). It colors exactly that mesh vertex
gold (#BB861C, matching its connector and FC row box) using point scalars, with no overlaid
circle or sphere; adjacent triangle colors interpolate from that vertex. For the
cortical circle, visible vertices with negative FC and blue rendered pixels near
the explicitly requested normalized screen position (0.64, 0.36) are considered;
the closest vertex determines its real parcel ID and corresponding matrix column.
There is no fabricated or arbitrarily shifted ROI-to-column correspondence.
Callout positions use the actual render camera; both connectors are straight.
The cortical image sits above the body-vertex connector to avoid crossing it.
The sagittal MRI plane maximizes AP intersection span over x=-31 to -20 mm at
0.5 mm intervals. The gold contour is the actual mesh/plane intersection.

The compact three-row 17-network legend matches the consumed GIFTI label table
and leaves more space for the two individual cortical examples. Panel 7 uses actual
sub-1016 FC and fixed HCP templates; all 1536 correlations and 512 argmax labels
are checked. Correlation maps share [-1,1]. Brain images are movable raster
objects; text, callouts and selection boxes are editable PowerPoint objects.

Runtime: py314 with NumPy, nibabel, SciPy, Matplotlib, BrainSpace/VTK, Pillow;
Xvfb for surfaces; Node with PptxGenJS 4.0.1. Limit numeric/VTK threads to one.

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 VTK_SMP_MAX_THREADS=1 xvfb-run -a /home/yojiao/.conda/envs/py314/bin/python scripts/visualize/flowchart/revise_assets.py
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 /home/yojiao/.conda/envs/py314/bin/python scripts/visualize/flowchart/compose_compact.py
node scripts/visualize/flowchart/build_compact.cjs
```

The 14 × 14.7 inch compact layout uses Arial for every native PPTX text run.
The local fontconfig Arial entry is a broken WOFF, so the separately authored
PNG/PDF previews explicitly use metric-compatible Liberation Sans; they are
not claimed to be Arial exports from PowerPoint. The PPTX contains one compact
combined figure; separate panel PNG previews are also supplied. Surface images remain raster;
all visible labels, legends, arrows, circles and matrix selection boxes
are independently editable. Matrix selections use exact original row/column
positions (no row sorting). Zero-based vertex IDs become one-based matrix rows;
ROI IDs are one-based column indices. Actual IDs are in `compact/asset_details.json`
and generated slide notes, not hard-coded in the layout.
FC matrix display limits are ±0.2; cortical seed FC uses standardized Fisher-z
profiles displayed at ±2.5 SD, while matching uses original numeric arrays.

Panel 5 uses orthographic rather than perspective projection; internal white
margins are cropped and axis-label coordinates transformed to the crop so the
scatter fills more of the panel without changing the data. The two rendered
Gradient 3 axis endpoints are checked to have identical screen x-coordinates.
Gradient 1/2/3 labels are native editable text beside their respective axes, and
the output caption is Three subregions. Panel 7 explicitly labels both shapes:
512 vertices × 400 ROIs and 3 templates × 400 ROIs. Undefined variable formulas
are replaced by comparison of each vertex profile with each template, followed
by selection of the highest-correlation template for each vertex.
Anterior/Middle/Posterior captions use the same colors in panels 6 and 7.
See `kember_matching_check.md` for the original-paper/author-code/local-output
comparison: Pearson + argmax agree, while feature space, Fisher-z use and
upstream clustering differ. This is not a claim of numerical replication.
The requested comparison scope starts after templates are available: the core
Pearson-profile-correlation and maximum-correlation assignment rule agrees.
Fisher-z input scaling and the author's small-cluster filtering/spatial dilation
remain distinctions; no equivalence of those extra steps is asserted.

Actual-slide QA uses isolated Aspose.Slides 26.8.1 in `.render_deps/` and Ubuntu
libgdiplus 6.0.4+dfsg-2 extracted into `.render_runtime/`. These do not change the
core-workflow environment. The QA render uses in-memory Liberation Sans font
substitution and contains the renderer's evaluation watermark; it is confined
to `compact/qa/` and is not a deliverable. The delivered PPTX is never rewritten
by the QA renderer. PNG/PDF previews have no evaluation watermark.
