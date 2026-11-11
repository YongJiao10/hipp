# Official HippUnfold v2.3.0 baseline and reconstruction scope

This assessment records the reconstruction using the official
[v2.3.0 release](https://github.com/khanlab/hippunfold/releases/tag/v2.3.0),
with the documented resource/inference runtime adapters only. Retired local
boundary, reference-plane and Workbench corrections are not part of this
method, and their candidate results do not establish the new release's
scientific outcomes. The current runtime is specified in the
[runtime manifest](hippo2_reproducibility_manifest.md).

## Upstream fixes

The maintainer confirmed two reported defects and included their fixes in
v2.3.0:

- [Issue580](https://github.com/khanlab/hippunfold/issues/580) and
  [PR583](https://github.com/khanlab/hippunfold/pull/583): the old contiguity
  operation could increase terminal-label fragmentation. The replacement fits
  four contiguous alternating AP/PD arcs on the outer boundary loop and
  propagates labels across its connected boundary band, with new source/sink QC.
- [Issue581](https://github.com/khanlab/hippunfold/issues/581) and
  [PR582](https://github.com/khanlab/hippunfold/pull/582): the old singleton
  reference plane missed the native midthickness vertices, so nonzero
  registration fields could have no applied effect. The release creates the
  reference at the configured midthickness height and projects all three
  surface layers onto that plane for displacement sampling.

The release also includes geometric-centre initialization and surface topology
QC. See the [complete tagged comparison](https://github.com/khanlab/hippunfold/compare/v2.2.0...v2.3.0).
The v2.3.0 reconstruction used the release rather than development main: the maintainer states that the
newer surface filtering requires a newly generated unfolded atlas. See the
[release guidance](https://github.com/khanlab/hippunfold/issues/580#issuecomment-6398866471).

## Required HCP166 reconstruction

The authoritative roster contains 166 subjects in
`manifests/hcp_7t_hippocampus_struct_complete_166.txt`. The formal output root is
`outputs_hcp/inputs/hippunfold/dense_corobl_batch/`, with complete participant
runs beneath `sub-<id>/hippunfold/`.

The upstream source comparison shows that preprocessing, CNN segmentation,
shape injection, volumetric coordinates, native physical surface generation
and native curvature/thickness algorithms did not change. When their exact
inputs remain available, they can be reused. However, the prior 01_02 cleanup
removed the tissue volumes, nnUNet outputs, GM masks, preprocessed T1w images
and transforms required for complete formal products. Retained native surfaces
and signed-distance metrics suffice only for a surface-chain continuation.
They cannot supply updated volumetric subfields, crop references and all
other required outputs.

Therefore the official complete participant DAG must regenerate missing
upstream prerequisites and every affected downstream product. It must remain
self-consistent; do not preserve old surface timestamps to bypass newly
computed dependencies.

## Completion and result interpretation

The complete official participant DAG has now been attempted for all 166
subjects, using 12 concurrent subjects and two GPU inference slots. Exactly
163 subjects initially satisfied the 96-product contract, matching v2.3.0 run
metadata and official error-level QC. All three failures remain recorded:

| Subject | Official stopping point | Observed failure |
| --- | --- | --- |
| 233326 | Surface `qc_verify` | Left dentate has two connected components |
| 249947 | Surface `qc_verify` | Right dentate has two connected components |
| 942658 | Right-dentate boundary assignment | `get_boundary_loop` raises `ValueError: max() iterable argument is empty` |

The coordinator continued the independent subjects and exited nonzero after
processing the full roster. The 156 completed participants present in the HCP
3T task dataset subsequently underwent approved 01_02 cleanup of anat, coords,
warps and CIFTI; their retained surfaces, metrics and QC are available, but
they no longer satisfy the full 96-product contract. The other seven completed
participants retain the full contract. A completed participant is not warning-free:
warning-level surface/source-sink findings remain in the official QC reports.

The established Dice screen requires both hemispheres to have
`desc-unetf3d_dice.tsv >= 0.70`. Reading the actual new outputs with the
existing `02_02_build_qc_manifest.py:read_dice` function gives:

| Population | Bilateral Dice pass | Rate |
| --- | ---: | ---: |
| All 166 attempted participants | 165/166 | 99.4% |
| The 163 complete official workflows | 162/163 | 99.4% |

The only Dice failure is 182739 (left 0.757747, right 0.694728). All three
workflow failures pass bilateral Dice. Exactly 162 participants therefore
satisfy both workflow completion and the Dice screen. This count does not
include the separate motion and functional-coverage criteria.

With the accepted mean Power FD <=0.50 mm threshold, all 162 structural/Dice
passes also pass the newly measured NIfTI support and runwise signal coverage.
Canonical BOLD resampling from the new v2.3.0 T1w surfaces, 2/4-mm smoothing
and both group-average surfaces are complete for these 162 participants.
The largest unusable-vertex counts are four on the left and eight on the
right, below the 20-vertex limit in this v2.3.0 assessment.
Execution logs are under `outputs_hcp/runs/hcp7t_v230_fd050_20261026/`.

Retained independent cortical references are reused. The missing standard
Kong2022 PFM labels for 901442 have been supplied and its reference has been
built. PFM FC, cortex-regressed participant time series and their FC are
complete for all 162 participants. Each FC route contains 648
participant/smoothing/hemisphere combinations (2/4 mm, left/right).

Group gradients are complete for both `pfm` and `pfm_ctxreg`, with all eight
route/smoothing/hemisphere manifests reporting 162 participants. All 24
group-atlas combinations are complete: eight each for `pfm` and `pfm_ctxreg`,
four for `pfm_17net`, and four for `pfm_weighted`. Every manifest matches the
162-participant roster; every gradient-spectral result has final K=3.
The canonical7 PFM routes include gradient-spectral and WTA, while family17
and weighted Fisher-z use gradient-spectral.

These products use the existing shifted-cosine gradient-spectral workflow.
The code also supports Ward/GMM/RBF methods, explicit preprocessing choices,
and OVR annotations. Exploratory runs used v2.3.0 inputs; execution of
`GRADIENT_PARCELLATION_REVISION_PLAN.md` on upgraded HippUnfold inputs remains pending.

The overview renderer preserves negative signs in signed Fisher-z cell
annotations, wraps long surface titles, and uses wider family17 heatmaps with
vertical network labels. These display changes do not alter stored FC or model
results.

Family-aware five-fold cross-validation is complete for all 162 participants:
20 training templates and 648 held-out participant mappings were generated.
Its robustness overview, disjoint-fold quantitative summary and Workbench
comparison, and Day 1/Day 2 comparison are complete under
`outputs_hcp/present/overviews/crossfit_template_matching/`. All four standard
robustness gates pass. Mean pairwise macro-Dice across completely disjoint
cohorts is 0.9662--0.9847; Day 1 versus Day 2 macro-Dice is 0.9505--0.9924.

Individual overviews await a display-label decision. Their current renderer
requires three distinct, stable canonical7 network winners across all 20
templates, which the measured profiles do not satisfy:

| AP-ordered source label | Highest mean Fisher-z network across templates |
| --- | --- |
| Anterior | Default: 17; Somatomotor: 3 |
| Middle | Default: 20 |
| Posterior | Default: 11; Somatomotor: 6; Control: 3 |

The stable spatial parcellation therefore has no supported one-to-one
VAN/SMN/DMN display mapping under this annotation rule. Using the source names
Anterior/Middle/Posterior is the proposed individual-legend choice.
Project code and documentation now use Anterior/Middle/Posterior consistently.
Existing generated labels and metadata still require an explicitly approved
in-place presentation/metadata refresh before their older middle-label name
matches the current terminology contract.

### Origin of the topology failures

Direct comparison with the unchanged native meshes staged before cleanup
distinguishes pre-existing defects from changes in the new reconstruction:

- 233326 left dentate has a main component with 11,180 vertices and 21,289
  faces, plus a small component with five vertices and six faces. Its old and
  new native meshes have exactly identical point and face arrays. The new
  `single_component` error-level QC now detects the pre-existing fragment.
- 249947 right dentate has a main component with 11,932 vertices and 22,801
  faces, plus one isolated triangle with three vertices. The staged old mesh
  had only one component, so this fragment appeared in the new reconstruction.
  In-memory replay of the official mesh-generation script on the retained
  current inputs reproduces the saved point and face arrays exactly. After
  largest-component extraction the mesh has one component and no non-manifold
  edges; hole filling introduces one non-manifold edge, and decimation produces
  the isolated triangle and two components.
- 942658 right dentate is connected but has one non-manifold edge. The staged
  old mesh also had a non-manifold edge and no cycle in its largest boundary
  component. Current mesh-generation replay reproduces the saved point and
  face arrays exactly: largest-component extraction has no non-manifold
  edges; hole filling introduces two, and decimation leaves one. The new
  boundary algorithm needs an ordered closed loop, unlike the old
  label-contiguity implementation; its `max()` over an empty cycle list raises
  the observed exception.

These findings identify reproducible mesh-processing topology defects and
new detection/algorithm requirements. They do not justify labelling the
original MRI scans as poor quality. Workflow error-level topology checks and
the older whole-segmentation Dice screen measure different properties.

A subject requires all 96 formal products: the existing 84-file structural
contract, four new source/sink/surface HTML and verification files, and eight
hemisphere/structure QC metric JSONs. Run with `--notemp` and check with
`--fail-on-missing --no-cleanup` for the full contract. The approved 01_02
cleanup for the 156 HCP 3T task participants removed anat, coords, warps and
CIFTI; their cleanup receipts certify the retained-output contract only.
Retain runtime scratch and failed outputs for diagnosis.
Official error-level QC failures or external-tool crashes remain failures;
do not silently select another scientific method or patched Workbench binary.

The fresh analysis uses actual newly generated native registered surfaces,
sampled T1w coordinates and each subject's new official RAS transform. The
complete cohort contains 652 hemisphere/structure middle surfaces; all differ
from their pre-registration counterparts. The largest corobl-to-T1w frame
control discrepancy is 0.00000811 mm.

For the 163 complete participants, the exact inverse-correspondence analysis
includes both ordinary triangles and continuous inverse sets in collapsed
triangles:

| Structure | Atlas targets evaluated | Subjects with inverse diameter >1 mm | Targets >1 mm | Maximum diameter in T1w |
| --- | ---: | ---: | ---: | ---: |
| Hippocampus | 166,912 | 40 | 49 | 1.695 mm |
| Dentate | 41,728 | 1 | 1 | 1.375 mm |

The structure sets overlap: their union is 40 participants and 50 targets.
Of these targets, 49 are on the atlas rectangle boundary and one is interior
(135124, left hippocampus, zero-based atlas vertex 350, diameter 1.013 mm).
The largest hippocampal case is 958976, right, vertex 11. The dentate case is
162935, left, vertex 127. These diameters describe separation among possible
inverse positions, not measured anatomical displacement or a new official
exclusion threshold. They do not by themselves show that the previously
reported source defect remains unfixed. The official QC source explicitly
notes that terminal-band constraints can collapse faces to zero unfolded area.

Independent raw-mesh review separates those 50 targets further: 47 exceed
1 mm because the full inverse includes collapsed boundary-band triangles;
only three have an ordinary, non-degenerate multi-sheet inverse spanning
more than 1 mm. Those three are 135124 left hippocampus vertex 350
(1.013 mm), 203418 left hippocampus vertex 159 (1.161 mm), and 162935 left
dentate vertex 127 (1.375 mm). They are geometry diagnostics, separate from
the three workflow failures listed above. At all 50 targets, the actual
Workbench output lies on a valid non-degenerate preimage to within
0.00000503 mm. The 40 participants therefore must not be treated as 40 failed
workflows or automatically excluded.

There are also 377 hippocampal and 789 dentate atlas targets with no exact
triangle preimage under the registered map. All are boundary targets; there
are no uncovered interior targets. Their official output coordinates exist,
but an exact-preimage diameter is undefined, so they are outside the >1 mm
count. The independent review confirms that all 1,166 targets were already
uncovered before unfolded registration. Disjoint terminal arcs produce
diagonal corner bevels in the native Laplace mesh; the gaps are not caused
by registration shrinkage. Observed Workbench outputs match nearest-triangle
or corner-vertex sampling at those targets. The maximum distance to the flat
mesh is 0.313 and 0.034 respectively in unfolded-coordinate units; these are
not T1w anatomical distances.

The v2.3.0 assessment distinguished complete participants from geometry
available in incomplete runs. An independent check covered all 208,640
complete-subject targets. Geometry or reproducibility alone does not establish
anatomical accuracy; downstream FC, gradients and atlas interpretation must
use the newly computed official products.

See the [runtime manifest](hippo2_reproducibility_manifest.md) for the runtime
gate, resource adapters and batch launch. The maintained stage-01 launcher
produces the canonical formal results.
