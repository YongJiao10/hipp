# fMRI QC

## Accepted policy and implementation status

The user approved this policy on 20 October 2026 for HCP7T rest, HCP3T
EMOTION/WM, Stanford 36P, BANDA and PDC. This is the current QC decision.
The HCP7T roster entry point, `02_02_build_qc_manifest.py`, now applies all
three criteria in order. It reads exact physical-run `Movement_Regressors.txt`
paths from `--motion-manifest` and each subject's declared
`surface/raw/sub-<subject>_coverage_summary.json`. It validates coverage run IDs,
labels and lengths against the motion frames, and checks recorded actual den-512
vertex counts. Missing inputs receive `unassessable` status; malformed or
mismatched declarations stop the operation before reports are written.

The passing roster is derived from current inputs, without a fixed subject
count or historical ID requirement. Established report filenames containing
`tsnr` remain an output interface, but no tSNR numerical gate is applied.
This roster implementation does not establish that all production sampling,
mask consumers, or other dataset entry points have been updated or rerun.
Group-input `qc` and `all` stages for other datasets or tasks stop before writing
until their production QC input contracts are implemented.
The research counts below are not new formal rosters; the historical HCP151
reference is not the new QC result.

## Participant and visit selection

Apply these criteria in order, stopping at the first failure:

1. **Motion:** overall mean Power FD <=0.30 mm. Perform this step before
   HippUnfold when processing new inputs.
2. **Anatomy:** HippUnfold Dice >=0.70 in both hemispheres.
3. **Hippocampal coverage:** at most 20 vertices without functional-GM support
   in each consumed den-512 hippocampal hemisphere (20/512 = 3.90625%).

Use one bilateral participant/visit roster per analysis branch. A failure on
either side excludes that participant/visit from the branch. Missing required
inputs remain unassessable and are not recorded as quality failures.

tSNR can be reported where its input baseline supports interpretation; it is
not a numerical exclusion criterion. Mean relative RMS <=0.5 mm (M05) adds
no exclusions among the currently assessed three-criterion passes and is not
an additional gate. The coverage threshold counts vertices, not physical area.

## Multiple runs and analysis units

Calculate Power FD within each physical run from adjacent motion samples:
sum absolute translations in mm and absolute rotations converted to arc
length at a 50-mm radius. HCP motion rotations are in degrees; Stanford uses
fMRIPrep Power FD. Set the first FD sample of each run to zero, and never
differentiate across run boundaries. Use the original motion frames before
censoring. The current comparison uses unfiltered, lag-one FD.

For multiple runs, use
`overall_mean_FD = sum(frames_r * mean_FD_r) / sum(frames_r)`.
Apply the 0.30-mm threshold once to this overall mean. There is no individual
run veto; for example, equal-length runs with means 0.40 and 0.10 pass at 0.25.
Dice is evaluated once per anatomical reconstruction, independently of run count.

Coverage uses actual trilinear weights at the registered hippocampal vertices.
The source GM set is the supplied CIFTI hippocampal BrainModel restricted to
voxels with finite, nonconstant signals. For multiple declared runs, first
intersect these valid source-voxel masks, then calculate their surface support.
A vertex is unsupported when its total GM weight is zero. Count unsupported
vertices once per hemisphere; do not add counts from different runs.

In the evaluated data, the common-source-mask operation removed no hippocampal
source voxels. All 166 HCP7T concatenated inputs were additionally checked in
four 900-frame blocks, with no nonfinite or constant hippocampal source series
in any block. Run count therefore added no coverage attrition in these records.

EMOTION and WM are evaluated separately; use their intersection if an analysis
requires both tasks. PDC is evaluated per visit, without pooling different
visits to pass motion QC. Cohort identity, denoising variants and declared run
selection remain explicit.

## Vertex handling and coordinates

Passing participant selection permits up to 20 unsupported vertices per side;
it does not make those vertices usable. Downstream smoothing, FC and activation
need an explicitly applied vertex mask combining positive GM support with
finite, nonconstant consumed signal. This mask integration remains production
work; the pass-count table does not demonstrate that it has been deployed.

Volume-to-surface sampling must use surfaces transformed into the exact BOLD
volume space. A `space-corobl` mesh cannot be used directly to sample MNI or
native T1w volumes. Use native T1w geometry for physical distances and smoothing;
displaying corresponding vertex values on a corobl mesh is a separate operation.
Coordinate sampling code has been repaired, but affected formal functional
products have not been rebuilt by this QC analysis.

## Evaluated retention

Rates below are cumulative percentages of the same complete-case N in each
row, using existing registered coverage and motion records. Missing cases are
not assigned an assumed final result.

| Dataset / unit | Complete-case N | FD pass | FD + Dice pass | All three pass | New exclusions: FD / Dice / coverage |
|---|---:|---:|---:|---:|---:|
| HCP7T rest | 166 | 156 (94.0%) | 155 (93.4%) | 148 (89.2%) | 10 / 1 / 7 |
| HCP3T EMOTION | 144 | 142 (98.6%) | 142 (98.6%) | 134 (93.1%) | 2 / 0 / 8 |
| HCP3T WM | 144 | 142 (98.6%) | 142 (98.6%) | 134 (93.1%) | 2 / 0 / 8 |
| Stanford 36P | 199 | 194 (97.5%) | 190 (95.5%) | 186 (93.5%) | 5 / 4 / 4 |
| BANDA | 202 | 183 (90.6%) | 183 (90.6%) | 179 (88.6%) | 19 / 0 / 4 |
| PDC visits with complete records | 207 | 197 (95.2%) | 197 (95.2%) | 196 (94.7%) | 10 / 0 / 1 |

BANDA has 203 roster members; one lacks a declared independent functional run.
PDC has 455 visits, of which 422 (92.7%) pass motion. The table uses the 207
visits with completed coverage assessments, not a final 455-visit selection.
Newly completed anatomical outputs do not acquire a coverage decision
automatically. The two-task HCP intersection is 133/144 (92.4%).

## Method basis

[Gao et al., Nature Communications 2026](https://www.nature.com/articles/s41467-026-71719-y)
use mean FD >0.30 mm for exclusion in cross-cohort functional-gradient work.
This supports the numerical choice; it does not establish an exact match of
FD implementation or aggregation. Different TRs and filtering affect FD, and
mean FD can conceal brief large movements.

[Kember et al., Nature Communications 2026](https://www.nature.com/articles/s41467-026-74572-1)
provide the anatomical and functional-GM coverage precedent. Their published
coverage criterion is >=20 mm² improperly interpolated surface, not 20 vertices.
The project's vertex-count rule and zero-weight classifier are explicit
adaptations for den-512 functional nodes. Positive support is not proof of
adequate SNR or individual tissue purity.

Kember defines low-motion frames by FD <=0.25 mm and requires at least 10
low-motion minutes. This duration rule was not adopted: the project's full
EMOTION and WM acquisitions last only 4.224 and 9.72 minutes. The accessible QC
description does not establish a specific temporal-censoring implementation.

Exploration reports, one-off analysis code/results and QC discussion sessions
are retained under `_archive/hippocampal_qc_exploration/` with their original
relative directory structure. They are historical evidence, not active
workflow instructions.
