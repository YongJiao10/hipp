# Hippocampal surface-vertex QC: scope and current decision

The primary target is functional signal at each hippocampal surface vertex
across HCP7T rest, HCP3T task, Stanford, BANDA and PDC. The user has excluded
the per-run R20 rule from further consideration. The retained supplementary
motion candidate is participant mean relative RMS<=0.5 mm; it is a whole-scan
rigid-body motion measure, not a hippocampal or vertex-specific signal measure.

Final gate selection is deferred until the confirmed corobl-to-MNI sampling
defect is repaired. The [five-person same-volume pilot](coordinate_pilot.md)
already establishes material changes in full-length HCP7T time series, FC and
gradients; further threshold research is not required before sampling repair.
The registered QC experiment is not a repair of existing FC or atlas products.

The 0.5-mm precedent is [Savage et al., Communications Biology 2024](https://www.nature.com/articles/s42003-024-06573-z),
*Dissecting task-based fMRI activity using normative modelling: an application
to the Emotional Face Matching Task*. Its Quality control section excludes
participants with mean relative RMS>0.5 mm. This is a published motion-screening
precedent, not validation of local hippocampal signal or a prevalence estimate.
Averaging included run means is the stated project aggregation convention.

## What each existing measurement assesses

| Measurement | Spatial target | Meaning for the requested QC |
|---|---|---|
| Mean RMS, R20, Power-FD burden | Whole-scan rigid-body motion | Supplementary run/participant screen; no local vertex mask |
| HippUnfold Dice | Hippocampal anatomical reconstruction | Anatomical prerequisite, not local BOLD signal quality |
| Surface V | Finite, nonconstant sampled vertex time series | Basic validity; a nonzero noisy or partially contaminated vertex can pass |
| Existing HCP/Stanford T25 selector | Previously sampled den-512 vertices | Reference counts; direct corobl-to-MNI sampling limits their anatomical interpretation |
| Registered surface T25 and Coverage20 | Individual midthickness vertices placed in the functional volume space | Main five-dataset surface comparison; definitions in the surface report |
| Earlier BANDA/PDC native-CIFTI T25 | Hippocampal volume voxels | Preliminary local volume-ROI comparison, distinct from the new surface test |

HCP7T's existing T25 is the explicitly labelled 10000/SD proxy. HCP3T uses
runwise task-model residual tSNR. Stanford uses matched-baseline/rest measurements.
These definitions remain distinct. The fact that tSNR is vertexwise does not
by itself validate 25 across different processing stages.

## Practical surface-specific comparison

Prioritize two complementary checks: functional surface coverage (Kember2026)
and local vertexwise signal/tSNR (the existing task/rest measurements, with
Barch2013 as a task-QA precedent). Retain the existing T25 selector as a
reference and measure at registered anatomical vertices for the main
[surface comparison](hipp_surface_qc.md). Retain midthickness sampling, actual consumed
surface density, anatomy and finite/nonconstant checks. Mean relative RMS is
supplementary motion QC. The [six-paper assessment](hipp_surface_qc_options.md)
distinguishes these choices from histological filters, microstructural-profile
alignment, optional gradient sensitivity analysis and unsupported exclusion rules.

[Kember et al., Nature Communications 2026](https://www.nature.com/articles/s41467-026-74572-1)
explicitly excludes participants with >=20 mm² of improperly interpolated
hippocampal surface, caused by insufficient overlap between reconstructed
surfaces and gray-matter voxels in preprocessed fMRI. This provides a concrete
surface-specific coverage criterion to test alongside vertexwise tSNR, without
inventing a new metric. It is a sampling/overlap test, not a direct test of every
source of low tSNR. The paper also uses Dice>=0.7 and sufficient low-motion data.
Its 20 mm² is physical surface area, not the project's 20% invalid-vertex rule.
The available text does not specify a per-hemisphere versus bilateral area
convention or a numerical definition of an improperly interpolated vertex.
Therefore, the proposed "any single hemisphere" wording is not a verified part
of the published rule; implementation must state its chosen convention. The
surface test defines zero actual trilinear functional-GM contribution as
unsupported and fails at >=20 mm2 in either hemisphere. Nearest-voxel support
is a separately labeled, stricter diagnostic.

For that comparison, use the exact consumed subject surface and its actual
functional support/mapping evidence. First identify unsupported vertices, then
sum their subject-specific surface areas under the paper's surface definition.
Do not infer functional coverage solely from temporal variability or silently
substitute a native-volume ROI for a missing surface series. The existing
vertex-T25 comparison is retained; the motion results do not justify demoting
it or restoring its threshold as a universally calibrated gate.

## Current result boundaries

The new [five-dataset surface test](hipp_surface_qc.md) explicitly transforms
native midthickness vertices to the supplied functional volume grids. It
includes true BANDA/PDC surface projections; their earlier native-CIFTI volume
ROI counts are not substituted. PDC still lacks anatomical surfaces for many
visits, which remain unassessable. Existing HCP/Stanford selector counts are
kept separately from the registered-vertex calculations.
Completed R20 and V+R20 results remain research measurements and are excluded
from candidate selection. They do not represent completed local surface QC.

The [five-dataset report](qc_five_datasets.md) preserves the numerical motion
and signal comparisons with these labels. HCP7T now has complete declared
motion inputs for 166 participants / 664 runs. Mean RMS0.5 passes 166/166
(100.00%). These supplementary motion results
do not replace its existing surface measurements. Codex and pi's default
Argon model agree on the distinction. Production masks, rosters, models and
formal results remain unchanged.
