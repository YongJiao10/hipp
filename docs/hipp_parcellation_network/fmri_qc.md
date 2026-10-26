# fMRI QC

## Accepted policy and implementation status

The current policy uses individual T1w/HippUnfold anatomy and the actually
sampled BOLD NIfTI for hippocampal localization and functional coverage. A
CIFTI hippocampal BrainModel is not an additional anatomical inclusion mask.
Scope is HCP7T rest, HCP3T EMOTION/WM, Stanford 36P, BANDA and PDC.
The HCP7T roster entry point, `02_02_build_qc_manifest.py`, now applies all
three criteria in order. It reads exact physical-run `Movement_Regressors.txt`
paths from `--motion-manifest` and each subject's
`surface/raw/sub-<subject>_coverage_summary.json`. It validates the NIfTI
coverage policy, physical run IDs, labels and lengths against the motion
frames, and actual den-512 vertex counts. It calculates unusable-vertex counts
from the exact `*_nifti_support.npy` arrays and consumed raw GIFTI signals,
using the same runwise validity calculation as the downstream vertex mask.
Missing inputs receive `unassessable` status; malformed or mismatched
declarations stop the operation before reports are written.

The passing roster is derived from current inputs, without a fixed subject
count or historical ID requirement. Established report filenames containing
`tsnr` remain an output interface, but no tSNR numerical gate is applied.
HCP7T rest raw sampling, vertex masking, participant QC and smoothing consume
the NIfTI coverage contract. This source change does not establish that
production outputs have been rebuilt or that other dataset entry points have
been implemented.
Group-input `qc` and `all` stages for other datasets or tasks stop before writing
until their production QC input contracts are implemented.
The current HCP7T functional roster below is generated from the completed
NIfTI coverage check on the official v2.3.0 surfaces.

## Participant and visit selection

Apply these criteria in order, stopping at the first failure:

1. **Motion:** overall mean Power FD <=0.50 mm. Perform this step before
   HippUnfold when processing new inputs.
2. **Anatomy:** HippUnfold Dice >=0.70 in both hemispheres.
3. **Hippocampal coverage:** at most 20 unusable vertices in each consumed
   den-512 hippocampal hemisphere (20/512 = 3.90625%). A vertex is unusable
   when it has no trilinear NIfTI image-grid support, or its sampled signal is
   nonfinite or constant in any included physical run.

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
Apply the 0.50-mm threshold once to this overall mean. There is no individual
run veto; for example, equal-length runs with means 0.40 and 0.10 pass at 0.25.
Dice is evaluated once per anatomical reconstruction, independently of run count.

Coverage uses the exact BOLD NIfTI shape and affine at the registered
hippocampal vertices. Sum the nonzero trilinear contributions whose voxel
indices lie inside each input volume. Save the minimum image-grid support
over the explicitly declared volume inputs as `*_nifti_support.npy`.
Positive support is required, but full support from all eight neighbours is
not an additional gate. These weights describe image-grid support, not tissue
membership or an independent gray-matter segmentation.

Assess finite, nonconstant signals on the actual sampled vertex time series
separately within every declared physical run. Intersect those vertex-validity
masks with positive NIfTI support. A constant run does not become valid because
another run varies or because their concatenation has a baseline jump. Count
the union of unusable vertices once per hemisphere; do not add runwise counts.
Zero-padded regions with constant sampled signals are invalid. This minimal
data-availability criterion does not establish tissue purity, accurate
registration or adequate BOLD sensitivity.

EMOTION and WM are evaluated separately; use their intersection if an analysis
requires both tasks. PDC is evaluated per visit, without pooling different
visits to pass motion QC. Cohort identity, denoising variants and declared run
selection remain explicit.

## Vertex handling and coordinates

Passing participant selection permits up to 20 unusable vertices per side;
it does not make those vertices usable. HCP7T downstream smoothing uses the
same positive NIfTI support and runwise finite/nonconstant mask as participant
QC. FC and activation must consume the resulting masked signals and declared
support. Other dataset entry points still require their production contracts.

The coverage policy is `bold_nifti_runwise_trilinear_validity`. Raw products,
vertex gates or smoothing caches using the preceding CIFTI coverage policy
must be rebuilt explicitly; they are not resumed or substituted. For one
concatenated BOLD NIfTI, provide `--bold-template` and
`--run-metadata-template`. For separate volumes, provide the exact
`--run-artifact-manifest-template`; hippocampal preparation consumes only its
declared BOLD artifacts. Cortical CIFTI reference extraction remains separate.

Volume-to-surface sampling must use surfaces transformed into the exact BOLD
volume space. A `space-corobl` mesh cannot be used directly to sample MNI or
native T1w volumes. Use native T1w geometry for physical distances and smoothing;
displaying corresponding vertex values on a corobl mesh is a separate operation.
HCP7T raw sampling, coverage masks and 2/4-mm smoothing have been rebuilt from
the official v2.3.0 surfaces for the current 162-participant roster. PFM FC and
group-analysis progress is recorded in the
[current result assessment](../environment/hippunfold_result_impact.md).

## Current assessed cohorts

The official HippUnfold v2.3.0 HCP7T reconstruction attempted all 166 subjects.
Exactly 163 completed the official workflow and 162 also pass bilateral
Dice >=0.70. All 162 pass the current overall mean Power FD <=0.50 mm rule;
their maximum mean is 0.487185 mm. All 162 also pass the newly sampled BOLD
NIfTI coverage check, with maximum unusable-vertex counts of four on the left
and eight on the right. The final functional roster contains 162 participants
at `outputs_hcp/inputs/kong2022/manifests/hcp_7t_hippocampus_fmri_qc_pass_t1only_latest.txt`.
The current report is
`outputs_hcp/inputs/kong2022/qc/t1only_hippunfold_tsnr_qc.json`. The three
incomplete structural workflows are excluded before this 163-subject QC
assessment; its single quality failure is 182739 for right-side Dice.

NIH rTMS currently has 43 functional participants and 66 resting-state scans;
D16 is anatomical-only. Using XCP-D's unfiltered Power `mean_fd` before
censoring and after dummy removal, 64/66 scans pass <=0.50 mm. The two failures
are D11 ses-1 (0.589030 mm) and D12 ses-1 (0.751863 mm). Of the 23 TRD
participants with both treatment visits, 21 retain both visits. Use session-level
decisions for treatment analyses; do not pool pre/post visits to rescue a
high-motion visit. The 42/43 pass count obtained by pooling each participant's
available scans is a separate descriptive aggregate, not the paired cohort.
Mean-FD participant/visit selection and XCP-D's existing per-frame censoring
at FD >0.50 mm are separate operations.

## Method basis

[Gao et al., Nature Communications 2026](https://www.nature.com/articles/s41467-026-71719-y)
use mean FD >0.30 mm for exclusion in cross-cohort functional-gradient work.
The project's accepted mean-FD threshold is <=0.50 mm by explicit user choice;
the cited study uses the stricter threshold. Different TRs and filtering
affect FD, and mean FD can conceal brief large movements.

[Kember et al., Nature Communications 2026](https://www.nature.com/articles/s41467-026-74572-1)
provide the anatomical and functional-GM coverage precedent. Their published
coverage criterion is >=20 mm² improperly interpolated surface, not 20 vertices.
The project's vertex-count threshold and NIfTI sampled-signal validity rule
are explicit adaptations for den-512 functional nodes. Kember's description
does not identify its functional gray-matter mask as a CIFTI hippocampal
BrainModel or specify the exact interpolation-support classifier. A NIfTI file
format itself does not define gray matter. Positive support is not proof of
adequate SNR or individual tissue purity; retain independent registration QC.

Kember defines low-motion frames by FD <=0.25 mm and requires at least 10
low-motion minutes. This duration rule was not adopted: the project's full
EMOTION and WM acquisitions last only 4.224 and 9.72 minutes. The accessible QC
description does not establish a specific temporal-censoring implementation.

Exploration reports, one-off analysis code/results and QC discussion sessions
are retained under `_archive/hippocampal_qc_exploration/` with their original
relative directory structure. They are historical evidence, not active
workflow instructions.
