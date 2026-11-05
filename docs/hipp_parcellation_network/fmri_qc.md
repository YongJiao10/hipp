# fMRI QC

## Accepted policy and implementation status

The current policy uses individual T1w/HippUnfold anatomy and the actually
sampled BOLD NIfTI for hippocampal localization and functional coverage. A
CIFTI hippocampal BrainModel is not an additional anatomical inclusion mask.
Scope is HCP7T rest, HCP3T EMOTION/WM, Stanford 36P, BANDA, PDC and NIH rTMS visits.
The shared roster entry point, `02_02_build_qc_manifest.py`, applies all
three criteria in order for every supported dataset and task. It reads exact
physical-run BOLD and motion declarations from `--qc-run-inputs` and each subject's
`surface/raw/sub-<subject>_coverage_summary.json`. It validates the NIfTI
coverage policy, physical run IDs, labels and lengths against the motion
frames, and actual den-512 vertex counts. It calculates unusable-vertex counts
from the exact `*_nifti_support.npy` arrays and consumed raw GIFTI signals,
using the same runwise validity calculation as the downstream vertex mask.
Missing inputs receive `unassessable` status; malformed or mismatched
declarations stop the operation before reports are written.

The passing roster is derived from current inputs, without a fixed subject
count or historical ID requirement.
Raw sampling, vertex masking, participant QC and smoothing consume the same
NIfTI coverage contract across HCP rest/tasks, BANDA, PDC, Stanford, Iowa and NIH rTMS.
Dataset registration and motion input formats are explicit input adapters;
dataset identity does not disable coverage QC. Existing HCP7T rest commands
may still provide their exact `--motion-manifest` table and BOLD/run metadata.
The HCP7T functional roster is generated from the completed
NIfTI coverage check on the official v2.4.0 surfaces.

## Shared run inputs and commands

`--qc-run-inputs` selects a TSV with one row per physical run:

Supply the complete participant or visit roster separately with `--subjects`
or `--subjects-manifest` to both preparation and direct QC calls. A participant
without a physical-run row is then reported as unassessable instead of omitted.

| Column | Meaning |
| --- | --- |
| `subject` | Participant or visit ID, with optional `sub-` prefix |
| `run_id` | Unique positive integer; runs are consumed in ascending order |
| `run_label` | Exact physical acquisition label |
| `bold_path` | Exact absolute 4D BOLD NIfTI path |
| `n_timepoints` | Frame count in the consumed physical-run BOLD |
| `motion_path` | Exact absolute original motion/confounds path |
| `motion_format` | `hcp_movement_degrees` or `fmriprep_confounds` |
| `temporal_mask_path` (optional) | Exact XCP-D outlier TSV when the consumed BOLD has censored frames |

The HCP motion format consumes the first six of twelve columns, with rotation
in degrees. The fMRIPrep format consumes `framewise_displacement` from the
original confounds TSV. Both set the first frame of each run to zero and
require the declared frame count. For censored XCP-D data, `n_timepoints`
describes the consumed BOLD frames; the explicit outlier TSV must contain one
row per original motion frame, with exactly `n_timepoints` zero-valued entries.
FD is averaged over original post-dummy-removal frames, before censoring;
coverage uses the shorter consumed series. An empty `bold_path` records a
missing source declaration explicitly: coverage is unassessable. It never
selects a CIFTI, reference image or mask in place of the BOLD series.

Task/non-HCP preparation requires separate `--route-root`, `--shared-root`
and, when building a group template, `--template-root`. Direct `02_02` calls
require separate `--shared-root`, `--out-dir` and `--manifest-dir` for those
cohorts. Run `02_01` stages `raw` and `validity`, then `qc`; `all` also
runs smoothing and the group-template stage. The hippocampal mask uses NIfTI
support and finite, nonconstant signal within every physical run.

For v2.3.0 cohorts, supply `--require-hippunfold-version 2.3.0` to the QC
entrypoint. Official completion metadata must match that release; an
incomplete reconstruction cannot enter through a high Dice score. Retained
post-cleanup surfaces remain usable for functional QC when their completed
run is recorded; this does not satisfy the full structural-resume contract.

Each route contains `qc/t1only_hippunfold_qc.{json,tsv}`. Passing rosters
are `<cohort>_hippocampus_fmri_qc_pass_t1only_<N|latest>.txt`, where cohort is
`hcp_7t`, `hcp_3t_emotion`, `hcp_3t_wm`, `banda`, `pdc`, or the selected dataset.
Reports use `incomplete_inputs` status when any participant is unassessable;
zero passing participants is a valid reported outcome and skips group work.

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

Mean relative RMS <=0.5 mm (M05) adds
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

NIH rTMS uses separate `ses-1` and `ses-2` routes below
`outputs_nih_rtms/inputs/kong2022/`; the same participant's visits are never
pooled for selection. Each visit uses its exact XCP-D denoised NIfTI, original
motion TSV and outlier TSV. The transform is derived from the exact fMRIPrep
preprocessed T1w consumed by HippUnfold, following its recorded input symlink.
The matching `from-MNI152NLin6Asym_to-T1w_mode-image_xfm.h5` shares that source's
directory and entity prefix. This preserves both longitudinal subject-level
and single-anatomical-visit session-level exports.

Passing participant selection permits up to 20 unusable vertices per side;
it does not make those vertices usable. HCP7T downstream smoothing uses the
same positive NIfTI support and runwise finite/nonconstant mask as participant
QC. FC and activation must consume the resulting masked signals and declared
support for their selected dataset/task branch.

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
The sampler joins the disconnected registered L/R meshes for one Workbench
mapping per physical run, then saves separate hemisphere signals in their
original vertex order. Trilinear interpolation still evaluates each vertex
independently. Run-level sampling saves the full vertex-by-time arrays as
`surface/raw/sub-<ID>_hemi-<L|R>_timeseries.npy` and GIFTIs, with run boundaries
in `surface_sampling_summary.json`. The current raw-resume and QC commands
still require the declared source BOLD files, even when those surface products
are present. Retain the volumes during this QC run.
HCP7T raw sampling and coverage have been rebuilt from the official v2.4.0
surfaces for the current 166-participant roster. Cortical references and
dependent FC products require regeneration under the current signal policy.

## Current assessed cohorts

The official HippUnfold v2.4.0 HCP7T reconstruction and functional QC cover
all 166 participants. The current report records 166 passes, zero quality
failures and zero unassessable participants. The accepted roster is
`outputs_hcp/inputs/kong2022/manifests/hcp_7t_hippocampus_fmri_qc_pass_t1only_latest.txt`.

HCP3T EMOTION includes 159 task participants. All 159 pass motion; 155 have
recorded official v2.3.0 completion and pass bilateral Dice. All 155 also pass
NIfTI coverage, with zero unusable vertices in either hemisphere. Raw sampling
and vertex masks are complete. The report is
`outputs_hcp/inputs/kong2022/shared_inputs/task-emotion/qc/t1only_hippunfold_qc.json`;
the final roster is `hcp_3t_emotion_hippocampus_fmri_qc_pass_t1only_latest.txt`
in the route's `manifests/` directory. The remaining four participants are
182739 (right Dice) and 233326, 249947, 942658 (no official completion).
HCP3T WM has the same 159-person input roster and 155 motion/anatomy passes;
all 155 also pass NIfTI coverage, with zero unusable vertices on each side.
Raw sampling and vertex masks are complete. The report and final roster are
under the corresponding `task-wm/qc/` and `task-wm/manifests/` directories,
with prefix `hcp_3t_wm`.

BANDA has 203 attempted structural participants, with 196 v2.3.0 completion
records. Among the 202 participants with declared resting runs, 199 pass
motion; BANDA028, BANDA116 and BANDA195 exceed mean FD 0.50 mm. Exactly 193
pass motion and completed anatomy/Dice. All 808 declared Legacy Volume BOLD
runs are present. Coverage sampling and vertex masks are complete for all 193
eligible participants; all pass, with zero unusable vertices in each hemisphere.
The final report records 193 passes, three motion failures and seven
unassessable participants.
BANDA199 has no run-level rest declaration; six other participants lack
structural completion records after motion selection. The current report is
`outputs_banda/inputs/kong2022/qc/t1only_hippunfold_qc.json`.

PDC has 455 attempted visits, with 442 v2.3.0 completion records. Motion passes
451 visits; e0020s01, k0082s01, s0063s01 and s0066s02 exceed mean FD 0.50 mm.
Exactly 438 pass motion and completed anatomy/Dice. There are 982 of 984
declared Legacy Volume BOLD runs present. Coverage sampling and vertex masks
are complete for all 437 eligible visits with all runs present; all pass, with
zero unusable vertices in each hemisphere. The report records 437 passes,
four motion failures and 14 unassessable visits. Thirteen of those visits lack
structural completion records. The current report is
`outputs_pdc/inputs/kong2022/qc/t1only_hippunfold_qc.json`.
The only missing BOLD runs are `k0008_01_MR` REST1 AP/PA, corresponding to
visit `k0008s01`; their Legacy Volume sources were not matched. QC retains
their exact expected paths and marks that visit unassessable until both runs
are available.

Both reports retain `incomplete_inputs` status because the required inputs
identified above remain missing.
The exact run tables are `<dataset>_qc_runs.tsv` in the adjacent `manifests/`
directories. They preserve all 808 BANDA and 984 PDC physical runs and now
declare the exact `<run_label>_hp0_clean.nii.gz` Legacy Volume paths beside
the original motion files. Missing files remain unassessable; no CIFTI-derived
volume is substituted.

### BANDA and PDC volume source

The local dataset READMEs record NDA personal packages 1250287 (`BANDA11Rec`)
and 1250294 (`PDCRec`). Their `metadata/datastructure_manifest.txt` inventories
declare cleaned MSMAll CIFTIs, original motion, reference images and masks;
they do not declare `rfMRI_REST*_hp0_clean.nii.gz` 4D volumes. The official
[HCP maintainer explanation](https://groups.google.com/a/humanconnectome.org/g/hcp-users/c/Dqe9fcBgIYM)
identifies **rfMRI Preprocessed Legacy Volume** as the source of the cleaned
volume, distinct from the Recommended CIFTI package. The
[NDA CRHD access page](https://nda.nih.gov/general-query.html?q=query%3Dfeatured-datasets%3AConnectomes+Related+to+Human+Disease)
supports selecting that processing subset or using the corresponding AllFiles
package. The exact declared files are stored in each dataset's original
`imaging/` hierarchy, with the two PDC missing runs listed above. Shared QC
consumes those paths directly. Do not convert the selected CIFTI grayordinates
into a replacement volume or use a reference image as the time series.

NIH rTMS currently has 43 functional participants and 66 resting-state scans;
D16 is anatomical-only. All 44 anatomical participants have official v2.3.0
completion records. The retained functional cohort comprises 42 ses-1 and
24 ses-2 scans; their visit-specific input tables are
`outputs_nih_rtms/inputs/kong2022/ses-<1|2>/manifests/nih_rtms_qc_runs.tsv`.
Raw sampling, vertex masks and final coverage assessment are complete for
all 64 motion/anatomy-passing scans. The final rosters contain **37/42 ses-1**
and **24/24 ses-2** scans, totaling **61/66**. The two motion exclusions are
D11 ses-1 (0.589030 mm) and D12 ses-1 (0.751863 mm), using XCP-D's unfiltered
Power FD before censoring and after dummy removal. Three further ses-1 scans
fail sampled-signal coverage: D08 has 55 unusable right vertices, D09 has 29
unusable left vertices, and D28 has 85 unusable left vertices. All their
vertices have positive image-grid support; the excluded vertices have invalid
sampled signals. All other assessed hemispheres satisfy the <=20 rule.

Each visit's report is `qc/t1only_hippunfold_qc.json` and its final roster
is `manifests/nih_rtms_hippocampus_fmri_qc_pass_t1only_latest.txt` within that
visit route. Both reports have `done` status, with no unassessable scans.
Of the 23 TRD participants with both treatment visits, **18 retain both
visits** after all three criteria. Use session-level decisions for treatment
analyses; do not pool pre/post visits to rescue an excluded visit. Mean-FD
participant/visit selection and XCP-D's existing per-frame censoring at
FD >0.50 mm are separate operations.

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
