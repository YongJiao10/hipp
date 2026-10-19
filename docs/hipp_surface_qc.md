# Five-dataset hippocampal surface QC test

This research comparison tests hippocampal surface tSNR25 against a
Kember-inspired functional-GM coverage selector and their intersection. The
registered tables are the main result; initial tables retain existing-selector
reference counts.
It leaves production surfaces, masks, scientific results and rosters unchanged.

## Definitions

- **Basic:** the existing bilateral Dice >=0.70 and finite/nonconstant signal
  checks, with <20% invalid vertices in each hemisphere.
- **Initial T25 reference:** the existing HCP/Stanford participant decision from
  `tsnr_qc_comparison_subjects.tsv`, including Basic. HCP7T is the legacy
  10000/SD proxy; HCP3T uses the project's OLS task-residual tSNR; Stanford
  uses the saved matched-baseline, detrended rest calculation. These are
  current-selector reference counts, not newly recomputed, uniformly defined
  tSNR at the transformed anatomical coordinates.
- **Registered T25:** recompute at the native-to-MNI transformed anatomical
  vertices. HCP7T preserves its declared nominal-10000 / concatenated sample-SD
  proxy because the supplied CIFTI is temporally centered. HCP3T uses the same
  task OLS/DCT128 model and ddof=1 as the current comparison, now sampling the
  CIFTI hippocampal GM representation at the registered vertices. Stanford
  samples the explicit preprocessed BOLD volume at the registered vertices and
  uses the declared 22P/36P retained original frames for mean / linearly
  detrended sample SD. Its delivered denoised CIFTI supplies finite/nonconstant
  signal validity, not a fabricated baseline. Basic is recomputed at the same
  registered vertices. This is a research comparison, not a production repair.
- **BANDA/PDC registered T25:** a new explicit surface projection of the supplied cleaned
  CIFTI hippocampal volume models. Trilinear weights sample each run at the
  transformed midthickness vertices. Divide the weighted series by its positive
  GM weight (a constant rescaling, which leaves mean/SD tSNR unchanged). Calculate
  temporal mean / sample SD, ddof=1; no additional smoothing or detrending.
  Each vertex must meet 25 in the included runs; <20% invalid vertices per
  hemisphere and Basic are required. This replaces neither production inputs
  nor the distinct older native-CIFTI voxel-ROI comparison.
- **Coverage20:** Basic plus <20 mm2 unsupported native folded surface area in
  **each** hemisphere. A vertex is unsupported when its actual trilinear
  interpolation weights sum to zero over the provided hippocampal functional
  GM voxels that have finite, nonconstant signal in the supplied runs. The
  functional GM set is the CIFTI hippocampal BrainModel, not HippUnfold's own
  segmentation or an unrelated brain mask. The same definition is used in
  all datasets. Report L/R areas and their bilateral sum.
- **Joint:** T25 AND Coverage20 at the participant/visit level; no extra gate.
- **Nearest20 diagnostic:** the same area calculation after classifying only
  the nearest voxel. It is deliberately labeled separately: an unmodeled
  nearest voxel can coexist with positive GM contributions from other neighbors.
  This stricter test is not the primary coverage definition.

Triangle areas are calculated on the individual folded **T1w** midthickness
mesh in mm2 and divided equally among each triangle's three vertices. Mesh
topology and vertex order remain den-512; no unfolding-area approximation,
equal-area-per-vertex approximation or remeshing is used.

[Kember et al., Nature Communications 2026](https://www.nature.com/articles/s41467-026-74572-1)
supports an exclusion threshold of >=20 mm2 improperly interpolated hippocampal
surface. It does not fully specify its vertex classifier or hemisphere
convention. The definitions above are an explicit **project implementation**,
not a claim of exact replication or a validated cross-acquisition threshold.
Positive GM weight establishes some supplied functional-GM contribution; it
does not establish adequate SNR, majority-GM composition or correct segmentation.
The sampled model is an atlas-defined functional hippocampus representation,
not an independent voxelwise measurement of each subject's tissue fraction.

## Coordinate handling and current-selector limitation

The inspected production summaries sample `space-corobl` meshes directly into
MNI BOLD volumes. A corobl filename/in-bounds coordinate does not establish MNI
registration. Native T1w meshes and declared anatomical transforms supply an
explicit registered route for this **new coverage analysis**:

- HCP: convert the provided `source/acpc_dc2standard.nii.gz` FNIRT warp to a
  NIfTI RAS displacement with Workbench. The inverse image warp is absent.
  Solve `MNI + displacement(MNI) = native` only at the required vertices;
  accept solutions only with maximum native-coordinate residual <0.001 mm.
- Stanford: native RAS -> LPS -> the explicit fMRIPrep
  `from-MNI152NLin6Asym_to-T1w_mode-image_xfm.h5` image composite using
  `antsApplyTransformsToPoints` -> RAS. As documented by
  [ANTs](https://github.com/ANTsX/ANTs/wiki/Forward-and-inverse-warps-for-warping-images%2C-pointsets-and-Jacobians),
  point and image transform directions are opposite.
- BANDA/PDC: Workbench `-surface-apply-warpfield` with
  `standard2acpc_dc.nii.gz -fnirt acpc_dc2standard.nii.gz` from the declared
  structural subject's `MNINonLinear/xfms`.

The initial HCP/Stanford T25 reference decisions retain their previous sampled
coordinates. They do not establish that existing low tSNR reflects correctly
localized hippocampal noise. `hipp_surface_tsnr_registered.py` completes the
test at registered vertices and retains the initial decisions in a separate
reference column. Its HCP/BANDA/PDC CIFTI GM projection differs from the earlier
unrestricted volume sampling, so changes cannot be attributed to coordinates
alone. HCP7T remains explicitly a proxy, not measured-baseline tSNR. All groups
use the same coverage-area criterion; the tSNR inputs retain their declared
baseline and task/rest definitions. Formal sampling outputs remain unchanged.

### Confirmed impact on functional time series

This is a functional-sampling defect, not just a change in a QC statistic.
In representative sub-100610, the median displacement from the old corobl
coordinates to registered MNI coordinates is 12.98 mm left / 12.31 mm right
(maximum 20.19 / 20.33 mm). For Stanford sub-1017 it is 13.00 / 12.71 mm
(maximum 20.32 / 20.74 mm). These are representative cases, not cohort-wide
displacement estimates.

A controlled check on sub-100610 EMOTION LR keeps the **same 176-frame,
2-mm BOLD volume, trilinear sampler and task GLM**, changing only surface
coordinates. Sampling the old corobl coordinates exactly reproduces the
existing production raw GIFTI's LR values (maximum absolute difference zero).
Thus the problem occurs in the consumed cache, not merely in a filename.

| Measure, same-volume check | Left | Right |
|---|---:|---:|
| Old coordinates: vertices with zero hippocampal CIFTI-model support | 318/512 | 314/512 |
| Registered coordinates: vertices with zero model support | 0/512 | 0/512 |
| Median temporal correlation, old versus registered series | -0.0028 | 0.0036 |
| Old median task residual tSNR | 30.18 | 29.09 |
| Registered median task residual tSNR | 32.25 | 33.29 |
| Old vertices failing tSNR25 | 117/512 | 144/512 |
| Registered vertices failing tSNR25 | 71/512 | 55/512 |

The support counts refer to the supplied hippocampal functional model, not
independent tissue-class labels. The near-zero temporal correlations show
that the sampled responses can change fundamentally even when median tSNR
changes moderately. This one-run check isolates coordinates; the full-cohort
registered comparison also changes HCP's functional representation, as stated
above. Research examples are retained under
`hipp_surface_qc_samples/coordinate_impact/sub-100610_EMOTION_LR/`.

The dependency chain is explicit in the current code: preparation selects
corobl meshes for direct BOLD sampling in
`02_01_prepare_group_inputs.py`; `04_01_materialize_kong2022_roi_fc.py`
loads the derived hippocampal time series; `05_01_compute_hippomaps_gradients.py`
loads those FC caches. Task activation and background FC likewise consume the
derived surface series. Consequently, functional QC, hippocampal FC/background
FC, task activation, and their dependent gradients, functional parcellations
and group summaries require recomputation **where they consume affected
caches**. Their numerical changes have not yet been measured cohort-wide.

Raw acquisition files, upstream volume preprocessing and HippUnfold anatomical
reconstructions are not invalidated by this particular sampling defect.
Repair starts at volume-to-surface sampling and its consumers. The registered
QC test did not repair existing formal FC, gradient, activation or atlas files.

The HCP7T resting-state raw cache for sub-100610 also agrees with direct corobl
sampling of its declared 1.6-mm BOLD volume: a four-frame check has maximum
absolute interpolation difference 0.0062. The current formal ctxreg summary
declares the shared hippocampal surface time series as its input. This extends
the confirmed input concern beyond task fMRI; it is still a representative
input/lineage check, not a cohort-wide recomputation of FC or gradients.

The conclusion is that affected hippocampal functional analyses need controlled
recalculation and reassessment. It does not establish that every project result
or every scientific conclusion is wrong. The direction and size of changes in
FC, gradients, parcellations and group effects remain unmeasured. Repair scope
must follow each formal branch's actual consumed inputs; existing result files
are retained for comparison.

## Populations and outputs

Authoritative parents: HCP7T 166; HCP3T EMOTION and WM, 144 identical people;
Stanford 199, with 22P/36P reported separately; BANDA 203; PDC 455 visits in
218 people. BANDA199 has no independently declared functional runs. Missing
surfaces/function/transform files are unassessable, not failed quality. PDC
still lacks many bilateral surface products, so this test uses those currently
available. All parent entries remain in the denominator.

PDC also reports people with at least one passing visit. That count describes
available data and does not authorize discarding visits within longitudinal
analyses. The two-task HCP3T summary requires both tasks to pass.

Run from the repository root:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
NUMEXPR_NUM_THREADS=1 ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS=1 \
/home/yojiao/.conda/envs/py314/bin/python docs/hipp_surface_qc.py

OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
NUMEXPR_NUM_THREADS=1 ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS=1 \
/home/yojiao/.conda/envs/py314/bin/python docs/hipp_surface_tsnr_registered.py
```

`--pilot` tests sub-100610, sub-1017, sub-BANDA001 and sub-e0010s01 across their
applicable branches. Numerical libraries and ANTs use one thread; two analysis
workers plus at most two external commands stay below IDAS's 10-process cap.

- `hipp_surface_qc_subjects.tsv`: individual decisions, areas, source paths and
  actionable missing reasons, initially compared with existing-selector T25.
- `hipp_surface_qc_summary.tsv`: total/pass/fail/unassessable and rates against
  both the parent and evaluable denominators.
- `hipp_surface_qc_registered_subjects.tsv` and
  `hipp_surface_qc_registered_summary.tsv`: main comparison using registered
  surface tSNR and signal validity, with existing T25 retained as a reference.
- `hipp_surface_qc_samples/`: research transformed geometry and vertex maps;
  HCP world-field conversions and Stanford point CSVs are retained inputs to
  this analysis. No temporary or user-owned files are deleted.

## Results

Completed 19 October 2026. Percentages below use the **evaluable** denominator;
the TSV also gives confirmed passes divided by the complete parent roster.
All three selectors include Basic. EMOTION/WM and Stanford 22P/36P are analysis
branches of the same respective parent datasets, not additional independent
cohorts.

| Dataset / unit | Parent N | Evaluable N | T25 pass (%) | Coverage20 pass (%) | Joint pass (%) |
|---|---:|---:|---:|---:|---:|
| HCP7T rest | 166 | 166 | 165 (99.4%) | 132 (79.5%) | 132 (79.5%) |
| HCP3T EMOTION | 144 | 144 | 128 (88.9%) | 114 (79.2%) | 103 (71.5%) |
| HCP3T WM | 144 | 144 | 132 (91.7%) | 114 (79.2%) | 106 (73.6%) |
| HCP3T both tasks | 144 | 144 | 126 (87.5%) | 114 (79.2%) | 101 (70.1%) |
| Stanford 22P | 199 | 197 | 188 (95.4%) | 168 (85.3%) | 164 (83.2%) |
| Stanford 36P | 199 | 199 | 188 (94.5%) | 169 (84.9%) | 164 (82.4%) |
| BANDA people | 203 | 202 | 195 (96.5%) | 174 (86.1%) | 170 (84.2%) |
| PDC visits | 455 | 207 | 198 (95.7%) | 196 (94.7%) | 187 (90.3%) |
| PDC people, any passing visit | 218 | 101 | 98 (97.0%) | 97 (96.0%) | 93 (92.1%) |

Missing products are not quality failures: Stanford 22P has two missing
functional inputs; BANDA199 has no independent-run declaration; 248 PDC visits
lack bilateral anatomical surfaces. The PDC person row describes availability
of at least one passing visit, not a complete longitudinal inclusion rule.

The central finding is that the earlier extreme task attrition was not a
reliable estimate for registered hippocampal surface data. EMOTION changes
from the existing selector's 39/144 to 128/144; WM from 36/144 to 132/144.
Both coordinate mapping and the explicit GM functional representation change,
so this is not an isolated experiment attributing every change to one cause.
The registered T25 test retains 88.9–99.4% of evaluable scan products across
the individual dataset/task branches. This supports continuing to evaluate
surface tSNR25 in these data; it does not independently calibrate 25 across
all acquisitions or preprocessing definitions.

Coverage detects a different defect. HCP7T Basic/T25 passes 165 people, whereas
Coverage20 passes 132. For PDC, T25 passes 198 visits, Coverage20 passes 196,
and their intersection passes 187: the failed sets add information rather
than being identical. The joint scheme is materially stricter, retaining
101/144 people if both HCP tasks must pass. These are candidate-selection
effects, not an automatic production policy.

The nearest-voxel-only 20-mm2 diagnostic passes **zero** evaluable cases in
every branch. This illustrates why its different GM-support condition must
not be confused with zero actual trilinear contribution, and why the citation
alone is insufficient to reproduce the unreported vertex classifier. The
primary definition was fixed before the full-cohort pass-count calculation;
neither numerical threshold was adjusted to obtain a preferred retention rate.

The run produced [individual results](hipp_surface_qc_registered_subjects.tsv)
and [all totals, failures, unknowns and both rate denominators](hipp_surface_qc_registered_summary.tsv).
Checks compare all measured vertex masks with participant decisions and parent
rosters, then independently recount the 45 summaries, task intersection and
PDC person aggregation. On the BANDA001 sample, sparse interpolation agrees
with Workbench to relative error 5.4e-7 and native vertex areas to <4e-7 mm2;
HCP inverse-coordinate residuals are <0.001 mm. Formal results remain unchanged.
