# Coordinate-only functional pilot

## Scope and design

This pilot measures the effect of correcting hippocampal volume-sampling
coordinates. It does not repair production outputs or choose a final QC gate.
Existing literature and the five-dataset QC comparison remain available in
[hipp_surface_qc.md](hipp_surface_qc.md).

HCP7T participants are fixed before measuring outcomes: 100610, 165436, 205220,
573249 and 966975, positions 0/37/75/113/150 in the existing 151-person reference
roster. Each contributes the entire original 3,600-frame, 1.6-mm MNI BOLD volume
declared in its raw surface-stage metadata. This is a five-person sensitivity
analysis, not a replacement for the HCP151 atlas or an estimate for all cohorts.

Both arms use unrestricted trilinear sampling of that same BOLD. The old arm
uses corobl physical coordinates. The registered arm uses native T1w vertices
transformed to MNI with the FNIRT chain already tested in the QC analysis.
Both use the exact same **shared valid mask**, cortex ROI references, 2-mm FWHM
corobl smoothing surface, run lengths and numerical methods. Shared validity
starts from the original mask and removes vertices with nonfinite or constant
registered signals, applying the identical resulting mask to both arms. This
is basic calculability, not a new numerical QC gate or participant exclusion.
The adjustment was required by a real NaN source voxel in sub-165436, described
below. Retaining the old
smoothing geometry is deliberate: **only sampling coordinates change**. Native
physical smoothing is a separate production-repair choice, outside this test.
No CIFTI-GM restriction, new denoising, threshold adjustment, cohort change,
cortex regression or cortex-neighbor exclusion is introduced.

## Measurements and interpretation

1. Reproduce existing raw caches with the declared volume and old coordinates;
   independently compare the new sparse sampler with Workbench on four frames.
   This detects a sampler discrepancy before interpreting downstream changes.
   Raw-cache agreement requires matching NaN locations; maximum absolute
   differences describe finite entries.
2. Compare old/new vertex time series, 400-ROI FC profiles and canonical-seven
   network WTA labels. Derive canonical-seven references from the same 400 ROI
   signals with the production family17 mapping and aggregation function.
3. Average subject Fisher-z FC with the existing 80% group-support convention;
   use the production BrainSpace Pearson embedding, five components, seed 0,
   column z-scoring and 0.9 sparsity. Run the production surface-adjacency
   gradient-spectral method on G1–G3, fixed K=3 and seed 0.
4. Report G1/G2/G3 rank correlations, explicitly sign invariant; also match
   components and compare their subspace to distinguish changes from component
   reordering. Report label-permutation-adjusted K=3 agreement and adjusted Rand
   index. Similar K=3 geometry alone cannot validate the underlying FC, because
   this method explicitly imposes surface adjacency.

Original-mask smoothing and FC are separately reproduced and compared directly
with existing formal caches; shared-mask old/new comparisons follow afterward.
Those comparisons distinguish the controlled reconstruction from claims about
an existing cache's actual contents. tSNR25 is not deployed; the
retained HCP nominal-10000/SD proxy is only a descriptive signal comparison.

## Execution and outputs

Run on IDAS with explicit py314; one analysis process and at most one Workbench
child, both single threaded. Inputs and formal output trees remain read-only.
The one-off driver is in `tmp/hippocampal_qc/scripts/`. All generated products
are under `tmp/hippocampal_qc/outputs/`, with per-subject arrays in its
`coordinate_pilot_data/` subdirectory. No files are deleted by the driver and
no upstream preprocessing is launched.

```bash
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1 ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS=1
export CORE_WORKFLOW_WB_THREADS=1
export CORE_WORKFLOW_WB_COMMAND=/home/yojiao/FASTANS/workbench/bin_linux64/wb_command
/home/yojiao/.conda/envs/py314/bin/python tmp/hippocampal_qc/scripts/coordinate_pilot.py \
  --stage sample --subjects 100610 --sampler-check
/home/yojiao/.conda/envs/py314/bin/python tmp/hippocampal_qc/scripts/coordinate_pilot.py \
  --stage sample --subjects 165436 205220 573249 966975
/home/yojiao/.conda/envs/py314/bin/python tmp/hippocampal_qc/scripts/coordinate_pilot.py --stage group
/home/yojiao/.conda/envs/py314/bin/python tmp/hippocampal_qc/scripts/coordinate_pilot.py --stage plot
```

`coordinate_pilot_subjects.tsv` and `coordinate_pilot_group.tsv` contain the
individual and group comparisons. Per-subject metrics, FC arrays and group
embeddings/labels are retained under the pilot data directory. Generated
numeric/binary outputs follow the repository's local-results Git policy.
`coordinate_pilot_maps.png` and `.pdf` display paired G1/G2/K=3 maps on identical
unfolded geometry, with gradient signs and cluster colors matched for display.

## Results

All five participants completed both hemispheres with the full 3,600 frames.
Old-coordinate sampling reproduces the original raw caches within interpolation
precision (maximum finite absolute difference 0.021; matching NaN locations).
Original-mask 2-mm smoothed time series are exactly reproduced in all ten
hemispheres; the maximum difference from existing formal 400-ROI FC is
5.96e-8. The old-input defect is therefore confirmed in consumed caches, rather
than inferred from source-code defaults alone.

Only three of 4,845 initially included participant-vertices need removal from
the **shared** comparison support: sub-165436 right vertex 26 and sub-966975
right vertices 469/470. Both arms exclude them. All five people remain; 4,842
participant-vertices have paired valid signals. The median coordinate
displacement in the ten individual hemispheres ranges from 11.59 to 13.74 mm;
the largest individual vertex displacement is 22.06 mm.

### Individual time series and connectivity

Each cell reports **left / right**. Temporal r is the median vertexwise Pearson
correlation of unsmoothed old/new time series. FC-profile r compares the 400
ROI correlations at each shared valid vertex after the matched 2-mm smoothing.
WTA is direct maximum-correlation assignment to the canonical seven networks;
it is a diagnostic, not a recomputed formal template-matching result. No equal
network frequencies or random-chance baseline are assumed.

| Participant | Temporal r | FC-profile r | Seven-network WTA agreement |
|---|---:|---:|---:|
| 100610 | 0.0019 / 0.0027 | 0.021 / 0.072 | 19.3% / 16.5% |
| 165436 | 0.0006 / -0.0006 | 0.056 / 0.041 | 22.5% / 15.6% |
| 205220 | 0.0026 / 0.0023 | 0.125 / 0.005 | 23.4% / 15.8% |
| 573249 | 0.0017 / 0.0041 | 0.038 / 0.163 | 13.7% / 26.1% |
| 966975 | 0.0027 / 0.0016 | -0.084 / -0.002 | 12.5% / 20.9% |

### Paired five-person group analysis

Both columns compare old versus corrected maps built from **the same five
participants**, not an old HCP151 atlas versus a new five-person atlas. Group
support is identical between arms: 501 left and 485 right vertices, each
supported by at least four of the five participants.

| Old/new comparison | Left | Right |
|---|---:|---:|
| Correlation across the entire group mean Fisher-z FC matrix | 0.052 | 0.058 |
| Median vertex FC-profile correlation | 0.056 | 0.095 |
| G1 absolute Spearman correlation, same component index | 0.597 | 0.466 |
| G2 absolute Spearman correlation, same component index | 0.227 | 0.090 |
| G3 absolute Spearman correlation, same component index | 0.127 | 0.056 |
| Mean absolute correlation after matching G1–G3 components | 0.317 | 0.273 |
| Three-dimensional gradient subspace cosines | 0.082 / 0.361 / 0.647 | 0.141 / 0.216 / 0.604 |
| K=3 agreement after matching label numbers | 446/501 (89.0%) | 472/485 (97.3%) |
| K=3 adjusted Rand index | 0.692 | 0.917 |

The underlying time series and FC change profoundly. G1 retains moderate
spatial correspondence, whereas G2/G3 and the combined gradient subspace show
substantial change; component sign/order alone cannot explain it. The coarse
K=3 partition remains much more similar, particularly on the right. This
gradient-spectral method explicitly constrains clustering to surface adjacency,
so partition agreement does not validate FC or the networks assigned to each
partition. This pilot does not identify how much of the partition similarity
comes from that constraint versus retained functional organization.

The [paired map figure](../tmp/hippocampal_qc/outputs/coordinate_pilot_maps.png) shows G1, G2 and K=3 on common
unfolded geometry. Signs and cluster colors are aligned for display; the
numerical comparisons explicitly account for those arbitrary choices. The
[PDF](../tmp/hippocampal_qc/outputs/coordinate_pilot_maps.pdf), [subject table](../tmp/hippocampal_qc/outputs/coordinate_pilot_subjects.tsv)
and [group table](../tmp/hippocampal_qc/outputs/coordinate_pilot_group.tsv) are retained locally.

### Decision supported by this pilot

Proceed with the sampling-code repair and rerun its existing validity checks;
keep final numerical QC selection deferred until corrected inputs are available.
The pilot already establishes a real, material defect in hippocampal functional
inputs. Repeating broader threshold searches or enlarging this pilot is not
required before fixing that source. Exact production rebuild scope follows
actual dependencies after the corrected sampling path is ready.

This is not evidence that every scientific conclusion reverses. It has not
remeasured HCP151, task activation, Stanford/BANDA/PDC connectivity, native TMS
beta, group differences or formal template-matching assignments. Raw data and
anatomical reconstructions are not invalidated by this downstream defect.
No production code, formal result tree, cohort or QC threshold was changed.

### Nonfinite source value encountered during the pilot

Sub-165436 right vertex 26 touches original MNI voxel (46, 61, 39), which is NaN
in all 3,600 frames and has positive interpolation weight 0.01936. Independent
Workbench sampling of four original frames returns NaN at this vertex as well.
It is not a discrepancy in the new sparse sampler. The original corobl-based
mask included this vertex. The shared-mask comparison excludes unmeasurable
vertices identically in both arms; final production validity masks must be
recomputed after correcting coordinates.
