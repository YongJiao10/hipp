# Five-dataset measurements: motion and hippocampal signal

HCP7T motion evaluated on IDAS, 19 October 2026; other completed measurements are reused unchanged. Scope: HCP7T rest, HCP3T task, Stanford,
BANDA and PDC. EMOTION/WM and Stanford 22P/36P are analysis branches, not extra
independent populations. PDC contains 218 people with 455 sessions.

**The primary target is hippocampal surface-vertex signal quality.** Participant
mean relative RMS<=0.5 mm is the retained supplementary motion candidate,
following [Savage et al., Communications Biology 2024, Quality control](https://www.nature.com/articles/s42003-024-06573-z).
It is a whole-scan motion measure and does not identify usable hippocampal
vertices. The user has excluded the per-run R20 rule from further consideration.
The tables below retain completed research measurements; R20 is not an active
selection proposal. See the [current surface-QC decision](tsnr_task_rest_qc_justification.md).

## Motion-only comparison

These counts isolate each motion rule, before anatomy or signal coverage.
M05 is mean relative RMS<=0.5 mm. R20 is the completed, excluded run-motion comparison. F25 permits
no more than 25% of frames with Power FD>0.5 mm in every run; it is a secondary
FD-motion comparison informed by [Tozzi et al., Nature Medicine 2024](https://pmc.ncbi.nlm.nih.gov/articles/PMC11271415/).
Its amplitude/proportion are linked across supplementary sections; it does not
reproduce that study's visual QC or full FD/DVARS treatment. See [source details](tsnr_qc_task_banda_primary_extension.md).

Every percentage uses the fixed parent denominator. U denotes missing evidence;
when U>0, the reported percentage is confirmed retention, not an estimated final
quality pass rate. No rate is assigned to an entirely unevaluated motion row.

| Dataset / analysis unit | Parent total | Mean RMS<=0.5 | R20: RMS0.25 / 20% | F25: FD0.5 / 25% |
|---|---:|---|---|---|
| HCP7T rest | 166 | 166/166 (100.00%) | 134/166 (80.72%) | 157/166 (94.58%) |
| HCP3T EMOTION | 144 | 144/144 (100.00%) | 139/144 (96.53%) | 142/144 (98.61%) |
| HCP3T WM | 144 | 144/144 (100.00%) | 142/144 (98.61%) | 142/144 (98.61%) |
| HCP3T both tasks | 144 | 144/144 (100.00%) | 138/144 (95.83%) | 141/144 (97.92%) |
| Stanford 22P | 199 | 198/199 (99.50%) | 193/199 (96.98%) | 196/199 (98.49%) |
| Stanford 36P | 199 | 198/199 (99.50%) | 193/199 (96.98%) | 196/199 (98.49%) |
| BANDA | 203 | 202/203 (99.51%); U=1 | 175/203 (86.21%); U=1 | 187/203 (92.12%); U=1 |
| PDC sessions | 455 | 455/455 (100.00%) | 400/455 (87.91%) | 434/455 (95.38%) |
| PDC people, at least one passing session | 218 | 218/218 (100.00%) | 195/218 (89.45%) | 210/218 (96.33%) |

HCP3T both tasks requires the same person to pass both EMOTION and WM; it is not
288 independent participants. Stanford branches use the same original motion
trace. PDC person counts describe availability of at least one passing session;
they neither choose a session nor define a complete longitudinal cohort.

## Relation to hippocampal surface QC

The main local-QC comparison is now the [registered surface test](hipp_surface_qc.md):
surface tSNR25, functional-GM unsupported area >=20 mm2, and their intersection.
It reports all five parent rosters, current anatomical availability and true
BANDA/PDC surface projections. Earlier native-CIFTI volume-ROI pass counts are
not surface results. The numerical TSVs linked below retain the supplementary
motion comparisons; they do not define the current local surface gate.

HCP7T has 166 parent participants. Existing anatomy/validity passes 165; the
historical 10000/SD25 proxy passes 151. The exact unpacked motion paths are
declared in `manifests/hcp7t_motion_transfer_20261019.tsv`: four 900-frame
runs per participant (664 runs total), TR=1 second, matching the consumed
3600-frame volume. All motion inputs are evaluable. Within the existing
151-person reference, M05 passes 151/151 (100.00%) and R20 passes 123/151
(81.46%). These are supplementary motion outcomes; the formal reference
roster and surface-vertex QC remain unchanged.
Stanford 22P has one unassessable combined R20 entry after known motion failures
are accounted for; Stanford 36P has none. The branches remain distinct.

## Measurement and interpretation

- R20 uses `count(relative RMS>0.25)/N` for each physical source run; exactly 20%
  passes. Any failed declared run rejects the current participant/session product.
- HCP task/BANDA/PDC read `Movement_RelativeRMS.txt`. Stanford reads fMRIPrep
  `rmsd` on the original source run before XCP-D censoring, not on its analysis
  splits or already retained frames. Its first undefined motion sample is set
  to zero; the denominator is the full declared run length.
- F25 uses fMRIPrep Power FD for Stanford. HCP-style parameters use the first
  three translation columns in mm and next three rotation columns in degrees:
  `sum(abs(delta translation)) + 50*pi/180*sum(abs(delta rotation))`, with the first
  FD zero. These traces are unfiltered, one-frame differences. RMS and FD are
  separate metrics. [HCP format](https://wiki.humanconnectome.org/docs/How%20To%20Handle%20Downloaded%20HCP%20Data.html)
- The tSNR alternatives retain their measurement labels: HCP7T historical
  10000/SD proxy, HCP3T task-model residual tSNR, Stanford matched baseline/rest
  measurements, and BANDA/PDC cleaned-run mean/sample-SD. Retention comparisons
  describe selection; they do not rank the intrinsic quality of five protocols.

The measured distinction is concrete: BANDA028 passes mean RMS0.5 (0.3420 mm),
but fails R20 because its worst run has 47.80% of frames above RMS0.25. Its peak
is 13.3658 mm. PDC s0110s01 has a 9.7194-mm peak but only 7.32% above RMS0.25 in
its worst run and passes R20. The published burden rule allows brief peaks;
it is a run-exclusion screen alongside existing denoising, not a claim that
every passing frame is free of artifact. No additional peak cutoff was invented.

The comparison contains 3,231 physical motion runs (664 HCP7T rest, 576 HCP
task, 199 Stanford, 808 BANDA, 984 PDC), with 1,510 original participant/session-by-branch entries.
Adding the two-task intersection and PDC person aggregation yields 1,872 table
entries, not 1,872 independent people. Production masks, rosters, models and
results were not changed; no archive was accessed. The
[Kember-inspired surface coverage test](hipp_surface_qc.md) measures native
folded unsupported area with an explicit interpolation/hemisphere convention;
20 mm2 remains distinct from the existing <20% invalid-vertex rule.

## Outputs and rerun

- [All schemes, totals, passes, failures and unassessable counts](../tmp/hippocampal_qc/outputs/qc_five_datasets_summary.tsv)
- [Individual decisions and measured criteria](../tmp/hippocampal_qc/outputs/qc_five_datasets_subjects.tsv)
- [Original-run motion measurements](../tmp/hippocampal_qc/outputs/qc_five_datasets_motion_runs.tsv)
- [Analysis script](../tmp/hippocampal_qc/scripts/qc_five_datasets.py)

The script refreshes all TSV tables from exact declared inputs and completed
signal measurements. To refresh HCP7T motion only while retaining other cohort
measurements, run:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \
/home/yojiao/.conda/envs/py314/bin/python -c \
'import sys; sys.path.insert(0, "tmp/hippocampal_qc/scripts"); from qc_five_datasets import update_hcp7t_motion; update_hcp7t_motion()'
```

To recalculate the full comparison:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \
/home/yojiao/.conda/envs/py314/bin/python -u tmp/hippocampal_qc/scripts/qc_five_datasets.py
```
