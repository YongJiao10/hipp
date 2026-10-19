# BANDA/PDC native-CIFTI volume-ROI comparison

Volume-ROI measurements from 18 October 2026. The main current comparison is
the [five-dataset registered surface test](hipp_surface_qc.md), which explicitly
projects BANDA/PDC functional GM to their individual midthickness vertices.
The measurements below remain a distinct volume-ROI analysis and must not be
used as surface-vertex pass counts. The retained supplementary motion candidate
is mean relative RMS<=0.5 mm; the user excluded per-run R20 from consideration.

## Inputs and representation

BANDA has 203 parent participants; 202 have 808 declared independent rest runs.
BANDA199 has no independent-run declaration; the concatenation is deliberately
not substituted. PDC has 455 sessions, 984 runs and 218 people. Run declarations
are `outputs_banda/inputs/banda/kong2022/manifests/banda_cortex_inputs.json` and
`outputs_pdc/inputs/pdc/kong2022/manifests/pdc_cortex_inputs.json`; the explicit
PDC `session_mapping.tsv` supplies person IDs.

The completed signal measurements use native cleaned-CIFTI hippocampal volume
ROIs: 764 left and 795 right voxels. These baseline-preserving signals support
runwise `mean(cleaned signal)/SD(cleaned signal, ddof=1)`. They are distinct from
den-512 surface samples and the earlier baseline/detrended rest definition.
The available declarations contain no matching BOLD time-series volumes or
completed den-512 functional samples. No new sampling or preprocessing was run.

Signal_V intersects finite/nonconstant masks across runs. Signal_T25 additionally
requires runwise tSNR>=25. Both require <20% invalid ROI voxels in each hemisphere;
this fraction is an exploratory native-ROI adaptation of the existing project
rule. Combined V adds bilateral Dice; V+R20 adds the published motion screen.

## Executed results

Percentages use fixed parent denominators; U means unassessable. With U>0 these
are confirmed retention percentages, not final quality pass-rate estimates.

| Criterion | BANDA participants | PDC sessions | PDC people, any passing session |
|---|---|---|---|
| Dice + signal validity | 159/203 (78.33%); U=44 | 156/455 (34.29%); U=299 | 83/218 (38.07%); U=135 |
| Motion: mean RMS<=0.5 | 202/203 (99.51%); U=1 | 455/455 (100.00%) | 218/218 (100.00%) |
| Completed R20 (excluded) | 175/203 (86.21%); U=1 | 400/455 (87.91%) | 195/218 (89.45%) |
| Motion: secondary F25 | 187/203 (92.12%); U=1 | 434/455 (95.38%) | 210/218 (96.33%) |
| Dice + signal validity + R20 | 138/203 (67.98%); U=38 | 136/455 (29.89%); U=264 | 74/218 (33.94%); U=121 |
| Dice + tSNR25 alternative | 11/203 (5.42%); U=2 | 28/455 (6.15%); U=65 | 18/218 (8.26%); U=38 |

Signal-only validity passes all 202 evaluable BANDA participants and all 455 PDC
sessions. Signal-only tSNR25 passes 12/202 (5.94%) and 93/455 (20.44%), respectively;
56/218 PDC people have at least one tSNR25-passing session. Current bilateral
Dice is available for 159 BANDA participants and 156 PDC sessions, all passing.
Missing anatomy is pending processing, not a failed reconstruction. Person-level
any-session counts describe availability; they do not select a session or define
a longitudinal analysis cohort.

BANDA028 illustrates the effect: mean RMS0.3420 mm passes the old 0.5 gate, but
47.80% of its worst run exceeds RMS0.25, so R20 fails it. Its peak is 13.3658 mm.
PDC s0110s01 passes R20 despite a 9.7194-mm peak because its worst-run exceedance
is 7.32%; a burden rule permits brief peaks. No extra peak cutoff was introduced.

The original 1,792 CIFTI runs had no signal/RMS read or frame-count error. BANDA
runs have 410 frames at TR 0.8 s; PDC has 983 runs of 478 frames and one of 418, also
TR 0.8 s. Current motion tests read the original per-frame traces before any
analysis exclusion, without censoring the delivered time series. The native-ROI tSNR result
does not settle the appropriate local surface-vertex signal criterion.

## Outputs

Use [the five-dataset summary](qc_five_datasets_summary.tsv),
[individual decisions](qc_five_datasets_subjects.tsv), and
[runwise motion measurements](qc_five_datasets_motion_runs.tsv) for refreshed
anatomy and the new R20/F25 decisions. The original
[runwise native signal measurements](banda_pdc_qc_runs.tsv) and
[signal analysis script](banda_pdc_qc.py) retain the native-CIFTI calculation.
Production masks, rosters and results are unchanged.
