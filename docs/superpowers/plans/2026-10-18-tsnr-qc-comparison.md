# Five-dataset QC comparison plan

The user authorized discussion with pi's default Argon model, agreement on a
practical published rule, and execution on HCP7T rest, HCP3T task, Stanford,
BANDA and PDC. Iowa7T and JJ_TMS are outside scope.

## Decision and inputs

The primary target is hippocampal surface vertices. The main local comparison
retains vertexwise T25 and the directly relevant Kember2026 surface-mapping
coverage criterion (>=20 mm² improperly interpolated area), alongside existing
anatomical prerequisites. Mean relative RMS<=0.5 mm is the retained
supplementary whole-scan motion candidate, following Savage2024. Per-run
R20 is excluded from further consideration at the user's request; completed
R20/FD results remain research comparisons. The completed motion counts
do not establish local surface signal quality. No retention target is used to
tune thresholds, and no new surface-area result is claimed.

- Fixed parents: HCP7T166; HCP3T144 per task; Stanford199; BANDA203;
  PDC455 sessions/218 people. Branches are not independent populations.
- HCP task/BANDA/PDC use native Movement_RelativeRMS. Stanford uses original
  fMRIPrep rmsd before XCP-D censoring; first undefined motion sample is zero.
- BANDA/PDC signal comparisons use explicitly labelled native CIFTI volume ROIs;
  den-512 functional surfaces are not inferred from these results.
- HCP7T motion uses the exact unpacked paths in
  `manifests/hcp7t_motion_transfer_20261019.tsv`: 166 participants, four 900-frame
  runs each at TR=1 second. Mean RMS0.5 passes 166/166; R20 passes 134/166.
  Preserve existing surface/tSNR measurements and the formal reference roster.

## Execution

- [x] Compare primary sources and resolve the actual metric/cutoff/proportion.
- [x] Compare published motion alternatives; retain mean RMS0.5 as the motion
  candidate and exclude per-run R20 under the current user instruction.
- [x] Read all 3,231 available physical motion runs using explicit py314 and
  numerical threads=1 on IDAS; preserve the 10-process cap.
- [x] Reuse completed signal measurements and refresh BANDA/PDC anatomy only.
- [x] Calculate motion-only and combined schemes, plus both-task intersection
  and explicitly labelled PDC any-session person counts.
- [x] Write individual/run/summary tables and the current English report.
- [x] Directly check aggregation, authoritative parent membership, original-trace
  examples and document consistency before reporting completion.

## Outputs and boundary

The motion/signal measurement report is `docs/qc_five_datasets.md`; its script and three TSV tables
share the `qc_five_datasets` prefix. These are research comparisons. Production
masks, rosters, denoising and formal results are unchanged. No archive access,
deletions, hashes, new QC-method workflow or pipeline rebuild is needed.
