# NIfTI Hippocampal Coverage Implementation Plan

**Goal:** Define hippocampal functional coverage from the individual HippUnfold
surface, its explicit registration, and the actually sampled BOLD NIfTI, without
a CIFTI hippocampal-membership requirement.

**Spec:** The user's current request and
`docs/hipp_parcellation_network/fmri_qc.md`, updated in this change.

**Architecture:** Keep the existing registered volume-to-surface sampler. Record
NIfTI spatial support at the consumed vertices and assess finite, nonconstant
sampled signals separately within each declared physical run. Reuse this same
validity calculation for participant QC and smoothing masks. Cortex CIFTI inputs
remain a separate analysis interface.

## Constraints

- Work in the current checkout and preserve the existing runtime changes.
- Use py314, one numerical thread, repository `tmp/`, and no parallel tests.
- Keep mean Power FD <=0.50 mm, bilateral Dice >=0.70 and at most 20 unusable
  vertices per actual 512-vertex hemisphere. Do not infer new retention counts.
- Preserve exact BOLD/run declarations; missing or mismatched inputs fail closed.
- Reject caches using the preceding CIFTI coverage policy; do not add a fallback.
- Do not require every interpolation neighbour to be valid, add a tissue
  segmentation package, or interpret finite variable signals as tissue purity.
- No production-tree replacement, dataset rerun, deletion, dependency change,
  commit or additional verification framework is part of this change.

## Implementation

1. Add a regression showing that HCP raw preparation succeeds with a declared
   BOLD NIfTI and physical-run metadata, without a CIFTI file. Cover a run that
   is constant despite variation in the concatenation and out-of-volume vertices.
2. Replace the CIFTI support helper in `lib/surface_sampling.py` with NIfTI
   spatial support and shared runwise vertex validity. Change `02_01` raw input
   selection to consume exact BOLD declarations only. Preserve the separate
   task/cortex interfaces.
3. Update `02_01` mask/cache/smoothing consumers and `02_02` participant QC to
   use the same NIfTI support plus current sampled-signal validity. Keep existing
   report filenames and the numerical threshold, while replacing the obsolete
   GM-specific coverage metadata and diagnostics.
4. Update existing QC, cache, ownership and run-manifest regressions, then run
   the relevant tests and repository suite. Test malformed run declarations,
   stale CIFTI-policy caches, changed raw signals, bilateral failures, and mask
   propagation to smoothing.
5. Update the active QC policy, core-workflow README, registry and rerun
   instructions in place. Clearly separate historical counts from a future
   NIfTI reassessment in scientific documentation.
6. Discuss the implemented algorithm with pi Argon using public methodological
   material only; obtain a fresh local code review and address material findings.

## Review focus

- NIfTI-only raw preparation must not inspect a missing CIFTI declaration.
- A varying concatenation must not rescue a constant physical run.
- QC counts and smoothing masks must agree on unusable vertices.
- The previous coverage policy must not resume as the new policy.
- NIfTI support and the sampler must use the same world-to-voxel geometry;
  neither a file format nor finite signal establishes a gray-matter label.

## Progress

- Implemented NIfTI-only raw input selection, geometric support, shared runwise
  signal validity, current-input QC counts, mask/smoothing propagation and
  rejection of the preceding CIFTI-policy caches. Active docs are synchronized.
- The NIfTI-only regression first failed on the CIFTI input requirement and
  then passed with real Workbench sampling. The focused QC/input/cache suite
  passed 102 tests before the additional manifest variant was included.
- Final repository suite: 1156 passed, 8 skipped, 13 failed. All failures are
  outside the QC/core-workflow tests: missing result/manifests or old
  visualization path expectations. Full failures and exceptions are in
  `tmp/qc_nifti_full_suite.log`. No output trees were rebuilt to satisfy tests.
- The fresh local source review found no material issue. Pi Argon
  (`google-vertex/gemini-4-argon`) agreed that NIfTI grid support plus actual
  per-run sampled-signal validity is a defensible minimal availability rule,
  without a CIFTI anatomical veto or an all-neighbours-valid requirement.
  The discussion used method descriptions only, without repository source,
  private data paths, participant identifiers or project statistics. Its
  response is `tmp/qc_nifti_argon_implementation_reply.txt`.
- This criterion does not establish gray-matter purity, adequate BOLD
  sensitivity or correct registration. Existing numerical cutoffs remain
  unchanged; no production rerun or new retention counts are claimed.

### Repository-suite failures outside this change

- `tests/exploratory_beta/test_multidataset_cluster_plots.py::test_multidataset_cluster_stacked_plots_run`
- `tests/exploratory_beta/test_multidataset_cluster_plots.py::test_render_multidataset_workbench_hippocampus_overlay_runs`
- `tests/visualize/test_cortex_group_contrasts.py::test_original_source_selection_is_explicit_and_only_iowa36p`
- `tests/visualize/test_cortex_network_area_analysis.py::test_subject_record_builder_uses_every_formal_roster_for_both_methods`
- `tests/visualize/test_cortex_network_area_core.py::test_formal_rosters_have_fixed_counts_and_exclude_3003`
- `tests/visualize/test_multidataset_grid_architecture.py::test_preflight_readiness_audit`
- `tests/visualize/test_multidataset_grid_architecture.py::test_7net_grid_spec_and_cohorts_resolution`
- `tests/visualize/test_template_matching_dominant_subregions.py::test_labels_use_exact_template_matching_semantics[standard]`
- `tests/visualize/test_template_matching_dominant_subregions.py::test_labels_use_exact_template_matching_semantics[ctxreg]`
- `tests/visualize/test_template_matching_dominant_subregions.py::test_stanford_cohort_counts_are_exact`
- `tests/visualize/test_template_matching_dominant_subregions.py::test_stanford_roster_fully_covered_by_demo_and_tm`
- `tests/visualize/test_tm_group_contrasts.py::test_tm_dominant_counts_match_legacy_csv`
- `tests/visualize/test_tm_group_contrasts.py::test_tm_ctxreg_dominant_is_deterministic_and_complete`
