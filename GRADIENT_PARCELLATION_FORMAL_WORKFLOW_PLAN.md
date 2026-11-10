# Formal Hippocampal Gradient Parcellation Workflow Plan

**Status:** Ready for execution; implementation and analysis have not started.

**Scope:** HCP PFM canonical7 workflow only. This plan does not change other datasets, atlas families, cortex routes, or existing historical results.

## 1. Purpose and decisions

The earlier comparison used four preprocessing/affinity routes (A–D) with several clustering methods. The formal workflow will retain two routes whose roles are easy to distinguish and keep one additional clustering method strictly for visual comparison:

| Formal name | Earlier label | Feature preparation and affinity | Clustering | Role |
|---|---|---|---|---|
| **TopFC** | B | Use the existing vertex-to-network FC features without column scaling; sparsify each feature row to its largest 10% of entries (`sparsity=0.90`) before affinity construction. | Shifted-cosine spectral clustering | Historical-method baseline; the primary method can later be switched back to this route through the formal selection/index, without rebuilding unrelated routes. |
| **TopSim** | C | Do not column-scale. Build the complete Pearson-correlation affinity first, then retain each row's largest 10% of affinities (`sparsity=0.90`). | Ward clustering with the existing binary connected-neighbor constraint | Current primary method. |
| **TopSim GMM** | C + GMM comparison | Same TopSim features and affinity as TopSim. | Gaussian mixture model | Comparison display only; it is not a formal primary candidate. |

Routes A and D are retired from the active formal workflow. Keep existing comparison results and research history in place; this plan does not delete or relocate them. Do not create new numbered workflow scripts for this work. Continue the current unnumbered library/entry-point structure and retain the existing formal task script `06_11_render_task_activation_products.py`.

Use **2 mm and 4 mm as separate analyses and separate figures**. Both smoothing levels use the same 2 mm left-hemisphere alignment reference, so any difference between their results is not caused by changing the alignment anchor.

Keep the native three gradient coordinates (G1–G3), five fitted components, seed `20260420`, minimum subject coverage `0.80`, and displayed `K=3`. Evaluate every `K=2,3,4,5,6,7,8`; do not reduce the analysis to K=3 alone. Calculate the redundancy diagnostic at threshold `0.90`, while keeping subject coverage at `0.80`.

## 2. Formal products and naming

Keep the formal workflow under the existing HCP PFM routes:

```text
outputs_hcp/work/hippomaps_gradients/pfm/
  topfc/{2mm,4mm}/hemi_{L,R}/
  topsim/{2mm,4mm}/hemi_{L,R}/

outputs_hcp/work/group_atlas/pfm/
  topfc/gradient-spectral/kong2022/{2mm,4mm}/cov_0p80/g1g2g3/hemi_{L,R}/
  topsim/gradient-ward/kong2022/{2mm,4mm}/cov_0p80/g1g2g3/hemi_{L,R}/
  topsim/gradient-gmm/kong2022/{2mm,4mm}/cov_0p80/g1g2g3/hemi_{L,R}/
  wta-top/kong2022/{2mm,4mm}/cov_0p80/hemi_{L,R}/

outputs_hcp/present/overviews/pfm/
  topfc/gradient-spectral/{2mm,4mm}/
  topsim/gradient-ward/{2mm,4mm}/
  topsim/gradient-gmm/{2mm,4mm}/
  wta-top/{2mm,4mm}/
```

The three clustering overviews in each method/smoothing directory are:

```text
roi-fisherz_ovr_overview.png
network-ts-fisherz_ovr_overview.png
top20_ovr_overview.png
```

Each WTA directory contains `wta_overview.png`. Render it from the original overview plotting source and preserve the original WTA content and definitions. It must remain a whole-hippocampus overview with the count panel, left/right long-axis bin curves, and native-surface maps; do not replace it with a hippocampus-only subset or a newly invented WTA plot.

There are 18 clustering overview figures (3 methods × 2 smoothing levels × 3 annotation pairs) and 2 WTA figures (one per smoothing level), for **20 overview figures** total. Formal overview and matrix selection must identify method, profile, smoothing, annotation/featuresource, hemisphere, and K explicitly. The primary formal selection is TopSim + Ward; TopFC + spectral remains the named baseline. Changing the primary selection later should change the selected index/default, not silently relabel or overwrite the scientific outputs.

## 3. Cohort, inputs, and gradient generation

Use the current authoritative HCP PFM canonical7 T1-only QC roster:

```text
outputs_hcp/inputs/kong2022/manifests/hcp_7t_hippocampus_fmri_qc_pass_t1only_latest.txt
```

The roster currently contains 166 subjects. Resolve each subject's FC, hippocampal time series, validity mask, ROI reference, and group surface from the declared input paths and preserve the existing exact-artifact selection rules. Do not infer a substitute input from a directory glob or select an older artifact if the required path/metadata is missing or mismatched.

Generate full-cohort gradients for TopFC and TopSim at both smoothing levels and both hemispheres (eight gradient combinations). Keep the three native coordinates G1–G3 and existing alignment/canonicalization behavior. Record the route's feature-preparation settings with the output metadata, including whether sparsification was applied before or after affinity creation. Use the same 2 mm left-hemisphere alignment reference for both 2 mm and 4 mm products.

## 4. Clustering and K evaluation

Run these three formal comparison combinations:

1. TopFC + shifted-cosine spectral clustering.
2. TopSim + connected Ward clustering.
3. TopSim + GMM.

Keep each method's existing algorithm definition and randomization settings. In particular, retain shifted-cosine spectral's existing initialization count and affinity implementation; retain Ward's binary connected-neighbor constraint; retain GMM's full-covariance configuration (`n_init=10`, `max_iter=1000`, `tol=1e-3`, `reg_covar=1e-6`). Do not make algorithm-specific settings drift between smoothing levels or hemispheres.

For every method × smoothing × hemisphere setting, compute and save one full-cohort result for each K from 2 through 8. This is **84 setting/K records** (3 methods × 2 smoothing levels × 2 hemispheres × 7 K values). Keep `K=3` as the displayed/default result, while retaining the automatic selection result as a separate diagnostic field; do not let an empty automatic-survivor pool prevent a valid K=3 result from being displayed.

### Split and null stability

For each of the 84 setting/K records, run 20 subject-split repetitions and 20 null repetitions: **1,680 split attempts and 1,680 null attempts** overall. Record each repetition's outcome. A scientifically invalid or failed repetition remains an explicit failure/NaN with its reason; do not encode it as a score of zero.

- Split the 166 subjects into shuffled halves of 83/83. Use deterministic repetition seeds `20260420 + 1 + 13*i`, for `i=0..19`; derive the two half-fit seeds from the split seed and half index.
- For each null repetition, independently permute each subject's FC feature rows while applying the same row permutation to that subject's feature mask. Use deterministic seeds `20260420 + 5000 + 17*i`, for `i=0..19`.
- Use the same repetition schedule for all profiles, methods, smoothing levels, and hemispheres so comparisons are paired. Fit each split/null half using the same 2 mm left-hemisphere alignment reference.
- Reuse the corresponding split/null preparation across K values; the K-specific fitting remains separate. Do not run nested process pools. Provide a bounded resampling worker setting (default 8) and keep per-K inner jobs serial (`jobs=1`), with total active work within the repository's IDAS cap of 16.
- Preserve the actual null distribution and its 5th percentile. Do not silently discard invalid attempts or report a fabricated fallback value.

Keep the existing FC-based cluster-profile coherence diagnostic and gradient-coordinate dispersion as separate measures. The FC-coherence calculation must consume the original FC feature bank, not gradient coordinates. Compute the cluster-redundancy diagnostic using the existing subjectwise vertex-FC profile mechanism, and set its threshold to `0.90`; do not substitute the network-time-series annotation inputs for this diagnostic.

## 5. Annotation and network-profile branches

For each clustering result, build three side-by-side annotation pairs. The cluster labels are shared; only the assignment/profile interpretation changes.

### A. ROI-equal Fisher-z + standard ROI OVR

- For each subject and cluster, average hippocampal time series over the cluster's valid vertices.
- Correlate each cluster mean time series with each of the 400 cortical ROI time series; Fisher-z transform the correlations.
- Average ROI Fisher-z values equally within each canonical7 network to create the network profile.
- Fit the standard canonical7 OVR interpretation from the 400-ROI subjectwise Fisher-z values, preserving the current logistic-regression settings and current label/assignment contract.

### B. Network-time-series Fisher-z + matching network-time-series OVR

- Construct each cortical network time series from its constituent ROI time series using the existing `n_grayordinates_used` weights, equivalent to averaging the valid cortical vertices represented by those ROIs.
- Correlate each subject/cluster hippocampal mean time series directly with each of the seven network time series and Fisher-z transform. This is a time-series calculation, not an average of already-computed ROI correlations.
- Fit a matching canonical7 OVR branch from these subjectwise cluster-to-network Fisher-z features. Its input must be the same seven-network feature table used by the network-time-series Fisher-z profile.
- Give this OVR product a distinct, explicit feature-source identity and output prefix, such as `network_ovr_canonical7_network_ts_*`, so it cannot overwrite the standard ROI OVR output.

### C. Normalized Top20 + standard ROI OVR

- Retain the current normalized Top20 assignment calculation from the subjectwise ROI Fisher-z features and the existing distinction between the Top20 vote calculation and the supplementary FDR-selected positive-count fields.
- Pair this profile with the same standard ROI OVR used in branch A; do not recompute or rename the OVR merely because the profile panel shows normalized Top20.

Use explicit registries/metadata to map each annotation variant to its feature source and output names. Do not infer the source by parsing or stripping a variant name. Preserve the current ROI-based OVR result names and coefficients. Add the new network-time-series OVR as a separate declared variant. Keep Fisher-z profiles, normalized Top20 assignments, standard ROI OVR, weighted/network-time-series Fisher-z, and network-time-series OVR as explicit coexisting products in each run; do not copy entire gradient trees to create the extra annotation branch.

The existing OVR classifier settings remain: canonical7 labels, current logistic-regression solver/regularization/tolerance/iteration/seed settings, no feature scaling or class weighting, and positive-beta winner rules. Apply the existing class-order and unassigned-class contract to both feature sources. Preserve the full signed coefficient/heatmap data even when a displayed association panel shows only positive betas.

## 6. Overview plotting requirements

Update the existing source plotting functions and CLI so each requested smoothing and annotation pair is rendered as its own overview. The existing plotting source remains authoritative for WTA. Retain existing behavior for callers that do not request the new single-smoothing/method/annotation selectors.

### Clustering overview content

Each clustering overview has two method-matched interpretation rows and four logical columns:

1. Fisher-z or normalized-Top20 cluster-assignment heatmap.
2. Cluster/ambiguity display for the corresponding assignment.
3. The corresponding OVR coefficient heatmap.
4. OVR winner/runner-up positive-beta association bars.

Include the existing quality diagnostics for ARI, cluster redundancy, and the existing confidence/ambiguity summaries in the overview's diagnostic rows. The top and lower interpretation rows each repeat the aligned diagnostic panels so the Fisher/Top20 assignment can be compared directly with its matching OVR interpretation. Keep method, smoothing, profile, annotation pair, K, and hemisphere clear in titles or captions.

### OVR association display

Replace the current vertical maximum-beta-only chart with the horizontal winner/runner-up style used by the assignment cluster/ambiguity panel:

- Display the two largest strictly positive raw OVR beta values, with the winning network filled and the runner-up outlined, using the existing canonical7 colors.
- Label the network(s) and show the beta difference when two positive values exist.
- If exactly one network has positive beta, show one bar and one label only.
- If no network has positive beta, preserve the cluster row and show a clear `No positive β` state.
- Keep raw beta values as beta values; do not apply the Fisher-z ambiguity margin to them.
- Keep the full signed-beta heatmap unchanged so the compact association panel does not hide negative coefficients.

### Axes, ranges, and readable layout

- Complete the K axis at `K=2..8` wherever K is shown.
- Plot redundancy with y-axis lower bound `0.50` (not zero), upper bound `1.02`, labeled ticks from `0.50` to `1.00`, and a horizontal threshold line at `0.90` read from result metadata.
- Keep subject coverage at `0.80`; the `0.90` value changes only the redundancy diagnostic threshold.
- Show split stability as mean and standard error and the null reference as the 5th percentile. Label the null statistic explicitly.
- Use comparable Fisher-z color ranges across the FC-derived profiles and methods; use separate shared ranges for the two OVR feature sources. Keep normalized Top20 on its defined 0–1 range. Keep the bar-panel range consistent with its corresponding shared coefficient range.
- Make the first two diagnostic columns equal width across all four diagnostic rows. Do not let colorbars shrink one panel's axes independently. Use a dedicated colorbar column or another layout that reserves the same space for all comparable panels.
- Use the same plotting/layout helper for the 12-panel and 32-panel variants. The rendered data axes must not change width or height merely because the figure has a different number of clusters; use a shared grid specification and consistent cell sizing.
- Start from a readable canvas near 30 × 24 inches at 300 dpi, with approximately 24 pt title, 17 pt panel headings, 15 pt axis labels, 13 pt ticks, and 11 pt annotations/legend text. Adjust the canvas and spacing to prevent overlap; do not shrink fonts to compensate for an undersized canvas.
- After drawing, compare corresponding data-axis bounding boxes. The first two columns' axes should match within 1% in width across rows; paired upper/lower heatmap axes should match within 1% in height after colorbar placement. If labels or legends collide, expand the canvas or adjust spacing while preserving those shared dimensions.

### WTA overview

Render the original WTA overview separately for 2 mm and 4 mm using its source implementation. Preserve all original overview panels and data definitions: whole-hippocampus winner counts, both hemispheres' long-axis curves, and the native-surface display. Do not add cluster-method panels to WTA, and do not use WTA to replace any clustering annotation branch.

## 7. Indexing, matrix integration, and documentation

Extend the existing formal group-matrix/index path to discover the HCP PFM products by explicit profile, clustering method, smoothing level, annotation/featuresource, hemisphere, and K. The selected/default display should point to TopSim + Ward at K=3, while the TopFC + spectral baseline and TopSim + GMM comparison remain addressable. WTA stays a separate clusterless product.

Do not alter default behavior for non-HCP datasets or existing intrinsic/network/spectral/WTA routes. Keep the selected identity in existing result metadata and formal index fields; do not add a new hash, manifest, compatibility layer, migration system, or copied result-root solely for selection.

Update the existing core overview and workflow/step registry documentation in the same implementation. The English documentation must describe the active profiles, roles, inputs, method settings, smoothing, K range, annotation branches, output locations, and commands needed to render the figures. It must be self-contained and must not depend on one-off scripts, reports, previews, or logs in `tmp/`.

## 8. Implementation sequence

1. Add a small maintained profile-definition module for TopFC/TopSim, their preprocessing order, method roles, and primary selection. Use it in the existing gradient and clustering entry points rather than duplicating route logic.
2. Extend the existing result metadata and output-path/index resolution for the two named profiles and three method combinations. Preserve current behavior outside HCP PFM.
3. Refactor the split/null resampling preparation so each setting's data preparation is shared across K, while each K is fitted and recorded separately. Correct empty/invalid repetition reporting to NaN/status plus reason.
4. Add the direct network-time-series Fisher-z and matching OVR annotation variant, with explicit feature-source metadata and non-colliding output names. Preserve the standard ROI branch and the normalized Top20 branch.
5. Add method/smoothing/annotation selectors to the overview entry point, implement the consistent shared-grid layout and positive-beta association panel, and render the original WTA source once per smoothing.
6. Extend formal overview/matrix discovery and update current English workflow documentation.
7. Generate the complete planned formal products and review all 20 overview figures at full resolution. Fix any clipping, overlap, mismatched panel dimensions, missing K, or incorrect source label before calling the run complete.

## 9. Acceptance criteria

The implementation and run are complete when all of the following hold:

- Formal profiles and roles resolve unambiguously as TopFC baseline, TopSim + Ward primary, and TopSim + GMM display comparison; A and D are absent from active formal selection.
- Both smoothing levels have separate gradient, atlas, and overview products and share the 2 mm left alignment reference.
- All 84 method/smoothing/hemisphere/K records (`K=2..8`) are present or have an explicit recorded scientific failure. No K is silently omitted.
- Each record reports its 20 split and 20 null outcomes; invalid outcomes are not represented as zero.
- The K=3 overview result is available for every method/smoothing/hemisphere combination when the fit itself is valid, independent of the separate automatic-selection diagnostic.
- The three annotation pairs are built from the specified sources. The network-time-series Fisher-z profile and its OVR use the same seven-network time-series feature table; the ROI OVR remains separately available and unchanged.
- Redundancy uses the 0.90 threshold, its plot begins at 0.50, and coverage remains 0.80.
- There are 18 clustering overviews and 2 original-source WTA overviews. The WTA panels retain the original whole-hippocampus curves, counts, and surface content.
- OVR association panels correctly show two, one, or zero positive-beta bars without dropping cluster rows; all overview panels remain legible and aligned at full resolution.
- Formal indexing resolves the requested product explicitly, and non-HCP behavior remains unchanged.
- The repository's focused existing tests and the new focused tests for profile selection, resampling reuse/failure values, weighted time-series calculations, OVR source coexistence, and overview layout pass under the required `py314` interpreter.
- Required English documentation is updated alongside the implementation. Historical comparison outputs remain untouched.

## 10. Operational constraints

- Identify the execution host before launching batch work. On IDAS, keep all concurrent jobs/processes within the repository cap of 16. Use at most 8 resampling workers by default, no nested per-K pools, and count any child processes in the total.
- Use `/home/yojiao/.conda/envs/py314/bin/python` for local workflow scripts and tests.
- Do not create new numbered workflow steps/scripts for testing or implementation.
- Do not delete or clean any prior comparison results as part of this plan. Existing historical outputs remain available and unchanged.
- Keep temporary implementation artifacts under the repository's `tmp/` workspace. Formal products and maintainable tools belong in their established project locations.
- This document records the implementation plan only. No analysis or code changes are authorized by saving this plan.
