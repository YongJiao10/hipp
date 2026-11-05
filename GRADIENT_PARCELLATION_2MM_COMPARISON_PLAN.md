# Hippocampal 2 mm Gradient Comparison and Dual-Interpretation Overview Plan

**Status:** Planned. Stop after saving this document; implementation and scientific computation are not authorized by the latest instruction.

**Goal:** Compare the existing shifted-cosine spectral method with the proposed preprocessing and surface-constrained Ward method, retain GMM comparisons, and show Fisher-z and canonical7 OVR interpretations together in familiar overview figures.

**Architecture:** Reuse the maintained gradient, clustering, annotation, and rendering functions. A maintained comparison entry point coordinates shared gradient fits, K diagnostics, annotations, and temporary outputs. Spatial cluster boundaries remain independent of network annotation.

**Runtime:** `/home/yojiao/.conda/envs/py314/bin/python`; existing NumPy, SciPy, BrainSpace, scikit-learn, nibabel, Matplotlib, and Workbench. No dependency upgrade is planned.

## 1. Approved scientific design

- Use the complete current HCP PFM QC roster, currently 166 participants, from `outputs_hcp/inputs/kong2022/manifests/hcp_7t_hippocampus_fmri_qc_pass_t1only_latest.txt`. Do not cap the roster or substitute an older cohort.
- Use the existing Kong2022 400-ROI subject FC, subject time series, validity masks, cortical references, and current group-average surfaces. Establish the 512-vertex density from consumed surface assets.
- Use **2 mm FWHM surface smoothing exclusively**, including the gradient alignment reference. The reference is the newly fitted **2 mm left hemisphere of the same preprocessing route**. Do not read 4 mm FC or gradients for this comparison.
- Preserve Fisher-z transformation before subject averaging, the Pearson gradient kernel, coverage threshold 0.80, and seed `20260420`. Compute five diffusion components and cluster with aligned G1-G3 at their native scale; do not apply component-wise unit-SD scaling.
- Preserve the existing algorithm definitions and parameters for shifted-cosine spectral clustering, surface-constrained Ward, and full-covariance GMM.

Preprocessing routes:

| Route | FC column scaling | Sparsification |
|---|---|---|
| A | Column z-score | Input feature rows, sparsity 0.9 |
| B | None | Input feature rows, sparsity 0.9 |
| C | None | Pearson affinity rows, sparsity 0.9 |
| D | None | None |

Comparison arms:

| Arm | Purpose |
|---|---|
| A + `gradient-spectral` | Existing preprocessing and shifted-cosine clustering control |
| A + `gradient-ward` | Change clustering while holding preprocessing fixed |
| C + `gradient-spectral` | Change preprocessing while holding clustering fixed |
| C + `gradient-ward` | Proposed primary approach |
| C + `gradient-gmm` | Alternative clustering on the proposed features |
| A + `gradient-gmm` | GMM preprocessing comparison |
| B + `gradient-gmm` | GMM preprocessing comparison |
| D + `gradient-gmm` | GMM preprocessing comparison |

The existing-method control also uses the common 2 mm reference. Describe it as a matched method control, rather than an exact reproduction of historical products aligned to 4 mm.

### K computation versus figure display

- Fit **K=2,3,4,5,6** for every arm and both hemispheres: **80 full-cohort fits**.
- Feed these results into the existing overview's K stability/null and redundancy panels. Preserve their existing plotting logic and metric meanings; provide real K=2-6 data instead of changing the panels into a new selection system.
- Fix the displayed spatial maps, FC heatmaps, and OVR results at **K=3**. Do not generate an overview for each K. Do not automatically change the displayed K.
- Use 20 subject-split resamples and 20 spatial-permutation null resamples, with identical seed lists and subject splits across arms and K values. Recompute gradients on each independent half. Null permutations must move each subject's FC and validity mask together.
- Fit each half's left alignment reference using that half's subjects only. Do not use a full-cohort reference to fit independent halves, and do not align different preprocessing routes to one another.
- Preserve GMM labels exactly. Report connected components per cluster and the count and fraction outside the largest component. Render small islands as observed; do not repair, merge, hide, or automatically exclude them.
- Retain explicit full-fit and resample failure records, reasons, and successful/requested denominators. Do not silently substitute methods, parameters, or outputs.

## 2. Implementation tasks

### Task 1: Explicit alignment-reference selection

- [ ] Add `--alignment-anchor-smoothing {2mm,4mm}` to the maintained gradient producer and group-atlas entry point. This comparison explicitly selects `2mm`; other workflows retain their existing default.
- [ ] Replace hard-coded 4 mm anchor selection, FC paths, metadata, and diagnostic messages with the explicitly requested reference. Keep the reference hemisphere fixed at L.
- [ ] Pass the reference selection through full-cohort, subset/split, and null fitting. Reuse the existing component-order and sign alignment implementation.
- [ ] Include the requested reference in existing gradient/feature-summary/resume checks, rejecting missing or conflicting declarations when a 2 mm reference is requested. Use existing metadata mechanisms; add no new hash or verification framework.
- [ ] Reuse 2 mm FC and time-series inputs, but recompute gradients, partitions, and resampling results. Previously completed 2 mm products aligned to a 4 mm reference are not reusable for this run.

### Task 2: Maintained comparison driver and complete K diagnostics

- [ ] Add `scripts/core_workflow/05_07_compare_gradient_parcellations.py`, calling maintained scientific functions rather than importing or executing a pre-existing temporary driver.
- [ ] Expose output root, worker count, seed, K range, displayed K, split count, and null count; use this plan's defaults: 8 workers, seed 20260420, K=2-6, display K=3, 20 splits, and 20 nulls.
- [ ] Generate the eight declared arms and all 80 full-cohort records. Compute existing FC-profile coherence and between-cluster profile similarity diagnostics for each K.
- [ ] Populate the existing `k_selection_summary.tsv/json` format with K=2-6 rows, including `group_ari_mean`, `group_ari_se`, `ARI_null_mean`, `ARI_null_5pct`, actual resample counts, and `max_profile_similarity`. Mark K=3 as the displayed result. Do not reinterpret the existing single-K `fixed-comparison` CLI contract as a multi-K selector.
- [ ] Annotate the final K=3 labels using subject-level cluster-mean-time-series FC. Export Fisher-z and `ovr_logistic_canonical7` inputs/results through existing contracts. Preserve any existing common Top-20 export requirements without displaying Top-20 in this comparison.
- [ ] Keep OVR's largest strictly positive raw-beta assignment rule. Preserve raw/per-SD coefficient and odds-ratio units; do not rank labels by per-SD effects or imply inferential, causal, or Kember-replication claims.

### Task 3: Single-smoothing, dual-interpretation overviews

- [ ] Extend `scripts/core_workflow/06_01_build_group_overview.py` with an explicitly selected dual-interpretation layout while retaining the established dual-smoothing layout for other callers.
- [ ] Load only the declared 2 mm L/R result manifests for this layout; adapt preflight and source metadata to the two-source grid rather than requiring four 2/4 mm sources.
- [ ] Replace the old 2 mm/4 mm rows with **2 mm Fisher-z / 2 mm canonical7 OVR** rows. Retain native-surface views and the familiar overview information density.
- [ ] Keep the existing K diagnostic panels supplied with K=2-6 curves and a K=3 display marker. All surface maps, annotation heatmaps, and coefficient panels remain K=3.
- [ ] Show Fisher-z connection profiles and ambiguity information in the Fisher-z row. Show raw-beta heatmaps and descriptive association panels with a zero reference in the OVR row; do not apply Fisher-z confidence thresholds to coefficients.
- [ ] Use the same spatial cluster labels for both rows. Keep viewpoints, network colors, figure sizes, and comparable numerical color scales consistent across arms.
- [ ] Render eight K=3 overviews, one per arm. Do not render WTA, family17, Top-20, or RBF overview products in this run.
- [ ] Add `method_comparison_k3.png` for the five main controls and `gmm_fragmentation_k3.png` for GMM routes A-D. Use original spatial-cluster identities and explicit component/island counts so shared network colors cannot hide boundaries.

### Task 4: Parallel execution and temporary deliverables

- [ ] Write outputs under `tmp/gradient_parcellation_comparison/anchor_2mm/`, keeping preprocessing/method distinctions explicit through existing output conventions.
- [ ] Use up to eight numerical worker processes, accounting for the coordinator and helper processes so the total remains at most ten. Set BLAS/OpenMP to one thread per worker; use no nested pools.
- [ ] Organize resampling work by preprocessing route and seed. Fit each half's gradients once and share them across that route's methods and K values. Do not recompute gradients separately for each K.
- [ ] Finish the numerical pool before serial Workbench rendering, avoiding overlap between numerical workers and rendering subprocesses.
- [ ] Provide an English README linking the main comparison, eight overview figures, and GMM fragmentation figure. Report before/after mechanisms, observed results, K diagnostics, and professional recommendations separately; do not invent a combined score or select a winner by expected network names or balanced parcel sizes.
- [ ] Update matching English workflow documentation in the same change. Preserve the completed `GRADIENT_PARCELLATION_REVISION_PLAN.md` and unrelated working-tree changes.

## 3. Verification and acceptance

- [ ] Add regression tests before implementation: explicit 2 mm anchor selection; full/split/null propagation; rejection of old-reference reuse; and unchanged default behavior for other callers.
- [ ] Verify that this comparison consumes no 4 mm scientific inputs and that independent halves use only their own subjects for their references.
- [ ] Verify the eight-arm matrix, K=2-6 rows, shared seeds/splits, deterministic serial/parallel results, and the bounded worker configuration.
- [ ] Verify that the overview K panels receive exactly K=2-6 while final labels and displayed annotations remain K=3.
- [ ] Verify that Fisher-z/OVR annotation leaves cluster boundaries unchanged, canonical7 class/network order remains correct, and GMM islands are retained and reported accurately.
- [ ] Run focused and affected workflow/visualization tests with the explicit py314 interpreter, directing temporary test artifacts beneath the repository's `tmp/` directory.
- [ ] Inspect all eight overviews and both comparison figures for complete curves, readable text, consistent views/colors, correct units, and absent clipping. Keep failed settings visible in the report rather than fabricating missing figures.

## 4. Execution boundaries

- Formal production result trees remain read-only. This plan does not authorize production migration, replacement, cleanup, default-method changes, or deletion of existing research outputs.
- Do not add dependencies, hashes, receipts, or persistent coordination artifacts. Use the requested plan document and existing scientific result metadata.
- This document is the approved handoff. The user's latest instruction is to **save the plan and stop**; do not implement or launch computation until a subsequent execution request.
