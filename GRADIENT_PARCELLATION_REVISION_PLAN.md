# Gradient Parcellation and Network Annotation Revision Plan

> **For agentic workers:** Implement this plan task by task using the `executing-plans` skill. Keep each behavior change paired with its tests and documentation update.

**Goal:** Make gradient preprocessing explicit and evaluate its numerical consequences, compare three gradient-based clustering methods under a shared input contract, and add Kember-style one-versus-rest (OVR) network annotation while preserving existing Fisher-z and normalized Top-20 assignments.

**Status:** Complete. The HCP PFM comparison, review fixes, and final affected-suite verification are complete.

**Architecture:** A shared gradient producer supports explicitly selected preprocessing settings. Within each setting, identical vertex features feed surface-constrained Ward clustering, Gaussian mixture modeling, and distance-weighted surface spectral clustering. Network assignments are computed after spatial clustering and never alter the cluster boundaries.

**Tech stack:** Existing py314 environment with NumPy, SciPy, BrainSpace, scikit-learn, nibabel, and current rendering tools. The inspected environment has scikit-learn 1.8.0 and BrainSpace 0.2.1. No package installation or environment upgrade is planned.

**Spec:** Decisions confirmed in the conversation: compare all three clustering methods before selecting a primary method; retain Fisher-z and Top-20; add Kember-style OVR using existing subject-level FC; fit canonical7 and family17 separately; provide coefficient plots and independent OVR network annotations; implement shared code and validate on HCP PFM.

## Execution Status

- **Input readiness:** The current HCP roster and functional QC record contain 166 participants. Both group-average surfaces use 512 vertices and the same roster. Existing records show 166 completed smoothing and cortical-reference stages and 664 completed subject/smoothing/hemisphere FC stages; the FC records postdate the current QC and declare a consistent 400-ROI order.
- **Implementation baseline:** The shared preprocessing, Ward/GMM/RBF clustering, OVR, and comparison interfaces are complete and verified. The relevant gradient, assignment, fixed-cluster reuse, surface-affinity, crossfit, and visualization tests passed: 266 passed with two Python import deprecation warnings.
- **Follow-up:** A resume manifest lacking `group_gradient_feature_summary` or its upstream gradient declaration, or declaring conflicting sources, could be accepted when its signature matched. The validator now rejects all three cases. Independent review also found that valid skipped results could fail on resume and that no-valid-K runs loaded gradients unnecessarily; both paths are fixed, with regression and first-run/resume integration tests. Fixed-cluster reuse now requires preprocessing declarations in both the feature summary and upstream gradient manifest, and RBF bandwidth reuse requires an explicitly declared exact parameter match. Regression tests failed before the fixes and passed afterward. The completed comparison did not use these resume or fixed-reuse paths. The final affected suite passed all 266 tests.
- **Current comparison:** The fresh full-roster run under `tmp/gradient_parcellation_comparison/current_nifti_qc/` exited successfully after all 160 candidate screens and 20 split plus 20 null resamples. All candidate records, the four assignment variants for each viable candidate, and the shared resampling seeds passed result-contract checks. The report now interprets the observed tradeoffs. Independent review identified three active-workflow issues in `05_02_run_group_atlas.py`: `strong_null_ari` permuted FC while leaving validity masks unpermuted when constructing split support, `--resume` could skip upstream gradient-manifest and feature-summary validation, and `--reuse-fixed-cluster-root` did not check clustering feature scaling or RBF bandwidth. The comparison driver already permuted FC and masks together and used isolated per-setting paths, so its results were unaffected; all three active-workflow issues are now fixed in `05_02_run_group_atlas.py` and covered by regression tests. The run used one worker and one numerical-library thread; its log is `tmp/gradient_comparison_runtime_PGjwbe/run.log`. The earlier exploratory report predates the current QC/FC inputs and is retained as prior research output. The previous temporary driver recorded retired `tsnr_gate` paths instead of its consumed validity masks; a separate temporary driver now records the current `surface/validity/` paths. A targeted check failed on the previous driver and passed on the corrected copy.
- **Formal outputs:** Read-only throughout this exploration. Comparison results and temporary work remain under the repository's `tmp/` directory.

## Global Constraints

- Compare column-wise FC z-scoring with no FC scaling; separately compare feature sparsification, affinity sparsification, and no sparsification. Retain Fisher-z transformation, subject averaging, and the Pearson kernel. The absence of column scaling in a reference tutorial does not establish that column scaling is intrinsically invalid.
- Compute five diffusion components. Compare native G1–G3 with component-wise unit-SD G1–G3 for clustering; do not conflate this choice with FC column scaling. Neither weighting is assumed universally correct.
- The initial comparison uses K=3, the complete authoritative subject roster, and identical data for all three methods.
- Keep spatial cluster identities independent of network annotation. Do not force distinct network labels or select methods to obtain expected network names or Kember-like boundaries.
- Update English project documentation whenever code changes affect behavior, paths, caches, launchers, environments, or workflow outputs.
- Use `/home/yojiao/.conda/envs/py314/bin/python` for repository Python work. Keep temporary comparison outputs under `<repository>/tmp/`.
- Keep formal result trees read-only in this change. Any later production rebuild or replacement requires an exact user-approved path list.

## Review Focus

- Cached gradients must match the explicitly requested FC scaling and affinity construction; missing declarations or mismatches must fail reuse checks. Column-scaled gradients are not intrinsically stale, but cannot serve a no-scaling request.
- Ward must reject a disconnected supported mesh before scikit-learn silently completes its connectivity graph.
- GMM output may be spatially disconnected and must report that fact without post-hoc repair.
- OVR coefficients must be mapped through the estimator's declared class order and remain tied to the correct taxonomy.
- RBF affinity must respond to differences in gradient magnitude and must not inherit duplicate triangle-edge weights.

## Review Evidence and Interpretation

The installed BrainSpace source confirms that `compute_affinity` defaults to `pre_sparsify=True`: with sparsity 0.9 it retains the largest 10% of input features per row before constructing Pearson affinity. This selection is by signed value, not absolute magnitude. With `pre_sparsify=False`, sparsification occurs after kernel construction. The installed `GradientMaps.fit` does not forward `pre_sparsify` to affinity construction; its extra keywords go to the embedding estimator. See the [BrainSpace source documentation](https://brainspace.readthedocs.io/en/latest/_modules/brainspace/gradient/kernels.html).

The local scikit-learn 1.8.0 runtime confirms that explicitly setting `penalty="l2"` emits a deprecation warning while omitting it retains default L2 without that warning. See the [1.8 API documentation](https://scikit-learn.org/1.8/modules/generated/sklearn.linear_model.LogisticRegression.html).

The external review reports severe HCP cluster imbalance and weak RBF connections. Its exact cluster counts, variances, and network percentages have not been independently reproduced in this plan review; Task 5 must reproduce and investigate them. Unequal cluster sizes, absence of a desired network label, and disagreement with anatomical thirds are not themselves evidence of invalid clustering. Large native-component variance directly affects Ward and Euclidean RBF distances. Full-covariance GMM is affine-equivariant in the ideal unregularized optimum; initialization, local optima, and finite covariance regularization can make practical fits scale-sensitive, but its behavior cannot be explained solely by Ward's variance-dominance argument.

The local Kember notebook Cell 6 exports `exp(beta)` on its supplied features. It does not establish that these inputs have unit SD across subject-by-cluster observations. Standardization at a different upstream level is not equivalent. Per-SD OR export here therefore does not establish numerical comparability with Kember Fig. 2.

---

## Task 1: Make gradient preprocessing and cache reuse explicit

**Execution status:** Complete. All 16 current full-cohort gradient manifests were checked against the requested preprocessing, complete 166-subject roster, random seed, 400-ROI input, and each setting's own 4 mm left alignment anchor; `--resume` and `--reuse-fixed-cluster-root` in `05_02_run_group_atlas.py` now also enforce upstream gradient-manifest, feature-summary, clustering-scaling, and RBF-bandwidth validation before reusing successful result artifacts. Matching skipped results retain their recorded status without requiring absent result artifacts.

**Files:**

- Modify `scripts/core_workflow/lib/hippomaps_adapter.py`.
- Modify `scripts/core_workflow/05_01_compute_hippomaps_gradients.py`, `scripts/core_workflow/05_02_run_group_atlas.py`, and individual-workflow callers of the shared gradient adapter.
- Update `tests/core_workflow/test_hippomaps_gradient_route.py`, affected individual-workflow tests, and workflow documentation.

**Interfaces:** Extend `fit_gradientmaps_pearson` with explicit keyword settings for `feature_scaling={none,col_zscore}`, `sparsify_stage={features,affinity,none}`, and sparsity; preserve its three-array return value. Expose matching caller options. Keep existing production defaults until comparison review. Do not combine FC scaling and sparsification order into a single ambiguous mode name.

- [x] Write tests for original FC in the no-scaling route, column scaling only when selected, and correct sparsification order. Run them before implementation to confirm the missing behavior.
- [x] Implement each explicit route in the shared adapter while preserving input validation and output extraction. Construct affinity with `compute_affinity` using the requested order, then use `GradientMaps(kernel=None).fit(affinity, sparsity=None)` to avoid a second kernel or sparsification pass. Do not pass `pre_sparsify` through `GradientMaps.fit(**kwargs)`.
- [x] Make nonnegative affinity handling, diagonal treatment, and symmetrization explicit and consistent across routes. Preserve BrainSpace's mean symmetrization of row-sparsified affinities, and test that the explicit pre-feature route matches the current library route. Check the resulting diffusion graph for disconnected support.
- [x] Update existing producer and subset-fit metadata with selected FC scaling, sparsification stage, effective sparsity, and affinity settings. Remove claims that column z-scoring matches the HippoMaps tutorial.
- [x] Extend gradient manifest and resume validation to reject missing or mismatched settings. Errors must identify the manifest and requested setting and require explicit regeneration; do not infer settings from filenames or silently reuse another variant.
- [x] Ensure crossfit, split-half, null, and full-cohort fits use the same declared route. Within each preprocessing setting, use newly computed PFM 4mm-left gradients as its alignment reference. Do not align distinct preprocessing settings to the existing column-scaled anchor merely to make their components agree.
- [x] Implement clustering feature scaling separately: `none` or per-component centering and division by SD (`ddof=0`) over supported vertices. Record component means and SDs; reject zero-variance components. Scale only the selected G1–G3 after alignment. Fit any held-out transform on training data only. Recompute the transform within each independent split fit.
- [x] Run the gradient-route and crossfit tests. Confirm existing Fisher-z-before-subject-averaging behavior is unchanged.

**Completion:** All active fit paths honor the selected preprocessing semantics, and artifacts with missing or mismatched settings cannot silently enter new or resumed analyses.

## Task 2: Add three explicit gradient clustering methods

**Execution status:** Complete. Ward, GMM, and RBF spectral dispatch and their focused tests are verified. All candidates have been screened. One validated route-A GMM example has five mesh components in its third cluster; the labels are retained without spatial repair, as specified. The active-workflow `strong_null_ari` validity-mask permutation mismatch has been resolved and verified by unit tests.

**Files:**

- Create `scripts/core_workflow/lib/gradient_clustering.py`.
- Modify method dispatch and clustering call sites in `scripts/core_workflow/05_02_run_group_atlas.py` and individual-workflow launchers.
- Create focused tests in `tests/core_workflow/test_gradient_clustering.py` and update `tests/core_workflow/test_surface_affinity.py` where needed.

**Interface:** Add a shared entry point accepting `method`, `features`, `adjacency`, `k`, `seed`, and `rbf_scale`; return integer cluster labels plus method-specific provenance. The feature matrix is `supported_vertices × selected_components`, already transformed using the declared clustering scaling, and adjacency uses the same vertex ordering.

Register these exact method names and primary settings:

| Method | Definition |
|---|---|
| `gradient-ward` | `AgglomerativeClustering(linkage="ward", metric="euclidean", connectivity=mesh_adjacency)` |
| `gradient-gmm` | `GaussianMixture(covariance_type="full", n_init=10, max_iter=1000, tol=1e-3, reg_covar=1e-6)` |
| `gradient-spectral-rbf` | Existing spectral solver using RBF-weighted mesh-edge affinity |

- [x] Write tests for method dispatch, K nonempty clusters, reproducible seeded fits, and method-specific failures before adding implementations.
- [x] For spatial methods, construct binary, symmetric, zero-diagonal adjacency after applying the support mask. Reject disconnected input graphs. Ensure Ward returns one connected component per cluster.
- [x] For GMM, use the same declared gradient features and do not supply mesh coordinates or adjacency. Reject nonconvergence and report each cluster's connected-component count; do not alter labels to make them contiguous. Record covariance conditioning and the effect of fixed `reg_covar` when comparing scales.
- [x] For RBF spectral clustering, use `W_ij = A_ij * exp(-||g_i-g_j||^2 / (2*sigma^2))`, with `sigma = rbf_scale * median(positive unique-undirected-mesh-edge distance)`. Use `rbf_scale=1.0` for the primary comparison, and preserve `assign_labels="kmeans"`, `n_init=50`, and `eigen_solver="arpack"`.
- [x] Construct RBF weights in float64. Fail if the scale is undefined or the positive-weight graph is disconnected. Also diagnose effective connectivity: on a separate copy, discard edges with `weight / max_weight < 1e-5`, eliminate explicit zeros, and count components. Treat disconnected effective support as an investigated failed candidate in this exploratory comparison, not a reason to silently change sigma. Keep the original affinity unchanged. This cutoff is an explicit experimental threshold, not a machine-precision constant or a universal solver guarantee.
- [x] Record edge-weight quantiles, weighted-degree minimum/median and their ratio, effective connected components, and returned cluster sizes. Flag clusters smaller than `ceil(0.03 * n_supported)` for investigation; do not force balanced sizes or automatically merge/relabel small clusters. Degree diagnostics complement connectivity because weak bridges can exist even when every vertex has a substantial degree. Threshold connectivity alone cannot guarantee stable eigensolutions; examine repeated-fit and bandwidth sensitivity in Task 5.
- [x] Keep the existing `gradient-spectral` shifted-cosine method distinct and available. Centralize method identification so only methods that build affinities enter spectral dispatch.
- [x] Route full-cohort, split-half, null, and crossfit fits through the shared dispatcher. Keep existing defaults until the comparison is reviewed.
- [x] Run the new tests plus existing surface-affinity, group-atlas, and crossfit tests.

**Completion:** Each candidate consumes the same features; each method's spatial assumptions are explicit; no method silently borrows another method's output or affinity.

## Task 3: Add subject-level OVR network analysis and annotations

**Execution status:** Complete. Both taxonomy models and their coefficient, class-order, missing-input, and fixed-label tests are verified. All 101 viable current-cohort candidates passed the existing artifact and lineage checks for Fisher-z, normalized Top-20, canonical7 OVR, and family17 OVR.

**Files:**

- Create `scripts/core_workflow/lib/network_ovr.py`.
- Modify subjectwise profile handling and assignment export in `scripts/core_workflow/05_02_run_group_atlas.py`.
- Extend `tests/core_workflow/test_gradient_assignment_contract.py` and add focused OVR tests.

**Interface:** Accept a subjectwise array shaped `subjects × clusters × 400 ROIs`, subject IDs, cluster IDs, the explicit ROI registry, and taxonomy. Return class-aligned coefficient/intercept arrays, odds ratios, and network assignment rows.

- [x] Reuse the existing `cluster_roi_profile_subjectwise(...)` output, computed from each subject's cluster-mean hippocampal time series and cortical ROI time series. Do not calculate OVR from group-mean profiles or substitute averaged vertex-level correlations.
- [x] Aggregate ROI Fisher-z profiles using authoritative `canonical7` and `family17` registries. Fit the taxonomies separately on the same cluster identities.
- [x] Flatten each subject-by-cluster-by-network table to one observation per subject and cluster. Use `OneVsRestClassifier(LogisticRegression(solver="liblinear", C=1.0, tol=1e-4, max_iter=1000, random_state=20260420))`, retaining default L2 with no class weighting or extra feature scaling. Omit the deprecated explicit `penalty` argument in scikit-learn 1.8.
- [x] Preserve current ROI aggregation semantics. Fail with subject, cluster, and network details if required network-level values are missing; do not impute or silently omit observations.
- [x] Resolve model outputs through the estimator's declared classes, including the binary two-cluster representation. Treat convergence failure as an error; do not change solver or regularization automatically.
- [x] Export the subject-level input table, beta, intercept, `exp(beta)`, class/network order, and subject IDs. Also export `feature_sd`, computed with `ddof=0` across the exact flattened fitting observations for each taxonomy, `standardized_beta = beta * feature_sd`, and `standardized_odds_ratio = exp(standardized_beta)`. State the population used for the SD. Raw OR is per one-unit Fisher-z change; standardized OR is per one SD with other features held fixed. This is a change of reporting units for the fitted model, not a refit on standardized predictors; L2-regularized refits generally differ.
- [x] State that OVR is descriptive conditional association, not out-of-sample accuracy, an inferential test, or a causal estimate. Rows from the same subject are dependent; any future uncertainty estimate must resample subjects together. Do not claim OR magnitude is directly comparable with Kember, whose features, standardization domain, and model inputs differ.
- [x] Add independent assignment variants `ovr_logistic_canonical7` and `ovr_logistic_family17`. Assign each cluster to its largest strictly positive raw coefficient; otherwise mark it unassigned. Retain the complete coefficient table even when exporting labels. Per-SD effect ranking is a separate diagnostic and may differ from raw-beta ranking; exporting it must not silently redefine these assignment variants.
- [x] Preserve the meaning of `--assignment-variant both` as Fisher-z plus normalized Top-20. Require explicit selection of OVR variants.
- [x] Test coefficient alignment against direct scikit-learn fits, both odds-ratio conversions and SD population, absence of the deprecated-penalty warning, taxonomy separation, missing-feature errors, unassigned rows, and invariance of cluster labels when annotation variants change.

Document this as Kember-style OVR using this project's existing network-level FC features. It does not reproduce Kember's cortical-vertex standardization, individualized hippocampal mapping, or exact Yeo network inputs.

**Completion:** Existing and new assignments annotate the same clusters; both taxonomy models have inspectable inputs and coefficients.

## Task 4: Integrate exports, consumers, and comparison mode

**Execution status:** Complete. The integration, fixed-cluster reuse, and renderer tests passed in the 266-test run. The active group-overview consumer's preflight also accepted all four 2/4-mm and L/R outputs for representative Ward, GMM, and RBF settings. Workflow documentation has been updated for the explicit methods, comparison mode, OVR variants, and cache/resume contracts. `06_05_render_group_gradient_heatmaps.py` has since been retired; `06_04_render_hippomaps_gradient_products.py` is the maintained continuous-gradient renderer, and cluster-method annotation is handled by the active group-overview consumers.

**Files:**

- Extend active consumers in `scripts/core_workflow/06_01_build_group_overview.py`, `06_02_materialize_group_labels.py`, `06_03_build_cluster_cortex_panels.py`, and `07_01_build_group_matrix.py` as required by their actual method/variant dispatch. The maintained continuous-gradient renderer is `06_04_render_hippomaps_gradient_products.py`.
- Modify overview and confidence renderers only where they assume a single gradient method or shifted-cosine provenance.
- Extend assignment and artifact-contract tests for the new method and variant paths.

- [x] Add all three method names to the producer, output-layout, manifest, and consumer method registries. Preserve exact manifest-declared selection and fail closed on missing or ambiguous files.
- [x] Extend renderers to distinguish Ward, GMM, and RBF spectral outputs. Attach shifted-cosine provenance only to the existing method; record RBF scale and mesh-edge rule for RBF spectral.
- [x] Render spatial cluster identity separately from Fisher-z, Top-20, and both OVR variants. Use a zero reference for coefficients and an odds-ratio-one reference for odds ratios; label raw and per-SD units explicitly. Never label coefficients as Fisher-z or imply that plotting per-SD OR reproduces Kember's scale.
- [x] Add `fixed-comparison` to the existing K-selection mode. It evaluates and exports the requested K, records diagnostics, and leaves automatic K selection unset. Data, contract, and fitting errors remain fatal.
- [x] Allow an explicitly declared dataset comparison to write under repository `tmp/`, while retaining dataset identity and source-input validation. Do not alter default formal output routing.
- [x] Ensure existing Lynch, HCP, Stanford, and Iowa caller paths either pass method selection through correctly or continue using their current defaults. OVR is supported only when the caller has the declared Kong2022 400-ROI subjectwise input contract; reject unsupported inputs explicitly.
- [x] Update behavior/path documentation and relevant environment documentation in the same changes. No dependency change is expected.

**Completion:** New methods and variants can be generated, read, validated, and rendered through active consumers without changing the identity or boundary of their spatial clusters.

## Task 5: Run the HCP PFM comparison

**Execution status:** Complete. Input validation, all 16 full-cohort gradient combinations, all 160 candidate screens, and all 20 split plus 20 null resamples completed. Of 96 primary settings, 78 passed and 18 failed; across primary and bandwidth-sensitivity settings, 101 passed and 59 failed. All 32 Ward and 32 GMM settings passed; all 59 failures occurred in RBF settings and retain their connectivity diagnostics. The first candidate (route A, native features, Ward, 4 mm left) passed its output contracts, with cluster sizes 144/264/104. All 20 split resamples finished for each viable setting: all Ward and GMM split comparisons passed, while 71 of 740 RBF split comparisons failed and were recorded. All candidate and assignment contracts and the shared seed lists passed; the report includes method tradeoffs and input-sensitivity findings, and the independent review findings have been resolved.

**Scratch root:** `tmp/gradient_parcellation_comparison/` (create during implementation only).

**Authoritative roster:** `outputs_hcp/inputs/kong2022/manifests/hcp_7t_hippocampus_fmri_qc_pass_t1only_latest.txt`. Use the complete roster resulting from the upgraded HippUnfold reconstruction; do not assume a historical count or cap the sample.

First run a full-roster numerical screen on left/right hemispheres and 2mm/4mm smoothing, using coverage 0.80, G1–G3, K=3, seed `20260420`, and identical inputs. Use these four explicit gradient routes, each with both native and unit-SD clustering features:

| Route | FC column scaling | Sparsification |
|---|---|---|
| A | Column z-score | Feature rows, sparsity 0.9 (current-route control) |
| B | None | Feature rows, sparsity 0.9 (original-plan control) |
| C | None | Pearson affinity rows, sparsity 0.9 |
| D | None | None (full nonnegative Pearson affinity control) |

This is 8 feature settings × 3 methods × 2 hemispheres × 2 smoothing levels = 96 primary candidate fits. RBF scale 1.0 is the prespecified primary bandwidth; examine 0.5 and 2.0 separately and report failed settings rather than silently choosing a successful scale. This is a targeted comparison, not a full factorial search or a search for anatomical thirds.

For numerically viable candidates, generate both OVR taxonomies and retain Fisher-z and Top-20 assignments. Then use 20 full-roster subject-split resamples and 20 null resamples with the same seed lists across methods/settings; recompute each split's gradients and scaling from that split. Keep failed candidates in the results with their failure reason; skip their downstream annotation and resampling. Keep all outputs explicitly separated by preprocessing, clustering scaling, method, and bandwidth within the scratch root, using existing metadata mechanisms rather than new hash or governance artifacts.

- [x] Report cluster sizes, connected components per cluster, the fraction outside each cluster's largest component, gradient dispersion, existing FC-profile coherence, split stability, null results, between-method ARI, and agreement with binary-mesh-only spectral clustering on identical support.
- [x] Report FC feature retention by network, diffusion-affinity connectivity, gradient SDs and variance shares, GMM convergence, RBF degree/effective-connectivity diagnostics and scale sensitivity, and all assignment variants separately. Treat geometry agreement and anatomical ordering as diagnostics, not selection thresholds. Large diffusion-coordinate variance is not automatically a bug, and equalizing it can amplify weak or noisy components.
- [x] Generate comparison tables and figures under the scratch root. Record complete results or an explicit investigated failure for every prespecified setting. Reproduce the external review's reported imbalance where possible, with exact input/support and parameter declarations; do not present its supplied numbers as independently verified before this run.
- [x] Do not combine results into an invented score or automatically choose a production method. The final comparison recommends tradeoffs; selecting the production method remains a separate user decision.
- [x] Use one heavy fit at a time with BLAS/OpenMP threads set to one. The host is IDAS; keep total concurrency at or below 10.

## Test Plan

- Gradient route: requested FC scaling and sparsification order are honored with no double kernel/sparsification; missing or mismatched declarations fail reuse; full and subset fits share settings; clustering feature scaling uses the declared population.
- Clustering: same-direction vectors with different magnitudes affect distance-based affinity; duplicate triangle edges do not change weights; tiny nonzero bridges trigger effective-connectivity diagnostics; small-cluster flags are recorded without relabeling; Ward remains connected; disconnected support fails before library edge completion; GMM reports discontiguous regions and nonconvergence.
- OVR: subject-level inputs are used; coefficient classes and raw/per-SD odds ratios match direct estimator output and defined SDs; the two taxonomies remain separate; missing inputs fail with identifiers; annotation selection leaves clusters unchanged.
- Integration: method/taxonomy paths resolve from exact manifests; old `both` behavior remains unchanged; comparison output requires explicit dataset identity; renderers distinguish raw FC from regression coefficients.
- Execute targeted tests first, then affected core-workflow, visualization, and individual-workflow suites using the explicit py314 interpreter. Direct all test artifacts to repository `tmp/`.
- Review `git diff` and `git status`; preserve unrelated pre-existing working-tree changes. Do not create hashes, persistent verification artifacts, or cleanup commands for pre-existing paths.

## Assumptions and Completion Criteria

- The existing subjectwise ROI profile function remains the source of truth for OVR input FC and ROI ordering.
- Ward is expected to enforce connected clusters on a connected supported mesh; GMM is allowed to produce disconnected regions; RBF spectral affinity retains mesh locality while encoding gradient-distance differences.
- The HCP comparison is exploratory and does not authorize formal output replacement or a default-method change.
- The change is complete when preprocessing, method dispatch, OVR, exporters, consumers, documentation, targeted/integration tests, and the HCP comparison all agree on their explicit contracts.
