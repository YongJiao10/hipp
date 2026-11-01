# Eliminate Silent Gradient Profile Fallbacks Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task with verification checkpoints.

**Goal:** Eliminate the silent raw-cluster fallback that can mislabel Anterior/Middle/Posterior gradient panels, make the canonical ROI profile contract explicit across producer and consumer, and add a repository rule forbidding unvalidated fallback logic.

**Architecture:** The group-atlas producer will generate and register the long-axis-relabelled `cluster_roi_profile.npy` for both Kong2022 atlas variants used by the visualizer (`kong2022` and `kong2022_7net`). The visualizer will accept only that registered artifact after validating manifest status, path, shape, and head-to-tail ordering; it will fail closed when the contract is absent or invalid. Gradient panel cache names will include a new contract namespace so stale raw-label panels cannot be reused after the fix.

**Tech Stack:** Python 3.14 (`/home/yojiao/.conda/envs/py314/bin/python`), NumPy, pytest, repository Markdown documentation, existing group-atlas and visualization scripts.

## Global Constraints

- Do not add a fallback or compatibility branch that silently substitutes another label/profile source.
- Preserve the user-owned untracked `.tmp_lynch8.Cl8ZVK/` directory.
- Use `apply_patch` for source and documentation edits.
- Run all repository Python commands through the explicit `py314` interpreter.
- Do not delete files or directories.

## Task 1: Add failing tests for the strict profile contract

**Files:**
- Modify: `tests/visualize/test_hcp_jjtms_7net_branch.py`
- Modify: `tests/core_workflow/test_kong2022_7net.py`

- [ ] Add a renderer test that constructs a valid registered `cluster_roi_profile.npy` manifest and verifies the loader returns the canonical profile.
- [ ] Add tests that assert the renderer raises a clear error when the profile path is absent, the manifest points to a different file, the profile shape is not `(3, 400)`, or the ordering metadata is not head-to-tail.
- [ ] Replace the existing test that blesses `expand_network_profile_to_roi_profile` as a valid 7-network path with a test asserting no network-level expansion is used by the gradient loader.
- [ ] Add a producer test showing `kong2022_7net` is included in the profile-producing atlas contract while unrelated atlas variants remain unchanged.
- [ ] Run only these tests and confirm they fail for the current implementation (RED), recording the failure shape before implementation.

## Task 2: Make the visualizer fail closed and invalidate stale panels

**Files:**
- Modify: `scripts/visualize/multidataset_hipp_cortex_gradient_wta/render_multidataset_hipp_cortex_gradient_wta.py`
- Modify: `tests/visualize/test_hcp_jjtms_7net_branch.py`

- [ ] Remove the `group_labels_k3.npy`/`group_labels_final.npy` selection path from gradient profile loading.
- [ ] Remove the network-profile-to-ROI expansion path from the gradient loader; a network profile is not an ROI profile and must never be used as one.
- [ ] Require the registered profile path from the group-atlas manifest and validate status, path identity, exact `(3, 400)` shape, `cluster_label_ordering == long_axis_head_to_tail`, and `[1, 2, 3]` head-to-tail IDs before use.
- [ ] Add an explicit gradient profile contract namespace to panel-cache keys and ensure the existing unversioned cache cannot satisfy a new request.
- [ ] Update source-contract tests to assert the strict path and new cache namespace.
- [ ] Run the focused visualization tests and confirm they pass (GREEN).

## Task 3: Make the producer create the canonical artifact for every supported Kong2022 atlas

**Files:**
- Modify: `scripts/core_workflow/05_02_run_group_atlas.py`
- Modify: `tests/core_workflow/test_kong2022_7net.py`

- [ ] Add one shared predicate/constant for atlas variants that produce ROI profiles: `kong2022` and `kong2022_7net`.
- [ ] Use that predicate only in the ROI-profile generation, artifact registration, manifest path, and resume-input/signature code paths; do not change unrelated network-level annotation semantics.
- [ ] Ensure the resume signature changes when the canonical 7-network ROI-profile contract is introduced so an old manifest cannot silently skip regeneration.
- [ ] Keep the artifact payload based on `labels_final` after long-axis relabelling.
- [ ] Add producer tests that inspect the relevant source/manifest contract and prevent regression to `atlas == kong2022`-only behavior.
- [ ] Run focused core-workflow tests.

## Task 4: Encode the no-fallback policy and synchronize documentation

**Files:**
- Modify: `AGENTS.md`
- Modify: `scripts/visualize/multidataset_hipp_cortex_gradient_wta/README.md`
- Modify: `docs/hipp_parcellation_network/core_workflow_overview.md` (only if needed to keep the producer/consumer contract explicit)

- [ ] Add a hard repository rule prohibiting silent fallbacks in rigorous analysis/workflow code.
- [ ] State that missing, stale, ambiguous, or mismatched artifacts must fail closed with actionable errors.
- [ ] State that any migration/compatibility path must be explicit, named, separately validated, and provenance-recorded; it must not be an implicit `if missing: use other file` branch.
- [ ] Document that gradient panels consume only the canonical long-axis-ordered ROI profile and that network profiles cannot be expanded into ROI rows.
- [ ] Run documentation/source contract checks.

## Task 5: Regenerate affected 7-network artifacts and verify the plotted row identities

**Files/outputs:**
- Existing Iowa/HCP group-atlas outputs under the configured output roots (generated by the existing producer command; no source deletion).
- Regenerated figure under `scripts/visualize/multidataset_hipp_cortex_gradient_wta/`.

- [ ] Run the producer for the affected 2mm 36P 7-network cohort outputs with resume invalidation active, using the repository’s existing command/configuration rather than an ad hoc label rewrite.
- [ ] Confirm every row used by the figure has a registered canonical ROI profile and no gradient row is loaded from raw labels or a network profile.
- [ ] Regenerate the comparison figure with the new cache namespace.
- [ ] Numerically verify the patient Middle row corresponds to the canonical Middle profile and that the displayed row order is Anterior, Middle, Posterior for all five cohorts.

## Task 6: Full verification and handoff

- [ ] Run the focused tests, then the relevant broader visualization/core-workflow test subsets.
- [ ] Run `git diff --check` and inspect the complete diff for accidental changes.
- [ ] Report the confirmed scope: whether only the patient row was affected in the supplied image, and why the underlying class of error is broader.
- [ ] Do not claim completion until fresh verification demonstrates the strict loader, producer contract, cache invalidation, and documentation rule are all in place.
