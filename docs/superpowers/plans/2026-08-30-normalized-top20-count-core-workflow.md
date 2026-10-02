# Normalized Top20 Count Core Workflow Implementation Plan

> Superseded on 2026-09-15: the active contract keeps `normalized_top20_count`
> as an explicit cluster-level comparison and the Kong2022 ROI-registry WTA
> assignment, while `fisherz` is the cluster-level default. The original plan
> is retained as historical provenance for the reverted default decision.

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [x]`) syntax for tracking.

**Goal:** Make `normalized_top20_count` the default Top-20 assignment for both Kong2022 cluster-level/Fisher-z-rendered outputs and `wta-top`, while removing raw `count` as a maintained `core_workflow` variant.

**Architecture:** Keep `fisherz` as an explicit cluster-level comparison variant, replace the cluster-level raw-count variant with the existing normalized Top-20 vote semantics, and point all default manifest fields and overview/matrix registries at `normalized_top20_count`. Store normalized network votes in status rows so renderers can reconstruct the normalized profile without relying on raw-count artifacts.

**Tech Stack:** Python 3.14, NumPy, pytest, JSON/Markdown workflow contracts.

**Spec:** User request in the current session: `fisherz` and `wta` both have Top-20 count versions; make both default to normalized count and remove raw-count versions from `core_workflow`.

## Global Constraints

- Active system is IDAS; do not launch more than 10 concurrent jobs/tasks/processes.
- Core workflow Python checks use `/home/yojiao/.conda/envs/py314/bin/python`; do not use bare `python`.
- Do not delete existing generated raw-count files without a separate explicit deletion confirmation; this change removes their generation/registration only.
- Keep `fisherz` as an explicit comparison variant; remove only the raw `count` assignment variant.
- Synchronize code, tests, manifests, and workflow documentation in the same change.

---

### Task 1: Lock the normalized variant contract with failing tests

**Files:**
- Modify: `tests/core_workflow/test_kong2022_cluster_annotation_shared.py`
- Modify: `tests/core_workflow/test_method_name_contract.py`
- Modify: `tests/core_workflow/test_functional_atlas_default_variant_contract.py`
- Modify: `tests/core_workflow/test_network_color_policy.py`

**Interfaces:**
- Consumes: Existing `cluster_assignment_variants_from_kong2022_roi_profile`, overview resolver, and label materializer APIs.
- Produces: Regression expectations that cluster variants are exactly `fisherz` and `normalized_top20_count`, that normalized voting corrects unequal network ROI opportunity, and that defaults/path names use the normalized variant.

- [x] **Step 1: Replace the raw-count cluster expectation with a normalized Top-20 expectation.**

Use a fixture with two Default ROIs and ten Visual ROIs, only three positive Visual ROIs, and assert that the normalized variant selects Default despite Visual having the larger raw vote. Assert `assignment_metric == "top20_count_normalized_by_network_roi_count"`, normalized fields, and absence of the `count` variant.

- [x] **Step 2: Change overview/matrix expectations from `count` to `normalized_top20_count`.**

Update exact variant lists, `--assignment-variant both` expansion, and expected normalized output filenames. Add an assertion that the overview assignment choices do not contain raw `count`.

- [x] **Step 3: Update default materialization fixtures to make normalized Top-20 the cluster-level default.**

Use `default_assignment_variant: "normalized_top20_count"` and suffixed normalized label/status paths in the default functional-atlas tests, while retaining one explicit Fisher-z resolution test.

- [x] **Step 4: Update label-table refresh fixtures to use the normalized variant name and files.**

Keep the test’s color-table purpose unchanged, but ensure the fixture no longer encodes a maintained raw-count variant.

- [x] **Step 5: Run the changed tests and verify they fail for the missing production behavior.**

Run:

```bash
/home/yojiao/.conda/envs/py314/bin/python -c "import sys; print(sys.executable)"
/home/yojiao/.conda/envs/py314/bin/python -m pytest tests/core_workflow/test_kong2022_cluster_annotation_shared.py tests/core_workflow/test_method_name_contract.py tests/core_workflow/test_functional_atlas_default_variant_contract.py tests/core_workflow/test_network_color_policy.py -q
```

Expected: failures specifically report the still-registered raw `count` variant and Fisher-z default paths; fix test setup errors before proceeding.

### Task 2: Replace raw cluster count generation with normalized Top-20 generation

**Files:**
- Modify: `scripts/core_workflow/05_02_run_group_atlas.py`

**Interfaces:**
- Consumes: `build_normalized_topn_count_vertex_rows` and subjectwise cluster ROI profiles.
- Produces: `cluster_assignment_variants_from_kong2022_roi_profile()` returning `fisherz` and `normalized_top20_count`; normalized status rows with `normalized_count_by_network`; manifests whose default assignment points to normalized files.

- [x] **Step 1: Make `assignment_status_from_cluster_roi_topn` compute normalized network votes.**

For each cluster, select the top positive ROI values up to `top_n_cap`, compute each network’s finite ROI opportunity denominator from the same finite mean profile, rank by count divided by opportunity with score-sum/count/name tie-breakers, and emit `top1_normalized_count`, `top2_normalized_count`, `margin_normalized_count`, availability counts, and `normalized_count_by_network`. Preserve significance metadata and the `top_positive_selected` audit list.

- [x] **Step 2: Rename the cluster variant key and metric.**

Change the shared variant builder and its callers from `count`/`topn_count_positive_roi` to `normalized_top20_count`/`top20_count_normalized_by_network_roi_count`, and remove `cluster_dominant_network_count.json` generation.

- [x] **Step 3: Make normalized Top-20 the cluster-level default.**

Set the cluster manifest’s `default_assignment_variant`, top-level functional label/map/status/dominant-network paths, `cluster_assignment_metric`, and `cluster_roi_topn_assignment_metric` to the normalized variant. Continue writing explicit Fisher-z status/labels as the comparison variant.

- [x] **Step 4: Run the Task 1 tests and verify the normalized cluster test passes.**

Run the same four-file pytest command from Task 1; expected result is green for the updated assignment and default contracts.

### Task 3: Remove raw-count registration from overview and matrix rendering

**Files:**
- Modify: `scripts/core_workflow/06_01_build_group_overview.py`
- Modify: `scripts/core_workflow/07_01_build_group_matrix.py`

**Interfaces:**
- Consumes: Normalized cluster status rows and manifest variant paths from Task 2.
- Produces: CLI choices and matrix jobs containing `fisherz` plus `normalized_top20_count`, with normalized confidence/profile rendering.

- [x] **Step 1: Replace raw `count` in assignment registries and defaults.**

Use `("fisherz", "normalized_top20_count", "weighted_fisherz")` in the overview registry, use Fisher-z plus normalized Top-20 for `both`, and make the matrix registry emit those two variants for maintained cluster branches. WTA’s `both` behavior remains its single normalized variant.

- [x] **Step 2: Add normalized profile reconstruction and validation.**

Replace `count_profile_from_status`/raw count validation with a normalized profile reconstruction from `normalized_count_by_network`, validating the normalized top network/value and accepting zero-filled networks absent from a row.

- [x] **Step 3: Render normalized confidence values and heatmaps.**

Make `confidence_rows` prefer normalized fields, use a normalized margin threshold, label the panel/cbar as normalized Top-N ROI count, and route the cluster heatmap through the normalized profile for `normalized_top20_count`. Remove the raw-count rendering branch while preserving Fisher-z and weighted Fisher-z behavior.

- [x] **Step 4: Run overview/matrix contract tests.**

Run:

```bash
/home/yojiao/.conda/envs/py314/bin/python -m pytest tests/core_workflow/test_method_name_contract.py tests/core_workflow/test_lynch2024_route_and_reference.py tests/core_workflow/test_functional_atlas_default_variant_contract.py -q
```

Expected: all variant choices, path names, and default resolution assertions pass, with no raw `count` assignment accepted by the core overview resolver.

### Task 4: Repair shared integrations and synchronize documentation

**Files:**
- Modify: `scripts/core_workflow/05_05_build_weighted_fisherz_branch.py`
- Modify: `scripts/individual_workflow/02_build_individual_overviews.py` only where the shared core variant key is consumed
- Modify: `scripts/core_workflow/README.md`
- Modify: `docs/hipp_parcellation_network/core_workflow_overview.md`
- Modify: `docs/hipp_parcellation_network/multi_branch_flow.md`
- Modify: `docs/hipp_parcellation_network/step_by_step_flows.md`
- Modify: `docs/hipp_parcellation_network/fixedk_overview_tuning.md`

**Interfaces:**
- Consumes: The Task 2 normalized variant key and output metric.
- Produces: No stale core-workflow instruction or shared caller that expects `assignment_variants["count"]` or advertises raw count as maintained.

- [x] **Step 1: Update the weighted branch provenance string.**

Record the normalized Top-20 assignment policy rather than `topn_count_positive_roi`; do not alter the weighted Fisher-z computation itself.

- [x] **Step 2: Update shared individual integration for the renamed core variant.**

Change only key/path/variant plumbing needed for the shared helper; preserve any explicitly separate individual-workflow compatibility behavior unless it would call the removed core `count` resolver.

- [x] **Step 3: Rewrite active and retained workflow docs.**

State that cluster-level maintained variants are Fisher-z and normalized Top-20 count, normalized is the default, raw count is not generated/registered by core workflow, and old generated raw files are historical artifacts not removed by this change. Update command examples and filenames from raw count to normalized names.

- [x] **Step 4: Search for stale maintained raw-count references.**

Run:

```bash
rg -n 'topn_count_positive_roi|assignment_variant.*count|assignment_variants.*count|group_functional_atlas_labels_count|cluster_assignment_status_count|cluster_dominant_network_count' scripts/core_workflow tests/core_workflow docs/hipp_parcellation_network
```

Review every hit; generic result counters and historical-output descriptions may remain only when clearly unrelated to assignment variants.

### Task 5: Verify the complete change before handoff

**Files:**
- Inspect: all modified files plus `git diff`

- [x] **Step 1: Run explicit-interpreter syntax checks.**

```bash
/home/yojiao/.conda/envs/py314/bin/python -m py_compile scripts/core_workflow/05_02_run_group_atlas.py scripts/core_workflow/06_01_build_group_overview.py scripts/core_workflow/07_01_build_group_matrix.py scripts/core_workflow/05_05_build_weighted_fisherz_branch.py
```

- [x] **Step 2: Run the complete core-workflow test suite.**

```bash
/home/yojiao/.conda/envs/py314/bin/python -m pytest tests/core_workflow -q
```

- [x] **Step 3: Inspect the final diff and confirm scope.**

```bash
git diff --check
git diff -- scripts/core_workflow tests/core_workflow docs/hipp_parcellation_network scripts/individual_workflow/02_build_individual_overviews.py
```

Confirm unrelated pre-existing worktree changes remain untouched and no deletion command or generated-output cleanup was performed.

- [x] **Step 4: Report only verified results.**

Include the exact test count/status, changed contract, and explicit note that existing raw-count artifacts were not deleted.

**Verification note:** The normalized targeted contracts passed (`73 passed`, then
`44 passed` after the direct-default assertion). The individual/Iowa7T normalized
contracts passed (`26 passed`, with one unrelated missing-document test excluded).
The complete available core suite was attempted with known missing maintenance
modules excluded: `301 passed, 14 failed`; the failures are pre-existing missing
`scripts/maintenance/*` modules and an unrelated Iowa7T preset token mismatch.
