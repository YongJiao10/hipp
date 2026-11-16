# Unified Group-Atlas Priority Branches Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fold the current Stanford group-atlas analysis into the shared hippocampal analysis run, enforce a strict high-before-low priority barrier, and restrict group-wise atlas contrasts to each group versus the HC reference atlas.

**Architecture:** The unified run owns one branch registry, one priority scheduler, one run manifest, and one report closure. Subject-level rest-FC branches and group-atlas map branches are represented in that registry with an explicit `analysis_family`, so they run together without incorrectly mixing group-map geometry rows with subject-level FC rows. The group atlas uses the existing `routea_hc_*` assets as the explicit HC reference and emits only HC-vs-one-group comparisons; the old all-pairs comparison path is removed from active execution but its directory is not deleted in this change.

**Tech Stack:** Python 3.14 at `/home/yojiao/.conda/envs/py314/bin/python`, pytest, NumPy, pandas, SciPy, scikit-learn, nibabel, matplotlib, JSON/CSV/TSV manifests, Markdown.

**Spec:** `stanford_jjtms_multiscale_hippocampal_exploration/rest_fc_primary_analysis_spec.md`, `cross_dataset_rest_fc/config/analysis_config.json`, and the user-approved branch policy in this request.

## Global Constraints

- The active hippocampal method axis is exactly `hcp_fixed`, `hcp_template_matching`, and `group_atlas_gradient_spectral`; `individual_gradient_k3_full` is never reintroduced.
- The group-atlas reference is the existing `HC` map (`routea_hc_full_k3_fisherz` and its weighted counterpart), and every other configured group is compared only to `HC`.
- Priority is strict: 8-network/2-mm work is tier P0; 8-network/4-mm and 17-network/2-mm work are tier P1; 17-network/4-mm work is tier P2. A lower tier cannot start until every higher-tier branch is complete.
- Group-atlas branches have K=3 region labels and no cortical network-resolution axis; their 2-mm cells are P0 and their 4-mm cells are P2.
- Existing rest-only path gates, Fisher-z requirements, missing-value handling, no-archive rule, and explicit `/home/yojiao/.conda/envs/py314/bin/python` execution remain unchanged.
- No file or directory deletion is part of this implementation; any later cleanup must present an exact target list and receive confirmation first.

---

### Task 1: Freeze the unified branch and HC-contrast contract with failing tests

**Files:**
- Create: `stanford_jjtms_multiscale_hippocampal_exploration/cross_dataset_rest_fc/tests/test_branch_plan.py`
- Modify: `stanford_jjtms_multiscale_hippocampal_exploration/external_group_atlas_analysis/tests/test_atlas_compare.py`

**Interfaces:**
- `branch_plan.build_branch_plan(config)` returns an ordered tuple of immutable branch records with `analysis_family`, `method`, `dataset`, `network_resolution`, `smoothing_mm`, `hemisphere_modes`, `group`, `assignment_variant`, and `priority_tier` fields.
- `branch_plan.can_start(branch_id, statuses, plan)` returns `True` only when every branch in a strictly higher tier has status `complete`.
- `atlas_compare.GROUP_ORDER` contains `HC` followed by the seven configured cohort groups; `atlas_compare.HC_COMPARISON_ORDER` contains exactly one `(group, HC)` pair for each non-HC group.

- [ ] **Step 1: Write the failing branch-plan tests.**

```python
def test_unified_plan_has_three_methods_and_excludes_individual_gradient() -> None:
    plan = build_branch_plan(CONFIG)
    methods = {branch.method for branch in plan}
    assert methods == {
        "hcp_fixed",
        "hcp_template_matching",
        "group_atlas_gradient_spectral",
    }
    assert all("individual_gradient" not in branch.method for branch in plan)


def test_priority_order_puts_all_8net_2mm_before_lower_tiers() -> None:
    plan = build_branch_plan(CONFIG)
    tiers = [branch.priority_tier for branch in plan]
    assert tiers == sorted(tiers)
    assert all(
        branch.priority_tier == 0
        for branch in plan
        if branch.analysis_family == "subject_fc"
        and branch.network_resolution == 8
        and branch.smoothing_mm == 2
    )
    assert all(
        branch.priority_tier > 0
        for branch in plan
        if branch.analysis_family == "subject_fc"
        and (branch.network_resolution == 17 or branch.smoothing_mm == 4)
    )


def test_lower_tier_is_blocked_until_every_higher_tier_is_complete() -> None:
    plan = build_branch_plan(CONFIG)
    low = next(branch for branch in plan if branch.priority_tier == 1)
    statuses = {branch.branch_id: "complete" for branch in plan if branch.priority_tier == 0}
    assert can_start(low.branch_id, statuses, plan)
    statuses[next(branch.branch_id for branch in plan if branch.priority_tier == 0)] = "running"
    assert not can_start(low.branch_id, statuses, plan)
```

- [ ] **Step 2: Run the new tests and verify they fail because the unified planner and HC-only registry do not yet exist.**

Run:

```bash
/home/yojiao/.conda/envs/py314/bin/python -c "import sys; print(sys.executable)"
/home/yojiao/.conda/envs/py314/bin/python -m pytest \
  stanford_jjtms_multiscale_hippocampal_exploration/cross_dataset_rest_fc/tests/test_branch_plan.py \
  stanford_jjtms_multiscale_hippocampal_exploration/external_group_atlas_analysis/tests/test_atlas_compare.py -q
```

Expected: collection or assertion failures identifying the missing planner, missing third method, and current all-pairs/7-group registry.

- [ ] **Step 3: Add failing HC-only contrast assertions.**

```python
def test_group_comparisons_use_hc_reference_only() -> None:
    assert len(HC_COMPARISON_ORDER) == 7
    assert all(second == "HC" for _, second in HC_COMPARISON_ORDER)
    assert {first for first, _ in HC_COMPARISON_ORDER} == set(GROUP_ORDER) - {"HC"}
    assert not any(
        first != "HC" and second != "HC"
        for first, second in HC_COMPARISON_ORDER
    )
```

- [ ] **Step 4: Re-run the focused tests and record the expected red state before production edits.**

---

### Task 2: Add the explicit unified branch policy and deterministic scheduler

**Files:**
- Create: `stanford_jjtms_multiscale_hippocampal_exploration/cross_dataset_rest_fc/src/branch_plan.py`
- Modify: `stanford_jjtms_multiscale_hippocampal_exploration/cross_dataset_rest_fc/config/analysis_config.json`
- Test: `stanford_jjtms_multiscale_hippocampal_exploration/cross_dataset_rest_fc/tests/test_branch_plan.py`

**Interfaces:**
- `Branch` is a frozen dataclass with a stable `branch_id` derived from all non-null axis values.
- `build_branch_plan(config)` expands subject-FC units for both datasets, both subject methods, 8/17 networks, 2/4 mm, and both raw hemispheres needed to produce bilateral/L/R/L-R views. It also expands group-atlas units for `HC` plus the seven configured groups, `plain/weighted`, 2/4 mm, and L/R.
- `build_branch_plan` applies the configured P0/P1/P2 lookup to subject-FC and group-atlas axes.
- `can_start(branch_id, statuses, plan)` raises `KeyError` for an unknown branch and otherwise enforces the strict higher-tier completion barrier.
- `run_priority_batches` writes complete/failed status records and never invokes a lower-tier callback while a higher-tier branch is pending, running, or failed.

- [ ] **Step 1: Add the configuration registry.** Add the following top-level object to `analysis_config.json` without changing the existing FC input registries:

```json
"unified_branch_policy": {
  "method_order": [
    "hcp_fixed",
    "hcp_template_matching",
    "group_atlas_gradient_spectral"
  ],
  "subject_fc_priority": [
    {"tier": 0, "network_resolution": 8, "smoothing_mm": 2},
    {"tier": 1, "network_resolution": 8, "smoothing_mm": 4},
    {"tier": 1, "network_resolution": 17, "smoothing_mm": 2},
    {"tier": 2, "network_resolution": 17, "smoothing_mm": 4}
  ],
  "group_atlas_priority": [
    {"tier": 0, "smoothing_mm": 2},
    {"tier": 2, "smoothing_mm": 4}
  ],
  "barrier": "strict"
},
"group_atlas": {
  "method": "group_atlas_gradient_spectral",
  "input_root": "outputs_stanford/work/group_atlas",
  "reference_group": "HC",
  "groups": [
    "HC",
    "NTHC",
    "TEHC",
    "NTS",
    "NTS Any MDD",
    "NTS Pure MDD",
    "TIS",
    "TIS PTSD"
  ],
  "assignment_variants": ["plain", "weighted"],
  "smoothings_mm": [2, 4],
  "hemispheres": ["L", "R"],
  "k": 3,
  "groupwise_contrast": "each_non_reference_group_vs_reference"
}
```

- [ ] **Step 2: Implement the immutable branch records, expansion, sorting, and barrier checks in `branch_plan.py`.** The sorting key is `(priority_tier, method_order, dataset, group, assignment_variant, smoothing_mm, network_resolution)`, with null axes sorted before numeric axes; no input order is allowed to determine execution order.

- [ ] **Step 3: Run only `test_branch_plan.py` and verify the planner tests pass.**

Run:

```bash+/home/yojiao/.conda/envs/py314/bin/python -m pytest \
  stanford_jjtms_multiscale_hippocampal_exploration/cross_dataset_rest_fc/tests/test_branch_plan.py -q
```

- [ ] **Step 4: Run the atlas tests and confirm they remain red only for the not-yet-updated HC registry/contrast behavior.**

---

### Task 3: Make FC feature materialization branch-selectable and priority-aware

**Files:**
- Modify: `stanford_jjtms_multiscale_hippocampal_exploration/cross_dataset_rest_fc/src/features.py`
- Modify: `stanford_jjtms_multiscale_hippocampal_exploration/cross_dataset_rest_fc/src/reporting.py`
- Modify: `stanford_jjtms_multiscale_hippocampal_exploration/cross_dataset_rest_fc/src/run_primary.py`
- Modify: `stanford_jjtms_multiscale_hippocampal_exploration/cross_dataset_rest_fc/tests/test_features.py`
- Modify: `stanford_jjtms_multiscale_hippocampal_exploration/cross_dataset_rest_fc/tests/test_reporting.py`

**Interfaces:**
- Extend `features.build_fc_tensor` with keyword-only filters `methods: Sequence[str] | None = None`, `smoothings_mm: Sequence[int] | None = None`, and `network_resolutions: Sequence[int] | None = None`; omitted filters preserve current all-axis behavior for fixture callers.
- Extend `_combo_rows` and `_empty_rows` so a filtered call emits only requested network resolutions while preserving explicit failure rows for every requested component.
- Add `reporting.run_priority_batches(config, rosters, plan, ...)` returning `(tensors, views, branch_statuses, branch_manifest)`; each subject-FC branch is materialized and marked complete before the next tier is entered.
- `reporting.run_analysis` uses `run_priority_batches`, concatenates completed tier outputs, and includes `branch_manifest.json` data in `run_manifest.json` and `technical_report.md`.

- [ ] **Step 1: Write a failing feature test for resolution filtering.**

```python
def test_build_fc_tensor_can_materialize_one_priority_resolution(tmp_path: Path) -> None:
    config = _write_fixture(tmp_path)
    tensor = build_fc_tensor(
        "jjtms",
        _roster(),
        config,
        methods=("synthetic_registry",),
        smoothings_mm=(2,),
        network_resolutions=(8,),
    )
    assert set(tensor["network_resolution"]) == {8}
    assert set(tensor["smoothing_mm"]) == {2}
```

- [ ] **Step 2: Run the focused feature test and confirm it fails because the new keyword filters are absent.**

- [ ] **Step 3: Implement the filters and preserve all existing unfiltered fixture behavior.**

- [ ] **Step 4: Add a reporting test that injects a failing P0 branch and asserts no P1/P2 callback is invoked.**

```python
def test_priority_batches_stop_before_lower_tier_after_high_tier_failure() -> None:
    calls: list[str] = []

    def execute(branch):
        calls.append(branch.branch_id)
        return "failed" if branch.priority_tier == 0 else "complete"

    with pytest.raises(RuntimeError, match="higher-priority"):
        run_priority_batches(CONFIG, ROSTERS, build_branch_plan(CONFIG), execute)
    p0_ids = {branch.branch_id for branch in build_branch_plan(CONFIG) if branch.priority_tier == 0}
    assert set(calls) <= p0_ids
```

- [ ] **Step 5: Implement the strict tier loop.** The loop must complete every branch in the current tier before opening the next tier; a failed branch stops the run and records its failure rather than allowing low-priority work to start.

- [ ] **Step 6: Run all affected feature/reporting tests and confirm green.**

---

### Task 4: Fold group atlas into the unified run and reduce comparisons to HC-only

**Files:**
- Modify: `stanford_jjtms_multiscale_hippocampal_exploration/external_group_atlas_analysis/src/atlas_compare.py`
- Modify: `stanford_jjtms_multiscale_hippocampal_exploration/external_group_atlas_analysis/src/run_atlas_analysis.py`
- Modify: `stanford_jjtms_multiscale_hippocampal_exploration/external_group_atlas_analysis/tests/test_atlas_compare.py`
- Modify: `stanford_jjtms_multiscale_hippocampal_exploration/cross_dataset_rest_fc/src/reporting.py`
- Modify: `stanford_jjtms_multiscale_hippocampal_exploration/cross_dataset_rest_fc/tests/test_reporting.py`

**Interfaces:**
- `atlas_compare.GROUP_ORDER` becomes `("HC", "NTHC", "TEHC", "NTS", "NTS Any MDD", "NTS Pure MDD", "TIS", "TIS PTSD")`.
- `_atlas_directory` maps `HC` to `routea_hc_full_k3_fisherz` or `routea_hc_weighted_fisherz_full_k3` and continues to map the seven cohort names to their current paths.
- `discover_atlases` validates exactly 64 cells: 8 groups × 2 variants × 2 smoothings × 2 hemispheres.
- `HC_COMPARISON_ORDER` contains exactly seven pairs `(group, "HC")`; `build_matched_group_pairs` emits only those seven comparisons per variant/smoothing/hemisphere, and no non-HC pair.
- `build_descriptive_contrasts` emits only the seven direct HC contrasts for the group-atlas family. The existing subject-level Stanford factorial interaction remains in the FC model and is not duplicated in the group-map comparison table.
- `run_analysis` accepts `groups`, `variants`, `smoothings_mm`, and `hemispheres` filters and an output root so the unified reporter can run the P0 group-atlas cells before P2 cells.

- [ ] **Step 1: Update the tests to expect the HC reference and seven HC-only comparisons.** Replace the current 21-pair assertion with `len(HC_COMPARISON_ORDER) == 7`; add assertions that no output `group_a` and `group_b` are both non-HC.

- [ ] **Step 2: Run the atlas tests and verify the expected red failures.**

- [ ] **Step 3: Implement HC path mapping, filtered discovery, HC-only matched comparisons, and HC-only descriptive contrasts.** Keep `within_group_spec_stability` for all eight maps, because it is not a group-wise contrast.

- [ ] **Step 4: Add a reporting integration test that checks the unified manifest contains `analysis_family == "group_atlas_geometry"`, `method == "group_atlas_gradient_spectral"`, and group-atlas P0 entries before any P2 entry.**

- [ ] **Step 5: Make the main `primary_screen` runner invoke the group-atlas executor through the same priority scheduler and write its tables under the current run's `results/primary_screen/group_atlas/`, figures under `figures/primary_screen/group_atlas/`, and report under `reports/primary_screen/group_atlas/`. The old standalone CLI remains a compatibility entry point but is no longer the authoritative execution route.**

- [ ] **Step 6: Run the affected atlas and reporting tests and confirm green.**

---

### Task 5: Synchronize scientific documentation and output contracts

**Files:**
- Modify: `stanford_jjtms_multiscale_hippocampal_exploration/rest_fc_primary_analysis_spec.md`
- Modify: `stanford_jjtms_multiscale_hippocampal_exploration/README.md`
- Modify: `stanford_jjtms_multiscale_hippocampal_exploration/cross_dataset_rest_fc/README.md`
- Modify: `stanford_jjtms_multiscale_hippocampal_exploration/docs/analysis_map.md`
- Modify: `stanford_jjtms_multiscale_hippocampal_exploration/study_spec.md`
- Modify: `stanford_jjtms_multiscale_hippocampal_exploration/external_group_atlas_analysis/README.md`
- Modify: `stanford_jjtms_multiscale_hippocampal_exploration/external_group_atlas_analysis/reports/atlas_descriptive/README.md`
- Modify: `stanford_jjtms_multiscale_hippocampal_exploration/cross_dataset_rest_fc/tests/test_documentation_contract.py` if its assertions cover the old independent-workflow language.

**Interfaces:**
- Every current document describes one unified hippocampal branch registry and one priority-gated run, with `group_atlas_gradient_spectral` as the third method.
- Documents explicitly state the priority matrix: P0 = 8net/2mm plus group-atlas 2mm; P1 = 8net/4mm and 17net/2mm; P2 = 17net/4mm plus group-atlas 4mm.
- Documents explicitly state that group-wise atlas comparisons are group-vs-HC only; all non-HC pairwise comparisons are out of scope.
- Documents distinguish the group-atlas geometry analysis family from subject-level FC rows without calling it an independent workflow.
- Documents contain no active `individual_gradient_k3_full` or `individual_gradient` method reference.

- [ ] **Step 1: Add documentation-contract tests for the method list, priority matrix, HC-only rule, and absence of individual-gradient references.**

- [ ] **Step 2: Run the documentation tests and verify the expected red state.**

- [ ] **Step 3: Update the documents and current run/report descriptions to match the implementation.**

- [ ] **Step 4: Re-run documentation tests and scan active non-archive files with `rg`; do not inspect or modify archive content.**

---

### Task 6: Rebuild current outputs only after the contracts pass

**Files:**
- Modify in place: `stanford_jjtms_multiscale_hippocampal_exploration/cross_dataset_rest_fc/results/primary_screen/`
- Modify in place: `stanford_jjtms_multiscale_hippocampal_exploration/cross_dataset_rest_fc/figures/primary_screen/`
- Modify in place: `stanford_jjtms_multiscale_hippocampal_exploration/cross_dataset_rest_fc/reports/primary_screen/`
- Modify in place: `stanford_jjtms_multiscale_hippocampal_exploration/external_group_atlas_analysis/results/atlas_descriptive/` only if the compatibility CLI is explicitly rerun.

**Interfaces:**
- The unified primary manifest records every branch, priority tier, start/end status, input signature, and output path.
- The primary result closure contains the FC tables plus the group-atlas HC-only tables; no separate authoritative atlas run is required.
- Expected group-atlas counts after integration are 64 inventory rows, 192 region-area rows, 112 HC-only matched-comparison rows, and 64 within-group stability rows. The exact count is asserted from the generated manifest rather than assumed from filenames.

- [ ] **Step 1: Run the focused unit tests for branch planning, features, reporting, and atlas comparison.**

- [ ] **Step 2: Run the current data smoke with the explicit py314 interpreter and verify P0 completion before P1/P2 entries in the manifest.**

- [ ] **Step 3: Run the full current `primary_screen` materialization only after the smoke and contract checks pass.**

- [ ] **Step 4: Verify output closure, branch ordering, HC-only contrast keys, and absence of individual-gradient method names.**

- [ ] **Step 5: Run `python -m py_compile` through the explicit py314 interpreter for every modified active Python file.**

- [ ] **Step 6: Review the diff for unrelated user changes; do not stage or revert the currently modified Kember files.**

- [ ] **Step 7: Commit only the unified branch-policy changes after fresh verification.**
