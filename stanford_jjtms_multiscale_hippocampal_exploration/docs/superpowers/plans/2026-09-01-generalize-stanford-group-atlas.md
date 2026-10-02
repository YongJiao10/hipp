# Generalize Stanford Group Atlas Analysis Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [x]`) syntax for tracking.

**Goal:** Rename the misleading four-group atlas package and make its stable atlas comparison cover the seven configured Stanford cohorts, while preserving the four-group trauma-by-symptom model as a separate inferential design.

**Architecture:** The atlas package remains an external, descriptive group-map workflow, but its cohort registry expands from `NTHC`, `TEHC`, `NTS`, and `TIS` to those four parent groups plus `NTS Any MDD`, `NTS Pure MDD`, and `TIS PTSD`. The package will discover 56 validated atlas cells, compare all cohort pairs descriptively, and emit explicit nested-cohort direct-to-NTHC summaries; `kember_metrics` remains the subject-level endpoint workflow and the four-group factorial contrast definition remains unchanged.

**Tech Stack:** Python 3.14 (`/home/yojiao/.conda/envs/py314/bin/python`), pytest, NumPy, pandas, SciPy, scikit-learn, nibabel, matplotlib, Markdown/JSON/CSV manifests, Git.

**Spec:** `stanford_jjtms_multiscale_hippocampal_exploration/docs/analysis_map.md`, `stanford_jjtms_multiscale_hippocampal_exploration/study_spec.md`, and the cohort definitions in `kember_metrics/config/analysis_config.json`.

## Global Constraints

- The four-group factorial design remains exactly `NTHC`, `TEHC`, `NTS`, and `TIS`; nested cohorts are not additional mutually exclusive factorial cells.
- The complete atlas cohort registry is exactly `NTHC`, `TEHC`, `NTS`, `NTS Any MDD`, `NTS Pure MDD`, `TIS`, and `TIS PTSD`.
- Atlas scope remains descriptive group-map geometry only; no p-values, q-values, uncertainty, or causal/clinical inference is added.
- Atlas area fields remain namespaced as `atlas_region_area_mm2`, `atlas_region_area_fraction`, and `atlas_total_surface_area_mm2`; they are never Kember subject-level area.
- Stable outputs are overwritten in place under `results/atlas_descriptive/`, `figures/atlas_descriptive/`, and `reports/atlas_descriptive/`; no result-version suffix is introduced.
- All repository Python commands use `/home/yojiao/.conda/envs/py314/bin/python`; direct HippUnfold is not part of this change.
- No archive content is accessed.

---

### Task 1: Lock the seven-cohort atlas contract with failing tests

**Files:**
- Modify: `stanford_jjtms_multiscale_hippocampal_exploration/external_group_atlas_analysis/tests/test_atlas_compare.py`
- Modify: `stanford_jjtms_multiscale_hippocampal_exploration/external_group_atlas_analysis/tests/test_area_namespace.py`

**Interfaces:**
- Consumes the existing `discover_atlases`, `build_matched_group_pairs`, and `run_analysis` interfaces.
- Produces test expectations for `GROUP_ORDER`, 56-cell discovery, 21 pairwise cohort combinations, nested direct summaries, and the renamed package import path.

- [x] **Step 1: Add a failing seven-cohort registry test.** Assert that the public registry contains the seven exact cohort names in the documented order, that `PAIR_ORDER` has `21` pairs, and that `_atlas_directory` resolves the MDD/PTSD slugs.

- [x] **Step 2: Add a failing discovery/output cardinality test.** Build a synthetic input tree with the seven cohort names, two variants, two smoothing values, and two hemispheres using the existing test fixture helpers; assert that discovery returns `56` records and that the inventory has `56` rows.

- [x] **Step 3: Add a failing nested-summary test.** Use synthetic records and assert that `build_descriptive_contrasts` emits `NTS Any MDD-NTHC`, `NTS Pure MDD-NTHC`, and `TIS PTSD-NTHC` in addition to the four parent-group contrasts and the four-group interaction.

- [x] **Step 4: Run only the modified atlas tests and confirm RED.**

Run:

```bash
/home/yojiao/.conda/envs/py314/bin/python -m pytest \
  stanford_jjtms_multiscale_hippocampal_exploration/external_group_atlas_analysis/tests/test_atlas_compare.py \
  stanford_jjtms_multiscale_hippocampal_exploration/external_group_atlas_analysis/tests/test_area_namespace.py -q
```

Expected: failure because the production registry is still four-group and the renamed package/output contract does not yet exist.

### Task 2: Generalize atlas code to all seven cohorts

**Files:**
- Modify: `stanford_jjtms_multiscale_hippocampal_exploration/external_group_atlas_analysis/src/atlas_compare.py`
- Modify: `stanford_jjtms_multiscale_hippocampal_exploration/external_group_atlas_analysis/src/run_atlas_analysis.py`

**Interfaces:**
- `GROUP_ORDER` becomes the seven-cohort tuple; `GROUP_SLUG`, `PAIR_ORDER`, discovery, stability, and plotting derive from it.
- `discover_atlases(input_root, expected_n_vertices=512)` returns exactly 56 validated `LoadedAtlas` records.
- `build_descriptive_contrasts(...)` keeps the four parent-group contrasts and interaction, and adds three nested direct-to-NTHC atlas summaries.
- Manifest/report text identifies the run as a seven-cohort Stanford group-atlas comparison and records the parent/nested cohort sets.

- [x] **Step 1: Replace the hard-coded four-cohort registry with the seven-cohort registry and add distinct colors for all seven plot lines.**

- [x] **Step 2: Keep all-pair geometry comparisons derived from `GROUP_ORDER`, yielding 21 pairs per matched specification without introducing inferential statistics.**

- [x] **Step 3: Add the three nested direct-to-NTHC descriptive contrasts to `build_descriptive_contrasts`; retain the symptom-by-trauma difference-in-differences only for the four parent groups.**

- [x] **Step 4: Update run manifest, technical summary, CLI docstring, report README, figure titles, and output cardinality text to say seven cohorts and 56 cells.**

- [x] **Step 5: Run the modified atlas tests and confirm GREEN.**

### Task 3: Rename the package and migrate all references

**Files:**
- Rename: `stanford_jjtms_multiscale_hippocampal_exploration/external_four_group_atlas_analysis/` → `stanford_jjtms_multiscale_hippocampal_exploration/external_group_atlas_analysis/`
- Modify: `stanford_jjtms_multiscale_hippocampal_exploration/README.md`
- Modify: `stanford_jjtms_multiscale_hippocampal_exploration/docs/analysis_map.md`
- Modify: `stanford_jjtms_multiscale_hippocampal_exploration/docs/restructure_record.md`
- Modify: `stanford_jjtms_multiscale_hippocampal_exploration/study_spec.md`
- Modify: `stanford_jjtms_multiscale_hippocampal_exploration/literature_and_kember_method_notes.md`
- Modify: `stanford_jjtms_multiscale_hippocampal_exploration/kember_metrics/tests/test_documentation_contract.py`

**Interfaces:**
- The canonical import/source path is `external_group_atlas_analysis/`.
- All active docs describe three workflows with the generalized external group-atlas name.
- The docs explicitly distinguish the seven cohort registry from the four-group factorial model and state the nested cohort overlap policy.

- [x] **Step 1: Rename the confirmed old package path `external_four_group_atlas_analysis/` to `external_group_atlas_analysis/` without creating a versioned copy.**

- [x] **Step 2: Update all active references, examples, test import paths, output paths, and headings.** Historical `docs/restructure_record.md` entries may retain the old path only when clearly labeled as historical, but its current-state section must use the new name.

- [x] **Step 3: Update the analysis map and README to list seven cohorts, the six Kember direct contrasts, and the separate four-group factorial role.**

- [x] **Step 4: Run a repository-wide active-reference scan and the renamed package tests.**

### Task 4: Rebuild stable seven-cohort atlas outputs

**Files:**
- Modify: `stanford_jjtms_multiscale_hippocampal_exploration/external_group_atlas_analysis/results/atlas_descriptive/*`
- Modify: `stanford_jjtms_multiscale_hippocampal_exploration/external_group_atlas_analysis/figures/atlas_descriptive/*`
- Modify: `stanford_jjtms_multiscale_hippocampal_exploration/external_group_atlas_analysis/reports/atlas_descriptive/*`

**Interfaces:**
- The existing validated group-atlas manifests under `outputs_stanford/work/group_atlas/` are the only input source.
- The stable run ID remains `atlas_descriptive`.
- Expected output counts are 56 inventory rows, 168 region-area rows, 168 matched pair rows (21 pairs × 2 variants × 2 smoothing levels × 2 hemispheres), and 56 stability rows (7 groups × 2 variants × 2 smoothing levels × 2 hemispheres).

- [x] **Step 1: Run the renamed CLI with `--run-id atlas_descriptive` using the explicit py314 interpreter.**

- [x] **Step 2: Validate output cardinalities, group names, namespaced area fields, no p/q inference, and all consumed geometry paths.**

- [x] **Step 3: Confirm rerun overwrite semantics by rerunning the same stable command and checking the manifest/output closure.**

### Task 5: Full verification and checkpoint commit

**Files:**
- Test: `stanford_jjtms_multiscale_hippocampal_exploration/external_group_atlas_analysis/tests/`
- Test: `stanford_jjtms_multiscale_hippocampal_exploration/kember_metrics/tests/`

- [x] **Step 1: Run the complete affected workflow test suites (run separately due top-level module names) with the explicit py314 interpreter.**

- [x] **Step 2: Compile all active Python sources with py314.**

- [x] **Step 3: Scan active files for stale `external_four_group_atlas_analysis` references and forbidden versioned result names, excluding the historical restructure record where appropriate.**

- [x] **Step 4: Review the staged diff for path correctness, docs/code synchronization, and no unrelated parent-repository changes.**

- [x] **Step 5: Commit the completed change with `refactor: generalize Stanford group atlas workflow`.**
