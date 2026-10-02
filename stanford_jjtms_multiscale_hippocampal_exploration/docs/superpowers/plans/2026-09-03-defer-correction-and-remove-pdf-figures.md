# Defer Multiplicity Correction and Remove PDF Figures Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Ensure multiplicity correction is never applied or presented before the final analysis-object set is frozen, and make every project figure writer PNG-only while removing all currently stored PDF figures after explicit path confirmation.

**Architecture:** Separate primary localisation from multiplicity finalisation: localisation will retain raw p-values, confidence intervals, coefficients, and effect sizes while emitting an explicit deferred-correction status and no q-values. A future correction step must be an explicit post-freeze operation, never an implicit part of model fitting or reporting. Centralise PNG-only rendering at every active figure writer and remove PDF paths from provenance, manifests, reports, and tests.

**Tech Stack:** Python 3.14 (`/home/yojiao/.conda/envs/py314/bin/python`), pandas, scipy, matplotlib, pytest, JSON/CSV/TSV manifests, Git.

**Spec:** User instruction: do not determine or apply correction until the final analysis-object collection is fixed; do not save PDF figures; remove all existing PDF figures under the project.

## Global Constraints

- Use `/home/yojiao/.conda/envs/py314/bin/python` for repository Python tasks.
- Preserve the user-owned Kember worktree changes; stage only this requested project change.
- Do not access Git history or the expired archive to infer scientific results; the archive is only touched for the explicitly confirmed PDF deletion list.
- Before every deletion, validate each target resolves inside the project root and obtain confirmation of the exact target list.
- Do not save or regenerate any `.pdf`; static PNG/Markdown/CSV/JSON outputs remain authoritative.
- Do not claim final FDR findings while the analysis-object set is not frozen.

---

### Task 1: Make correction explicitly deferred in primary localisation

**Files:**
- Modify: `cross_dataset_rest_fc/src/localisation.py`
- Test: `cross_dataset_rest_fc/tests/test_localisation.py`

**Interfaces:**
- `fit_local_effects(tensor, contrasts)` continues to return the row-level localisation table.
- The returned table retains `p_raw`, `q_bh`, and adds an explicit correction-state field; `q_bh` is null for every row until a separately implemented finalisation step receives a frozen analysis-object set.
- No implicit `_apply_planned_bh` call remains in model fitting.

- [ ] **Step 1: Write the failing regression tests**

  Replace tests that expect planned BH values with assertions that a complete family still has null `q_bh`, an explicit `deferred` correction status, and metadata stating that the analysis-object set is not frozen. Add a test that the result does not expose planned family sizes as evidence of a final correction scope.

- [ ] **Step 2: Run the focused localisation tests and confirm the expected failure**

  Run:

  ```bash
  /home/yojiao/.conda/envs/py314/bin/python -m pytest stanford_jjtms_multiscale_hippocampal_exploration/cross_dataset_rest_fc/tests/test_localisation.py -q
  ```

  Expected: the old implementation fails because it populates `q_bh` and exposes the old BH-family assumptions.

- [ ] **Step 3: Implement the minimal deferred-correction behavior**

  Remove the implicit planned-BH application and its `planned_sizes` bookkeeping from `fit_local_effects`. Keep raw model inference unchanged. Add a stable output status such as `deferred_until_analysis_set_freeze` and result attributes documenting `multiplicity_correction="deferred"` and `analysis_object_set="not_frozen"`. Keep `q_bh` in the schema as null for compatibility, but do not calculate it.

- [ ] **Step 4: Run the focused tests and the full target test suites**

  Run the focused test first, then:

  ```bash
  /home/yojiao/.conda/envs/py314/bin/python -m pytest stanford_jjtms_multiscale_hippocampal_exploration/cross_dataset_rest_fc/tests stanford_jjtms_multiscale_hippocampal_exploration/external_group_atlas_analysis/tests -q
  ```

  Expected: all tests pass with no q-values assigned by localisation.

---

### Task 2: Remove q-based final-claim language from reporting and exploratory review

**Files:**
- Modify: `cross_dataset_rest_fc/src/reporting.py`
- Modify: `cross_dataset_rest_fc/src/plotting.py`
- Modify: `cross_dataset_rest_fc/src/plot_core_results.py`
- Modify: `cross_dataset_rest_fc/src/build_broad_signal_review.py`
- Modify: `cross_dataset_rest_fc/tests/test_reporting.py`
- Modify: `cross_dataset_rest_fc/tests/test_localisation.py`

**Interfaces:**
- Reports say that q-values are intentionally deferred and do not count q<.05/q<.10 rows.
- Figures show effects, confidence intervals, and raw p-values only; no q-threshold marker legend or q-based selection remains.
- Exploratory signal review uses effect magnitude and direction stability only; it cannot select a secondary layer using q-values.

- [ ] **Step 1: Add failing report/selection tests**

  Assert that the technical report contains the deferred-correction statement, that q-threshold language is absent from chart metadata, and that broad-review selections remain valid when every `q_bh` is null.

- [ ] **Step 2: Run the focused reporting tests and confirm failure**

  Run:

  ```bash
  /home/yojiao/.conda/envs/py314/bin/python -m pytest stanford_jjtms_multiscale_hippocampal_exploration/cross_dataset_rest_fc/tests/test_reporting.py -q
  ```

  Expected: current report and exploratory code still refers to q thresholds and q-supported secondary selection.

- [ ] **Step 3: Implement deferred-correction reporting**

  Update report text, artifact tooltips, chart maps, captions, and broad-review selection rules so raw p-values are clearly non-corrected and q-values are not interpreted. Preserve effect/CI/direction-stability summaries as exploratory candidates.

- [ ] **Step 4: Run the focused and full target tests**

  Expected: all reporting, localisation, atlas, and integration tests pass.

---

### Task 3: Enforce PNG-only figure generation and remove PDF metadata

**Files:**
- Modify: `cross_dataset_rest_fc/src/plotting.py`
- Modify: `cross_dataset_rest_fc/src/plot_core_results.py`
- Modify: `cross_dataset_rest_fc/src/build_broad_signal_review.py`
- Modify: `external_group_atlas_analysis/src/atlas_compare.py`
- Modify: `kember_metrics/src/plot_contrasts.py`
- Modify: `kember_metrics/src/contrasts.py`
- Modify: `external_group_atlas_analysis/tests/test_atlas_compare.py`
- Modify: `kember_metrics/tests/test_contrasts.py`
- Modify: `kember_metrics/tests/test_area_summary.py`

**Interfaces:**
- Every figure renderer returns and records PNG paths only.
- Figure provenance and run manifests contain no PDF path, PDF hash, or PDF output key.
- Re-running any active renderer cannot create a PDF sibling.

- [ ] **Step 1: Add failing PNG-only regression assertions**

  Update/add tests that render representative cross-FC, atlas, and Kember figures and assert a PNG exists, the returned/provenance schema has no PDF fields, and the PDF sibling does not exist.

- [ ] **Step 2: Run the focused renderer tests and confirm failure**

  Run the affected test nodes; current writers are expected to create PDF files and return PDF metadata.

- [ ] **Step 3: Remove all PDF save calls and provenance fields**

  Keep one PNG `savefig` per figure. Remove PDF paths from Kember figure manifests, direct-contrast tables, and run-manifest outputs. Update report figure lists and active documentation to say PNG-only.

- [ ] **Step 4: Run renderer tests and compile all changed modules**

  Use the explicit py314 interpreter and verify no active source path contains a PDF save call.

---

### Task 4: Regenerate active outputs without q-values or PDFs

**Files:**
- Regenerate: `cross_dataset_rest_fc/results/primary_screen/`
- Regenerate: `cross_dataset_rest_fc/reports/primary_screen/`
- Regenerate: `cross_dataset_rest_fc/figures/primary_screen/`
- Regenerate: `cross_dataset_rest_fc/results/broad_signal_review/`
- Regenerate: `cross_dataset_rest_fc/reports/broad_signal_review/`
- Regenerate: `cross_dataset_rest_fc/figures/broad_signal_review/`
- Regenerate: current Kember/atlas PNG-only manifests if their runners are invoked by the changed tests or workflow.

**Interfaces:**
- Primary output retains raw p, coefficient, CI, and effect-size fields but has null q-values plus deferred status.
- Output closure and manifests contain PNG/Markdown/CSV/JSON only.

- [ ] **Step 1: Run the corrected primary workflow with the canonical input snapshot**

  Run the existing primary entry point with `SYSTEM_NAME=idas`, the explicit py314 interpreter, `--run-id primary_screen`, seed `20260824`, and `n_perm=499`; verify the runner completes without PDF creation.

- [ ] **Step 2: Regenerate the effect-first review from the corrected primary table**

  Run its existing explicit py314 entry point and verify no q-based selection or PDF output is created.

- [ ] **Step 3: Update synchronized documentation and manifests**

  Update the project README, cross-dataset README, technical report, source notes, chart map, restructure record, and Kember/atlas figure documentation so correction is deferred and PNG-only output is authoritative.

- [ ] **Step 4: Validate output closure**

  Assert all primary q-values are null, all correction statuses are deferred, no output closure path ends in `.pdf`, no active project PDF remains after the confirmed deletion, and output hashes match the regenerated manifest.

---

### Task 5: Delete the confirmed PDF target list and commit

**Files:**
- Delete only the exact PDF paths presented to the user for confirmation under the project root, including active and explicitly included archived PDF figures.

- [ ] **Step 1: Present the exact target list and wait for confirmation**

  Do not run any deletion command until the user confirms the complete list. Validate every target with `realpath` and ensure it is inside `stanford_jjtms_multiscale_hippocampal_exploration`.

- [ ] **Step 2: Delete only the confirmed paths**

  Use explicit paths, never a broad wildcard or recursive delete. Recheck that each target is absent and no non-PDF figure was affected.

- [ ] **Step 3: Run final verification**

  Run the full target pytest suite, py_compile, `git diff --check`, output-hash validation, and a repository PDF inventory returning zero paths under the project.

- [ ] **Step 4: Stage only this requested project change and commit**

  Preserve unrelated Kember worktree changes unless they are part of the explicit PNG-only implementation; commit with a message describing deferred multiplicity correction and PNG-only figures.

