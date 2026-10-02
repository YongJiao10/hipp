# Stanford Exploration Structure Rebuild Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Rebuild the Stanford/JJTMS exploration directory into three non-overlapping workflows, remove the ambiguous `fast_rest` branch, and make every Kember surface-area result use one article-aligned native-area implementation.

**Architecture:** Keep `cross_dataset_rest_fc/` as the primary Rest-FC workflow. Merge the Kember metric extraction, native-area audit, direct contrasts, and figures into one `kember_metrics/` workflow; move the current Kember code out of `fast_rest/` and remove the obsolete Stanford-only Task 1–4 workflow. Keep `external_four_group_atlas_analysis/` as a separate group-level geometric description, with output fields explicitly namespaced as atlas geometry rather than Kember metrics.

**Tech Stack:** Python 3.14 environment at `/home/yojiao/.conda/envs/py314/bin/python`, NumPy, pandas, SciPy, nibabel, Connectome Workbench `wb_command`, pytest, JSON/TSV/CSV manifests, Markdown documentation.

**Spec:** `stanford_jjtms_multiscale_hippocampal_exploration/study_spec.md`, the Kember method evidence in `stanford_jjtms_multiscale_hippocampal_exploration/literature_and_kember_method_notes.md`, and the primary-source research memo produced during this task.

## Global Constraints

- Kember system surface area is the sum of consumed native HippUnfold per-vertex `surfareacorobl` values in mm² after label transfer to the native surface.
- The consumed native area metric, native surface, native labels, label-transfer method, density evidence, and input hashes must be recorded together.
- The midthickness triangle one-third-area calculation is not a Kember endpoint; it may only be used by a separately named group-geometry description.
- Kember outputs contain `surface_area_mm2`, `surface_area_fraction`, `total_surface_area_mm2`, and explicit area provenance; they do not contain `transition_area_*` as Kember metrics.
- Active local workflow paths use semantic names only; no `_vN`, `_attemptN`, or date-suffixed run directory is a current output target.
- The upstream source tag/commit in `source_lock.json` is provenance and remains recorded even though local workflow directory names do not contain version labels.
- Existing input roots remain read-only; no archive content is accessed.
- All Python repository commands use `/home/yojiao/.conda/envs/py314/bin/python`.
- No file or directory deletion occurs until the exact target list has been shown to the user and explicitly confirmed in the current session.
- The checkpoint commit `7dc8011` is the rollback baseline for this rebuild.

---

### Task 1: Freeze the evidence-based method decision

**Files:**
- Create: `stanford_jjtms_multiscale_hippocampal_exploration/docs/analysis_map.md`
- Create: `stanford_jjtms_multiscale_hippocampal_exploration/docs/kember_area_method.md`
- Create: `stanford_jjtms_multiscale_hippocampal_exploration/docs/restructure_record.md`
- Modify: `stanford_jjtms_multiscale_hippocampal_exploration/README.md`
- Modify: `stanford_jjtms_multiscale_hippocampal_exploration/literature_and_kember_method_notes.md`
- Modify: `stanford_jjtms_multiscale_hippocampal_exploration/study_spec.md`
- Test: `stanford_jjtms_multiscale_hippocampal_exploration/kember_metrics/tests/test_document_status.py`

**Interfaces:**
- Produces the single written decision used by all later code: `area_source = hippunfold_native_surfareacorobl`.
- Produces the workflow map: `cross_dataset_rest_fc` = primary FC, `kember_metrics` = Kember subject metrics, `external_four_group_atlas_analysis` = group-level geometric description.

- [ ] **Step 1: Record the primary-source evidence.**

  Write a short table in `docs/kember_area_method.md` comparing:

  | Candidate | Input | Operation | Status |
  | --- | --- | --- | --- |
  | `fast_rest/src/mesh.py` | consumed midthickness coordinates/faces | triangle area / 3 per vertex, then sum by label | geometric approximation, not Kember |
  | `kember_area/src/kember_method_area.py` | native HippUnfold `surfareacorobl` and native labels | sum native mm² values by system | canonical Kember implementation |

  Cite the Kember Nature Communications article, the author release/source data, and HippUnfold documentation. State explicitly that the native metric is selected because it is the metric consumed by the Kember-aligned workflow and preserves the upstream per-vertex area convention.

- [ ] **Step 2: Define the field contract.**

  Document these exact fields:

  ```text
  surface_area_mm2
  surface_area_fraction
  total_surface_area_mm2
  total_valid_surface_area_mm2
  total_labeled_area_mm2
  unassigned_surface_area_mm2
  vertex_area_source
  vertex_area_metric_path
  vertex_area_convention
  ```

  Define `surface_area_fraction = surface_area_mm2 / total_valid_surface_area_mm2`; state that the fraction is secondary context and the absolute area is the Kember-aligned primary endpoint.

- [ ] **Step 3: Write the final workflow map.**

  `docs/analysis_map.md` must contain one table with source code, inputs, estimand, current outputs, and forbidden cross-usage for all three workflows. It must say that group-atlas `atlas_region_area_mm2` is not interchangeable with Kember `surface_area_mm2`.

- [ ] **Step 4: Add document contract tests.**

  Test that the method document names `surfareacorobl`, rejects `transition_area` as a Kember endpoint, and states that `compute_vertex_area` is geometric-only. Test that the analysis map has exactly the three active workflow names.

- [ ] **Step 5: Run the document tests.**

  Run:

  ```bash
  /home/yojiao/.conda/envs/py314/bin/python -m pytest \
    stanford_jjtms_multiscale_hippocampal_exploration/kember_metrics/tests/test_document_status.py -q
  ```

  Expected: the new tests fail before the documents and pass after the documents are written.

---

### Task 2: Create the unified Kember workflow package

**Files:**
- Rename: `stanford_jjtms_multiscale_hippocampal_exploration/kember_area/` → `stanford_jjtms_multiscale_hippocampal_exploration/kember_metrics/`
- Move and modify: `fast_rest/src/kember_template_features.py` → `kember_metrics/src/feature_build.py`
- Move and modify: `fast_rest/src/direct_contrasts.py` → `kember_metrics/src/contrasts.py`
- Move and modify: `fast_rest/src/kember_direct_contrast_visualization.py` → `kember_metrics/src/plot_contrasts.py`
- Move and modify: `fast_rest/src/inventory.py` → `kember_metrics/src/inputs.py`
- Move and modify: `fast_rest/src/clinical_rosters.py` → `kember_metrics/src/rosters.py`
- Move and modify: `kember_area/src/kember_method_area.py` → `kember_metrics/src/surface_area.py`
- Move and modify: `kember_area/src/run_kember_method_area.py` → `kember_metrics/src/native_label_alignment.py`
- Move and modify: `kember_area/src/input_contract.py` → `kember_metrics/src/input_contract.py`
- Move and modify: `kember_area/src/source_lock.py` → `kember_metrics/src/source_lock.py`
- Create: `kember_metrics/config/analysis_config.json`
- Modify: `kember_metrics/source_lock.json`
- Modify: all moved tests under `kember_metrics/tests/`

**Interfaces:**
- `surface_area.compute_system_surface_area(vertex_area_mm2, labels, valid_mask=None) -> dict[str, Any]` remains the sole Kember area aggregation function.
- `native_label_alignment.align_labels_to_native(...) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]` transfers 512-source labels to the native surface with `ADAP_BARY_AREA` and returns the native valid ROI plus provenance.
- `feature_build.compute_kember_metrics(...)` receives a native area vector explicitly; it never calls a mesh-area reconstruction function.
- `feature_build.run_feature_build(config_path, output_dir, ...)` writes one combined subject-level feature table containing both `hcp151_template_matching` and `hcp151_fixed` rows.
- `contrasts.run_direct_contrasts(feature_table, config, output_dir)` writes one combined direct-contrast table with a `method` column rather than parallel duplicate output trees.

- [ ] **Step 1: Add the failing native-area integration test.**

  Add a test using coordinates/faces whose geometric area differs from a supplied native vector. Assert that the Kember feature result equals the supplied native vector sum, records `vertex_area_source == "hippunfold_native_surfareacorobl"`, and does not equal the geometric reconstruction.

- [ ] **Step 2: Run the new integration test to verify it fails.**

  Run:

  ```bash
  /home/yojiao/.conda/envs/py314/bin/python -m pytest \
    stanford_jjtms_multiscale_hippocampal_exploration/kember_metrics/tests/test_native_area_integration.py -q
  ```

  Expected: FAIL because the current Kember feature builder still imports `compute_vertex_area` from `fast_rest/src/mesh.py`.

- [ ] **Step 3: Implement the shared native-area path.**

  Adapt `surface_area.py` from the current pure aggregation function. Preserve its validation for finite nonnegative one-dimensional native metric values and integer labels. Make the returned dictionary include the exact field names from Task 1.

- [ ] **Step 4: Integrate native labels and native area into feature extraction.**

  For every supported `method × subject × smoothing × hemisphere` combination:

  1. load the consumed 512-source labels and source valid mask;
  2. load the native coronal midthickness surface and native `surfareacorobl` metric;
  3. transfer labels to the native surface with `ADAP_BARY_AREA` using unfolded midthickness registration surfaces;
  4. compute Kember area from native labels and native metric;
  5. retain the existing 512 source time-series path for the functional metrics, with separate source/native vertex-count fields;
  6. write input paths, density evidence, transfer provenance, and metric hashes into the feature manifest.

  The feature builder must have no import or call to `compute_vertex_area`.

- [ ] **Step 5: Consolidate configuration.**

  Create `kember_metrics/config/analysis_config.json` with the current Stanford input roots, 2/4 mm smoothing, L/R hemispheres, the two supported label methods, six direct contrasts, native metric filename pattern, and stable output directories:

  ```text
  kember_metrics/work/features/
  kember_metrics/results/direct_contrasts/
  kember_metrics/figures/direct_contrasts/
  kember_metrics/reports/direct_contrasts/
  ```

  Remove the fixed-only legacy contrast configuration from the active code path.

- [ ] **Step 6: Migrate and adapt tests.**

  Keep pure area tests, native input-contract tests, source-lock tests, Kember functional metric tests, direct contrast tests, visualization tests, and end-to-end manifest tests. Update imports and assertions to the new package name and stable output paths. Add a contract asserting that only `hcp151_template_matching` and `hcp151_fixed` are active in the current combined feature table.

- [ ] **Step 7: Run the unified Kember unit suite.**

  Run:

  ```bash
  /home/yojiao/.conda/envs/py314/bin/python -m pytest \
    stanford_jjtms_multiscale_hippocampal_exploration/kember_metrics/tests -q
  ```

  Expected: all unit and contract tests pass; no test may read archive content.

---

### Task 3: Rebuild current Kember outputs under one stable path

**Files:**
- Create at runtime: `stanford_jjtms_multiscale_hippocampal_exploration/kember_metrics/work/features/`
- Create at runtime: `stanford_jjtms_multiscale_hippocampal_exploration/kember_metrics/results/direct_contrasts/`
- Create at runtime: `stanford_jjtms_multiscale_hippocampal_exploration/kember_metrics/figures/direct_contrasts/`
- Create at runtime: `stanford_jjtms_multiscale_hippocampal_exploration/kember_metrics/reports/direct_contrasts/`
- Modify: `kember_metrics/README.md`
- Modify: `kember_metrics/source_lock.json`

**Interfaces:**
- The feature manifest is the sole source for the direct-contrast table.
- The direct-contrast table contains one row per `contrast × metric × method × smoothing × hemisphere × system` and retains all six case-vs-NTHC contrasts.
- Reports and figures reference only the stable `kember_metrics` paths.

- [ ] **Step 1: Add output-closure tests.**

  Assert that every generated output path is below `kember_metrics/`, that the feature table and contrast table have matching method/smoothing/hemisphere/system keys, and that every area row has native metric provenance.

- [ ] **Step 2: Run the full feature build with the explicit interpreter.**

  Run:

  ```bash
  /home/yojiao/.conda/envs/py314/bin/python \
    stanford_jjtms_multiscale_hippocampal_exploration/kember_metrics/src/feature_build.py \
    --config stanford_jjtms_multiscale_hippocampal_exploration/kember_metrics/config/analysis_config.json \
    --output-dir stanford_jjtms_multiscale_hippocampal_exploration/kember_metrics/work/features
  ```

  Expected: one current feature table, status table, and manifest; no `_vN`, `_attemptN`, or dated output directory is created.

- [ ] **Step 3: Run the combined direct contrasts.**

  Run:

  ```bash
  /home/yojiao/.conda/envs/py314/bin/python \
    stanford_jjtms_multiscale_hippocampal_exploration/kember_metrics/src/contrasts.py \
    --config stanford_jjtms_multiscale_hippocampal_exploration/kember_metrics/config/analysis_config.json \
    --feature-table stanford_jjtms_multiscale_hippocampal_exploration/kember_metrics/work/features/subject_features.tsv \
    --output-dir stanford_jjtms_multiscale_hippocampal_exploration/kember_metrics/results/direct_contrasts
  ```

  Expected: the result table, coverage, membership, figure sources, manifest, and report all describe the same current feature-table hash.

- [ ] **Step 4: Render figures and report.**

  Generate figures below `kember_metrics/figures/direct_contrasts/` and the analysis summary/provenance below `kember_metrics/reports/direct_contrasts/`. The report must state that absolute native area is the Kember-aligned endpoint, relative area is contextual, and transition/geometric boundary measures are absent.

- [ ] **Step 5: Verify the rebuilt outputs.**

  Run the unified Kember tests plus a manifest check using the explicit interpreter. Expected: all current output paths exist, hashes match, no stale `fast_rest` or `kember_area` path is indexed, and no Kember row has `vertex_area_source == "consumed_midthickness_geometry"`.

---

### Task 4: Namespace the separate group-atlas geometry workflow

**Files:**
- Modify: `stanford_jjtms_multiscale_hippocampal_exploration/external_four_group_atlas_analysis/src/atlas_compare.py`
- Modify: `stanford_jjtms_multiscale_hippocampal_exploration/external_four_group_atlas_analysis/src/run_atlas_analysis.py`
- Modify: `stanford_jjtms_multiscale_hippocampal_exploration/external_four_group_atlas_analysis/README.md`
- Modify: `stanford_jjtms_multiscale_hippocampal_exploration/external_four_group_atlas_analysis/tests/test_atlas_compare.py`
- Regenerate: `external_four_group_atlas_analysis/results/atlas_descriptive/`
- Regenerate: `external_four_group_atlas_analysis/figures/atlas_descriptive/`
- Regenerate: `external_four_group_atlas_analysis/reports/atlas_descriptive/`

**Interfaces:**
- Group-atlas area fields are named `atlas_region_area_mm2` and `atlas_region_area_fraction`.
- The README and manifest state `area_source = group_atlas_consumed_geometry`.
- No group-atlas output is imported by the Kember feature builder.

- [ ] **Step 1: Add field-name tests.**

  Test that the atlas descriptive table uses `atlas_region_area_mm2`, never bare `area_mm2`, and that its documentation says it is not a Kember endpoint.

- [ ] **Step 2: Rename the geometry-only fields in code and report builders.**

  Keep the existing geometric triangle-area calculation only inside the atlas workflow, but use explicit names and provenance so a search for `surface_area_mm2` identifies only Kember’s native metric endpoint.

- [ ] **Step 3: Regenerate and verify the current atlas output.**

  Run the existing atlas command with `--run-id atlas_descriptive` using py314. Verify current tables/figures/reports have the new names and no old `region_area.csv` schema is retained.

---

### Task 5: Remove obsolete workflow names and stale artifacts after confirmation

**Files:**
- Modify: `stanford_jjtms_multiscale_hippocampal_exploration/README.md`
- Modify: `stanford_jjtms_multiscale_hippocampal_exploration/rest_fc_primary_analysis_spec.md`
- Modify: `stanford_jjtms_multiscale_hippocampal_exploration/cross_dataset_rest_fc/README.md`
- Modify: `stanford_jjtms_multiscale_hippocampal_exploration/external_four_group_atlas_analysis/README.md`
- Create: `stanford_jjtms_multiscale_hippocampal_exploration/docs/restructure_record.md`

**Deletion gate:** Before any deletion command, show the user the exact resolved target list and wait for explicit confirmation. The proposed list is:

```text
stanford_jjtms_multiscale_hippocampal_exploration/fast_rest/
stanford_jjtms_multiscale_hippocampal_exploration/kember_area/results/pilot_sub-1001_2mm_native_area_20260828T000000Z/
stanford_jjtms_multiscale_hippocampal_exploration/cross_dataset_rest_fc/results/primary_screen_v1_20260824_attempt4/
stanford_jjtms_multiscale_hippocampal_exploration/cross_dataset_rest_fc/results/primary_screen_v1_20260824_attempt4_attempt2/
stanford_jjtms_multiscale_hippocampal_exploration/cross_dataset_rest_fc/results/primary_screen_v1_20260824_attempt5/
stanford_jjtms_multiscale_hippocampal_exploration/cross_dataset_rest_fc/results/primary_screen_v1_20260824_attempt7/
stanford_jjtms_multiscale_hippocampal_exploration/cross_dataset_rest_fc/figures/primary_screen/core_results_network_localisation_v2.pdf
stanford_jjtms_multiscale_hippocampal_exploration/cross_dataset_rest_fc/figures/primary_screen/core_results_network_localisation_v2.png
stanford_jjtms_multiscale_hippocampal_exploration/cross_dataset_rest_fc/figures/primary_screen/core_results_network_localisation_v3.pdf
stanford_jjtms_multiscale_hippocampal_exploration/cross_dataset_rest_fc/figures/primary_screen/core_results_network_localisation_v3.png
stanford_jjtms_multiscale_hippocampal_exploration/cross_dataset_rest_fc/figures/primary_screen/core_results_network_localisation_v4.pdf
stanford_jjtms_multiscale_hippocampal_exploration/cross_dataset_rest_fc/figures/primary_screen/core_results_network_localisation_v4.png
stanford_jjtms_multiscale_hippocampal_exploration/implementation_plan.md
stanford_jjtms_multiscale_hippocampal_exploration/rest_fc_primary_analysis_implementation_plan.md
stanford_jjtms_multiscale_hippocampal_exploration/rest_fmri_promising_first_spec.md
stanford_jjtms_multiscale_hippocampal_exploration/rest_fmri_promising_first_plan.md
stanford_jjtms_multiscale_hippocampal_exploration/branch_spec_corrections.md
```

The two current Kember result families under `fast_rest` are rebuilt before this gate, so deleting `fast_rest/` removes stale code and stale outputs rather than the canonical rebuilt result. The source-lock upstream tag is not in this deletion list.

- [ ] **Step 1: Update all active documentation and source references.**

  Replace `fast_rest` references with `kember_metrics` only where they refer to migrated Kember code; remove references to the deleted old Task 1–4 workflow. Preserve scientific conclusions in the new method and restructure documents.

- [ ] **Step 2: Present the resolved deletion list to the user.**

  Resolve every path from the repository root, verify each is below `/home/yojiao/hpchome/hipp`, and report the list and reason. Do not delete until the user confirms this exact list.

- [ ] **Step 3: Delete only the confirmed targets.**

  Use explicit paths restricted to the current working tree. Record the deletion operation in `docs/restructure_record.md` with the confirmation text and checkpoint commit `7dc8011`.

- [ ] **Step 4: Run a stale-name scan.**

  Run:

  ```bash
  rg -n --glob '*.py' --glob '*.md' --glob '*.json' --glob '*.tsv' \
    'fast_rest|kember_area|_v[0-9]+|_attempt[0-9]+' \
    stanford_jjtms_multiscale_hippocampal_exploration
  ```

  Expected: no active code, current manifest, or current documentation points to deleted local workflow names. The only allowed version-like text is an explicitly labeled upstream provenance tag/commit in `kember_metrics/source_lock.json` and its method documentation.

---

### Task 6: Final verification and checkpoint

**Files:**
- Modify: `stanford_jjtms_multiscale_hippocampal_exploration/docs/restructure_record.md`
- Verify: all active workflow tests and output manifests

- [ ] **Step 1: Run the required interpreter-context check.**

  Run:

  ```bash
  /home/yojiao/.conda/envs/py314/bin/python -c "import sys; print(sys.executable)"
  ```

  Expected: `/home/yojiao/.conda/envs/py314/bin/python`.

- [ ] **Step 2: Run focused tests.**

  Run the Kember, atlas, and primary Rest-FC suites with py314. Expected: all tests that do not require unavailable external input pass; any unavailable-input failure must be recorded as an input availability issue rather than hidden.

- [ ] **Step 3: Run static and manifest checks.**

  Verify JSON parsing, SHA-256 references, output closure, no duplicate active output targets, no stale local version directories, and `git diff --check`.

- [ ] **Step 4: Update the restructure record.**

  Record final directory map, area decision, exact deletions, verification commands, pass/fail counts, and any remaining missing-input limitation.

- [ ] **Step 5: Commit the completed rebuild.**

  After all verifications pass or remaining limitations are explicitly documented, create:

  ```bash
  git add stanford_jjtms_multiscale_hippocampal_exploration
  git commit -m "refactor: unify exploration workflows and Kember area"
  ```

  The final handoff must include the new commit hash and point to the analysis map and Kember area decision documents.
