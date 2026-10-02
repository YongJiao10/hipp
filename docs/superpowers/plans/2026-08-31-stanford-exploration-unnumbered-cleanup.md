# Stanford Exploration Unnumbered Cleanup Implementation Plan

> **For agentic workers:** Execute this plan inline in the current session. The user has explicitly confirmed the deletion list in the conversation. Use test-first changes for workflow behavior and verify every affected branch before claiming completion.

**Goal:** Make the Stanford/JJ_TMS exploration use one mutable, unversioned current-output location per workflow, remove the obsolete error/patch artifacts confirmed by the user, and allow corrected runs to overwrite current outputs.

**Architecture:** Keep the existing branch layout (`src`, `tests`, `config`, `work`, `results`, `figures`, `reports`) but replace versioned current run IDs with stable semantic names. Generated manifests and documentation must use those stable names. Historical version directories that are not promoted to a current location are removed only after the user-confirmed cleanup step.

**Tech Stack:** Python 3.14 at `/home/yojiao/.conda/envs/py314/bin/python`, pytest, pandas, JSON/CSV/TSV manifests, Markdown documentation.

**Spec:** User request in the current conversation: change the no-overwrite behavior, clear the confirmed old/error contents, and stop using local `vN` run names. Preserve `source_lock.json`'s upstream Kember tag `v1` as external provenance.

## Global Constraints

- Do not access archive content.
- Do not use bare `python`; use `/home/yojiao/.conda/envs/py314/bin/python` for repository Python tasks.
- Keep every deletion within `/home/yojiao/hpchome/hipp/stanford_jjtms_multiscale_hippocampal_exploration` and only delete the paths confirmed by the user.
- Preserve the external Kember source tag `v1` in `source_lock.json`; remove `v1` only from local workflow/directory naming and explanatory prose where it denotes a local revision.
- Existing user changes outside the confirmed targets must remain untouched.

### Task 1: Lock the desired overwrite and naming behavior with failing tests

**Files:**
- Modify: `stanford_jjtms_multiscale_hippocampal_exploration/external_four_group_atlas_analysis/tests/test_atlas_compare.py`
- Modify: `stanford_jjtms_multiscale_hippocampal_exploration/cross_dataset_rest_fc/tests/test_reporting.py`
- Modify: `stanford_jjtms_multiscale_hippocampal_exploration/fast_rest/tests/test_end_to_end.py`
- Modify: `stanford_jjtms_multiscale_hippocampal_exploration/fast_rest/tests/test_direct_contrasts.py`
- Modify: `stanford_jjtms_multiscale_hippocampal_exploration/fast_rest/tests/test_kember_template_metrics.py`
- Modify: `stanford_jjtms_multiscale_hippocampal_exploration/kember_v1_area/tests/test_run_kember_method_area.py`

- [ ] Add one test per output writer that creates a target with an existing output file, invokes the writer again with the same stable run name, and asserts the new content replaces the old content.
- [ ] Replace tests that expect `FileExistsError` for ordinary existing output directories with assertions that path-scope validation still rejects unsafe paths but valid current paths are writable.
- [ ] Add assertions that the new default/current IDs contain no local `vN` or `attemptN` suffix.
- [ ] Run the focused tests with `/home/yojiao/.conda/envs/py314/bin/python -m pytest ... -q` and confirm they fail for the expected old overwrite/version behavior.

### Task 2: Implement stable unversioned current outputs

**Files:**
- Modify: `external_four_group_atlas_analysis/src/atlas_compare.py`
- Modify: `external_four_group_atlas_analysis/src/run_atlas_analysis.py`
- Modify: `cross_dataset_rest_fc/src/reporting.py`
- Modify: `cross_dataset_rest_fc/src/plotting.py`
- Modify: `cross_dataset_rest_fc/src/plot_core_results.py`
- Modify: `cross_dataset_rest_fc/src/build_broad_signal_review.py`
- Modify: `fast_rest/src/handoff.py`
- Modify: `fast_rest/src/direct_contrasts.py`
- Modify: `fast_rest/src/kember_template_features.py`
- Modify: `fast_rest/src/core_visualization.py`
- Modify: `fast_rest/src/plot_area_2mm_summary.py`
- Modify: `kember_v1_area/src/run_kember_method_area.py`

- [ ] Replace local default IDs with stable semantic IDs: `atlas_descriptive`, `primary_screen`, `broad_signal_review`, `task2_features_20260823`, `task3_screen_20260823`, `core_results_20260823`, and `task1_inventory_20260823`.
- [ ] Remove ordinary `FileExistsError` guards from valid output writers; keep path traversal, absolute-target, and input-scope validation.
- [ ] Make valid current output writers create parent directories with `exist_ok=True` and overwrite their declared files deterministically.
- [ ] Update generated manifest/report strings so they no longer describe runs as versioned or sealed attempts.
- [ ] Run the newly changed focused tests and confirm they pass before broader verification.

### Task 3: Update documentation and source references

**Files:**
- Modify: `stanford_jjtms_multiscale_hippocampal_exploration/README.md`
- Modify: `external_four_group_atlas_analysis/README.md`
- Modify: `cross_dataset_rest_fc/README.md`
- Modify: `fast_rest/README.md`
- Modify: `kember_v1_area/README.md`
- Modify: `kember_v1_area/source_audit.md`
- Modify: all non-archive Markdown/JSON/Python references under the Stanford exploration directory that point to the old local run IDs or `kember_v1_area`.

- [ ] Replace “never overwrite”, “no-overwrite”, “versioned run”, and `_attemptN` instructions with the stable-current-output policy.
- [ ] Identify the current output by its semantic directory name, not by `vN`.
- [ ] State that the only retained `v1` is the pinned external Kember source tag in `source_lock.json`.
- [ ] Remove references to the obsolete attempt-specific recovery workflow once its confirmed files are removed.

### Task 4: Promote current outputs and remove confirmed obsolete contents

**Files/paths:**
- Promote the authoritative atlas v7 content to `external_four_group_atlas_analysis/{results,figures,reports}/atlas_descriptive`.
- Promote cross-dataset attempt8 content to `cross_dataset_rest_fc/{results,figures,reports}/primary_screen`.
- Promote broad review content to `cross_dataset_rest_fc/{results,figures,reports}/broad_signal_review`.
- Promote the latest fast-rest v4/v5/v3/correction contents to the stable names listed in Task 2.
- Rename `kember_v1_area` to `kember_area` and update all local references.
- Delete exactly the old version directories, obsolete attempt directories, stale recovery/reporting fragments, patch rejects, and failed screenshot listed by the user in the confirmation immediately preceding this plan.

- [ ] Verify all promoted manifests point to the promoted paths and have no stale local version/attempt IDs.
- [ ] Verify no unapproved deletion target is included.

### Task 5: Full verification and handoff

**Files:**
- No additional source files; inspect the final diff and status.

- [ ] Run each branch's focused pytest suite with the explicit py314 interpreter.
- [ ] Run repository-wide text checks for local `v[0-9]+`, `attempt[0-9]+`, `never overwrite`, `no-overwrite`, and `refusing to overwrite`, excluding the preserved external `source_lock.json` tag and unrelated repository content.
- [ ] Check that all canonical output directories exist and old confirmed targets do not.
- [ ] Report the final changed paths, verification commands, and any remaining intentional upstream provenance version.
