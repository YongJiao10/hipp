# Individual Atlas Shallow Layout Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Store individual atlas outputs under a shallow subject-first directory tree.

**Architecture:** Centralize the new layout in the existing output-path helpers so computation and overview code continue sharing one contract. Move run metadata to top-level `_runs`, migrate the Iowa7T result tree, then verify artifacts before deleting obsolete parents.

**Tech Stack:** Python 3.14, pathlib, pytest, NumPy, nibabel.

---

### Task 1: Lock The New Path Contract

**Files:**
- Modify: `tests/individual_workflow/test_routea_individual_gradients_wta.py`

- [ ] Change path expectations to `sub-<id>/gradient/<smoothing>/<hemi>` and `sub-<id>/wta/<smoothing>/<hemi>`.
- [ ] Add a manifest discovery test using the new gradient path.
- [ ] Run the focused tests and confirm they fail against the old implementation.

### Task 2: Implement The Shallow Layout

**Files:**
- Modify: `scripts/individual_workflow/routea_individual_gradients_wta.py`
- Modify: `scripts/individual_workflow/01_monitor_individual_run.py`

- [ ] Update the shared output-path helper.
- [ ] Update manifest scanning to search `sub-*/gradient/*/*/combo_manifest.json`.
- [ ] Move run metadata paths to `individual_atlas/_runs`.
- [ ] Run focused and full individual workflow tests.

### Task 3: Synchronize Documentation

**Files:**
- Modify: `scripts/individual_workflow/README.md`
- Modify: `docs/hipp_parcellation_network/individual_workflow_routea_gradients_wta.md`
- Modify: `docs/hipp_parcellation_network/core_workflow_overview.md`

- [ ] Replace method-first examples with the subject-first layout.
- [ ] Confirm no maintained document references `gradient_spectral_k3/subjects` or `wta_top/subjects`.

