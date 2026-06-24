# Kong2022 Shared Input Root Rename Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make `shared_inputs/routea` and `shared_inputs/surface` the canonical cross-dataset defaults and migrate the Iowa7T data accordingly.

**Architecture:** Change the two central resolver functions so all maintained callers inherit the new defaults. Move the Iowa7T subject trees into the already-empty canonical roots, rerun the individual workflow to refresh absolute paths, rebuild overviews, verify all artifacts, then remove obsolete `_151` parents.

**Tech Stack:** Python 3.14, pathlib, pytest, NumPy, nibabel, filesystem move operations.

---

### Task 1: Lock The Resolver Contract

**Files:**
- Modify: `tests/core_workflow/test_path_templates.py`
- Modify: `tests/individual_workflow/test_routea_individual_gradients_wta.py`

- [ ] Add assertions that `resolve_kong2022_routea_root()` ends in `shared_inputs/routea` and `resolve_kong2022_surface_root()` ends in `shared_inputs/surface`.
- [ ] Assert the individual parser defaults use those resolver outputs.
- [ ] Run the focused tests and confirm they fail while the resolver still returns `_151` paths.

### Task 2: Change Canonical Defaults

**Files:**
- Modify: `scripts/core_workflow/lib/output_layout.py`
- Modify: `scripts/core_workflow/05_01_compute_hippomaps_gradients.py`

- [ ] Return `routea` and `surface` from the two base resolver functions.
- [ ] Change the RouteA gradient route-to-directory mapping from `routea_151` to `routea`.
- [ ] Keep `routea_ctxreg_151`, `routea_ctxdrop_151`, and `routeb_151` unchanged.
- [ ] Run focused and complete path/individual workflow tests.

### Task 3: Synchronize Maintained Documentation

**Files:**
- Modify: `scripts/individual_workflow/README.md`
- Modify: `scripts/core_workflow/README.md`
- Modify: `docs/hipp_parcellation_network/core_workflow_overview.md`
- Modify: `docs/hipp_parcellation_network/multi_branch_flow.md`
- Modify: `docs/hipp_parcellation_network/core_workflow_step_registry.md`
- Modify: `docs/hipp_parcellation_network/core_workflow_artifact_lineage.md`
- Modify: `docs/hipp_parcellation_network/individual_workflow_routea_gradients_wta.md`
- Modify: `docs/hipp_parcellation_network/step_by_step_flows.md`

- [ ] Replace canonical base RouteA and surface references with `routea` and `surface`.
- [ ] Preserve `_151` names for ctxreg, ctxdrop, and routeb branches.
- [ ] Confirm maintained documentation no longer describes `routea_151` or `surface_151` as the base defaults.

### Task 4: Migrate Iowa7T Data

- [ ] Confirm `routea` and `surface` contain no subject directories.
- [ ] Move `routea_151/sub-04172026` to `routea/sub-04172026`.
- [ ] Move `surface_151/sub-04172026` to `surface/sub-04172026`.
- [ ] Verify reference, FC, surface time-series, and mask files through the new paths.
