# Selective CtxReg Reconciliation Implementation Plan

> **For agentic workers:** Execute the tasks in order, with a review gate after each task. Do not commit: the user will inspect the resulting working tree.

**Goal:** Integrate only the demonstrably correct, reusable CtxReg capabilities from `ctxreg_worktree` into the formal repository while preserving every unrelated or superior main-tree capability and keeping code, documentation, manifests, and formal products consistent.

**Architecture:** Treat `8cb3537` as the shared baseline and perform a hunk-level three-way reconciliation, not a Git branch merge. Main remains authoritative for unrelated paths and for stronger existing contracts; worktree contributes only independently verified CtxReg behavior. Formal products are downstream of the selected source contract: a PNG, JSON, TSV, NPY, or cache is retained only when its exact manifest proves compatible with the selected code, inputs, route, assignment variant, and output root.

**Tech Stack:** Python 3.14 at `/home/yojiao/.conda/envs/py314/bin/python`, pytest, NumPy, nibabel, Workbench, JSON manifests, Git working-tree diffs.

**Spec:** `AGENTS.md`, `docs/environment/agent_runtime_contract.md`, and the user's reconciliation criteria in this conversation.

## Global Constraints

- Do not create commits, stage files, merge branches, move outputs, or delete anything.
- Preserve the main-tree staged archive relocation and never inspect archive content.
- An unchanged main-tree path remains unchanged unless a reviewed worktree hunk demonstrably corrects it.
- A newer timestamp is only chronology evidence; choose behavior using contracts, provenance, tests, and scientific semantics.
- Prohibit parent-directory output-root fallbacks, candidate-order selection, glob-order selection, and implicit artifact substitution.
- Formal HCP control routes are exactly `routea_ctxreg` and `routea_ctxdrop`; do not create dated, recovered, regenerated, or suffix-based result roots.
- Any behavior/path/workflow change must receive matching documentation in the same change.
- Output trees are read-only. A product that needs rebuilding must be listed for user approval before any write is attempted.
- Preserve 4mm, CtxDrop, and Stanford capabilities; add CtxReg as an explicit branch rather than narrowing existing scope.

---

### Task 1: Establish the reconciliation guardrails

**Files:**

- Modify: `tests/core_workflow/test_hippomaps_gradient_route.py`
- Modify: `tests/core_workflow/test_ctxreg_variable_runs.py`
- Verify: `scripts/core_workflow/lib/output_layout.py`

**Interfaces:**

- `resolve_outputs_root(dataset: str) -> Path` must always resolve below the checkout that imports it.
- A selected CtxReg run must use explicit, contiguous recorded run metadata.

- [ ] **Step 1: Add a failing worktree-isolation assertion.**

```python
def test_outputs_root_is_local_to_importing_checkout() -> None:
    assert output_layout.resolve_outputs_root("hcp") == REPO_ROOT / "outputs_hcp"
```

- [ ] **Step 2: Run the two focused tests before implementation.**

Run:

```bash
env OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 \
  /home/yojiao/.conda/envs/py314/bin/python -m pytest -q -p no:cacheprovider \
  tests/core_workflow/test_hippomaps_gradient_route.py \
  tests/core_workflow/test_ctxreg_variable_runs.py
```

Expected: the old worktree output-root fallback fails the isolation assertion.

- [ ] **Step 3: Keep main's exact `resolve_outputs_root` implementation.**

```python
def resolve_outputs_root(dataset: str = DEFAULT_DATASET) -> Path:
    return REPO_ROOT / f"outputs_{_validate_dataset(dataset)}"
```

Do not transfer the worktree `REPO_ROOT.parent`, `.git`-file, or `exists()` branches.

- [ ] **Step 4: Re-run the focused tests.**

Expected: both pass while a worktree can no longer target main outputs implicitly.

### Task 2: Add explicit signal-branch primitives without narrowing existing routes

**Files:**

- Create: `scripts/core_workflow/lib/branch_config.py`
- Modify: `scripts/core_workflow/lib/route_tags.py`
- Test: `tests/core_workflow/test_hippomaps_gradient_route.py`
- Test: `tests/core_workflow/test_lynch_signal_branch_routing.py`

**Interfaces:**

- `normalize_signal_branch(value: str) -> str` accepts exactly `standard` and `ctxreg`.
- `validate_signal_branch_route(branch: str, route_tag: str) -> str` enforces `standard -> routea` and `ctxreg -> routea_ctxreg`.
- `validate_group_atlas_root_for_route(..., signal_branch: str | None = None)` accepts deliberate nested CtxReg Lynch roots and rejects route mismatch.

- [ ] **Step 1: Add failing tests for standard/CtxReg route agreement and nested CtxReg Lynch output roots.**

```python
with pytest.raises(ValueError, match="Signal branch/route mismatch"):
    validate_signal_branch_route("ctxreg", "routea")
assert validate_signal_branch_route("ctxreg", "routea_ctxreg") == "ctxreg"
```

- [ ] **Step 2: Implement only the pure mapping and nested-root validation hunk.**

Keep comments scope-neutral: core branch primitives must not claim that 4mm, CtxDrop, or Stanford are retired.

- [ ] **Step 3: Run the two focused route tests.**

Expected: explicit branch mismatches fail closed; existing non-CtxReg routes retain their behavior.

### Task 3: Make CtxReg residual and FC materialization exact-input stages

**Files:**

- Modify: `scripts/core_workflow/04_00_materialize_cortex_regressed_surface_timeseries.py`
- Modify: `scripts/core_workflow/04_01_materialize_kong2022_roi_fc.py`
- Test: `tests/core_workflow/test_ctxreg_variable_runs.py`
- Test: `tests/core_workflow/test_lynch_reference_run_metadata.py`

**Interfaces:**

- `--run-metadata-template` selects one caller-declared per-subject JSON path.
- The selected JSON must contain explicit positive run IDs, contiguous start/stop segments, and an exact total equal to time-series frames.
- All validation completes before the stage creates a result directory or writes an FC/residual artifact.

- [ ] **Step 1: Add failing tests for omitted/ambiguous metadata and write-before-validation.**

```python
with pytest.raises(ValueError, match="requires explicit run IDs"):
    load_run_segments(subject_root, run_metadata=metadata_without_ids)
with pytest.raises(ValueError, match="do not cover all"):
    validate_run_segments(bad_segments, n_timepoints=100)
assert not output_dir.exists()
```

- [ ] **Step 2: Replace candidate discovery with the selected template.**

```python
metadata_path = Path(run_metadata_template.format(subject=subject, sub_subject=f"sub-{subject}"))
segments = load_run_segments(subject_root, run_metadata=metadata_path)
validate_run_segments(segments, n_timepoints=time_series.shape[1])
```

Remove the metadata candidate lists and `run-*` directory fallback from formal execution. A legacy compatibility mode, if ever required, must be a separately named CLI mode with a test and provenance field.

- [ ] **Step 3: Preserve explicit mask behavior.**

The formal CtxReg path must require both cortical validity masks. Do not expose all-ones masks through an implicit default.

- [ ] **Step 4: Run the focused run-metadata and CtxReg tests.**

Expected: unequal valid runs pass; missing IDs, mismatched totals, and incomplete masks fail before writes.

### Task 4: Reconcile gradients, group atlas, and overview contracts hunk by hunk

**Files:**

- Modify: `scripts/core_workflow/05_01_compute_hippomaps_gradients.py`
- Modify: `scripts/core_workflow/05_02_run_group_atlas.py`
- Modify: `scripts/core_workflow/06_01_build_group_overview.py`
- Create: `scripts/core_workflow/lib/gradient_assignment_contract.py`
- Test: `tests/core_workflow/test_hippomaps_gradient_route.py`
- Test: `tests/core_workflow/test_lynch_signal_branch_routing.py`
- Test: `tests/visualize/test_gradient_assignment_contract.py`

**Interfaces:**

- A gradient/group-atlas manifest carries both atlas route and `signal_branch`.
- CtxReg hippocampal signals use `routea_ctxreg`; cortex references remain the explicitly mapped standard `routea` sibling.
- `validate_gradient_assignment(run_dir, manifest, variant)` validates the exact declared status JSON, NPY, label-map JSON, GIFTI, and profile before rendering.

- [ ] **Step 1: Add failing anchor-selection tests.**

```python
with pytest.raises(ValueError, match="alignment anchor"):
    select_alignment_anchor(results_for_2mm_only, requested_anchor=None)
anchor, reason = select_alignment_anchor(results_for_2mm_only, requested_anchor=("2mm", "L"))
assert (anchor.smoothing, anchor.hemi, reason) == ("2mm", "L", "explicit")
```

- [ ] **Step 2: Implement an explicit alignment anchor.**

Retain the declared `4mm/L` default only when that exact requested combo exists. For any subset such as 2mm CtxReg, require `--alignment-anchor 2mm:L`; never use `results[0]`.

- [ ] **Step 3: Transfer CtxReg lineage hunks only.**

Transfer signal-branch validation, exact standard-reference sibling mapping, branch-qualified output route, gradient manifest branch checks, canonical manifest normalization, assignment validation, and overview source sidecars.

- [ ] **Step 4: Explicitly retain main's stronger behavior.**

Do not transfer worktree support for `experimental` K selection, active `canonical8`, parent-directory dataset-root scanning, duplicated label-map helper removal, or relaxed CtxDrop provenance. CtxDrop continues to require exactly 2mm/4mm by L/R; an optional one-smoothing overview is permitted only for non-CtxDrop calls.

- [ ] **Step 5: Limit assignment variants to approved formal semantics.**

Initially register only `normalized_top20_count` and `fisherz`. Do not import `weighted_fisherz` into the formal contract.

- [ ] **Step 6: Run the focused gradient, routing, and assignment tests.**

Expected: every requested branch/variant is explicit; mismatch, missing GIFTI/NPY/status, or stale route provenance fails closed.

### Task 5: Add template matching as a verified CtxReg extension

**Files:**

- Modify: `scripts/individual_workflow/03_01_build_hcp151_fixed_template.py`
- Create: `scripts/jj_tms/jj_tms_ctxreg_fixed_template_matching.py`
- Modify: `scripts/jj_tms/README.md`
- Test: `tests/individual_workflow/test_jj_tms_ctxreg_fixed_template_matching.py`

**Interfaces:**

- `build_template(..., template_id: str)` produces a manifest carrying the selected template identity.
- JJ_TMS matching requires a complete `hcp151_routea_ctxreg` template, exact 2mm CtxReg FC files, standard validity masks, and exact den-512 surfaces.

- [ ] **Step 1: Replace environment-dependent tests with temporary fixtures.**

```python
template_root = tmp_path / "template"
write_complete_template_fixture(template_root, template_id="hcp151_routea_ctxreg")
assert load_template(template_root)["manifest"]["template_id"] == "hcp151_routea_ctxreg"
```

- [ ] **Step 2: Make template identity and resume behavior fail closed.**

Require `template_id == "hcp151_routea_ctxreg"`; a malformed or incompatible resume manifest raises an actionable error rather than being silently ignored or overwritten.

- [ ] **Step 3: Preflight all selected subject inputs before creating per-subject output directories.**

The default roster is the existing HC-67 plus Patient-26 manifests; do not add a redundant union roster unless it becomes the caller-selected formal source.

- [ ] **Step 4: Run the hermetic template matching tests.**

Expected: template/FC/mask/surface mismatches fail without reading existing result trees.

### Task 6: Extend visualization for CtxReg without retiring the established grid

**Files:**

- Modify: `scripts/visualize/lib/multidataset_cohorts.py`
- Modify: `scripts/visualize/lib/cortex_group_contrasts.py`
- Modify: `scripts/visualize/run_multidataset_suite.py`
- Modify: `scripts/visualize/template_matching_dominant_subregions/render_template_matching_dominant_subregions.py`
- Create or modify: `scripts/visualize/lib/gradient_surface_panel.py`
- Test: `tests/visualize/test_multidataset_ctxreg_contract.py`
- Test: `tests/visualize/test_multidataset_grid_architecture.py`
- Test: `tests/visualize/test_template_matching_dominant_subregions.py`

**Interfaces:**

- Existing standard grid tags retain 36P and 2mm/4mm behavior.
- CtxReg is an explicit signal branch with exact `routea_ctxreg` paths and no Standard fallback.
- Every cache/contrast key includes signal branch, smoothing, preprocessing variant, method, and selected assignment semantics.

- [ ] **Step 1: Add failing compatibility tests.**

```python
assert GridSpec.from_tag("4mm_36p").signal_branch == "standard"
assert GridSpec.from_tag("2mm_36p_ctxreg").signal_branch == "ctxreg"
with pytest.raises(ValueError, match="routea_ctxreg"):
    resolve_cohort_paths("hcp_7t", GridSpec.from_tag("2mm_36p_ctxreg"))["grad_subpath"].relative_to(ctxreg_root)
```

- [ ] **Step 2: Keep a code-root/data-root boundary explicit.**

Remove all `.git`-file plus parent-output-root inference. A worktree invocation must receive an explicit `--repo-root`/data-root pair; no resolver may silently point at its parent.

- [ ] **Step 3: Preserve existing launcher modes and add CtxReg options.**

Retain legacy and comparison modes, all supported standard grids, and existing Stanford/CtxDrop behavior. Add a CtxReg branch selector; do not restrict the CLI to HCP/JJ_TMS or 2mm/36P.

- [ ] **Step 4: Transfer only strict provenance behavior.**

Keep exact branch lineage checks, assignment validation, source fingerprint sidecars, and branch-qualified cache tags. Reject automatic cache rebuilds during read-only rendering; rebuilding must be a caller-selected command.

- [ ] **Step 5: Generalize template-matching presentation.**

Retain the four standard grid products and add the 2mm/36P CtxReg product. Validate fixed-template semantics from exact combo manifests; do not derive numeric cluster meaning from a different group-atlas assignment.

- [ ] **Step 6: Run visualization contract tests with no rendering writes.**

Expected: standard routes remain selectable, CtxReg paths cannot resolve Standard artifacts, and stale/ambiguous caches fail closed.

### Task 7: Synchronize workflow documentation and formal registries

**Files:**

- Modify: `scripts/core_workflow/README.md`
- Modify: `scripts/core_workflow/step_registry.json`
- Modify: `docs/hipp_parcellation_network/core_workflow_overview.md`
- Modify: `docs/hipp_parcellation_network/core_workflow_step_registry.md`
- Modify: `docs/hipp_parcellation_network/step_by_step_flows.md`
- Modify: `scripts/visualize/README.md`
- Modify: `docs/rendering/gradient_surface_consistency.md` only if its claims survive the final renderer design
- Preserve: `docs/environment/agent_runtime_contract.md` main-tree rule against worktree links and parent output fallbacks

**Interfaces:**

- Documentation names the exact selected run-metadata input, signal branch, route, assignment variant, output root, and rebuild gate.

- [ ] **Step 1: Retain main's strict Lynch metadata documentation.**

Keep the rule that missing run IDs, contradictory metadata, noninteger lengths, and frame-total mismatches fail before outputs are written.

- [ ] **Step 2: Add the final CtxReg invocation contract.**

Document explicit run metadata, CtxReg hippocampal input roots, standard reference/mask roots, explicit alignment anchor for 2mm-only calls, and the branch-qualified formal output paths.

- [ ] **Step 3: Remove unverified release claims.**

Do not copy statements that a figure is generated, a product is READY, or production is approved unless Task 8 validates that exact product.

- [ ] **Step 4: Update the step registry and visual README in the same patch.**

The registry, CLI examples, and output semantics must match the selected implementation exactly; do not document worktree symlinks as a formal runtime feature.

### Task 8: Validate products and decide their disposition without output writes

**Files:**

- Inspect only: `outputs_hcp/`, `outputs_iowa_jj_tms/`, `outputs_stanford/`
- Inspect only: worktree generated assets under `scripts/visualize/`
- Create only if needed: a source-control review note, not a result-tree file

**Interfaces:**

- A retained product must have an exact manifest that matches the selected route, signal branch, assignment variant, source manifests, and formal output root.

- [ ] **Step 1: Build a per-product eligibility table.**

For each CtxReg PNG/manifest/TSV/NPY, record `retain`, `needs approved rebuild`, or `exclude`, with exact reason and source paths.

- [ ] **Step 2: Reject the known invalid worktree products.**

Reject products whose manifest points into `ctxreg_worktree`, whose preflight reports missing inputs, whose selected code path used a parent-root fallback, or whose branch/assignment provenance is incomplete.

- [ ] **Step 3: Treat current main CtxReg products as read-only candidates.**

Do not overwrite or delete them. Retain only those whose manifest verifies against the selected source contract; otherwise classify them as requiring a user-approved rebuild at their exact formal paths.

- [ ] **Step 4: Do not promote `weighted_fisherz`.**

The worktree post-hoc copy-and-rewrite branch remains excluded because it creates a nonformal output root, discovers/copies existing run trees, and changes assignment semantics outside the core CtxReg contract.

### Task 9: Verify the selected integration and prepare a deletion decision

**Files:**

- Verify: all touched source, docs, manifests, and tests
- Do not delete: `ctxreg_worktree` or any output until the user approves an exact target list

**Interfaces:**

- The main working tree contains every accepted source change and no unreviewed worktree-only dependency.

- [ ] **Step 1: Run source integrity checks.**

Run:

```bash
/home/yojiao/.conda/envs/py314/bin/python -m py_compile \
  scripts/core_workflow/04_00_materialize_cortex_regressed_surface_timeseries.py \
  scripts/core_workflow/04_01_materialize_kong2022_roi_fc.py \
  scripts/core_workflow/05_01_compute_hippomaps_gradients.py \
  scripts/core_workflow/05_02_run_group_atlas.py \
  scripts/core_workflow/06_01_build_group_overview.py
```

- [ ] **Step 2: Run the focused core and visualization pytest suites with the explicit py314 interpreter.**

Expected: all selected contracts pass, including worktree-isolation, explicit metadata, signal routing, CtxDrop provenance, assignment validation, and standard-grid compatibility.

- [ ] **Step 3: Inspect the final unstaged diff and product eligibility table with the user.**

No commit is created. Flag every remaining worktree-only source, generated asset, and document as `integrated`, `intentionally excluded`, or `needs approved rebuild`.

- [ ] **Step 4: Present an exact deletion target list.**

Only after all accepted changes are in main and the user explicitly confirms the exact list may the worktree or stale experimental assets be deleted.

## Self-Review

- Scope coverage: the plan covers source selection, output isolation, CtxReg inputs, gradients/atlas/overview lineage, template matching, visualization compatibility, docs, product eligibility, tests, and eventual deletion.
- Main-tree protection: every task retains unrelated main behavior and forbids whole-file overwrites.
- Product consistency: Task 8 blocks artifact promotion until manifests match the selected source contract, and Task 7 prevents release claims before validation.
- No-commit constraint: each task leaves changes unstaged for user inspection.
