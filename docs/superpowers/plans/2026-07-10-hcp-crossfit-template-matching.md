# HCP Cross-Fitted Template Matching Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a low-compute, family-aware five-fold RouteA/Kong2022 template-matching workflow that produces leakage-free individualized HCP K=3 hippocampal labels and a quantitative fold-robustness overview.

**Architecture:** A dedicated individual-workflow entrypoint creates deterministic family-grouped folds, fits five training-only group templates with the existing group-gradient and surface-spectral APIs, and maps each held-out subject by vertex-to-prototype Pearson correlation. A separate renderer consumes only saved fold artifacts to calculate Dice/ARI/prototype/agreement summaries and a compact long-axis overview.

**Tech Stack:** Python 3.14 environment at `/home/yojiao/.conda/envs/py314/bin/python`, NumPy, SciPy, scikit-learn, nibabel, matplotlib, existing HippoMaps/BrainSpace and core-workflow helpers, pytest.

## Global Constraints

- Work in the cleaned `main` checkout requested by the user; do not create a worktree.
- Maximum concurrent jobs/tasks/processes is 10; the new computation is sequential by default.
- Do not use bare `python`; use `/home/yojiao/.conda/envs/py314/bin/python`.
- HCP production runs require a non-committed external `subject,family_id` map unless `--allow-subject-only-folds` is explicit.
- Every subject must be held out from the functional template that labels it.
- Use the consumed `space-corobl_den-512` surface path and recorded density evidence; do not infer actual density from CLI arguments alone.
- Do not overwrite current `outputs_hcp/work/individual_atlas/` or full-cohort group-atlas products.
- Runtime behavior, paths, and outputs must be documented in the same change.

---

### Task 1: Cross-fit fold and matching primitives

**Files:**
- Create: `scripts/individual_workflow/routea_crossfit_template_matching.py`
- Create: `tests/individual_workflow/test_routea_crossfit_template_matching.py`
- Modify: `scripts/core_workflow/lib/output_layout.py`

**Interfaces:**
- Produces: `assign_grouped_folds(subjects, family_by_subject, n_folds, seed) -> dict[str, int]`
- Produces: `validate_fold_assignment(subjects, family_by_subject, fold_by_subject, n_folds) -> None`
- Produces: `build_cluster_prototypes(group_fc_full, labels, support_mask, k=3) -> np.ndarray`
- Produces: `match_vertices_to_prototypes(fc_fisherz, valid_mask, prototypes) -> tuple[np.ndarray, np.ndarray, np.ndarray]`
- Produces: `resolve_crossfit_template_matching_root(dataset="hcp")` and `resolve_crossfit_template_matching_overview_root(dataset="hcp")`

- [ ] **Step 1: Write failing tests for grouped folds and output roots**

```python
def test_grouped_folds_are_deterministic_balanced_and_keep_families_together():
    subjects = ["01", "02", "03", "04", "05", "06"]
    families = {"01": "a", "02": "a", "03": "b", "04": "c", "05": "d", "06": "e"}
    first = workflow.assign_grouped_folds(subjects, families, n_folds=3, seed=17)
    second = workflow.assign_grouped_folds(subjects, families, n_folds=3, seed=17)
    assert first == second
    assert first["01"] == first["02"]
    counts = np.bincount(list(first.values()), minlength=3)
    assert int(counts.max() - counts.min()) <= 1

def test_crossfit_output_roots_are_isolated_from_existing_individual_atlas():
    assert output_layout.resolve_crossfit_template_matching_root().name == "crossfit_template_matching"
    assert output_layout.resolve_crossfit_template_matching_overview_root().name == "crossfit_template_matching"
```

- [ ] **Step 2: Run the fold/root tests and verify RED**

Run:

```bash
/home/yojiao/.conda/envs/py314/bin/python -m pytest -q tests/individual_workflow/test_routea_crossfit_template_matching.py -k 'grouped_folds or output_roots'
```

Expected: collection or attribute failures because the workflow and resolvers do not exist.

- [ ] **Step 3: Implement deterministic grouped folds and isolated roots**

Implement a seeded, largest-family-first greedy allocator. Normalize `sub-`
prefixes, reject missing family entries, assign tied family sizes in seeded
order, and always place the next family into the currently smallest fold.
Add these exact root definitions:

```python
def resolve_crossfit_template_matching_root(dataset: str = DEFAULT_DATASET) -> Path:
    return resolve_work_root(dataset) / "crossfit_template_matching"

def resolve_crossfit_template_matching_overview_root(dataset: str = DEFAULT_DATASET) -> Path:
    return resolve_overviews_dir(dataset) / "crossfit_template_matching"
```

- [ ] **Step 4: Add failing tests for prototype construction and vertex matching**

```python
def test_cluster_prototypes_average_only_supported_ordered_cluster_vertices():
    fc = np.asarray([[1., 0.], [3., 2.], [9., 9.], [0., 4.]], dtype=np.float32)
    labels = np.asarray([1, 1, 0, 2], dtype=np.int32)
    support = np.asarray([1, 1, 0, 1], dtype=bool)
    got = workflow.build_cluster_prototypes(fc, labels, support, k=2)
    assert np.allclose(got, [[2., 1.], [0., 4.]])

def test_vertex_matching_returns_labels_scores_and_top1_top2_margin():
    fc = np.asarray([[1., 0., -1.], [-1., 0., 1.], [5., 5., 5.]], dtype=np.float32)
    valid = np.asarray([1, 1, 1], dtype=bool)
    prototypes = np.asarray([[1., 0., -1.], [-1., 0., 1.]], dtype=np.float32)
    labels, scores, margin = workflow.match_vertices_to_prototypes(fc, valid, prototypes)
    assert labels.tolist() == [1, 2, 0]
    assert scores.shape == (3, 2)
    assert np.allclose(margin[:2], [2., 2.])
    assert np.isnan(margin[2])
```

- [ ] **Step 5: Run the numeric tests and verify RED**

Run the two test names directly. Expected: missing-function failures.

- [ ] **Step 6: Implement finite-feature Pearson matching**

For every vertex/template pair, use only jointly finite features, mean-center
both vectors, and return NaN when fewer than two finite features or either norm
is zero. Assign label 0 when all template scores are invalid; otherwise assign
one-based `argmax` and save the difference between the two largest finite
scores.

- [ ] **Step 7: Run Task 1 tests and commit**

Run:

```bash
/home/yojiao/.conda/envs/py314/bin/python -m pytest -q tests/individual_workflow/test_routea_crossfit_template_matching.py
```

Expected: all Task 1 tests pass.

Commit:

```bash
git add scripts/individual_workflow/routea_crossfit_template_matching.py scripts/core_workflow/lib/output_layout.py tests/individual_workflow/test_routea_crossfit_template_matching.py
git commit -m "feat: add crossfit template matching primitives"
```

### Task 2: Fold-template fitting and held-out subject artifacts

**Files:**
- Modify: `scripts/individual_workflow/routea_crossfit_template_matching.py`
- Modify: `tests/individual_workflow/test_routea_crossfit_template_matching.py`

**Interfaces:**
- Consumes: Task 1 fold, prototype, matching, and root APIs.
- Produces: `fit_fold_template(config, fold_id, smoothing, hemi) -> FoldTemplateResult`
- Produces: `write_heldout_subject_outputs(config, template, subject) -> dict[str, object]`
- Produces: CLI flags `--family-map`, `--allow-subject-only-folds`, `--n-folds`, `--fold-seed`, `--smoothings`, `--hemis`, `--coverage-threshold`, `--dry-run`, `--resume`, and path overrides.

- [ ] **Step 1: Write failing tests for the family-data gate and disjoint fold manifest**

```python
def test_hcp_family_map_is_required_without_explicit_waiver(tmp_path):
    args = workflow.build_arg_parser().parse_args(["--subjects-manifest", str(tmp_path / "subjects.txt")])
    with pytest.raises(ValueError, match="family"):
        workflow.resolve_family_map(args, ["01", "02"])

def test_fold_manifest_has_disjoint_training_and_heldout_subjects():
    payload = workflow.fold_manifest_payload({"01": 0, "02": 1, "03": 0}, n_folds=2)
    for row in payload["folds"]:
        assert set(row["training_subjects"]).isdisjoint(row["heldout_subjects"])
```

- [ ] **Step 2: Verify RED, then implement family TSV/CSV parsing and manifest payloads**

The parser requires columns `subject,family_id`, rejects duplicate/conflicting
subjects, and rejects cohort subjects absent from the map. When the explicit
waiver is used, set `family_id=subject` and record
`family_aware=false`, `subject_only_waiver=true`.

- [ ] **Step 3: Write a failing synthetic fold-template test**

Create a tiny GIFTI surface with six vertices, monkeypatch
`fit_gradientmaps_pearson` to deterministic three-column gradients, and assert:

```python
assert result.training_subjects == ["01", "02"]
assert "03" not in result.fc_paths
assert result.prototypes.shape == (3, n_features)
assert set(np.unique(result.labels_full)) == {1, 2, 3}
assert result.density_evidence["consumed_asset"].endswith("den-512_label-hipp_midthickness.surf.gii")
```

- [ ] **Step 4: Verify RED, then implement fold fitting by reusing core APIs**

Reuse `05_01_compute_hippomaps_gradients.group_mean_fisherz_fc`,
`lib.hippomaps_adapter.fit_gradientmaps_pearson`, and the `05_02` surface,
adjacency, fixed-K clustering, and long-axis ordering functions. Do not call the
full `05_02` CLI or K-selection path. Save group support/coverage, group mean
FC, gradients, ordered K3 labels, `3 x 400` prototypes, named template GIFTI,
and a manifest under `templates/fold-XX/<smoothing>/hemi_<hemi>/`.

- [ ] **Step 5: Write a failing held-out artifact test**

Use a two-prototype synthetic template and assert that the output directory
contains:

```python
expected = {
    "individual_labels_k3.npy",
    "template_correlations.npy",
    "correlation_margin.npy",
    "combo_manifest.json",
}
assert expected.issubset({p.name for p in combo_dir.iterdir()})
manifest = json.loads((combo_dir / "combo_manifest.json").read_text())
assert manifest["subject_was_in_template"] is False
assert manifest["template_fold"] == 1
```

- [ ] **Step 6: Verify RED, then implement held-out writing and CLI orchestration**

The orchestration builds each fold/combo once, maps only that fold's held-out
subjects, supports resume only after manifest/input-parameter agreement, and
writes top-level fold assignments plus a crossfit manifest. `--dry-run` audits
required FC/mask/surface inputs and prints the exact count of template fits and
subject-combo mappings without fitting gradients or writing official labels.

- [ ] **Step 7: Run Task 2 tests and commit**

Run:

```bash
/home/yojiao/.conda/envs/py314/bin/python -m pytest -q tests/individual_workflow/test_routea_crossfit_template_matching.py
```

Expected: all tests pass.

Commit:

```bash
git add scripts/individual_workflow/routea_crossfit_template_matching.py tests/individual_workflow/test_routea_crossfit_template_matching.py
git commit -m "feat: build held-out HCP template labels"
```

### Task 3: Quantitative five-fold robustness overview

**Files:**
- Create: `scripts/individual_workflow/03_build_crossfit_robustness_overview.py`
- Create: `tests/individual_workflow/test_crossfit_robustness_overview.py`

**Interfaces:**
- Consumes: saved `group_labels_k3.npy`, support masks, `template_profiles.npy`, fold manifests, and canonical template surfaces.
- Produces: `pairwise_label_metrics(labels_by_fold, support_by_fold) -> dict[str, np.ndarray]`
- Produces: `vertex_agreement(labels_by_fold) -> np.ndarray`
- Produces: overview PNG, `robustness_summary.tsv`, and `robustness_manifest.json`.

- [ ] **Step 1: Write failing metric tests**

```python
def test_pairwise_metrics_and_vertex_agreement_detect_one_discordant_fold():
    labels = np.asarray([[1, 1, 2, 3], [1, 1, 2, 3], [1, 2, 2, 3]], dtype=np.int32)
    support = np.ones_like(labels, dtype=bool)
    metrics = overview.pairwise_label_metrics(labels, support, k=3)
    assert np.allclose(np.diag(metrics["macro_dice"]), 1.0)
    assert metrics["macro_dice"][0, 1] == pytest.approx(1.0)
    agreement = overview.vertex_agreement(labels)
    assert agreement.tolist() == pytest.approx([1.0, 2.0 / 3.0, 1.0, 1.0])
```

- [ ] **Step 2: Verify RED, then implement Dice, ARI, prototype correlation, and agreement helpers**

Pairwise label metrics use the intersection of nonzero support for each fold
pair. Macro-Dice averages labels 1..3, assigning 1.0 only when both folds lack
the same label and failing when the comparison support is empty. Prototype
correlation compares corresponding AP-ordered rows with finite-feature Pearson
correlation.

- [ ] **Step 3: Write a failing renderer test using five synthetic folds**

Construct saved arrays/manifests for `2mm/hemi_L`, invoke the renderer, and
assert a nonempty PNG, one TSV row with finite mean/min metrics, and a manifest
listing exactly five source folds.

- [ ] **Step 4: Verify RED, then implement the overview**

Use the canonical surface `long_axis_order` to show five categorical label
strips. Add pairwise macro-Dice and ARI heatmaps, corresponding-prototype
correlation summaries, and the vertex-agreement trace. One row represents each
requested smoothing/hemisphere combo. Save both numeric summary and complete
source paths so the PNG never becomes the sole robustness record.

- [ ] **Step 5: Run renderer tests and commit**

Run:

```bash
/home/yojiao/.conda/envs/py314/bin/python -m pytest -q tests/individual_workflow/test_crossfit_robustness_overview.py
```

Expected: all tests pass.

Commit:

```bash
git add scripts/individual_workflow/03_build_crossfit_robustness_overview.py tests/individual_workflow/test_crossfit_robustness_overview.py
git commit -m "feat: render crossfit template robustness"
```

### Task 4: Workflow documentation and contracts

**Files:**
- Create: `docs/hipp_parcellation_network/crossfit_template_matching.md`
- Modify: `scripts/individual_workflow/README.md`
- Modify: `scripts/README.md`
- Modify: `docs/hipp_parcellation_network/core_workflow_artifact_lineage.md`
- Modify: `tests/individual_workflow/test_routea_crossfit_template_matching.py`

**Interfaces:**
- Documents the production family-map gate, dry-run/full commands, output
  layout, algorithm, interpretation, robustness overview, and explicit
  non-run-level scope.

- [ ] **Step 1: Add a failing documentation-contract test**

```python
def test_crossfit_docs_name_family_gate_outputs_and_overview():
    text = (REPO_ROOT / "docs/hipp_parcellation_network/crossfit_template_matching.md").read_text()
    for token in ["subject,family_id", "--allow-subject-only-folds", "individual_labels_k3.npy", "robustness_summary.tsv"]:
        assert token in text
```

- [ ] **Step 2: Verify RED, then write synchronized documentation**

Document these exact execution forms:

```bash
/home/yojiao/.conda/envs/py314/bin/python scripts/individual_workflow/routea_crossfit_template_matching.py --family-map /secure/path/hcp_family_map.tsv --dry-run
/home/yojiao/.conda/envs/py314/bin/python scripts/individual_workflow/routea_crossfit_template_matching.py --family-map /secure/path/hcp_family_map.tsv --resume
/home/yojiao/.conda/envs/py314/bin/python scripts/individual_workflow/03_build_crossfit_robustness_overview.py
```

State that the full concat FC is intentional, no run-level claim is made, the
labels are cross-fitted at subject/family level, and restricted family data are
inputs rather than repository artifacts.

- [ ] **Step 3: Run documentation and workflow tests and commit**

Run:

```bash
/home/yojiao/.conda/envs/py314/bin/python -m pytest -q tests/individual_workflow/test_routea_crossfit_template_matching.py tests/individual_workflow/test_crossfit_robustness_overview.py
```

Expected: all tests pass.

Commit:

```bash
git add docs/hipp_parcellation_network/crossfit_template_matching.md docs/hipp_parcellation_network/core_workflow_artifact_lineage.md scripts/README.md scripts/individual_workflow/README.md tests/individual_workflow/test_routea_crossfit_template_matching.py
git commit -m "docs: document HCP crossfit template matching"
```

### Task 5: Verification and handoff

**Files:**
- Modify only files required by failures reproduced in this task.

**Interfaces:**
- Produces final test evidence, dry-run audit evidence, clean Git status, and a
  concise run handoff without launching the 151-subject full computation.

- [ ] **Step 1: Verify the explicit interpreter**

Run:

```bash
/home/yojiao/.conda/envs/py314/bin/python -c "import sys; print(sys.executable)"
```

Expected: `/home/yojiao/.conda/envs/py314/bin/python`.

- [ ] **Step 2: Run syntax and focused regression tests**

```bash
/home/yojiao/.conda/envs/py314/bin/python -m py_compile scripts/individual_workflow/routea_crossfit_template_matching.py scripts/individual_workflow/03_build_crossfit_robustness_overview.py
/home/yojiao/.conda/envs/py314/bin/python -m pytest -q tests/individual_workflow/test_routea_crossfit_template_matching.py tests/individual_workflow/test_crossfit_robustness_overview.py tests/individual_workflow/test_routea_individual_gradients_wta.py tests/core_workflow/test_hippomaps_gradient_route.py tests/core_workflow/test_surface_consensus_and_routea_guards.py
```

Expected: all tests pass; existing deprecation warnings may remain unchanged.

- [ ] **Step 3: Run a dry-run input/workload audit**

Use the real QC manifest plus the user-supplied secure family map. Expected
workload for both smoothings and hemispheres is 20 fold-template fits and 604
held-out subject-combo mappings, with zero train/held-out overlap. If a family
map is not available in this session, run the same audit with
`--allow-subject-only-folds` and report that this is interface validation only,
not the preferred production fold definition.

- [ ] **Step 4: Inspect changes and run final diff checks**

```bash
git diff --check
git status --short
```

Expected: no whitespace errors; only intended crossfit code, tests, plans, and
documentation are modified.

- [ ] **Step 5: Request code review and address only verified findings**

Review fold leakage, family containment, numeric null handling, output
isolation, density evidence, overview metric definitions, and documentation
parity. Reproduce each accepted issue with a failing test before changing code.

- [ ] **Step 6: Commit final verified adjustments**

```bash
git add docs scripts tests
git commit -m "test: verify HCP crossfit template workflow"
```

Skip this commit when no post-review files changed.
