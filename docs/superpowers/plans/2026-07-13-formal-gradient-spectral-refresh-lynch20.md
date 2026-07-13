# Formal Gradient-spectral Refresh and Lynch20 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make every maintained active gradient-spectral result traceable to shifted-cosine and replace the misnamed taxonomy with an exact 20-network, Noise-excluded Lynch20 workflow.

**Architecture:** Rename the Lynch route at shared configuration seams, rebuild the Lynch20 input lineage rather than rewriting old manifests, propagate affinity provenance through weighted branches and presentation manifests, and enforce the release contract with one read-only audit module. A fixed `jobs=1` sequential launcher performs only the approved HCP weighted, Stanford HC/Symp, Lynch20/Lynch8, and overview stages.

**Tech Stack:** Python 3.14 (`/home/yojiao/.conda/envs/py314/bin/python`), NumPy/SciPy, nibabel, matplotlib, pytest, Git.

## Global Constraints

- Use `/home/yojiao/.conda/envs/py314/bin/python` for every repository Python command; first print the informational bare-Python executable.
- `gradient-spectral` has one implementation: `shifted-cosine`; no relu-cosine selector or compatibility mode.
- Lynch formal names are `lynch2024_20net` and `lynch20`; do not retain any retired runtime alias, fallback, or symlink.
- Lynch20 must exclude `Noise` and contain exactly 20 ordered networks/features.
- Diagnostic K range is `[2, 8]`; final K is `3`.
- Production execution is strictly sequential with `jobs=1`, safe run IDs, stop-on-first-failure status, and low-frequency monitoring.
- Before retained HCP branches are consumed, a non-computational all-or-nothing validation pass atomically enriches their eight JSON manifests with dataset/root and verified artifact-fingerprint provenance; no binary output is rewritten.
- The eight-manifest commit stages every JSON update first and restores all exact original bytes on any mid-replacement failure; launcher preflight runs the same complete read-only preparation before detaching.
- Do not recompute RouteB, ctxreg, ctxdrop, or archived experimental gradient-spectral branches.
- Do not alter continuous HippoMaps products except the Lynch gradient-bank route rename/rebuild.
- Preserve unrelated root dirty deletions and the uncommitted root handoff.
- Update code and matching active documentation in the same commit.
- Do not browse archive content or move/delete old misnamed-taxonomy paths until the exact inventory is presented and the user explicitly confirms it.
- Maximum total concurrency is 10; this plan never requests more than one production worker.

---

### Task 1: Replace the misnamed taxonomy with a strict Lynch20 route

**Files:**
- Modify: `scripts/core_workflow/lib/lynch2024_reference.py`
- Modify: `scripts/core_workflow/lib/branch_config.py`
- Modify: `scripts/core_workflow/lib/output_layout.py`
- Modify: `scripts/core_workflow/lib/route_tags.py`
- Modify: `scripts/core_workflow/02_03_build_lynch2024_nsi_manifest.py`
- Modify: `scripts/core_workflow/03_03_build_lynch2024_reference.py`
- Modify: `scripts/core_workflow/04_01_materialize_kong2022_roi_fc.py`
- Modify: `scripts/core_workflow/05_01_compute_hippomaps_gradients.py`
- Modify: `scripts/core_workflow/05_02_run_group_atlas.py`
- Modify: `scripts/core_workflow/06_01_build_group_overview.py`
- Modify: `scripts/core_workflow/step_registry.json`
- Test: `tests/core_workflow/test_lynch2024_route_and_reference.py`
- Test: `tests/core_workflow/test_hippomaps_gradient_route.py`
- Test: `tests/core_workflow/test_pipeline_numbering_contract.py`

**Interfaces:**
- Produces: `LYNCH2024_20NET_ATLAS = "lynch2024_20net"`, `lynch2024_20net_order(config: Path) -> list[str]`, `resolve_lynch2024_20net_group_atlas_root(dataset: str = "hcp") -> Path`, and `gradient_route_for_route_tag("lynch2024_8net") == "lynch2024_20net"`.
- Consumes: the existing merge config and hard-label PFM component tables, with `Noise` removed before order validation.

- [ ] **Step 1: Write failing route and order tests**

Add tests that require the breaking public contract:

```python
def test_lynch20_order_is_exactly_twenty_and_excludes_noise() -> None:
    order = lynch2024_20net_order(REPO_ROOT / "config/cross_atlas_network_merge.json")
    assert len(order) == 20
    assert "Noise" not in order
    assert len(set(order)) == 20


def test_retired_runtime_names_are_removed() -> None:
    assert "lynch2024_20net" in route_tags.ROUTE_TAG_CHOICES
    assert "lynch20" in route_tags.NETWORK_TAXONOMY_CHOICES
    assert all("20net" in name or "8net" in name or "lynch" not in name for name in route_tags.ROUTE_TAG_CHOICES)


def test_lynch8_reuses_lynch20_gradient_bank() -> None:
    assert atlas_mod.gradient_route_for_route_tag("lynch2024_8net") == "lynch2024_20net"
```

- [ ] **Step 2: Run the tests and verify red**

Run:

```bash
python -c "import sys; print(sys.executable)"
/home/yojiao/.conda/envs/py314/bin/python -m pytest -q \
  tests/core_workflow/test_lynch2024_route_and_reference.py \
  tests/core_workflow/test_hippomaps_gradient_route.py \
  tests/core_workflow/test_pipeline_numbering_contract.py
```

Expected: failures naming missing `lynch2024_20net`/`lynch20` and remaining retired choices.

- [ ] **Step 3: Implement the canonical names and strict order validator**

Rename the public constants/functions and validate immediately after Noise removal:

```python
LYNCH2024_20NET_ATLAS = "lynch2024_20net"


def lynch2024_20net_order(merge_config: Path) -> list[str]:
    names = [name for name in load_lynch_parent_order(merge_config) if name != "Noise"]
    if len(names) != 20 or len(set(names)) != 20:
        raise ValueError(
            "Lynch2024 formal taxonomy must contain exactly 20 unique non-Noise networks: "
            f"count={len(names)} names={names}"
        )
    return names
```

Change every runtime choice, route inference, output resolver, atlas selector,
stage registry entry, and gradient-route mapping from the misnamed taxonomy to
`20net`/`lynch20`. Use `lynch2024_20net_nsi043_full_k3_normalized_wta` as the
formal mixed-method group root and `outputs_hcp/work/hippomaps_gradients/lynch2024_20net`
as the gradient bank. Do not accept old spellings in argparse choices.

- [ ] **Step 4: Add manifest evidence and FC dimensionality validation**

In reference/FC/gradient manifests record:

```python
"network_taxonomy": "lynch20",
"network_count": 20,
"noise_excluded": True,
"network_order": network_order,
```

Before saving Lynch20 FC or consuming it for gradients, require
`fc.shape[-1] == 20`; raise a path-specific `ValueError` otherwise.

- [ ] **Step 5: Run focused tests and compile checks**

Run the Step 2 pytest command plus:

```bash
/home/yojiao/.conda/envs/py314/bin/python -m py_compile \
  scripts/core_workflow/lib/lynch2024_reference.py \
  scripts/core_workflow/lib/output_layout.py \
  scripts/core_workflow/lib/route_tags.py \
  scripts/core_workflow/03_03_build_lynch2024_reference.py \
  scripts/core_workflow/04_01_materialize_kong2022_roi_fc.py \
  scripts/core_workflow/05_01_compute_hippomaps_gradients.py \
  scripts/core_workflow/05_02_run_group_atlas.py
```

Expected: focused tests pass; compilation exits 0.

- [ ] **Step 6: Commit the taxonomy seam**

```bash
git add scripts/core_workflow tests/core_workflow scripts/core_workflow/step_registry.json
git commit -m "Correct Lynch2024 formal taxonomy to 20 networks"
```

---

### Task 2: Propagate shifted-cosine into weighted and presentation provenance

**Files:**
- Modify: `scripts/core_workflow/05_05_build_weighted_fisherz_branch.py`
- Modify: `scripts/core_workflow/06_01_build_group_overview.py`
- Modify: `scripts/core_workflow/lib/surface_affinity.py`
- Test: `tests/core_workflow/test_kong2022_cluster_annotation_shared.py`
- Test: `tests/core_workflow/test_crossfit_robustness_overview.py`
- Create: `tests/core_workflow/test_group_overview_affinity_provenance.py`

**Interfaces:**
- Produces: `gradient_affinity_provenance(manifests: list[dict[str, object]]) -> dict[str, str]` and overview item fields `surface_affinity`, `affinity_description`, `source_group_manifests`.
- Consumes: group manifests from Task 1/current HCP results.

- [ ] **Step 1: Write failing provenance tests**

Create fixtures with four shifted group manifests and assert:

```python
item = build_overview_item_for_test(group_manifests)
assert item["surface_affinity"] == "shifted-cosine"
assert item["affinity_description"] == GRADIENT_SPECTRAL_AFFINITY_DESCRIPTION
assert len(item["source_group_manifests"]) == 4
assert "shifted-cosine" in rendered_title
```

Add negative cases where one manifest is relu-cosine or missing affinity; expect
`ValueError("Inconsistent gradient-spectral affinity provenance")`.

- [ ] **Step 2: Verify the tests fail**

```bash
/home/yojiao/.conda/envs/py314/bin/python -m pytest -q \
  tests/core_workflow/test_group_overview_affinity_provenance.py \
  tests/core_workflow/test_kong2022_cluster_annotation_shared.py
```

Expected: missing provenance helper/item fields.

- [ ] **Step 3: Implement group-to-presentation provenance**

Add one helper in `06_01_build_group_overview.py`:

```python
def gradient_affinity_provenance(manifests: list[dict[str, object]]) -> dict[str, str]:
    affinities = {str(item.get("surface_affinity", "")) for item in manifests}
    descriptions = {str(item.get("affinity_description", "")) for item in manifests}
    if affinities != {GRADIENT_SPECTRAL_AFFINITY} or descriptions != {
        GRADIENT_SPECTRAL_AFFINITY_DESCRIPTION
    }:
        raise ValueError("Inconsistent gradient-spectral affinity provenance")
    return {
        "surface_affinity": GRADIENT_SPECTRAL_AFFINITY,
        "affinity_description": GRADIENT_SPECTRAL_AFFINITY_DESCRIPTION,
    }
```

Collect the exact manifest paths used by each overview item. Append
`| shifted-cosine` to the visible gradient-spectral title while preserving the
PNG filename. Do not add affinity fields to WTA items.

- [ ] **Step 4: Preserve affinity through weighted branch construction**

Require each copied gradient source manifest to be shifted-cosine. The weighted
branch manifest records:

```python
"clustering_surface_affinity": "shifted-cosine",
"clustering_affinity_description": GRADIENT_SPECTRAL_AFFINITY_DESCRIPTION,
"weighted_assignment_only": True,
```

Do not relabel WTA provenance as gradient provenance.

- [ ] **Step 5: Run focused tests**

Run:

```bash
/home/yojiao/.conda/envs/py314/bin/python -m pytest -q \
  tests/core_workflow/test_group_overview_affinity_provenance.py \
  tests/core_workflow/test_kong2022_cluster_annotation_shared.py \
  tests/core_workflow/test_crossfit_robustness_overview.py
```

Expected: all pass.

- [ ] **Step 6: Commit provenance propagation**

```bash
git add scripts/core_workflow/05_05_build_weighted_fisherz_branch.py \
  scripts/core_workflow/06_01_build_group_overview.py \
  tests/core_workflow
git commit -m "Record shifted-cosine presentation provenance"
```

---

### Task 3: Add a read-only formal gradient release audit

**Files:**
- Create: `scripts/core_workflow/audit_formal_gradient_spectral_release.py`
- Create: `tests/core_workflow/test_audit_formal_gradient_spectral_release.py`

**Interfaces:**
- Produces: `audit_release(repo_root: Path, contract: dict[str, object]) -> dict[str, object]` returning `status`, `group_combinations`, `overview_items`, and `failures`.
- Consumes: shifted group manifests, weighted branch manifests, and overview manifests from Tasks 1–2.

- [ ] **Step 1: Write the failing audit tests**

Use a minimal repository fixture and assert the audit rejects each exact fault:

```python
def test_audit_rejects_legacy_group_manifest(tmp_path: Path) -> None:
    fixture = build_formal_fixture(tmp_path, surface_affinity="relu-cosine")
    report = module.audit_release(tmp_path, fixture.contract)
    assert report["status"] == "failed"
    assert any("surface_affinity" in item for item in report["failures"])


def test_audit_rejects_overview_with_missing_lineage(tmp_path: Path) -> None:
    fixture = build_formal_fixture(tmp_path, surface_affinity="shifted-cosine")
    fixture.overview_manifest.unlink()
    report = module.audit_release(tmp_path, fixture.contract)
    assert report["status"] == "failed"


def test_audit_accepts_complete_release(tmp_path: Path) -> None:
    fixture = build_formal_fixture(tmp_path, surface_affinity="shifted-cosine")
    assert module.audit_release(tmp_path, fixture.contract)["status"] == "ok"
```

- [ ] **Step 2: Run the test and verify red**

```bash
/home/yojiao/.conda/envs/py314/bin/python -m pytest -q \
  tests/core_workflow/test_audit_formal_gradient_spectral_release.py
```

Expected: module missing.

- [ ] **Step 3: Implement the explicit release contract**

The default contract enumerates only:

```python
FORMAL_GROUPS = (
    ("outputs_hcp", "routea_full_k3_fisherz", "kong2022"),
    ("outputs_hcp", "routea_17net_full_k3_fisherz", "kong2022"),
    ("outputs_hcp", "routea_weighted_fisherz_full_k3", "kong2022"),
    ("outputs_hcp", "lynch2024_20net_nsi043_full_k3_normalized_wta", "lynch2024_20net"),
    ("outputs_hcp", "lynch2024_8net_nsi043_full_k3_normalized_wta", "lynch2024_8net"),
    ("outputs_stanford", "routea_hc_full_k3_fisherz", "kong2022"),
    ("outputs_stanford", "routea_symp_full_k3_fisherz", "kong2022"),
    ("outputs_stanford", "routea_hc_weighted_fisherz_full_k3", "kong2022"),
    ("outputs_stanford", "routea_symp_weighted_fisherz_full_k3", "kong2022"),
)
```

For each 2mm/4mm × L/R combination require status, affinity, K, graph, source
paths, and expected assignment variant. Resolve every overview manifest source
path and require it belongs to the matching formal group. Report all failures in
one pass; never modify outputs.

- [ ] **Step 4: Add CLI JSON output**

Support `--repo-root`, `--output-json`, and `--allow-incomplete`. Default exit is
0 only for `status=ok`; incomplete/legacy provenance exits 1. Writing an output
JSON is optional and atomic.

- [ ] **Step 5: Run tests and compile**

```bash
/home/yojiao/.conda/envs/py314/bin/python -m pytest -q \
  tests/core_workflow/test_audit_formal_gradient_spectral_release.py
/home/yojiao/.conda/envs/py314/bin/python -m py_compile \
  scripts/core_workflow/audit_formal_gradient_spectral_release.py
```

Expected: pass and exit 0.

- [ ] **Step 6: Commit the audit**

```bash
git add scripts/core_workflow/audit_formal_gradient_spectral_release.py \
  tests/core_workflow/test_audit_formal_gradient_spectral_release.py
git commit -m "Audit formal gradient-spectral provenance"
```

---

### Task 4: Build the fixed sequential refresh runner

**Files:**
- Create: `scripts/core_workflow/run_formal_gradient_spectral_refresh.py`
- Create: `tests/core_workflow/test_run_formal_gradient_spectral_refresh.py`
- Reuse: `scripts/core_workflow/run_hcp_routea_gradient_spectral_background.py`

**Interfaces:**
- Produces: `build_stage_plan(repo_root: Path, python_exe: Path, jobs: int) -> list[dict[str, object]]`, `validate_release_settings(run_id: str, jobs: int) -> None`, and detached launch/status behavior matching the existing HCP runner.
- Consumes: numbered workflow scripts and audit CLI from Tasks 1–3.

- [ ] **Step 1: Write exact stage-plan tests**

Assert the ordered names are:

```python
assert [stage["name"] for stage in stages] == [
    "lynch20_reference",
    "lynch20_fc",
    "lynch8_fc",
    "lynch20_gradient_bank",
    "lynch20_group",
    "lynch20_overview",
    "lynch8_group",
    "lynch8_overview",
    "hcp_weighted_branch",
    "hcp_weighted_overview",
    "stanford_hc_group",
    "stanford_hc_weighted",
    "stanford_hc_overviews",
    "stanford_symp_group",
    "stanford_symp_weighted",
    "stanford_symp_overviews",
    "hcp_routea_overviews",
    "hcp_family17_overviews",
    "formal_release_audit",
]
```

Also assert every group stage contains `--methods gradient-spectral`,
`--k-min 2 --k-max 8 --force-final-k 3 --jobs 1`; every command uses canonical Lynch20 naming and excludes
RouteB, ctxreg, or ctxdrop.

- [ ] **Step 2: Verify tests fail**

```bash
/home/yojiao/.conda/envs/py314/bin/python -m pytest -q \
  tests/core_workflow/test_run_formal_gradient_spectral_refresh.py
```

Expected: runner module missing.

- [ ] **Step 3: Implement fixed settings and path-safe status files**

Copy only the small detached-run/status pattern from the existing HCP runner.
Keep:

```python
if jobs != 1:
    raise ValueError("The formal release runner is fixed at --jobs 1")
if re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}", run_id) is None:
    raise ValueError(f"Unsafe --run-id: {run_id!r}")
```

Pass the explicit `--repo-root` to the worker. Set BLAS/OpenMP thread variables
to 1. Never launch more than one child stage at a time.

- [ ] **Step 4: Encode exact formal commands and preflight**

Use the existing documented HCP/Stanford shared roots, subgroup subject
manifests, templates, and Lynch NSI pass manifest. Every required input path and
script must exist before launch. Overview commands use method-specific
assignment variants and stable canonical filenames. The final stage runs:

```bash
/home/yojiao/.conda/envs/py314/bin/python \
  scripts/core_workflow/audit_formal_gradient_spectral_release.py \
  --output-json outputs_hcp/work/group_atlas/logs/<run-id>.audit.json
```

- [ ] **Step 5: Add resume and failure-status tests**

Mock `subprocess.run` so a middle stage fails. Assert later stages do not run,
the status becomes `failed`, and `completed_stages` contains only prior names.
Assert a successful mock run reaches `status=ok` only after audit completion.

- [ ] **Step 6: Run focused tests**

```bash
/home/yojiao/.conda/envs/py314/bin/python -m pytest -q \
  tests/core_workflow/test_run_formal_gradient_spectral_refresh.py \
  tests/core_workflow/test_run_hcp_routea_gradient_spectral_background.py
```

Expected: pass.

- [ ] **Step 7: Commit the runner**

```bash
git add scripts/core_workflow/run_formal_gradient_spectral_refresh.py \
  tests/core_workflow/test_run_formal_gradient_spectral_refresh.py
git commit -m "Run the formal gradient refresh sequentially"
```

---

### Task 5: Synchronize documentation and verify the implementation

**Files:**
- Modify: `scripts/core_workflow/README.md`
- Modify: `docs/hipp_parcellation_network/core_workflow_overview.md`
- Modify: `docs/hipp_parcellation_network/core_workflow_artifact_lineage.md`
- Modify: `docs/hipp_parcellation_network/core_workflow_step_registry.md`
- Modify: `docs/hipp_parcellation_network/multi_branch_flow.md`
- Modify: `docs/hipp_parcellation_network/step_by_step_flows.md`
- Modify: `docs/rendering/wb_locked_rendering.md`
- Modify: `scripts/core_workflow/step_registry.json`

**Interfaces:**
- Consumes: final runtime names, paths, manifest schema, audit command, and runner from Tasks 1–4.
- Produces: one active documented contract with no retired taxonomy runtime instructions or tracked spellings.

- [ ] **Step 1: Replace active Lynch naming and document migration status**

Document Lynch20 as 20 non-Noise networks and Lynch8 as its downstream merge.
Historical references use the generic phrase “misnamed taxonomy”; active
commands and paths use only `20net`/`lynch20`.

- [ ] **Step 2: Document formal affinity and audit contracts**

Add the exact audit command, formal branch list, overview affinity fields,
stable filename/visible-title policy, `jobs=1` runner, and archive-confirmation
gate.

- [ ] **Step 3: Run active-doc stale-name checks**

```bash
git grep -n -i -E '<retired-taxonomy-pattern>' -- \
  scripts/core_workflow \
  docs/hipp_parcellation_network docs/rendering
```

Expected: no active runtime occurrence; only explicitly marked historical text
inside the approved design/plan may remain.

- [ ] **Step 4: Run full verification**

```bash
git diff --check
/home/yojiao/.conda/envs/py314/bin/python -m py_compile \
  scripts/core_workflow/*.py scripts/core_workflow/lib/*.py
/home/yojiao/.conda/envs/py314/bin/python -m pytest -q tests
```

Expected: diff/compile exit 0 and full suite passes.

- [ ] **Step 5: Request independent code review and address blockers**

Review since `1998738` against the confirmed design spec. Require separate
Standards and Spec conclusions; rerun affected tests after every correction.

- [ ] **Step 6: Commit docs and final corrections**

```bash
git add scripts/core_workflow docs tests
git commit -m "Finalize formal gradient-spectral release workflow"
```

---

### Task 6: Execute, validate, and hand off the production refresh

**Files:**
- Create at runtime: `outputs_hcp/work/group_atlas/logs/<run-id>.status.json`
- Create at runtime: `outputs_hcp/work/group_atlas/logs/<run-id>.audit.json`
- Create: `HANDOFF_formal_gradient_spectral_refresh_20260713.md`
- Generate after success: an exact proposed inventory under a user-approved misnamed-taxonomy archive root

**Interfaces:**
- Consumes: verified runner and audit from Tasks 3–5.
- Produces: formal shifted-cosine result set plus a user-reviewable old-path inventory; it does not move the old paths without confirmation.

- [ ] **Step 1: Run launcher preflight without mutation**

Invoke the runner's preflight/stage-plan mode and verify all required HCP,
Stanford, and Lynch inputs exist; planned concurrency must print `1`.

- [ ] **Step 2: Launch the detached sequential worker**

```bash
/home/yojiao/.conda/envs/py314/bin/python \
  scripts/core_workflow/run_formal_gradient_spectral_refresh.py \
  --launch --jobs 1
```

Record the emitted run ID, PID, status, stdout, and stderr paths. Do not poll at
high frequency.

- [ ] **Step 3: On terminal success, verify the audit independently**

Run the audit again with a second output JSON and compare its status/counts to
the worker audit. Spot-check HCP, Stanford HC/Symp, Lynch20, Lynch8, and weighted
manifests for shifted-cosine, `[2,8]`, K=3, and connected graphs. Confirm overview
titles and manifests expose shifted-cosine.

- [ ] **Step 4: Produce but do not apply the old misnamed-taxonomy inventory**

List exact active paths containing the retired taxonomy route across Lynch
reference, FC, gradient, group, presentation, logs, and
manifests. Exclude archive content and unrelated prose. Hash the proposed source
trees and save a reviewed plan under the proposed archive root only after the
user approves creating it.

- [ ] **Step 5: Ask for exact archive confirmation**

Present the complete move list, destination
the proposed misnamed-taxonomy archive root, and reason. Wait for explicit
confirmation before any rename/removal operation.

- [ ] **Step 6: Write the root handoff**

Record commits, run/audit paths and status, formal counts, validation evidence,
old-path inventory status, and the rule that archive execution remains blocked
until confirmation. Leave the handoff uncommitted unless the user explicitly
requests a commit.
