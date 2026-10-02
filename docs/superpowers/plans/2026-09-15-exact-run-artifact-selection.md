# Exact Run-Artifact Selection Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use `executing-plans` task-by-task. Do not commit or stage: the user will inspect the working tree.

**Goal:** Remove automatic run-file discovery and inferred run segmentation from the maintained core paths so every consumed run is explicitly declared, fingerprinted, and provenance-recorded before outputs are created.

**Architecture:** Add one small manifest reader in `scripts/core_workflow/lib/io_utils.py`. The reader consumes a caller-selected per-subject JSON receipt containing an ordered run list and exact artifact records; it never globs, sorts filesystem results, or guesses IDs. Migrate the run-level modes in the group-input and reference launchers to this reader, while retaining a separately explicit single-file mode only when its segment metadata is also caller supplied.

**Tech Stack:** Python 3.14 at `/home/yojiao/.conda/envs/py314/bin/python`, pytest, JSON, SHA-256 receipts, nibabel.

**Spec:** `AGENTS.md`, `docs/environment/agent_runtime_contract.md`, and the user’s selective-reconciliation criteria.

## Global Constraints

- Do not commit, stage, delete, move, or write any formal output-tree artifact.
- Do not browse archive content.
- No glob, candidate, existence-order, or inferred run-ID/segment fallback may remain in formal execution.
- A manifest declares an ordered positive `run_id` list and a complete `path`/`sha256`/`size_bytes` record for each consumed artifact.
- Validate all selected manifest entries before creating a result directory.
- Preserve explicit HCP single-CIFTI mode only with a caller-selected run-segment metadata receipt; do not infer equal four-way boundaries.
- Update CLI/help/runbook text in the same source change.

---

### Task 1: Define the exact run-artifact receipt reader

**Files:**

- Modify: `scripts/core_workflow/lib/io_utils.py`
- Modify: `tests/core_workflow/test_cifti_in_memory_inputs.py`

**Interfaces:**

- Produces `load_run_artifact_manifest(path: Path, subject: str, required_artifacts: Sequence[str]) -> list[dict[str, object]]`.
- Manifest schema:

```json
{
  "subject": "sub-1027",
  "runs": [
    {
      "run_id": 2,
      "dtseries": {"path": "/absolute/run-02.dtseries.nii", "sha256": "...", "size_bytes": 123},
      "bold": {"path": "/absolute/run-02_bold.nii.gz", "sha256": "...", "size_bytes": 456}
    }
  ]
}
```

- [ ] **Step 1: Write failing tests for exact ordered receipt loading.**

```python
def test_load_run_artifact_manifest_uses_declared_order_and_receipts(tmp_path):
    manifest = write_run_manifest(tmp_path, subject="1027", run_ids=[2, 3])
    runs = io_utils.load_run_artifact_manifest(
        manifest, subject="1027", required_artifacts=("dtseries",)
    )
    assert [row["run_id"] for row in runs] == [2, 3]
```

- [ ] **Step 2: Write failing tests for absent IDs, duplicate IDs, relative paths, and SHA drift.**

```python
with pytest.raises(ValueError, match="requires positive explicit run_id"):
    io_utils.load_run_artifact_manifest(bad_manifest, subject="1027", required_artifacts=("dtseries",))
```

- [ ] **Step 3: Implement the reader without globbing or path-derived IDs.**

For every required artifact, require an absolute recorded path, regular file, matching SHA-256 and size. Preserve JSON list order after rejecting duplicate/non-increasing IDs; do not sort it.

- [ ] **Step 4: Run the focused receipt tests.**

```bash
env OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 \
  /home/yojiao/.conda/envs/py314/bin/python -m pytest -q -p no:cacheprovider \
  tests/core_workflow/test_cifti_in_memory_inputs.py
```

### Task 2: Migrate run-level input callers to explicit receipts

**Files:**

- Modify: `scripts/core_workflow/02_01_prepare_group_inputs.py`
- Modify: `scripts/core_workflow/03_01_build_kong2022_reference.py`
- Modify: `scripts/core_workflow/03_02_build_kong2022_routea_reference.py`
- Modify: `scripts/core_workflow/03_04_run_lynch2024_fastans_pfm.py`
- Modify: their directly targeted tests under `tests/core_workflow/`

**Interfaces:**

- Replace formal `--run-bold-glob`, `--run-dtseries-glob`, and `--dtseries-run-template` modes with one caller-selected `--run-artifact-manifest-template`.
- The template formats only `{subject}`/`{sub_subject}`; it resolves one exact manifest path per requested subject.
- `02_01` requires both `bold` and `dtseries`; reference builders require `dtseries`.

- [ ] **Step 1: Write failing launcher tests proving a manifest controls exact paths.**

```python
with pytest.raises(ValueError, match="run artifact manifest"):
    workflow.run_subject(..., run_artifact_manifest_template=None)
assert "--run-dtseries" in workflow.command_for_manifest_subject(...)
```

- [ ] **Step 2: Migrate `02_01` before any output directory creation.**

Resolve and validate the per-subject receipt, verify bold/dtseries IDs align and frame counts match, then create `surface/raw`. Do not retain a glob alternative in formal execution.

- [ ] **Step 3: Migrate the three reference launchers.**

Use receipt-declared dtseries paths in recorded order. Preserve the explicitly selected single-file HCP option only when Task 3’s metadata path is supplied.

- [ ] **Step 4: Run only focused caller tests.**

```bash
env OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 \
  /home/yojiao/.conda/envs/py314/bin/python -m pytest -q -p no:cacheprovider \
  tests/core_workflow/test_cifti_in_memory_inputs.py \
  tests/core_workflow/test_lynch_no_temporary_inputs.py
```

### Task 3: Require recorded run boundaries

**Files:**

- Modify: `scripts/core_workflow/lib/io_utils.py`
- Modify: `scripts/core_workflow/03_03_build_lynch2024_reference.py`
- Modify: `scripts/core_workflow/lib/extract_cortex_kong2022_roi_timeseries.py`
- Modify: `scripts/core_workflow/lib/extract_cortex_kong2022_route_a_roi_timeseries.py`
- Modify: `tests/core_workflow/test_lynch_reference_run_metadata.py`
- Modify: `tests/core_workflow/test_cifti_in_memory_inputs.py`

**Interfaces:**

- `parse_run_segments` receives explicit positive `run_ids` and `run_lengths`; `None` is rejected.
- A single CIFTI call must pass `--run-metadata-template` selecting one exact metadata JSON with matching run IDs/lengths.
- Run-level manifest paths carry their own list order and frame lengths are checked against actual CIFTI headers before output creation.

- [ ] **Step 1: Write failing tests for missing segment metadata and pre-write failure.**

```python
with pytest.raises(ValueError, match="requires explicit run IDs"):
    parse_run_segments(None, 120)
assert not output_dir.exists()
```

- [ ] **Step 2: Make every caller pass selected run IDs and lengths.**

For receipt runs, use the declared order and observed header lengths. For a selected single CIFTI, load the explicit metadata JSON and require its total to equal CIFTI frames. Remove the equal-four-way default text and behavior.

- [ ] **Step 3: Run segment and Lynch regression tests.**

```bash
env OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 \
  /home/yojiao/.conda/envs/py314/bin/python -m pytest -q -p no:cacheprovider \
  tests/core_workflow/test_lynch_reference_run_metadata.py \
  tests/core_workflow/test_cifti_in_memory_inputs.py
```

### Task 4: Synchronize formal documentation and audit all changed paths

**Files:**

- Modify: `scripts/core_workflow/README.md`
- Modify: `scripts/core_workflow/step_registry.json`
- Modify: `docs/ctxreg_selective_integration.md`
- Modify: relevant workflow runbooks that currently name glob/template selection

**Interfaces:**

- Docs name the exact receipt schema, caller-selected manifest template, explicit segment metadata, and failure-before-write guarantee.
- `docs/ctxreg_selective_integration.md` lists exact kept main paths, exact deferred worktree paths, focused test commands/results, and product dispositions; it must not claim an unreproducible test count.

- [ ] **Step 1: Update CLI and workflow examples to use explicit manifest templates.**

Every execution example uses `/home/yojiao/.conda/envs/py314/bin/python` and names the receipt input. Remove glob-based examples.

- [ ] **Step 2: Write the exact selective-integration allowlist/defer list.**

Include the main-only stronger Lynch metadata implementation, the worktree-only coupled branch files deferred as a unit, unrelated `hcp_cortex_control_gradient_wta` preserved in main, and all generated worktree products excluded.

- [ ] **Step 3: Run documentation-contract and focused source tests.**

```bash
env OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 \
  /home/yojiao/.conda/envs/py314/bin/python -m pytest -q -p no:cacheprovider \
  tests/core_workflow/test_cifti_in_memory_inputs.py \
  tests/core_workflow/test_lynch_reference_run_metadata.py \
  tests/core_workflow/test_lynch_no_temporary_inputs.py
```

- [ ] **Step 4: Review unstaged diff with the user.**

Do not stage, commit, run formal workflows, migrate outputs, or delete the worktree.
