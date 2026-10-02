# Locked HippUnfold Run Contract Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the permissive local HippUnfold launch surface with a fixed, dataset-scoped T1w/512 contract and make output validation reject unfold-only partial DAGs.

**Architecture:** Put every caller-visible decision behind one Python launch-contract module whose interface is only `dataset`, `subject`, and `dry_run`. Dataset presets own canonical BIDS roots, output roots, and T1w extensions; the module owns all HippUnfold/Snakemake arguments and runtime environment. Keep shell files only as strict two-argument compatibility adapters, and add independent retained-output anchors to the checker so a partial Snakemake log cannot define “complete.”

**Tech Stack:** Python 3.14, argparse, dataclasses, subprocess, Bash compatibility adapters, pytest.

**Spec:** `AGENTS.md`, `docs/environment/hippo2_reproducibility_manifest.md`, and the 2026-08-19 user requirement to fix HippUnfold inputs/outputs and remove free-form launch parameters.

## Global Constraints

- Core workflow Python and tests use `/home/yojiao/.conda/envs/py314/bin/python`.
- Direct HippUnfold execution uses the validated repo-local `hippo2` executable at `/home/yojiao/hpchome/hipp/.cache/conda_envs/hippo2/bin/hippunfold`, version 2.0.0.
- The fixed processing contract is T1w, density 512, `output_spaces=[corobl]`, bilateral hippocampus and dentate, unfolded registration enabled, T1 registration to template disabled, GPU enabled, and one Snakemake core.
- `--omit-from download_nnunet_model`, arbitrary Snakemake extras, arbitrary model paths, and caller-controlled modality/density/space/output roots are forbidden.
- Active HCP, Stanford, and Iowa JJ-TMS paths are dataset presets; archive content is out of scope.
- Code, launcher behavior, workflow docs, and the environment manifest change together.

---

### Task 1: Fixed launch-contract module and public entrypoint

**Files:**
- Create: `scripts/core_workflow/lib/hippunfold_run_contract.py`
- Modify: `scripts/core_workflow/01_01_run_hippunfold_t1w_local.py`
- Modify: `scripts/core_workflow/lib/hippunfold_shell/run_hippunfold_local.sh`
- Test: `tests/core_workflow/test_hippunfold_cache_contract.py`

**Interfaces:**
- Consumes: `dataset: Literal["hcp", "stanford", "iowa_jj_tms"]`, digit-only `subject: str`, `dry_run: bool`.
- Produces: `resolve_run_contract(dataset, subject) -> HippUnfoldRunContract`, deterministic command/environment JSON, and a real run through the fixed `hippo2` executable.

- [ ] **Step 1: Write failing behavioral tests**

  Add literal expectations for all three dataset presets, exact HippUnfold argv, forbidden DAG-pruning flags, rejected unknown datasets/unsafe subjects, and the public CLI dry-run JSON. Add an adapter test proving the single shell entrypoint rejects argument counts other than exactly dataset + subject.

- [ ] **Step 2: Run the focused tests and verify RED**

  Run: `/home/yojiao/.conda/envs/py314/bin/python -m pytest tests/core_workflow/test_hippunfold_cache_contract.py -q`

  Expected: failure because `hippunfold_run_contract` and the locked CLI do not yet exist and the current command contains `--omit-from`.

- [ ] **Step 3: Implement the minimal locked module and entrypoint**

  Implement immutable dataset presets, subject validation, canonical paths, exact argv/environment generation, dry-run JSON, local model preflight, and subprocess execution. Replace shell configurability with strict compatibility delegation to the public Python entrypoint.

- [ ] **Step 4: Run the focused tests and verify GREEN**

  Run: `/home/yojiao/.conda/envs/py314/bin/python -m pytest tests/core_workflow/test_hippunfold_cache_contract.py -q`

  Expected: all focused tests pass.

### Task 2: Independent full-output anchors

**Files:**
- Modify: `scripts/core_workflow/01_02_check_hippunfold_outputs.py`
- Modify: `tests/core_workflow/test_hippunfold_output_checker.py`

**Interfaces:**
- Consumes: the cached successful-log output template list.
- Produces: a union contract that always requires bilateral corobl/T1w/unfold den-512 hippocampal midthickness surfaces, the den-512 hippocampal surface spec, and bilateral nnUNet Dice QC files.

- [ ] **Step 1: Write a failing partial-DAG regression test**

  Build a subject fixture with only the four template-copied unfold surfaces plus a successful-looking log/cache and assert `check_subject` reports the fixed missing anchors. Keep a complete fixture green.

- [ ] **Step 2: Run the focused checker test and verify RED**

  Run: `/home/yojiao/.conda/envs/py314/bin/python -m pytest tests/core_workflow/test_hippunfold_output_checker.py -q`

  Expected: the unfold-only fixture is incorrectly accepted before the change.

- [ ] **Step 3: Union fixed anchors into every manifest contract**

  Add subject-templated anchors before hashing, receipt validation, cleanup, and reporting so old partial receipts no longer validate and complete retained HCP/Stanford outputs remain acceptable.

- [ ] **Step 4: Run the focused checker test and verify GREEN**

  Run: `/home/yojiao/.conda/envs/py314/bin/python -m pytest tests/core_workflow/test_hippunfold_output_checker.py -q`

  Expected: all checker tests pass.

### Task 3: Documentation and workflow registry synchronization

**Files:**
- Modify: `scripts/core_workflow/README.md`
- Modify: `docs/environment/hippo2_reproducibility_manifest.md`
- Modify: `scripts/core_workflow/step_registry.json`

**Interfaces:**
- Consumes: the fixed launcher and checker contracts from Tasks 1-2.
- Produces: one documented invocation form, dataset path table, fixed setting table, local-model behavior, and explicit prohibition of free-form HippUnfold/Snakemake parameters.

- [ ] **Step 1: Update the active runbook and environment manifest**

  Document exact HCP, Stanford, and JJ-TMS presets; explain that `output_spaces=corobl` still yields HippUnfold’s internal T1w/unfold surfaces; replace the incorrect `--omit-from` claim; and document the checker anchors.

- [ ] **Step 2: Update stage registry wording**

  Make stage 01 describe the locked multi-dataset T1w/512 launch contract and independent completeness guard.

- [ ] **Step 3: Run contract/help tests**

  Run: `/home/yojiao/.conda/envs/py314/bin/python -m pytest tests/core_workflow/test_core_workflow_traceability_contract.py tests/core_workflow/test_pipeline_numbering_contract.py -q`

  Expected: all contract/help tests pass.

### Task 4: Final verification

**Files:**
- Verify only; no new production files.

**Interfaces:**
- Consumes: all changes from Tasks 1-3.
- Produces: fresh evidence that the fixed launcher, output checker, docs contracts, and shell syntax are valid.

- [ ] **Step 1: Run Python interpreter context check**

  Run: `python -c "import sys; print(sys.executable)"`

- [ ] **Step 2: Run focused and integration tests with py314**

  Run: `/home/yojiao/.conda/envs/py314/bin/python -m pytest tests/core_workflow/test_hippunfold_cache_contract.py tests/core_workflow/test_hippunfold_output_checker.py tests/core_workflow/test_core_workflow_traceability_contract.py tests/core_workflow/test_pipeline_numbering_contract.py -q`

- [ ] **Step 3: Run syntax checks**

  Run: `/home/yojiao/.conda/envs/py314/bin/python -m py_compile scripts/core_workflow/01_01_run_hippunfold_t1w_local.py scripts/core_workflow/01_02_check_hippunfold_outputs.py scripts/core_workflow/lib/hippunfold_run_contract.py`

  Run: `bash -n scripts/core_workflow/lib/hippunfold_shell/run_hippunfold_local.sh`

- [ ] **Step 4: Inspect final diff and workspace status**

  Run: `git diff --check`

  Run: `git status --short`
