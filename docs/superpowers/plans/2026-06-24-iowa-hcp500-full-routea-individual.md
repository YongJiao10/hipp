# Iowa HCP500 Full RouteA Individual Workflow Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a maintained `iowa_hcp500` dataset path from completed QuNex/HCP sessions through local HippUnfold, RouteA preparation, manual Argon ArealMSHBM exchange, and subject-level gradient-spectral K=3/WTA products.

**Architecture:** Centralize HCPpipeline500 session discovery, path templates, lightweight HippUnfold input bridging, and RouteA label validation in one library. Existing core and individual entrypoints become dataset-aware but continue to execute the current scientific algorithms. A thin orchestration CLI performs preflight and emits or executes stage commands; Argon wrappers gain an `iowa_hcp500` preset but never submit from the local agent.

**Tech Stack:** Python 3.14 (`/home/yojiao/.conda/envs/py314/bin/python`), pytest, Bash/SGE, nibabel, NumPy, existing core workflow and HippUnfold 2.0.0 `hippo2` runtime.

---

## File Structure

**Create**

- `scripts/core_workflow/lib/iowa_hcp500.py` — dataset constants, completed-session discovery, manifest writing, narrow HippUnfold BIDS bridge, and RouteA label validation.
- `scripts/core_workflow/00_iowa_hcp500_workflow.py` — preflight and stage command orchestration; dry-run by default.
- `tests/core_workflow/test_iowa_hcp500_workflow.py` — unit and CLI contract tests for discovery, paths, manifests, label validation, and command construction.
- `docs/hipp_parcellation_network/iowa_hcp500_runbook.md` — complete local/Argon handoff and recovery runbook.

**Modify**

- `.gitignore` — retain the existing user change ignoring `outputs_iowa_hcp500/`.
- `scripts/core_workflow/lib/output_layout.py` — retain and test the existing `iowa_hcp500` dataset key.
- `scripts/core_workflow/02_01_prepare_group_inputs.py` — add dynamic `--dataset` defaults and HCP REST template support.
- `scripts/core_workflow/02_02_build_qc_manifest.py` — add dynamic `--dataset` defaults.
- `scripts/core_workflow/03_02_build_kong2022_routea_reference.py` — add dynamic `--dataset` defaults.
- `scripts/core_workflow/04_01_materialize_kong2022_roi_fc.py` — add dynamic `--dataset` defaults.
- `scripts/individual_workflow/routea_individual_gradients_wta.py` — add dataset-aware roots and dataset provenance.
- `scripts/individual_workflow/01_monitor_individual_run.py` — add dataset-aware default work root.
- `scripts/individual_workflow/02_build_individual_overviews.py` — add dataset-aware work/present/adapter/input roots.
- `scripts/individual_workflow/run_routea_individual_gradients_wta_background.sh` — accept a dataset and resolve the correct output root.
- `tests/core_workflow/test_path_templates.py` — extend output-layout isolation coverage.
- `tests/individual_workflow/test_routea_individual_gradients_wta.py` — test dataset-aware individual defaults and provenance.
- `tests/core_workflow/test_routea_flat_label_contract.py` — test the Argon `iowa_hcp500` preset and documentation contract.
- `scripts/ArealMSHBM_Kong2022_Iowa_Stanford/prepare_kong2022_areal_mshbm_subject_inputs.py` — resolve HCPpipeline500 session paths.
- `scripts/ArealMSHBM_Kong2022_Iowa_Stanford/submit_kong2022_areal_mshbm.sh` — add `DATASET=iowa_hcp500`.
- `scripts/ArealMSHBM_Kong2022_Iowa_Stanford/run_kong2022_areal_mshbm_subject_sge.sh` — add the preset and safe temporary path.
- `scripts/ArealMSHBM_Kong2022_Iowa_Stanford/report_kong2022_areal_mshbm_status.sh` — add the preset.
- `scripts/ArealMSHBM_Kong2022_Iowa_Stanford/README.md` — document exact manual Argon commands.
- `scripts/core_workflow/README.md` — finish the existing dataset documentation update.
- `scripts/individual_workflow/README.md` — document dataset-aware individual execution.
- `docs/hipp_parcellation_network/individual_workflow_routea_gradients_wta.md` — document `iowa_hcp500` roots and commands.
- `docs/hipp_parcellation_network/multi_branch_flow.md` — add the dataset path without changing algorithms.
- `docs/hipp_parcellation_network/step_by_step_flows.md` — add the complete stage sequence.
- `docs/environment/agent_runtime_contract.md` — document explicit py314 and hippo2 commands used by the new launcher.

The pre-existing changes to `.gitignore`, `scripts/core_workflow/README.md`, and
`scripts/core_workflow/lib/output_layout.py` belong to the user. Inspect and
extend them; do not replace or revert them. Do not include
`HANDOFF_stanford_full_pipeline.md` in this feature unless a directly conflicting
statement must be corrected.

### Task 1: Lock the dataset output-layout contract

**Files:**

- Modify: `tests/core_workflow/test_path_templates.py`
- Modify: `scripts/core_workflow/lib/output_layout.py`
- Modify: `.gitignore`

- [ ] **Step 1: Add failing output-isolation tests**

Append:

```python
from scripts.core_workflow.lib.output_layout import (
    DATASET_KEYS,
    resolve_hippunfold_batch_root,
    resolve_kong2022_manifest_root,
    resolve_outputs_root,
    resolve_overviews_dir,
    resolve_work_root,
)


def test_iowa_hcp500_dataset_roots_are_isolated():
    assert "iowa_hcp500" in DATASET_KEYS
    assert resolve_outputs_root("iowa_hcp500") == REPO_ROOT / "outputs_iowa_hcp500"
    assert resolve_hippunfold_batch_root("iowa_hcp500").parts[-4:] == (
        "inputs", "iowa_hcp500", "hippunfold", "dense_corobl_batch"
    )
    assert resolve_kong2022_manifest_root("iowa_hcp500").parts[-4:] == (
        "inputs", "iowa_hcp500", "kong2022", "manifests"
    )
    assert resolve_work_root("iowa_hcp500") == REPO_ROOT / "outputs_iowa_hcp500" / "work"
    assert resolve_overviews_dir("iowa_hcp500") == (
        REPO_ROOT / "outputs_iowa_hcp500" / "present" / "overviews"
    )


def test_iowa_hcp500_does_not_alias_existing_iowa():
    assert resolve_outputs_root("iowa_hcp500") != resolve_outputs_root("iowa")
```

Add `REPO_ROOT = Path(__file__).resolve().parents[2]` if the test module does
not already define it.

- [ ] **Step 2: Run the tests and verify the intended red state**

Run:

```bash
python -c "import sys; print(sys.executable)"
/home/yojiao/.conda/envs/py314/bin/python -m pytest tests/core_workflow/test_path_templates.py -q
```

Expected: the tests fail if the current uncommitted dataset-key implementation
is absent or incomplete. If they already pass because of the user's existing
change, record that this task is characterization of pre-existing work and do
not manufacture a failure by reverting it.

- [ ] **Step 3: Complete the minimal output-layout implementation**

Ensure:

```python
DATASET_KEYS = ("hcp", "iowa", "stanford", "iowa_hcp500")
```

Keep `resolve_outputs_root()` generic:

```python
def resolve_outputs_root(dataset: str = DEFAULT_DATASET) -> Path:
    return REPO_ROOT / f"outputs_{_validate_dataset(dataset)}"
```

Ensure `.gitignore` contains exactly one:

```gitignore
outputs_iowa_hcp500/
```

- [ ] **Step 4: Run the focused tests**

Run the Task 1 pytest command again.

Expected: PASS.

- [ ] **Step 5: Commit only Task 1 files**

```bash
git add .gitignore scripts/core_workflow/lib/output_layout.py tests/core_workflow/test_path_templates.py
git commit -m "feat: register iowa hcp500 output layout"
```

### Task 2: Implement completed-session discovery and manifests

**Files:**

- Create: `scripts/core_workflow/lib/iowa_hcp500.py`
- Create: `tests/core_workflow/test_iowa_hcp500_workflow.py`

- [ ] **Step 1: Write failing discovery tests**

Create fixtures with this shape:

```python
def make_session(root: Path, subject: str, *, t1=True, t2=True, rest=True) -> Path:
    base = root / subject / "hcp" / subject
    required = {
        "t1": base / "T1w" / "T1w_acpc_dc_restore.nii.gz",
        "t2": base / "T1w" / "T2w_acpc_dc_restore.nii.gz",
        "rest": base / "MNINonLinear" / "Results" / "REST"
        / "REST_Atlas_MSMAll_hp0_clean_rclean.dtseries.nii",
    }
    for enabled, path in ((t1, required["t1"]), (t2, required["t2"]), (rest, required["rest"])):
        if enabled:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"x")
    return base


def test_discovery_includes_only_complete_sessions(tmp_path):
    make_session(tmp_path, "04172026")
    make_session(tmp_path, "05012026", t2=False)
    rows = discover_sessions(tmp_path)
    assert [row.subject for row in rows] == ["04172026", "05012026"]
    assert rows[0].eligible is True
    assert rows[0].missing == ()
    assert rows[1].eligible is False
    assert rows[1].missing == ("t2w",)


def test_manifest_writer_is_deterministic_and_smoke_is_fixed(tmp_path):
    sessions = tmp_path / "sessions"
    make_session(sessions, "06012026")
    make_session(sessions, "04172026")
    out = tmp_path / "manifests"
    paths = write_discovery_products(sessions_root=sessions, manifest_dir=out)
    assert paths.smoke.read_text().splitlines() == ["04172026"]
    assert paths.full.read_text().splitlines() == ["04172026", "06012026"]
    assert "missing_t1w" in paths.status.read_text() or "eligible" in paths.status.read_text()
```

- [ ] **Step 2: Verify red**

```bash
/home/yojiao/.conda/envs/py314/bin/python -m pytest \
  tests/core_workflow/test_iowa_hcp500_workflow.py -q
```

Expected: FAIL because `lib.iowa_hcp500` does not exist.

- [ ] **Step 3: Implement the discovery library**

Define:

```python
DEFAULT_HCP500_ROOT = Path("/home/yojiao/hpchome/HCPpipeline500")
DEFAULT_STUDY_ROOT = DEFAULT_HCP500_ROOT / "work" / "qunex_iowa7t_hcppipeline500"
DEFAULT_SESSIONS_ROOT = DEFAULT_STUDY_ROOT / "sessions"
SMOKE_SUBJECT = "04172026"


@dataclass(frozen=True)
class SessionPaths:
    subject: str
    subject_root: Path
    t1w: Path
    t2w: Path
    rest_dtseries: Path


@dataclass(frozen=True)
class SessionStatus:
    subject: str
    paths: SessionPaths
    eligible: bool
    missing: tuple[str, ...]


@dataclass(frozen=True)
class DiscoveryProducts:
    status: Path
    smoke: Path
    full: Path
```

Implement `session_paths()`, `discover_sessions()`, and
`write_discovery_products()`. Scan with `Path.iterdir()` only at the immediate
session level. Treat non-files and zero-byte files as missing. Write:

```text
iowa_hcp500_session_status.tsv
iowa_hcp500_smoke_04172026.txt
iowa_hcp500_eligible_sessions.txt
```

The TSV columns are:

```text
subject eligible missing t1w t2w rest_dtseries
```

Raise a clear error if `04172026` is not eligible when writing the smoke
manifest.

- [ ] **Step 4: Verify green**

Run the Task 2 pytest command.

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add scripts/core_workflow/lib/iowa_hcp500.py tests/core_workflow/test_iowa_hcp500_workflow.py
git commit -m "feat: discover completed iowa hcp500 sessions"
```

### Task 3: Add the narrow HippUnfold input bridge and label validator

**Files:**

- Modify: `scripts/core_workflow/lib/iowa_hcp500.py`
- Modify: `tests/core_workflow/test_iowa_hcp500_workflow.py`

- [ ] **Step 1: Add failing bridge and validation tests**

Add tests:

```python
def test_materialize_hippunfold_input_bridge_links_only_t1w_and_t2w(tmp_path):
    sessions = tmp_path / "sessions"
    source = make_session(sessions, "04172026")
    status = discover_sessions(sessions)[0]
    bids_root = tmp_path / "bids"
    outputs = materialize_hippunfold_input(status, bids_root)
    assert outputs.t1w == bids_root / "sub-04172026" / "anat" / "sub-04172026_T1w.nii.gz"
    assert outputs.t2w == bids_root / "sub-04172026" / "anat" / "sub-04172026_T2w.nii.gz"
    assert outputs.t1w.is_symlink()
    assert outputs.t1w.resolve() == source / "T1w" / "T1w_acpc_dc_restore.nii.gz"
    assert outputs.t2w.resolve() == source / "T1w" / "T2w_acpc_dc_restore.nii.gz"
    assert len(list((bids_root / "sub-04172026").rglob("*"))) == 3


def test_validate_routea_labels_requires_two_integer_32k_vectors(tmp_path):
    subject_dir = tmp_path / "sub-04172026"
    subject_dir.mkdir()
    np.savetxt(subject_dir / "lh_labels.txt", np.ones(32492), fmt="%d")
    np.savetxt(subject_dir / "rh_labels.txt", np.ones(32492), fmt="%d")
    result = validate_routea_labels(subject_dir)
    assert result.valid is True
    assert result.left_count == 32492
    assert result.right_count == 32492


def test_validate_routea_labels_reports_missing_or_noninteger_data(tmp_path):
    subject_dir = tmp_path / "sub-04172026"
    subject_dir.mkdir()
    np.savetxt(subject_dir / "lh_labels.txt", [1.5, 2.0])
    result = validate_routea_labels(subject_dir)
    assert result.valid is False
    assert "missing_rh" in result.errors
    assert "lh_non_integer" in result.errors
```

- [ ] **Step 2: Verify red**

Run the Task 2 pytest command.

Expected: FAIL for missing bridge/validator APIs.

- [ ] **Step 3: Implement minimal bridge and validator**

Add:

```python
@dataclass(frozen=True)
class HippunfoldBridge:
    subject: str
    root: Path
    t1w: Path
    t2w: Path


@dataclass(frozen=True)
class RouteALabelValidation:
    valid: bool
    left_count: int
    right_count: int
    errors: tuple[str, ...]
```

`materialize_hippunfold_input()` creates only:

```text
sub-<id>/anat/sub-<id>_T1w.nii.gz
sub-<id>/anat/sub-<id>_T2w.nii.gz
```

using `Path.symlink_to()` and refuses to replace a path that points to a
different source. `validate_routea_labels()` uses NumPy, expects 32,492 values
per hemisphere, finite values, and integer-like values via
`np.allclose(values, np.rint(values))`.

- [ ] **Step 4: Verify green**

Run the Task 2 pytest command.

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add scripts/core_workflow/lib/iowa_hcp500.py tests/core_workflow/test_iowa_hcp500_workflow.py
git commit -m "feat: prepare iowa hcp500 hippunfold inputs"
```

### Task 4: Build the dry-run-first local orchestration CLI

**Files:**

- Create: `scripts/core_workflow/00_iowa_hcp500_workflow.py`
- Modify: `tests/core_workflow/test_iowa_hcp500_workflow.py`

- [ ] **Step 1: Add failing CLI and command-construction tests**

Load the script with `importlib.util` and test:

```python
def test_smoke_stage_commands_use_iowa_hcp500_roots(tmp_path):
    layout = build_layout(repo_root=REPO_ROOT, hcp500_root=tmp_path / "HCPpipeline500")
    commands = build_stage_commands(
        layout=layout,
        manifest=tmp_path / "smoke.txt",
        stages=["hippunfold", "surface", "routea_reference", "roi_fc", "individual", "overview"],
    )
    text = "\n".join(shlex.join(cmd) for cmd in commands)
    assert "--dataset iowa_hcp500" in text
    assert "outputs_iowa_hcp500" in text
    assert "REST_Atlas_MSMAll_hp0_clean_rclean.dtseries.nii" in text
    assert "outputs_hcp" not in text


def test_cli_defaults_to_dry_run():
    args = build_arg_parser().parse_args(["discover"])
    assert args.execute is False
```

- [ ] **Step 2: Verify red**

Run the Task 2 pytest command.

Expected: FAIL because the orchestration module does not exist.

- [ ] **Step 3: Implement the CLI**

Subcommands:

```text
discover
prepare-hippunfold-inputs
validate-routea-labels
commands
run
```

Shared arguments:

```text
--dataset iowa_hcp500
--hcp500-root /home/yojiao/hpchome/HCPpipeline500
--subjects-manifest <path>
--scope smoke|full
--execute
```

`discover` and `prepare-hippunfold-inputs` perform lightweight local writes.
`validate-routea-labels` reads the returned label root and exits non-zero on
invalid subjects. `commands` prints shell-quoted commands. `run` executes only
when `--execute` is present.

Construct commands with argument arrays, never shell strings. Use:

```python
PY314 = Path("/home/yojiao/.conda/envs/py314/bin/python")
DATASET = "iowa_hcp500"
REST_TEMPLATE = (
    "{data_root}/{subject}/hcp/{subject}/MNINonLinear/Results/REST/"
    "REST_Atlas_MSMAll_hp0_clean_rclean.dtseries.nii"
)
```

The command order is:

1. single-subject `01_01_run_hippunfold_t1w_local.py`;
2. `01_02_check_hippunfold_outputs.py`;
3. `02_01_prepare_group_inputs.py`;
4. `02_02_build_qc_manifest.py`;
5. local RouteA-label validation gate;
6. `03_02_build_kong2022_routea_reference.py`;
7. `04_01_materialize_kong2022_roi_fc.py`;
8. individual compute;
9. overview builder.

Do not add group-gradient or group-atlas stages.

- [ ] **Step 4: Verify green and CLI help**

```bash
/home/yojiao/.conda/envs/py314/bin/python -m pytest \
  tests/core_workflow/test_iowa_hcp500_workflow.py -q
/home/yojiao/.conda/envs/py314/bin/python \
  scripts/core_workflow/00_iowa_hcp500_workflow.py --help
```

Expected: PASS and exit 0.

- [ ] **Step 5: Commit**

```bash
git add scripts/core_workflow/00_iowa_hcp500_workflow.py tests/core_workflow/test_iowa_hcp500_workflow.py
git commit -m "feat: orchestrate iowa hcp500 local workflow"
```

### Task 5: Make core RouteA stages dataset-aware

**Files:**

- Modify: `scripts/core_workflow/02_01_prepare_group_inputs.py`
- Modify: `scripts/core_workflow/02_02_build_qc_manifest.py`
- Modify: `scripts/core_workflow/03_02_build_kong2022_routea_reference.py`
- Modify: `scripts/core_workflow/04_01_materialize_kong2022_roi_fc.py`
- Modify: `tests/core_workflow/test_iowa_hcp500_workflow.py`

- [ ] **Step 1: Add failing parser-default tests**

For each module, parse:

```python
args = module.build_arg_parser().parse_args(["--dataset", "iowa_hcp500"])
args = module.apply_dataset_defaults(args)
```

Assert:

```python
assert "outputs_iowa_hcp500" in str(args.shared_root)
assert "outputs_hcp" not in str(args.shared_root)
```

Use stage-specific assertions for `batch_root`, `route_root`, `template_root`,
`manifest_dir`, `route_a_root`, and `hippunfold_batch_root`.

- [ ] **Step 2: Verify red**

Run:

```bash
/home/yojiao/.conda/envs/py314/bin/python -m pytest \
  tests/core_workflow/test_iowa_hcp500_workflow.py -q
```

Expected: FAIL because these parsers lack `--dataset` and
`apply_dataset_defaults()`.

- [ ] **Step 3: Implement dynamic defaults**

In each module:

```python
parser.add_argument("--dataset", default=DEFAULT_DATASET, choices=DATASET_KEYS)
```

Change dataset-scoped root defaults to `None`, then fill only missing values in
`apply_dataset_defaults(args)`. Example:

```python
def apply_dataset_defaults(args: argparse.Namespace) -> argparse.Namespace:
    dataset = str(args.dataset)
    if args.shared_root is None:
        args.shared_root = str(resolve_kong2022_surface_root(dataset))
    if args.hippunfold_batch_root is None:
        args.hippunfold_batch_root = str(resolve_hippunfold_batch_root(dataset))
    return args
```

Call this immediately after parsing. Explicit user-provided roots always win.
For `02_01`, keep `--bold-root` explicit and pass the HCPpipeline500
`--bold-template` from orchestration.

- [ ] **Step 4: Run focused and regression tests**

```bash
/home/yojiao/.conda/envs/py314/bin/python -m pytest \
  tests/core_workflow/test_iowa_hcp500_workflow.py \
  tests/core_workflow/test_path_templates.py \
  tests/core_workflow/test_routea_flat_label_contract.py -q
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add \
  scripts/core_workflow/02_01_prepare_group_inputs.py \
  scripts/core_workflow/02_02_build_qc_manifest.py \
  scripts/core_workflow/03_02_build_kong2022_routea_reference.py \
  scripts/core_workflow/04_01_materialize_kong2022_roi_fc.py \
  tests/core_workflow/test_iowa_hcp500_workflow.py
git commit -m "feat: resolve core routea stages by dataset"
```

### Task 6: Make individual compute, monitor, launcher, and overview dataset-aware

**Files:**

- Modify: `scripts/individual_workflow/routea_individual_gradients_wta.py`
- Modify: `scripts/individual_workflow/01_monitor_individual_run.py`
- Modify: `scripts/individual_workflow/02_build_individual_overviews.py`
- Modify: `scripts/individual_workflow/run_routea_individual_gradients_wta_background.sh`
- Modify: `tests/individual_workflow/test_routea_individual_gradients_wta.py`

- [ ] **Step 1: Add failing dataset-default tests**

Add:

```python
def test_iowa_hcp500_parser_defaults_are_isolated():
    workflow = load_module()
    args = workflow.build_arg_parser().parse_args(["--dataset", "iowa_hcp500"])
    args = workflow.apply_dataset_defaults(args)
    roots = [
        args.subjects_manifest,
        args.shared_root,
        args.surface_root,
        args.hippunfold_batch_root,
        args.out_root,
    ]
    assert all("outputs_iowa_hcp500" in str(path) for path in roots)
    assert all("outputs_hcp" not in str(path) for path in roots)


def test_combo_manifest_records_dataset(tmp_path, monkeypatch):
    # Use the existing process_combo fixture/mocking pattern.
    # Assert the written manifest contains:
    assert manifest["dataset"] == "iowa_hcp500"
```

Add importable parser tests for monitor and overview modules, asserting their
default roots resolve to `outputs_iowa_hcp500` after
`apply_dataset_defaults()`.

- [ ] **Step 2: Verify red**

```bash
/home/yojiao/.conda/envs/py314/bin/python -m pytest \
  tests/individual_workflow/test_routea_individual_gradients_wta.py -q
```

Expected: FAIL because dataset-aware defaults are absent.

- [ ] **Step 3: Implement dynamic individual defaults**

Import `DATASET_KEYS` and resolver functions. Add:

```python
parser.add_argument("--dataset", default=DEFAULT_DATASET, choices=DATASET_KEYS)
```

Use `None` for dataset-scoped root defaults and fill them in
`apply_dataset_defaults()`. The individual defaults are:

```python
subjects_manifest = resolve_kong2022_manifest_root(dataset) / "iowa_hcp500_eligible_sessions.txt"
shared_root = resolve_kong2022_routea_root(dataset)
surface_root = resolve_kong2022_surface_root(dataset)
hippunfold_batch_root = resolve_hippunfold_batch_root(dataset)
out_root = resolve_work_root(dataset) / "individual_atlas"
overview_root = resolve_overviews_dir(dataset) / "individual"
adapter_root = out_root / "_overview_core_adapter"
adapter_present_root = adapter_root / "present_tmp"
```

For HCP, preserve the existing latest HCP subject-manifest filename rather than
using the Iowa filename. Add `"dataset": config["dataset"]` to successful and
failed combo manifests and include dataset in run handoff text.

The background launcher accepts:

```bash
DATASET="${2:-hcp}"
```

and computes paths through a short py314 resolver call or accepts explicit
`OUT_ROOT`; do not hard-code `outputs_hcp`.

- [ ] **Step 4: Run tests and shell syntax**

```bash
/home/yojiao/.conda/envs/py314/bin/python -m pytest \
  tests/individual_workflow/test_routea_individual_gradients_wta.py -q
bash -n scripts/individual_workflow/run_routea_individual_gradients_wta_background.sh
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add \
  scripts/individual_workflow/routea_individual_gradients_wta.py \
  scripts/individual_workflow/01_monitor_individual_run.py \
  scripts/individual_workflow/02_build_individual_overviews.py \
  scripts/individual_workflow/run_routea_individual_gradients_wta_background.sh \
  tests/individual_workflow/test_routea_individual_gradients_wta.py
git commit -m "feat: add dataset-aware individual routea workflow"
```

### Task 7: Add the Argon `iowa_hcp500` preset

**Files:**

- Modify: `scripts/ArealMSHBM_Kong2022_Iowa_Stanford/prepare_kong2022_areal_mshbm_subject_inputs.py`
- Modify: `scripts/ArealMSHBM_Kong2022_Iowa_Stanford/submit_kong2022_areal_mshbm.sh`
- Modify: `scripts/ArealMSHBM_Kong2022_Iowa_Stanford/run_kong2022_areal_mshbm_subject_sge.sh`
- Modify: `scripts/ArealMSHBM_Kong2022_Iowa_Stanford/report_kong2022_areal_mshbm_status.sh`
- Modify: `tests/core_workflow/test_routea_flat_label_contract.py`
- Modify: `tests/core_workflow/test_iowa_hcp500_workflow.py`

- [ ] **Step 1: Add failing Python input-preparer test**

Create a minimal valid CIFTI dtseries fixture using nibabel or monkeypatch
`dtseries_n_timepoints()`, then run the preparer with:

```text
--dataset iowa_hcp500
--subject 04172026
--data-root <tmp>/sessions
```

Assert the selected path is:

```text
<tmp>/sessions/04172026/hcp/04172026/MNINonLinear/Results/REST/
REST_Atlas_MSMAll_hp0_clean_rclean.dtseries.nii
```

and the manifest records one run.

- [ ] **Step 2: Add failing shell preset contract tests**

Read the three shell scripts and assert they contain:

```text
iowa_hcp500
/old_Users/yojiao/HCPpipeline500/work/qunex_iowa7t_hcppipeline500/sessions
outputs_iowa_hcp500/inputs/iowa_hcp500/kong2022/cortex_references/routea_areal_mshbm
```

Also run the submitter in a temporary fixture:

```bash
DATASET=iowa_hcp500
DATA_ROOT=<tmp>/sessions
OUT_ROOT=<tmp>/labels
SUBJECT_IDS_CSV=04172026
DRY_RUN=1
```

Assert exit 0 and printed qsub commands, with no qsub execution.

- [ ] **Step 3: Verify red**

```bash
/home/yojiao/.conda/envs/py314/bin/python -m pytest \
  tests/core_workflow/test_iowa_hcp500_workflow.py \
  tests/core_workflow/test_routea_flat_label_contract.py -q
```

Expected: FAIL because the preset is incomplete.

- [ ] **Step 4: Implement the input-preparer preset**

Replace the glob-only mapping with a dataset spec:

```python
DATASET_SPECS = {
    "jj_tms": {"subject_prefix": "sub-", "relative_globs": ["func/..."]},
    "stanford": {"subject_prefix": "sub-", "relative_globs": ["func/..."]},
    "iowa7t": {"subject_prefix": "sub-", "relative_globs": ["func/..."]},
    "iowa_hcp500": {
        "subject_prefix": "",
        "relative_globs": [
            "hcp/{subject}/MNINonLinear/Results/REST/"
            "REST_Atlas_MSMAll_hp0_clean_rclean.dtseries.nii"
        ],
    },
}
```

Resolve paths relative to `data_root / subject_id` for HCP500 and retain the
current `sub-<id>/func` behavior for existing presets.

- [ ] **Step 5: Implement shell presets**

Use these Argon defaults:

```bash
default_data_root="/old_Users/yojiao/HCPpipeline500/work/qunex_iowa7t_hcppipeline500/sessions"
default_out_root="${repo_root}/outputs_iowa_hcp500/inputs/iowa_hcp500/kong2022/cortex_references/routea_areal_mshbm"
job_prefix="ia500_k22"
manifest_prefix="iowa_hcp500"
tmp_prefix="iowa_hcp500_kong2022"
```

When auto-discovering subjects for this preset, inspect only immediate session
directories and require the final REST dtseries. Extend the cleanup allowlist
with `/tmp/iowa_hcp500_kong2022_*`.

- [ ] **Step 6: Verify tests and shell syntax**

```bash
/home/yojiao/.conda/envs/py314/bin/python -m pytest \
  tests/core_workflow/test_iowa_hcp500_workflow.py \
  tests/core_workflow/test_routea_flat_label_contract.py -q
bash -n scripts/ArealMSHBM_Kong2022_Iowa_Stanford/submit_kong2022_areal_mshbm.sh
bash -n scripts/ArealMSHBM_Kong2022_Iowa_Stanford/run_kong2022_areal_mshbm_subject_sge.sh
bash -n scripts/ArealMSHBM_Kong2022_Iowa_Stanford/report_kong2022_areal_mshbm_status.sh
```

Expected: PASS.

- [ ] **Step 7: Commit**

```bash
git add \
  scripts/ArealMSHBM_Kong2022_Iowa_Stanford/prepare_kong2022_areal_mshbm_subject_inputs.py \
  scripts/ArealMSHBM_Kong2022_Iowa_Stanford/submit_kong2022_areal_mshbm.sh \
  scripts/ArealMSHBM_Kong2022_Iowa_Stanford/run_kong2022_areal_mshbm_subject_sge.sh \
  scripts/ArealMSHBM_Kong2022_Iowa_Stanford/report_kong2022_areal_mshbm_status.sh \
  tests/core_workflow/test_routea_flat_label_contract.py \
  tests/core_workflow/test_iowa_hcp500_workflow.py
git commit -m "feat: add iowa hcp500 argon routea preset"
```

### Task 8: Synchronize workflow and handoff documentation

**Files:**

- Create: `docs/hipp_parcellation_network/iowa_hcp500_runbook.md`
- Modify: `scripts/core_workflow/README.md`
- Modify: `scripts/individual_workflow/README.md`
- Modify: `docs/hipp_parcellation_network/individual_workflow_routea_gradients_wta.md`
- Modify: `docs/hipp_parcellation_network/multi_branch_flow.md`
- Modify: `docs/hipp_parcellation_network/step_by_step_flows.md`
- Modify: `docs/environment/agent_runtime_contract.md`
- Modify: `scripts/ArealMSHBM_Kong2022_Iowa_Stanford/README.md`
- Modify: `tests/core_workflow/test_iowa_hcp500_workflow.py`

- [ ] **Step 1: Add failing documentation-contract test**

Assert the combined documentation contains:

```text
iowa_hcp500
04172026
REST_Atlas_MSMAll_hp0_clean_rclean.dtseries.nii
00_iowa_hcp500_workflow.py
DATASET=iowa_hcp500
DRY_RUN=1
TASK_DRY_RUN=1
outputs_iowa_hcp500
/home/yojiao/.conda/envs/py314/bin/python
```

Assert the runbook has explicit headings:

```text
Local Preflight
HippUnfold Gate
Argon Manual Steps
Return RouteA Labels
Resume Local Workflow
Verification
```

- [ ] **Step 2: Verify red**

Run the Task 2 pytest command.

Expected: FAIL because the runbook and synchronized documentation are absent.

- [ ] **Step 3: Write the runbook and update maintained docs**

The Argon section must present these steps as user-operated:

```bash
# On Argon: inspect generated qsub commands only
DATASET=iowa_hcp500 \
SUBJECT_IDS_CSV=04172026 \
DRY_RUN=1 \
scripts/ArealMSHBM_Kong2022_Iowa_Stanford/submit_kong2022_areal_mshbm.sh

# On Argon: submit compute-node preflight only
DATASET=iowa_hcp500 \
SUBJECT_IDS_CSV=04172026 \
TASK_DRY_RUN=1 \
scripts/ArealMSHBM_Kong2022_Iowa_Stanford/submit_kong2022_areal_mshbm.sh

# On Argon: real one-subject smoke, only after reviewing the dry-run
DATASET=iowa_hcp500 \
SUBJECT_IDS_CSV=04172026 \
SMOKE_N=1 \
scripts/ArealMSHBM_Kong2022_Iowa_Stanford/submit_kong2022_areal_mshbm.sh
```

Document that full submission uses the generated eligible-session manifest and
requires user approval after the smoke outputs exist. Include exact local label
destination and validator command.

Document the hippo2 gate exactly as required by `AGENTS.md`; do not use package
metadata as version proof.

- [ ] **Step 4: Verify documentation tests**

Run the Task 2 pytest command.

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add \
  docs/hipp_parcellation_network/iowa_hcp500_runbook.md \
  scripts/core_workflow/README.md \
  scripts/individual_workflow/README.md \
  docs/hipp_parcellation_network/individual_workflow_routea_gradients_wta.md \
  docs/hipp_parcellation_network/multi_branch_flow.md \
  docs/hipp_parcellation_network/step_by_step_flows.md \
  docs/environment/agent_runtime_contract.md \
  scripts/ArealMSHBM_Kong2022_Iowa_Stanford/README.md \
  tests/core_workflow/test_iowa_hcp500_workflow.py
git commit -m "docs: add iowa hcp500 routea runbook"
```

### Task 9: Run local smoke preflight without starting substantial computation

**Files:**

- Runtime outputs only under `outputs_iowa_hcp500/`

- [ ] **Step 1: Verify the repository interpreter context**

```bash
python -c "import sys; print(sys.executable)"
/home/yojiao/.conda/envs/py314/bin/python -c \
  "import sys, nibabel, brainspace; print(sys.executable, nibabel.__version__, brainspace.__version__)"
```

Expected: the second command reports the py314 executable and valid package
versions.

- [ ] **Step 2: Discover sessions and create smoke/full manifests**

```bash
/home/yojiao/.conda/envs/py314/bin/python \
  scripts/core_workflow/00_iowa_hcp500_workflow.py discover \
  --scope smoke
```

Expected: `04172026` eligible; smoke and full manifests are written under
`outputs_iowa_hcp500/inputs/iowa_hcp500/kong2022/manifests/`.

- [ ] **Step 3: Prepare the narrow HippUnfold input bridge**

```bash
/home/yojiao/.conda/envs/py314/bin/python \
  scripts/core_workflow/00_iowa_hcp500_workflow.py prepare-hippunfold-inputs \
  --scope smoke
```

Expected: only T1w/T2w links are created under the dataset-scoped HippUnfold
input root.

- [ ] **Step 4: Print the smoke commands**

```bash
/home/yojiao/.conda/envs/py314/bin/python \
  scripts/core_workflow/00_iowa_hcp500_workflow.py commands \
  --scope smoke
```

Expected: commands refer to `04172026`, `outputs_iowa_hcp500`, and the final
HCP REST dtseries; none refer to `outputs_hcp`.

- [ ] **Step 5: Run the Argon launcher dry-run locally against explicit fixture paths**

Use a temporary output root and explicit subject list so no scheduler call is
made:

```bash
DATASET=iowa_hcp500 \
DATA_ROOT=/home/yojiao/hpchome/HCPpipeline500/work/qunex_iowa7t_hcppipeline500/sessions \
OUT_ROOT=/tmp/iowa_hcp500_argon_dryrun \
SUBJECT_IDS_CSV=04172026 \
DRY_RUN=1 \
scripts/ArealMSHBM_Kong2022_Iowa_Stanford/submit_kong2022_areal_mshbm.sh
```

Expected: exit 0 and printed UI qsub command; no qsub submission.

Do not remove `/tmp/iowa_hcp500_argon_dryrun` without presenting the deletion
target to the user and receiving explicit confirmation.

### Task 10: Full verification and requirements audit

**Files:**

- All changed files

- [ ] **Step 1: Run syntax and compile checks**

```bash
/home/yojiao/.conda/envs/py314/bin/python -m py_compile \
  scripts/core_workflow/lib/iowa_hcp500.py \
  scripts/core_workflow/00_iowa_hcp500_workflow.py \
  scripts/core_workflow/02_01_prepare_group_inputs.py \
  scripts/core_workflow/02_02_build_qc_manifest.py \
  scripts/core_workflow/03_02_build_kong2022_routea_reference.py \
  scripts/core_workflow/04_01_materialize_kong2022_roi_fc.py \
  scripts/individual_workflow/routea_individual_gradients_wta.py \
  scripts/individual_workflow/01_monitor_individual_run.py \
  scripts/individual_workflow/02_build_individual_overviews.py \
  scripts/ArealMSHBM_Kong2022_Iowa_Stanford/prepare_kong2022_areal_mshbm_subject_inputs.py
bash -n scripts/individual_workflow/run_routea_individual_gradients_wta_background.sh
bash -n scripts/ArealMSHBM_Kong2022_Iowa_Stanford/submit_kong2022_areal_mshbm.sh
bash -n scripts/ArealMSHBM_Kong2022_Iowa_Stanford/run_kong2022_areal_mshbm_subject_sge.sh
bash -n scripts/ArealMSHBM_Kong2022_Iowa_Stanford/report_kong2022_areal_mshbm_status.sh
```

Expected: all exit 0.

- [ ] **Step 2: Run focused tests**

```bash
/home/yojiao/.conda/envs/py314/bin/python -m pytest \
  tests/core_workflow/test_iowa_hcp500_workflow.py \
  tests/core_workflow/test_path_templates.py \
  tests/core_workflow/test_routea_flat_label_contract.py \
  tests/individual_workflow/test_routea_individual_gradients_wta.py -q
```

Expected: all pass.

- [ ] **Step 3: Run the full relevant workflow test suites**

```bash
/home/yojiao/.conda/envs/py314/bin/python -m pytest \
  tests/core_workflow \
  tests/individual_workflow -q
```

Expected: all pass. Do not claim completion if any failure remains.

- [ ] **Step 4: Verify CLI help**

```bash
/home/yojiao/.conda/envs/py314/bin/python scripts/core_workflow/00_iowa_hcp500_workflow.py --help
/home/yojiao/.conda/envs/py314/bin/python scripts/core_workflow/02_01_prepare_group_inputs.py --help
/home/yojiao/.conda/envs/py314/bin/python scripts/core_workflow/03_02_build_kong2022_routea_reference.py --help
/home/yojiao/.conda/envs/py314/bin/python scripts/core_workflow/04_01_materialize_kong2022_roi_fc.py --help
/home/yojiao/.conda/envs/py314/bin/python scripts/individual_workflow/routea_individual_gradients_wta.py --help
/home/yojiao/.conda/envs/py314/bin/python scripts/individual_workflow/02_build_individual_overviews.py --help
```

Expected: all exit 0 and list `iowa_hcp500` where dataset choices are shown.

- [ ] **Step 5: Audit paths and requirements**

```bash
git diff --check
git status --short
rg -n "outputs_hcp" \
  scripts/core_workflow/00_iowa_hcp500_workflow.py \
  docs/hipp_parcellation_network/iowa_hcp500_runbook.md
```

Expected: no whitespace errors. Any `outputs_hcp` occurrence in the new
dataset-specific files must be an explicit negative example, not a runtime
default.

Review each acceptance criterion from:

```text
docs/superpowers/specs/2026-06-24-iowa-hcp500-full-routea-individual-design.md
```

Report separately:

- locally implemented and verified;
- blocked pending the user's Argon smoke;
- commands the user must execute on Argon;
- exact label paths to return before local RouteA reference/FC and individual
  computation can continue.

- [ ] **Step 6: Commit final verification-only fixes, if any**

Stage only files changed by those fixes and use:

```bash
git commit -m "test: verify iowa hcp500 workflow contracts"
```

Do not create an empty commit.
