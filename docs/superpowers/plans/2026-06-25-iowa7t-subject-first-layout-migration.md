# Iowa7T Subject-First Layout Migration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement and apply the fixed Iowa7T 36P subject-first hippocampal output layout for `04172026` and prepare `06112026` with separate `multiband/` and `multi_echo/` branches, without creating empty future directories.

**Architecture:** Add Iowa7T-specific canonical path helpers to the existing layout authority, then add a manifest-driven migration planner that copies legacy outputs into canonical locations and verifies them before any old path is removed. The migration creates only target parent directories for real transferred files or symlinks; `group/` is not created unless a non-empty group-scoped source is migrated.

**Tech Stack:** Python stdlib (`pathlib`, `dataclasses`, `json`, `shutil`, `filecmp`), pytest, existing py314 runtime (`/home/yojiao/.conda/envs/py314/bin/python`), Jianglab local/Argon path mapping.

---

## Preconditions

- Current branch is `main`. Do not execute code-changing implementation tasks on `main` until the user explicitly confirms main-branch implementation or requests an isolated worktree.
- Production Jianglab paths:
  - Local: `/home/yojiao/LSS/jianglab/3_Data_Working/fmriprep_processed_Iowa7T/DataOrg_PFM_36P/hipp_outputs`
  - Argon: `/Shared/jianglab/3_Data_Working/fmriprep_processed_Iowa7T/DataOrg_PFM_36P/hipp_outputs`
- Do not access archive content.
- Do not delete anything. The final obsolete legacy path list must be presented in a later message and explicitly approved before deletion.
- Do not pre-create empty directories. Directory creation is allowed only as the parent of a concrete file or symlink being copied, or for a run manifest/log file that is actually written.

## File Structure

- Modify `scripts/core_workflow/lib/output_layout.py`: add Iowa7T 36P-specific helpers for Jianglab root resolution, subject validation, branch validation, subject roots, shared HippUnfold root, branch roots, and runtime roots while keeping existing HCP/Stanford/HCP500 APIs intact.
- Create `scripts/iowa7t/migrate_hipp_outputs_layout.py`: build legacy-to-canonical file mappings, dry-run/copy files and symlinks, skip empty group-scoped source trees, write migration manifests only when a dry-run/apply command actually runs.
- Create `tests/core_workflow/test_iowa7t_36p_layout.py`: unit tests for canonical path rules and no `variants/` wrapper.
- Create `tests/iowa7t/test_migrate_hipp_outputs_layout.py`: tempdir tests for mapping, dry-run/copy, collision detection, and no empty `group/` creation.
- Modify `docs/superpowers/specs/2026-06-25-iowa7t-subject-first-output-layout-design.md`: keep the no-empty-directory rule explicit.
- Create/update `docs/iowa7t_36p_hipp_outputs_layout.md`: operational runbook for future agents and colleagues.

---

### Task 1: Confirm execution workspace

**Files:** none

- [ ] **Step 1: Ask for branch/worktree decision before code changes**

Ask the user exactly:

```text
当前仓库在 main 分支。请确认：允许我直接在 main 上实施这些代码/迁移辅助脚本改动，还是要我先创建独立 worktree？
```

Expected: user explicitly chooses main or worktree.

- [ ] **Step 2: If user chooses worktree, create/check it before implementation**

Use the `using-git-worktrees` skill before creating an isolated workspace. Do not create any directory until that skill has been read and its guardrails followed.

Expected: implementation happens outside the dirty main checkout, or user explicitly approved main.

---

### Task 2: Add Iowa7T 36P layout tests

**Files:**
- Create: `tests/core_workflow/test_iowa7t_36p_layout.py`

- [ ] **Step 1: Write failing tests**

Create tests covering:

```python
from pathlib import Path

import pytest

from scripts.core_workflow.lib import output_layout as layout


def test_iowa7t_36p_root_uses_local_jianglab_prefix(monkeypatch):
    monkeypatch.setattr(layout.Path, "exists", lambda self: str(self) == "/home/yojiao/LSS/jianglab")
    assert layout.resolve_iowa7t_36p_root() == Path("/home/yojiao/LSS/jianglab/3_Data_Working/fmriprep_processed_Iowa7T/DataOrg_PFM_36P")


def test_iowa7t_36p_root_uses_argon_prefix_when_local_missing(monkeypatch):
    monkeypatch.setattr(layout.Path, "exists", lambda self: str(self) == "/Shared/jianglab")
    assert layout.resolve_iowa7t_36p_root() == Path("/Shared/jianglab/3_Data_Working/fmriprep_processed_Iowa7T/DataOrg_PFM_36P")


def test_iowa7t_subject_layout_has_shared_hippunfold_and_direct_branches():
    subject = layout.iowa7t_subject_layout("06112026", hipp_outputs_root=Path("/tmp/hipp_outputs"))
    assert subject.subject_root == Path("/tmp/hipp_outputs/subjects/sub-06112026")
    assert subject.hippunfold_root == Path("/tmp/hipp_outputs/subjects/sub-06112026/hippunfold")
    assert subject.branch_root("multiband") == Path("/tmp/hipp_outputs/subjects/sub-06112026/multiband")
    assert subject.branch_root("multi_echo") == Path("/tmp/hipp_outputs/subjects/sub-06112026/multi_echo")
    assert "variants" not in subject.branch_root("multi_echo").parts


def test_iowa7t_branch_layout_paths_are_branch_scoped():
    branch = layout.iowa7t_branch_layout("06112026", "multi_echo", hipp_outputs_root=Path("/tmp/hipp_outputs"))
    assert branch.root == Path("/tmp/hipp_outputs/subjects/sub-06112026/multi_echo")
    assert branch.cortex == branch.root / "cortex"
    assert branch.surface == branch.root / "surface"
    assert branch.fc == branch.root / "fc"
    assert branch.atlas == branch.root / "atlas"
    assert branch.figures == branch.root / "figures"
    assert branch.provenance == branch.root / "provenance.json"


def test_iowa7t_layout_rejects_bad_subject_and_branch():
    with pytest.raises(ValueError, match="Invalid Iowa7T subject"):
        layout.iowa7t_subject_layout("sub-abc", hipp_outputs_root=Path("/tmp/hipp_outputs"))
    with pytest.raises(ValueError, match="Invalid Iowa7T branch"):
        layout.iowa7t_branch_layout("06112026", "variants", hipp_outputs_root=Path("/tmp/hipp_outputs"))


def test_iowa7t_layout_never_resolves_to_repo_outputs_iowa():
    root = layout.resolve_iowa7t_hipp_outputs_root(jianglab_root=Path("/tmp/jianglab"))
    assert root == Path("/tmp/jianglab/3_Data_Working/fmriprep_processed_Iowa7T/DataOrg_PFM_36P/hipp_outputs")
    assert "outputs_iowa" not in root.parts
```

- [ ] **Step 2: Run tests and verify they fail because APIs are missing**

Run:

```bash
/home/yojiao/.conda/envs/py314/bin/python -m pytest tests/core_workflow/test_iowa7t_36p_layout.py -q
```

Expected: FAIL with `AttributeError` for missing Iowa7T layout helpers.

---

### Task 3: Implement Iowa7T 36P canonical path helpers

**Files:**
- Modify: `scripts/core_workflow/lib/output_layout.py`

- [ ] **Step 1: Add dataclasses and constants**

Add to `output_layout.py`:

```python
from dataclasses import dataclass

IOWA7T_BRANCHES = ("multiband", "multi_echo")
LOCAL_JIANGLAB_ROOT = Path("/home/yojiao/LSS/jianglab")
ARGON_JIANGLAB_ROOT = Path("/Shared/jianglab")
IOWA7T_36P_RELATIVE_ROOT = Path("3_Data_Working/fmriprep_processed_Iowa7T/DataOrg_PFM_36P")


@dataclass(frozen=True)
class Iowa7TBranchLayout:
    subject: str
    branch: str
    root: Path
    cortex: Path
    surface: Path
    fc: Path
    atlas: Path
    qc: Path
    figures: Path
    provenance: Path


@dataclass(frozen=True)
class Iowa7TSubjectLayout:
    subject: str
    subject_root: Path
    hippunfold_root: Path

    def branch_root(self, branch: str) -> Path:
        return self.subject_root / validate_iowa7t_branch(branch)
```

- [ ] **Step 2: Add resolver functions**

Add before the existing dataset-scoped root functions:

```python
def validate_iowa7t_subject(subject: str) -> str:
    raw = str(subject).strip()
    subject_id = raw.removeprefix("sub-")
    if not (len(subject_id) == 8 and subject_id.isdigit()):
        raise ValueError(f"Invalid Iowa7T subject: {subject!r}; expected MMDDYYYY digits with optional sub- prefix")
    return subject_id


def validate_iowa7t_branch(branch: str) -> str:
    key = str(branch).strip().lower().replace("-", "_")
    if key not in IOWA7T_BRANCHES:
        raise ValueError(f"Invalid Iowa7T branch: {branch!r}; expected one of {IOWA7T_BRANCHES}")
    return key


def resolve_jianglab_root(*, require_exists: bool = True) -> Path:
    if LOCAL_JIANGLAB_ROOT.exists():
        return LOCAL_JIANGLAB_ROOT
    if ARGON_JIANGLAB_ROOT.exists():
        return ARGON_JIANGLAB_ROOT
    if require_exists:
        raise FileNotFoundError("Jianglab root not found. Local uses /home/yojiao/LSS/jianglab; Argon uses /Shared/jianglab.")
    return LOCAL_JIANGLAB_ROOT


def resolve_iowa7t_36p_root(*, jianglab_root: Path | None = None, require_exists: bool = True) -> Path:
    root = Path(jianglab_root) if jianglab_root is not None else resolve_jianglab_root(require_exists=require_exists)
    return root / IOWA7T_36P_RELATIVE_ROOT


def resolve_iowa7t_hipp_outputs_root(*, jianglab_root: Path | None = None, require_exists: bool = True) -> Path:
    return resolve_iowa7t_36p_root(jianglab_root=jianglab_root, require_exists=require_exists) / "hipp_outputs"


def iowa7t_subject_layout(subject: str, *, hipp_outputs_root: Path | None = None) -> Iowa7TSubjectLayout:
    subject_id = validate_iowa7t_subject(subject)
    root = Path(hipp_outputs_root) if hipp_outputs_root is not None else resolve_iowa7t_hipp_outputs_root()
    subject_root = root / "subjects" / f"sub-{subject_id}"
    return Iowa7TSubjectLayout(subject=subject_id, subject_root=subject_root, hippunfold_root=subject_root / "hippunfold")


def iowa7t_branch_layout(subject: str, branch: str, *, hipp_outputs_root: Path | None = None) -> Iowa7TBranchLayout:
    subject_layout = iowa7t_subject_layout(subject, hipp_outputs_root=hipp_outputs_root)
    branch_key = validate_iowa7t_branch(branch)
    root = subject_layout.branch_root(branch_key)
    return Iowa7TBranchLayout(
        subject=subject_layout.subject,
        branch=branch_key,
        root=root,
        cortex=root / "cortex",
        surface=root / "surface",
        fc=root / "fc",
        atlas=root / "atlas",
        qc=root / "qc",
        figures=root / "figures",
        provenance=root / "provenance.json",
    )
```

- [ ] **Step 3: Verify tests**

Run:

```bash
/home/yojiao/.conda/envs/py314/bin/python -m pytest tests/core_workflow/test_iowa7t_36p_layout.py tests/core_workflow/test_path_templates.py -q
```

Expected: PASS.

---

### Task 4: Add migration planner tests

**Files:**
- Create: `tests/iowa7t/test_migrate_hipp_outputs_layout.py`

- [ ] **Step 1: Write failing migration tests**

Create tests covering these exact behaviors:

```python
from pathlib import Path

import pytest

from scripts.iowa7t import migrate_hipp_outputs_layout as mig


def write_file(path: Path, text: str = "x") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def test_build_mapping_places_04172026_multiband_outputs_without_variants(tmp_path):
    legacy = tmp_path / "hipp_outputs"
    write_file(legacy / "inputs/hippunfold/dense_corobl_batch/sub-04172026/hippunfold/sub-04172026/surf/a.func.gii")
    write_file(legacy / "inputs/kong2022/shared_inputs/surface/sub-04172026/surface/raw/b.func.gii")
    write_file(legacy / "inputs/kong2022/shared_inputs/routea/sub-04172026/fc/c.npy")
    write_file(legacy / "work/individual_atlas/sub-04172026/gradient/d.json")
    write_file(legacy / "work/individual_atlas/sub-04172026/wta/e.json")
    write_file(legacy / "present/overviews/individual/gradient/f.png")
    mappings = mig.build_mappings(legacy, subjects=("04172026",))
    destinations = {m.destination.relative_to(legacy) for m in mappings}
    assert Path("subjects/sub-04172026/hippunfold/sub-04172026/surf/a.func.gii") in destinations
    assert Path("subjects/sub-04172026/multiband/surface/raw/b.func.gii") in destinations
    assert Path("subjects/sub-04172026/multiband/fc/c.npy") in destinations
    assert Path("subjects/sub-04172026/multiband/atlas/gradient/d.json") in destinations
    assert Path("subjects/sub-04172026/multiband/atlas/wta/e.json") in destinations
    assert Path("subjects/sub-04172026/multiband/figures/gradient/f.png") in destinations
    assert all("variants" not in path.parts for path in destinations)


def test_empty_group_sources_are_not_mapped_and_do_not_create_group(tmp_path):
    legacy = tmp_path / "hipp_outputs"
    (legacy / "inputs/kong2022/templates/hipp_group_average_t1only_151").mkdir(parents=True)
    (legacy / "inputs/kong2022/qc").mkdir(parents=True)
    (legacy / "inputs/kong2022/manifests").mkdir(parents=True)
    mappings = mig.build_mappings(legacy, subjects=("04172026",))
    mig.apply_mappings(mappings, mode="copy", write_manifest=False)
    assert mappings == []
    assert not (legacy / "group").exists()


def test_nonempty_group_sources_are_mapped(tmp_path):
    legacy = tmp_path / "hipp_outputs"
    write_file(legacy / "inputs/kong2022/manifests/status.tsv")
    mappings = mig.build_mappings(legacy, subjects=("04172026",))
    assert [m.destination.relative_to(legacy) for m in mappings] == [Path("group/manifests/status.tsv")]


def test_collision_detection_refuses_different_destination_content(tmp_path):
    legacy = tmp_path / "hipp_outputs"
    write_file(legacy / "inputs/kong2022/shared_inputs/routea/sub-04172026/fc/a.txt", "old")
    write_file(legacy / "subjects/sub-04172026/multiband/fc/a.txt", "new")
    mappings = mig.build_mappings(legacy, subjects=("04172026",))
    with pytest.raises(FileExistsError, match="Destination exists with different content"):
        mig.validate_mappings(mappings)
```

- [ ] **Step 2: Run tests and verify they fail because module is missing**

Run:

```bash
/home/yojiao/.conda/envs/py314/bin/python -m pytest tests/iowa7t/test_migrate_hipp_outputs_layout.py -q
```

Expected: FAIL with `ModuleNotFoundError` or missing functions.

---

### Task 5: Implement manifest-driven migration script

**Files:**
- Create: `scripts/iowa7t/__init__.py`
- Create: `scripts/iowa7t/migrate_hipp_outputs_layout.py`

- [ ] **Step 1: Create package marker**

Create `scripts/iowa7t/__init__.py`:

```python
"""Iowa7T 36P workflow helpers."""
```

- [ ] **Step 2: Implement migration module**

Create a module with:

```python
from __future__ import annotations

import argparse
import filecmp
import json
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

from scripts.core_workflow.lib.output_layout import resolve_iowa7t_hipp_outputs_root, validate_iowa7t_subject
from scripts.core_workflow.lib.workflow_utils import utc_timestamp_for_filename


@dataclass(frozen=True)
class Mapping:
    source: Path
    destination: Path
    kind: str

    def to_json(self) -> dict[str, str]:
        return {key: str(value) for key, value in asdict(self).items()}


def _iter_payload_files(root: Path) -> list[Path]:
    if not root.exists():
        return []
    if root.is_file() or root.is_symlink():
        return [root]
    return sorted(path for path in root.rglob("*") if path.is_file() or path.is_symlink())


def _map_tree(source_root: Path, destination_root: Path) -> list[Mapping]:
    return [Mapping(path, destination_root / path.relative_to(source_root), "symlink" if path.is_symlink() else "file") for path in _iter_payload_files(source_root)]


def build_mappings(hipp_outputs_root: Path, subjects: Iterable[str]) -> list[Mapping]:
    root = Path(hipp_outputs_root)
    mappings: list[Mapping] = []
    for raw_subject in subjects:
        subject = validate_iowa7t_subject(raw_subject)
        sub = f"sub-{subject}"
        mappings.extend(_map_tree(root / "inputs/hippunfold/dense_corobl_batch" / sub / "hippunfold", root / "subjects" / sub / "hippunfold"))
        mappings.extend(_map_tree(root / "inputs/kong2022/shared_inputs/surface" / sub / "surface", root / "subjects" / sub / "multiband" / "surface"))
        mappings.extend(_map_tree(root / "inputs/kong2022/shared_inputs/routea" / sub / "reference", root / "subjects" / sub / "multiband" / "cortex"))
        mappings.extend(_map_tree(root / "inputs/kong2022/shared_inputs/routea" / sub / "fc", root / "subjects" / sub / "multiband" / "fc"))
        mappings.extend(_map_tree(root / "work/individual_atlas" / sub / "gradient", root / "subjects" / sub / "multiband" / "atlas" / "gradient"))
        mappings.extend(_map_tree(root / "work/individual_atlas" / sub / "wta", root / "subjects" / sub / "multiband" / "atlas" / "wta"))
        mappings.extend(_map_tree(root / "present/overviews/individual/gradient", root / "subjects" / sub / "multiband" / "figures" / "gradient"))
        mappings.extend(_map_tree(root / "present/overviews/individual/wta", root / "subjects" / sub / "multiband" / "figures" / "wta"))
    mappings.extend(_map_tree(root / "work/individual_atlas/_overview_core_adapter", root / "runs/cache/overview_core_adapter"))
    mappings.extend(_map_tree(root / "work/individual_atlas/_runs", root / "runs"))
    mappings.extend(_map_tree(root / "inputs/kong2022/templates/hipp_group_average_t1only_151", root / "group/hippocampal_template"))
    mappings.extend(_map_tree(root / "inputs/kong2022/qc", root / "group/qc"))
    mappings.extend(_map_tree(root / "inputs/kong2022/manifests", root / "group/manifests"))
    return sorted(mappings, key=lambda mapping: str(mapping.destination))


def _same_payload(source: Path, destination: Path) -> bool:
    if source.is_symlink() or destination.is_symlink():
        return source.resolve(strict=True) == destination.resolve(strict=True)
    return source.stat().st_size == destination.stat().st_size and filecmp.cmp(source, destination, shallow=False)


def validate_mappings(mappings: list[Mapping]) -> None:
    seen: dict[Path, Path] = {}
    for mapping in mappings:
        if mapping.destination in seen and seen[mapping.destination] != mapping.source:
            raise FileExistsError(f"Multiple sources map to one destination: {seen[mapping.destination]} and {mapping.source} -> {mapping.destination}")
        seen[mapping.destination] = mapping.source
        if mapping.destination.exists() and not _same_payload(mapping.source, mapping.destination):
            raise FileExistsError(f"Destination exists with different content: {mapping.destination}")


def apply_mappings(mappings: list[Mapping], *, mode: str, write_manifest: bool = True, manifest_root: Path | None = None) -> Path | None:
    if mode not in {"dry-run", "copy"}:
        raise ValueError(f"Unsupported mode: {mode}")
    validate_mappings(mappings)
    manifest_path = None
    if write_manifest and manifest_root is not None:
        manifest_root.mkdir(parents=True, exist_ok=True)
        manifest_path = manifest_root / f"iowa7t_layout_migration_{utc_timestamp_for_filename()}_{mode}.json"
        manifest_path.write_text(json.dumps([m.to_json() for m in mappings], indent=2) + "
", encoding="utf-8")
    if mode == "dry-run":
        return manifest_path
    for mapping in mappings:
        if mapping.destination.exists():
            continue
        mapping.destination.parent.mkdir(parents=True, exist_ok=True)
        if mapping.source.is_symlink():
            mapping.destination.symlink_to(mapping.source.resolve(strict=True))
        else:
            shutil.copy2(mapping.source, mapping.destination)
    return manifest_path
```

Also add CLI `parse_args()` and `main()` that default to `resolve_iowa7t_hipp_outputs_root()`, accept `--subjects` and `--mode dry-run|copy`, write manifests under `hipp_outputs/runs/manifests/`, and print JSON with `mode`, `mapping_count`, and `manifest`.

- [ ] **Step 3: Run migration tests**

Run:

```bash
/home/yojiao/.conda/envs/py314/bin/python -m pytest tests/iowa7t/test_migrate_hipp_outputs_layout.py -q
```

Expected: PASS.

---

### Task 6: Add operational runbook and update design doc

**Files:**
- Modify: `docs/superpowers/specs/2026-06-25-iowa7t-subject-first-output-layout-design.md`
- Create: `docs/iowa7t_36p_hipp_outputs_layout.md`

- [ ] **Step 1: Ensure spec says no empty future directories**

Confirm the spec contains:

```text
Directories are created only when they receive real content.
no empty `group/` subtree is created for future intent alone.
```

- [ ] **Step 2: Create runbook**

Create `docs/iowa7t_36p_hipp_outputs_layout.md`:

```markdown
# Iowa7T 36P Hippocampal Output Layout

Canonical production root:

- Local: `/home/yojiao/LSS/jianglab/3_Data_Working/fmriprep_processed_Iowa7T/DataOrg_PFM_36P/hipp_outputs`
- Argon: `/Shared/jianglab/3_Data_Working/fmriprep_processed_Iowa7T/DataOrg_PFM_36P/hipp_outputs`

Rules:

- Iowa7T means the Jianglab 36P fMRIPrep tree, not HCPpipeline500.
- `outputs_iowa/` in the repository is not a production output root.
- Subject structural HippUnfold output is shared at `subjects/sub-<ID>/hippunfold/`.
- Functional branches are direct siblings: `multiband/` and `multi_echo/`.
- There is no `variants/` wrapper.
- Do not pre-create empty future directories such as `group/`.
- Create directories only as parents for concrete files/symlinks/manifests being written.
- Do not delete legacy paths until the exact deletion list is approved in the current session.
```

- [ ] **Step 3: Commit documentation and code together after tests pass**

Run:

```bash
git add scripts/core_workflow/lib/output_layout.py scripts/iowa7t tests/core_workflow/test_iowa7t_36p_layout.py tests/iowa7t/test_migrate_hipp_outputs_layout.py docs/superpowers/specs/2026-06-25-iowa7t-subject-first-output-layout-design.md docs/iowa7t_36p_hipp_outputs_layout.md
git commit -m "feat: add Iowa7T 36P subject-first output layout"
```

Expected: one focused commit; unrelated dirty files remain untouched.

---

### Task 7: Dry-run migration on Jianglab output tree

**Files/Data:**
- Read/write manifest only under `/home/yojiao/LSS/jianglab/3_Data_Working/fmriprep_processed_Iowa7T/DataOrg_PFM_36P/hipp_outputs/runs/manifests/`

- [ ] **Step 1: Run dry-run migration**

Run:

```bash
/home/yojiao/.conda/envs/py314/bin/python scripts/iowa7t/migrate_hipp_outputs_layout.py --subjects 04172026 --mode dry-run
```

Expected: JSON with nonzero `mapping_count` and a manifest path under `hipp_outputs/runs/manifests/`.

- [ ] **Step 2: Inspect resulting top-level directories without deep traversal**

Run:

```bash
timeout 60 find /home/yojiao/LSS/jianglab/3_Data_Working/fmriprep_processed_Iowa7T/DataOrg_PFM_36P/hipp_outputs -maxdepth 1 2>/dev/null | sort
```

Expected: dry-run creates only a real manifest under `runs/manifests/`; it does not create empty `group/`.

---

### Task 8: Apply copy migration and verify

**Files/Data:**
- Writes canonical copies under `/home/yojiao/LSS/jianglab/.../hipp_outputs/subjects/`, plus manifests under `runs/manifests/`.
- Does not delete legacy paths.

- [ ] **Step 1: Run copy migration**

Run only after dry-run looks correct:

```bash
/home/yojiao/.conda/envs/py314/bin/python scripts/iowa7t/migrate_hipp_outputs_layout.py --subjects 04172026 --mode copy
```

Expected: JSON with same `mapping_count` as dry-run.

- [ ] **Step 2: Verify canonical subject paths exist**

Run:

```bash
timeout 60 find /home/yojiao/LSS/jianglab/3_Data_Working/fmriprep_processed_Iowa7T/DataOrg_PFM_36P/hipp_outputs/subjects/sub-04172026 -maxdepth 2 2>/dev/null | sort
```

Expected includes shared `hippunfold/` and branch `multiband/` with `atlas/`, `fc/`, and `surface/`.

- [ ] **Step 3: Verify no `variants/` directory was created**

Run:

```bash
timeout 60 find /home/yojiao/LSS/jianglab/3_Data_Working/fmriprep_processed_Iowa7T/DataOrg_PFM_36P/hipp_outputs/subjects -maxdepth 4 -type d -name variants 2>/dev/null
```

Expected: no output.

- [ ] **Step 4: Verify no empty `group/` was created by policy violation**

Run:

```bash
timeout 60 find /home/yojiao/LSS/jianglab/3_Data_Working/fmriprep_processed_Iowa7T/DataOrg_PFM_36P/hipp_outputs/group -maxdepth 1 2>/dev/null | head -20
```

Expected: either no `group/` directory, or only real migrated payload-bearing directories.

---

### Task 9: Prepare `06112026` canonical branch targets without empty pre-creation

**Files/Data:**
- No directory is created in this task unless a real processing command writes output.

- [ ] **Step 1: Resolve branch targets programmatically**

Run:

```bash
/home/yojiao/.conda/envs/py314/bin/python -c "from scripts.core_workflow.lib.output_layout import iowa7t_subject_layout, iowa7t_branch_layout; s=iowa7t_subject_layout('06112026'); print(s.hippunfold_root); print(iowa7t_branch_layout('06112026','multiband').root); print(iowa7t_branch_layout('06112026','multi_echo').root)"
```

Expected:

```text
/home/yojiao/LSS/jianglab/3_Data_Working/fmriprep_processed_Iowa7T/DataOrg_PFM_36P/hipp_outputs/subjects/sub-06112026/hippunfold
/home/yojiao/LSS/jianglab/3_Data_Working/fmriprep_processed_Iowa7T/DataOrg_PFM_36P/hipp_outputs/subjects/sub-06112026/multiband
/home/yojiao/LSS/jianglab/3_Data_Working/fmriprep_processed_Iowa7T/DataOrg_PFM_36P/hipp_outputs/subjects/sub-06112026/multi_echo
```

- [ ] **Step 2: Use these paths when launching HippUnfold and branch-specific processing**

Do not create placeholder directories. The processing launcher should create `hippunfold/`, `multiband/`, or `multi_echo/` only when writing actual outputs.

Expected: future processing for `06112026` has one shared HippUnfold output and two direct branch outputs.

---

### Task 10: Produce deletion-candidate report only

**Files/Data:**
- Read-only inventory of legacy wrappers.
- No deletion.

- [ ] **Step 1: Generate exact obsolete path candidates after canonical verification**

Candidate list should include only paths that are fully represented in canonical layout and are not needed by active code:

```text
/home/yojiao/LSS/jianglab/3_Data_Working/fmriprep_processed_Iowa7T/DataOrg_PFM_36P/hipp_outputs/inputs/qc
/home/yojiao/LSS/jianglab/3_Data_Working/fmriprep_processed_Iowa7T/DataOrg_PFM_36P/hipp_outputs/inputs/hippunfold
/home/yojiao/LSS/jianglab/3_Data_Working/fmriprep_processed_Iowa7T/DataOrg_PFM_36P/hipp_outputs/inputs/kong2022
/home/yojiao/LSS/jianglab/3_Data_Working/fmriprep_processed_Iowa7T/DataOrg_PFM_36P/hipp_outputs/work/individual_atlas
/home/yojiao/LSS/jianglab/3_Data_Working/fmriprep_processed_Iowa7T/DataOrg_PFM_36P/hipp_outputs/present/overviews/individual
```

- [ ] **Step 2: Present exact deletion list to user**

Ask for explicit approval of the exact paths. Do not run any `rm`, `rmdir`, or deletion script in this plan.

Expected: deletion remains a separate, explicitly approved operation.

---

## Self-Review

- Spec coverage: subject-first layout, direct `multiband/`/`multi_echo/`, shared HippUnfold, no `variants/`, no empty `group/`, Jianglab local/Argon path mapping, dry-run/copy/verify/delete-later flow are covered.
- Placeholder scan: no `TBD`, `TODO`, or unspecified implementation step remains.
- Type consistency: tests use `Iowa7TSubjectLayout`, `Iowa7TBranchLayout`, `build_mappings`, `validate_mappings`, and `apply_mappings`, all defined in implementation tasks.
