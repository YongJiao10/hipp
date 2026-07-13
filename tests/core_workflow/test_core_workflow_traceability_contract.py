#!/usr/bin/env python3
from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
import tempfile
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
CORE_ROOT = REPO_ROOT / "scripts" / "core_workflow"
STEP_REGISTRY = CORE_ROOT / "step_registry.json"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
PREFERRED_PYTHON = Path("/home/yojiao/.conda/envs/py314/bin/python")
TEST_PYTHON = str(PREFERRED_PYTHON if PREFERRED_PYTHON.exists() else Path(sys.executable).resolve())
ACTIVE_CANONICAL_DOCS = [
    REPO_ROOT / "docs" / "README.md",
    REPO_ROOT / "docs" / "hipp_parcellation_network" / "core_workflow_overview.md",
    REPO_ROOT / "docs" / "hipp_parcellation_network" / "core_workflow_step_registry.md",
    REPO_ROOT / "docs" / "hipp_parcellation_network" / "multi_branch_flow.md",
    REPO_ROOT / "docs" / "hipp_parcellation_network" / "step_by_step_flows.md",
    REPO_ROOT / "scripts" / "README.md",
    REPO_ROOT / "scripts" / "core_workflow" / "README.md",
]
ACTIVE_CORE_PY = sorted((CORE_ROOT.glob("*.py"))) + sorted((CORE_ROOT / "lib").glob("*.py"))


def _load_registry() -> dict:
    return json.loads(STEP_REGISTRY.read_text(encoding="utf-8"))


def _load_audit_module():
    path = CORE_ROOT / "audit_formal_gradient_spectral_release.py"
    spec = importlib.util.spec_from_file_location("audit_formal_gradient_spectral_release", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_step_registry_references_existing_paths_and_modules() -> None:
    payload = _load_registry()
    steps = payload["steps"]
    assert steps, "step registry must contain at least one step"
    for item in steps:
        entry = REPO_ROOT / item["entry_script"]
        assert entry.exists(), f"missing entry script: {entry}"
        impl_path = REPO_ROOT / str(item["implementation_script"])
        assert impl_path.exists(), f"missing implementation module file: {impl_path}"
    for doc in ACTIVE_CANONICAL_DOCS:
        assert doc.exists(), f"missing canonical doc: {doc}"


def test_release_utilities_register_exact_audit_formal_groups() -> None:
    registry = _load_registry()
    audit = _load_audit_module()
    expected = [
        {"dataset": dataset, "group_root": group_root, "atlas": atlas}
        for dataset, group_root, atlas in audit.FORMAL_GROUPS
    ]
    utilities = {item["step_id"]: item for item in registry["utilities"]}
    for utility_id in (
        "audit_formal_gradient_spectral_release",
        "run_formal_gradient_spectral_refresh",
    ):
        assert utilities[utility_id]["formal_groups"] == expected


def test_every_numbered_step_help_succeeds() -> None:
    payload = _load_registry()
    for item in payload["steps"]:
        entry = REPO_ROOT / item["entry_script"]
        proc = subprocess.run(
            [TEST_PYTHON, str(entry), "--help"],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
        )
        assert proc.returncode == 0, f"--help failed for {entry}\nstdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"


def test_every_numbered_step_emits_traceability_manifest_fields() -> None:
    payload = _load_registry()
    for item in payload["steps"]:
        entry = REPO_ROOT / item["entry_script"]
        step_id = str(item["step_id"])
        help_proc = subprocess.run(
            [TEST_PYTHON, str(entry), "--help"],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
        )
        assert help_proc.returncode == 0, (
            f"--help failed for {entry}\nstdout:\n{help_proc.stdout}\nstderr:\n{help_proc.stderr}"
        )
        if "--step-manifest-out" not in help_proc.stdout:
            continue
        with tempfile.TemporaryDirectory(prefix="core_workflow_manifest_") as td:
            out_json = Path(td) / f"{step_id}.json"
            cmd = [
                TEST_PYTHON,
                str(entry),
                "--workflow-branch",
                "kong2022_routeb",
                "--upstream-step-id",
                "00_00_upstream_placeholder",
                "--input-artifact",
                "inputs/example_a",
                "--output-artifact",
                "outputs/example_b",
                "--step-manifest-out",
                str(out_json),
                "--dry-run",
            ]
            proc = subprocess.run(cmd, cwd=REPO_ROOT, capture_output=True, text=True)
            assert proc.returncode == 0, (
                f"dry-run failed for {entry}\nstdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"
            )
            manifest = json.loads(out_json.read_text(encoding="utf-8"))
            assert manifest["step_id"] == step_id
            assert manifest["workflow_branch"] == "kong2022_routeb"
            assert "00_00_upstream_placeholder" in manifest["upstream_step_ids"]
            assert "inputs/example_a" in manifest["input_artifacts"]
            assert "outputs/example_b" in manifest["output_artifacts"]
            assert isinstance(manifest.get("command"), str) and manifest["command"]
            assert isinstance(manifest.get("git_commit"), str) and manifest["git_commit"]


def test_active_core_workflow_has_no_legacy_wrapper_or_runtime_refs() -> None:
    blocked_tokens = [
        "target_relpath",
        "scripts/hipp_parcellation_network/",
        "scripts/cortex/",
        "scripts/common",
        "scripts/workbench",
        "scripts/wb_command",
        'REPO_ROOT / "scripts" / "wb_command"',
        'Path(__file__).resolve().parents[1] / "wb_command"',
        "run_hippunfold_local_t2w_t1reg_isolated_cache.sh",
    ]
    for path in ACTIVE_CORE_PY + [STEP_REGISTRY] + ACTIVE_CANONICAL_DOCS:
        txt = Path(path).read_text(encoding="utf-8")
        for token in blocked_tokens:
            assert token not in txt, f"blocked token in {path}: {token}"


def main() -> int:
    test_step_registry_references_existing_paths_and_modules()
    test_release_utilities_register_exact_audit_formal_groups()
    test_every_numbered_step_help_succeeds()
    test_every_numbered_step_emits_traceability_manifest_fields()
    test_active_core_workflow_has_no_legacy_wrapper_or_runtime_refs()
    print("ok: core workflow traceability contract checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
