#!/usr/bin/env python3
from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
CORE_ROOT = REPO_ROOT / "scripts" / "core_workflow"
STEP_REGISTRY = CORE_ROOT / "step_registry.json"
ACTIVE_CANONICAL_DOCS = [
    REPO_ROOT / "docs" / "README.md",
    REPO_ROOT / "docs" / "hipp_parcellation_network" / "core_workflow_overview.md",
    REPO_ROOT / "docs" / "hipp_parcellation_network" / "core_workflow_step_registry.md",
    REPO_ROOT / "docs" / "hipp_parcellation_network" / "multi_branch_flow.md",
    REPO_ROOT / "docs" / "hipp_parcellation_network" / "step_by_step_flows.md",
    REPO_ROOT / "scripts" / "README.md",
]


def _load_registry() -> dict:
    return json.loads(STEP_REGISTRY.read_text(encoding="utf-8"))


def test_step_registry_references_existing_paths() -> None:
    payload = _load_registry()
    steps = payload["steps"]
    assert steps, "step registry must contain at least one step"
    for item in steps:
        entry = REPO_ROOT / item["entry_script"]
        target = REPO_ROOT / item["target_script"]
        assert entry.exists(), f"missing entry script: {entry}"
        assert target.exists(), f"missing target script: {target}"
    for doc in ACTIVE_CANONICAL_DOCS:
        assert doc.exists(), f"missing canonical doc: {doc}"


def test_every_numbered_step_emits_traceability_manifest_fields() -> None:
    payload = _load_registry()
    for item in payload["steps"]:
        entry = REPO_ROOT / item["entry_script"]
        step_id = str(item["step_id"])
        with tempfile.TemporaryDirectory(prefix="core_workflow_manifest_") as td:
            out_json = Path(td) / f"{step_id}.json"
            cmd = [
                sys.executable,
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


def test_canonical_docs_and_registry_do_not_depend_on_archived_t2_launcher() -> None:
    blocked_tokens = [
        "run_hippunfold_local_t2w_t1reg_isolated_cache.sh",
        "_archive/deprecated_2026-05-11/scripts",
    ]
    for doc in ACTIVE_CANONICAL_DOCS:
        txt = doc.read_text(encoding="utf-8")
        for token in blocked_tokens:
            assert token not in txt, f"blocked archived token in canonical doc {doc}: {token}"
    registry_txt = STEP_REGISTRY.read_text(encoding="utf-8")
    for token in blocked_tokens:
        assert token not in registry_txt, f"blocked archived token in step registry: {token}"


def main() -> int:
    test_step_registry_references_existing_paths()
    test_every_numbered_step_emits_traceability_manifest_fields()
    test_canonical_docs_and_registry_do_not_depend_on_archived_t2_launcher()
    print("ok: core workflow traceability contract checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
