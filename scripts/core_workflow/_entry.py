#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_BRANCH = "kong2022_routeb"
BRANCH_CHOICES = ("kong2019", "kong2022_routea", "kong2022_routeb")


@dataclass(frozen=True)
class StepSpec:
    step_id: str
    purpose: str
    target_relpath: str
    target_type: str = "python"
    upstream_defaults: tuple[str, ...] = ()


def _now_utc_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _get_git_commit() -> str:
    proc = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode != 0:
        return "unknown"
    return proc.stdout.strip()


def _build_command(spec: StepSpec, passthrough: Sequence[str]) -> list[str]:
    target = REPO_ROOT / spec.target_relpath
    if spec.target_type == "python":
        return [sys.executable, str(target), *passthrough]
    if spec.target_type == "shell":
        return ["bash", str(target), *passthrough]
    raise ValueError(f"Unsupported target_type: {spec.target_type}")


def _write_manifest(
    manifest_out: Path,
    *,
    spec: StepSpec,
    workflow_branch: str,
    upstream_step_ids: Sequence[str],
    input_artifacts: Sequence[str],
    output_artifacts: Sequence[str],
    command: Sequence[str],
    returncode: int | None,
    dry_run: bool,
) -> None:
    payload = {
        "step_id": spec.step_id,
        "purpose": spec.purpose,
        "workflow_branch": workflow_branch,
        "upstream_step_ids": list(upstream_step_ids),
        "input_artifacts": list(input_artifacts),
        "output_artifacts": list(output_artifacts),
        "command": " ".join(shlex.quote(x) for x in command),
        "git_commit": _get_git_commit(),
        "timestamp_utc": _now_utc_iso(),
        "target_script": spec.target_relpath,
        "status": "dry_run" if dry_run else ("ok" if returncode == 0 else "failed"),
        "returncode": returncode,
    }
    manifest_out.parent.mkdir(parents=True, exist_ok=True)
    manifest_out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def run_step(spec: StepSpec) -> int:
    parser = argparse.ArgumentParser(
        description=(
            f"{spec.step_id}: {spec.purpose}. "
            "Pass target-script arguments directly to this wrapper (unknown args are forwarded)."
        )
    )
    parser.add_argument(
        "--workflow-branch",
        choices=BRANCH_CHOICES,
        default=DEFAULT_BRANCH,
        help="Active branch label written to the step manifest.",
    )
    parser.add_argument(
        "--upstream-step-id",
        action="append",
        default=list(spec.upstream_defaults),
        help="Repeatable upstream step ID. Defaults are prefilled by this wrapper.",
    )
    parser.add_argument(
        "--input-artifact",
        action="append",
        default=[],
        help="Repeatable input artifact path for traceability manifest.",
    )
    parser.add_argument(
        "--output-artifact",
        action="append",
        default=[],
        help="Repeatable output artifact path for traceability manifest.",
    )
    parser.add_argument(
        "--step-manifest-out",
        type=Path,
        default=None,
        help="Optional output path for standardized step manifest JSON.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print and record command without executing target.",
    )
    args, passthrough = parser.parse_known_args()

    cmd = _build_command(spec, passthrough)
    if args.dry_run:
        print("DRY RUN:", " ".join(shlex.quote(x) for x in cmd))
        if args.step_manifest_out is not None:
            _write_manifest(
                args.step_manifest_out,
                spec=spec,
                workflow_branch=args.workflow_branch,
                upstream_step_ids=args.upstream_step_id,
                input_artifacts=args.input_artifact,
                output_artifacts=args.output_artifact,
                command=cmd,
                returncode=0,
                dry_run=True,
            )
        return 0

    proc = subprocess.run(cmd, cwd=REPO_ROOT, check=False)
    if args.step_manifest_out is not None:
        _write_manifest(
            args.step_manifest_out,
            spec=spec,
            workflow_branch=args.workflow_branch,
            upstream_step_ids=args.upstream_step_id,
            input_artifacts=args.input_artifact,
            output_artifacts=args.output_artifact,
            command=cmd,
            returncode=proc.returncode,
            dry_run=False,
        )
    return proc.returncode
