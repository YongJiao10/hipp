from __future__ import annotations

import argparse
import shlex
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Sequence

from .branch_config import BRANCH_CHOICES, DEFAULT_BRANCH
from .workflow_utils import git_commit, utc_now_iso, write_json

REPO_ROOT = Path(__file__).resolve().parents[3]


@dataclass(frozen=True)
class StageSpec:
    step_id: str
    purpose: str
    stage_main: Callable[[], int | None]
    upstream_defaults: tuple[str, ...] = ()


def _normalize_passthrough(unknown: Sequence[str]) -> list[str]:
    if unknown and unknown[0] == "--":
        return list(unknown[1:])
    return list(unknown)


def _invoke_with_passthrough(stage_main: Callable[[], int | None], passthrough: Sequence[str]) -> int:
    old_argv = sys.argv
    sys.argv = [old_argv[0], *passthrough]
    try:
        rc = stage_main()
    finally:
        sys.argv = old_argv
    return int(rc) if rc is not None else 0


def _write_manifest(
    manifest_out: Path,
    *,
    spec: StageSpec,
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
        "git_commit": git_commit(short=False, repo_root=REPO_ROOT) or "unknown",
        "timestamp_utc": utc_now_iso(zulu=True),
        "status": "dry_run" if dry_run else ("ok" if returncode == 0 else "failed"),
        "returncode": returncode,
    }
    write_json(manifest_out, payload, indent=2, sort_keys=True, trailing_newline=True)


def run_stage(spec: StageSpec) -> int:
    parser = argparse.ArgumentParser(
        description=(
            f"{spec.step_id}: {spec.purpose}. "
            "Pass stage-specific arguments after -- (or as unknown flags) to forward them to stage implementation."
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
        help="Repeatable upstream step ID. Defaults are prefilled by this entrypoint.",
    )
    parser.add_argument(
        "--input-artifact",
        action="append",
        default=[],
        help="Repeatable explicit input artifact path/identifier.",
    )
    parser.add_argument(
        "--output-artifact",
        action="append",
        default=[],
        help="Repeatable explicit output artifact path/identifier.",
    )
    parser.add_argument(
        "--step-manifest-out",
        default=None,
        help="Optional JSON path to write standardized traceability manifest.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Do not execute stage implementation; only emit manifest metadata if requested.",
    )
    args, unknown = parser.parse_known_args()
    passthrough = _normalize_passthrough(unknown)

    command = [sys.executable, str(Path(sys.argv[0]).resolve()), *passthrough]
    returncode: int | None = None

    if not args.dry_run:
        returncode = _invoke_with_passthrough(spec.stage_main, passthrough)

    if args.step_manifest_out:
        _write_manifest(
            Path(args.step_manifest_out).resolve(),
            spec=spec,
            workflow_branch=args.workflow_branch,
            upstream_step_ids=args.upstream_step_id,
            input_artifacts=args.input_artifact,
            output_artifacts=args.output_artifact,
            command=command,
            returncode=returncode,
            dry_run=bool(args.dry_run),
        )

    if args.dry_run:
        return 0
    if returncode is None:
        return 0
    return int(returncode)
