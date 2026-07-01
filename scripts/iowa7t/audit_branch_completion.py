#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Iterable

REPO_ROOT = Path(__file__).resolve().parents[2]
CORE_WORKFLOW_DIR = REPO_ROOT / "scripts" / "core_workflow"
for import_root in (REPO_ROOT, CORE_WORKFLOW_DIR):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from lib.output_layout import IOWA7T_BRANCHES, IOWA7T_VARIANTS, resolve_iowa7t_hipp_outputs_root
from scripts.iowa7t.run_branch_downstream import normalize_subject


def parse_job(value: str) -> tuple[str, str]:
    if ":" not in value:
        raise argparse.ArgumentTypeError("Expected --job value as <subject>:<multiband|multi_echo>")
    subject, branch = value.split(":", 1)
    subject_id = normalize_subject(subject)
    branch_key = branch.strip().lower().replace("-", "_")
    if branch_key not in IOWA7T_BRANCHES:
        raise argparse.ArgumentTypeError(f"Unsupported branch {branch!r}; expected one of {IOWA7T_BRANCHES}")
    return subject_id, branch_key


def _overview_pngs(root: Path, subject: str, branch: str) -> list[Path]:
    branch_root = Path(root) / "subjects" / f"sub-{subject}" / branch / "figures"
    files: list[Path] = []
    for figure_dir in (branch_root / "gradient", branch_root / "wta"):
        if figure_dir.exists():
            files.extend(path for path in figure_dir.glob("*overview.png") if path.is_file())
    return sorted(files, key=lambda path: path.as_posix())


def audit_jobs(
    *,
    jobs: Iterable[tuple[str, str]],
    output_root: Path,
    compare_output_root: Path,
) -> dict[str, object]:
    output_root = Path(output_root)
    compare_output_root = Path(compare_output_root)
    rows: list[dict[str, object]] = []
    ok = True
    for raw_subject, raw_branch in jobs:
        subject = normalize_subject(raw_subject)
        branch = raw_branch.strip().lower().replace("-", "_")
        expected = _overview_pngs(compare_output_root, subject, branch)
        missing: list[str] = []
        present: list[str] = []
        for source_path in expected:
            rel = source_path.relative_to(compare_output_root)
            target_path = output_root / rel
            if target_path.exists() and target_path.is_file():
                present.append(str(target_path))
            else:
                missing.append(str(target_path))
        if missing:
            ok = False
        rows.append(
            {
                "subject": f"sub-{subject}",
                "branch": branch,
                "ok": not missing,
                "expected_overviews": [str(path) for path in expected],
                "present_overviews": present,
                "missing_overviews": missing,
            }
        )
    return {
        "ok": ok,
        "output_root": str(output_root),
        "compare_output_root": str(compare_output_root),
        "jobs": rows,
    }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Audit Iowa7T branch overview completion by comparing relative PNG inventory against another variant."
    )
    parser.add_argument("--variant", choices=IOWA7T_VARIANTS, default="noxcpd")
    parser.add_argument("--compare-variant", choices=IOWA7T_VARIANTS, default="36p")
    parser.add_argument("--output-root", default=None)
    parser.add_argument("--compare-output-root", default=None)
    parser.add_argument("--job", action="append", type=parse_job, required=True)
    return parser


def main() -> int:
    args = build_arg_parser().parse_args()
    output_root = (
        Path(args.output_root)
        if args.output_root is not None
        else resolve_iowa7t_hipp_outputs_root(variant=str(args.variant), require_exists=False)
    )
    compare_output_root = (
        Path(args.compare_output_root)
        if args.compare_output_root is not None
        else resolve_iowa7t_hipp_outputs_root(variant=str(args.compare_variant), require_exists=False)
    )
    summary = audit_jobs(
        jobs=args.job,
        output_root=output_root,
        compare_output_root=compare_output_root,
    )
    print(json.dumps(summary, indent=2), flush=True)
    return 0 if summary["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
