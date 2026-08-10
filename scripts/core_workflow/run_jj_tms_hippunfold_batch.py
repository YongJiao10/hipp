#!/home/yojiao/.conda/envs/py314/bin/python
"""CLI for the fixed JJ_TMS full HippUnfold corobl rerun."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import jj_tms_hippunfold as workflow


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run and strictly validate the fixed 70-subject JJ_TMS corobl DAG."
    )
    parser.add_argument("--input-root", type=Path, default=workflow.DEFAULT_INPUT_ROOT)
    parser.add_argument("--batch-root", type=Path, default=workflow.DEFAULT_BATCH_ROOT)
    parser.add_argument(
        "--subjects-file", type=Path, default=workflow.DEFAULT_SUBJECTS_FILE
    )
    parser.add_argument(
        "--concurrency", type=int, default=workflow.CONCURRENCY,
        help="Fixed at 4 for this workflow.",
    )
    parser.add_argument(
        "--preflight-only", action="store_true",
        help="Validate the fixed roster, inputs, entry point, and model without running jobs.",
    )
    parser.add_argument(
        "--print-command", metavar="SUBJECT",
        help="Print the standard full-DAG command for one subject and exit.",
    )
    args = parser.parse_args()
    return args


def main() -> int:
    args = _parse_args()
    try:
        subjects = workflow.read_production_subject_manifest(args.subjects_file)
        if args.print_command:
            print(" ".join(workflow.build_subject_command(
                args.print_command,
                input_root=args.input_root,
                batch_root=args.batch_root,
            )))
            return 0
        preflight = workflow.preflight(args.input_root, subjects)
        print(json.dumps(preflight, indent=2, sort_keys=True))
        if args.preflight_only:
            return 0 if preflight["status"] == "pass" else 2
        if preflight["status"] != "pass":
            return 2
        status, report = workflow.run_batch(
            input_root=args.input_root,
            batch_root=args.batch_root,
            subjects_file=args.subjects_file,
            concurrency=args.concurrency,
        )
        print(json.dumps({
            "status": report.get("status"),
            "complete_count": report.get("complete_count"),
            "subject_count": report.get("subject_count"),
            "final_manifest": report.get("run_id"),
        }, indent=2, sort_keys=True))
        return status
    except (FileNotFoundError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
