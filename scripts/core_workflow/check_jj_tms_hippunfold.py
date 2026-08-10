#!/home/yojiao/.conda/envs/py314/bin/python
"""CLI for strict JJ_TMS HippUnfold output acceptance."""

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
        description="Require complete corobl den-512 HippUnfold outputs for all fixed JJ_TMS subjects."
    )
    parser.add_argument("--batch-root", type=Path, default=workflow.DEFAULT_BATCH_ROOT)
    parser.add_argument(
        "--subjects-file", type=Path, default=workflow.DEFAULT_SUBJECTS_FILE
    )
    parser.add_argument("--json-out", type=Path)
    parser.add_argument("--downstream-root", type=Path)
    parser.add_argument(
        "--run-started-at", type=float,
        help="Unix timestamp; restrict logs/metadata to the current rerun.",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    try:
        subjects = workflow.read_production_subject_manifest(args.subjects_file)
        report = workflow.build_batch_report(
            args.batch_root,
            subjects,
            expected_subject_count=workflow.EXPECTED_SUBJECT_COUNT,
            run_started_at=args.run_started_at,
            downstream_root=args.downstream_root,
        )
    except (FileNotFoundError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    if args.json_out:
        workflow.write_report(report, args.json_out)
    print(json.dumps({
        "status": report["status"],
        "complete_count": report["complete_count"],
        "subject_count": report["subject_count"],
        "expected_subject_count": report["expected_subject_count"],
    }, indent=2, sort_keys=True))
    if report["status"] != "complete":
        for row in report["subjects"]:
            if row["status"] != "complete":
                print(
                    f"sub-{row['subject']}: "
                    f"{'; '.join(row['errors'])}; "
                    f"missing={len(row['missing_files'])}",
                    file=sys.stderr,
                )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
