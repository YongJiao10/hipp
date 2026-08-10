#!/home/yojiao/.conda/envs/py314/bin/python
"""Validate one JJ_TMS standard-entry HippUnfold DAG without executing jobs."""

from __future__ import annotations

import argparse
import re
from datetime import datetime, timezone
import json
import subprocess
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import jj_tms_hippunfold as workflow


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run a representative full-target HippUnfold dry-run and validate its DAG."
    )
    parser.add_argument("--subject", default="1001")
    parser.add_argument("--input-root", type=Path, default=workflow.DEFAULT_INPUT_ROOT)
    parser.add_argument(
        "--batch-root", type=Path,
        help="Fresh probe batch root; defaults to a unique retained directory under /tmp.",
    )
    args = parser.parse_args()

    subject = workflow.normalize_subject(args.subject)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    batch_root = args.batch_root or (
        Path("/tmp") / f"jj_tms_hippunfold_dag_preflight_{subject}_{timestamp}"
    )
    probe_output = batch_root / f"sub-{subject}" / "hippunfold"
    if probe_output.exists():
        print(json.dumps({
            "status": "fail",
            "subject": subject,
            "probe_root": str(batch_root),
            "errors": ["probe output already exists; fresh DAG validation is mandatory"],
        }, indent=2, sort_keys=True))
        return 2
    probe_output.parent.mkdir(parents=True)
    command = workflow.build_subject_command(
        subject, input_root=args.input_root, batch_root=batch_root
    )
    environment = workflow.build_subject_environment(subject)
    environment["HIPPUNFOLD_EXTRA_ARGS"] = "--dry-run --printshellcmds"
    process = subprocess.run(
        command,
        cwd=str(workflow.REPO_ROOT),
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )
    transcript = f"{process.stdout}\n{process.stderr}"
    errors = workflow.validate_dag_output(transcript)
    if process.returncode != 0:
        errors.insert(0, f"standard entry returned {process.returncode}")
    rule_counts = {
        rule: len(re.findall(rf"(?:local)?rule\s+{re.escape(rule)}:", transcript))
        for rule in workflow.REQUIRED_DAG_RULE_COUNTS
    }
    result = {
        "status": "pass" if not errors else "fail",
        "subject": subject,
        "probe_root": str(batch_root),
        "probe_root_retained": True,
        "rule_counts": rule_counts,
        "command": command,
        "errors": errors,
        "returncode": process.returncode,
    }
    print(json.dumps(result, indent=2, sort_keys=True))
    if errors:
        print(transcript, file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
