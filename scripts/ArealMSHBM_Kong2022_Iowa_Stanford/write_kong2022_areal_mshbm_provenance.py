#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path


def read_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Write provenance for a completed Kong2022 ArealMSHBM subject branch."
    )
    parser.add_argument("--input-manifest", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--provenance", required=True, type=Path)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--branch", default="")
    parser.add_argument("--subject", required=True)
    parser.add_argument("--job-id", required=True)
    parser.add_argument("--task-id", required=True)
    parser.add_argument("--host", required=True)
    parser.add_argument("--matlab-script", required=True)
    parser.add_argument("--cbig-code-dir", required=True)
    parser.add_argument("--group-prior", required=True)
    parser.add_argument("--wb-command", required=True)
    parser.add_argument("--matlab-mode", required=True)
    parser.add_argument("--overwrite", required=True)
    parser.add_argument("--run-start-utc", required=True)
    parser.add_argument("--run-end-utc", required=True)
    parser.add_argument("--runtime-seconds", required=True, type=float)
    args = parser.parse_args()

    manifest = read_json(args.input_manifest)
    runs = manifest.get("runs", [])
    input_ciftis = [row["dtseries"] for row in runs]
    timepoints = [int(row["timepoints"]) for row in runs]

    payload = {
        "schema_version": 1,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset": args.dataset,
        "branch": args.branch or None,
        "subject": args.subject,
        "input_cifti": input_ciftis[0] if len(input_ciftis) == 1 else input_ciftis,
        "input_ciftis": input_ciftis,
        "output_dir": str(args.output_dir),
        "labels": {
            "lh": str(args.output_dir / "lh_labels.txt"),
            "rh": str(args.output_dir / "rh_labels.txt"),
        },
        "qsub": {
            "job_id": args.job_id,
            "task_id": args.task_id,
            "host": args.host,
        },
        "matlab": {
            "script": args.matlab_script,
            "mode": args.matlab_mode,
        },
        "parameters": {
            "target_mesh": "fs_LR_32k",
            "model": "gMSHBM",
            "num_ROIs": "400",
            "w": "30",
            "c": "30",
            "beta": "50",
            "group_prior": args.group_prior,
            "cbig_code_dir": args.cbig_code_dir,
            "wb_command": args.wb_command,
            "overwrite": args.overwrite,
        },
        "timepoints": timepoints[0] if len(timepoints) == 1 else timepoints,
        "runs": runs,
        "run_list": manifest.get("run_list"),
        "censor_list": manifest.get("censor_list"),
        "timing": {
            "run_start_utc": args.run_start_utc,
            "run_end_utc": args.run_end_utc,
            "runtime_seconds": args.runtime_seconds,
        },
    }

    args.provenance.parent.mkdir(parents=True, exist_ok=True)
    args.provenance.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
