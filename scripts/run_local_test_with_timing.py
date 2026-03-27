#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
from datetime import datetime
from pathlib import Path
from time import perf_counter

from parse_snakemake_timing import parse_log


def format_seconds(seconds: float) -> str:
    total = int(round(seconds))
    minutes, secs = divmod(total, 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{hours:d}:{minutes:02d}:{secs:02d}"
    return f"{minutes:d}:{secs:02d}"


def build_table(rows: list[dict[str, object]]) -> str:
    headers = ["Step", "Status", "Duration", "Details"]
    widths = [len(h) for h in headers]
    rendered: list[list[str]] = []
    for row in rows:
        cols = [
            str(row["step"]),
            str(row["status"]),
            str(row["duration"]),
            str(row["details"]),
        ]
        rendered.append(cols)
        widths = [max(widths[i], len(cols[i])) for i in range(len(headers))]

    lines = []
    header_line = "  ".join(headers[i].ljust(widths[i]) for i in range(len(headers)))
    sep_line = "  ".join("-" * widths[i] for i in range(len(headers)))
    lines.append(header_line)
    lines.append(sep_line)
    for cols in rendered:
        lines.append(
            "  ".join(
                [
                    cols[0].ljust(widths[0]),
                    cols[1].ljust(widths[1]),
                    cols[2].rjust(widths[2]),
                    cols[3].ljust(widths[3]),
                ]
            )
        )
    return "\n".join(lines)


def run_stage(cmd: list[str], log_prefix: Path) -> dict[str, object]:
    start_dt = datetime.now()
    start = perf_counter()
    proc = subprocess.run(cmd, text=True, capture_output=True)
    duration = perf_counter() - start
    stdout_path = log_prefix.with_suffix(".stdout.log")
    stderr_path = log_prefix.with_suffix(".stderr.log")
    stdout_path.write_text(proc.stdout, encoding="utf-8", errors="replace")
    stderr_path.write_text(proc.stderr, encoding="utf-8", errors="replace")
    return {
        "command": cmd,
        "status": "success" if proc.returncode == 0 else "failed",
        "returncode": int(proc.returncode),
        "start_time": start_dt.isoformat(),
        "duration_seconds": duration,
        "stdout_log": str(stdout_path),
        "stderr_log": str(stderr_path),
    }


def latest_snakemake_log(hippunfold_dir: Path) -> Path | None:
    log_dir = hippunfold_dir / ".snakemake" / "log"
    logs = sorted(log_dir.glob("*.snakemake.log"))
    return logs[-1] if logs else None


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the local single-subject test flow with timing capture")
    parser.add_argument("--subject", default="100610")
    parser.add_argument("--input-dir", default="data/hippunfold_input")
    parser.add_argument("--bids-dir", dest="input_dir_legacy", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--out-root", default="outputs/timing_runs")
    parser.add_argument("--label", default=None, help="Optional run label suffix")
    args = parser.parse_args()
    input_dir = args.input_dir_legacy or args.input_dir

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"sub-{args.subject}_{stamp}" if not args.label else f"sub-{args.subject}_{stamp}_{args.label}"
    run_dir = Path(args.out_root) / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    hippunfold_dir = run_dir / "hippunfold"
    logs_dir = run_dir / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)

    stage_records: list[dict[str, object]] = []
    hipp_stage = run_stage(
        [
            "zsh",
            "/Users/jy/Documents/HippoMaps/scripts/run_hippunfold_local.sh",
            args.subject,
            input_dir,
            str(hippunfold_dir),
        ],
        logs_dir / "01_hippunfold",
    )
    stage_records.append(
        {
            "step": "hippunfold_local_test",
            "status": hipp_stage["status"],
            "duration_seconds": hipp_stage["duration_seconds"],
            "returncode": hipp_stage["returncode"],
            "stdout_log": hipp_stage["stdout_log"],
            "stderr_log": hipp_stage["stderr_log"],
        }
    )

    snakemake_report = None
    log_path = latest_snakemake_log(hippunfold_dir)
    if log_path is not None:
        snakemake_report = parse_log(log_path)
        (run_dir / "hippunfold_rule_timing.json").write_text(json.dumps(snakemake_report, indent=2), encoding="utf-8")

    report = {
        "subject": args.subject,
        "run_dir": str(run_dir),
        "hippunfold_dir": str(hippunfold_dir),
        "stage_records": stage_records,
        "snakemake_log": str(log_path) if log_path is not None else None,
        "snakemake_rule_timing_json": str(run_dir / "hippunfold_rule_timing.json") if snakemake_report else None,
    }
    (run_dir / "timing_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    table_rows = [
        {
            "step": stage["step"],
            "status": stage["status"],
            "duration": format_seconds(float(stage["duration_seconds"])),
            "details": f"returncode={stage['returncode']}",
        }
        for stage in stage_records
    ]
    if snakemake_report:
        for item in snakemake_report["summary_by_rule"][:12]:
            table_rows.append(
                {
                    "step": f"rule:{item['rule']}",
                    "status": "mixed" if int(item["failure_count"]) else "ok",
                    "duration": format_seconds(float(item["total_seconds"])),
                    "details": (
                        f"count={item['count']}, success={item['success_count']}, failure={item['failure_count']}"
                    ),
                }
            )

    summary_md = (
        f"# Timing Summary\n\n"
        f"Run directory: `{run_dir}`\n\n"
        f"```text\n{build_table(table_rows)}\n```\n"
    )
    (run_dir / "timing_summary.md").write_text(summary_md, encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
