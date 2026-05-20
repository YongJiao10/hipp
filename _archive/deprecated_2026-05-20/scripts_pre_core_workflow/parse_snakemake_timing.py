#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path


TIMESTAMP_RE = re.compile(r"^\[(.+)\]$")
RULE_START_RE = re.compile(r"^(?:localrule|rule)\s+([A-Za-z0-9_]+):$")
JOBID_RE = re.compile(r"^\s*jobid:\s*(\d+)\s*$")
FINISH_RE = re.compile(r"^Finished jobid:\s*(\d+)\s+\(Rule:\s*([A-Za-z0-9_]+)\)$")
ERROR_RULE_RE = re.compile(r"^Error in rule\s+([A-Za-z0-9_]+):$")


@dataclass
class ActiveJob:
    rule: str
    start_time: datetime


def parse_timestamp(text: str) -> datetime:
    return datetime.strptime(text, "%a %b %d %H:%M:%S %Y")


def summarize_records(records: list[dict[str, object]]) -> list[dict[str, object]]:
    grouped: dict[str, dict[str, object]] = defaultdict(
        lambda: {"rule": "", "count": 0, "total_seconds": 0.0, "success_count": 0, "failure_count": 0}
    )
    for record in records:
        rule = str(record["rule"])
        item = grouped[rule]
        item["rule"] = rule
        item["count"] = int(item["count"]) + 1
        item["total_seconds"] = float(item["total_seconds"]) + float(record["duration_seconds"])
        if record["status"] == "success":
            item["success_count"] = int(item["success_count"]) + 1
        else:
            item["failure_count"] = int(item["failure_count"]) + 1

    summary = []
    for item in grouped.values():
        count = max(1, int(item["count"]))
        total = float(item["total_seconds"])
        summary.append(
            {
                "rule": item["rule"],
                "count": int(item["count"]),
                "success_count": int(item["success_count"]),
                "failure_count": int(item["failure_count"]),
                "total_seconds": total,
                "mean_seconds": total / count,
            }
        )
    summary.sort(key=lambda x: (-float(x["total_seconds"]), str(x["rule"])))
    return summary


def parse_log(log_path: Path) -> dict[str, object]:
    lines = log_path.read_text(encoding="utf-8", errors="replace").splitlines()
    active: dict[str, ActiveJob] = {}
    records: list[dict[str, object]] = []
    last_timestamp: datetime | None = None
    pending_rule: tuple[str, datetime] | None = None
    pending_error_rule: tuple[str, datetime] | None = None
    first_timestamp: datetime | None = None
    final_timestamp: datetime | None = None

    for line in lines:
        ts_match = TIMESTAMP_RE.match(line.strip())
        if ts_match:
            last_timestamp = parse_timestamp(ts_match.group(1))
            final_timestamp = last_timestamp
            if first_timestamp is None:
                first_timestamp = last_timestamp
            continue

        start_match = RULE_START_RE.match(line.strip())
        if start_match and last_timestamp is not None:
            pending_rule = (start_match.group(1), last_timestamp)
            pending_error_rule = None
            continue

        error_match = ERROR_RULE_RE.match(line.strip())
        if error_match and last_timestamp is not None:
            pending_error_rule = (error_match.group(1), last_timestamp)
            pending_rule = None
            continue

        jobid_match = JOBID_RE.match(line)
        if jobid_match:
            jobid = jobid_match.group(1)
            if pending_rule is not None:
                rule, start_time = pending_rule
                active[jobid] = ActiveJob(rule=rule, start_time=start_time)
                pending_rule = None
                continue
            if pending_error_rule is not None:
                rule, fail_time = pending_error_rule
                job = active.pop(jobid, None)
                start_time = job.start_time if job is not None else fail_time
                records.append(
                    {
                        "jobid": int(jobid),
                        "rule": rule,
                        "status": "failed",
                        "start_time": start_time.isoformat(),
                        "end_time": fail_time.isoformat(),
                        "duration_seconds": max(0.0, (fail_time - start_time).total_seconds()),
                    }
                )
                pending_error_rule = None
                continue

        finish_match = FINISH_RE.match(line.strip())
        if finish_match and last_timestamp is not None:
            jobid, rule = finish_match.groups()
            job = active.pop(jobid, None)
            if job is None:
                continue
            records.append(
                {
                    "jobid": int(jobid),
                    "rule": rule,
                    "status": "success",
                    "start_time": job.start_time.isoformat(),
                    "end_time": last_timestamp.isoformat(),
                    "duration_seconds": max(0.0, (last_timestamp - job.start_time).total_seconds()),
                }
            )

    for jobid, job in sorted(active.items(), key=lambda item: int(item[0])):
        end_time = final_timestamp or job.start_time
        records.append(
            {
                "jobid": int(jobid),
                "rule": job.rule,
                "status": "unfinished",
                "start_time": job.start_time.isoformat(),
                "end_time": end_time.isoformat(),
                "duration_seconds": max(0.0, (end_time - job.start_time).total_seconds()),
            }
        )

    return {
        "log_path": str(log_path),
        "n_records": len(records),
        "wall_clock_seconds": (
            max(0.0, (final_timestamp - first_timestamp).total_seconds())
            if first_timestamp is not None and final_timestamp is not None
            else 0.0
        ),
        "records": sorted(records, key=lambda x: (x["start_time"], x["jobid"])),
        "summary_by_rule": summarize_records(records),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Parse Snakemake log timestamps into per-rule timing records")
    parser.add_argument("--log", required=True)
    parser.add_argument("--out", default=None, help="Optional JSON output path")
    args = parser.parse_args()

    report = parse_log(Path(args.log))
    text = json.dumps(report, indent=2)
    if args.out:
        Path(args.out).write_text(text, encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
