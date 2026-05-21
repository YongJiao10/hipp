#!/usr/bin/env python3
"""Validate per-subject HippUnfold outputs against the Snakemake rule all targets."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

SUCCESS_MARKER = "Finished jobid: 0 (Rule: all)"
FAIL_MARKER = "At least one job did not complete successfully."


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(
        description="Check whether each subject has a full HippUnfold final output set."
    )
    parser.add_argument(
        "--input-root",
        default=str(repo_root / "data" / "hippunfold_input"),
        help="Input root containing sub-* directories.",
    )
    parser.add_argument(
        "--batch-root",
        default=str(repo_root / "outputs_core" / "inputs" / "hcp7t_hippocampus" / "hippunfold" / "dense_corobl_batch"),
        help="Batch output root containing sub-*/hippunfold.",
    )
    parser.add_argument(
        "--subject",
        action="append",
        default=[],
        help="Subject id without sub- prefix. Repeat for multiple subjects.",
    )
    parser.add_argument(
        "--manifest-cache",
        default=None,
        help="Optional manifest cache JSON path (default: <batch-root>/_batch_logs/output_integrity_manifest.json).",
    )
    parser.add_argument(
        "--json-out",
        default=None,
        help="Optional JSON output path for full report.",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress summary printing.",
    )
    parser.add_argument(
        "--fail-on-missing",
        action="store_true",
        help="Exit non-zero when any checked subject has missing outputs.",
    )
    return parser.parse_args()


def list_subjects(input_root: Path) -> list[str]:
    return sorted(
        p.name.replace("sub-", "", 1)
        for p in input_root.glob("sub-*")
        if p.is_dir()
    )


def get_subject_log_paths(batch_root: Path, sid: str) -> list[Path]:
    log_dir = batch_root / f"sub-{sid}" / "hippunfold" / ".snakemake" / "log"
    if not log_dir.is_dir():
        return []
    return sorted(
        log_dir.glob("*.snakemake.log"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )


def latest_log_info(batch_root: Path, sid: str) -> dict[str, Any]:
    logs = get_subject_log_paths(batch_root, sid)
    if not logs:
        return {"path": None, "has_success_marker": False, "has_fail_marker": False}
    text = logs[0].read_text(encoding="utf-8", errors="ignore")
    return {
        "path": str(logs[0]),
        "has_success_marker": SUCCESS_MARKER in text,
        "has_fail_marker": FAIL_MARKER in text,
    }


def parse_all_rule_inputs(log_text: str) -> list[str]:
    lines = log_text.splitlines()
    for i, line in enumerate(lines):
        if line.strip() != "localrule all:":
            continue
        chunks: list[str] = []
        saw_input = False
        for j in range(i + 1, len(lines)):
            current = lines[j]
            if current.startswith("    input:"):
                chunks.append(current.split("input:", 1)[1].strip())
                saw_input = True
                continue
            if saw_input and re.match(r"^\s{8,}\S", current):
                chunks.append(current.strip())
                continue
            if saw_input:
                break
        if saw_input:
            joined = " ".join(chunks)
            return [x.strip() for x in joined.split(",") if x.strip()]
    return []


def build_manifest_from_success_logs(
    batch_root: Path, subjects: list[str]
) -> dict[str, Any]:
    for sid in subjects:
        for log_path in get_subject_log_paths(batch_root, sid):
            text = log_path.read_text(encoding="utf-8", errors="ignore")
            if SUCCESS_MARKER not in text or FAIL_MARKER in text:
                continue
            paths = parse_all_rule_inputs(text)
            subject_paths = [p for p in paths if p.startswith(f"sub-{sid}/")]
            if not subject_paths:
                continue
            templates = []
            seen = set()
            for p in subject_paths:
                t = p.replace(f"sub-{sid}", "sub-{subject}")
                if t not in seen:
                    seen.add(t)
                    templates.append(t)
            return {
                "baseline_subject": sid,
                "baseline_log": str(log_path),
                "expected_outputs": templates,
            }
    raise RuntimeError(
        "Failed to build output manifest from successful logs: no parseable rule all input list found."
    )


def load_or_create_manifest(
    cache_path: Path, batch_root: Path, subjects: list[str]
) -> dict[str, Any]:
    if cache_path.exists():
        data = json.loads(cache_path.read_text(encoding="utf-8"))
        outputs = data.get("expected_outputs", [])
        if isinstance(outputs, list) and outputs:
            return data
    manifest = build_manifest_from_success_logs(batch_root, subjects)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def check_subject(
    batch_root: Path, sid: str, expected_templates: list[str]
) -> dict[str, Any]:
    output_root = batch_root / f"sub-{sid}" / "hippunfold"
    latest = latest_log_info(batch_root, sid)
    missing = []
    for template in expected_templates:
        rel = template.replace("{subject}", sid)
        if not (output_root / rel).exists():
            missing.append(rel)
    status = "complete_all_outputs_present" if not missing else "missing_outputs"
    if not output_root.exists():
        status = "missing_output_root"
    return {
        "subject": sid,
        "status": status,
        "expected_outputs": len(expected_templates),
        "missing_outputs": missing,
        "missing_count": len(missing),
        "latest_log": latest["path"],
        "latest_log_has_success_marker": latest["has_success_marker"],
        "latest_log_has_fail_marker": latest["has_fail_marker"],
    }


def print_summary(report: list[dict[str, Any]], manifest: dict[str, Any]) -> None:
    total = len(report)
    complete = sum(1 for r in report if r["status"] == "complete_all_outputs_present")
    missing = total - complete
    print(
        "subjects_total={} complete={} incomplete={} expected_outputs_per_subject={} baseline_sub={}".format(
            total,
            complete,
            missing,
            len(manifest["expected_outputs"]),
            manifest["baseline_subject"],
        )
    )
    if missing:
        top = sorted(
            (r for r in report if r["status"] != "complete_all_outputs_present"),
            key=lambda x: x["missing_count"],
            reverse=True,
        )[:10]
        print("top_missing_subjects:")
        for row in top:
            print(
                "  sub-{} missing_count={} status={}".format(
                    row["subject"], row["missing_count"], row["status"]
                )
            )


def main() -> int:
    args = parse_args()
    input_root = Path(args.input_root).resolve()
    batch_root = Path(args.batch_root).resolve()
    subjects = args.subject or list_subjects(input_root)
    if not subjects:
        raise RuntimeError("No subjects found to check.")

    default_cache = batch_root / "_batch_logs" / "output_integrity_manifest.json"
    cache_path = Path(args.manifest_cache).resolve() if args.manifest_cache else default_cache
    manifest = load_or_create_manifest(cache_path, batch_root, subjects)
    expected_templates = manifest["expected_outputs"]
    report = [check_subject(batch_root, sid, expected_templates) for sid in subjects]

    if args.json_out:
        out = Path(args.json_out).resolve()
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, indent=2), encoding="utf-8")

    if not args.quiet:
        print_summary(report, manifest)

    has_missing = any(r["status"] != "complete_all_outputs_present" for r in report)
    if args.fail_on_missing and has_missing:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
