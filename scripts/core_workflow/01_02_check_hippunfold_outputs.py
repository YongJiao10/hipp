#!/usr/bin/env python3
"""
Process Step Description:
Check HippUnfold outputs, then remove core-workflow-unconsumed bulky directories.
The first pass requires the complete cached output contract. Only after that
validation succeeds does the checker remove anat, coords, warps, and cifti.
It writes a fingerprint-bound subject receipt so later checks validate the
retained output contract without reporting the approved cleanup as missing.
Historical schema-1 cleanup receipts with an older contract fingerprint are
reconciled only when their cleanup provenance is complete, all current
retained outputs are present, and no approved-pruned directory remains.

Main Flow Path:
- 2nd step in the numbered pipeline sequence:
  01_01_run_hippunfold_t1w_local.py -> 01_02_check_hippunfold_outputs.py (this step) -> 02_01_prepare_group_inputs.py -> 02_02_build_qc_manifest.py -> 03_01/03_02 reference building -> 04_01/04_02 FC materialization -> 05_01 gradient computation -> 05_02 group atlas parcellation -> 06_xx presentation rendering -> 07_01 group matrix overview.

Inputs:
- Command-line arguments:
  - --input-root: Input root containing structural raw sub-* directories (default: `data/hippunfold_input`).
  - --batch-root: Batch output root containing sub-*/hippunfold outputs (default: `outputs_hcp/inputs/hcp/hippunfold/dense_corobl_batch`).
  - --subject: Subject ID without "sub-" prefix. Can be specified multiple times.
  - --subjects-from-batch: Enumerate valid `sub-*` payloads from the selected
    batch instead of using the input-root roster.
  - --manifest-cache: Path to manifest cache JSON file (default: `<batch-root>/_batch_logs/output_integrity_manifest.json`).
  - --contract-source-cache: Validated full-output cache used only to seed a
    missing dataset-local cache, for datasets whose retained run logs are gone.
  - --json-out: Path for full validation JSON report.
  - --quiet: Suppress stdout logging.
  - --fail-on-missing: Fail with non-zero exit code if any checked subject is missing outputs.
  - --no-cleanup: Validate only; skip the default post-validation cleanup.
- Physical inputs:
  - Subject-level HippUnfold output directory structures: `{batch_root}/sub-<subject>/hippunfold/`.

Outputs:
- Command-line output / file outputs:
  - Manifest cache file: `{manifest_cache}` or `{batch_root}/_batch_logs/output_integrity_manifest.json` containing validation logs and file integrity status.
  - Optional JSON report at `{json_out}`.
  - Subject cleanup receipt at
    `{batch_root}/sub-<subject>/hippunfold/core_workflow_cleanup_receipt.json`.
  - After full validation, removal of subject payload directories `anat/`,
    `coords/`, `warps/`, and `cifti/`; `surf/`, `metric/`, and `qc/` are retained.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import tempfile
import time
from pathlib import Path
from typing import Any

from lib.output_layout import (
    DATASET_KEYS,
    DEFAULT_DATASET,
    resolve_hippunfold_batch_root,
)

SUCCESS_MARKER = "Finished jobid: 0 (Rule: all)"
FAIL_MARKER = "At least one job did not complete successfully."
COMPLETE_STEPS_PATTERN = re.compile(
    r"\b(?P<completed>\d+) of (?P<total>\d+) steps \(100%\) done\b"
)
NOOP_MARKER = "Nothing to be done (all requested files are present and up to date)."

PRUNED_OUTPUT_DIRS = ("anat", "coords", "warps", "cifti")
CLEANUP_RECEIPT_NAME = "core_workflow_cleanup_receipt.json"
CLEANUP_RECEIPT_SCHEMA_VERSION = 1
FIXED_HIPPUNFOLD_SPACE_ANCHORS = (
    "sub-{subject}/surf/sub-{subject}_hemi-L_space-corobl_den-512_label-hipp_midthickness.surf.gii",
    "sub-{subject}/surf/sub-{subject}_hemi-R_space-corobl_den-512_label-hipp_midthickness.surf.gii",
    "sub-{subject}/surf/sub-{subject}_hemi-L_space-T1w_den-512_label-hipp_midthickness.surf.gii",
    "sub-{subject}/surf/sub-{subject}_hemi-R_space-T1w_den-512_label-hipp_midthickness.surf.gii",
    "sub-{subject}/surf/sub-{subject}_hemi-L_space-unfold_den-512_label-hipp_midthickness.surf.gii",
    "sub-{subject}/surf/sub-{subject}_hemi-R_space-unfold_den-512_label-hipp_midthickness.surf.gii",
    "sub-{subject}/surf/sub-{subject}_den-512_label-hipp_surfaces.spec",
    "sub-{subject}/qc/sub-{subject}_hemi-L_desc-unetf3d_dice.tsv",
    "sub-{subject}/qc/sub-{subject}_hemi-R_desc-unetf3d_dice.tsv",
)
COMPLETE_STATUSES = {
    "complete_all_outputs_present",
    "complete_retained_outputs_after_cleanup",
}


def is_successful_log(log_text: str) -> bool:
    """Accept legacy and HippUnfold 2.x Snakemake success log formats."""
    if FAIL_MARKER in log_text:
        return False
    if SUCCESS_MARKER in log_text or NOOP_MARKER in log_text:
        return True
    return any(
        match.group("completed") == match.group("total")
        for match in COMPLETE_STEPS_PATTERN.finditer(log_text)
    )


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(
        description="Check whether each subject has a full HippUnfold final output set."
    )
    parser.add_argument("--dataset", default=DEFAULT_DATASET, choices=DATASET_KEYS,
                        help="Dataset key (default: hcp)")
    parser.add_argument(
        "--input-root",
        default=str(repo_root / "data" / "hippunfold_input"),
        help="Input root containing sub-* directories.",
    )
    parser.add_argument(
        "--batch-root",
        default=str(resolve_hippunfold_batch_root()),
        help="Batch output root containing sub-*/hippunfold.",
    )
    parser.add_argument(
        "--subject",
        action="append",
        default=[],
        help="Subject id without sub- prefix. Repeat for multiple subjects.",
    )
    parser.add_argument(
        "--subjects-from-batch",
        action="store_true",
        help="Check every valid sub-* payload in --batch-root.",
    )
    parser.add_argument(
        "--manifest-cache",
        default=None,
        help="Optional manifest cache JSON path (default: <batch-root>/_batch_logs/output_integrity_manifest.json).",
    )
    parser.add_argument(
        "--contract-source-cache",
        default=None,
        help=(
            "Optional validated full-output cache used only to seed a missing "
            "dataset-local manifest cache."
        ),
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
    parser.add_argument(
        "--no-cleanup",
        action="store_true",
        help=(
            "Validate only. By default, a subject that passes the complete-output "
            "contract is pruned of anat, coords, warps, and cifti directories."
        ),
    )
    return parser.parse_args()


def list_subjects(input_root: Path) -> list[str]:
    return sorted(
        p.name.replace("sub-", "", 1)
        for p in input_root.glob("sub-*")
        if p.is_dir()
    )


def list_batch_subjects(batch_root: Path) -> list[str]:
    subjects: list[str] = []
    for path in batch_root.glob("sub-*"):
        if not path.is_dir():
            continue
        subject = path.name.removeprefix("sub-")
        if not subject:
            continue
        if (path / "hippunfold" / f"sub-{subject}").is_dir():
            subjects.append(subject)
    return sorted(subjects)


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
        "has_success_marker": is_successful_log(text),
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
            if not is_successful_log(text):
                continue
            paths = parse_all_rule_inputs(text)
            subject_paths = [p for p in paths if p.startswith(f"sub-{sid}/")]
            if not subject_paths:
                subject_root = batch_root / f"sub-{sid}" / "hippunfold"
                subject_paths = [
                    p.relative_to(subject_root).as_posix()
                    for p in subject_root.rglob("*")
                    if p.is_file() and ".snakemake" not in p.parts and "logs" not in p.parts
                ]
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
                "expected_outputs": required_check_templates(templates),
            }
    raise RuntimeError(
        "Failed to build output manifest from successful logs: no parseable rule all input list found."
    )


def load_or_create_manifest(
    cache_path: Path,
    batch_root: Path,
    subjects: list[str],
    source_cache_path: Path | None = None,
) -> dict[str, Any]:
    if cache_path.exists():
        data = json.loads(cache_path.read_text(encoding="utf-8"))
        outputs = data.get("expected_outputs", [])
        if isinstance(outputs, list) and outputs:
            effective = required_check_templates([str(item) for item in outputs])
            if effective != outputs:
                data = {**data, "expected_outputs": effective}
                _atomic_write_json(cache_path, data)
            return data
    if source_cache_path is not None:
        source = json.loads(source_cache_path.read_text(encoding="utf-8"))
        outputs = source.get("expected_outputs", [])
        if not isinstance(outputs, list) or not outputs:
            raise ValueError(
                f"Contract source cache has no expected_outputs: {source_cache_path}"
            )
        effective = required_check_templates([str(item) for item in outputs])
        manifest = {
            "baseline_subject": source.get("baseline_subject", "external-contract"),
            "baseline_log": source.get("baseline_log"),
            "expected_outputs": effective,
            "seeded_from_contract_cache": str(source_cache_path.resolve()),
            "seeded_contract_sha256": manifest_contract_sha256(effective),
        }
        _atomic_write_json(cache_path, manifest)
        return manifest
    manifest = build_manifest_from_success_logs(batch_root, subjects)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def output_category(template: str) -> str | None:
    parts = Path(str(template)).parts
    if len(parts) < 2 or not parts[0].startswith("sub-"):
        return None
    return parts[1]


def required_check_templates(expected_templates: list[str]) -> list[str]:
    """Return the complete fixed HippUnfold 2.0 output contract."""
    templates = [str(template) for template in expected_templates]
    return list(dict.fromkeys([*templates, *FIXED_HIPPUNFOLD_SPACE_ANCHORS]))


def manifest_contract_sha256(expected_templates: list[str]) -> str:
    payload = json.dumps(
        required_check_templates(expected_templates),
        ensure_ascii=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def retained_expected_templates(expected_templates: list[str]) -> list[str]:
    pruned = set(PRUNED_OUTPUT_DIRS)
    return [
        str(template)
        for template in expected_templates
        if output_category(str(template)) not in pruned
    ]


def cleanup_receipt_path(batch_root: Path, sid: str) -> Path:
    return batch_root / f"sub-{sid}" / "hippunfold" / CLEANUP_RECEIPT_NAME


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=str(path.parent),
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_name, path)
    except BaseException:
        try:
            Path(temp_name).unlink()
        except FileNotFoundError:
            pass
        raise


def _read_cleanup_receipt(batch_root: Path, sid: str) -> dict[str, Any] | None:
    path = cleanup_receipt_path(batch_root, sid)
    if not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


def _cleanup_receipt_has_identity(payload: dict[str, Any], sid: str) -> bool:
    approved = payload.get("approved_pruned_directories")
    return (
        payload.get("schema_version") == CLEANUP_RECEIPT_SCHEMA_VERSION
        and payload.get("subject") == sid
        and isinstance(approved, list)
        and tuple(approved) == PRUNED_OUTPUT_DIRS
    )


def load_valid_cleanup_receipt(
    batch_root: Path,
    sid: str,
    expected_templates: list[str],
) -> dict[str, Any] | None:
    payload = _read_cleanup_receipt(batch_root, sid)
    if payload is None:
        return None
    if not _cleanup_receipt_has_identity(payload, sid):
        return None
    if payload.get("manifest_contract_sha256") != manifest_contract_sha256(expected_templates):
        return None
    if payload.get("status") not in {"validated_cleanup_pending", "cleanup_complete"}:
        return None
    return payload


def load_legacy_cleanup_receipt(
    batch_root: Path,
    sid: str,
    expected_templates: list[str],
) -> dict[str, Any] | None:
    """Return a completed historical receipt for retained-output reconciliation."""
    payload = _read_cleanup_receipt(batch_root, sid)
    if payload is None or not _cleanup_receipt_has_identity(payload, sid):
        return None
    if payload.get("status") != "cleanup_complete":
        return None
    if payload.get("manifest_contract_sha256") == manifest_contract_sha256(expected_templates):
        return None
    digest = payload.get("manifest_contract_sha256")
    if not isinstance(digest, str) or len(digest) != 64:
        return None
    try:
        int(digest, 16)
    except ValueError:
        return None
    full_count = payload.get("full_expected_outputs")
    retained_count = payload.get("retained_expected_outputs")
    if (
        isinstance(full_count, bool)
        or not isinstance(full_count, int)
        or full_count <= 0
        or isinstance(retained_count, bool)
        or not isinstance(retained_count, int)
        or retained_count <= 0
        or retained_count > full_count
    ):
        return None
    def is_nonnegative_int(value: Any) -> bool:
        return isinstance(value, int) and not isinstance(value, bool) and value >= 0

    aggregate_keys = (
        "allocated_bytes_removed",
        "apparent_bytes_removed",
        "files_removed",
    )
    if any(not is_nonnegative_int(payload.get(key)) for key in aggregate_keys):
        return None
    removed = payload.get("removed_directories")
    if not isinstance(removed, list):
        return None
    removed_names: list[str] = []
    totals = {key: 0 for key in aggregate_keys}
    for item in removed:
        if not isinstance(item, dict):
            return None
        name = item.get("directory")
        if not isinstance(name, str):
            return None
        if not isinstance(item.get("path"), str) or not item["path"]:
            return None
        if not isinstance(item.get("existed"), bool):
            return None
        if any(not is_nonnegative_int(item.get(key)) for key in aggregate_keys):
            return None
        removed_names.append(name)
        for key in aggregate_keys:
            totals[key] += item[key]
    if tuple(removed_names) != PRUNED_OUTPUT_DIRS:
        return None
    if any(payload[key] != totals[key] for key in aggregate_keys):
        return None
    return payload


def _tree_usage(path: Path) -> tuple[int, int, int]:
    apparent_bytes = 0
    allocated_bytes = 0
    file_count = 0
    if not path.exists() and not path.is_symlink():
        return apparent_bytes, allocated_bytes, file_count
    if path.is_symlink() or path.is_file():
        stat = path.lstat()
        return int(stat.st_size), int(getattr(stat, "st_blocks", 0) * 512), 1
    for root, dirs, files in os.walk(path, followlinks=False):
        root_path = Path(root)
        for name in [*dirs, *files]:
            item = root_path / name
            try:
                stat = item.lstat()
            except FileNotFoundError:
                continue
            apparent_bytes += int(stat.st_size)
            allocated_bytes += int(getattr(stat, "st_blocks", 0) * 512)
            if item.is_file() or item.is_symlink():
                file_count += 1
    return apparent_bytes, allocated_bytes, file_count


def _remove_pruned_directories(batch_root: Path, sid: str) -> list[dict[str, Any]]:
    output_root = (batch_root / f"sub-{sid}" / "hippunfold").resolve()
    subject_root = output_root / f"sub-{sid}"
    results: list[dict[str, Any]] = []
    for name in PRUNED_OUTPUT_DIRS:
        target = subject_root / name
        if target.parent != subject_root or target.name != name:
            raise RuntimeError(f"Unsafe cleanup target rejected: {target}")
        existed = target.exists() or target.is_symlink()
        apparent_bytes, allocated_bytes, file_count = _tree_usage(target)
        if target.is_symlink() or target.is_file():
            target.unlink()
        elif target.is_dir():
            shutil.rmtree(target)
        results.append(
            {
                "directory": name,
                "path": str(target),
                "existed": bool(existed),
                "apparent_bytes_removed": int(apparent_bytes),
                "allocated_bytes_removed": int(allocated_bytes),
                "files_removed": int(file_count),
            }
        )
    return results


def _pruned_directories_present(batch_root: Path, sid: str) -> bool:
    subject_root = batch_root / f"sub-{sid}" / "hippunfold" / f"sub-{sid}"
    return any(
        (subject_root / name).exists() or (subject_root / name).is_symlink()
        for name in PRUNED_OUTPUT_DIRS
    )


def _check_templates(
    batch_root: Path,
    sid: str,
    templates: list[str],
) -> list[str]:
    output_root = batch_root / f"sub-{sid}" / "hippunfold"
    missing: list[str] = []
    for template in templates:
        rel = str(template).replace("{subject}", sid)
        if not (output_root / rel).exists():
            missing.append(rel)
    return missing


def check_subject(
    batch_root: Path, sid: str, expected_templates: list[str]
) -> dict[str, Any]:
    output_root = batch_root / f"sub-{sid}" / "hippunfold"
    latest = latest_log_info(batch_root, sid)
    full_templates = required_check_templates(expected_templates)
    receipt = load_valid_cleanup_receipt(batch_root, sid, full_templates)
    templates = full_templates
    validation_mode = "full_output_contract"
    if receipt is not None:
        templates = retained_expected_templates(full_templates)
        validation_mode = "retained_outputs_with_valid_cleanup_receipt"
    else:
        legacy_receipt = load_legacy_cleanup_receipt(batch_root, sid, full_templates)
        legacy_templates = retained_expected_templates(full_templates)
        legacy_missing = _check_templates(batch_root, sid, legacy_templates)
        if (
            legacy_receipt is not None
            and not legacy_missing
            and not _pruned_directories_present(batch_root, sid)
        ):
            receipt = legacy_receipt
            templates = legacy_templates
            validation_mode = "retained_outputs_with_legacy_cleanup_receipt"
    missing = _check_templates(batch_root, sid, templates)
    if receipt is not None and not missing:
        status = "complete_retained_outputs_after_cleanup"
    else:
        status = "complete_all_outputs_present" if not missing else "missing_outputs"
    if not output_root.exists():
        status = "missing_output_root"
    return {
        "subject": sid,
        "status": status,
        "validation_mode": validation_mode,
        "expected_outputs": len(templates),
        "full_expected_outputs": len(full_templates),
        "missing_outputs": missing,
        "missing_count": len(missing),
        "cleanup_receipt": str(cleanup_receipt_path(batch_root, sid)) if receipt else None,
        "cleanup_receipt_status": receipt.get("status") if receipt else None,
        "latest_log": latest["path"],
        "latest_log_has_success_marker": latest["has_success_marker"],
        "latest_log_has_fail_marker": latest["has_fail_marker"],
    }


def validate_and_cleanup_subject(
    batch_root: Path,
    sid: str,
    expected_templates: list[str],
    *,
    manifest_cache: Path,
    cleanup_enabled: bool,
) -> dict[str, Any]:
    full_templates = required_check_templates(expected_templates)
    report = check_subject(batch_root, sid, full_templates)
    if report["status"] not in COMPLETE_STATUSES or not cleanup_enabled:
        return report

    receipt_path = cleanup_receipt_path(batch_root, sid)
    receipt = load_valid_cleanup_receipt(batch_root, sid, full_templates)
    if receipt is None:
        receipt = load_legacy_cleanup_receipt(batch_root, sid, full_templates)
    if (
        receipt is not None
        and receipt.get("status") == "cleanup_complete"
        and not _pruned_directories_present(batch_root, sid)
    ):
        report["cleanup"] = {
            "performed": False,
            "already_complete": True,
            "allocated_bytes_removed": 0,
            "apparent_bytes_removed": 0,
            "files_removed": 0,
            "removed_directories": [],
        }
        return report
    if receipt is None:
        receipt = {
            "schema_version": CLEANUP_RECEIPT_SCHEMA_VERSION,
            "status": "validated_cleanup_pending",
            "subject": sid,
            "validated_unix_time": time.time(),
            "manifest_cache": str(manifest_cache),
            "manifest_contract_sha256": manifest_contract_sha256(full_templates),
            "full_expected_outputs": len(full_templates),
            "retained_expected_outputs": len(retained_expected_templates(full_templates)),
            "approved_pruned_directories": list(PRUNED_OUTPUT_DIRS),
        }
        _atomic_write_json(receipt_path, receipt)

    removed = _remove_pruned_directories(batch_root, sid)
    current_allocated = int(sum(item["allocated_bytes_removed"] for item in removed))
    current_apparent = int(sum(item["apparent_bytes_removed"] for item in removed))
    current_files = int(sum(item["files_removed"] for item in removed))
    previous_allocated = int(receipt.get("allocated_bytes_removed", 0))
    previous_apparent = int(receipt.get("apparent_bytes_removed", 0))
    previous_files = int(receipt.get("files_removed", 0))
    receipt.update(
        {
            "status": "cleanup_complete",
            "cleanup_completed_unix_time": time.time(),
            "removed_directories": removed,
            "allocated_bytes_removed": previous_allocated + current_allocated,
            "apparent_bytes_removed": previous_apparent + current_apparent,
            "files_removed": previous_files + current_files,
        }
    )
    _atomic_write_json(receipt_path, receipt)

    final_report = check_subject(batch_root, sid, full_templates)
    final_report["cleanup"] = {
        "performed": True,
        "allocated_bytes_removed": current_allocated,
        "apparent_bytes_removed": current_apparent,
        "files_removed": current_files,
        "removed_directories": removed,
    }
    return final_report


def print_summary(report: list[dict[str, Any]], manifest: dict[str, Any]) -> None:
    total = len(report)
    complete = sum(1 for r in report if r["status"] in COMPLETE_STATUSES)
    trimmed = sum(
        1 for r in report if r["status"] == "complete_retained_outputs_after_cleanup"
    )
    missing = total - complete
    allocated_removed = sum(
        int(r.get("cleanup", {}).get("allocated_bytes_removed", 0)) for r in report
    )
    print(
        "subjects_total={} complete={} trimmed={} incomplete={} expected_outputs_per_subject={} baseline_sub={} allocated_bytes_removed={}".format(
            total,
            complete,
            trimmed,
            missing,
            len(manifest["expected_outputs"]),
            manifest["baseline_subject"],
            allocated_removed,
        )
    )
    if missing:
        top = sorted(
            (r for r in report if r["status"] not in COMPLETE_STATUSES),
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
    if args.subjects_from_batch and args.subject:
        raise ValueError("--subjects-from-batch cannot be combined with --subject")
    subjects = (
        list_batch_subjects(batch_root)
        if args.subjects_from_batch
        else (args.subject or list_subjects(input_root))
    )
    if not subjects:
        raise RuntimeError("No subjects found to check.")

    default_cache = batch_root / "_batch_logs" / "output_integrity_manifest.json"
    cache_path = Path(args.manifest_cache).resolve() if args.manifest_cache else default_cache
    source_cache_path = (
        Path(args.contract_source_cache).resolve()
        if args.contract_source_cache
        else None
    )
    manifest = load_or_create_manifest(
        cache_path,
        batch_root,
        subjects,
        source_cache_path=source_cache_path,
    )
    expected_templates = manifest["expected_outputs"]
    report = [
        validate_and_cleanup_subject(
            batch_root,
            sid,
            expected_templates,
            manifest_cache=cache_path,
            cleanup_enabled=not args.no_cleanup,
        )
        for sid in subjects
    ]

    if args.json_out:
        out = Path(args.json_out).resolve()
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, indent=2), encoding="utf-8")

    if not args.quiet:
        print_summary(report, manifest)

    has_missing = any(r["status"] not in COMPLETE_STATUSES for r in report)
    if args.fail_on_missing and has_missing:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
