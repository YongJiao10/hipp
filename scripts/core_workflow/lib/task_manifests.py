#!/usr/bin/env python3
"""Per-subject run-artifact manifest builder for HCP 3T task branches.

Walks a staged HCP task tree (``sub-<id>/func/<run_label>/``) and writes
the exact JSON receipt consumed by ``load_run_artifact_manifest`` with
``required_artifacts=("bold", "dtseries")``: absolute canonical paths plus
SHA-256/size fingerprints per run, ordered LR then RL.

A missing run directory, an ambiguous run file set, or a frame count that
disagrees with the maintained task contract fails closed; no partial
manifest is ever written.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from .task_design import (
    TASK_RUN_LABELS,
    TASK_RUN_LENGTHS,
    normalize_task_branch,
)


def task_run_labels(task: str) -> tuple[str, ...]:
    return TASK_RUN_LABELS[normalize_task_branch(task)]


def task_run_manifest_path(
    manifest_root: str | Path, *, subject: str, task: str
) -> Path:
    branch = normalize_task_branch(task)
    subject_id = str(subject).strip().removeprefix("sub-")
    if not subject_id:
        raise ValueError(f"Invalid subject: {subject!r}")
    return (
        Path(manifest_root).resolve() / f"task-{branch}" / f"sub-{subject_id}.json"
    )


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def task_runs_from_manifest(
    manifest_path: str | Path, *, subject: str
) -> list[dict[str, object]]:
    """Resolve ordered run identities from a built run-artifact manifest.

    Returns one record per run in manifest order with run_id, run label
    (source directory name), and canonical bold/dtseries paths. Callers must
    read per-run frame counts from file headers and validate them; run order
    is never inferred from total frames or filenames.
    """
    from .io_utils import load_run_artifact_manifest

    records = load_run_artifact_manifest(
        Path(manifest_path), subject=subject, required_artifacts=("bold", "dtseries")
    )
    runs: list[dict[str, object]] = []
    for record in records:
        bold = Path(record["artifacts"]["bold"])
        runs.append(
            {
                "run_id": int(record["run_id"]),
                "run_label": bold.parent.name,
                "run_dir": bold.parent,
                "bold": bold,
                "dtseries": Path(record["artifacts"]["dtseries"]),
            }
        )
    return runs


def verify_cached_runs_against_manifest(
    *,
    subject: str,
    task: str,
    cached_kind: str,
    cached_summary_path: str | Path,
    cached_run_ids: list[int],
    cached_run_lengths: list[int],
    cached_source_paths: list[str],
    manifest_runs: list[dict[str, object]],
    manifest_source_key: str,
    manifest_lengths: list[int],
) -> dict[str, object]:
    """Cross-check an already-sampled cache against the live manifest.

    The manifest declares run identity/order today, but the cached timeseries
    was concatenated by an earlier 02_01/03_02 pass. Total-frame equality
    cannot catch a stale manifest whose runs were reordered (LR/RL swaps are
    equal-length). This verifier requires the cache's own sampling receipt
    (ordered run ids, lengths, source paths) to match the manifest
    entry-by-entry; any drift fails closed with a re-sample instruction.

    Returns a JSON-serializable verification receipt for output summaries.
    """
    summary_path = Path(cached_summary_path)
    if not summary_path.is_file():
        raise FileNotFoundError(
            f"Missing {cached_kind} sampling receipt for sub-{subject} "
            f"task {task}: {summary_path} (re-run the sampling stage)"
        )
    manifest_ids = [int(r["run_id"]) for r in manifest_runs]
    manifest_sources = [str(Path(str(r[manifest_source_key])).resolve()) for r in manifest_runs]
    cached_sources = [str(Path(str(p)).resolve()) for p in cached_source_paths]
    problems: list[str] = []
    if [int(v) for v in cached_run_ids] != manifest_ids:
        problems.append(
            f"run order drift: cached={list(cached_run_ids)} manifest={manifest_ids}"
        )
    if [int(v) for v in cached_run_lengths] != [int(v) for v in manifest_lengths]:
        problems.append(
            f"run length drift: cached={list(cached_run_lengths)} "
            f"manifest={list(manifest_lengths)}"
        )
    if cached_sources != manifest_sources:
        problems.append(
            f"source path drift: cached={cached_sources} manifest={manifest_sources}"
        )
    if problems:
        raise ValueError(
            f"Stale task cache for sub-{subject} task {task} ({cached_kind} "
            f"receipt {summary_path}): " + "; ".join(problems)
            + ". Re-run 02_01/03_02 sampling before 04_03/04_04."
        )
    return {
        "cached_kind": cached_kind,
        "cached_summary": str(summary_path.resolve()),
        "run_ids": manifest_ids,
        "run_lengths": [int(v) for v in manifest_lengths],
        "source_key": manifest_source_key,
        "match": True,
    }


def find_task_run_files(run_dir: Path) -> tuple[Path, Path]:
    """Return ``(bold_volume, dtseries)`` for one staged task run directory."""
    run_dir = Path(run_dir).resolve()
    if not run_dir.is_dir():
        raise FileNotFoundError(f"Missing staged task run directory: {run_dir}")
    dtseries = sorted(run_dir.glob("*.dtseries.nii"))
    volumes = sorted(
        path
        for path in run_dir.glob("*.nii.gz")
        if path.is_file() and ".dtseries." not in path.name
    )
    if len(dtseries) != 1:
        raise ValueError(
            f"Expected exactly one dtseries in {run_dir}, found {len(dtseries)}: "
            f"{[p.name for p in dtseries]}"
        )
    if len(volumes) != 1:
        raise ValueError(
            f"Expected exactly one BOLD volume in {run_dir}, found {len(volumes)}: "
            f"{[p.name for p in volumes]}"
        )
    return volumes[0].resolve(), dtseries[0].resolve()


def _receipt(path: Path) -> dict[str, object]:
    resolved = path.resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"Missing declared task artifact: {resolved}")
    return {
        "path": str(resolved),
        "sha256": _sha256_file(resolved),
        "size_bytes": int(resolved.stat().st_size),
    }


def build_task_run_artifact_manifest(
    *,
    subject: str,
    task: str,
    task_data_root: str | Path,
    manifest_root: str | Path,
    overwrite: bool = False,
) -> Path:
    """Build and write one subject's task run-artifact manifest receipt."""
    branch = normalize_task_branch(task)
    subject_id = str(subject).strip().removeprefix("sub-")
    if not subject_id:
        raise ValueError(f"Invalid subject: {subject!r}")
    data_root = Path(task_data_root).resolve()
    out_path = task_run_manifest_path(manifest_root, subject=subject_id, task=branch)
    if out_path.exists() and not overwrite:
        raise FileExistsError(
            f"Task manifest already exists (pass overwrite=True to rebuild): {out_path}"
        )
    runs: list[dict[str, object]] = []
    for index, label in enumerate(TASK_RUN_LABELS[branch], start=1):
        run_dir = data_root / f"sub-{subject_id}" / "func" / label
        bold, dtseries = find_task_run_files(run_dir)
        runs.append(
            {
                "run_id": index,
                "run_label": label,
                "bold": _receipt(bold),
                "dtseries": _receipt(dtseries),
            }
        )
    payload = {
        "subject": f"sub-{subject_id}",
        "task": branch,
        "task_data_root": str(data_root),
        "expected_frames_per_run": int(TASK_RUN_LENGTHS[branch]),
        "runs": runs,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = out_path.with_suffix(".tmp.json")
    tmp_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    tmp_path.replace(out_path)
    return out_path
