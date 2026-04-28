#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
RUN_ID_ROOT = REPO_ROOT / "outputs_experiments" / "hippunfold_recommended_t1w_20260423"
DEFAULT_BATCH_ROOT = RUN_ID_ROOT / "dense_corobl_batch"
DEFAULT_ROUTE_ROOT = RUN_ID_ROOT / "kong2022_route_b"
DEFAULT_SHARED_ROOT = DEFAULT_ROUTE_ROOT / "hipp_group_inputs_shared"
DEFAULT_GROUP_ROOT = DEFAULT_ROUTE_ROOT / "hipp_group_merged_canonical_atlas"
DEFAULT_PRESENT_DIR = REPO_ROOT / "present_group_t1only"
DEFAULT_MANIFEST = DEFAULT_ROUTE_ROOT / "manifests" / "hcp_7t_hippocampus_fmri_qc_pass_t1only_latest.txt"
DEFAULT_GROUP_AVG = DEFAULT_GROUP_ROOT / "group_average_surfaces"

PREPARE = REPO_ROOT / "scripts" / "hipp_parcellation_network" / "prepare_t1only_group_inputs.py"
KONG2022_REF = REPO_ROOT / "scripts" / "cortex" / "run_cortex_kong2022_reference_batch.py"
KONG2022_FC = REPO_ROOT / "scripts" / "hipp_parcellation_network" / "materialize_kong2022_vertex_to_roi_fc.py"
INTRINSIC_FC = REPO_ROOT / "scripts" / "hipp_parcellation_network" / "materialize_intrinsic_vertex_to_vertex_fc.py"
GROUP_ATLAS = REPO_ROOT / "scripts" / "hipp_parcellation_network" / "run_merged_canonical_group_atlas.py"
PRESENT = REPO_ROOT / "scripts" / "hipp_parcellation_network" / "build_present_group_overview_merged_canonical.py"
CLUSTER_ROI_CORTEX = REPO_ROOT / "scripts" / "hipp_parcellation_network" / "build_cluster_roi_cortex_independent.py"

STAGES = ["prepare", "kong2022-reference", "kong2022-fc", "intrinsic-fc", "group-atlas", "present", "cluster-roi-cortex"]
DEFAULT_MAX_CONCURRENCY = 10
DEFAULT_PIPELINE_JOBS = 4
DEFAULT_GROUP_ATLAS_JOBS = 4


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def git_commit() -> str | None:
    proc = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=REPO_ROOT, text=True, capture_output=True)
    return proc.stdout.strip() if proc.returncode == 0 else None


def save_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def selected_stages(from_step: str | None, until_step: str | None) -> list[str]:
    start = STAGES.index(from_step) if from_step else 0
    stop = STAGES.index(until_step) if until_step else len(STAGES) - 1
    if start > stop:
        raise ValueError(f"Invalid stage range: from={from_step} until={until_step}")
    return STAGES[start : stop + 1]


def resolve_max_concurrency() -> int:
    raw = os.environ.get("HIPP_MAX_CONCURRENCY", "").strip()
    if not raw:
        return DEFAULT_MAX_CONCURRENCY
    try:
        value = int(raw)
    except ValueError as exc:
        raise ValueError(f"Invalid HIPP_MAX_CONCURRENCY={raw!r}; expected integer >= 1") from exc
    if value < 1:
        raise ValueError(f"Invalid HIPP_MAX_CONCURRENCY={raw!r}; expected integer >= 1")
    return value


def run_stage(name: str, cmd: list[str], log_dir: Path) -> dict[str, object]:
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"{name}.log"
    t0 = time.monotonic()
    print(f"[stage] {name}", flush=True)
    with log_path.open("w", encoding="utf-8") as log:
        log.write("$ " + " ".join(cmd) + "\n")
        log.flush()
        proc = subprocess.run(cmd, cwd=REPO_ROOT, text=True, stdout=log, stderr=subprocess.STDOUT)
    elapsed = time.monotonic() - t0
    row = {
        "stage": name,
        "command": cmd,
        "log": str(log_path),
        "returncode": int(proc.returncode),
        "elapsed_seconds": float(elapsed),
    }
    if proc.returncode != 0:
        raise RuntimeError(f"Stage failed: {name}; see {log_path}")
    return row


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the full T1only Kong2022 Route B pipeline.")
    parser.add_argument("--hippunfold-batch-root", default=str(DEFAULT_BATCH_ROOT))
    parser.add_argument("--route-root", default=str(DEFAULT_ROUTE_ROOT))
    parser.add_argument("--shared-root", default=str(DEFAULT_SHARED_ROOT))
    parser.add_argument("--group-root", default=str(DEFAULT_GROUP_ROOT))
    parser.add_argument("--present-dir", default=str(DEFAULT_PRESENT_DIR))
    parser.add_argument("--jobs", type=int, default=DEFAULT_PIPELINE_JOBS)
    parser.add_argument("--from-step", choices=STAGES, default=None)
    parser.add_argument("--until-step", choices=STAGES, default=None)
    parser.add_argument("--resume", action="store_true", default=True)
    args = parser.parse_args()

    max_concurrency = resolve_max_concurrency()
    requested_jobs = int(args.jobs)
    jobs = max(1, min(requested_jobs, max_concurrency))
    if requested_jobs > max_concurrency:
        print(
            f"[warn] requested jobs={requested_jobs} exceeds max_concurrency={max_concurrency}; using jobs={jobs}",
            flush=True,
        )
    group_atlas_jobs = max(1, min(jobs, DEFAULT_GROUP_ATLAS_JOBS))
    if group_atlas_jobs != jobs:
        print(
            f"[info] group-atlas K-eval jobs capped at {group_atlas_jobs} (pipeline jobs={jobs})",
            flush=True,
        )
    batch_root = Path(args.hippunfold_batch_root).resolve()
    route_root = Path(args.route_root).resolve()
    shared_root = Path(args.shared_root).resolve()
    group_root = Path(args.group_root).resolve()
    present_dir = Path(args.present_dir).resolve()
    manifest = route_root / "manifests" / DEFAULT_MANIFEST.name
    group_avg = group_root / "group_average_surfaces"
    log_dir = route_root / "logs" / f"pipeline_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"

    stage_rows: list[dict[str, object]] = []
    for stage in selected_stages(args.from_step, args.until_step):
        if stage == "prepare":
            cmd = [
                sys.executable,
                str(PREPARE),
                "--hippunfold-batch-root",
                str(batch_root),
                "--route-root",
                str(route_root),
                "--shared-root",
                str(shared_root),
                "--jobs",
                str(jobs),
            ]
            if args.resume:
                cmd.append("--resume")
        elif stage == "kong2022-reference":
            cmd = [
                sys.executable,
                str(KONG2022_REF),
                "--shared-root",
                str(shared_root),
                "--subjects-manifest",
                str(manifest),
                "--jobs",
                str(jobs),
                "--resume",
            ]
        elif stage == "kong2022-fc":
            cmd = [
                sys.executable,
                str(KONG2022_FC),
                "--shared-root",
                str(shared_root),
                "--subjects-manifest",
                str(manifest),
                "--jobs",
                str(jobs),
            ]
        elif stage == "intrinsic-fc":
            cmd = [
                sys.executable,
                str(INTRINSIC_FC),
                "--shared-root",
                str(shared_root),
                "--subjects-manifest",
                str(manifest),
                "--jobs",
                str(jobs),
            ]
        elif stage == "group-atlas":
            cmd = [
                sys.executable,
                str(GROUP_ATLAS),
                "--shared-root",
                str(shared_root),
                "--subjects-manifest",
                str(manifest),
                "--hippunfold-batch-root",
                str(batch_root),
                "--group-average-surfaces-dir",
                str(group_avg),
                "--out-root",
                str(group_root),
                "--jobs",
                str(group_atlas_jobs),
                "--resume",
                "--coverage-thresholds",
                "0.80",
                "--atlases",
                "kong2022",
                "--methods",
                "group-intrinsic-spectral",
                "group-network-top",
                "group-network-all",
            ]
        elif stage == "present":
            cmd = [
                sys.executable,
                str(PRESENT),
                "--group-root",
                str(group_root),
                "--present-dir",
                str(present_dir),
                "--atlases",
                "kong2022",
                "--methods",
                "group-intrinsic-spectral",
                "group-network-top",
                "group-network-all",
                "--coverage-threshold",
                "0.80",
            ]
        elif stage == "cluster-roi-cortex":
            cmd = [
                sys.executable,
                str(CLUSTER_ROI_CORTEX),
                "--group-root",
                str(group_root),
                "--shared-root",
                str(shared_root),
                "--present-dir",
                str(present_dir),
                "--atlas",
                "kong2022",
                "--methods",
                "group-intrinsic-spectral",
                "group-network-top",
                "group-network-all",
                "--smoothings",
                "2mm",
                "4mm",
                "--hemis",
                "L",
                "R",
                "--coverage-threshold",
                "0.80",
                "--fdr-q",
                "0.05",
                "--max-rois-per-sign",
                "20",
            ]
        else:
            raise AssertionError(stage)
        stage_rows.append(run_stage(stage, cmd, log_dir))

    payload = {
        "stage": "t1only_kong2022_route_b_pipeline",
        "status": "done",
        "timestamp_utc": now_utc(),
        "git_commit": git_commit(),
        "hippunfold_batch_root": str(batch_root),
        "route_root": str(route_root),
        "shared_root": str(shared_root),
        "group_root": str(group_root),
        "present_dir": str(present_dir),
        "subjects_manifest": str(manifest),
        "group_average_surfaces_dir": str(group_avg),
        "jobs": int(jobs),
        "group_atlas_jobs": int(group_atlas_jobs),
        "max_concurrency": int(max_concurrency),
        "stages": stage_rows,
    }
    save_json(route_root / "pipeline_manifest.json", payload)
    print(json.dumps({"status": "ok", "manifest": str(route_root / "pipeline_manifest.json")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
