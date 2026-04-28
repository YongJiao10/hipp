#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
GROUP_ATLAS = REPO_ROOT / "scripts" / "hipp_parcellation_network" / "run_merged_canonical_group_atlas.py"
PRESENT = REPO_ROOT / "scripts" / "hipp_parcellation_network" / "build_present_group_overview_merged_canonical.py"

DEFAULT_BATCH_ROOT = REPO_ROOT / "outputs_experiments" / "hippunfold_recommended_t1w_20260423" / "dense_corobl_batch"
DEFAULT_ROUTE_ROOT = REPO_ROOT / "outputs_experiments" / "hippunfold_recommended_t1w_20260423" / "kong2022_route_b"
DEFAULT_SHARED_ROOT = DEFAULT_ROUTE_ROOT / "hipp_group_inputs_shared"
DEFAULT_GROUP_ROOT = DEFAULT_ROUTE_ROOT / "hipp_group_merged_canonical_atlas"
DEFAULT_PRESENT_DIR = REPO_ROOT / "present_group_t1only"
DEFAULT_MANIFEST = DEFAULT_ROUTE_ROOT / "manifests" / "hcp_7t_hippocampus_fmri_qc_pass_t1only_latest.txt"
DEFAULT_GROUP_AVG = DEFAULT_GROUP_ROOT / "group_average_surfaces"

MAX_REPO_CONCURRENCY = 10
DEFAULT_COMBO_JOBS = 4
PRESENT_PYTHON = "/home/yojiao/.conda/envs/py314/bin/python"
METHOD_ORDER = ["group-intrinsic-spectral", "group-network-spectral"]
SMOOTHINGS = ["2mm", "4mm"]
HEMIS = ["L", "R"]
THREAD_ENV_OVERRIDES = {
    "OMP_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1",
}


@dataclass(frozen=True)
class Combo:
    method: str
    smoothing: str
    hemi: str


def build_group_atlas_cmd(
    *,
    python_exe: str,
    combo: Combo,
    shared_root: Path,
    group_root: Path,
    manifest: Path,
    batch_root: Path,
    group_avg: Path,
    coverage_threshold: float,
    subject_fc_top_fraction: float,
    subject_fc_binarize: bool,
) -> list[str]:
    binarize_flag = "--subject-fc-binarize" if bool(subject_fc_binarize) else "--no-subject-fc-binarize"
    return [
        python_exe,
        str(GROUP_ATLAS),
        "--shared-root",
        str(shared_root),
        "--out-root",
        str(group_root),
        "--subjects-manifest",
        str(manifest),
        "--hippunfold-batch-root",
        str(batch_root),
        "--group-average-surfaces-dir",
        str(group_avg),
        "--jobs",
        "1",
        "--coverage-thresholds",
        f"{coverage_threshold:.2f}",
        "--k-selection-mode",
        "redundancy-aware",
        "--redundancy-early-stop",
        "--subject-fc-top-fraction",
        f"{float(subject_fc_top_fraction):.6g}",
        binarize_flag,
        "--atlases",
        "kong2022",
        "--methods",
        combo.method,
        "--smoothings",
        combo.smoothing,
        "--hemis",
        combo.hemi,
    ]


def build_present_cmd(
    *,
    python_exe: str,
    group_root: Path,
    present_dir: Path,
    method: str,
    coverage_threshold: float,
) -> list[str]:
    return [
        PRESENT_PYTHON,
        str(PRESENT),
        "--group-root",
        str(group_root),
        "--present-dir",
        str(present_dir),
        "--atlases",
        "kong2022",
        "--methods",
        method,
        "--coverage-threshold",
        f"{coverage_threshold:.2f}",
    ]


def run_method_combo_parallel(
    *,
    python_exe: str,
    method: str,
    shared_root: Path,
    group_root: Path,
    manifest: Path,
    batch_root: Path,
    group_avg: Path,
    present_dir: Path,
    combo_jobs: int,
    coverage_threshold: float,
    subject_fc_top_fraction: float,
    subject_fc_binarize: bool,
    log_dir: Path,
) -> None:
    combos = [Combo(method=method, smoothing=smoothing, hemi=hemi) for smoothing in SMOOTHINGS for hemi in HEMIS]
    child_env = os.environ.copy()
    child_env.update(THREAD_ENV_OVERRIDES)
    procs: list[tuple[Combo, subprocess.Popen[bytes], Path]] = []
    for combo in combos:
        combo_tag = f"{method}_{combo.smoothing}_{combo.hemi}"
        log_path = log_dir / f"{combo_tag}.log"
        cmd = build_group_atlas_cmd(
            python_exe=python_exe,
            combo=combo,
            shared_root=shared_root,
            group_root=group_root,
            manifest=manifest,
            batch_root=batch_root,
            group_avg=group_avg,
            coverage_threshold=coverage_threshold,
            subject_fc_top_fraction=subject_fc_top_fraction,
            subject_fc_binarize=subject_fc_binarize,
        )
        with log_path.open("wb") as fout:
            fout.write(("CMD: " + " ".join(cmd) + "\n\n").encode("utf-8"))
        fout = log_path.open("ab")
        proc = subprocess.Popen(cmd, stdout=fout, stderr=subprocess.STDOUT, cwd=str(REPO_ROOT), env=child_env)
        procs.append((combo, proc, log_path))
        print(f"[launch] {method} {combo.smoothing} {combo.hemi} pid={proc.pid} log={log_path}", flush=True)
        if len(procs) >= combo_jobs:
            break

    pending = combos[len(procs):]
    while procs:
        still_running: list[tuple[Combo, subprocess.Popen[bytes], Path]] = []
        for combo, proc, log_path in procs:
            rc = proc.poll()
            if rc is None:
                still_running.append((combo, proc, log_path))
                continue
            print(f"[done] {method} {combo.smoothing} {combo.hemi} rc={rc}", flush=True)
            if rc != 0:
                raise RuntimeError(f"Combo failed: {method} {combo.smoothing} {combo.hemi}; see {log_path}")
            if pending:
                next_combo = pending.pop(0)
                next_tag = f"{method}_{next_combo.smoothing}_{next_combo.hemi}"
                next_log = log_dir / f"{next_tag}.log"
                next_cmd = build_group_atlas_cmd(
                    python_exe=python_exe,
                    combo=next_combo,
                    shared_root=shared_root,
                    group_root=group_root,
                    manifest=manifest,
                    batch_root=batch_root,
                    group_avg=group_avg,
                    coverage_threshold=coverage_threshold,
                    subject_fc_top_fraction=subject_fc_top_fraction,
                    subject_fc_binarize=subject_fc_binarize,
                )
                with next_log.open("wb") as fout:
                    fout.write(("CMD: " + " ".join(next_cmd) + "\n\n").encode("utf-8"))
                fout = next_log.open("ab")
                next_proc = subprocess.Popen(next_cmd, stdout=fout, stderr=subprocess.STDOUT, cwd=str(REPO_ROOT), env=child_env)
                still_running.append((next_combo, next_proc, next_log))
                print(
                    f"[launch] {method} {next_combo.smoothing} {next_combo.hemi} pid={next_proc.pid} log={next_log}",
                    flush=True,
                )
        procs = still_running
        if procs:
            time.sleep(5.0)

    present_cmd = build_present_cmd(
        python_exe=python_exe,
        group_root=group_root,
        present_dir=present_dir,
        method=method,
        coverage_threshold=coverage_threshold,
    )
    print(f"[present] {method}", flush=True)
    subprocess.run(present_cmd, check=True, cwd=str(REPO_ROOT), env=child_env)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Canonical fast rerun entrypoint for T1only spectral methods: intrinsic first, then network, with 2mm/4mm x L/R combo parallelism."
    )
    parser.add_argument("--hippunfold-batch-root", default=str(DEFAULT_BATCH_ROOT))
    parser.add_argument("--shared-root", default=str(DEFAULT_SHARED_ROOT))
    parser.add_argument("--group-root", default=str(DEFAULT_GROUP_ROOT))
    parser.add_argument("--subjects-manifest", default=str(DEFAULT_MANIFEST))
    parser.add_argument("--group-average-surfaces-dir", default=str(DEFAULT_GROUP_AVG))
    parser.add_argument("--present-dir", default=str(DEFAULT_PRESENT_DIR))
    parser.add_argument("--coverage-threshold", type=float, default=0.80)
    parser.add_argument("--subject-fc-top-fraction", type=float, default=0.10)
    parser.add_argument("--subject-fc-binarize", dest="subject_fc_binarize", action="store_true")
    parser.add_argument("--no-subject-fc-binarize", dest="subject_fc_binarize", action="store_false")
    parser.set_defaults(subject_fc_binarize=True)
    parser.add_argument("--combo-jobs", type=int, default=DEFAULT_COMBO_JOBS)
    args = parser.parse_args()

    combo_jobs = max(1, min(int(args.combo_jobs), len(SMOOTHINGS) * len(HEMIS), MAX_REPO_CONCURRENCY))
    if not (0.0 <= float(args.subject_fc_top_fraction) <= 1.0):
        raise ValueError(
            f"--subject-fc-top-fraction must be within [0,1], got {args.subject_fc_top_fraction}"
        )
    batch_root = Path(args.hippunfold_batch_root).resolve()
    shared_root = Path(args.shared_root).resolve()
    group_root = Path(args.group_root).resolve()
    manifest = Path(args.subjects_manifest).resolve()
    group_avg = Path(args.group_average_surfaces_dir).resolve()
    present_dir = Path(args.present_dir).resolve()
    route_root = manifest.parent.parent
    log_dir = route_root / "logs" / f"spectral_fast_rerun_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
    log_dir.mkdir(parents=True, exist_ok=True)

    print(
        "[config] "
        f"combo_jobs={combo_jobs} "
        f"method_order={METHOD_ORDER} "
        f"coverage_threshold={float(args.coverage_threshold):.2f} "
        f"subject_fc_top_fraction={float(args.subject_fc_top_fraction):.6g} "
        f"subject_fc_binarize={bool(args.subject_fc_binarize)} "
        f"thread_env={THREAD_ENV_OVERRIDES}",
        flush=True,
    )
    for method in METHOD_ORDER:
        print(f"[method-start] {method}", flush=True)
        run_method_combo_parallel(
            python_exe=sys.executable,
            method=method,
            shared_root=shared_root,
            group_root=group_root,
            manifest=manifest,
            batch_root=batch_root,
            group_avg=group_avg,
            present_dir=present_dir,
            combo_jobs=combo_jobs,
            coverage_threshold=float(args.coverage_threshold),
            subject_fc_top_fraction=float(args.subject_fc_top_fraction),
            subject_fc_binarize=bool(args.subject_fc_binarize),
            log_dir=log_dir,
        )
        print(f"[method-done] {method}", flush=True)
    print(f"[done] logs={log_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
