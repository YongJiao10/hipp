#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from lib.output_layout import DEFAULT_CORTEX_ROUTEA_ROOT, DEFAULT_KONG2022_MANIFEST_ROOT, DEFAULT_KONG2022_ROUTEA_ROOT


REPO_ROOT = Path(__file__).resolve().parents[2]
PYTHON_EXE = sys.executable or "python"
EXTRACT_MODULE = "scripts.core_workflow.lib.extract_cortex_kong2022_route_a_roi_timeseries"
DEFAULT_SHARED_ROOT = DEFAULT_KONG2022_ROUTEA_ROOT
DEFAULT_ROUTE_A_ROOT = DEFAULT_CORTEX_ROUTEA_ROOT
DEFAULT_DATA_ROOT = REPO_ROOT / "data" / "hippunfold_input"
DEFAULT_MANIFEST = DEFAULT_KONG2022_MANIFEST_ROOT / "hcp_7t_hippocampus_fmri_qc_pass_t1only_latest.txt"
MAX_CONCURRENCY = 10
SOURCE_TAG = "route_a_kong2022_areal_mshbm"
FORBIDDEN_ROUTEB_TOKEN = "kong2022_route_b"


def validate_routea_path(path: Path, arg_name: str) -> None:
    if FORBIDDEN_ROUTEB_TOKEN in path.parts:
        raise RuntimeError(
            f"RouteA compliance violation: {arg_name} points into a RouteB tree: {path}. "
            "Use route-agnostic/shared kong2022 manifests and RouteA-compatible roots."
        )


def load_subjects_from_manifest(path: Path) -> list[str]:
    out: list[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        s = line.strip()
        if not s:
            continue
        out.append(s.replace("sub-", ""))
    if not out:
        raise RuntimeError(f"No subjects found in manifest: {path}")
    return out


def subject_is_done(shared_root: Path, subject: str) -> bool:
    ref_dir = shared_root / f"sub-{subject}" / "reference" / "kong2022"
    summary = ref_dir / "reference_summary.json"
    stage = ref_dir / "stage_manifest.json"
    if not summary.exists() or not stage.exists():
        return False
    try:
        payload = json.loads(summary.read_text(encoding="utf-8"))
    except Exception:
        return False
    return str(payload.get("reference_label_source", "")) == SOURCE_TAG


def run_subject(
    *,
    subject: str,
    shared_root: Path,
    route_a_root: Path,
    data_root: Path,
    kong2022_dlabel: Path | None,
    overwrite: bool,
) -> tuple[str, bool, str]:
    cmd = [
        PYTHON_EXE,
        "-m",
        EXTRACT_MODULE,
        "--subject",
        str(subject),
        "--shared-root",
        str(shared_root),
        "--route-a-root",
        str(route_a_root),
        "--dtseries",
        str(data_root / f"sub-{subject}" / "func" / f"sub-{subject}_task-rest_run-concat.dtseries.nii"),
    ]
    if kong2022_dlabel is not None:
        cmd.extend(["--kong2022-dlabel", str(kong2022_dlabel)])
    if overwrite:
        cmd.append("--overwrite")
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        tail = (proc.stderr or proc.stdout)[-4000:]
        return subject, False, tail
    return subject, True, "ok"


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Batch bridge Route A individualized labels into Route B-compatible reference/kong2022 arrays."
    )
    parser.add_argument("--shared-root", default=str(DEFAULT_SHARED_ROOT))
    parser.add_argument("--route-a-root", default=str(DEFAULT_ROUTE_A_ROOT))
    parser.add_argument("--data-root", default=str(DEFAULT_DATA_ROOT))
    parser.add_argument("--kong2022-dlabel", default=None)
    parser.add_argument("--subjects-manifest", default=str(DEFAULT_MANIFEST))
    parser.add_argument("--subjects", nargs="*", default=None, help="Optional explicit subject IDs.")
    parser.add_argument("--jobs", type=int, default=4, help="Thread-pool concurrency (hard-capped at 10).")
    parser.add_argument("--resume", action="store_true", default=True)
    parser.add_argument("--overwrite", action="store_true", default=False)
    args = parser.parse_args()

    shared_root = Path(args.shared_root).resolve()
    route_a_root = Path(args.route_a_root).resolve()
    data_root = Path(args.data_root).resolve()
    kong2022_dlabel = Path(args.kong2022_dlabel).resolve() if args.kong2022_dlabel else None
    subjects_manifest = Path(args.subjects_manifest).resolve()
    validate_routea_path(shared_root, "--shared-root")
    validate_routea_path(subjects_manifest, "--subjects-manifest")
    subjects = (
        [str(s).replace("sub-", "") for s in args.subjects]
        if args.subjects
        else load_subjects_from_manifest(subjects_manifest)
    )

    if args.resume:
        todo = [s for s in subjects if not subject_is_done(shared_root, s)]
    else:
        todo = list(subjects)

    print(f"[info] total={len(subjects)} todo={len(todo)} resume={bool(args.resume)}", flush=True)
    if not todo:
        print("[info] all subjects already done", flush=True)
        return 0

    jobs = max(1, min(int(args.jobs), MAX_CONCURRENCY, len(todo)))
    failed: list[str] = []
    with ThreadPoolExecutor(max_workers=jobs) as pool:
        futs = {
            pool.submit(
                run_subject,
                subject=s,
                shared_root=shared_root,
                route_a_root=route_a_root,
                data_root=data_root,
                kong2022_dlabel=kong2022_dlabel,
                overwrite=bool(args.overwrite),
            ): s
            for s in todo
        }
        done = 0
        for fut in as_completed(futs):
            done += 1
            subject, ok, msg = fut.result()
            if ok:
                print(f"[done {done}/{len(todo)}] sub-{subject}", flush=True)
            else:
                failed.append(subject)
                print(f"[FAIL {done}/{len(todo)}] sub-{subject}\n{msg}", flush=True)

    if failed:
        print(f"[error] failed_subjects={len(failed)} {failed}", flush=True)
        return 1
    print(f"[ok] completed={len(todo)} jobs={jobs}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
