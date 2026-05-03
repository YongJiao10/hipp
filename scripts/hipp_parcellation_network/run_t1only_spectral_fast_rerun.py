#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
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
DEFAULT_ROUTE_ROOT = REPO_ROOT / "outputs_experiments" / "hippunfold_recommended_t1w_20260423" / "kong2022"
DEFAULT_SHARED_ROOT = DEFAULT_ROUTE_ROOT / "shared_inputs" / "routeb_151"
DEFAULT_SURFACE_ROOT = DEFAULT_ROUTE_ROOT / "shared_inputs" / "surface_151"
DEFAULT_GROUP_ROOT = DEFAULT_ROUTE_ROOT / "hipp_group_merged_canonical_atlas"
DEFAULT_PRESENT_DIR = REPO_ROOT / "present_group_t1only"
DEFAULT_MANIFEST = (
    REPO_ROOT
    / "outputs_experiments"
    / "hippunfold_recommended_t1w_20260423"
    / "kong2022"
    / "manifests"
    / "hcp_7t_hippocampus_fmri_qc_pass_t1only_latest.txt"
)
DEFAULT_GROUP_AVG = DEFAULT_ROUTE_ROOT / "templates" / "hipp_group_average_t1only_151" / "group_average_surfaces"

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
FORBIDDEN_ROUTEB_TOKEN = "kong2022_route_b"


@dataclass(frozen=True)
class Combo:
    method: str
    smoothing: str
    hemi: str


def load_subjects(path: Path) -> list[str]:
    out: list[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        s = line.strip()
        if not s:
            continue
        out.append(s.replace("sub-", ""))
    if not out:
        raise RuntimeError(f"No subjects found in manifest: {path}")
    return out


def detect_route_source(shared_root: Path, manifest: Path) -> str:
    subjects = load_subjects(manifest)
    tags: set[str] = set()
    for sid in subjects:
        ref = shared_root / f"sub-{sid}" / "reference" / "kong2022" / "reference_summary.json"
        if not ref.exists():
            raise FileNotFoundError(f"Missing reference summary for source detection: {ref}")
        payload = json.loads(ref.read_text(encoding="utf-8"))
        src = str(payload.get("reference_label_source", "")).lower()
        tags.add("routea" if "route_a" in src else "routeb")
    if len(tags) != 1:
        raise RuntimeError(f"Mixed route sources detected in shared_root={shared_root}: {sorted(tags)}")
    return next(iter(tags))


def with_tokens(path: Path, route_tag: str, mode_tag: str) -> Path:
    name = path.name
    for token in (route_tag, mode_tag):
        if token not in name.lower():
            name = f"{name}_{token}"
    return path.with_name(name)


def enforce_provenance_stamp(root: Path, *, route_tag: str, mode_tag: str, kind: str) -> None:
    root.mkdir(parents=True, exist_ok=True)
    stamp = root / "route_isolation_manifest.json"
    payload = {"kind": kind, "route_source": route_tag, "mode": mode_tag}
    if stamp.exists():
        old = json.loads(stamp.read_text(encoding="utf-8"))
        if str(old.get("route_source")) != route_tag or str(old.get("mode")) != mode_tag:
            raise RuntimeError(
                f"{kind} provenance mismatch at {root}: existing={old}, requested={payload}. "
                "Refusing cross-route or cross-mode reuse."
            )
    stamp.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def enforce_routea_path_compliance(shared_root: Path, manifest: Path, group_root_raw: Path) -> None:
    violations: list[str] = []
    for name, path in [
        ("--shared-root", shared_root),
        ("--subjects-manifest", manifest),
        ("--group-root", group_root_raw),
    ]:
        if FORBIDDEN_ROUTEB_TOKEN in path.parts:
            violations.append(f"{name}={path}")
    if violations:
        raise RuntimeError(
            "RouteA compliance violation: routea mode forbids RouteB-rooted paths for shared-root/manifest/group-root.\n"
            + "\n".join(violations)
        )


def build_group_atlas_cmd(
    *,
    python_exe: str,
    combo: Combo,
    shared_root: Path,
    surface_root: Path,
    group_root: Path,
    manifest: Path,
    batch_root: Path,
    group_avg: Path,
    coverage_threshold: float,
    subject_fc_top_fraction: float,
    subject_fc_binarize: bool,
    redundancy_early_stop: bool,
) -> list[str]:
    binarize_flag = "--subject-fc-binarize" if bool(subject_fc_binarize) else "--no-subject-fc-binarize"
    cmd = [
        python_exe,
        str(GROUP_ATLAS),
        "--shared-root",
        str(shared_root),
        "--surface-root",
        str(surface_root),
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
    if bool(redundancy_early_stop):
        cmd.append("--redundancy-early-stop")
    return cmd


def build_present_cmd(
    *,
    python_exe: str,
    group_root: Path,
    present_dir: Path,
    method: str,
    coverage_threshold: float,
    route_tag: str,
    mode_tag: str,
) -> list[str]:
    manifest_name = f"group_overview_manifest_{route_tag}_{mode_tag}_{method}.json"
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
        "--route-tag",
        route_tag,
        "--mode-tag",
        mode_tag,
        "--manifest-name",
        manifest_name,
        "--skip-readme",
    ]


def run_method_combo_parallel(
    *,
    python_exe: str,
    method: str,
    shared_root: Path,
    surface_root: Path,
    group_root: Path,
    manifest: Path,
    batch_root: Path,
    group_avg: Path,
    present_dir: Path,
    combo_jobs: int,
    coverage_threshold: float,
    subject_fc_top_fraction: float,
    subject_fc_binarize: bool,
    redundancy_early_stop: bool,
    log_dir: Path,
    route_tag: str,
    mode_tag: str,
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
            surface_root=surface_root,
            group_root=group_root,
            manifest=manifest,
            batch_root=batch_root,
            group_avg=group_avg,
            coverage_threshold=coverage_threshold,
            subject_fc_top_fraction=subject_fc_top_fraction,
            subject_fc_binarize=subject_fc_binarize,
            redundancy_early_stop=redundancy_early_stop,
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
                    surface_root=surface_root,
                    group_root=group_root,
                    manifest=manifest,
                    batch_root=batch_root,
                    group_avg=group_avg,
                    coverage_threshold=coverage_threshold,
                    subject_fc_top_fraction=subject_fc_top_fraction,
                    subject_fc_binarize=subject_fc_binarize,
                    redundancy_early_stop=redundancy_early_stop,
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
        route_tag=route_tag,
        mode_tag=mode_tag,
    )
    print(f"[present] {method}", flush=True)
    subprocess.run(present_cmd, check=True, cwd=str(REPO_ROOT), env=child_env)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Canonical fast rerun entrypoint for T1only spectral methods: intrinsic first, then network, with 2mm/4mm x L/R combo parallelism."
    )
    parser.add_argument("--hippunfold-batch-root", default=str(DEFAULT_BATCH_ROOT))
    parser.add_argument("--shared-root", default=str(DEFAULT_SHARED_ROOT))
    parser.add_argument("--surface-root", default=str(DEFAULT_SURFACE_ROOT))
    parser.add_argument("--group-root", default=str(DEFAULT_GROUP_ROOT))
    parser.add_argument("--subjects-manifest", default=str(DEFAULT_MANIFEST))
    parser.add_argument("--group-average-surfaces-dir", default=str(DEFAULT_GROUP_AVG))
    parser.add_argument("--present-dir", default=str(DEFAULT_PRESENT_DIR))
    parser.add_argument("--coverage-threshold", type=float, default=0.80)
    parser.add_argument("--subject-fc-top-fraction", type=float, default=0.10)
    parser.add_argument("--subject-fc-binarize", dest="subject_fc_binarize", action="store_true")
    parser.add_argument("--no-subject-fc-binarize", dest="subject_fc_binarize", action="store_false")
    parser.set_defaults(subject_fc_binarize=True)
    parser.add_argument(
        "--redundancy-early-stop",
        dest="redundancy_early_stop",
        action="store_true",
        help="Speed mode: stop larger K once first redundant K is observed (default: enabled).",
    )
    parser.add_argument(
        "--no-redundancy-early-stop",
        dest="redundancy_early_stop",
        action="store_false",
        help="Full mode: evaluate all candidate K values without early stopping.",
    )
    parser.set_defaults(redundancy_early_stop=True)
    parser.add_argument("--combo-jobs", type=int, default=DEFAULT_COMBO_JOBS)
    parser.add_argument(
        "--route-source",
        choices=["auto", "routea", "routeb"],
        default="auto",
        help="Expected upstream source. `auto` detects from shared_root reference summaries.",
    )
    args = parser.parse_args()

    combo_jobs = max(1, min(int(args.combo_jobs), len(SMOOTHINGS) * len(HEMIS), MAX_REPO_CONCURRENCY))
    if not (0.0 <= float(args.subject_fc_top_fraction) <= 1.0):
        raise ValueError(
            f"--subject-fc-top-fraction must be within [0,1], got {args.subject_fc_top_fraction}"
        )
    batch_root = Path(args.hippunfold_batch_root).resolve()
    shared_root = Path(args.shared_root).resolve()
    surface_root = Path(args.surface_root).resolve()
    manifest = Path(args.subjects_manifest).resolve()
    group_root_raw = Path(args.group_root).resolve()
    route_tag_detected = detect_route_source(shared_root, manifest)
    route_tag = route_tag_detected if args.route_source == "auto" else str(args.route_source)
    if str(args.route_source) != "auto" and route_tag != route_tag_detected:
        raise RuntimeError(
            f"Requested --route-source={args.route_source} but detected upstream source={route_tag_detected}. "
            "Refusing to run with mismatched route provenance."
        )
    if route_tag == "routea":
        enforce_routea_path_compliance(shared_root, manifest, group_root_raw)
    mode_tag = "allon" if bool(args.subject_fc_binarize) else "alloff"
    group_root = with_tokens(group_root_raw, route_tag, mode_tag)
    present_dir = Path(args.present_dir).resolve()
    group_avg = Path(args.group_average_surfaces_dir).resolve()
    enforce_provenance_stamp(group_root, route_tag=route_tag, mode_tag=mode_tag, kind="group_root")
    present_dir.mkdir(parents=True, exist_ok=True)
    route_root = manifest.parent.parent
    log_dir = route_root / "logs" / (
        f"spectral_fast_rerun_{route_tag}_{mode_tag}_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
    )
    log_dir.mkdir(parents=True, exist_ok=True)

    print(
        "[config] "
        f"combo_jobs={combo_jobs} "
        f"method_order={METHOD_ORDER} "
        f"coverage_threshold={float(args.coverage_threshold):.2f} "
        f"subject_fc_top_fraction={float(args.subject_fc_top_fraction):.6g} "
        f"subject_fc_binarize={bool(args.subject_fc_binarize)} "
        f"redundancy_early_stop={bool(args.redundancy_early_stop)} "
        f"route_source={route_tag} "
        f"group_root={group_root} "
        f"present_dir={present_dir} "
        f"thread_env={THREAD_ENV_OVERRIDES}",
        flush=True,
    )
    for method in METHOD_ORDER:
        print(f"[method-start] {method}", flush=True)
        run_method_combo_parallel(
            python_exe=sys.executable,
            method=method,
            shared_root=shared_root,
            surface_root=surface_root,
            group_root=group_root,
            manifest=manifest,
            batch_root=batch_root,
            group_avg=group_avg,
            present_dir=present_dir,
            combo_jobs=combo_jobs,
            coverage_threshold=float(args.coverage_threshold),
            subject_fc_top_fraction=float(args.subject_fc_top_fraction),
            subject_fc_binarize=bool(args.subject_fc_binarize),
            redundancy_early_stop=bool(args.redundancy_early_stop),
            log_dir=log_dir,
            route_tag=route_tag,
            mode_tag=mode_tag,
        )
        print(f"[method-done] {method}", flush=True)
    print(f"[done] logs={log_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
