#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
PY314 = Path("/home/yojiao/.conda/envs/py314/bin/python")
THREAD_ENV = {
    "OMP_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1",
    "VECLIB_MAXIMUM_THREADS": "1",
}


def validate_release_settings(*, run_id: str, jobs: int) -> None:
    if int(jobs) != 1:
        raise ValueError("The release runner is intentionally fixed at --jobs 1")
    if (
        len(run_id) > 128
        or run_id in {".", ".."}
        or re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", run_id) is None
    ):
        raise ValueError(f"Unsafe --run-id: {run_id!r}")


def expected_overviews(repo_root: Path) -> list[Path]:
    base = Path(repo_root) / "outputs_hcp" / "present" / "overviews"
    stem = "kong2022_gradient-spectral_group_overview_g1g2g3"
    return [
        base / f"{stem}_routea_fisherz.png",
        base / f"{stem}_routea_count.png",
        base / f"{stem}_routea_17net_fisherz.png",
        base / f"{stem}_routea_17net_count.png",
    ]


def build_stage_plan(*, repo_root: Path, python_exe: Path, jobs: int) -> list[dict[str, object]]:
    if int(jobs) != 1:
        raise ValueError("The release stage plan is fixed at one job")
    root = Path(repo_root).resolve()
    python = str(Path(python_exe))
    shared = root / "outputs_hcp/inputs/hcp/kong2022/shared_inputs/routea"
    surface = root / "outputs_hcp/inputs/hcp/kong2022/shared_inputs/surface"
    gradients = root / "outputs_hcp/work/hippomaps_gradients"
    manifest = root / "outputs_hcp/inputs/hcp/kong2022/manifests/hcp_7t_hippocampus_fmri_qc_pass_t1only_latest.txt"
    batch = root / "outputs_hcp/inputs/hcp/hippunfold/dense_corobl_batch"
    group_surfaces = root / "outputs_hcp/inputs/hcp/kong2022/templates/hipp_group_average_t1only_151/group_average_surfaces"
    present = root / "outputs_hcp/present/overviews"
    group_script = root / "scripts/core_workflow/05_02_run_group_atlas.py"
    overview_script = root / "scripts/core_workflow/06_01_build_group_overview.py"

    def group_command(out_root: Path, route_tag: str, taxonomy: str) -> list[str]:
        return [
            python,
            str(group_script),
            "--shared-root", str(shared),
            "--surface-root", str(surface),
            "--reference-root", str(shared),
            "--gradient-root", str(gradients),
            "--out-root", str(out_root),
            "--route-tag", route_tag,
            "--network-taxonomy", taxonomy,
            "--subjects-manifest", str(manifest),
            "--hippunfold-batch-root", str(batch),
            "--group-average-surfaces-dir", str(group_surfaces),
            "--methods", "gradient-spectral",
            "--atlases", "kong2022",
            "--smoothings", "2mm", "4mm",
            "--hemis", "L", "R",
            "--coverage-thresholds", "0.80",
            "--k-min", "2", "--k-max", "8", "--default-k", "3",
            "--jobs", str(int(jobs)),
            "--resume",
        ]

    def overview_command(group_root: Path, route_tag: str) -> list[str]:
        return [
            python,
            str(overview_script),
            "--group-root", str(group_root),
            "--present-dir", str(present),
            "--route-tag", route_tag,
            "--methods", "gradient-spectral",
            "--atlases", "kong2022",
            "--assignment-variant", "both",
        ]

    routea_root = root / "outputs_hcp/work/group_atlas/routea_full_k3_fisherz"
    family17_root = root / "outputs_hcp/work/group_atlas/routea_17net_full_k3_fisherz"
    return [
        {"name": "routea_canonical8_group", "command": group_command(routea_root, "routea", "canonical8")},
        {"name": "routea_canonical8_overviews", "command": overview_command(routea_root, "routea")},
        {"name": "routea_family17_group", "command": group_command(family17_root, "routea_17net", "family17")},
        {"name": "routea_family17_overviews", "command": overview_command(family17_root, "routea_17net")},
    ]


def _run_paths(repo_root: Path, run_id: str) -> dict[str, Path]:
    validate_release_settings(run_id=run_id, jobs=1)
    prefix = Path(repo_root) / "outputs_hcp" / "work" / "group_atlas" / "logs" / run_id
    return {
        "status": prefix.with_suffix(".status.json"),
        "stdout": prefix.with_suffix(".stdout.log"),
        "stderr": prefix.with_suffix(".stderr.log"),
        "pid": prefix.with_suffix(".pid"),
    }


def _write_status(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def preflight(repo_root: Path, stages: list[dict[str, object]]) -> None:
    root = Path(repo_root).resolve()
    required = [
        PY314,
        root / "scripts/core_workflow/05_02_run_group_atlas.py",
        root / "scripts/core_workflow/06_01_build_group_overview.py",
        root / "outputs_hcp/inputs/hcp/kong2022/shared_inputs/routea",
        root / "outputs_hcp/inputs/hcp/kong2022/shared_inputs/surface",
        root / "outputs_hcp/work/hippomaps_gradients/routea",
        root / "outputs_hcp/inputs/hcp/kong2022/manifests/hcp_7t_hippocampus_fmri_qc_pass_t1only_latest.txt",
        root / "outputs_hcp/inputs/hcp/kong2022/templates/hipp_group_average_t1only_151/group_average_surfaces",
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing RouteA release inputs:\n" + "\n".join(missing))
    if len(stages) != 4:
        raise ValueError(f"Expected exactly four release stages, got {len(stages)}")


def validate_release_outputs(repo_root: Path) -> dict[str, object]:
    root = Path(repo_root).resolve()
    manifests: list[Path] = []
    for branch in ("routea_full_k3_fisherz", "routea_17net_full_k3_fisherz"):
        branch_root = root / "outputs_hcp/work/group_atlas" / branch / "gradient-spectral/kong2022"
        for smoothing in ("2mm", "4mm"):
            for hemi in ("L", "R"):
                manifest_path = branch_root / smoothing / "cov_0p80/g1g2g3" / f"hemi_{hemi}" / "group_atlas_manifest.json"
                payload = json.loads(manifest_path.read_text(encoding="utf-8"))
                if payload.get("surface_affinity") != "shifted-cosine":
                    raise ValueError(f"Unexpected affinity in {manifest_path}")
                if payload.get("candidate_k_range_requested") != [2, 8]:
                    raise ValueError(f"Unexpected K range in {manifest_path}")
                if int(payload.get("chosen_k_final", -1)) != 3:
                    raise ValueError(f"Unexpected final K in {manifest_path}")
                variants = payload.get("assignment_variants", {})
                if not {"fisherz", "count"}.issubset(variants):
                    raise ValueError(f"Missing assignment variants in {manifest_path}")
                graph = payload.get("affinity_graph_diagnostics", {})
                if int(graph.get("n_connected_components", -1)) != 1:
                    raise ValueError(f"Disconnected affinity graph in {manifest_path}")
                manifests.append(manifest_path)
    overviews = expected_overviews(root)
    missing_overviews = [str(path) for path in overviews if not path.is_file() or path.stat().st_size <= 0]
    if missing_overviews:
        raise FileNotFoundError("Missing RouteA release overviews:\n" + "\n".join(missing_overviews))
    return {"group_manifests": len(manifests), "overviews": [str(path) for path in overviews]}


def build_worker_command(
    *,
    script_path: Path,
    repo_root: Path,
    run_id: str,
    jobs: int,
) -> list[str]:
    return [
        str(PY314),
        str(Path(script_path).resolve()),
        "--worker",
        "--repo-root", str(Path(repo_root).resolve()),
        "--run-id", run_id,
        "--jobs", str(int(jobs)),
    ]


def run_worker(*, repo_root: Path, run_id: str, jobs: int) -> int:
    validate_release_settings(run_id=run_id, jobs=jobs)
    root = Path(repo_root).resolve()
    paths = _run_paths(root, run_id)
    stages = build_stage_plan(repo_root=root, python_exe=PY314, jobs=jobs)
    env = dict(os.environ)
    env.update(THREAD_ENV)
    completed: list[str] = []
    base_status: dict[str, object] = {
        "run_id": run_id,
        "pid": os.getpid(),
        "jobs": int(jobs),
        "stages": stages,
        "expected_overviews": [str(path) for path in expected_overviews(root)],
    }
    try:
        preflight(root, stages)
        _write_status(paths["status"], {**base_status, "status": "running", "completed_stages": completed})
        for stage in stages:
            subprocess.run(stage["command"], cwd=root, env=env, check=True)
            completed.append(str(stage["name"]))
            _write_status(paths["status"], {**base_status, "status": "running", "completed_stages": completed})
        evidence = validate_release_outputs(root)
        _write_status(
            paths["status"],
            {
                **base_status,
                "status": "ok",
                "completed_stages": completed,
                "completed_utc": datetime.now(timezone.utc).isoformat(),
                "evidence": evidence,
            },
        )
        return 0
    except Exception as exc:
        _write_status(
            paths["status"],
            {
                **base_status,
                "status": "failed",
                "completed_stages": completed,
                "failed_utc": datetime.now(timezone.utc).isoformat(),
                "error": f"{type(exc).__name__}: {exc}",
            },
        )
        raise


def launch(*, repo_root: Path, run_id: str, jobs: int) -> dict[str, object]:
    validate_release_settings(run_id=run_id, jobs=jobs)
    root = Path(repo_root).resolve()
    paths = _run_paths(root, run_id)
    stages = build_stage_plan(repo_root=root, python_exe=PY314, jobs=jobs)
    preflight(root, stages)
    collisions = [str(path) for path in paths.values() if path.exists()]
    if collisions:
        raise FileExistsError("Run record collision:\n" + "\n".join(collisions))
    _write_status(
        paths["status"],
        {"status": "launching", "run_id": run_id, "jobs": int(jobs), "stages": stages},
    )
    env = dict(os.environ)
    env.update(THREAD_ENV)
    command = build_worker_command(
        script_path=Path(__file__),
        repo_root=root,
        run_id=run_id,
        jobs=jobs,
    )
    with paths["stdout"].open("w", encoding="utf-8") as stdout, paths["stderr"].open("w", encoding="utf-8") as stderr:
        process = subprocess.Popen(
            command,
            cwd=root,
            env=env,
            stdin=subprocess.DEVNULL,
            stdout=stdout,
            stderr=stderr,
            start_new_session=True,
        )
    paths["pid"].write_text(f"{process.pid}\n", encoding="utf-8")
    return {
        "status": "launched",
        "run_id": run_id,
        "pid": process.pid,
        **{key: str(path) for key, path in paths.items()},
    }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--launch", action="store_true")
    mode.add_argument("--worker", action="store_true")
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--jobs", type=int, default=1)
    return parser


def main() -> int:
    args = build_arg_parser().parse_args()
    run_id = args.run_id or f"routea_gradient_spectral_release_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
    validate_release_settings(run_id=run_id, jobs=int(args.jobs))
    if args.worker:
        return run_worker(repo_root=args.repo_root, run_id=run_id, jobs=int(args.jobs))
    result = launch(repo_root=args.repo_root, run_id=run_id, jobs=int(args.jobs))
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
