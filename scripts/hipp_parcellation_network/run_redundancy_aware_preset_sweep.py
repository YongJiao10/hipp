#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ROUTE_ROOT = REPO_ROOT / "outputs_experiments" / "hippunfold_recommended_t1w_20260423" / "kong2022_route_b"
DEFAULT_SHARED_ROOT = DEFAULT_ROUTE_ROOT / "hipp_group_inputs_shared"
DEFAULT_GROUP_ROOT = DEFAULT_ROUTE_ROOT / "hipp_group_merged_canonical_atlas"
DEFAULT_BATCH_ROOT = REPO_ROOT / "outputs_experiments" / "hippunfold_recommended_t1w_20260423" / "dense_corobl_batch"
DEFAULT_SUBJECTS = DEFAULT_ROUTE_ROOT / "manifests" / "hcp_7t_hippocampus_fmri_qc_pass_t1only_latest.txt"
GROUP_ATLAS_SCRIPT = REPO_ROOT / "scripts" / "hipp_parcellation_network" / "run_merged_canonical_group_atlas.py"

METHODS = ["group-network-spectral", "group-intrinsic-spectral"]
SMOOTHINGS = ["2mm", "4mm"]
HEMIS = ["L", "R"]
ATLAS = "kong2022"
TAU_TAG = "cov_0p80"
COMBOS = [(m, s, h) for m in METHODS for s in SMOOTHINGS for h in HEMIS]

PRESETS: list[dict[str, object]] = [
    {
        "name": "baseline",
        "feature_denoise_mode": "none",
        "feature_pca_var": 0.95,
        "feature_pca_max_components": 128,
        "spatial_smooth_iters": 0,
        "spatial_smooth_alpha": 0.0,
    },
    {
        "name": "pca95",
        "feature_denoise_mode": "pca",
        "feature_pca_var": 0.95,
        "feature_pca_max_components": 128,
        "spatial_smooth_iters": 0,
        "spatial_smooth_alpha": 0.0,
    },
    {
        "name": "pca95_smooth2",
        "feature_denoise_mode": "pca",
        "feature_pca_var": 0.95,
        "feature_pca_max_components": 128,
        "spatial_smooth_iters": 2,
        "spatial_smooth_alpha": 0.35,
    },
    {
        "name": "pca90_smooth2",
        "feature_denoise_mode": "pca",
        "feature_pca_var": 0.90,
        "feature_pca_max_components": 128,
        "spatial_smooth_iters": 2,
        "spatial_smooth_alpha": 0.35,
    },
    {
        "name": "pca90_smooth4",
        "feature_denoise_mode": "pca",
        "feature_pca_var": 0.90,
        "feature_pca_max_components": 128,
        "spatial_smooth_iters": 4,
        "spatial_smooth_alpha": 0.35,
    },
    {
        "name": "pca85_smooth2_alpha50",
        "feature_denoise_mode": "pca",
        "feature_pca_var": 0.85,
        "feature_pca_max_components": 128,
        "spatial_smooth_iters": 2,
        "spatial_smooth_alpha": 0.50,
    },
]


def load_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def k3_summary_ok(run_dir: Path) -> tuple[bool, str]:
    manifest = load_json(run_dir / "group_atlas_manifest.json")
    if str(manifest.get("status")) != "ok":
        return False, f"status={manifest.get('status')}"
    if int(manifest.get("chosen_k_final", -1)) != 3:
        return False, f"chosen_k_final={manifest.get('chosen_k_final')}"
    k_summary = load_json(run_dir / "k_selection_summary.json")
    rows = [dict(r) for r in k_summary.get("rows", [])]
    row3 = next((r for r in rows if int(r.get("k", -1)) == 3), None)
    if row3 is None:
        return False, "missing_k3_row"
    if int(row3.get("selected_final", 0)) != 1:
        return False, "k3_not_selected_final"
    if int(row3.get("redundant_pair_count", -1)) != 0:
        return False, f"redundant_pair_count={row3.get('redundant_pair_count')}"
    return True, "ok"


def run_one_combo(
    *,
    python_bin: str,
    preset: dict[str, object],
    shared_root: Path,
    group_root: Path,
    batch_root: Path,
    subjects_manifest: Path,
    method: str,
    smoothing: str,
    hemi: str,
    jobs: int,
    instability_resamples: int,
    null_resamples: int,
) -> int:
    cmd = [
        python_bin,
        str(GROUP_ATLAS_SCRIPT),
        "--shared-root",
        str(shared_root),
        "--out-root",
        str(group_root),
        "--subjects-manifest",
        str(subjects_manifest),
        "--hippunfold-batch-root",
        str(batch_root),
        "--group-average-surfaces-dir",
        str(group_root / "group_average_surfaces"),
        "--atlases",
        ATLAS,
        "--methods",
        method,
        "--coverage-thresholds",
        "0.80",
        "--smoothings",
        smoothing,
        "--hemis",
        hemi,
        "--jobs",
        str(int(jobs)),
        "--instability-resamples",
        str(int(instability_resamples)),
        "--null-resamples",
        str(int(null_resamples)),
        "--k-selection-mode",
        "redundancy-aware",
        "--feature-denoise-mode",
        str(preset["feature_denoise_mode"]),
        "--feature-pca-var",
        str(float(preset["feature_pca_var"])),
        "--feature-pca-max-components",
        str(int(preset["feature_pca_max_components"])),
        "--spatial-smooth-iters",
        str(int(preset["spatial_smooth_iters"])),
        "--spatial-smooth-alpha",
        str(float(preset["spatial_smooth_alpha"])),
    ]
    env = dict(os.environ)
    env["PYTHONWARNINGS"] = "ignore"
    proc = subprocess.run(cmd, cwd=REPO_ROOT, env=env)
    return int(proc.returncode)


def save_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Run global redundancy-aware preset sweep for spectral K selection.")
    parser.add_argument("--python-bin", default="/home/yojiao/.conda/envs/py314/bin/python")
    parser.add_argument("--shared-root", default=str(DEFAULT_SHARED_ROOT))
    parser.add_argument("--group-root", default=str(DEFAULT_GROUP_ROOT))
    parser.add_argument("--hippunfold-batch-root", default=str(DEFAULT_BATCH_ROOT))
    parser.add_argument("--subjects-manifest", default=str(DEFAULT_SUBJECTS))
    parser.add_argument("--jobs", type=int, default=4)
    parser.add_argument("--instability-resamples", type=int, default=20)
    parser.add_argument("--null-resamples", type=int, default=20)
    args = parser.parse_args()

    jobs = max(1, min(int(args.jobs), 10))
    shared_root = Path(args.shared_root).resolve()
    group_root = Path(args.group_root).resolve()
    batch_root = Path(args.hippunfold_batch_root).resolve()
    subjects_manifest = Path(args.subjects_manifest).resolve()

    table: list[dict[str, object]] = []
    winner: str | None = None
    for preset in PRESETS:
        preset_name = str(preset["name"])
        print("[preset] " + preset_name, flush=True)
        checks: list[dict[str, object]] = []
        preset_failed = False
        run_failed = False
        for method, smoothing, hemi in COMBOS:
            print(f"[combo] {method} {smoothing} hemi-{hemi}", flush=True)
            rc = run_one_combo(
                python_bin=str(args.python_bin),
                preset=preset,
                shared_root=shared_root,
                group_root=group_root,
                batch_root=batch_root,
                subjects_manifest=subjects_manifest,
                method=method,
                smoothing=smoothing,
                hemi=hemi,
                jobs=jobs,
                instability_resamples=int(args.instability_resamples),
                null_resamples=int(args.null_resamples),
            )
            run_dir = group_root / method / ATLAS / smoothing / TAU_TAG / f"hemi_{hemi}"
            if rc != 0:
                checks.append(
                    {
                        "method": method,
                        "smoothing": smoothing,
                        "hemi": hemi,
                        "ok": False,
                        "detail": f"run_failed_rc={rc}",
                        "run_dir": str(run_dir),
                    }
                )
                run_failed = True
                break
            ok, detail = k3_summary_ok(run_dir)
            checks.append(
                {
                    "method": method,
                    "smoothing": smoothing,
                    "hemi": hemi,
                    "ok": bool(ok),
                    "detail": str(detail),
                    "run_dir": str(run_dir),
                }
            )
            if not ok:
                preset_failed = True
                break
        if run_failed:
            table.append({"preset": preset_name, "status": "run_failed", "checks": checks})
            continue
        if not preset_failed and len(checks) == len(COMBOS):
            table.append({"preset": preset_name, "status": "pass", "checks": checks})
            winner = preset_name
            break
        table.append({"preset": preset_name, "status": "fail", "checks": checks})

    payload = {
        "status": "ok" if winner is not None else "no_preset_passed",
        "winner": winner,
        "target": {
            "methods": METHODS,
            "atlas": ATLAS,
            "smoothings": SMOOTHINGS,
            "hemis": HEMIS,
            "coverage_threshold": 0.80,
            "required_k_final": 3,
            "instability_resamples": int(args.instability_resamples),
            "null_resamples": int(args.null_resamples),
        },
        "table": table,
    }
    out_path = group_root / "_redundancy_aware_preset_sweep.json"
    save_json(out_path, payload)
    print(json.dumps({"status": payload["status"], "winner": winner, "summary": str(out_path)}, indent=2))
    return 0 if winner is not None else 2


if __name__ == "__main__":
    raise SystemExit(main())
