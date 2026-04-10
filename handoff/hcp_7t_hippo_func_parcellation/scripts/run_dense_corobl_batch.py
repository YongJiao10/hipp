#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from time import perf_counter

REPO_ROOT = Path(__file__).resolve().parents[1]
COMMON_DIR = REPO_ROOT / "scripts" / "common"
if str(COMMON_DIR) not in sys.path:
    sys.path.insert(0, str(COMMON_DIR))

from hipp_density_assets import load_surface_density_from_pipeline_config


def run(cmd: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(cmd, text=True, capture_output=True)


def run_step(cmd: list[str], step: str, timings: list[dict[str, object]]) -> None:
    start = perf_counter()
    proc = run(cmd)
    duration = perf_counter() - start
    timings.append(
        {
            "step": step,
            "command": cmd,
            "returncode": int(proc.returncode),
            "duration_seconds": duration,
            "status": "success" if proc.returncode == 0 else "failed",
            "stdout": proc.stdout[-4000:],
            "stderr": proc.stderr[-4000:],
        }
    )
    if proc.returncode != 0:
        raise RuntimeError(
            f"Step failed ({step}, rc={proc.returncode}): {' '.join(cmd)}\nSTDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}"
        )


def main() -> int:
    default_density = load_surface_density_from_pipeline_config(REPO_ROOT / "config" / "hippo_pipeline.toml")
    parser = argparse.ArgumentParser(description="Sequentially rerun multiple subjects and render dense corobl figures")
    parser.add_argument("--subjects", nargs="+", required=True)
    parser.add_argument("--input-dir", default=str(REPO_ROOT / "data" / "hippunfold_input"))
    parser.add_argument("--out-root", default=str(REPO_ROOT / "outputs" / "dense_corobl_batch"))
    parser.add_argument("--density", default=default_density)
    args = parser.parse_args()

    python_exe = sys.executable or "python"
    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    batch_summary: dict[str, object] = {
        "subjects": args.subjects,
        "input_dir": args.input_dir,
        "out_root": str(out_root),
        "results": [],
    }

    for subject in args.subjects:
        subject_root = out_root / f"sub-{subject}"
        hippunfold_dir = subject_root / "hippunfold"
        post_dir = subject_root / "post_dense_corobl"
        subject_root.mkdir(parents=True, exist_ok=True)

        timings: list[dict[str, object]] = []
        result: dict[str, object] = {
            "subject": subject,
            "hippunfold_dir": str(hippunfold_dir),
            "post_dir": str(post_dir),
            "timings": timings,
        }

        try:
            run_step(
                [
                    "zsh",
                    str(REPO_ROOT / "scripts" / "run_hippunfold_local.sh"),
                    subject,
                    args.input_dir,
                    str(hippunfold_dir),
                ],
                "hippunfold",
                timings,
            )

            post_cmd = [
                python_exe,
                str(REPO_ROOT / "scripts" / "run_post_hippunfold_pipeline.py"),
                "--subject",
                subject,
                "--dtseries",
                f"{args.input_dir}/sub-{subject}/func/sub-{subject}_task-rest_run-concat.dtseries.nii",
                "--bold",
                f"{args.input_dir}/sub-{subject}/func/sub-{subject}_task-rest_run-concat_bold.nii.gz",
                "--brain-mask",
                f"{args.input_dir}/sub-{subject}/func/sub-{subject}_task-rest_run-concat_desc-brain_mask.nii.gz",
                "--hippunfold-dir",
                str(hippunfold_dir),
                "--space",
                "corobl",
                "--density",
                args.density,
                "--outdir",
                str(post_dir),
            ]
            run_step(post_cmd, "post_dense_corobl", timings)
            result["status"] = "success"
        except Exception as exc:
            result["status"] = "failed"
            result["error"] = str(exc)
        finally:
            (subject_root / "batch_subject_summary.json").write_text(
                json.dumps(result, indent=2), encoding="utf-8"
            )
            batch_summary["results"].append(result)

    (out_root / "batch_summary.json").write_text(json.dumps(batch_summary, indent=2), encoding="utf-8")
    print(json.dumps(batch_summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
