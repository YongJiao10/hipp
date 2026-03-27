#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from time import perf_counter


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
    parser = argparse.ArgumentParser(description="Sequentially rerun multiple subjects and render dense corobl figures")
    parser.add_argument("--subjects", nargs="+", required=True)
    parser.add_argument("--input-dir", default="data/hippunfold_input")
    parser.add_argument("--reference-dir", default="outputs/100610/reference")
    parser.add_argument("--out-root", default="outputs/dense_corobl_batch")
    parser.add_argument("--skip-volume-backproject", action="store_true")
    args = parser.parse_args()

    python_exe = sys.executable or "python"
    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    batch_summary: dict[str, object] = {
        "subjects": args.subjects,
        "input_dir": args.input_dir,
        "reference_dir": args.reference_dir,
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
                    "scripts/run_hippunfold_local.sh",
                    subject,
                    args.input_dir,
                    str(hippunfold_dir),
                ],
                "hippunfold",
                timings,
            )

            post_cmd = [
                python_exe,
                "scripts/run_post_hippunfold_pipeline.py",
                "--subject",
                subject,
                "--bold",
                f"{args.input_dir}/sub-{subject}/func/sub-{subject}_task-rest_run-concat_bold.nii.gz",
                "--brain-mask",
                f"{args.input_dir}/sub-{subject}/func/sub-{subject}_task-rest_run-concat_desc-brain_mask.nii.gz",
                "--hippunfold-dir",
                str(hippunfold_dir),
                "--reference-dir",
                args.reference_dir,
                "--space",
                "corobl",
                "--outdir",
                str(post_dir),
            ]
            if args.skip_volume_backproject:
                post_cmd.append("--skip-volume-backproject")

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
