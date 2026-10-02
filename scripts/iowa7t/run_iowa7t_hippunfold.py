#!/usr/bin/env python3
"""Run HippUnfold for Iowa7T subjects and run 01_02 cleanup upon completion."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.core_workflow.lib.hippunfold_run_contract import (
    CANONICAL_CACHE_DIR,
    CANONICAL_MODEL_PATH,
    CANONICAL_RUNTIME_SOURCE,
    HIPPO2_PREFIX,
    HIPPUNFOLD_EXECUTABLE,
    _join_path,
)
from scripts.core_workflow.lib.output_layout import (
    IOWA7T_VARIANTS,
    resolve_iowa7t_data_root,
    resolve_iowa7t_hipp_outputs_root,
    validate_iowa7t_subject,
    validate_iowa7t_variant,
)

PYTHON_EXE = "/home/yojiao/.conda/envs/py314/bin/python"
RUNTIME_PY = REPO_ROOT / ".runtime_py"


def setup_bids_staging(
    staging_dir: Path,
    subject: str,
    t1w_source: Path,
) -> None:
    if not t1w_source.is_file():
        raise FileNotFoundError(f"T1w image not found: {t1w_source}")

    staging_dir.mkdir(parents=True, exist_ok=True)
    dataset_desc = staging_dir / "dataset_description.json"
    if not dataset_desc.exists():
        dataset_desc.write_text(
            json.dumps(
                {
                    "Name": "UIowa 7T BIDS Staging",
                    "BIDSVersion": "1.4.0",
                    "DatasetType": "raw",
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )

    anat_dir = staging_dir / f"sub-{subject}" / "anat"
    anat_dir.mkdir(parents=True, exist_ok=True)
    t1w_link = anat_dir / f"sub-{subject}_T1w.nii.gz"
    if t1w_link.exists() or t1w_link.is_symlink():
        t1w_link.unlink()
    t1w_link.symlink_to(t1w_source.resolve())


def build_hippunfold_env(runtime_root: Path) -> dict[str, str]:
    env = dict(os.environ)
    for key in tuple(env):
        if key.startswith("HIPPUNFOLD_"):
            del env[key]
    env.update(
        {
            "HIPPUNFOLD_CACHE_DIR": CANONICAL_CACHE_DIR,
            "HIPPUNFOLD_RUNTIME_SOURCE_CACHE": CANONICAL_RUNTIME_SOURCE,
            "HIPPUNFOLD_MODEL_PATH": CANONICAL_MODEL_PATH,
            "HIPPUNFOLD_CACHE_ROOT": str(runtime_root),
            "XDG_CACHE_HOME": str(runtime_root / "xdg_cache"),
            "PIP_CACHE_DIR": str(runtime_root / "pip_cache"),
            "TMPDIR": str(runtime_root / "tmp"),
            "PATH": _join_path(
                str(REPO_ROOT / "scripts"),
                str(HIPPO2_PREFIX / "bin"),
                env.get("PATH", ""),
            ),
            "LD_LIBRARY_PATH": _join_path(
                str(HIPPO2_PREFIX / "lib"), env.get("LD_LIBRARY_PATH", "")
            ),
            "OMP_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "VECLIB_MAXIMUM_THREADS": "1",
            "nnUNet_n_proc_DA": "1",
            "KMP_DUPLICATE_LIB_OK": "TRUE",
            "TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD": "1",
            "PYTHONPATH": f"{RUNTIME_PY}:{env.get('PYTHONPATH', '')}",
        }
    )
    return env


def run_hippunfold(
    staging_dir: Path,
    output_dir: Path,
    subject: str,
    runtime_root: Path,
) -> int:
    output_dir.mkdir(parents=True, exist_ok=True)
    for subpath in ("xdg_cache", "pip_cache", "tmp"):
        (runtime_root / subpath).mkdir(parents=True, exist_ok=True)

    cmd = [
        str(HIPPUNFOLD_EXECUTABLE),
        str(staging_dir),
        str(output_dir),
        "participant",
        "--modality",
        "T1w",
        "--output-density",
        "512",
        "--output-spaces",
        "corobl",
        "--autotop_labels",
        "hipp",
        "dentate",
        "--participant-label",
        subject,
        "--runtime-source-cache-path",
        CANONICAL_RUNTIME_SOURCE,
        "--workflow-profile",
        "none",
        "--sdm",
        "env-modules",
        "--cores",
        "1",
        "--notemp",
        "--filter-T1w",
        "extension=.nii.gz",
        "--use_gpu",
    ]

    env = build_hippunfold_env(runtime_root)
    print(f"Executing HippUnfold: {' '.join(cmd)}", flush=True)
    t0 = time.time()
    proc = subprocess.run(cmd, env=env, cwd=str(REPO_ROOT), check=False)
    elapsed = time.time() - t0
    print(
        f"HippUnfold finished with return code {proc.returncode} in {elapsed:.1f}s",
        flush=True,
    )
    return int(proc.returncode)


def run_cleanup_01_02(
    batch_root: Path,
    subject: str,
    contract_source_cache: Path | None = None,
) -> int:
    checker_script = REPO_ROOT / "scripts" / "core_workflow" / "01_02_check_hippunfold_outputs.py"
    cmd = [
        PYTHON_EXE,
        str(checker_script),
        "--batch-root",
        str(batch_root),
        "--subject",
        subject,
        "--fail-on-missing",
    ]
    if contract_source_cache is not None and contract_source_cache.is_file():
        cmd.extend(["--contract-source-cache", str(contract_source_cache)])

    print(f"Executing 01_02 cleanup: {' '.join(cmd)}", flush=True)
    proc = subprocess.run(cmd, cwd=str(REPO_ROOT), check=False)
    return int(proc.returncode)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run HippUnfold and 01_02 cleanup for an Iowa7T subject."
    )
    parser.add_argument(
        "--subject",
        required=True,
        help="Iowa7T subject ID (e.g. 06112026).",
    )
    parser.add_argument(
        "--variant",
        choices=IOWA7T_VARIANTS,
        default="22p",
        help="Iowa7T variant (default: 22p).",
    )
    parser.add_argument(
        "--staging-dir",
        type=Path,
        default=None,
        help="Custom isolated BIDS staging directory.",
    )
    parser.add_argument(
        "--skip-cleanup",
        action="store_true",
        help="Skip 01_02 cleanup step.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    subject = validate_iowa7t_subject(args.subject)
    variant = validate_iowa7t_variant(args.variant)

    data_root = resolve_iowa7t_data_root(variant=variant)
    hipp_outputs_root = resolve_iowa7t_hipp_outputs_root(variant=variant)
    batch_root = hipp_outputs_root / "subjects"
    subject_output_dir = batch_root / f"sub-{subject}" / "hippunfold"

    t1w_file = data_root / f"sub-{subject}" / "anat" / f"sub-{subject}_T1w.nii.gz"
    staging_dir = (
        args.staging_dir
        if args.staging_dir is not None
        else Path(f"/tmp/iowa7t_{variant}_hippunfold_bids_staging")
    )
    runtime_root = REPO_ROOT / "runtime" / "hippunfold_workdirs" / f"iowa7t_{variant}" / f"sub-{subject}"

    print(f"Subject: {subject}")
    print(f"Variant: {variant}")
    print(f"Data Root: {data_root}")
    print(f"T1w File: {t1w_file}")
    print(f"Staging Dir: {staging_dir}")
    print(f"Output Dir: {subject_output_dir}")
    print(f"Runtime Work Dir: {runtime_root}")

    setup_bids_staging(staging_dir, subject, t1w_file)
    ret = run_hippunfold(staging_dir, subject_output_dir, subject, runtime_root)
    if ret != 0:
        print(f"Error: HippUnfold failed with code {ret}", file=sys.stderr)
        return ret

    if not args.skip_cleanup:
        contract_source = (
            resolve_iowa7t_hipp_outputs_root(variant="36p")
            / "subjects"
            / "_batch_logs"
            / "output_integrity_manifest.json"
        )
        cleanup_ret = run_cleanup_01_02(
            batch_root=batch_root,
            subject=subject,
            contract_source_cache=contract_source if contract_source.is_file() else None,
        )
        if cleanup_ret != 0:
            print(f"Error: 01_02 cleanup failed with code {cleanup_ret}", file=sys.stderr)
            return cleanup_ret

    print(f"Successfully finished HippUnfold and 01_02 cleanup for sub-{subject} ({variant})!")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
