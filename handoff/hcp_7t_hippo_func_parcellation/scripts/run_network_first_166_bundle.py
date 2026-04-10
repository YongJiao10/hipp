#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
PYTHON_EXE = sys.executable or "python3"
DEFAULT_SUBJECTS_FILE = REPO_ROOT / "manifests" / "hcp_7t_hippocampus_struct_complete_166.txt"
ATLAS_METHOD = {
    "lynch2024": "Lynch2024",
    "kong2019": "Kong2019",
}
DEFAULT_BRANCHES = ["network-gradient", "network-prob-cluster-nonneg"]
DEFAULT_ATLASES = ["lynch2024", "kong2019"]


def run(cmd: list[str]) -> None:
    proc = subprocess.run(cmd, text=True, capture_output=True)
    if proc.returncode != 0:
        raise RuntimeError(
            f"Command failed ({proc.returncode}): {' '.join(cmd)}\nSTDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}"
        )


def load_subjects(subjects: list[str], subjects_file: str) -> list[str]:
    if subjects:
        return [str(item) for item in subjects]
    return [line.strip() for line in Path(subjects_file).read_text(encoding="utf-8").splitlines() if line.strip()]


def require_path(path: Path, label: str) -> Path:
    if not path.exists():
        raise FileNotFoundError(f"Missing {label}: {path}")
    return path


def anat_path(input_root: Path, subject: str, suffix: str) -> Path:
    return input_root / f"sub-{subject}" / "anat" / f"sub-{subject}_{suffix}"


def stage_inputs(subjects: list[str], source_template: str, input_root: Path) -> None:
    for subject in subjects:
        source_dir = Path(source_template.format(subject=subject)).resolve()
        require_path(source_dir, f"stage source dir for sub-{subject}")
        run(
            [
                PYTHON_EXE,
                str(REPO_ROOT / "scripts" / "stage_hippunfold_inputs.py"),
                "--subject",
                subject,
                "--source-dir",
                str(source_dir),
                "--input-dir",
                str(input_root),
            ]
        )


def run_dense_corobl(subjects: list[str], input_root: Path, hippunfold_root: Path) -> None:
    run(
        [
            PYTHON_EXE,
            str(REPO_ROOT / "scripts" / "run_dense_corobl_batch.py"),
            "--subjects",
            *subjects,
            "--input-dir",
            str(input_root),
            "--out-root",
            str(hippunfold_root),
        ]
    )


def run_cortex_pfm(subjects: list[str], atlases: list[str], input_root: Path, cortex_root: Path, fastans_root: Path) -> None:
    for subject in subjects:
        dtseries = require_path(
            input_root / f"sub-{subject}" / "func" / f"sub-{subject}_task-rest_run-concat.dtseries.nii",
            f"dtseries for sub-{subject}",
        )
        left_midthickness = require_path(
            anat_path(input_root, subject, "hemi-L_space-fsLR_den-32k_desc-MSMAll_midthickness.surf.gii"),
            f"left midthickness for sub-{subject}",
        )
        right_midthickness = require_path(
            anat_path(input_root, subject, "hemi-R_space-fsLR_den-32k_desc-MSMAll_midthickness.surf.gii"),
            f"right midthickness for sub-{subject}",
        )
        left_inflated = require_path(
            anat_path(input_root, subject, "hemi-L_space-fsLR_den-32k_desc-MSMAll_inflated.surf.gii"),
            f"left inflated for sub-{subject}",
        )
        right_inflated = require_path(
            anat_path(input_root, subject, "hemi-R_space-fsLR_den-32k_desc-MSMAll_inflated.surf.gii"),
            f"right inflated for sub-{subject}",
        )
        sulc_dscalar = require_path(
            anat_path(input_root, subject, "space-fsLR_den-32k_desc-MSMAll_sulc.dscalar.nii"),
            f"sulc dscalar for sub-{subject}",
        )
        methods = [ATLAS_METHOD[atlas_slug] for atlas_slug in atlases]
        run(
            [
                PYTHON_EXE,
                str(REPO_ROOT / "scripts" / "cortex" / "run_cortex_pfm_subject.py"),
                "--subject",
                subject,
                "--dtseries",
                str(dtseries),
                "--left-midthickness",
                str(left_midthickness),
                "--right-midthickness",
                str(right_midthickness),
                "--left-inflated",
                str(left_inflated),
                "--right-inflated",
                str(right_inflated),
                "--sulc-dscalar",
                str(sulc_dscalar),
                "--fastans-root",
                str(fastans_root),
                "--out-root",
                str(cortex_root),
                "--methods",
                *methods,
            ]
        )
        for atlas_slug in atlases:
            run(
                [
                    PYTHON_EXE,
                    str(REPO_ROOT / "scripts" / "cortex" / "derive_cortex_roi_components.py"),
                    "--subject",
                    subject,
                    "--method",
                    ATLAS_METHOD[atlas_slug],
                    "--data-root",
                    str(input_root),
                    "--out-root",
                    str(cortex_root),
                ]
            )


def run_network_parcellation(
    subjects: list[str],
    subjects_file: str,
    branches: list[str],
    atlases: list[str],
    input_root: Path,
    hippunfold_root: Path,
    cortex_root: Path,
    parcellation_root: Path,
    resume_mode: str,
    retain_level: str,
    cleanup_level: str,
    views: str,
    layout: str,
) -> None:
    run(
        [
            PYTHON_EXE,
            str(REPO_ROOT / "scripts" / "experiments" / "hipp_functional_parcellation_network" / "run_batch.py"),
            "--subjects-file",
            subjects_file,
            "--subjects",
            *subjects,
            "--branches",
            *branches,
            "--atlases",
            *atlases,
            "--input-root",
            str(input_root),
            "--hippunfold-root",
            str(hippunfold_root),
            "--cortex-root",
            str(cortex_root),
            "--out-root",
            str(parcellation_root),
            "--resume-mode",
            resume_mode,
            "--retain-level",
            retain_level,
            "--cleanup-level",
            cleanup_level,
            "--views",
            views,
            "--layout",
            layout,
        ]
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Top-level HPC driver for the 166-subject network-first bundle")
    parser.add_argument("--subjects", nargs="+", default=None)
    parser.add_argument("--subjects-file", default=str(DEFAULT_SUBJECTS_FILE))
    parser.add_argument("--branches", nargs="+", default=DEFAULT_BRANCHES, choices=DEFAULT_BRANCHES)
    parser.add_argument("--atlases", nargs="+", default=DEFAULT_ATLASES, choices=DEFAULT_ATLASES)
    parser.add_argument("--input-root", default=str(REPO_ROOT / "data" / "hippunfold_input"))
    parser.add_argument("--stage-source-dir-template", default=None)
    parser.add_argument("--hippunfold-root", default=str(REPO_ROOT / "outputs" / "dense_corobl_batch"))
    parser.add_argument("--cortex-root", default=str(REPO_ROOT / "outputs" / "cortex_pfm"))
    parser.add_argument("--parcellation-root", default=str(REPO_ROOT / "outputs" / "hipp_functional_parcellation_network"))
    parser.add_argument("--fastans-root", default=str(REPO_ROOT / "external" / "FASTANS"))
    parser.add_argument("--resume-mode", choices=["resume", "force"], default="resume")
    parser.add_argument("--retain-level", choices=["label", "render", "feature", "all"], default="render")
    parser.add_argument("--cleanup-level", choices=["none", "label", "render", "feature"], default="none")
    parser.add_argument("--views", default="ventral,dorsal")
    parser.add_argument("--layout", choices=["1x2", "2x2"], default="2x2")
    parser.add_argument("--skip-stage", action="store_true")
    parser.add_argument("--skip-dense-corobl", action="store_true")
    parser.add_argument("--skip-cortex", action="store_true")
    parser.add_argument("--skip-parcellation", action="store_true")
    args = parser.parse_args()

    subjects = load_subjects(args.subjects or [], args.subjects_file)
    input_root = Path(args.input_root).resolve()
    hippunfold_root = Path(args.hippunfold_root).resolve()
    cortex_root = Path(args.cortex_root).resolve()
    parcellation_root = Path(args.parcellation_root).resolve()
    fastans_root = Path(args.fastans_root).resolve()

    if not args.skip_stage:
        if not args.stage_source_dir_template:
            raise ValueError("--stage-source-dir-template is required unless --skip-stage is set")
        stage_inputs(subjects, args.stage_source_dir_template, input_root)

    if not args.skip_dense_corobl:
        run_dense_corobl(subjects, input_root, hippunfold_root)

    if not args.skip_cortex:
        run_cortex_pfm(subjects, list(args.atlases), input_root, cortex_root, fastans_root)

    if not args.skip_parcellation:
        run_network_parcellation(
            subjects=subjects,
            subjects_file=args.subjects_file,
            branches=list(args.branches),
            atlases=list(args.atlases),
            input_root=input_root,
            hippunfold_root=hippunfold_root,
            cortex_root=cortex_root,
            parcellation_root=parcellation_root,
            resume_mode=args.resume_mode,
            retain_level=args.retain_level,
            cleanup_level=args.cleanup_level,
            views=args.views,
            layout=args.layout,
        )

    summary = {
        "subjects": subjects,
        "subjects_file": args.subjects_file,
        "branches": list(args.branches),
        "atlases": list(args.atlases),
        "input_root": str(input_root),
        "hippunfold_root": str(hippunfold_root),
        "cortex_root": str(cortex_root),
        "parcellation_root": str(parcellation_root),
        "fastans_root": str(fastans_root),
        "resume_mode": args.resume_mode,
        "retain_level": args.retain_level,
        "cleanup_level": args.cleanup_level,
        "views": args.views,
        "layout": args.layout,
        "skip_stage": bool(args.skip_stage),
        "skip_dense_corobl": bool(args.skip_dense_corobl),
        "skip_cortex": bool(args.skip_cortex),
        "skip_parcellation": bool(args.skip_parcellation),
    }
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
