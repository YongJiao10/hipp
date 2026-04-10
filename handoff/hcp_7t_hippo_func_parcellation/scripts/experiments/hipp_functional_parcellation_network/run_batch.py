#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]
COMMON_DIR = REPO_ROOT / "scripts" / "common"
if str(COMMON_DIR) not in sys.path:
    sys.path.insert(0, str(COMMON_DIR))

from hipp_density_assets import load_surface_density_from_pipeline_config

PYTHON_EXE = sys.executable or "/opt/miniconda3/envs/py314/bin/python"
BRANCHES = [
    "network-gradient",
    "network-prob-cluster-nonneg",
]
ATLASES = ["lynch2024", "kong2019"]
SUPPORTED_ATLASES = ["lynch2024", "kong2019"]
DEFAULT_SUBJECTS_FILE = REPO_ROOT / "manifests" / "hcp_7t_hippocampus_struct_complete_166.txt"
SUMMARY_KEEP = {
    "hipp_functional_parcellation_network_overview.png",
    "k_selection_curves.png",
    "network_probability_heatmaps.png",
    "final_selection_summary.json",
    "final_selection_core.json",
    "summary_manifest.json",
}


def run(cmd: list[str]) -> None:
    proc = subprocess.run(cmd, text=True, capture_output=True)
    if proc.returncode != 0:
        raise RuntimeError(
            f"Command failed ({proc.returncode}): {' '.join(cmd)}\nSTDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}"
        )


def move_path(src: Path, dst: Path, manifest: list[dict[str, str]]) -> None:
    if not src.exists():
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        if dst.is_dir():
            shutil.rmtree(dst)
        else:
            dst.unlink()
    shutil.move(str(src), str(dst))
    manifest.append({"src": str(src), "dst": str(dst)})


def cleanup_subject_outputs(branch_slug: str, atlas_slug: str, subject: str, cleanup_level: str, out_root: Path) -> None:
    if cleanup_level == "none":
        return
    root = out_root / branch_slug / atlas_slug / f"sub-{subject}"
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    archive_root = (
        REPO_ROOT
        / "_archive"
        / "hipp_functional_parcellation_network"
        / "cleanup"
        / f"{stamp}_parcellation_cleanup"
        / cleanup_level
        / branch_slug
        / atlas_slug
        / f"sub-{subject}"
    )
    if cleanup_level == "label":
        archive_names = {"surface", "fc", "features", "clustering", "soft_outputs", "renders"}
    elif cleanup_level == "render":
        archive_names = {"fc", "features", "clustering", "soft_outputs"}
    elif cleanup_level == "feature":
        archive_names = set()
    else:
        raise ValueError(f"Unsupported cleanup_level: {cleanup_level}")

    manifest: list[dict[str, str]] = []
    for child in sorted(root.iterdir()):
        if child.name in SUMMARY_KEEP:
            continue
        if child.name.startswith("."):
            continue
        if archive_names and child.name not in archive_names:
            continue
        dst = archive_root / child.name
        move_path(child, dst, manifest)
    archive_root.mkdir(parents=True, exist_ok=True)
    (archive_root / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def run_subject_atlas(
    *,
    branch_slug: str,
    atlas_slug: str,
    subject: str,
    input_root: str,
    hippunfold_root: str,
    cortex_root: str,
    out_root: str,
    resume_mode: str,
    retain_level: str,
    views: str,
    layout: str,
    hipp_density: str,
    shared_surface_store_root: str | None,
    summaries_only: bool,
    rebuild_shortlist: bool,
) -> Path:
    out_root_path = Path(out_root).resolve()
    subject_root = out_root_path / branch_slug / atlas_slug / f"sub-{subject}"
    if not summaries_only:
        run(
            [
                PYTHON_EXE,
                str(REPO_ROOT / "scripts" / "experiments" / "hipp_functional_parcellation_network" / "run_subject.py"),
                "--subject",
                subject,
                "--branch",
                branch_slug,
                "--atlas-slug",
                atlas_slug,
                "--input-root",
                input_root,
                "--hippunfold-root",
                hippunfold_root,
                "--cortex-root",
                cortex_root,
                "--out-root",
                str(out_root_path),
                "--resume-mode",
                resume_mode,
                "--retain-level",
                retain_level,
                "--views",
                views,
                "--layout",
                layout,
                "--hipp-density",
                hipp_density,
            ]
            + (
                ["--shared-surface-store-root", shared_surface_store_root]
                if shared_surface_store_root
                else []
            )
        )
    elif not subject_root.exists():
        raise FileNotFoundError(f"--summaries-only requested but subject root is missing: {subject_root}")
    run(
        [
            PYTHON_EXE,
            str(REPO_ROOT / "scripts" / "experiments" / "hipp_functional_parcellation_network" / "summarize_outputs.py"),
            "--root",
            str(subject_root),
        ]
        + (["--rebuild-shortlist"] if rebuild_shortlist else [])
    )
    return subject_root

def load_subjects(subjects_arg: list[str], subjects_file: str | None) -> list[str]:
    if subjects_arg:
        return [str(item) for item in subjects_arg]
    path = Path(subjects_file).resolve() if subjects_file else DEFAULT_SUBJECTS_FILE
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def validate(branches: list[str], atlases: list[str], subjects: list[str], out_root: Path) -> None:
    for branch_slug in branches:
        for atlas_slug in atlases:
            for subject in subjects:
                root = out_root / branch_slug / atlas_slug / f"sub-{subject}"
                for name in ["final_selection_summary.json", "summary_manifest.json", "hipp_functional_parcellation_network_overview.png"]:
                    if not (root / name).exists():
                        raise RuntimeError(f"Missing required output for {branch_slug}/{atlas_slug}/sub-{subject}: {name}")


def main() -> int:
    default_density = load_surface_density_from_pipeline_config(REPO_ROOT / "config" / "hippo_pipeline.toml")
    parser = argparse.ArgumentParser(description="Batch run network-first hippocampal functional parcellation with resumable stage-aware outputs")
    parser.add_argument("--branches", nargs="+", default=BRANCHES)
    parser.add_argument("--atlases", nargs="+", default=ATLASES, choices=SUPPORTED_ATLASES)
    parser.add_argument("--subjects", nargs="+", default=None)
    parser.add_argument("--subjects-file", default=str(DEFAULT_SUBJECTS_FILE))
    parser.add_argument("--input-root", default=str(REPO_ROOT / "data" / "hippunfold_input"))
    parser.add_argument("--hippunfold-root", default=str(REPO_ROOT / "outputs" / "dense_corobl_batch"))
    parser.add_argument("--cortex-root", default=str(REPO_ROOT / "outputs" / "cortex_pfm"))
    parser.add_argument("--out-root", default=str(REPO_ROOT / "outputs" / "hipp_functional_parcellation_network"))
    parser.add_argument("--resume-mode", choices=["resume", "force"], default="resume")
    parser.add_argument("--retain-level", choices=["label", "render", "feature", "all"], default="render")
    parser.add_argument("--views", default="ventral,dorsal")
    parser.add_argument("--layout", choices=["1x2", "2x2"], default="2x2")
    parser.add_argument("--shared-surface-store-root", default=None)
    parser.add_argument("--hipp-density", default=default_density)
    parser.add_argument("--cleanup-level", choices=["none", "label", "render", "feature"], default="none")
    parser.add_argument(
        "--summaries-only",
        action="store_true",
        help="Skip run_subject and only regenerate summarize_outputs from existing outputs.",
    )
    parser.add_argument(
        "--rebuild-shortlist",
        action="store_true",
        help="Forward to summarize_outputs: force regenerate _overview_shortlist renders.",
    )
    args = parser.parse_args()

    branches = [str(item) for item in args.branches]
    atlases = [str(item) for item in args.atlases]
    subjects = load_subjects(args.subjects or [], args.subjects_file)
    out_root = Path(args.out_root).resolve()

    for branch_slug in branches:
        for atlas_slug in atlases:
            for subject in subjects:
                run_subject_atlas(
                    branch_slug=branch_slug,
                    atlas_slug=atlas_slug,
                    subject=subject,
                    input_root=args.input_root,
                    hippunfold_root=args.hippunfold_root,
                    cortex_root=args.cortex_root,
                    out_root=str(out_root),
                    resume_mode=args.resume_mode,
                    retain_level=args.retain_level,
                    views=args.views,
                    layout=args.layout,
                    hipp_density=args.hipp_density,
                    shared_surface_store_root=args.shared_surface_store_root,
                    summaries_only=bool(args.summaries_only),
                    rebuild_shortlist=bool(args.rebuild_shortlist),
                )
                cleanup_subject_outputs(branch_slug, atlas_slug, subject, args.cleanup_level, out_root)
    validate(branches, atlases, subjects, out_root)
    summary = {
        "branches": branches,
        "atlases": atlases,
        "subjects": subjects,
        "subjects_file": args.subjects_file,
        "input_root": args.input_root,
        "hippunfold_root": args.hippunfold_root,
        "cortex_root": args.cortex_root,
        "out_root": str(out_root),
        "resume_mode": args.resume_mode,
        "retain_level": args.retain_level,
        "cleanup_level": args.cleanup_level,
        "summaries_only": bool(args.summaries_only),
        "rebuild_shortlist": bool(args.rebuild_shortlist),
        "shared_surface_store_root": args.shared_surface_store_root,
        "layout": args.layout,
        "views": args.views,
    }
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
