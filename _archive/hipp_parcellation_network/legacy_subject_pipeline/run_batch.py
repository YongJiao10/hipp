#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
COMMON_DIR = REPO_ROOT / "scripts" / "common"
if str(COMMON_DIR) not in sys.path:
    sys.path.insert(0, str(COMMON_DIR))

from hipp_density_assets import load_surface_density_from_pipeline_config

PYTHON_EXE = sys.executable or "/opt/miniconda3/envs/py314/bin/python"
DEFAULT_OUT_ROOT = REPO_ROOT / "outputs_migration" / "hipp_functional_parcellation_network"
DEFAULT_PRESENT_DIR = REPO_ROOT / "present_network_migration"
BRANCHES = [
    "network-gradient",
    "network-prob-cluster-nonneg",
    "network-spectral",
    "network-spectral-nonneg",
    "intrinsic-spectral",
    "intrinsic-spectral-nonneg",
]
ATLASES = ["lynch2024", "kong2019"]
SUPPORTED_ATLASES = ["lynch2024", "hermosillo2024", "kong2019"]
SUBJECTS = ["100610", "102311", "102816"]
SUMMARY_KEEP = {
    "hipp_functional_parcellation_network_overview.png",
    "k_selection_curves.png",
    "network_probability_heatmaps.png",
    "final_selection_summary.json",
    "final_selection_core.json",
    "summary_manifest.json",
}


_KNOWN_SKIP_MESSAGES: dict[str, str] = {
    "No K survived local-minimum, 1-SE, and min-parcel constraints": (
        "experimental k-selection (mark_instability_decisions) found no K satisfying "
        "local-minimum filter, 1-SE rule, and min_parcel_ok.\n\n"
        "All evaluated K values (2-10) were eliminated by the minimum parcel-size rule. "
        "Common cause: very small meshes or severe data quality issues."
    ),
    "No K survived local-minimum and 1-SE screening": (
        "experimental k-selection (mark_instability_decisions) found no K satisfying "
        "both the local-minimum filter and the 1-SE rule.\n\n"
        "The instability curve has no local minimum within 1 standard error of the global minimum."
    ),
}


def _known_skip_reason(exc_str: str) -> str | None:
    for pattern, message in _KNOWN_SKIP_MESSAGES.items():
        if pattern in exc_str:
            return message
    return None


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
        archive_names = {"surface", "features", "clustering", "soft_outputs", "renders"}
    elif cleanup_level == "render":
        archive_names = {"features", "clustering", "soft_outputs"}
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
    resume_mode: str,
    retain_level: str,
    views: str,
    layout: str,
    k_selection_mode: str,
    run_split_mode: str,
    hipp_density: str,
    shared_surface_store_root: str | None,
    summaries_only: bool,
    out_root: Path,
) -> Path:
    subject_root = out_root / branch_slug / atlas_slug / f"sub-{subject}"
    if not summaries_only:
        run(
            [
                PYTHON_EXE,
                str(REPO_ROOT / "scripts" / "hipp_parcellation_network" / "run_subject.py"),
                "--subject",
                subject,
                "--branch",
                branch_slug,
                "--atlas-slug",
                atlas_slug,
                "--out-root",
                str(out_root),
                "--resume-mode",
                resume_mode,
                "--retain-level",
                retain_level,
                "--views",
                views,
                "--layout",
                layout,
                "--k-selection-mode",
                k_selection_mode,
                "--run-split-mode",
                run_split_mode,
                "--hipp-density",
                hipp_density,
                "--skip-summary",
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
            str(REPO_ROOT / "scripts" / "hipp_parcellation_network" / "summarize_outputs.py"),
            "--root",
            str(subject_root),
        ]
    )
    return subject_root


def copy_present(branch_slug: str, atlas_slug: str, subject: str, out_root: Path, present_dir: Path) -> None:
    subject_root = out_root / branch_slug / atlas_slug / f"sub-{subject}"
    src = subject_root / "hipp_functional_parcellation_network_overview.png"
    if not src.exists():
        raise FileNotFoundError(f"Missing overview for present copy: {src}")
    present_dir.mkdir(parents=True, exist_ok=True)
    dst = present_dir / f"sub-{subject}_{atlas_slug}_{branch_slug}_overview.png"
    placeholder = present_dir / f"sub-{subject}_{atlas_slug}_{branch_slug}_overview.md"
    if placeholder.exists():
        placeholder.unlink()
    shutil.copyfile(src, dst)


def clear_present_overviews(present_dir: Path) -> None:
    if not present_dir.exists():
        return
    for path in present_dir.glob("sub-*_overview.png"):
        path.unlink()
    for path in present_dir.glob("sub-*_overview.md"):
        path.unlink()


def write_skip_placeholder(tag: str, k_selection_mode: str, present_dir: Path, subject: str, atlas_slug: str, branch_slug: str, reason_body: str) -> None:
    placeholder = present_dir / f"sub-{subject}_{atlas_slug}_{branch_slug}_overview.md"
    present_dir.mkdir(parents=True, exist_ok=True)
    placeholder.write_text(
        f"# SKIPPED: {tag}\n\n"
        f"**Time:** {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}\n\n"
        f"**k-selection-mode:** {k_selection_mode}\n\n"
        f"## Reason\n\n"
        f"{reason_body}\n",
        encoding="utf-8",
    )


def validate(branches: list[str], atlases: list[str], subjects: list[str], out_root: Path, present_dir: Path) -> None:
    expected_present = {
        f"sub-{subject}_{atlas}_{branch}_overview.png"
        for branch in branches
        for atlas in atlases
        for subject in subjects
    }
    actual_present = {path.name for path in present_dir.glob("*_overview.png")}
    missing = expected_present - actual_present
    extras = actual_present - expected_present
    if missing or extras:
        raise RuntimeError(
            "present_network_migration/ overview set mismatch.\n"
            f"Missing: {sorted(missing)}\n"
            f"Extras: {sorted(extras)}\n"
            f"Actual: {sorted(actual_present)}"
        )
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
    parser.add_argument("--subjects", nargs="+", default=SUBJECTS)
    parser.add_argument("--resume-mode", choices=["resume", "force"], default="resume")
    parser.add_argument("--retain-level", choices=["label", "render", "feature", "all"], default="render")
    parser.add_argument("--views", default="ventral,dorsal")
    parser.add_argument("--layout", choices=["1x2", "2x2"], default="2x2")
    parser.add_argument("--k-selection-mode", choices=["mainline", "experimental"], default="mainline")
    parser.add_argument(
        "--run-split-mode",
        choices=["none", "runwise"],
        default="none",
        help=(
            "none: use concat timeseries for all run slots; runwise: derive run boundaries from dtseries and split "
            "already-extracted hippocampal surface timeseries only (no raw volume BOLD run-splitting)."
        ),
    )
    parser.add_argument("--shared-surface-store-root", default=None)
    parser.add_argument("--hipp-density", default=default_density)
    parser.add_argument("--cleanup-level", choices=["none", "label", "render", "feature"], default="none")
    parser.add_argument("--out-root", default=str(DEFAULT_OUT_ROOT))
    parser.add_argument("--present-dir", default=str(DEFAULT_PRESENT_DIR))
    parser.add_argument("--clear-present", action="store_true")
    parser.add_argument(
        "--summaries-only",
        action="store_true",
        help="Skip run_subject and only regenerate summarize_outputs + present copies from existing outputs.",
    )
    args = parser.parse_args()

    branches = [str(item) for item in args.branches]
    atlases = [str(item) for item in args.atlases]
    subjects = [str(item) for item in args.subjects]
    out_root = Path(args.out_root).resolve()
    present_dir = Path(args.present_dir).resolve()

    if args.clear_present:
        clear_present_overviews(present_dir)
    skipped: list[str] = []
    for branch_slug in branches:
        for atlas_slug in atlases:
            for subject in subjects:
                tag = f"{branch_slug}/{atlas_slug}/sub-{subject}"
                try:
                    run_subject_atlas(
                        branch_slug=branch_slug,
                        atlas_slug=atlas_slug,
                        subject=subject,
                        resume_mode=args.resume_mode,
                        retain_level=args.retain_level,
                        views=args.views,
                        layout=args.layout,
                        k_selection_mode=args.k_selection_mode,
                        run_split_mode=args.run_split_mode,
                        hipp_density=args.hipp_density,
                        shared_surface_store_root=args.shared_surface_store_root,
                        summaries_only=bool(args.summaries_only),
                        out_root=out_root,
                    )
                    copy_present(branch_slug, atlas_slug, subject, out_root, present_dir)
                    cleanup_subject_outputs(branch_slug, atlas_slug, subject, args.cleanup_level, out_root)
                except Exception as exc:
                    known = _known_skip_reason(str(exc))
                    if known is None:
                        raise RuntimeError(f"Unexpected failure for {tag}") from exc
                    print(f"[SKIP] {tag}: {exc}", flush=True)
                    skipped.append(tag)
                    write_skip_placeholder(
                        tag,
                        args.k_selection_mode,
                        present_dir,
                        subject,
                        atlas_slug,
                        branch_slug,
                        known,
                    )
    if skipped:
        print(f"Skipped {len(skipped)} combination(s): {skipped}", flush=True)
    summary = {
        "branches": branches,
        "atlases": atlases,
        "subjects": subjects,
        "resume_mode": args.resume_mode,
        "retain_level": args.retain_level,
        "cleanup_level": args.cleanup_level,
        "summaries_only": bool(args.summaries_only),
        "shortlist_policy": "always_rebuild",
        "shared_surface_store_root": args.shared_surface_store_root,
        "layout": args.layout,
        "k_selection_mode": args.k_selection_mode,
        "run_split_mode": args.run_split_mode,
        "run_split_behavior": (
            "split_hipp_surface_from_concat_by_dtseries_run_lengths"
            if args.run_split_mode == "runwise"
            else "no_runwise_split_concat_only"
        ),
        "views": args.views,
        "out_root": str(out_root),
        "present": str(present_dir),
    }
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
