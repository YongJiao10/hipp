#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.iowa7t.run_branch_downstream import DEFAULT_OUTPUT_ROOT, BranchJob, resolve_branch_job


PY314 = Path("/home/yojiao/.conda/envs/py314/bin/python")
CACHE_NAME = "iowa7t_branch_individual_overviews"
METHODS = ["gradient_spectral_k3", "gradient_count", "gradient_weighted_fisherz", "wta_top", "wta_weighted_fisherz", "wta_normalized_top20_count", "wta_family17_normalized_top20_count"]
GRADIENT_METHODS = ["gradient_spectral_k3", "gradient_count", "gradient_weighted_fisherz"]
NETWORK_COLUMNS = ("canonical8", "family17")
WEIGHTED_GRADIENT_FILES = (
    "cluster_network_profile_weighted_fisherz.npy",
    "cluster_assignment_status_weighted_fisherz.json",
    "cluster_dominant_network_weighted_fisherz.json",
    "group_functional_atlas_labels_weighted_fisherz.npy",
    "group_functional_atlas_label_map_weighted_fisherz.json",
)
WEIGHTED_WTA_FILES = (
    "vertex_network_profile_weighted_fisherz.npy",
    "wta_top_labels_weighted_fisherz.npy",
    "vertex_assignment_status_weighted_fisherz.json",
    "wta_top_label_map_weighted_fisherz.json",
)
NORMALIZED_WTA_FILES = (
    "wta_top_labels_normalized_top20_count.npy",
    "vertex_assignment_status_normalized_top20_count.json",
    "wta_top_label_map_normalized_top20_count.json",
    "wta_top_labels_family17_normalized_top20_count.npy",
    "vertex_assignment_status_family17_normalized_top20_count.json",
    "wta_top_label_map_family17_normalized_top20_count.json",
)


@dataclass(frozen=True)
class BranchOverviewPaths:
    cache_root: Path
    work_root: Path
    overview_root: Path
    adapter_root: Path
    adapter_present_root: Path
    compat_shared_root: Path
    compat_surface_root: Path
    final_gradient_root: Path
    final_wta_root: Path
    final_gradient_figures: Path
    final_wta_figures: Path


def cache_name_for_network_column(network_column: str) -> str:
    column = str(network_column)
    if column == "canonical8":
        return CACHE_NAME
    if column == "family17":
        return f"{CACHE_NAME}_family17"
    raise ValueError(f"Unsupported network column: {network_column!r}")


def resolve_branch_overview_paths(job: BranchJob, *, network_column: str = "canonical8") -> BranchOverviewPaths:
    cache_root = job.output_root / "runs" / "cache" / cache_name_for_network_column(network_column) / f"sub-{job.subject}_{job.branch}"
    return BranchOverviewPaths(
        cache_root=cache_root,
        work_root=cache_root / "work",
        overview_root=cache_root / "overviews",
        adapter_root=cache_root / "overview_core_adapter",
        adapter_present_root=cache_root / "overview_core_adapter" / "present_tmp",
        compat_shared_root=cache_root / "inputs" / "routea",
        compat_surface_root=cache_root / "inputs" / "surface",
        final_gradient_root=job.branch_root / "atlas" / "gradient",
        final_wta_root=job.branch_root / "atlas" / "wta",
        final_gradient_figures=job.branch_root / "figures" / "gradient",
        final_wta_figures=job.branch_root / "figures" / "wta",
    )


def _ensure_link_or_dir_link(link_path: Path, target: Path) -> None:
    target = Path(target).resolve()
    link_path.parent.mkdir(parents=True, exist_ok=True)
    if link_path.is_symlink():
        try:
            current_target = link_path.resolve(strict=True)
        except FileNotFoundError:
            current_target = None
        if current_target != target:
            link_path.unlink()
            link_path.symlink_to(target, target_is_directory=target.is_dir())
        return
    if link_path.exists():
        if link_path.resolve() == target:
            return
        raise FileExistsError(f"Compatibility path already exists and is not the expected link: {link_path}")
    link_path.symlink_to(target, target_is_directory=target.is_dir())


def prepare_compat_inputs(job: BranchJob, paths: BranchOverviewPaths) -> None:
    sub = f"sub-{job.subject}"
    _ensure_link_or_dir_link(paths.compat_shared_root / sub / "fc", job.branch_root / "fc")
    _ensure_link_or_dir_link(paths.compat_shared_root / sub / "reference", job.branch_root / "cortex")
    _ensure_link_or_dir_link(paths.compat_surface_root / sub / "surface", job.branch_root / "surface")


def run_command(cmd: list[str]) -> None:
    completed = subprocess.run(cmd, cwd=str(REPO_ROOT), text=True)
    if completed.returncode != 0:
        raise RuntimeError(f"Command failed with exit code {completed.returncode}: {' '.join(cmd)}")

def copy_tree_payload(src: Path, dst: Path, *, overwrite: bool) -> None:
    if not src.exists():
        raise FileNotFoundError(f"Missing source payload: {src}")
    if dst.exists() and not overwrite:
        return
    dst.mkdir(parents=True, exist_ok=True)
    for item in src.rglob("*"):
        rel = item.relative_to(src)
        if ".ipynb_checkpoints" in rel.parts:
            continue
        out = dst / rel
        if item.is_dir():
            out.mkdir(parents=True, exist_ok=True)
        elif item.is_file() or item.is_symlink():
            out.parent.mkdir(parents=True, exist_ok=True)
            if out.exists() and not overwrite:
                continue
            shutil.copy2(item, out)


def copy_file_payload(src: Path, dst: Path, *, overwrite: bool) -> None:
    if not src.exists():
        raise FileNotFoundError(f"Missing source payload: {src}")
    if dst.exists() and not overwrite:
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def copy_file_existing_parent(src: Path, dst: Path, *, overwrite: bool) -> None:
    if not src.exists():
        raise FileNotFoundError(f"Missing source payload: {src}")
    if not dst.parent.exists():
        raise FileNotFoundError(f"Destination parent does not exist: {dst.parent}")
    if dst.exists() and not overwrite:
        return
    shutil.copy2(src, dst)


def sync_weighted_outputs_to_cache(job: BranchJob, paths: BranchOverviewPaths, *, overwrite: bool) -> None:
    sub = f"sub-{job.subject}"
    for smoothing in ("2mm", "4mm"):
        for hemi in ("L", "R"):
            src_grad = paths.final_gradient_root / smoothing / hemi
            dst_grad = paths.work_root / sub / "gradient" / smoothing / hemi
            for name in WEIGHTED_GRADIENT_FILES:
                copy_file_existing_parent(src_grad / name, dst_grad / name, overwrite=overwrite)
            grad_labels = sorted((src_grad / "workbench_assets").glob("*_weighted_fisherz.label.gii"))
            if not grad_labels:
                raise FileNotFoundError(f"Missing weighted gradient label GIFTI under {src_grad / 'workbench_assets'}")
            copy_file_existing_parent(grad_labels[0], dst_grad / "workbench_assets" / grad_labels[0].name, overwrite=overwrite)

            src_wta = paths.final_wta_root / smoothing / hemi
            dst_wta = paths.work_root / sub / "wta" / smoothing / hemi
            for name in WEIGHTED_WTA_FILES:
                copy_file_existing_parent(src_wta / name, dst_wta / name, overwrite=overwrite)
            for name in NORMALIZED_WTA_FILES:
                if not (dst_wta / name).exists():
                    copy_file_existing_parent(src_wta / name, dst_wta / name, overwrite=overwrite)
            for tag in ("weighted_fisherz",):
                wta_labels = sorted((src_wta / "workbench_assets").glob(f"*_{tag}.label.gii"))
                if not wta_labels:
                    raise FileNotFoundError(f"Missing {tag} WTA label GIFTI under {src_wta / 'workbench_assets'}")
                copy_file_existing_parent(wta_labels[0], dst_wta / "workbench_assets" / wta_labels[0].name, overwrite=overwrite)
            for tag in ("normalized_top20_count", "family17_normalized_top20_count"):
                if sorted((dst_wta / "workbench_assets").glob(f"*_{tag}.label.gii")):
                    continue
                wta_labels = sorted((src_wta / "workbench_assets").glob(f"*_{tag}.label.gii"))
                if not wta_labels:
                    raise FileNotFoundError(f"Missing {tag} WTA label GIFTI under {src_wta / 'workbench_assets'} and {dst_wta / 'workbench_assets'}")
                copy_file_existing_parent(wta_labels[0], dst_wta / "workbench_assets" / wta_labels[0].name, overwrite=overwrite)


def materialize_final_outputs(job: BranchJob, paths: BranchOverviewPaths, *, overwrite: bool, network_column: str = "canonical8") -> None:
    sub = f"sub-{job.subject}"
    column = str(network_column)
    if column == "canonical8":
        copy_tree_payload(paths.work_root / sub / "gradient", paths.final_gradient_root, overwrite=overwrite)
        copy_tree_payload(paths.work_root / sub / "wta", paths.final_wta_root, overwrite=overwrite)
        copy_file_payload(paths.overview_root / "gradient" / f"{sub}_gradient_fisherz_overview.png", paths.final_gradient_figures / f"{sub}_gradient_fisherz_overview.png", overwrite=overwrite)
        copy_file_payload(paths.overview_root / "gradient" / f"{sub}_gradient_count_overview.png", paths.final_gradient_figures / f"{sub}_gradient_count_overview.png", overwrite=overwrite)
        copy_file_payload(paths.overview_root / "gradient" / f"{sub}_gradient_weighted_fisherz_overview.png", paths.final_gradient_figures / f"{sub}_gradient_weighted_fisherz_overview.png", overwrite=overwrite)
        copy_file_payload(paths.overview_root / "wta" / f"{sub}_wta_overview.png", paths.final_wta_figures / f"{sub}_wta_overview.png", overwrite=overwrite)
        copy_file_payload(paths.overview_root / "wta" / f"{sub}_wta_weighted_fisherz_overview.png", paths.final_wta_figures / f"{sub}_wta_weighted_fisherz_overview.png", overwrite=overwrite)
        copy_file_payload(paths.overview_root / "wta" / f"{sub}_wta_normalized_top20_count_overview.png", paths.final_wta_figures / f"{sub}_wta_normalized_top20_count_overview.png", overwrite=overwrite)
        copy_file_payload(paths.overview_root / "wta" / f"{sub}_wta_family17_normalized_top20_count_overview.png", paths.final_wta_figures / f"{sub}_wta_family17_normalized_top20_count_overview.png", overwrite=overwrite)
    elif column == "family17":
        copy_file_payload(paths.overview_root / "gradient" / f"{sub}_gradient_family17_fisherz_overview.png", paths.final_gradient_figures / f"{sub}_gradient_family17_fisherz_overview.png", overwrite=overwrite)
        copy_file_payload(paths.overview_root / "gradient" / f"{sub}_gradient_family17_count_overview.png", paths.final_gradient_figures / f"{sub}_gradient_family17_count_overview.png", overwrite=overwrite)
        copy_file_payload(paths.overview_root / "gradient" / f"{sub}_gradient_family17_weighted_fisherz_overview.png", paths.final_gradient_figures / f"{sub}_gradient_family17_weighted_fisherz_overview.png", overwrite=overwrite)
    else:
        raise ValueError(f"Unsupported network column: {network_column!r}")
    manifest = {
        "status": "ok",
        "workflow": "iowa7t_branch_individual_overviews",
        "subject": sub,
        "branch": job.branch,
        "network_column": column,
        "cache_root": str(paths.cache_root.resolve()),
        "final_outputs": {
            "atlas_gradient": str(paths.final_gradient_root.resolve()),
            "atlas_wta": str(paths.final_wta_root.resolve()),
            "figures_gradient": str(paths.final_gradient_figures.resolve()),
            "figures_wta": str(paths.final_wta_figures.resolve()),
        },
    }
    manifest_name = "overview_manifest.json" if column == "canonical8" else f"overview_manifest_{column}.json"
    (job.branch_root / "figures" / manifest_name).write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


def run_branch_overviews(job: BranchJob, *, overwrite: bool, jobs: int, network_column: str = "canonical8") -> dict[str, object]:
    column = str(network_column)
    if column not in NETWORK_COLUMNS:
        raise ValueError(f"Unsupported network column: {network_column!r}")
    paths = resolve_branch_overview_paths(job, network_column=column)
    prepare_compat_inputs(job, paths)
    subjects_arg = ["--subjects", job.subject]
    common_input_args = [
        "--shared-root", str(paths.compat_shared_root),
        "--surface-root", str(paths.compat_surface_root),
        "--hippunfold-batch-root", str(job.output_root / "subjects"),
    ]
    run_command([
        str(PY314),
        "scripts/individual_workflow/routea_individual_gradients_wta.py",
        *subjects_arg,
        *common_input_args,
        "--out-root", str(paths.work_root),
        "--jobs", str(max(1, min(int(jobs), 10))),
        "--network-column", column,
    ] + (["--resume"] if not overwrite else []))
    if column == "canonical8":
        sync_weighted_outputs_to_cache(job, paths, overwrite=overwrite)
    run_command([
        str(PY314),
        "scripts/individual_workflow/02_build_individual_overviews.py",
        *subjects_arg,
        *common_input_args,
        "--work-root", str(paths.work_root),
        "--overview-root", str(paths.overview_root),
        "--adapter-root", str(paths.adapter_root),
        "--adapter-present-root", str(paths.adapter_present_root),
        "--methods", *(METHODS if column == "canonical8" else GRADIENT_METHODS),
        "--network-column", column,
        "--jobs", str(max(1, min(int(jobs), 10))),
    ] + (["--resume"] if not overwrite else []))
    materialize_final_outputs(job, paths, overwrite=overwrite, network_column=column)
    return {
        "subject": job.subject,
        "branch": job.branch,
        "cache_root": str(paths.cache_root),
        "atlas_gradient": str(paths.final_gradient_root),
        "atlas_wta": str(paths.final_wta_root),
        "figures_gradient": str(paths.final_gradient_figures),
        "figures_wta": str(paths.final_wta_figures),
        "network_column": column,
    }


def parse_job(value: str) -> tuple[str, str]:
    if ":" not in value:
        raise argparse.ArgumentTypeError("Expected --job value as <subject>:<multiband|multi_echo>")
    subject, branch = value.split(":", 1)
    return subject.replace("sub-", ""), branch


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run fixed-layout Iowa7T branch individual gradient/WTA overviews.")
    parser.add_argument("--job", action="append", type=parse_job, required=True)
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--jobs", type=int, default=1)
    parser.add_argument("--network-column", choices=NETWORK_COLUMNS, default="canonical8")
    parser.add_argument("--overwrite", action="store_true", default=False)
    return parser


def main() -> int:
    args = build_arg_parser().parse_args()
    if int(args.jobs) < 1:
        raise ValueError("--jobs must be >= 1")
    if int(args.jobs) > 10:
        raise ValueError("--jobs exceeds repository hard cap 10")
    output_root = Path(args.output_root)
    results = []
    for subject, branch in args.job:
        job = resolve_branch_job(subject, branch, output_root=output_root)
        print(f"[start] sub-{job.subject} {job.branch} individual overview", flush=True)
        result = run_branch_overviews(job, overwrite=bool(args.overwrite), jobs=int(args.jobs), network_column=str(args.network_column))
        results.append(result)
        print(f"[done] sub-{job.subject} {job.branch}", flush=True)
    print(json.dumps({"jobs": results}, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
