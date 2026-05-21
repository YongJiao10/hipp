#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
from dataclasses import dataclass
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
from lib.output_layout import DEFAULT_GROUP_ATLAS_ROOT, DEFAULT_KONG2022_ROOT, DEFAULT_OVERVIEWS_DIR
BUILD_PRESENT_SCRIPT = REPO_ROOT / "scripts" / "core_workflow" / "06_01_build_group_overview.py"
WTA_PROFILE_SCRIPT = REPO_ROOT / "scripts" / "core_workflow" / "06_02_build_wta_long_axis_profiles.py"
DEFAULT_PYTHON = "/home/yojiao/.conda/envs/py314/bin/python"
DEFAULT_ROUTE_ROOT = DEFAULT_KONG2022_ROOT
DEFAULT_CORE_GROUP_ATLAS_ROOT = DEFAULT_GROUP_ATLAS_ROOT
DEFAULT_PRESENT_DIR = DEFAULT_OVERVIEWS_DIR
DEFAULT_MANIFEST_NAME = "group_overview_matrix_manifest.json"
DEFAULT_ROUTES = ("routea",)
METHODS = ["intrinsic-spectral", "network-spectral", "gradient-spectral", "wta-top"]
ATLAS = "kong2022"
CLUSTERLESS_METHODS = {"wta-top"}
GRADIENT_METHODS = {"gradient-spectral"}
ASSIGNMENT_VARIANTS = ("fisherz", "count")
WTA_PROFILE_BINS = (20, 25)


@dataclass(frozen=True)
class RenderJob:
    method: str
    route_tag: str
    mode_tag: str | None
    group_root: Path
    assignment_variant: str | None = None


def build_render_jobs(route_root: Path, routes: tuple[str, ...] = DEFAULT_ROUTES) -> list[RenderJob]:
    requested_routes = tuple(str(route) for route in routes)
    unknown_routes = sorted(set(requested_routes).difference({"routea", "routeb"}))
    if unknown_routes:
        raise ValueError(f"Unknown routes: {unknown_routes}")
    jobs: list[RenderJob] = []
    routea_full = route_root / "hipp_group_merged_canonical_atlas_routea_full"
    routeb_full = route_root / "hipp_group_merged_canonical_atlas_routeb_full"
    routea_fixed_k3_fisherz = DEFAULT_CORE_GROUP_ATLAS_ROOT / "routea_full_k3_fisherz"
    route_roots_by_method = {
        "intrinsic-spectral": [("routea", None, routea_fixed_k3_fisherz), ("routeb", None, routeb_full)],
        "network-spectral": [("routea", None, routea_fixed_k3_fisherz), ("routeb", None, routeb_full)],
        "gradient-spectral": [("routea", None, routea_fixed_k3_fisherz), ("routeb", None, routeb_full)],
        "wta-top": [("routea", None, routea_fixed_k3_fisherz), ("routeb", None, routeb_full)],
    }
    for method in METHODS:
        for route_tag, mode_tag, group_root in route_roots_by_method[method]:
            if str(route_tag) not in requested_routes:
                continue
            variants: tuple[str | None, ...]
            if method in CLUSTERLESS_METHODS:
                variants = (None,)
            elif str(route_tag) == "routea":
                variants = ASSIGNMENT_VARIANTS
            else:
                variants = ("fisherz",)
            for assignment_variant in variants:
                jobs.append(
                    RenderJob(
                        method=method,
                        route_tag=str(route_tag),
                        mode_tag=(None if mode_tag is None else str(mode_tag)),
                        group_root=Path(group_root).resolve(),
                        assignment_variant=assignment_variant,
                    )
                )
    return jobs


def expected_overview_path(
    present_dir: Path,
    *,
    method: str,
    route_tag: str,
    mode_tag: str | None,
    assignment_variant: str | None = None,
) -> Path:
    bits = [ATLAS, method, "group_overview", route_tag]
    if mode_tag:
        bits.append(mode_tag)
    if assignment_variant:
        bits.append(assignment_variant)
    return present_dir / f"{'_'.join(bits)}.png"


def wta_profile_command(
    *,
    python_exe: str,
    group_root: Path,
    present_dir: Path,
    route_tag: str,
    coverage_threshold: float,
) -> list[str]:
    return [
        str(python_exe),
        str(WTA_PROFILE_SCRIPT),
        "--group-root",
        str(Path(group_root).resolve()),
        "--present-dir",
        str(Path(present_dir).resolve()),
        "--atlas",
        ATLAS,
        "--route-tag",
        str(route_tag),
        "--coverage-threshold",
        f"{float(coverage_threshold):.2f}",
        "--bins",
        *[str(x) for x in WTA_PROFILE_BINS],
    ]


def write_readme(present_dir: Path, items: list[dict[str, str]]) -> None:
    lines = [
        "# HCP 7T hippocampus overview matrix",
        "",
        "This directory is the canonical maintained overview export root for all methods.",
        "All maintained presentation outputs are rendered from core_workflow-produced work roots by default.",
        "No 10perc/full, legacy fixedk_work, or auto-K presentation split is maintained in this exporter.",
        "Each method exports one overview per route into this directory.",
        "",
        "| route | mode | assignment | method | overview | source_group_root |",
        "|---|---|---|---|---|---|",
    ]
    for item in items:
        png_name = Path(item["overview_png"]).name
        lines.append(
            f"| {item['route_tag']} | {item['mode_tag']} | {item.get('assignment_variant', '')} | {item['method']} | "
            f"[{png_name}]({png_name}) | `{item['source_group_root']}` |"
        )

    wta_profiles = sorted(present_dir.glob("kong2022_wta-top_long_axis_network_profile_routea_*.png"))
    if wta_profiles:
        lines.extend(
            [
                "",
                "## WTA Long-Axis Profiles",
                "",
                "These WTA-top profile products are generated by `06_02_build_wta_long_axis_profiles.py` from active WTA labels and summarize network-label composition by hippocampal long-axis bin.",
                "",
                "| profile | data |",
                "|---|---|",
            ]
        )
        for png in wta_profiles:
            if "checkpoint" in png.name:
                continue
            tsv = png.with_suffix(".tsv")
            tsv_cell = f"[{tsv.name}]({tsv.name})" if tsv.exists() else ""
            lines.append(f"| [{png.name}]({png.name}) | {tsv_cell} |")
    (present_dir / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    p = argparse.ArgumentParser(description="One-shot exporter for the maintained HCP 7T hippocampus overview matrix.")
    p.add_argument("--python-exe", default=DEFAULT_PYTHON)
    p.add_argument("--route-root", default=str(DEFAULT_ROUTE_ROOT))
    p.add_argument("--present-dir", default=str(DEFAULT_PRESENT_DIR))
    p.add_argument("--routes", nargs="+", choices=["routea", "routeb"], default=list(DEFAULT_ROUTES))
    p.add_argument("--coverage-threshold", type=float, default=0.80)
    p.add_argument("--manifest-name", default=DEFAULT_MANIFEST_NAME)
    p.add_argument("--scene", default=None, help="Optional override for wb scene path.")
    args = p.parse_args()

    route_root = Path(args.route_root).resolve()
    present_dir = Path(args.present_dir).resolve()
    present_dir.mkdir(parents=True, exist_ok=True)

    records: list[dict[str, str]] = []
    jobs = build_render_jobs(route_root, routes=tuple(args.routes))
    for job in jobs:
        group_root = Path(job.group_root).resolve()
        if not group_root.exists():
            raise FileNotFoundError(f"Missing group_root: {group_root}")
        cmd = [
            args.python_exe,
            str(BUILD_PRESENT_SCRIPT),
            "--group-root",
            str(group_root),
            "--present-dir",
            str(present_dir),
            "--atlases",
            ATLAS,
            "--methods",
            job.method,
            "--coverage-threshold",
            f"{float(args.coverage_threshold):.2f}",
            "--route-tag",
            job.route_tag,
            "--skip-manifest",
            "--skip-readme",
        ]
        if job.mode_tag:
            cmd.extend(["--mode-tag", job.mode_tag])
        if job.assignment_variant:
            cmd.extend(["--assignment-variant", job.assignment_variant])
        if args.scene:
            cmd.extend(["--scene", str(Path(args.scene).resolve())])
        subprocess.run(cmd, check=True, cwd=str(REPO_ROOT))
        overview_png = expected_overview_path(
            present_dir,
            method=job.method,
            route_tag=job.route_tag,
            mode_tag=job.mode_tag,
            assignment_variant=job.assignment_variant,
        ).resolve()
        if not overview_png.exists():
            raise FileNotFoundError(f"Expected overview not found: {overview_png}")
        records.append(
            {
                "route_tag": job.route_tag,
                "mode_tag": "" if job.mode_tag is None else job.mode_tag,
                "method": job.method,
                "assignment_variant": "" if job.assignment_variant is None else job.assignment_variant,
                "source_group_root": str(group_root),
                "overview_png": str(overview_png),
            }
        )

    if "routea" in {str(route) for route in args.routes}:
        subprocess.run(
            wta_profile_command(
                python_exe=str(args.python_exe),
                group_root=DEFAULT_CORE_GROUP_ATLAS_ROOT / "routea_full_k3_fisherz",
                present_dir=present_dir,
                route_tag="routea",
                coverage_threshold=float(args.coverage_threshold),
            ),
            check=True,
            cwd=str(REPO_ROOT),
        )

    expected = len(jobs)
    if len(records) != expected:
        raise RuntimeError(f"Expected {expected} overview outputs, got {len(records)}")

    payload = {
        "present_dir": str(present_dir),
        "count": len(records),
        "items": records,
    }
    (present_dir / args.manifest_name).write_text(json.dumps(payload, indent=2), encoding="utf-8")
    write_readme(present_dir, records)
    print(json.dumps({"present_dir": str(present_dir), "manifest": args.manifest_name, "count": len(records)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
