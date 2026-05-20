#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
from dataclasses import dataclass
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
from lib.output_layout import DEFAULT_OVERVIEWS_DIR
BUILD_PRESENT_SCRIPT = REPO_ROOT / "scripts" / "core_workflow" / "06_01_build_group_overview.py"
DEFAULT_PYTHON = "/home/yojiao/.conda/envs/py314/bin/python"
DEFAULT_ROUTE_ROOT = REPO_ROOT / "outputs_experiments" / "hippunfold_recommended_t1w_20260423" / "kong2022"
DEFAULT_PRESENT_DIR = DEFAULT_OVERVIEWS_DIR
DEFAULT_MANIFEST_NAME = "group_overview_matrix_manifest.json"
METHODS = ["intrinsic-spectral", "network-spectral", "gradient-spectral", "wta-top"]
ATLAS = "kong2022"
CLUSTERLESS_METHODS = {"wta-top"}
GRADIENT_METHODS = {"gradient-spectral"}


@dataclass(frozen=True)
class RenderJob:
    method: str
    route_tag: str
    mode_tag: str | None
    group_root: Path


def build_render_jobs(route_root: Path) -> list[RenderJob]:
    jobs: list[RenderJob] = []
    route_roots = [
        ("routea", None, route_root / "hipp_group_merged_canonical_atlas_routea_full"),
        ("routeb", None, route_root / "hipp_group_merged_canonical_atlas_routeb_full"),
    ]
    for method in METHODS:
        for route_tag, mode_tag, group_root in route_roots:
            jobs.append(
                RenderJob(
                    method=method,
                    route_tag=str(route_tag),
                    mode_tag=(None if mode_tag is None else str(mode_tag)),
                    group_root=Path(group_root).resolve(),
                )
            )
    return jobs


def expected_overview_path(present_dir: Path, *, method: str, route_tag: str, mode_tag: str | None) -> Path:
    bits = [ATLAS, method, "group_overview", route_tag]
    if mode_tag:
        bits.append(mode_tag)
    return present_dir / f"{'_'.join(bits)}.png"


def write_readme(present_dir: Path, items: list[dict[str, str]]) -> None:
    lines = [
        "# HCP 7T hippocampus overview matrix",
        "",
        "This directory is the canonical maintained overview export root for all methods.",
        "All methods are rendered from the fixed full-FC, fixed K=3 route roots by default.",
        "No `10perc/full` or auto-K presentation split is maintained in this exporter.",
        "Each method exports one overview per route into this directory.",
        "",
        "| route | mode | method | overview | source_group_root |",
        "|---|---|---|---|---|",
    ]
    for item in items:
        png_name = Path(item["overview_png"]).name
        lines.append(
            f"| {item['route_tag']} | {item['mode_tag']} | {item['method']} | "
            f"[{png_name}]({png_name}) | `{item['source_group_root']}` |"
        )
    (present_dir / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    p = argparse.ArgumentParser(description="One-shot exporter for the maintained HCP 7T hippocampus overview matrix.")
    p.add_argument("--python-exe", default=DEFAULT_PYTHON)
    p.add_argument("--route-root", default=str(DEFAULT_ROUTE_ROOT))
    p.add_argument("--present-dir", default=str(DEFAULT_PRESENT_DIR))
    p.add_argument("--coverage-threshold", type=float, default=0.80)
    p.add_argument("--manifest-name", default=DEFAULT_MANIFEST_NAME)
    p.add_argument("--scene", default=None, help="Optional override for wb scene path.")
    args = p.parse_args()

    route_root = Path(args.route_root).resolve()
    present_dir = Path(args.present_dir).resolve()
    present_dir.mkdir(parents=True, exist_ok=True)

    records: list[dict[str, str]] = []
    jobs = build_render_jobs(route_root)
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
        if args.scene:
            cmd.extend(["--scene", str(Path(args.scene).resolve())])
        subprocess.run(cmd, check=True, cwd=str(REPO_ROOT))
        overview_png = expected_overview_path(
            present_dir,
            method=job.method,
            route_tag=job.route_tag,
            mode_tag=job.mode_tag,
        ).resolve()
        if not overview_png.exists():
            raise FileNotFoundError(f"Expected overview not found: {overview_png}")
        records.append(
            {
                "route_tag": job.route_tag,
                "mode_tag": "" if job.mode_tag is None else job.mode_tag,
                "method": job.method,
                "source_group_root": str(group_root),
                "overview_png": str(overview_png),
            }
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
