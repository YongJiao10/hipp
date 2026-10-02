#!/usr/bin/env python3
from __future__ import annotations

"""
Process Step Description:
Build group matrix summary. Integrates method results from different branches by executing downstream presentation scripts to construct the final group-level comparison matrix, summary index markdown, and overview manifests.

Main Flow Path:
- 10th (final) step in the numbered mainline pipeline sequence:
  01_01_run_hippunfold_t1w_local.py -> 01_02_check_hippunfold_outputs.py -> 02_01_prepare_group_inputs.py -> 02_02_build_qc_manifest.py -> 03_01/03_02 reference building -> 04_01/04_02 FC materialization -> 05_01 gradient computation -> 05_02 group atlas parcellation -> 06_01 through 06_07 presentation generation -> 07_01_build_group_matrix.py (this step).

Inputs:
- Command-line arguments:
  - --python-exe: Explicit python interpreter path used to execute child scripts (default: `/home/yojiao/.conda/envs/py314/bin/python`).
  - --group-atlas-root: Base directory containing branch-level group-atlas outputs (default: `outputs_hcp/work/group_atlas`).
  - --present-dir: Output directory for final matrix presentation figures and manifest (default: `outputs_hcp/present/overviews`).
  - --routes: Routes/branches to process (choices: "routea", "routea_ctxreg", "routea_ctxdrop", "routeb", default: ["routea"]).
  - --methods: Methods to export (default: all supported methods).
  - --coverage-threshold: Coverage threshold used to filter outputs (default: 0.80).
  - --manifest-name: Filename for the generated overview matrix manifest JSON (default: "group_overview_matrix_manifest.json").
  - --scene: Connectome Workbench template scene file override path.
- Physical inputs:
  - Group atlas directories: maintained science branches keep their method/taxonomy suffixes; the HCP cortex-control routes resolve explicitly to the formal direct roots `routea_ctxreg` and `routea_ctxdrop`.
  - Scripts to execute:
    - `scripts/core_workflow/06_01_build_group_overview.py`

Outputs:
- File outputs:
  - Overview panel PNG figures for each method and route combination under `{present_dir}/`.
  - Group overview matrix manifest JSON file: `{present_dir}/{manifest_name}`.
  - Matrix comparison index file: `{present_dir}/README.md`.
"""
import argparse
import json
import subprocess
from dataclasses import dataclass
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
from lib.output_layout import (
    DEFAULT_GROUP_ATLAS_ROOT,
    DEFAULT_KONG2022_ROOT,
    DEFAULT_OVERVIEWS_DIR,
    DEFAULT_ROUTEA_CTXDROP_GROUP_ATLAS_ROOT,
    DEFAULT_ROUTEA_CTXREG_GROUP_ATLAS_ROOT,
    DEFAULT_ROUTEA_GROUP_ATLAS_ROOT,
    DEFAULT_ROUTEA_17NET_GROUP_ATLAS_ROOT,
    DEFAULT_ROUTEB_GROUP_ATLAS_ROOT,
)
BUILD_PRESENT_SCRIPT = REPO_ROOT / "scripts" / "core_workflow" / "06_01_build_group_overview.py"
DEFAULT_PYTHON = "/home/yojiao/.conda/envs/py314/bin/python"
DEFAULT_GROUP_ATLAS_BASE_ROOT = DEFAULT_GROUP_ATLAS_ROOT
DEFAULT_CORE_GROUP_ATLAS_ROOT = DEFAULT_GROUP_ATLAS_ROOT
DEFAULT_PRESENT_DIR = DEFAULT_OVERVIEWS_DIR
DEFAULT_MANIFEST_NAME = "group_overview_matrix_manifest.json"
DEFAULT_ROUTES = ("routea",)
ROUTE_CHOICES = ("routea", "routea_17net", "routea_ctxreg", "routea_ctxdrop", "routeb")
METHODS = ["intrinsic-spectral", "network-spectral", "gradient-spectral", "wta-top"]
ATLAS = "kong2022"
CLUSTERLESS_METHODS = {"wta-top"}
GRADIENT_METHODS = {"gradient-spectral"}
DEFAULT_GRADIENT_COMPONENT_TAG = "g1g2g3"
ASSIGNMENT_VARIANTS = ("fisherz", "normalized_top20_count")


@dataclass(frozen=True)
class RenderJob:
    method: str
    route_tag: str
    mode_tag: str | None
    group_root: Path
    assignment_variant: str | None = None


def resolve_group_atlas_base_root(group_atlas_root: Path) -> Path:
    base = Path(group_atlas_root).resolve()
    if base == DEFAULT_KONG2022_ROOT.resolve():
        return DEFAULT_CORE_GROUP_ATLAS_ROOT.resolve()
    return base


def canonical_group_roots(group_atlas_root: Path) -> dict[str, Path]:
    base = resolve_group_atlas_base_root(group_atlas_root)
    default_base = DEFAULT_CORE_GROUP_ATLAS_ROOT.resolve()
    if base == default_base:
        return {
            "routea": DEFAULT_ROUTEA_GROUP_ATLAS_ROOT.resolve(),
            "routea_17net": DEFAULT_ROUTEA_17NET_GROUP_ATLAS_ROOT.resolve(),
            "routea_ctxreg": DEFAULT_ROUTEA_CTXREG_GROUP_ATLAS_ROOT.resolve(),
            "routea_ctxdrop": DEFAULT_ROUTEA_CTXDROP_GROUP_ATLAS_ROOT.resolve(),
            "routeb": DEFAULT_ROUTEB_GROUP_ATLAS_ROOT.resolve(),
        }
    return {
        "routea": (base / "routea_full_k3_fisherz").resolve(),
        "routea_17net": (base / "routea_17net_full_k3_fisherz").resolve(),
        "routea_ctxreg": (base / "routea_ctxreg_full_k3_fisherz").resolve(),
        "routea_ctxdrop": (base / "routea_ctxdrop_full_k3_fisherz").resolve(),
        "routeb": (base / "routeb_full_k3_fisherz").resolve(),
    }


def build_render_jobs(
    group_atlas_root: Path = DEFAULT_GROUP_ATLAS_BASE_ROOT,
    routes: tuple[str, ...] = DEFAULT_ROUTES,
    methods: tuple[str, ...] = tuple(METHODS),
) -> list[RenderJob]:
    requested_routes = tuple(str(route) for route in routes)
    unknown_routes = sorted(set(requested_routes).difference(set(ROUTE_CHOICES)))
    if unknown_routes:
        raise ValueError(f"Unknown routes: {unknown_routes}")
    requested_methods = tuple(str(method) for method in methods)
    unknown_methods = sorted(set(requested_methods).difference(set(METHODS)))
    if unknown_methods:
        raise ValueError(f"Unknown methods: {unknown_methods}")
    jobs: list[RenderJob] = []
    route_roots = canonical_group_roots(Path(group_atlas_root))
    route_roots_by_method = {
        "intrinsic-spectral": [("routea", None, route_roots["routea"]), ("routea_ctxreg", None, route_roots["routea_ctxreg"]), ("routeb", None, route_roots["routeb"])],
        "network-spectral": [("routea", None, route_roots["routea"]), ("routea_ctxreg", None, route_roots["routea_ctxreg"]), ("routeb", None, route_roots["routeb"])],
        "gradient-spectral": [("routea", None, route_roots["routea"]), ("routea_17net", None, route_roots["routea_17net"]), ("routea_ctxreg", None, route_roots["routea_ctxreg"]), ("routea_ctxdrop", None, route_roots["routea_ctxdrop"]), ("routeb", None, route_roots["routeb"])],
        "wta-top": [("routea", None, route_roots["routea"]), ("routea_ctxreg", None, route_roots["routea_ctxreg"]), ("routea_ctxdrop", None, route_roots["routea_ctxdrop"]), ("routeb", None, route_roots["routeb"])],
    }
    for method in requested_methods:
        for route_tag, mode_tag, group_root in route_roots_by_method[method]:
            if str(route_tag) not in requested_routes:
                continue
            variants: tuple[str | None, ...]
            if method in CLUSTERLESS_METHODS:
                variants = (None,)
            elif str(route_tag) in {"routea", "routea_17net", "routea_ctxreg", "routea_ctxdrop"}:
                variants = ASSIGNMENT_VARIANTS
            else:
                variants = ASSIGNMENT_VARIANTS
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
    bits = [ATLAS, method, "group_overview"]
    if method in GRADIENT_METHODS:
        bits.append(DEFAULT_GRADIENT_COMPONENT_TAG)
    bits.append(route_tag)
    if mode_tag:
        bits.append(mode_tag)
    if assignment_variant:
        bits.append(assignment_variant)
    return present_dir / f"{'_'.join(bits)}.png"


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

    (present_dir / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    p = argparse.ArgumentParser(description="One-shot exporter for the maintained HCP 7T hippocampus overview matrix.")
    p.add_argument("--python-exe", default=DEFAULT_PYTHON)
    p.add_argument("--group-atlas-root", "--route-root", dest="group_atlas_root", default=str(DEFAULT_GROUP_ATLAS_BASE_ROOT), help="Base root containing canonical branch group-atlas directories; --route-root is a legacy alias.")
    p.add_argument("--present-dir", default=str(DEFAULT_PRESENT_DIR))
    p.add_argument("--routes", nargs="+", choices=list(ROUTE_CHOICES), default=list(DEFAULT_ROUTES))
    p.add_argument("--methods", nargs="+", choices=METHODS, default=list(METHODS))
    p.add_argument("--coverage-threshold", type=float, default=0.80)
    p.add_argument("--manifest-name", default=DEFAULT_MANIFEST_NAME)
    p.add_argument("--scene", default=None, help="Optional override for wb scene path.")
    args = p.parse_args()

    group_atlas_root = resolve_group_atlas_base_root(Path(args.group_atlas_root))
    present_dir = Path(args.present_dir).resolve()
    present_dir.mkdir(parents=True, exist_ok=True)

    records: list[dict[str, str]] = []
    jobs = build_render_jobs(group_atlas_root, routes=tuple(args.routes), methods=tuple(args.methods))
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
