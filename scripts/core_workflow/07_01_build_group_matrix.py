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
  - --routes: Routes/branches to process (choices: "pfm", "pfm_17net", "pfm_ctxreg", "pfm_ctxreg_10mm", "fixed", "fixed_17net", default: ["pfm"]).
  - --methods: Methods to export (default: all supported methods).
  - --coverage-threshold: Coverage threshold used to filter outputs (default: 0.80).
  - --manifest-name: Filename for the generated overview matrix manifest JSON (default: "group_overview_matrix_manifest.json").
  - --scene: Connectome Workbench template scene file override path.
- Physical inputs:
  - Group atlas directories: maintained science branches keep their method/taxonomy suffixes; the HCP cortex-control routes resolve explicitly to the formal direct roots `pfm_ctxreg` (20 mm) and `pfm_ctxreg_10mm`.
  - Scripts to execute:
    - `scripts/core_workflow/06_01_build_group_overview.py`

Outputs:
- File outputs:
  - Overview panel PNG figures for each method and route combination under `{present_dir}/{route}/`.
  - Group overview matrix manifest JSON file: `{present_dir}/{manifest_name}`.
  - Matrix comparison index file: `{present_dir}/README.md`.
"""
import argparse
import json
import subprocess
from dataclasses import dataclass
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
from lib.stale_products import print_stale_figures
from lib.gradient_clustering import GRADIENT_METHODS
from lib.network_ovr import OVR_VARIANTS
from lib.output_layout import (
    ANNOTATION_PAIR_BY_ASSIGNMENT_VARIANT,
    annotation_pair_overview_name,
    DEFAULT_FIXED_17NET_GROUP_ATLAS_ROOT,
    DEFAULT_FIXED_GROUP_ATLAS_ROOT,
    DEFAULT_GROUP_ATLAS_ROOT,
    DEFAULT_KONG2022_ROOT,
    DEFAULT_OVERVIEWS_DIR,
    DEFAULT_PFM_17NET_GROUP_ATLAS_ROOT,
    DEFAULT_PFM_CTXREG_10MM_GROUP_ATLAS_ROOT,
    DEFAULT_PFM_CTXREG_GROUP_ATLAS_ROOT,
    DEFAULT_PFM_GROUP_ATLAS_ROOT,
)
BUILD_PRESENT_SCRIPT = REPO_ROOT / "scripts" / "core_workflow" / "06_01_build_group_overview.py"
DEFAULT_PYTHON = "/home/yojiao/.conda/envs/py314/bin/python"
DEFAULT_GROUP_ATLAS_BASE_ROOT = DEFAULT_GROUP_ATLAS_ROOT
DEFAULT_CORE_GROUP_ATLAS_ROOT = DEFAULT_GROUP_ATLAS_ROOT
DEFAULT_PRESENT_DIR = DEFAULT_OVERVIEWS_DIR
DEFAULT_MANIFEST_NAME = "group_overview_matrix_manifest.json"
DEFAULT_ROUTES = ("pfm",)
ROUTE_CHOICES = ("pfm", "pfm_17net", "pfm_ctxreg", "pfm_ctxreg_10mm", "fixed", "fixed_17net")
METHODS = ["intrinsic-spectral", "network-spectral", "gradient-spectral", "wta-top"]
ATLAS = "kong2022"
CLUSTERLESS_METHODS = {"wta-top"}
METHOD_CHOICES = METHODS + sorted(GRADIENT_METHODS - set(METHODS))
DEFAULT_GRADIENT_COMPONENT_TAG = "g1g2g3"
ASSIGNMENT_VARIANTS = ("network_ts_fisherz", "fisherz", "normalized_top20_count")
# "both": the Kong2022 cluster-level default (network-TS Fisher-z) plus normalized Top-20.
DEFAULT_ASSIGNMENT_VARIANTS = ("network_ts_fisherz", "normalized_top20_count")


@dataclass(frozen=True)
class RenderJob:
    method: str
    route_tag: str
    mode_tag: str | None
    group_root: Path
    smoothing: str
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
            "pfm": DEFAULT_PFM_GROUP_ATLAS_ROOT.resolve(),
            "pfm_17net": DEFAULT_PFM_17NET_GROUP_ATLAS_ROOT.resolve(),
            "pfm_ctxreg": DEFAULT_PFM_CTXREG_GROUP_ATLAS_ROOT.resolve(),
            "pfm_ctxreg_10mm": DEFAULT_PFM_CTXREG_10MM_GROUP_ATLAS_ROOT.resolve(),
            "fixed": DEFAULT_FIXED_GROUP_ATLAS_ROOT.resolve(),
            "fixed_17net": DEFAULT_FIXED_17NET_GROUP_ATLAS_ROOT.resolve(),
        }
    return {
        "pfm": (base / "pfm").resolve(),
        "pfm_17net": (base / "pfm_17net").resolve(),
        "pfm_ctxreg": (base / "pfm_ctxreg").resolve(),
        "pfm_ctxreg_10mm": (base / "pfm_ctxreg_10mm").resolve(),
        "fixed": (base / "fixed").resolve(),
        "fixed_17net": (base / "fixed_17net").resolve(),
    }


def build_render_jobs(
    group_atlas_root: Path = DEFAULT_GROUP_ATLAS_BASE_ROOT,
    routes: tuple[str, ...] = DEFAULT_ROUTES,
    methods: tuple[str, ...] = tuple(METHODS),
    assignment_variants: tuple[str, ...] = DEFAULT_ASSIGNMENT_VARIANTS,
) -> list[RenderJob]:
    requested_routes = tuple(str(route) for route in routes)
    unknown_routes = sorted(set(requested_routes).difference(set(ROUTE_CHOICES)))
    if unknown_routes:
        raise ValueError(f"Unknown routes: {unknown_routes}")
    requested_methods = tuple(str(method) for method in methods)
    unknown_methods = sorted(set(requested_methods).difference(set(METHOD_CHOICES)))
    if unknown_methods:
        raise ValueError(f"Unknown methods: {unknown_methods}")
    jobs: list[RenderJob] = []
    route_roots = canonical_group_roots(Path(group_atlas_root))
    route_roots_by_method = {
        "intrinsic-spectral": [("pfm", None, route_roots["pfm"]), ("pfm_ctxreg", None, route_roots["pfm_ctxreg"]), ("fixed", None, route_roots["fixed"])],
        "network-spectral": [("pfm", None, route_roots["pfm"]), ("pfm_ctxreg", None, route_roots["pfm_ctxreg"]), ("fixed", None, route_roots["fixed"])],
        "gradient-spectral": [
            ("pfm", None, route_roots["pfm"]),
            ("pfm_17net", None, route_roots["pfm_17net"]),
            ("pfm_ctxreg", None, route_roots["pfm_ctxreg"]),
            ("pfm_ctxreg_10mm", None, route_roots["pfm_ctxreg_10mm"]),
            ("fixed", None, route_roots["fixed"]),
            ("fixed_17net", None, route_roots["fixed_17net"]),
        ],
        "wta-top": [
            ("pfm", None, route_roots["pfm"]),
            ("pfm_17net", None, route_roots["pfm_17net"]),
            ("pfm_ctxreg", None, route_roots["pfm_ctxreg"]),
            ("pfm_ctxreg_10mm", None, route_roots["pfm_ctxreg_10mm"]),
            ("fixed", None, route_roots["fixed"]),
            ("fixed_17net", None, route_roots["fixed_17net"]),
        ],
    }
    route_roots_by_method.update({
        method: list(route_roots_by_method["gradient-spectral"])
        for method in GRADIENT_METHODS - {"gradient-spectral"}
    })
    for method in requested_methods:
        for route_tag, mode_tag, group_root in route_roots_by_method[method]:
            if str(route_tag) not in requested_routes:
                continue
            variants: tuple[str | None, ...]
            if method in CLUSTERLESS_METHODS:
                variants = (None,)
            elif str(route_tag) in {"pfm", "pfm_17net", "pfm_ctxreg", "pfm_ctxreg_10mm"}:
                variants = assignment_variants
            else:
                variants = assignment_variants
            for assignment_variant in variants:
                for smoothing in ("2mm", "4mm"):
                    jobs.append(
                        RenderJob(
                            method=method,
                            route_tag=str(route_tag),
                            mode_tag=(None if mode_tag is None else str(mode_tag)),
                            group_root=Path(group_root).resolve(),
                            assignment_variant=assignment_variant,
                            smoothing=smoothing,
                        )
                    )
    return jobs


def annotation_pair_for_job(method: str, assignment_variant: str | None) -> str | None:
    """Select the exact maintained interpretation for each gradient assignment."""
    if method not in GRADIENT_METHODS:
        return None
    if assignment_variant in ANNOTATION_PAIR_BY_ASSIGNMENT_VARIANT:
        return ANNOTATION_PAIR_BY_ASSIGNMENT_VARIANT[assignment_variant]
    raise ValueError(f"Unsupported gradient overview assignment variant: {assignment_variant!r}")


def expected_overview_path(
    present_dir: Path,
    *,
    method: str,
    route_tag: str,
    mode_tag: str | None,
    assignment_variant: str | None = None,
    smoothing: str,
) -> Path:
    if smoothing not in {"2mm", "4mm"}:
        raise ValueError(f"Unsupported smoothing: {smoothing!r}")
    pair = annotation_pair_for_job(method, assignment_variant)
    if pair is not None:
        filename = annotation_pair_overview_name(pair)
    elif method in CLUSTERLESS_METHODS:
        filename = "wta_overview.png"
    elif assignment_variant is not None:
        bits = [ATLAS, method, "group_overview"]
        if method in GRADIENT_METHODS:
            bits.append(DEFAULT_GRADIENT_COMPONENT_TAG)
        bits.append(route_tag)
        if mode_tag:
            bits.append(mode_tag)
        bits.append(assignment_variant)
        filename = f"{'_'.join(bits)}.png"
    else:
        raise ValueError(f"Missing assignment variant for {method}")
    return present_dir / route_tag / method / smoothing / filename


def discover_hcp_profile_overviews(present_dir: Path) -> list[dict[str, object]]:
    items: list[dict[str, object]] = []
    overviews_base = Path(present_dir).resolve()
    pfm_root = overviews_base / "pfm"
    if not pfm_root.is_dir():
        return items

    for profile, method, role in [
        ("topsim", "gradient-ward", "primary"),
        ("topfc", "gradient-spectral", "historical_baseline"),
        ("topsim", "gradient-gmm", "comparison_display"),
    ]:
        for smoothing in ("2mm", "4mm"):
            dir_path = pfm_root / profile / method / smoothing
            if not dir_path.is_dir():
                continue
            for pair in ("network-ts-fisherz", "roi-fisherz", "top20"):
                png_path = dir_path / annotation_pair_overview_name(pair)
                if png_path.is_file():
                    items.append({
                        "profile": profile,
                        "method": method,
                        "role": role,
                        "smoothing": smoothing,
                        "annotation_pair": pair,
                        "k": 3,
                        "selected_default": (profile == "topsim" and method == "gradient-ward" and smoothing == "2mm" and pair == "network-ts-fisherz"),
                        "overview_png": str(png_path.resolve()),
                    })

    for smoothing in ("2mm", "4mm"):
        dir_path = pfm_root / "wta-top" / smoothing
        if dir_path.is_dir():
            png_path = dir_path / "wta_overview.png"
            if png_path.is_file():
                items.append({
                    "profile": None,
                    "method": "wta-top",
                    "role": "clusterless",
                    "smoothing": smoothing,
                    "annotation_pair": None,
                    "k": None,
                    "selected_default": False,
                    "overview_png": str(png_path.resolve()),
                })
    return items


discover_formal_hcp_overviews = discover_hcp_profile_overviews


def write_readme(present_dir: Path, items: list[dict[str, str]], formal_items: list[dict[str, object]] | None = None) -> None:
    lines = [
        "# HCP 7T hippocampus overview matrix",
        "",
        "This directory is the canonical maintained overview export root for all methods.",
        "All maintained presentation outputs are rendered from core_workflow-produced work roots by default.",
        "No 10perc/full, legacy fixedk_work, or auto-K presentation split is maintained in this exporter.",
        "Each method exports its overviews into the corresponding route subdirectory.",
        "",
    ]
    if formal_items:
        lines.extend([
            "## Formal HCP PFM Parcellation Products",
            "",
            "Primary selection: **TopSim + Ward** at K=3 (2mm).",
            "Baseline: **TopFC + spectral**; Comparison: **TopSim + GMM**; Clusterless: **WTA**.",
            "",
            "| profile | method | role | smoothing | annotation_pair | selected_default | overview |",
            "|---|---|---|---|---|---|---|",
        ])
        for fitem in formal_items:
            png_name = Path(str(fitem["overview_png"])).relative_to(present_dir).as_posix()
            lines.append(
                f"| {fitem.get('profile') or '-'} | {fitem['method']} | {fitem['role']} | "
                f"{fitem['smoothing']} | {fitem.get('annotation_pair') or '-'} | "
                f"{'**PRIMARY**' if fitem.get('selected_default') else 'no'} | [{png_name}]({png_name}) |"
            )
        lines.append("")

    lines.extend([
        "## General Route Matrix",
        "",
        "| route | mode | assignment | method | overview | source_group_root |",
        "|---|---|---|---|---|---|",
    ])
    for item in items:
        png_name = Path(item["overview_png"]).relative_to(present_dir).as_posix()
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
    p.add_argument("--methods", nargs="+", choices=METHOD_CHOICES, default=list(METHODS))
    p.add_argument("--assignment-variant", choices=["both", *ASSIGNMENT_VARIANTS, *OVR_VARIANTS], default="both")
    p.add_argument("--coverage-threshold", type=float, default=0.80)
    p.add_argument("--manifest-name", default=DEFAULT_MANIFEST_NAME)
    p.add_argument("--scene", default=None, help="Optional override for wb scene path.")
    p.add_argument("--index-only", action="store_true", help="Index existing overview products without re-running 06_01 render jobs.")
    p.add_argument(
        "--check-sources", nargs="*", default=None, metavar="PRESENT_ROOT",
        help="Only list overview and gradient figures that are older than the work files they were rendered "
             "from, then exit nonzero if any is stale. Defaults to --present-dir and its sibling gradients/.",
    )
    args = p.parse_args()
    if args.check_sources is not None:
        roots = args.check_sources or [args.present_dir, Path(args.present_dir).resolve().parent / "gradients"]
        return 1 if print_stale_figures(roots) else 0

    group_atlas_root = resolve_group_atlas_base_root(Path(args.group_atlas_root))
    present_dir = Path(args.present_dir).resolve()
    present_dir.mkdir(parents=True, exist_ok=True)

    records: list[dict[str, str]] = []
    jobs = build_render_jobs(group_atlas_root, routes=tuple(args.routes), methods=tuple(args.methods),
                             assignment_variants=DEFAULT_ASSIGNMENT_VARIANTS if args.assignment_variant == "both" else (args.assignment_variant,))
    if args.index_only:
        for job in jobs:
            overview_png = expected_overview_path(
                present_dir,
                method=job.method,
                route_tag=job.route_tag,
                mode_tag=job.mode_tag,
                assignment_variant=job.assignment_variant,
                smoothing=job.smoothing,
            ).resolve()
            if overview_png.exists():
                records.append(
                    {
                        "route_tag": job.route_tag,
                        "mode_tag": "" if job.mode_tag is None else job.mode_tag,
                        "method": job.method,
                        "smoothing": job.smoothing,
                        "assignment_variant": "" if job.assignment_variant is None else job.assignment_variant,
                        "source_group_root": str(job.group_root),
                        "overview_png": str(overview_png),
                    }
                )
    else:
        for job in jobs:
            group_root = Path(job.group_root).resolve()
            if not group_root.exists():
                raise FileNotFoundError(f"Missing group_root: {group_root}")
            route_present_dir = present_dir / job.route_tag / job.method / job.smoothing
            cmd = [
                args.python_exe,
                str(BUILD_PRESENT_SCRIPT),
                "--group-root",
                str(group_root),
                "--present-dir",
                str(route_present_dir),
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
                "--smoothing",
                job.smoothing,
            ]
            if job.mode_tag:
                cmd.extend(["--mode-tag", job.mode_tag])
            pair = annotation_pair_for_job(job.method, job.assignment_variant)
            if pair is not None:
                cmd.extend(["--annotation-pair", pair])
            elif job.assignment_variant:
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
                smoothing=job.smoothing,
            ).resolve()
            if not overview_png.exists():
                raise FileNotFoundError(f"Expected overview not found: {overview_png}")
            records.append(
                {
                    "route_tag": job.route_tag,
                    "mode_tag": "" if job.mode_tag is None else job.mode_tag,
                    "method": job.method,
                    "smoothing": job.smoothing,
                    "assignment_variant": "" if job.assignment_variant is None else job.assignment_variant,
                    "source_group_root": str(group_root),
                    "overview_png": str(overview_png),
                }
            )

        expected = len(jobs)
        if len(records) != expected:
            raise RuntimeError(f"Expected {expected} overview outputs, got {len(records)}")

    formal_items = discover_formal_hcp_overviews(present_dir)
    payload = {
        "present_dir": str(present_dir),
        "count": len(records),
        "items": records,
        "hcp_pfm_profile_products": formal_items,
        "primary_profile_selection": next(
            (item for item in formal_items if item.get("selected_default")), None
        ),
        "formal_hcp_pfm_products": formal_items,
        "formal_primary_selection": next(
            (item for item in formal_items if item.get("selected_default")), None
        ),
    }
    (present_dir / args.manifest_name).write_text(json.dumps(payload, indent=2), encoding="utf-8")
    write_readme(present_dir, records, formal_items=formal_items)
    print(json.dumps({"present_dir": str(present_dir), "manifest": args.manifest_name, "count": len(records)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
