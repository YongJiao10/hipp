#!/usr/bin/env python3
"""Unified Batch Runner and Orchestrator for Multi-Dataset Hippocampal Visualizations.

Executes the 6 participating multi-dataset visualizations across the 36p grid
(smoothing: 2mm and 4mm).

Signal branches: the default 'standard' mode uses the 36p grid
with unsuffixed names. '--signal-branch ctxreg' renders only the canonical
2mm/36p ctxreg cell; '--signal-branch all' renders the standard grid plus
the ctxreg cell with branch-qualified (_standard/_ctxreg) names.

Excluded Visualizations (hard-guarded, not run):
  - multidataset_gradient_subregion_top20_fixed_hcp7t_2col
  - iowa7t_cortex_master_2row
  - kong2022_standard_lateral
  - multidataset_cluster_confidence_negative_stacked
  - multidataset_subfield_composition_2mm_2col/archive_v1_v3
"""

from __future__ import annotations

import argparse
import concurrent.futures
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
LIB_DIR = REPO_ROOT / "scripts" / "visualize" / "lib"
if str(LIB_DIR) not in sys.path:
    sys.path.insert(0, str(LIB_DIR))

# Repo-local imports must follow the LIB_DIR sys.path setup above.
from cortex_group_contrasts import load_contrasts, prepare_contrasts  # noqa: E402
from multidataset_cohorts import (  # noqa: E402
    COMPARISON_VERSION,
    GRID_VARIANTS,
    GridSpec,
    check_comparison_readiness,
    check_grid_readiness,
    get_comparison_output_tag,
)
from tm_group_contrasts import load_tm_contrasts, prepare_tm_contrasts  # noqa: E402

INTERPRETER = Path("/home/yojiao/.conda/envs/py314/bin/python")
if not INTERPRETER.is_file():
    INTERPRETER = Path(sys.executable)

# Registry of participating visualization targets
VIS_SCRIPTS: dict[str, dict[str, str]] = {
    "cluster_confidence_stacked": {
        "dir": "multidataset_cluster_confidence_stacked",
        "script": "plot_multidataset_cluster_positive_stacked.py",
        "desc": "Cluster confidence stacked positive profiles",
    },
    "gradient_subregion_top20_stacked_2col": {
        "dir": "multidataset_gradient_subregion_top20_stacked_2col",
        "script": "render_multidataset_gradient_subregion_top20_stacked_2col.py",
        "desc": "Gradient subregion Top-20 normalized stacked 2-column",
    },
    "hipp_cortex_gradient_wta": {
        "dir": "multidataset_hipp_cortex_gradient_wta",
        "script": "render_multidataset_hipp_cortex_gradient_wta.py",
        "desc": "Hippocampus + cortex gradient and WTA overview",
    },
    "subfield_composition_2col": {
        "dir": "multidataset_subfield_composition_2mm_2col",
        "script": "render_multidataset_subfield_composition_2mm_2col.py",
        "desc": "Hippocampal subfield composition across cohorts",
    },
    "workbench_hippocampus_overlay": {
        "dir": "multidataset_workbench_hippocampus_overlay",
        "script": "render_multidataset_workbench_hippocampus_overlay.py",
        "desc": "Workbench 4-view hippocampus overlay and vertex stats",
    },
    "wta_normalized_longaxis_stacked_2col": {
        "dir": "multidataset_wta_normalized_longaxis_stacked_2col",
        "script": "render_multidataset_wta_normalized_longaxis_stacked_2col.py",
        "desc": "WTA normalized long-axis composition 2-column",
    },
}


def print_readiness_report(grid_specs: list[GridSpec], version: str = "legacy") -> bool:
    """Print preflight readiness table for requested grid combinations."""
    print("\n" + "=" * 78)
    print("PREFLIGHT DATA READINESS AUDIT")
    print("=" * 78)
    all_ready = True
    for spec in grid_specs:
        report = (
            check_comparison_readiness(spec)
            if version == COMPARISON_VERSION
            else check_grid_readiness(spec)
        )
        ready_cnt = report["ready_count"]
        tot = (
            report["total_entries"]
            if version == COMPARISON_VERSION
            else report["total_cohorts"]
        )
        status_str = f"[{'READY' if report['all_ready'] else 'PARTIAL'}]"
        if version == COMPARISON_VERSION:
            variant_desc = (
                f"Kong2022 + Lynch2024 7-network; HCP/JJ_TMS only; {spec.variant}"
            )
        else:
            variant_desc = f"Iowa & Stanford: {spec.variant}, HCP 7T: ICA-FIX"
        print(
            f"{status_str:9} Grid: {spec.tag:8} ({spec.smoothing}; {variant_desc}): {ready_cnt}/{tot} cohorts ready"
        )
        if not report["all_ready"]:
            all_ready = False
            for cid, info in report["cohorts"].items():
                if not info["ready"]:
                    print(
                        f"   -> Missing {cid} ({info['name']}): {', '.join(info['missing'])}"
                    )
    print("=" * 78 + "\n")
    return all_ready


def _branch_qualified_tag(spec: GridSpec, qualify_standard: bool) -> str:
    """Return the file tag, always branch-suffixed in branch-pair mode.

    Default suite runs keep the established unsuffixed standard names.
    Branch-pair runs (``--signal-branch ctxreg``/``all``) qualify every cell
    so standard and ctxreg products form a disjoint namespace.
    """
    base = f"{spec.smoothing}_{spec.variant}"
    if spec.atlas == "canonical8":
        base = f"8net_{base}"
    if spec.signal_branch != "standard" or qualify_standard:
        return f"{base}_{spec.signal_branch}"
    return base


def run_figure(
    vis_key: str,
    spec: GridSpec,
    dry_run: bool = False,
    version: str = "legacy",
    contrast_manifest: Path | None = None,
    branch_qualified: bool = False,
    tm_contrast_manifest: Path | None = None,
) -> dict[str, Any]:
    """Execute a single visualization script for a given GridSpec."""
    cfg = VIS_SCRIPTS[vis_key]
    script_path = REPO_ROOT / "scripts" / "visualize" / cfg["dir"] / cfg["script"]
    if not script_path.is_file():
        raise FileNotFoundError(f"Missing script: {script_path}")

    # Canonical results directory inside each figure folder
    out_dir = REPO_ROOT / "scripts" / "visualize" / cfg["dir"]
    if branch_qualified:
        cell_tag = _branch_qualified_tag(spec, qualify_standard=True)
        if vis_key == "hipp_cortex_gradient_wta":
            output_tag = cell_tag
        elif version == COMPARISON_VERSION:
            output_tag = f"{COMPARISON_VERSION}_{cell_tag}"
        else:
            output_tag = spec.file_tag
    elif vis_key == "hipp_cortex_gradient_wta":
        output_tag = spec.file_tag
    else:
        output_tag = (
            get_comparison_output_tag(spec)
            if version == COMPARISON_VERSION
            else spec.file_tag
        )
    out_png = out_dir / f"{cfg['dir']}_{output_tag}.png"

    cmd = [
        str(INTERPRETER),
        "-c",
        "import runpy, sys; path = sys.argv.pop(1); sys.argv[0] = path; "
        "runpy.run_path(path, run_name='suite_renderer')['_render_grid']()",
        str(script_path),
        "--smoothing",
        spec.smoothing,
        "--variant",
        spec.variant,
        "--signal-branch",
        spec.signal_branch,
        "--atlas",
        spec.atlas,
        "--out-png",
        str(out_png),
    ]
    if vis_key != "hipp_cortex_gradient_wta" and version != "legacy":
        cmd.extend(["--version", version])

    if vis_key == "hipp_cortex_gradient_wta":
        if contrast_manifest is None and not dry_run:
            raise ValueError("Missing explicit all-grid contrast preparation manifest")
        cmd.extend(
            [
                "--contrast-manifest",
                str(contrast_manifest or "<prepared-all-grid-manifest>"),
            ]
        )
    if (
        tm_contrast_manifest is not None
        and (
            vis_key == "hipp_cortex_gradient_wta"
            or (
                vis_key == "workbench_hippocampus_overlay"
                and version == COMPARISON_VERSION
            )
        )
        and (spec.smoothing, spec.variant) == ("2mm", "36p")
    ):
        cmd.extend(["--tm-contrast-manifest", str(tm_contrast_manifest)])

    print(f"==> Launching {vis_key} [{output_tag}]: {' '.join(cmd)}")
    if dry_run:
        return {
            "vis_key": vis_key,
            "grid": output_tag,
            "status": "dry_run",
            "out_png": str(out_png),
        }

    t0 = time.time()
    env = os.environ.copy()
    env["PYTHONPATH"] = f"{REPO_ROOT}:{LIB_DIR}:{env.get('PYTHONPATH', '')}"

    proc = subprocess.run(
        cmd, cwd=REPO_ROOT, env=env, capture_output=True, text=True, check=False
    )
    elapsed = time.time() - t0

    success = proc.returncode == 0
    if not success:
        print(
            f"[FAIL] {vis_key} [{output_tag}] exit={proc.returncode}\nSTDERR:\n{proc.stderr}\nSTDOUT:\n{proc.stdout}"
        )
    else:
        print(
            f"[SUCCESS] {vis_key} [{output_tag}] finished in {elapsed:.1f}s -> {out_png}"
        )

    return {
        "vis_key": vis_key,
        "grid": output_tag,
        "status": "success" if success else "failed",
        "returncode": proc.returncode,
        "elapsed": elapsed,
        "out_png": str(out_png),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Run multi-dataset hippocampal visualization suite across 36p grids."
    )
    parser.add_argument(
        "--version",
        choices=["legacy", COMPARISON_VERSION],
        default="legacy",
        help=(
            "Visualization data branch. 'legacy' preserves the original HCP/Iowa/Stanford suite; "
            f"'{COMPARISON_VERSION}' adds HCP/JJ_TMS Kong2022-vs-Lynch2024 columns."
        ),
    )
    parser.add_argument(
        "--atlas",
        choices=["kong2022", "7net"],
        default="7net",
        help="Atlas / network taxonomy version. Default: 7net (canonical7, unsuffixed); kong2022 is an equivalent path alias.",
    )
    parser.add_argument(
        "--figures",
        nargs="+",
        choices=list(VIS_SCRIPTS.keys()) + ["all"],
        default=["all"],
        help=f"Visualizations to generate. Choices: {list(VIS_SCRIPTS.keys())} or 'all'.",
    )
    parser.add_argument(
        "--check-readiness",
        action="store_true",
        help="Perform preflight data availability audit without running renders.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print commands without executing renders.",
    )
    parser.add_argument(
        "--rebuild-contrast-profiles",
        action="store_true",
        help="Explicitly rebuild subject FC exports from validated canonical inputs; otherwise require valid saved exports",
    )
    parser.add_argument(
        "--iowa36p-source",
        choices=["canonical", "original-denoised"],
        default="canonical",
        help="Explicit approved original-denoised recovery for Iowa36P only",
    )
    parser.add_argument(
        "--signal-branch",
        choices=["standard", "ctxreg", "all"],
        default="standard",
        help=(
            "Hippocampal signal branch cells to render. 'standard' preserves the "
            "36p grids with unsuffixed names; 'ctxreg' renders only the "
            "canonical 2mm/36p ctxreg cell; 'all' renders the standard grids plus "
            "the ctxreg cell with branch-qualified (_standard/_ctxreg) names."
        ),
    )
    args = parser.parse_args()

    if args.version == COMPARISON_VERSION and args.atlas not in {"kong2022", "7net"}:
        parser.error(
            f"--atlas is not used by --version {COMPARISON_VERSION}; omit it to select both 7-network methods"
        )

    # Determine GridSpecs. Default mode uses the standard 36p grids;
    # grid; branch modes add the canonical 2mm/36p ctxreg cell (ctxreg is only
    # defined for 2mm/36p/7net by the GridSpec contract).
    branch_qualified = args.signal_branch != "standard"
    if args.signal_branch == "standard":
        grid_specs = tuple(
            GridSpec.from_tag(v, atlas=args.atlas) for v in GRID_VARIANTS
        )
    elif args.signal_branch == "ctxreg":
        grid_specs = (GridSpec("2mm", "36p", signal_branch="ctxreg", atlas=args.atlas),)
    else:
        grid_specs = tuple(
            GridSpec.from_tag(v, atlas=args.atlas) for v in GRID_VARIANTS
        ) + (GridSpec("2mm", "36p", signal_branch="ctxreg", atlas=args.atlas),)

    # Determine figures to run
    if "all" in args.figures:
        figures = list(VIS_SCRIPTS.keys())
    else:
        figures = args.figures

    # Preflight readiness check
    all_ready = print_readiness_report(list(grid_specs), version=args.version)
    if args.check_readiness:
        return 0 if all_ready else 1

    # Execute the selected grids concurrently.
    print(
        f"[RUNNER] Executing {len(figures)} figure(s) across {len(grid_specs)} grid cell(s)..."
    )
    contrast_manifest = None
    if "hipp_cortex_gradient_wta" in figures and not args.dry_run:
        c_ver = "canonical"
        manifest_path = (
            REPO_ROOT
            / "scripts/visualize/multidataset_hipp_cortex_gradient_wta/contrasts"
            / f"{c_ver}_all_grids.manifest.json"
        )
        if manifest_path.is_file() and not args.rebuild_contrast_profiles:
            try:
                for s in grid_specs:
                    load_contrasts(manifest_path, s, c_ver)
                contrast_manifest = manifest_path
            except Exception:
                contrast_manifest = None
        if contrast_manifest is None:
            contrast_manifest = prepare_contrasts(
                list(grid_specs),
                version=c_ver,
                rebuild_profiles=args.rebuild_contrast_profiles,
                source_mode=args.iowa36p_source,
            )
    tm_manifests: dict[str, Path] = {}
    if (
        "hipp_cortex_gradient_wta" in figures
        or (
            "workbench_hippocampus_overlay" in figures
            and args.version == COMPARISON_VERSION
        )
    ) and not args.dry_run:
        c_ver = "canonical"
        tm_dir = (
            REPO_ROOT
            / "scripts/visualize/multidataset_hipp_cortex_gradient_wta/contrasts"
        )
        tm_branch_specs: dict[str, list[GridSpec]] = {}
        for s in grid_specs:
            if (s.smoothing, s.variant) == ("2mm", "36p"):
                tm_branch_specs.setdefault(s.signal_branch, []).append(s)
        for branch, specs in tm_branch_specs.items():
            if branch == "ctxreg":
                tm_path = tm_dir / f"{c_ver}_2mm_36p_ctxreg_tm.manifest.json"
                tm_source = "canonical"
            else:
                tm_path = tm_dir / f"{c_ver}_2mm_36p_tm.manifest.json"
                tm_source = args.iowa36p_source
            tm_ready: Path | None = None
            if tm_path.is_file() and not args.rebuild_contrast_profiles:
                try:
                    for s in specs:
                        load_tm_contrasts(tm_path, s, c_ver)
                    tm_ready = tm_path
                except Exception:
                    tm_ready = None
            if tm_ready is None:
                prepared_tm: Path = prepare_tm_contrasts(
                    list(specs),
                    version=c_ver,
                    rebuild_profiles=args.rebuild_contrast_profiles,
                    source_mode=tm_source,
                    manifest_path=tm_path,
                )
                tm_ready = prepared_tm
            for s in specs:
                tm_manifests[s.tag] = tm_ready
    results = []
    for fkey in figures:
        base_kwargs: dict[str, Any] = (
            {"contrast_manifest": contrast_manifest}
            if fkey == "hipp_cortex_gradient_wta"
            else {}
        )
        submit_specs: list[tuple[Any, dict[str, Any]]] = []
        for spec in grid_specs:
            skwargs = dict(base_kwargs)
            if fkey == "hipp_cortex_gradient_wta" or (
                fkey == "workbench_hippocampus_overlay"
                and args.version == COMPARISON_VERSION
            ):
                skwargs["tm_contrast_manifest"] = tm_manifests.get(spec.tag)
            submit_specs.append((spec, skwargs))
        with concurrent.futures.ThreadPoolExecutor(max_workers=min(4, len(grid_specs))) as executor:
            future_to_spec = {
                executor.submit(
                    run_figure,
                    fkey,
                    spec,
                    dry_run=args.dry_run,
                    version=args.version,
                    branch_qualified=branch_qualified,
                    **skwargs,
                ): spec
                for spec, skwargs in submit_specs
            }
            spec_to_result = {}
            for future in concurrent.futures.as_completed(future_to_spec):
                spec = future_to_spec[future]
                try:
                    res = future.result()
                    spec_to_result[spec.tag] = res
                except Exception as exc:
                    print(f"[FAIL] {fkey} [{spec.tag}] worker raised exception: {exc}")
                    spec_to_result[spec.tag] = {
                        "vis_key": fkey,
                        "grid": spec.tag,
                        "status": "failed",
                        "error": str(exc),
                    }
            for spec in grid_specs:
                results.append(spec_to_result[spec.tag])

    print("\n" + "=" * 78)
    print("RUN SUMMARY RECEIPT")
    print("=" * 78)
    for r in results:
        status = r["status"].upper()
        print(f"[{status:7}] {r['vis_key']:38} [{r['grid']:7}] -> {r['out_png']}")
    print("=" * 78 + "\n")

    any_fail = any(r.get("status") == "failed" for r in results)
    return 1 if any_fail else 0


if __name__ == "__main__":
    sys.exit(main())
