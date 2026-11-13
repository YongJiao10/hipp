"""Report presentation figures that are older than the work files they show.

A figure is stale when a file it was rendered from is missing or has a later
modification time: the group-atlas or gradient manifest it names, or, one step
upstream, the gradient or parent atlas that group atlas was built from. Stage
manifests are rewritten on every run, so their modification time marks the run.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

OVERVIEW_SIDECAR_GLOB = "*.sources.json"
GRADIENT_PRODUCT_MANIFEST_GLOB = "hippomaps_gradient_products_*_manifest.json"


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _outdated_by(product: Path, sources: Iterable[Path | str]) -> list[str]:
    """Sources that are missing or newer than the product built from them."""
    problems = []
    for raw in sources:
        source = Path(raw)
        if not source.is_file():
            problems.append(f"missing source: {source}")
        elif source.stat().st_mtime_ns > product.stat().st_mtime_ns:
            problems.append(f"{source} is newer than {product}")
    return problems


def _group_atlas_parents(manifest_path: Path) -> list[str]:
    """The gradient and parent-atlas manifests a group atlas declares as its inputs."""
    manifest = _load(manifest_path)
    reuse = manifest.get("fixed_cluster_reuse") or {}
    declared = (manifest.get("gradient_source_manifest"), manifest.get("source_group_atlas_manifest"),
                reuse.get("source_manifest"))
    return [str(path) for path in declared if path]


def _overview_problems(sidecar: Path) -> tuple[Path, list[str]]:
    payload = _load(sidecar)
    figure = Path(payload["overview_png"])
    if not figure.is_file():
        return figure, [f"missing figure: {figure}"]
    atlas_manifests = [Path(path) for path in payload["source_manifests"]]
    problems = _outdated_by(figure, atlas_manifests)
    for manifest in atlas_manifests:
        if manifest.is_file():
            problems.extend(_outdated_by(manifest, _group_atlas_parents(manifest)))
    return figure, problems


def _gradient_figure_problems(product_manifest: Path) -> tuple[Path, list[str]]:
    payload = _load(product_manifest)
    figure = Path(payload["outputs"]["gradient_figure"])
    if not figure.is_file():
        return figure, [f"missing figure: {figure}"]
    return figure, _outdated_by(figure, payload["source_manifests"].values())


def check_present_products(present_roots: Iterable[Path | str]) -> list[dict[str, object]]:
    """Check every overview and gradient figure recorded under the given roots."""
    results = []
    for raw in present_roots:
        root = Path(raw).resolve()
        if not root.is_dir():
            raise FileNotFoundError(f"Presentation root does not exist: {root}")
        checks = [_overview_problems(path) for path in sorted(root.rglob(OVERVIEW_SIDECAR_GLOB))]
        checks += [_gradient_figure_problems(path) for path in sorted(root.rglob(GRADIENT_PRODUCT_MANIFEST_GLOB))]
        results.extend({"figure": str(figure), "stale": bool(problems), "problems": problems}
                       for figure, problems in checks)
    return results


def print_stale_figures(present_roots: Iterable[Path | str]) -> int:
    """Print each stale figure with its reason; return the number of stale figures."""
    results = check_present_products(present_roots)
    stale = [item for item in results if item["stale"]]
    for item in stale:
        print(f"[stale] {item['figure']}")
        for problem in item["problems"]:
            print(f"    {problem}")
    print(json.dumps({"checked": len(results), "stale": len(stale)}, indent=2))
    return len(stale)


def remind_stale_figures(work_stage_root: Path | str) -> None:
    """After a work stage rewrites `<dataset>/work/<stage>`, name the figures it outdated."""
    stage_root = Path(work_stage_root).resolve()
    present = stage_root.parent.parent / "present"
    if stage_root.parent.name != "work" or not present.is_dir():
        return
    roots = [root for root in (present / "overviews", present / "gradients") if root.is_dir()]
    stale = [item for item in check_present_products(roots) if item["stale"]]
    if stale:
        print(f"[stale] {len(stale)} presentation figure(s) under {present} are now older than their sources; "
              "re-render them (list: 07_01_build_group_matrix.py --check-sources).", flush=True)
