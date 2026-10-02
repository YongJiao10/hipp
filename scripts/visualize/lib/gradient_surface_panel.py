"""Render a gradient surface panel from the same assignment as its bars."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np
from PIL import Image

from lib.gradient_assignment_contract import validate_gradient_assignment
from lib.template_assets import validate_template_assets
from lib.workbench_grid import build_grid_canvas
from lib.workbench_scene_batch import render_scene_batch
from surface_panel_cache import panel_inputs, publish_cache


ROOT = Path(__file__).resolve().parents[3]


def validate_bar_composition(
    entries: list[dict[str, Any]],
    validated: Mapping[str, Any],
    *,
    context: str,
) -> None:
    """Bind displayed bar percentages to the validated surface composition."""
    ids = [item.get("cluster_id") for item in entries]
    if ids != [1, 2, 3]:
        raise ValueError(
            f"Gradient bar composition requires ordered clusters 1/2/3: "
            f"{context}"
        )
    expected_compositions = validated.get("compositions")
    winners = validated.get("winners")
    if not isinstance(expected_compositions, dict) or not isinstance(
        winners, dict
    ):
        raise ValueError(
            f"Gradient assignment result is missing compositions/winners: "
            f"{context}"
        )
    if set(ids) != set(expected_compositions):
        raise ValueError(
            f"Gradient bar composition cluster IDs disagree with validated "
            f"assignment: {context}"
        )

    for entry in entries:
        cid = int(entry["cluster_id"])
        expected = expected_compositions[cid]
        got = entry.get("comp")
        if not isinstance(expected, dict) or not isinstance(got, dict):
            raise ValueError(
                f"Gradient bar/surface composition mismatch cluster {cid}: "
                f"{context}"
            )
        if set(expected) != set(got):
            raise ValueError(
                f"Gradient bar/surface composition mismatch cluster {cid}: "
                f"{context}"
            )
        try:
            mismatch = any(
                not np.isfinite(float(got[name]))
                or not np.isclose(
                    float(expected[name]),
                    float(got[name]),
                    rtol=0,
                    atol=1e-8,
                )
                for name in expected
            )
        except (TypeError, ValueError):
            mismatch = True
        if mismatch:
            raise ValueError(
                f"Gradient bar/surface composition mismatch cluster {cid}: "
                f"{context}"
            )
        if cid not in winners:
            raise ValueError(
                f"Gradient assignment result is missing winner for cluster "
                f"{cid}: {context}"
            )
        entry["dominant_network"] = str(winners[cid])


def _read_manifest(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Invalid gradient group-atlas manifest: {path}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"Gradient group-atlas manifest must be an object: {path}")
    return payload


def render_gradient_surface_panel(
    cohort: Mapping[str, Any],
    subregions: Mapping[str, list[dict[str, Any]]],
    *,
    variant: str,
    smoothing: str = "2mm",
    outdir: Path,
) -> Image.Image:
    """Render explicit canonical inputs and retain provenance sidecars.

    Overview PNGs are never used as scientific surface inputs. The caller
    owns the output directory; this helper only creates or overwrites its
    derived panel and audit files.
    """
    outdir = Path(outdir).resolve()
    grad_subpath = Path(str(cohort["grad_subpath"])).resolve()
    runs = {
        hemi: grad_subpath / f"hemi_{hemi}"
        for hemi in ("L", "R")
    }
    results: dict[str, dict[str, Any]] = {}
    manifests: dict[str, dict[str, Any]] = {}
    assets: dict[str, tuple[Path, str | None, Path | None]] = {}
    sources: list[Path] = [Path(__file__).resolve()]

    for hemi, run_dir in runs.items():
        manifest_path = run_dir / "group_atlas_manifest.json"
        manifest = _read_manifest(manifest_path)
        manifests[hemi] = manifest
        results[hemi] = validate_gradient_assignment(
            run_dir, manifest, variant
        )
        entries = subregions.get(hemi)
        if not isinstance(entries, list):
            raise ValueError(
                f"Missing {hemi} subregion composition entries: {run_dir}"
            )
        validate_bar_composition(
            entries, results[hemi], context=str(run_dir)
        )
        assets[hemi] = validate_template_assets(
            manifest, hemi=hemi, run_dir=run_dir
        )
        sources.extend(
            [
                manifest_path,
                *[Path(path) for path in results[hemi]["paths"]],
                assets[hemi][0],
            ]
        )

    if assets["L"][1:] != assets["R"][1:]:
        raise ValueError(
            f"Gradient surface template subject/spec mismatch: {runs}"
        )
    subject, spec = assets["L"][1:]
    if subject is None or spec is None:
        raise ValueError(
            f"Gradient surface template subject/spec is missing: {runs}"
        )

    scene = ROOT / "config" / "wb_locked_native_view_lateral_medial.scene"
    sources.extend(
        [
            spec,
            scene,
            ROOT / "scripts" / "core_workflow" / "lib" / "gradient_assignment_contract.py",
            ROOT / "scripts" / "core_workflow" / "lib" / "surface_utils.py",
            ROOT / "scripts" / "core_workflow" / "lib" / "template_assets.py",
            ROOT / "scripts" / "core_workflow" / "lib" / "workbench.py",
        ]
    )
    selection = {
        "assignment_variant": variant,
        "cohort": str(cohort["id"]),
        "method": str(cohort["method"]),
        "signal_branch": str(cohort["signal_branch"]),
        "smoothing": str(smoothing),
        "n_subjects": int(cohort["n_subjects"]),
    }
    inputs = panel_inputs(sources, **selection)
    pngs = render_scene_batch(
        scene=scene,
        subjects=[subject],
        outdir=outdir / "scene",
        width=1600,
        height=1024,
        renderer="OSMesa",
        name="gradient_assignment",
        left_label_template=str(results["L"]["label_path"]),
        right_label_template=str(results["R"]["label_path"]),
        left_surface_template=str(assets["L"][0]),
        right_surface_template=str(assets["R"][0]),
        spec_template=str(spec),
        no_template_scene=True,
        retain_scene=True,
    )
    if inputs != panel_inputs(sources, **selection):
        raise ValueError(
            f"Gradient inputs changed during render; refusing publication: "
            f"{outdir}"
        )

    with Image.open(pngs[0]) as native:
        grid = build_grid_canvas(native, None, "1x2").convert("RGB")
    png = outdir / "surface.png"
    grid.save(png)
    publish_cache(png, inputs)
    audit = {
        "selection": selection,
        "hemispheres": {
            hemi: {
                "winners": results[hemi]["winners"],
                "compositions": results[hemi]["compositions"],
            }
            for hemi in ("L", "R")
        },
    }
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "assignment_audit.json").write_text(
        json.dumps(audit, indent=2) + "\n",
        encoding="utf-8",
    )
    return grid
