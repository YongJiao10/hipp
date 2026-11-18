"""Annotate completed joint model fits using the established Kong2022 pipeline."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import compare as baseline
import nibabel as nib
import numpy as np

from lib import kong2022_label_utils
from lib.network_colors import require_network_colors
from network_annotation import annotate_partition
from compare_subject_models import METHODS, SENSITIVITY

PRIMARY_METHODS = {"shen_joint": "Shen joint spectral", "hierarch_bayes": "HierarchBayesParcel"}
SHEN_DIAGNOSTIC_ROUTES = ("pfm_ctxreg_10mm", "pfm_ctxreg")


def annotate_fit(run, labels, subjects, fc_root, ts_root, ref_root, masks, smoothing, hemi,
                 names, registry, colors):
    cluster_ids = np.unique(labels[labels > 0])
    observed = np.asarray([
        [np.any(masks[sub] & (labels == cid)) for cid in cluster_ids] for sub in subjects
    ], dtype=bool)
    baseline.annotate(
        run, labels, subjects, fc_root, ts_root, ref_root, masks, smoothing, hemi,
        observed_profiles=observed,
    )
    return annotate_partition(
        run, subjects, ts_root, ref_root, masks, smoothing, hemi, names, registry, colors,
        observed_profiles=observed,
    )


def annotate(out, routes, methods):
    atlas = baseline.atlas
    table = []
    for route in routes:
        route_root = out / route
        settings = json.loads((route_root / "run_settings.json").read_text())
        subjects = settings["subjects"]
        reference_root = Path(settings["reference_root"])
        ts_root = Path(settings["hippocampal_timeseries_root"])
        atlas.KONG2022_DLABEL_OVERRIDE = Path(settings["kong2022_dlabel"]).resolve(strict=True)
        kong2022_label_utils.KONG2022_DLABEL_PATH = atlas.KONG2022_DLABEL_OVERRIDE
        names = list(atlas.canonical_families_7_order())
        registry = atlas.load_kong2022_roi_registry()
        colors = require_network_colors(names)
        for hemi in ("L", "R"):
            masks = {
                sub: baseline.gradients.subject_valid_mask(Path(settings["mask_root"]), sub, hemi)
                for sub in subjects
            }
            for smoothing in settings["smoothing"]:
                condition = route_root / smoothing / f"hemi_{hemi}"
                failures = json.loads((condition / "fit_failures.json").read_text())
                blocked = {row["method"] for row in failures if row["fold"] == "all"}
                if "shen_graphs" in blocked:
                    blocked.update(("shen_joint", "shen_mean_graph"))
                for method in methods:
                    if method in blocked or (method == SENSITIVITY and (route, smoothing) != ("pfm", "2mm")):
                        continue
                    print(f"[{route} {smoothing} {hemi}] annotating {method}", flush=True)
                    run = condition / method
                    labels = np.load(run / "group_labels_final.npy")
                    variants = annotate_fit(
                        run, labels, subjects, Path(settings["fc_root"]), ts_root,
                        reference_root, masks, smoothing, hemi, names, registry, colors,
                    )
                    for variant, rows in variants.items():
                        for row in rows:
                            table.append({
                                "route": route, "smoothing": smoothing, "hemisphere": hemi, "method": method,
                                "variant": variant, "cluster_id": row["cluster_id"], "network": row["top1_network"],
                                "status": row["status"],
                                "score": row.get("top1_beta" if variant.startswith("ovr_") else "top1_fisher_z"),
                            })
    baseline.write_table(out / "network_labels.tsv", table)
    (out / "network_annotation_unavailable.json").write_text("[]\n")


def annotate_shen_diagnostics(out, routes, diagnostic):
    atlas = baseline.atlas
    for route in SHEN_DIAGNOSTIC_ROUTES:
        if route not in routes:
            continue
        run = diagnostic / route
        source = run / "shen_converged_hemi-L.label.gii"
        labels = np.asarray(nib.load(source).darrays[0].data, dtype=np.int32)
        if set(np.unique(labels)) != {1, 2}:
            raise ValueError(f"Expected the two populated Shen parcels at {source}")

        settings = json.loads((out / route / "run_settings.json").read_text())
        subjects = settings["subjects"]
        ts_root = Path(settings["hippocampal_timeseries_root"])
        ref_root = Path(settings["reference_root"])
        atlas.KONG2022_DLABEL_OVERRIDE = Path(settings["kong2022_dlabel"]).resolve(strict=True)
        kong2022_label_utils.KONG2022_DLABEL_PATH = atlas.KONG2022_DLABEL_OVERRIDE
        names = list(atlas.canonical_families_7_order())
        registry = atlas.load_kong2022_roi_registry()
        colors = require_network_colors(names)
        masks = {
            sub: baseline.gradients.subject_valid_mask(Path(settings["mask_root"]), sub, "L")
            for sub in subjects
        }
        print(f"[{route} 2mm L] annotating the converged Shen diagnostic", flush=True)
        np.save(run / "group_labels_final.npy", labels)
        annotate_fit(run, labels, subjects, Path(settings["fc_root"]), ts_root,
                     ref_root, masks, "2mm", "L", names, registry, colors)


def render(out, routes, *, shen_diagnostic=None, variants=None):
    from figure import render_panel
    from PIL import Image
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
    from network_annotation import VARIANTS, network_label_path
    from lib.workbench_grid import build_2x2_composite
    from lib.network_colors import NETWORK_DISPLAY_NAMES

    fit_failures = {
        (r["route"], r["smoothing"], r["hemisphere"], m)
        for r in json.loads((out / "fit_failures.json").read_text()) if r["fold"] == "all"
        for m in (("shen_joint",) if r["method"] == "shen_graphs" else (r["method"],))
    }
    annotation_failures = {
        (r["route"], r["smoothing"], r["hemisphere"], r["method"], variant): len(r["missing_subject_parcels"])
        for r in json.loads((out / "network_annotation_unavailable.json").read_text())
        for variant in r["variants"]
    }

    columns = []
    for route in routes:
        settings = json.loads((out / route / "run_settings.json").read_text())
        for smoothing in settings["smoothing"]:
            columns.append((route, smoothing, settings["surfaces"]))
    titles = {None: "Parcels ordered along the hippocampal long axis", **VARIANTS}
    for variant in titles if variants is None else variants:
        title = titles[variant]
        fig, axes = plt.subplots(len(PRIMARY_METHODS), len(columns), squeeze=False,
                                 figsize=(2.7 * len(columns) + 1, 6.7), layout="constrained")
        for col, (route, smoothing, surfaces) in enumerate(columns):
            axes[0, col].set_title(f"{baseline.ROUTES[route]} · {smoothing.replace('mm', ' mm')}", fontsize=10)
            for row, (method, label) in enumerate(PRIMARY_METHODS.items()):
                ax = axes[row, col]
                ax.set_axis_off()
                use_diagnostic = (
                    shen_diagnostic is not None and route in SHEN_DIAGNOSTIC_ROUTES
                    and smoothing == "2mm" and method == "shen_joint"
                )
                diagnostic_runs = {"L": shen_diagnostic / route} if use_diagnostic else {}
                absent, label_paths, notes = [], {}, []
                for hemi in ("L", "R"):
                    key = (route, smoothing, hemi, method)
                    if hemi in diagnostic_runs:
                        run = diagnostic_runs[hemi]
                        if key not in fit_failures:
                            raise ValueError(f"Expected a failed Shen three-parcel fit at {key}")
                        label_paths[hemi] = (
                            run / "shen_converged_hemi-L.label.gii" if variant is None
                            else network_label_path(run, variant)
                        )
                    elif key in fit_failures or (*key, variant) in annotation_failures:
                        absent.append(hemi)
                        missing = annotation_failures.get((*key, variant))
                        if missing is None:
                            message = "partition unavailable"
                        elif missing == 1:
                            message = "OVR unavailable (one missing profile)"
                        else:
                            message = f"OVR unavailable ({missing} missing profiles)"
                        notes.append(f"{hemi}  {message}")
                        continue
                    else:
                        run = out / route / smoothing / f"hemi_{hemi}" / method

                    if variant is not None:
                        status = json.loads((run / f"cluster_assignment_status_{variant}.json").read_text())
                        networks = [item["top1_network"] for item in status["items"]]
                        network_names = " · ".join(NETWORK_DISPLAY_NAMES.get(x, x) for x in networks)
                        notes.append(f"{hemi}  {network_names}")
                    if hemi in diagnostic_runs:
                        notes.append(f"{hemi}  2 of 3 labels occupied (diagnostic)")

                if shen_diagnostic is None or diagnostic_runs or absent:
                    panel = render_panel(
                        out, route, smoothing, method, surfaces, variant,
                        unavailable=tuple(absent), label_paths=label_paths,
                    )
                else:
                    # Targeted regeneration reuses the exact existing colored panels.
                    native = (
                        out / "panels" / route / smoothing / method / (variant or "clusters")
                        / "sub-group-average_partition_surface.png"
                    )
                    with Image.open(native) as image:
                        panel = build_2x2_composite(image.convert("RGBA"))
                ax.imshow(panel)
                if notes:
                    ax.text(.5, -.01, "\n".join(notes), transform=ax.transAxes,
                            ha="center", va="top", fontsize=7)
                if col == 0:
                    ax.text(-.03, .50, label, transform=ax.transAxes, ha="right", va="center", fontsize=10)
        fig.suptitle(f"{title} · HCP 7T, Kong2022 Canonical7", fontsize=14)
        if variant is None:
            fig.legend([Patch(color=np.asarray(c) / 255) for c in baseline.CLUSTER_COLORS[1:]],
                       ["C1", "C2", "C3"], loc="outside lower center", ncol=3)
        fig.savefig(out / f"subject_models_{variant or 'clusters'}.png", dpi=200)
        plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--routes", nargs="+", choices=tuple(baseline.ROUTES), default=list(baseline.ROUTES))
    parser.add_argument("--methods", nargs="+", choices=(*METHODS, SENSITIVITY), default=list(PRIMARY_METHODS))
    parser.add_argument("--render-only", action="store_true")
    parser.add_argument("--shen-diagnostic", type=Path, help="Explicit source of converged Shen diagnostic labels")
    args = parser.parse_args()
    out = args.out.resolve(strict=True)
    if not out.is_relative_to(baseline.REPO_ROOT / "tmp"):
        parser.error("--out must be under this worktree's tmp directory")
    source = args.shen_diagnostic.resolve(strict=True) if args.shen_diagnostic is not None else None
    if source is not None and not source.is_relative_to(out):
        parser.error("--shen-diagnostic must be under the experiment output")
    if args.render_only:
        render(out, args.routes, shen_diagnostic=source)
    elif source is not None:
        annotate_shen_diagnostics(out, args.routes, source)
    else:
        annotate(out, args.routes, args.methods)


if __name__ == "__main__":
    main()
