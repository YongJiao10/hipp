"""Annotate completed joint model fits using the established Kong2022 pipeline."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import compare as baseline
import numpy as np

from lib import kong2022_label_utils
from lib.network_colors import require_network_colors
from network_annotation import OVR_NET_TS, OVR_ROI, annotate_partition
from compare_subject_models import METHODS, SENSITIVITY

PRIMARY_METHODS = {"shen_joint": "Shen joint spectral", "hierarch_bayes": "HierarchBayesParcel"}


def annotate(out, routes, methods):
    atlas = baseline.atlas
    table, unavailable = [], []
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
                    missing = [
                        (sub, cid) for sub, mask in masks.items() for cid in range(1, baseline.K + 1)
                        if not np.any(mask & (labels == cid))
                    ]
                    if missing:
                        _, mean_profile, _, _ = atlas.cluster_roi_profile_subjectwise(
                            group_labels_final=labels, support_mask=labels > 0, subjects=subjects,
                            shared_root=Path(settings["fc_root"]), surface_root=ts_root,
                            reference_root=reference_root, smoothing=smoothing, hemi=hemi,
                            valid_masks_by_subject=masks,
                        )
                        np.save(run / "cluster_roi_profile.npy", mean_profile)
                        unavailable.append({
                            "route": route, "smoothing": smoothing, "hemisphere": hemi, "method": method,
                            "variants": [OVR_NET_TS, OVR_ROI], "missing_subject_parcels": missing,
                        })
                        print(f"  OVR requires the full roster, missing {len(missing)} subject-parcel rows", flush=True)
                    else:
                        baseline.annotate(run, labels, subjects, Path(settings["fc_root"]), ts_root,
                                          reference_root, masks, smoothing, hemi)
                    variants = annotate_partition(
                        run, subjects, ts_root, reference_root, masks, smoothing, hemi, names, registry, colors,
                        include_ovr=not missing,
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
    (out / "network_annotation_unavailable.json").write_text(json.dumps(unavailable, indent=2) + "\n")


def render(out, routes):
    from figure import render_panel
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
    from network_annotation import VARIANTS
    from lib.network_colors import NETWORK_DISPLAY_NAMES

    fit_failures = {
        (r["route"], r["smoothing"], r["hemisphere"], m)
        for r in json.loads((out / "fit_failures.json").read_text()) if r["fold"] == "all"
        for m in (("shen_joint",) if r["method"] == "shen_graphs" else (r["method"],))
    }
    annotation_failures = {
        (r["route"], r["smoothing"], r["hemisphere"], r["method"], variant)
        for r in json.loads((out / "network_annotation_unavailable.json").read_text())
        for variant in r["variants"]
    }

    columns = []
    for route in routes:
        settings = json.loads((out / route / "run_settings.json").read_text())
        for smoothing in settings["smoothing"]:
            columns.append((route, smoothing, settings["surfaces"]))
    variants = {None: "C1–C3 along the hippocampal long axis", **VARIANTS}
    for variant, title in variants.items():
        fig, axes = plt.subplots(len(PRIMARY_METHODS), len(columns), squeeze=False,
                                 figsize=(2.7 * len(columns) + 1, 6.7), layout="constrained")
        for col, (route, smoothing, surfaces) in enumerate(columns):
            axes[0, col].set_title(f"{baseline.ROUTES[route]} · {smoothing.replace('mm', ' mm')}", fontsize=10)
            for row, (method, label) in enumerate(PRIMARY_METHODS.items()):
                ax = axes[row, col]
                ax.set_axis_off()
                absent = []
                for hemi in ("L", "R"):
                    key = (route, smoothing, hemi, method)
                    if key in fit_failures or (*key, variant) in annotation_failures:
                        absent.append(hemi)
                panel = render_panel(out, route, smoothing, method, surfaces, variant, unavailable=absent)
                ax.imshow(panel)
                if absent:
                    ax.text(.5, -.01, " / ".join(absent) + " unavailable", transform=ax.transAxes,
                            ha="center", va="top", fontsize=8)
                elif variant:
                    names = []
                    for hemi in ("L", "R"):
                        run = out / route / smoothing / f"hemi_{hemi}" / method
                        status = json.loads((run / f"cluster_assignment_status_{variant}.json").read_text())
                        parcel_names = [item["top1_network"] for item in status["items"]]
                        names.append(hemi + "  " + " · ".join(NETWORK_DISPLAY_NAMES.get(x, x) for x in parcel_names))
                    ax.text(.5, -.01, "\n".join(names), transform=ax.transAxes, ha="center", va="top", fontsize=7)
                if col == 0:
                    ax.text(-.03, .50, label, transform=ax.transAxes, ha="right", va="center", fontsize=10)
        fig.suptitle(f"{title} · HCP 7T, Kong2022 Canonical7", fontsize=14)
        if variant is None:
            fig.legend([Patch(color=np.asarray(c) / 255) for c in baseline.CLUSTER_COLORS[1:]],
                       ["C1 Anterior", "C2 Middle", "C3 Posterior"], loc="outside lower center", ncol=3)
        fig.savefig(out / f"subject_models_{variant or 'clusters'}.png", dpi=200)
        plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--routes", nargs="+", choices=tuple(baseline.ROUTES), default=list(baseline.ROUTES))
    parser.add_argument("--methods", nargs="+", choices=(*METHODS, SENSITIVITY), default=list(PRIMARY_METHODS))
    parser.add_argument("--render-only", action="store_true")
    args = parser.parse_args()
    out = args.out.resolve(strict=True)
    if not out.is_relative_to(baseline.REPO_ROOT / "tmp"):
        parser.error("--out must be under this worktree's tmp directory")
    if args.render_only:
        render(out, args.routes)
    else:
        annotate(out, args.routes, args.methods)


if __name__ == "__main__":
    main()
