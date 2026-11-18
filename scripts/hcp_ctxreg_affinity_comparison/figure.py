"""Compose the comparison using the project's standard hippocampal surface panels."""
from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from PIL import Image

from lib.network_colors import NETWORK_DISPLAY_NAMES, load_network_colors
from lib.surface_utils import render_hippocampal_grid

from lib.workbench_grid import build_2x2_composite

from network_annotation import VARIANTS, network_label_path

METHODS = {"mean_fc": "Avg FC", "mean_similarity": "Avg Sim"}


def partition_directory(out, route, smoothing, hemi, method):
    return out / route / smoothing / f"hemi_{hemi}" / method


def render_panel(out, route, smoothing, method, surfaces, variant):
    def label_path(hemi):
        run = partition_directory(out, route, smoothing, hemi, method)
        if variant is None:
            return run / "group_labels_final.label.gii"
        return network_label_path(run, variant)

    render = render_hippocampal_grid(
        subject="group-average",
        outdir=out / "panels" / route / smoothing / method / (variant or "clusters"),
        name="partition",
        title="",
        left_labels=label_path("L"),
        right_labels=label_path("R"),
        left_surface=surfaces["L"],
        right_surface=surfaces["R"],
    )
    with Image.open(render["surface_png"]) as image:
        return build_2x2_composite(image.convert("RGBA"))


def render_comparison(out: Path, route_names, cluster_colors):
    rows = []
    surfaces = {}
    for route in route_names:
        with (out / route / "comparison.tsv").open(newline="") as stream:
            rows.extend(csv.DictReader(stream, delimiter="\t"))
        settings = json.loads((out / route / "run_settings.json").read_text())
        surfaces[route] = {hemi: Path(settings["surfaces"][hemi]) for hemi in ("L", "R")}

    table = {(row["route"], row["smoothing"], row["hemisphere"]): row for row in rows}
    with (out / "comparison.tsv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, list(rows[0]), delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)
    for variant in (None, *VARIANTS):
        render_figure(out, route_names, cluster_colors, surfaces, table, variant)


def render_figure(out, route_names, cluster_colors, surfaces, table, variant):
    combinations = [(route, smoothing) for smoothing in ("2mm", "4mm") for route in route_names]
    has_network_labels = variant is not None
    fig, axes = plt.subplots(2, len(combinations), figsize=(2.22 * len(combinations) + 1, 8.4),
                             squeeze=False, gridspec_kw={"hspace": .24 if has_network_labels else .13, "wspace": .20})
    fig.subplots_adjust(left=.093, right=.98, top=.84, bottom=.22 if has_network_labels else .18)
    ink = "#0b2447"
    visible_networks = set()
    for col, (route, smoothing) in enumerate(combinations):
        for row, method in enumerate(METHODS):
            panel = render_panel(out, route, smoothing, method, surfaces[route], variant)
            ax = axes[row, col]
            ax.imshow(panel)
            ax.set_axis_off()
            if variant is not None:
                labels = []
                for hemi in ("L", "R"):
                    run = partition_directory(out, route, smoothing, hemi, method)
                    status = json.loads((run / f"cluster_assignment_status_{variant}.json").read_text())
                    names = [item["top1_network"] for item in status["items"]]
                    visible_networks.update(names)
                    labels.append(" · ".join(NETWORK_DISPLAY_NAMES.get(name, name) for name in names))
                ax.text(.5, -.055, f"L  {labels[0]}\nR  {labels[1]}", transform=ax.transAxes,
                        ha="center", va="top", color=ink, fontsize=8.5, linespacing=1.3)
        ax = axes[0, col]
        ax.set_title(route_names[route], color=ink, fontsize=10, weight="bold", pad=10)
        box = axes[1, col].get_position()
        left, right = (table[(route, smoothing, hemi)] for hemi in ("L", "R"))
        fig.text(
            (box.x0 + box.x1) / 2, .145,
            f"ARI  {float(left['between_method_ari']):.2f} / {float(right['between_method_ari']):.2f}\n"
            f"Δ gap  {float(left['held_out_gap_delta']):+.4f} / {float(right['held_out_gap_delta']):+.4f}",
            ha="center", va="top", color=ink, fontsize=8.5, linespacing=1.4,
        )
    for row, label in enumerate(METHODS.values()):
        box = axes[row, 0].get_position()
        fig.text(.035, (box.y0 + box.y1) / 2, label, va="center", color=ink, fontsize=11, weight="bold")
    for block, smoothing in enumerate(("2 mm", "4 mm")):
        first = axes[0, block * len(route_names)].get_position()
        last = axes[0, (block + 1) * len(route_names) - 1].get_position()
        fig.text((first.x0 + last.x1) / 2, .898, f"{smoothing} smoothing",
                 ha="center", color=ink, fontsize=12, weight="bold")

    title = VARIANTS[variant] if variant is not None else "Hippocampal partitions from mean FC and mean similarity"
    fig.text(.035, .962, title,
             va="top", fontsize=18, weight="bold", color=ink)
    fig.text(.035, .918, "HCP 7T · Kong2022 Canonical7 · TopSim / Ward", fontsize=10, color=ink)
    if variant is None:
        colors = dict(zip(("C1 Anterior", "C2 Middle", "C3 Posterior"), cluster_colors[1:4], strict=True))
        caption = "Dorsal above, ventral below. Metrics are listed left / right."
    else:
        palette = load_network_colors()
        palette["unassigned"] = (20, 20, 20, 255)
        colors = {NETWORK_DISPLAY_NAMES.get(name, name): color
                  for name, color in palette.items() if name in visible_networks}
        caption = "Dorsal above, ventral below. Networks are listed C1 to C3. Metrics are left / right."
    fig.legend(
        [Patch(color=tuple(c / 255 for c in rgba)) for rgba in colors.values()],
        list(colors), loc="lower left", bbox_to_anchor=(.028, .053),
        ncol=len(colors), frameon=False, fontsize=9,
    )
    fig.text(
        .035, .03,
        caption + " Δ gap is the change in held-out within-parcel minus cross-boundary similarity.",
        color=ink, fontsize=8.5,
    )
    suffix = f"_{variant}" if variant is not None else ""
    fig.savefig(out / f"ctxreg_affinity_averaging{suffix}.png", dpi=220)
    plt.close(fig)
