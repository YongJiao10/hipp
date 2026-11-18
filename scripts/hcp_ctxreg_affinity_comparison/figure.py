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

from lib.surface_utils import render_hippocampal_grid

from lib.workbench_grid import build_2x2_composite

METHODS = {"mean_fc": "Avg FC", "mean_similarity": "Avg Sim"}


def render_panel(out, route, smoothing, method, surfaces):
    def label_path(hemi):
        return out / route / smoothing / f"hemi_{hemi}" / method / "group_labels_final.label.gii"

    render = render_hippocampal_grid(
        subject="group-average",
        outdir=out / "panels" / route / smoothing / method,
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
    combinations = [(route, smoothing) for smoothing in ("2mm", "4mm") for route in route_names]
    fig, axes = plt.subplots(2, len(combinations), figsize=(2.22 * len(combinations) + 1, 8.4),
                             squeeze=False, gridspec_kw={"hspace": .13, "wspace": .20})
    fig.subplots_adjust(left=.093, right=.98, top=.84, bottom=.18)
    ink = "#0b2447"
    for col, (route, smoothing) in enumerate(combinations):
        for row, method in enumerate(METHODS):
            panel = render_panel(out, route, smoothing, method, surfaces[route])
            ax = axes[row, col]
            ax.imshow(panel)
            ax.set_axis_off()
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

    fig.text(.035, .962, "Hippocampal partitions from mean FC and mean similarity",
             va="top", fontsize=18, weight="bold", color=ink)
    fig.text(.035, .918, "HCP 7T · Kong2022 cortical ROIs · TopSim / Ward", fontsize=10, color=ink)
    labels = ("C1 Anterior", "C2 Middle", "C3 Posterior")
    fig.legend(
        [Patch(color=tuple(c / 255 for c in rgba)) for rgba in cluster_colors[1:4]],
        labels, loc="lower left", bbox_to_anchor=(.028, .053), ncol=3, frameon=False, fontsize=9,
    )
    fig.text(
        .035, .03,
        "Dorsal above, ventral below. Metrics are listed left / right. "
        "Δ gap is the change in held-out within-parcel minus cross-boundary similarity.",
        color=ink, fontsize=8.5,
    )
    fig.savefig(out / "ctxreg_affinity_averaging.png", dpi=220)
    plt.close(fig)
    with (out / "comparison.tsv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, list(rows[0]), delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)
