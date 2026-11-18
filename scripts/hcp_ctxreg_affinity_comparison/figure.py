"""Render the paired hippocampal maps from the affinity-averaging comparison."""
from __future__ import annotations

import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import PolyCollection
from matplotlib.patches import Patch


def draw_surface(ax, coords, triangles, labels, colors):
    """Project the coronal-oblique mesh with anterior at left."""
    corners = coords[triangles]
    midpoints = (corners + np.roll(corners, -1, axis=1)) / 2
    centroids = np.broadcast_to(corners.mean(axis=1, keepdims=True), corners.shape)
    polygons = np.stack((corners, midpoints, centroids, np.roll(midpoints, 1, axis=1)), axis=2)
    polygons = polygons.reshape(-1, 4, 3)
    # Both hemispheres are viewed from their medial side.
    depth = polygons[:, :, 0].mean(axis=1) * -np.sign(coords[:, 0].mean())
    order = np.argsort(depth)
    projected = polygons[order][:, :, [1, 2]] * [-1, 1]
    fill = colors[labels[triangles].ravel()[order]]
    ax.add_collection(PolyCollection(projected, facecolors=fill, edgecolors=fill, linewidths=0.2))
    ax.autoscale_view()
    ax.set_aspect("equal")
    ax.axis("off")


def render_comparison(out: Path, route_names, methods, surfaces, cluster_colors):
    rows = []
    for route in route_names:
        with (out / route / "comparison.tsv").open(newline="") as stream:
            rows.extend(csv.DictReader(stream, delimiter="\t"))
    with (out / "comparison.tsv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, list(rows[0]), delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)

    table = {(row["route"], row["smoothing"], row["hemisphere"]): row for row in rows}
    n_rows = 2 * len(route_names)
    fig, axs = plt.subplots(n_rows, 7, figsize=(16, n_rows * 1.72 + 1.35), squeeze=False,
                            gridspec_kw={"width_ratios": [1.35, 2.1, 2.1, 1.3, 2.1, 2.1, 1.3]})
    fig.patch.set_facecolor("#FFFFFF")
    colors = np.vstack(([0.88, 0.90, 0.92, 1.0], np.asarray(cluster_colors[1:4]) / 255))
    for i, (route, smoothing) in enumerate((r, sm) for r in route_names for sm in ("2mm", "4mm")):
        axs[i, 0].axis("off")
        axs[i, 0].text(0, .53, f"{route_names[route]}\n{smoothing.replace('mm', ' mm')} smoothing",
                       va="center", color="#162638", fontsize=11, weight="semibold", linespacing=1.5)
        for start, hemi in ((1, "L"), (4, "R")):
            coords, faces = surfaces[hemi]
            for j, method in enumerate(methods):
                path = out / route / smoothing / f"hemi_{hemi}" / method / "group_labels_final.npy"
                draw_surface(axs[i, start + j], coords, faces, np.load(path), colors)
            row = table[(route, smoothing, hemi)]
            ax = axs[i, start + 2]
            ax.axis("off")
            delta = float(row["held_out_gap_delta"])
            ax.text(.02, .57, f"ARI  {float(row['between_method_ari']):.2f}\nΔ gap  {delta:+.4f}",
                    va="center", fontsize=10, color="#162638", linespacing=1.5)
    for index, title in enumerate(("", "Left · Mean FC", "Left · Mean similarity", "",
                                   "Right · Mean FC", "Right · Mean similarity", "")):
        axs[0, index].set_title(title, fontweight="semibold", color="#162638", fontsize=11, pad=5)
    fig.text(.04, .976, "How averaging order changes hippocampal parcel boundaries",
             va="top", fontsize=20, weight="semibold", color="#162638")
    fig.text(.04, .921, "HCP 7T · Kong2022 cortical ROIs · TopSim / Ward · Three parcels per hemisphere",
             va="top", fontsize=11, color="#506174")
    legend = ["C1 Anterior", "C2 Middle", "C3 Posterior", "Low coverage"]
    fig.legend([Patch(color=colors[i]) for i in (1, 2, 3, 0)], legend,
               loc="lower left", bbox_to_anchor=(.037, .045), ncol=4, frameon=False, fontsize=10)
    fig.text(.04, .028, "Medial projection, anterior at left.  ARI compares the paired maps.  "
             "Δ gap = change in held-out within-parcel minus cross-boundary similarity.",
             fontsize=10, color="#506174")
    fig.subplots_adjust(left=.04, right=.99, top=.887, bottom=.093, hspace=.17, wspace=.06)
    fig.savefig(out / "ctxreg_affinity_averaging.png", dpi=220)
    plt.close(fig)
