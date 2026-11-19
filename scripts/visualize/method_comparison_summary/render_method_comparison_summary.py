#!/usr/bin/env python3
"""One-page comparison of hippocampal parcellation methods across routes and smoothings.

Every condition is a card with the same four panels used by the per-condition
group overviews (06_01): subject-split ARI over K, the redundancy gate, the
dominant-network bars, and the unfolded/folded hippocampus. The card's upper
row is the network time-series Fisher-z annotation and the lower row the
network time-series OVR annotation of the same partition; ARI and redundancy
describe the partition and are drawn once.

Rows are methods (gradient methods per component set); columns are routes
(Standard, CtxReg 10 mm, CtxReg 20 mm) within each smoothing. Each run is
resolved by its exact run directory and validated against a single shared
comparison contract before anything is drawn.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import multiprocessing
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch, Rectangle  # noqa: E402
from PIL import Image  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[3]
CORE = REPO_ROOT / "scripts" / "core_workflow"
sys.path.insert(0, str(CORE))

from lib.group_atlas_lineage import validate_group_atlas_lineage  # noqa: E402
from lib.surface_utils import long_axis_order, render_hippocampal_grid  # noqa: E402


def _load_overview_module():
    spec = importlib.util.spec_from_file_location("group_overview", CORE / "06_01_build_group_overview.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


ov = _load_overview_module()

ATLAS = "kong2022"
COVERAGE_TAG = "cov_0p80"
TAXONOMY = "canonical7"
REDUNDANCY_THRESHOLD = 0.80
VARIANTS = (
    ("network_ts_fisherz", "Network-TS Fisher-z"),
    ("ovr_logistic_canonical7_network_ts", "Network-TS OVR"),
)
ROUTES = (("pfm", "Standard"), ("pfm_ctxreg_10mm", "CtxReg 10 mm"), ("pfm_ctxreg", "CtxReg 20 mm"))
SMOOTHINGS = ("2mm", "4mm")
# Settings that must agree for every consumed run.
SHARED_PAYLOAD_KEYS = (
    "subjects_manifest", "instability_resamples", "null_resamples", "k_final_rule", "default_k", "seed",
    "candidate_k_values",
)


@dataclass(frozen=True)
class MethodRow:
    key: str
    label: str
    method: str
    comp_tag: str  # gradient component directory; empty for non-gradient methods
    family: str  # "gradient" or "graph"


METHOD_ROWS = (
    MethodRow("ward_g1", "Ward · avg FC\nG1", "gradient-ward", "g1", "gradient"),
    MethodRow("ward_g1g2", "Ward · avg FC\nG1+G2", "gradient-ward", "g1g2", "gradient"),
    MethodRow("ward_g1g2g3", "Ward · avg FC\nG1–G3", "gradient-ward", "g1g2g3", "gradient"),
    MethodRow("avgsim_g1", "Ward · avg sim\nG1", "mean-similarity-ward", "g1", "gradient"),
    MethodRow("avgsim_g1g2", "Ward · avg sim\nG1+G2", "mean-similarity-ward", "g1g2", "gradient"),
    MethodRow("avgsim_g1g2g3", "Ward · avg sim\nG1–G3", "mean-similarity-ward", "g1g2g3", "gradient"),
    MethodRow("shen", "Shen\nmultigraph", "multigraph-spectral", "", "graph"),
    MethodRow("mean_graph", "Mean-graph\nspectral", "mean-graph-spectral", "", "graph"),
)
FIGURES = {
    "all": ("method_comparison_summary", ("gradient", "graph")),
    "gradient": ("method_comparison_summary_gradient", ("gradient",)),
    "graph": ("method_comparison_summary_graph", ("graph",)),
}

FS = 6.0
HEMI_COLORS = {"L": "#1f77b4", "R": "#d62728"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass
class Condition:
    row: MethodRow
    route: str
    smoothing: str
    run_dirs: dict[str, Path]
    manifests: dict[str, dict]
    rows: dict[str, list[dict[str, str]]]
    surfaces: tuple[Path, Path, str, Path]
    annotation: dict[str, dict]  # variant -> {"label_paths", "bars", "net_names"}

    @property
    def k(self) -> dict[str, int]:
        return {h: int(self.manifests[h]["chosen_k_final"]) for h in "LR"}


def validate_manifest(m: dict, path: Path, *, method: str, route: str, smoothing: str, hemi: str) -> None:
    expected = {
        "status": "ok", "method": method, "atlas": ATLAS, "smoothing": smoothing, "hemi": hemi,
        "signal_control_route": route, "coverage_threshold": 0.8, "network_taxonomy": TAXONOMY,
        "redundancy_threshold": REDUNDANCY_THRESHOLD,
    }
    bad = [f"{key}={m.get(key)!r} (expected {value!r})" for key, value in expected.items() if m.get(key) != value]
    if int(m.get("n_subjects", -1)) != len(m.get("subjects", [])):
        bad.append(f"n_subjects={m.get('n_subjects')} disagrees with subjects ({len(m.get('subjects', []))})")
    if bad:
        raise ValueError(f"Comparison contract mismatch in {path}: " + "; ".join(bad))


def load_condition(roots: dict[str, Path], row: MethodRow, route: str, smoothing: str) -> Condition:
    root = roots[row.family] / route
    run_dirs = {h: ov._build_run_dir(root, row.method, ATLAS, smoothing, COVERAGE_TAG, h, gradient_comp_tag=row.comp_tag)
                for h in "LR"}
    manifests = {}
    for h, run_dir in run_dirs.items():
        path = run_dir / "group_atlas_manifest.json"
        if not path.is_file():
            raise FileNotFoundError(f"Missing group atlas manifest for {row.key} {route} {smoothing} hemi-{h}: {path}")
        m = json.loads(path.read_text(encoding="utf-8"))
        validate_manifest(m, path, method=row.method, route=route, smoothing=smoothing, hemi=h)
        if row.family == "gradient":
            validate_group_atlas_lineage(path, expected_network_taxonomy=TAXONOMY, expected_method=row.method)
        ov.ensure_gradient_assignment_variants(path, m)
        manifests[h] = m
    surfaces = ov.resolve_template_pair_assets(manifests["L"], manifests["R"],
                                               left_run_dir=run_dirs["L"], right_run_dir=run_dirs["R"])
    if surfaces[2] is None or surfaces[3] is None:
        raise ValueError(f"No template render spec for {row.key} {route} {smoothing}")
    k_rows = {h: ov.read_tsv(run_dirs[h] / "k_selection_summary.tsv") for h in "LR"}
    annotation = {}
    for variant, _ in VARIANTS:
        label_paths, bars, net_names = {}, {}, None
        for h in "LR":
            m, d = manifests[h], run_dirs[h]
            validated = ov.validate_gradient_assignment(d, m, variant)
            status_path, label_path = ov.resolve_assignment_variant_paths(d, m, variant)
            labels = np.load(d / "group_labels_final.npy").astype(np.int32)
            k = int(m["chosen_k_final"])
            surface = surfaces[0] if h == "L" else surfaces[1]
            cids = ov.resolve_cluster_order_from_manifest(m, labels, long_axis_order(surface), k)
            names = [str(x) for x in validated["network_names"]]
            if net_names is not None and names != net_names:
                raise ValueError(f"Left/right network orders differ for {row.key} {route} {smoothing} {variant}")
            net_names = names
            if variant == "network_ts_fisherz":
                profile = ov.reorder_profile(np.load(d / "cluster_network_ts_profile.npy").astype(np.float32), cids)
                status = json.loads(status_path.read_text(encoding="utf-8"))
                ov.validate_fisher_status_matches_profile(
                    profile_ordered=profile, status_payload=status, network_names=names, cluster_ids=cids,
                    context=f"{row.key} {route} {smoothing} hemi-{h} {variant}",
                )
                bars[h] = ov.confidence_rows(status, h, cids)
            else:
                bars[h] = ov.reorder_profile(np.asarray(validated["profile"], dtype=float), cids)
            label_paths[h] = label_path
        annotation[variant] = {"label_paths": label_paths, "bars": bars, "net_names": net_names}
    return Condition(row, route, smoothing, run_dirs, manifests, k_rows, surfaces, annotation)


def validate_shared_settings(conditions: list[Condition]) -> dict:
    reference = None
    for cond in conditions:
        for h in "LR":
            m = cond.manifests[h]
            payload = m.get("resume_input_signature_payload", {})
            settings = {key: payload.get(key) for key in SHARED_PAYLOAD_KEYS}
            settings["subjects"] = list(m["subjects"])
            if reference is None:
                reference = (settings, cond.run_dirs[h])
            elif settings != reference[0]:
                diff = sorted(k for k in settings if settings[k] != reference[0][k])
                raise ValueError(
                    f"Runs disagree on comparison settings {diff}: {cond.run_dirs[h]} vs {reference[1]}"
                )
    shared = dict(reference[0])
    shared["n_subjects"] = len(shared.pop("subjects"))
    return shared


def crop_surface_legend(image: np.ndarray) -> np.ndarray:
    """Drop the per-panel network legend on the right of the fold/unfold render."""
    rgb = image[..., :3].astype(float)
    if rgb.max() > 1.0:
        rgb = rgb / 255.0
    content = (rgb < 0.97).any(axis=2).any(axis=0)
    columns = np.flatnonzero(content)
    # The legend is the last content block after a blank gutter of at least 2% of the width.
    gaps = np.flatnonzero(np.diff(columns) > max(8, int(0.02 * image.shape[1])))
    if gaps.size == 0:
        raise ValueError("Could not locate the legend gutter in the hippocampal render")
    right_edge = columns[gaps[-1]] + 6
    left_edge = max(0, columns[0] - 6)
    kept = image[:, left_edge:right_edge]
    rows = np.flatnonzero((rgb[:, left_edge:right_edge] < 0.97).any(axis=2).any(axis=1))
    return kept[max(0, rows[0] - 4): rows[-1] + 5]


def render_surface_job(job: tuple) -> tuple[str, str]:
    key, subject, left_labels, right_labels, left_surface, right_surface, spec_path, outdir, cache = job
    unfolded = ov.resolve_overview_unfolded_assets(Path(left_surface), Path(right_surface), Path(spec_path))
    render = render_hippocampal_grid(
        subject=subject, outdir=Path(outdir), name=key, title="",
        left_labels=Path(left_labels), right_labels=Path(right_labels),
        left_surface=Path(left_surface), right_surface=Path(right_surface),
        legend_display_names=ov.NETWORK_DISPLAY_NAMES, surface_cache_root=Path(cache), unfolded=unfolded,
    )
    cropped = crop_surface_legend(plt.imread(render["foldunfold_png"]))
    png = Path(outdir) / f"{key}.png"
    plt.imsave(png, cropped)
    return key, str(png)


def condition_key(cond: Condition, variant: str) -> str:
    return f"{cond.route}_{cond.smoothing}_{cond.row.key}_{variant}"


def render_surfaces(conditions: list[Condition], panel_dir: Path, jobs: int) -> dict[str, Path]:
    cache = REPO_ROOT / "tmp" / "method-comparison" / "surface_cache"
    tasks = []
    for cond in conditions:
        surf_l, surf_r, subject, spec = cond.surfaces
        for variant, _ in VARIANTS:
            labels = cond.annotation[variant]["label_paths"]
            tasks.append((condition_key(cond, variant), subject, str(labels["L"]), str(labels["R"]),
                          str(surf_l), str(surf_r), str(spec), str(panel_dir), str(cache)))
    panel_dir.mkdir(parents=True, exist_ok=True)
    # Fork before any VTK state exists in the parent; workers only render and write PNGs.
    with ProcessPoolExecutor(max_workers=jobs, mp_context=multiprocessing.get_context("fork")) as pool:
        return {key: Path(path) for key, path in pool.map(render_surface_job, tasks)}


def _shrink(ax) -> None:
    legend = ax.get_legend()
    if legend is not None:
        legend.remove()
    ax.set_title("")
    ax.tick_params(labelsize=FS - 1, length=2, pad=1)
    ax.xaxis.label.set_size(FS - 0.5)
    ax.yaxis.label.set_size(FS - 0.5)
    for text in ax.texts:
        text.set_fontsize(FS - 1.2)


def draw_ari(ax, cond: Condition, *, show_labels: bool) -> None:
    for h in "LR":
        if not ov.plot_ari_summary_curve(ax, cond.rows[h], hemi=h, color=HEMI_COLORS[h]):
            ax.text(0.5, 0.5, f"ARI unavailable hemi-{h}", transform=ax.transAxes, ha="center", fontsize=FS)
        m = cond.manifests[h]
        ax.axvline(int(m["chosen_k_final"]), color=HEMI_COLORS[h], ls="--", lw=0.8)
        auto = m.get("automatic_k_final")
        if auto is not None and int(auto) != int(m["chosen_k_final"]):
            ax.axvline(int(auto), color=HEMI_COLORS[h], ls="-.", lw=0.8)
    ax.set_ylim(0.0, 1.0)
    ax.set_yticks([0, 0.5, 1])
    ax.set_xlabel("K" if show_labels else "")
    ax.set_ylabel("Split-half ARI" if show_labels else "")
    _shrink(ax)
    for line in ax.lines:
        line.set_markersize(2.0)
        line.set_linewidth(min(line.get_linewidth(), 1.0))


def draw_redundancy(ax, cond: Condition, *, show_labels: bool, ylim: tuple[float, float]) -> None:
    ov.plot_k_gate_panel(
        ax, rows_l=cond.rows["L"], rows_r=cond.rows["R"], k_l=cond.k["L"], k_r=cond.k["R"],
        smoothing=cond.smoothing, redundancy_threshold=REDUNDANCY_THRESHOLD, k_label="Primary K",
    )
    ax.set_ylim(*ylim)
    ticks = np.round(np.arange(np.ceil(ylim[0] * 2) / 2, 1.01, 0.5), 1)
    ax.set_yticks(ticks, [f"{t:.1f}" for t in ticks])
    ax.set_xlabel("K" if show_labels else "")
    ax.set_ylabel("Max profile sim." if show_labels else "")
    _shrink(ax)
    for line in ax.lines:
        line.set_markersize(2.0)
        line.set_linewidth(min(line.get_linewidth(), 1.0))


def draw_bars(ax, cond: Condition, variant: str, xmax: float) -> None:
    ann = cond.annotation[variant]
    if variant == "network_ts_fisherz":
        ov.plot_confidence_panel(ax, ann["bars"]["L"] + ann["bars"]["R"], "")
        ax.set_xlabel("Fisher-z")
    else:
        profile = np.vstack([ann["bars"]["L"], ann["bars"]["R"]])
        labels = [f"L C{i + 1}" for i in range(len(ann["bars"]["L"]))] + \
                 [f"R C{i + 1}" for i in range(len(ann["bars"]["R"]))]
        ov.plot_ovr_association_panel(
            ax, profile=profile, row_positions=list(range(len(labels))), row_labels=labels,
            smoothing=cond.smoothing, network_names=ann["net_names"],
        )
        ax.set_xlabel("OVR raw β")
    ax.set_xlim(0.0, xmax)
    compact_bar_labels(ax)
    _shrink(ax)


def compact_bar_labels(ax) -> None:
    """Keep 'L C1' as the tick; the bar colour is the winner, the runner-up is named beside the margin."""
    ticks = ax.get_yticks()
    runner_up = {}
    short = []
    for y, label in zip(ticks, ax.get_yticklabels(), strict=True):
        parts = label.get_text().split(" ", 2)
        short.append(" ".join(parts[:2]))
        rest = parts[2] if len(parts) > 2 else ""
        runner_up[round(float(y), 3)] = rest.split(" | ", 1)[1] if " | " in rest else ""
    ax.set_yticks(ticks, short)
    for text in ax.texts:
        name = runner_up.get(round(float(text.get_position()[1]), 3), "")
        if name and text.get_text():
            text.set_text(f"{text.get_text()} vs {name}")


def draw_surface(ax, png: Path, cond: Condition, variant: str, highlighted: bool = False) -> None:
    ax.imshow(plt.imread(png))
    ax.axis("off")
    k_text = "K " + "/".join(
        f"{cond.k[h]}" + ("*" if cond.manifests[h].get("primary_k_source") == "default_k_undersized_fallback" else "")
        for h in "LR"
    )
    tag = "Fz" if variant == VARIANTS[0][0] else "OVR"
    label = f"{tag} · {k_text}" if variant == VARIANTS[0][0] else tag
    ax.text(0.0, 1.0, label + ("  ✓ L=R" if highlighted else ""), transform=ax.transAxes,
            ha="left", va="bottom", fontsize=FS + 0.5, fontweight="bold",
            color=HIGHLIGHT_COLOR if highlighted else "black")


HIGHLIGHT_COLOR = "#1a9850"
HIGHLIGHT_K = 3


def cluster_networks(cond: Condition, variant: str, hemi: str) -> list[str | None]:
    """Dominant network of each anterior-to-posterior cluster under one annotation."""
    ann = cond.annotation[variant]
    if variant == "network_ts_fisherz":
        return [None if str(r["top1_network"]) in {"", "none", "unassigned"} else str(r["top1_network"])
                for r in ann["bars"][hemi]]
    names = ann["net_names"]
    return [names[int(np.argmax(row))] if np.max(row) > 0 else None for row in ann["bars"][hemi]]


def symmetric_nonredundant(cond: Condition, variant: str) -> bool:
    """Both hemispheres export K=3, pass the redundancy gate there, and match network by cluster."""
    for h in "LR":
        if cond.k[h] != HIGHLIGHT_K:
            return False
        row = next((r for r in cond.rows[h] if int(r["k"]) == HIGHLIGHT_K), None)
        if row is None or str(row.get("nonredundant_ok")) != "1":
            return False
    left, right = cluster_networks(cond, variant, "L"), cluster_networks(cond, variant, "R")
    return None not in left and left == right


def draw_highlight(fig, surface_ax, bar_ax, card_right: float) -> None:
    """Frame the surface/bar row, including its tag above and the bar axis labels below (inches)."""
    w, h = fig.get_size_inches()
    sb, bb = surface_ax.get_position(), bar_ax.get_position()
    x0 = sb.x0 - 0.04 / w
    x1 = card_right - 0.05 / w
    y0 = min(sb.y0, bb.y0) - 0.27 / h
    y1 = max(sb.y1, bb.y1) + 0.13 / h
    fig.add_artist(Rectangle((x0, y0), x1 - x0, y1 - y0, transform=fig.transFigure, fill=False,
                             edgecolor=HIGHLIGHT_COLOR, lw=2.0, zorder=10))


def bar_xmax(conditions: list[Condition], variant: str) -> float:
    values = []
    for cond in conditions:
        bars = cond.annotation[variant]["bars"]
        if variant == "network_ts_fisherz":
            values += [float(r["top1_value"]) for h in "LR" for r in bars[h]]
        else:
            values += [float(np.max(p)) for h in "LR" for p in bars[h] if np.max(p) > 0]
    return max(values) * 1.45 if values else 1.0


def compose(conditions: dict[tuple, Condition], panels: dict[str, Path], rows: list[MethodRow],
            out_png: Path, shared: dict, title: str) -> None:
    columns = [(s, route, name) for s in SMOOTHINGS for route, name in ROUTES]
    card_w, card_h, label_w = 4.6, 2.9, 1.0
    gap = 0.35
    n_groups = sum(1 for i, r in enumerate(rows) if i == 0 or rows[i - 1].family != r.family) - 1
    header_h, legend_h = 0.95, 1.05
    fig_w = label_w + len(columns) * card_w + gap
    fig_h = header_h + len(rows) * card_h + n_groups * 0.25 + legend_h
    fig = plt.figure(figsize=(fig_w, fig_h))
    flat = list(conditions.values())
    xmax = {variant: bar_xmax(flat, variant) for variant, _ in VARIANTS}
    sims = [float(r["max_profile_similarity"]) for c in flat for h in "LR" for r in c.rows[h]
            if r.get("max_profile_similarity") not in (None, "", "nan")]
    red_ylim = (min(0.0, float(np.floor(min(sims) * 5) / 5)) - 0.03, 1.03)

    def x0(col: int) -> float:
        return (label_w + col * card_w + (gap if col >= len(ROUTES) else 0.0)) / fig_w

    fig.text(0.5, 1 - 0.12 / fig_h, title, ha="center", va="top", fontsize=11, fontweight="bold")
    fig.text(0.5, 1 - 0.34 / fig_h,
             f"Each card: ARI and redundancy gate (left); upper row Fz = {VARIANTS[0][1]}, "
             f"lower row OVR = {VARIANTS[1][1]} annotation of the same partition", ha="center", va="top",
             fontsize=7.5, color="#333333")
    for si, smoothing in enumerate(SMOOTHINGS):
        left, right = x0(si * len(ROUTES)), x0((si + 1) * len(ROUTES) - 1) + card_w / fig_w
        y = 1 - 0.58 / fig_h
        fig.text((left + right) / 2, y, smoothing.replace("mm", " mm"), ha="center", va="center", fontsize=10,
                 fontweight="bold")
        fig.add_artist(Line2D([left + 0.01, right - 0.01], [y - 0.12 / fig_h] * 2, color="#555555", lw=0.8))
    for col, (_, _, route_name) in enumerate(columns):
        fig.text(x0(col) + card_w / fig_w / 2, 1 - 0.84 / fig_h, route_name, ha="center", va="center", fontsize=8.5)

    y_top = 1 - header_h / fig_h
    for ri, row in enumerate(rows):
        if ri and rows[ri - 1].family != row.family:
            y_top -= 0.25 / fig_h
            fig.add_artist(Line2D([0.01, 0.99], [y_top + 0.12 / fig_h] * 2, color="#999999", lw=0.8, ls="--"))
        y_bottom = y_top - card_h / fig_h
        fig.text(0.008, (y_top + y_bottom) / 2, row.label, ha="left", va="center", fontsize=8,
                 fontweight="bold", linespacing=1.15)
        for col, (smoothing, route, _) in enumerate(columns):
            cond = conditions[(row.key, route, smoothing)]
            left = x0(col)
            gs = fig.add_gridspec(
                2, 3, left=left + 0.36 / fig_w, right=left + (card_w - 0.12) / fig_w,
                bottom=y_bottom + 0.30 / fig_h, top=y_top - 0.10 / fig_h,
                width_ratios=[0.95, 1.3, 0.95], wspace=0.32, hspace=0.38,
            )
            draw_ari(fig.add_subplot(gs[0, 0]), cond, show_labels=col == 0)
            draw_redundancy(fig.add_subplot(gs[1, 0]), cond, show_labels=col == 0, ylim=red_ylim)
            for vi, (variant, _) in enumerate(VARIANTS):
                key = condition_key(cond, variant)
                highlighted = symmetric_nonredundant(cond, variant)
                surface_ax = fig.add_subplot(gs[vi, 1])
                draw_surface(surface_ax, panels[key], cond, variant, highlighted)
                bar_ax = fig.add_subplot(gs[vi, 2])
                draw_bars(bar_ax, cond, variant, xmax[variant])
                bar_ax.tick_params(axis="y", pad=1)
                if highlighted:
                    draw_highlight(fig, surface_ax, bar_ax, left + (card_w - 0.03) / fig_w)
            fig.add_artist(Rectangle((left + 0.03 / fig_w, y_bottom + 0.04 / fig_h),
                                     (card_w - 0.06) / fig_w, (card_h - 0.06) / fig_h,
                                     transform=fig.transFigure, fill=False, edgecolor="#d8d8d8", lw=0.6))
        y_top = y_bottom

    networks = sorted({name for cond in flat for v, _ in VARIANTS for name in cond.annotation[v]["net_names"]},
                      key=lambda n: list(ov.NETWORK_BAR_COLORS).index(n) if n in ov.NETWORK_BAR_COLORS else 99)
    net_handles = [Patch(facecolor=ov.NETWORK_BAR_COLORS[n], label=f"{ov.network_display_name(n)} = {n}")
                   for n in networks]
    line_handles = [
        Line2D([0], [0], color=HEMI_COLORS["L"], marker="o", ms=3, lw=1, label="Left hemisphere"),
        Line2D([0], [0], color=HEMI_COLORS["R"], marker="o", ms=3, lw=1, label="Right hemisphere"),
        Line2D([0], [0], color="#555555", ls=":", lw=1, label="ARI null 95th percentile"),
        Line2D([0], [0], color="#b00020", ls=":", lw=1, label=f"Redundancy threshold {REDUNDANCY_THRESHOLD:.2f}"),
        Line2D([0], [0], color="#555555", ls="--", lw=1, label="Exported K"),
        Line2D([0], [0], color="#555555", ls="-.", lw=1, label="Automatic K (if different)"),
        Rectangle((0, 0), 1, 1, facecolor="#777777", label="Bar: winning network"),
        Rectangle((0, 0), 1, 1, facecolor="none", edgecolor="#111111", label="Outline: runner-up"),
        Rectangle((0, 0), 1, 1, facecolor="none", edgecolor=HIGHLIGHT_COLOR, lw=2.4,
                  label=f"✓ L=R: both K={HIGHLIGHT_K}, same network per cluster, nonredundant at K={HIGHLIGHT_K}"),
    ]
    fig.legend(handles=net_handles, loc="lower left", bbox_to_anchor=(0.01, 0.004), ncol=len(net_handles),
               fontsize=7, frameon=False, title=f"Kong2022 {TAXONOMY} networks", title_fontsize=7.5)
    fig.legend(handles=line_handles, loc="lower right", bbox_to_anchor=(0.99, 0.004), ncol=3, fontsize=7,
               frameon=False)
    fig.text(
        0.5, 0.004 + 0.80 / fig_h,
        f"HCP 7T, N={shared['n_subjects']}; K scan {min(shared['candidate_k_values'])}–{max(shared['candidate_k_values'])}, "
        f"default K={shared['default_k']} (* = K=2 exported because a K=3 parcel was below the minimum size); "
        f"{shared['instability_resamples']} split-half and {shared['null_resamples']} null resamples; "
        "surfaces show L unfolded | L folded | R folded | R unfolded, anterior at top.",
        ha="center", va="bottom", fontsize=6.5, color="#333333",
    )
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=200, facecolor="white")
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--gradient-root", type=Path, required=True,
                        help="Group-atlas comparison root holding <route>/gradient-ward and mean-similarity-ward runs")
    parser.add_argument("--graph-root", type=Path, required=True,
                        help="Group-atlas comparison root holding <route>/multigraph-spectral and mean-graph-spectral runs")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--jobs", type=int, default=16)
    args = parser.parse_args()
    roots = {"gradient": args.gradient_root.resolve(strict=True), "graph": args.graph_root.resolve(strict=True)}
    out = args.out.resolve()

    conditions = {
        (row.key, route, smoothing): load_condition(roots, row, route, smoothing)
        for row in METHOD_ROWS for route, _ in ROUTES for smoothing in SMOOTHINGS
    }
    shared = validate_shared_settings(list(conditions.values()))
    panels = render_surfaces(list(conditions.values()), out / "panels", args.jobs)

    outputs = {}
    for name, (stem, families) in FIGURES.items():
        rows = [row for row in METHOD_ROWS if row.family in families]
        subset = {key: c for key, c in conditions.items() if c.row in rows}
        png = out / f"{stem}.png"
        compose(subset, panels, rows, png, shared,
                "Hippocampal parcellation methods — Standard vs CtxReg 10 mm vs CtxReg 20 mm (HCP, Kong2022 canonical7)")
        outputs[name] = str(png)

    manifest = {
        "figures": outputs,
        "n_subjects": shared["n_subjects"],
        "shared_settings": shared,
        "redundancy_threshold": REDUNDANCY_THRESHOLD,
        "annotation_variants": [v for v, _ in VARIANTS],
        "roots": {k: str(v) for k, v in roots.items()},
        "conditions": [
            {
                "row": c.row.key, "method": c.row.method, "gradient_components": c.row.comp_tag or None,
                "route": c.route, "smoothing": c.smoothing,
                "chosen_k_final": c.k,
                "primary_k_source": {h: c.manifests[h].get("primary_k_source") for h in "LR"},
                "automatic_k_final": {h: c.manifests[h].get("automatic_k_final") for h in "LR"},
                "manifests": {h: str(c.run_dirs[h] / "group_atlas_manifest.json") for h in "LR"},
                "label_sha256": {v: {h: sha256(Path(c.annotation[v]["label_paths"][h])) for h in "LR"}
                                 for v, _ in VARIANTS},
                "cluster_networks": {v: {h: cluster_networks(c, v, h) for h in "LR"} for v, _ in VARIANTS},
                "symmetric_nonredundant_k3": {v: symmetric_nonredundant(c, v) for v, _ in VARIANTS},
            }
            for c in conditions.values()
        ],
    }
    (out / "method_comparison_summary_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    for c in conditions.values():
        passed = [name for v, name in VARIANTS if symmetric_nonredundant(c, v)]
        if passed:
            print(f"L=R K={HIGHLIGHT_K}: {c.row.key:14s} {c.route:16s} {c.smoothing}  {', '.join(passed)}  "
                  f"{cluster_networks(c, VARIANTS[0][0] if passed[0] == VARIANTS[0][1] else VARIANTS[1][0], 'L')}")
    print(json.dumps(outputs, indent=2))
    return 0


if __name__ == "__main__":
    os.environ.setdefault("LP_NUM_THREADS", "1")
    os.environ.setdefault("VTK_SMP_MAX_THREADS", "1")
    raise SystemExit(main())
