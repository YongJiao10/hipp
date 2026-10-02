#!/usr/bin/env python3
from __future__ import annotations

"""
Process Step Description:
Render task-activation group products. Paints group-mean contrast and beta
maps onto group-average midthickness surfaces as GIFTI overlays and renders
one summary PNG per task (profile / bars rows stacked per contrast):
summary top row long-axis profile lines (weighted beta sides + contrast,
no CI bands; bins numbered 1-20 head to tail),
bottom row bar+scatter cluster plots (contrast, hcp151_atlas + template_matching,
Ant/Mid/Post, light fills with dark dots, single red 95% CI error bars,
FDR markers, black paired-subregion brackets with range-scaled end caps,
one BH-FDR family; axes tops expand so markers never overflow).
Hemispheres share y-scales; panels annotate N, coverage, correction;
bins are PCA-ordered equal-vertex bins (not equidistant anatomy); beta units
are raw BOLD (a.u.), never % signal change.

Significance markers (BH-FDR two-tailed): '+' 0.01<=p<0.05, '*' 0.001<=p<0.01,
'**' 0.0001<=p<0.001, '***' p<0.0001. Analysis and visualization cover 2mm
only; overlays sit directly under the task present root (no smoothing folder).

Main Flow Path:
- Stage-06 task branch (task presentation):
  05_06 (group maps + tables) -> 06_11_render_task_activation_products.py (this step).

Inputs:
- Command-line arguments:
  - --task: Task branch (required).
  - --group-root: 05_06 group output root (default: `outputs_hcp/work/task_activation/task-<task>/group/`).
  - --group-average-surfaces-dir: Group-average midthickness surfaces.
  - --out-root: Stable presentation root (default: `outputs_hcp/present/task_activation/task-<task>/`). Reruns refresh the same files.
  - --hemis / --contrasts: Subsets to render; each defaults to auto-discovery from the group root.
- Physical inputs:
  - Group maps, `group_cluster_summary.tsv`, and per-subject data backing 05_06.

Outputs:
- Command-line output / file outputs:
  - GIFTI overlays: `{out_root}/hemi_{hemi}_{group_contrast_{name}_mean,group_beta_{stem}_mean}.func.gii`.
  - Summary PNG: `{out_root}/task_{task}.png` (per contrast: profiles; bars with paired-subregion brackets).
  - `{out_root}/task_activation_products_manifest.json`.
"""
import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.gridspec as gridspec
import matplotlib.patches as patches
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
from nibabel.gifti import GiftiDataArray, GiftiImage
from scipy.stats import t as t_dist

from lib.surface_utils import equal_vertex_long_axis_bins, long_axis_order
from lib.task_design import TASK_CONTRASTS, normalize_task_branch
from lib.task_group import (
    PARCELLATION_HCP151_ATLAS,
    PARCELLATION_TEMPLATE_MATCHING,
    fdr_adjust,
    sig_marker,
)
from lib.workflow_utils import file_meta, utc_now_iso, write_json


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_GROUP_AVG_SURFACES_DIR = (
    REPO_ROOT
    / "outputs_hcp"
    / "inputs"
    / "hcp"
    / "kong2022"
    / "templates"
    / "hipp_group_average_t1only_151"
    / "group_average_surfaces"
)
N_LONG_AXIS_BINS = 20
CLUSTER_ORDER = ["Anterior", "Body", "Posterior"]
CLUSTER_SHORT = {"Anterior": "Ant", "Body": "Mid", "Posterior": "Post"}
ZONE_ORDER = [PARCELLATION_HCP151_ATLAS, PARCELLATION_TEMPLATE_MATCHING]
ZONE_SHORT = {
    PARCELLATION_HCP151_ATLAS: "hcp151",
    PARCELLATION_TEMPLATE_MATCHING: "tm",
}
ZONE_COLORS = {
    PARCELLATION_HCP151_ATLAS: "#93C5FD",
    PARCELLATION_TEMPLATE_MATCHING: "#FDE68A",
}
ZONE_EDGE_COLORS = {
    PARCELLATION_HCP151_ATLAS: "#1D4ED8",
    PARCELLATION_TEMPLATE_MATCHING: "#D97706",
}
ZONE_DOT_COLORS = {
    PARCELLATION_HCP151_ATLAS: "#1E3A8A",
    PARCELLATION_TEMPLATE_MATCHING: "#92400E",
}
# Bracket palette: black = subregion difference within one method.
REGION_BRACKET_COLOR = "black"
NATURE_FONT = {"family": "sans-serif", "sans-serif": ["Arial", "Helvetica", "DejaVu Sans"]}
BETA_DISPLAY = {"fear": "Emotional faces", "neut": "Shapes"}
CATEGORY_THEMES = {
    "faces": {"text": "#9F1239", "bg": "#FFE4E6", "border": "#FDA4AF", "title": "Faces"},
    "places": {"text": "#065F46", "bg": "#D1FAE5", "border": "#6EE7B7", "title": "Places"},
    "tools": {"text": "#92400E", "bg": "#FEF3C7", "border": "#FCD34D", "title": "Tools"},
    "body": {"text": "#5B21B6", "bg": "#EDE9FE", "border": "#C4B5FD", "title": "Body"},
}


def default_group_root(task: str) -> Path:
    return REPO_ROOT / "outputs_hcp" / "work" / "task_activation" / f"task-{task}" / "group"


def default_out_root(task: str) -> Path:
    return REPO_ROOT / "outputs_hcp" / "present" / "task_activation" / f"task-{task}"


def discover_combos(group_root: Path) -> list[tuple[str, str, str]]:
    combos: list[tuple[str, str, str]] = []
    for mean_path in sorted(group_root.glob("2mm/*/group_contrast_*_mean.npy")):
        hemi = mean_path.parent.name.removeprefix("hemi_")
        contrast = mean_path.name.removeprefix("group_contrast_").removesuffix("_mean.npy")
        combos.append(("2mm", hemi, contrast))
    return combos


def discover_beta_stems(group_root: Path) -> list[str]:
    stems: list[str] = []
    for mean_path in sorted(group_root.glob("2mm/*/group_beta_*_mean.npy")):
        stem = mean_path.name.removeprefix("group_beta_").removesuffix("_mean.npy")
        if stem not in stems:
            stems.append(stem)
    return stems


def make_func_gifti(data: np.ndarray) -> GiftiImage:
    return GiftiImage(
        darrays=[
            GiftiDataArray(
                data=np.asarray(data, dtype=np.float32),
                intent="NIFTI_INTENT_TIME_SERIES",
                datatype="NIFTI_TYPE_FLOAT32",
            )
        ]
    )


def read_cluster_summary(group_root: Path) -> list[dict[str, str]]:
    path = group_root / "group_cluster_summary.tsv"
    if not path.is_file():
        raise FileNotFoundError(f"Missing group cluster summary: {path}")
    with path.open(encoding="utf-8") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def group_roster(group_root: Path, *, smoothing: str, hemi: str, contrast: str) -> list[str]:
    manifest = json.loads((group_root / "group_build_manifest.json").read_text(encoding="utf-8"))
    for combo in manifest["combos"]:
        if (
            combo["smoothing"] == smoothing
            and combo["hemi"] == hemi
            and combo["contrast"] == contrast
        ):
            return [s.replace("sub-", "") for s in combo["roster"]]
    raise ValueError(f"No roster for {contrast} {smoothing}/hemi-{hemi} in group manifest")


def contrast_display(contrast: str) -> str:
    """Plain-language panel titles (keys stay as filenames/manifest IDs)."""
    return {
        "twobk_gt_zerobk": "Working memory load (2-back > 0-back)",
        "faces_gt_shapes": "Emotional faces > Shapes",
        "rel_gt_match": "Relational > Matching",
        "faces_gt_others": "Faces > Others (Places/Tools/Body)",
        "places_gt_others": "Places > Others (Faces/Tools/Body)",
        "tools_gt_others": "Tools > Others (Faces/Places/Body)",
        "body_gt_others": "Body > Others (Faces/Places/Tools)",
        "load_x_faces": "Load × Faces interaction",
        "load_x_places": "Load × Places interaction",
        "load_x_tools": "Load × Tools interaction",
        "load_x_body": "Load × Body interaction",
    }.get(contrast, contrast)


def contrast_physiological_sides(
    task: str, contrast: str
) -> tuple[dict[str, float], dict[str, float], str, str]:
    """Return physiological decomposition of a contrast into two sides and descriptive curve labels.

    For working memory:
    - Load (twobk_gt_zerobk): 2-back (mean) vs 0-back (mean) across categories
    - Category contrasts (<cat>_gt_others): <Cat> (mean 0bk+2bk) vs Others (mean 0bk+2bk across 3 categories)
    - Category x Load interactions (load_x_<cat>): Δ <Cat> (2bk - 0bk) vs Δ Others (2bk - 0bk across 3 categories)
    Difference (side1 - side2) algebraically reconstructs the contrast map exactly.
    """
    branch = normalize_task_branch(task)
    if branch == "wm":
        categories = ("faces", "places", "tools", "body")
        if contrast == "twobk_gt_zerobk":
            side1 = {f"2bk_{c}": 0.25 for c in categories}
            side2 = {f"0bk_{c}": 0.25 for c in categories}
            return side1, side2, "2-back (mean)", "0-back (mean)"
        for cat in categories:
            if contrast == f"{cat}_gt_others":
                others = [c for c in categories if c != cat]
                side1 = {f"0bk_{cat}": 0.5, f"2bk_{cat}": 0.5}
                side2 = {}
                for c in others:
                    side2[f"0bk_{c}"] = 1.0 / 6.0
                    side2[f"2bk_{c}"] = 1.0 / 6.0
                return side1, side2, f"{cat.capitalize()} (mean)", "Others (mean)"
            if contrast == f"load_x_{cat}":
                others = [c for c in categories if c != cat]
                side1 = {f"2bk_{cat}": 0.5, f"0bk_{cat}": -0.5}
                side2 = {}
                for c in others:
                    side2[f"2bk_{c}"] = 0.5 / 3.0
                    side2[f"0bk_{c}"] = -0.5 / 3.0
                return (
                    side1,
                    side2,
                    f"Δ {cat.capitalize()} (2bk-0bk)",
                    "Δ Others (2bk-0bk)",
                )
    elif branch == "emotion":
        if contrast == "faces_gt_shapes":
            return {"fear": 1.0}, {"neut": 1.0}, "Emotional faces", "Shapes"
    elif branch == "relational":
        if contrast == "rel_gt_match":
            return {"rel": 1.0}, {"match": 1.0}, "Relational", "Matching"

    weights = TASK_CONTRASTS[branch][contrast]
    pos = {k: float(v) for k, v in weights.items() if float(v) > 0}
    neg = {k: float(v) for k, v in weights.items() if float(v) < 0}
    pos_total = sum(pos.values())
    neg_total = sum(-v for v in neg.values())
    return (
        {k: v / pos_total for k, v in pos.items()},
        {k: -v / neg_total for k, v in neg.items()},
        "weighted +side",
        "weighted -side",
    )


def binned_profile(mean_map: np.ndarray, bins: list[np.ndarray]) -> np.ndarray:
    """Bin one group-mean map along the long axis (no CI bands)."""
    values = np.asarray(mean_map, dtype=np.float64).reshape(-1)
    n_vertices = sum(vertices.size for vertices in bins)
    if values.size != n_vertices:
        raise ValueError(f"Map/surface vertex mismatch: {values.size} vs {n_vertices}")
    profile = np.full(len(bins), np.nan)
    for b, vertices in enumerate(bins):
        vals = values[vertices]
        vals = vals[np.isfinite(vals)]
        if vals.size:
            profile[b] = float(np.mean(vals))
    return profile


ZONE_ORDER = [PARCELLATION_HCP151_ATLAS, PARCELLATION_TEMPLATE_MATCHING]
ZONE_LABELS = {
    PARCELLATION_HCP151_ATLAS: "hcp151",
    PARCELLATION_TEMPLATE_MATCHING: "tm",
}


def contrast_beta_groups(task: str, contrast: str) -> tuple[list[str], list[str]]:
    """Split contrast regressors into positive/negative weight stem groups."""
    weights = TASK_CONTRASTS[normalize_task_branch(task)][contrast]
    pos = sorted(k for k, v in weights.items() if float(v) > 0)
    neg = sorted(k for k, v in weights.items() if float(v) < 0)
    if not pos or not neg:
        raise ValueError(f"Contrast {contrast!r} needs both sides for beta grouping")
    return pos, neg


def contrast_side_weights(task: str, contrast: str) -> tuple[dict[str, float], dict[str, float]]:
    """Actual contrast weights per side, normalized to sum to 1.

    Profile side curves use these weights (not arithmetic means), so their
    difference reconstructs the plotted contrast exactly for any weight
    scheme (e.g. load_x_places with 0.5 vs 1/6 weights).
    """
    weights = TASK_CONTRASTS[normalize_task_branch(task)][contrast]
    pos = {k: float(v) for k, v in weights.items() if float(v) > 0}
    neg = {k: float(v) for k, v in weights.items() if float(v) < 0}
    if not pos or not neg:
        raise ValueError(f"Contrast {contrast!r} needs both sides for beta grouping")
    pos_total = sum(pos.values())
    neg_total = sum(-v for v in neg.values())
    return (
        {k: v / pos_total for k, v in pos.items()},
        {k: -v / neg_total for k, v in neg.items()},
    )


def weighted_side_map(group_root: Path, *, weights: dict[str, float], hemi: str) -> np.ndarray:
    """Vertex-wise weighted combination of group-mean beta maps.

    Keep the declared weights fixed. A non-finite required condition makes
    the combined vertex NaN; remaining conditions never replace missing ones.
    """
    first = group_beta_mean(group_root, stem=sorted(weights)[0], hemi=hemi)
    total = np.zeros_like(first, dtype=np.float64)
    valid = np.ones_like(first, dtype=bool)
    for stem, weight in sorted(weights.items()):
        beta = group_beta_mean(group_root, stem=stem, hemi=hemi)
        if beta.shape != total.shape:
            raise ValueError(
                f"Beta map shape mismatch for stem {stem!r}: {beta.shape} vs {total.shape}"
            )
        finite = np.isfinite(beta)
        valid &= finite
        total[finite] += float(weight) * beta[finite]
    total[~valid] = np.nan
    return total


def parse_bh_fdr_p(row: dict[str, str], *, contrast: str, hemi: str) -> float:
    """Require the BH-FDR adjusted p; never fall back to the raw p.

    A missing/empty ``group_p_fdr`` (stale or partial summary table) fails
    closed so uncorrected significance is never drawn as corrected.
    """
    raw = row.get("group_p_fdr")
    if raw is None or str(raw).strip() == "":
        raise ValueError(
            f"Missing BH-FDR group_p_fdr for {contrast} hemi-{hemi} "
            f"{row.get('parcellation')}/{row.get('cluster_name')}: "
            "re-run 05_06 instead of plotting raw p as FDR"
        )
    return float(raw)


def group_beta_mean(group_root: Path, *, stem: str, hemi: str) -> np.ndarray:
    path = group_root / "2mm" / f"hemi_{hemi}" / f"group_beta_{stem}_mean.npy"
    if not path.is_file():
        raise FileNotFoundError(f"Missing group beta map: {path}")
    return np.load(str(path)).astype(np.float64).reshape(-1)


def panel_coverage(
    subject_table: list[dict[str, str]], *, contrast: str, hemi: str, zone: str
) -> float:
    fracs = [
        float(r["n_finite"]) / max(1, int(r["n_vertices"]))
        for r in subject_table
        if r["parcellation"] == zone
        and r["contrast"] == contrast
        and r["hemi"] == hemi
    ]
    if not fracs:
        return float("nan")
    return float(np.mean(fracs))


def render_task_figure(
    *,
    task: str,
    contrast: str,
    hemis: list[str],
    group_root: Path,
    surfaces_dir: Path,
    summary_rows: list[dict[str, str]],
    subject_table: list[dict[str, str]],
    n_subjects: int,
    profile_axes: list,
    bars_axes: list,
    bracket_tests: list[dict[str, object]],
    is_card: bool = False,
    unit_category: str | None = None,
) -> None:
    pos_weights, neg_weights, label1, label2 = contrast_physiological_sides(task, contrast)
    for col, hemi in enumerate(hemis):
        ax = profile_axes[col]
        surface_path = (
            surfaces_dir
            / f"group-average_hemi-{hemi}_space-corobl_den-512_label-hipp_midthickness.surf.gii"
        )
        bins = equal_vertex_long_axis_bins(long_axis_order(surface_path), N_LONG_AXIS_BINS)
        pos_map = weighted_side_map(group_root, weights=pos_weights, hemi=hemi)
        neg_map = weighted_side_map(group_root, weights=neg_weights, hemi=hemi)
        contrast_map = np.load(
            str(group_root / "2mm" / f"hemi_{hemi}" / f"group_contrast_{contrast}_mean.npy")
        ).astype(np.float64).reshape(-1)

        contrast_label = "Contrast" if is_card else f"contrast {contrast_display(contrast)}"
        lines = [
            (binned_profile(pos_map, bins), "#2563EB", 1.8, label1),
            (binned_profile(neg_map, bins), "#D97706", 1.8, label2),
            (binned_profile(contrast_map, bins), "#111827", 2.2, contrast_label),
        ]
        bin_numbers = np.arange(1, N_LONG_AXIS_BINS + 1)
        for mean, color, width, label in lines:
            ax.plot(bin_numbers, mean, color=color, linewidth=width, label=label)
        ax.axhline(0.0, color="#64748B", linestyle="--", linewidth=0.7)
        ax.set_xlim(0.8, N_LONG_AXIS_BINS + 0.2)
        ax.set_xticks([1, 5, 10, 15, 20])
        panel_n = len(group_roster(group_root, smoothing="2mm", hemi=hemi, contrast=contrast))
        panel_cov = panel_coverage(subject_table, contrast=contrast, hemi=hemi, zone=PARCELLATION_HCP151_ATLAS)

        if is_card:
            ax.set_title(f"hemi-{hemi}", fontsize=9, fontweight="bold", pad=4)
            ax.set_xlabel("Bin (1=Ant → 20=Post)", fontsize=7.5, labelpad=2)
            if col == 0:
                ax.set_ylabel("Activation (a.u.)", fontsize=7.5, labelpad=2)
            ax.tick_params(labelsize=7, pad=2)
            ax.legend(
                fontsize=6.5,
                loc="best",
                frameon=True,
                facecolor="white",
                edgecolor="#E2E8F0",
                framealpha=0.9,
                borderpad=0.3,
            )
        else:
            ax.set_title(
                f"{contrast_display(contrast)}\nhemi-{hemi} profile (N={panel_n}, coverage={panel_cov:.0%})",
                fontsize=10,
            )
            ax.set_xlabel(
                f"long axis bin number (1 = head/anterior -> {N_LONG_AXIS_BINS} = tail/posterior)",
                fontsize=9,
            )
            ax.set_ylabel("beta / contrast (a.u.)", fontsize=9)
            ax.tick_params(labelsize=8)
            ax.legend(fontsize=8, frameon=False)

    ylims = [ax.get_ylim() for ax in profile_axes]
    shared = (min(lo for lo, _ in ylims), max(hi for _, hi in ylims))
    for ax in profile_axes:
        ax.set_ylim(shared)

    for col, hemi in enumerate(hemis):
        ax = bars_axes[col]
        roster = group_roster(group_root, smoothing="2mm", hemi=hemi, contrast=contrast)
        bar_info: dict[tuple[str, str], dict[str, object]] = {}
        for zone in ZONE_ORDER:
            rows = [
                r for r in summary_rows
                if r["parcellation"] == zone and r["contrast"] == contrast and r["hemi"] == hemi
            ]
            rows = sorted(
                rows,
                key=lambda r: CLUSTER_ORDER.index(r["cluster_name"]) if r["cluster_name"] in CLUSTER_ORDER else 99,
            )
            if [r["cluster_name"] for r in rows] != CLUSTER_ORDER:
                raise ValueError(
                    f"Expected Anterior/Body/Posterior rows for {contrast} {zone} hemi-{hemi}, "
                    f"got {[r['cluster_name'] for r in rows]}"
                )
            for row in rows:
                subjects_vals = np.array([
                    float(r["mean"]) for r in subject_table
                    if r["parcellation"] == zone
                    and r["contrast"] == contrast
                    and r["hemi"] == hemi
                    and r["cluster_name"] == row["cluster_name"]
                    and r["subject"].replace("sub-", "") in roster
                ])
                subjects_vals = subjects_vals[np.isfinite(subjects_vals)]
                mean = float(row["group_mean"])
                n = int(float(row["n_subjects"]))
                se = float(row["group_sd"]) / np.sqrt(max(n, 1)) if n >= 2 else float("nan")
                half = se * float(t_dist.ppf(0.975, max(n - 1, 1))) if n >= 2 else float("nan")
                bar_info[(zone, row["cluster_name"])] = {
                    "mean": mean,
                    "half": half,
                    "n": n,
                    "values": subjects_vals,
                    "p": parse_bh_fdr_p(row, contrast=contrast, hemi=hemi),
                }
        for ci, cluster in enumerate(CLUSTER_ORDER):
            for zi, zone in enumerate(ZONE_ORDER):
                pos = ci * 3.0 + zi * 0.9
                info = bar_info[(zone, cluster)]
                mean = float(info["mean"])
                half = float(info["half"])
                color = ZONE_COLORS[zone]
                edge_color = ZONE_EDGE_COLORS[zone]
                dot_color = ZONE_DOT_COLORS[zone]

                ax.bar(
                    pos,
                    mean,
                    width=0.8,
                    color=color,
                    alpha=0.9,
                    edgecolor=edge_color,
                    linewidth=0.9,
                    label=ZONE_SHORT[zone] if (ci == 0 and not is_card) else (ZONE_SHORT[zone] if ci == 0 and col == 0 else None),
                )
                ax.text(
                    pos,
                    1.01,
                    f"n={info['n']}",
                    transform=ax.get_xaxis_transform(),
                    ha="center",
                    va="bottom",
                    fontsize=6 if is_card else 7,
                    color="#475569",
                    bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.8, "pad": 0.5 if is_card else 1},
                )
                if np.isfinite(half):
                    ax.errorbar(
                        pos,
                        mean,
                        yerr=half,
                        color="#DC2626",
                        capsize=3 if is_card else 4,
                        linewidth=1.3 if is_card else 1.5,
                        zorder=5,
                    )
                subjects_vals = np.asarray(info["values"])
                rng = np.random.default_rng(abs(hash((task, contrast, hemi, zone, cluster))) % (2**32))
                jitter = (rng.random(subjects_vals.size) - 0.5) * (0.22 if is_card else 0.24)
                ax.scatter(
                    np.full(subjects_vals.size, pos) + jitter,
                    subjects_vals,
                    s=7 if is_card else 12,
                    color=dot_color,
                    edgecolors="none",
                    alpha=0.45 if is_card else 0.50,
                    zorder=3,
                )
                marker = sig_marker(float(info["p"]))
                if marker:
                    my = mean + (half if np.isfinite(half) else 0.0) if mean >= 0 else mean - (half if np.isfinite(half) else 0.0)
                    va = "bottom" if mean >= 0 else "top"
                    pad = (0.03 if is_card else 0.02) * max(1.0, abs(mean) + (half if np.isfinite(half) else 0.0))
                    ax.text(
                        pos,
                        my + (pad if va == "bottom" else -pad),
                        marker,
                        ha="center",
                        va=va,
                        fontsize=11 if is_card else 13,
                        fontweight="bold",
                        color="black",
                        zorder=6,
                        path_effects=[pe.withStroke(linewidth=2.0, foreground="white")],
                    )
        for zone in ZONE_ORDER:
            anchor = {
                cluster: ci * 3.0 + ZONE_ORDER.index(zone) * 0.9 for ci, cluster in enumerate(CLUSTER_ORDER)
            }
            zone_vals: dict[str, dict[str, float]] = {}
            for r in subject_table:
                if not (r["parcellation"] == zone and r["contrast"] == contrast and r["hemi"] == hemi):
                    continue
                if r["subject"].replace("sub-", "") not in roster:
                    continue
                if np.isfinite(float(r["mean"])):
                    zone_vals.setdefault(r["subject"], {})[r["cluster_name"]] = float(r["mean"])
            pairs = [("Anterior", "Body"), ("Body", "Posterior"), ("Anterior", "Posterior")]
            for a, b in pairs:
                paired = np.array([
                    (v[a], v[b]) for v in zone_vals.values() if a in v and b in v
                ])
                if len(paired) < 2:
                    continue
                diff = paired[:, 0] - paired[:, 1]
                n = len(diff)
                tval = float(np.mean(diff) / (np.std(diff, ddof=1) / np.sqrt(n))) if np.std(diff, ddof=1) > 0 else 0.0
                pval = float(2.0 * t_dist.sf(abs(tval), n - 1))
                bracket_tests.append(
                    {
                        "ax": ax,
                        "zone": zone,
                        "a": a,
                        "b": b,
                        "p": pval,
                        "anchor": anchor,
                        "bar_info": bar_info,
                        "bars_axes": bars_axes,
                    }
                )
        ax.set_xticks([ci * 3.0 + 0.45 for ci in range(3)])
        ax.set_xticklabels([CLUSTER_SHORT[c] for c in CLUSTER_ORDER], fontsize=7.5 if is_card else 9)
        ax.tick_params(labelsize=7 if is_card else 8, pad=2)
        ax.axhline(0.0, color="#64748B", linestyle="--", linewidth=0.7)
        if is_card:
            ax.set_title(f"hemi-{hemi} clusters", fontsize=8.5, fontweight="bold", pad=12)
            if col == 0:
                ax.set_ylabel("Cluster mean (a.u.)", fontsize=7.5, labelpad=2)
                ax.legend(
                    fontsize=6.5,
                    loc="lower right",
                    frameon=True,
                    facecolor="white",
                    edgecolor="#E2E8F0",
                    framealpha=0.85,
                )
        else:
            panel_cov = panel_coverage(
                subject_table, contrast=contrast, hemi=hemi, zone=PARCELLATION_HCP151_ATLAS
            )
            ax.set_title(
                f"{contrast_display(contrast)}\nhemi-{hemi} clusters (effective n per bar, coverage={panel_cov:.0%})",
                pad=20,
                fontsize=10,
            )
            ax.set_ylabel("area-weighted cluster mean (a.u.)", fontsize=9)
            ax.legend(fontsize=8, loc="best", frameon=False)

    bar_ylims = [ax.get_ylim() for ax in bars_axes]
    shared_bar_ylim = (min(lo for lo, _ in bar_ylims), max(hi for _, hi in bar_ylims))
    for ax in bars_axes:
        ax.set_ylim(shared_bar_ylim)


def draw_brackets(
    bracket_tests: list[dict[str, object]],
    *,
    is_card: bool = False,
) -> None:
    """FDR-adjust collected subregion-bracket p-values across the whole figure.

    One combined BH-FDR family covers all black region brackets (subregion
    difference within one method). To prevent collisions when multiple comparisons
    are significant on the same panel:
      1. Brackets are grouped per axis and sorted hierarchically: shorter span
         pairs (adjacent subregions) first, longer span pairs (Ant-Post) later.
      2. Brackets on the same axis share a global collision-avoidance stack;
         horizontally overlapping brackets are assigned distinct vertical tiers
         with enough clearance so horizontal lines never cut through markers.
      3. Top headroom is expanded dynamically to guarantee asterisk markers
         never touch, overlap, or exceed the top axis frame boundary.
    """
    if not bracket_tests:
        return
    bracket_fdr = fdr_adjust([float(t["p"]) for t in bracket_tests])
    tests_by_ax: dict[object, list[dict[str, object]]] = {}
    axes_pairs: dict[int, list] = {}
    for test, adjusted in zip(bracket_tests, bracket_fdr):
        marker = sig_marker(adjusted)
        if not marker:
            continue
        ax = test["ax"]
        t = dict(test)
        t["adjusted"] = adjusted
        t["marker"] = marker
        tests_by_ax.setdefault(ax, []).append(t)
        if "bars_axes" in test and isinstance(test["bars_axes"], list):
            for a in test["bars_axes"]:
                axes_pairs[id(a)] = test["bars_axes"]

    needed_top: dict[int, tuple[object, float]] = {}

    for ax, tests in tests_by_ax.items():
        ylo, yhi = ax.get_ylim()
        span = max(yhi - ylo, 1e-9)

        prepared = []
        for t in tests:
            zone = str(t["zone"])
            a, b = str(t["a"]), str(t["b"])
            anchor = t["anchor"]
            bar_info = t["bar_info"]
            assert isinstance(anchor, dict)
            assert isinstance(bar_info, dict)
            xa, xb = float(anchor[a]), float(anchor[b])
            if xa > xb:
                xa, xb = xb, xa
            tops = []
            for cluster in (a, b):
                info = bar_info[(zone, cluster)]
                assert isinstance(info, dict)
                mean = float(info["mean"])
                half = float(info["half"])
                vals = np.asarray(info["values"])
                tops.append(mean + (half if np.isfinite(half) else 0.0) + span * 0.04)
                if vals.size:
                    tops.append(float(np.max(vals)))
            if ("Anterior" in (a, b)) and ("Posterior" in (a, b)):
                for mid_c in ("Body", "Mid"):
                    if (zone, mid_c) in bar_info:
                        info_m = bar_info[(zone, mid_c)]
                        assert isinstance(info_m, dict)
                        mean_m = float(info_m["mean"])
                        half_m = float(info_m["half"])
                        vals_m = np.asarray(info_m["values"])
                        tops.append(mean_m + (half_m if np.isfinite(half_m) else 0.0) + span * 0.04)
                        if vals_m.size:
                            tops.append(float(np.max(vals_m)))
            data_top = max(tops) if tops else yhi
            prepared.append(
                {
                    "test": t,
                    "xa": xa,
                    "xb": xb,
                    "span_x": xb - xa,
                    "data_top": data_top,
                    "zone": zone,
                    "marker": t["marker"],
                }
            )

        def sort_key(item: dict[str, object]) -> tuple[float, float, int]:
            z = str(item["zone"])
            zone_idx = ZONE_ORDER.index(z) if z in ZONE_ORDER else 99
            return (round(float(item["span_x"]), 2), round(float(item["xa"]), 2), zone_idx)

        prepared.sort(key=sort_key)

        placed: list[dict[str, object]] = []
        cap = span * 0.020
        delta_y = span * (0.080 if is_card else 0.090)
        text_offset = span * 0.008

        for item in prepared:
            xa, xb = float(item["xa"]), float(item["xb"])
            marker = str(item["marker"])
            y = float(item["data_top"]) + span * (0.040 if is_card else 0.045)
            for p in placed:
                p_xa, p_xb, p_y = float(p["xa"]), float(p["xb"]), float(p["y"])
                overlap = max(xa, p_xa) <= min(xb, p_xb) + 0.3
                if overlap:
                    y = max(y, p_y + delta_y)

            ax.plot(
                [xa, xa, xb, xb],
                [y - cap, y, y, y - cap],
                color=REGION_BRACKET_COLOR,
                linewidth=1.2 if is_card else 1.3,
            )
            ax.text(
                (xa + xb) / 2,
                y + text_offset,
                marker,
                ha="center",
                va="bottom",
                fontsize=8.0 if is_card else 9.5,
                fontweight="bold",
                color=REGION_BRACKET_COLOR,
                path_effects=[pe.withStroke(linewidth=1.8, foreground="white")],
            )
            placed.append({"xa": xa, "xb": xb, "y": y, "marker": marker})
            headroom = span * (0.11 if is_card else 0.13)
            want = y + headroom
            prev = needed_top.get(id(ax))
            if prev is None or want > prev[1]:
                needed_top[id(ax)] = (ax, want)

    if needed_top:
        for ax_id, (ax, want) in needed_top.items():
            lo, hi = ax.get_ylim()
            new_hi = max(hi, want)
            pair = axes_pairs.get(ax_id, [ax])
            for p_ax in pair:
                p_lo, p_hi = p_ax.get_ylim()
                if new_hi > p_hi:
                    p_ax.set_ylim(p_lo, new_hi)


def read_subject_table(group_root: Path) -> list[dict[str, str]]:
    path = group_root / "subject_cluster_table.tsv"
    if not path.is_file():
        raise FileNotFoundError(f"Missing subject cluster table: {path}")
    with path.open(encoding="utf-8") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def render_wm_figures(
    *,
    group_root: Path,
    surfaces_dir: Path,
    out_root: Path,
    summary_rows: list[dict[str, str]],
    subject_table: list[dict[str, str]],
    n_subjects: int,
    hemis: list[str],
) -> list[Path]:
    """Render working memory figures: standalone load figure and 2x4 category layout."""
    plt.rcParams.update(
        {
            "font.family": NATURE_FONT["family"],
            "font.sans-serif": NATURE_FONT["sans-serif"],
            "axes.linewidth": 0.8,
            "xtick.major.width": 0.8,
            "ytick.major.width": 0.8,
        }
    )

    generated_pngs: list[Path] = []

    # 1. STANDALONE LOAD FIGURE
    fig_load, axes_load = plt.subplots(2, len(hemis), figsize=(11, 7.5), squeeze=False)
    bracket_tests_load: list[dict[str, object]] = []
    render_task_figure(
        task="wm",
        contrast="twobk_gt_zerobk",
        hemis=hemis,
        group_root=group_root,
        surfaces_dir=surfaces_dir,
        summary_rows=summary_rows,
        subject_table=subject_table,
        n_subjects=n_subjects,
        profile_axes=list(axes_load[0]),
        bars_axes=list(axes_load[1]),
        bracket_tests=bracket_tests_load,
        is_card=False,
    )
    draw_brackets(bracket_tests_load)
    fig_load.suptitle(
        f"Working Memory Cognitive Load: 2-back > 0-back (N={n_subjects}, 2mm; markers BH-FDR)",
        fontsize=13,
        fontweight="bold",
        y=0.98,
    )
    fig_load.text(
        0.5,
        0.01,
        "Markers: +=p<.05, *=p<.01, **=p<.001, ***=p<.0001 (BH-FDR, two-tailed); bins = equal-vertex bins 1-20 (Ant → Post)\n"
        "Beta/Contrast in raw BOLD units (a.u.); black brackets = pairwise subregion difference within parcellation method",
        ha="center",
        fontsize=7.5,
        style="italic",
        color="#334155",
    )
    fig_load.tight_layout(rect=[0, 0.03, 1, 0.95])
    out_load_png = out_root / "task_wm_load.png"
    fig_load.savefig(out_load_png, dpi=300)
    plt.close(fig_load)
    generated_pngs.append(out_load_png)

    # 2. 2x4 CATEGORIES FIGURE
    categories = ["faces", "places", "tools", "body"]
    row_contrasts = [
        [f"{c}_gt_others" for c in categories],
        [f"load_x_{c}" for c in categories],
    ]

    fig_cat = plt.figure(figsize=(28, 15.0))
    outer_gs = gridspec.GridSpec(
        2,
        4,
        figure=fig_cat,
        left=0.035,
        right=0.985,
        top=0.90,
        bottom=0.045,
        wspace=0.26,
        hspace=0.40,
    )
    bracket_tests_cat: list[dict[str, object]] = []
    prof_axes_by_row: list[list] = [[], []]
    bar_axes_by_row: list[list] = [[], []]

    for r in range(2):
        for c in range(4):
            contrast = row_contrasts[r][c]
            cat = categories[c]
            sub_gs = gridspec.GridSpecFromSubplotSpec(
                2,
                2,
                subplot_spec=outer_gs[r, c],
                wspace=0.20,
                hspace=0.34,
            )
            prof_L = fig_cat.add_subplot(sub_gs[0, 0])
            prof_R = fig_cat.add_subplot(sub_gs[0, 1])
            bar_L = fig_cat.add_subplot(sub_gs[1, 0])
            bar_R = fig_cat.add_subplot(sub_gs[1, 1])
            prof_axes_by_row[r].extend([prof_L, prof_R])
            bar_axes_by_row[r].extend([bar_L, bar_R])

            render_task_figure(
                task="wm",
                contrast=contrast,
                hemis=hemis,
                group_root=group_root,
                surfaces_dir=surfaces_dir,
                summary_rows=summary_rows,
                subject_table=subject_table,
                n_subjects=n_subjects,
                profile_axes=[prof_L, prof_R],
                bars_axes=[bar_L, bar_R],
                bracket_tests=bracket_tests_cat,
                is_card=True,
                unit_category=cat,
            )

    # 1. Unify profile Y-axis limits row-wise across all 4 categories
    for r in range(2):
        r_ylims = [ax.get_ylim() for ax in prof_axes_by_row[r]]
        shared_prof = (min(lo for lo, _ in r_ylims), max(hi for _, hi in r_ylims))
        for ax in prof_axes_by_row[r]:
            ax.set_ylim(shared_prof)

    # 2. Draw brackets (which adjusts axes tops for significance brackets)
    draw_brackets(bracket_tests_cat, is_card=True)

    # 3. Unify cluster bar Y-axis limits row-wise across all 4 categories so heights are directly comparable
    for r in range(2):
        r_bar_ylims = [ax.get_ylim() for ax in bar_axes_by_row[r]]
        shared_bar = (min(lo for lo, _ in r_bar_ylims), max(hi for _, hi in r_bar_ylims))
        for ax in bar_axes_by_row[r]:
            ax.set_ylim(shared_bar)

    # Draw card boxes and distinct headers
    for r in range(2):
        for c in range(4):
            cat = categories[c]
            theme = CATEGORY_THEMES[cat]
            bb = outer_gs[r, c].get_position(fig_cat)
            pad_x = 0.007
            pad_y_bottom = 0.010
            pad_y_top = 0.032
            card_rect = patches.FancyBboxPatch(
                (bb.x0 - pad_x, bb.y0 - pad_y_bottom),
                bb.width + 2 * pad_x,
                bb.height + pad_y_bottom + pad_y_top,
                boxstyle="round,pad=0.005,rounding_size=0.008",
                transform=fig_cat.transFigure,
                facecolor="#FFFFFF" if r == 0 else "#F8FAFC",
                edgecolor="#CBD5E1",
                linewidth=1.2,
                zorder=-1,
            )
            fig_cat.patches.append(card_rect)

            title_text = f"{theme['title']} > Others" if r == 0 else f"Load × {theme['title']}"
            fig_cat.text(
                bb.x0 + bb.width / 2.0,
                bb.y1 + 0.016,
                title_text,
                ha="center",
                va="center",
                fontsize=11,
                fontweight="bold",
                color=theme["text"],
                bbox=dict(
                    boxstyle="round,pad=0.3",
                    facecolor=theme["bg"],
                    edgecolor=theme["border"],
                    linewidth=1.0,
                ),
                zorder=10,
            )

    # Section banners
    fig_cat.text(
        0.5,
        0.958,
        "A. CATEGORY SPECIALIZATION (Category > Others across 0-back & 2-back)",
        ha="center",
        va="center",
        fontsize=12,
        fontweight="bold",
        color="#0F172A",
        bbox=dict(boxstyle="square,pad=0.35", facecolor="#E2E8F0", edgecolor="#CBD5E1", linewidth=0.8),
    )
    fig_cat.text(
        0.5,
        0.485,
        "B. COGNITIVE LOAD × CATEGORY INTERACTION (Differential Sensitivity: 2-back vs 0-back)",
        ha="center",
        va="center",
        fontsize=12,
        fontweight="bold",
        color="#0F172A",
        bbox=dict(boxstyle="square,pad=0.35", facecolor="#E2E8F0", edgecolor="#CBD5E1", linewidth=0.8),
    )

    fig_cat.suptitle(
        f"HCP 3T Task-fMRI Working Memory: Category Specialization & Load Interactions (N={n_subjects}, 2mm; markers BH-FDR)",
        fontsize=14.5,
        fontweight="bold",
        y=0.992,
        color="#0F172A",
    )
    fig_cat.text(
        0.5,
        0.008,
        "Markers: +=p<.05, *=p<.01, **=p<.001, ***=p<.0001 (BH-FDR, two-tailed); bins = equal-vertex bins 1-20 (Ant → Post)\n"
        "Beta in raw BOLD units (a.u.); black brackets = pairwise subregion difference within parcellation method (BH-FDR family across whole figure)",
        ha="center",
        fontsize=7.5,
        style="italic",
        color="#475569",
    )

    out_cat_png = out_root / "task_wm_categories.png"
    fig_cat.savefig(out_cat_png, dpi=300)
    plt.close(fig_cat)
    generated_pngs.append(out_cat_png)
    return generated_pngs


def main() -> int:
    parser = argparse.ArgumentParser(description="Render task-activation products.")
    parser.add_argument("--task", required=True)
    parser.add_argument("--group-root", default=None)
    parser.add_argument("--group-average-surfaces-dir", default=str(DEFAULT_GROUP_AVG_SURFACES_DIR))
    parser.add_argument("--out-root", default=None)
    parser.add_argument("--hemis", nargs="*", default=None)
    parser.add_argument("--contrasts", nargs="*", default=None)
    args = parser.parse_args()

    task = normalize_task_branch(args.task)
    group_root = Path(args.group_root) if args.group_root else default_group_root(task)
    out_root = Path(args.out_root) if args.out_root else default_out_root(task)
    surfaces_dir = Path(args.group_average_surfaces_dir)
    manifest_path = group_root / "group_build_manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Missing 05_06 group manifest: {manifest_path}")
    group_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    discovered = [combo for combo in discover_combos(group_root) if combo[0] == "2mm"]
    hemis = sorted({h for _, h, _ in discovered if args.hemis is None or h in args.hemis})
    contrasts = sorted({c for _, _, c in discovered if args.contrasts is None or c in args.contrasts})
    if not (hemis and contrasts):
        raise ValueError(f"No group combos discovered under {group_root}")
    summary_rows = [r for r in read_cluster_summary(group_root) if r["smoothing"] == "2mm"]
    subject_table = [r for r in read_subject_table(group_root) if r["smoothing"] == "2mm"]
    beta_stems = discover_beta_stems(group_root)
    out_root.mkdir(parents=True, exist_ok=True)
    products: list[dict[str, object]] = []

    for _, hemi, contrast in discovered:
        if hemi not in hemis or contrast not in contrasts:
            continue
        if contrast not in TASK_CONTRASTS[task]:
            raise ValueError(f"Contrast {contrast!r} not maintained for task {task!r}")
        for filename, kind in (
            [(f"group_contrast_{contrast}_mean.npy", f"group_contrast_{contrast}_mean")]
            + [(f"group_beta_{stem}_mean.npy", f"group_beta_{stem}_mean") for stem in beta_stems]
        ):
            src = group_root / "2mm" / f"hemi_{hemi}" / filename
            if not src.is_file():
                raise FileNotFoundError(f"Missing group map: {src}")
            mean_map = np.load(str(src)).astype(np.float32)
            overlay = out_root / f"hemi_{hemi}_{kind}.func.gii"
            nib.save(make_func_gifti(mean_map), str(overlay))
            products.append({"path": str(overlay.resolve()), "meta": file_meta(overlay)})

    for contrast in contrasts:
        if contrast not in TASK_CONTRASTS[task]:
            raise ValueError(f"Contrast {contrast!r} not maintained for task {task!r}")

    # Render Figures
    if task == "wm" and len(contrasts) >= 9:
        wm_pngs = render_wm_figures(
            group_root=group_root,
            surfaces_dir=surfaces_dir,
            out_root=out_root,
            summary_rows=summary_rows,
            subject_table=subject_table,
            n_subjects=int(group_manifest["n_subjects_total"]),
            hemis=hemis,
        )
        for png_path in wm_pngs:
            products.append({"path": str(png_path.resolve()), "meta": file_meta(png_path)})
            print(json.dumps({"figure": str(png_path)}), flush=True)
    else:
        plt.rcParams.update(
            {
                "font.family": NATURE_FONT["family"],
                "font.sans-serif": NATURE_FONT["sans-serif"],
                "axes.linewidth": 0.8,
                "xtick.major.width": 0.8,
                "ytick.major.width": 0.8,
            }
        )
        n_rows = len(contrasts) * 2
        fig_size = (11, 7.5) if len(contrasts) == 1 else (6 * len(hemis), 4 * n_rows)
        fig, axes = plt.subplots(
            n_rows, len(hemis), figsize=fig_size, squeeze=False
        )
        bracket_tests: list[dict[str, object]] = []
        for ri, contrast in enumerate(contrasts):
            base = ri * 2
            render_task_figure(
                task=task,
                contrast=contrast,
                hemis=hemis,
                group_root=group_root,
                surfaces_dir=surfaces_dir,
                summary_rows=summary_rows,
                subject_table=subject_table,
                n_subjects=int(group_manifest["n_subjects_total"]),
                profile_axes=list(axes[base]),
                bars_axes=list(axes[base + 1]),
                bracket_tests=bracket_tests,
                is_card=False,
            )
            print(json.dumps({"contrast": contrast, "row": base}), flush=True)
        draw_brackets(bracket_tests)
        out_png = out_root / f"task_{task}.png"
        task_label = {
            "emotion": "Emotion: Emotional Faces > Shapes",
            "relational": "Relational: Relational > Matching",
        }.get(task, f"{task.capitalize()}: {contrast_display(contrasts[0]) if len(contrasts) == 1 else ''}")
        fig.suptitle(
            f"HCP 3T Task-fMRI {task_label} (N={group_manifest['n_subjects_total']}, 2mm; markers BH-FDR)",
            fontsize=13,
            fontweight="bold",
            y=0.98,
        )
        fig.text(
            0.5,
            0.01,
            "Markers: +=p<.05, *=p<.01, **=p<.001, ***=p<.0001 (BH-FDR, two-tailed); bins = equal-vertex bins 1-20 (Ant → Post)\n"
            "Beta/Contrast in raw BOLD units (a.u.); black brackets = pairwise subregion difference within parcellation method",
            ha="center",
            fontsize=7.5,
            style="italic",
            color="#334155",
        )
        fig.tight_layout(rect=[0, 0.03, 1, 0.95])
        fig.savefig(out_png, dpi=300)
        plt.close(fig)
        products.append({"path": str(out_png.resolve()), "meta": file_meta(out_png)})
        print(json.dumps({"figure": str(out_png)}), flush=True)

    current_names = {Path(p["path"]).name for p in products}
    current_names.add("task_activation_products_manifest.json")
    legacy_pngs = sorted(
        str(path)
        for path in list(out_root.glob("*.png")) + list(out_root.glob("*/*.png"))
        if path.is_file() and path.name not in current_names
    )
    write_json(
        out_root / "task_activation_products_manifest.json",
        {
            "stage": "task_activation_products",
            "status": "done",
            "timestamp_utc": utc_now_iso(),
            "task": task,
            "smoothing": "2mm",
            "sig_markers": "+: 0.01<=p<0.05; *: 0.001<=p<0.01; **: 0.0001<=p<0.001; ***: p<0.0001 (two-tailed)",
            "group_manifest": str(manifest_path.resolve()),
            "n_subjects_total": group_manifest["n_subjects_total"],
            "contrasts": contrasts,
            "hemis": hemis,
            "legacy_pngs": legacy_pngs,
            "products": products,
        },
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

