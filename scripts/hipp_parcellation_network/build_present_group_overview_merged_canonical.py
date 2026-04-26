#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import tempfile
from pathlib import Path

import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
from matplotlib.colors import ListedColormap
from matplotlib.patches import Rectangle
from mpl_toolkits.axes_grid1 import make_axes_locatable

from surface_utils import long_axis_order, render_locked_grid


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_GROUP_ROOT = REPO_ROOT / "outputs_migration" / "hipp_group_merged_canonical_atlas"
DEFAULT_PRESENT_DIR = REPO_ROOT / "present_group"
DEFAULT_SCENE = REPO_ROOT / "config" / "wb_locked_native_view_lateral_medial.scene"
METHOD_CHOICES = [
    "group-network-spectral",
    "group-network-partial-spectral",
    "group-intrinsic-spectral",
    "group-intrinsic-partial-spectral",
]
METHODS_DEFAULT = [
    "group-network-spectral",
    "group-intrinsic-spectral",
]
ATLASES = ["kong2019", "kong2019_8net", "kong2022"]
SMOOTHINGS = ["2mm", "4mm"]
FS_TITLE = 11
FS_LABEL = 10
FS_TICK = 9
FS_SMALL = 8.5
FS_CELL = 6
FS_MISSING = 13
FS_BIG_TITLE = 13
FS_SUPTITLE = 18


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f, delimiter="\t"))


def style_imshow(ax, data: np.ndarray, title: str, *, vmin: float | None = None, vmax: float | None = None, cmap: str = "magma"):
    arr = np.asarray(data, dtype=np.float32)
    cm = plt.get_cmap(cmap).copy()
    cm.set_bad("#f2f2f2")
    im = ax.imshow(arr, aspect="auto", interpolation="nearest", cmap=cm, vmin=vmin, vmax=vmax)
    ax.set_title(title, fontsize=FS_TITLE)
    return im


def add_cbar(fig, ax, im, label: str, ticks: list[float] | None = None, ticklabels: list[str] | None = None) -> None:
    div = make_axes_locatable(ax)
    cax = div.append_axes("right", size="3%", pad=0.06)
    cb = fig.colorbar(im, cax=cax, ticks=ticks)
    if ticklabels is not None:
        cb.set_ticklabels(ticklabels)
    cb.ax.tick_params(labelsize=FS_TICK)
    cb.set_label(label, fontsize=FS_LABEL)


def concat_lr(a: np.ndarray, la: list[str], b: np.ndarray, lb: list[str]) -> tuple[np.ndarray, list[int], list[str]]:
    sep = np.full((1, a.shape[1]), np.nan, dtype=np.float32)
    mat = np.vstack([a, sep, b]).astype(np.float32)
    yt = list(range(a.shape[0])) + list(range(a.shape[0] + 1, a.shape[0] + 1 + b.shape[0]))
    yl = la + lb
    return mat, yt, yl


def ordered_cluster_ids_by_position(labels: np.ndarray, order: np.ndarray, k: int) -> list[int]:
    x = labels[order]
    rows: list[tuple[float, int]] = []
    for cid in range(1, int(k) + 1):
        idx = np.flatnonzero(x == cid)
        pos = float(np.mean(idx)) if idx.size else float("inf")
        rows.append((pos, int(cid)))
    rows.sort(key=lambda t: (t[0], t[1]))
    return [cid for _, cid in rows]


def resolve_cluster_order_from_manifest(manifest: dict, labels: np.ndarray, order: np.ndarray, k: int) -> list[int]:
    fixed = manifest.get("cluster_order_head_to_tail")
    if isinstance(fixed, list):
        vals = [int(x) for x in fixed]
        expect = set(range(1, int(k) + 1))
        if set(vals) == expect and len(vals) == int(k):
            return vals
    return ordered_cluster_ids_by_position(labels, order, k)


def labels_to_indicator_by_ids(labels: np.ndarray, order: np.ndarray, cluster_ids: list[int]) -> np.ndarray:
    out = np.zeros((len(cluster_ids), int(order.size)), dtype=np.float32)
    x = labels[order]
    for i, cid in enumerate(cluster_ids):
        out[i, :] = (x == int(cid)).astype(np.float32)
    return out


def reorder_profile(profile: np.ndarray, cluster_ids: list[int]) -> np.ndarray:
    idx = [int(cid) - 1 for cid in cluster_ids]
    return profile[np.asarray(idx, dtype=np.int32), :].astype(np.float32)


def annotate_profile_values(ax, data: np.ndarray, vmax: float) -> None:
    if vmax <= 0:
        return
    for r in range(data.shape[0]):
        for c in range(data.shape[1]):
            val = float(data[r, c])
            if not np.isfinite(val):
                continue
            color = "white" if abs(val) >= (0.55 * vmax) else "#222222"
            ax.text(c, r, f"{val:+.2f}", ha="center", va="center", fontsize=FS_CELL, color=color)


def highlight_top1_profile_cells(ax, data: np.ndarray) -> None:
    for r in range(data.shape[0]):
        row = data[r, :]
        if not np.isfinite(row).any():
            continue
        c = int(np.nanargmax(row))
        ax.add_patch(Rectangle((c - 0.5, r - 0.5), 1.0, 1.0, fill=False, edgecolor="#111111", linewidth=1.1))


def confidence_rows(status_payload: dict, hemi: str, cluster_ids: list[int]) -> list[dict[str, object]]:
    by_cluster = {int(row["cluster_id"]): row for row in status_payload.get("items", [])}
    out: list[dict[str, object]] = []
    for display_id, old_id in enumerate(cluster_ids, start=1):
        if int(old_id) not in by_cluster:
            continue
        row = by_cluster[int(old_id)]
        status = str(row.get("status", "dominant"))
        margin = float(row.get("margin", 0.0))
        low_margin = bool(int(row.get("low_margin_lt_0p05", 1 if margin < 0.05 else 0)))
        out.append(
            {
                "label": f"{hemi} C{int(display_id)}",
                "status": status,
                "top1_network": str(row["top1_network"]),
                "top2_network": str(row["top2_network"]),
                "top1_fisher_z": float(row["top1_fisher_z"]),
                "top2_fisher_z": float(row["top2_fisher_z"]),
                "margin": margin,
                "ambiguous": bool(status != "dominant" or low_margin),
            }
        )
    return out


def plot_confidence_panel(ax, rows: list[dict[str, object]], title: str) -> None:
    if not rows:
        ax.axis("off")
        ax.text(0.5, 0.5, "missing", ha="center", va="center", fontsize=FS_MISSING)
        return
    y = np.arange(len(rows), dtype=float)
    top1 = np.array([float(r["top1_fisher_z"]) for r in rows], dtype=float)
    top2 = np.array([float(r["top2_fisher_z"]) for r in rows], dtype=float)
    colors = ["#c97c00" if bool(r["ambiguous"]) else "#2c7fb8" for r in rows]
    labels = [f"{r['label']} {r['top1_network']} | {r['top2_network']}" for r in rows]
    ax.barh(y, top1, height=0.62, color=colors, alpha=0.88, label="1st")
    ax.barh(y, top2, height=0.32, color="none", edgecolor="#111111", linewidth=0.9, label="2nd")
    for yi, row in zip(y, rows, strict=False):
        ax.text(
            max(float(row["top1_fisher_z"]), float(row["top2_fisher_z"])) + 0.005,
            yi,
            f"Δ={float(row['margin']):.3f}",
            va="center",
            ha="left",
            fontsize=FS_TICK,
            color="#c97c00" if bool(row["ambiguous"]) else "#333333",
        )
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=FS_TICK)
    ax.invert_yaxis()
    xmax = max(0.12, float(np.max(top1)) * 1.22)
    ax.set_xlim(0.0, xmax)
    ax.set_xlabel("Fisher-z", fontsize=FS_LABEL)
    ax.set_title(title, fontsize=FS_TITLE)
    ax.tick_params(labelsize=FS_TICK)
    ax.axvline(0.05, color="#999999", lw=0.8, ls=":")
    ax.grid(axis="x", color="#dddddd", lw=0.5, alpha=0.7)
    ax.legend(fontsize=FS_TICK, loc="lower right")


def render_combo(
    *,
    group_root: Path,
    present_dir: Path,
    method: str,
    atlas: str,
    coverage_tag: str,
    scene: Path,
) -> dict[str, object]:
    fig = plt.figure(figsize=(15.4, 12.2), dpi=180, constrained_layout=True)
    gs = fig.add_gridspec(nrows=2, ncols=2, width_ratios=[1.40, 1.30], hspace=0.06, wspace=0.05)
    rows_meta: list[dict[str, object]] = []

    for ridx, smoothing in enumerate(SMOOTHINGS):
        left_manifest = group_root / method / atlas / smoothing / coverage_tag / "hemi_L" / "group_atlas_manifest.json"
        right_manifest = group_root / method / atlas / smoothing / coverage_tag / "hemi_R" / "group_atlas_manifest.json"
        if not (left_manifest.exists() and right_manifest.exists()):
            sub = gs[ridx, 0].subgridspec(2, 2, wspace=0.08, hspace=0.035)
            for rr in range(2):
                for cc in range(2):
                    ax = fig.add_subplot(sub[rr, cc])
                    ax.axis("off")
                    ax.text(0.5, 0.5, f"{smoothing} missing", ha="center", va="center", fontsize=FS_MISSING)
            ax_big = fig.add_subplot(gs[ridx, 1])
            ax_big.axis("off")
            ax_big.text(0.5, 0.5, f"{smoothing} missing", ha="center", va="center", fontsize=FS_MISSING)
            continue

        m_l = load_json(left_manifest)
        m_r = load_json(right_manifest)
        d_l = left_manifest.parent
        d_r = right_manifest.parent
        prof_l = np.load(d_l / "cluster_network_profile.npy").astype(np.float32)
        prof_r = np.load(d_r / "cluster_network_profile.npy").astype(np.float32)
        lbl_l = np.load(d_l / "group_labels_final.npy").astype(np.int32)
        lbl_r = np.load(d_r / "group_labels_final.npy").astype(np.int32)
        status_l = load_json(d_l / "cluster_assignment_status.json")
        status_r = load_json(d_r / "cluster_assignment_status.json")

        surf_l = Path(str(m_l["template_left_or_right_surface"]))
        surf_r = Path(str(m_r["template_left_or_right_surface"]))
        order_l = long_axis_order(surf_l)
        order_r = long_axis_order(surf_r)

        k_l = int(m_l["chosen_k_final"])
        k_r = int(m_r["chosen_k_final"])
        cids_l = resolve_cluster_order_from_manifest(m_l, lbl_l, order_l, k_l)
        cids_r = resolve_cluster_order_from_manifest(m_r, lbl_r, order_r, k_r)
        ind_l = labels_to_indicator_by_ids(lbl_l, order_l, cids_l)
        ind_r = labels_to_indicator_by_ids(lbl_r, order_r, cids_r)
        prof_l_o = reorder_profile(prof_l, cids_l)
        prof_r_o = reorder_profile(prof_r, cids_r)
        row_lbl_l = [f"L-C{i}" for i in range(1, len(cids_l) + 1)]
        row_lbl_r = [f"R-C{i}" for i in range(1, len(cids_r) + 1)]
        ind_lr, yt_ind, yl_ind = concat_lr(ind_l, row_lbl_l, ind_r, row_lbl_r)

        sub = gs[ridx, 0].subgridspec(2, 2, wspace=0.08, hspace=0.035)
        ax0 = fig.add_subplot(sub[0, 0])
        rows_l = read_tsv(d_l / "k_selection_summary.tsv")
        rows_r = read_tsv(d_r / "k_selection_summary.tsv")
        for hemi, rows, color in [("L", rows_l, "#1f77b4"), ("R", rows_r, "#d62728")]:
            rows = sorted(rows, key=lambda r: int(r["k"]))
            ks = np.array([int(r["k"]) for r in rows], dtype=int)
            mean = np.array([float(r["group_ari_mean"]) for r in rows], dtype=float)
            se = np.array([float(r["group_ari_se"]) for r in rows], dtype=float)
            ax0.plot(ks, mean, marker="o", lw=1.3, color=color, label=hemi)
            ax0.fill_between(ks, mean - se, mean + se, color=color, alpha=0.18)
            null_ref = np.array([float(r["ARI_null_5pct"]) for r in rows], dtype=float)
            ax0.plot(ks, null_ref, ls=":", lw=1.0, color=color, alpha=0.85)
        ax0.axvline(k_l, color="#1f77b4", ls="--", lw=1.0)
        ax0.axvline(k_r, color="#d62728", ls="--", lw=1.0)
        ax0.text(
            0.02,
            0.02,
            "dotted = strong-null ARI 5th pct\nvertical dashed = selected K*",
            transform=ax0.transAxes,
            ha="left",
            va="bottom",
            fontsize=FS_SMALL,
            color="#333333",
            bbox={"facecolor": "white", "alpha": 0.72, "edgecolor": "none", "pad": 1.5},
        )
        ax0.set_title(f"{smoothing} | K selection (ARI)", fontsize=FS_TITLE)
        ax0.set_xlabel("K", fontsize=FS_LABEL)
        ax0.set_ylabel("Subject-split ARI", fontsize=FS_LABEL)
        ax0.tick_params(labelsize=FS_TICK)
        ax0.legend(fontsize=FS_TICK, loc="upper left")

        ax1 = fig.add_subplot(sub[0, 1])
        prof_lr, yt_prof, yl_prof = concat_lr(prof_l_o, row_lbl_l, prof_r_o, row_lbl_r)
        vmax = float(np.nanmax(np.abs(prof_lr))) if np.isfinite(prof_lr).any() else 1.0
        im1 = style_imshow(ax1, prof_lr, f"{smoothing} | cluster_network_profile (signed)", vmin=-vmax, vmax=vmax, cmap="coolwarm")
        ax1.set_yticks(yt_prof)
        ax1.set_yticklabels(yl_prof, fontsize=FS_LABEL)
        nets = [str(x) for x in load_json(d_l / "group_feature_summary.json")["network_names"]]
        ax1.set_xticks(np.arange(len(nets)))
        ax1.set_xticklabels(nets, fontsize=FS_TICK, rotation=35, ha="right")
        annotate_profile_values(ax1, prof_lr, vmax)
        highlight_top1_profile_cells(ax1, prof_lr)
        add_cbar(fig, ax1, im1, "Fisher-z")

        ax2 = fig.add_subplot(sub[1, 0])
        im2 = style_imshow(
            ax2,
            ind_lr,
            f"{smoothing} | cluster_indicator (KxV, C-order head->tail)",
            vmin=0.0,
            vmax=1.0,
            cmap=ListedColormap(["#d9e4f5", "#1f77b4"]),
        )
        ax2.set_yticks(yt_ind)
        ax2.set_yticklabels(yl_ind, fontsize=FS_LABEL)
        ax2.set_xlabel("Vertex (long-axis)", fontsize=FS_LABEL)
        ax2.tick_params(labelsize=FS_TICK)
        add_cbar(fig, ax2, im2, "Indicator", ticks=[0.0, 1.0], ticklabels=["0", "1"])

        ax3 = fig.add_subplot(sub[1, 1])
        conf_rows = confidence_rows(status_l, "L", cids_l) + confidence_rows(status_r, "R", cids_r)
        plot_confidence_panel(ax3, conf_rows, f"{smoothing} | cluster confidence / ambiguity")

        ax_big = fig.add_subplot(gs[ridx, 1])
        left_label = Path(str(m_l.get("template_functional_atlas_label", m_l["template_cluster_label"])))
        right_label = Path(str(m_r.get("template_functional_atlas_label", m_r["template_cluster_label"])))
        spec_path = Path(str(m_l["template_spec_path"]))
        template_subject = str(m_l["template_subject"]).replace("sub-", "")
        with tempfile.TemporaryDirectory(prefix="present_group_surface_render_") as tmp:
            render = render_locked_grid(
                subject=template_subject,
                scene=scene,
                outdir=Path(tmp),
                name=f"merged_group_cluster_wta_{method}_{atlas}_{smoothing}",
                title=f"Group functional atlas {method} {atlas} {smoothing}",
                left_labels=left_label,
                right_labels=right_label,
                left_surface=surf_l,
                right_surface=surf_r,
                spec_path=spec_path,
            )
            big = Path(str(render["biglegend_png"]))
            ax_big.imshow(plt.imread(big))
        ax_big.set_title(f"{smoothing} | functional atlas surface", fontsize=FS_BIG_TITLE)
        ax_big.axis("off")

        rows_meta.append(
            {
                "smoothing": smoothing,
                "k_final_L": k_l,
                "k_final_R": k_r,
                "n_subjects": int(m_l["n_subjects"]),
            }
        )

    out_png = present_dir / f"{atlas}_{method}_group_overview.png"
    fig.suptitle(f"Group Overview | {atlas} | {method}", fontsize=FS_SUPTITLE)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=300)
    plt.close(fig)
    return {"overview_png": str(out_png), "rows": rows_meta}


def write_readme(present_dir: Path, items: list[dict[str, object]]) -> None:
    lines = [
        "# present_group merged-canonical overview",
        "",
        "| atlas | method | overview | smoothings | key metrics |",
        "|---|---|---|---|---|",
    ]
    for item in sorted(items, key=lambda x: (str(x["atlas"]), str(x["method"]))):
        png = Path(str(item["overview_png"])).name
        sm = ", ".join(str(s) for s in item.get("smoothings", []))
        metric = str(item.get("metric_summary", ""))
        lines.append(f"| {item['atlas']} | {item['method']} | [{png}]({png}) | {sm} | {metric} |")
    (present_dir / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    p = argparse.ArgumentParser(description="Build present_group overviews from merged-canonical group atlas outputs.")
    p.add_argument("--group-root", default=str(DEFAULT_GROUP_ROOT))
    p.add_argument("--present-dir", default=str(DEFAULT_PRESENT_DIR))
    p.add_argument("--methods", nargs="+", default=METHODS_DEFAULT, choices=METHOD_CHOICES)
    p.add_argument("--atlases", nargs="+", default=ATLASES)
    p.add_argument("--coverage-threshold", type=float, default=0.80)
    p.add_argument("--scene", default=str(DEFAULT_SCENE))
    args = p.parse_args()

    group_root = Path(args.group_root).resolve()
    present_dir = Path(args.present_dir).resolve()
    present_dir.mkdir(parents=True, exist_ok=True)
    tau_tag = f"cov_{args.coverage_threshold:.2f}".replace(".", "p")

    items: list[dict[str, object]] = []
    for atlas in args.atlases:
        for method in args.methods:
            meta = render_combo(
                group_root=group_root,
                present_dir=present_dir,
                method=method,
                atlas=atlas,
                coverage_tag=tau_tag,
                scene=Path(args.scene).resolve(),
            )
            row_meta = meta["rows"]
            metric = "; ".join([f"{r['smoothing']}: K(L/R)={r['k_final_L']}/{r['k_final_R']} N={r['n_subjects']}" for r in row_meta]) if row_meta else "missing"
            items.append(
                {
                    "atlas": atlas,
                    "method": method,
                    "overview_png": meta["overview_png"],
                    "smoothings": [r["smoothing"] for r in row_meta],
                    "metric_summary": metric,
                }
            )

    manifest = {"present_dir": str(present_dir), "count": len(items), "items": items}
    (present_dir / "group_overview_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    write_readme(present_dir, items)
    print(json.dumps({"present_dir": str(present_dir), "count": len(items)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
