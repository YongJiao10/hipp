#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import pickle
import shutil
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from mpl_toolkits.axes_grid1 import make_axes_locatable


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SPECTRAL_ROOT = REPO_ROOT / "outputs_migration" / "hipp_group_prior_fastpfm"
DEFAULT_PRESENT_DIR = REPO_ROOT / "present_group"
NETWORK_STYLE_JSON = REPO_ROOT / "config" / "hipp_network_style.json"
SMOOTHINGS = ["2mm", "4mm"]


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def resolve_existing_path(path_str: str) -> Path:
    p = Path(path_str)
    if p.exists():
        return p
    raise FileNotFoundError(f"Missing file path in manifest: {path_str}")


def load_network_names_from_group_manifest(group_manifest: Path) -> list[str]:
    payload = load_json(group_manifest)
    left_prior = resolve_existing_path(str(payload["hemi_priors"]["L"]["prior_pickle"]))
    prior_obj = pickle.loads(left_prior.read_bytes())
    return [str(x) for x in prior_obj["profile_networks"]]


def combine_lr(a: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, list[float], list[str]]:
    if a.ndim != 2 or b.ndim != 2:
        raise ValueError("combine_lr expects 2D arrays")
    rows = max(a.shape[0], b.shape[0])

    def pad_rows(x: np.ndarray) -> np.ndarray:
        if x.shape[0] == rows:
            return x
        pad = np.full((rows - x.shape[0], x.shape[1]), np.nan, dtype=np.float32)
        return np.vstack([x, pad])

    a2 = pad_rows(a.astype(np.float32))
    b2 = pad_rows(b.astype(np.float32))
    sep = np.full((1, a2.shape[1]), np.nan, dtype=np.float32)
    stacked = np.vstack([a2, sep, b2])
    yticks = [rows / 2 - 0.5, rows + 0.5 + rows / 2 - 0.5]
    ylabels = ["L", "R"]
    return stacked, yticks, ylabels


def combine_lr_with_named_rows(
    a: np.ndarray,
    b: np.ndarray,
    *,
    left_prefix: str,
    right_prefix: str,
    base_names: list[str],
) -> tuple[np.ndarray, list[float], list[str]]:
    """Stack L/R matrices with one separator row and per-row labels."""
    if a.ndim != 2 or b.ndim != 2:
        raise ValueError("combine_lr_with_named_rows expects 2D arrays")
    rows = max(a.shape[0], b.shape[0], len(base_names))
    a2 = np.full((rows, a.shape[1]), np.nan, dtype=np.float32)
    b2 = np.full((rows, b.shape[1]), np.nan, dtype=np.float32)
    a2[: a.shape[0], :] = a.astype(np.float32)
    b2[: b.shape[0], :] = b.astype(np.float32)
    sep = np.full((1, a2.shape[1]), np.nan, dtype=np.float32)
    stacked = np.vstack([a2, sep, b2])

    names = list(base_names) + [""] * max(0, rows - len(base_names))
    ylabels = [f"{left_prefix}{n}" if n else "" for n in names] + [""] + [f"{right_prefix}{n}" if n else "" for n in names]
    yticks = list(np.arange(stacked.shape[0]))
    return stacked, yticks, ylabels


def read_group_k_rows(tsv_path: Path) -> list[dict[str, str]]:
    with tsv_path.open("r", encoding="utf-8") as f:
        return list(csv.DictReader(f, delimiter="\t"))


def collect_spectral_manifests(spectral_root: Path) -> dict[tuple[str, str, str], Path]:
    out: dict[tuple[str, str, str], Path] = {}
    for manifest in spectral_root.glob("*/*/*/group_prior_manifest.json"):
        branch = manifest.parents[2].name
        atlas = manifest.parents[1].name
        smoothing = manifest.parents[0].name
        out[(branch, atlas, smoothing)] = manifest
    return out


def style_and_imshow(ax, mat: np.ndarray, title: str, cmap: str = "magma"):
    data = np.array(mat, dtype=np.float32)
    if np.isfinite(data).any():
        vmax = float(np.nanmax(data))
    else:
        vmax = 1.0
    cm = plt.get_cmap(cmap).copy()
    cm.set_bad("#f2f2f2")
    im = ax.imshow(data, aspect="auto", interpolation="nearest", cmap=cm, vmin=0.0, vmax=max(vmax, 1e-6))
    ax.set_title(title, fontsize=10)
    return im


def add_compact_colorbar(fig, ax, im, *, label: str = "P") -> None:
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="3%", pad=0.08)
    cbar = fig.colorbar(im, cax=cax)
    cbar.ax.tick_params(labelsize=7)
    cbar.set_label(label, fontsize=8)


def draw_pnet_panel(
    *,
    fig,
    ax,
    p_net_v: np.ndarray,
    title: str,
    yticks: list[float] | np.ndarray,
    yticklabels: list[str],
    xlabel: str = "Vertex",
) -> None:
    im = style_and_imshow(ax, p_net_v, title)
    ax.set_yticks(yticks)
    ax.set_yticklabels(yticklabels, fontsize=7)
    ax.set_xlabel(xlabel, fontsize=8)
    ax.tick_params(labelsize=7)
    add_compact_colorbar(fig, ax, im, label="P")


def save_pnet_component_png(
    *,
    p_net_v: np.ndarray,
    title: str,
    out_png: Path,
    yticklabels: list[str],
) -> None:
    save_heatmap_png(
        p_net_v,
        title,
        out_png,
        yticklabels=yticklabels,
        add_colorbar=True,
    )


def annotate_heatmap_values(ax, data: np.ndarray, *, fmt: str = "{:.2f}", fontsize: int = 6) -> None:
    arr = np.asarray(data)
    if arr.ndim != 2:
        return
    for i in range(arr.shape[0]):
        for j in range(arr.shape[1]):
            v = arr[i, j]
            if np.isfinite(v):
                ax.text(j, i, fmt.format(float(v)), ha="center", va="center", fontsize=fontsize, color="white")


def save_heatmap_png(
    mat: np.ndarray,
    title: str,
    out_png: Path,
    *,
    cmap: str = "magma",
    yticklabels: list[str] | None = None,
    annotate_values: bool = False,
    value_fmt: str = "{:.2f}",
    add_colorbar: bool = False,
) -> None:
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(12, 3.6), dpi=220)
    im = style_and_imshow(ax, mat, title, cmap=cmap)
    if yticklabels is not None:
        ax.set_yticks(np.arange(len(yticklabels)))
        ax.set_yticklabels(yticklabels, fontsize=8)
    if annotate_values:
        annotate_heatmap_values(ax, np.asarray(mat), fmt=value_fmt, fontsize=6)
    if add_colorbar:
        cbar = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
        cbar.ax.tick_params(labelsize=7)
    ax.set_xlabel("Vertex/Network axis")
    fig.tight_layout()
    fig.savefig(out_png, dpi=300)
    plt.close(fig)


def save_k_curve_png(rows: list[dict[str, str]], title: str, out_png: Path, *, kL: int, kR: int) -> None:
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6.8, 3.6), dpi=220)
    for hemi, color in [("L", "#1f77b4"), ("R", "#d62728")]:
        hrows = [r for r in rows if r["hemi"] == hemi]
        hrows = sorted(hrows, key=lambda r: int(r["k"]))
        ks = np.array([int(r["k"]) for r in hrows], dtype=int)
        mean = np.array([float(r["group_instability_mean"]) for r in hrows], dtype=float)
        se = np.array([float(r["group_instability_se"]) for r in hrows], dtype=float)
        ax.plot(ks, mean, marker="o", lw=1.8, color=color, label=hemi)
        ax.fill_between(ks, mean - se, mean + se, color=color, alpha=0.18)
    ax.axvline(kL, color="#1f77b4", ls="--", lw=1.0)
    ax.axvline(kR, color="#d62728", ls="--", lw=1.0)
    ax.set_title(title)
    ax.set_xlabel("K")
    ax.set_ylabel("Instability mean±SE")
    ax.legend(loc="upper left", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_png, dpi=300)
    plt.close(fig)


def render_spectral_overview(
    *,
    atlas: str,
    method: str,
    manifests_by_smoothing: dict[str, Path],
    out_png: Path,
) -> dict[str, object]:
    fig = plt.figure(figsize=(15.4, 12.2), dpi=180, constrained_layout=True)
    gs = fig.add_gridspec(nrows=2, ncols=2, width_ratios=[1.40, 1.30], hspace=0.10, wspace=0.05)

    summary_rows = []
    comp_root = out_png.parent / "components" / atlas / method
    for ridx, smoothing in enumerate(SMOOTHINGS):
        manifest = manifests_by_smoothing.get(smoothing)
        if manifest is None:
            sub = gs[ridx, 0].subgridspec(2, 2, wspace=0.08, hspace=0.08)
            for rr in range(2):
                for cc in range(2):
                    ax = fig.add_subplot(sub[rr, cc])
                    ax.axis("off")
                    ax.text(0.5, 0.5, f"{smoothing} missing", ha="center", va="center", fontsize=12)
            ax_big = fig.add_subplot(gs[ridx, 1])
            ax_big.axis("off")
            ax_big.text(0.5, 0.5, f"{smoothing} missing", ha="center", va="center", fontsize=12)
            continue

        sub = gs[ridx, 0].subgridspec(2, 2, wspace=0.08, hspace=0.08)

        m = load_json(manifest)
        pri_l = resolve_existing_path(str(m["hemi_priors"]["L"]["prior_pickle"]))
        pri_r = resolve_existing_path(str(m["hemi_priors"]["R"]["prior_pickle"]))
        p_l = pickle.loads(pri_l.read_bytes())
        p_r = pickle.loads(pri_r.read_bytes())

        prior_l = np.asarray(p_l["prior_matrix"], dtype=np.float32)
        prior_r = np.asarray(p_r["prior_matrix"], dtype=np.float32)
        cnp_l = np.asarray(p_l["cluster_network_probs"], dtype=np.float32)
        cnp_r = np.asarray(p_r["cluster_network_probs"], dtype=np.float32)
        pnet_l = (cnp_l.T @ prior_l).astype(np.float32)
        pnet_r = (cnp_r.T @ prior_r).astype(np.float32)

        net_names = [str(x) for x in p_l["profile_networks"]]

        order_l_path = manifest.parent / "priors" / f"p_net_v_{method}_{atlas}_{smoothing}_hemi-L_longaxis_order.npy"
        order_r_path = manifest.parent / "priors" / f"p_net_v_{method}_{atlas}_{smoothing}_hemi-R_longaxis_order.npy"
        if order_l_path.exists():
            o_l = np.load(order_l_path)
            prior_l = prior_l[:, o_l]
            pnet_l = pnet_l[:, o_l]
        if order_r_path.exists():
            o_r = np.load(order_r_path)
            prior_r = prior_r[:, o_r]
            pnet_r = pnet_r[:, o_r]

        # Panel 1: group K curve (L/R instability)
        ax0 = fig.add_subplot(sub[0, 0])
        tsv_rows = read_group_k_rows(resolve_existing_path(str(m["group_k_selection_tsv"])))
        for hemi, color in [("L", "#1f77b4"), ("R", "#d62728")]:
            rows = [r for r in tsv_rows if r["hemi"] == hemi]
            rows = sorted(rows, key=lambda r: int(r["k"]))
            ks = np.array([int(r["k"]) for r in rows], dtype=int)
            mean = np.array([float(r["group_instability_mean"]) for r in rows], dtype=float)
            se = np.array([float(r["group_instability_se"]) for r in rows], dtype=float)
            ax0.plot(ks, mean, marker="o", lw=1.4, color=color, label=hemi)
            ax0.fill_between(ks, mean - se, mean + se, color=color, alpha=0.18)

        kL = int(m["hemi_priors"]["L"]["k_final"])
        kR = int(m["hemi_priors"]["R"]["k_final"])
        ax0.axvline(kL, color="#1f77b4", ls="--", lw=1.0)
        ax0.axvline(kR, color="#d62728", ls="--", lw=1.0)
        ax0.set_title(f"{smoothing} | Group K selection", fontsize=9)
        ax0.set_xlabel("K", fontsize=8)
        ax0.set_ylabel("Instability", fontsize=8)
        ax0.tick_params(labelsize=8)
        ax0.legend(loc="upper left", fontsize=7)

        # Panel 2: cluster_network_probs(KxN), L/R stacked + numeric labels
        ax1 = fig.add_subplot(sub[0, 1])
        cnp_lr, yt1, yl1 = combine_lr(cnp_l, cnp_r)
        style_and_imshow(ax1, cnp_lr, f"{smoothing} | cluster_network_probs (KxN)")
        ax1.set_yticks(yt1)
        ax1.set_yticklabels(yl1, fontsize=8)
        ax1.set_xlabel("Network", fontsize=8)
        ax1.tick_params(labelsize=7)
        annotate_heatmap_values(ax1, cnp_lr, fmt="{:.2f}", fontsize=6)

        # Panel 3: P_net_v(NxV), L/R stacked + network names
        ax2 = fig.add_subplot(sub[1, 0])
        pnet_lr, yt2, yl2 = combine_lr_with_named_rows(
            pnet_l,
            pnet_r,
            left_prefix="L-",
            right_prefix="R-",
            base_names=net_names,
        )
        draw_pnet_panel(
            fig=fig,
            ax=ax2,
            p_net_v=pnet_lr,
            title=f"{smoothing} | P_net_v (NxV)",
            yticks=yt2,
            yticklabels=yl2,
            xlabel="Vertex",
        )

        # Panel 4: WTA surface
        ax3 = fig.add_subplot(sub[1, 1])
        template_subject = str(m.get("template_subject", "100610"))
        pnet_render = manifest.parent / "template" / "renders" / f"sub-{template_subject}_wb_p_net_v_wta_{method}_{atlas}_{smoothing}_biglegend.png"
        if not pnet_render.exists():
            pnet_render = manifest.parent / "template" / "renders" / f"sub-{template_subject}_wb_group_netprob_wta_{method}_{atlas}_{smoothing}_biglegend.png"
        prior_lr, yt3, yl3 = combine_lr(prior_l, prior_r)
        style_and_imshow(ax3, prior_lr, f"{smoothing} | prior_matrix (KxV), L/R")
        ax3.set_yticks(yt3)
        ax3.set_yticklabels(yl3, fontsize=8)
        ax3.set_xlabel("Vertex", fontsize=8)
        ax3.tick_params(labelsize=7)

        ax_big = fig.add_subplot(gs[ridx, 1])
        if pnet_render.exists():
            ax_big.imshow(plt.imread(pnet_render))
            ax_big.set_title(f"{smoothing} | WTA surface", fontsize=11)
            ax_big.axis("off")
        else:
            ax_big.axis("off")
            ax_big.text(0.5, 0.5, f"Missing WTA render\n{pnet_render.name}", ha="center", va="center", fontsize=9)

        comp_sm = comp_root / smoothing
        save_k_curve_png(tsv_rows, f"{atlas} {method} {smoothing} group-K", comp_sm / "k_selection_curve.png", kL=kL, kR=kR)
        save_heatmap_png(prior_l, f"{atlas} {method} {smoothing} prior_matrix hemi-L", comp_sm / "prior_matrix_hemi-L.png")
        save_heatmap_png(prior_r, f"{atlas} {method} {smoothing} prior_matrix hemi-R", comp_sm / "prior_matrix_hemi-R.png")
        save_heatmap_png(
            cnp_l,
            f"{atlas} {method} {smoothing} cluster_network_probs hemi-L",
            comp_sm / "cluster_network_probs_hemi-L.png",
            annotate_values=True,
            value_fmt="{:.2f}",
        )
        save_heatmap_png(
            cnp_r,
            f"{atlas} {method} {smoothing} cluster_network_probs hemi-R",
            comp_sm / "cluster_network_probs_hemi-R.png",
            annotate_values=True,
            value_fmt="{:.2f}",
        )
        save_pnet_component_png(
            p_net_v=pnet_l,
            title=f"{atlas} {method} {smoothing} p_net_v hemi-L",
            out_png=comp_sm / "p_net_v_hemi-L.png",
            yticklabels=net_names,
        )
        save_pnet_component_png(
            p_net_v=pnet_r,
            title=f"{atlas} {method} {smoothing} p_net_v hemi-R",
            out_png=comp_sm / "p_net_v_hemi-R.png",
            yticklabels=net_names,
        )
        if pnet_render.exists():
            shutil.copy2(pnet_render, comp_sm / "wta_surface.png")

        summary_rows.append(
            {
                "smoothing": smoothing,
                "k_final_L": kL,
                "k_final_R": kR,
                "n_subjects": int([r for r in tsv_rows if r["hemi"] == "L"][0]["n_subjects"]),
                "components": {
                    "k_curve": str(comp_sm / "k_selection_curve.png"),
                    "prior_hemi_L": str(comp_sm / "prior_matrix_hemi-L.png"),
                    "prior_hemi_R": str(comp_sm / "prior_matrix_hemi-R.png"),
                    "cluster_network_probs_hemi_L": str(comp_sm / "cluster_network_probs_hemi-L.png"),
                    "cluster_network_probs_hemi_R": str(comp_sm / "cluster_network_probs_hemi-R.png"),
                    "p_net_v_hemi_L": str(comp_sm / "p_net_v_hemi-L.png"),
                    "p_net_v_hemi_R": str(comp_sm / "p_net_v_hemi-R.png"),
                    "wta_surface": str(comp_sm / "wta_surface.png"),
                },
            }
        )

    fig.suptitle(f"Group Overview | {atlas} | {method}", fontsize=16)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=300)
    plt.close(fig)
    return {"rows": summary_rows}


def write_readme(*, present_dir: Path, items: list[dict[str, object]]) -> None:
    lines = [
        "# present_group overview",
        "",
        "| atlas | method | overview | smoothings | key metrics |",
        "|---|---|---|---|---|",
    ]
    for item in sorted(items, key=lambda x: (str(x["atlas"]), str(x["method"]))):
        atlas = str(item["atlas"])
        method = str(item["method"])
        png_name = Path(str(item["overview_png"])).name
        smoothings = ", ".join(str(s) for s in item.get("smoothings", []))
        metric = str(item.get("metric_summary", ""))
        comp_dir = Path(str(item.get("components_dir", ""))).name
        comp_link = f"[components](components/{atlas}/{method}/)" if comp_dir else ""
        lines.append(f"| {atlas} | {method} | [{png_name}]({png_name}) {comp_link} | {smoothings} | {metric} |")
    (present_dir / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Build atlas+method group overviews into present_group")
    parser.add_argument("--spectral-root", default=str(DEFAULT_SPECTRAL_ROOT))
    parser.add_argument("--present-dir", default=str(DEFAULT_PRESENT_DIR))
    args = parser.parse_args()

    spectral_root = Path(args.spectral_root).resolve()
    present_dir = Path(args.present_dir).resolve()
    present_dir.mkdir(parents=True, exist_ok=True)

    items_out: list[dict[str, object]] = []



    spectral_manifests = collect_spectral_manifests(spectral_root)

    atlases = sorted({a for (_b, a, _s) in spectral_manifests.keys()})

    # spectral methods
    for method in ["network-spectral", "intrinsic-spectral"]:
        for atlas in atlases:
            by_sm: dict[str, Path] = {}
            for sm in SMOOTHINGS:
                m = spectral_manifests.get((method, atlas, sm))
                if m is not None:
                    by_sm[sm] = m
            if not by_sm:
                continue

            out_png = present_dir / f"{atlas}_{method}_group_overview.png"
            meta = render_spectral_overview(atlas=atlas, method=method, manifests_by_smoothing=by_sm, out_png=out_png)
            smoothings = [r["smoothing"] for r in meta["rows"]]
            metric = "; ".join(
                [
                    f"{r['smoothing']}: K(L/R)={r['k_final_L']}/{r['k_final_R']} N={r['n_subjects']}"
                    for r in meta["rows"]
                ]
            )
            items_out.append(
                {
                    "atlas": atlas,
                    "method": method,
                    "overview_png": str(out_png),
                    "smoothings": smoothings,
                    "metric_summary": metric,
                    "row_details": meta["rows"],
                    "components_dir": str((present_dir / "components" / atlas / method).resolve()),
                }
            )

    manifest = {
        "present_dir": str(present_dir),
        "count": len(items_out),
        "items": items_out,
    }
    (present_dir / "group_overview_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    write_readme(present_dir=present_dir, items=items_out)

    print(json.dumps({"present_dir": str(present_dir), "count": len(items_out)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
