#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import pickle
import re
import subprocess
from pathlib import Path

import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
from nibabel.gifti import GiftiDataArray, GiftiImage, GiftiLabel, GiftiLabelTable


REPO_ROOT = Path(__file__).resolve().parents[2]
import sys
PYTHON_EXE = sys.executable or "python3"
RENDER_SCENE_BATCH = REPO_ROOT / "scripts" / "workbench" / "render_wb_scene_batch.py"
COMPOSE_WB_GRID = REPO_ROOT / "scripts" / "workbench" / "compose_wb_grid_with_legend.py"
NETWORK_STYLE_JSON = REPO_ROOT / "config" / "hipp_network_style.json"
DEFAULT_SCENE = REPO_ROOT / "config" / "wb_locked_native_view_lateral_medial.scene"
DEFAULT_GROUP_ROOT = REPO_ROOT / "outputs_migration" / "hipp_group_prior_fastpfm"


def run(cmd: list[str]) -> None:
    proc = subprocess.run(cmd, text=True, capture_output=True)
    if proc.returncode != 0:
        raise RuntimeError(
            f"Command failed ({proc.returncode}): {' '.join(cmd)}\\nSTDOUT:\\n{proc.stdout}\\nSTDERR:\\n{proc.stderr}"
        )


def load_network_colors(path: Path) -> dict[str, tuple[int, int, int, int]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    out: dict[str, tuple[int, int, int, int]] = {}
    for _, row in payload.items():
        name = str(row["name"])
        rgba = tuple(int(v) for v in row["rgba"])
        out[name] = rgba
    return out


def long_axis_order(surface_path: Path) -> np.ndarray:
    img = nib.load(str(surface_path))
    coords = None
    for da in img.darrays:
        if da.intent == "NIFTI_INTENT_POINTSET":
            coords = np.asarray(da.data, dtype=np.float64)
            break
    if coords is None:
        coords = np.asarray(img.darrays[0].data, dtype=np.float64)
    centered = coords - np.mean(coords, axis=0, keepdims=True)
    _u, _s, vt = np.linalg.svd(centered, full_matrices=False)
    axis_scores = centered @ vt[0]
    return np.argsort(axis_scores, kind="stable").astype(np.int32)


def infer_density_from_surface(path: Path) -> str:
    m = re.search(r"_den-([A-Za-z0-9]+)_label-", path.name)
    if not m:
        raise ValueError(f"Unable to infer density from surface path: {path}")
    return m.group(1)


def infer_spec_path(subject: str, surface_path: Path) -> Path:
    density = infer_density_from_surface(surface_path)
    spec_path = surface_path.parent / f"sub-{subject}_den-{density}_label-hipp_surfaces.spec"
    if not spec_path.exists():
        raise FileNotFoundError(f"Missing spec file: {spec_path}")
    return spec_path


def make_network_label_gifti(
    *,
    labels: np.ndarray,
    network_names: list[str],
    network_colors: dict[str, tuple[int, int, int, int]],
) -> GiftiImage:
    table = GiftiLabelTable()
    null_label = GiftiLabel(key=0, red=0.62, green=0.62, blue=0.62, alpha=1.0)
    null_label.label = "Null"
    table.labels.append(null_label)

    for idx, name in enumerate(network_names, start=1):
        rgba = network_colors.get(name, (128, 128, 128, 255))
        label = GiftiLabel(
            key=int(idx),
            red=float(rgba[0]) / 255.0,
            green=float(rgba[1]) / 255.0,
            blue=float(rgba[2]) / 255.0,
            alpha=float(rgba[3]) / 255.0,
        )
        label.label = str(name)
        table.labels.append(label)

    arr = GiftiDataArray(data=labels.astype(np.int32), intent="NIFTI_INTENT_LABEL", datatype="NIFTI_TYPE_INT32")
    return GiftiImage(darrays=[arr], labeltable=table)


def render_locked_grid(
    *,
    subject: str,
    scene: Path,
    outdir: Path,
    name: str,
    title: str,
    left_labels: Path,
    right_labels: Path,
    left_surface: Path,
    right_surface: Path,
    spec_path: Path,
) -> dict[str, str]:
    outdir.mkdir(parents=True, exist_ok=True)
    native_root = outdir / "native"
    native_root.mkdir(parents=True, exist_ok=True)
    cmd_render = [
        PYTHON_EXE,
        str(RENDER_SCENE_BATCH),
        "--scene",
        str(scene),
        "--subjects",
        subject,
        "--outdir",
        str(native_root / "batch"),
        "--scene-index",
        "1",
        "--width",
        "1600",
        "--height",
        "1024",
        "--renderer",
        "OSMesa",
        "--name",
        f"{name}_native",
        "--no-template-scene",
        "--left-label-template",
        str(left_labels),
        "--right-label-template",
        str(right_labels),
        "--left-surface-template",
        str(left_surface),
        "--right-surface-template",
        str(right_surface),
        "--spec-template",
        str(spec_path),
    ]
    run(cmd_render)
    native_png = native_root / "batch" / f"sub-{subject}" / f"sub-{subject}_wb_{name}_native_native.png"
    final_png = outdir / f"sub-{subject}_wb_{name}_biglegend.png"
    cmd_compose = [
        PYTHON_EXE,
        str(COMPOSE_WB_GRID),
        "--ventral-image",
        str(native_png),
        "--layout",
        "1x2",
        "--left-labels",
        str(left_labels),
        "--right-labels",
        str(right_labels),
        "--legend-group",
        "label",
        "--title",
        title,
        "--out",
        str(final_png),
    ]
    run(cmd_compose)
    return {"native_scene_png": str(native_png), "biglegend_png": str(final_png)}


def save_heatmap_png(
    *,
    p_net_v: np.ndarray,
    network_names: list[str],
    out_png: Path,
) -> None:
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig_h = max(3.0, 0.35 * p_net_v.shape[0] + 1.8)
    vmax = float(np.nanmax(p_net_v)) if np.isfinite(p_net_v).any() else 1.0
    fig, ax = plt.subplots(figsize=(12, fig_h), dpi=180)
    im = ax.imshow(
        p_net_v,
        aspect="auto",
        interpolation="nearest",
        cmap="magma",
        vmin=0.0,
        vmax=vmax,
    )
    ax.set_xlabel("Vertex (Long-axis ordered)")
    ax.set_ylabel("Network")
    ax.set_yticks(np.arange(len(network_names)))
    ax.set_yticklabels(network_names, fontsize=8)
    ax.set_title(out_png.stem)
    cbar = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    cbar.set_label("Probability Mass")
    fig.tight_layout()
    fig.savefig(out_png)
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description="Render long-axis P_net_v heatmaps and probability-based WTA surface maps.")
    parser.add_argument("--group-root", default=str(DEFAULT_GROUP_ROOT))
    parser.add_argument("--scene", default=str(DEFAULT_SCENE))
    args = parser.parse_args()

    group_root = Path(args.group_root).resolve()
    scene = Path(args.scene).resolve()
    if not group_root.exists():
        raise FileNotFoundError(f"Missing group root: {group_root}")
    if not scene.exists():
        raise FileNotFoundError(f"Missing scene file: {scene}")

    network_colors = load_network_colors(NETWORK_STYLE_JSON)
    manifests = sorted(group_root.glob("*/*/*/group_prior_manifest.json"))
    if not manifests:
        raise RuntimeError(f"No group_prior_manifest.json found under: {group_root}")

    summary: list[dict[str, object]] = []
    for manifest_path in manifests:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        branch = str(manifest["branch"])
        atlas = str(manifest["atlas"])
        smoothing = str(manifest["smoothing"])
        template_subject = str(manifest["template_subject"])
        template_assets = manifest["template_workbench_assets"]
        left_surface = Path(str(template_assets["left_surface"]))
        right_surface = Path(str(template_assets["right_surface"]))
        spec_path = infer_spec_path(template_subject, left_surface)

        combo_root = manifest_path.parent
        priors_dir = combo_root / "priors"
        renders_dir = combo_root / "template" / "renders"
        wb_assets_dir = combo_root / "template" / "workbench_assets"
        wb_assets_dir.mkdir(parents=True, exist_ok=True)

        hemi_label_paths: dict[str, Path] = {}
        hemi_heatmaps: dict[str, str] = {}

        for hemi in ["L", "R"]:
            prior_pickle = Path(str(manifest["hemi_priors"][hemi]["prior_pickle"]))
            payload = pickle.loads(prior_pickle.read_bytes())
            prior_matrix = np.asarray(payload["prior_matrix"], dtype=np.float64)
            cluster_network_probs = np.asarray(payload["cluster_network_probs"], dtype=np.float64)
            network_names = [str(x) for x in payload["profile_networks"]]

            p_net_v = cluster_network_probs.T @ prior_matrix
            surface_path = left_surface if hemi == "L" else right_surface
            order = long_axis_order(surface_path)
            p_net_v_longaxis = p_net_v[:, order]

            pnet_stem = f"p_net_v_{branch}_{atlas}_{smoothing}_hemi-{hemi}"
            pnet_npy = priors_dir / f"{pnet_stem}.npy"
            order_npy = priors_dir / f"{pnet_stem}_longaxis_order.npy"
            pnet_png = renders_dir / f"{pnet_stem}.png"
            pnet_npy.parent.mkdir(parents=True, exist_ok=True)

            np.save(pnet_npy, p_net_v.astype(np.float32))
            np.save(order_npy, order.astype(np.int32))
            save_heatmap_png(p_net_v=p_net_v_longaxis, network_names=network_names, out_png=pnet_png)
            hemi_heatmaps[hemi] = str(pnet_png)

            colsum = p_net_v.sum(axis=0)
            labels = np.zeros(p_net_v.shape[1], dtype=np.int32)
            valid = colsum > 0
            labels[valid] = np.argmax(p_net_v[:, valid], axis=0).astype(np.int32) + 1

            label_img = make_network_label_gifti(
                labels=labels,
                network_names=network_names,
                network_colors=network_colors,
            )
            density = infer_density_from_surface(surface_path)
            label_path = wb_assets_dir / (
                f"sub-{template_subject}_hemi-{hemi}_space-corobl_den-{density}_label-"
                f"p_net_v_wta_{branch}_{atlas}_{smoothing}.label.gii"
            )
            nib.save(label_img, str(label_path))
            hemi_label_paths[hemi] = label_path

        render_name = f"p_net_v_wta_{branch}_{atlas}_{smoothing}"
        render_title = f"P_net_v WTA {branch} {atlas} {smoothing}"
        render_paths = render_locked_grid(
            subject=template_subject,
            scene=scene,
            outdir=renders_dir,
            name=render_name,
            title=render_title,
            left_labels=hemi_label_paths["L"],
            right_labels=hemi_label_paths["R"],
            left_surface=left_surface,
            right_surface=right_surface,
            spec_path=spec_path,
        )

        summary.append(
            {
                "manifest": str(manifest_path),
                "heatmaps": hemi_heatmaps,
                "wta_label_left": str(hemi_label_paths["L"]),
                "wta_label_right": str(hemi_label_paths["R"]),
                "wta_render": render_paths,
            }
        )

    out_summary = group_root / "p_net_v_products_manifest.json"
    out_summary.write_text(json.dumps({"items": summary}, indent=2), encoding="utf-8")
    print(f"Wrote {out_summary}")
    print(f"Processed {len(summary)} group-prior combos")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
