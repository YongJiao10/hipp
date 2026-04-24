#!/usr/bin/env python3
from __future__ import annotations

import json
import pickle
import subprocess
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
from nibabel.gifti import GiftiDataArray, GiftiImage, GiftiLabel, GiftiLabelTable

REPO_ROOT = Path(__file__).resolve().parents[1]
CORTEX_DIR = REPO_ROOT / "scripts" / "cortex"
if str(CORTEX_DIR) not in sys.path:
    sys.path.insert(0, str(CORTEX_DIR))

import run_cortex_pfm_subject as cortex


PRIOR_ROOT = REPO_ROOT / "external" / "FASTANS" / "resources" / "PFM" / "priors" / "Lynch2024"
PICKLE_PATH = PRIOR_ROOT / "Lynch2024_priors.pickle"
LABEL_PATH = PRIOR_ROOT / "Lynch2024_LabelList.txt"
OUT_ROOT = REPO_ROOT / "outputs_migration" / "group_priors" / "lynch2024"
PRESENT_ROOT = REPO_ROOT / "present"
TEMPLATE_SUBJECT = "100610"
SOURCE_SCENE = REPO_ROOT / "config" / "manual_wb_scenes" / "cortex_manual.scene"


def load_label_rows(path: Path) -> list[dict[str, object]]:
    lines = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    rows: list[dict[str, object]] = []
    for idx in range(0, len(lines), 2):
        name = lines[idx]
        parts = lines[idx + 1].split()
        rows.append(
            {
                "key": int(parts[0]),
                "name": name,
                "rgba": tuple(int(v) for v in parts[1:5]),
            }
        )
    return rows


def make_label_gifti(labels: np.ndarray, label_rows: list[dict[str, object]]) -> GiftiImage:
    table = GiftiLabelTable()
    unknown = GiftiLabel(key=0, red=0.0, green=0.0, blue=0.0, alpha=0.0)
    unknown.label = "???"
    table.labels.append(unknown)
    for row in label_rows:
        rgba = row["rgba"]
        item = GiftiLabel(
            key=int(row["key"]),
            red=float(rgba[0]) / 255.0,
            green=float(rgba[1]) / 255.0,
            blue=float(rgba[2]) / 255.0,
            alpha=float(rgba[3]) / 255.0,
        )
        item.label = str(row["name"])
        table.labels.append(item)
    arr = GiftiDataArray(data=labels.astype(np.int32), intent="NIFTI_INTENT_LABEL", datatype="NIFTI_TYPE_INT32")
    return GiftiImage(darrays=[arr], labeltable=table)


def run(cmd: list[str]) -> None:
    proc = subprocess.run(cmd, text=True, capture_output=True)
    if proc.returncode != 0:
        raise RuntimeError(f"Command failed ({proc.returncode}): {' '.join(cmd)}\nSTDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}")


def save_wta_assets(priors: np.ndarray, label_rows: list[dict[str, object]], outdir: Path) -> dict[str, Path]:
    outdir.mkdir(parents=True, exist_ok=True)
    n_vertices_total = priors.shape[1]
    if n_vertices_total != 64984:
        raise ValueError(f"Expected fsLR 32k total vertex count 64984, got {n_vertices_total}")
    left_count = 32492
    prob_sum = priors.sum(axis=0)
    labels = np.zeros(n_vertices_total, dtype=np.int32)
    nz = prob_sum > 0
    labels[nz] = priors[:, nz].argmax(axis=0).astype(np.int32) + 1
    left_labels = labels[:left_count]
    right_labels = labels[left_count:]

    left_path = outdir / "Lynch2024_group_prior_wta.L.label.gii"
    right_path = outdir / "Lynch2024_group_prior_wta.R.label.gii"
    dlabel_path = outdir / "Lynch2024_group_prior_wta.dlabel.nii"
    nib.save(make_label_gifti(left_labels, label_rows), str(left_path))
    nib.save(make_label_gifti(right_labels, label_rows), str(right_path))
    run(
        [
            cortex.WB_COMMAND,
            "-cifti-create-label",
            str(dlabel_path),
            "-left-label",
            str(left_path),
            "-right-label",
            str(right_path),
        ]
    )
    return {"left_label": left_path, "right_label": right_path, "dlabel": dlabel_path}


def render_shape_figure(label_rows: list[dict[str, object]], assets: dict[str, Path], outdir: Path, nonzero_vertices: int) -> Path:
    subject = TEMPLATE_SUBJECT
    anat_dir = REPO_ROOT / "data" / "hippunfold_input" / f"sub-{subject}" / "anat"
    left_inflated = anat_dir / f"sub-{subject}_hemi-L_space-fsLR_den-32k_desc-MSMAll_inflated.surf.gii"
    right_inflated = anat_dir / f"sub-{subject}_hemi-R_space-fsLR_den-32k_desc-MSMAll_inflated.surf.gii"
    sulc_dscalar = anat_dir / f"sub-{subject}_space-fsLR_den-32k_desc-MSMAll_sulc.dscalar.nii"
    assets_dir = REPO_ROOT / "outputs_migration" / "cortex_pfm" / f"sub-{subject}" / "assets"
    left_sulc = assets_dir / f"sub-{subject}_hemi-L_space-fsLR_den-32k_desc-MSMAll_sulc.func.gii"
    right_sulc = assets_dir / f"sub-{subject}_hemi-R_space-fsLR_den-32k_desc-MSMAll_sulc.func.gii"

    views_dir = outdir / "views"
    views_dir.mkdir(parents=True, exist_ok=True)
    rendered_views: list[tuple[str, Path]] = []
    for index, view in enumerate(cortex.VIEW_SPECS):
        out_scene = outdir / f"wb_lynch2024_group_prior_{view['name']}.scene" if index == 0 else views_dir / f"wb_lynch2024_group_prior_{view['name']}.scene"
        out_png = views_dir / f"wb_lynch2024_group_prior_{view['name']}.png"
        cortex.render_view(
            source_scene=SOURCE_SCENE,
            left_inflated=left_inflated,
            right_inflated=right_inflated,
            left_sulc=left_sulc,
            right_sulc=right_sulc,
            sulc_dscalar=sulc_dscalar,
            dlabel=assets["dlabel"],
            left_label=assets["left_label"],
            right_label=assets["right_label"],
            out_scene=out_scene,
            out_png=out_png,
            rotation_axis=view["axis"],
            rotation_deg=view["deg"],
        )
        rendered_views.append((view["name"], out_png))

    out_png = outdir / "Lynch2024_group_priors_shape.png"
    cortex.compose_multiview(
        subject="group",
        title="Lynch2024 Group Prior WTA Shape",
        subtitle=f"20 prior channels, nonzero cortex vertices: {nonzero_vertices}",
        legend_items=label_rows,
        view_pngs=rendered_views,
        out_png=out_png,
    )
    return out_png


def render_probability_figure(priors: np.ndarray, label_rows: list[dict[str, object]], outdir: Path) -> Path:
    mask = priors.sum(axis=0) > 0
    priors_nz = priors[:, mask]
    network_names = [str(row["name"]) for row in label_rows]
    network_colors = [tuple(v / 255.0 for v in row["rgba"][:3]) for row in label_rows]

    winner = priors_nz.argmax(axis=0)
    max_prob = priors_nz.max(axis=0)
    entropy = -(priors_nz * np.log(np.clip(priors_nz, 1e-12, None))).sum(axis=0) / np.log(priors_nz.shape[0])
    occupancy = np.bincount(winner, minlength=priors_nz.shape[0]).astype(float)
    occupancy_frac = occupancy / occupancy.sum()
    winner_strengths = [priors_nz[idx, winner == idx] for idx in range(priors_nz.shape[0])]

    order = np.argsort(occupancy_frac)[::-1]
    ordered_names = [network_names[i] for i in order]
    ordered_colors = [network_colors[i] for i in order]
    ordered_occ = occupancy_frac[order]
    ordered_strengths = [winner_strengths[i] for i in order]

    plt.rcParams.update({"font.size": 12})
    fig = plt.figure(figsize=(18, 10), constrained_layout=True)
    gs = fig.add_gridspec(2, 2, height_ratios=[1.0, 1.25])
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[1, :])

    ax1.hist(max_prob, bins=np.linspace(0.0, 1.0, 26), color="#3d6aa2", edgecolor="white")
    ax1.axvline(float(np.median(max_prob)), color="black", linestyle="--", linewidth=1.5, label=f"median={np.median(max_prob):.2f}")
    ax1.set_title("Max Prior Probability Across Cortex")
    ax1.set_xlabel("Winner probability")
    ax1.set_ylabel("Vertex count")
    ax1.legend(frameon=False, fontsize=11)

    ax2.hist(entropy, bins=np.linspace(0.0, 1.0, 26), color="#c9793a", edgecolor="white")
    ax2.axvline(float(np.median(entropy)), color="black", linestyle="--", linewidth=1.5, label=f"median={np.median(entropy):.2f}")
    ax2.set_title("Normalized Entropy Across Cortex")
    ax2.set_xlabel("Entropy (0=sharp, 1=diffuse)")
    ax2.set_ylabel("Vertex count")
    ax2.legend(frameon=False, fontsize=11)

    positions = np.arange(len(ordered_names))
    box = ax3.boxplot(
        ordered_strengths,
        vert=False,
        positions=positions,
        patch_artist=True,
        widths=0.7,
        showfliers=False,
        medianprops={"color": "black", "linewidth": 1.5},
        whiskerprops={"color": "#444444", "linewidth": 1.0},
        capprops={"color": "#444444", "linewidth": 1.0},
    )
    for patch, color in zip(box["boxes"], ordered_colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.8)
        patch.set_edgecolor("#222222")

    ax3.set_yticks(positions)
    ax3.set_yticklabels([f"{name}  ({frac*100:.1f}%)" for name, frac in zip(ordered_names, ordered_occ)])
    ax3.set_xlabel("Winner-network prior probability on its assigned vertices")
    ax3.set_title("Per-Network Winner Probability Distribution (ordered by WTA occupancy)")
    ax3.grid(axis="x", alpha=0.25)

    fig.suptitle("Lynch2024 Group Priors: Probability Distribution Summary", fontsize=18)
    out_png = outdir / "Lynch2024_group_priors_probability_distribution.png"
    fig.savefig(out_png, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return out_png


def main() -> int:
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    PRESENT_ROOT.mkdir(parents=True, exist_ok=True)

    priors = np.asarray(pickle.loads(PICKLE_PATH.read_bytes()), dtype=np.float64)
    label_rows_all = load_label_rows(LABEL_PATH)
    label_rows = label_rows_all[: priors.shape[0]]
    omitted_labels = [row["name"] for row in label_rows_all[priors.shape[0] :]]
    nonzero_vertices = int((priors.sum(axis=0) > 0).sum())
    zero_vertices = int((priors.sum(axis=0) == 0).sum())

    assets = save_wta_assets(priors, label_rows, OUT_ROOT / "workbench_assets")
    shape_png = render_shape_figure(label_rows, assets, OUT_ROOT, nonzero_vertices)
    prob_png = render_probability_figure(priors, label_rows, OUT_ROOT)

    summary = {
        "matrix_shape": list(priors.shape),
        "n_prior_networks": int(priors.shape[0]),
        "n_total_vertices": int(priors.shape[1]),
        "n_nonzero_vertices": nonzero_vertices,
        "n_zero_vertices": zero_vertices,
        "max_probability_mean": float(priors.max(axis=0).mean()),
        "max_probability_median_nonzero": float(np.median(priors[:, priors.sum(axis=0) > 0].max(axis=0))),
        "omitted_label_rows": omitted_labels,
        "shape_png": str(shape_png),
        "probability_png": str(prob_png),
    }
    (OUT_ROOT / "Lynch2024_group_priors_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    for src, name in [
        (shape_png, "Lynch2024_group_priors_shape.png"),
        (prob_png, "Lynch2024_group_priors_probability_distribution.png"),
    ]:
        dst = PRESENT_ROOT / name
        dst.write_bytes(src.read_bytes())

    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
