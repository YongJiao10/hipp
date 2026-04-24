#!/usr/bin/env python3
from __future__ import annotations

import argparse
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


ATLAS_SPECS = {
    "kong2019": {"name": "Kong2019", "prior_dir": "Kong2019"},
    "hermosillo2024": {"name": "Hermosillo2024", "prior_dir": "Hermosillo2024"},
    "lynch2024": {"name": "Lynch2024", "prior_dir": "Lynch2024"},
}
LEFT_VERTEX_COUNT = 32492
TOTAL_VERTEX_COUNT = 64984
TEMPLATE_SUBJECT = "100610"
FOUR_PANEL_OUT_NAME = "group_priors_canonical_merged_4panel_ordered.png"
EXPLICIT_LABEL_PANEL = "Schaefer400_Kong2022_Deterministic_canonical_merged_shape.png"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render group-level canonical-merged cortical labels for Kong2019/Hermosillo2024/Lynch2024"
    )
    parser.add_argument(
        "--atlases",
        nargs="+",
        default=["kong2019", "hermosillo2024", "lynch2024"],
        choices=sorted(ATLAS_SPECS),
    )
    parser.add_argument(
        "--merge-config",
        default=str(REPO_ROOT / "config" / "cross_atlas_network_merge.json"),
    )
    parser.add_argument(
        "--source-scene",
        default=str(REPO_ROOT / "config" / "manual_wb_scenes" / "cortex_manual.scene"),
    )
    parser.add_argument(
        "--out-root",
        default=str(REPO_ROOT / "outputs_migration" / "group_priors"),
    )
    parser.add_argument(
        "--present-root",
        default=str(REPO_ROOT / "present"),
    )
    return parser.parse_args()


def run(cmd: list[str]) -> None:
    proc = subprocess.run(cmd, text=True, capture_output=True)
    if proc.returncode != 0:
        raise RuntimeError(f"Command failed ({proc.returncode}): {' '.join(cmd)}\\nSTDOUT:\\n{proc.stdout}\\nSTDERR:\\n{proc.stderr}")


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


def load_merge_config(path: Path) -> tuple[list[str], set[str], dict[str, dict[str, str]], dict[str, tuple[int, int, int, int]]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    canonical_order = [str(item) for item in payload["canonical_network_order"]]
    exclude_labels = {str(item) for item in payload.get("exclude_labels", [])}
    atlas_mapping = {
        str(atlas_slug): {str(key): str(value) for key, value in atlas_spec["mapping"].items()}
        for atlas_slug, atlas_spec in payload["atlases"].items()
    }
    shared_colors = {
        str(name): tuple(int(v) for v in rgba)
        for name, rgba in payload.get("shared_colors_rgba", {}).items()
    }
    return canonical_order, exclude_labels, atlas_mapping, shared_colors


def merge_priors_to_canonical(
    priors: np.ndarray,
    original_names: list[str],
    canonical_order: list[str],
    exclude_labels: set[str],
    mapping: dict[str, str],
) -> tuple[np.ndarray, list[str]]:
    missing = sorted(set(original_names) - set(mapping))
    if missing:
        raise KeyError(f"Missing cross-atlas mapping for labels: {missing}")

    canonical_sum = {name: np.zeros(priors.shape[1], dtype=np.float64) for name in canonical_order}
    for idx, original_name in enumerate(original_names):
        target = mapping[original_name]
        if target in exclude_labels:
            continue
        if target not in canonical_sum:
            raise KeyError(f"Canonical label '{target}' missing from canonical_network_order")
        canonical_sum[target] += priors[idx]

    used_canonical = [name for name in canonical_order if name not in exclude_labels and float(canonical_sum[name].sum()) > 0.0]
    if not used_canonical:
        raise RuntimeError("No canonical networks remained after merge")
    merged = np.asarray([canonical_sum[name] for name in used_canonical], dtype=np.float64)
    return merged, used_canonical


def save_wta_assets(
    merged_priors: np.ndarray,
    canonical_names: list[str],
    shared_colors: dict[str, tuple[int, int, int, int]],
    outdir: Path,
    atlas_name: str,
) -> tuple[dict[str, Path], list[dict[str, object]], dict[str, float]]:
    outdir.mkdir(parents=True, exist_ok=True)

    if merged_priors.shape[1] != TOTAL_VERTEX_COUNT:
        raise ValueError(f"Expected fsLR 32k total vertex count {TOTAL_VERTEX_COUNT}, got {merged_priors.shape[1]}")

    label_rows: list[dict[str, object]] = []
    for idx, canonical_name in enumerate(canonical_names, start=1):
        if canonical_name not in shared_colors:
            raise KeyError(f"Missing shared color for canonical network: {canonical_name}")
        label_rows.append({"key": idx, "name": canonical_name, "rgba": shared_colors[canonical_name]})

    prob_sum = merged_priors.sum(axis=0)
    labels = np.zeros(TOTAL_VERTEX_COUNT, dtype=np.int32)
    nz = prob_sum > 0
    labels[nz] = merged_priors[:, nz].argmax(axis=0).astype(np.int32) + 1

    counts = np.bincount(labels[nz], minlength=len(canonical_names) + 1).astype(np.float64)
    occupancy_fraction = {
        canonical_names[idx - 1]: float(counts[idx] / counts[1:].sum()) if counts[1:].sum() > 0 else 0.0
        for idx in range(1, len(canonical_names) + 1)
    }

    left_labels = labels[:LEFT_VERTEX_COUNT]
    right_labels = labels[LEFT_VERTEX_COUNT:]

    stem = f"{atlas_name}_group_prior_canonical_merged_wta"
    left_path = outdir / f"{stem}.L.label.gii"
    right_path = outdir / f"{stem}.R.label.gii"
    dlabel_path = outdir / f"{stem}.dlabel.nii"

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

    return {"left_label": left_path, "right_label": right_path, "dlabel": dlabel_path}, label_rows, occupancy_fraction


def render_shape_figure(
    *,
    source_scene: Path,
    atlas_name: str,
    assets: dict[str, Path],
    label_rows: list[dict[str, object]],
    outdir: Path,
    nonzero_vertices: int,
    n_canonical: int,
) -> Path:
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
    atlas_slug = atlas_name.lower()

    for index, view in enumerate(cortex.VIEW_SPECS):
        out_scene = (
            outdir / f"wb_{atlas_slug}_group_prior_canonical_merged_{view['name']}.scene"
            if index == 0
            else views_dir / f"wb_{atlas_slug}_group_prior_canonical_merged_{view['name']}.scene"
        )
        out_png = views_dir / f"wb_{atlas_slug}_group_prior_canonical_merged_{view['name']}.png"
        cortex.render_view(
            source_scene=source_scene,
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

    out_png = outdir / f"{atlas_name}_group_priors_canonical_merged_shape.png"
    cortex.compose_multiview(
        subject="group",
        title=f"{atlas_name} Canonical-Merged Labels",
        subtitle=f"{n_canonical} merged canonical networks, nonzero cortex vertices: {nonzero_vertices}",
        legend_items=label_rows,
        view_pngs=rendered_views,
        out_png=out_png,
        font_scale=2.0,
    )
    return out_png


def render_one_atlas(
    *,
    atlas_slug: str,
    merge_config_path: Path,
    source_scene: Path,
    out_root: Path,
    present_root: Path,
) -> dict[str, object]:
    spec = ATLAS_SPECS[atlas_slug]
    atlas_name = spec["name"]
    prior_dir = spec["prior_dir"]

    canonical_order, exclude_labels, atlas_mapping, shared_colors = load_merge_config(merge_config_path)
    if atlas_slug not in atlas_mapping:
        raise KeyError(f"Atlas '{atlas_slug}' not found in merge config")

    priors_root = REPO_ROOT / "external" / "FASTANS" / "resources" / "PFM" / "priors" / prior_dir
    pickle_path = priors_root / f"{atlas_name}_priors.pickle"
    label_path = priors_root / f"{atlas_name}_LabelList.txt"

    priors = np.asarray(pickle.loads(pickle_path.read_bytes()), dtype=np.float64)
    label_rows_all = load_label_rows(label_path)
    original_rows = label_rows_all[: priors.shape[0]]
    original_names = [str(row["name"]) for row in original_rows]

    merged_priors, canonical_names = merge_priors_to_canonical(
        priors,
        original_names,
        canonical_order,
        exclude_labels,
        atlas_mapping[atlas_slug],
    )

    outdir = out_root / atlas_slug / "canonical_merged"
    assets, canonical_label_rows, occupancy_fraction = save_wta_assets(
        merged_priors,
        canonical_names,
        shared_colors,
        outdir / "workbench_assets",
        atlas_name,
    )

    nonzero_vertices = int((merged_priors.sum(axis=0) > 0).sum())
    zero_vertices = int((merged_priors.sum(axis=0) == 0).sum())
    shape_png = render_shape_figure(
        source_scene=source_scene,
        atlas_name=atlas_name,
        assets=assets,
        label_rows=canonical_label_rows,
        outdir=outdir,
        nonzero_vertices=nonzero_vertices,
        n_canonical=len(canonical_names),
    )

    summary = {
        "atlas_slug": atlas_slug,
        "atlas_name": atlas_name,
        "merge_config": str(merge_config_path.resolve()),
        "matrix_shape_original": [int(priors.shape[0]), int(priors.shape[1])],
        "matrix_shape_canonical_merged": [int(merged_priors.shape[0]), int(merged_priors.shape[1])],
        "n_original_networks": int(priors.shape[0]),
        "n_canonical_networks_used": int(len(canonical_names)),
        "canonical_networks_used": canonical_names,
        "n_nonzero_vertices": nonzero_vertices,
        "n_zero_vertices": zero_vertices,
        "network_occupancy_fraction": occupancy_fraction,
        "shape_png": str(shape_png.resolve()),
        "dlabel": str(assets["dlabel"].resolve()),
        "left_label": str(assets["left_label"].resolve()),
        "right_label": str(assets["right_label"].resolve()),
    }

    summary_path = outdir / f"{atlas_name}_group_priors_canonical_merged_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    present_root.mkdir(parents=True, exist_ok=True)
    present_png = present_root / f"{atlas_name}_group_priors_canonical_merged_shape.png"
    present_png.write_bytes(shape_png.read_bytes())
    summary["summary_json"] = str(summary_path.resolve())
    summary["present_shape_png"] = str(present_png.resolve())
    return summary


def compose_present_4panel(present_root: Path, run_summary: dict[str, object]) -> Path:
    atlas_order = ["kong2019", "hermosillo2024", "lynch2024"]
    atlas_label = {
        "kong2019": "Kong2019 priors + WTA",
        "hermosillo2024": "Hermosillo2024 priors + WTA",
        "lynch2024": "Lynch2024 priors + WTA",
    }
    panel_paths: list[Path] = [present_root / EXPLICIT_LABEL_PANEL]
    panel_titles: list[str] = ["Schaefer400 deterministic explicit labels"]

    atlas_payload = run_summary.get("atlases", {})
    if not isinstance(atlas_payload, dict):
        raise RuntimeError("run_summary['atlases'] must be a dictionary")
    for slug in atlas_order:
        atlas_info = atlas_payload.get(slug)
        if not isinstance(atlas_info, dict):
            raise RuntimeError(f"Missing atlas summary for {slug}")
        png_path = Path(str(atlas_info["present_shape_png"]))
        panel_paths.append(png_path)
        panel_titles.append(atlas_label[slug])

    for path in panel_paths:
        if not path.exists():
            raise FileNotFoundError(f"Missing 4-panel input image: {path}")

    fig, axes = plt.subplots(4, 1, figsize=(13.0, 24.0), constrained_layout=True)
    for idx, ax in enumerate(axes):
        img = plt.imread(panel_paths[idx])
        ax.imshow(img)
        ax.set_title(panel_titles[idx], fontsize=20, pad=10)
        ax.axis("off")

    out_png = present_root / FOUR_PANEL_OUT_NAME
    fig.savefig(out_png, dpi=220, facecolor="white")
    plt.close(fig)
    return out_png


def main() -> int:
    args = parse_args()
    merge_config_path = Path(args.merge_config).resolve()
    source_scene = Path(args.source_scene).resolve()
    out_root = Path(args.out_root).resolve()
    present_root = Path(args.present_root).resolve()

    run_summary: dict[str, object] = {
        "merge_config": str(merge_config_path),
        "source_scene": str(source_scene),
        "atlases": {},
    }

    for atlas_slug in args.atlases:
        run_summary["atlases"][atlas_slug] = render_one_atlas(
            atlas_slug=atlas_slug,
            merge_config_path=merge_config_path,
            source_scene=source_scene,
            out_root=out_root,
            present_root=present_root,
        )

    four_panel_png = compose_present_4panel(present_root, run_summary)
    run_summary["present_4panel_png"] = str(four_panel_png.resolve())

    print(json.dumps(run_summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
