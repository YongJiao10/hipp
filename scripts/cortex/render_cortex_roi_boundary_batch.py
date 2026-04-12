#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from PIL import Image

import run_cortex_pfm_subject as cortex
from derive_cortex_roi_components import derive_roi_components


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Derive ROI components, summarize counts, and render cortex ROI-boundary overlays."
    )
    parser.add_argument("--subjects", nargs="+", required=True)
    parser.add_argument("--methods", nargs="+", default=["Lynch2024", "Hermosillo2024"])
    parser.add_argument("--scene", default=str(cortex.REPO_ROOT / "config" / "manual_wb_scenes" / "cortex_manual.scene"))
    parser.add_argument("--data-root", default=str(cortex.REPO_ROOT / "data" / "hippunfold_input"))
    parser.add_argument("--out-root", default=str(cortex.REPO_ROOT / "outputs_migration" / "cortex_pfm"))
    parser.add_argument("--roi-min-area-mm2", type=float, default=25.0)
    return parser.parse_args()


def boundary_mask_from_render(boundary_png: Path) -> "np.ndarray":
    import numpy as np

    image = np.asarray(Image.open(boundary_png).convert("RGB"), dtype=np.uint8)
    return (image[:, :, 0] >= 200) & (image[:, :, 1] <= 80) & (image[:, :, 2] >= 200)


def skeletonize_mask(mask: "np.ndarray") -> "np.ndarray":
    import numpy as np
    from scipy import ndimage as ndi

    structure = np.ones((3, 3), dtype=bool)
    mask = mask.astype(bool, copy=True)
    skeleton = np.zeros_like(mask, dtype=bool)
    while mask.any():
        eroded = ndi.binary_erosion(mask, structure=structure)
        opened = ndi.binary_dilation(eroded, structure=structure)
        skeleton |= mask & ~opened
        mask = eroded
    return skeleton


def overlay_boundaries(base_png: Path, boundary_png: Path, out_png: Path) -> None:
    import numpy as np
    from scipy import ndimage as ndi

    base = np.asarray(Image.open(base_png).convert("RGB"), dtype=np.uint8)
    boundary = skeletonize_mask(boundary_mask_from_render(boundary_png))
    # Thicken the 1-pixel skeleton slightly so parcel borders remain legible in the montage.
    boundary = ndi.binary_dilation(boundary, structure=np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=bool), iterations=1)
    composite = base.copy()
    composite[boundary] = np.array([20, 20, 20], dtype=np.uint8)
    Image.fromarray(composite).save(out_png)


def write_summary(summary_rows: list[dict[str, object]], out_root: Path) -> None:
    csv_path = out_root / "roi_component_summary.csv"
    md_path = out_root / "roi_component_summary.md"
    csv_fields = [
        "subject",
        "method",
        "row_type",
        "hemisphere",
        "network",
        "raw_component_count",
        "kept_roi_count",
    ]
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=csv_fields)
        writer.writeheader()
        for row in summary_rows:
            writer.writerow(row)

    totals = [row for row in summary_rows if row["row_type"] == "summary"]
    lines = [
        "# ROI Component Summary",
        "",
        "```text",
        f"{'Subject':<8}  {'Method':<14}  {'Raw':>5}  {'Kept':>5}",
    ]
    for row in totals:
        lines.append(
            f"{row['subject']:<8}  {row['method']:<14}  {int(row['raw_component_count']):>5}  {int(row['kept_roi_count']):>5}"
        )
    lines.extend(["```", "", f"Detailed long-form counts: `{csv_path}`"])
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    scene = Path(args.scene).resolve()
    data_root = Path(args.data_root).resolve()
    out_root = Path(args.out_root).resolve()

    summary_rows: list[dict[str, object]] = []
    for subject in args.subjects:
        rendered_pngs: dict[str, Path] = {}
        for method in args.methods:
            stats = derive_roi_components(
                subject=subject,
                method=method,
                data_root=data_root,
                out_root=out_root,
                roi_min_area_mm2=args.roi_min_area_mm2,
            )
            method_name = stats["method"]
            slug = stats["method_slug"]
            method_dir = out_root / f"sub-{subject}" / slug
            roi_dir = method_dir / "roi_components"
            roi_views_dir = roi_dir / "views"
            roi_views_dir.mkdir(parents=True, exist_ok=True)

            left_inflated = data_root / f"sub-{subject}" / "anat" / f"sub-{subject}_hemi-L_space-fsLR_den-32k_desc-MSMAll_inflated.surf.gii"
            right_inflated = data_root / f"sub-{subject}" / "anat" / f"sub-{subject}_hemi-R_space-fsLR_den-32k_desc-MSMAll_inflated.surf.gii"
            left_sulc = out_root / f"sub-{subject}" / "assets" / f"sub-{subject}_hemi-L_space-fsLR_den-32k_desc-MSMAll_sulc.func.gii"
            right_sulc = out_root / f"sub-{subject}" / "assets" / f"sub-{subject}_hemi-R_space-fsLR_den-32k_desc-MSMAll_sulc.func.gii"
            sulc_dscalar = data_root / f"sub-{subject}" / "anat" / f"sub-{subject}_space-fsLR_den-32k_desc-MSMAll_sulc.dscalar.nii"
            left_boundary_label = Path(stats["hemispheres"]["L"]["boundary_label"])
            right_boundary_label = Path(stats["hemispheres"]["R"]["boundary_label"])

            overlay_views: list[tuple[str, Path]] = []
            for index, view in enumerate(cortex.VIEW_SPECS):
                roi_scene = roi_views_dir / f"wb_{slug}_{view['name']}_roi_boundary.scene"
                roi_png = roi_views_dir / f"wb_{slug}_{view['name']}_roi_boundary.png"
                overlay_png = roi_views_dir / f"wb_{slug}_{view['name']}_roi_boundaries.png"
                cortex.render_view(
                    source_scene=scene,
                    left_inflated=left_inflated,
                    right_inflated=right_inflated,
                    left_sulc=left_sulc,
                    right_sulc=right_sulc,
                    sulc_dscalar=sulc_dscalar,
                    dlabel=method_dir / f"PFM_{method_name}priors.dlabel.nii",
                    left_label=left_boundary_label,
                    right_label=right_boundary_label,
                    out_scene=roi_scene,
                    out_png=roi_png,
                    rotation_axis=view["axis"],
                    rotation_deg=view["deg"],
                )
                base_png = method_dir / "views" / f"wb_{slug}_{view['name']}.png"
                overlay_boundaries(base_png, roi_png, overlay_png)
                overlay_views.append((view["name"], overlay_png))

            legend_items = cortex.parse_label_legend(
                cortex.DEFAULT_FASTANS_ROOT / "resources" / "PFM" / "priors" / method_name / cortex.METHODS[method_name]["labels_file"]
            )
            montage_png = method_dir / f"wb_{slug}_inflated_roi_boundaries.png"
            cortex.compose_multiview(
                subject=subject,
                title=cortex.METHODS[method_name]["display_name"],
                legend_items=legend_items,
                view_pngs=overlay_views,
                out_png=montage_png,
                subtitle=f"Kept ROI parcels: {int(stats['kept_roi_count'])}",
            )
            rendered_pngs[method_name] = montage_png

            summary_rows.append(
                {
                    "subject": subject,
                    "method": method_name,
                    "row_type": "summary",
                    "hemisphere": "",
                    "network": "",
                    "raw_component_count": int(stats["raw_component_count"]),
                    "kept_roi_count": int(stats["kept_roi_count"]),
                }
            )
            for hemisphere, hemi_info in stats["hemispheres"].items():
                for network, net_info in hemi_info["networks"].items():
                    summary_rows.append(
                        {
                            "subject": subject,
                            "method": method_name,
                            "row_type": "network",
                            "hemisphere": hemisphere,
                            "network": network,
                            "raw_component_count": int(net_info["raw_component_count"]),
                            "kept_roi_count": int(net_info["kept_roi_count"]),
                        }
                    )

            print(
                json.dumps(
                    {
                        "subject": subject,
                        "method": method_name,
                        "raw_component_count": stats["raw_component_count"],
                        "kept_roi_count": stats["kept_roi_count"],
                        "overlay_montage": str(montage_png),
                    }
                )
            )

        if "Lynch2024" in rendered_pngs and "Hermosillo2024" in rendered_pngs:
            cortex.build_comparison(
                subject,
                rendered_pngs["Lynch2024"],
                rendered_pngs["Hermosillo2024"],
                out_root / f"sub-{subject}" / "comparison" / "wb_lynch2024_vs_hermosillo2024_roi_boundaries.png",
            )

    write_summary(summary_rows, out_root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
