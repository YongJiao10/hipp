#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
CORE_WORKFLOW_DIR = REPO_ROOT / "scripts" / "core_workflow"
DEFAULT_SCENE = REPO_ROOT / "config" / "manual_wb_scenes" / "cortex_manual.scene"

# Load 06_09_plot_cortex_kong2022_mshbm.py using importlib
spec = importlib.util.spec_from_file_location(
    "plot_cortex_mshbm",
    CORE_WORKFLOW_DIR / "06_09_plot_cortex_kong2022_mshbm.py"
)
plot_cortex_mshbm = importlib.util.module_from_spec(spec)
spec.loader.exec_module(plot_cortex_mshbm)

def main() -> int:
    parser = argparse.ArgumentParser(description="Render Iowa7T subject cortical brain maps.")
    parser.add_argument("--subjects", nargs="+", default=["06112026", "06122026"])
    parser.add_argument("--variants", nargs="+", default=["0p", "32p", "36p"])
    parser.add_argument("--branches", nargs="+", default=["multiband", "multi_echo"])
    parser.add_argument("--renderer", default="OSMesa")
    args = parser.parse_args()

    base_dir = Path("/home/yojiao/LSS/jianglab/3_Data_Working/fmriprep_processed_Iowa7T")
    render_scene_script = CORE_WORKFLOW_DIR / "lib" / "render_cortex_pfm_scene.py"

    bundle = plot_cortex_mshbm.load_kong2022_bundle()
    network_colors = plot_cortex_mshbm.load_network_colors()

    for subject in args.subjects:
        for var in args.variants:
            var_upper = var.upper()
            for branch in args.branches:
                # Resolve paths
                var_dir = base_dir / f"DataOrg_PFM_{var_upper}"
                sub_dir = var_dir / "hipp_outputs" / "subjects" / f"sub-{subject}"
                branch_dir = sub_dir / branch
                
                lh_txt = branch_dir / "cortex" / "areal_mshbm" / "lh_labels.txt"
                rh_txt = branch_dir / "cortex" / "areal_mshbm" / "rh_labels.txt"
                
                if not lh_txt.exists() or not rh_txt.exists():
                    print(f"Skipping sub-{subject} | {var_upper} | {branch} (labels missing)")
                    continue
                
                print(f"Rendering sub-{subject} | {var_upper} | {branch} ...")
                
                # Use standard template subject 100610 surfaces to avoid tilted/crooked orientations
                template_anat_dir = REPO_ROOT / "data" / "hippunfold_input" / "sub-100610" / "anat"
                lh_surf = template_anat_dir / "sub-100610_hemi-L_space-fsLR_den-32k_desc-MSMAll_inflated.surf.gii"
                rh_surf = template_anat_dir / "sub-100610_hemi-R_space-fsLR_den-32k_desc-MSMAll_inflated.surf.gii"
                sulc_dscalar = template_anat_dir / "sub-100610_space-fsLR_den-32k_desc-MSMAll_sulc.dscalar.nii"

                if not lh_surf.exists() or not rh_surf.exists() or not sulc_dscalar.exists():
                    print(f"Error: Standard template surfaces/sulc missing under {template_anat_dir}")
                    continue
                
                # Setup output folders inside branch_dir
                out_dir = branch_dir
                figures_dir = out_dir / "figures"
                scenes_dir = out_dir / "scenes"
                figures_dir.mkdir(parents=True, exist_ok=True)
                scenes_dir.mkdir(parents=True, exist_ok=True)
                
                # 1. Generate Gifti labels and Cifti dlabels
                left_label_gii, right_label_gii = plot_cortex_mshbm.generate_gifti_labels(
                    subject=subject,
                    lh_txt=lh_txt,
                    rh_txt=rh_txt,
                    out_dir=out_dir,
                    bundle=bundle,
                    network_colors=network_colors,
                )
                
                dlabel_path = plot_cortex_mshbm.create_cifti_dlabel(
                    subject=subject,
                    left_label_gii=left_label_gii,
                    right_label_gii=right_label_gii,
                    out_dir=out_dir,
                )
                
                # 2. Separate sulc
                left_sulc_gii, right_sulc_gii = plot_cortex_mshbm.separate_sulc(
                    subject=subject,
                    sulc_dscalar=sulc_dscalar,
                    out_dir=out_dir,
                )
                
                # 3. Render individual views
                for spec in plot_cortex_mshbm.VIEW_SPECS:
                    if spec["name"] != "lateral":
                        continue

                    out_scene = scenes_dir / f"sub-{subject}_atlas-kong2022_areal_mshbm_{spec['name']}.scene"
                    out_png = figures_dir / f"sub-{subject}_atlas-kong2022_areal_mshbm_{spec['name']}.png"

                    cmd = [
                        sys.executable,
                        str(render_scene_script),
                        "--source-scene",
                        str(DEFAULT_SCENE),
                        "--left-surface",
                        str(lh_surf),
                        "--right-surface",
                        str(rh_surf),
                        "--left-label",
                        str(left_label_gii),
                        "--right-label",
                        str(right_label_gii),
                        "--left-sulc",
                        str(left_sulc_gii),
                        "--right-sulc",
                        str(right_sulc_gii),
                        "--sulc-dscalar",
                        str(sulc_dscalar),
                        "--dlabel",
                        str(dlabel_path),
                        "--out-scene",
                        str(out_scene),
                        "--out-png",
                        str(out_png),
                        "--scene-index",
                        "1",
                        "--renderer",
                        args.renderer,
                        "--width",
                        "1200",
                        "--height",
                        "900",
                    ]

                    if spec["axis"] is not None:
                        cmd.extend(["--rotation-axis", str(spec["axis"]), "--rotation-deg", str(spec["deg"])])

                    subprocess.run(cmd, check=True)

                    # Add side legend to the lateral view and save it back
                    img_with_legend = plot_cortex_mshbm.add_legend_to_view(out_png, subject, "lateral", network_colors)
                    img_with_legend.save(out_png)
                    print(f"Successfully generated: {out_png}")

                # Clean up intermediate directories to only keep the final figures
                import shutil
                shutil.rmtree(scenes_dir, ignore_errors=True)
                shutil.rmtree(out_dir / "assets", ignore_errors=True)
                
    return 0

if __name__ == "__main__":
    sys.exit(main())
