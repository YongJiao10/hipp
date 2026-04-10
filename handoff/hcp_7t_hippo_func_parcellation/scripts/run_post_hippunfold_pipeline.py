#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from time import perf_counter

REPO_ROOT = Path(__file__).resolve().parents[1]
COMMON_DIR = REPO_ROOT / "scripts" / "common"
if str(COMMON_DIR) not in sys.path:
    sys.path.insert(0, str(COMMON_DIR))

from hipp_density_assets import (
    detect_space_strict,
    find_surface_asset_strict,
    load_surface_density_from_pipeline_config,
    subject_surf_dir,
)


def run(cmd: list[str], step_name: str, timings: list[dict[str, object]]) -> None:
    start = perf_counter()
    proc = subprocess.run(cmd, text=True, capture_output=True)
    duration = perf_counter() - start
    timings.append(
        {
            "step": step_name,
            "command": cmd,
            "returncode": int(proc.returncode),
            "duration_seconds": duration,
            "status": "success" if proc.returncode == 0 else "failed",
        }
    )
    if proc.returncode != 0:
        raise RuntimeError(
            f"Command failed ({proc.returncode}): {' '.join(cmd)}\nSTDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}"
        )


def require_existing(path: Path, label: str) -> Path:
    if not path.exists():
        raise FileNotFoundError(f"Missing {label}: {path}")
    return path


def find_subject_surface(
    hippunfold_dir: Path,
    subject: str,
    hemi: str,
    space: str,
    density: str,
    surface_name: str = "midthickness",
) -> Path:
    surf_dir = subject_surf_dir(hippunfold_dir, subject)
    return find_surface_asset_strict(
        surf_dir=surf_dir,
        subject=subject,
        hemi=hemi,
        space=space,
        density=density,
        suffix=f"{surface_name}.surf.gii",
    )


def find_structural_label(hippunfold_dir: Path, subject: str, hemi: str, space: str, density: str) -> Path:
    surf_dir = subject_surf_dir(hippunfold_dir, subject)
    return find_surface_asset_strict(
        surf_dir=surf_dir,
        subject=subject,
        hemi=hemi,
        space=space,
        density=density,
        suffix="atlas-multihist7_subfields.label.gii",
    )


def main() -> int:
    default_density = load_surface_density_from_pipeline_config(REPO_ROOT / "config" / "hippo_pipeline.toml")
    parser = argparse.ArgumentParser(description="Run the post-HippUnfold surface-first HippoMaps pipeline")
    parser.add_argument("--subject", required=True)
    parser.add_argument("--bold", required=True)
    parser.add_argument("--brain-mask", required=True)
    parser.add_argument("--dtseries", required=True)
    parser.add_argument("--hippunfold-dir", required=True)
    parser.add_argument("--density", default=default_density)
    parser.add_argument("--space", default="auto")
    parser.add_argument("--outdir", required=True)
    args = parser.parse_args()
    python_exe = sys.executable or "python"
    view_config = json.loads((REPO_ROOT / "config" / "native_surface_3d_view.json").read_text(encoding="utf-8"))
    layout_template_json = REPO_ROOT / "config" / "native_surface_layout_template.json"
    atlas_dlabel = REPO_ROOT / "data" / "atlas" / "schaefer400" / "Schaefer2018_400Parcels_7Networks_order.dlabel.nii"

    outdir = Path(args.outdir)
    surf_dir = outdir / "surface"
    ref_dir = outdir / "reference"
    grad_dir = outdir / "gradients"
    for path in [surf_dir, ref_dir, grad_dir]:
        path.mkdir(parents=True, exist_ok=True)

    timings: list[dict[str, object]] = []
    surf_source = subject_surf_dir(Path(args.hippunfold_dir), args.subject)
    resolved_space = detect_space_strict(
        surf_dir=surf_source,
        subject=args.subject,
        density=args.density,
        preferred=args.space,
        candidates=["corobl", "T2w", "T1w", "nativepro"],
    )

    summary: dict[str, object] = {
        "subject": args.subject,
        "space": resolved_space,
        "surface_dir": str(surf_dir),
        "reference_dir": str(ref_dir),
        "gradient_dir": str(grad_dir),
        "timings": timings,
    }

    try:
        require_existing(Path(args.dtseries), "surface dtseries")
        require_existing(atlas_dlabel, "Schaefer dlabel atlas")

        run(
            [
                python_exe,
                str(REPO_ROOT / "scripts" / "extract_schaefer_cifti_reference.py"),
                "--dtseries",
                args.dtseries,
                "--atlas-dlabel",
                str(atlas_dlabel),
                "--outdir",
                str(ref_dir),
            ],
            "extract_schaefer_cifti_reference",
            timings,
        )

        run(
            [
                python_exe,
                str(REPO_ROOT / "scripts" / "common" / "sample_hipp_surface_timeseries.py"),
                "--bold",
                args.bold,
                "--hippunfold-dir",
                args.hippunfold_dir,
                "--subject",
                args.subject,
                "--density",
                args.density,
                "--space",
                resolved_space,
                "--outdir",
                str(surf_dir),
            ],
            "sample_hipp_surface_timeseries",
            timings,
        )

        structural_native_png = outdir / f"sub-{args.subject}_hipp_structural_native.png"
        run(
            [
                python_exe,
                str(REPO_ROOT / "scripts" / "workbench" / "render_native_surface_label_map_3d.py"),
                "--subject",
                args.subject,
                "--left-surface",
                str(find_subject_surface(Path(args.hippunfold_dir), args.subject, "L", resolved_space, args.density)),
                "--right-surface",
                str(find_subject_surface(Path(args.hippunfold_dir), args.subject, "R", resolved_space, args.density)),
                "--left-labels",
                str(find_structural_label(Path(args.hippunfold_dir), args.subject, "L", resolved_space, args.density)),
                "--right-labels",
                str(find_structural_label(Path(args.hippunfold_dir), args.subject, "R", resolved_space, args.density)),
                "--vertical-axis",
                str(view_config["vertical_axis"]),
                "--elev",
                str(view_config["elev"]),
                "--azim",
                str(view_config["azim"]),
                "--roll",
                str(view_config["roll"]),
                "--zoom",
                str(view_config["zoom"]),
                "--pad",
                str(view_config["pad"]),
                "--dpi",
                str(view_config["dpi"]),
                "--fig-width",
                str(view_config["fig_width"]),
                "--fig-height",
                str(view_config["fig_height"]),
                "--background",
                str(view_config["background"]),
                "--layout-template-json",
                str(layout_template_json),
                "--out",
                str(structural_native_png),
            ],
            "render_structural_native_map",
            timings,
        )
        summary["structural_native_png"] = str(structural_native_png)

        parcel_ts = str(ref_dir / "schaefer400_parcel_timeseries.npy")
        for hemi in ["L", "R"]:
            hemi_grad = grad_dir / f"hemi-{hemi}"
            hemi_grad.mkdir(parents=True, exist_ok=True)
            hemi_ts = (
                surf_dir / f"sub-{args.subject}_hemi-{hemi}_space-{resolved_space}_den-{args.density}_label-hipp_bold.npy"
            )
            run(
                [
                    python_exe,
                    str(REPO_ROOT / "scripts" / "common" / "compute_fc_gradients.py"),
                    "--hipp-ts",
                    str(hemi_ts),
                    "--parcel-ts",
                    parcel_ts,
                    "--surface",
                    str(find_subject_surface(Path(args.hippunfold_dir), args.subject, hemi, resolved_space, args.density)),
                    "--outdir",
                    str(hemi_grad),
                ],
                f"compute_fc_gradients_hemi_{hemi}",
                timings,
            )

        gradient_native_png = outdir / f"sub-{args.subject}_hipp_fc_gradient1_native.png"
        run(
            [
                python_exe,
                str(REPO_ROOT / "scripts" / "workbench" / "render_native_surface_scalar_map_3d.py"),
                "--subject",
                args.subject,
                "--left-surface",
                str(find_subject_surface(Path(args.hippunfold_dir), args.subject, "L", resolved_space, args.density)),
                "--right-surface",
                str(find_subject_surface(Path(args.hippunfold_dir), args.subject, "R", resolved_space, args.density)),
                "--left-scalars",
                str(grad_dir / "hemi-L" / "hipp_fc_gradient1.npy"),
                "--right-scalars",
                str(grad_dir / "hemi-R" / "hipp_fc_gradient1.npy"),
                "--colorbar-label",
                "Gradient 1",
                "--vertical-axis",
                str(view_config["vertical_axis"]),
                "--elev",
                str(view_config["elev"]),
                "--azim",
                str(view_config["azim"]),
                "--roll",
                str(view_config["roll"]),
                "--zoom",
                str(view_config["zoom"]),
                "--pad",
                str(view_config["pad"]),
                "--dpi",
                str(view_config["dpi"]),
                "--fig-width",
                str(view_config["fig_width"]),
                "--fig-height",
                str(view_config["fig_height"]),
                "--background",
                str(view_config["background"]),
                "--layout-template-json",
                str(layout_template_json),
                "--out",
                str(gradient_native_png),
            ],
            "render_gradient_native_map",
            timings,
        )
        summary["gradient_native_png"] = str(gradient_native_png)
        summary["formal_functional_outputs"] = [
            str(grad_dir / "hemi-L" / "hipp_fc_gradients.npy"),
            str(grad_dir / "hemi-R" / "hipp_fc_gradients.npy"),
            str(gradient_native_png),
        ]
        summary["status"] = "success"
    except Exception as exc:
        summary["status"] = "failed"
        summary["error"] = str(exc)
        raise
    finally:
        (outdir / "post_hippunfold_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        (outdir / "post_pipeline_timing.json").write_text(json.dumps(timings, indent=2), encoding="utf-8")

    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
