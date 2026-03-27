#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from time import perf_counter


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


def detect_space(hippunfold_dir: Path, subject: str, density: str, preferred: str | None) -> str:
    if preferred and preferred != "auto":
        return preferred
    surf_dir = hippunfold_dir / f"sub-{subject}" / "surf"
    for candidate in ["corobl", "T2w", "T1w", "nativepro"]:
        patterns = [
            f"sub-{subject}_hemi-L_space-{candidate}_den-{density}_label-hipp_midthickness.surf.gii",
            f"sub-{subject}_hemi-L_space-{candidate}_label-hipp_midthickness.surf.gii",
        ]
        for pattern in patterns:
            if list(surf_dir.glob(pattern)):
                return candidate
    raise FileNotFoundError(
        f"Could not auto-detect a folded hippocampal surface in {surf_dir} for density={density}"
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
    surf_dir = hippunfold_dir / f"sub-{subject}" / "surf"
    patterns = [
        f"sub-{subject}_hemi-{hemi}_space-{space}_den-{density}_label-hipp_{surface_name}.surf.gii",
        f"sub-{subject}_hemi-{hemi}_space-{space}_label-hipp_{surface_name}.surf.gii",
    ]
    for pattern in patterns:
        matches = sorted(surf_dir.glob(pattern))
        if matches:
            return matches[0]
    raise FileNotFoundError(
        f"Missing {surface_name} surface for subject={subject}, hemi={hemi}, space={space}, density={density}"
    )


def find_structural_label(hippunfold_dir: Path, subject: str, hemi: str, space: str, density: str) -> Path:
    surf_dir = hippunfold_dir / f"sub-{subject}" / "surf"
    patterns = [
        f"sub-{subject}_hemi-{hemi}_space-{space}_den-{density}_label-hipp_atlas-multihist7_subfields.label.gii",
        f"sub-{subject}_hemi-{hemi}_space-{space}_label-hipp_atlas-multihist7_subfields.label.gii",
    ]
    for pattern in patterns:
        matches = sorted(surf_dir.glob(pattern))
        if matches:
            return matches[0]
    raise FileNotFoundError(
        f"Missing structural label for subject={subject}, hemi={hemi}, space={space}, density={density}"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the post-HippUnfold volume-based HippoMaps pipeline")
    parser.add_argument("--subject", required=True)
    parser.add_argument("--bold", required=True)
    parser.add_argument("--brain-mask", required=True)
    parser.add_argument("--hippunfold-dir", required=True)
    parser.add_argument("--reference-dir", required=True)
    parser.add_argument("--density", default="2mm")
    parser.add_argument("--space", default="auto")
    parser.add_argument("--skip-volume-backproject", action="store_true")
    parser.add_argument("--outdir", required=True)
    args = parser.parse_args()
    python_exe = sys.executable or "python"
    hippo_python = "/opt/miniconda3/envs/hippo/bin/python"
    surface_to_volume_python = hippo_python if Path(hippo_python).exists() else python_exe
    view_config = json.loads(Path("config/native_surface_3d_view.json").read_text(encoding="utf-8"))
    layout_template_json = Path("config/native_surface_layout_template.json")

    outdir = Path(args.outdir)
    surf_dir = outdir / "surface"
    wta_dir = outdir / "wta"
    vol_dir = outdir / "volume"
    seed_dir = outdir / "seed_fc"
    for path in [surf_dir, wta_dir, vol_dir, seed_dir]:
        path.mkdir(parents=True, exist_ok=True)

    timings: list[dict[str, object]] = []
    resolved_space = detect_space(Path(args.hippunfold_dir), args.subject, args.density, args.space)

    summary: dict[str, object] = {
        "subject": args.subject,
        "space": resolved_space,
        "surface_dir": str(surf_dir),
        "wta_dir": str(wta_dir),
        "volume_dir": str(vol_dir),
        "seed_fc_dir": str(seed_dir),
        "timings": timings,
    }

    try:
        run(
            [
                python_exe,
                "scripts/sample_hipp_surface_timeseries.py",
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
                "scripts/render_native_surface_label_map_3d.py",
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

        network_ts = str(Path(args.reference_dir) / "schaefer7_network_timeseries.npy")
        hemi_label_paths: dict[str, str] = {}
        for hemi in ["L", "R"]:
            hemi_wta = wta_dir / f"hemi-{hemi}"
            hemi_wta.mkdir(parents=True, exist_ok=True)
            hemi_ts = (
                surf_dir / f"sub-{args.subject}_hemi-{hemi}_space-{resolved_space}_den-{args.density}_label-hipp_bold.npy"
            )
            run(
                [
                    python_exe,
                    "scripts/compute_wta_labels.py",
                    "--hipp-ts",
                    str(hemi_ts),
                    "--network-ts",
                    network_ts,
                    "--outdir",
                    str(hemi_wta),
                ],
                f"compute_wta_labels_hemi_{hemi}",
                timings,
            )

        wta_native_png = outdir / f"sub-{args.subject}_hipp_wta_native.png"
        run(
            [
                python_exe,
                "scripts/render_native_surface_label_map_3d.py",
                "--subject",
                args.subject,
                "--left-surface",
                str(find_subject_surface(Path(args.hippunfold_dir), args.subject, "L", resolved_space, args.density)),
                "--right-surface",
                str(find_subject_surface(Path(args.hippunfold_dir), args.subject, "R", resolved_space, args.density)),
                "--left-labels",
                str(wta_dir / "hemi-L" / "hipp_wta_labels.npy"),
                "--right-labels",
                str(wta_dir / "hemi-R" / "hipp_wta_labels.npy"),
                "--style-json",
                "config/hipp_network_style.json",
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
                str(wta_native_png),
            ],
            "render_wta_native_map",
            timings,
        )
        summary["wta_native_png"] = str(wta_native_png)

        if args.skip_volume_backproject:
            summary["status"] = "success"
            summary["skipped"] = ["surface_labels_to_volume", "combine_hemi_labels_to_bold", "compute_seed_fc"]
            print(json.dumps(summary, indent=2))
            return 0

        for hemi in ["L", "R"]:
            hemi_wta = wta_dir / f"hemi-{hemi}"
            hemi_labels_t1w = vol_dir / f"sub-{args.subject}_hemi-{hemi}_space-{resolved_space}_desc-wta_labels.nii.gz"
            hemi_label_paths[hemi] = str(hemi_labels_t1w)
            run(
                [
                    surface_to_volume_python,
                    "scripts/surface_labels_to_volume.py",
                    "--surf-labels",
                    str(hemi_wta / "hipp_wta_labels.npy"),
                    "--density",
                    args.density,
                    "--hippunfold-dir",
                    args.hippunfold_dir,
                    "--subject",
                    args.subject,
                    "--hemi",
                    hemi,
                    "--space",
                    resolved_space,
                    "--out",
                    str(hemi_labels_t1w),
                ],
                f"surface_labels_to_volume_hemi_{hemi}",
                timings,
            )

        merged_bold = vol_dir / f"sub-{args.subject}_space-bold_desc-wta_labels.nii.gz"
        summary["merged_bold_labels"] = str(merged_bold)
        run(
            [
                python_exe,
                "scripts/combine_hemi_labels_to_bold.py",
                "--left-labels",
                hemi_label_paths["L"],
                "--right-labels",
                hemi_label_paths["R"],
                "--bold-ref",
                args.bold,
                "--out",
                str(merged_bold),
            ],
            "combine_hemi_labels_to_bold",
            timings,
        )

        run(
            [
                python_exe,
                "scripts/compute_seed_fc.py",
                "--bold",
                args.bold,
                "--brain-mask",
                args.brain_mask,
                "--seed-labels",
                str(merged_bold),
                "--outdir",
                str(seed_dir),
            ],
            "compute_seed_fc",
            timings,
        )
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
