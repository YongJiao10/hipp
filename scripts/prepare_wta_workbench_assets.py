#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

import nibabel as nib
import numpy as np
from nibabel.gifti import GiftiDataArray, GiftiImage, GiftiLabel, GiftiLabelTable


def load_style(path: Path) -> dict[int, dict[str, object]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    return {int(k): v for k, v in data.items()}


def find_surface(hippunfold_dir: Path, subject: str, hemi: str, density: str, space: str) -> Path:
    candidates = [
        hippunfold_dir / f"sub-{subject}" / "surf" / f"sub-{subject}_hemi-{hemi}_space-{space}_den-{density}_label-hipp_midthickness.surf.gii",
        hippunfold_dir / f"sub-{subject}" / "surf" / f"sub-{subject}_hemi-{hemi}_space-{space}_label-hipp_midthickness.surf.gii",
        hippunfold_dir / "work" / f"sub-{subject}" / "surf" / f"sub-{subject}_hemi-{hemi}_space-{space}_den-{density}_label-hipp_midthickness.surf.gii",
        hippunfold_dir / "work" / f"sub-{subject}" / "surf" / f"sub-{subject}_hemi-{hemi}_space-{space}_label-hipp_midthickness.surf.gii",
    ]
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError(f"Could not find surface for hemi={hemi}, space={space}, density={density}")


def make_label_gifti(labels: np.ndarray, style: dict[int, dict[str, object]]) -> GiftiImage:
    label_table = GiftiLabelTable()
    unknown = GiftiLabel(key=0, red=0.0, green=0.0, blue=0.0, alpha=0.0)
    unknown.label = "???"
    label_table.labels.append(unknown)
    for key in sorted(style):
        rgba = style[key]["rgba"]
        lab = GiftiLabel(
            key=int(key),
            red=float(rgba[0]) / 255.0,
            green=float(rgba[1]) / 255.0,
            blue=float(rgba[2]) / 255.0,
            alpha=float(rgba[3]) / 255.0,
        )
        lab.label = str(style[key]["name"])
        label_table.labels.append(lab)

    arr = GiftiDataArray(data=labels.astype(np.int32), intent="NIFTI_INTENT_LABEL", datatype="NIFTI_TYPE_INT32")
    img = GiftiImage(darrays=[arr], labeltable=label_table)
    return img


def run(cmd: list[str]) -> None:
    proc = subprocess.run(cmd, text=True, capture_output=True)
    if proc.returncode != 0:
        raise RuntimeError(
            f"Command failed ({proc.returncode}): {' '.join(cmd)}\nSTDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}"
        )


def main() -> int:
    parser = argparse.ArgumentParser(description="Prepare Workbench label assets from hippocampal WTA numpy labels")
    parser.add_argument("--subject", required=True)
    parser.add_argument("--hippunfold-dir", required=True)
    parser.add_argument("--left-labels", required=True, help="left hemi WTA labels .npy")
    parser.add_argument("--right-labels", required=True, help="right hemi WTA labels .npy")
    parser.add_argument("--density", default="512")
    parser.add_argument("--style-json", default="config/hipp_network_style.json")
    parser.add_argument("--spaces", nargs="+", default=["unfold", "T2w"], help="surface spaces to emit")
    parser.add_argument("--outdir", required=True)
    parser.add_argument("--wb-command", default="scripts/wb_command")
    args = parser.parse_args()

    hippunfold_dir = Path(args.hippunfold_dir)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    style = load_style(Path(args.style_json))

    hemi_labels = {
        "L": np.load(args.left_labels).astype(np.int32),
        "R": np.load(args.right_labels).astype(np.int32),
    }

    summary: dict[str, object] = {"subject": args.subject, "density": args.density, "spaces": {}}
    for space in args.spaces:
        space_dir = outdir / space
        space_dir.mkdir(parents=True, exist_ok=True)
        left_out = space_dir / f"sub-{args.subject}_hemi-L_space-{space}_den-{args.density}_label-hipp_wta.label.gii"
        right_out = space_dir / f"sub-{args.subject}_hemi-R_space-{space}_den-{args.density}_label-hipp_wta.label.gii"

        for hemi, out_path in [("L", left_out), ("R", right_out)]:
            surf_path = find_surface(hippunfold_dir, args.subject, hemi, args.density, space)
            n_vertices = nib.load(str(surf_path)).agg_data("pointset").shape[0]
            labels = hemi_labels[hemi]
            if labels.shape[0] != n_vertices:
                raise ValueError(
                    f"Vertex count mismatch for hemi={hemi}, space={space}: labels={labels.shape[0]} vs surf={n_vertices}"
                )
            nib.save(make_label_gifti(labels, style), out_path)

        dlabel_out = space_dir / f"sub-{args.subject}_space-{space}_den-{args.density}_label-hipp_wta.dlabel.nii"
        run(
            [
                args.wb_command,
                "-cifti-create-label",
                str(dlabel_out),
                "-left-label",
                str(left_out),
                "-right-label",
                str(right_out),
            ]
        )

        label_table_out = space_dir / f"sub-{args.subject}_space-{space}_den-{args.density}_label-hipp_wta_labeltable.txt"
        run([args.wb_command, "-cifti-label-export-table", str(dlabel_out), "1", str(label_table_out)])
        summary["spaces"][space] = {
            "left_label_gii": str(left_out),
            "right_label_gii": str(right_out),
            "dlabel": str(dlabel_out),
            "label_table": str(label_table_out),
        }

    (outdir / "workbench_assets_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
