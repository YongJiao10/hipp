#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path

import numpy as np


def load_surface_to_volume():
    module_path = Path("/opt/miniconda3/envs/hippo/lib/python3.11/site-packages/hippomaps/utils.py")
    spec = importlib.util.spec_from_file_location("hippomaps_utils_local", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load spec for {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.surface_to_volume


def detect_space(hippunfold_dir: Path, subject: str, hemi: str, density: str, preferred: str | None) -> str:
    if preferred and preferred != "auto":
        return preferred
    surf_dir = hippunfold_dir / f"sub-{subject}" / "surf"
    patterns = [
        "sub-{subject}_hemi-{hemi}_space-{space}_den-{density}_label-hipp_midthickness.surf.gii",
        "sub-{subject}_hemi-{hemi}_space-{space}_label-hipp_midthickness.surf.gii",
    ]
    for candidate in ["T2w", "T1w", "nativepro", "corobl"]:
        for pattern in patterns:
            resolved = pattern.format(subject=subject, hemi=hemi, space=candidate, density=density)
            if list(surf_dir.glob(resolved)):
                return candidate
    raise FileNotFoundError(
        f"Could not auto-detect folded surface space in {surf_dir} for hemi={hemi}, density={density}"
    )


def ensure_coords_bridge(hippunfold_dir: Path, subject: str) -> None:
    subject_dir = hippunfold_dir / f"sub-{subject}"
    coords_dir = subject_dir / "coords"
    if coords_dir.is_symlink() and not coords_dir.exists():
        coords_dir.unlink()
    if coords_dir.exists():
        return

    work_coords_dir = hippunfold_dir / "work" / f"sub-{subject}" / "coords"
    if not work_coords_dir.exists():
        return

    subject_dir.mkdir(parents=True, exist_ok=True)
    try:
        coords_dir.symlink_to(work_coords_dir.resolve())
    except FileExistsError:
        return


def main() -> int:
    parser = argparse.ArgumentParser(description="Project hippocampal surface labels back to volume")
    parser.add_argument("--surf-labels", required=True)
    parser.add_argument("--density", required=True)
    parser.add_argument("--hippunfold-dir", required=True)
    parser.add_argument("--subject", required=True)
    parser.add_argument("--hemi", required=True, choices=["L", "R"])
    parser.add_argument("--space", default="auto")
    parser.add_argument("--label", default="hipp")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    surf_labels = np.load(args.surf_labels)
    surface_to_volume = load_surface_to_volume()
    hippunfold_dir = Path(args.hippunfold_dir)
    ensure_coords_bridge(hippunfold_dir, args.subject)
    resolved_space = detect_space(hippunfold_dir, args.subject, args.hemi, args.density, args.space)
    surface_to_volume(
        surf_data=surf_labels.astype(np.float32),
        indensity=args.density,
        hippunfold_dir=args.hippunfold_dir,
        sub=args.subject,
        ses="",
        hemi=args.hemi,
        space=resolved_space,
        label=args.label,
        save_out_name=args.out,
        method="nearest",
    )
    print(args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
