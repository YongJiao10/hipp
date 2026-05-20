#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.metadata
import importlib.util
import sys
from pathlib import Path

import numpy as np

COMMON_DIR = Path(__file__).resolve().parent / "common"
if str(COMMON_DIR) not in sys.path:
    sys.path.insert(0, str(COMMON_DIR))

from hipp_density_assets import detect_space_strict, subject_surf_dir


def load_surface_to_volume():
    module_path = Path(importlib.metadata.distribution("hippomaps").locate_file("hippomaps/utils.py")).resolve()
    spec = importlib.util.spec_from_file_location("hippomaps_utils_local", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load spec for {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.surface_to_volume


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
    surf_dir = subject_surf_dir(hippunfold_dir, args.subject)
    resolved_space = detect_space_strict(
        surf_dir=surf_dir,
        subject=args.subject,
        density=args.density,
        preferred=args.space,
        candidates=["T2w", "T1w", "nativepro", "corobl"],
    )
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
