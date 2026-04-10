#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path


PATCH_MARKER = "HIPPOMAPS_MACOS_COMPAT_NEUROVIS"
REPLACEMENT = """name: neurovis
channels:
  - conda-forge
dependencies:
  - python=3.11
  - numpy
  - scipy
  - nibabel
  - pandas
  - matplotlib
  - seaborn
  - nilearn
# HIPPOMAPS_MACOS_COMPAT_NEUROVIS: modern dependency set compatible with Snakemake 9 scripts
"""


def patch_neurovis_yaml(path: Path) -> bool:
    text = path.read_text(encoding="utf-8")
    if text.strip() == REPLACEMENT.strip():
        return False
    if (
        PATCH_MARKER not in text
        and "python=3.9" not in text
        and "matplotlib=3.4.2" not in text
    ):
        return False
    path.write_text(REPLACEMENT, encoding="utf-8")
    return True


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Patch HippUnfold neurovis env to use a Python version compatible with Snakemake 9 scripts"
    )
    parser.add_argument("--hippunfold-site-root", required=True)
    parser.add_argument("--runtime-source-cache", required=True)
    args = parser.parse_args()

    candidates = [
        Path(args.hippunfold_site_root) / "workflow" / "envs" / "neurovis.yaml",
        Path(args.runtime_source_cache)
        / "file"
        / Path(args.hippunfold_site_root).relative_to("/")
        / "workflow"
        / "envs"
        / "neurovis.yaml",
    ]

    patched = []
    skipped = []
    for path in candidates:
        if not path.exists():
            continue
        try:
            if patch_neurovis_yaml(path):
                patched.append(str(path))
        except PermissionError:
            skipped.append(f"{path} (permission denied)")

    if patched:
        print("Patched neurovis compatibility files:")
        for path in patched:
            print(path)
    if skipped:
        print("Skipped neurovis compatibility files:")
        for path in skipped:
            print(path)
    if not patched and not skipped:
        print("No neurovis compatibility patches were needed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
