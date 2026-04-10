#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path


REPLACEMENT = """name: pyvista
channels:
  - conda-forge
dependencies:
  - python=3.11
  - numpy
  - nibabel
  - pyvista
  - vtk
  - scipy
  - pandas
  - networkx
# HIPPOMAPS_MACOS_COMPAT_PYVISTA_ENV: modern dependency set compatible with Snakemake 9 scripts
"""


def patch_file(path: Path) -> bool:
    if not path.exists():
        return False
    text = path.read_text(encoding="utf-8")
    if text.strip() == REPLACEMENT.strip():
        return False
    if "pyvista" not in text or "python=3.9" not in text:
        return False
    path.write_text(REPLACEMENT, encoding="utf-8")
    return True


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Patch HippUnfold pyvista env to use a Python version compatible with Snakemake 9 scripts"
    )
    parser.add_argument("--hippunfold-site-root", required=True)
    parser.add_argument("--runtime-source-cache", required=True)
    args = parser.parse_args()

    site_root = Path(args.hippunfold_site_root)
    runtime_source_cache = Path(args.runtime_source_cache)
    candidates = [
        site_root / "workflow" / "envs" / "pyvista.yaml",
        runtime_source_cache
        / "file"
        / site_root.relative_to("/")
        / "workflow"
        / "envs"
        / "pyvista.yaml",
    ]

    patched = []
    skipped = []
    for path in candidates:
        if not path.exists():
            continue
        try:
            if patch_file(path):
                patched.append(str(path))
        except PermissionError:
            skipped.append(f"{path} (permission denied)")

    if patched:
        print("Patched pyvista env compatibility files:")
        for path in patched:
            print(path)
    if skipped:
        print("Skipped pyvista env compatibility files:")
        for path in skipped:
            print(path)
    if not patched and not skipped:
        print("No pyvista env compatibility patches were needed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
