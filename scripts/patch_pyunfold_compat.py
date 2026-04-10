#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path


REPLACEMENT = """name: pyunfold
channels:
  - conda-forge
dependencies:
  - python=3.11
  - numpy
  - scipy
  - nibabel
  - pandas
  - astropy
  - scikit-fmm
# HIPPOMAPS_MACOS_COMPAT_PYUNFOLD: modern dependency set compatible with Snakemake 9 scripts
"""


def _candidate_paths(hippunfold_site_root: Path, runtime_source_cache: Path) -> list[Path]:
    candidates = [
        hippunfold_site_root / "workflow" / "envs" / "pyunfold.yaml",
        runtime_source_cache
        / "file"
        / hippunfold_site_root.relative_to("/")
        / "workflow"
        / "envs"
        / "pyunfold.yaml",
    ]
    if runtime_source_cache.exists():
        candidates.extend(runtime_source_cache.rglob("pyunfold.yaml"))

    unique: list[Path] = []
    seen: set[Path] = set()
    for path in candidates:
        if path not in seen:
            unique.append(path)
            seen.add(path)
    return unique


def patch_file(path: Path) -> bool:
    if not path.exists():
        return False
    text = path.read_text()
    if text == REPLACEMENT:
        return False
    path.write_text(REPLACEMENT)
    return True


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--hippunfold-site-root", type=Path, required=True)
    parser.add_argument("--runtime-source-cache", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    changed = False
    for path in _candidate_paths(args.hippunfold_site_root, args.runtime_source_cache):
        if patch_file(path):
            print(f"patched {path}")
            changed = True
    if not changed:
        print("pyunfold compat patch already applied")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
