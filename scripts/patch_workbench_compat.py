#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path


PATCH_MARKER = "HIPPOMAPS_MACOS_COMPAT_WORKBENCH"
REPLACEMENT = """name: workbench
channels:
  - conda-forge
dependencies:
  - python=3.9
# HIPPOMAPS_MACOS_COMPAT_WORKBENCH: rely on wb_command from PATH on macOS
"""


def patch_workbench_yaml(path: Path) -> bool:
    text = path.read_text(encoding="utf-8")
    if PATCH_MARKER in text:
        return False
    if "dev-connectome-workbench" not in text:
        return False
    path.write_text(REPLACEMENT, encoding="utf-8")
    return True


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Patch HippUnfold workbench env to reuse local wb_command on macOS"
    )
    parser.add_argument("--hippunfold-site-root", required=True)
    parser.add_argument("--runtime-source-cache", required=True)
    args = parser.parse_args()

    candidates = [
        Path(args.hippunfold_site_root) / "workflow" / "envs" / "workbench.yaml",
        Path(args.runtime_source_cache)
        / "file"
        / Path(args.hippunfold_site_root).relative_to("/")
        / "workflow"
        / "envs"
        / "workbench.yaml",
    ]

    patched = []
    skipped = []
    for path in candidates:
        if not path.exists():
            continue
        try:
            if patch_workbench_yaml(path):
                patched.append(str(path))
        except PermissionError:
            skipped.append(f"{path} (permission denied)")

    if patched:
        print("Patched workbench compatibility files:")
        for path in patched:
            print(path)
    if skipped:
        print("Skipped workbench compatibility files:")
        for path in skipped:
            print(path)
    if not patched and not skipped:
        print("No workbench compatibility patches were needed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
