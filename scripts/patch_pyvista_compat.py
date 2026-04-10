#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path


PATCH_MARKER = "HIPPOMAPS_MACOS_COMPAT_PYVISTA"


def patch_get_boundary_vertices(path: Path) -> bool:
    text = path.read_text(encoding="utf-8")
    if PATCH_MARKER in text:
        return False

    needle = 'logger.info(f"Surface loaded: {surface.n_points} vertices, {surface.n_faces} faces.")\n'
    replacement = (
        f"# {PATCH_MARKER}: pyvista removed non-strict PolyData.n_faces in newer releases\n"
        'logger.info(f"Surface loaded: {surface.n_points} vertices, {surface.n_cells} faces.")\n'
    )
    if needle not in text:
        return False
    text = text.replace(needle, replacement, 1)
    path.write_text(text, encoding="utf-8")
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description="Patch HippUnfold pyvista API compatibility for local macOS testing")
    parser.add_argument("--hippunfold-site-root", required=True)
    parser.add_argument("--runtime-source-cache", required=True)
    args = parser.parse_args()

    candidates = [
        Path(args.hippunfold_site_root) / "workflow" / "scripts" / "get_boundary_vertices.py",
        Path(args.hippunfold_site_root) / "workflow" / "scripts" / "postproc_boundary_vertices.py",
        Path(args.runtime_source_cache)
        / "file"
        / Path(args.hippunfold_site_root).relative_to("/")
        / "workflow"
        / "scripts"
        / "get_boundary_vertices.py",
        Path(args.runtime_source_cache)
        / "file"
        / Path(args.hippunfold_site_root).relative_to("/")
        / "workflow"
        / "scripts"
        / "postproc_boundary_vertices.py",
    ]

    patched = []
    skipped = []
    for path in candidates:
        if not path.exists():
            continue
        try:
            if patch_get_boundary_vertices(path):
                patched.append(str(path))
        except PermissionError:
            skipped.append(f"{path} (permission denied)")

    if patched:
        print("Patched pyvista compatibility files:")
        for path in patched:
            print(path)
    if skipped:
        print("Skipped pyvista compatibility files:")
        for path in skipped:
            print(path)
    if not patched and not skipped:
        print("No pyvista compatibility patches were needed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
