#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path


PATCH_MARKER = "HIPPOMAPS_MACOS_COMPAT_SUBFIELDS_SHELL"
OLD = (
    '"wb_command -label-to-volume-mapping {input.label_gii} {input.midthickness_surf} {input.ref_nii} {output.nii_label}"\n'
    '        " -ribbon-constrained {input.inner_surf} {input.outer_surf} &>> {log}"\n'
)
NEW = (
    '"wb_command -label-to-volume-mapping {input.label_gii} {input.midthickness_surf} {input.ref_nii} {output.nii_label}"\n'
    '        " -ribbon-constrained {input.inner_surf} {input.outer_surf} >> {log} 2>&1"\n'
    "        # HIPPOMAPS_MACOS_COMPAT_SUBFIELDS_SHELL: use portable append redirection syntax\n"
)


def patch_file(path: Path) -> bool:
    if not path.exists():
        return False
    text = path.read_text(encoding="utf-8")
    if PATCH_MARKER in text:
        return False
    if OLD not in text:
        return False
    path.write_text(text.replace(OLD, NEW), encoding="utf-8")
    return True


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Patch HippUnfold subfields shell redirection to a portable form"
    )
    parser.add_argument("--hippunfold-site-root", required=True)
    parser.add_argument("--runtime-source-cache", required=True)
    args = parser.parse_args()

    site_root = Path(args.hippunfold_site_root)
    runtime_source_cache = Path(args.runtime_source_cache)
    candidates = [
        site_root / "workflow" / "rules" / "subfields.smk",
        runtime_source_cache
        / "file"
        / site_root.relative_to("/")
        / "workflow"
        / "rules"
        / "subfields.smk",
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
        print("Patched subfields shell compatibility files:")
        for path in patched:
            print(path)
    if skipped:
        print("Skipped subfields shell compatibility files:")
        for path in skipped:
            print(path)
    if not patched and not skipped:
        print("No subfields shell compatibility patches were needed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
