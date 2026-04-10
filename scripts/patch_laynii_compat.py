#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path


PATCH_MARKER = "HIPPOMAPS_MACOS_COMPAT_LAYNII"


def patch_coords_smk(path: Path) -> bool:
    text = path.read_text(encoding="utf-8")
    if PATCH_MARKER in text:
        return False

    old_equidist_variants = [
        (
        '        "cp {input} dseg.nii.gz && "\n'
        '        "LN2_LAYERS  -rim dseg.nii.gz && "\n'
        '        "cp dseg_metric_equidist.nii.gz {output.equidist}"\n'
        ),
        (
        '        "cp {input} dseg.nii.gz && "\n'
        '        "LN2_LAYERS  -rim dseg.nii.gz &> {log} && "\n'
        '        "cp dseg_metric_equidist.nii.gz {output.equidist}"\n'
        ),
    ]
    new_equidist = (
        '        "cp {input} dseg.nii.gz && "\n'
        f'        "gunzip -c dseg.nii.gz > dseg.nii && "  # {PATCH_MARKER}: macOS local LAYNII cannot read .nii.gz reliably\n'
        '        "LN2_LAYERS  -rim dseg.nii'
        + (' &> {log}' if "{log}" in text else "")
        + ' && "\n'
        '        "gzip -f dseg_metric_equidist.nii && "\n'
        '        "cp dseg_metric_equidist.nii.gz {output.equidist}"\n'
    )

    old_equivol_variants = [
        (
        '        "cp {input} dseg.nii.gz && "\n'
        '        "LN2_LAYERS  -rim dseg.nii.gz -equivol && "\n'
        '        "cp dseg_metric_equivol.nii.gz {output.equivol}"\n'
        ),
        (
        '        "cp {input} dseg.nii.gz && "\n'
        '        "LN2_LAYERS  -rim dseg.nii.gz -equivol &> {log} && "\n'
        '        "cp dseg_metric_equivol.nii.gz {output.equivol}"\n'
        ),
    ]
    new_equivol = (
        '        "cp {input} dseg.nii.gz && "\n'
        f'        "gunzip -c dseg.nii.gz > dseg.nii && "  # {PATCH_MARKER}: macOS local LAYNII cannot read .nii.gz reliably\n'
        '        "LN2_LAYERS  -rim dseg.nii -equivol'
        + (' &> {log}' if "{log}" in text else "")
        + ' && "\n'
        '        "gzip -f dseg_metric_equivol.nii && "\n'
        '        "cp dseg_metric_equivol.nii.gz {output.equivol}"\n'
    )

    old_equidist = next((variant for variant in old_equidist_variants if variant in text), None)
    old_equivol = next((variant for variant in old_equivol_variants if variant in text), None)
    if old_equidist is None or old_equivol is None:
        return False

    text = text.replace(old_equidist, new_equidist, 1)
    text = text.replace(old_equivol, new_equivol, 1)
    path.write_text(text, encoding="utf-8")
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description="Patch HippUnfold LAYNII rules for macOS local compatibility")
    parser.add_argument("--hippunfold-site-root", required=True)
    parser.add_argument("--runtime-source-cache", required=True)
    args = parser.parse_args()

    candidates = [
        Path(args.hippunfold_site_root) / "workflow" / "rules" / "coords.smk",
        Path(args.runtime_source_cache)
        / "file"
        / Path(args.hippunfold_site_root).relative_to("/")
        / "workflow"
        / "rules"
        / "coords.smk",
    ]

    patched = []
    skipped = []
    for path in candidates:
        if not path.exists():
            continue
        try:
            if patch_coords_smk(path):
                patched.append(str(path))
        except PermissionError:
            skipped.append(f"{path} (permission denied)")

    if patched:
        print("Patched LAYNII compatibility files:")
        for path in patched:
            print(path)
    if skipped:
        print("Skipped LAYNII compatibility files:")
        for path in skipped:
            print(path)
    if not patched and not skipped:
        print("No LAYNII compatibility patches were needed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
