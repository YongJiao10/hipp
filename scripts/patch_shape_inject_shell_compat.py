#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path


PATCH_MARKER = "HIPPOMAPS_MACOS_COMPAT_SHAPE_INJECT_SHELL"
OLD = (
    '"greedy -threads {threads} {params.general_opts} {params.affine_opts} {params.img_pairs} -o {output.matrix}  &> {log} && "\n'
    '        "greedy -threads {threads} {params.general_opts} {params.greedy_opts} {params.img_pairs} -it {output.matrix} -o {output.warp} &>> {log}"'
)
NEW = (
    '"greedy -threads {threads} {params.general_opts} {params.affine_opts} {params.img_pairs} -o {output.matrix} > {log} 2>&1 && "\n'
    '        "greedy -threads {threads} {params.general_opts} {params.greedy_opts} {params.img_pairs} -it {output.matrix} -o {output.warp} >> {log} 2>&1"\n'
    "        # HIPPOMAPS_MACOS_COMPAT_SHAPE_INJECT_SHELL: use portable redirection syntax matching legacy shell behavior"
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
        description="Patch HippUnfold shape_inject shell redirections to a portable form"
    )
    parser.add_argument("--hippunfold-site-root", required=True)
    parser.add_argument("--runtime-source-cache", required=True)
    args = parser.parse_args()

    site_root = Path(args.hippunfold_site_root)
    runtime_source_cache = Path(args.runtime_source_cache)
    candidates = [
        site_root / "workflow" / "rules" / "shape_inject.smk",
        runtime_source_cache
        / "file"
        / site_root.relative_to("/")
        / "workflow"
        / "rules"
        / "shape_inject.smk",
    ]

    patched = []
    for path in candidates:
        if patch_file(path):
            patched.append(str(path))

    if patched:
        print("Patched shape_inject shell compatibility files:")
        for path in patched:
            print(path)
    else:
        print("No shape_inject shell compatibility patches were needed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
