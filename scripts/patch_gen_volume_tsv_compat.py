#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path


PATCH_MARKER = "HIPPOMAPS_MACOS_COMPAT_PANDAS_APPEND"
OLD = """# create the output dataframe

df = pd.DataFrame(columns=["subject", "hemi"] + names)


for in_img, hemi in zip(snakemake.input.segs, hemis):
    img_nib = nib.load(in_img)
    img = img_nib.get_fdata()
    zooms = img_nib.header.get_zooms()

    # voxel size in mm^3
    voxel_mm3 = np.prod(zooms)

    new_entry = {
        "subject": "sub-{subject}".format(subject=snakemake.wildcards["subject"]),
        "hemi": hemi,
    }
    for index, name in zip(indices, names):
        # add volume as value, name as key
        new_entry[name] = np.sum(img == index) * voxel_mm3

    # now create a dataframe from it
    df = df.append(new_entry, ignore_index=True)

df.to_csv(snakemake.output.tsv, sep="\\t", index=False)
"""

NEW = """# create the output dataframe
rows = []


for in_img, hemi in zip(snakemake.input.segs, hemis):
    img_nib = nib.load(in_img)
    img = img_nib.get_fdata()
    zooms = img_nib.header.get_zooms()

    # voxel size in mm^3
    voxel_mm3 = np.prod(zooms)

    new_entry = {
        "subject": "sub-{subject}".format(subject=snakemake.wildcards["subject"]),
        "hemi": hemi,
    }
    for index, name in zip(indices, names):
        # add volume as value, name as key
        new_entry[name] = np.sum(img == index) * voxel_mm3

    rows.append(new_entry)

df = pd.DataFrame(rows, columns=["subject", "hemi"] + names)
# HIPPOMAPS_MACOS_COMPAT_PANDAS_APPEND: pandas removed DataFrame.append
df.to_csv(snakemake.output.tsv, sep="\\t", index=False)
"""


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
        description="Patch HippUnfold gen_volume_tsv for pandas append removal"
    )
    parser.add_argument("--hippunfold-site-root", required=True)
    parser.add_argument("--runtime-source-cache", required=True)
    args = parser.parse_args()

    site_root = Path(args.hippunfold_site_root)
    runtime_source_cache = Path(args.runtime_source_cache)
    candidates = [
        site_root / "workflow" / "scripts" / "gen_volume_tsv.py",
        runtime_source_cache
        / "file"
        / site_root.relative_to("/")
        / "workflow"
        / "scripts"
        / "gen_volume_tsv.py",
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
        print("Patched gen_volume_tsv compatibility files:")
        for path in patched:
            print(path)
    if skipped:
        print("Skipped gen_volume_tsv compatibility files:")
        for path in skipped:
            print(path)
    if not patched and not skipped:
        print("No gen_volume_tsv compatibility patches were needed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
