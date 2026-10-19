#!/usr/bin/env python3
"""Standardized Reference Paper Ingestion Tool (Pattern A).

Implements self-contained reference packages under `refs/`:
    refs/<citation_key>/
        ├── <citation_key>.pdf
        ├── <citation_key>.md
        ├── <citation_key>.json
        └── images/ (optional)

And maintains a central index in `refs/manifest.tsv`.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import shutil
import sys
from pathlib import Path


def sanitize_key(text: str) -> str:
    """Convert arbitrary author/title strings into a clean citation key."""
    # Replace non-alphanumeric chars with underscore
    clean = re.sub(r"[^\w\s-]", "", text)
    clean = re.sub(r"[\s-]+", "_", clean.strip())
    # Collapse consecutive underscores
    clean = re.sub(r"_+", "_", clean)
    return clean.strip("_")


def infer_citation_key_from_filename(filename: str) -> tuple[str, str, str, str]:
    """Infer (citation_key, author, year, title) from typical filename formats.

    Examples:
    - 'Buckner et al. - 2011 - The organization of the human cerebellum...'
    - 'Yeo_2011_Cerebral_Cortex'
    - 'Tozzi2024_NatureMedicine_SupplementaryInformation'
    """
    stem = Path(filename).stem

    # Pattern: Author et al. - Year - Title
    zotero_match = re.match(r"^([^-]+)\s*-\s*(\d{4})\s*-\s*(.+)$", stem)
    if zotero_match:
        raw_author, year, raw_title = zotero_match.groups()
        author = re.sub(r"\s*(et\s+al\.?|等。?|等)\s*", "", raw_author).strip()
        author = sanitize_key(author)
        # Take first 4-5 meaningful words of title
        title_words = [w for w in re.split(r"[\s_-]+", raw_title) if len(w) > 2][:4]
        title_slug = "_".join(title_words)
        citation_key = f"{author}_{year}_{sanitize_key(title_slug)}"
        return citation_key, author, year, raw_title.strip()

    # Pattern: AuthorYear_Title
    ay_match = re.match(r"^([A-Za-z]+)(\d{4})_(.+)$", stem)
    if ay_match:
        author, year, rest = ay_match.groups()
        citation_key = f"{author}_{year}_{sanitize_key(rest)}"
        return citation_key, author, year, rest.replace("_", " ")

    # Fallback
    citation_key = sanitize_key(stem)
    return citation_key, "Unknown", "Unknown", stem


def convert_pdf(
    pdf_path: Path,
    output_dir: Path,
    extract_images: bool = False,
) -> tuple[Path, Path, Path | None]:
    """Parse PDF with opendataloader_pdf into markdown and json."""
    try:
        import opendataloader_pdf
    except ImportError as err:
        raise RuntimeError(
            "opendataloader_pdf is required. Run with /home/yojiao/.conda/envs/py314/bin/python."
        ) from err

    output_dir.mkdir(parents=True, exist_ok=True)
    images_dir = output_dir / "images" if extract_images else None

    if extract_images:
        images_dir.mkdir(exist_ok=True)

    opendataloader_pdf.convert(
        input_path=str(pdf_path),
        output_dir=str(output_dir),
        format=["markdown", "json"],
        image_output="external" if extract_images else "off",
        image_dir=str(images_dir) if extract_images else None,
        quiet=False,
    )

    stem = pdf_path.stem
    md_file = output_dir / f"{stem}.md"
    json_file = output_dir / f"{stem}.json"

    if not md_file.is_file() or not json_file.is_file():
        raise FileNotFoundError(
            f"Expected outputs not found after conversion: {md_file} or {json_file}"
        )

    return md_file, json_file, images_dir if extract_images else None


def update_manifest(
    refs_dir: Path,
    record: dict[str, str],
) -> None:
    """Append or update a record in refs/manifest.tsv."""
    manifest_path = refs_dir / "manifest.tsv"
    fieldnames = [
        "citation_key",
        "author",
        "year",
        "title",
        "directory",
        "pdf_path",
        "md_path",
        "json_path",
        "has_images",
    ]

    records: dict[str, dict[str, str]] = {}
    if manifest_path.is_file():
        with open(manifest_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f, delimiter="\t")
            for row in reader:
                records[row["citation_key"]] = row

    records[record["citation_key"]] = record

    with open(manifest_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        for key in sorted(records.keys()):
            writer.writerow(records[key])


def sync_manifest(refs_dir: Path) -> int:
    """Scan refs/ for subdirectories matching Pattern A and update manifest.tsv."""
    manifest_path = refs_dir / "manifest.tsv"
    count = 0
    records: dict[str, dict[str, str]] = {}

    for item in sorted(refs_dir.iterdir()):
        if not item.is_dir() or item.name.startswith(".") or item.name.startswith("_"):
            continue
        # Require Pattern A standard: <folder_name>/<folder_name>.(pdf|md)
        pdf_path = item / f"{item.name}.pdf"
        md_path = item / f"{item.name}.md"
        json_path = item / f"{item.name}.json"
        if not (pdf_path.is_file() or md_path.is_file()):
            continue

        images_dir = item / "images"


        stem = item.name
        inferred_key, inferred_author, inferred_year, inferred_title = (
            infer_citation_key_from_filename(stem)
        )
        citation_key = item.name


        # Check existing records first
        existing_rec = records.get(citation_key, {})
        auth = existing_rec.get("author") or ""
        yr = existing_rec.get("year") or ""
        ti = existing_rec.get("title") or ""

        # Check legacy map
        for _old, info in LEGACY_MAP.items():
            if info["key"] == citation_key:
                auth = auth or info["author"]
                yr = yr or info["year"]
                ti = ti or info["title"]
                break

        records[citation_key] = {
            "citation_key": citation_key,
            "author": auth or inferred_author,
            "year": yr or inferred_year,
            "title": ti or inferred_title,
            "directory": str(item.relative_to(refs_dir.parent)),
            "pdf_path": str(pdf_path.relative_to(refs_dir.parent)),
            "md_path": str(md_path.relative_to(refs_dir.parent))
            if md_path.is_file()
            else "",
            "json_path": str(json_path.relative_to(refs_dir.parent))
            if json_path.is_file()
            else "",
            "has_images": "true" if (images_dir.is_dir() and any(images_dir.iterdir())) else "false",
        }
        count += 1


    fieldnames = [
        "citation_key",
        "author",
        "year",
        "title",
        "directory",
        "pdf_path",
        "md_path",
        "json_path",
        "has_images",
    ]

    with open(manifest_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        for key in sorted(records.keys()):
            writer.writerow(records[key])

    print(f"Synced {count} entries to {manifest_path}")
    return count


def ingest_file(
    input_pdf: Path,
    refs_dir: Path,
    citation_key: str | None = None,
    author: str | None = None,
    year: str | None = None,
    title: str | None = None,
    extract_images: bool = False,
    move_source: bool = False,
) -> Path:
    """Ingest a PDF file into refs/<citation_key>/ following Pattern A."""
    input_pdf = input_pdf.resolve()
    if not input_pdf.is_file():
        raise FileNotFoundError(f"Input PDF not found: {input_pdf}")

    inf_key, inf_author, inf_year, inf_title = infer_citation_key_from_filename(
        input_pdf.name
    )
    final_key = citation_key or inf_key
    final_author = author or inf_author
    final_year = year or inf_year
    final_title = title or inf_title

    paper_dir = refs_dir / final_key
    paper_dir.mkdir(parents=True, exist_ok=True)

    target_pdf = paper_dir / f"{final_key}.pdf"

    if input_pdf != target_pdf:
        if move_source:
            shutil.move(str(input_pdf), str(target_pdf))
        else:
            shutil.copy2(str(input_pdf), str(target_pdf))

    print(f"Parsing {target_pdf.name} with opendataloader_pdf (images={'on' if extract_images else 'off'})...")
    md_file, json_file, images_dir = convert_pdf(
        pdf_path=target_pdf,
        output_dir=paper_dir,
        extract_images=extract_images,
    )

    manifest_record = {
        "citation_key": final_key,
        "author": final_author,
        "year": final_year,
        "title": final_title,
        "directory": str(paper_dir.relative_to(refs_dir.parent)),
        "pdf_path": str(target_pdf.relative_to(refs_dir.parent)),
        "md_path": str(md_file.relative_to(refs_dir.parent)),
        "json_path": str(json_file.relative_to(refs_dir.parent)),
        "has_images": "true" if extract_images else "false",
    }
    update_manifest(refs_dir, manifest_record)
    print(f"Ingested {final_key} into {paper_dir}")
    return paper_dir


LEGACY_MAP: dict[str, dict[str, str]] = {
    "Buckner et al. - 2011 - The organization of the human cerebellum estimated by intrinsic functional connectivity": {
        "key": "Buckner_2011_Cerebellum_FC",
        "author": "Buckner",
        "year": "2011",
        "title": "The organization of the human cerebellum estimated by intrinsic functional connectivity",
    },
    "DeKraker et al. - 2025 - HippoMaps multiscale cartography of human hippocampal organization": {
        "key": "DeKraker_2025_HippoMaps",
        "author": "DeKraker",
        "year": "2025",
        "title": "HippoMaps multiscale cartography of human hippocampal organization",
    },
    "Ezama et al. - 2021 - Functional connectivity of the hippocampus and its subfields in resting-state networks": {
        "key": "Ezama_2021_Hippocampus_Subfields_RSFC",
        "author": "Ezama",
        "year": "2021",
        "title": "Functional connectivity of the hippocampus and its subfields in resting-state networks",
    },
    "Kember et al. - 2026 - The hippocampus becomes topographically and functionally specialized along the longitudinal axis wit": {
        "key": "Kember_2026_Hippocampus_Longitudinal_Axis",
        "author": "Kember",
        "year": "2026",
        "title": "The hippocampus becomes topographically and functionally specialized along the longitudinal axis",
    },
    "Kong et al. - 2019 - Spatial Topography of Individual-Specific Cortical Networks Predicts Human Cognition, Personality, a": {
        "key": "Kong_2019_Individual_Cortical_Networks",
        "author": "Kong",
        "year": "2019",
        "title": "Spatial Topography of Individual-Specific Cortical Networks Predicts Human Cognition, Personality, and Emotion",
    },
    "Kong et al. - 2021 - Individual-Specific Areal-Level Parcellations Improve Functional Connectivity Prediction of Behavior": {
        "key": "Kong_2021_Areal_Parcellations_Behavior",
        "author": "Kong",
        "year": "2021",
        "title": "Individual-Specific Areal-Level Parcellations Improve Functional Connectivity Prediction of Behavior",
    },
    "Li et al. - 2026 - Multimodal evidence for hippocampal engagement and modulation by functional connectivity-guided pari": {
        "key": "Li_2026_Hippocampal_Engagement_Modulation",
        "author": "Li",
        "year": "2026",
        "title": "Multimodal evidence for hippocampal engagement and modulation by functional connectivity-guided parietal stimulation",
    },
    "Lv et al. - 2026 - Three parsimonious spatiotemporal patterns in cerebellum reveal individual traits in function and be": {
        "key": "Lv_2026_Cerebellum_Spatiotemporal_Patterns",
        "author": "Lv",
        "year": "2026",
        "title": "Three parsimonious spatiotemporal patterns in cerebellum reveal individual traits in function and behavior",
    },
    "Lynch et al. - 2024 - Frontostriatal salience network expansion in indiv": {
        "key": "Lynch_2024_Frontostriatal_Salience_Expansion",
        "author": "Lynch",
        "year": "2024",
        "title": "Frontostriatal salience network expansion in individuals",
    },
    "Lynch et al. - 2026 - Objective quality assessment for precision functional MRI data": {
        "key": "Lynch_2026_Objective_QA_Precision_fMRI",
        "author": "Lynch",
        "year": "2026",
        "title": "Objective quality assessment for precision functional MRI data",
    },
    "The organization of the human striatum estimated by intrinsic functional connectivity  Journal of N": {
        "key": "Choi_2012_Striatum_FC",
        "author": "Choi",
        "year": "2012",
        "title": "The organization of the human striatum estimated by intrinsic functional connectivity",
    },
    "Thomas Yeo 等。 - 2011 - The organization of the human cerebral cortex esti": {
        "key": "Yeo_2011_Cerebral_Cortex_Organization",
        "author": "Yeo",
        "year": "2011",
        "title": "The organization of the human cerebral cortex estimated by intrinsic functional connectivity",
    },
    "Tozzi2024_NatureMedicine_SupplementaryInformation": {
        "key": "Tozzi_2024_NatureMedicine_Supplement",
        "author": "Tozzi",
        "year": "2024",
        "title": "Personalized brain circuit models identify clinically relevant biotypes in depression and anxiety - Supplementary Information",
    },
}


def migrate_legacy_flat_refs(refs_dir: Path) -> None:
    """Migrate flat files in refs/ into Pattern A self-contained packages."""
    print(f"Beginning legacy migration in {refs_dir}...")
    for old_stem, info in LEGACY_MAP.items():
        key = info["key"]
        target_dir = refs_dir / key
        target_dir.mkdir(parents=True, exist_ok=True)

        old_pdf = refs_dir / f"{old_stem}.pdf"
        old_md = refs_dir / f"{old_stem}.md"
        old_json = refs_dir / f"{old_stem}.json"
        old_img_dir = refs_dir / f"{old_stem}_images"

        new_pdf = target_dir / f"{key}.pdf"
        new_md = target_dir / f"{key}.md"
        new_json = target_dir / f"{key}.json"
        new_img_dir = target_dir / "images"

        if old_pdf.is_file() and not new_pdf.exists():
            shutil.move(str(old_pdf), str(new_pdf))
            print(f"  [PDF]  {old_pdf.name} -> {target_dir.name}/{new_pdf.name}")

        if old_md.is_file() and not new_md.exists():
            shutil.move(str(old_md), str(new_md))
            print(f"  [MD]   {old_md.name} -> {target_dir.name}/{new_md.name}")

        if old_json.is_file() and not new_json.exists():
            shutil.move(str(old_json), str(new_json))
            print(f"  [JSON] {old_json.name} -> {target_dir.name}/{new_json.name}")

        if old_img_dir.is_dir() and not new_img_dir.exists():
            shutil.move(str(old_img_dir), str(new_img_dir))
            print(f"  [IMG]  {old_img_dir.name} -> {target_dir.name}/images")

        record = {
            "citation_key": key,
            "author": info["author"],
            "year": info["year"],
            "title": info["title"],
            "directory": str(target_dir.relative_to(refs_dir.parent)),
            "pdf_path": str(new_pdf.relative_to(refs_dir.parent)) if new_pdf.is_file() else "",
            "md_path": str(new_md.relative_to(refs_dir.parent)) if new_md.is_file() else "",
            "json_path": str(new_json.relative_to(refs_dir.parent)) if new_json.is_file() else "",
            "has_images": "true" if new_img_dir.is_dir() else "false",
        }
        update_manifest(refs_dir, record)
    print("Migration of flat files completed.")



def main() -> None:
    parser = argparse.ArgumentParser(
        description="Standardized Literature Ingestion Tool (Pattern A)"
    )
    parser.add_argument(
        "pdf_path",
        nargs="?",
        type=Path,
        help="Path to input PDF file to ingest",
    )
    parser.add_argument(
        "--key",
        type=str,
        default=None,
        help="Explicit citation key (e.g. Yeo_2011_Cerebral_Cortex_Organization)",
    )
    parser.add_argument("--author", type=str, default=None, help="First author")
    parser.add_argument("--year", type=str, default=None, help="Publication year")
    parser.add_argument("--title", type=str, default=None, help="Paper title")
    parser.add_argument(
        "--extract-images",
        action="store_true",
        default=False,
        help="Extract figures/images into images/ subfolder (default: False)",
    )
    parser.add_argument(
        "--move",
        action="store_true",
        default=False,
        help="Move input PDF instead of copying",
    )
    parser.add_argument(
        "--refs-dir",
        type=Path,
        default=Path("refs"),
        help="Root references directory (default: refs)",
    )
    parser.add_argument(
        "--sync-manifest",
        action="store_true",
        default=False,
        help="Scan refs directory and update refs/manifest.tsv",
    )
    parser.add_argument(
        "--migrate-flat-refs",
        action="store_true",
        default=False,
        help="Migrate existing flat legacy literature in refs/ into Pattern A subfolders",
    )

    args = parser.parse_args()

    refs_dir = args.refs_dir.resolve()
    if args.sync_manifest:
        sync_manifest(refs_dir)
        return

    if args.migrate_flat_refs:
        migrate_legacy_flat_refs(refs_dir)
        sync_manifest(refs_dir)
        return


    if not args.pdf_path:
        parser.print_help()
        sys.exit(1)

    ingest_file(
        input_pdf=args.pdf_path,
        refs_dir=refs_dir,
        citation_key=args.key,
        author=args.author,
        year=args.year,
        title=args.title,
        extract_images=args.extract_images,
        move_source=args.move,
    )


if __name__ == "__main__":
    main()
