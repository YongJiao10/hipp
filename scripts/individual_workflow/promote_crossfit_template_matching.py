#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import re
import shutil
import uuid
from datetime import datetime, timezone
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RERUN_ROOT = REPO_ROOT / "outputs_hcp" / "reruns" / "shifted_cosine_crossfit_20260711"


def _load_json(path: Path) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return payload


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _replace_strings(value: object, replacements: tuple[tuple[str, str], ...]) -> object:
    if isinstance(value, str):
        out = value
        for old, new in replacements:
            out = out.replace(old, new)
        return out
    if isinstance(value, list):
        return [_replace_strings(item, replacements) for item in value]
    if isinstance(value, dict):
        return {key: _replace_strings(item, replacements) for key, item in value.items()}
    return value


TEXT_SUFFIXES = {".json", ".tsv", ".csv", ".txt", ".md"}
EXPECTED_ROBUSTNESS_THRESHOLDS = {
    "max_affinity_graph_components": 1,
    "min_cluster_fraction": 0.1,
    "min_pairwise_ari": 0.7,
    "min_pairwise_macro_dice": 0.85,
    "min_pairwise_prototype_correlation": 0.5,
}


def _rewrite_text_tree(root: Path, replacements: tuple[tuple[str, str], ...]) -> int:
    count = 0
    for path in sorted(item for item in root.rglob("*") if item.is_file() and item.suffix.lower() in TEXT_SUFFIXES):
        text = path.read_text(encoding="utf-8")
        rewritten = text
        for old, new in replacements:
            rewritten = rewritten.replace(old, new)
        if path.suffix.lower() == ".json":
            payload = json.loads(rewritten)
            rewritten = json.dumps(payload, indent=2, sort_keys=True) + "\n"
        path.write_text(rewritten, encoding="utf-8")
        count += 1
    return count


def _scan_text_contract(root: Path, *, forbidden_roots: tuple[str, ...] = ()) -> None:
    family_pattern = re.compile(r"family[\s_-]*id", re.IGNORECASE)
    for path in sorted(item for item in root.rglob("*") if item.is_file() and item.suffix.lower() in TEXT_SUFFIXES):
        text = path.read_text(encoding="utf-8", errors="replace")
        if family_pattern.search(text):
            raise ValueError(f"Raw family identifier field leaked into cross-fit output: {path}")
        for forbidden in forbidden_roots:
            if forbidden and forbidden in text:
                raise ValueError(f"Source-root reference remains after promotion: {path}")


def _validate_robustness_summary(path: Path) -> int:
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    expected_combos = {("2mm", "L"), ("2mm", "R"), ("4mm", "L"), ("4mm", "R")}
    if {(row.get("smoothing"), row.get("hemi")) for row in rows} != expected_combos:
        raise ValueError(f"Robustness summary must contain exactly four 2mm/4mm L/R rows: {path}")
    for row in rows:
        for key, value in row.items():
            if key in {"smoothing", "hemi"} or value in {None, ""}:
                continue
            try:
                number = float(value)
            except ValueError:
                continue
            if not math.isfinite(number):
                raise ValueError(f"Non-finite robustness metric {key}={value!r}: {path}")
    return len(rows)


def validate_source(
    source_root: Path,
    source_present: Path,
    *,
    expected_template_fits: int,
    expected_mappings: int,
    expected_lineage_root: Path | None = None,
) -> dict[str, object]:
    manifest_path = source_root / "crossfit_manifest.json"
    manifest = _load_json(manifest_path)
    expected = {
        "status": "ok",
        "n_subjects": 151,
        "n_folds": 5,
        "template_fits_completed": int(expected_template_fits),
        "subject_combo_mappings_completed": int(expected_mappings),
        "surface_affinity": "shifted-cosine",
    }
    for key, value in expected.items():
        if manifest.get(key) != value:
            label = "mapping" if key == "subject_combo_mappings_completed" else key
            raise ValueError(f"Cross-fit {label} mismatch: expected={value!r} actual={manifest.get(key)!r}")
    template_count = len(list((source_root / "templates").rglob("template_manifest.json")))
    mapping_count = len(list((source_root / "subjects").rglob("combo_manifest.json")))
    if template_count != int(expected_template_fits):
        raise ValueError(f"Template manifest count mismatch: {template_count}")
    if mapping_count != int(expected_mappings):
        raise ValueError(f"Mapping manifest count mismatch: {mapping_count}")
    _scan_text_contract(source_root)
    _scan_text_contract(source_present)
    robustness_rows = _validate_robustness_summary(source_present / "robustness_summary.tsv")
    robustness_manifest = _load_json(source_present / "robustness_manifest.json")
    if robustness_manifest.get("status") != "ok":
        raise ValueError("Robustness manifest status is not ok")
    lineage_root = source_root if expected_lineage_root is None else expected_lineage_root
    if Path(str(robustness_manifest.get("crossfit_root", ""))).resolve() != Path(lineage_root).resolve():
        raise ValueError("Robustness manifest crossfit_root does not match promoted source")
    if robustness_manifest.get("n_folds") != 5:
        raise ValueError("Robustness manifest must contain five folds")
    gate = robustness_manifest.get("robustness_gate")
    if not isinstance(gate, dict) or gate.get("passed") is not True or gate.get("failures") != []:
        raise ValueError("Robustness gate did not pass cleanly")
    if gate.get("thresholds") != EXPECTED_ROBUSTNESS_THRESHOLDS:
        raise ValueError("Robustness thresholds differ from the approved release gate")
    combinations = robustness_manifest.get("combinations")
    if not isinstance(combinations, list) or len(combinations) != 4:
        raise ValueError("Robustness manifest must contain four smoothing/hemisphere combinations")
    for combination in combinations:
        if not isinstance(combination, dict):
            raise ValueError("Invalid robustness combination")
        matrix = combination.get("pairwise_adjusted_rand_index")
        if not isinstance(matrix, list) or len(matrix) != 5 or any(not isinstance(row, list) or len(row) != 5 for row in matrix):
            raise ValueError("Each robustness combination must contain a 5-fold pairwise ARI matrix")
    overview = source_present / "hcp_routea_crossfit_k3_robustness_overview.png"
    if not overview.is_file() or overview.stat().st_size <= 0:
        raise ValueError(f"Missing robustness overview: {overview}")
    return {
        "template_manifest_count": template_count,
        "mapping_manifest_count": mapping_count,
        "robustness_rows": robustness_rows,
    }


def promote_crossfit(
    *,
    repo_root: Path,
    source_root: Path,
    source_present: Path,
    destination_root: Path,
    destination_present: Path,
    expected_template_fits: int = 20,
    expected_mappings: int = 604,
) -> dict[str, object]:
    root = Path(repo_root).resolve()
    paths = [Path(path).resolve() for path in (source_root, source_present, destination_root, destination_present)]
    for path in paths:
        try:
            path.relative_to(root)
        except ValueError as exc:
            raise ValueError(f"Promotion path must resolve within repository: {path}") from exc
    src, src_present, dest, dest_present = paths
    if dest.exists() or dest_present.exists():
        raise FileExistsError(f"Canonical promotion destination already exists: {dest} or {dest_present}")
    evidence = validate_source(
        src,
        src_present,
        expected_template_fits=expected_template_fits,
        expected_mappings=expected_mappings,
    )
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest_present.parent.mkdir(parents=True, exist_ok=True)
    nonce = uuid.uuid4().hex
    staged = dest.parent / f".{dest.name}.promotion-staging-{nonce}"
    staged_present = dest_present.parent / f".{dest_present.name}.promotion-staging-{nonce}"
    shutil.copytree(src, staged)
    shutil.copytree(src_present, staged_present)
    replacements = (
        (str(src), str(dest)),
        (str(src_present), str(dest_present)),
    )
    rewritten = _rewrite_text_tree(staged, replacements) + _rewrite_text_tree(staged_present, replacements)
    _scan_text_contract(staged, forbidden_roots=(str(src), str(src_present)))
    _scan_text_contract(staged_present, forbidden_roots=(str(src), str(src_present)))
    # Both complete trees are validated before either canonical destination is exposed.
    validate_source(
        staged,
        staged_present,
        expected_template_fits=expected_template_fits,
        expected_mappings=expected_mappings,
        expected_lineage_root=dest,
    )
    staged.rename(dest)
    staged_present.rename(dest_present)
    promotion_manifest: dict[str, object] = {
        "status": "ok",
        "promoted_utc": datetime.now(timezone.utc).isoformat(),
        "source_root": str(src),
        "source_present": str(src_present),
        "destination_root": str(dest),
        "destination_present": str(dest_present),
        "surface_affinity": "shifted-cosine",
        "text_files_rewritten": rewritten,
        **evidence,
    }
    _write_json(dest / "promotion_manifest.json", promotion_manifest)
    validate_source(
        dest,
        dest_present,
        expected_template_fits=expected_template_fits,
        expected_mappings=expected_mappings,
    )
    return promotion_manifest


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    parser.add_argument("--source-root", type=Path, default=DEFAULT_RERUN_ROOT / "crossfit_template_matching")
    parser.add_argument(
        "--source-present",
        type=Path,
        default=DEFAULT_RERUN_ROOT / "present" / "crossfit_template_matching",
    )
    parser.add_argument(
        "--destination-root",
        type=Path,
        default=REPO_ROOT / "outputs_hcp" / "work" / "crossfit_template_matching",
    )
    parser.add_argument(
        "--destination-present",
        type=Path,
        default=REPO_ROOT / "outputs_hcp" / "present" / "overviews" / "crossfit_template_matching",
    )
    return parser


def main() -> int:
    args = build_arg_parser().parse_args()
    manifest = promote_crossfit(
        repo_root=args.repo_root,
        source_root=args.source_root,
        source_present=args.source_present,
        destination_root=args.destination_root,
        destination_present=args.destination_present,
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
