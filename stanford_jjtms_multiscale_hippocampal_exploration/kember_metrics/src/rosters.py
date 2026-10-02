"""Derive Stanford clinical subgroup rosters from the canonical Master Database."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


SOURCE_ID_FIELD = "CC_Post_Intake_ID"
DIAGNOSIS_FIELD = "Current_PTSD_Diagnosis"
REPO_ROOT = Path(__file__).resolve().parents[3]


@dataclass(frozen=True)
class ClinicalRosterResult:
    """The source join and diagnosis selection needed to reproduce a roster."""

    subject_ids: tuple[str, ...]
    tis_subject_ids: tuple[str, ...]
    excluded_subject_ids: tuple[str, ...]
    missing_subject_ids: tuple[str, ...]
    duplicate_subject_ids: tuple[str, ...]
    source_rows: int
    source_id_field: str
    diagnosis_field: str
    positive_code: str
    diagnosis_counts: dict[str, int]
    master_csv: Path
    tis_manifest: Path
    source_sha256: str
    tis_manifest_sha256: str

    @property
    def tis_n(self) -> int:
        return len(self.tis_subject_ids)

    @property
    def tis_ptsd_n(self) -> int:
        return len(self.subject_ids)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _normalise_subject_id(value: Any) -> str:
    raw = str(value).strip()
    if raw.lower().startswith("sub-"):
        raw = raw[4:]
    if raw.endswith(".0") and raw[:-2].isdigit():
        raw = raw[:-2]
    if not raw or not raw.isdigit():
        raise ValueError(f"invalid subject ID: {value!r}")
    return f"{int(raw):04d}"


def _read_tis_manifest(path: Path) -> tuple[str, ...]:
    subject_ids: list[str] = []
    seen: set[str] = set()
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        parts = stripped.split()
        if len(parts) != 1:
            raise ValueError(f"TIS manifest line {line_number} must contain one subject ID")
        subject_id = _normalise_subject_id(parts[0])
        if subject_id in seen:
            raise ValueError(f"duplicate subject ID in TIS manifest: {subject_id}")
        seen.add(subject_id)
        subject_ids.append(subject_id)
    if not subject_ids:
        raise ValueError(f"TIS manifest is empty: {path}")
    return tuple(sorted(subject_ids))


def _repo_path(path: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return str(resolved)


def derive_tis_ptsd_roster(
    master_csv: Path,
    tis_manifest: Path,
    positive_code: str = "1",
) -> ClinicalRosterResult:
    """Select current-PTSD TIS subjects from the canonical CSV."""

    master_csv = Path(master_csv)
    tis_manifest = Path(tis_manifest)
    tis_subject_ids = _read_tis_manifest(tis_manifest)

    with master_csv.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = reader.fieldnames or []
        required = {SOURCE_ID_FIELD, DIAGNOSIS_FIELD}
        missing_fields = sorted(required.difference(fieldnames))
        if missing_fields:
            raise ValueError(f"Master Database is missing required fields: {missing_fields}")
        rows = list(reader)

    rows_by_subject: dict[str, list[tuple[int, dict[str, str]]]] = {}
    for row_number, row in enumerate(rows, start=2):
        raw_subject_id = (row.get(SOURCE_ID_FIELD) or "").strip()
        if not raw_subject_id:
            continue
        subject_id = _normalise_subject_id(raw_subject_id)
        rows_by_subject.setdefault(subject_id, []).append((row_number, row))

    missing_subject_ids = tuple(subject_id for subject_id in tis_subject_ids if subject_id not in rows_by_subject)
    duplicate_subject_ids = tuple(
        subject_id for subject_id in tis_subject_ids if len(rows_by_subject.get(subject_id, [])) > 1
    )
    if missing_subject_ids or duplicate_subject_ids:
        problems: list[str] = []
        if missing_subject_ids:
            problems.append(f"not found in master CSV: {list(missing_subject_ids)}")
        if duplicate_subject_ids:
            problems.append(f"duplicate source rows: {list(duplicate_subject_ids)}")
        raise ValueError("TIS-to-Master-Database join failed; " + "; ".join(problems))

    diagnosis_counts: dict[str, int] = {}
    selected: list[str] = []
    for subject_id in tis_subject_ids:
        row = rows_by_subject[subject_id][0][1]
        diagnosis = (row.get(DIAGNOSIS_FIELD) or "").strip()
        if diagnosis not in {"0", "1"}:
            raise ValueError(
                f"unexpected {DIAGNOSIS_FIELD} code for {subject_id}: {diagnosis!r}; expected 0 or 1"
            )
        diagnosis_counts[diagnosis] = diagnosis_counts.get(diagnosis, 0) + 1
        if diagnosis == positive_code:
            selected.append(subject_id)

    selected_ids = tuple(sorted(selected))
    excluded_ids = tuple(subject_id for subject_id in tis_subject_ids if subject_id not in selected_ids)
    return ClinicalRosterResult(
        subject_ids=selected_ids,
        tis_subject_ids=tis_subject_ids,
        excluded_subject_ids=excluded_ids,
        missing_subject_ids=missing_subject_ids,
        duplicate_subject_ids=duplicate_subject_ids,
        source_rows=len(rows),
        source_id_field=SOURCE_ID_FIELD,
        diagnosis_field=DIAGNOSIS_FIELD,
        positive_code=positive_code,
        diagnosis_counts=dict(sorted(diagnosis_counts.items())),
        master_csv=master_csv,
        tis_manifest=tis_manifest,
        source_sha256=_sha256(master_csv),
        tis_manifest_sha256=_sha256(tis_manifest),
    )


def _provenance(result: ClinicalRosterResult, manifest_path: Path) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "cohort": "tis_ptsd",
        "source": {
            "path": _repo_path(result.master_csv),
            "sha256": result.source_sha256,
            "data_rows": result.source_rows,
            "id_field": result.source_id_field,
            "diagnosis_field": result.diagnosis_field,
            "positive_code": result.positive_code,
        },
        "eligibility": {
            "tis_manifest": _repo_path(result.tis_manifest),
            "sha256": result.tis_manifest_sha256,
            "tis_n": result.tis_n,
        },
        "result": {
            "tis_n": result.tis_n,
            "tis_ptsd_n": result.tis_ptsd_n,
            "subject_ids": list(result.subject_ids),
            "excluded_tis_subject_ids": list(result.excluded_subject_ids),
            "diagnosis_counts": result.diagnosis_counts,
            "missing_subject_ids": list(result.missing_subject_ids),
            "duplicate_subject_ids": list(result.duplicate_subject_ids),
            "manifest_path": _repo_path(manifest_path),
        },
        "derivation": {
            "join_key": result.source_id_field,
            "rule": f"{result.diagnosis_field} == {result.positive_code!r}",
            "nested_within": "TIS",
            "statement": (
                "tis_ptsd is the intersection of the existing QC-passed TIS roster "
                "with rows having Current_PTSD_Diagnosis == 1 in the canonical Master Database."
            ),
        },
    }


def write_roster_outputs(
    result: ClinicalRosterResult,
    manifest_path: Path,
    provenance_path: Path,
) -> None:
    """Write the numeric roster and deterministic provenance evidence."""

    manifest_path = Path(manifest_path)
    provenance_path = Path(provenance_path)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    provenance_path.parent.mkdir(parents=True, exist_ok=True)

    manifest_path.write_text("".join(f"{subject_id}\n" for subject_id in result.subject_ids), encoding="utf-8")
    provenance_path.write_text(
        json.dumps(_provenance(result, manifest_path), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--master-csv", type=Path, required=True)
    parser.add_argument("--tis-manifest", type=Path, required=True)
    parser.add_argument("--manifest-out", type=Path, required=True)
    parser.add_argument("--provenance-out", type=Path, required=True)
    parser.add_argument("--positive-code", default="1")
    return parser


def main() -> None:
    args = _parser().parse_args()
    result = derive_tis_ptsd_roster(args.master_csv, args.tis_manifest, args.positive_code)
    write_roster_outputs(result, args.manifest_out, args.provenance_out)
    print(
        f"tis_ptsd_n={result.tis_ptsd_n} tis_n={result.tis_n} "
        f"current_ptsd_counts={json.dumps(result.diagnosis_counts, sort_keys=True)}"
    )


if __name__ == "__main__":
    main()
