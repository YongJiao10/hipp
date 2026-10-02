#!/usr/bin/env python3
from __future__ import annotations

"""
Build the Lynch2024 all-base-subject manifest and NSI audit table.

This step starts the Lynch2024 FASTANS/PFM comparison branch from the
authoritative base QC-passed subject manifest. Subject identifiers are
normalized with or without the ``sub-`` prefix. The NSI table is retained for
provenance and auditing, but it is not a cohort-selection rule.
"""

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path

from lib.io_utils import load_subject_ids, normalize_subject_id, write_tsv
from lib.output_layout import DEFAULT_LYNCH2024_MANIFEST_ROOT
from lib.workflow_utils import utc_now_iso, write_json


DEFAULT_PASS_NAME = "hcp_7t_lynch2024_pass_151.txt"
DEFAULT_AUDIT_NAME = "hcp_7t_lynch2024_nsi_audit_151.tsv"
DEFAULT_RUN_NAME = "hcp_7t_lynch2024_manifest.json"


@dataclass(frozen=True)
class LynchNSIManifestResult:
    pass_subjects: list[str]
    excluded_subjects: list[str]
    pass_manifest: Path
    audit_tsv: Path
    run_manifest: Path


def _sniff_delimiter(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return ","
    return "\t"


def read_nsi_table(path: Path) -> dict[str, float]:
    delimiter = _sniff_delimiter(path)
    with Path(path).open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f, delimiter=delimiter)
        if reader.fieldnames is None:
            raise RuntimeError(f"NSI table has no header: {path}")
        required = {"subject", "nsi"}
        missing = sorted(required - set(reader.fieldnames))
        if missing:
            raise RuntimeError(f"NSI table missing required columns {missing}: {path}")
        out: dict[str, float] = {}
        for row in reader:
            subject = normalize_subject_id(str(row.get("subject", "")))
            if not subject:
                continue
            try:
                nsi = float(str(row.get("nsi", "")).strip())
            except ValueError as exc:
                raise ValueError(f"Invalid NSI value for sub-{subject}: {row.get('nsi')!r}") from exc
            out[subject] = nsi
    if not out:
        raise RuntimeError(f"NSI table had no usable rows: {path}")
    return out


def build_lynch2024_nsi_manifest(
    *,
    base_subjects_manifest: Path,
    nsi_tsv: Path,
    out_dir: Path,
) -> LynchNSIManifestResult:
    base_path = Path(base_subjects_manifest).resolve()
    nsi_path = Path(nsi_tsv).resolve()
    out_root = Path(out_dir).resolve()
    subjects = load_subject_ids(base_path)
    nsi_by_subject = read_nsi_table(nsi_path)

    pass_subjects: list[str] = list(subjects)
    audit_rows: list[dict[str, object]] = []
    for subject in subjects:
        nsi = nsi_by_subject.get(subject)
        if nsi is None:
            audit_rows.append(
                {
                    "subject": f"sub-{subject}",
                    "nsi": "",
                    "status": "missing_nsi",
                }
            )
            continue
        audit_rows.append(
            {
                "subject": f"sub-{subject}",
                "nsi": f"{float(nsi):.12g}",
                "status": "included",
            }
        )

    out_root.mkdir(parents=True, exist_ok=True)
    pass_manifest = out_root / DEFAULT_PASS_NAME
    audit_tsv = out_root / DEFAULT_AUDIT_NAME
    run_manifest = out_root / DEFAULT_RUN_NAME

    pass_manifest.write_text(
        "".join(f"sub-{subject}\n" for subject in pass_subjects),
        encoding="utf-8",
    )
    write_tsv(
        audit_tsv,
        audit_rows,
        ["subject", "nsi", "status"],
    )
    manifest = {
        "stage": "02_03_build_lynch2024_nsi_manifest",
        "status": "ok",
        "timestamp_utc": utc_now_iso(),
        "base_subjects_manifest": str(base_path),
        "nsi_tsv": str(nsi_path),
        "selection_rule": "all_base_subjects",
        "n_base_subjects": int(len(subjects)),
        "n_pass": int(len(pass_subjects)),
        "n_excluded": 0,
        "n_missing_nsi": int(sum(row["status"] == "missing_nsi" for row in audit_rows)),
        "pass_subjects": [f"sub-{subject}" for subject in pass_subjects],
        "outputs": {
            "pass_manifest": str(pass_manifest),
            "audit_tsv": str(audit_tsv),
            "run_manifest": str(run_manifest),
        },
    }
    write_json(run_manifest, manifest, sort_keys=True, trailing_newline=True)
    return LynchNSIManifestResult(
        pass_subjects=pass_subjects,
        excluded_subjects=[],
        pass_manifest=pass_manifest,
        audit_tsv=audit_tsv,
        run_manifest=run_manifest,
    )


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build Lynch2024 all-base-subject manifest and NSI audit.")
    parser.add_argument("--base-subjects-manifest", required=True)
    parser.add_argument("--nsi-tsv", required=True, help="TSV/CSV with required columns: subject,nsi")
    parser.add_argument("--out-dir", default=str(DEFAULT_LYNCH2024_MANIFEST_ROOT))
    return parser


def main() -> int:
    args = build_arg_parser().parse_args()
    result = build_lynch2024_nsi_manifest(
        base_subjects_manifest=Path(args.base_subjects_manifest),
        nsi_tsv=Path(args.nsi_tsv),
        out_dir=Path(args.out_dir),
    )
    print(
        json.dumps(
            {
                "status": "ok",
                "n_pass": len(result.pass_subjects),
                "pass_manifest": str(result.pass_manifest),
                "audit_tsv": str(result.audit_tsv),
                "run_manifest": str(result.run_manifest),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
