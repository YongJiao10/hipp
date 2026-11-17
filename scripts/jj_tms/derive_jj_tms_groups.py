#!/usr/bin/env python3
"""Derive JJ_TMS cohort rosters from the subject workbook and compare them with the frozen manifests.

Rules (workbook ``Sheet1``; columns A id, C MDD, D PTSD, E GAD, K QC):

* A subject is in the imaging roster iff ``QC == "Yes"``.
* ``JJ_TMS_1xxx`` are healthy controls (``hc``); ``JJ_TMS_3xxx`` are patients.
* A patient joins the ``patient`` roster iff the MDD cell lists ``Current``.
  Other QC-passed patients (for example past-only MDD) are reported as excluded.
* GAD counts as confirmed only for ``Yes (Current)``; the cell
  ``No (Current diagnosis, but no on MINI)`` is clinical anxiety without MINI
  confirmation and is *not* GAD. PTSD counts only for ``Yes (Current)``.
* ``pure_mdd``: current MDD, no PTSD, GAD not confirmed.
  ``comorbid_mdd``: current MDD with PTSD and/or confirmed GAD.

Roster sizes are results of these rules, never inputs. Unknown cell values stop
the run with the row and value named; nothing is guessed.

``--mode check`` compares the derived rosters with the frozen manifests and exits
1 on any difference. ``--mode write`` writes the derived rosters into a new
directory and never overwrites an existing file.
"""
from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MANIFEST_DIR = REPO_ROOT / "manifests"
SHEET = "Sheet1"
COL_ID, COL_MDD, COL_PTSD, COL_GAD, COL_QC = 0, 2, 3, 4, 10
MDD_TOKENS = {"Past", "Current", "Recurrent"}
PTSD_VALUES = {"No": False, "Yes (Current)": True}
GAD_VALUES = {
    "No": False,
    "Yes (Current)": True,
    "No (Current diagnosis, but no on MINI)": False,
}
GROUPS = ("all", "hc", "patient", "pure_mdd", "comorbid_mdd")
MANIFEST_NAMES = {
    "all": "jj_tms_subjects.txt",
    "hc": "jj_tms_hc_subjects.txt",
    "patient": "jj_tms_patient_subjects.txt",
    "pure_mdd": "jj_tms_pure_mdd_subjects.txt",
    "comorbid_mdd": "jj_tms_comorbid_mdd_subjects.txt",
}
GROUP_DESCRIPTIONS = {
    "all": "QC-passed imaging roster",
    "hc": "healthy controls",
    "patient": "patients with current MDD",
    "pure_mdd": "current MDD without PTSD or MINI-confirmed GAD",
    "comorbid_mdd": "current MDD with PTSD and/or MINI-confirmed GAD",
}


@dataclass(frozen=True)
class Subject:
    subject: str
    current_mdd: bool
    ptsd: bool
    gad_confirmed: bool


def _cell(value: object) -> str | None:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    return str(value).strip()


def parse_current_mdd(value: str | None, *, subject: str) -> bool:
    if value == "No":
        return False
    match = re.fullmatch(r"Yes \(([^)]*)\)", value or "")
    if match is None:
        raise ValueError(f"{subject}: unrecognised MDD value {value!r}")
    tokens = {token.strip() for token in match.group(1).split(",")}
    if not tokens <= MDD_TOKENS:
        raise ValueError(f"{subject}: unrecognised MDD value {value!r}")
    return "Current" in tokens


def _parse_choice(value: str | None, allowed: dict[str, bool], *, subject: str, column: str) -> bool:
    if value not in allowed:
        raise ValueError(f"{subject}: unrecognised {column} value {value!r}")
    return allowed[value]


def read_subjects(workbook: Path) -> list[Subject]:
    """Return the QC-passed subjects with their parsed diagnosis flags."""
    frame = pd.read_excel(workbook, sheet_name=SHEET, header=None)
    subjects: list[Subject] = []
    seen: set[str] = set()
    for _, row in frame.iterrows():
        raw_id = _cell(row[COL_ID])
        if raw_id is None or not raw_id.startswith("JJ_TMS_"):
            continue
        subject = raw_id.removeprefix("JJ_TMS_")
        if not re.fullmatch(r"[13]\d{3}", subject):
            raise ValueError(f"{raw_id}: subject ID must be 1xxx (HC) or 3xxx (patient)")
        if subject in seen:
            raise ValueError(f"{raw_id}: duplicate workbook row")
        seen.add(subject)
        qc = _cell(row[COL_QC])
        if qc not in (None, "Yes"):
            raise ValueError(f"{raw_id}: unrecognised QC value {qc!r}")
        if qc != "Yes":
            continue
        subjects.append(
            Subject(
                subject=subject,
                current_mdd=parse_current_mdd(_cell(row[COL_MDD]), subject=raw_id),
                ptsd=_parse_choice(_cell(row[COL_PTSD]), PTSD_VALUES, subject=raw_id, column="PTSD"),
                gad_confirmed=_parse_choice(_cell(row[COL_GAD]), GAD_VALUES, subject=raw_id, column="GAD"),
            )
        )
    if not subjects:
        raise ValueError(f"No QC == 'Yes' subjects in {workbook}")
    return subjects


def derive_groups(subjects: list[Subject]) -> tuple[dict[str, list[str]], list[tuple[str, str]]]:
    groups: dict[str, list[str]] = {name: [] for name in GROUPS}
    excluded: list[tuple[str, str]] = []
    for item in sorted(subjects, key=lambda s: int(s.subject)):
        groups["all"].append(item.subject)
        if item.subject.startswith("1"):
            if item.current_mdd or item.ptsd or item.gad_confirmed:
                raise ValueError(f"JJ_TMS_{item.subject}: healthy-control ID with a current diagnosis")
            groups["hc"].append(item.subject)
            continue
        if not item.current_mdd:
            excluded.append((item.subject, "patient without current MDD"))
            continue
        groups["patient"].append(item.subject)
        groups["comorbid_mdd" if (item.ptsd or item.gad_confirmed) else "pure_mdd"].append(item.subject)
    return groups, excluded


def read_manifest(path: Path) -> list[str]:
    if not path.is_file():
        raise FileNotFoundError(f"Missing frozen manifest: {path}")
    values = [
        line.strip().removeprefix("sub-")
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    if len(set(values)) != len(values):
        raise ValueError(f"Duplicate subject IDs in {path}")
    return values


def compare(groups: dict[str, list[str]], manifest_dir: Path) -> dict[str, tuple[list[str], list[str]]]:
    """Return per-group (in workbook only, in manifest only)."""
    differences: dict[str, tuple[list[str], list[str]]] = {}
    for name in GROUPS:
        frozen = set(read_manifest(manifest_dir / MANIFEST_NAMES[name]))
        derived = set(groups[name])
        added = sorted(derived - frozen, key=int)
        removed = sorted(frozen - derived, key=int)
        if added or removed:
            differences[name] = (added, removed)
    return differences


def write_rosters(groups: dict[str, list[str]], out_dir: Path, manifest_dir: Path) -> list[Path]:
    out_dir = Path(out_dir).resolve()
    if out_dir == Path(manifest_dir).resolve():
        raise ValueError("--out-dir must not be the frozen manifest directory")
    targets = {name: out_dir / MANIFEST_NAMES[name] for name in GROUPS}
    existing = [str(path) for path in targets.values() if path.exists()]
    if existing:
        raise FileExistsError("Refusing to overwrite: " + ", ".join(existing))
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, path in targets.items():
        header = (
            f"# JJ_TMS {GROUP_DESCRIPTIONS[name]}; derived by scripts/jj_tms/derive_jj_tms_groups.py "
            f'from subject_infomation.xlsx {SHEET} (QC == "Yes")\n'
        )
        path.write_text(header + "".join(f"{subject}\n" for subject in groups[name]), encoding="utf-8")
    return list(targets.values())


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--workbook", type=Path, required=True, help="subject_infomation.xlsx")
    parser.add_argument("--manifest-dir", type=Path, default=DEFAULT_MANIFEST_DIR)
    parser.add_argument("--mode", choices=("check", "write"), default="check")
    parser.add_argument("--out-dir", type=Path, help="new directory for --mode write")
    args = parser.parse_args(argv)

    groups, excluded = derive_groups(read_subjects(args.workbook))
    for name in GROUPS:
        print(f"{name}: {len(groups[name])}")
    for subject, reason in excluded:
        print(f"excluded JJ_TMS_{subject}: {reason}")

    if args.mode == "write":
        if args.out_dir is None:
            parser.error("--mode write requires --out-dir")
        for path in write_rosters(groups, args.out_dir, args.manifest_dir):
            print(f"wrote {path}")
        return 0

    differences = compare(groups, args.manifest_dir)
    if not differences:
        print("frozen manifests match the workbook")
        return 0
    for name, (added, removed) in differences.items():
        print(f"{name}: in workbook only {added}; in frozen manifest only {removed}")
    return 1


if __name__ == "__main__":
    sys.exit(main())
