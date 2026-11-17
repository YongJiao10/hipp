from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class CohortContract:
    """Subject-ID syntax of a cohort; roster size is never part of a contract."""

    key: str
    bare_id_pattern: re.Pattern[str]


CONTRACTS = {
    "hcp": CohortContract("hcp", re.compile(r"\d{6}")),
    "jjtms_patient": CohortContract("jjtms_patient", re.compile(r"3\d{3}")),
}


def validate_subject_lines(lines: list[str], cohort_key: str, *, source: object) -> list[str]:
    contract = CONTRACTS[cohort_key]
    raw_subjects = [
        str(line).strip()
        for line in lines
        if str(line).strip() and not str(line).strip().startswith("#")
    ]
    malformed = [
        subject
        for subject in raw_subjects
        if contract.bare_id_pattern.fullmatch(subject.removeprefix("sub-")) is None
    ]
    if malformed:
        raise ValueError(f"Formal cohort {source} has invalid subject syntax: {malformed[:3]}")
    subjects = [f"sub-{subject.removeprefix('sub-')}" for subject in raw_subjects]
    if len(set(subjects)) != len(subjects):
        raise ValueError(f"Formal cohort {source} contains duplicate subjects")
    if not subjects:
        raise ValueError(f"Formal cohort {source} contains no subjects")
    return subjects


def load_formal_subjects(path: Path, cohort_key: str) -> list[str]:
    manifest = Path(path).resolve()
    if not manifest.is_file():
        raise FileNotFoundError(f"Missing formal cohort manifest: {manifest}")
    return validate_subject_lines(
        manifest.read_text(encoding="utf-8").splitlines(), cohort_key, source=manifest
    )
