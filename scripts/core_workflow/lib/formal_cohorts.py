from __future__ import annotations

import hashlib
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
    "lynch": CohortContract("lynch", re.compile(r"\d{6}")),
    "stanford_hc": CohortContract("stanford_hc", re.compile(r"[12]\d{3}")),
    "stanford_symp": CohortContract("stanford_symp", re.compile(r"[34]\d{3}")),
    "jjtms_hc": CohortContract("jjtms_hc", re.compile(r"1\d{3}")),
    "jjtms_patient": CohortContract("jjtms_patient", re.compile(r"3\d{3}")),
    "jjtms": CohortContract("jjtms", re.compile(r"[13]\d{3}")),
}
MANIFEST_NAME_CONTRACTS = {
    "hcp_7t_hippocampus_fmri_qc_pass_t1only_latest.txt": "hcp",
    "hcp_7t_lynch2024_pass_latest.txt": "lynch",
    "hcp_7t_lynch2024_nsi043_pass_latest.txt": "lynch",
    "stanford_hc_1xxx2xxx_qc_pass_t1only_latest.txt": "stanford_hc",
    "stanford_symp_3xxx4xxx_qc_pass_t1only_latest.txt": "stanford_symp",
    "jj_tms_hc_subjects.txt": "jjtms_hc",
    "jj_tms_patient_subjects.txt": "jjtms_patient",
    "jj_tms_subjects.txt": "jjtms",
}


def cohort_key_for_group(dataset: str, branch: str) -> str:
    if dataset == "outputs_hcp" and branch.startswith("lynch"):
        return "lynch"
    if dataset == "outputs_hcp":
        return "hcp"
    if dataset == "outputs_stanford" and (branch.startswith(("pfm_hc", "routea_hc")) or branch == "hc"):
        return "stanford_hc"
    if dataset == "outputs_stanford" and (branch.startswith(("pfm_symp", "routea_symp")) or branch == "symp"):
        return "stanford_symp"
    if dataset in ("outputs_iowa_jj_tms", "iowa_jj_tms") and (branch.startswith(("pfm_hc", "routea_hc")) or "hc" in branch):
        return "jjtms_hc"
    if dataset in ("outputs_iowa_jj_tms", "iowa_jj_tms") and (branch.startswith(("pfm_patient", "pfm_symp", "routea_patient", "routea_symp")) or "patient" in branch or "symp" in branch):
        return "jjtms_patient"
    if dataset in ("outputs_iowa_jj_tms", "iowa_jj_tms"):
        return "jjtms"
    raise ValueError(f"No formal cohort contract for dataset={dataset!r} branch={branch!r}")


def cohort_key_for_manifest(path: Path) -> str:
    try:
        return MANIFEST_NAME_CONTRACTS[Path(path).name]
    except KeyError as exc:
        raise ValueError(f"No formal cohort contract for manifest: {path}") from exc


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


def subject_list_sha256(subjects: list[str]) -> str:
    return hashlib.sha256(("\n".join(subjects) + "\n").encode("utf-8")).hexdigest()


def validate_recorded_subjects(
    manifest: dict[str, object], subjects: list[str], *, source: Path
) -> None:
    recorded = manifest.get("subjects")
    if not isinstance(recorded, list) or [str(item) for item in recorded] != subjects:
        raise ValueError(f"Formal manifest subjects do not match cohort manifest: {source}")
    if manifest.get("n_subjects") != len(subjects):
        raise ValueError(
            f"Formal manifest n_subjects={manifest.get('n_subjects')!r} does not match cohort size {len(subjects)}: {source}"
        )
