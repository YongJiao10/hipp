from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class CohortContract:
    key: str
    expected_size: int
    subject_pattern: re.Pattern[str]


CONTRACTS = {
    "hcp151": CohortContract("hcp151", 151, re.compile(r"sub-\d{6}")),
    "lynch78": CohortContract("lynch78", 78, re.compile(r"sub-\d{6}")),
    "stanford_hc91": CohortContract("stanford_hc91", 91, re.compile(r"sub-[12]\d{3}")),
    "stanford_symp103": CohortContract("stanford_symp103", 103, re.compile(r"sub-[34]\d{3}")),
}
MANIFEST_NAME_CONTRACTS = {
    "hcp_7t_hippocampus_fmri_qc_pass_t1only_151.txt": "hcp151",
    "hcp_7t_hippocampus_fmri_qc_pass_t1only_latest.txt": "hcp151",
    "hcp_7t_lynch2024_nsi043_pass_78.txt": "lynch78",
    "stanford_hc_1xxx2xxx_qc_pass_t1only_91.txt": "stanford_hc91",
    "stanford_symp_3xxx4xxx_qc_pass_t1only_103.txt": "stanford_symp103",
}


def cohort_key_for_group(dataset: str, branch: str) -> str:
    if dataset == "outputs_hcp" and branch.startswith("lynch"):
        return "lynch78"
    if dataset == "outputs_hcp":
        return "hcp151"
    if dataset == "outputs_stanford" and branch.startswith("routea_hc"):
        return "stanford_hc91"
    if dataset == "outputs_stanford" and branch.startswith("routea_symp"):
        return "stanford_symp103"
    raise ValueError(f"No formal cohort contract for dataset={dataset!r} branch={branch!r}")


def cohort_key_for_manifest(path: Path) -> str:
    try:
        return MANIFEST_NAME_CONTRACTS[Path(path).name]
    except KeyError as exc:
        raise ValueError(f"No formal cohort contract for manifest: {path}") from exc


def validate_subject_lines(lines: list[str], cohort_key: str, *, source: object) -> list[str]:
    contract = CONTRACTS[cohort_key]
    subjects = [str(line).strip() for line in lines if str(line).strip()]
    malformed = [subject for subject in subjects if contract.subject_pattern.fullmatch(subject) is None]
    if malformed:
        raise ValueError(f"Formal cohort {source} has non-canonical subject syntax: {malformed[:3]}")
    if len(set(subjects)) != len(subjects):
        raise ValueError(f"Formal cohort {source} contains duplicate subjects")
    if len(subjects) != contract.expected_size:
        raise ValueError(
            f"Formal cohort {source} must contain exactly {contract.expected_size} subjects, got {len(subjects)}"
        )
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
