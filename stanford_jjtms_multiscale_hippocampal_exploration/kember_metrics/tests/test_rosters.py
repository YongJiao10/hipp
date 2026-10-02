"""Tests for source-derived Stanford clinical subgroup rosters."""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import pytest


SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))

from rosters import derive_tis_ptsd_roster, write_roster_outputs  # noqa: E402


REPO_ROOT = Path(__file__).resolve().parents[3]
MASTER_CSV = REPO_ROOT / "data" / "CC_RC_Master_Database_10.23.18_Data.csv"
TIS_MANIFEST = (
    REPO_ROOT
    / "outputs_stanford"
    / "inputs"
    / "stanford"
    / "kong2022"
    / "manifests"
    / "stanford_tis_4xxx_qc_pass_t1only_47.txt"
)


def _write_master(path: Path, rows: list[dict[str, str]]) -> None:
    fieldnames = ["CC_Post_Intake_ID", "Current_PTSD_Diagnosis", "CC_Post_Intake_Grouping"]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _write_manifest(path: Path, ids: list[str]) -> None:
    path.write_text("\n".join(ids) + "\n", encoding="utf-8")


def test_derives_current_ptsd_only_inside_tis(tmp_path: Path) -> None:
    master = tmp_path / "master.csv"
    tis = tmp_path / "tis.txt"
    _write_master(
        master,
        [
            {"CC_Post_Intake_ID": "0001", "Current_PTSD_Diagnosis": "1", "CC_Post_Intake_Grouping": "4"},
            {"CC_Post_Intake_ID": "0002", "Current_PTSD_Diagnosis": "0", "CC_Post_Intake_Grouping": "4"},
            {"CC_Post_Intake_ID": "0003", "Current_PTSD_Diagnosis": "1", "CC_Post_Intake_Grouping": "4"},
            {"CC_Post_Intake_ID": "3001", "Current_PTSD_Diagnosis": "1", "CC_Post_Intake_Grouping": "3"},
        ],
    )
    _write_manifest(tis, ["0001", "0002", "0003"])

    result = derive_tis_ptsd_roster(master, tis)

    assert result.subject_ids == ("0001", "0003")
    assert result.tis_subject_ids == ("0001", "0002", "0003")
    assert result.excluded_subject_ids == ("0002",)
    assert result.tis_n == 3
    assert result.source_rows == 4
    assert result.source_id_field == "CC_Post_Intake_ID"
    assert result.diagnosis_field == "Current_PTSD_Diagnosis"
    assert result.diagnosis_counts == {"0": 1, "1": 2}


def test_rejects_duplicate_tis_post_ids(tmp_path: Path) -> None:
    master = tmp_path / "master.csv"
    tis = tmp_path / "tis.txt"
    _write_master(
        master,
        [
            {"CC_Post_Intake_ID": "0001", "Current_PTSD_Diagnosis": "1", "CC_Post_Intake_Grouping": "4"},
            {"CC_Post_Intake_ID": "0001", "Current_PTSD_Diagnosis": "0", "CC_Post_Intake_Grouping": "4"},
        ],
    )
    _write_manifest(tis, ["0001"])

    with pytest.raises(ValueError, match="duplicate"):
        derive_tis_ptsd_roster(master, tis)


def test_rejects_missing_tis_post_id(tmp_path: Path) -> None:
    master = tmp_path / "master.csv"
    tis = tmp_path / "tis.txt"
    _write_master(
        master,
        [{"CC_Post_Intake_ID": "0001", "Current_PTSD_Diagnosis": "1", "CC_Post_Intake_Grouping": "4"}],
    )
    _write_manifest(tis, ["0001", "0002"])

    with pytest.raises(ValueError, match="not found"):
        derive_tis_ptsd_roster(master, tis)


def test_repository_tis_ptsd_derivation_has_35_members() -> None:
    result = derive_tis_ptsd_roster(MASTER_CSV, TIS_MANIFEST)

    assert result.tis_n == 47
    assert len(result.subject_ids) == 35
    assert result.source_rows == 332
    assert result.source_id_field == "CC_Post_Intake_ID"
    assert result.diagnosis_field == "Current_PTSD_Diagnosis"
    assert result.diagnosis_counts == {"0": 12, "1": 35}
    assert not result.missing_subject_ids
    assert not result.duplicate_subject_ids


def test_writes_deterministic_manifest_and_provenance(tmp_path: Path) -> None:
    result = derive_tis_ptsd_roster(MASTER_CSV, TIS_MANIFEST)
    manifest = tmp_path / "tis_ptsd.txt"
    provenance = tmp_path / "tis_ptsd_provenance.json"

    write_roster_outputs(result, manifest, provenance)
    first_manifest = manifest.read_bytes()
    first_provenance = provenance.read_bytes()
    write_roster_outputs(result, manifest, provenance)

    assert manifest.read_bytes() == first_manifest
    assert provenance.read_bytes() == first_provenance
    evidence = json.loads(provenance.read_text(encoding="utf-8"))
    assert evidence["result"]["tis_n"] == 47
    assert evidence["result"]["tis_ptsd_n"] == 35
    assert evidence["source"]["diagnosis_field"] == "Current_PTSD_Diagnosis"
    assert evidence["derivation"]["nested_within"] == "TIS"
