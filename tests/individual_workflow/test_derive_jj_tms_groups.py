from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[2]


def load_module():
    spec = importlib.util.spec_from_file_location(
        "derive_jj_tms_groups", ROOT / "scripts/jj_tms/derive_jj_tms_groups.py"
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


CURRENT = "Yes (Past, Current)"
NO_MINI = "No (Current diagnosis, but no on MINI)"
# id, MDD, PTSD, GAD, QC, note
ROWS = [
    ("JJ_TMS_1001", "No", "No", "No", "Yes"),
    ("JJ_TMS_1002", "No", "No", "No", "Yes"),
    ("JJ_TMS_1003", "No", "No", "No", None),  # not QC-passed
    ("JJ_TMS_3001", CURRENT, "No", "Yes (Current)", "Yes"),  # comorbid via GAD
    ("JJ_TMS_3002", "Yes (Current, Past)", "Yes (Current)", "No", "Yes"),  # comorbid via PTSD
    ("JJ_TMS_3003", "Yes (Past)", "No", "Yes (Current)", "Yes"),  # past-only: excluded
    ("JJ_TMS_3004", CURRENT, "No", "No", "Yes"),  # pure
    ("JJ_TMS_3005", "Yes (Past, Current, Recurrent)", "No", NO_MINI, "Yes"),  # pure: not MINI-confirmed
    ("JJ_TMS_3006", CURRENT, "No", "Yes (Current)", None),  # not QC-passed
]


def write_workbook(path: Path, rows=ROWS) -> Path:
    header = [
        [None, "Background", None, None, None, "TMS Target", "Data Upload", None, "Data Processing", None, None, None],
        [None, "Gender", "MDD", "PTSD", "GAD", None, "Visit 1", "Visit 2", "Visit 1", "Visit 2", "QC", None],
    ]
    body = [[sid, "F", mdd, ptsd, gad, "LDLPFC", "Yes", "Yes", "Yes", "Yes", qc, None] for sid, mdd, ptsd, gad, qc in rows]
    footer = [[None] * 12, ["Updated by test"] + [None] * 11]
    pd.DataFrame(header + body + footer).to_excel(path, sheet_name="Sheet1", header=False, index=False)
    return path


def test_groups_follow_diagnosis_rules(tmp_path: Path):
    module = load_module()
    subjects = module.read_subjects(write_workbook(tmp_path / "info.xlsx"))
    groups, excluded = module.derive_groups(subjects)

    assert groups["all"] == ["1001", "1002", "3001", "3002", "3003", "3004", "3005"]
    assert groups["hc"] == ["1001", "1002"]
    assert groups["patient"] == ["3001", "3002", "3004", "3005"]
    assert groups["comorbid_mdd"] == ["3001", "3002"]
    assert groups["pure_mdd"] == ["3004", "3005"]
    assert excluded == [("3003", "patient without current MDD")]
    assert set(groups["pure_mdd"]) | set(groups["comorbid_mdd"]) == set(groups["patient"])
    assert not set(groups["pure_mdd"]) & set(groups["comorbid_mdd"])


def test_roster_size_follows_the_workbook(tmp_path: Path):
    module = load_module()
    rows = ROWS + [("JJ_TMS_3007", CURRENT, "No", "Yes (Current)", "Yes")]
    groups, _ = module.derive_groups(module.read_subjects(write_workbook(tmp_path / "info.xlsx", rows)))
    assert groups["comorbid_mdd"] == ["3001", "3002", "3007"]


@pytest.mark.parametrize(
    ("row", "message"),
    [
        (("JJ_TMS_3009", "Maybe", "No", "No", "Yes"), "MDD"),
        (("JJ_TMS_3009", CURRENT, "Yes (Past)", "No", "Yes"), "PTSD"),
        (("JJ_TMS_3009", CURRENT, "No", "Probable", "Yes"), "GAD"),
        (("JJ_TMS_3009", CURRENT, "No", "No", "Maybe"), "QC"),
        (("JJ_TMS_2009", "No", "No", "No", "Yes"), "subject ID"),
        (("JJ_TMS_1001", "No", "No", "No", "Yes"), "duplicate"),
    ],
)
def test_unknown_values_stop_with_the_row_named(tmp_path: Path, row, message):
    module = load_module()
    with pytest.raises(ValueError, match=message):
        module.read_subjects(write_workbook(tmp_path / "info.xlsx", ROWS + [row]))


def test_healthy_control_with_diagnosis_is_rejected(tmp_path: Path):
    module = load_module()
    rows = [("JJ_TMS_1001", CURRENT, "No", "No", "Yes")]
    with pytest.raises(ValueError, match="healthy-control"):
        module.derive_groups(module.read_subjects(write_workbook(tmp_path / "info.xlsx", rows)))


def test_check_mode_reports_drift_and_matches_when_frozen(tmp_path: Path, capsys):
    module = load_module()
    workbook = write_workbook(tmp_path / "info.xlsx")
    frozen = tmp_path / "frozen"
    assert module.main(["--workbook", str(workbook), "--mode", "write", "--out-dir", str(frozen)]) == 0
    assert module.main(["--workbook", str(workbook), "--manifest-dir", str(frozen)]) == 0

    (frozen / "jj_tms_comorbid_mdd_subjects.txt").write_text("sub-3001\n", encoding="utf-8")
    assert module.main(["--workbook", str(workbook), "--manifest-dir", str(frozen)]) == 1
    assert "comorbid_mdd: in workbook only ['3002']" in capsys.readouterr().out


def test_write_mode_never_overwrites_and_rejects_manifest_dir(tmp_path: Path):
    module = load_module()
    workbook = write_workbook(tmp_path / "info.xlsx")
    out_dir = tmp_path / "out"
    assert module.main(["--workbook", str(workbook), "--mode", "write", "--out-dir", str(out_dir)]) == 0
    with pytest.raises(FileExistsError):
        module.main(["--workbook", str(workbook), "--mode", "write", "--out-dir", str(out_dir)])
    with pytest.raises(ValueError, match="manifest directory"):
        module.main(
            ["--workbook", str(workbook), "--manifest-dir", str(out_dir), "--mode", "write", "--out-dir", str(out_dir)]
        )
    text = (out_dir / "jj_tms_pure_mdd_subjects.txt").read_text(encoding="utf-8")
    assert text.splitlines()[1:] == ["3004", "3005"]
