from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_DIR = REPO_ROOT / "scripts" / "ArealMSHBM_Kong2022_Iowa_Stanford"


def test_iowa_hcp500_input_preparer_selects_hcp_rest_dtseries(
    tmp_path: Path, monkeypatch
) -> None:
    sys.path.insert(0, str(SCRIPT_DIR))
    import prepare_kong2022_areal_mshbm_subject_inputs as preparer

    sessions = tmp_path / "sessions"
    dtseries = (
        sessions
        / "04172026"
        / "hcp"
        / "04172026"
        / "MNINonLinear"
        / "Results"
        / "REST"
        / "REST_Atlas_MSMAll_hp0_clean_rclean.dtseries.nii"
    )
    dtseries.parent.mkdir(parents=True)
    dtseries.write_bytes(b"fixture")
    output_dir = tmp_path / "prepared"

    monkeypatch.setattr(preparer, "dtseries_n_timepoints", lambda _: 3)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "prepare",
            "--dataset",
            "iowa_hcp500",
            "--subject",
            "04172026",
            "--data-root",
            str(sessions),
            "--subject-output-dir",
            str(output_dir),
        ],
    )

    assert preparer.main() == 0
    manifest = json.loads((output_dir / "inputs" / "input_manifest.json").read_text())
    assert manifest["run_count"] == 1
    assert manifest["runs"][0]["dtseries"] == str(dtseries)


def test_iowa_hcp500_launcher_dry_run_prints_one_ui_qsub(tmp_path: Path) -> None:
    env = os.environ.copy()
    env.update(
        {
            "DATASET": "iowa_hcp500",
            "DATA_ROOT": str(tmp_path / "sessions"),
            "OUT_ROOT": str(tmp_path / "labels"),
            "SUBJECT_IDS_CSV": "04172026",
            "DRY_RUN": "1",
        }
    )
    result = subprocess.run(
        [str(SCRIPT_DIR / "submit_kong2022_areal_mshbm.sh")],
        cwd=REPO_ROOT,
        env=env,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert result.stderr.count("qsub -N") == 1
    assert "-q UI" in result.stderr
    assert "-t 1-1" in result.stderr
    manifests = list((tmp_path / "labels" / "_manifests").glob("iowa_hcp500_ui_*.txt"))
    assert len(manifests) == 1
    assert manifests[0].read_text(encoding="utf-8") == "04172026\n"
    assert "dry_run=1" in result.stderr
