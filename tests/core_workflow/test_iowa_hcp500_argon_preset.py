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


def test_iowa7t_input_preparer_selects_branch_exact_dtseries(
    tmp_path: Path, monkeypatch
) -> None:
    sys.path.insert(0, str(SCRIPT_DIR))
    import prepare_kong2022_areal_mshbm_subject_inputs as preparer

    data_root = tmp_path / "DataOrg_PFM_36P"
    func = data_root / "sub-06112026" / "func"
    func.mkdir(parents=True)
    single = func / "sub-06112026_task-rest_dir-concat_space-fsLR_den-91k_desc-denoised_bold.dtseries.nii"
    multi = func / "sub-06112026_task-rest_run-concat_space-fsLR_den-32k_desc-MEICA_XCPD36P_denoised_bold.dtseries.nii"
    single.write_bytes(b"single")
    multi.write_bytes(b"multi")

    monkeypatch.setattr(preparer, "dtseries_n_timepoints", lambda _: 7)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "prepare",
            "--dataset",
            "iowa7t",
            "--branch",
            "single_echo",
            "--subject",
            "06112026",
            "--data-root",
            str(data_root),
            "--subject-output-dir",
            str(tmp_path / "single_prepared"),
        ],
    )

    assert preparer.main() == 0
    single_manifest = json.loads(
        (tmp_path / "single_prepared" / "inputs" / "input_manifest.json").read_text()
    )
    assert single_manifest["branch"] == "single_echo"
    assert single_manifest["runs"][0]["dtseries"] == str(single)

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "prepare",
            "--dataset",
            "iowa7t",
            "--branch",
            "multi_echo",
            "--subject",
            "06112026",
            "--data-root",
            str(data_root),
            "--subject-output-dir",
            str(tmp_path / "multi_prepared"),
        ],
    )

    assert preparer.main() == 0
    multi_manifest = json.loads(
        (tmp_path / "multi_prepared" / "inputs" / "input_manifest.json").read_text()
    )
    assert multi_manifest["branch"] == "multi_echo"
    assert multi_manifest["runs"][0]["dtseries"] == str(multi)


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


def test_iowa7t_launcher_dry_run_uses_canonical_branch_layout(tmp_path: Path) -> None:
    env = os.environ.copy()
    env.update(
        {
            "DATASET": "iowa7t",
            "IOWA7T_BRANCH": "single_echo",
            "DATA_ROOT": str(tmp_path / "DataOrg_PFM_36P"),
            "OUT_ROOT": str(tmp_path / "hipp_outputs"),
            "SUBJECT_IDS_CSV": "06112026,06122026",
            "RUN_ID": "iowa7t_single_test",
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
    assert "run_id=iowa7t_single_test" in result.stderr
    assert str(tmp_path / "hipp_outputs" / "runs" / "logs" / "iowa7t_single_test") in result.stderr
    assert "outputs_iowa" not in result.stderr
    assert "HCPpipeline500" not in result.stderr
    manifests = list(
        (tmp_path / "hipp_outputs" / "runs" / "manifests" / "iowa7t_single_test").glob(
            "iowa7t_single_*.txt"
        )
    )
    assert len(manifests) == 2
