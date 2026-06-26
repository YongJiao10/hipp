from pathlib import Path

import pytest

from scripts.iowa7t.run_branch_downstream import resolve_branch_job


def test_resolve_branch_job_writes_direct_branch_paths(tmp_path: Path) -> None:
    job = resolve_branch_job(
        "06112026",
        "single_echo",
        data_root=tmp_path / "data",
        output_root=tmp_path / "hipp_outputs",
    )

    assert job.branch_root == tmp_path / "hipp_outputs" / "subjects" / "sub-06112026" / "single_echo"
    assert job.surface_root == job.branch_root / "surface"
    assert job.cortex_kong2022_root == job.branch_root / "cortex" / "kong2022"
    assert job.fc_root == job.branch_root / "fc" / "kong2022"
    assert job.areal_mshbm_label_dir == job.branch_root / "cortex" / "areal_mshbm"
    assert "single_echo/sub-06112026" not in str(job.surface_root)


def test_resolve_branch_job_uses_branch_specific_input_templates(tmp_path: Path) -> None:
    single = resolve_branch_job(
        "06112026",
        "single_echo",
        data_root=tmp_path / "data",
        output_root=tmp_path / "hipp_outputs",
    )
    multi = resolve_branch_job(
        "06112026",
        "multi_echo",
        data_root=tmp_path / "data",
        output_root=tmp_path / "hipp_outputs",
    )

    assert single.bold_volume.name == "sub-06112026_task-rest_dir-concat_space-MNI152NLin6Asym_res-2_desc-denoised_bold.nii.gz"
    assert single.dtseries.name == "sub-06112026_task-rest_dir-concat_space-fsLR_den-91k_desc-denoised_bold.dtseries.nii"
    assert multi.bold_volume.name == "sub-06112026_task-rest_run-concat_space-MNI152NLin6Asym_res-2_desc-MEICA_XCPD36P_denoised_bold.nii.gz"
    assert multi.dtseries.name == "sub-06112026_task-rest_run-concat_space-fsLR_den-32k_desc-MEICA_XCPD36P_denoised_bold.dtseries.nii"


def test_resolve_branch_job_rejects_unknown_branch(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="Unsupported Iowa7T branch"):
        resolve_branch_job("06112026", "echo2", data_root=tmp_path / "data", output_root=tmp_path / "hipp_outputs")
