from pathlib import Path

import json

import numpy as np
import pytest

import scripts.iowa7t.run_branch_downstream as downstream
from scripts.iowa7t.run_branch_downstream import (
    fixed_400_roi_reference,
    load_run_segments_direct,
    mark_missing_roi_fc_columns,
    resolve_branch_job,
)


def test_resolve_branch_job_writes_direct_branch_paths(tmp_path: Path) -> None:
    job = resolve_branch_job(
        "06112026",
        "multiband",
        data_root=tmp_path / "data",
        output_root=tmp_path / "hipp_outputs",
    )

    assert job.branch_root == tmp_path / "hipp_outputs" / "subjects" / "sub-06112026" / "multiband"
    assert job.surface_root == job.branch_root / "surface"
    assert job.cortex_kong2022_root == job.branch_root / "cortex" / "kong2022"
    assert job.fc_root == job.branch_root / "fc" / "kong2022"
    assert job.areal_mshbm_label_dir == job.branch_root / "cortex" / "areal_mshbm"
    assert "multiband/sub-06112026" not in str(job.surface_root)


def test_resolve_branch_job_uses_branch_specific_input_templates(tmp_path: Path) -> None:
    single = resolve_branch_job(
        "06112026",
        "multiband",
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
    assert multi.dtseries.name == "sub-06112026_task-rest_run-concat_space-fsLR_den-91k_desc-MEICA_XCPD36P_denoised_bold.dtseries.nii"


def test_resolve_branch_job_uses_noxcpd_input_templates(tmp_path: Path) -> None:
    single = resolve_branch_job(
        "06112026",
        "multiband",
        variant="noxcpd",
        data_root=tmp_path / "DataOrg_PFM_NoXCPD",
        output_root=tmp_path / "DataOrg_PFM_NoXCPD" / "hipp_outputs",
    )
    multi = resolve_branch_job(
        "06112026",
        "multi_echo",
        variant="noxcpd",
        data_root=tmp_path / "DataOrg_PFM_NoXCPD",
        output_root=tmp_path / "DataOrg_PFM_NoXCPD" / "hipp_outputs",
    )

    assert single.bold_volume.name == "sub-06112026_task-rest_dir-concat_space-MNI152NLin6Asym_res-2_desc-NoXCPD_bold.nii.gz"
    assert single.dtseries.name == "sub-06112026_task-rest_dir-concat_space-fsLR_den-91k_desc-NoXCPD_bold.dtseries.nii"
    assert multi.bold_volume.name == "sub-06112026_task-rest_run-concat_space-MNI152NLin6Asym_res-2_desc-MEICA_NoXCPD_bold.nii.gz"
    assert multi.dtseries.name == "sub-06112026_task-rest_run-concat_space-fsLR_den-91k_desc-MEICA_NoXCPD_bold.dtseries.nii"


def test_resolve_branch_job_defaults_noxcpd_hippunfold_to_own_subject_tree(tmp_path: Path) -> None:
    job = resolve_branch_job(
        "06112026",
        "multiband",
        variant="noxcpd",
        data_root=tmp_path / "DataOrg_PFM_NoXCPD",
        output_root=tmp_path / "DataOrg_PFM_NoXCPD" / "hipp_outputs",
        default_36p_output_root=tmp_path / "DataOrg_PFM_36P" / "hipp_outputs",
    )

    assert job.hippunfold_dir == tmp_path / "DataOrg_PFM_NoXCPD" / "hipp_outputs" / "subjects" / "sub-06112026" / "hippunfold"


def test_resolve_branch_job_accepts_explicit_hippunfold_batch_root(tmp_path: Path) -> None:
    job = resolve_branch_job(
        "06112026",
        "multi_echo",
        variant="noxcpd",
        data_root=tmp_path / "DataOrg_PFM_NoXCPD",
        output_root=tmp_path / "DataOrg_PFM_NoXCPD" / "hipp_outputs",
        hippunfold_batch_root=tmp_path / "shared_subjects",
    )

    assert job.hippunfold_dir == tmp_path / "shared_subjects" / "sub-06112026" / "hippunfold"


def test_resolve_branch_job_uses_0p_32p_input_templates(tmp_path: Path) -> None:
    cases = {
        "0p": "MEICA_XCPD0P_denoised",
        "32p": "MEICA_XCPD32P_denoised",
    }
    for variant, multi_echo_desc in cases.items():
        data_root = tmp_path / f"DataOrg_PFM_{variant.upper()}"
        output_root = data_root / "hipp_outputs"
        single = resolve_branch_job(
            "06122026",
            "multiband",
            variant=variant,
            data_root=data_root,
            output_root=output_root,
        )
        multi = resolve_branch_job(
            "06122026",
            "multi_echo",
            variant=variant,
            data_root=data_root,
            output_root=output_root,
        )

        assert single.bold_volume.name == "sub-06122026_task-rest_dir-concat_space-MNI152NLin6Asym_res-2_desc-denoised_bold.nii.gz"
        assert single.dtseries.name == "sub-06122026_task-rest_dir-concat_space-fsLR_den-91k_desc-denoised_bold.dtseries.nii"
        assert multi.bold_volume.name == f"sub-06122026_task-rest_run-concat_space-MNI152NLin6Asym_res-2_desc-{multi_echo_desc}_bold.nii.gz"
        assert multi.dtseries.name == f"sub-06122026_task-rest_run-concat_space-fsLR_den-91k_desc-{multi_echo_desc}_bold.dtseries.nii"
        assert multi.hippunfold_dir == output_root / "subjects" / "sub-06122026" / "hippunfold"


def test_resolve_branch_job_rejects_unknown_branch(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="Unsupported Iowa7T branch"):
        resolve_branch_job("06112026", "echo2", data_root=tmp_path / "data", output_root=tmp_path / "hipp_outputs")


def test_fixed_400_roi_reference_keeps_missing_roi_as_nan() -> None:
    roi_registry = [
        {
            "roi_id": 1,
            "roi_name": "L_missing",
            "family17": "DefaultA",
            "canonical8": "Default",
            "hemi": "L",
            "subregion": "FPole",
            "rank": 1,
        },
        {
            "roi_id": 2,
            "roi_name": "L_present",
            "family17": "DefaultA",
            "canonical8": "Default",
            "hemi": "L",
            "subregion": "PFC",
            "rank": 2,
        },
        {
            "roi_id": 201,
            "roi_name": "R_present",
            "family17": "DefaultA",
            "canonical8": "Default",
            "hemi": "R",
            "subregion": "PFC",
            "rank": 1,
        },
    ]
    left_rows = [
        {
            "parcel_id": 2,
            "parcel_name": "L_present",
            "family17": "DefaultA",
            "canonical8": "Default",
            "hemisphere": "L",
            "subregion": "PFC",
            "rank": 2,
            "n_grayordinates_total": 4,
            "n_grayordinates_used": 4,
            "excluded_by_tsnr_gate": False,
        }
    ]
    right_rows = [
        {
            "parcel_id": 201,
            "parcel_name": "R_present",
            "family17": "DefaultA",
            "canonical8": "Default",
            "hemisphere": "R",
            "subregion": "PFC",
            "rank": 1,
            "n_grayordinates_total": 5,
            "n_grayordinates_used": 5,
            "excluded_by_tsnr_gate": False,
        }
    ]

    parcel_ts, rows, missing_ids = fixed_400_roi_reference(
        roi_registry=roi_registry,
        left_ts=[np.asarray([1.0, 2.0, 3.0], dtype=np.float32)],
        left_rows=left_rows,
        right_ts=[np.asarray([4.0, 5.0, 6.0], dtype=np.float32)],
        right_rows=right_rows,
        n_timepoints=3,
    )

    assert parcel_ts.shape == (3, 3)
    assert rows[0]["parcel_id"] == 1
    assert rows[0]["included_in_timeseries"] is False
    assert rows[0]["excluded_by_no_vertices"] is True
    assert missing_ids == [1]
    assert np.isnan(parcel_ts[0]).all()
    np.testing.assert_array_equal(parcel_ts[1], np.asarray([1.0, 2.0, 3.0], dtype=np.float32))
    np.testing.assert_array_equal(parcel_ts[2], np.asarray([4.0, 5.0, 6.0], dtype=np.float32))


def test_mark_missing_roi_fc_columns_restores_nan_after_corrcoef() -> None:
    fc = np.zeros((2, 4), dtype=np.float32)
    out = mark_missing_roi_fc_columns(fc, missing_roi_ids=[2, 4], roi_ids=[1, 2, 3, 4])

    assert np.all(out[:, 0] == 0.0)
    assert np.isnan(out[:, 1]).all()
    assert np.all(out[:, 2] == 0.0)
    assert np.isnan(out[:, 3]).all()

def test_load_run_segments_direct_honors_explicit_ids(tmp_path: Path) -> None:
    job = resolve_branch_job(
        "1001",
        "multiband",
        data_root=tmp_path / "data",
        output_root=tmp_path / "hipp_outputs",
    )
    metadata = tmp_path / "run_metadata.json"
    metadata.write_text(
        json.dumps({"run_ids": [2, 7], "run_lengths": [2, 3], "concat_timepoints": 5}),
        encoding="utf-8",
    )
    runs_dir = job.cortex_common_root / "runs"
    for run_id, run_start, run_length in ((2, 0, 2), (7, 2, 3)):
        run_dir = runs_dir / f"run-{run_id}"
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "reference_summary.json").write_text(
            json.dumps(
                {
                    "run_id": str(run_id),
                    "run_start": run_start,
                    "run_stop": run_start + run_length,
                    "run_length": run_length,
                }
            ),
            encoding="utf-8",
        )

    segments = load_run_segments_direct(job, run_metadata_path=metadata)

    assert [row["run_id"] for row in segments] == [2, 7]
    assert [(row["run_start"], row["run_stop"]) for row in segments] == [(0, 2), (2, 5)]


def test_run_job_rejects_missing_metadata_before_running_stages(monkeypatch, tmp_path: Path) -> None:
    job = resolve_branch_job(
        "1001",
        "multiband",
        data_root=tmp_path / "data",
        output_root=tmp_path / "hipp_outputs",
    )
    monkeypatch.setattr(downstream, "require_inputs", lambda _: None)

    with pytest.raises(ValueError, match="requires an explicit run metadata path"):
        downstream.run_job(job, stage="all", overwrite=False)
    assert not job.branch_root.exists()
