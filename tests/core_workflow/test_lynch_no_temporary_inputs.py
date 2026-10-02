from __future__ import annotations

import importlib.util
import hashlib
import json
import subprocess
from pathlib import Path
import sys

import pytest


CORE_DIR = Path(__file__).resolve().parents[2] / "scripts" / "core_workflow"
if str(CORE_DIR) not in sys.path:
    sys.path.insert(0, str(CORE_DIR))


MODULE_PATH = Path(__file__).resolve().parents[2] / "scripts" / "core_workflow" / "03_04_run_lynch2024_fastans_pfm.py"
spec = importlib.util.spec_from_file_location("lynch_fastans_no_tmp", MODULE_PATH)
assert spec and spec.loader
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_ephemeral_path_detector_rejects_tmp_variants() -> None:
    for raw in (
        "/tmp/merged/sub-1001.dtseries.nii",
        "/var/tmp/run/sub-1001.dtseries.nii",
        "/home/user/.tmp_lynch8.Cl8ZVK/merged/sub-1001.dtseries.nii",
        "/home/user/tmp_lynch8/merged/sub-1001.dtseries.nii",
    ):
        assert module.is_ephemeral_path(Path(raw))


def test_persistent_input_validator_accepts_canonical_source() -> None:
    source = Path(
        "/home/yojiao/LSS/jianglab/3_Data_Working/fmriprep_processed_JJ_TMS/"
        "xcpd_surf/sub-1001/func/sub-1001_task-rest_run-01_space-fsLR_"
        "den-91k_desc-denoised_bold.dtseries.nii"
    )
    assert module.validate_persistent_input_path(source, label="dtseries") == source


def test_persistent_concat_source_is_rejected() -> None:
    with pytest.raises(ValueError, match="in-memory"):
        module.validate_persistent_input_path(
            Path(
                "/home/yojiao/LSS/jianglab/3_Data_Working/fmriprep_processed_JJ_TMS/"
                "xcpd_surf_concat/sub-1001/func/sub-1001_task-rest_run-concat_space-fsLR_"
                "den-91k_desc-denoised_bold.dtseries.nii"
            ),
            label="dtseries",
        )


def test_hcp_canonical_single_run_concat_is_not_the_iowa_concat_tree() -> None:
    # HCP supplies one externally prepared canonical CIFTI.  The Iowa
    # prohibition is scoped to the generated xcpd_surf_concat derivative tree.
    assert not module.is_persistent_concat_path(
        Path(
            "/home/yojiao/hpchome/hipp/data/hippunfold_input/sub-1001/func/"
            "sub-1001_task-rest_run-concat.dtseries.nii"
        )
    )


def test_persistent_input_validator_fails_closed_for_ephemeral_source() -> None:
    with pytest.raises(ValueError, match="temporary"):
        module.validate_persistent_input_path(
            Path("/home/yojiao/hpchome/hipp/.tmp_lynch8.Cl8ZVK/merged/sub-1001.dtseries.nii"),
            label="dtseries",
        )


def test_pfm_run_template_resolves_ordered_inputs_without_concat_artifact(tmp_path: Path) -> None:
    func_dir = tmp_path / "sub-3002" / "func"
    func_dir.mkdir(parents=True)
    declared_first = func_dir / "source-alpha.dtseries.nii"
    declared_second = func_dir / "source-beta.dtseries.nii"
    declared_first.write_bytes(b"declared run two")
    declared_second.write_bytes(b"declared run seven")

    def receipt(path: Path) -> dict[str, object]:
        return {
            "path": str(path.resolve()),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "size_bytes": path.stat().st_size,
        }

    manifest = tmp_path / "sub-3002_run_artifacts.json"
    manifest.write_text(
        json.dumps(
            {
                "subject": "sub-3002",
                "runs": [
                    {"run_id": 2, "dtseries": receipt(declared_first)},
                    {"run_id": 7, "dtseries": receipt(declared_second)},
                ],
            }
        ),
        encoding="utf-8",
    )

    paths, mode, run_ids, receipts = module.resolve_pfm_dtseries_inputs(
        data_root=tmp_path,
        subject="3002",
        dtseries_template="{data_root}/{sub_subject}/func/{sub_subject}_task-rest_run-concat.dtseries.nii",
        run_artifact_manifest_template="{data_root}/{sub_subject}_run_artifacts.json",
    )

    assert paths == [declared_first.resolve(), declared_second.resolve()]
    assert mode == "in_memory_run_concat"
    assert run_ids == [2, 7]
    assert [row["path"] for row in receipts] == [
        str(declared_first.resolve()),
        str(declared_second.resolve()),
    ]


def test_pfm_missing_run_manifest_fails_before_subject_output_creation(tmp_path: Path) -> None:
    output_root = tmp_path / "outputs"

    with pytest.raises(FileNotFoundError, match="Missing explicit run artifact manifest"):
        module.run_fastans_lynch2024_subject(
            subject="3002",
            data_root=tmp_path / "data",
            output_root=output_root,
            resources_root=tmp_path / "resources",
            dtseries_template="{data_root}/{sub_subject}_run-concat.dtseries.nii",
            run_artifact_manifest_template="{data_root}/{sub_subject}_run_artifacts.json",
            left_surface_template="{data_root}/{sub_subject}_L.surf.gii",
            right_surface_template="{data_root}/{sub_subject}_R.surf.gii",
            wb_command=tmp_path / "wb_command",
        )

    assert not output_root.exists()


def test_pfm_cli_exposes_exact_manifest_selection() -> None:
    proc = subprocess.run(
        [sys.executable, str(MODULE_PATH), "--help"],
        check=False,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, proc.stderr
    assert "--run-artifact-manifest-template" in proc.stdout
