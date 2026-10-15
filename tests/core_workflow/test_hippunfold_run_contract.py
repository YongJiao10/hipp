"""Behavioral tests for the fixed HippUnfold stage-01 launch contract."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from scripts.core_workflow.lib.hippunfold_run_contract import (
    CANONICAL_ASSET_ROOT,
    CANONICAL_MODEL_PATH,
    CANONICAL_RUNTIME_SOURCE,
    DATASET_PRESETS,
    build_run_environment,
    resolve_run_contract,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
CORE_ENTRY = REPO_ROOT / "scripts/core_workflow/01_01_run_hippunfold_t1w_local.py"
BASE_WRAPPER = REPO_ROOT / "scripts/core_workflow/lib/hippunfold_shell/run_hippunfold_local.sh"
PY314 = Path("/home/yojiao/.conda/envs/py314/bin/python")


def test_hcp_preset_has_fixed_historical_paths_and_parameters() -> None:
    contract = resolve_run_contract("hcp", "102311")

    assert contract.bids_dir == REPO_ROOT / "data/hippunfold_input"
    assert contract.output_dir == (
        REPO_ROOT
        / "outputs_hcp/inputs/hcp/hippunfold/dense_corobl_batch/sub-102311/hippunfold"
    )
    assert contract.t1w_filter == "extension=.nii.gz"
    assert contract.hippunfold_argv[3:] == (
        "participant",
        "--modality",
        "T1w",
        "--output-density",
        "512",
        "--output-spaces",
        "corobl",
        "--autotop_labels",
        "hipp",
        "dentate",
        "--participant-label",
        "102311",
        "--runtime-source-cache-path",
        CANONICAL_RUNTIME_SOURCE,
        "--workflow-profile",
        "none",
        "--sdm",
        "env-modules",
        "--cores",
        "1",
        "--notemp",
        "--filter-T1w",
        "extension=.nii.gz",
        "--use_gpu",
    )
    assert "--omit-from" not in contract.hippunfold_argv
    assert "download_nnunet_model" not in contract.hippunfold_argv


@pytest.mark.parametrize(
    ("dataset", "subject", "bids_dir", "output_dir", "filter_value"),
    [
        (
            "stanford",
            "1001",
            "/home/yojiao/LSS/jianglab/3_Data_Working/fmriprep_processed_CausalConnectome/rest_fmriprep/DataOrg_PFM_36P",
            "outputs_stanford/inputs/stanford/hippunfold/dense_corobl_batch/sub-1001/hippunfold",
            "extension=.nii.gz",
        ),
        (
            "iowa_jj_tms",
            "1028",
            "/home/yojiao/LSS/jianglab/3_Data_Working/fmriprep_processed_JJ_TMS/raw",
            "outputs_iowa_jj_tms/inputs/iowa_jj_tms/hippunfold/dense_corobl_batch/sub-1028/hippunfold",
            "extension=.nii",
        ),
    ],
)
def test_non_hcp_presets_only_change_dataset_paths_and_extension(
    dataset: str,
    subject: str,
    bids_dir: str,
    output_dir: str,
    filter_value: str,
) -> None:
    contract = resolve_run_contract(dataset, subject)
    hcp = resolve_run_contract("hcp", "102311")

    assert str(contract.bids_dir) == bids_dir
    assert contract.output_dir == REPO_ROOT / output_dir
    assert contract.t1w_filter == filter_value
    assert contract.hippunfold_argv[3:-2][:11] == hcp.hippunfold_argv[3:-2][:11]
    assert contract.hippunfold_argv[3:-2][12:] == hcp.hippunfold_argv[3:-2][12:]
    assert "--omit-from" not in contract.hippunfold_argv


def test_contract_rejects_unsafe_or_unknown_inputs() -> None:
    with pytest.raises(ValueError, match="Unknown HippUnfold dataset"):
        resolve_run_contract("free_form", "1001")
    with pytest.raises(ValueError, match="alphanumeric"):
        resolve_run_contract("hcp", "sub-1001")
    with pytest.raises(ValueError, match="alphanumeric"):
        resolve_run_contract("hcp", "1001/../../other")


def test_environment_scrubs_all_free_form_hippunfold_overrides() -> None:
    contract = resolve_run_contract("hcp", "102311")
    env = build_run_environment(
        contract,
        base_env={
            "HIPPUNFOLD_EXTRA_ARGS": "--omit-from download_nnunet_model",
            "HIPPUNFOLD_OUTPUT_DENSITY": "18k",
            "HIPPUNFOLD_CACHE_DIR": "/tmp/agent-choice",
            "HIPPUNFOLD_MODEL_PATH": "/tmp/agent-model.tar",
            "PATH": ":/agent/bin::",
            "LD_LIBRARY_PATH": "::/agent/lib:",
        },
    )

    assert env["HIPPUNFOLD_CACHE_DIR"] == f"{CANONICAL_ASSET_ROOT}/hippunfold_assets"
    assert env["HIPPUNFOLD_RUNTIME_SOURCE_CACHE"] == CANONICAL_RUNTIME_SOURCE
    assert env["HIPPUNFOLD_MODEL_PATH"] == CANONICAL_MODEL_PATH
    assert env["XDG_CACHE_HOME"] == str(contract.runtime_root / "xdg_cache")
    assert env["PIP_CACHE_DIR"] == str(contract.runtime_root / "pip_cache")
    assert env["TMPDIR"] == str(contract.runtime_root / "tmp")
    assert not env["PATH"].endswith(":")
    assert not env["LD_LIBRARY_PATH"].endswith(":")
    assert "::" not in env["PATH"]
    assert "::" not in env["LD_LIBRARY_PATH"]
    assert not env["PATH"].startswith(":")
    assert not env["LD_LIBRARY_PATH"].startswith(":")
    assert "HIPPUNFOLD_EXTRA_ARGS" not in env
    assert "HIPPUNFOLD_OUTPUT_DENSITY" not in env


def test_public_dry_run_emits_fixed_json_contract() -> None:
    proc = subprocess.run(
        [str(PY314), str(CORE_ENTRY), "1001", "--dataset", "stanford", "--dry-run"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert proc.returncode == 0, proc.stderr
    payload = json.loads(proc.stdout)
    assert payload["dataset"] == "stanford"
    assert payload["subject"] == "1001"
    assert payload["output_spaces"] == ["corobl"]
    assert payload["expected_surface_spaces"] == ["corobl", "T1w", "unfold"]
    assert payload["output_density"] == "512"
    assert payload["modality"] == "T1w"
    assert payload["t1_reg_template"] is False
    assert payload["unfolded_registration"] is True
    assert payload["hippunfold_argv"].count("--omit-from") == 0


def test_shell_compatibility_adapter_rejects_free_form_argument_counts() -> None:
    proc = subprocess.run(
        ["bash", str(BASE_WRAPPER), "hcp", "1001", "unexpected"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert proc.returncode == 2
    assert "exactly two arguments" in proc.stderr


def test_dataset_presets_are_closed_and_complete() -> None:
    assert tuple(DATASET_PRESETS) == ("hcp", "stanford", "iowa_jj_tms", "banda", "pdc")


@pytest.mark.parametrize("dataset", ("hcp", "stanford", "iowa_jj_tms"))
def test_established_presets_keep_their_n4_launch_behavior(dataset: str) -> None:
    contract = resolve_run_contract(dataset, "1001")

    assert "--skip-preproc" not in contract.hippunfold_argv
    assert contract.as_json().get("skip_preproc") is False


@pytest.mark.parametrize(
    ("dataset", "subject"),
    (("banda", "BANDA001"), ("pdc", "e0010s01")),
)
def test_bias_corrected_inputs_skip_n4_in_the_public_dry_run(
    dataset: str, subject: str
) -> None:
    proc = subprocess.run(
        [str(PY314), str(CORE_ENTRY), subject, "--dataset", dataset, "--dry-run"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert proc.returncode == 0, proc.stderr
    payload = json.loads(proc.stdout)
    assert payload["dataset"] == dataset
    assert payload["subject"] == subject
    assert payload["hippunfold_argv"].count("--skip-preproc") == 1
    assert payload.get("skip_preproc") is True
    assert payload["modality"] == "T1w"
    assert "--no-reg-template" not in payload["hippunfold_argv"]
