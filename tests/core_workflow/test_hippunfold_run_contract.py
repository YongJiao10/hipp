"""Behavioral tests for the fixed HippUnfold stage-01 launch contract."""

from __future__ import annotations

import importlib.util
import json
import io
import subprocess
import threading
import time
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

from scripts.core_workflow.lib import hippunfold_run_contract as run_contract_module
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


@pytest.fixture
def stage01(monkeypatch: pytest.MonkeyPatch) -> ModuleType:
    monkeypatch.syspath_prepend(str(CORE_ENTRY.parent))
    spec = importlib.util.spec_from_file_location("hippunfold_stage01", CORE_ENTRY)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(module, "prepare_runtime_directories", lambda env: None)
    monkeypatch.setattr(module, "prepare_runtime_python", lambda: None)
    monkeypatch.setattr(module, "prepare_subject_environment", lambda contract, env: dict(env))
    monkeypatch.setattr(module, "validate_local_assets", lambda env: None)
    monkeypatch.setattr(module, "detect_system_name", lambda: "idas")
    return module


def test_hcp_preset_has_fixed_historical_paths_and_parameters() -> None:
    contract = resolve_run_contract("hcp", "102311")

    assert contract.bids_dir == REPO_ROOT / "data/hippunfold_input"
    assert contract.output_dir == (
        REPO_ROOT
        / "outputs_hcp/inputs/hippunfold/dense_corobl_batch/sub-102311/hippunfold"
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
        "--inner_outer_reg_method",
        "greedy",
        "--inner_outer_reg_smoothing",
        "3.0",
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
            "outputs_stanford/inputs/hippunfold/dense_corobl_batch/sub-1001/hippunfold",
            "extension=.nii.gz",
        ),
        (
            "iowa_jj_tms",
            "1028",
            "/home/yojiao/LSS/jianglab/3_Data_Working/fmriprep_processed_JJ_TMS/raw",
            "outputs_iowa_jj_tms/inputs/hippunfold/dense_corobl_batch/sub-1028/hippunfold",
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
    assert contract.hippunfold_argv[3:-6][:11] == hcp.hippunfold_argv[3:-6][:11]
    assert contract.hippunfold_argv[3:-6][12:] == hcp.hippunfold_argv[3:-6][12:]
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


def test_runtime_gate_uses_the_same_fixed_environment_as_the_run(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    executable = tmp_path / "hippo2" / "bin" / "hippunfold"
    executable.parent.mkdir(parents=True)
    executable.write_text("", encoding="utf-8")
    monkeypatch.setattr(run_contract_module, "HIPPUNFOLD_EXECUTABLE", executable)
    monkeypatch.setattr(run_contract_module, "HIPPO2_PREFIX", executable.parent.parent)

    env = {
        "HIPPUNFOLD_CACHE_DIR": str(tmp_path / "cache"),
        "XDG_CACHE_HOME": str(tmp_path / "xdg"),
        "PIP_CACHE_DIR": str(tmp_path / "pip"),
        "TMPDIR": str(tmp_path / "tmp"),
    }
    received_envs: list[dict[str, str] | None] = []

    def fake_run(args: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        received_envs.append(kwargs.get("env"))  # type: ignore[arg-type]
        if args[0] == "/bin/bash":
            stdout = str(executable)
        elif args[-1] == "--version":
            stdout = "2.3.0"
        elif args[-1] == "--help":
            stdout = "--output-density {native,512,2k} --inner_outer_reg_method --inner_outer_reg_smoothing"
        else:
            stdout = ""
        return subprocess.CompletedProcess(args, 0, stdout=stdout, stderr="")

    monkeypatch.setattr(run_contract_module.subprocess, "run", fake_run)

    run_contract_module.validate_local_assets(env)

    assert len(received_envs) == 4
    assert all(received is env for received in received_envs)


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


def test_manifest_mode_validates_once_and_checks_each_subject(
    stage01: ModuleType, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    subjects = ["k0114s02", "k0114s03", "k0114s04"]
    manifest = tmp_path / "subjects.txt"
    manifest.write_text("\n".join(subjects), encoding="utf-8")
    validated = []
    executed = []
    checked = []
    monkeypatch.setattr(stage01, "_is_completed", lambda contract: False)
    monkeypatch.setattr(stage01, "validate_local_assets", validated.append)
    def start(contract, env):
        executed.append((contract.subject, env))
        return subprocess.Popen([str(PY314), '-c', 'pass']), io.StringIO()
    monkeypatch.setattr(stage01, "_start_subject", start)
    monkeypatch.setattr(stage01, "_check_subject", lambda contract: checked.append(contract.subject))

    assert stage01.run_t1w_local(["--dataset", "pdc", "--manifest", str(manifest)]) == 0

    assert len(validated) == 1
    assert executed[0][1] is validated[0]
    assert [subject for subject, env in executed] == subjects
    assert checked == subjects
    assert [env["HIPPUNFOLD_CACHE_ROOT"] for subject, env in executed] == [
        str(resolve_run_contract("pdc", subject).runtime_root) for subject in subjects
    ]


def test_batch_runs_two_subjects_and_checks_outputs_on_the_coordinator(
    stage01: ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
    contracts = [resolve_run_contract("pdc", f"k0114s0{index}") for index in range(1, 6)]
    processes = []
    max_active = 0
    check_threads = []
    coordinator = threading.get_ident()
    monkeypatch.setattr(stage01, "_is_completed", lambda contract: False)

    def execute(contract, env):
        nonlocal max_active
        process = subprocess.Popen([str(PY314), '-c', 'import time; time.sleep(0.4)'])
        processes.append(process)
        max_active = max(max_active, sum(p.poll() is None for p in processes))
        return process, io.StringIO()

    monkeypatch.setattr(stage01, "_start_subject", execute)
    monkeypatch.setattr(stage01, "_check_subject", lambda contract: check_threads.append(threading.get_ident()))

    stage01._run_batch(contracts, jobs=2)

    assert max_active == 2
    assert check_threads == [coordinator] * len(contracts)


def test_hcp_accepts_the_twelve_subject_profile(stage01, monkeypatch):
    executed = []
    checked = []
    monkeypatch.setattr(stage01, "_is_completed", lambda contract: False)

    class FinishedProcess:
        returncode = 0

        def poll(self):
            return 0

    def start(contract, env):
        executed.append(contract.subject)
        return FinishedProcess(), io.StringIO()

    monkeypatch.setattr(stage01, "_start_subject", start)
    monkeypatch.setattr(stage01, "_check_subject", lambda contract: checked.append(contract.subject))
    stage01._run_batch([resolve_run_contract("hcp", "1001")], jobs=12)
    assert executed == checked == ["1001"]


def test_hcp_has_two_gpu_slots_while_other_presets_remain_serial_on_gpu():
    hcp = build_run_environment(resolve_run_contract("hcp", "1001"))
    pdc = build_run_environment(resolve_run_contract("pdc", "1001"))
    assert hcp["HIPPUNFOLD_GPU_SLOTS"] == "2"
    assert pdc["HIPPUNFOLD_GPU_SLOTS"] == "1"


def test_completed_batch_does_not_run_the_runtime_gate(
    stage01: ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
    validated = []
    monkeypatch.setattr(stage01, "_is_completed", lambda contract: True)
    monkeypatch.setattr(stage01, "validate_local_assets", validated.append)

    stage01._run_batch([resolve_run_contract("pdc", "k0114s01")], jobs=1)

    assert validated == []


@pytest.mark.parametrize(("dataset", "jobs", "limit"), [("pdc", 3, 2), ("banda", 2, 1)])
def test_batch_rejects_excess_jobs_before_validation_or_execution(
    stage01: ModuleType, monkeypatch: pytest.MonkeyPatch,
    dataset: str, jobs: int, limit: int,
) -> None:
    calls = []
    monkeypatch.setattr(stage01, "validate_local_assets", lambda env: calls.append("gate"))
    monkeypatch.setattr(stage01, "_start_subject", lambda contract, env: calls.append("run"))

    with pytest.raises(ValueError, match=f"{dataset} allows between 1 and {limit} jobs"):
        stage01._run_batch([resolve_run_contract(dataset, "1001")], jobs=jobs)

    assert calls == []


@pytest.mark.parametrize("failure_stage", ["execution", "checking"])
def test_batch_failure_stops_dispatching_new_subjects(
    stage01: ModuleType, monkeypatch: pytest.MonkeyPatch, failure_stage: str,
) -> None:
    contracts = [resolve_run_contract("pdc", f"k0114s0{index}") for index in range(1, 4)]
    executed = []
    monkeypatch.setattr(stage01, "_is_completed", lambda contract: False)

    def execute(contract, env):
        executed.append(contract.subject)
        return subprocess.Popen([str(PY314), '-c', 'pass']), io.StringIO()

    def check(contract) -> None:
        raise RuntimeError("subject failed")

    if failure_stage == 'execution':
        def execute(contract, env):
            executed.append(contract.subject)
            raise RuntimeError('subject failed')
    monkeypatch.setattr(stage01, "_start_subject", execute)
    monkeypatch.setattr(stage01, "_check_subject", check)

    with pytest.raises(RuntimeError, match="subject failed"):
        stage01._run_batch(contracts, jobs=1)

    assert executed == [contracts[0].subject]


def test_single_subject_keeps_one_gate_and_returns_the_process_status(
    stage01: ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
    validated = []
    executed = []
    monkeypatch.setattr(stage01, "validate_local_assets", validated.append)

    def execute(contract, env) -> int:
        executed.append((contract.subject, env))
        return 7

    monkeypatch.setattr(stage01, "run_hippunfold", execute)

    assert stage01.run_t1w_local(["k0114s02", "--dataset", "pdc"]) == 7
    assert len(validated) == 1
    assert executed == [("k0114s02", validated[0])]


def test_batch_checker_fails_on_missing_outputs(
    stage01: ModuleType, monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    contract = SimpleNamespace(subject="1001", output_dir=tmp_path, preset=SimpleNamespace(output_root=tmp_path))
    calls = []

    def fail_check(args, **kwargs):
        calls.append((args, kwargs))
        raise subprocess.CalledProcessError(1, args)

    monkeypatch.setattr(stage01.subprocess, "run", fail_check)

    with pytest.raises(subprocess.CalledProcessError):
        stage01._check_subject(contract)

    assert "--fail-on-missing" in calls[0][0]
    assert "--no-cleanup" in calls[0][0]
    assert calls[0][1]["check"] is True
    assert "[done" not in capsys.readouterr().out


def test_manifest_dry_run_uses_the_existing_public_entry_point(tmp_path: Path) -> None:
    manifest = tmp_path / "subjects.txt"
    manifest.write_text("k0114s02\nk0114s03\n", encoding="utf-8")
    proc = subprocess.run(
        [str(PY314), str(CORE_ENTRY), "--dataset", "pdc", "--manifest", str(manifest),
         "--jobs", "2", "--dry-run"],
        cwd=REPO_ROOT, capture_output=True, text=True, check=False,
    )

    assert proc.returncode == 0, proc.stderr
    assert [row["subject"] for row in json.loads(proc.stdout)] == ["k0114s02", "k0114s03"]


def test_surfaces_without_output_check_receipt_do_not_skip_a_subject(
    stage01: ModuleType, tmp_path: Path,
) -> None:
    output_dir = tmp_path / "hippunfold"
    surfaces = output_dir / "sub-1001" / "surf"
    surfaces.mkdir(parents=True)
    for hemi in ("L", "R"):
        (surfaces / f"sub-1001_hemi-{hemi}_space-corobl_den-512_label-hipp_midthickness.surf.gii").write_text("", encoding="utf-8")
    contract = SimpleNamespace(subject="1001", output_dir=output_dir)

    assert stage01._is_completed(contract) is False

    (output_dir / "core_workflow_cleanup_receipt.json").write_text("{}", encoding="utf-8")
    assert stage01._is_completed(contract) is False


def test_v220_complete_run_cannot_be_reused_as_v230(
    stage01: ModuleType, monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    monkeypatch.setattr(run_contract_module, "DATASET_PRESETS", {
        "hcp": run_contract_module.DatasetPreset(
            bids_dir=tmp_path / "bids", output_root=tmp_path,
            t1w_filter="extension=.nii.gz",
        ),
    })
    contract = resolve_run_contract("hcp", "1001")
    expected = "sub-{subject}/surf/midthickness.surf.gii"
    surface = contract.output_dir / expected.format(subject="1001")
    surface.parent.mkdir(parents=True)
    surface.touch()
    manifest = tmp_path / "_batch_logs/output_integrity_manifest.json"
    manifest.parent.mkdir()
    manifest.write_text(json.dumps({"expected_outputs": [expected]}))
    old_contract = contract.as_json()
    old_contract["hippunfold_version"] = "2.2.0"
    (contract.output_dir / "output_check.json").write_text(json.dumps([{
        "status": "complete_all_outputs_present", "missing_count": 0,
        "run_contract": old_contract,
    }]))

    with pytest.raises(RuntimeError, match="mismatched run contract"):
        stage01._is_completed(contract)


def test_batch_keep_going_checks_later_subjects_after_a_subject_fails(
    stage01: ModuleType, monkeypatch: pytest.MonkeyPatch,
) -> None:
    contracts = [resolve_run_contract("pdc", f"k0114s0{i}") for i in range(1, 4)]
    checked = []
    monkeypatch.setattr(stage01, "_is_completed", lambda contract: False)

    class FinishedProcess:
        returncode = 0

        def poll(self):
            return 0

    monkeypatch.setattr(stage01, "_start_subject", lambda contract, env: (FinishedProcess(), io.StringIO()))

    def check(contract):
        checked.append(contract.subject)
        if contract.subject == contracts[0].subject:
            raise RuntimeError("first subject failed QC")

    monkeypatch.setattr(stage01, "_check_subject", check)
    with pytest.raises(RuntimeError, match="first subject failed QC"):
        stage01._run_batch(contracts, jobs=1, keep_going=True)
    assert checked == [contract.subject for contract in contracts]
