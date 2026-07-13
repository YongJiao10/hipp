from __future__ import annotations

import importlib.util
import json
import subprocess
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = REPO_ROOT / "scripts/core_workflow/run_formal_gradient_spectral_refresh.py"
PY314 = Path("/home/yojiao/.conda/envs/py314/bin/python")


def load_runner():
    spec = importlib.util.spec_from_file_location("formal_refresh_runner", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_stage_plan_is_exact_sequential_formal_scope() -> None:
    runner = load_runner()
    stages = runner.build_stage_plan(repo_root=REPO_ROOT, python_exe=PY314, jobs=1)

    assert [stage["name"] for stage in stages] == [
        "lynch20_reference",
        "lynch20_fc",
        "lynch8_fc",
        "lynch20_gradient_bank",
        "lynch20_group",
        "lynch20_overview",
        "lynch8_group",
        "lynch8_overview",
        "hcp_weighted_branch",
        "hcp_weighted_overview",
        "stanford_hc_group",
        "stanford_hc_weighted",
        "stanford_hc_overviews",
        "stanford_symp_group",
        "stanford_symp_weighted",
        "stanford_symp_overviews",
        "hcp_routea_overviews",
        "hcp_family17_overviews",
        "formal_release_audit",
    ]
    group_names = {"lynch20_group", "lynch8_group", "stanford_hc_group", "stanford_symp_group"}
    for stage in stages:
        commands = [list(stage["command"]), *(list(item) for item in stage.get("additional_commands", []))]
        command = commands[0]
        if stage["name"] in group_names:
            assert _option(command, "--methods") == "gradient-spectral"
            assert _option(command, "--k-min") == "2"
            assert _option(command, "--k-max") == "8"
            assert _option(command, "--force-final-k") == "3"
            assert _option(command, "--jobs") == "1"
        for nested_command in commands:
            text = " ".join(nested_command).lower()
            assert "21net" not in text
            assert "routeb" not in text
            assert "ctxreg" not in text
            assert "ctxdrop" not in text
            if nested_command[1].endswith("06_01_build_group_overview.py"):
                assert _option(nested_command, "--manifest-name").endswith(
                    "group_overview_manifest.json"
                )
                assert _option(nested_command, "--scene").endswith(
                    "config/wb_locked_native_view_lateral_medial.scene"
                )
            if nested_command[1].endswith("05_05_build_weighted_fisherz_branch.py"):
                assert _option(nested_command, "--network-style").endswith(
                    "config/hipp_network_style.json"
                )

    audit = list(stages[-1]["command"])
    assert audit[0] == str(PY314)
    assert audit[1].endswith("scripts/core_workflow/audit_formal_gradient_spectral_release.py")
    assert _option(audit, "--output-json").endswith("formal.audit.json")

    single_overview_names = {
        "lynch20_overview",
        "lynch8_overview",
        "hcp_weighted_overview",
        "stanford_hc_overviews",
        "stanford_symp_overviews",
    }
    for stage in stages:
        if stage["name"] in single_overview_names:
            assert _option(list(stage["command"]), "--output-name").endswith(".png")

    by_name = {str(stage["name"]): stage for stage in stages}
    assert _option(list(by_name["lynch20_overview"]["command"]), "--output-name") == (
        "lynch2024_20net_gradient-spectral_group_overview_g1g2g3_lynch2024_20net_fisherz.png"
    )
    assert _option(list(by_name["lynch8_overview"]["command"]), "--output-name") == (
        "lynch2024_8net_gradient-spectral_group_overview_g1g2g3_lynch2024_8net_fisherz.png"
    )
    for name in ("hcp_routea_overviews", "hcp_family17_overviews"):
        assert "--output-name" not in list(by_name[name]["command"])


def _option(command: list[str], option: str) -> str:
    return command[command.index(option) + 1]


@pytest.mark.parametrize("jobs", [0, 2, 10])
def test_release_settings_reject_parallel_jobs(jobs: int) -> None:
    runner = load_runner()
    with pytest.raises(ValueError, match="fixed at --jobs 1"):
        runner.validate_release_settings(run_id="formal", jobs=jobs)


@pytest.mark.parametrize("run_id", ["../escape", "/absolute", "bad id", "a" * 129, ""])
def test_release_settings_reject_unsafe_run_ids(run_id: str) -> None:
    runner = load_runner()
    with pytest.raises(ValueError, match="Unsafe --run-id"):
        runner.validate_release_settings(run_id=run_id, jobs=1)


def test_worker_stops_on_first_failure_and_records_only_prior_stages(tmp_path, monkeypatch) -> None:
    runner = load_runner()
    stages = [
        {"name": "first", "command": ["first"]},
        {"name": "middle", "command": ["middle"]},
        {"name": "later", "command": ["later"]},
    ]
    calls: list[list[str]] = []

    monkeypatch.setattr(runner, "build_stage_plan", lambda **_kwargs: stages)
    monkeypatch.setattr(runner, "preflight", lambda *_args: None)

    def fake_run(command, **kwargs):
        calls.append(list(command))
        assert kwargs["check"] is True
        assert all(kwargs["env"][name] == "1" for name in runner.THREAD_ENV)
        if command == ["middle"]:
            raise subprocess.CalledProcessError(9, command)

    monkeypatch.setattr(runner.subprocess, "run", fake_run)

    assert runner.run_worker(repo_root=tmp_path, run_id="formal", jobs=1) == 1
    assert calls == [["first"], ["middle"]]
    status = json.loads(runner.run_paths(tmp_path, "formal")["status"].read_text())
    assert status["status"] == "failed"
    assert status["completed_stages"] == ["first"]
    assert status["failed_stage"] == "middle"


def test_worker_is_ok_only_after_audit_completes(tmp_path, monkeypatch) -> None:
    runner = load_runner()
    stages = [
        {"name": "group", "command": ["group"]},
        {"name": "formal_release_audit", "command": ["audit"]},
    ]
    observed_statuses: list[tuple[list[str], str, list[str]]] = []

    monkeypatch.setattr(runner, "build_stage_plan", lambda **_kwargs: stages)
    monkeypatch.setattr(runner, "preflight", lambda *_args: None)

    def fake_run(command, **_kwargs):
        path = runner.run_paths(tmp_path, "formal")["status"]
        payload = json.loads(path.read_text())
        observed_statuses.append((list(command), payload["status"], payload["completed_stages"]))

    monkeypatch.setattr(runner.subprocess, "run", fake_run)

    assert runner.run_worker(repo_root=tmp_path, run_id="formal", jobs=1) == 0
    assert observed_statuses == [
        (["group"], "running", []),
        (["audit"], "running", ["group"]),
    ]
    status = json.loads(runner.run_paths(tmp_path, "formal")["status"].read_text())
    assert status["status"] == "ok"
    assert status["completed_stages"] == ["group", "formal_release_audit"]


def test_worker_command_forwards_explicit_repo_root(tmp_path) -> None:
    runner = load_runner()
    command = runner.build_worker_command(
        script_path=SCRIPT,
        repo_root=tmp_path,
        run_id="formal",
        jobs=1,
    )
    assert _option(command, "--repo-root") == str(tmp_path.resolve())


def test_status_paths_preserve_dotted_run_id(tmp_path) -> None:
    runner = load_runner()
    paths = runner.run_paths(tmp_path, "formal.v1")
    assert paths["status"].name == "formal.v1.status.json"
    assert paths["audit"].name == "formal.v1.audit.json"
