from __future__ import annotations

import importlib.util
import json
import subprocess
from collections import Counter
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
            retired = "21" + "net"
            assert retired not in text
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
            if nested_command[1].endswith("05_02_run_group_atlas.py"):
                expected_dataset = "outputs_stanford" if "stanford" in stage["name"] else "outputs_hcp"
                assert _option(nested_command, "--dataset") == expected_dataset

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


def test_preflight_rejects_missing_exact_hcp_group_manifest(tmp_path, monkeypatch) -> None:
    runner = load_runner()
    missing = (
        tmp_path
        / "outputs_hcp/work/group_atlas/routea_full_k3_fisherz"
        / "gradient-spectral/kong2022/2mm/cov_0p80/g1g2g3/hemi_L"
        / "group_atlas_manifest.json"
    )
    missing.parent.mkdir(parents=True)
    required = set(runner.required_preexisting_paths(tmp_path))
    for branch in ("routea_full_k3_fisherz", "routea_17net_full_k3_fisherz"):
        for smoothing in ("2mm", "4mm"):
            for hemi in ("L", "R"):
                assert (
                    tmp_path
                    / "outputs_hcp/work/group_atlas"
                    / branch
                    / "gradient-spectral/kong2022"
                    / smoothing
                    / "cov_0p80/g1g2g3"
                    / f"hemi_{hemi}/group_atlas_manifest.json"
                ) in required

    monkeypatch.setattr(runner, "required_preexisting_paths", lambda _root: [missing])

    with pytest.raises(FileNotFoundError) as exc_info:
        runner.preflight(tmp_path, [])

    assert str(exc_info.value) == f"Missing formal refresh inputs:\n{missing}"


def test_preflight_derives_subject_specific_stanford_and_lynch_inputs(tmp_path) -> None:
    runner = load_runner()
    stanford_manifests = tmp_path / "outputs_stanford/inputs/stanford/kong2022/manifests"
    stanford_manifests.mkdir(parents=True)
    (stanford_manifests / "stanford_hc_1xxx2xxx_qc_pass_t1only_91.txt").write_text(
        "".join(f"sub-{1001 + i}\n" for i in range(91))
    )
    (stanford_manifests / "stanford_symp_3xxx4xxx_qc_pass_t1only_103.txt").write_text(
        "".join(f"sub-{3001 + i}\n" for i in range(103))
    )
    lynch_manifest = (
        tmp_path
        / "outputs_hcp/inputs/hcp/lynch2024_fast_pfm_t1only/manifests"
        / "hcp_7t_lynch2024_nsi043_pass_78.txt"
    )
    lynch_manifest.parent.mkdir(parents=True)
    lynch_manifest.write_text("".join(f"sub-{100610 + i}\n" for i in range(78)))

    required = set(runner.required_preexisting_paths(tmp_path))

    assert (
        tmp_path
        / "outputs_stanford/inputs/stanford/kong2022/shared_inputs/routea"
        / "sub-1001/fc/kong2022/2mm/hemi_L/hipp_vertex_to_network_fc.npy"
    ) in required
    assert (
        tmp_path
        / "outputs_stanford/work/hippomaps_gradients/routea/4mm/hemi_R"
        / "hippomaps_gradient_manifest.json"
    ) in required
    assert (
        tmp_path
        / "outputs_stanford/inputs/stanford/kong2022/templates"
        / "hipp_group_average_t1only_151/group_average_surfaces"
        / "group-average_hemi-L_space-corobl_den-512_label-hipp_midthickness.surf.gii"
    ) in required
    assert (
        tmp_path
        / "outputs_hcp/work/cortex_pfm/sub-100610/lynch2024"
        / "cortex_roi_component_timeseries.npy"
    ) in required
    assert (
        tmp_path
        / "outputs_hcp/work/cortex_pfm/sub-100610/lynch2024/cortex_roi_components.tsv"
    ) in required


@pytest.mark.parametrize(
    ("name", "lines", "message"),
    [
        ("hcp_7t_lynch2024_nsi043_pass_78.txt", ["sub-100610"] * 78, "duplicate"),
        ("stanford_hc_1xxx2xxx_qc_pass_t1only_91.txt", ["1001"] * 91, "canonical"),
        ("stanford_symp_3xxx4xxx_qc_pass_t1only_103.txt", [f"sub-{3001+i}" for i in range(102)], "103"),
    ],
)
def test_subject_ids_reject_invalid_formal_cohort_manifests(
    tmp_path: Path, name: str, lines: list[str], message: str
) -> None:
    runner = load_runner()
    manifest = tmp_path / name
    manifest.write_text("\n".join(lines) + "\n")

    with pytest.raises(ValueError, match=message):
        runner._subject_ids(manifest)


def test_runner_formal_groups_match_audit_and_both_registry_entries() -> None:
    runner = load_runner()
    audit_path = REPO_ROOT / "scripts/core_workflow/audit_formal_gradient_spectral_release.py"
    spec = importlib.util.spec_from_file_location("formal_audit_for_runner_test", audit_path)
    assert spec is not None and spec.loader is not None
    audit = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(audit)
    registry = json.loads((REPO_ROOT / "scripts/core_workflow/step_registry.json").read_text())
    registered = {
        item["step_id"]: {
            (row["dataset"], row["group_root"], row["atlas"])
            for row in item["formal_groups"]
        }
        for item in registry["utilities"]
        if "formal_groups" in item
    }

    expected = set(audit.FORMAL_GROUPS)
    assert set(runner.FORMAL_GROUPS) == expected
    assert registered["audit_formal_gradient_spectral_release"] == expected
    assert registered["run_formal_gradient_spectral_refresh"] == expected


def test_stage_plan_derived_groups_match_audit_and_upgrade_precedes_weighted() -> None:
    runner = load_runner()
    stages = runner.build_stage_plan(repo_root=REPO_ROOT, python_exe=PY314, jobs=1)
    audit_path = REPO_ROOT / "scripts/core_workflow/audit_formal_gradient_spectral_release.py"
    spec = importlib.util.spec_from_file_location("formal_audit_for_stage_derivation", audit_path)
    assert spec is not None and spec.loader is not None
    audit = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(audit)

    assert Counter(runner.derive_formal_group_identities(stages, REPO_ROOT)) == Counter(audit.FORMAL_GROUPS)
    registry = json.loads((REPO_ROOT / "scripts/core_workflow/step_registry.json").read_text())
    derived = runner.derive_formal_group_identities(stages, REPO_ROOT)
    for utility in registry["utilities"]:
        if utility["step_id"] in {
            "audit_formal_gradient_spectral_release",
            "run_formal_gradient_spectral_refresh",
        }:
            assert Counter(
                (row["dataset"], row["group_root"], row["atlas"])
                for row in utility["formal_groups"]
            ) == Counter(derived)
    weighted = next(stage for stage in stages if stage["name"] == "hcp_weighted_branch")
    commands = runner._stage_commands(weighted)
    assert commands[0][1].endswith("upgrade_retained_hcp_formal_provenance.py")
    assert commands[1][1].endswith("05_05_build_weighted_fisherz_branch.py")


def test_derived_group_identities_expose_extra_and_duplicate_stages() -> None:
    runner = load_runner()
    stages = runner.build_stage_plan(repo_root=REPO_ROOT, python_exe=PY314, jobs=1)
    baseline = runner.derive_formal_group_identities(stages, REPO_ROOT)
    assert len(baseline) == 9

    duplicate = [*stages, dict(next(stage for stage in stages if stage["name"] == "lynch20_group"))]
    duplicate_identities = runner.derive_formal_group_identities(duplicate, REPO_ROOT)
    assert len(duplicate_identities) == 10
    assert duplicate_identities.count(("outputs_hcp", "lynch2024_20net_nsi043_full_k3_normalized_wta", "lynch2024_20net")) == 2
    assert Counter(duplicate_identities) != Counter(baseline)

    duplicate_consumer = [
        *stages,
        dict(next(stage for stage in stages if stage["name"] == "lynch20_overview")),
    ]
    duplicate_consumer_identities = runner.derive_formal_group_identities(
        duplicate_consumer, REPO_ROOT
    )
    assert duplicate_consumer_identities.count(
        ("outputs_hcp", "lynch2024_20net_nsi043_full_k3_normalized_wta", "lynch2024_20net")
    ) == 2

    extra_root = REPO_ROOT / "outputs_hcp/work/group_atlas/unplanned_full_k3"
    extra_command = list(next(stage for stage in stages if stage["name"] == "hcp_routea_overviews")["command"])
    extra_command[extra_command.index("--group-root") + 1] = str(extra_root)
    extra = [*stages, {"name": "unplanned_overview", "command": extra_command}]
    extra_identities = runner.derive_formal_group_identities(extra, REPO_ROOT)
    assert ("outputs_hcp", "unplanned_full_k3", "kong2022") in extra_identities
    assert Counter(extra_identities) != Counter(baseline)


def test_launch_reuses_complete_retained_validation_before_popen(
    tmp_path: Path, monkeypatch
) -> None:
    runner = load_runner()
    fixture_path = REPO_ROOT / "tests/core_workflow/test_upgrade_retained_hcp_formal_provenance.py"
    fixture_spec = importlib.util.spec_from_file_location("retained_fixture_for_preflight", fixture_path)
    assert fixture_spec is not None and fixture_spec.loader is not None
    fixture_module = importlib.util.module_from_spec(fixture_spec)
    fixture_spec.loader.exec_module(fixture_module)
    _subjects_manifest, manifests, _binaries = fixture_module.build_fixture(tmp_path)
    payload = json.loads(manifests[0].read_text())
    missing = tmp_path / "outputs_hcp/inputs/consumed_missing.npy"
    payload["actual_consumed_input_paths"]["fixture"] = str(missing)
    manifests[0].write_text(json.dumps(payload))
    called = False
    monkeypatch.setattr(runner, "required_preexisting_paths", lambda _root: [])
    monkeypatch.setattr(
        runner,
        "build_stage_plan",
        lambda **_kwargs: [
            {"name": f"stage_{index}", "command": [str(PY314), str(SCRIPT)]}
            for index in range(19)
        ],
    )
    def fake_popen(*_args, **_kwargs):
        nonlocal called
        called = True

    monkeypatch.setattr(runner.subprocess, "Popen", fake_popen)

    with pytest.raises(FileNotFoundError, match="actual_consumed_input_paths"):
        runner.launch(repo_root=tmp_path, run_id="formal", jobs=1)
    assert called is False


def test_launch_forwards_worker_command_and_writes_detached_paths(tmp_path, monkeypatch) -> None:
    runner = load_runner()
    observed: dict[str, object] = {}

    monkeypatch.setattr(runner, "preflight", lambda *_args: None)

    class Process:
        pid = 24680

    def fake_popen(command, **kwargs):
        observed["command"] = list(command)
        observed["kwargs"] = kwargs
        return Process()

    monkeypatch.setattr(runner.subprocess, "Popen", fake_popen)

    result = runner.launch(repo_root=tmp_path, run_id="formal.v1", jobs=1)

    command = observed["command"]
    assert _option(command, "--repo-root") == str(tmp_path.resolve())
    assert _option(command, "--run-id") == "formal.v1"
    assert observed["kwargs"]["cwd"] == tmp_path.resolve()
    assert all(observed["kwargs"]["env"][name] == "1" for name in runner.THREAD_ENV)
    assert result["status"] == "launched"
    paths = runner.run_paths(tmp_path, "formal.v1")
    assert command == runner.build_worker_command(
        script_path=SCRIPT,
        repo_root=tmp_path,
        run_id="formal.v1",
        jobs=1,
    )
    for key, path in paths.items():
        assert result[f"{key}_path"] == str(path)
    assert json.loads(paths["status"].read_text())["status"] == "launching"
    assert paths["pid"].read_text() == "24680\n"


@pytest.mark.parametrize("collision_key", ["status", "stdout", "stderr", "pid", "audit"])
def test_launch_rejects_any_detached_record_collision(
    tmp_path, monkeypatch, collision_key: str
) -> None:
    runner = load_runner()
    monkeypatch.setattr(runner, "preflight", lambda *_args: None)
    paths = runner.run_paths(tmp_path, "formal")
    collision = paths[collision_key]
    collision.parent.mkdir(parents=True)
    collision.write_text("existing\n")
    called = False

    def fake_popen(*_args, **_kwargs):
        nonlocal called
        called = True

    monkeypatch.setattr(runner.subprocess, "Popen", fake_popen)

    with pytest.raises(FileExistsError, match="Run record collision") as exc_info:
        runner.launch(repo_root=tmp_path, run_id="formal", jobs=1)

    assert str(collision) in str(exc_info.value)
    assert called is False
