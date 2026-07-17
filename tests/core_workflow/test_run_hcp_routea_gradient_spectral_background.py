from __future__ import annotations

import importlib.util
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
MODULE_PATH = REPO_ROOT / "scripts" / "core_workflow" / "run_hcp_routea_gradient_spectral_background.py"


def load_module():
    spec = importlib.util.spec_from_file_location("run_hcp_routea_gradient_spectral_background", MODULE_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_stage_plan_runs_only_routea_8_and_17net_gradient_outputs() -> None:
    module = load_module()
    stages = module.build_stage_plan(
        repo_root=REPO_ROOT,
        python_exe=Path("/home/yojiao/.conda/envs/py314/bin/python"),
        jobs=1,
    )

    assert [stage["name"] for stage in stages] == [
        "routea_canonical8_group",
        "routea_canonical8_overviews",
        "routea_family17_group",
        "routea_family17_overviews",
    ]
    joined = "\n".join(" ".join(stage["command"]) for stage in stages)
    assert joined.count("--methods gradient-spectral") == 4
    assert joined.count("--k-min 2 --k-max 8 --default-k 3") == 2
    assert joined.count("--assignment-variant both") == 2
    assert "--network-taxonomy canonical8" in joined
    assert "--network-taxonomy family17" in joined
    assert "--route-tag routea_17net" in joined
    assert "ctxreg" not in joined and "ctxdrop" not in joined
    assert "--gradient-components" not in joined


def test_expected_overviews_are_exactly_fisherz_and_count_for_both_taxonomies() -> None:
    module = load_module()

    got = [path.relative_to(REPO_ROOT).as_posix() for path in module.expected_overviews(REPO_ROOT)]

    assert got == [
        "outputs_hcp/present/overviews/kong2022_gradient-spectral_group_overview_g1g2g3_routea_fisherz.png",
        "outputs_hcp/present/overviews/kong2022_gradient-spectral_group_overview_g1g2g3_routea_count.png",
        "outputs_hcp/present/overviews/kong2022_gradient-spectral_group_overview_g1g2g3_routea_17net_fisherz.png",
        "outputs_hcp/present/overviews/kong2022_gradient-spectral_group_overview_g1g2g3_routea_17net_count.png",
    ]


def test_launcher_defaults_to_one_job() -> None:
    module = load_module()
    args = module.build_arg_parser().parse_args(["--launch"])
    assert args.jobs == 1


def test_worker_command_preserves_explicit_repo_root() -> None:
    module = load_module()
    command = module.build_worker_command(
        script_path=MODULE_PATH,
        repo_root=Path("/tmp/explicit-repo"),
        run_id="release-test",
        jobs=1,
    )
    joined = " ".join(command)
    assert "--repo-root /tmp/explicit-repo" in joined


def test_release_runner_rejects_parallel_jobs_and_unsafe_run_ids() -> None:
    module = load_module()
    for run_id, jobs in (("safe", 2), ("../escape", 1), ("a/b", 1)):
        try:
            module.validate_release_settings(run_id=run_id, jobs=jobs)
        except ValueError:
            pass
        else:
            raise AssertionError(f"release settings must be rejected: {run_id=} {jobs=}")
