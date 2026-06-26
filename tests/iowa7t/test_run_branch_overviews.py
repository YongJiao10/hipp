from pathlib import Path

from scripts.iowa7t.run_branch_downstream import resolve_branch_job
from scripts.iowa7t.run_branch_overviews import resolve_branch_overview_paths


def test_branch_overview_paths_are_direct_and_cache_scoped(tmp_path: Path) -> None:
    job = resolve_branch_job("06112026", "multi_echo", data_root=tmp_path / "data", output_root=tmp_path / "hipp_outputs")

    paths = resolve_branch_overview_paths(job)

    assert paths.cache_root == tmp_path / "hipp_outputs" / "runs" / "cache" / "iowa7t_branch_individual_overviews" / "sub-06112026_multi_echo"
    assert paths.work_root == paths.cache_root / "work"
    assert paths.overview_root == paths.cache_root / "overviews"
    assert paths.compat_shared_root == paths.cache_root / "inputs" / "routea"
    assert paths.compat_surface_root == paths.cache_root / "inputs" / "surface"
    assert paths.final_gradient_root == job.branch_root / "atlas" / "gradient"
    assert paths.final_wta_root == job.branch_root / "atlas" / "wta"
    assert paths.final_gradient_figures == job.branch_root / "figures" / "gradient"
    assert paths.final_wta_figures == job.branch_root / "figures" / "wta"
    assert f"multi_echo/sub-{job.subject}" not in str(paths.final_gradient_root)
