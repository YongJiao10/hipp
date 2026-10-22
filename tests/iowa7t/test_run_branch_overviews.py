from pathlib import Path

from scripts.iowa7t.run_branch_downstream import resolve_branch_job
from scripts.iowa7t.run_branch_overviews import METHODS, build_arg_parser, resolve_branch_overview_paths


def test_branch_overview_paths_are_direct_and_cache_scoped(tmp_path: Path) -> None:
    job = resolve_branch_job("06112026", "multi_echo", data_root=tmp_path / "data", output_root=tmp_path / "hipp_outputs")

    paths = resolve_branch_overview_paths(job)

    assert paths.cache_root == tmp_path / "hipp_outputs" / "runs" / "cache" / "iowa7t_branch_individual_overviews" / "sub-06112026_multi_echo"
    assert paths.work_root == paths.cache_root / "work"
    assert paths.overview_root == paths.cache_root / "overviews"
    assert paths.compat_shared_root == paths.cache_root / "inputs" / "pfm"
    assert paths.compat_surface_root == paths.cache_root / "inputs" / "surface"
    assert paths.final_gradient_root == job.branch_root / "atlas" / "gradient"
    assert paths.final_wta_root == job.branch_root / "atlas" / "wta"
    assert paths.final_gradient_figures == job.branch_root / "figures" / "gradient"
    assert paths.final_wta_figures == job.branch_root / "figures" / "wta"
    assert f"multi_echo/sub-{job.subject}" not in str(paths.final_gradient_root)


def test_family17_branch_overview_paths_use_isolated_cache(tmp_path: Path) -> None:
    job = resolve_branch_job("06112026", "multi_echo", data_root=tmp_path / "data", output_root=tmp_path / "hipp_outputs")

    paths = resolve_branch_overview_paths(job, network_column="family17")

    assert paths.cache_root == tmp_path / "hipp_outputs" / "runs" / "cache" / "iowa7t_branch_individual_overviews_family17" / "sub-06112026_multi_echo"
    assert paths.final_gradient_figures == job.branch_root / "figures" / "gradient"

def test_default_branch_overview_contract_is_canonical7(tmp_path: Path) -> None:
    job = resolve_branch_job("06112026", "multi_echo", data_root=tmp_path / "data", output_root=tmp_path / "hipp_outputs")

    paths = resolve_branch_overview_paths(job)
    args = build_arg_parser().parse_args(["--job", "06112026:multiband"])

    assert paths.cache_root.name == "sub-06112026_multi_echo"
    assert paths.cache_root.parent.name == "iowa7t_branch_individual_overviews"
    assert args.network_column == "canonical7"


def test_canonical8_branch_overview_is_explicitly_isolated(tmp_path: Path) -> None:
    job = resolve_branch_job("06112026", "multi_echo", data_root=tmp_path / "data", output_root=tmp_path / "hipp_outputs")

    paths = resolve_branch_overview_paths(job, network_column="canonical8")

    assert paths.cache_root.parent.name == "iowa7t_branch_individual_overviews_canonical8"

def test_branch_overview_methods_include_only_canonical_wta() -> None:
    assert "gradient_weighted_fisherz" in METHODS
    assert {
        method for method in METHODS if method.startswith("wta_")
    } == {
        "wta_normalized_top20_count",
    }


def test_branch_overview_parser_accepts_family17_network_column() -> None:
    args = build_arg_parser().parse_args(["--job", "06112026:multiband", "--network-column", "family17"])

    assert args.network_column == "family17"


def test_branch_overview_parser_accepts_noxcpd_variant() -> None:
    args = build_arg_parser().parse_args(["--variant", "noxcpd", "--job", "06112026:multiband"])

    assert args.variant == "noxcpd"


def test_copy_file_payload_propagates_copy2_metadata_errors(tmp_path: Path, monkeypatch) -> None:
    import pytest
    from scripts.iowa7t import run_branch_overviews

    src = tmp_path / "src.txt"
    dst = tmp_path / "out" / "dst.txt"
    src.write_text("payload", encoding="utf-8")

    def raise_permission_error(src_path, dst_path):
        raise PermissionError("metadata denied")

    monkeypatch.setattr(run_branch_overviews.shutil, "copy2", raise_permission_error)

    with pytest.raises(PermissionError, match="metadata denied"):
        run_branch_overviews.copy_file_payload(src, dst, overwrite=True)

    assert not dst.exists()


def test_copy_tree_payload_skips_notebook_checkpoint_artifacts(tmp_path: Path) -> None:
    from scripts.iowa7t import run_branch_overviews

    src = tmp_path / "src"
    dst = tmp_path / "dst"
    (src / ".ipynb_checkpoints").mkdir(parents=True)
    (src / ".ipynb_checkpoints" / "bad-checkpoint.json").write_text("bad", encoding="utf-8")
    (src / "valid.json").write_text("ok", encoding="utf-8")

    run_branch_overviews.copy_tree_payload(src, dst, overwrite=True)

    assert (dst / "valid.json").read_text(encoding="utf-8") == "ok"
    assert not (dst / ".ipynb_checkpoints").exists()
