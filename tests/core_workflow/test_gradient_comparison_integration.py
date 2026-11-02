from pathlib import Path
import importlib.util
import sys

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts/core_workflow"))
from lib.gradient_clustering import GRADIENT_METHODS, gradient_method_provenance
from lib.network_ovr import OVR_VARIANTS


def stage(name):
    spec = importlib.util.spec_from_file_location(name.removesuffix(".py"), ROOT / "scripts/core_workflow" / name)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_producer_and_consumers_recognize_new_gradient_methods_without_default_changes():
    atlas = stage("05_02_run_group_atlas.py")
    overview = stage("06_01_build_group_overview.py")
    panels = stage("06_03_build_cluster_cortex_panels.py")
    matrix = stage("07_01_build_group_matrix.py")
    for method in GRADIENT_METHODS:
        assert method in atlas.METHOD_CHOICES
        assert method in overview.METHOD_CHOICES
        assert method in panels.METHOD_CHOICES
        assert method in matrix.METHOD_CHOICES
        assert overview._is_gradient_method(method)
        assert panels._is_gradient_method(method)
    assert atlas.METHODS_DEFAULT == ["intrinsic-spectral", "network-spectral", "wta-top"]
    assert overview.resolve_render_assignment_variants("gradient-ward", "both", atlas="kong2022") == ("fisherz", "normalized_top20_count")
    for variant in OVR_VARIANTS:
        assert overview.resolve_render_assignment_variants("gradient-gmm", variant, atlas="kong2022") == (variant,)


def test_new_methods_have_their_own_consistent_provenance():
    overview = stage("06_01_build_group_overview.py")
    for method in GRADIENT_METHODS:
        source = gradient_method_provenance(method)
        result = overview.overview_source_provenance(method=method, manifest_paths=[Path(f"/pfm/{i}/manifest.json") for i in range(4)], manifests=[source] * 4)
        assert result["surface_affinity"] == source["surface_affinity"]
        if method != "gradient-spectral":
            assert "shifted" not in result["affinity_description"]
    with pytest.raises(ValueError, match="provenance"):
        overview.overview_source_provenance(method="gradient-ward", manifest_paths=[], manifests=[gradient_method_provenance("gradient-gmm")])


def test_fixed_comparison_requests_one_k_and_leaves_automatic_k_unset():
    atlas = stage("05_02_run_group_atlas.py")
    args = atlas.build_arg_parser().parse_args(["--k-selection-mode", "fixed-comparison", "--k-min", "4", "--k-max", "4"])
    assert atlas.comparison_k_values(args) == [4]
    assert atlas.comparison_primary_k(4) == {"chosen_k_final": 4, "automatic_k_final": None, "primary_k_source": "fixed_comparison"}
    args.k_max = 5
    with pytest.raises(ValueError, match="single.*K"):
        atlas.comparison_k_values(args)


def test_fixed_comparison_fails_when_requested_k_cannot_fit_support():
    atlas = stage("05_02_run_group_atlas.py")
    with pytest.raises(ValueError, match="requested K.*supported"):
        atlas.effective_k_values([3], n_support=3, v_min=1, mode="fixed-comparison")
    assert atlas.effective_k_values([3], n_support=12, v_min=8, mode="fixed-comparison")[1] == [3]
    assert atlas.effective_k_values([3], n_support=12, v_min=8, mode="redundancy-aware")[1] == []


def test_run_anchor_changes_invalidate_the_existing_gradient_resume_contract(tmp_path):
    atlas = stage("05_02_run_group_atlas.py")
    fc = tmp_path / "hemi_L/hipp_vertex_to_network_fc.npy"
    run_fc = fc.parent / "runs/run-1" / fc.name
    run_fc.parent.mkdir(parents=True)
    fc.write_bytes(b"fc")
    run_fc.write_bytes(b"old")
    options = {"alignment_fc_paths":{"subject":str(fc)}, "alignment_run_ids":[1]}
    before = atlas.gradient_resume_inputs(gradient_root=tmp_path, cohort=None, variant="36p", gradient_options=options)
    run_fc.write_bytes(b"changed run anchor")
    after = atlas.gradient_resume_inputs(gradient_root=tmp_path, cohort=None, variant="36p", gradient_options=options)
    assert before != after


def test_tmp_comparison_requires_explicit_dataset_and_correct_source_roots(tmp_path):
    atlas = stage("05_02_run_group_atlas.py")
    out = tmp_path / "tmp/comparison"
    roots = [tmp_path / "outputs_hcp/inputs/kong2022/shared_inputs/pfm"]
    assert atlas.validate_dataset_output_contract(out, "outputs_hcp", source_roots=roots, comparison=True, repo_root=tmp_path)[0] == "outputs_hcp"
    with pytest.raises(ValueError, match="explicit.*dataset"):
        atlas.validate_dataset_output_contract(out, "auto", source_roots=roots, comparison=True, repo_root=tmp_path)
    with pytest.raises(ValueError, match="source.*dataset"):
        atlas.validate_dataset_output_contract(out, "outputs_hcp", source_roots=[tmp_path / "outputs_stanford/inputs"], comparison=True, repo_root=tmp_path)
    with pytest.raises(ValueError, match="outside"):
        atlas.validate_dataset_output_contract(out, "outputs_hcp", source_roots=roots, comparison=False, repo_root=tmp_path)
