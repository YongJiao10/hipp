import argparse
import json
from pathlib import Path
import sys
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts/core_workflow"))
from lib.hippomaps_adapter import gradient_preprocessing_settings

import importlib.util

def stage(name):
    spec = importlib.util.spec_from_file_location(name.removesuffix(".py"), ROOT / "scripts/core_workflow" / name)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_cli_alignment_anchor_smoothing_arguments_and_defaults():
    mod_05_01 = stage("05_01_compute_hippomaps_gradients.py")
    p1 = mod_05_01.build_arg_parser().parse_args([])
    assert p1.alignment_anchor_smoothing == "4mm"
    p2 = mod_05_01.build_arg_parser().parse_args(["--alignment-anchor-smoothing", "2mm"])
    assert p2.alignment_anchor_smoothing == "2mm"

    parser_05_02 = stage("05_02_run_group_atlas.py").build_arg_parser()
    parsed_05_02 = parser_05_02.parse_args([])
    assert parsed_05_02.alignment_anchor_smoothing == "4mm"
    
    parsed_05_02_custom = parser_05_02.parse_args(["--alignment-anchor-smoothing", "2mm"])
    assert parsed_05_02_custom.alignment_anchor_smoothing == "2mm"


def test_05_01_select_alignment_anchor_and_validation(monkeypatch):
    mod_05_01 = stage("05_01_compute_hippomaps_gradients.py")
    from types import SimpleNamespace
    
    res_2mm_L = SimpleNamespace(smoothing="2mm", hemi="L", route="pfm")
    res_2mm_R = SimpleNamespace(smoothing="2mm", hemi="R", route="pfm")
    res_4mm_L = SimpleNamespace(smoothing="4mm", hemi="L", route="pfm")
    res_4mm_R = SimpleNamespace(smoothing="4mm", hemi="R", route="pfm")
    
    results = [res_2mm_L, res_2mm_R, res_4mm_L, res_4mm_R]
    
    # 2mm anchor selection
    anchor, desc = mod_05_01.select_alignment_anchor(results, anchor_smoothing="2mm")
    assert anchor is res_2mm_L
    assert desc == "preferred_2mm_hemi_L"
    
    # 4mm anchor selection (default)
    anchor, desc = mod_05_01.select_alignment_anchor(results, anchor_smoothing="4mm")
    assert anchor is res_4mm_L
    assert desc == "preferred_4mm_hemi_L"
    
    # Missing anchor raises
    with pytest.raises(ValueError, match="Alignment requires the newly fitted 2mm hemi L anchor"):
        mod_05_01.select_alignment_anchor([res_4mm_L, res_4mm_R], anchor_smoothing="2mm")

    # main() requires requested anchor in smoothings and L in hemis
    monkeypatch.setattr(sys, "argv", [
        "05_01_compute_hippomaps_gradients.py",
        "--smoothings", "2mm",
        "--hemis", "R",
        "--alignment-anchor-smoothing", "2mm",
    ])
    with pytest.raises(ValueError, match="requires explicit 2mm hemi L input"):
        mod_05_01.main()

    monkeypatch.setattr(sys, "argv", [
        "05_01_compute_hippomaps_gradients.py",
        "--smoothings", "4mm",
        "--hemis", "L",
        "--alignment-anchor-smoothing", "2mm",
    ])
    with pytest.raises(ValueError, match="requires explicit 2mm hemi L input"):
        mod_05_01.main()


def test_05_02_load_hippomaps_gradient_features_anchor_validation(tmp_path):
    mod_05_02 = stage("05_02_run_group_atlas.py")
    run_dir = tmp_path / "pfm" / "2mm" / "hemi_L"
    run_dir.mkdir(parents=True)
    manifest_path = run_dir / "hippomaps_gradient_manifest.json"
    prep = gradient_preprocessing_settings(400, feature_scaling="col_zscore", sparsify_stage="features", sparsity=0.9)
    grad_full = np.arange(50, dtype=np.float32).reshape(10, 5)
    grad_path = run_dir / "hippomaps_gradients_full.npy"
    np.save(grad_path, grad_full)

    base_payload = {
        "status": "ok",
        "route": "pfm",
        "group_fc_shape": [10, 400],
        "feature_count": 400,
        "gradient_preprocessing": prep,
        "gradient_random_state": 20260420,
        "outputs": {"hippomaps_gradients_full": str(grad_path)},
    }

    # 1. Manifest with 4mm anchor, request 2mm -> must reject
    payload_4mm = dict(base_payload)
    payload_4mm["gradient_alignment"] = {"enabled": True, "anchor_smoothing": "4mm", "anchor_hemi": "L"}
    manifest_path.write_text(json.dumps(payload_4mm), encoding="utf-8")
    with pytest.raises(ValueError, match="alignment anchor mismatch"):
        mod_05_02.load_hippomaps_gradient_features(
            gradient_root=tmp_path,
            route="pfm",
            smoothing="2mm",
            hemi="L",
            support_mask=np.ones(10, dtype=bool),
            alignment_anchor_smoothing="2mm",
            gradient_random_state=20260420,
        )

    # 2. Manifest with no gradient_alignment, request 2mm -> must reject
    payload_no_align = dict(base_payload)
    manifest_path.write_text(json.dumps(payload_no_align), encoding="utf-8")
    with pytest.raises(ValueError, match="lacks gradient_alignment declaration"):
        mod_05_02.load_hippomaps_gradient_features(
            gradient_root=tmp_path,
            route="pfm",
            smoothing="2mm",
            hemi="L",
            support_mask=np.ones(10, dtype=bool),
            alignment_anchor_smoothing="2mm",
            gradient_random_state=20260420,
        )

    # 3. Manifest with 2mm anchor, request 4mm -> must reject symmetric mismatch
    payload_2mm = dict(base_payload)
    payload_2mm["gradient_alignment"] = {"enabled": True, "anchor_smoothing": "2mm", "anchor_hemi": "L"}
    manifest_path.write_text(json.dumps(payload_2mm), encoding="utf-8")
    with pytest.raises(ValueError, match="alignment anchor mismatch"):
        mod_05_02.load_hippomaps_gradient_features(
            gradient_root=tmp_path,
            route="pfm",
            smoothing="2mm",
            hemi="L",
            support_mask=np.ones(10, dtype=bool),
            alignment_anchor_smoothing="4mm",
            gradient_random_state=20260420,
        )

    # 4. Manifest with 2mm anchor, request 2mm -> succeeds and returns meta with anchor_smoothing="2mm"
    features, meta = mod_05_02.load_hippomaps_gradient_features(
        gradient_root=tmp_path,
        route="pfm",
        smoothing="2mm",
        hemi="L",
        support_mask=np.ones(10, dtype=bool),
        alignment_anchor_smoothing="2mm",
        gradient_random_state=20260420,
    )
    assert meta["gradient_alignment"]["anchor_smoothing"] == "2mm"


def test_05_02_resume_validation_symmetric_checks(tmp_path):
    mod_05_02 = stage("05_02_run_group_atlas.py")
    manifest_path = tmp_path / "group_atlas_manifest.json"
    summary_path = tmp_path / "group_gradient_feature_summary.json"
    upstream_manifest_path = tmp_path / "hippomaps_gradient_manifest.json"
    prep = gradient_preprocessing_settings(400, feature_scaling="col_zscore", sparsify_stage="features", sparsity=0.9)

    def write_fixtures(up_anchor, sum_anchor):
        up_data = {
            "status": "ok",
            "group_fc_shape": [10, 400],
            "gradient_preprocessing": prep,
            "gradient_random_state": 20260420,
        }
        if up_anchor is not None:
            up_data["gradient_alignment"] = {"enabled": True, "anchor_smoothing": up_anchor, "anchor_hemi": "L"}
        upstream_manifest_path.write_text(json.dumps(up_data), encoding="utf-8")

        sum_data = {
            "gradient_preprocessing": prep,
            "clustering_feature_transform": {"scaling": "none"},
            "hippomaps_gradient_manifest": str(upstream_manifest_path),
        }
        if sum_anchor is not None:
            sum_data["gradient_alignment"] = {"enabled": True, "anchor_smoothing": sum_anchor, "anchor_hemi": "L"}
        summary_path.write_text(json.dumps(sum_data), encoding="utf-8")

        payload = {
            "status": "ok",
            "method": "gradient-spectral",
            "group_gradient_feature_summary": str(summary_path),
            "gradient_source_manifest": str(upstream_manifest_path),
        }
        manifest_path.write_text(json.dumps(payload), encoding="utf-8")
        return payload

    # 1. Upstream 4mm, Summary 4mm, requested 2mm -> reject
    payload = write_fixtures("4mm", "4mm")
    with pytest.raises(ValueError, match="alignment_anchor_smoothing mismatch"):
        mod_05_02._validate_gradient_resume_contract(
            manifest_path,
            payload,
            gradient_options={"alignment_anchor_smoothing": "2mm", "random_state": 20260420},
        )

    # 2. Upstream 2mm, Summary 2mm, requested 4mm -> reject (symmetric check)
    payload = write_fixtures("2mm", "2mm")
    with pytest.raises(ValueError, match="alignment_anchor_smoothing mismatch"):
        mod_05_02._validate_gradient_resume_contract(
            manifest_path,
            payload,
            gradient_options={"alignment_anchor_smoothing": "4mm", "random_state": 20260420},
        )

    # 3. Partial mismatch: summary 2mm, upstream 4mm, requested 2mm -> reject
    payload = write_fixtures("4mm", "2mm")
    with pytest.raises(ValueError, match="alignment_anchor_smoothing mismatch"):
        mod_05_02._validate_gradient_resume_contract(
            manifest_path,
            payload,
            gradient_options={"alignment_anchor_smoothing": "2mm", "random_state": 20260420},
        )

    # 4. Matching 2mm -> succeeds
    payload = write_fixtures("2mm", "2mm")
    mod_05_02._validate_gradient_resume_contract(
        manifest_path,
        payload,
        gradient_options={"alignment_anchor_smoothing": "2mm", "random_state": 20260420},
    )


def test_05_02_build_group_gradient_features_diagnostics_and_anchor_option(tmp_path, monkeypatch):
    mod_05_02 = stage("05_02_run_group_atlas.py")
    fc_path = tmp_path / "fc.npy"
    mask_path = tmp_path / "mask.npy"
    np.save(fc_path, np.zeros((10, 4), dtype=np.float32))
    np.save(mask_path, np.ones(8, dtype=bool)) # shape mismatch (8 vs 10)

    # Mock diffusion gradients so we isolate the anchor FC verification
    monkeypatch.setattr(
        mod_05_02,
        "brainspace_diffusion_gradients",
        lambda *args, **kwargs: (np.zeros((8, 5), dtype=np.float32), np.zeros(5), np.zeros(5)),
    )

    options = {
        "alignment_anchor_smoothing": "2mm",
        "alignment_fc_paths": {"s1": str(fc_path)},
        "alignment_mask_paths": {"s1": str(mask_path)},
        "alignment_coverage_threshold": 0.8,
    }
    with pytest.raises(ValueError, match="2mm-left anchor FC/mask mismatch"):
        mod_05_02.build_group_gradient_features(
            feature_bank={"s1": np.zeros((8, 4), dtype=np.float32)},
            subjects_subset=["s1"],
            support_mask=np.ones(8, dtype=bool),
            gradient_options=options,
        )


def test_load_fixed_cluster_reuse_rejects_anchor_mismatch(tmp_path, monkeypatch):
    mod_05_02 = stage("05_02_run_group_atlas.py")
    monkeypatch.setattr(mod_05_02, "validate_group_atlas_lineage", lambda *a, **k: {})
    source_dir = tmp_path / "source_run"
    source_dir.mkdir()

    source_manifest_path = source_dir / "group_atlas_manifest.json"
    source_summary_path = source_dir / "group_gradient_feature_summary.json"
    np.save(source_dir / "group_labels_final.npy", np.array([1, 2, 3], dtype=np.int32))
    np.save(source_dir / "group_support_mask.npy", np.ones(3, dtype=bool))
    (source_dir / "k_selection_summary.json").write_text(json.dumps({"rows": [{"k": 3}]}), encoding="utf-8")

    source_summary_path.write_text(json.dumps({
        "gradient_alignment": {"anchor_smoothing": "4mm", "anchor_hemi": "L"},
        "clustering_feature_transform": {"scaling": "none"},
    }), encoding="utf-8")

    source_manifest_path.write_text(json.dumps({
        "status": "ok",
        "method": "gradient-ward",
        "atlas": "kong2022",
        "network_taxonomy": "canonical7",
        "smoothing": "2mm",
        "hemi": "L",
        "coverage_threshold": 0.80,
        "chosen_k_final": 3,
        "clustering_recomputed": True,
        "gradient_recomputed": True,
        "group_gradient_feature_summary": str(source_summary_path),
    }), encoding="utf-8")

    target_gradient_feature_summary = {
        "gradient_alignment": {"anchor_smoothing": "2mm", "anchor_hemi": "L"},
        "clustering_feature_transform": {"scaling": "none"},
    }

    with pytest.raises(ValueError, match="Fixed-cluster source alignment anchor mismatch"):
        mod_05_02.load_fixed_cluster_reuse(
            source_run=source_dir,
            target_method="gradient-ward",
            target_atlas="kong2022",
            smoothing="2mm",
            hemi="L",
            coverage_threshold=0.80,
            k_values=[3],
            force_final_k=None,
            default_k=3,
            k_selection_fields=[],
            n_vertices=3,
            support_mask=np.ones(3, dtype=bool),
            target_gradient_feature_summary=target_gradient_feature_summary,
        )
