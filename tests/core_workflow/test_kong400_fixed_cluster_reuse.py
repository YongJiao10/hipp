from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts/core_workflow"))
SPEC = importlib.util.spec_from_file_location("kong_atlas_reuse", ROOT / "scripts/core_workflow/05_02_run_group_atlas.py")
assert SPEC and SPEC.loader
atlas = importlib.util.module_from_spec(SPEC)
sys.modules["kong_atlas_reuse"] = atlas
SPEC.loader.exec_module(atlas)


def test_family17_can_reuse_canonical7_ward_clustering(monkeypatch):
    class ReachedOutputSetup(Exception):
        pass

    def stop_before_writing(*args, **kwargs):
        raise ReachedOutputSetup

    monkeypatch.setattr(Path, "mkdir", stop_before_writing)
    monkeypatch.setattr(sys, "argv", [
        "05_02_run_group_atlas.py", "--route-tag", "pfm_17net",
        "--network-taxonomy", "family17", "--profile", "topsim",
        "--methods", "gradient-ward", "--reuse-fixed-cluster-root", "source",
    ])
    with pytest.raises(ReachedOutputSetup):
        atlas.main()


def test_family17_reuse_preserves_all_source_k_diagnostics(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    # The automatically selected K can differ from the displayed default K.
    rows = [{"k": 2, "selected_final": 1}, {"k": 3, "selected_final": 0}]
    (source / "k_selection_summary.json").write_text(json.dumps({
        "rows": rows, "k_final": 3, "chosen_k_final": 3,
        "automatic_k_final": 2, "primary_k_source": "default_k",
    }))
    copied_rows, primary = atlas.family17_reused_k_diagnostics(source, 3)
    assert copied_rows == rows
    assert primary["automatic_k_final"] == 2
    assert primary["primary_k_source"] == "default_k"


def make_source(tmp_path: Path, feature_count: int = 400) -> Path:
    run = tmp_path / "source" / "gradient-spectral" / "kong2022" / "2mm" / "cov_0p80" / "g1g2g3" / "hemi_L"
    run.mkdir(parents=True)
    np.save(run / "group_labels_final.npy", np.asarray([1, 1, 2, 3], dtype=np.int32))
    np.save(run / "group_support_mask.npy", np.asarray([1, 1, 1, 1], dtype=np.uint8))
    gradient_array = run / "hippomaps_gradients_full.npy"
    np.save(gradient_array, np.arange(12, dtype=np.float32).reshape(4, 3))
    np.save(run / "group_gradient_features_active.npy", np.arange(12, dtype=np.float32).reshape(4, 3))
    np.save(run / "cluster_roi_profile.npy", np.zeros((3, 400), dtype=np.float32))
    fc_array = run / "group_mean_fisherz_fc_active.npy"
    np.save(fc_array, np.arange(4 * feature_count, dtype=np.float32).reshape(4, feature_count))
    preprocessing = atlas.gradient_preprocessing_settings(
        feature_count, feature_scaling="col_zscore", sparsify_stage="features",
    )
    transform = atlas.scale_gradient_features(np.arange(12, dtype=np.float32).reshape(4, 3), scaling="none")[1]
    gradient_manifest = run / "gradient_manifest.json"
    gradient_manifest.write_text(json.dumps({
        "status": "ok", "method": "hippomaps-gradient", "atlas": "kong2022",
        "route": "routea", "smoothing": "2mm", "hemi": "L",
        "feature_taxonomy": "kong2022_roi400", "feature_family": "kong2022_400roi_fc",
        "group_fc_shape": [4, feature_count], "feature_count": feature_count,
        "gradient_shape_full": [4, 3], "gradient_components": [1, 2, 3],
        "selected_components": ["G1", "G2", "G3"], "gradientmaps_kernel": "pearson",
        "gradient_preprocessing": preprocessing,
        "outputs": {"hippomaps_gradients_full": str(gradient_array),
                    "group_mean_fisherz_fc_active": str(fc_array)},
    }))
    gradient_summary = run / "group_gradient_feature_summary.json"
    gradient_summary.write_text(json.dumps({
        "hippomaps_gradient_manifest": str(gradient_manifest),
        "hippomaps_gradients_full": str(gradient_array),
        "group_fc_shape": [4, feature_count], "feature_count": feature_count,
        "feature_family": "group_vertex_to_kong2022_400roi_fc_diffusion_gradients_g1g2g3",
        "gradient_shape_full": [4, 3], "selected_components": ["G1", "G2", "G3"],
        "gradient_components": [1, 2, 3], "smoothing": "2mm", "hemi": "L", "route": "routea",
        "gradient_preprocessing": preprocessing, "clustering_feature_transform": transform,
    }))
    row = {"k": 3, "group_ari_mean": 0.42, "group_ari_se": 0.02, "selected_final": 1, "nontrivial_min_size_ok": 1}
    (run / "k_selection_summary.json").write_text(json.dumps({"rows": [row], "chosen_k_final": 3}))
    manifest = {
        "status": "ok", "method": "gradient-spectral", "atlas": "kong2022",
        "network_taxonomy": "canonical7", "gradient_recomputed": True, "clustering_recomputed": True,
        "density_evidence": {"n_vertices": 4}, "coverage_threshold": 0.8,
        "smoothing": "2mm", "hemi": "L", "chosen_k_final": 3,
        "subjects": ["sub-01", "sub-02"], "n_subjects": 2,
        "group_gradient_feature_summary": str(gradient_summary),
    }
    (run / "group_atlas_manifest.json").write_text(json.dumps(manifest))
    return tmp_path / "source"


def test_kong_reuse_resolves_only_explicit_path(tmp_path: Path) -> None:
    root = make_source(tmp_path)
    run = atlas.resolve_fixed_cluster_source_run(
        source_root=root, method="gradient-spectral", atlas="kong2022",
        smoothing="2mm", hemi="L", coverage_tag="cov_0p80", gradient_components=(1, 2, 3),
    )
    assert run.name == "hemi_L"


def test_kong_reuse_preserves_source_k_diagnostics(tmp_path: Path) -> None:
    root = make_source(tmp_path)
    run = atlas.resolve_fixed_cluster_source_run(
        source_root=root, method="gradient-spectral", atlas="kong2022",
        smoothing="2mm", hemi="L", coverage_tag="cov_0p80", gradient_components=(1, 2, 3),
    )
    k, labels, row, evidence = atlas.load_fixed_cluster_reuse(
        source_run=run, target_method="gradient-spectral", target_atlas="kong2022",
        smoothing="2mm", hemi="L", coverage_threshold=0.8,
        support_mask=np.ones(4, dtype=bool), n_vertices=4, k_values=[2, 3],
        force_final_k=3, default_k=3, k_selection_fields=["k", "group_ari_mean"],
        target_subjects=["01", "02"],
        target_gradient_feature_summary=json.loads(
            (run / "group_gradient_feature_summary.json").read_text(encoding="utf-8")
        ),
    )
    assert k == 3
    assert labels.tolist() == [1, 1, 2, 3]
    assert row["group_ari_mean"] == pytest.approx(0.42)
    assert evidence["gradient_recomputed"] is False


def test_kong_reuse_rejects_seven_feature_gradient_source(tmp_path: Path) -> None:
    root = make_source(tmp_path, feature_count=7)
    run = atlas.resolve_fixed_cluster_source_run(
        source_root=root, method="gradient-spectral", atlas="kong2022",
        smoothing="2mm", hemi="L", coverage_tag="cov_0p80", gradient_components=(1, 2, 3),
    )
    with pytest.raises(ValueError, match="400-ROI"):
        atlas.load_fixed_cluster_reuse(
            source_run=run, target_method="gradient-spectral", target_atlas="kong2022",
            smoothing="2mm", hemi="L", coverage_threshold=0.8,
            support_mask=np.ones(4, dtype=bool), n_vertices=4, k_values=[2, 3],
            force_final_k=3, default_k=3, k_selection_fields=["k"], target_subjects=["01", "02"],
            target_gradient_feature_summary=json.loads(
                (run / "group_gradient_feature_summary.json").read_text(encoding="utf-8")
            ),
        )


def test_kong_reuse_rejects_target_gradient_branch_mismatch(tmp_path: Path) -> None:
    root = make_source(tmp_path)
    run = atlas.resolve_fixed_cluster_source_run(
        source_root=root, method="gradient-spectral", atlas="kong2022",
        smoothing="2mm", hemi="L", coverage_tag="cov_0p80", gradient_components=(1, 2, 3),
    )
    target_summary = json.loads((run / "group_gradient_feature_summary.json").read_text(encoding="utf-8"))
    target_manifest = tmp_path / "target_branch_manifest.json"
    target_manifest_payload = json.loads((run / "gradient_manifest.json").read_text(encoding="utf-8"))
    target_manifest_payload["route"] = "routea_ctxreg"
    target_manifest.write_text(json.dumps(target_manifest_payload), encoding="utf-8")
    target_summary["hippomaps_gradient_manifest"] = str(target_manifest)
    with pytest.raises(ValueError, match="gradient identity mismatch"):
        atlas.load_fixed_cluster_reuse(
            source_run=run, target_method="gradient-spectral", target_atlas="kong2022",
            smoothing="2mm", hemi="L", coverage_threshold=0.8,
            support_mask=np.ones(4, dtype=bool), n_vertices=4, k_values=[2, 3],
            force_final_k=3, default_k=3, k_selection_fields=["k"], target_subjects=["01", "02"],
            target_gradient_feature_summary=target_summary,
        )


def test_kong_reuse_rejects_target_gradient_array_mismatch(tmp_path: Path) -> None:
    root = make_source(tmp_path)
    run = atlas.resolve_fixed_cluster_source_run(
        source_root=root, method="gradient-spectral", atlas="kong2022",
        smoothing="2mm", hemi="L", coverage_tag="cov_0p80", gradient_components=(1, 2, 3),
    )
    target_array = tmp_path / "target_gradients.npy"
    np.save(target_array, np.full((4, 3), 999.0, dtype=np.float32))
    target_summary = json.loads((run / "group_gradient_feature_summary.json").read_text(encoding="utf-8"))
    target_summary["hippomaps_gradients_full"] = str(target_array)
    target_manifest = tmp_path / "target_gradient_manifest.json"
    target_manifest_payload = json.loads(
        (run / "gradient_manifest.json").read_text(encoding="utf-8")
    )
    target_manifest_payload["outputs"]["hippomaps_gradients_full"] = str(target_array)
    target_manifest.write_text(json.dumps(target_manifest_payload), encoding="utf-8")
    target_summary["hippomaps_gradient_manifest"] = str(target_manifest)
    with pytest.raises(ValueError, match="selected gradient array mismatch"):
        atlas.load_fixed_cluster_reuse(
            source_run=run, target_method="gradient-spectral", target_atlas="kong2022",
            smoothing="2mm", hemi="L", coverage_threshold=0.8,
            support_mask=np.ones(4, dtype=bool), n_vertices=4, k_values=[2, 3],
            force_final_k=3, default_k=3, k_selection_fields=["k"], target_subjects=["01", "02"],
            target_gradient_feature_summary=target_summary,
        )


@pytest.mark.parametrize("case", ["missing_summary", "missing_upstream", "mismatched_upstream"])
def test_kong_reuse_rejects_missing_or_mismatched_preprocessing(tmp_path: Path, case: str) -> None:
    root = make_source(tmp_path)
    run = atlas.resolve_fixed_cluster_source_run(
        source_root=root, method="gradient-spectral", atlas="kong2022",
        smoothing="2mm", hemi="L", coverage_tag="cov_0p80", gradient_components=(1, 2, 3),
    )
    summary_path = run / "group_gradient_feature_summary.json"
    source_summary = json.loads(summary_path.read_text(encoding="utf-8"))
    target_summary = dict(source_summary)
    if case == "missing_summary":
        del source_summary["gradient_preprocessing"]
        summary_path.write_text(json.dumps(source_summary), encoding="utf-8")
    else:
        manifest_path = run / "gradient_manifest.json"
        gradient_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if case == "missing_upstream":
            del gradient_manifest["gradient_preprocessing"]
        else:
            gradient_manifest["gradient_preprocessing"]["feature_scaling"] = "none"
        manifest_path.write_text(json.dumps(gradient_manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="gradient_preprocessing|Gradient preprocessing"):
        atlas.load_fixed_cluster_reuse(
            source_run=run, target_method="gradient-spectral", target_atlas="kong2022",
            smoothing="2mm", hemi="L", coverage_threshold=0.8,
            support_mask=np.ones(4, dtype=bool), n_vertices=4, k_values=[2, 3],
            force_final_k=3, default_k=3, k_selection_fields=["k"], target_subjects=["01", "02"],
            target_gradient_feature_summary=target_summary,
        )


def test_kong_rbf_reuse_requires_declared_bandwidth(tmp_path: Path) -> None:
    root = make_source(tmp_path)
    run = atlas.resolve_fixed_cluster_source_run(
        source_root=root, method="gradient-spectral", atlas="kong2022",
        smoothing="2mm", hemi="L", coverage_tag="cov_0p80", gradient_components=(1, 2, 3),
    )
    manifest_path = run / "group_atlas_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["method"] = "gradient-spectral-rbf"
    manifest["gradient_clustering"] = {"rbf_scale": 1.0}
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    summary = json.loads((run / "group_gradient_feature_summary.json").read_text(encoding="utf-8"))
    with pytest.raises(ValueError, match="rbf_scale"):
        atlas.load_fixed_cluster_reuse(
            source_run=run, target_method="gradient-spectral-rbf", target_atlas="kong2022",
            smoothing="2mm", hemi="L", coverage_threshold=0.8,
            support_mask=np.ones(4, dtype=bool), n_vertices=4, k_values=[2, 3],
            force_final_k=3, default_k=3, k_selection_fields=["k"], target_subjects=["01", "02"],
            target_gradient_feature_summary=summary, target_rbf_scale=1.0,
        )


@pytest.mark.parametrize("source_rbf_scale", [0.5, 1.000001])
def test_kong_reuse_rejects_clustering_scaling_and_rbf_scale_mismatch(tmp_path: Path, source_rbf_scale: float) -> None:
    root = make_source(tmp_path)
    run = atlas.resolve_fixed_cluster_source_run(
        source_root=root, method="gradient-spectral", atlas="kong2022",
        smoothing="2mm", hemi="L", coverage_tag="cov_0p80", gradient_components=(1, 2, 3),
    )
    raw_grad = np.arange(12, dtype=np.float32).reshape(4, 3)
    source_summary = json.loads((run / "group_gradient_feature_summary.json").read_text(encoding="utf-8"))
    source_summary["clustering_feature_transform"] = atlas.scale_gradient_features(raw_grad, scaling="none")[1]
    (run / "group_gradient_feature_summary.json").write_text(json.dumps(source_summary), encoding="utf-8")
    target_summary = dict(source_summary)
    target_summary["clustering_feature_transform"] = atlas.scale_gradient_features(raw_grad, scaling="unit_sd")[1]
    with pytest.raises(ValueError, match="clustering_feature_scaling|clustering_feature_transform"):
        atlas.load_fixed_cluster_reuse(
            source_run=run, target_method="gradient-spectral", target_atlas="kong2022",
            smoothing="2mm", hemi="L", coverage_threshold=0.8,
            support_mask=np.ones(4, dtype=bool), n_vertices=4, k_values=[2, 3],
            force_final_k=3, default_k=3, k_selection_fields=["k"], target_subjects=["01", "02"],
            target_gradient_feature_summary=target_summary,
            target_clustering_feature_scaling="unit_sd",
        )

    # Also test rbf_scale mismatch on gradient-spectral-rbf
    rbf_manifest = json.loads((run / "group_atlas_manifest.json").read_text(encoding="utf-8"))
    rbf_manifest["method"] = "gradient-spectral-rbf"
    rbf_manifest["rbf_scale"] = source_rbf_scale
    (run / "group_atlas_manifest.json").write_text(json.dumps(rbf_manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="rbf_scale"):
        atlas.load_fixed_cluster_reuse(
            source_run=run, target_method="gradient-spectral-rbf", target_atlas="kong2022",
            smoothing="2mm", hemi="L", coverage_threshold=0.8,
            support_mask=np.ones(4, dtype=bool), n_vertices=4, k_values=[2, 3],
            force_final_k=3, default_k=3, k_selection_fields=["k"], target_subjects=["01", "02"],
            target_gradient_feature_summary=source_summary,
            target_rbf_scale=1.0,
        )
