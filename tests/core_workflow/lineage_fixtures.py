"""Small, complete on-disk Kong gradient fixtures shared by consumer tests."""
import json
from pathlib import Path
import numpy as np


def make_run(run: Path, *, omit_profile=False, fields=None, with_artifacts=True):
    run.mkdir(parents=True, exist_ok=True)
    manifest = {
        "status": "ok", "atlas": "kong2022", "method": "gradient-spectral",
        "network_taxonomy": "canonical7", "gradient_recomputed": True,
        "clustering_recomputed": True, "chosen_k_final": 3,
        "subjects": ["sub-01", "sub-02"], "n_subjects": 2,
        "smoothing": "2mm", "hemi": "L", "coverage_threshold": .8,
        "density_evidence": {"n_vertices": 3}, **(fields or {}),
    }
    path = run / "group_atlas_manifest.json"
    path.write_text(json.dumps(manifest))
    if not with_artifacts:
        return path
    np.save(run / "group_labels_final.npy", np.array([1, 2, 3], dtype=np.int32))
    np.save(run / "group_support_mask.npy", np.ones(3, dtype=bool))
    np.save(run / "group_functional_atlas_labels_fisherz.npy", np.array([1, 2, 3]))
    (run / "k_selection_summary.json").write_text(json.dumps({"rows": [{"k": 3}]}))
    if not omit_profile:
        np.save(run / "cluster_roi_profile.npy", np.zeros((3, 400), dtype=np.float32))
    gradients = np.arange(15, dtype=np.float32).reshape(3, 5)
    np.save(run / "hippomaps_gradients_full.npy", gradients)
    np.save(run / "group_gradient_features_active.npy", gradients[:, :3])
    np.save(run / "group_mean_fisherz_fc_active.npy", np.zeros((3, 400), dtype=np.float32))
    (run / "gradient_manifest.json").write_text(json.dumps({
        "feature_count": 400, "group_fc_shape": [3, 400],
        "outputs": {"hippomaps_gradients_full": str(run / "hippomaps_gradients_full.npy"),
                    "group_mean_fisherz_fc_active": str(run / "group_mean_fisherz_fc_active.npy")},
    }))
    (run / "group_gradient_feature_summary.json").write_text(json.dumps({
        "hippomaps_gradient_manifest": str(run / "gradient_manifest.json"),
        "hippomaps_gradients_full": str(run / "hippomaps_gradients_full.npy"),
        "gradient_components": [1, 2, 3],
    }))
    return path


