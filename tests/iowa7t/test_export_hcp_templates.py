"""Portable HCP templates must retain their scientific recipe and row identity."""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest
from scipy.io import loadmat


ROOT = Path(__file__).resolve().parents[2]


def module():
    spec = importlib.util.spec_from_file_location(
        "export_hcp_templates", ROOT / "scripts/iowa7t/export_hcp_templates.py"
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def test_direct_wta_competes_against_all_networks_before_selecting_rows():
    mod = module()
    names = ["Default", "Visual", "Somatomotor", "DorsalAttention",
             "VentralAttention", "Control", "Language"]
    scores = np.array([[1, 0, 0, 0, 0, 2, 0],
                       [0, 0, 0, 0, 3, 0, 0],
                       [4, 0, 0, 0, 0, 0, 0],
                       [0, 0, 5, 0, 0, 0, 0]], dtype=np.float32)
    labels = mod.direct_labels(scores, np.array([0, 1, 3, 4]), 6, names)
    assert labels.tolist() == [0, 2, 0, 1, 3, 0]


def test_subject_profile_correlates_the_mean_time_series_and_keeps_empty_rows():
    mod = module()
    x = np.array([1, -1, 1, -1], dtype=np.float32)
    y = np.array([1, 1, -1, -1], dtype=np.float32)
    ts = np.stack([x, y, -x])
    roi = np.stack([x, y, np.ones(4, dtype=np.float32)])
    profile = mod.subject_profile(ts, roi, np.array([True, True, False]),
                                  np.array([1, 1, 2]), 3)
    assert profile.shape == (3, 3)
    np.testing.assert_allclose(profile[0, :2], [0.8813736, 0.8813736], atol=1e-6)
    assert np.isnan(profile[:, 2]).all()
    assert np.isnan(profile[1:]).all()


def test_cohort_mismatch_is_rejected_with_the_exact_source_path(tmp_path):
    mod = module()
    path = tmp_path / "group_atlas_manifest.json"
    path.write_text(json.dumps({"status": "ok", "subjects": ["sub-197348"],
                                "n_subjects": 1}))
    with pytest.raises(ValueError, match="group_atlas_manifest.json"):
        mod.read_source_manifest(path, ["sub-100610", "sub-197348"])


def test_numpy_and_matlab_exports_preserve_rows_and_missing_values(tmp_path):
    mod = module()
    profiles = np.arange(4 * 3 * 400, dtype=np.float32).reshape(4, 3, 400)
    profiles[2, 1, 9] = np.nan
    mod.write_profile_files(tmp_path, profiles, ("Default", "VentralAttention", "Somatomotor"),
                            ("DMN", "VAN", "SMN"), gradient=False)
    with np.load(tmp_path / "template_profiles.npz", allow_pickle=False) as data:
        np.testing.assert_equal(data["profiles"], profiles)
        assert data["row_keys"].tolist() == ["2mm_L", "2mm_R", "4mm_L", "4mm_R"]
        assert data["network_names"].tolist() == ["Default", "VentralAttention", "Somatomotor"]
    mat = loadmat(tmp_path / "template_profiles.mat", simplify_cells=True)
    np.testing.assert_equal(mat["profiles"], profiles)
    assert [str(s).strip() for s in mat["network_names"]] == ["Default", "VentralAttention", "Somatomotor"]
