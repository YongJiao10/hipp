from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
MODULE_PATH = REPO_ROOT / "scripts" / "iowa7t" / "audit_family17_gradient_overviews.py"


def load_module():
    spec = importlib.util.spec_from_file_location("audit_family17_gradient_overviews", MODULE_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_fisher_profile_status_check_rejects_mismatched_value():
    audit = load_module()
    profile = np.asarray([[0.1, 0.5]], dtype=np.float32)
    status = {"items": [{"cluster_id": 1, "top1_network": "B", "top1_fisher_z": 0.2}]}

    try:
        audit.check_fisher_profile_status(profile, status, ["A", "B"], [1], context="unit")
        raise AssertionError("expected mismatch to fail")
    except RuntimeError as exc:
        assert "Fisher-z mismatch" in str(exc)


def test_count_profile_status_check_accepts_tied_counts_when_status_top_is_max():
    audit = load_module()
    profile = np.asarray([[1.0, 1.0, 0.0]], dtype=np.float32)
    status = {"items": [{"cluster_id": 1, "top1_network": "B", "top1_count": 1}]}

    audit.check_count_profile_status(profile, status, ["A", "B", "C"], [1], context="unit")


def test_count_profile_status_check_rejects_nonmax_status_top():
    audit = load_module()
    profile = np.asarray([[2.0, 1.0, 0.0]], dtype=np.float32)
    status = {"items": [{"cluster_id": 1, "top1_network": "B", "top1_count": 1}]}

    try:
        audit.check_count_profile_status(profile, status, ["A", "B", "C"], [1], context="unit")
        raise AssertionError("expected mismatch to fail")
    except RuntimeError as exc:
        assert "Count mismatch" in str(exc)


def test_fisher_profile_status_check_rejects_all_nan_weighted_profile():
    audit = load_module()
    profile = np.asarray([[np.nan, np.nan]], dtype=np.float32)
    status = {"items": [{"cluster_id": 1, "status": "unassigned", "top1_network": "unassigned", "top1_fisher_z": 0.0}]}

    try:
        audit.check_fisher_profile_status(profile, status, ["A", "B"], [1], context="weighted unit", require_finite=True)
        raise AssertionError("expected all-NaN weighted profile to fail")
    except RuntimeError as exc:
        assert "no finite Fisher-z values" in str(exc)
