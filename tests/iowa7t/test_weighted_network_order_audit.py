from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
MODULE_PATH = REPO_ROOT / "scripts" / "iowa7t" / "audit_repair_weighted_network_order.py"


def load_module():
    spec = importlib.util.spec_from_file_location("audit_repair_weighted_network_order", MODULE_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_reorders_weighted_profile_from_recorded_order_to_canonical_order():
    audit = load_module()
    recorded_order = [
        "Auditory",
        "Control",
        "Default",
        "DorsalAttention",
        "Language",
        "Somatomotor",
        "VentralAttention",
        "Visual",
    ]
    old_profile = np.array(
        [[0.50, 0.10, 0.20, 0.00, 0.30, 0.40, 0.90, -0.20]],
        dtype=np.float32,
    )

    repaired = audit.reorder_profile_columns(
        old_profile,
        current_order=recorded_order,
        target_order=audit.CANONICAL8_NETWORK_ORDER,
    )
    rows, dominant = audit.assignment_status_from_profile(repaired, audit.CANONICAL8_NETWORK_ORDER)

    expected = np.array(
        [[0.20, -0.20, 0.40, 0.00, 0.90, 0.10, 0.50, 0.30]],
        dtype=np.float32,
    )
    assert np.allclose(repaired, expected, atol=1e-6)
    assert rows[0]["top1_network"] == "VentralAttention"
    assert rows[0]["top2_network"] == "Auditory"
    assert dominant == ["VentralAttention"]


def test_order_check_rejects_label_map_profile_length_mismatch():
    audit = load_module()
    profile = np.zeros((2, 8), dtype=np.float32)

    try:
        audit.reorder_profile_columns(profile, current_order=["Default"], target_order=audit.CANONICAL8_NETWORK_ORDER)
        raise AssertionError("expected mismatch to fail")
    except ValueError as exc:
        assert "current_order" in str(exc)
