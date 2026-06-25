from __future__ import annotations

from pathlib import Path

import numpy as np

from scripts.core_workflow.lib.iowa_hcp500 import (
    SMOKE_SUBJECT,
    discover_sessions,
    materialize_hippunfold_input,
    validate_routea_labels,
    write_discovery_products,
)


def make_session(
    root: Path,
    subject: str,
    *,
    t1: bool = True,
    t2: bool = True,
    rest_dtseries: bool = True,
    rest_volume: bool = True,
) -> Path:
    base = root / subject / "hcp" / subject
    paths = {
        "t1": base / "T1w" / "T1w_acpc_dc_restore.nii.gz",
        "t2": base / "T1w" / "T2w_acpc_dc_restore.nii.gz",
        "rest_dtseries": (
            base
            / "MNINonLinear"
            / "Results"
            / "REST"
            / "REST_Atlas_MSMAll_hp0_clean_rclean.dtseries.nii"
        ),
        "rest_volume": (
            base
            / "MNINonLinear"
            / "Results"
            / "REST"
            / "REST_hp0_clean_rclean.nii.gz"
        ),
    }
    enabled = {
        "t1": t1,
        "t2": t2,
        "rest_dtseries": rest_dtseries,
        "rest_volume": rest_volume,
    }
    for key, path in paths.items():
        if enabled[key]:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"x")
    return base


def test_discovery_requires_structural_dtseries_and_rest_volume(tmp_path: Path) -> None:
    make_session(tmp_path, SMOKE_SUBJECT)
    make_session(tmp_path, "05012026", rest_volume=False)

    rows = discover_sessions(tmp_path)

    assert [row.subject for row in rows] == [SMOKE_SUBJECT, "05012026"]
    assert rows[0].eligible is True
    assert rows[0].missing == ()
    assert rows[1].eligible is False
    assert rows[1].missing == ("rest_volume",)


def test_discovery_products_are_deterministic(tmp_path: Path) -> None:
    sessions = tmp_path / "sessions"
    make_session(sessions, "06012026")
    make_session(sessions, SMOKE_SUBJECT)

    products = write_discovery_products(
        sessions_root=sessions,
        manifest_dir=tmp_path / "manifests",
    )

    assert products.smoke.read_text(encoding="utf-8").splitlines() == [SMOKE_SUBJECT]
    assert products.full.read_text(encoding="utf-8").splitlines() == [
        SMOKE_SUBJECT,
        "06012026",
    ]
    assert "rest_volume" in products.status.read_text(encoding="utf-8").splitlines()[0]


def test_hippunfold_bridge_links_only_t1w_and_t2w(tmp_path: Path) -> None:
    sessions = tmp_path / "sessions"
    source = make_session(sessions, SMOKE_SUBJECT)
    status = discover_sessions(sessions)[0]

    bridge = materialize_hippunfold_input(status, tmp_path / "bids")

    assert bridge.t1w.resolve() == source / "T1w" / "T1w_acpc_dc_restore.nii.gz"
    assert bridge.t2w.resolve() == source / "T1w" / "T2w_acpc_dc_restore.nii.gz"
    assert bridge.t1w.is_symlink()
    assert bridge.t2w.is_symlink()
    assert len(list((tmp_path / "bids" / f"sub-{SMOKE_SUBJECT}").rglob("*"))) == 3


def test_routea_label_validation_accepts_kong2022_hemisphere_ranges(
    tmp_path: Path,
) -> None:
    subject_dir = tmp_path / f"sub-{SMOKE_SUBJECT}"
    subject_dir.mkdir()
    left = np.resize(np.arange(0, 201), 32492)
    right = np.resize(np.concatenate(([0], np.arange(201, 401))), 32492)
    np.savetxt(subject_dir / "lh_labels.txt", left, fmt="%d")
    np.savetxt(subject_dir / "rh_labels.txt", right, fmt="%d")

    result = validate_routea_labels(subject_dir)

    assert result.valid is True
    assert result.left_count == 32492
    assert result.right_count == 32492
    assert result.errors == ()


def test_routea_label_validation_rejects_wrong_ranges(tmp_path: Path) -> None:
    subject_dir = tmp_path / f"sub-{SMOKE_SUBJECT}"
    subject_dir.mkdir()
    np.savetxt(subject_dir / "lh_labels.txt", np.full(32492, 201), fmt="%d")
    np.savetxt(subject_dir / "rh_labels.txt", np.full(32492, 1), fmt="%d")

    result = validate_routea_labels(subject_dir)

    assert result.valid is False
    assert "lh_out_of_range" in result.errors
    assert "rh_out_of_range" in result.errors
