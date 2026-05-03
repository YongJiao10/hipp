#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path
from tempfile import TemporaryDirectory

import nibabel as nib
import numpy as np
from nibabel.gifti import GiftiDataArray, GiftiImage

from run_merged_canonical_group_atlas import build_subject_surface_consensus
from run_t1only_spectral_fast_rerun import enforce_routea_path_compliance

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "scripts" / "cortex"))
from run_cortex_kong2022_route_a_reference_batch import validate_routea_path


def _write_surface(path: Path, coords: np.ndarray, faces: np.ndarray) -> None:
    img = GiftiImage(
        darrays=[
            GiftiDataArray(
                data=np.asarray(coords, dtype=np.float32),
                intent="NIFTI_INTENT_POINTSET",
            ),
            GiftiDataArray(
                data=np.asarray(faces, dtype=np.int32),
                intent="NIFTI_INTENT_TRIANGLE",
            ),
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    nib.save(img, str(path))


def _with_fake_surface_map(mapping: dict[tuple[str, str], Path]):
    import run_merged_canonical_group_atlas as atlas_mod

    original = atlas_mod.infer_surface_path

    def fake_infer_surface_path(subject: str, hemi: str, batch_root: Path | None = None) -> Path:
        key = (subject, hemi)
        if key not in mapping:
            raise FileNotFoundError(f"missing fake mapping for {key}")
        return mapping[key]

    atlas_mod.infer_surface_path = fake_infer_surface_path
    return atlas_mod, original


def test_subject_surface_consensus_passes_when_topology_and_axis_match() -> None:
    with TemporaryDirectory() as td:
        root = Path(td)
        faces = np.asarray([[0, 1, 2], [1, 2, 3]], dtype=np.int32)
        coords_a = np.asarray(
            [[0.0, 3.0, 0.0], [0.0, 2.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 0.0]],
            dtype=np.float32,
        )
        coords_b = np.asarray(
            [[0.1, 3.1, 0.0], [0.1, 2.1, 0.0], [0.1, 1.1, 0.0], [0.1, 0.1, 0.0]],
            dtype=np.float32,
        )
        sa = root / "sub-a_L.surf.gii"
        sb = root / "sub-b_L.surf.gii"
        _write_surface(sa, coords_a, faces)
        _write_surface(sb, coords_b, faces)
        atlas_mod, original = _with_fake_surface_map({("a", "L"): sa, ("b", "L"): sb})
        try:
            surfaces, audit, axis = build_subject_surface_consensus(["a", "b"], ["L"], Path("/unused"))
            assert surfaces["L"] == sa, surfaces
            assert int(audit["L"]["topology_consistent_subjects"]) == 2, audit
            assert int(audit["L"]["long_axis_order_consistent_subjects"]) == 2, audit
            assert axis["L"].shape == (4,), axis["L"]
        finally:
            atlas_mod.infer_surface_path = original


def test_subject_surface_consensus_fails_on_topology_mismatch() -> None:
    with TemporaryDirectory() as td:
        root = Path(td)
        coords = np.asarray(
            [[0.0, 3.0, 0.0], [0.0, 2.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 0.0]],
            dtype=np.float32,
        )
        faces_a = np.asarray([[0, 1, 2], [1, 2, 3]], dtype=np.int32)
        faces_b = np.asarray([[0, 1, 3], [1, 2, 3]], dtype=np.int32)
        sa = root / "sub-a_L.surf.gii"
        sb = root / "sub-b_L.surf.gii"
        _write_surface(sa, coords, faces_a)
        _write_surface(sb, coords, faces_b)
        atlas_mod, original = _with_fake_surface_map({("a", "L"): sa, ("b", "L"): sb})
        try:
            try:
                build_subject_surface_consensus(["a", "b"], ["L"], Path("/unused"))
                raise AssertionError("expected topology mismatch failure")
            except RuntimeError as exc:
                assert "topology mismatch" in str(exc), exc
                assert "sub-b" in str(exc), exc
        finally:
            atlas_mod.infer_surface_path = original


def test_subject_surface_consensus_fails_on_long_axis_mismatch() -> None:
    with TemporaryDirectory() as td:
        root = Path(td)
        faces = np.asarray([[0, 1, 2], [1, 2, 3]], dtype=np.int32)
        coords_a = np.asarray(
            [[0.0, 3.0, 0.0], [0.0, 2.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 0.0]],
            dtype=np.float32,
        )
        coords_b = np.asarray(
            [[0.0, 2.0, 0.0], [0.0, 3.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 0.0]],
            dtype=np.float32,
        )
        sa = root / "sub-a_L.surf.gii"
        sb = root / "sub-b_L.surf.gii"
        _write_surface(sa, coords_a, faces)
        _write_surface(sb, coords_b, faces)
        atlas_mod, original = _with_fake_surface_map({("a", "L"): sa, ("b", "L"): sb})
        try:
            try:
                build_subject_surface_consensus(["a", "b"], ["L"], Path("/unused"))
                raise AssertionError("expected long-axis mismatch failure")
            except RuntimeError as exc:
                assert "long-axis order mismatch" in str(exc), exc
                assert "sub-b" in str(exc), exc
        finally:
            atlas_mod.infer_surface_path = original


def test_routea_path_guards_reject_routeb_paths() -> None:
    bad = REPO_ROOT / "outputs_experiments" / "hippunfold_recommended_t1w_20260423" / "kong2022_route_b" / "x"
    good = REPO_ROOT / "outputs_experiments" / "hippunfold_recommended_t1w_20260423" / "kong2022" / "manifests" / "y.txt"

    try:
        validate_routea_path(bad, "--subjects-manifest")
        raise AssertionError("expected validate_routea_path failure")
    except RuntimeError as exc:
        assert "--subjects-manifest" in str(exc), exc

    enforce_routea_path_compliance(
        shared_root=good.parent,
        manifest=good,
        group_root_raw=REPO_ROOT / "outputs_migration" / "hipp_group_merged_canonical_atlas",
    )
    try:
        enforce_routea_path_compliance(
            shared_root=bad,
            manifest=good,
            group_root_raw=REPO_ROOT / "outputs_migration" / "hipp_group_merged_canonical_atlas",
        )
        raise AssertionError("expected enforce_routea_path_compliance failure")
    except RuntimeError as exc:
        assert "--shared-root" in str(exc), exc


def main() -> int:
    test_subject_surface_consensus_passes_when_topology_and_axis_match()
    test_subject_surface_consensus_fails_on_topology_mismatch()
    test_subject_surface_consensus_fails_on_long_axis_mismatch()
    test_routea_path_guards_reject_routeb_paths()
    print("ok: subject-surface consensus and routea compliance checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
