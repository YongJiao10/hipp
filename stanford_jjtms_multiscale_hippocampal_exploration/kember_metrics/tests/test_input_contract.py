"""Behavioral contracts for the Stanford Kember-method input audit."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest


SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))

from input_contract import audit_stanford_input_contract  # noqa: E402


def _write(path: Path, text: str = "asset") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _write_combo_manifest(path: Path, *, n_vertices: int | None = 512) -> None:
    density_evidence: dict[str, object] = {
        "consumed_asset": "/frozen/surface.surf.gii",
        "surface_fingerprint": {"sha256": "a" * 64},
    }
    if n_vertices is not None:
        density_evidence["n_vertices"] = n_vertices
    _write(
        path,
        json.dumps(
            {
                "density_evidence": density_evidence,
                "label_semantics": {
                    "1": "Anterior/VAN",
                    "2": "Middle/SMN",
                    "3": "Posterior/DMN",
                },
            }
        ),
    )


def _config(root: Path) -> Path:
    config_path = root / "config.json"
    config_path.write_text(
        json.dumps(
            {
                "smoothings_mm": [2],
                "hemispheres": ["L", "R"],
                "rosters": {"all_subjects": "rosters/all_subjects.txt"},
                "paths": {
                    "hippunfold_root": "hippunfold",
                    "hcp151_template_root": "fixed",
                    "template_matching_root": "matching",
                },
            }
        ),
        encoding="utf-8",
    )
    _write(root / "rosters/all_subjects.txt", "1001\n1002\n")
    for hemi in ("L", "R"):
        _write(root / f"fixed/hcp151_2mm_hemi-{hemi}_labels.npy")
        _write(root / f"fixed/hcp151_2mm_hemi-{hemi}_support.npy")
    return config_path


def _write_subject_assets(
    root: Path,
    subject: str,
    *,
    include_template: bool = True,
    include_native_area: bool = True,
) -> None:
    for hemi in ("L", "R"):
        surf = root / "hippunfold" / subject / "hippunfold" / subject / "surf"
        for space, density in (
            ("corobl", "512"),
            ("corobl", "native"),
            ("unfold", "512"),
            ("unfold", "native"),
        ):
            _write(
                surf
                / f"{subject}_hemi-{hemi}_space-{space}_den-{density}_label-hipp_midthickness.surf.gii"
            )
        if include_native_area:
            _write(
                root
                / "hippunfold"
                / subject
                / "hippunfold"
                / subject
                / "metric"
                / f"{subject}_hemi-{hemi}_den-native_label-hipp_surfareacorobl.shape.gii"
            )
        if include_template:
            matching = root / "matching" / "subjects" / subject / "matching" / "2mm" / f"hemi-{hemi}"
            _write(matching / "labels.npy")
            _write(matching / "valid_mask.npy")
            _write_combo_manifest(matching / "combo_manifest.json")


def test_audit_freezes_only_subjects_with_all_retained_branch_inputs(tmp_path: Path) -> None:
    """Fails if a subject missing template labels enters the common Kember cohort."""

    config_path = _config(tmp_path)
    _write_subject_assets(tmp_path, "sub-1001")
    _write_subject_assets(tmp_path, "sub-1002", include_template=False)

    contract = audit_stanford_input_contract(config_path, root=tmp_path)

    assert contract["common_subjects"] == ["sub-1001"]
    assert contract["candidate_subject_count"] == 2
    assert contract["common_subject_count"] == 1
    assert contract["excluded_subjects"]["sub-1002"] == [
        "template_matching:labels",
        "template_matching:valid_mask",
        "template_matching:combo_manifest",
    ]
    assert set(contract["branches"]) == {
        "hcp151_fixed",
        "hcp151_template_matching",
    }


def test_audit_excludes_subject_missing_native_hippunfold_area_metric(
    tmp_path: Path,
) -> None:
    """A 512 label map cannot stand in for the native per-vertex area asset."""

    config_path = _config(tmp_path)
    _write_subject_assets(tmp_path, "sub-1001")
    _write_subject_assets(tmp_path, "sub-1002", include_native_area=False)

    contract = audit_stanford_input_contract(config_path, root=tmp_path)

    assert contract["common_subjects"] == ["sub-1001"]
    assert contract["excluded_subjects"]["sub-1002"] == ["native_area_metric"]


def test_audit_rejects_template_manifest_without_asset_backed_density_evidence(
    tmp_path: Path,
) -> None:
    """Fails if density is inferred without a consumed asset and vertex-count record."""

    config_path = _config(tmp_path)
    _write_subject_assets(tmp_path, "sub-1001")
    _write_subject_assets(tmp_path, "sub-1002")
    bad_manifest = (
        tmp_path
        / "matching/subjects/sub-1002/matching/2mm/hemi-L/combo_manifest.json"
    )
    _write_combo_manifest(bad_manifest, n_vertices=None)

    with pytest.raises(ValueError, match="asset-backed density evidence"):
        audit_stanford_input_contract(config_path, root=tmp_path)
