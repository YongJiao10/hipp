"""Explicit Stanford input audit for the Kember-method area workflow.

The audit only probes paths named by the unified workflow configuration.
It freezes a common cohort for fixed HCP151 and individual template matching
without recursively discovering data.
"""

from __future__ import annotations

import json
import re
from collections.abc import Mapping
from pathlib import Path
from typing import Any


_SHA256 = re.compile(r"^[0-9a-f]{64}$")


def _resolve(root: Path, value: object) -> Path:
    path = Path(str(value))
    return path if path.is_absolute() else root / path


def _normalise_subject(value: str) -> str:
    value = value.strip()
    if value.lower().startswith("sub-"):
        value = value[4:]
    if not value.isdigit():
        raise ValueError(f"invalid subject identifier: {value!r}")
    return f"sub-{int(value):04d}"


def _read_subjects(path: Path) -> list[str]:
    if not path.is_file():
        raise FileNotFoundError(f"configured all-subjects roster is missing: {path}")
    subjects: list[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or stripped.lower() == "subject":
            continue
        subjects.append(_normalise_subject(stripped.split()[0]))
    if not subjects:
        raise ValueError(f"configured all-subjects roster is empty: {path}")
    if len(subjects) != len(set(subjects)):
        raise ValueError(f"configured all-subjects roster has duplicate IDs: {path}")
    return subjects


def _require_density_evidence(path: Path) -> int:
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid template combo manifest: {path}") from exc
    if not isinstance(manifest, Mapping):
        raise ValueError(f"invalid template combo manifest: {path}")
    density = manifest.get("density_evidence")
    if not isinstance(density, Mapping):
        raise ValueError(f"template combo manifest lacks asset-backed density evidence: {path}")
    consumed_asset = density.get("consumed_asset")
    n_vertices = density.get("n_vertices")
    fingerprint = density.get("surface_fingerprint")
    sha256 = fingerprint.get("sha256") if isinstance(fingerprint, Mapping) else None
    if (
        not isinstance(consumed_asset, str)
        or not consumed_asset
        or not isinstance(n_vertices, int)
        or n_vertices <= 0
        or not isinstance(sha256, str)
        or not _SHA256.fullmatch(sha256)
    ):
        raise ValueError(f"template combo manifest lacks asset-backed density evidence: {path}")
    return n_vertices


def _surface_path(
    surface_root: Path,
    subject: str,
    hemisphere: str,
    *,
    space: str,
    density: str,
) -> Path:
    return (
        surface_root
        / subject
        / "hippunfold"
        / subject
        / "surf"
        / f"{subject}_hemi-{hemisphere}_space-{space}_den-{density}_label-hipp_midthickness.surf.gii"
    )


def _native_area_metric_path(
    surface_root: Path, subject: str, hemisphere: str
) -> Path:
    return (
        surface_root
        / subject
        / "hippunfold"
        / subject
        / "metric"
        / f"{subject}_hemi-{hemisphere}_den-native_label-hipp_surfareacorobl.shape.gii"
    )


def _unique(values: list[str]) -> list[str]:
    return list(dict.fromkeys(values))


def audit_stanford_input_contract(
    config_path: Path, *, root: Path
) -> dict[str, Any]:
    """Audit explicit assets and freeze the cross-branch Stanford cohort.

    The returned JSON-serializable contract treats missing retained-branch
    assets as exclusions.
    """

    config = json.loads(config_path.read_text(encoding="utf-8"))
    if not isinstance(config, Mapping):
        raise ValueError(f"input configuration must be an object: {config_path}")
    paths = config.get("paths")
    rosters = config.get("rosters")
    smoothings = config.get("smoothings_mm")
    hemispheres = config.get("hemispheres")
    if not isinstance(paths, Mapping) or not isinstance(rosters, Mapping):
        raise ValueError("input configuration needs paths and rosters objects")
    if not isinstance(smoothings, list) or not isinstance(hemispheres, list):
        raise ValueError("input configuration needs smoothing and hemisphere lists")
    required_paths = (
        "hippunfold_root",
        "hcp151_template_root",
        "template_matching_root",
    )
    missing_paths = [name for name in required_paths if name not in paths]
    if missing_paths:
        raise ValueError(f"input configuration lacks path settings: {missing_paths}")
    if "all_subjects" not in rosters:
        raise ValueError("input configuration lacks rosters.all_subjects")

    surface_root = _resolve(root, paths["hippunfold_root"])
    fixed_root = _resolve(root, paths["hcp151_template_root"])
    matching_root = _resolve(root, paths["template_matching_root"]) / "subjects"
    subjects = _read_subjects(_resolve(root, rosters["all_subjects"]))
    combinations = [(int(smoothing), str(hemisphere)) for smoothing in smoothings for hemisphere in hemispheres]

    fixed_assets: list[str] = []
    for smoothing, hemisphere in combinations:
        for kind in ("labels", "support"):
            path = fixed_root / f"hcp151_{smoothing}mm_hemi-{hemisphere}_{kind}.npy"
            if not path.is_file():
                raise FileNotFoundError(f"required fixed HCP151 asset is missing: {path}")
            fixed_assets.append(str(path))

    common_subjects: list[str] = []
    excluded_subjects: dict[str, list[str]] = {}
    density_vertex_counts: set[int] = set()
    for subject in subjects:
        missing: list[str] = []
        for smoothing, hemisphere in combinations:
            required_subject_assets = (
                (
                    "surface_midthickness_512",
                    _surface_path(
                        surface_root,
                        subject,
                        hemisphere,
                        space="corobl",
                        density="512",
                    ),
                ),
                (
                    "native_surface_midthickness",
                    _surface_path(
                        surface_root,
                        subject,
                        hemisphere,
                        space="corobl",
                        density="native",
                    ),
                ),
                (
                    "unfold_surface_512",
                    _surface_path(
                        surface_root,
                        subject,
                        hemisphere,
                        space="unfold",
                        density="512",
                    ),
                ),
                (
                    "native_unfold_surface",
                    _surface_path(
                        surface_root,
                        subject,
                        hemisphere,
                        space="unfold",
                        density="native",
                    ),
                ),
                (
                    "native_area_metric",
                    _native_area_metric_path(surface_root, subject, hemisphere),
                ),
            )
            for name, path in required_subject_assets:
                if not path.is_file():
                    missing.append(name)

            matching = matching_root / subject / "matching" / f"{smoothing}mm" / f"hemi-{hemisphere}"
            label = matching / "labels.npy"
            valid_mask = matching / "valid_mask.npy"
            combo_manifest = matching / "combo_manifest.json"
            for name, path in (
                ("template_matching:labels", label),
                ("template_matching:valid_mask", valid_mask),
                ("template_matching:combo_manifest", combo_manifest),
            ):
                if not path.is_file():
                    missing.append(name)
            if label.is_file() and valid_mask.is_file() and combo_manifest.is_file():
                density_vertex_counts.add(_require_density_evidence(combo_manifest))

        missing = _unique(missing)
        if missing:
            excluded_subjects[subject] = missing
        else:
            common_subjects.append(subject)

    return {
        "schema_version": 1,
        "candidate_subject_count": len(subjects),
        "common_subject_count": len(common_subjects),
        "common_subjects": common_subjects,
        "excluded_subjects": excluded_subjects,
        "combinations_per_subject": len(combinations),
        "branches": {
            "hcp151_fixed": {
                "status": "available",
                "global_asset_count": len(fixed_assets),
            },
            "hcp151_template_matching": {
                "status": "available",
                "asset_backed_density_vertex_counts": sorted(density_vertex_counts),
            },
        },
    }
