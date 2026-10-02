from __future__ import annotations

from pathlib import Path
from typing import Any

from .output_layout import (
    DATASET_KEYS,
    DEFAULT_DATASET,
    REPO_ROOT,
    resolve_hippunfold_batch_root,
    resolve_kong2022_template_root,
)


_TEMPLATE_SURFACE_NAME = (
    "group-average_hemi-{hemi}_space-corobl_den-512_label-hipp_midthickness.surf.gii"
)
_TEMPLATE_SPEC_NAME = "sub-{subject}_den-512_label-hipp_surfaces.spec"

def canonical_dataset_key(value: object | None, *, default: str = DEFAULT_DATASET) -> str:
    """Normalize manifest dataset labels to output_layout dataset keys."""
    if value is None or str(value).strip() == "":
        return default
    key = str(value).strip().lower()
    if key.startswith("outputs_"):
        key = key[len("outputs_") :]
    if key == "iowa":
        key = "iowa_estt"
    if key not in DATASET_KEYS:
        raise ValueError(f"Unknown template-asset dataset: {value!r}")
    return key


def dataset_key_from_path(path: Path | str | None) -> str | None:
    """Return the dataset encoded by an outputs_<dataset> path, if present."""
    if path is None:
        return None
    parts = set(Path(path).resolve().parts)
    for key in DATASET_KEYS:
        if f"outputs_{key}" in parts:
            return key
    return None


def infer_dataset_key(
    manifest: dict[str, Any] | None = None,
    *,
    run_dir: Path | None = None,
    dataset: str | None = None,
) -> str:
    """Infer the target dataset, giving the target run path precedence."""
    if dataset is not None:
        return canonical_dataset_key(dataset)
    path_key = dataset_key_from_path(run_dir)
    if path_key is not None:
        return path_key
    if isinstance(manifest, dict):
        raw_dataset = manifest.get("dataset")
        if raw_dataset is not None and str(raw_dataset).strip() != "":
            return canonical_dataset_key(raw_dataset)
    return DEFAULT_DATASET


def _resolve_manifest_path(raw: object | None) -> Path | None:
    if raw is None:
        return None
    text = str(raw).strip()
    if text == "" or text.lower() in {"none", "null"}:
        return None
    path = Path(text).expanduser()
    return (path if path.is_absolute() else REPO_ROOT / path).resolve()


def _normal_subject(value: object | None) -> str | None:
    if value is None:
        return None
    subject = str(value).strip()
    if subject == "" or subject.lower() in {"none", "null", "subject-consensus"}:
        return None
    subject = subject.removeprefix("sub-")
    return subject or None


def _subject_candidates(manifest: dict[str, Any]) -> list[str]:
    """Return metadata subjects without trusting any source absolute path."""
    values: list[object | None] = [manifest.get("template_subject")]
    subjects = manifest.get("subjects")
    if isinstance(subjects, (list, tuple)):
        values.extend(subjects)
    result: list[str] = []
    for value in values:
        subject = _normal_subject(value)
        if subject is not None and subject not in result:
            result.append(subject)
    return result


def _path_is_equal(left: Path | None, right: Path) -> bool:
    return left is not None and left.resolve() == right.resolve()


def _variant_from_manifest(manifest: dict[str, Any]) -> str:
    return str(manifest.get("preprocessing_variant", "36p")).strip().lower()


def _canonical_template_surface_path(
    *,
    dataset_key: str,
    variant: str,
    hemi: str,
) -> Path:
    hemi_key = str(hemi).strip().upper()
    if hemi_key not in {"L", "R"}:
        raise ValueError(f"Invalid template surface hemisphere: {hemi!r}")
    candidate = (
        resolve_kong2022_template_root(dataset_key, variant=variant)
        / "group_average_surfaces"
        / _TEMPLATE_SURFACE_NAME.format(hemi=hemi_key)
    ).resolve()
    if not candidate.exists():
        raise FileNotFoundError(
            f"Missing canonical Kong2022 template surface for dataset={dataset_key}, "
            f"variant={variant}, hemi={hemi_key}: {candidate}"
        )
    return candidate


def _canonical_template_spec(
    manifest: dict[str, Any],
    *,
    dataset_key: str,
) -> tuple[str | None, Path | None]:
    subjects = _subject_candidates(manifest)
    if not subjects:
        return None, None
    root = resolve_hippunfold_batch_root(dataset_key).resolve()
    for subject in subjects:
        name = _TEMPLATE_SPEC_NAME.format(subject=subject)
        candidates = (
            root / name,
            root / f"sub-{subject}" / "hippunfold" / f"sub-{subject}" / "surf" / name,
            root / f"sub-{subject}" / "surf" / name,
        )
        for candidate in candidates:
            if candidate.exists():
                return subject, candidate.resolve()
    return None, None


def canonical_template_assets(
    manifest: dict[str, Any],
    *,
    hemi: str,
    dataset: str | None = None,
    run_dir: Path | None = None,
) -> tuple[Path, str | None, Path | None]:
    """Return the only valid template assets for a target dataset/variant.

    This function intentionally ignores source absolute asset paths.  It is the
    producer-side contract used before a manifest is written; callers must
    provide dataset/variant metadata and the function derives paths from the
    canonical layout.
    """
    dataset_key = infer_dataset_key(manifest, run_dir=run_dir, dataset=dataset)
    variant = _variant_from_manifest(manifest)
    surface = _canonical_template_surface_path(
        dataset_key=dataset_key,
        variant=variant,
        hemi=hemi,
    )
    subject, spec = _canonical_template_spec(manifest, dataset_key=dataset_key)
    if subject is None or spec is None:
        raise FileNotFoundError(
            f"Missing canonical HippUnfold template spec for dataset={dataset_key}, "
            f"variant={variant}; template_subject={manifest.get('template_subject')!r}, "
            f"subjects={manifest.get('subjects')!r}"
        )
    return surface, subject, spec


def canonicalize_template_manifest(
    manifest: dict[str, Any],
    *,
    hemi: str,
    dataset: str | None = None,
    run_dir: Path | None = None,
) -> dict[str, Any]:
    """Write dataset-canonical template fields into a producer manifest.

    A manifest that declares any template field must resolve all template fields
    from the target dataset.  No foreign or caller-supplied absolute path is
    used as a fallback.
    """
    surface, subject, spec = canonical_template_assets(
        manifest,
        hemi=hemi,
        dataset=dataset,
        run_dir=run_dir,
    )
    manifest["template_left_or_right_surface"] = str(surface)
    manifest["template_subject"] = f"sub-{subject}"
    manifest["template_spec_path"] = str(spec)
    density_evidence = manifest.get("density_evidence")
    if isinstance(density_evidence, dict):
        density_evidence["surface_path"] = str(surface)
    manifest["template_asset_contract"] = {
        "dataset": f"outputs_{infer_dataset_key(manifest, run_dir=run_dir, dataset=dataset)}",
        "preprocessing_variant": _variant_from_manifest(manifest),
        "canonicalized_by": "group-atlas-producer",
    }
    return manifest


def validate_template_assets(
    manifest: dict[str, Any],
    *,
    hemi: str,
    dataset: str | None = None,
    run_dir: Path | None = None,
) -> tuple[Path, str | None, Path | None]:
    """Validate already-written manifest paths without repairing them."""
    surface, subject, spec = canonical_template_assets(
        manifest,
        hemi=hemi,
        dataset=dataset,
        run_dir=run_dir,
    )
    declared_surface = _resolve_manifest_path(manifest.get("template_left_or_right_surface"))
    declared_spec = _resolve_manifest_path(manifest.get("template_spec_path"))
    declared_subject = _normal_subject(manifest.get("template_subject"))
    if not _path_is_equal(declared_surface, surface):
        raise ValueError(
            "Manifest template_left_or_right_surface is not canonical for its "
            f"target dataset/variant: declared={declared_surface}, expected={surface}"
        )
    if not _path_is_equal(declared_spec, spec):
        raise ValueError(
            "Manifest template_spec_path is not canonical for its target "
            f"dataset/subject: declared={declared_spec}, expected={spec}"
        )
    if declared_subject != subject:
        raise ValueError(
            "Manifest template_subject is not canonical for its target spec: "
            f"declared={declared_subject!r}, expected={subject!r}"
        )
    return surface, subject, spec


def validate_template_surface_path(
    manifest_path: Path | str | None,
    *,
    dataset: str | None = None,
    run_dir: Path | None = None,
    hemi: str | None = None,
) -> Path:
    """Validate a written surface path against the target dataset/variant."""
    dataset_key = infer_dataset_key(run_dir=run_dir, dataset=dataset)
    variant = "36p"
    if run_dir is not None:
        manifest_path_obj = Path(run_dir) / "group_atlas_manifest.json"
        if manifest_path_obj.exists():
            import json

            variant = _variant_from_manifest(json.loads(manifest_path_obj.read_text(encoding="utf-8")))
    expected = _canonical_template_surface_path(
        dataset_key=dataset_key,
        variant=variant,
        hemi=str(hemi or ""),
    )
    declared = _resolve_manifest_path(manifest_path)
    if not _path_is_equal(declared, expected):
        raise ValueError(
            "Manifest template surface is not canonical for its target "
            f"dataset/variant: declared={declared}, expected={expected}"
        )
    return expected


def resolve_template_surface_path(
    manifest_path: Path | str | None,
    *,
    dataset: str | None = None,
    run_dir: Path | None = None,
    hemi: str | None = None,
    fallback_roots: tuple[Path, ...] = (),
    template_root: Path | None = None,
) -> Path:
    """Compatibility name for strict validation; never performs fallback repair."""
    if fallback_roots or template_root is not None:
        raise ValueError("Fallback template roots are not supported; canonicalize the producer manifest")
    return validate_template_surface_path(
        manifest_path,
        dataset=dataset,
        run_dir=run_dir,
        hemi=hemi,
    )


def resolve_template_assets(
    manifest: dict[str, Any],
    *,
    hemi: str,
    dataset: str | None = None,
    run_dir: Path | None = None,
    surface_fallback_roots: tuple[Path, ...] = (),
    spec_fallback_roots: tuple[Path, ...] = (),
    template_root: Path | None = None,
) -> tuple[Path, str | None, Path | None]:
    """Compatibility name for strict validation; never repairs stale fields."""
    if surface_fallback_roots or spec_fallback_roots or template_root is not None:
        raise ValueError("Fallback template roots are not supported; canonicalize the producer manifest")
    return validate_template_assets(
        manifest,
        hemi=hemi,
        dataset=dataset,
        run_dir=run_dir,
    )
