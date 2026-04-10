#!/usr/bin/env python3
from __future__ import annotations

import re
from pathlib import Path

SPACE_CANDIDATES = ["corobl", "T2w", "T1w", "nativepro", "unfold"]


class DensityAssetError(RuntimeError):
    pass


def _density_contract_msg(density: str) -> str:
    return (
        f"Input density = {density}. Consumed asset must use den-{density}. "
        "Legacy assets without den token are rejected."
    )


def load_surface_density_from_pipeline_config(config_path: Path) -> str:
    if not config_path.exists():
        raise DensityAssetError(f"Missing pipeline config: {config_path}")
    for raw in config_path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("surface_density") and "=" in line:
            _, value = line.split("=", 1)
            density = value.strip().strip('"')
            if density:
                return density
    raise DensityAssetError(f"Missing required key `surface_density` in config: {config_path}")


def subject_surf_dir(hippunfold_dir: Path, subject: str) -> Path:
    surf_dir = hippunfold_dir / f"sub-{subject}" / "surf"
    if not surf_dir.exists():
        raise DensityAssetError(
            f"Missing canonical hippocampal surface directory: {surf_dir}. "
            "Only <hippunfold-dir>/sub-<id>/surf is allowed."
        )
    return surf_dir


def _ensure_single(matches: list[Path], what: str) -> Path:
    if not matches:
        raise DensityAssetError(what)
    if len(matches) > 1:
        rendered = "\n".join(str(path) for path in matches)
        raise DensityAssetError(f"Ambiguous density asset candidates:\n{rendered}")
    return matches[0]


def _assert_no_legacy_without_density(surf_dir: Path, legacy_pattern: str, density: str) -> None:
    legacy_matches = sorted(surf_dir.glob(legacy_pattern))
    if legacy_matches:
        rendered = "\n".join(str(path) for path in legacy_matches)
        raise DensityAssetError(
            f"Legacy asset(s) without den token detected while density={density}:\n{rendered}\n"
            f"{_density_contract_msg(density)} "
            "Please regenerate HippUnfold outputs with the target density."
        )


def _assert_no_mixed_density(surf_dir: Path, den_pattern: str, density: str) -> None:
    den_matches = sorted(surf_dir.glob(den_pattern))
    density_tokens: set[str] = set()
    token_pattern = re.compile(r"_den-([^_]+)_label-hipp_")
    for path in den_matches:
        match = token_pattern.search(path.name)
        if match:
            density_tokens.add(match.group(1))
    if density_tokens and (density not in density_tokens or len(density_tokens) > 1):
        rendered = "\n".join(str(path) for path in den_matches)
        raise DensityAssetError(
            f"Mixed or mismatched density assets detected (requested density={density}, found={sorted(density_tokens)}):\n{rendered}\n"
            f"{_density_contract_msg(density)} "
            "Do not mix densities in one step; regenerate and keep a single den token."
        )


def detect_space_strict(
    *,
    surf_dir: Path,
    subject: str,
    density: str,
    preferred: str | None,
    candidates: list[str] | None = None,
) -> str:
    if preferred and preferred != "auto":
        _ = find_surface_asset_strict(
            surf_dir=surf_dir,
            subject=subject,
            hemi="L",
            space=preferred,
            density=density,
            suffix="midthickness.surf.gii",
        )
        return preferred

    scan_spaces = candidates or SPACE_CANDIDATES
    found: list[str] = []
    for space in scan_spaces:
        try:
            _ = find_surface_asset_strict(
                surf_dir=surf_dir,
                subject=subject,
                hemi="L",
                space=space,
                density=density,
                suffix="midthickness.surf.gii",
            )
            found.append(space)
        except DensityAssetError:
            continue

    if not found:
        raise DensityAssetError(
            f"Could not detect folded hippocampal surface for subject={subject}, density={density} under {surf_dir}. "
            f"Expected files like sub-{subject}_hemi-L_space-<space>_den-{density}_label-hipp_midthickness.surf.gii. "
            f"{_density_contract_msg(density)}"
        )
    if len(found) > 1:
        raise DensityAssetError(
            f"Ambiguous space for subject={subject}, density={density}: {found}. "
            "Pass --space explicitly to avoid implicit mixed selection."
        )
    return found[0]


def find_surface_asset_strict(
    *,
    surf_dir: Path,
    subject: str,
    hemi: str,
    space: str,
    density: str,
    suffix: str,
) -> Path:
    den_pattern = f"sub-{subject}_hemi-{hemi}_space-{space}_den-*_label-hipp_{suffix}"
    _assert_no_mixed_density(surf_dir, den_pattern, density)
    legacy_pattern = f"sub-{subject}_hemi-{hemi}_space-{space}_label-hipp_{suffix}"
    _assert_no_legacy_without_density(surf_dir, legacy_pattern, density)
    matches = sorted(
        surf_dir.glob(f"sub-{subject}_hemi-{hemi}_space-{space}_den-{density}_label-hipp_{suffix}")
    )
    return _ensure_single(
        matches,
        (
            f"Missing strict density asset for subject={subject}, hemi={hemi}, space={space}, density={density}, suffix={suffix}. "
            f"Required pattern: sub-{subject}_hemi-{hemi}_space-{space}_den-{density}_label-hipp_{suffix}. "
            f"{_density_contract_msg(density)}"
        ),
    )


def find_cifti_asset_strict(*, cifti_dir: Path, subject: str, density: str, suffix: str) -> Path:
    den_pattern = f"sub-{subject}_den-*_label-hipp_{suffix}"
    matches = sorted(cifti_dir.glob(den_pattern))
    density_tokens: set[str] = set()
    token_pattern = re.compile(r"_den-([^_]+)_label-hipp_")
    for path in matches:
        match = token_pattern.search(path.name)
        if match:
            density_tokens.add(match.group(1))
    if density_tokens and (density not in density_tokens or len(density_tokens) > 1):
        rendered = "\n".join(str(path) for path in matches)
        raise DensityAssetError(
            f"Mixed or mismatched CIFTI densities for subject={subject} (requested={density}, found={sorted(density_tokens)}):\n{rendered}\n"
            f"{_density_contract_msg(density)}"
        )
    legacy_matches = sorted(cifti_dir.glob(f"sub-{subject}_label-hipp_{suffix}"))
    if legacy_matches:
        rendered = "\n".join(str(path) for path in legacy_matches)
        raise DensityAssetError(
            f"Legacy CIFTI asset(s) without den token detected for density={density}:\n{rendered}\n"
            f"{_density_contract_msg(density)} Regenerate with den-{density}."
        )
    return _ensure_single(
        sorted(cifti_dir.glob(f"sub-{subject}_den-{density}_label-hipp_{suffix}")),
        (
            f"Missing strict CIFTI asset for subject={subject}, density={density}, suffix={suffix}. "
            f"{_density_contract_msg(density)}"
        ),
    )


def find_surface_sampling_metric_strict(
    *,
    surface_source_dir: Path,
    subject: str,
    hemi: str,
    density: str,
    space: str = "corobl",
) -> Path:
    den_pattern = f"sub-{subject}_hemi-{hemi}_space-{space}_den-*_label-hipp_bold.func.gii"
    matches = sorted(surface_source_dir.glob(den_pattern))
    tokens: set[str] = set()
    token_pattern = re.compile(r"_den-([^_]+)_label-hipp_bold\.func\.gii$")
    for path in matches:
        match = token_pattern.search(path.name)
        if match:
            tokens.add(match.group(1))
    if tokens and (density not in tokens or len(tokens) > 1):
        rendered = "\n".join(str(path) for path in matches)
        raise DensityAssetError(
            f"Mixed or mismatched surface sampling metrics under {surface_source_dir} for hemi={hemi}. "
            f"requested={density}, found={sorted(tokens)}\n{rendered}\n"
            f"{_density_contract_msg(density)}"
        )
    legacy = sorted(surface_source_dir.glob(f"sub-{subject}_hemi-{hemi}_space-{space}_label-hipp_bold.func.gii"))
    if legacy:
        rendered = "\n".join(str(path) for path in legacy)
        raise DensityAssetError(
            f"Legacy raw metric(s) without den token detected:\n{rendered}\n"
            f"{_density_contract_msg(density)} "
            f"Regenerate post-HippUnfold outputs with density={density}."
        )
    return _ensure_single(
        sorted(surface_source_dir.glob(f"sub-{subject}_hemi-{hemi}_space-{space}_den-{density}_label-hipp_bold.func.gii")),
        (
            f"Missing strict raw metric for subject={subject}, hemi={hemi}, density={density} in {surface_source_dir}. "
            f"Required: sub-{subject}_hemi-{hemi}_space-{space}_den-{density}_label-hipp_bold.func.gii. "
            f"{_density_contract_msg(density)}"
        ),
    )
