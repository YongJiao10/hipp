"""Reusable Workbench cortex surfaces for group overviews."""
from __future__ import annotations

import hashlib
import os
import tempfile
from importlib.metadata import version
from pathlib import Path

import numpy as np
from PIL import Image

from lib.cluster_roi_profile_contract import validate_cluster_roi_profile_manifest


def bilateral_cluster_roi_profile(manifest_paths: list[Path], manifests: list[dict]) -> np.ndarray | None:
    """Read and validate the exact left/right cluster ROI profiles before averaging."""
    if any(manifest.get("cluster_roi_profile") is None or manifest.get("chosen_k_final") != 3 for manifest in manifests):
        return None
    profiles = []
    for manifest_path, manifest in zip(manifest_paths, manifests, strict=True):
        profile_path = manifest_path.parent / "cluster_roi_profile.npy"
        profile = np.load(profile_path).astype(np.float64)
        validate_cluster_roi_profile_manifest(manifest, profile_path=profile_path, profile_shape=profile.shape)
        profiles.append(profile)
    return np.mean(profiles, axis=0)


def hemisphere_cluster_roi_profiles(manifest_paths: list[Path], manifests: list[dict]) -> list[np.ndarray] | None:
    """Read and validate each hemisphere's cluster ROI profile at that run's own exported K."""
    if any(manifest.get("cluster_roi_profile") is None for manifest in manifests):
        return None
    profiles = []
    for manifest_path, manifest in zip(manifest_paths, manifests, strict=True):
        profile_path = manifest_path.parent / "cluster_roi_profile.npy"
        profile = np.load(profile_path).astype(np.float64)
        validate_cluster_roi_profile_manifest(
            manifest, profile_path=profile_path, profile_shape=profile.shape,
            n_clusters=int(manifest["chosen_k_final"]),
        )
        profiles.append(profile)
    return profiles


_RUNTIME_PACKAGES = ("numpy", "nibabel", "matplotlib", "Pillow")


def _cache_directory(cache_root: Path, inputs: list[Path]) -> Path:
    """One directory per runtime stack and exact resolved input paths."""
    runtime = "_".join(f"{name}-{version(name)}" for name in _RUNTIME_PACKAGES)
    directory = Path(cache_root) / runtime
    for index, path in enumerate(inputs):
        path = Path(path).resolve(strict=True)
        directory = directory / f"input-{index}" / path.relative_to(path.anchor)
    return directory


def _dependency_digest(inputs: list[Path], code: list[Path]) -> str:
    """SHA-256 over the content of every input, renderer source file and this module."""
    digest = hashlib.sha256()
    for path in [*inputs, *code, Path(__file__)]:
        path = Path(path).resolve(strict=True)
        digest.update(str(path).encode("utf-8") + b"\0" + hashlib.sha256(path.read_bytes()).digest())
    return digest.hexdigest()


def _write_atomically(directory: Path, destination: Path, suffix: str, data: bytes) -> None:
    with tempfile.NamedTemporaryFile(dir=directory, suffix=suffix, delete=False) as tmp:
        temporary = Path(tmp.name)
        tmp.write(data)
    os.replace(temporary, destination)


def cortex_surface_cache_paths(*, cache_root: Path, inputs: list[Path], code: list[Path], size: tuple[int, int],
                               count: int = 3) -> list[Path | None]:
    """Return each valid cached surface, or None.

    A surface is valid only when its recorded dependency digest equals the current content digest of
    all inputs and renderer code, and the PNG decodes at the expected size.
    """
    directory = _cache_directory(cache_root, inputs)
    expected = _dependency_digest(inputs, code)
    result = []
    for index in range(int(count)):
        candidate = directory / f"cluster{index + 1}.png"
        record = candidate.with_suffix(".png.sha256")
        valid = candidate.is_file() and record.is_file() and record.read_text(encoding="utf-8").strip() == expected
        if valid:
            try:
                with Image.open(candidate) as img:
                    valid = img.format == "PNG" and img.size == size
                    img.verify()
            except (OSError, ValueError, SyntaxError):
                valid = False
        result.append(candidate if valid else None)
    return result


def store_cortex_surface(*, cache_root: Path, inputs: list[Path], code: list[Path], index: int, source: Path) -> Path:
    """Atomically publish a surface and its dependency digest so concurrent overviews never read partial files."""
    directory = _cache_directory(cache_root, inputs)
    directory.mkdir(parents=True, exist_ok=True)
    destination = directory / f"cluster{index + 1}.png"
    record = destination.with_suffix(".png.sha256")
    record.unlink(missing_ok=True)
    _write_atomically(directory, destination, ".png", Path(source).read_bytes())
    _write_atomically(directory, record, ".sha256", (_dependency_digest(inputs, code) + "\n").encode("utf-8"))
    return destination
