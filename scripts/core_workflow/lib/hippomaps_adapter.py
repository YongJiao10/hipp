from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_HIPPOMAPS_ROOT = REPO_ROOT / "external" / "hippomaps"


def ensure_hippomaps_on_path(hippomaps_root: Path = DEFAULT_HIPPOMAPS_ROOT) -> Path:
    root = Path(hippomaps_root).resolve()
    if not (root / "hippomaps").is_dir():
        raise FileNotFoundError(f"Missing HippoMaps package directory: {root / 'hippomaps'}")
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    return root


def hippomaps_commit(hippomaps_root: Path = DEFAULT_HIPPOMAPS_ROOT) -> str:
    root = Path(hippomaps_root).resolve()
    proc = subprocess.run(
        ["git", "-C", str(root), "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode != 0:
        return "unknown"
    return proc.stdout.strip()


def hippomaps_import_path(hippomaps_root: Path = DEFAULT_HIPPOMAPS_ROOT) -> str:
    root = ensure_hippomaps_on_path(hippomaps_root)
    return str((root / "hippomaps").resolve())


def ev_fractions_from_lambdas(lambdas: np.ndarray) -> np.ndarray:
    vals = np.asarray(lambdas, dtype=np.float64).reshape(-1)
    if vals.size == 0:
        raise ValueError("Lambda vector is empty")
    denom = float(np.sum(vals))
    if not np.isfinite(denom) or denom <= 0.0:
        raise ValueError(f"Lambda sum must be positive, got {denom}")
    return (vals / denom).astype(np.float64)


def feature_zscore_columns(matrix: np.ndarray) -> np.ndarray:
    arr = np.asarray(matrix, dtype=np.float32)
    if arr.ndim != 2:
        raise ValueError(f"Expected 2D feature matrix, got shape={arr.shape}")
    mean = np.nanmean(arr, axis=0, keepdims=True)
    std = np.nanstd(arr, axis=0, keepdims=True)
    out = (arr - mean) / np.where(std > 0.0, std, 1.0)
    out[~np.isfinite(out)] = 0.0
    return out.astype(np.float32)


def _extract_gradientmaps_array(model: object, primary: str, fallback: str, label: str) -> np.ndarray:
    raw: Any = getattr(model, primary, None)
    if raw is None:
        raw = getattr(model, fallback, None)
    if raw is None:
        raise RuntimeError(f"GradientMaps output missing {primary}/{fallback} for {label}")
    if isinstance(raw, list):
        if len(raw) != 1:
            raise RuntimeError(f"Expected one GradientMaps dataset for {label}, got {len(raw)}")
        raw = raw[0]
    arr = np.asarray(raw)
    if arr.size == 0:
        raise RuntimeError(f"GradientMaps returned empty {label}")
    return arr


def fit_gradientmaps_pearson(
    group_fc: np.ndarray,
    *,
    n_components: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    matrix = np.asarray(group_fc, dtype=np.float32)
    if matrix.ndim != 2:
        raise ValueError(f"Expected 2D group FC matrix, got shape={matrix.shape}")
    n_comp = int(n_components)
    if n_comp <= 0:
        raise ValueError(f"n_components must be positive, got {n_components}")
    if int(matrix.shape[0]) <= n_comp:
        raise RuntimeError(
            f"Not enough supported vertices for GradientMaps: n_vertices={matrix.shape[0]} n_components={n_comp}"
        )
    from brainspace.gradient import GradientMaps

    matrix_z = feature_zscore_columns(matrix)
    model = GradientMaps(n_components=n_comp, kernel="pearson")
    fitted = model.fit(matrix_z)
    model_out = fitted if fitted is not None else model
    gradients = _extract_gradientmaps_array(model_out, "gradients_", "gradients", "gradients")
    lambdas = _extract_gradientmaps_array(model_out, "lambdas_", "lambdas", "lambdas").reshape(-1)
    gradients = np.asarray(gradients, dtype=np.float32)
    lambdas = np.asarray(lambdas, dtype=np.float64)
    if gradients.ndim != 2:
        raise RuntimeError(f"Expected 2D gradients, got shape={gradients.shape}")
    if gradients.shape[1] < n_comp:
        raise RuntimeError(f"GradientMaps returned {gradients.shape[1]} gradients, expected {n_comp}")
    if lambdas.size < n_comp:
        raise RuntimeError(f"GradientMaps returned {lambdas.size} lambdas, expected at least {n_comp}")
    gradients = gradients[:, :n_comp]
    gradients[~np.isfinite(gradients)] = 0.0
    if not np.all(np.isfinite(lambdas)):
        raise RuntimeError(f"GradientMaps lambdas contain non-finite values: {lambdas.tolist()}")
    ev = ev_fractions_from_lambdas(lambdas)
    return gradients.astype(np.float32), lambdas.astype(np.float64), ev.astype(np.float64)


def reconstruct_full_surface(active: np.ndarray, support_mask: np.ndarray, vertex_count: int) -> np.ndarray:
    arr = np.asarray(active, dtype=np.float32)
    mask = np.asarray(support_mask, dtype=bool).reshape(-1)
    n_vertices = int(vertex_count)
    if mask.size != n_vertices:
        raise ValueError(f"Support mask length mismatch: mask={mask.size} vertex_count={n_vertices}")
    if arr.ndim == 1:
        if arr.shape[0] != int(np.sum(mask)):
            raise ValueError(f"Active/support mismatch: active={arr.shape} support={int(np.sum(mask))}")
        full = np.full((n_vertices,), np.nan, dtype=np.float32)
        full[mask] = arr
        return full
    if arr.ndim != 2:
        raise ValueError(f"Expected 1D or 2D active array, got shape={arr.shape}")
    if arr.shape[0] != int(np.sum(mask)):
        raise ValueError(f"Active/support mismatch: active_rows={arr.shape[0]} support={int(np.sum(mask))}")
    full2 = np.full((n_vertices, arr.shape[1]), np.nan, dtype=np.float32)
    full2[mask, :] = arr
    return full2
