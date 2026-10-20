from __future__ import annotations

import subprocess
import sys
import os
import tempfile
import time
from pathlib import Path
from typing import Any, Callable

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_HIPPOMAPS_ROOT = REPO_ROOT / "external" / "hippomaps"
DEFAULT_GRADIENT_RANDOM_STATE = 0


def ensure_hippomaps_on_path(hippomaps_root: Path = DEFAULT_HIPPOMAPS_ROOT) -> Path:
    root = Path(hippomaps_root).resolve()
    if not (root / "hippomaps").is_dir():
        raise FileNotFoundError(f"Missing HippoMaps package directory: {root / 'hippomaps'}")
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    return root


def wait_for_stable_file(path: Path, *, checks: int = 3, interval: float = 0.15) -> None:
    last: tuple[int, int] | None = None
    stable = 0
    for _ in range(max(1, checks * 10)):
        if path.exists():
            st = path.stat()
            current = (int(st.st_size), int(st.st_mtime_ns))
            if current == last and current[0] > 0:
                stable += 1
                if stable >= checks:
                    return
            else:
                stable = 0
                last = current
        time.sleep(float(interval))
    if not path.exists() or path.stat().st_size <= 0:
        raise TypeError(f"Render target did not stabilize: {path}")


def normalize_hemis(hemis: list[str] | tuple[str, ...]) -> list[str]:
    out = [str(hemi).upper() for hemi in hemis]
    if not out:
        raise ValueError("At least one hemisphere is required")
    invalid = [hemi for hemi in out if hemi not in ("L", "R")]
    if invalid:
        raise ValueError(f"Invalid hemisphere(s): {invalid}; expected one or more of {('L', 'R')}")
    return out


def enable_den512_plotting_v2(hippomaps_root: Path) -> bool:
    ensure_hippomaps_on_path(hippomaps_root)
    import hippomaps.plotting as hm_plotting

    v2_densities = getattr(hm_plotting, "_V2_DENS", None)
    if not isinstance(v2_densities, set):
        raise TypeError("Expected hippomaps.plotting._V2_DENS to be a set")
    if "512" in v2_densities:
        return False
    v2_densities.add("512")
    return True


def load_hippomaps_plotter(hippomaps_root: Path = DEFAULT_HIPPOMAPS_ROOT) -> Callable[..., object]:
    ensure_hippomaps_on_path(hippomaps_root)
    enable_den512_plotting_v2(hippomaps_root)
    from hippomaps.plotting import surfplot_canonical_foldunfold

    return surfplot_canonical_foldunfold


def stitch_hemi_images(left_png: Path, right_png: Path, output_png: Path) -> Path:
    from PIL import Image

    left_img = Image.open(left_png).convert("RGB")
    right_img = Image.open(right_png).convert("RGB")
    out = Image.new("RGB", (left_img.size[0] + right_img.size[0],
                           max(left_img.size[1], right_img.size[1])), (255, 255, 255))
    out.paste(left_img, (0, 0))
    out.paste(right_img, (left_img.size[0], 0))
    output_png.parent.mkdir(parents=True, exist_ok=True)
    out.save(output_png)
    return output_png


def capture_hippomaps_plot(
    plotter: Callable[..., object], cdata: np.ndarray, *, output_png: Path, **plotter_kwargs: Any,
) -> Path:
    """Capture the supported HippoMaps return types using one shared adapter."""
    output_png.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        prefix=f".{output_png.stem}.", suffix=output_png.suffix, dir=output_png.parent, delete=False
    ) as tmp:
        render_png = Path(tmp.name)
    fig = plotter(cdata, **plotter_kwargs, filename=str(render_png))
    final_render = render_png
    if fig is None:
        if not render_png.exists() or render_png.stat().st_size <= 0:
            raise TypeError("HippoMaps plotter returned None and did not create the requested screenshot")
    elif isinstance(fig, str):
        got = Path(fig)
        if got.exists() and got.stat().st_size > 0:
            final_render = got
        elif not render_png.exists() or render_png.stat().st_size <= 0:
            raise TypeError(f"HippoMaps plotter returned screenshot path but file is missing: {fig}")
    elif hasattr(fig, "savefig"):
        import matplotlib.pyplot as plt

        fig.savefig(render_png, dpi=220, bbox_inches="tight")
        try:
            plt.close(fig)
        except TypeError:
            pass
    elif hasattr(fig, "screenshot"):
        fig.screenshot(str(render_png))
    else:
        raise TypeError(f"Unsupported HippoMaps plotter return type: {type(fig)!r}")
    if not final_render.exists() or final_render.stat().st_size <= 0:
        raise TypeError(f"HippoMaps render target was not created: {final_render}")
    os.replace(final_render, output_png)
    wait_for_stable_file(output_png)
    return output_png


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


def brainspace_sparsity_for_feature_count(
    n_features: int,
    *,
    default: float = 0.9,
    full_graph_feature_threshold: int = 50,
) -> float:
    n = int(n_features)
    if n <= 0:
        raise ValueError(f"n_features must be positive, got {n_features}")
    if n <= int(full_graph_feature_threshold):
        return 0.0
    return float(default)


def fit_gradientmaps_pearson(
    group_fc: np.ndarray,
    *,
    n_components: int,
    random_state: int | None = DEFAULT_GRADIENT_RANDOM_STATE,
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
    sparsity = brainspace_sparsity_for_feature_count(matrix_z.shape[1])
    model = GradientMaps(n_components=n_comp, kernel="pearson", random_state=random_state)
    fitted = model.fit(matrix_z, sparsity=sparsity)
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
