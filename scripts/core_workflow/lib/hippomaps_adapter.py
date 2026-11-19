from __future__ import annotations

import subprocess
import sys
import os
import tempfile
import time
from pathlib import Path
from typing import Any, Callable

import numpy as np


from .output_layout import DATA_ROOT

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_HIPPOMAPS_ROOT = DATA_ROOT / "external" / "hippomaps"
GRADIENT_EIGENSOLVER = "symmetric_eigh"
DIFFUSION_ALPHA = 0.5


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


def gradient_preprocessing_settings(
    n_features: int,
    *,
    feature_scaling: str = "col_zscore",
    sparsify_stage: str = "features",
    sparsity: float | None = None,
) -> dict[str, Any]:
    """Resolve independent FC scaling and affinity-construction choices."""
    if feature_scaling not in {"none", "col_zscore"}:
        raise ValueError(f"Unknown FC feature_scaling: {feature_scaling!r}")
    if sparsify_stage not in {"features", "affinity", "none"}:
        raise ValueError(f"Unknown sparsify_stage: {sparsify_stage!r}")
    effective = brainspace_sparsity_for_feature_count(n_features) if sparsity is None else float(sparsity)
    if not np.isfinite(effective) or not 0 <= effective < 1:
        raise ValueError(f"sparsity must be finite in [0, 1), got {sparsity!r}")
    return {
        "feature_scaling": feature_scaling,
        "sparsify_stage": sparsify_stage,
        "effective_sparsity": None if sparsify_stage == "none" else effective,
        "kernel": "pearson",
        "non_negative": True,
        "symmetrization": "arithmetic_mean",
        "diagonal": "retain_kernel_diagonal",
        "diffusion_alpha": DIFFUSION_ALPHA,
        "diffusion_time": 0,
        "eigensolver": GRADIENT_EIGENSOLVER,
    }


def validate_gradient_preprocessing(
    manifest: dict[str, Any], *, settings: dict[str, Any], manifest_path: Path,
) -> None:
    recorded = manifest.get("gradient_preprocessing")
    if not isinstance(recorded, dict):
        raise ValueError(f"Missing gradient_preprocessing in {manifest_path}; explicitly regenerate the gradients")
    mismatches = [
        f"{key}: requested={value!r}, recorded={recorded.get(key)!r}"
        for key, value in settings.items() if key not in recorded or recorded[key] != value
    ]
    if mismatches:
        raise ValueError(f"Gradient preprocessing mismatch in {manifest_path}: {'; '.join(mismatches)}; explicitly regenerate the gradients")


def build_pearson_affinity(
    group_fc: np.ndarray, *, feature_scaling: str = "col_zscore",
    sparsify_stage: str = "features", sparsity: float | None = None,
) -> np.ndarray:
    from brainspace.gradient.kernels import compute_affinity
    from scipy.sparse.csgraph import connected_components

    matrix = np.asarray(group_fc, dtype=np.float32)
    if matrix.ndim != 2 or matrix.shape[0] < 2 or matrix.shape[1] < 2:
        raise ValueError(f"Expected a vertex-by-feature FC matrix, got {matrix.shape}")
    if not np.isfinite(matrix).all():
        raise ValueError("Gradient FC matrix contains non-finite values")
    settings = gradient_preprocessing_settings(
        matrix.shape[1], feature_scaling=feature_scaling,
        sparsify_stage=sparsify_stage, sparsity=sparsity,
    )
    features = feature_zscore_columns(matrix) if feature_scaling == "col_zscore" else matrix.copy()
    affinity = compute_affinity(
        features, kernel="pearson", sparsity=settings["effective_sparsity"],
        pre_sparsify=sparsify_stage != "affinity", non_negative=True,
    )
    if not np.isfinite(affinity).all():
        raise ValueError("Pearson affinity contains non-finite values; inspect constant vertex profiles")
    # DiffusionMaps uses this mean symmetrization for row-sparsified input.
    # Preserve the Pearson diagonal, including the legacy default route.
    affinity = (affinity + affinity.T) / 2
    n_components = connected_components(affinity, directed=False, return_labels=False)
    if n_components != 1:
        raise ValueError(f"Diffusion affinity is disconnected: {n_components} components")
    return affinity


def scale_gradient_features(
    features: np.ndarray, *, scaling: str = "none", transform: dict[str, Any] | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Fit on supported training vertices, or apply that exact fitted transform."""
    matrix = np.asarray(features, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[0] == 0 or not np.isfinite(matrix).all():
        raise ValueError("Clustering features must be a nonempty finite 2D matrix")
    if scaling not in {"none", "unit_sd"}:
        raise ValueError(f"Unknown clustering feature scaling: {scaling!r}")
    if transform is None:
        means, sds = matrix.mean(axis=0), matrix.std(axis=0, ddof=0)
        if np.any(sds == 0):
            raise ValueError(f"Clustering features contain zero-variance components: {np.flatnonzero(sds == 0).tolist()}")
        transform = {"scaling": scaling, "component_means": means.tolist(),
                     "component_sds": sds.tolist(), "ddof": 0,
                     "n_vertices_fit": int(matrix.shape[0]),
                     "fit_population": "supported_training_vertices"}
    elif transform.get("scaling") != scaling:
        raise ValueError("Clustering transform disagrees with requested scaling")
    means = np.asarray(transform["component_means"], dtype=np.float64)
    sds = np.asarray(transform["component_sds"], dtype=np.float64)
    if means.shape != (matrix.shape[1],) or sds.shape != means.shape or np.any(sds <= 0):
        raise ValueError("Clustering transform has invalid component means or SDs")
    scaled = (matrix - means) / sds if scaling == "unit_sd" else matrix.copy()
    return scaled, dict(transform)


def diffusion_map_embedding(affinity: np.ndarray, *, n_components: int) -> tuple[np.ndarray, np.ndarray]:
    """Exact multi-scale diffusion-map embedding of a symmetric, connected affinity.

    The affinity is first normalised by degree^-DIFFUSION_ALPHA on both sides.
    The resulting diffusion operator P = D^-1 W is not symmetric, but it shares
    its eigenvalues with S = D^-1/2 W D^-1/2 and its right eigenvectors are
    D^-1/2 times those of S. S is decomposed completely with a direct symmetric
    solver, so the result has no start vector and no random seed.

    The components are the eigenvectors with the algebraically largest
    eigenvalues after the trivial one. Each eigenvector has unit norm under the
    stationary distribution pi = degree / sum(degree), the standard diffusion-map
    normalisation, and is scaled by its multi-scale eigenvalue w / (1 - w), so
    Euclidean distance between rows is the multi-scale diffusion distance. The
    largest-magnitude element of every component is positive. BrainSpace
    normalises each eigenvector to unit unweighted norm instead, which rescales
    components by a few percent relative to these.
    """
    weights = np.array(affinity, dtype=np.float64)
    n_comp = int(n_components)
    if weights.ndim != 2 or weights.shape[0] != weights.shape[1]:
        raise ValueError(f"Diffusion affinity must be square, got {weights.shape}")
    if not 0 < n_comp < weights.shape[0]:
        raise ValueError(f"n_components must be in [1, {weights.shape[0] - 1}], got {n_components}")
    if not np.isfinite(weights).all() or np.any(weights < 0):
        raise ValueError("Diffusion affinity must be finite and nonnegative")
    if not np.allclose(weights, weights.T, rtol=0.0, atol=1e-10):
        raise ValueError("Diffusion affinity must be symmetric")
    scale = np.power(weights.sum(axis=1), -DIFFUSION_ALPHA)
    weights = weights * scale[:, None] * scale[None, :]
    degree = weights.sum(axis=1)
    if np.any(degree <= 0):
        raise ValueError("Diffusion affinity has a vertex without positive weight")
    root = np.sqrt(degree)
    symmetric = weights / root[:, None] / root[None, :]
    symmetric = (symmetric + symmetric.T) / 2
    values, vectors = np.linalg.eigh(symmetric)
    values = values[::-1][: n_comp + 1]
    vectors = vectors[:, ::-1][:, : n_comp + 1] / root[:, None]
    vectors = vectors / vectors[:, [0]]
    values = values / values[0]
    lambdas = values[1:]
    if not np.all(lambdas < 1 - 1e-10):
        raise ValueError("Diffusion affinity has a repeated leading eigenvalue; the graph is disconnected")
    lambdas = lambdas / (1 - lambdas)
    embedding = vectors[:, 1:] * lambdas[None, :]
    embedding = embedding * np.sign(embedding[np.abs(embedding).argmax(axis=0), np.arange(n_comp)])
    return embedding, lambdas


MEAN_SIMILARITY_EMBEDDING_CALL = (
    "nanmean over subjects of within-subject Pearson vertex similarity of Fisher-z 400-ROI fingerprints; "
    "diagonal=1; compute_affinity(kernel=None, sparsity=declared, non_negative=True); mean_symmetrize; "
    "diffusion_map_embedding(affinity, alpha=0.5, diffusion_time=0)"
)


def build_mean_similarity_affinity(subject_similarities: np.ndarray, *, sparsity: float) -> np.ndarray:
    """Average participant vertex similarities, then row-sparsify like TopSim.

    Each vertex pair is averaged over the participants who observe both vertices,
    so every participant with a defined similarity counts equally. A supported
    pair that no participant observes stops the fit.
    """
    from brainspace.gradient.kernels import compute_affinity
    from scipy.sparse.csgraph import connected_components

    stack = np.asarray(subject_similarities)
    if stack.ndim != 3 or stack.shape[1] != stack.shape[2] or stack.shape[1] < 2:
        raise ValueError(f"Expected subjects x vertices x vertices similarities, got {stack.shape}")
    defined = np.isfinite(stack)
    counts = defined.sum(axis=0)
    if not np.all(counts > 0):
        a, b = np.argwhere(counts == 0)[0]
        raise ValueError(f"No participant observes supported vertex pair ({int(a)}, {int(b)})")
    similarity = np.where(defined, stack, 0.0).sum(axis=0, dtype=np.float64) / counts
    np.fill_diagonal(similarity, 1.0)
    affinity = compute_affinity(similarity, kernel=None, sparsity=float(sparsity), non_negative=True)
    if not np.isfinite(affinity).all():
        raise ValueError("Mean-similarity affinity contains non-finite values")
    affinity = (affinity + affinity.T) / 2
    n_components = connected_components(affinity, directed=False, return_labels=False)
    if n_components != 1:
        raise ValueError(f"Mean-similarity affinity is disconnected: {n_components} components")
    return affinity


def fit_mean_similarity_gradients(
    subject_similarities: np.ndarray, *, n_components: int, sparsity: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n_comp = int(n_components)
    if int(np.asarray(subject_similarities).shape[1]) <= n_comp:
        raise RuntimeError(
            f"Not enough supported vertices for gradients: shape={np.shape(subject_similarities)} n_components={n_comp}"
        )
    affinity = build_mean_similarity_affinity(subject_similarities, sparsity=sparsity)
    gradients, lambdas = diffusion_map_embedding(affinity, n_components=n_comp)
    if not np.all(np.isfinite(gradients)) or not np.all(np.isfinite(lambdas)):
        raise RuntimeError("Diffusion-map embedding returned non-finite values")
    ev = ev_fractions_from_lambdas(lambdas)
    return gradients.astype(np.float32), lambdas.astype(np.float64), ev.astype(np.float64)


def fit_diffusion_gradients(
    group_fc: np.ndarray,
    *,
    n_components: int,
    feature_scaling: str = "col_zscore",
    sparsify_stage: str = "features",
    sparsity: float | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    matrix = np.asarray(group_fc, dtype=np.float32)
    if matrix.ndim != 2:
        raise ValueError(f"Expected 2D group FC matrix, got shape={matrix.shape}")
    n_comp = int(n_components)
    if n_comp <= 0:
        raise ValueError(f"n_components must be positive, got {n_components}")
    if int(matrix.shape[0]) <= n_comp:
        raise RuntimeError(
            f"Not enough supported vertices for gradients: n_vertices={matrix.shape[0]} n_components={n_comp}"
        )
    affinity = build_pearson_affinity(
        matrix, feature_scaling=feature_scaling, sparsify_stage=sparsify_stage, sparsity=sparsity,
    )
    gradients, lambdas = diffusion_map_embedding(affinity, n_components=n_comp)
    if not np.all(np.isfinite(gradients)) or not np.all(np.isfinite(lambdas)):
        raise RuntimeError("Diffusion-map embedding returned non-finite values")
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
