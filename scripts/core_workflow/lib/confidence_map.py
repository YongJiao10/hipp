from __future__ import annotations

import numpy as np
from sklearn.metrics import silhouette_samples


def compute_silhouette_map(
    features: np.ndarray,
    labels: np.ndarray,
    metric: str = "cosine",
) -> np.ndarray:
    """Compute per-vertex silhouette values following Yeo et al. 2011."""
    unique_labels = np.unique(labels)
    if len(unique_labels) < 2:
        return np.zeros(labels.shape, dtype=np.float32)

    sil = silhouette_samples(features, labels, metric=metric)
    return sil.astype(np.float32)


def publication_display_upper(
    values: np.ndarray,
    *,
    vmax: float | None = None,
) -> float:
    """Choose a nonnegative, paper-style upper bound for silhouette display."""
    if vmax is not None:
        out = float(vmax)
        if not np.isfinite(out) or out <= 0:
            raise ValueError(f"vmax must be a positive finite value, got {vmax!r}")
        return out

    arr = np.asarray(values, dtype=np.float32)
    finite = arr[np.isfinite(arr)]
    positive = finite[finite > 0]
    if positive.size == 0:
        return 1.0
    return float(np.max(positive))


def publication_display_data(values: np.ndarray, *, vmax: float | None = None) -> np.ndarray:
    """Clip signed silhouette values for nonnegative confidence-map display."""
    upper = publication_display_upper(values, vmax=vmax)
    arr = np.asarray(values, dtype=np.float32)
    out = np.array(arr, copy=True, dtype=np.float32)
    finite = np.isfinite(out)
    out[finite] = np.clip(out[finite], 0.0, upper)
    return out


def silhouette_to_full_surface(
    silhouette_active: np.ndarray,
    support_mask: np.ndarray,
    fill_value: float = np.nan,
) -> np.ndarray:
    """Expand active-vertex silhouette to full surface array."""
    full_surface = np.full(support_mask.shape, fill_value, dtype=np.float32)
    mask_bool = support_mask.astype(bool)
    if mask_bool.sum() != len(silhouette_active):
        raise ValueError(
            f"Active count mismatch: mask active count={mask_bool.sum()}, "
            f"silhouette_active length={len(silhouette_active)}"
        )
    full_surface[mask_bool] = silhouette_active
    return full_surface


def silhouette_summary(
    silhouette_active: np.ndarray,
    labels_active: np.ndarray,
) -> dict:
    """Compute per-cluster and global silhouette statistics."""
    if len(silhouette_active) == 0:
        return {
            "global_mean": 0.0,
            "global_std": 0.0,
            "global_median": 0.0,
            "global_min": 0.0,
            "global_max": 0.0,
            "global_frac_negative": 0.0,
            "per_cluster": [],
        }

    global_mean = float(np.mean(silhouette_active))
    global_std = float(np.std(silhouette_active))
    global_median = float(np.median(silhouette_active))
    global_min = float(np.min(silhouette_active))
    global_max = float(np.max(silhouette_active))
    global_frac_negative = float(np.mean(silhouette_active < 0))

    unique_labels = sorted(np.unique(labels_active))
    per_cluster = []
    for cid in unique_labels:
        cluster_mask = labels_active == cid
        c_sil = silhouette_active[cluster_mask]
        if len(c_sil) > 0:
            per_cluster.append(
                {
                    "cluster_id": int(cid),
                    "mean": float(np.mean(c_sil)),
                    "std": float(np.std(c_sil)),
                    "median": float(np.median(c_sil)),
                    "min": float(np.min(c_sil)),
                    "max": float(np.max(c_sil)),
                    "n_vertices": int(len(c_sil)),
                    "frac_negative": float(np.mean(c_sil < 0)),
                }
            )
        else:
            per_cluster.append(
                {
                    "cluster_id": int(cid),
                    "mean": 0.0,
                    "std": 0.0,
                    "median": 0.0,
                    "min": 0.0,
                    "max": 0.0,
                    "n_vertices": 0,
                    "frac_negative": 0.0,
                }
            )

    return {
        "global_mean": global_mean,
        "global_std": global_std,
        "global_median": global_median,
        "global_min": global_min,
        "global_max": global_max,
        "global_frac_negative": global_frac_negative,
        "per_cluster": per_cluster,
    }
