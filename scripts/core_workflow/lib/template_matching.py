from __future__ import annotations

import numpy as np


def build_cluster_prototypes(
    group_fc: np.ndarray,
    labels: np.ndarray,
    support: np.ndarray,
    k: int = 3,
) -> np.ndarray:
    """Average supported vertex FC profiles within ordered clusters."""
    fc = np.asarray(group_fc, dtype=np.float32)
    ordered_labels = np.asarray(labels, dtype=np.int32).reshape(-1)
    support_mask = np.asarray(support, dtype=bool).reshape(-1)
    if fc.ndim != 2:
        raise ValueError(f"Expected 2D group FC, got shape={fc.shape}")
    if fc.shape[0] != ordered_labels.size or ordered_labels.size != support_mask.size:
        raise ValueError(
            "FC/label/support vertex mismatch: "
            f"fc={fc.shape[0]} labels={ordered_labels.size} support={support_mask.size}"
        )
    if int(k) < 1:
        raise ValueError("k must be at least 1")

    prototypes = np.full((int(k), fc.shape[1]), np.nan, dtype=np.float32)
    for cluster_id in range(1, int(k) + 1):
        vertex_mask = support_mask & (ordered_labels == cluster_id)
        if not np.any(vertex_mask):
            raise ValueError(f"No supported vertices for cluster {cluster_id}")
        values = fc[vertex_mask]
        finite = np.isfinite(values)
        counts = np.sum(finite, axis=0)
        sums = np.sum(np.where(finite, values, 0.0), axis=0, dtype=np.float64)
        present = counts > 0
        prototypes[cluster_id - 1, present] = (
            sums[present] / counts[present]
        ).astype(np.float32)
    return prototypes


def _finite_pearson(left: np.ndarray, right: np.ndarray) -> float:
    a = np.asarray(left, dtype=np.float64).reshape(-1)
    b = np.asarray(right, dtype=np.float64).reshape(-1)
    finite = np.isfinite(a) & np.isfinite(b)
    if int(np.sum(finite)) < 2:
        return float("nan")
    x = a[finite] - float(np.mean(a[finite]))
    y = b[finite] - float(np.mean(b[finite]))
    denominator = float(np.linalg.norm(x) * np.linalg.norm(y))
    if not np.isfinite(denominator) or denominator <= 0.0:
        return float("nan")
    correlation = float(np.dot(x, y) / denominator)
    if not np.isfinite(correlation):
        return float("nan")
    return float(np.clip(correlation, -1.0, 1.0))


def match_vertices_to_prototypes(
    fisherz_fc: np.ndarray,
    valid_mask: np.ndarray,
    prototypes: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Match valid vertex profiles to ordered prototypes by finite Pearson r."""
    fc = np.asarray(fisherz_fc, dtype=np.float32)
    valid = np.asarray(valid_mask, dtype=bool).reshape(-1)
    template = np.asarray(prototypes, dtype=np.float32)
    if fc.ndim != 2 or template.ndim != 2:
        raise ValueError(
            f"FC and prototypes must be 2D, got fc={fc.shape} prototypes={template.shape}"
        )
    if fc.shape[0] != valid.size:
        raise ValueError(f"FC/mask vertex mismatch: fc={fc.shape[0]} mask={valid.size}")
    if fc.shape[1] != template.shape[1]:
        raise ValueError(
            f"FC/prototype feature mismatch: fc={fc.shape[1]} prototypes={template.shape[1]}"
        )
    if template.shape[0] < 1:
        raise ValueError("At least one prototype is required")

    labels = np.zeros(valid.size, dtype=np.int32)
    scores = np.full((valid.size, template.shape[0]), np.nan, dtype=np.float32)
    margin = np.full(valid.size, np.nan, dtype=np.float32)
    for vertex_index in np.flatnonzero(valid):
        for prototype_index in range(template.shape[0]):
            scores[vertex_index, prototype_index] = _finite_pearson(
                fc[vertex_index],
                template[prototype_index],
            )
        finite_indices = np.flatnonzero(np.isfinite(scores[vertex_index]))
        if finite_indices.size == 0:
            continue
        best_index = int(
            finite_indices[np.argmax(scores[vertex_index, finite_indices])]
        )
        labels[vertex_index] = best_index + 1
        if finite_indices.size >= 2:
            ordered = np.sort(scores[vertex_index, finite_indices])
            margin[vertex_index] = float(ordered[-1] - ordered[-2])
    return labels, scores, margin
