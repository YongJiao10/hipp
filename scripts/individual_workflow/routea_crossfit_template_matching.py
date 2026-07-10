#!/usr/bin/env python3
from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence

import numpy as np


def _normalize_subject(raw: object) -> str:
    subject = str(raw).strip().removeprefix("sub-")
    if not subject:
        raise ValueError("Empty subject id")
    return subject


def _normalize_family_map(family_by_subject: Mapping[str, object]) -> dict[str, str]:
    normalized: dict[str, str] = {}
    for raw_subject, raw_family in family_by_subject.items():
        subject = _normalize_subject(raw_subject)
        family = str(raw_family).strip()
        if not family:
            raise ValueError(f"Empty family entry for subject {subject}")
        if subject in normalized and normalized[subject] != family:
            raise ValueError(f"Conflicting family entries for subject {subject}")
        normalized[subject] = family
    return normalized


def _normalize_subjects_and_families(
    subjects: Sequence[str],
    family_by_subject: Mapping[str, object],
) -> tuple[list[str], dict[str, str]]:
    normalized_subjects = [_normalize_subject(subject) for subject in subjects]
    if len(set(normalized_subjects)) != len(normalized_subjects):
        raise ValueError("Duplicate subjects after normalizing sub- prefixes")

    normalized_families = _normalize_family_map(family_by_subject)
    missing = [subject for subject in normalized_subjects if subject not in normalized_families]
    if missing:
        raise ValueError(f"Missing family entries for subjects: {', '.join(missing)}")
    return normalized_subjects, normalized_families


def validate_fold_assignment(
    subjects: Sequence[str],
    family_by_subject: Mapping[str, object],
    fold_by_subject: Mapping[str, int],
    n_folds: int,
) -> None:
    if n_folds < 1:
        raise ValueError("n_folds must be at least 1")

    normalized_subjects, normalized_families = _normalize_subjects_and_families(
        subjects,
        family_by_subject,
    )
    normalized_folds: dict[str, int] = {}
    for raw_subject, raw_fold in fold_by_subject.items():
        subject = _normalize_subject(raw_subject)
        fold = int(raw_fold)
        if subject in normalized_folds and normalized_folds[subject] != fold:
            raise ValueError(f"Conflicting fold entries for subject {subject}")
        normalized_folds[subject] = fold

    expected = set(normalized_subjects)
    assigned = set(normalized_folds)
    if assigned != expected:
        missing = sorted(expected - assigned)
        extra = sorted(assigned - expected)
        raise ValueError(f"Fold assignment subject mismatch; missing={missing}, extra={extra}")

    invalid = {
        subject: normalized_folds[subject]
        for subject in normalized_subjects
        if not 0 <= normalized_folds[subject] < n_folds
    }
    if invalid:
        raise ValueError(f"Fold ids must be in [0, {n_folds}); got {invalid}")

    fold_by_family: dict[str, int] = {}
    for subject in normalized_subjects:
        family = normalized_families[subject]
        fold = normalized_folds[subject]
        if family in fold_by_family and fold_by_family[family] != fold:
            raise ValueError(f"Family {family!r} is split across folds")
        fold_by_family[family] = fold


def assign_grouped_folds(
    subjects: Sequence[str],
    family_by_subject: Mapping[str, object],
    n_folds: int,
    seed: int,
) -> dict[str, int]:
    if n_folds < 1:
        raise ValueError("n_folds must be at least 1")

    normalized_subjects, normalized_families = _normalize_subjects_and_families(
        subjects,
        family_by_subject,
    )
    members_by_family: dict[str, list[str]] = defaultdict(list)
    for subject in normalized_subjects:
        members_by_family[normalized_families[subject]].append(subject)

    family_ids = sorted(members_by_family)
    rng = np.random.default_rng(seed)
    seeded_family_order = [family_ids[index] for index in rng.permutation(len(family_ids))]
    ordered_families = sorted(
        seeded_family_order,
        key=lambda family: -len(members_by_family[family]),
    )

    fold_counts = [0] * n_folds
    fold_by_subject: dict[str, int] = {}
    for family in ordered_families:
        fold = min(range(n_folds), key=lambda index: (fold_counts[index], index))
        for subject in members_by_family[family]:
            fold_by_subject[subject] = fold
        fold_counts[fold] += len(members_by_family[family])

    validate_fold_assignment(
        normalized_subjects,
        normalized_families,
        fold_by_subject,
        n_folds,
    )
    return fold_by_subject


def build_cluster_prototypes(
    group_fc_full: np.ndarray,
    labels: np.ndarray,
    support_mask: np.ndarray,
    *,
    k: int = 3,
) -> np.ndarray:
    fc = np.asarray(group_fc_full, dtype=np.float32)
    ordered_labels = np.asarray(labels, dtype=np.int32).reshape(-1)
    support = np.asarray(support_mask, dtype=bool).reshape(-1)
    if fc.ndim != 2:
        raise ValueError(f"Expected 2D group FC, got shape={fc.shape}")
    if fc.shape[0] != ordered_labels.size or ordered_labels.size != support.size:
        raise ValueError(
            "FC/label/support vertex mismatch: "
            f"fc={fc.shape[0]} labels={ordered_labels.size} support={support.size}"
        )
    if int(k) < 1:
        raise ValueError("k must be at least 1")

    prototypes = np.full((int(k), fc.shape[1]), np.nan, dtype=np.float32)
    for cluster_id in range(1, int(k) + 1):
        vertex_mask = support & (ordered_labels == cluster_id)
        if not np.any(vertex_mask):
            raise ValueError(f"No supported vertices for cluster {cluster_id}")
        values = fc[vertex_mask]
        finite = np.isfinite(values)
        counts = np.sum(finite, axis=0)
        sums = np.nansum(values, axis=0, dtype=np.float64)
        present = counts > 0
        prototypes[cluster_id - 1, present] = (
            sums[present] / counts[present]
        ).astype(np.float32)
    return prototypes


def _finite_pearson(a: np.ndarray, b: np.ndarray) -> float:
    left = np.asarray(a, dtype=np.float64).reshape(-1)
    right = np.asarray(b, dtype=np.float64).reshape(-1)
    finite = np.isfinite(left) & np.isfinite(right)
    if int(np.sum(finite)) < 2:
        return float("nan")
    x = left[finite] - float(np.mean(left[finite]))
    y = right[finite] - float(np.mean(right[finite]))
    denom = float(np.linalg.norm(x) * np.linalg.norm(y))
    if not np.isfinite(denom) or denom <= 0.0:
        return float("nan")
    corr = float(np.dot(x, y) / denom)
    return float(np.clip(corr, -1.0, 1.0)) if np.isfinite(corr) else float("nan")


def match_vertices_to_prototypes(
    fc_fisherz: np.ndarray,
    valid_mask: np.ndarray,
    prototypes: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    fc = np.asarray(fc_fisherz, dtype=np.float32)
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
            finite_indices[
                np.argmax(scores[vertex_index, finite_indices])
            ]
        )
        labels[vertex_index] = best_index + 1
        if finite_indices.size >= 2:
            ordered = np.sort(scores[vertex_index, finite_indices])
            margin[vertex_index] = float(ordered[-1] - ordered[-2])
    return labels, scores, margin
