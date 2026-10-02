#!/usr/bin/env python3
"""Group-level task-activation helpers shared by stage 05_06/06_11.

Design notes for seamless increments (no hardcoded rosters):

- The subject roster is never fixed here. Callers resolve it explicitly via
  ``--subjects`` / ``--subjects-manifest``, or fall back to auto-discovery:
  every subject directory under the task shared root holding the required
  contrast file for a smoothing/hemi combo.
- Group outputs live at stable paths. Rerunning after new subjects finish
  per-subject stages deterministically refreshes the same files; the roster
  and per-input receipts are recorded in the group
  manifest, so provenance stays exact without dated directories.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy.stats import t as t_dist

from .surface_utils import long_axis_order, vertex_area_weights
from .fc_math import bh_fdr
from .task_design import TASK_CONTRASTS, normalize_task_branch

ANATOMICAL_CLUSTER_NAMES = ("Anterior", "Body", "Posterior")

# Subregion zone branches. hcp151_atlas = rest group K=3 labels shared by all
# subjects; template_matching = crossfit individualized K=3 labels per subject.
# Names mirror the JJ_TMS PARCELLATIONS convention for cross-project consistency.
PARCELLATION_HCP151_ATLAS = "hcp151_atlas"
PARCELLATION_TEMPLATE_MATCHING = "template_matching"
PARCELLATION_CHOICES = (
    PARCELLATION_HCP151_ATLAS,
    PARCELLATION_TEMPLATE_MATCHING,
)


def two_tailed_p(tstat: float, n: int) -> float:
    """Two-tailed one-sample t-test p-value (NaN-aware, needs n>=2)."""
    if not np.isfinite(tstat) or n < 2:
        return float("nan")
    return float(2.0 * t_dist.sf(abs(float(tstat)), int(n) - 1))


def sig_marker(p: float) -> str:
    """Significance marker: '+' 0.01<=p<0.05, '*' 0.001<=p<0.01,
    '**' 0.0001<=p<0.001, '***' p<0.0001, '' otherwise."""
    if not np.isfinite(p):
        return ""
    if p < 0.0001:
        return "***"
    if p < 0.001:
        return "**"
    if p < 0.01:
        return "*"
    if p < 0.05:
        return "+"
    return ""


def fdr_adjust(p_values: list[float]) -> list[float]:
    """Benjamini-Hochberg FDR over one family of p-values (NaN-safe)."""
    adjusted = bh_fdr(np.asarray(p_values, dtype=np.float64))
    return [float(v) for v in adjusted]


def subject_contrast_path(
    shared_root: Path, *, subject: str, task: str, smoothing: str, hemi: str, contrast: str
) -> Path:
    return (
        Path(shared_root)
        / f"sub-{subject}"
        / "task"
        / task
        / smoothing
        / f"hemi_{hemi}"
        / f"contrast_{contrast}.npy"
    )


def discover_subjects(
    shared_root: Path,
    *,
    task: str,
    smoothing: str,
    hemi: str,
    contrast: str,
) -> list[str]:
    """Auto-discover subjects with a complete contrast file for one combo."""
    branch = normalize_task_branch(task)
    found: list[str] = []
    for subject_dir in sorted(Path(shared_root).glob("sub-*")):
        if not subject_dir.is_dir():
            continue
        subject = subject_dir.name.removeprefix("sub-")
        path = subject_contrast_path(
            shared_root,
            subject=subject,
            task=branch,
            smoothing=smoothing,
            hemi=hemi,
            contrast=contrast,
        )
        if path.is_file():
            found.append(subject)
    return found


def discover_contrasts(
    shared_root: Path, *, task: str, smoothing: str, hemi: str
) -> list[str]:
    """Auto-discover maintained contrast names with at least one subject file."""
    branch = normalize_task_branch(task)
    names = sorted(TASK_CONTRASTS[branch].keys())
    kept: list[str] = []
    probe_subjects = sorted(
        p.name.removeprefix("sub-")
        for p in Path(shared_root).glob("sub-*")
        if p.is_dir()
    )
    for name in names:
        for subject in probe_subjects:
            if subject_contrast_path(
                shared_root,
                subject=subject,
                task=branch,
                smoothing=smoothing,
                hemi=hemi,
                contrast=name,
            ).is_file():
                kept.append(name)
                break
    return kept


def load_subject_stack(
    shared_root: Path,
    *,
    subjects: list[str],
    task: str,
    smoothing: str,
    hemi: str,
    contrast: str,
) -> tuple[np.ndarray, list[str]]:
    """Stack per-subject contrast maps; fail closed on shape mismatch."""
    branch = normalize_task_branch(task)
    rows: list[np.ndarray] = []
    used: list[str] = []
    expected: tuple[int, ...] | None = None
    for subject in subjects:
        path = subject_contrast_path(
            shared_root,
            subject=subject,
            task=branch,
            smoothing=smoothing,
            hemi=hemi,
            contrast=contrast,
        )
        if not path.is_file():
            raise FileNotFoundError(
                f"Roster subject lacks contrast file: sub-{subject} {task}/{contrast} "
                f"{smoothing}/hemi-{hemi}: {path}"
            )
        arr = np.load(str(path)).astype(np.float64).reshape(-1)
        if expected is None:
            expected = arr.shape
        elif arr.shape != expected:
            raise ValueError(
                f"Contrast shape mismatch for sub-{subject}: {arr.shape} vs {expected}"
            )
        rows.append(arr)
        used.append(subject)
    if not rows:
        raise ValueError(f"Empty roster for {task}/{contrast} {smoothing}/hemi-{hemi}")
    return np.stack(rows, axis=0), used


def subject_beta_dir(
    shared_root: Path, *, subject: str, task: str, smoothing: str, hemi: str
) -> Path:
    return (
        Path(shared_root)
        / f"sub-{subject}"
        / "task"
        / normalize_task_branch(task)
        / smoothing
        / f"hemi_{hemi}"
    )


def load_subject_betas(
    shared_root: Path,
    *,
    subjects: list[str],
    task: str,
    smoothing: str,
    hemi: str,
) -> tuple[dict[str, np.ndarray], list[str]]:
    """Stack run-averaged per-condition betas; returns {stem: n_subj x n_vert}.

    Regressor order comes from each subject's ``task_activation_summary.json``
    and must agree across the roster; any disagreement fails closed.
    """
    branch = normalize_task_branch(task)
    stems: list[str] | None = None
    columns: dict[str, list[np.ndarray]] = {}
    for subject in subjects:
        task_dir = subject_beta_dir(
            shared_root, subject=subject, task=branch, smoothing=smoothing, hemi=hemi
        )
        betas_path = task_dir / "task_betas.npy"
        summary_path = task_dir / "task_activation_summary.json"
        if not betas_path.is_file():
            raise FileNotFoundError(f"Missing task betas for sub-{subject}: {betas_path}")
        if not summary_path.is_file():
            raise FileNotFoundError(f"Missing task activation summary for sub-{subject}: {summary_path}")
        names = list(json.loads(summary_path.read_text(encoding="utf-8"))["regressor_names"])
        if stems is None:
            stems = names
            for stem in stems:
                columns[stem] = []
        elif names != stems:
            raise ValueError(
                f"Regressor order mismatch for sub-{subject}: {names} vs {stems}"
            )
        betas = np.load(str(betas_path)).astype(np.float64)
        if betas.ndim != 2 or betas.shape[1] != len(stems):
            raise ValueError(
                f"Beta shape mismatch for sub-{subject}: {betas.shape} vs stems={stems}"
            )
        expected_vertices: tuple[int, ...] | None = (
            columns[stems[0]][0].shape if columns[stems[0]] else None
        )
        for index, stem in enumerate(stems):
            column = betas[:, index].reshape(-1)
            if expected_vertices is not None and column.shape != expected_vertices:
                raise ValueError(f"Beta vertex mismatch for sub-{subject} {stem}")
            columns[stem].append(column)
    if stems is None:
        raise ValueError(f"Empty roster for {task} betas {smoothing}/hemi-{hemi}")
    assert stems is not None
    return {stem: np.stack(columns[stem], axis=0) for stem in stems}, stems


def group_summary_stats(stack: np.ndarray) -> dict[str, np.ndarray]:
    """Mean, one-sample t, and Cohen's d per vertex (NaN-aware)."""
    x = np.asarray(stack, dtype=np.float64)
    if x.ndim != 2 or x.shape[0] < 2:
        raise ValueError(f"Need >=2 subjects for group stats, got shape {x.shape}")
    n = np.sum(np.isfinite(x), axis=0).astype(np.float64)
    mean = np.nanmean(x, axis=0)
    sd = np.nanstd(x, axis=0, ddof=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        tstat = np.where(
            (sd > 0) & (n >= 2), mean / (sd / np.sqrt(np.maximum(n, 1.0))), np.nan
        )
        cohens_d = np.where(sd > 0, mean / sd, np.nan)
    tstat[n < 2] = np.nan
    return {"mean": mean, "tstat": tstat, "cohens_d": cohens_d, "n": n}


def load_group_labels(
    group_atlas_root: Path, *, smoothing: str, hemi: str, k: int = 3
) -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
    """Load final K-cluster labels, support mask, and manifest for one combo.

    Uses ``group_labels_final.npy`` (not ``group_labels_k{k}.npy``, whose
    k-means IDs are arbitrary) and fails closed unless the manifest's
    ``chosen_k_final`` and ``cluster_order_head_to_tail`` match.
    """
    combo_dir = (
        Path(group_atlas_root)
        / "gradient-spectral"
        / "kong2022"
        / smoothing
        / "cov_0p80"
        / "g1g2g3"
        / f"hemi_{hemi}"
    )
    labels_path = combo_dir / "group_labels_final.npy"
    mask_path = combo_dir / "group_support_mask.npy"
    manifest_path = combo_dir / "group_atlas_manifest.json"
    for path in (labels_path, mask_path, manifest_path):
        if not path.is_file():
            raise FileNotFoundError(f"Missing group-atlas asset: {path}")
    labels = np.asarray(np.load(str(labels_path)), dtype=np.int32).reshape(-1)
    mask = np.asarray(np.load(str(mask_path))).reshape(-1).astype(bool)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if labels.shape != mask.shape:
        raise ValueError(f"Label/mask shape mismatch: {labels.shape} vs {mask.shape}")
    if int(manifest.get("chosen_k_final", -1)) != int(k):
        raise ValueError(
            f"Group atlas chosen_k_final={manifest.get('chosen_k_final')} != requested k={k}: "
            f"{manifest_path}"
        )
    return labels, mask, manifest


def anatomical_cluster_names(
    *,
    manifest: dict[str, object],
    group_labels: np.ndarray,
    surface_path: Path,
) -> dict[int, str]:
    """Map cluster IDs to Anterior/Body/Posterior via the verified head-to-tail order.

    Mirrors the 06_10 contract: ``cluster_order_head_to_tail`` from the group
    manifest, validated against ``long_axis_order`` centroids, named
    Anterior/Body/Posterior for K=3.
    """
    labels = np.asarray(group_labels, dtype=np.int32)
    order = [int(v) for v in manifest.get("cluster_order_head_to_tail", [])]
    active_ids = sorted(int(v) for v in np.unique(labels) if int(v) > 0)
    if not order:
        raise ValueError("Missing cluster_order_head_to_tail for anatomical naming")
    if set(order) != set(active_ids) or len(order) != len(active_ids):
        raise ValueError(
            f"cluster_order_head_to_tail disagrees with labels: order={order} ids={active_ids}"
        )
    axis = np.asarray(long_axis_order(surface_path), dtype=np.int32)
    if axis.ndim != 1 or axis.size != labels.size or set(axis.tolist()) != set(
        range(labels.size)
    ):
        raise ValueError("Invalid long-axis vertex order for head-to-tail validation")
    rank = np.empty(axis.size, dtype=np.float64)
    rank[axis] = np.arange(axis.size, dtype=np.float64)
    centroids = {
        cluster_id: float(np.mean(rank[labels == cluster_id])) for cluster_id in active_ids
    }
    positions = [centroids[c] for c in order]
    if any(right <= left for left, right in zip(positions, positions[1:])):
        raise ValueError(f"Head-to-tail order violated: positions={positions}")
    if len(order) == 3:
        names = ANATOMICAL_CLUSTER_NAMES
    else:
        names = tuple(f"C{cluster_id:02d}" for cluster_id in order)
    return {cluster_id: str(name) for cluster_id, name in zip(order, names)}


def subject_cluster_rows(
    stack: np.ndarray,
    subjects: list[str],
    *,
    group_labels: np.ndarray,
    support_mask: np.ndarray,
    names_by_id: dict[int, str],
    vertex_areas: dict[str, np.ndarray],
    min_valid_area_fraction: float = 0.10,
) -> list[dict[str, object]]:
    """Per-subject native-area-weighted mean contrast per cluster.

    Aggregation is area-weighted only (no unweighted twin): each subject's
    own midthickness vertex areas weight its support vertices, so every
    cluster mean is a tissue-area mean, not a vertex-count mean.

    Coverage gate: a subject-zone entry joins statistics only when its
    finite (valid) tissue area reaches ``min_valid_area_fraction`` of the
    zone's total support area. Gated-out entries are NaN with
    ``excluded_low_coverage=True`` so a single surviving vertex can never
    stand in for a whole partition.
    """
    if not 0.0 < float(min_valid_area_fraction) <= 1.0:
        raise ValueError(
            f"min_valid_area_fraction must be in (0, 1], got {min_valid_area_fraction}"
        )
    rows: list[dict[str, object]] = []
    for subject, values in zip(subjects, stack):
        areas = np.asarray(vertex_areas[subject], dtype=np.float64).reshape(-1)
        if areas.shape != values.shape:
            raise ValueError(
                f"Area/contrast shape mismatch for sub-{subject}: "
                f"{areas.shape} vs {values.shape}"
            )
        for cluster_id in sorted(names_by_id):
            sel = support_mask & (group_labels == cluster_id)
            vertex_values = np.asarray(values[sel], dtype=np.float64)
            vertex_weights = np.asarray(areas[sel], dtype=np.float64)
            finite = np.isfinite(vertex_values) & np.isfinite(vertex_weights)
            finite_weights = vertex_weights[finite]
            finite_values = vertex_values[finite]
            area_total = float(np.sum(vertex_weights[np.isfinite(vertex_weights)]))
            weight_total = float(np.sum(finite_weights))
            valid_frac = weight_total / area_total if area_total > 0 else 0.0
            gated_out = valid_frac < float(min_valid_area_fraction)
            if finite_values.size and weight_total > 0 and not gated_out:
                mean = float(np.sum(finite_values * finite_weights) / weight_total)
                var = float(
                    np.sum(finite_weights * (finite_values - mean) ** 2) / weight_total
                )
                sd = float(np.sqrt(max(var, 0.0)))
            else:
                mean, sd = float("nan"), float("nan")
            rows.append(
                {
                    "subject": f"sub-{subject}",
                    "cluster_id": int(cluster_id),
                    "cluster_name": names_by_id[cluster_id],
                    "n_vertices": int(np.sum(sel)),
                    "n_finite": int(np.sum(finite)),
                    "area_mm2": float(weight_total),
                    "area_total_mm2": float(area_total),
                    "valid_area_fraction": float(valid_frac),
                    "excluded_low_coverage": bool(gated_out),
                    "mean": mean,
                    "sd": sd,
                }
            )
    return rows


def native_midthickness_surface(
    hippunfold_batch_root: Path, *, subject: str, hemi: str
) -> Path:
    """Subject-native den-512 midthickness surface (same asset 02_01 samples)."""
    subject_id = str(subject).strip().removeprefix("sub-")
    path = (
        Path(hippunfold_batch_root)
        / f"sub-{subject_id}"
        / "hippunfold"
        / f"sub-{subject_id}"
        / "surf"
        / f"sub-{subject_id}_hemi-{hemi}_space-corobl_den-512_label-hipp_midthickness.surf.gii"
    )
    if not path.is_file():
        raise FileNotFoundError(
            f"Missing native midthickness surface for sub-{subject_id} hemi-{hemi}: {path}"
        )
    return path


def native_vertex_areas(
    hippunfold_batch_root: Path, *, subject: str, hemi: str
) -> np.ndarray:
    """Native per-vertex tissue areas for one subject/hemi."""
    areas = vertex_area_weights(
        native_midthickness_surface(hippunfold_batch_root, subject=subject, hemi=hemi)
    )
    if areas.shape != (512,):
        raise ValueError(f"Expected 512 native vertex areas, got {areas.shape}")
    return areas


def individual_labels_path(
    crossfit_subjects_root: Path, *, subject: str, smoothing: str, hemi: str
) -> Path:
    subject_id = str(subject).strip().removeprefix("sub-")
    return (
        Path(crossfit_subjects_root)
        / f"sub-{subject_id}"
        / smoothing
        / f"hemi_{hemi}"
        / "individual_labels_k3.npy"
    )


def resolve_zone_subjects(
    *,
    zone: str,
    used: list[str],
    smoothing: str,
    hemi: str,
    crossfit_root: Path,
) -> tuple[list[str], dict[str, str]]:
    """Resolve the effective roster and provenance notes for one zone.

    Template-matching labels are functional-identity labels: a vertex labeled
    2 functions like template cluster 2 (Body) wherever it sits, so no
    spatial monotonicity is required or checked. Subjects without a TM label
    file are excluded from this zone only and recorded; the group zone keeps
    the full roster.
    """
    if zone == PARCELLATION_HCP151_ATLAS:
        return list(used), {s: "group K=3 labels" for s in used}
    if zone != PARCELLATION_TEMPLATE_MATCHING:
        raise ValueError(f"Unknown zone: {zone!r}; expected one of {PARCELLATION_CHOICES}")
    kept: list[str] = []
    notes: dict[str, str] = {}
    for subject in used:
        label_path = individual_labels_path(
            crossfit_root, subject=subject, smoothing=smoothing, hemi=hemi
        )
        if not label_path.is_file():
            notes[subject] = f"excluded from {zone}: missing {label_path}"
            continue
        kept.append(subject)
        notes[subject] = "individual K=3 labels"
    return kept, notes
