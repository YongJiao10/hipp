#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import sys
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import nibabel as nib
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
CORE_WORKFLOW_DIR = REPO_ROOT / "scripts" / "core_workflow"
if str(CORE_WORKFLOW_DIR) not in sys.path:
    sys.path.insert(0, str(CORE_WORKFLOW_DIR))

from lib.gifti_labels import make_named_label_gifti
from lib.hippomaps_adapter import (
    DEFAULT_GRADIENT_RANDOM_STATE,
    DEFAULT_HIPPOMAPS_ROOT,
    ensure_hippomaps_on_path,
    fit_gradientmaps_pearson,
    reconstruct_full_surface,
)
from lib.fc_math import fisher_z_transform_fc
from lib.output_layout import (
    DEFAULT_HIPPUNFOLD_BATCH_ROOT,
    DEFAULT_KONG2022_MANIFEST_ROOT,
    DEFAULT_KONG2022_ROUTEA_ROOT,
    DEFAULT_KONG2022_SURFACE_ROOT,
    DEFAULT_KONG2022_TEMPLATE_ROOT,
    resolve_crossfit_template_matching_root,
)
from lib.surface_utils import long_axis_order


DEFAULT_SUBJECTS_MANIFEST = (
    DEFAULT_KONG2022_MANIFEST_ROOT
    / "hcp_7t_hippocampus_fmri_qc_pass_t1only_latest.txt"
)
DEFAULT_N_FOLDS = 5
DEFAULT_FOLD_SEED = 20260710
DEFAULT_K = 3
CLUSTER_LABEL_ITEMS = [
    (0, "Null", (158, 158, 158, 255)),
    (1, "Anterior", (220, 80, 75, 255)),
    (2, "Body", (235, 175, 70, 255)),
    (3, "Posterior", (70, 115, 190, 255)),
]


@dataclass(frozen=True)
class CrossfitConfig:
    shared_root: Path
    surface_root: Path
    template_surfaces_dir: Path
    hippunfold_batch_root: Path
    out_root: Path
    fold_by_subject: Mapping[str, int]
    coverage_threshold: float = 0.80
    n_components: int = 5
    gradient_random_state: int = 0
    cluster_seed: int = 0


@dataclass(frozen=True)
class FoldTemplateResult:
    fold_index: int
    smoothing: str
    hemi: str
    training_subjects: tuple[str, ...]
    heldout_subjects: tuple[str, ...]
    surface_path: Path
    support_mask: np.ndarray
    coverage_count: np.ndarray
    coverage_fraction: np.ndarray
    group_fc_full: np.ndarray
    gradients_full: np.ndarray
    labels_full: np.ndarray
    prototypes: np.ndarray
    fc_paths: Mapping[str, str]
    density_evidence: Mapping[str, str]
    template_dir: Path


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

    empty_folds = sorted(set(range(int(n_folds))) - set(normalized_folds.values()))
    if empty_folds:
        raise ValueError(f"Fold assignment contains empty folds: {empty_folds}")

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


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build family-aware cross-fitted RouteA hippocampal K=3 labels."
    )
    parser.add_argument("--subjects-manifest", type=Path, default=DEFAULT_SUBJECTS_MANIFEST)
    parser.add_argument("--family-map", type=Path)
    parser.add_argument("--allow-subject-only-folds", action="store_true")
    parser.add_argument("--n-folds", type=int, default=DEFAULT_N_FOLDS)
    parser.add_argument("--fold-seed", type=int, default=DEFAULT_FOLD_SEED)
    parser.add_argument("--shared-root", type=Path, default=DEFAULT_KONG2022_ROUTEA_ROOT)
    parser.add_argument("--surface-root", type=Path, default=DEFAULT_KONG2022_SURFACE_ROOT)
    parser.add_argument(
        "--hippunfold-batch-root",
        type=Path,
        default=DEFAULT_HIPPUNFOLD_BATCH_ROOT,
    )
    parser.add_argument(
        "--template-surfaces-dir",
        type=Path,
        default=DEFAULT_KONG2022_TEMPLATE_ROOT / "group_average_surfaces",
    )
    parser.add_argument(
        "--out-root",
        type=Path,
        default=resolve_crossfit_template_matching_root(),
    )
    parser.add_argument("--hippomaps-root", type=Path, default=DEFAULT_HIPPOMAPS_ROOT)
    parser.add_argument("--smoothings", nargs="+", default=["2mm", "4mm"])
    parser.add_argument("--hemis", nargs="+", choices=["L", "R"], default=["L", "R"])
    parser.add_argument("--coverage-threshold", type=float, default=0.80)
    parser.add_argument("--n-components", type=int, default=5)
    parser.add_argument(
        "--gradient-random-state",
        type=int,
        default=DEFAULT_GRADIENT_RANDOM_STATE,
    )
    parser.add_argument("--cluster-seed", type=int, default=0)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--resume", action="store_true")
    return parser


def _read_family_rows(path: Path) -> dict[str, str]:
    family_path = Path(path)
    if not family_path.exists():
        raise FileNotFoundError(f"Missing family map: {family_path}")
    delimiter = "\t" if family_path.suffix.lower() == ".tsv" else ","
    with family_path.open("r", encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream, delimiter=delimiter)
        fields = set(reader.fieldnames or [])
        required = {"subject", "family_id"}
        if not required.issubset(fields):
            raise ValueError(
                f"Family map requires columns subject,family_id; got {sorted(fields)}"
            )
        family_by_subject: dict[str, str] = {}
        for row in reader:
            subject = _normalize_subject(row["subject"])
            family_id = str(row["family_id"]).strip()
            if not family_id:
                raise ValueError(f"Empty family_id for subject {subject}")
            old = family_by_subject.get(subject)
            if old is not None and old != family_id:
                raise ValueError(f"Conflicting family entries for subject {subject}")
            family_by_subject[subject] = family_id
    return family_by_subject


def resolve_family_map(
    args: argparse.Namespace,
    subjects: Sequence[str],
) -> tuple[dict[str, str], dict[str, object]]:
    normalized_subjects = [_normalize_subject(subject) for subject in subjects]
    family_path = getattr(args, "family_map", None)
    if family_path is None:
        if not bool(getattr(args, "allow_subject_only_folds", False)):
            raise ValueError(
                "HCP cross-fitting requires --family-map with subject,family_id; "
                "use --allow-subject-only-folds only as an explicit waiver"
            )
        return (
            {subject: subject for subject in normalized_subjects},
            {
                "family_aware": False,
                "subject_only_waiver": True,
                "family_map_path": None,
            },
        )

    family_by_subject = _read_family_rows(Path(family_path))
    missing = [subject for subject in normalized_subjects if subject not in family_by_subject]
    if missing:
        raise ValueError(f"Missing family entries for subjects: {missing}")
    return (
        {subject: family_by_subject[subject] for subject in normalized_subjects},
        {
            "family_aware": True,
            "subject_only_waiver": False,
            "family_map_path": str(Path(family_path).resolve()),
        },
    )


def fold_manifest_payload(
    fold_by_subject: Mapping[str, int],
    *,
    n_folds: int,
    family_metadata: Mapping[str, object] | None = None,
    fold_seed: int | None = None,
) -> dict[str, object]:
    normalized = {
        _normalize_subject(subject): int(fold)
        for subject, fold in fold_by_subject.items()
    }
    if len(normalized) != len(fold_by_subject):
        raise ValueError("Duplicate subjects after normalizing fold assignment")
    invalid = {
        subject: fold
        for subject, fold in normalized.items()
        if not 0 <= fold < int(n_folds)
    }
    if invalid:
        raise ValueError(f"Fold ids must be in [0, {n_folds}); got {invalid}")
    all_subjects = sorted(normalized)
    folds: list[dict[str, object]] = []
    for fold_index in range(int(n_folds)):
        heldout = [
            subject for subject in all_subjects if normalized[subject] == fold_index
        ]
        training = [
            subject for subject in all_subjects if normalized[subject] != fold_index
        ]
        if not heldout or not training:
            raise ValueError(
                f"Fold {fold_index} has an empty training or held-out partition"
            )
        folds.append(
            {
                "fold_index": int(fold_index),
                "fold_id": f"fold-{fold_index + 1:02d}",
                "training_subjects": training,
                "heldout_subjects": heldout,
                "train_test_disjoint": set(training).isdisjoint(heldout),
            }
        )
    return {
        "n_folds": int(n_folds),
        "fold_seed": int(fold_seed) if fold_seed is not None else None,
        "n_subjects": len(all_subjects),
        "family": dict(family_metadata or {}),
        "fold_by_subject": normalized,
        "folds": folds,
    }


def load_group_gradient_module():
    module_name = "_crossfit_group_gradient"
    if module_name in sys.modules:
        return sys.modules[module_name]
    path = CORE_WORKFLOW_DIR / "05_01_compute_hippomaps_gradients.py"
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load group-gradient module: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def load_group_atlas_module():
    module_name = "_crossfit_group_atlas"
    if module_name in sys.modules:
        return sys.modules[module_name]
    path = CORE_WORKFLOW_DIR / "05_02_run_group_atlas.py"
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load group-atlas module: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _subject_valid_mask_path(surface_root: Path, subject: str, hemi: str) -> Path:
    return (
        Path(surface_root)
        / f"sub-{subject}"
        / "surface"
        / "tsnr_gate"
        / f"sub-{subject}_hemi-{hemi}_valid_mask.npy"
    )


def _subject_fc_path(shared_root: Path, subject: str, smoothing: str, hemi: str) -> Path:
    return (
        Path(shared_root)
        / f"sub-{subject}"
        / "fc"
        / "kong2022"
        / str(smoothing)
        / f"hemi_{hemi}"
        / "hipp_vertex_to_network_fc.npy"
    )


def _load_full_subject_fc(path: Path, valid_mask: np.ndarray) -> np.ndarray:
    if not Path(path).exists():
        raise FileNotFoundError(f"Missing subject FC: {path}")
    fc = np.load(path).astype(np.float32)
    valid = np.asarray(valid_mask, dtype=bool).reshape(-1)
    if fc.ndim != 2:
        raise ValueError(f"Expected 2D subject FC: {path} shape={fc.shape}")
    if fc.shape[0] == valid.size:
        return fc
    if fc.shape[0] == int(np.sum(valid)):
        full = np.full((valid.size, fc.shape[1]), np.nan, dtype=np.float32)
        full[np.flatnonzero(valid)] = fc
        return full
    raise ValueError(
        f"FC/mask vertex mismatch for {path}: rows={fc.shape[0]} "
        f"vertices={valid.size} valid={int(np.sum(valid))}"
    )


def _template_surface_path(template_surfaces_dir: Path, hemi: str) -> Path:
    return (
        Path(template_surfaces_dir)
        / f"group-average_hemi-{hemi}_space-corobl_den-512_label-hipp_midthickness.surf.gii"
    )


def _fold_partitions(
    fold_by_subject: Mapping[str, int],
    fold_index: int,
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    normalized = {
        _normalize_subject(subject): int(fold)
        for subject, fold in fold_by_subject.items()
    }
    heldout = tuple(sorted(subject for subject, fold in normalized.items() if fold == int(fold_index)))
    training = tuple(sorted(subject for subject, fold in normalized.items() if fold != int(fold_index)))
    if not heldout or not training:
        raise ValueError(f"Fold {fold_index} has an empty training or held-out partition")
    return training, heldout


def _fold_template_dir(
    out_root: Path,
    fold_index: int,
    smoothing: str,
    hemi: str,
) -> Path:
    return (
        Path(out_root)
        / "templates"
        / f"fold-{int(fold_index) + 1:02d}"
        / str(smoothing)
        / f"hemi_{hemi}"
    )


def validate_template_manifest(
    manifest: Mapping[str, object],
    config: CrossfitConfig,
    fold_index: int,
    smoothing: str,
    hemi: str,
) -> None:
    training, heldout = _fold_partitions(config.fold_by_subject, fold_index)
    expected_surface = _template_surface_path(config.template_surfaces_dir, hemi).resolve()
    expected = {
        "status": "ok",
        "fold_index": int(fold_index),
        "smoothing": str(smoothing),
        "hemi": str(hemi),
        "training_subjects": [f"sub-{subject}" for subject in training],
        "heldout_subjects": [f"sub-{subject}" for subject in heldout],
        "train_test_disjoint": True,
        "coverage_threshold": float(config.coverage_threshold),
        "n_components": int(config.n_components),
        "gradient_random_state": int(config.gradient_random_state),
        "cluster_seed": int(config.cluster_seed),
        "k_final": DEFAULT_K,
    }
    mismatches: list[str] = []
    for key, value in expected.items():
        actual = manifest.get(key)
        if isinstance(value, float):
            if actual is None or not np.isclose(float(actual), value):
                mismatches.append(f"{key}: expected={value!r} actual={actual!r}")
        elif actual != value:
            mismatches.append(f"{key}: expected={value!r} actual={actual!r}")
    density = manifest.get("density_evidence")
    consumed = density.get("consumed_asset") if isinstance(density, Mapping) else None
    if consumed is None or Path(str(consumed)).resolve() != expected_surface:
        mismatches.append(
            f"density_evidence.consumed_asset: expected={expected_surface} actual={consumed!r}"
        )
    if mismatches:
        raise ValueError("Resume template mismatch: " + "; ".join(mismatches))


def load_fold_template(
    config: CrossfitConfig,
    fold_index: int,
    smoothing: str,
    hemi: str,
) -> FoldTemplateResult:
    template_dir = _fold_template_dir(
        config.out_root,
        fold_index,
        smoothing,
        hemi,
    )
    manifest_path = template_dir / "template_manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Missing resume template manifest: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    validate_template_manifest(
        manifest,
        config,
        fold_index,
        smoothing,
        hemi,
    )
    array_paths = {
        "support": template_dir / "group_support_mask.npy",
        "coverage_count": template_dir / "coverage_count.npy",
        "coverage_fraction": template_dir / "coverage_fraction.npy",
        "group_fc": template_dir / "group_mean_fisherz_fc.npy",
        "gradients": template_dir / "group_gradients.npy",
        "labels": template_dir / "group_labels_k3.npy",
        "prototypes": template_dir / "template_profiles.npy",
    }
    missing = [str(path) for path in array_paths.values() if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing resume template arrays: {missing}")
    training, heldout = _fold_partitions(config.fold_by_subject, fold_index)
    surface_path = _template_surface_path(config.template_surfaces_dir, hemi).resolve()
    fc_paths_raw = manifest.get("fc_paths", {})
    fc_paths = {
        _normalize_subject(subject): str(path)
        for subject, path in dict(fc_paths_raw).items()
    }
    return FoldTemplateResult(
        fold_index=int(fold_index),
        smoothing=str(smoothing),
        hemi=str(hemi),
        training_subjects=training,
        heldout_subjects=heldout,
        surface_path=surface_path,
        support_mask=np.load(array_paths["support"]).astype(bool),
        coverage_count=np.load(array_paths["coverage_count"]).astype(np.int32),
        coverage_fraction=np.load(array_paths["coverage_fraction"]).astype(np.float32),
        group_fc_full=np.load(array_paths["group_fc"]).astype(np.float32),
        gradients_full=np.load(array_paths["gradients"]).astype(np.float32),
        labels_full=np.load(array_paths["labels"]).astype(np.int32),
        prototypes=np.load(array_paths["prototypes"]).astype(np.float32),
        fc_paths=fc_paths,
        density_evidence=dict(manifest["density_evidence"]),
        template_dir=template_dir.resolve(),
    )


def fit_or_load_fold_template(
    config: CrossfitConfig,
    fold_index: int,
    smoothing: str,
    hemi: str,
    *,
    resume: bool,
) -> FoldTemplateResult:
    manifest_path = (
        _fold_template_dir(config.out_root, fold_index, smoothing, hemi)
        / "template_manifest.json"
    )
    if resume and manifest_path.exists():
        return load_fold_template(config, fold_index, smoothing, hemi)
    return fit_fold_template(config, fold_index, smoothing, hemi)


def fit_fold_template(
    config: CrossfitConfig,
    fold_id: int,
    smoothing: str,
    hemi: str,
) -> FoldTemplateResult:
    fold_index = int(fold_id)
    training_subjects, heldout_subjects = _fold_partitions(
        config.fold_by_subject,
        fold_index,
    )
    if int(config.n_components) < DEFAULT_K:
        raise ValueError(f"n_components must be at least {DEFAULT_K}")
    surface_path = _template_surface_path(config.template_surfaces_dir, hemi)
    if not surface_path.exists():
        raise FileNotFoundError(f"Missing group template surface: {surface_path}")

    core = load_group_atlas_module()
    coords, faces = core.load_surface(surface_path)
    vertex_count = int(coords.shape[0])
    valid_masks: dict[str, np.ndarray] = {}
    feature_bank: dict[str, np.ndarray] = {}
    fc_paths: dict[str, str] = {}
    for subject in training_subjects:
        mask_path = _subject_valid_mask_path(config.surface_root, subject, hemi)
        if not mask_path.exists():
            raise FileNotFoundError(f"Missing subject valid mask: {mask_path}")
        valid_mask = np.load(mask_path).astype(bool).reshape(-1)
        if valid_mask.size != vertex_count:
            raise ValueError(
                f"Mask/template vertex mismatch for sub-{subject}: "
                f"mask={valid_mask.size} template={vertex_count}"
            )
        fc_path = _subject_fc_path(config.shared_root, subject, smoothing, hemi)
        valid_masks[subject] = valid_mask
        feature_bank[subject] = _load_full_subject_fc(fc_path, valid_mask)
        fc_paths[subject] = str(fc_path.resolve())

    coverage_count = np.sum(
        np.stack([valid_masks[subject] for subject in training_subjects], axis=0),
        axis=0,
    ).astype(np.int32)
    coverage_fraction = (
        coverage_count.astype(np.float32) / float(len(training_subjects))
    ).astype(np.float32)
    support_mask = coverage_fraction >= float(config.coverage_threshold)
    if not np.any(support_mask):
        raise RuntimeError(f"Fold {fold_index} support mask has zero vertices")

    group_gradient = load_group_gradient_module()
    group_fc_active = group_gradient.group_mean_fisherz_fc(
        feature_bank=feature_bank,
        subjects=list(training_subjects),
        support_mask=support_mask,
    )
    gradients_active, lambdas, ev_fractions = fit_gradientmaps_pearson(
        group_fc_active,
        n_components=int(config.n_components),
        random_state=int(config.gradient_random_state),
    )
    group_fc_full = reconstruct_full_surface(group_fc_active, support_mask, vertex_count)
    gradients_full = reconstruct_full_surface(gradients_active, support_mask, vertex_count)
    adjacency = core.build_surface_adjacency(faces, vertex_count)
    _labels_active, labels_full = core.cluster_from_precomputed_group_features(
        group_features=np.asarray(gradients_active[:, :DEFAULT_K], dtype=np.float32),
        support_mask=support_mask,
        adjacency_full=adjacency,
        k=DEFAULT_K,
        seed=int(config.cluster_seed),
    )
    labels_full = core.reorder_labels_by_long_axis(
        labels_full.astype(np.int32),
        long_axis_order(surface_path),
    ).astype(np.int32)
    prototypes = build_cluster_prototypes(
        group_fc_full,
        labels_full,
        support_mask,
        k=DEFAULT_K,
    )

    template_dir = _fold_template_dir(
        config.out_root,
        fold_index,
        smoothing,
        hemi,
    )
    template_dir.mkdir(parents=True, exist_ok=True)
    np.save(template_dir / "group_support_mask.npy", support_mask.astype(np.uint8))
    np.save(template_dir / "coverage_count.npy", coverage_count)
    np.save(template_dir / "coverage_fraction.npy", coverage_fraction)
    np.save(template_dir / "group_mean_fisherz_fc.npy", group_fc_full.astype(np.float32))
    np.save(template_dir / "group_gradients.npy", gradients_full.astype(np.float32))
    np.save(template_dir / "gradient_lambdas.npy", np.asarray(lambdas, dtype=np.float64))
    np.save(template_dir / "gradient_ev_fractions.npy", np.asarray(ev_fractions, dtype=np.float64))
    np.save(template_dir / "group_labels_k3.npy", labels_full)
    np.save(template_dir / "template_profiles.npy", prototypes)
    nib.save(
        make_named_label_gifti(labels_full, CLUSTER_LABEL_ITEMS),
        str(template_dir / "group_labels_k3.label.gii"),
    )
    density_evidence = {
        "consumed_asset": str(surface_path.resolve()),
        "filename_density": "512",
        "evidence_source": "consumed_template_surface_path",
    }
    manifest = {
        "status": "ok",
        "fold_index": fold_index,
        "fold_id": f"fold-{fold_index + 1:02d}",
        "smoothing": str(smoothing),
        "hemi": str(hemi),
        "training_subjects": [f"sub-{subject}" for subject in training_subjects],
        "heldout_subjects": [f"sub-{subject}" for subject in heldout_subjects],
        "train_test_disjoint": set(training_subjects).isdisjoint(heldout_subjects),
        "coverage_threshold": float(config.coverage_threshold),
        "n_components": int(config.n_components),
        "gradient_random_state": int(config.gradient_random_state),
        "cluster_seed": int(config.cluster_seed),
        "k_final": DEFAULT_K,
        "cluster_label_semantics": {"1": "Anterior", "2": "Body", "3": "Posterior"},
        "fc_paths": {f"sub-{subject}": path for subject, path in fc_paths.items()},
        "density_evidence": density_evidence,
    }
    _write_json(template_dir / "template_manifest.json", manifest)
    return FoldTemplateResult(
        fold_index=fold_index,
        smoothing=str(smoothing),
        hemi=str(hemi),
        training_subjects=training_subjects,
        heldout_subjects=heldout_subjects,
        surface_path=surface_path.resolve(),
        support_mask=support_mask,
        coverage_count=coverage_count,
        coverage_fraction=coverage_fraction,
        group_fc_full=group_fc_full.astype(np.float32),
        gradients_full=gradients_full.astype(np.float32),
        labels_full=labels_full,
        prototypes=prototypes,
        fc_paths=fc_paths,
        density_evidence=density_evidence,
        template_dir=template_dir.resolve(),
    )


def _subject_surface_path(
    hippunfold_batch_root: Path,
    subject: str,
    hemi: str,
) -> Path:
    return (
        Path(hippunfold_batch_root)
        / f"sub-{subject}"
        / "hippunfold"
        / f"sub-{subject}"
        / "surf"
        / f"sub-{subject}_hemi-{hemi}_space-corobl_den-512_label-hipp_midthickness.surf.gii"
    )


def _subject_combo_dir(
    out_root: Path,
    subject: str,
    smoothing: str,
    hemi: str,
) -> Path:
    return (
        Path(out_root)
        / "subjects"
        / f"sub-{_normalize_subject(subject)}"
        / str(smoothing)
        / f"hemi_{hemi}"
    )


def validate_subject_manifest(
    manifest: Mapping[str, object],
    config: CrossfitConfig,
    template: FoldTemplateResult,
    subject: str,
) -> None:
    subject_id = _normalize_subject(subject)
    expected_inputs = {
        "fc": str(
            _subject_fc_path(
                config.shared_root,
                subject_id,
                template.smoothing,
                template.hemi,
            ).resolve()
        ),
        "valid_mask": str(
            _subject_valid_mask_path(
                config.surface_root,
                subject_id,
                template.hemi,
            ).resolve()
        ),
        "subject_surface": str(
            _subject_surface_path(
                config.hippunfold_batch_root,
                subject_id,
                template.hemi,
            ).resolve()
        ),
        "template_profiles": str(
            (template.template_dir / "template_profiles.npy").resolve()
        ),
    }
    expected = {
        "status": "ok",
        "subject": f"sub-{subject_id}",
        "smoothing": str(template.smoothing),
        "hemi": str(template.hemi),
        "template_fold": int(template.fold_index),
        "template_training_subjects": [
            f"sub-{training_subject}" for training_subject in template.training_subjects
        ],
        "subject_was_in_template": False,
    }
    mismatches = [
        f"{key}: expected={value!r} actual={manifest.get(key)!r}"
        for key, value in expected.items()
        if manifest.get(key) != value
    ]
    actual_inputs = manifest.get("inputs")
    if not isinstance(actual_inputs, Mapping):
        mismatches.append("inputs: missing or not an object")
    else:
        for key, value in expected_inputs.items():
            actual = actual_inputs.get(key)
            if actual is None or Path(str(actual)).resolve() != Path(value):
                mismatches.append(f"inputs.{key}: expected={value!r} actual={actual!r}")
    if mismatches:
        raise ValueError("Resume subject mismatch: " + "; ".join(mismatches))


def load_existing_heldout_subject_outputs(
    config: CrossfitConfig,
    template: FoldTemplateResult,
    subject: str,
) -> dict[str, object]:
    combo_dir = _subject_combo_dir(
        config.out_root,
        subject,
        template.smoothing,
        template.hemi,
    )
    manifest_path = combo_dir / "combo_manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Missing resumed subject manifest: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    validate_subject_manifest(manifest, config, template, subject)
    outputs = manifest.get("outputs")
    if not isinstance(outputs, Mapping):
        raise ValueError(f"Resume subject mismatch: missing outputs in {manifest_path}")
    required = {
        "labels",
        "template_correlations",
        "correlation_margin",
        "label_gifti",
    }
    missing_keys = sorted(required - set(outputs))
    if missing_keys:
        raise ValueError(
            f"Resume subject mismatch: missing output keys {missing_keys} in {manifest_path}"
        )
    resolved = {key: Path(str(outputs[key])).resolve() for key in required}
    missing_files = [str(path) for path in resolved.values() if not path.exists()]
    if missing_files:
        raise FileNotFoundError(f"Missing resumed subject outputs: {missing_files}")
    resolved["manifest"] = manifest_path.resolve()
    return resolved


def write_or_load_heldout_subject_outputs(
    config: CrossfitConfig,
    template: FoldTemplateResult,
    subject: str,
    *,
    resume: bool,
) -> dict[str, object]:
    manifest_path = (
        _subject_combo_dir(
            config.out_root,
            subject,
            template.smoothing,
            template.hemi,
        )
        / "combo_manifest.json"
    )
    if resume and manifest_path.exists():
        return load_existing_heldout_subject_outputs(config, template, subject)
    return write_heldout_subject_outputs(config, template, subject)


def write_heldout_subject_outputs(
    config: CrossfitConfig,
    template: FoldTemplateResult,
    subject: str,
) -> dict[str, object]:
    subject_id = _normalize_subject(subject)
    if subject_id in set(template.training_subjects):
        raise ValueError(
            f"Refusing to label sub-{subject_id} with a template that includes the subject"
        )
    if subject_id not in set(template.heldout_subjects):
        raise ValueError(
            f"Subject sub-{subject_id} is not held out by fold {template.fold_index}"
        )

    mask_path = _subject_valid_mask_path(config.surface_root, subject_id, template.hemi)
    if not mask_path.exists():
        raise FileNotFoundError(f"Missing held-out valid mask: {mask_path}")
    valid_mask = np.load(mask_path).astype(bool).reshape(-1)
    fc_path = _subject_fc_path(
        config.shared_root,
        subject_id,
        template.smoothing,
        template.hemi,
    )
    fc_raw = _load_full_subject_fc(fc_path, valid_mask)
    fc_fisherz = fisher_z_transform_fc(fc_raw)
    labels, scores, margin = match_vertices_to_prototypes(
        fc_fisherz,
        valid_mask,
        template.prototypes,
    )
    subject_surface = _subject_surface_path(
        config.hippunfold_batch_root,
        subject_id,
        template.hemi,
    )
    if not subject_surface.exists():
        raise FileNotFoundError(f"Missing held-out subject surface: {subject_surface}")

    combo_dir = _subject_combo_dir(
        config.out_root,
        subject_id,
        template.smoothing,
        template.hemi,
    )
    combo_dir.mkdir(parents=True, exist_ok=True)
    labels_path = combo_dir / "individual_labels_k3.npy"
    scores_path = combo_dir / "template_correlations.npy"
    margin_path = combo_dir / "correlation_margin.npy"
    label_gifti_path = combo_dir / (
        f"sub-{subject_id}_hemi-{template.hemi}_space-corobl_den-512_"
        f"label-crossfit_template_matching_k3_{template.smoothing}.label.gii"
    )
    np.save(labels_path, labels.astype(np.int32))
    np.save(scores_path, scores.astype(np.float32))
    np.save(margin_path, margin.astype(np.float32))
    nib.save(
        make_named_label_gifti(labels, CLUSTER_LABEL_ITEMS),
        str(label_gifti_path),
    )

    finite_margin = margin[np.isfinite(margin)]
    label_counts = {
        str(label): int(np.sum(labels == label))
        for label in range(0, DEFAULT_K + 1)
    }
    density_evidence = {
        "consumed_subject_surface": str(subject_surface.resolve()),
        "consumed_template_surface": str(template.surface_path.resolve()),
        "filename_density": "512",
        "evidence_source": "consumed_surface_paths",
    }
    manifest = {
        "status": "ok",
        "subject": f"sub-{subject_id}",
        "smoothing": template.smoothing,
        "hemi": template.hemi,
        "template_fold": int(template.fold_index),
        "template_fold_id": f"fold-{template.fold_index + 1:02d}",
        "template_training_subjects": [
            f"sub-{training_subject}" for training_subject in template.training_subjects
        ],
        "subject_was_in_template": subject_id in set(template.training_subjects),
        "assignment_metric": "max_pearson_correlation_to_training_cluster_profile",
        "label_semantics": {"1": "Anterior", "2": "Body", "3": "Posterior"},
        "label_counts": label_counts,
        "margin_summary": {
            "n_finite": int(finite_margin.size),
            "mean": float(np.mean(finite_margin)) if finite_margin.size else None,
            "median": float(np.median(finite_margin)) if finite_margin.size else None,
            "fraction_below_0p05": (
                float(np.mean(finite_margin < 0.05)) if finite_margin.size else None
            ),
        },
        "inputs": {
            "fc": str(fc_path.resolve()),
            "valid_mask": str(mask_path.resolve()),
            "subject_surface": str(subject_surface.resolve()),
            "template_manifest": str(
                (template.template_dir / "template_manifest.json").resolve()
            ),
            "template_profiles": str(
                (template.template_dir / "template_profiles.npy").resolve()
            ),
        },
        "density_evidence": density_evidence,
        "outputs": {
            "labels": str(labels_path.resolve()),
            "template_correlations": str(scores_path.resolve()),
            "correlation_margin": str(margin_path.resolve()),
            "label_gifti": str(label_gifti_path.resolve()),
        },
    }
    manifest_path = combo_dir / "combo_manifest.json"
    _write_json(manifest_path, manifest)
    return {
        "labels": labels_path.resolve(),
        "template_correlations": scores_path.resolve(),
        "correlation_margin": margin_path.resolve(),
        "label_gifti": label_gifti_path.resolve(),
        "manifest": manifest_path.resolve(),
    }


def dry_run_payload(
    args: argparse.Namespace,
    subjects: Sequence[str],
    fold_by_subject: Mapping[str, int],
) -> dict[str, object]:
    normalized_subjects = tuple(_normalize_subject(subject) for subject in subjects)
    fold_payload = fold_manifest_payload(
        fold_by_subject,
        n_folds=int(args.n_folds),
        fold_seed=int(args.fold_seed),
    )
    missing: set[str] = set()
    for hemi in args.hemis:
        template_surface = _template_surface_path(args.template_surfaces_dir, hemi)
        if not template_surface.exists():
            missing.add(str(template_surface.resolve()))
        for subject in normalized_subjects:
            mask_path = _subject_valid_mask_path(args.surface_root, subject, hemi)
            if not mask_path.exists():
                missing.add(str(mask_path.resolve()))
            subject_surface = _subject_surface_path(
                args.hippunfold_batch_root,
                subject,
                hemi,
            )
            if not subject_surface.exists():
                missing.add(str(subject_surface.resolve()))
            for smoothing in args.smoothings:
                fc_path = _subject_fc_path(args.shared_root, subject, smoothing, hemi)
                if not fc_path.exists():
                    missing.add(str(fc_path.resolve()))

    overlap_count = 0
    for row in fold_payload["folds"]:
        overlap_count += len(
            set(row["training_subjects"]) & set(row["heldout_subjects"])
        )
    return {
        "status": "ok" if not missing else "missing_inputs",
        "n_subjects": len(normalized_subjects),
        "n_folds": int(args.n_folds),
        "smoothings": [str(value) for value in args.smoothings],
        "hemis": [str(value) for value in args.hemis],
        "template_fits": int(args.n_folds) * len(args.smoothings) * len(args.hemis),
        "subject_combo_mappings": (
            len(normalized_subjects) * len(args.smoothings) * len(args.hemis)
        ),
        "train_test_overlap_count": int(overlap_count),
        "missing_inputs": sorted(missing),
        "out_root": str(Path(args.out_root).resolve()),
        "writes_outputs": False,
        "max_concurrent_tasks": 1,
    }


def _load_subject_ids(path: Path) -> list[str]:
    manifest_path = Path(path)
    if not manifest_path.exists():
        raise FileNotFoundError(f"Missing subjects manifest: {manifest_path}")
    subjects = [
        _normalize_subject(line)
        for line in manifest_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not subjects:
        raise ValueError(f"No subjects in manifest: {manifest_path}")
    if len(set(subjects)) != len(subjects):
        raise ValueError(f"Duplicate subjects in manifest: {manifest_path}")
    return subjects


def _write_fold_assignments_tsv(
    path: Path,
    subjects: Sequence[str],
    family_by_subject: Mapping[str, str],
    fold_by_subject: Mapping[str, int],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream, delimiter="\t", lineterminator="\n")
        writer.writerow(["subject", "family_id", "fold_index", "fold_id"])
        for subject in sorted(_normalize_subject(value) for value in subjects):
            fold_index = int(fold_by_subject[subject])
            writer.writerow(
                [
                    f"sub-{subject}",
                    family_by_subject[subject],
                    fold_index,
                    f"fold-{fold_index + 1:02d}",
                ]
            )


def validate_output_root_policy(out_root: Path, *, resume: bool) -> None:
    manifest_path = Path(out_root) / "crossfit_manifest.json"
    if manifest_path.exists() and not resume:
        raise FileExistsError(
            f"Existing cross-fit run manifest requires --resume: {manifest_path}"
        )


def main(argv: Sequence[str] | None = None) -> int:
    args = build_arg_parser().parse_args(list(argv) if argv is not None else None)
    subjects = _load_subject_ids(args.subjects_manifest)
    family_by_subject, family_metadata = resolve_family_map(args, subjects)
    fold_by_subject = assign_grouped_folds(
        subjects,
        family_by_subject,
        n_folds=int(args.n_folds),
        seed=int(args.fold_seed),
    )
    audit = dry_run_payload(args, subjects, fold_by_subject)
    if args.dry_run:
        print(json.dumps(audit, indent=2, sort_keys=True))
        return 0 if audit["status"] == "ok" else 2
    if audit["status"] != "ok":
        print(json.dumps(audit, indent=2, sort_keys=True))
        return 2

    validate_output_root_policy(Path(args.out_root), resume=bool(args.resume))
    ensure_hippomaps_on_path(Path(args.hippomaps_root))
    out_root = Path(args.out_root)
    fold_payload = fold_manifest_payload(
        fold_by_subject,
        n_folds=int(args.n_folds),
        family_metadata=family_metadata,
        fold_seed=int(args.fold_seed),
    )
    fold_root = out_root / "folds"
    _write_fold_assignments_tsv(
        fold_root / "fold_assignments.tsv",
        subjects,
        family_by_subject,
        fold_by_subject,
    )
    _write_json(fold_root / "fold_manifest.json", fold_payload)
    config = CrossfitConfig(
        shared_root=Path(args.shared_root),
        surface_root=Path(args.surface_root),
        template_surfaces_dir=Path(args.template_surfaces_dir),
        hippunfold_batch_root=Path(args.hippunfold_batch_root),
        out_root=out_root,
        fold_by_subject=fold_by_subject,
        coverage_threshold=float(args.coverage_threshold),
        n_components=int(args.n_components),
        gradient_random_state=int(args.gradient_random_state),
        cluster_seed=int(args.cluster_seed),
    )
    template_fits = 0
    mapping_outputs: list[dict[str, str]] = []
    for fold_index in range(int(args.n_folds)):
        for smoothing in args.smoothings:
            for hemi in args.hemis:
                template = fit_or_load_fold_template(
                    config,
                    fold_index,
                    str(smoothing),
                    str(hemi),
                    resume=bool(args.resume),
                )
                template_fits += 1
                for subject in template.heldout_subjects:
                    outputs = write_or_load_heldout_subject_outputs(
                        config,
                        template,
                        subject,
                        resume=bool(args.resume),
                    )
                    mapping_outputs.append(
                        {key: str(value) for key, value in outputs.items()}
                    )
    run_manifest = {
        "status": "ok",
        "workflow": "routea_crossfit_template_matching_k3",
        "subjects_manifest": str(Path(args.subjects_manifest).resolve()),
        "n_subjects": len(subjects),
        "n_folds": int(args.n_folds),
        "fold_seed": int(args.fold_seed),
        "family": family_metadata,
        "smoothings": [str(value) for value in args.smoothings],
        "hemis": [str(value) for value in args.hemis],
        "coverage_threshold": float(args.coverage_threshold),
        "n_components": int(args.n_components),
        "gradient_random_state": int(args.gradient_random_state),
        "cluster_seed": int(args.cluster_seed),
        "k_final": DEFAULT_K,
        "template_fits_completed": int(template_fits),
        "subject_combo_mappings_completed": len(mapping_outputs),
        "fold_manifest": str((fold_root / "fold_manifest.json").resolve()),
        "fold_assignments": str((fold_root / "fold_assignments.tsv").resolve()),
        "subject_manifests": [row["manifest"] for row in mapping_outputs],
    }
    _write_json(out_root / "crossfit_manifest.json", run_manifest)
    print(json.dumps(run_manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


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
        sums = np.sum(np.where(finite, values, 0.0), axis=0, dtype=np.float64)
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
