#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path

import nibabel as nib
import numpy as np
from scipy import sparse
from scipy import special
from scipy.optimize import brentq, linear_sum_assignment

from surface_utils import (
    infer_density_from_surface,
    infer_spec_path,
    load_network_colors,
    long_axis_order,
    make_network_label_gifti,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SHARED_ROOT = REPO_ROOT / "outputs_migration" / "hipp_functional_parcellation_network"
DEFAULT_MODEL_ROOT = REPO_ROOT / "outputs_migration" / "hipp_mshbm_kong2019"
DEFAULT_SUBJECT_MANIFEST = REPO_ROOT / "manifests" / "hcp_7t_hippocampus_fmri_qc_pass_153.txt"
DEFAULT_DENSE_COROBL_ROOT = REPO_ROOT / "outputs_migration" / "dense_corobl_batch"
NETWORK_STYLE_JSON = REPO_ROOT / "config" / "hipp_network_style.json"
CROSS_ATLAS_MERGE_JSON = REPO_ROOT / "config" / "cross_atlas_network_merge.json"
DEFAULT_ATLASES = ["kong2019", "lynch2024"]
DEFAULT_SMOOTHINGS = ["2mm", "4mm"]
RUN_IDS = ["1", "2", "3", "4"]
HEMIS = ["L", "R"]
MODEL_FAMILY = "hipp_network_mshbm_kong2019_cbig_like_adaptation"
MODEL_VERSION = "best"
ARTIFACT_VERSION = "2026-04-19"
NETWORK_COLORS = load_network_colors(NETWORK_STYLE_JSON)
TOPK_FRACTION = 0.10
DEFAULT_NUM_INITIALIZATION = 300
DEFAULT_W_GRID = [10.0, 25.0, 50.0, 100.0, 200.0]
DEFAULT_C_GRID = [0.0, 10.0, 25.0, 50.0]
_MERGE_CONFIG = json.loads(CROSS_ATLAS_MERGE_JSON.read_text(encoding="utf-8"))


class DataConsistencyError(RuntimeError):
    pass


@dataclass(frozen=True)
class Combo:
    atlas: str
    smoothing: str
    hemi: str


@dataclass(frozen=True)
class SplitSpec:
    train: list[str]
    validation: list[str]
    test: list[str]
    path: Path


@dataclass(frozen=True)
class AtlasReference:
    roi_labels: list[str]
    parent_networks: list[str]
    roi_parent_index: np.ndarray
    roi_timeseries: np.ndarray
    run_bounds: list[tuple[int, int]]


@dataclass(frozen=True)
class SubjectProfiles:
    subject: str
    atlas: str
    smoothing: str
    hemi: str
    valid_mask: np.ndarray
    profiles: np.ndarray  # T x V x D
    session_present: np.ndarray  # T
    parent_networks: list[str]
    roi_labels: list[str]


@dataclass(frozen=True)
class GroupArtifacts:
    params_path: Path
    compat_npz_path: Path
    manifest_path: Path
    summary_path: Path
    theta_wta_path: Path


def current_manifest_meta() -> dict[str, str]:
    return {
        "model_family": MODEL_FAMILY,
        "model_version": MODEL_VERSION,
        "artifact_version": ARTIFACT_VERSION,
    }


def payload_matches_current(payload: dict[str, object] | None) -> bool:
    if not payload:
        return False
    return (
        str(payload.get("model_family", "")) == MODEL_FAMILY
        and str(payload.get("artifact_version", "")) == ARTIFACT_VERSION
    )


@dataclass(frozen=True)
class TuneArtifacts:
    best_w: float
    best_c: float
    best_path: Path
    summary_path: Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Best-faith hippocampal adaptation of CBIG Kong2019 MS-HBM using "
            "shared seed-basis ROI profiles, atlas-native parent-network labels, "
            "and official-style step1/step2/tune/step3 stages."
        )
    )
    parser.add_argument("--stage", choices=["step1", "step2", "tune", "step3", "all"], default="all")
    parser.add_argument("--shared-root", default=str(DEFAULT_SHARED_ROOT))
    parser.add_argument("--model-root", default=str(DEFAULT_MODEL_ROOT))
    parser.add_argument("--subject-manifest", default=str(DEFAULT_SUBJECT_MANIFEST))
    parser.add_argument("--subjects", nargs="+", default=None)
    parser.add_argument("--split-manifest", default=None)
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument("--train-frac", type=float, default=0.70)
    parser.add_argument("--validation-frac", type=float, default=0.15)
    parser.add_argument("--atlases", nargs="+", default=DEFAULT_ATLASES)
    parser.add_argument("--smoothings", nargs="+", default=DEFAULT_SMOOTHINGS)
    parser.add_argument("--hemis", nargs="+", choices=HEMIS, default=HEMIS)
    parser.add_argument("--target-subjects", nargs="+", default=None)
    parser.add_argument("--subject-set", choices=["validation", "test"], default="test")
    parser.add_argument("--topk-fraction", type=float, default=TOPK_FRACTION)
    parser.add_argument("--num-initialization", type=int, default=DEFAULT_NUM_INITIALIZATION)
    parser.add_argument("--init-max-iters", type=int, default=40)
    parser.add_argument("--step2-max-iters", type=int, default=12)
    parser.add_argument("--intra-iters", type=int, default=4)
    parser.add_argument("--inter-iters", type=int, default=4)
    parser.add_argument("--infer-iters", type=int, default=10)
    parser.add_argument("--theta-floor", type=float, default=1e-5)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--w-grid", nargs="+", type=float, default=DEFAULT_W_GRID)
    parser.add_argument("--c-grid", nargs="+", type=float, default=DEFAULT_C_GRID)
    parser.add_argument("--resume-mode", choices=["force", "resume"], default="resume")
    return parser.parse_args()


def save_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def load_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_table_rows(path: Path) -> list[dict[str, str]]:
    lines = [line.rstrip("\n") for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not lines:
        raise DataConsistencyError(f"Empty TSV file: {path}")
    header = lines[0].split("\t")
    rows = []
    for line in lines[1:]:
        values = line.split("\t")
        rows.append({key: values[idx] if idx < len(values) else "" for idx, key in enumerate(header)})
    return rows


def load_subject_manifest(args: argparse.Namespace) -> list[str]:
    if args.subjects:
        return [str(x) for x in args.subjects]
    path = Path(args.subject_manifest)
    if not path.exists():
        raise FileNotFoundError(f"Missing subject manifest: {path}")
    subjects = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not subjects:
        raise ValueError(f"Subject manifest is empty: {path}")
    return subjects


def deterministic_split(subjects: list[str], seed: int, train_frac: float, validation_frac: float) -> tuple[list[str], list[str], list[str]]:
    total = len(subjects)
    if total < 3:
        raise ValueError("Need at least 3 subjects for train/validation/test split")
    shuffled = list(subjects)
    rng = np.random.default_rng(seed)
    rng.shuffle(shuffled)
    n_train = max(1, int(round(total * train_frac)))
    n_validation = max(1, int(round(total * validation_frac)))
    if n_train + n_validation >= total:
        n_validation = max(1, total - n_train - 1)
    n_test = total - n_train - n_validation
    if n_test <= 0:
        n_test = 1
        if n_train > n_validation:
            n_train -= 1
        else:
            n_validation -= 1
    train = sorted(shuffled[:n_train])
    validation = sorted(shuffled[n_train:n_train + n_validation])
    test = sorted(shuffled[n_train + n_validation:])
    if not train or not validation or not test:
        raise RuntimeError("Deterministic split unexpectedly produced an empty split")
    return train, validation, test


def resolve_split(args: argparse.Namespace, model_root: Path) -> SplitSpec:
    if args.split_manifest:
        path = Path(args.split_manifest).resolve()
        payload = load_json(path)
        train = [str(x) for x in payload["train_subjects"]]
        validation = [str(x) for x in payload["validation_subjects"]]
        test = [str(x) for x in payload["test_subjects"]]
        return SplitSpec(train=train, validation=validation, test=test, path=path)

    subjects = load_subject_manifest(args)
    train, validation, test = deterministic_split(subjects, args.split_seed, args.train_frac, args.validation_frac)
    split_dir = model_root / "splits"
    split_path = split_dir / (
        f"hcp7t_qc153_seed-{args.split_seed}_train-{len(train)}_validation-{len(validation)}_test-{len(test)}.json"
    )
    if not split_path.exists() or args.resume_mode == "force":
        save_json(
            split_path,
            {
                "source_subject_manifest": str(Path(args.subject_manifest).resolve()) if not args.subjects else "cli_subjects",
                "split_seed": int(args.split_seed),
                "train_frac": float(args.train_frac),
                "validation_frac": float(args.validation_frac),
                "test_frac": float(max(0.0, 1.0 - args.train_frac - args.validation_frac)),
                "n_subjects": int(len(subjects)),
                "train_subjects": train,
                "validation_subjects": validation,
                "test_subjects": test,
            },
        )
    return SplitSpec(train=train, validation=validation, test=test, path=split_path)


def get_subjects_for_step(split: SplitSpec, subject_set: str, target_subjects: list[str] | None) -> list[str]:
    base = split.validation if subject_set == "validation" else split.test
    if not target_subjects:
        return base
    target = {str(x) for x in target_subjects}
    selected = [sub for sub in base if sub in target]
    if not selected:
        raise ValueError(f"No target subjects remain in {subject_set} split: {sorted(target)}")
    return selected


def zscore_rows(ts: np.ndarray) -> np.ndarray:
    arr = np.asarray(ts, dtype=np.float32)
    arr = arr - np.nanmean(arr, axis=1, keepdims=True)
    sd = np.nanstd(arr, axis=1, keepdims=True)
    sd = np.where(sd < 1e-6, 1.0, sd)
    return (arr / sd).astype(np.float32)


def normalize_rows(mat: np.ndarray) -> np.ndarray:
    arr = np.asarray(mat, dtype=np.float32)
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    norms = np.where(norms < 1e-6, 1.0, norms)
    return (arr / norms).astype(np.float32)


def stable_softmax(logits: np.ndarray, axis: int = 1) -> np.ndarray:
    shifted = logits - np.max(logits, axis=axis, keepdims=True)
    expv = np.exp(shifted)
    denom = np.sum(expv, axis=axis, keepdims=True)
    denom = np.where(denom <= 0.0, 1.0, denom)
    return (expv / denom).astype(np.float32)


def inv_ad(dim: int, rbar: np.ndarray) -> np.ndarray:
    values = np.clip(np.asarray(rbar, dtype=np.float64), 1e-6, 0.999999)
    out = np.zeros(values.shape, dtype=np.float64)
    order = float(dim) / 2.0 - 1.0

    def a_d(kappa: float) -> float:
        num = float(special.ive(order + 1.0, kappa))
        den = float(special.ive(order, kappa))
        if den == 0.0 or not np.isfinite(num) or not np.isfinite(den):
            raise FloatingPointError("ive instability while evaluating A_D")
        return num / den

    it = np.nditer(values, flags=["multi_index"])
    for r in it:
        r_val = float(r)
        approx = (float(dim) - 1.0) * r_val / max(1e-8, 1.0 - r_val * r_val) + float(dim) / max(1.0, float(dim) - 1.0) * r_val
        approx = max(1e-6, approx)
        try:
            upper = max(approx, 1.0)
            while (a_d(upper) - r_val) < 0.0 and upper < 1e6:
                upper *= 2.0
            out[it.multi_index] = brentq(lambda x: a_d(x) - r_val, 1e-8, upper, maxiter=200)
        except Exception:
            out[it.multi_index] = max(1e-6, approx - (float(dim) / max(1.0, float(dim) - 1.0)) * r_val / 2.0)
    return out.astype(np.float32)


def cdln(kappa: np.ndarray, dim: int) -> np.ndarray:
    kap = np.clip(np.asarray(kappa, dtype=np.float64), 1e-8, None)
    order = float(dim) / 2.0 - 1.0
    log_bessel = np.log(np.clip(special.ive(order, kap), 1e-300, None)) + kap
    return (order * np.log(kap) - log_bessel).astype(np.float32)


def ini_concentration(dim: int) -> float:
    if dim < 1200:
        return 500.0
    if dim < 1800:
        return 650.0
    raise ValueError(f"Observation dimension too high for current adaptation: D={dim}")


def load_atlas_reference(shared_root: Path, subject: str, atlas: str) -> AtlasReference:
    ref_dir = shared_root / "_shared" / f"sub-{subject}" / "reference" / atlas
    roi_ts_path = ref_dir / "cortex_roi_parcel_timeseries.npy"
    roi_rows_path = ref_dir / "cortex_roi_parcels.tsv"
    parent_rows_path = ref_dir / "cortex_parent_networks.tsv"
    missing = [path for path in [roi_ts_path, roi_rows_path, parent_rows_path] if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing atlas-native reference inputs:\n" + "\n".join(str(p.resolve()) for p in missing))

    roi_ts = np.load(roi_ts_path).astype(np.float32)
    roi_rows = load_table_rows(roi_rows_path)
    parent_rows = load_table_rows(parent_rows_path)
    if roi_ts.shape[0] != len(roi_rows):
        raise DataConsistencyError(f"ROI parcel table mismatch: {roi_ts.shape[0]} rows vs {len(roi_rows)} TSV entries for {roi_ts_path}")
    atlas_cfg = _MERGE_CONFIG["atlases"].get(str(atlas))
    if atlas_cfg is None:
        raise KeyError(f"Atlas {atlas} missing in merge config: {CROSS_ATLAS_MERGE_JSON}")
    mapping = {str(k): str(v) for k, v in atlas_cfg.get("mapping", {}).items()}
    exclude_labels = {str(x) for x in _MERGE_CONFIG.get("exclude_labels", [])}
    canonical_order = [str(x) for x in _MERGE_CONFIG.get("canonical_network_order", [])]
    mapped_per_roi: list[str] = []
    for row in roi_rows:
        raw_parent = str(row["parent_network"])
        canonical_parent = mapping.get(raw_parent, raw_parent)
        if canonical_parent in exclude_labels:
            raise DataConsistencyError(
                f"Excluded canonical label {canonical_parent} present in ROI table for atlas={atlas}; "
                f"upstream noise exclusion should have removed it."
            )
        mapped_per_roi.append(canonical_parent)
    present = set(mapped_per_roi)
    parent_networks = [name for name in canonical_order if name in present]
    if not parent_networks:
        raise DataConsistencyError(f"No canonical parent networks resolved for atlas={atlas}")
    parent_index = {name: idx for idx, name in enumerate(parent_networks)}
    roi_parent_index = np.asarray([parent_index[name] for name in mapped_per_roi], dtype=np.int32)

    run_bounds: list[tuple[int, int]] = []
    for run_id in RUN_IDS:
        stage_manifest = ref_dir / "runs" / f"run-{run_id}" / "stage_manifest.json"
        if not stage_manifest.exists():
            raise FileNotFoundError(f"Missing run reference manifest: {stage_manifest}")
        payload = load_json(stage_manifest)
        params = payload.get("params", {})
        start = int(params["run_start"])
        stop = int(params["run_stop"])
        run_bounds.append((start, stop))
    if int(run_bounds[-1][1]) != int(roi_ts.shape[1]):
        raise DataConsistencyError(
            f"Run-boundary sum mismatch for sub-{subject} {atlas}: last_stop={run_bounds[-1][1]} vs roi_timepoints={roi_ts.shape[1]}"
        )

    return AtlasReference(
        roi_labels=[str(row["parcel_name"]) for row in roi_rows],
        parent_networks=parent_networks,
        roi_parent_index=roi_parent_index,
        roi_timeseries=roi_ts,
        run_bounds=run_bounds,
    )


def load_runwise_hipp_timeseries(shared_root: Path, subject: str, smoothing: str, hemi: str) -> tuple[np.ndarray, list[np.ndarray]]:
    base = shared_root / "_shared" / f"sub-{subject}" / "surface"
    valid_mask_path = base / "tsnr_gate" / f"sub-{subject}_hemi-{hemi}_valid_mask.npy"
    if not valid_mask_path.exists():
        raise FileNotFoundError(f"Missing hippocampal valid mask: {valid_mask_path}")
    valid_mask = np.load(valid_mask_path).astype(bool)
    runs = []
    for run_id in RUN_IDS:
        path = base / "runs_from_concat" / f"run-{run_id}" / smoothing / f"sub-{subject}_hemi-{hemi}_timeseries.npy"
        if not path.exists():
            raise FileNotFoundError(f"Missing runwise hippocampal timeseries: {path}")
        runs.append(np.load(path).astype(np.float32))
    for run in runs:
        if run.shape[0] != valid_mask.size:
            raise DataConsistencyError(
                f"Vertex-count mismatch for sub-{subject} {smoothing} hemi-{hemi}: run={run.shape[0]} mask={valid_mask.size}"
            )
    return valid_mask, runs


def seed_basis_manifest_path(combo_root: Path) -> Path:
    return combo_root / "step1" / "seed_basis.json"


def ensure_seed_basis(
    *,
    combo_root: Path,
    shared_root: Path,
    atlas: str,
    subjects: list[str],
    resume_mode: str,
) -> dict[str, object]:
    path = seed_basis_manifest_path(combo_root)
    if resume_mode == "resume" and path.exists():
        cached = load_json(path)
        if payload_matches_current(cached):
            return cached

    refs = [load_atlas_reference(shared_root, subject, atlas) for subject in subjects]
    parent_networks = refs[0].parent_networks
    for item in refs[1:]:
        if item.parent_networks != parent_networks:
            raise DataConsistencyError(f"Inconsistent parent-network order while building shared seed basis for atlas={atlas}")
    common = set(refs[0].roi_labels)
    for item in refs[1:]:
        common &= set(item.roi_labels)
    ordered_labels = [label for label in refs[0].roi_labels if label in common]
    if not ordered_labels:
        raise RuntimeError(f"No common ROI labels remain for atlas={atlas} across subjects={subjects}")
    first_index = {label: idx for idx, label in enumerate(refs[0].roi_labels)}
    roi_parent_index = refs[0].roi_parent_index[np.asarray([first_index[label] for label in ordered_labels], dtype=np.int32)]
    counts = np.bincount(roi_parent_index, minlength=len(parent_networks))
    if np.any(counts <= 0):
        missing = [parent_networks[idx] for idx, val in enumerate(counts) if val <= 0]
        raise RuntimeError(f"Shared seed basis lost all ROIs for parent networks: {missing}")
    dropped_by_subject = {}
    for subject, ref in zip(subjects, refs, strict=True):
        dropped = [label for label in ref.roi_labels if label not in common]
        dropped_by_subject[str(subject)] = {
            "roi_count_original": int(len(ref.roi_labels)),
            "roi_count_retained": int(len(ordered_labels)),
            "n_dropped": int(len(dropped)),
            "dropped_roi_labels": dropped,
        }
    payload = {
        **current_manifest_meta(),
        "atlas_slug": atlas,
        "subjects": subjects,
        "roi_labels": ordered_labels,
        "parent_networks": parent_networks,
        "roi_parent_index": roi_parent_index.astype(int).tolist(),
        "observation_dim": int(len(ordered_labels)),
        "roi_count_intersection": int(len(ordered_labels)),
        "dropped_by_subject": dropped_by_subject,
    }
    save_json(path, payload)
    return payload


def compute_binarized_correlation_profile(
    hipp_ts: np.ndarray,
    roi_ts: np.ndarray,
    valid_mask: np.ndarray,
    topk_fraction: float,
) -> np.ndarray:
    n_vertices = int(hipp_ts.shape[0])
    n_roi = int(roi_ts.shape[0])
    out = np.zeros((n_vertices, n_roi), dtype=np.float32)
    active = np.flatnonzero(valid_mask)
    if active.size == 0:
        return out
    z_hipp = zscore_rows(hipp_ts[active, :])
    z_roi = zscore_rows(roi_ts)
    corr = (z_hipp @ z_roi.T) / float(max(1, hipp_ts.shape[1]))
    k = max(1, int(math.ceil(float(n_roi) * float(topk_fraction))))
    idx = np.argpartition(corr, kth=max(0, n_roi - k), axis=1)[:, -k:]
    prof = np.zeros_like(corr, dtype=np.float32)
    prof[np.arange(corr.shape[0])[:, None], idx] = 1.0
    prof = normalize_rows(prof)
    out[active, :] = prof
    return out


def step1_subject_dir(combo_root: Path, subject: str) -> Path:
    return combo_root / "step1" / "profiles" / f"sub-{subject}"


def step1_subject_manifest_path(combo_root: Path, subject: str) -> Path:
    return step1_subject_dir(combo_root, subject) / "profile_manifest.json"


def step1_done(combo_root: Path, subject: str, resume_mode: str) -> bool:
    if resume_mode != "resume":
        return False
    path = step1_subject_manifest_path(combo_root, subject)
    if not path.exists():
        return False
    return payload_matches_current(load_json(path))


def ensure_step1_subject(
    *,
    combo: Combo,
    combo_root: Path,
    shared_root: Path,
    subject: str,
    seed_basis: dict[str, object],
    topk_fraction: float,
    resume_mode: str,
) -> dict[str, object]:
    out_dir = step1_subject_dir(combo_root, subject)
    manifest_path = step1_subject_manifest_path(combo_root, subject)
    if step1_done(combo_root, subject, resume_mode):
        return load_json(manifest_path)

    atlas_ref = load_atlas_reference(shared_root, subject, combo.atlas)
    if atlas_ref.parent_networks != [str(x) for x in seed_basis["parent_networks"]]:
        raise DataConsistencyError(f"Parent-network order mismatch for sub-{subject} atlas={combo.atlas}")
    label_to_idx = {label: idx for idx, label in enumerate(atlas_ref.roi_labels)}
    try:
        selected_idx = np.asarray([label_to_idx[str(label)] for label in seed_basis["roi_labels"]], dtype=np.int32)
    except KeyError as exc:
        raise DataConsistencyError(f"Subject sub-{subject} is missing shared seed ROI label: {exc}") from exc
    valid_mask, hipp_runs = load_runwise_hipp_timeseries(shared_root, subject, combo.smoothing, combo.hemi)

    if len(atlas_ref.run_bounds) != len(hipp_runs):
        raise DataConsistencyError(f"Run-count mismatch for sub-{subject} {combo.atlas} {combo.smoothing} hemi-{combo.hemi}")

    out_dir.mkdir(parents=True, exist_ok=True)
    run_outputs = []
    session_present = []
    for run_idx, run_id in enumerate(RUN_IDS):
        start, stop = atlas_ref.run_bounds[run_idx]
        roi_run = atlas_ref.roi_timeseries[selected_idx, start:stop]
        hipp_run = hipp_runs[run_idx]
        if int(roi_run.shape[1]) != int(hipp_run.shape[1]):
            raise DataConsistencyError(
                f"Runwise timepoint mismatch for sub-{subject} run-{run_id}: roi={roi_run.shape[1]} hipp={hipp_run.shape[1]}"
            )
        profile = compute_binarized_correlation_profile(hipp_run, roi_run, valid_mask, topk_fraction)
        path = out_dir / f"run-{run_id}_roi_profile.npy"
        np.save(path, profile.astype(np.float32))
        run_outputs.append(str(path.resolve()))
        session_present.append(bool(np.any(valid_mask)))

    valid_mask_path = out_dir / f"sub-{subject}_hemi-{combo.hemi}_valid_mask.npy"
    np.save(valid_mask_path, valid_mask.astype(np.uint8))
    payload = {
        **current_manifest_meta(),
        "subject": subject,
        "atlas_slug": combo.atlas,
        "smoothing": combo.smoothing,
        "hemi": combo.hemi,
        "topk_fraction": float(topk_fraction),
        "n_vertices": int(valid_mask.size),
        "n_vertices_valid": int(valid_mask.sum()),
        "observation_dim": int(len(seed_basis["roi_labels"])),
        "n_parent_networks": int(len(seed_basis["parent_networks"])),
        "roi_labels": [str(x) for x in seed_basis["roi_labels"]],
        "parent_networks": [str(x) for x in seed_basis["parent_networks"]],
        "roi_parent_index": [int(x) for x in seed_basis["roi_parent_index"]],
        "run_profile_paths": run_outputs,
        "run_bounds": [[int(a), int(b)] for a, b in atlas_ref.run_bounds],
        "valid_mask": str(valid_mask_path.resolve()),
        "session_present": [bool(x) for x in session_present],
    }
    save_json(manifest_path, payload)
    return payload


def load_step1_subject(combo_root: Path, subject: str) -> SubjectProfiles:
    payload = load_json(step1_subject_manifest_path(combo_root, subject))
    run_paths = [Path(p) for p in payload["run_profile_paths"]]
    profiles = np.asarray([np.load(path).astype(np.float32) for path in run_paths], dtype=np.float32)
    valid_mask = np.load(Path(str(payload["valid_mask"]))).astype(bool)
    return SubjectProfiles(
        subject=str(payload["subject"]),
        atlas=str(payload["atlas_slug"]),
        smoothing=str(payload["smoothing"]),
        hemi=str(payload["hemi"]),
        valid_mask=valid_mask,
        profiles=profiles,
        session_present=np.asarray(payload["session_present"], dtype=bool),
        parent_networks=[str(x) for x in payload["parent_networks"]],
        roi_labels=[str(x) for x in payload["roi_labels"]],
    )


def validate_step1_subjects(subject_profiles: list[SubjectProfiles]) -> tuple[list[str], list[str], int]:
    if not subject_profiles:
        raise ValueError("No subject profiles loaded")
    parent_networks = subject_profiles[0].parent_networks
    roi_labels = subject_profiles[0].roi_labels
    n_vertices = int(subject_profiles[0].valid_mask.size)
    for item in subject_profiles[1:]:
        if item.parent_networks != parent_networks:
            raise DataConsistencyError("Parent-network order differs across subjects")
        if item.roi_labels != roi_labels:
            raise DataConsistencyError("ROI seed-basis order differs across subjects")
        if int(item.valid_mask.size) != n_vertices:
            raise DataConsistencyError("Vertex count differs across subjects")
    return parent_networks, roi_labels, n_vertices


def mean_profile_over_subject_sessions(subject_profiles: list[SubjectProfiles]) -> tuple[np.ndarray, np.ndarray]:
    n_vertices = int(subject_profiles[0].valid_mask.size)
    dim = int(subject_profiles[0].profiles.shape[-1])
    accum = np.zeros((n_vertices, dim), dtype=np.float32)
    counts = np.zeros((n_vertices, 1), dtype=np.float32)
    for item in subject_profiles:
        for run_idx in range(item.profiles.shape[0]):
            if not item.session_present[run_idx]:
                continue
            prof = item.profiles[run_idx]
            active = item.valid_mask
            accum[active, :] += prof[active, :]
            counts[active, 0] += 1.0
    mean = np.zeros_like(accum)
    np.divide(accum, np.clip(counts, 1.0, None), out=mean, where=counts > 0)
    mean = normalize_rows(mean)
    valid_any = counts[:, 0] > 0
    return mean, valid_any.astype(bool)


def build_parent_seed_prototypes(roi_parent_index: np.ndarray, n_labels: int, dim: int) -> np.ndarray:
    ref = np.zeros((n_labels, dim), dtype=np.float32)
    for label_idx in range(n_labels):
        mask = roi_parent_index == label_idx
        if np.any(mask):
            ref[label_idx, mask] = 1.0
    return normalize_rows(ref)


def spherical_kmeans(
    data: np.ndarray,
    n_clusters: int,
    n_init: int,
    max_iters: int,
    random_state: int,
) -> tuple[np.ndarray, np.ndarray, float]:
    if data.shape[0] < n_clusters:
        raise RuntimeError(f"Too few vertices for spherical clustering: {data.shape[0]} < {n_clusters}")
    rng = np.random.default_rng(random_state)
    best_labels = None
    best_centroids = None
    best_obj = -np.inf
    for init_idx in range(max(1, int(n_init))):
        choose = rng.choice(data.shape[0], size=n_clusters, replace=False)
        centroids = data[choose, :].copy()
        centroids = normalize_rows(centroids)
        labels = np.zeros((data.shape[0],), dtype=np.int32)
        for _ in range(max(1, int(max_iters))):
            sim = data @ centroids.T
            new_labels = np.argmax(sim, axis=1).astype(np.int32)
            if np.array_equal(new_labels, labels):
                break
            labels = new_labels
            new_centroids = np.zeros_like(centroids)
            for k in range(n_clusters):
                members = data[labels == k]
                if members.size == 0:
                    new_centroids[k, :] = data[rng.integers(0, data.shape[0]), :]
                else:
                    new_centroids[k, :] = np.nanmean(members, axis=0).astype(np.float32)
            centroids = normalize_rows(new_centroids)
        obj = float(np.sum(np.max(data @ centroids.T, axis=1)))
        if obj > best_obj:
            best_obj = obj
            best_labels = labels.copy()
            best_centroids = centroids.copy()
        rng = np.random.default_rng(random_state + init_idx + 1)
    assert best_labels is not None and best_centroids is not None
    return best_labels, best_centroids, best_obj


def map_clusters_to_parent_networks(centroids: np.ndarray, parent_seed_ref: np.ndarray) -> dict[int, int]:
    cost = -(centroids @ parent_seed_ref.T).astype(np.float64)
    rows, cols = linear_sum_assignment(cost)
    mapping = {int(r): int(c) for r, c in zip(rows, cols, strict=True)}
    unused = [idx for idx in range(parent_seed_ref.shape[0]) if idx not in mapping.values()]
    for row in range(centroids.shape[0]):
        if row not in mapping:
            mapping[row] = int(unused.pop(0)) if unused else int(row)
    return mapping


def save_step1_group_init(
    *,
    combo_root: Path,
    group_mean: np.ndarray,
    valid_any: np.ndarray,
    parent_networks: list[str],
    roi_parent_index: np.ndarray,
    n_init: int,
    max_iters: int,
    random_state: int,
    split: SplitSpec,
) -> dict[str, object]:
    step1_root = combo_root / "step1"
    group_path = step1_root / "group" / "group_init.npz"
    summary_path = step1_root / "group" / "group_init_summary.json"

    active = np.flatnonzero(valid_any)
    data = group_mean[active, :]
    labels_raw, centroids_raw, objective = spherical_kmeans(
        data=data,
        n_clusters=len(parent_networks),
        n_init=n_init,
        max_iters=max_iters,
        random_state=random_state,
    )
    seed_ref = build_parent_seed_prototypes(roi_parent_index, len(parent_networks), group_mean.shape[1])
    mapping = map_clusters_to_parent_networks(centroids_raw, seed_ref)
    centroids = np.zeros_like(centroids_raw)
    for raw_idx, mapped_idx in mapping.items():
        centroids[mapped_idx, :] = centroids_raw[raw_idx, :]
    labels_active = np.asarray([mapping[int(x)] + 1 for x in labels_raw], dtype=np.int32)
    labels_full = np.zeros((group_mean.shape[0],), dtype=np.int32)
    labels_full[active] = labels_active
    lambda_init = np.zeros((group_mean.shape[0], len(parent_networks)), dtype=np.float32)
    lambda_init[active, labels_active - 1] = 1.0

    group_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        group_path,
        clustered_mtc=centroids.astype(np.float32),
        clustered_lambda=lambda_init.astype(np.float32),
        clustered_lowerbound=np.asarray([objective], dtype=np.float32),
        group_init_labels=labels_full.astype(np.int32),
        mean_profile=group_mean.astype(np.float32),
        valid_vertex_any_subject=valid_any.astype(np.uint8),
    )
    summary = {
        **current_manifest_meta(),
        "step": "step1",
        "init_method": "spherical_kmeans_on_group_average_seed_basis_profiles",
        "n_subjects": int(len(split.train)),
        "n_vertices": int(group_mean.shape[0]),
        "n_vertices_any_valid_subject": int(valid_any.sum()),
        "observation_dim": int(group_mean.shape[1]),
        "n_networks": int(len(parent_networks)),
        "atlas_native_networks": parent_networks,
        "training_subjects": split.train,
        "group_init_path": str(group_path.resolve()),
        "objective": float(objective),
        "cluster_to_parent_mapping": {str(k + 1): int(v + 1) for k, v in mapping.items()},
    }
    save_json(summary_path, summary)
    return {
        "group_init_path": str(group_path.resolve()),
        "group_init_summary": str(summary_path.resolve()),
        "n_subjects": int(len(split.train)),
        "n_vertices_any_valid_subject": int(valid_any.sum()),
    }


def run_step1_for_combo(
    *,
    combo: Combo,
    combo_root: Path,
    shared_root: Path,
    split: SplitSpec,
    topk_fraction: float,
    n_init: int,
    init_max_iters: int,
    random_state: int,
    resume_mode: str,
) -> dict[str, object]:
    all_subjects = split.train + split.validation + split.test
    seed_basis = ensure_seed_basis(
        combo_root=combo_root,
        shared_root=shared_root,
        atlas=combo.atlas,
        subjects=all_subjects,
        resume_mode=resume_mode,
    )
    first_manifest = ensure_step1_subject(
        combo=combo,
        combo_root=combo_root,
        shared_root=shared_root,
        subject=split.train[0],
        seed_basis=seed_basis,
        topk_fraction=topk_fraction,
        resume_mode=resume_mode,
    )
    roi_parent_index = np.asarray(first_manifest["roi_parent_index"], dtype=np.int32)
    for subject in split.train[1:] + split.validation + split.test:
        ensure_step1_subject(
            combo=combo,
            combo_root=combo_root,
            shared_root=shared_root,
            subject=subject,
            seed_basis=seed_basis,
            topk_fraction=topk_fraction,
            resume_mode=resume_mode,
        )
    train_profiles = [load_step1_subject(combo_root, subject) for subject in split.train]
    parent_networks, _, _ = validate_step1_subjects(train_profiles)
    group_mean, valid_any = mean_profile_over_subject_sessions(train_profiles)
    step1_summary = save_step1_group_init(
        combo_root=combo_root,
        group_mean=group_mean,
        valid_any=valid_any,
        parent_networks=parent_networks,
        roi_parent_index=roi_parent_index,
        n_init=n_init,
        max_iters=init_max_iters,
        random_state=random_state,
        split=split,
    )
    manifest_path = combo_root / "step1" / "step1_manifest.json"
    payload = {
        **current_manifest_meta(),
        "atlas_slug": combo.atlas,
        "smoothing": combo.smoothing,
        "hemi": combo.hemi,
        "split_manifest": str(split.path.resolve()),
        "train_subjects": split.train,
        "validation_subjects": split.validation,
        "test_subjects": split.test,
        "topk_fraction": float(topk_fraction),
        "seed_basis_manifest": str(seed_basis_manifest_path(combo_root).resolve()),
        "group_init_path": step1_summary["group_init_path"],
        "group_init_summary": step1_summary["group_init_summary"],
        "subject_profile_manifests": [str(step1_subject_manifest_path(combo_root, sub).resolve()) for sub in (split.train + split.validation + split.test)],
    }
    save_json(manifest_path, payload)
    return payload


def load_group_init(combo_root: Path) -> dict[str, np.ndarray]:
    path = Path(str(load_json(combo_root / "step1" / "step1_manifest.json")["group_init_path"]))
    with np.load(path, allow_pickle=False) as npz:
        return {str(k): npz[k] for k in npz.files}


def estimate_subject_lambda(subject_profiles: SubjectProfiles, nu: np.ndarray, kappa: np.ndarray, theta: np.ndarray) -> np.ndarray:
    n_vertices = int(subject_profiles.valid_mask.size)
    n_labels = int(nu.shape[1])
    scores = np.full((n_vertices, n_labels), -1e9, dtype=np.float32)
    log_theta = np.log(np.clip(theta.astype(np.float32), 1e-8, None))
    cdln_term = cdln(kappa, nu.shape[0])[None, :]
    for run_idx in range(subject_profiles.profiles.shape[0]):
        if not subject_profiles.session_present[run_idx]:
            continue
        X = subject_profiles.profiles[run_idx]
        scores[subject_profiles.valid_mask, :] = np.where(
            np.isfinite(scores[subject_profiles.valid_mask, :]),
            scores[subject_profiles.valid_mask, :],
            0.0,
        )
        scores[subject_profiles.valid_mask, :] += (X[subject_profiles.valid_mask, :] @ nu[:, :, run_idx]) * kappa[None, :] + cdln_term
    scores[subject_profiles.valid_mask, :] += log_theta[subject_profiles.valid_mask, :]
    lam = np.zeros((n_vertices, n_labels), dtype=np.float32)
    if np.any(subject_profiles.valid_mask):
        lam[subject_profiles.valid_mask, :] = stable_softmax(scores[subject_profiles.valid_mask, :], axis=1)
    return lam


def compute_group_theta(lambdas: np.ndarray, valid_masks: np.ndarray, theta_floor: float) -> np.ndarray:
    counts = np.sum(valid_masks.astype(np.float32), axis=0, keepdims=True).T
    theta = np.sum(lambdas, axis=0)
    np.divide(theta, np.clip(counts, 1.0, None), out=theta, where=counts > 0)
    theta = np.clip(theta, float(theta_floor), None)
    theta = theta / np.sum(theta, axis=1, keepdims=True)
    theta[counts[:, 0] <= 0, :] = 0.0
    return theta.astype(np.float32)


def expected_v_lambda(adjacency: sparse.csr_matrix, lam: np.ndarray) -> np.ndarray:
    same_prob = adjacency @ lam
    degree = np.asarray(adjacency.sum(axis=1), dtype=np.float32).reshape(-1, 1)
    return np.maximum(0.0, degree - same_prob).astype(np.float32)


def relative_cost_change(new_cost: float, old_cost: float | None) -> float:
    if old_cost is None or not np.isfinite(old_cost):
        return float("inf")
    return float(abs(new_cost - old_cost) / max(abs(old_cost), 1e-8))


def compute_group_em_cost(
    *,
    subject_profiles: list[SubjectProfiles],
    s_t_nu: np.ndarray,
    s_lambda: np.ndarray,
    kappa: np.ndarray,
    theta: np.ndarray,
) -> tuple[np.ndarray, float]:
    log_theta = np.log(np.clip(theta.astype(np.float32), 1e-8, None))
    cdln_term = cdln(kappa, s_t_nu.shape[0])[None, :]
    per_subject = np.zeros((len(subject_profiles),), dtype=np.float32)
    for sidx, item in enumerate(subject_profiles):
        active = item.valid_mask
        if not np.any(active):
            continue
        log_lambda_prop = np.zeros((int(np.sum(active)), s_t_nu.shape[1]), dtype=np.float32)
        for tidx in range(item.profiles.shape[0]):
            if not item.session_present[tidx]:
                continue
            X = item.profiles[tidx][active, :]
            log_lambda_prop += (X @ s_t_nu[:, :, tidx, sidx]) * kappa[None, :] + cdln_term
        lam_active = s_lambda[active, :, sidx]
        log_lam = np.log(np.clip(lam_active, 1e-8, None))
        per_subject[sidx] = float(
            np.sum(lam_active * log_lambda_prop)
            + np.sum(lam_active * log_theta[active, :])
            - np.sum(lam_active * log_lam)
        )
    return per_subject.astype(np.float32), float(np.sum(per_subject))


def compute_intra_inter_cost(
    *,
    mu: np.ndarray,
    epsil: np.ndarray,
    sigma: np.ndarray,
    s_psi: np.ndarray,
    s_t_nu: np.ndarray,
    em_total: float,
) -> tuple[float, float]:
    dim = mu.shape[0]
    sigma_similarity = np.sum(s_psi[:, :, None, :] * s_t_nu, axis=0)
    sigma_term = float(
        np.sum(
            cdln(sigma, dim)[:, None, None]
            + sigma[:, None, None] * sigma_similarity
        )
    )
    eps_similarity = np.sum(mu[:, :, None] * s_psi, axis=0)
    eps_term = float(
        np.sum(
            cdln(epsil, dim)[:, None]
            + epsil[:, None] * eps_similarity
        )
    )
    total = sigma_term + eps_term + float(em_total)
    return total, total


def compute_individual_em_cost(
    *,
    subject_profiles: SubjectProfiles,
    s_t_nu: np.ndarray,
    lam: np.ndarray,
    kappa: np.ndarray,
    theta: np.ndarray,
    w: float,
    c: float,
    adjacency: sparse.csr_matrix,
) -> tuple[float, float, float]:
    active = subject_profiles.valid_mask
    if not np.any(active):
        return 0.0, 0.0, 0.0
    log_lambda_prop = np.zeros((int(np.sum(active)), s_t_nu.shape[1]), dtype=np.float32)
    cdln_term = cdln(kappa, s_t_nu.shape[0])[None, :]
    for tidx in range(subject_profiles.profiles.shape[0]):
        if not subject_profiles.session_present[tidx]:
            continue
        X = subject_profiles.profiles[tidx][active, :]
        log_lambda_prop += (X @ s_t_nu[:, :, tidx]) * kappa[None, :] + cdln_term
    lam_active = lam[active, :]
    log_theta = np.log(np.clip(theta[active, :], 1e-8, None))
    log_lam = np.log(np.clip(lam_active, 1e-8, None))
    v_temp = expected_v_lambda(adjacency[active][:, active], lam_active) if float(c) > 0.0 else np.zeros_like(lam_active)
    total = float(
        np.sum(lam_active * log_lambda_prop)
        + float(w) * np.sum(lam_active * log_theta)
        - np.sum(lam_active * log_lam)
        - float(c) * np.sum(lam_active * v_temp)
    )
    return total, float(np.sum(lam_active * v_temp)), float(np.sum(lam_active * log_lambda_prop))


def update_subject_session_profiles(
    X: np.ndarray,
    lam: np.ndarray,
    subject_proto: np.ndarray,
    sigma: np.ndarray,
    kappa: np.ndarray,
) -> np.ndarray:
    weighted = X.T @ lam
    update = weighted * kappa[None, :] + subject_proto * sigma[None, :]
    return normalize_rows(update.T).T


def update_subject_prototype(session_proto: np.ndarray, mu: np.ndarray, sigma: np.ndarray, epsil: np.ndarray) -> np.ndarray:
    update = np.sum(session_proto * sigma[None, :, None], axis=2) + mu * epsil[None, :]
    return normalize_rows(update.T).T


def update_mu(subject_proto: np.ndarray, epsil: np.ndarray) -> np.ndarray:
    update = np.sum(subject_proto * epsil[None, :, None], axis=2)
    return normalize_rows(update.T).T


def compute_concentration_from_similarity(sim: np.ndarray, dim: int, minimum: float) -> np.ndarray:
    kappa = inv_ad(dim, np.clip(sim.astype(np.float32), 1e-5, 0.999))
    kappa = np.where(np.isfinite(kappa), kappa, minimum)
    kappa = np.maximum(kappa, minimum)
    return kappa.astype(np.float32)


def train_group_priors(
    *,
    subject_profiles: list[SubjectProfiles],
    init_payload: dict[str, np.ndarray],
    theta_floor: float,
    max_iters: int,
    intra_iters: int,
    inter_iters: int,
) -> tuple[dict[str, np.ndarray], dict[str, object]]:
    parent_networks, roi_labels, n_vertices = validate_step1_subjects(subject_profiles)
    n_labels = len(parent_networks)
    dim = len(roi_labels)
    n_subjects = len(subject_profiles)
    n_sessions = len(RUN_IDS)
    minimum = ini_concentration(dim)

    mu = normalize_rows(np.asarray(init_payload["clustered_mtc"], dtype=np.float32)).T
    theta = np.asarray(init_payload["clustered_lambda"], dtype=np.float32)
    theta = np.clip(theta, float(theta_floor), None)
    theta = theta / np.sum(theta, axis=1, keepdims=True)

    epsil = np.full((n_labels,), minimum, dtype=np.float32)
    sigma = np.full((n_labels,), minimum, dtype=np.float32)
    kappa = np.full((n_labels,), minimum, dtype=np.float32)
    s_psi = np.repeat(mu[:, :, None], n_subjects, axis=2).astype(np.float32)
    s_t_nu = np.repeat(mu[:, :, None, None], n_sessions, axis=2)
    s_t_nu = np.repeat(s_t_nu, n_subjects, axis=3).astype(np.float32)
    s_lambda = np.zeros((n_vertices, n_labels, n_subjects), dtype=np.float32)
    valid_masks = np.stack([item.valid_mask for item in subject_profiles], axis=0)

    history: list[dict[str, object]] = []
    prev_cost_inter: float | None = None
    for sidx, item in enumerate(subject_profiles):
        s_lambda[:, :, sidx] = estimate_subject_lambda(item, s_t_nu[:, :, :, sidx], kappa, theta)
    theta = compute_group_theta(s_lambda.transpose(2, 0, 1), valid_masks, theta_floor)

    for iter_idx in range(max(1, int(max_iters))):
        for _ in range(max(1, int(intra_iters))):
            for _ in range(max(1, int(inter_iters))):
                for sidx, item in enumerate(subject_profiles):
                    for tidx in range(n_sessions):
                        if not item.session_present[tidx]:
                            continue
                        X = item.profiles[tidx]
                        s_t_nu[:, :, tidx, sidx] = update_subject_session_profiles(
                            X=X[item.valid_mask, :],
                            lam=s_lambda[item.valid_mask, :, sidx],
                            subject_proto=s_psi[:, :, sidx],
                            sigma=sigma,
                            kappa=kappa,
                        )
                numer = 0.0
                denom = 0.0
                for label_idx in range(n_labels):
                    for sidx, item in enumerate(subject_profiles):
                        for tidx in range(n_sessions):
                            if not item.session_present[tidx]:
                                continue
                            X = item.profiles[tidx][item.valid_mask, :]
                            lam = s_lambda[item.valid_mask, label_idx, sidx]
                            if lam.size == 0 or float(np.sum(lam)) <= 0.0:
                                continue
                            sim = X @ s_t_nu[:, label_idx, tidx, sidx]
                            numer += float(np.sum(lam * sim))
                            denom += float(np.sum(lam))
                kappa_scalar = float(
                    compute_concentration_from_similarity(
                        np.asarray([numer / max(denom, 1e-6)], dtype=np.float32),
                        dim,
                        minimum,
                    )[0]
                )
                kappa = np.full((n_labels,), kappa_scalar, dtype=np.float32)
            for sidx in range(n_subjects):
                s_psi[:, :, sidx] = update_subject_prototype(s_t_nu[:, :, :, sidx], mu, sigma, epsil)
            sigmas = []
            for label_idx in range(n_labels):
                sims = []
                for sidx in range(n_subjects):
                    for tidx in range(n_sessions):
                        if subject_profiles[sidx].session_present[tidx]:
                            sims.append(float(np.dot(s_t_nu[:, label_idx, tidx, sidx], s_psi[:, label_idx, sidx])))
                sigmas.append(float(compute_concentration_from_similarity(np.asarray([np.mean(sims) if sims else 1e-5], dtype=np.float32), dim, minimum)[0]))
            sigma = np.asarray(sigmas, dtype=np.float32)
        mu = update_mu(s_psi, epsil)
        eps = []
        for label_idx in range(n_labels):
            sims = [float(np.dot(s_psi[:, label_idx, sidx], mu[:, label_idx])) for sidx in range(n_subjects)]
            eps.append(float(compute_concentration_from_similarity(np.asarray([np.mean(sims) if sims else 1e-5], dtype=np.float32), dim, minimum)[0]))
        epsil = np.asarray(eps, dtype=np.float32)
        for sidx, item in enumerate(subject_profiles):
            s_lambda[:, :, sidx] = estimate_subject_lambda(item, s_t_nu[:, :, :, sidx], kappa, theta)
        theta = compute_group_theta(s_lambda.transpose(2, 0, 1), valid_masks, theta_floor)

        cost_em_by_subject, cost_em_total = compute_group_em_cost(
            subject_profiles=subject_profiles,
            s_t_nu=s_t_nu,
            s_lambda=s_lambda,
            kappa=kappa,
            theta=theta,
        )
        cost_intra, cost_inter = compute_intra_inter_cost(
            mu=mu,
            epsil=epsil,
            sigma=sigma,
            s_psi=s_psi,
            s_t_nu=s_t_nu,
            em_total=cost_em_total,
        )
        rel_cost_change = relative_cost_change(cost_inter, prev_cost_inter)
        history.append(
            {
                "iteration": int(iter_idx + 1),
                "cost_em_subject_mean": float(np.mean(cost_em_by_subject)),
                "cost_em_total": float(cost_em_total),
                "cost_intra": float(cost_intra),
                "cost_inter": float(cost_inter),
                "relative_cost_change": rel_cost_change,
                "mean_epsil": float(np.mean(epsil)),
                "mean_sigma": float(np.mean(sigma)),
                "mean_kappa": float(np.mean(kappa)),
                "shared_kappa": 1,
            }
        )
        if rel_cost_change <= 1e-4:
            break
        prev_cost_inter = cost_inter

    group_proto = normalize_rows(mu.T @ mu)
    inter_sim = np.zeros((n_labels,), dtype=np.float32)
    intra_sim = np.zeros((n_labels,), dtype=np.float32)
    for label_idx in range(n_labels):
        inter_sim[label_idx] = float(np.mean([np.dot(s_psi[:, label_idx, sidx], mu[:, label_idx]) for sidx in range(n_subjects)]))
        values = []
        for sidx in range(n_subjects):
            for tidx in range(n_sessions):
                if subject_profiles[sidx].session_present[tidx]:
                    values.append(float(np.dot(s_t_nu[:, label_idx, tidx, sidx], s_psi[:, label_idx, sidx])))
        intra_sim[label_idx] = float(np.mean(values)) if values else 0.0

    params = {
        "mu": mu.astype(np.float32),
        "epsil": epsil.astype(np.float32),
        "sigma": sigma.astype(np.float32),
        "theta": theta.astype(np.float32),
        "kappa": kappa.astype(np.float32),
        "s_lambda": s_lambda.astype(np.float32),
        "s_psi": s_psi.astype(np.float32),
        "s_t_nu": s_t_nu.astype(np.float32),
        "group_prototypes": group_proto.astype(np.float32),
        "inter_subject_variability": inter_sim.astype(np.float32),
        "intra_subject_variability": intra_sim.astype(np.float32),
        "theta_wta_labels": (np.argmax(theta, axis=1).astype(np.int32) + 1),
        "valid_vertex_any_subject": np.asarray(init_payload["valid_vertex_any_subject"], dtype=np.uint8),
        "group_init_labels": np.asarray(init_payload["group_init_labels"], dtype=np.int32),
    }
    summary = {
        **current_manifest_meta(),
        "step": "step2",
        "n_subjects": int(n_subjects),
        "n_vertices": int(n_vertices),
        "n_vertices_any_valid_subject": int(np.sum(np.asarray(init_payload["valid_vertex_any_subject"]) > 0)),
        "observation_dim": int(dim),
        "n_networks": int(n_labels),
        "atlas_native_networks": parent_networks,
        "roi_labels": roi_labels,
        "params_fields": ["mu", "epsil", "sigma", "theta", "kappa", "s_lambda", "s_psi", "s_t_nu"],
        "history": history,
        "convergence_metric": "relative_cost_change",
        "mu_shape": [int(x) for x in mu.shape],
        "theta_shape": [int(x) for x in theta.shape],
        "s_lambda_shape": [int(x) for x in s_lambda.shape],
        "s_psi_shape": [int(x) for x in s_psi.shape],
        "s_t_nu_shape": [int(x) for x in s_t_nu.shape],
    }
    return params, summary


def surface_vertex_count(surface_path: Path) -> int:
    img = nib.load(str(surface_path))
    for da in img.darrays:
        if da.intent == "NIFTI_INTENT_POINTSET":
            return int(np.asarray(da.data).shape[0])
    return int(np.asarray(img.darrays[0].data).shape[0])


def find_template_surface(*, subject: str, hemi: str, n_vertices: int) -> Path:
    surf_dir = DEFAULT_DENSE_COROBL_ROOT / f"sub-{subject}" / "hippunfold" / f"sub-{subject}" / "surf"
    if not surf_dir.exists():
        raise FileNotFoundError(f"Missing hippocampal surf dir: {surf_dir}")
    candidates = sorted(surf_dir.glob(f"sub-{subject}_hemi-{hemi}_space-corobl_den-*_label-hipp_midthickness.surf.gii"))
    matching = [path for path in candidates if surface_vertex_count(path) == n_vertices]
    if len(matching) != 1:
        found = ", ".join(f"{path.name}:{surface_vertex_count(path)}" for path in candidates)
        raise RuntimeError(
            f"Expected exactly one surface with {n_vertices} vertices for subject={subject} hemi={hemi}; candidates={found}"
        )
    return matching[0]


def resolve_template_assets(*, subject: str, n_vertices: int) -> dict[str, Path]:
    left_surface = find_template_surface(subject=subject, hemi="L", n_vertices=n_vertices)
    right_surface = find_template_surface(subject=subject, hemi="R", n_vertices=n_vertices)
    return {
        "left_surface": left_surface,
        "right_surface": right_surface,
        "spec_path": infer_spec_path(subject, left_surface),
    }


def load_surface(path: Path) -> tuple[np.ndarray, np.ndarray]:
    img = nib.load(str(path))
    coords = np.asarray(img.agg_data("pointset"), dtype=np.float32)
    faces = np.asarray(img.agg_data("triangle"), dtype=np.int32)
    return coords, faces


def build_surface_adjacency(faces: np.ndarray, n_vertices: int) -> sparse.csr_matrix:
    row_parts = []
    col_parts = []
    for tri in faces:
        a, b, c = (int(tri[0]), int(tri[1]), int(tri[2]))
        row_parts.extend([a, b, b, c, c, a])
        col_parts.extend([b, a, c, b, a, c])
    data = np.ones(len(row_parts), dtype=np.float32)
    graph = sparse.csr_matrix((data, (row_parts, col_parts)), shape=(n_vertices, n_vertices), dtype=np.float32)
    graph = graph.maximum(graph.T)
    graph.setdiag(0)
    graph.eliminate_zeros()
    return graph


def save_group_label_assets(
    *,
    combo_root: Path,
    combo: Combo,
    template_subject: str,
    theta_wta: np.ndarray,
    networks: list[str],
) -> tuple[Path, Path, dict[str, str]]:
    template_assets = resolve_template_assets(subject=template_subject, n_vertices=int(theta_wta.size))
    surface_path = template_assets["left_surface"] if combo.hemi == "L" else template_assets["right_surface"]
    density = infer_density_from_surface(surface_path)
    order = long_axis_order(surface_path)
    order_path = combo_root / "group_priors" / f"group_theta_hemi-{combo.hemi}_longaxis_order.npy"
    order_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(order_path, order.astype(np.int32))

    label_dir = combo_root / "template" / "workbench_assets"
    label_dir.mkdir(parents=True, exist_ok=True)
    label_path = label_dir / (
        f"sub-{template_subject}_hemi-{combo.hemi}_space-corobl_den-{density}_label-"
        f"group_prior_network-mshbm_{combo.atlas}_{combo.smoothing}.label.gii"
    )
    nib.save(
        make_network_label_gifti(
            labels=theta_wta.astype(np.int32),
            network_names=networks,
            network_colors=NETWORK_COLORS,
        ),
        str(label_path),
    )
    return order_path, label_path, {
        "left_surface": str(template_assets["left_surface"].resolve()),
        "right_surface": str(template_assets["right_surface"].resolve()),
        "spec_path": str(template_assets["spec_path"].resolve()),
        "hemi_label": str(label_path.resolve()),
        "density": str(density),
    }


def run_step2_for_combo(
    *,
    combo: Combo,
    combo_root: Path,
    split: SplitSpec,
    theta_floor: float,
    step2_max_iters: int,
    intra_iters: int,
    inter_iters: int,
    resume_mode: str,
) -> GroupArtifacts:
    priors_root = combo_root / "priors"
    params_path = priors_root / "Params_Final.npz"
    compat_npz_path = combo_root / "group_priors" / f"group_priors_hemi-{combo.hemi}.npz"
    summary_path = combo_root / "group_init_summary.json"
    manifest_path = combo_root / "group_prior_manifest.json"
    theta_wta_path = combo_root / "group_priors" / f"group_theta_wta_hemi-{combo.hemi}.npy"
    if resume_mode == "resume" and params_path.exists() and compat_npz_path.exists() and manifest_path.exists():
        cached_manifest = load_json(manifest_path)
        if payload_matches_current(cached_manifest):
            return GroupArtifacts(
                params_path=params_path,
                compat_npz_path=compat_npz_path,
                manifest_path=manifest_path,
                summary_path=summary_path,
                theta_wta_path=theta_wta_path,
            )

    train_profiles = [load_step1_subject(combo_root, subject) for subject in split.train]
    init_payload = load_group_init(combo_root)
    params, summary = train_group_priors(
        subject_profiles=train_profiles,
        init_payload=init_payload,
        theta_floor=theta_floor,
        max_iters=step2_max_iters,
        intra_iters=intra_iters,
        inter_iters=inter_iters,
    )
    parent_networks = train_profiles[0].parent_networks
    template_subject = split.test[0] if split.test else split.train[0]
    theta_wta = np.asarray(params["theta_wta_labels"], dtype=np.int32)
    theta_wta_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(theta_wta_path, theta_wta)
    order_path, _, template_assets_payload = save_group_label_assets(
        combo_root=combo_root,
        combo=combo,
        template_subject=template_subject,
        theta_wta=theta_wta,
        networks=parent_networks,
    )

    params_payload = {
        "artifact_version": np.asarray([ARTIFACT_VERSION], dtype=object),
        "Params.mu": params["mu"],
        "Params.epsil": params["epsil"],
        "Params.sigma": params["sigma"],
        "Params.theta": params["theta"],
        "Params.kappa": params["kappa"],
        "Params.s_lambda": params["s_lambda"],
        "Params.s_psi": params["s_psi"],
        "Params.s_t_nu": params["s_t_nu"],
        "atlas_native_networks": np.asarray(parent_networks, dtype=object),
        "observation_roi_labels": np.asarray(train_profiles[0].roi_labels, dtype=object),
        "train_subjects": np.asarray(split.train, dtype=object),
        "runs_used": np.asarray([f"run-{x}" for x in RUN_IDS], dtype=object),
    }
    compat_payload = {
        "model_family": np.asarray([MODEL_FAMILY], dtype=object),
        "model_version": np.asarray([MODEL_VERSION], dtype=object),
        "artifact_version": np.asarray([ARTIFACT_VERSION], dtype=object),
        "atlas_slug": np.asarray([combo.atlas], dtype=object),
        "smoothing": np.asarray([combo.smoothing], dtype=object),
        "hemi": np.asarray([combo.hemi], dtype=object),
        "canonical_networks": np.asarray(parent_networks, dtype=object),
        "atlas_native_networks": np.asarray(parent_networks, dtype=object),
        "group_prototypes": params["group_prototypes"],
        "theta": params["theta"].T,
        "theta_wta_labels": theta_wta,
        "inter_subject_variability": params["inter_subject_variability"],
        "intra_subject_variability": params["intra_subject_variability"],
        "group_init_labels": params["group_init_labels"],
        "valid_vertex_any_subject": params["valid_vertex_any_subject"],
    }
    priors_root.mkdir(parents=True, exist_ok=True)
    compat_npz_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(params_path, **params_payload)
    np.savez_compressed(compat_npz_path, **compat_payload)

    summary.update(
        {
            "atlas_slug": combo.atlas,
            "smoothing": combo.smoothing,
            "hemi": combo.hemi,
            "train_subjects": split.train,
            "validation_subjects": split.validation,
            "test_subjects": split.test,
            "group_prior_npz": str(compat_npz_path.resolve()),
            "params_final": str(params_path.resolve()),
        }
    )
    save_json(summary_path, summary)
    manifest_payload = {
        **current_manifest_meta(),
        "atlas_slug": combo.atlas,
        "smoothing": combo.smoothing,
        "hemi": combo.hemi,
        "split_manifest": str(split.path.resolve()),
        "train_subjects": split.train,
        "validation_subjects": split.validation,
        "test_subjects": split.test,
        "group_prior_npz": str(compat_npz_path.resolve()),
        "params_final": str(params_path.resolve()),
        "group_theta_wta": str(theta_wta_path.resolve()),
        "longaxis_order": str(order_path.resolve()),
        "group_init_summary": str(summary_path.resolve()),
        "canonical_networks": parent_networks,
        "atlas_native_networks": parent_networks,
        "template_subject": template_subject,
        "template_workbench_assets": template_assets_payload,
        "official_reference": str((REPO_ROOT / "external" / "CBIG" / "stable_projects" / "brain_parcellation" / "Kong2019_MSHBM").resolve()),
    }
    save_json(manifest_path, manifest_payload)
    return GroupArtifacts(
        params_path=params_path,
        compat_npz_path=compat_npz_path,
        manifest_path=manifest_path,
        summary_path=summary_path,
        theta_wta_path=theta_wta_path,
    )


def load_params_final(path: Path) -> dict[str, np.ndarray | list[str]]:
    with np.load(path, allow_pickle=True) as npz:
        payload: dict[str, np.ndarray | list[str]] = {}
        for key in npz.files:
            arr = npz[key]
            if arr.dtype == object:
                payload[str(key)] = [str(x) for x in arr.tolist()]
            else:
                payload[str(key)] = arr
        return payload


def compute_wcss(features: np.ndarray, labels: np.ndarray) -> float:
    wcss = 0.0
    for label in np.unique(labels):
        members = features[labels == label]
        if members.size == 0:
            continue
        centroid = np.mean(members, axis=0, keepdims=True)
        wcss += float(np.sum((members - centroid) ** 2))
    return wcss


def compute_homogeneity(features: np.ndarray, labels: np.ndarray) -> float:
    tss = float(np.sum((features - np.mean(features, axis=0, keepdims=True)) ** 2))
    if tss <= 1e-12:
        return 0.0
    return float(max(0.0, 1.0 - (compute_wcss(features, labels) / tss)))


def infer_single_subject(
    *,
    subject_profiles: SubjectProfiles,
    params: dict[str, np.ndarray | list[str]],
    w: float,
    c: float,
    adjacency: sparse.csr_matrix,
    infer_iters: int,
) -> dict[str, np.ndarray | float | int]:
    mu = np.asarray(params["Params.mu"], dtype=np.float32)
    epsil = np.asarray(params["Params.epsil"], dtype=np.float32)
    sigma = np.asarray(params["Params.sigma"], dtype=np.float32)
    theta = np.asarray(params["Params.theta"], dtype=np.float32)
    n_sessions = subject_profiles.profiles.shape[0]
    n_labels = mu.shape[1]
    minimum = ini_concentration(mu.shape[0])

    s_psi = mu.copy()
    s_t_nu = np.repeat(mu[:, :, None], n_sessions, axis=2).astype(np.float32)
    kappa = np.full((n_labels,), minimum, dtype=np.float32)
    lam = estimate_subject_lambda(subject_profiles, s_t_nu, kappa, theta)
    active = subject_profiles.valid_mask
    active_adj = adjacency[active][:, active]
    conf = np.zeros((active.size,), dtype=np.float32)
    infer_history: list[dict[str, float]] = []
    prev_cost: float | None = None

    for iter_idx in range(max(1, int(infer_iters))):
        for tidx in range(n_sessions):
            if not subject_profiles.session_present[tidx]:
                continue
            X = subject_profiles.profiles[tidx][active, :]
            s_t_nu[:, :, tidx] = update_subject_session_profiles(
                X=X,
                lam=lam[active, :],
                subject_proto=s_psi,
                sigma=sigma,
                kappa=kappa,
            )
        s_psi = update_subject_prototype(s_t_nu, mu, sigma, epsil)
        numer = 0.0
        denom = 0.0
        for label_idx in range(n_labels):
            for tidx in range(n_sessions):
                if not subject_profiles.session_present[tidx]:
                    continue
                X = subject_profiles.profiles[tidx][active, :]
                ll = lam[active, label_idx]
                if float(np.sum(ll)) <= 0.0:
                    continue
                numer += float(np.sum(ll * (X @ s_t_nu[:, label_idx, tidx])))
                denom += float(np.sum(ll))
        kappa_scalar = float(
            compute_concentration_from_similarity(
                np.asarray([numer / max(denom, 1e-6)], dtype=np.float32),
                mu.shape[0],
                minimum,
            )[0]
        )
        kappa = np.full((n_labels,), kappa_scalar, dtype=np.float32)

        scores = np.zeros((int(active.sum()), n_labels), dtype=np.float32)
        cdln_term = cdln(kappa, mu.shape[0])[None, :]
        for tidx in range(n_sessions):
            if not subject_profiles.session_present[tidx]:
                continue
            X = subject_profiles.profiles[tidx][active, :]
            scores += (X @ s_t_nu[:, :, tidx]) * kappa[None, :] + cdln_term
        scores += float(w) * np.log(np.clip(theta[active, :], 1e-8, None))
        if float(c) > 0.0:
            scores += -2.0 * float(c) * expected_v_lambda(active_adj, lam[active, :])
        lam_active = stable_softmax(scores, axis=1)
        lam[active, :] = lam_active
        top2 = np.partition(lam_active, kth=max(0, n_labels - 2), axis=1)[:, -2:]
        conf[active] = (top2[:, -1] - top2[:, 0]).astype(np.float32)
        total_cost, pairwise_vdiff, vmf_term = compute_individual_em_cost(
            subject_profiles=subject_profiles,
            s_t_nu=s_t_nu,
            lam=lam,
            kappa=kappa,
            theta=theta,
            w=w,
            c=c,
            adjacency=adjacency,
        )
        rel_cost_change = relative_cost_change(total_cost, prev_cost)
        infer_history.append(
            {
                "iteration": int(iter_idx + 1),
                "cost_total": float(total_cost),
                "relative_cost_change": float(rel_cost_change),
                "pairwise_vdiff_mass": float(pairwise_vdiff),
                "vmf_term": float(vmf_term),
            }
        )
        if rel_cost_change <= 1e-4:
            break
        prev_cost = total_cost

    labels = np.zeros((active.size,), dtype=np.int32)
    labels[active] = np.argmax(lam[active, :], axis=1).astype(np.int32) + 1
    return {
        "labels": labels,
        "posterior": lam.astype(np.float32),
        "confidence": conf.astype(np.float32),
        "subject_prototype": s_psi.astype(np.float32),
        "session_prototypes": s_t_nu.astype(np.float32),
        "kappa": kappa.astype(np.float32),
        "infer_history": infer_history,
    }


def save_subject_outputs(
    *,
    combo_root: Path,
    subject: str,
    hemi: str,
    result: dict[str, np.ndarray | float | int],
    group_prior_path: Path,
    networks: list[str],
    subject_set: str,
    w: float,
    c: float,
) -> dict[str, object]:
    root = combo_root / "individual_posteriors" / subject_set / f"sub-{subject}"
    root.mkdir(parents=True, exist_ok=True)
    posterior_path = root / f"sub-{subject}_hemi-{hemi}_posterior.npy"
    labels_path = root / f"sub-{subject}_hemi-{hemi}_labels.npy"
    confidence_path = root / f"sub-{subject}_hemi-{hemi}_confidence.npy"
    session_proto_path = root / f"sub-{subject}_hemi-{hemi}_session_prototypes.npy"
    subject_proto_path = root / f"sub-{subject}_hemi-{hemi}_subject_prototype.npy"
    np.save(posterior_path, np.asarray(result["posterior"], dtype=np.float32))
    np.save(labels_path, np.asarray(result["labels"], dtype=np.int32))
    np.save(confidence_path, np.asarray(result["confidence"], dtype=np.float32))
    np.save(session_proto_path, np.asarray(result["session_prototypes"], dtype=np.float32))
    np.save(subject_proto_path, np.asarray(result["subject_prototype"], dtype=np.float32))
    summary_path = root / f"sub-{subject}_hemi-{hemi}_summary.json"
    save_json(
        summary_path,
        {
            **current_manifest_meta(),
            "subject": subject,
            "subject_set": subject_set,
            "hemi": hemi,
            "atlas_native_networks": networks,
            "group_prior": str(group_prior_path.resolve()),
            "posterior": str(posterior_path.resolve()),
            "labels": str(labels_path.resolve()),
            "confidence": str(confidence_path.resolve()),
            "session_prototypes": str(session_proto_path.resolve()),
            "subject_prototype": str(subject_proto_path.resolve()),
            "w": float(w),
            "c": float(c),
            "n_vertices_labeled": int(np.sum(np.asarray(result["labels"], dtype=np.int32) > 0)),
            "convergence_metric": "relative_cost_change",
            "infer_history": result.get("infer_history", []),
        },
    )
    return {
        "subject": subject,
        "hemi": hemi,
        "posterior": str(posterior_path.resolve()),
        "labels": str(labels_path.resolve()),
        "confidence": str(confidence_path.resolve()),
        "session_prototypes": str(session_proto_path.resolve()),
        "subject_prototype": str(subject_proto_path.resolve()),
        "summary": str(summary_path.resolve()),
    }


def tune_wc_for_combo(
    *,
    combo: Combo,
    combo_root: Path,
    split: SplitSpec,
    infer_iters: int,
    w_grid: list[float],
    c_grid: list[float],
    target_subjects: list[str] | None,
    resume_mode: str,
) -> TuneArtifacts:
    tune_root = combo_root / "tune"
    best_path = tune_root / "best_wc.json"
    summary_path = tune_root / "tuning_summary.json"
    if resume_mode == "resume" and best_path.exists() and summary_path.exists():
        best = load_json(best_path)
        if payload_matches_current(best):
            return TuneArtifacts(best_w=float(best["best_w"]), best_c=float(best["best_c"]), best_path=best_path, summary_path=summary_path)

    params = load_params_final(combo_root / "priors" / "Params_Final.npz")
    subjects = get_subjects_for_step(split, "validation", target_subjects)
    n_vertices = int(np.asarray(params["Params.theta"]).shape[0])
    surface = find_template_surface(subject=subjects[0], hemi=combo.hemi, n_vertices=n_vertices)
    _, faces = load_surface(surface)
    adjacency = build_surface_adjacency(faces, n_vertices)

    rows = []
    best_score = -np.inf
    best_w = None
    best_c = None
    for w in w_grid:
        for c in c_grid:
            scores = []
            for subject in subjects:
                data = load_step1_subject(combo_root, subject)
                result = infer_single_subject(
                    subject_profiles=data,
                    params=params,
                    w=float(w),
                    c=float(c),
                    adjacency=adjacency,
                    infer_iters=infer_iters,
                )
                mean_profile = normalize_rows(np.nanmean(data.profiles, axis=0).astype(np.float32))
                active = data.valid_mask & (np.asarray(result["labels"], dtype=np.int32) > 0)
                if int(np.sum(active)) > 1:
                    score = compute_homogeneity(mean_profile[active, :], np.asarray(result["labels"], dtype=np.int32)[active])
                    scores.append(score)
            row = {
                "w": float(w),
                "c": float(c),
                "n_subjects": int(len(scores)),
                "homogeneity_mean": float(np.mean(scores)) if scores else float("nan"),
                "homogeneity_values": [float(x) for x in scores],
            }
            rows.append(row)
            if scores and float(np.mean(scores)) > best_score:
                best_score = float(np.mean(scores))
                best_w = float(w)
                best_c = float(c)
    if best_w is None or best_c is None:
        raise RuntimeError(f"No validation tuning result available for {combo.atlas} {combo.smoothing} hemi-{combo.hemi}")
    tune_root.mkdir(parents=True, exist_ok=True)
    save_json(
        best_path,
        {
            **current_manifest_meta(),
            "best_w": best_w,
            "best_c": best_c,
            "best_homogeneity": best_score,
            "validation_subjects": subjects,
        },
    )
    save_json(
        summary_path,
        {
            **current_manifest_meta(),
            "atlas_slug": combo.atlas,
            "smoothing": combo.smoothing,
            "hemi": combo.hemi,
            "validation_subjects": subjects,
            "rows": rows,
            "best_w": best_w,
            "best_c": best_c,
            "best_homogeneity": best_score,
        },
    )
    return TuneArtifacts(best_w=best_w, best_c=best_c, best_path=best_path, summary_path=summary_path)


def run_step3_for_combo(
    *,
    combo: Combo,
    combo_root: Path,
    split: SplitSpec,
    subject_set: str,
    infer_iters: int,
    w_grid: list[float],
    c_grid: list[float],
    target_subjects: list[str] | None,
    resume_mode: str,
) -> dict[str, object]:
    tune = tune_wc_for_combo(
        combo=combo,
        combo_root=combo_root,
        split=split,
        infer_iters=infer_iters,
        w_grid=w_grid,
        c_grid=c_grid,
        target_subjects=None,
        resume_mode=resume_mode,
    )
    params = load_params_final(combo_root / "priors" / "Params_Final.npz")
    networks = [str(x) for x in params["atlas_native_networks"]]
    subjects = get_subjects_for_step(split, subject_set, target_subjects)
    n_vertices = int(np.asarray(params["Params.theta"]).shape[0])
    surface = find_template_surface(subject=subjects[0], hemi=combo.hemi, n_vertices=n_vertices)
    _, faces = load_surface(surface)
    adjacency = build_surface_adjacency(faces, n_vertices)

    outputs = []
    for subject in subjects:
        result_dir = combo_root / "individual_posteriors" / subject_set / f"sub-{subject}"
        summary_file = result_dir / f"sub-{subject}_hemi-{combo.hemi}_summary.json"
        if resume_mode == "resume" and summary_file.exists():
            cached_summary = load_json(summary_file)
            if payload_matches_current(cached_summary):
                outputs.append(cached_summary)
                continue
        data = load_step1_subject(combo_root, subject)
        result = infer_single_subject(
            subject_profiles=data,
            params=params,
            w=tune.best_w,
            c=tune.best_c,
            adjacency=adjacency,
            infer_iters=infer_iters,
        )
        outputs.append(
            save_subject_outputs(
                combo_root=combo_root,
                subject=subject,
                hemi=combo.hemi,
                result=result,
                group_prior_path=combo_root / "priors" / "Params_Final.npz",
                networks=networks,
                subject_set=subject_set,
                w=tune.best_w,
                c=tune.best_c,
            )
        )
    manifest_path = combo_root / f"individual_manifest_{subject_set}.json"
    save_json(
        manifest_path,
        {
            **current_manifest_meta(),
            "atlas_slug": combo.atlas,
            "smoothing": combo.smoothing,
            "hemi": combo.hemi,
            "subject_set": subject_set,
            "best_w": tune.best_w,
            "best_c": tune.best_c,
            "subjects": outputs,
        },
    )
    if subject_set == "test":
        save_json(
            combo_root / "individual_manifest.json",
            {
                **current_manifest_meta(),
                "atlas_slug": combo.atlas,
                "smoothing": combo.smoothing,
                "hemi": combo.hemi,
                "subject_set": subject_set,
                "best_w": tune.best_w,
                "best_c": tune.best_c,
                "subjects": outputs,
            },
        )
    return {"individual_manifest": str(manifest_path.resolve()), "best_w": tune.best_w, "best_c": tune.best_c}


def process_combo(args: argparse.Namespace, combo: Combo, split: SplitSpec, shared_root: Path, model_root: Path) -> dict[str, object]:
    combo_root = model_root / combo.atlas / combo.smoothing / f"hemi_{combo.hemi}"
    combo_root.mkdir(parents=True, exist_ok=True)
    results: dict[str, object] = {
        "atlas": combo.atlas,
        "smoothing": combo.smoothing,
        "hemi": combo.hemi,
        "split_manifest": str(split.path.resolve()),
    }

    if args.stage in {"step1", "all", "step2", "tune", "step3"}:
        step1_manifest = combo_root / "step1" / "step1_manifest.json"
        step1_manifest_valid = step1_manifest.exists() and payload_matches_current(load_json(step1_manifest))
        if args.stage == "step1" or not step1_manifest_valid or args.resume_mode == "force":
            results["step1"] = run_step1_for_combo(
                combo=combo,
                combo_root=combo_root,
                shared_root=shared_root,
                split=split,
                topk_fraction=args.topk_fraction,
                n_init=args.num_initialization,
                init_max_iters=args.init_max_iters,
                random_state=args.random_state,
                resume_mode=args.resume_mode,
            )
        else:
            results["step1"] = load_json(step1_manifest)
    if args.stage in {"step2", "all", "tune", "step3"}:
        group = run_step2_for_combo(
            combo=combo,
            combo_root=combo_root,
            split=split,
            theta_floor=args.theta_floor,
            step2_max_iters=args.step2_max_iters,
            intra_iters=args.intra_iters,
            inter_iters=args.inter_iters,
            resume_mode=args.resume_mode,
        )
        results["step2"] = {"params_final": str(group.params_path.resolve()), "group_prior_manifest": str(group.manifest_path.resolve())}
    if args.stage in {"tune", "all", "step3"}:
        tune = tune_wc_for_combo(
            combo=combo,
            combo_root=combo_root,
            split=split,
            infer_iters=args.infer_iters,
            w_grid=[float(x) for x in args.w_grid],
            c_grid=[float(x) for x in args.c_grid],
            target_subjects=args.target_subjects,
            resume_mode=args.resume_mode,
        )
        results["tune"] = {"best_w": tune.best_w, "best_c": tune.best_c, "summary": str(tune.summary_path.resolve())}
    if args.stage in {"step3", "all"}:
        results["step3"] = run_step3_for_combo(
            combo=combo,
            combo_root=combo_root,
            split=split,
            subject_set=args.subject_set,
            infer_iters=args.infer_iters,
            w_grid=[float(x) for x in args.w_grid],
            c_grid=[float(x) for x in args.c_grid],
            target_subjects=args.target_subjects,
            resume_mode=args.resume_mode,
        )
    return results


def main() -> int:
    args = parse_args()
    shared_root = Path(args.shared_root).resolve()
    model_root = Path(args.model_root).resolve()
    split = resolve_split(args, model_root)

    results = []
    for atlas in args.atlases:
        for smoothing in args.smoothings:
            for hemi in args.hemis:
                combo = Combo(atlas=str(atlas), smoothing=str(smoothing), hemi=str(hemi))
                results.append(process_combo(args, combo, split, shared_root, model_root))
    print(json.dumps({"results": results}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
