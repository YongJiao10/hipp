from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


DEFAULT_HCP500_ROOT = Path("/home/yojiao/hpchome/HCPpipeline500")
DEFAULT_STUDY_ROOT = DEFAULT_HCP500_ROOT / "work" / "qunex_iowa7t_hcppipeline500"
DEFAULT_SESSIONS_ROOT = DEFAULT_STUDY_ROOT / "sessions"
SMOKE_SUBJECT = "04172026"
FSLR_32K_VERTEX_COUNT = 32492


@dataclass(frozen=True)
class SessionPaths:
    subject: str
    subject_root: Path
    t1w: Path
    t2w: Path
    rest_dtseries: Path
    rest_volume: Path


@dataclass(frozen=True)
class SessionStatus:
    subject: str
    paths: SessionPaths
    eligible: bool
    missing: tuple[str, ...]


@dataclass(frozen=True)
class DiscoveryProducts:
    status: Path
    smoke: Path
    full: Path


@dataclass(frozen=True)
class HippunfoldBridge:
    subject: str
    root: Path
    t1w: Path
    t2w: Path


@dataclass(frozen=True)
class RouteALabelValidation:
    valid: bool
    left_count: int
    right_count: int
    errors: tuple[str, ...]


def session_paths(sessions_root: Path, subject: str) -> SessionPaths:
    subject_id = str(subject).removeprefix("sub-")
    subject_root = Path(sessions_root) / subject_id / "hcp" / subject_id
    rest_root = subject_root / "MNINonLinear" / "Results" / "REST"
    return SessionPaths(
        subject=subject_id,
        subject_root=subject_root,
        t1w=subject_root / "T1w" / "T1w_acpc_dc_restore.nii.gz",
        t2w=subject_root / "T1w" / "T2w_acpc_dc_restore.nii.gz",
        rest_dtseries=rest_root / "REST_Atlas_MSMAll_hp0_clean_rclean.dtseries.nii",
        rest_volume=rest_root / "REST_hp0_clean_rclean.nii.gz",
    )


def _is_nonempty_file(path: Path) -> bool:
    return path.is_file() and path.stat().st_size > 0


def discover_sessions(sessions_root: Path) -> list[SessionStatus]:
    root = Path(sessions_root)
    if not root.is_dir():
        raise FileNotFoundError(f"Iowa HCP500 sessions root not found: {root}")
    rows: list[SessionStatus] = []
    for entry in sorted(root.iterdir(), key=lambda path: path.name):
        if not entry.is_dir():
            continue
        paths = session_paths(root, entry.name)
        checks = (
            ("t1w", paths.t1w),
            ("t2w", paths.t2w),
            ("rest_dtseries", paths.rest_dtseries),
            ("rest_volume", paths.rest_volume),
        )
        missing = tuple(name for name, path in checks if not _is_nonempty_file(path))
        rows.append(
            SessionStatus(
                subject=paths.subject,
                paths=paths,
                eligible=not missing,
                missing=missing,
            )
        )
    return rows


def write_discovery_products(
    *,
    sessions_root: Path,
    manifest_dir: Path,
) -> DiscoveryProducts:
    rows = discover_sessions(sessions_root)
    out = Path(manifest_dir)
    out.mkdir(parents=True, exist_ok=True)
    status_path = out / "iowa_hcp500_session_status.tsv"
    smoke_path = out / f"iowa_hcp500_smoke_{SMOKE_SUBJECT}.txt"
    full_path = out / "iowa_hcp500_eligible_sessions.txt"

    header = (
        "subject\teligible\tmissing\tt1w\tt2w\trest_dtseries\trest_volume\n"
    )
    body = "".join(
        "\t".join(
            (
                row.subject,
                "1" if row.eligible else "0",
                ",".join(row.missing),
                str(row.paths.t1w),
                str(row.paths.t2w),
                str(row.paths.rest_dtseries),
                str(row.paths.rest_volume),
            )
        )
        + "\n"
        for row in rows
    )
    status_path.write_text(header + body, encoding="utf-8")

    eligible = [row.subject for row in rows if row.eligible]
    if SMOKE_SUBJECT not in eligible:
        raise RuntimeError(
            f"Required smoke subject {SMOKE_SUBJECT} is not eligible; "
            f"see {status_path}"
        )
    smoke_path.write_text(f"{SMOKE_SUBJECT}\n", encoding="utf-8")
    full_path.write_text("".join(f"{subject}\n" for subject in eligible), encoding="utf-8")
    return DiscoveryProducts(status=status_path, smoke=smoke_path, full=full_path)


def _ensure_symlink(destination: Path, source: Path) -> None:
    source_resolved = source.resolve(strict=True)
    if destination.is_symlink():
        if destination.resolve(strict=True) == source_resolved:
            return
        raise FileExistsError(
            f"Refusing to replace symlink {destination}; it points to "
            f"{destination.resolve(strict=False)}, expected {source_resolved}"
        )
    if destination.exists():
        raise FileExistsError(f"Refusing to replace existing path: {destination}")
    destination.symlink_to(source_resolved)


def materialize_hippunfold_input(
    status: SessionStatus,
    bids_root: Path,
) -> HippunfoldBridge:
    if not status.eligible:
        raise ValueError(
            f"Subject {status.subject} is not eligible: {', '.join(status.missing)}"
        )
    root = Path(bids_root) / f"sub-{status.subject}"
    anat = root / "anat"
    anat.mkdir(parents=True, exist_ok=True)
    t1w = anat / f"sub-{status.subject}_T1w.nii.gz"
    t2w = anat / f"sub-{status.subject}_T2w.nii.gz"
    _ensure_symlink(t1w, status.paths.t1w)
    _ensure_symlink(t2w, status.paths.t2w)
    return HippunfoldBridge(
        subject=status.subject,
        root=root,
        t1w=t1w,
        t2w=t2w,
    )


def _load_label_vector(path: Path, prefix: str, errors: list[str]) -> np.ndarray:
    if not path.is_file():
        errors.append(f"missing_{prefix}")
        return np.asarray([], dtype=np.float64)
    try:
        values = np.loadtxt(path, dtype=np.float64, ndmin=1)
    except Exception:
        errors.append(f"{prefix}_unreadable")
        return np.asarray([], dtype=np.float64)
    values = np.ravel(values)
    if values.size != FSLR_32K_VERTEX_COUNT:
        errors.append(f"{prefix}_wrong_count")
    if not np.all(np.isfinite(values)):
        errors.append(f"{prefix}_non_finite")
    if not np.allclose(values, np.rint(values)):
        errors.append(f"{prefix}_non_integer")
    return values


def validate_routea_labels(subject_dir: Path) -> RouteALabelValidation:
    root = Path(subject_dir)
    errors: list[str] = []
    left = _load_label_vector(root / "lh_labels.txt", "lh", errors)
    right = _load_label_vector(root / "rh_labels.txt", "rh", errors)
    if left.size and not np.all((left >= 0) & (left <= 200)):
        errors.append("lh_out_of_range")
    if right.size and not np.all((right == 0) | ((right >= 201) & (right <= 400))):
        errors.append("rh_out_of_range")
    return RouteALabelValidation(
        valid=not errors,
        left_count=int(left.size),
        right_count=int(right.size),
        errors=tuple(errors),
    )
