#!/opt/miniconda3/envs/py314/bin/python
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

import nibabel as nib
import numpy as np
from scipy.ndimage import map_coordinates


MIN_INSIDE_RATIO = 0.95
MIN_NONZERO_RATIO = 0.50
MIN_MEAN_ABS = 1e-6
MIN_INSIDE_IMPROVEMENT = 0.10
MIN_NONZERO_IMPROVEMENT = 0.25


@dataclass(frozen=True)
class QCMetrics:
    inside_ratio: float
    nonzero_ratio: float
    mean_abs: float
    variance: float


def run_real_wb_command(wb_app: str, wb_args: list[str]) -> int:
    cmd = [wb_app, *wb_args]
    if sys.platform == "darwin" and os.uname().machine == "arm64":
        cmd = ["arch", "-x86_64", *cmd]
    proc = subprocess.run(cmd)
    return proc.returncode


def load_surface_points(path: Path) -> np.ndarray:
    return np.asarray(
        nib.load(str(path)).agg_data("NIFTI_INTENT_POINTSET"), dtype=np.float64
    )


def load_volume_proxy(volume_path: Path) -> tuple[nib.Nifti1Image, np.ndarray] | None:
    volume_img = nib.load(str(volume_path))
    data = volume_img.dataobj
    shape = tuple(int(x) for x in volume_img.shape)
    if len(shape) == 3:
        proxy = np.asarray(data, dtype=np.float32)
    elif len(shape) == 4:
        n_frames = min(int(shape[3]), 10)
        proxy = np.zeros(shape[:3], dtype=np.float32)
        for idx in range(n_frames):
            proxy += np.abs(np.asarray(data[..., idx], dtype=np.float32))
        proxy /= float(n_frames)
    else:
        # Workbench can legitimately map non-scalar volumes here, such as
        # vector warp fields with shape (X, Y, Z, 1, 3). Our scalar QC
        # heuristic is not meaningful for those cases, so we should not
        # block the real command.
        print(
            f"[wb_command guard] Skipping scalar QC for unsupported volume dimensionality: {shape}",
            file=sys.stderr,
        )
        return None
    return volume_img, proxy


def sample_proxy_at_surface(
    volume_img: nib.Nifti1Image,
    proxy_volume: np.ndarray,
    surface_points_world: np.ndarray,
    order: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    vox = nib.affines.apply_affine(np.linalg.inv(volume_img.affine), surface_points_world)
    coords = np.vstack([vox[:, 0], vox[:, 1], vox[:, 2]])
    values = map_coordinates(proxy_volume, coords, order=order, mode="constant", cval=0.0)
    inside = (
        (vox[:, 0] >= 0.0)
        & (vox[:, 0] <= volume_img.shape[0] - 1)
        & (vox[:, 1] >= 0.0)
        & (vox[:, 1] <= volume_img.shape[1] - 1)
        & (vox[:, 2] >= 0.0)
        & (vox[:, 2] <= volume_img.shape[2] - 1)
    )
    return values.astype(np.float32), vox, inside


def compute_qc_metrics(values: np.ndarray, inside: np.ndarray) -> QCMetrics:
    finite = np.isfinite(values)
    safe_values = np.where(finite, values, 0.0)
    abs_values = np.abs(safe_values)
    return QCMetrics(
        inside_ratio=float(np.mean(inside)),
        nonzero_ratio=float(np.mean(abs_values > MIN_MEAN_ABS)),
        mean_abs=float(np.mean(abs_values)),
        variance=float(np.var(safe_values)),
    )


def qc_passes(metrics: QCMetrics) -> bool:
    return (
        metrics.inside_ratio >= MIN_INSIDE_RATIO
        and metrics.nonzero_ratio >= MIN_NONZERO_RATIO
        and metrics.mean_abs > MIN_MEAN_ABS
    )


def metrics_to_str(metrics: QCMetrics) -> str:
    return (
        f"inside={metrics.inside_ratio:.3f}, "
        f"nonzero={metrics.nonzero_ratio:.3f}, "
        f"mean_abs={metrics.mean_abs:.6g}, "
        f"var={metrics.variance:.6g}"
    )


def mapping_order(wb_args: list[str]) -> int:
    return 1 if "-trilinear" in wb_args else 0


def negative_x_shift(surface_points: np.ndarray, volume_img: nib.Nifti1Image) -> np.ndarray:
    shifted = surface_points.copy()
    shifted[:, 0] -= volume_img.shape[0] * abs(float(volume_img.affine[0, 0]))
    return shifted


def shifted_metrics_meaningfully_better(direct: QCMetrics, shifted: QCMetrics) -> bool:
    return (
        shifted.inside_ratio
        >= max(MIN_INSIDE_RATIO, direct.inside_ratio + MIN_INSIDE_IMPROVEMENT)
        and shifted.nonzero_ratio
        >= max(MIN_NONZERO_RATIO, direct.nonzero_ratio + MIN_NONZERO_IMPROVEMENT)
        and shifted.mean_abs > direct.mean_abs
    )


def write_shifted_surface(surface_path: Path, shifted_points: np.ndarray, out_path: Path) -> None:
    surface_img = nib.load(str(surface_path))
    pointset = np.asarray(shifted_points, dtype=np.float32)
    surface_img.darrays[0].data = pointset
    nib.save(surface_img, str(out_path))


def prepare_surface_for_mapping(
    volume_path: Path, surface_path: Path, wb_args: list[str]
) -> tuple[Path | None, str]:
    loaded = load_volume_proxy(volume_path)
    if loaded is None:
        return None, "direct-unsupported-qc"

    volume_img, proxy_volume = loaded
    surface_points = load_surface_points(surface_path)
    order = mapping_order(wb_args)

    direct_values, _, direct_inside = sample_proxy_at_surface(
        volume_img, proxy_volume, surface_points, order
    )
    direct_metrics = compute_qc_metrics(direct_values, direct_inside)
    print(f"[wb_command guard] direct QC: {metrics_to_str(direct_metrics)}", file=sys.stderr)
    if qc_passes(direct_metrics):
        return None, "direct"

    if float(volume_img.affine[0, 0]) < 0.0:
        shifted_points = negative_x_shift(surface_points, volume_img)
        shifted_values, _, shifted_inside = sample_proxy_at_surface(
            volume_img, proxy_volume, shifted_points, order
        )
        shifted_metrics = compute_qc_metrics(shifted_values, shifted_inside)
        print(f"[wb_command guard] shifted QC: {metrics_to_str(shifted_metrics)}", file=sys.stderr)
        if qc_passes(shifted_metrics) and shifted_metrics_meaningfully_better(
            direct_metrics, shifted_metrics
        ):
            tmpdir = tempfile.mkdtemp(prefix="wb_v2s_guard_")
            shifted_surface = Path(tmpdir) / surface_path.name
            write_shifted_surface(surface_path, shifted_points, shifted_surface)
            return shifted_surface, "negative_x_corrected"

    raise RuntimeError(
        "volume-to-surface QC failed before Workbench mapping; "
        f"direct metrics were {metrics_to_str(direct_metrics)} and no supported automatic correction applied"
    )


def parse_args() -> tuple[str, list[str]]:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--wb-app", required=True)
    ns, remaining = parser.parse_known_args()
    return ns.wb_app, remaining


def main() -> int:
    wb_app, wb_args = parse_args()
    if not wb_args:
        return run_real_wb_command(wb_app, wb_args)

    if wb_args[0] != "-volume-to-surface-mapping" or len(wb_args) < 4:
        return run_real_wb_command(wb_app, wb_args)

    volume_path = Path(wb_args[1])
    surface_path = Path(wb_args[2])

    try:
        prepared_surface, strategy = prepare_surface_for_mapping(
            volume_path, surface_path, wb_args
        )
    except Exception as exc:
        print(f"[wb_command guard] {exc}", file=sys.stderr)
        return 2

    guarded_args = list(wb_args)
    if prepared_surface is not None:
        guarded_args[2] = str(prepared_surface)
        print(
            f"[wb_command guard] Using pre-corrected negative-x surface for mapping: {prepared_surface}",
            file=sys.stderr,
        )
    else:
        print(f"[wb_command guard] Using original surface mapping path: {strategy}", file=sys.stderr)

    return run_real_wb_command(wb_app, guarded_args)


if __name__ == "__main__":
    raise SystemExit(main())
