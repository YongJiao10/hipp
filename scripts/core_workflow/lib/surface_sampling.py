#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import tempfile
from collections.abc import Sequence
from pathlib import Path

import nibabel as nib
import numpy as np

from lib.hipp_density_assets import find_surface_asset_strict, subject_surf_dir
from lib.task_quality import task_signal_validity
from lib.surface_coordinates import (METRIC_SPACE, REPO_ROOT, SurfaceRegistration,
                                     registered_sampling_surfaces)
from lib.workbench import run_workbench


NIFTI_COVERAGE_POLICY = "bold_nifti_runwise_trilinear_validity"


def _trilinear_grid_support(shape: Sequence[int], affine: np.ndarray, points: np.ndarray) -> np.ndarray:
    voxel = nib.affines.apply_affine(np.linalg.inv(affine), points)
    if not np.isfinite(voxel).all():
        raise ValueError("Registered sampling vertices contain nonfinite coordinates")
    lower = np.floor(voxel).astype(np.int64)
    fraction = voxel - lower
    weights = np.zeros(len(points), dtype=np.float64)
    for x in (0, 1):
        for y in (0, 1):
            for z in (0, 1):
                offset = np.asarray([x, y, z])
                corner = lower + offset
                inside = np.all((corner >= 0) & (corner < np.asarray(shape)), axis=1)
                contribution = np.prod(np.where(offset, fraction, 1 - fraction), axis=1)
                weights[inside] += contribution[inside]
    return weights.astype(np.float32)


def sample_hippocampal_nifti_support(
    *, bold_inputs: Sequence[Path], run_ids: Sequence[int],
    run_lengths: Sequence[int], sampling_surfaces: dict[str, Path],
) -> tuple[dict[str, np.ndarray], dict[str, dict[str, object]]]:
    """Measure trilinear image-grid support from the exact consumed BOLD NIfTIs.

    This is geometric support, not a tissue label. Actual sampled signals are
    assessed separately within each physical run by hippocampal_vertex_validity.
    Only image headers and the registered vertices are read here.
    """
    paths = [Path(path) for path in bold_inputs]
    ids, lengths = list(run_ids), list(run_lengths)
    if (not paths or not ids or len(ids) != len(lengths)
            or ids != sorted(set(ids))
            or any(not isinstance(n, (int, np.integer)) or isinstance(n, bool) or n < 2 for n in lengths)
            or len(paths) not in (1, len(ids))):
        raise ValueError(f"NIfTI inputs require explicit physical run IDs and lengths: ids={ids}, lengths={lengths}, paths={paths}")
    points = {hemi: np.asarray(nib.load(sampling_surfaces[hemi]).agg_data("pointset"), dtype=np.float64)
              for hemi in ("L", "R")}
    weights = {hemi: np.ones(len(points[hemi]), dtype=np.float32) for hemi in ("L", "R")}
    per_input: dict[str, list[int]] = {hemi: [] for hemi in ("L", "R")}
    for index, path in enumerate(paths):
        image = nib.load(path)
        if not isinstance(image, (nib.Nifti1Image, nib.Nifti2Image)) or len(image.shape) != 4:
            raise ValueError(f"Expected a 4D BOLD NIfTI: {path}")
        blocks = lengths if len(paths) == 1 else [lengths[index]]
        if sum(blocks) != image.shape[3]:
            raise ValueError(f"Physical run timepoints do not cover NIfTI frames at {path}: lengths={blocks}, frames={image.shape[3]}")
        for hemi in ("L", "R"):
            support = _trilinear_grid_support(image.shape[:3], image.affine, points[hemi])
            weights[hemi] = np.minimum(weights[hemi], support)
            per_input[hemi].append(int(np.count_nonzero(support > 0)))
    summaries: dict[str, dict[str, object]] = {}
    for hemi in ("L", "R"):
        summaries[hemi] = {
            "n_vertices_with_grid_support_per_input": per_input[hemi],
            "n_vertices_without_grid_support": int(np.count_nonzero(weights[hemi] == 0)),
            "n_vertices_with_partial_grid_support": int(np.count_nonzero((weights[hemi] > 0) & (weights[hemi] < 1))),
        }
    return weights, summaries


def hippocampal_vertex_validity(data: np.ndarray, run_lengths: Sequence[int], support: np.ndarray) -> np.ndarray:
    """Use the same NIfTI support and per-run sampled-signal rule in all consumers."""
    values = np.asarray(data)
    weight = np.asarray(support)
    if (values.ndim != 2 or weight.shape != (values.shape[0],)
            or not np.isfinite(weight).all() or np.any((weight < 0) | (weight > 1))):
        raise ValueError(f"Invalid NIfTI coverage support: support={weight.shape}, signal={values.shape}")
    return (weight > 0) & task_signal_validity(values, run_lengths)


def load_nifti_support(raw_dir: Path, subject: str, hemi: str, coverage: dict) -> tuple[np.ndarray, Path]:
    path = raw_dir / f"sub-{subject}_hemi-{hemi}_nifti_support.npy"
    declared = coverage['hemispheres'][hemi].get('nifti_support_path')
    if not isinstance(declared, str) or Path(declared).resolve() != path.resolve():
        raise ValueError(f"Coverage NIfTI support declaration mismatch for sub-{subject} hemi-{hemi}: {declared}; expected {path}")
    return np.load(path), path


def load_hippocampal_coverage_validity(raw_dir: Path, subject: str, hemi: str, coverage: dict) -> tuple[np.ndarray, list[Path]]:
    support, support_path = load_nifti_support(raw_dir, subject, hemi, coverage)
    signal_path = raw_dir / f"sub-{subject}_hemi-{hemi}_space-T1w_den-512_label-hipp_bold.func.gii"
    data = _load_metric_gifti(signal_path,
        n_vertices=coverage['hemispheres'][hemi]['n_vertices_total'],
        expected_timepoints=sum(coverage['run_lengths']))
    return hippocampal_vertex_validity(data, coverage['run_lengths'], support), [signal_path, support_path]


def _load_metric_gifti(path: Path, *, n_vertices: int, expected_timepoints: int) -> np.ndarray:
    metric = np.asarray(nib.load(str(path)).agg_data(), dtype=np.float32)
    if metric.ndim == 1:
        metric = metric[:, None]
    if metric.ndim != 2:
        raise ValueError(f"Expected 2D metric GIFTI: {path} shape={metric.shape}")
    if metric.shape[0] != n_vertices and metric.shape[1] == n_vertices:
        metric = metric.T
    expected_shape = (int(n_vertices), int(expected_timepoints))
    if metric.shape != expected_shape:
        raise ValueError(f"Metric GIFTI shape mismatch: {path} shape={metric.shape} expected={expected_shape}")
    return metric.astype(np.float32, copy=False)


def sample_volume_to_surface(
    *, bold: Path, surface: Path, out_metric: Path,
    mapping_method: str = "trilinear", wb_command: Path | None = None,
) -> np.ndarray:
    """Sample one registered physical mesh and load its vertex-by-time metric."""
    out_metric.parent.mkdir(parents=True, exist_ok=True)
    run_workbench([
        "-volume-to-surface-mapping", str(bold), str(surface), str(out_metric),
        f"-{mapping_method}",
    ], wb_command=wb_command)
    n_vertices = int(np.asarray(nib.load(surface).agg_data('pointset')).shape[0])
    return _load_metric_gifti(out_metric, n_vertices=n_vertices,
                             expected_timepoints=int(nib.load(bold).shape[3]))


def _prepare_surfaces(*, hippunfold_dir, subject, density, outdir, space, registration):
    if space != METRIC_SPACE:
        raise ValueError('Volume sampling requires native T1w surfaces and an explicit registration; corobl/auto are not physical BOLD spaces')
    if registration is None:
        raise ValueError('MNI volume sampling requires an explicit SurfaceRegistration')
    surf_dir = subject_surf_dir(hippunfold_dir, subject)
    native = {hemi: find_surface_asset_strict(surf_dir=surf_dir, subject=subject, hemi=hemi,
              space=METRIC_SPACE, density=density, suffix='midthickness.surf.gii') for hemi in ('L', 'R')}
    mapped = registered_sampling_surfaces(native, registration=registration, outdir=outdir/'sampling_surfaces')
    return native, mapped


def sample_hipp_surface_timeseries(
    *,
    bold: Path,
    hippunfold_dir: Path,
    subject: str,
    density: str,
    outdir: Path,
    space: str = "T1w",
    registration: SurfaceRegistration | None = None,
    mapping_method: str = "trilinear",
    smooth_iters: int = 0,
) -> list[dict[str, object]]:
    native, mapped = _prepare_surfaces(hippunfold_dir=hippunfold_dir, subject=subject,
        density=density, outdir=outdir, space=space, registration=registration)
    return _sample_prepared_surfaces(bold=bold, native=native, mapped=mapped, subject=subject,
        density=density, outdir=outdir, registration=registration,
        mapping_method=mapping_method, smooth_iters=smooth_iters)


def _sample_prepared_surfaces(*, bold, native, mapped, subject, density, outdir,
                              registration, mapping_method, smooth_iters):
    declared = re.search(r'_space-([^_]+)_', Path(bold).name)
    if declared and declared.group(1) != registration.target_space:
        raise ValueError(f'BOLD/registration space mismatch: {bold}; transform targets {registration.target_space}')
    outdir.mkdir(parents=True, exist_ok=True)

    summary: list[dict[str, object]] = []
    for hemi in ["L", "R"]:
        mid = mapped[hemi]
        out_metric = outdir / f"sub-{subject}_hemi-{hemi}_space-{METRIC_SPACE}_den-{density}_label-hipp_bold.func.gii"
        metric = sample_volume_to_surface(
            bold=bold, surface=mid, out_metric=out_metric, mapping_method=mapping_method,
        )

        summary.append(
            {
                "hemi": hemi,
                "surface": str(mid),
                "metric_gii": str(out_metric),
                "n_vertices": int(metric.shape[0]),
                "n_timepoints": int(metric.shape[1]),
                "space": METRIC_SPACE,
                "source_surface": str(native[hemi]),
                "sampling_space": registration.target_space,
                "registration_method": registration.method,
                "image_transform": str(registration.image_transform),
                "mapping_method": mapping_method,
                "smooth_iters": int(smooth_iters),
            }
        )

    return summary


def persistent_sampled_metric_summary(sampled: Sequence[dict[str, object]]) -> list[dict[str, object]]:
    """Remove paths to per-run metrics that only exist inside a temporary directory."""
    return [{key: value for key, value in record.items() if key != "metric_gii"} for record in sampled]


def sample_hipp_surface_run_timeseries(
    *,
    runs: Sequence[tuple[int, Path]],
    hippunfold_dir: Path,
    subject: str,
    density: str,
    outdir: Path,
    space: str = "T1w",
    registration: SurfaceRegistration | None = None,
    mapping_method: str = "trilinear",
    smooth_iters: int = 0,
) -> tuple[dict[str, np.ndarray], dict[str, object]]:
    """Sample run-level volumes and concatenate only the arrays in memory.

    Workbench receives one source volume at a time.  Its small temporary
    per-run GIFTIs are removed when the call returns; no concatenated volume,
    CIFTI, or run-level output is persisted.
    """
    outdir.mkdir(parents=True, exist_ok=True)
    normalized_runs = [(int(run_id), Path(path).resolve()) for run_id, path in runs]
    if not normalized_runs:
        raise ValueError("At least one run volume is required")
    run_ids = [run_id for run_id, _ in normalized_runs]
    if len(set(run_ids)) != len(run_ids):
        raise ValueError(f"Duplicate run IDs in surface-sampling inputs: {run_ids}")
    if run_ids != sorted(run_ids):
        raise ValueError(f"Run volumes must be ordered by numeric run ID: {run_ids}")

    native, mapped = _prepare_surfaces(hippunfold_dir=hippunfold_dir, subject=subject,
        density=density, outdir=outdir, space=space, registration=registration)
    surface_vertices: dict[str, int] = {}
    for hemi in ("L", "R"):
        mid = native[hemi]
        surface_vertices[hemi] = int(np.asarray(nib.load(str(mid)).agg_data("pointset")).shape[0])

    by_hemi: dict[str, list[np.ndarray]] = {"L": [], "R": []}
    run_lengths: list[int] = []
    run_records: list[dict[str, object]] = []
    scratch = REPO_ROOT / 'tmp'
    scratch.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=f"sub-{subject}_run_surface_", dir=str(scratch)) as tmp:
        tmp_root = Path(tmp)
        for run_id, bold in normalized_runs:
            if not bold.is_file():
                raise FileNotFoundError(f"Missing run volume for sub-{subject} run-{run_id}: {bold}")
            bold_img = nib.load(str(bold))
            if len(bold_img.shape) != 4:
                raise ValueError(f"Expected 4D run volume for sub-{subject} run-{run_id}: {bold_img.shape}")
            n_timepoints = int(bold_img.shape[3])
            run_tmp = tmp_root / f"run-{run_id}"
            run_tmp.mkdir(parents=True, exist_ok=True)
            sampled = _sample_prepared_surfaces(
                bold=bold,
                native=native,
                mapped=mapped,
                subject=subject,
                density=density,
                outdir=run_tmp,
                registration=registration,
                mapping_method=mapping_method,
                smooth_iters=smooth_iters,
            )
            for hemi in ("L", "R"):
                metric_path = run_tmp / (
                    f"sub-{subject}_hemi-{hemi}_space-{METRIC_SPACE}_den-{density}_"
                    "label-hipp_bold.func.gii"
                )
                by_hemi[hemi].append(
                    _load_metric_gifti(
                        metric_path,
                        n_vertices=surface_vertices[hemi],
                        expected_timepoints=n_timepoints,
                    )
                )
            run_lengths.append(n_timepoints)
            run_records.append(
                {
                    "run_id": run_id,
                    "run_label": f"run-{run_id}",
                    "bold": str(bold),
                    "n_timepoints": n_timepoints,
                    "sampled_metrics": persistent_sampled_metric_summary(sampled),
                }
            )

    concatenated = {
        hemi: np.concatenate(blocks, axis=1).astype(np.float32, copy=False)
        for hemi, blocks in by_hemi.items()
    }
    total_timepoints = int(sum(run_lengths))
    if any(int(data.shape[1]) != total_timepoints for data in concatenated.values()):
        raise RuntimeError("Run-level surface arrays do not share the same concatenated time axis")
    return concatenated, {
        "subject": str(subject).replace("sub-", ""),
        "input_mode": "in_memory_run_volume_concat",
        "space": METRIC_SPACE,
        "sampling_space": registration.target_space,
        "sampling_surfaces": persistent_sampled_metric_summary(sampled),
        "density": str(density),
        "mapping_method": mapping_method,
        "smooth_iters": int(smooth_iters),
        "run_ids": run_ids,
        "run_lengths": run_lengths,
        "n_timepoints": total_timepoints,
        "runs": run_records,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sample 4D BOLD timeseries onto hippocampal surfaces with wb_command")
    parser.add_argument("--bold", required=True)
    parser.add_argument("--hippunfold-dir", required=True)
    parser.add_argument("--subject", required=True)
    parser.add_argument("--density", required=True)
    parser.add_argument("--space", choices=['T1w'], default='T1w')
    parser.add_argument('--dataset', required=True, choices=['hcp', 'stanford', 'banda', 'pdc', 'iowa_jj_tms', 'iowa7t'])
    parser.add_argument("--mapping-method", choices=["enclosing", "trilinear"], default="trilinear")
    parser.add_argument("--smooth-iters", type=int, default=0)
    parser.add_argument("--outdir", required=True)
    return parser.parse_args()


def main() -> int:
    from lib.surface_coordinates import dataset_registration
    args = parse_args()
    summary = sample_hipp_surface_timeseries(
        bold=Path(args.bold).resolve(),
        hippunfold_dir=Path(args.hippunfold_dir).resolve(),
        subject=str(args.subject),
        density=str(args.density),
        outdir=Path(args.outdir).resolve(),
        space=str(args.space),
        registration=dataset_registration(args.dataset, args.subject),
        mapping_method=str(args.mapping_method),
        smooth_iters=int(args.smooth_iters),
    )
    outdir = Path(args.outdir).resolve()
    (outdir / "surface_sampling_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
