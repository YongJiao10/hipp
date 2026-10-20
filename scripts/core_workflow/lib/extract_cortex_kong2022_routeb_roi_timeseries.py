#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

try:
    from .extract_cortex_merged_network_timeseries import (
        aggregate_hemi_roi_timeseries,
        extract_structure_data,
        load_required_mask,
    )
except ImportError:  # pragma: no cover - supports direct script execution during local debugging.
    from extract_cortex_merged_network_timeseries import (
        aggregate_hemi_roi_timeseries,
        extract_structure_data,
        load_required_mask,
    )
from .io_utils import (
    compute_cortex_tsnr_gate,
    load_cifti_dtseries_inputs,
)
from .kong2022_label_utils import (
    KONG2022_DLABEL_PATH,
    KONG2022_MERGE_CONFIG,
    canonical_families_17_order,
    load_kong2022_bundle,
)
from .kong2022_reference import (
    ReferenceRequest,
    materialize_reference_timeseries,
    resolve_reference_cifti_inputs,
    validate_cifti_run_lengths,
    write_reference_manifests,
)
from .output_layout import DEFAULT_KONG2022_ROUTEB_ROOT
from .task_quality import parse_tsnr_threshold


REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_DATA_ROOT = REPO_ROOT / "data" / "hippunfold_input"
DEFAULT_SHARED_ROOT = DEFAULT_KONG2022_ROUTEB_ROOT
DEFAULT_RUN_IDS = [1, 2, 3, 4]
ATLAS_SLUG = "kong2022"
COMMON_REFERENCE_SLUG = "common"
TSNR_THRESHOLD = 25.0
HCP_MODE = 10000.0


def ensure_vertices_aligned(
    *,
    dt_vertices: np.ndarray,
    dlabel_vertices: np.ndarray,
    dlabel_parcel_ids: np.ndarray,
    hemi: str,
) -> np.ndarray:
    dt_v = np.asarray(dt_vertices, dtype=np.int32).reshape(-1)
    dl_v = np.asarray(dlabel_vertices, dtype=np.int32).reshape(-1)
    dl_pid = np.asarray(dlabel_parcel_ids, dtype=np.int32).reshape(-1)
    if dl_v.shape[0] != dl_pid.shape[0]:
        raise ValueError(f"dlabel vertex/parcel length mismatch hemi-{hemi}: {dl_v.shape} vs {dl_pid.shape}")
    if dt_v.shape[0] == dl_v.shape[0] and np.array_equal(dt_v, dl_v):
        return dl_pid.astype(np.int32, copy=False)
    mapping = {int(v): int(pid) for v, pid in zip(dl_v, dl_pid, strict=True)}
    aligned = np.asarray([mapping.get(int(v), -1) for v in dt_v], dtype=np.int32)
    if np.any(aligned < 0):
        missing = int(np.sum(aligned < 0))
        raise RuntimeError(
            f"Failed to align dlabel parcel ids to dtseries vertices hemi-{hemi}: missing={missing}"
        )
    return aligned



def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Extract cortex Kong2022 ROI timeseries from public dlabel and derive the default canonical-7 network timeseries."
    )
    parser.add_argument("--subject", required=True, help="Subject ID without sub- prefix.")
    parser.add_argument("--dtseries", default=None)
    parser.add_argument(
        "--run-dtseries",
        nargs="+",
        default=None,
        help="Run-level CIFTI dtseries files; concatenate in memory without writing a concat file.",
    )
    parser.add_argument("--run-ids", nargs="+", type=int, default=None)
    parser.add_argument("--run-artifact-manifest", default=None)
    parser.add_argument("--run-metadata-template", default=None)
    parser.add_argument("--kong2022-dlabel", default=str(KONG2022_DLABEL_PATH))
    parser.add_argument("--cross-atlas-merge-json", default=str(KONG2022_MERGE_CONFIG))
    parser.add_argument("--shared-root", default=str(DEFAULT_SHARED_ROOT))
    parser.add_argument("--left-tsnr-mask", default=None)
    parser.add_argument("--right-tsnr-mask", default=None)
    parser.add_argument("--run-lengths", nargs="*", type=int, default=None)
    parser.add_argument("--outdir", default=None)
    return parser


def build_kong2022_reference(*, request: ReferenceRequest) -> int:
    """Build the public-dlabel HCP rest reference without CLI parsing."""
    args = request
    if (args.dataset != "hcp" or args.task != "rest"
            or parse_tsnr_threshold(args.tsnr_threshold) != 25.0
            or args.tsnr_input_template is not None or args.atlas_slug != "kong2022"):
        raise ValueError("Route B references require HCP rest with tSNR25 and public Kong2022 labels")

    subject = str(args.subject).replace("sub-", "")
    shared_root = Path(args.shared_root).resolve()
    resolved = resolve_reference_cifti_inputs(
        args, subject=subject, shared_root=shared_root, required_artifacts=("dtseries",),
    )
    outdir = (
        Path(args.outdir).resolve()
        if args.outdir
        else (shared_root / f"sub-{subject}" / "reference" / ATLAS_SLUG).resolve()
    )
    common_ref_dir = (shared_root / f"sub-{subject}" / "reference" / COMMON_REFERENCE_SLUG).resolve()

    bundle = load_kong2022_bundle(
        dlabel_path=Path(args.kong2022_dlabel).resolve(),
        config_path=Path(args.cross_atlas_merge_json).resolve(),
    )
    roi_registry = list(bundle["roi_registry"])
    canonical7_order = list(bundle["canonical7_order"])
    family17_order = canonical_families_17_order()
    dlabel_vertices = bundle["vertex_arrays"]

    dt_data, dt_axis, source_run_lengths = load_cifti_dtseries_inputs(resolved.dtseries_inputs)
    dt_data_t = dt_data.T
    n_timepoints = int(dt_data.shape[0])
    source_run_lengths = validate_cifti_run_lengths(
        dtseries_input_mode=resolved.dtseries_input_mode,
        declared_run_lengths=resolved.declared_run_lengths,
        source_run_lengths=source_run_lengths,
        n_timepoints=n_timepoints,
    )
    outdir.mkdir(parents=True, exist_ok=True)
    common_ref_dir.mkdir(parents=True, exist_ok=True)
    left_vertices, left_dt = extract_structure_data(dt_axis, dt_data_t, "CORTEX_LEFT")
    right_vertices, right_dt = extract_structure_data(dt_axis, dt_data_t, "CORTEX_RIGHT")
    tsnr_dir = common_ref_dir / "tsnr_gate"
    left_tsnr_mask = (
        Path(args.left_tsnr_mask).resolve()
        if args.left_tsnr_mask
        else compute_cortex_tsnr_gate(data=left_dt, outdir=tsnr_dir, hemi="left", threshold=TSNR_THRESHOLD).resolve()
    )
    right_tsnr_mask = (
        Path(args.right_tsnr_mask).resolve()
        if args.right_tsnr_mask
        else compute_cortex_tsnr_gate(data=right_dt, outdir=tsnr_dir, hemi="right", threshold=TSNR_THRESHOLD).resolve()
    )
    left_valid = load_required_mask(str(left_tsnr_mask), int(left_vertices.shape[0]), "left cortex tSNR")
    right_valid = load_required_mask(str(right_tsnr_mask), int(right_vertices.shape[0]), "right cortex tSNR")

    left_parcel_ids = ensure_vertices_aligned(
        dt_vertices=left_vertices,
        dlabel_vertices=dlabel_vertices["L"][0],
        dlabel_parcel_ids=dlabel_vertices["L"][1],
        hemi="L",
    )
    right_parcel_ids = ensure_vertices_aligned(
        dt_vertices=right_vertices,
        dlabel_vertices=dlabel_vertices["R"][0],
        dlabel_parcel_ids=dlabel_vertices["R"][1],
        hemi="R",
    )

    reference = materialize_reference_timeseries(
        outdir=outdir, left_dt=left_dt, right_dt=right_dt,
        left_parcel_ids=left_parcel_ids, right_parcel_ids=right_parcel_ids,
        left_valid=left_valid, right_valid=right_valid,
        roi_registry=roi_registry, canonical7_order=canonical7_order, family17_order=family17_order,
    )
    write_reference_manifests(
        outdir=outdir,
        common_ref_dir=common_ref_dir,
        subject=subject,
        atlas_slug=ATLAS_SLUG,
        reference_scope=COMMON_REFERENCE_SLUG,
        dataset="hcp",
        task="rest",
        validity_policy="hcp_10000_over_std",
        tsnr_threshold=float(TSNR_THRESHOLD),
        resolved=resolved,
        source_run_lengths=source_run_lengths,
        n_timepoints=n_timepoints,
        reference=reference,
        canonical7_order=canonical7_order,
        family17_order=family17_order,
        kong2022_dlabel=Path(args.kong2022_dlabel).resolve(),
        cross_atlas_merge_json=Path(args.cross_atlas_merge_json).resolve(),
        left_tsnr_mask=left_tsnr_mask,
        right_tsnr_mask=right_tsnr_mask,
        include_dlabel_in_stage_inputs=True,
    )
    return 0


def main() -> int:
    return build_kong2022_reference(request=ReferenceRequest(**vars(build_arg_parser().parse_args())))


if __name__ == "__main__":
    raise SystemExit(main())
