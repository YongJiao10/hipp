#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
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
from .output_layout import DEFAULT_CORTEX_ROUTEA_ROOT, DEFAULT_KONG2022_ROUTEA_ROOT
from .kong2022_label_utils import (
    KONG2022_DLABEL_PATH,
    KONG2022_MERGE_CONFIG,
    canonical_families_17_order,
    load_kong2022_bundle,
)
from .task_design import TASK_RUN_LENGTHS, normalize_task_branch
from .task_quality import QC_DATASETS, parse_tsnr_threshold, signal_validity_policy
from .kong2022_reference import (
    ReferenceRequest,
    materialize_reference_timeseries,
    resolve_reference_cifti_inputs,
    validate_cifti_run_lengths,
    write_reference_manifests,
)


REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_DATA_ROOT = REPO_ROOT / "data" / "hippunfold_input"
DEFAULT_SHARED_ROOT = DEFAULT_KONG2022_ROUTEA_ROOT
DEFAULT_ROUTE_A_ROOT = DEFAULT_CORTEX_ROUTEA_ROOT
DEFAULT_RUN_IDS = [1, 2, 3, 4]
ATLAS_SLUG = "kong2022"
COMMON_REFERENCE_SLUG = "common"
TSNR_THRESHOLD = 25.0
HCP_MODE = 10000.0
ROUTE_A_SOURCE_TAG = "route_a_kong2022_areal_mshbm"


def route_a_label_paths(route_a_root: Path, subject: str) -> tuple[Path, Path]:
    subject = str(subject).replace("sub-", "")
    subject_dir = Path(route_a_root) / f"sub-{subject}"
    lh = subject_dir / "lh_labels.txt"
    rh = subject_dir / "rh_labels.txt"
    missing = [p for p in (lh, rh) if not p.exists()]
    if missing:
        expected = f"{subject_dir}/lh_labels.txt and {subject_dir}/rh_labels.txt"
        raise FileNotFoundError(f"Missing flat RouteA label files for sub-{subject}; expected {expected}")
    return lh, rh


def load_surface_label_vector(path: Path, expected_len: int = 32492) -> np.ndarray:
    arr = np.loadtxt(str(path), dtype=np.int32)
    if arr.ndim != 1:
        arr = arr.reshape(-1)
    if int(arr.shape[0]) != int(expected_len):
        raise ValueError(f"Expected {expected_len} labels in {path}, got {arr.shape[0]}")
    return arr.astype(np.int32, copy=False)


def align_surface_labels_to_dt_vertices(full_surface_labels: np.ndarray, dt_vertices: np.ndarray, hemi: str) -> np.ndarray:
    v = np.asarray(dt_vertices, dtype=np.int32).reshape(-1)
    if v.size == 0:
        raise RuntimeError(f"No dtseries vertices for hemi-{hemi}")
    v_max = int(np.max(v))
    if v_max >= int(full_surface_labels.shape[0]):
        raise ValueError(
            f"Vertex index out of bounds for hemi-{hemi}: max_vertex={v_max} label_len={full_surface_labels.shape[0]}"
        )
    return full_surface_labels[v].astype(np.int32, copy=False)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Bridge Route A individualized labels to Route B-compatible reference/kong2022 arrays."
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
    parser.add_argument(
        "--atlas-slug",
        choices=("kong2022",),
        default=ATLAS_SLUG,
        help="The unsuffixed Kong2022 reference is direct canonical7 from Route-A labels.",
    )
    parser.add_argument("--shared-root", default=str(DEFAULT_SHARED_ROOT))
    parser.add_argument("--route-a-root", default=str(DEFAULT_ROUTE_A_ROOT))
    parser.add_argument("--left-tsnr-mask", default=None)
    parser.add_argument("--right-tsnr-mask", default=None)
    parser.add_argument("--run-lengths", nargs="*", type=int, default=None)
    parser.add_argument(
        "--task",
        default="rest",
        help=(
            "Condition branch recorded in summaries; emotion|wm|relational "
            "validates every run length against the task frame contract."
        ),
    )
    parser.add_argument("--dataset", choices=QC_DATASETS, required=True)
    parser.add_argument(
        "--tsnr-threshold",
        default="25",
        help=(
            "Arbitrary screening criterion: any positive number gates at that "
            "value; 'none' keeps finite, nonconstant vertices with no tSNR screening."
        ),
    )
    parser.add_argument("--tsnr-input-template", help="Baseline cortex NPZ template with {subject}, {hemi}")
    parser.add_argument("--outdir", default=None)
    parser.add_argument("--overwrite", action="store_true", default=False)
    return parser


def build_kong2022_reference(*, request: ReferenceRequest) -> int:
    """Build one subject without CLI parsing or process-global state changes."""
    args = request
    ATLAS_SLUG = str(args.atlas_slug)
    if args.task != "rest" and not args.run_dtseries:
        raise ValueError("Task references require --run-dtseries and --run-artifact-manifest with original BOLD inputs")
    subject = str(args.subject).replace("sub-", "")
    shared_root = Path(args.shared_root).resolve()
    route_a_root = Path(args.route_a_root).resolve()
    resolved = resolve_reference_cifti_inputs(
        args,
        subject=subject,
        shared_root=shared_root,
        required_artifacts=("bold", "dtseries") if args.task != "rest" else ("dtseries",),
    )
    outdir = (
        Path(args.outdir).resolve()
        if args.outdir
        else (shared_root / f"sub-{subject}" / "reference" / ATLAS_SLUG).resolve()
    )
    common_ref_dir = (shared_root / f"sub-{subject}" / "reference" / COMMON_REFERENCE_SLUG).resolve()

    existing_summary = outdir / "reference_summary.json"
    if existing_summary.exists() and not args.overwrite:
        existing = json.loads(existing_summary.read_text(encoding="utf-8"))
        if str(existing.get("reference_label_source", "")) != ROUTE_A_SOURCE_TAG:
            raise RuntimeError(
                f"Refusing to overwrite non-Route-A reference at {existing_summary}. "
                "Use a different --shared-root or pass --overwrite."
            )

    lh_labels_path, rh_labels_path = route_a_label_paths(route_a_root=route_a_root, subject=subject)
    route_out_dir = lh_labels_path.parent
    lh_full = load_surface_label_vector(lh_labels_path)
    rh_full = load_surface_label_vector(rh_labels_path)

    bundle = load_kong2022_bundle(
        dlabel_path=Path(args.kong2022_dlabel).resolve(),
        config_path=Path(args.cross_atlas_merge_json).resolve(),
    )
    roi_registry = list(bundle["roi_registry"])
    canonical7_order = list(bundle["canonical7_order"])
    family17_order = canonical_families_17_order()

    dt_data, dt_axis, source_run_lengths = load_cifti_dtseries_inputs(resolved.dtseries_inputs)
    dt_data_t = dt_data.T
    n_timepoints = int(dt_data.shape[0])
    source_run_lengths = validate_cifti_run_lengths(
        dtseries_input_mode=resolved.dtseries_input_mode,
        declared_run_lengths=resolved.declared_run_lengths,
        source_run_lengths=source_run_lengths,
        n_timepoints=n_timepoints,
    )
    task_branch = normalize_task_branch(args.task) if str(args.task) != "rest" else "rest"
    policy = signal_validity_policy(args.dataset, task_branch)
    if task_branch != "rest":
        expected = int(TASK_RUN_LENGTHS[task_branch])
        if [int(v) for v in source_run_lengths] != [expected] * len(source_run_lengths):
            raise ValueError(
                f"Task {task_branch} expects {expected} frames per run, "
                f"got headers={tuple(int(v) for v in source_run_lengths)}"
            )
    outdir.mkdir(parents=True, exist_ok=True)
    common_ref_dir.mkdir(parents=True, exist_ok=True)
    left_vertices, left_dt = extract_structure_data(dt_axis, dt_data_t, "CORTEX_LEFT")
    right_vertices, right_dt = extract_structure_data(dt_axis, dt_data_t, "CORTEX_RIGHT")
    left_parcel_ids = align_surface_labels_to_dt_vertices(lh_full, left_vertices, hemi="L")
    right_parcel_ids = align_surface_labels_to_dt_vertices(rh_full, right_vertices, hemi="R")

    if policy != "hcp_10000_over_std" and (args.left_tsnr_mask or args.right_tsnr_mask):
        raise ValueError("Selected dataset/task must compute signal validity from its declared inputs; external tSNR masks are not supported")
    task_bold_paths = [r["artifacts"]["bold"] for r in resolved.records] if task_branch != "rest" else None
    def qc_input(hemi: str) -> Path | None:
        return Path(args.tsnr_input_template.format(subject=subject, hemi=hemi)).resolve() if args.tsnr_input_template else None
    tsnr_dir = common_ref_dir / "tsnr_gate"
    task_threshold = parse_tsnr_threshold(str(getattr(args, "tsnr_threshold", "25")))
    left_tsnr_mask = (
        Path(args.left_tsnr_mask).resolve()
        if args.left_tsnr_mask
        else compute_cortex_tsnr_gate(data=left_dt, outdir=tsnr_dir, hemi="left", threshold=task_threshold, task=task_branch, run_lengths=source_run_lengths, dataset=args.dataset, task_bold_paths=task_bold_paths, tsnr_input=qc_input("L"), subject=subject, vertex_ids=left_vertices).resolve()
    )
    right_tsnr_mask = (
        Path(args.right_tsnr_mask).resolve()
        if args.right_tsnr_mask
        else compute_cortex_tsnr_gate(data=right_dt, outdir=tsnr_dir, hemi="right", threshold=task_threshold, task=task_branch, run_lengths=source_run_lengths, dataset=args.dataset, task_bold_paths=task_bold_paths, tsnr_input=qc_input("R"), subject=subject, vertex_ids=right_vertices).resolve()
    )
    left_valid = load_required_mask(str(left_tsnr_mask), int(left_vertices.shape[0]), "left cortex tSNR")
    right_valid = load_required_mask(str(right_tsnr_mask), int(right_vertices.shape[0]), "right cortex tSNR")

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
        dataset=args.dataset,
        task=task_branch,
        validity_policy=policy,
        tsnr_threshold=task_threshold,
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
        reference_label_source=ROUTE_A_SOURCE_TAG,
        route_a_extra_summary={
            "route_a_root": str(route_a_root),
            "route_a_label_dir": str(route_out_dir),
            "route_a_lh_labels": str(lh_labels_path),
            "route_a_rh_labels": str(rh_labels_path),
        },
        extra_input_files=(lh_labels_path, rh_labels_path),
        include_dlabel_in_stage_inputs=False,
    )
    return 0


def main() -> int:
    return build_kong2022_reference(request=ReferenceRequest(**vars(build_arg_parser().parse_args())))


if __name__ == "__main__":
    raise SystemExit(main())
