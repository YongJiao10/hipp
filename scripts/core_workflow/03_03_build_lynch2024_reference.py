#!/usr/bin/env python3
from __future__ import annotations

"""
Build Lynch2024 FASTANS/PFM reference inputs for the all-base-subject branch.

This maintained entrypoint consumes per-subject Lynch2024 component time series
and component metadata produced by external FASTANS/PFM assets, then writes the
two maintained atlas references:

- ``reference/lynch2024_20net``: original Lynch parent networks, excluding Noise.
- ``reference/lynch2024_7net``: merged canonical networks from
  ``config/cross_atlas_network_merge.json``.
"""

import argparse
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np

from lib.io_utils import load_subject_ids, load_run_segment_metadata, parse_run_segments
from lib.lynch2024_reference import (
    build_lynch_network_references,
    lynch2024_20net_order,
    lynch2024_7net_reference_contract,
)
from lib.output_layout import (
    DEFAULT_CORTEX_ASSETS_ROOT,
    DEFAULT_LYNCH2024_SHARED_INPUTS_ROOT,
)
from lib.path_templates import format_subject_path_template
from lib.workflow_utils import capped_jobs, utc_now_iso, write_json


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MERGE_CONFIG = REPO_ROOT / "config" / "cross_atlas_network_merge.json"
DEFAULT_COMPONENT_TIMESERIES_TEMPLATE = (
    "{cortex_pfm_root}/{sub_subject}/lynch2024/cortex_roi_component_timeseries.npy"
)
DEFAULT_COMPONENT_TSV_TEMPLATE = (
    "{cortex_pfm_root}/{sub_subject}/lynch2024/cortex_roi_components.tsv"
)
MAX_CONCURRENCY = 10


def _write_common_run_manifests(
    *,
    subject: str,
    shared_root: Path,
    n_timepoints: int,
    run_lengths: list[int] | tuple[int, ...],
    run_ids: tuple[int, ...],
    run_boundary_source: str,
) -> list[dict[str, object]]:
    common_dir = shared_root / f"sub-{subject}" / "reference" / "common" / "runs"
    segments = parse_run_segments(list(run_lengths), int(n_timepoints), run_ids=run_ids)
    for seg in segments:
        rid = int(seg["run_id"])
        run_dir = common_dir / f"run-{rid}"
        run_dir.mkdir(parents=True, exist_ok=True)
        summary = {
            "subject": subject,
            "reference_scope": "common",
            "atlas_slug": "lynch2024",
            "run_id": str(rid),
            "run_label": str(seg["run_label"]),
            "run_start": int(seg["run_start"]),
            "run_stop": int(seg["run_stop"]),
            "run_length": int(seg["run_length"]),
            "n_timepoints": int(seg["run_length"]),
            "run_boundary_source": str(run_boundary_source),
        }
        write_json(run_dir / "reference_summary.json", summary)
        write_json(
            run_dir / "stage_manifest.json",
            {
                "stage": "reference_run_slice",
                "status": "done",
                "timestamp_utc": utc_now_iso(),
                "params": {
                    "subject": subject,
                    "atlas_slug": "lynch2024",
                    "reference_scope": "common",
                    "run_id": str(rid),
                },
                "outputs": [str((run_dir / "reference_summary.json").resolve())],
            },
        )
    return segments


def process_subject(
    *,
    subject: str,
    shared_root: Path,
    cortex_pfm_root: Path,
    component_timeseries_template: str,
    component_tsv_template: str,
    merge_config: Path,
    run_lengths: list[int] | None,
    run_metadata_template: str | None = None,
) -> dict[str, object]:
    subject = str(subject).replace("sub-", "")
    component_timeseries_path = format_subject_path_template(
        component_timeseries_template,
        subject=subject,
        cortex_pfm_root=cortex_pfm_root,
    )
    component_tsv_path = format_subject_path_template(
        component_tsv_template,
        subject=subject,
        cortex_pfm_root=cortex_pfm_root,
    )
    component_ts = np.load(component_timeseries_path).astype(np.float32)
    if component_ts.ndim != 2:
        raise ValueError(
            f"Expected 2D Lynch component time series: {component_timeseries_path}"
        )
    reference_root = shared_root / f"sub-{subject}" / "reference"
    if run_metadata_template is None:
        raise ValueError(
            "Explicit run metadata is required; pass --run-metadata-template "
            "for ordered run IDs and lengths"
        )
    metadata_path = format_subject_path_template(
        run_metadata_template,
        subject=subject,
        shared_root=shared_root,
        cortex_pfm_root=cortex_pfm_root,
    )
    run_ids, metadata_lengths = load_run_segment_metadata(metadata_path)
    if (
        run_lengths is not None
        and tuple(int(x) for x in run_lengths) != metadata_lengths
    ):
        raise ValueError(f"Explicit --run-lengths disagree with {metadata_path}")
    run_lengths = list(metadata_lengths)
    # Fail before any reference output is created or replaced.
    if run_lengths is not None and sum(run_lengths) != int(component_ts.shape[1]):
        raise ValueError(
            f"Run lengths disagree with component frame count: metadata={metadata_path}, "
            f"components={component_timeseries_path}, run_total={sum(run_lengths)}, "
            f"frames={component_ts.shape[1]}"
        )
    parse_run_segments(run_lengths, int(component_ts.shape[1]), run_ids=run_ids)
    outputs = build_lynch_network_references(
        component_timeseries=component_ts,
        component_tsv=component_tsv_path,
        out_dir=reference_root,
        merge_config=merge_config,
    )
    segments = _write_common_run_manifests(
        subject=subject,
        shared_root=shared_root,
        n_timepoints=int(component_ts.shape[1]),
        run_lengths=run_lengths,
        run_ids=run_ids,
        run_boundary_source="explicit_run_metadata_template",
    )
    summary = {
        "subject": subject,
        "atlas_slug": "lynch2024",
        "reference_label_source": "lynch2024_fastans_pfm_external_assets",
        "component_timeseries": str(component_timeseries_path),
        "component_tsv": str(component_tsv_path),
        "cross_atlas_merge_json": str(Path(merge_config).resolve()),
        "n_components_input": int(component_ts.shape[0]),
        "n_timepoints": int(component_ts.shape[1]),
        "run_lengths": [int(seg["run_length"]) for seg in segments],
        "run_ids": list(run_ids),
        "run_segments": segments,
        "run_metadata": str(metadata_path) if metadata_path is not None else None,
        "outputs": {k: str(v) for k, v in outputs.items() if isinstance(v, Path)},
    }
    network_order = lynch2024_20net_order(merge_config)
    contract_7 = lynch2024_7net_reference_contract(merge_config)
    for atlas_slug in ("lynch2024_20net", "lynch2024_7net"):
        atlas_dir = reference_root / atlas_slug
        taxonomy_evidence = (
            {
                "network_taxonomy": "lynch20",
                "network_count": 20,
                "noise_excluded": True,
                "network_order": network_order,
                "network_registry_column": "lynch20_network",
                "network_name_column": "lynch20_network",
                "network_reference_tsv": "cortex_lynch20_networks.tsv",
                "network_reference_timeseries": "cortex_lynch20_network_timeseries.npy",
            }
            if atlas_slug == "lynch2024_20net"
            else {**contract_7}
        )
        write_json(
            atlas_dir / "reference_summary.json",
            {**summary, "atlas_slug": atlas_slug, **taxonomy_evidence},
        )
        write_json(
            atlas_dir / "stage_manifest.json",
            {
                "stage": "reference",
                "status": "done",
                "timestamp_utc": utc_now_iso(),
                **taxonomy_evidence,
                "params": {
                    "subject": subject,
                    "atlas_slug": atlas_slug,
                    "reference_label_source": "lynch2024_fastans_pfm_external_assets",
                },
                "inputs": [
                    str(component_timeseries_path),
                    str(component_tsv_path),
                    str(Path(merge_config).resolve()),
                    *([str(metadata_path)] if metadata_path is not None else []),
                ],
                "outputs": [
                    str((atlas_dir / "reference_summary.json").resolve()),
                    str((atlas_dir / "stage_manifest.json").resolve()),
                ],
            },
        )
    return {
        "subject": f"sub-{subject}",
        "n_components_input": int(component_ts.shape[0]),
        "n_timepoints": int(component_ts.shape[1]),
    }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build Lynch2024 PFM reference inputs."
    )
    parser.add_argument(
        "--shared-root", default=str(DEFAULT_LYNCH2024_SHARED_INPUTS_ROOT)
    )
    parser.add_argument("--subjects-manifest", required=True)
    parser.add_argument("--subjects", nargs="*", default=None)
    parser.add_argument("--cortex-pfm-root", default=str(DEFAULT_CORTEX_ASSETS_ROOT))
    parser.add_argument(
        "--component-timeseries-template", default=DEFAULT_COMPONENT_TIMESERIES_TEMPLATE
    )
    parser.add_argument(
        "--component-tsv-template", default=DEFAULT_COMPONENT_TSV_TEMPLATE
    )
    parser.add_argument("--cross-atlas-merge-json", default=str(DEFAULT_MERGE_CONFIG))
    parser.add_argument(
        "--run-lengths",
        nargs="*",
        type=int,
        default=None,
        help="Optional cross-check against the selected run metadata; it is not a standalone selector.",
    )
    parser.add_argument(
        "--run-metadata-template",
        required=True,
        help="Required per-subject JSON template containing ordered run IDs and lengths.",
    )
    parser.add_argument(
        "--jobs",
        type=int,
        default=1,
        help="Subject-level concurrency; hard-capped at 10.",
    )
    return parser


def main() -> int:
    args = build_arg_parser().parse_args()
    shared_root = Path(args.shared_root).resolve()
    cortex_pfm_root = Path(args.cortex_pfm_root).resolve()
    merge_config = Path(args.cross_atlas_merge_json).resolve()
    subjects = (
        [str(s).replace("sub-", "") for s in args.subjects]
        if args.subjects
        else load_subject_ids(Path(args.subjects_manifest).resolve())
    )
    jobs = capped_jobs(args.jobs, len(subjects))
    results: list[dict[str, object]] = []
    if jobs == 1:
        for subject in subjects:
            result = process_subject(
                subject=subject,
                shared_root=shared_root,
                cortex_pfm_root=cortex_pfm_root,
                component_timeseries_template=str(args.component_timeseries_template),
                component_tsv_template=str(args.component_tsv_template),
                merge_config=merge_config,
                run_lengths=args.run_lengths,
                run_metadata_template=args.run_metadata_template,
            )
            results.append(result)
            print(f"[done] {result['subject']}", flush=True)
    else:
        with ThreadPoolExecutor(max_workers=min(jobs, MAX_CONCURRENCY)) as pool:
            futs = [
                pool.submit(
                    process_subject,
                    subject=subject,
                    shared_root=shared_root,
                    cortex_pfm_root=cortex_pfm_root,
                    component_timeseries_template=str(
                        args.component_timeseries_template
                    ),
                    component_tsv_template=str(args.component_tsv_template),
                    merge_config=merge_config,
                    run_lengths=args.run_lengths,
                    run_metadata_template=args.run_metadata_template,
                )
                for subject in subjects
            ]
            for fut in as_completed(futs):
                result = fut.result()
                results.append(result)
                print(f"[done] {result['subject']}", flush=True)
    print(
        json.dumps(
            {"status": "ok", "subjects": len(results), "shared_root": str(shared_root)},
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
