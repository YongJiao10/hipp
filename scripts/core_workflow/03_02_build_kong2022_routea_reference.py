#!/usr/bin/env python3
"""
Process Step Description:
Build Kong2022 RouteA reference data. RouteA-specific step to generate reference template data for Kong2022 cortical networks.

Main Flow Path:
- 5th step in the numbered pipeline sequence (RouteA reference building for kong2022_routea):
  01_01_run_hippunfold_t1w_local.py -> 01_02_check_hippunfold_outputs.py -> 02_01_prepare_group_inputs.py -> 02_02_build_qc_manifest.py -> 03_02_build_kong2022_routea_reference.py (this step) -> 04_01/04_02 FC materialization -> 05_01 gradient computation -> 05_02 group atlas parcellation -> 06_xx presentation rendering -> 07_01 group matrix overview.

Inputs:
- Command-line arguments:
  - --shared-root: Path to output shared processed files (default: `outputs_hcp/inputs/hcp/kong2022/shared_inputs/routea`).
  - --route-a-root: RouteA specific reference outputs root (default: `outputs_hcp/inputs/hcp/kong2022/shared_inputs/reference_151`).
  - --data-root: Root directory containing raw structural BIDS folders (default: `data/hippunfold_input`).
  - --kong2022-dlabel: Path to standard Kong2022 dlabel.nii cortical atlas file.
  - --subjects-manifest: Path to QC-passed subject list manifest (default: `outputs_hcp/inputs/hcp/kong2022/manifests/hcp_7t_hippocampus_fmri_qc_pass_t1only_latest.txt`).
  - --subjects: Space-separated list of explicit subject IDs.
  - --jobs: Thread-pool concurrency cap (default: 4, hard-capped at 10).
  - --resume: Skip already completed subjects (default: True).
  - --overwrite: Overwrite existing files (default: False).
- Physical inputs:
  - A legacy single-file fMRI dtseries or an exact per-subject JSON run-artifact
    manifest. The manifest declares the ordered dtseries paths, run IDs, sizes,
    and SHA256 receipts; run concatenation remains in memory.
  - Kong2022 standard cortical dlabel atlas.

Outputs:
- Command-line output / file outputs:
  - RouteA cortical network reference products under `{shared_root}/sub-<subject>/reference/{atlas_slug}/`.
  - `--atlas-slug kong2022` writes the default direct canonical7 reference branch; it does not use canonical8 artifacts.
"""
from __future__ import annotations

import argparse
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from lib.extract_cortex_kong2022_route_a_roi_timeseries import build_kong2022_reference
from lib.io_utils import load_subject_ids
from lib.kong2022_label_utils import KONG2022_DLABEL_PATH, KONG2022_MERGE_CONFIG
from lib.kong2022_reference import ReferenceRequest, resolve_batch_subject_dtseries
from lib.output_layout import (
    DEFAULT_CORTEX_ROUTEA_ROOT,
    DEFAULT_KONG2022_MANIFEST_ROOT,
    DEFAULT_KONG2022_ROUTEA_ROOT,
)
from lib.task_design import normalize_task_branch
from lib.task_manifests import task_run_manifest_path
from lib.task_quality import (
    QC_DATASETS,
    parse_tsnr_threshold,
    require_cortex_reference_policy,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SHARED_ROOT = DEFAULT_KONG2022_ROUTEA_ROOT
DEFAULT_ROUTE_A_ROOT = DEFAULT_CORTEX_ROUTEA_ROOT
DEFAULT_DATA_ROOT = REPO_ROOT / "data" / "hippunfold_input"
DEFAULT_DTSERIES_TEMPLATE = "{data_root}/{sub_subject}/func/{sub_subject}_task-rest_run-concat.dtseries.nii"
DEFAULT_MANIFEST = DEFAULT_KONG2022_MANIFEST_ROOT / "hcp_7t_hippocampus_fmri_qc_pass_t1only_latest.txt"
MAX_CONCURRENCY = 10
SOURCE_TAG = "route_a_kong2022_areal_mshbm"
FORBIDDEN_ROUTEB_TOKEN = "kong2022_route_b"


def validate_routea_path(path: Path, arg_name: str) -> None:
    if FORBIDDEN_ROUTEB_TOKEN in path.parts:
        raise RuntimeError(
            f"RouteA compliance violation: {arg_name} points into a RouteB tree: {path}. "
            "Use route-agnostic/shared kong2022 manifests and RouteA-compatible roots."
        )





def subject_is_done(
    shared_root: Path, subject: str, atlas_slug: str, *, dataset: str,
    tsnr_threshold: str, expected_inputs: dict[str, object],
    tsnr_input_template: str | None = None, task: str = "rest",
) -> bool:
    """Reuse only a reference made from the requested policy and source selection."""
    ref_dir = shared_root / f"sub-{subject}" / "reference" / str(atlas_slug)
    stage = ref_dir / "stage_manifest.json"
    if not stage.is_file():
        return False
    try:
        payload = require_cortex_reference_policy(ref_dir, dataset=dataset, task=task)
        if parse_tsnr_threshold(payload["tsnr_threshold"]) != parse_tsnr_threshold(tsnr_threshold):
            return False
        if (payload.get("reference_label_source") != SOURCE_TAG
                or payload.get("atlas_slug") != str(atlas_slug)):
            return False
        if any(payload.get(field) != value for field, value in expected_inputs.items()):
            return False
        for side, hemi in (("left", "L"), ("right", "R")):
            gate_path = ref_dir.parent / "common" / "tsnr_gate" / f"cortex_{side}_tsnr_gate_summary.json"
            gate = json.loads(gate_path.read_text(encoding="utf-8"))
            expected_baseline = (
                str(Path(tsnr_input_template.format(subject=subject, hemi=hemi)).resolve())
                if tsnr_input_template is not None else None
            )
            if gate.get("baseline_input") != expected_baseline:
                return False
    except (OSError, ValueError, KeyError, TypeError):
        return False
    return True


def run_subject(
    *,
    subject: str,
    shared_root: Path,
    route_a_root: Path,
    data_root: Path,
    dtseries_template: str,
    run_artifact_manifest_template: str | None,
    kong2022_dlabel: Path | None,
    atlas_slug: str,
    overwrite: bool,
    run_metadata_template: str | None = None,
    task: str = "rest",
    task_manifest_root: str | None = None,
    dataset: str,
    tsnr_input_template: str | None = None,
    tsnr_threshold: str = "25",
    resume: bool = True,
) -> tuple[str, bool, str]:
    options: dict[str, object] = {
        "subject": subject,
        "shared_root": shared_root,
        "route_a_root": route_a_root,
        "atlas_slug": atlas_slug,
    }
    branch = normalize_task_branch(task) if task != "rest" else "rest"
    expected_inputs: dict[str, object] = {
        "route_a_lh_labels": str((route_a_root / f"sub-{subject}" / "lh_labels.txt").resolve()),
        "route_a_rh_labels": str((route_a_root / f"sub-{subject}" / "rh_labels.txt").resolve()),
        "kong2022_dlabel": str((kong2022_dlabel or KONG2022_DLABEL_PATH).resolve()),
        "cross_atlas_merge_json": str(KONG2022_MERGE_CONFIG.resolve()),
    }
    direct_manifest: Path | None = None
    if branch != "rest":
        if not task_manifest_root:
            return subject, False, (
                f"--task {branch} requires --task-manifest-root with a prebuilt "
                "per-subject run-artifact manifest (see lib.task_manifests)"
            )
        direct_manifest = task_run_manifest_path(task_manifest_root, subject=subject, task=branch)
    try:
        dt_options, dt_expected = resolve_batch_subject_dtseries(
            subject=subject,
            data_root=data_root,
            dtseries_template=dtseries_template,
            run_artifact_manifest_template=run_artifact_manifest_template,
            run_metadata_template=run_metadata_template,
            direct_manifest=direct_manifest,
            record_expected_lengths=True,
        )
    except (FileNotFoundError, ValueError, KeyError, TypeError) as exc:
        return subject, False, str(exc)
    expected_inputs.update(dt_expected)
    options.update(dt_options)
    if kong2022_dlabel is not None:
        options["kong2022_dlabel"] = kong2022_dlabel
    options.update(task=branch, dataset=dataset, tsnr_threshold=str(tsnr_threshold))
    if tsnr_input_template is not None:
        options["tsnr_input_template"] = tsnr_input_template
    options["overwrite"] = overwrite
    if resume and not overwrite and branch == "rest" and subject_is_done(
        shared_root, subject, atlas_slug, dataset=dataset, task=branch,
        tsnr_threshold=tsnr_threshold, tsnr_input_template=tsnr_input_template,
        expected_inputs=expected_inputs,
    ):
        return subject, True, "reused"
    try:
        build_kong2022_reference(request=ReferenceRequest(**options))  # type: ignore[arg-type]
    except Exception as exc:
        return subject, False, str(exc)[-4000:]
    return subject, True, "ok"


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Batch bridge Route A individualized labels into Route B-compatible reference/kong2022 arrays."
    )
    parser.add_argument("--shared-root", default=str(DEFAULT_SHARED_ROOT))
    parser.add_argument("--route-a-root", default=str(DEFAULT_ROUTE_A_ROOT))
    parser.add_argument("--data-root", default=str(DEFAULT_DATA_ROOT))
    parser.add_argument(
        "--dtseries-template",
        default=DEFAULT_DTSERIES_TEMPLATE,
        help=(
            "Per-subject dtseries template. Variables: {data_root}, {subject}, "
            "{sub_subject}. Default preserves the legacy HCP layout."
        ),
    )
    parser.add_argument(
        "--run-artifact-manifest-template",
        default=None,
        help=(
            "Per-subject JSON receipt declaring exact ordered run dtseries files, "
            "run IDs, sizes, and SHA256 values. Variables: {data_root}, {subject}, {sub_subject}."
        ),
    )
    parser.add_argument(
        "--run-metadata-template",
        default=None,
        help="Per-subject JSON template containing explicit run IDs and lengths for single-CIFTI inputs.",
    )
    parser.add_argument("--kong2022-dlabel", default=None)
    parser.add_argument(
        "--atlas-slug",
        choices=("kong2022",),
        default="kong2022",
        help="Reference branch; the unsuffixed Kong2022 branch is direct canonical7 from Route-A labels.",
    )
    parser.add_argument("--subjects-manifest", default=str(DEFAULT_MANIFEST))
    parser.add_argument("--subjects", nargs="*", default=None, help="Optional explicit subject IDs.")
    parser.add_argument("--jobs", type=int, default=4, help="Thread-pool concurrency (hard-capped at 10).")
    parser.add_argument(
        "--task",
        default="rest",
        help=(
            "Condition branch: rest preserves legacy behavior; emotion|wm|relational "
            "requires --task-manifest-root and forwards --task to the extractor, "
            "which validates run frames against the task contract."
        ),
    )
    parser.add_argument(
        "--task-manifest-root",
        default=None,
        help="Root holding task-<task>/sub-<subject>.json run-artifact manifests.",
    )
    parser.add_argument("--dataset", choices=QC_DATASETS, required=True)
    parser.add_argument(
        "--tsnr-threshold",
        default="25",
        help="Arbitrary screening criterion forwarded to the extractor (any positive number or 'none').",
    )
    parser.add_argument("--tsnr-input-template", help="Baseline cortex NPZ template with {subject}, {hemi}")
    parser.add_argument("--resume", action="store_true", default=True)
    parser.add_argument("--overwrite", action="store_true", default=False)
    args = parser.parse_args()

    shared_root = Path(args.shared_root).resolve()
    route_a_root = Path(args.route_a_root).resolve()
    data_root = Path(args.data_root).resolve()
    kong2022_dlabel = Path(args.kong2022_dlabel).resolve() if args.kong2022_dlabel else None
    subjects_manifest = Path(args.subjects_manifest).resolve()
    validate_routea_path(shared_root, "--shared-root")
    validate_routea_path(subjects_manifest, "--subjects-manifest")
    subjects = (
        [str(s).replace("sub-", "") for s in args.subjects]
        if args.subjects
        else load_subject_ids(subjects_manifest)
    )

    # Source selection must be resolved before deciding whether a cache matches.
    todo = list(subjects)

    print(f"[info] total={len(subjects)} todo={len(todo)} resume={bool(args.resume)}", flush=True)
    if not todo:
        print("[info] all subjects already done", flush=True)
        return 0

    jobs = max(1, min(int(args.jobs), MAX_CONCURRENCY, len(todo)))
    failed: list[str] = []
    with ThreadPoolExecutor(max_workers=jobs) as pool:
        futs = {
            pool.submit(
                run_subject,
                subject=s,
                shared_root=shared_root,
                route_a_root=route_a_root,
                data_root=data_root,
                dtseries_template=args.dtseries_template,
                run_artifact_manifest_template=args.run_artifact_manifest_template,
                run_metadata_template=args.run_metadata_template,
                kong2022_dlabel=kong2022_dlabel,
                atlas_slug=str(args.atlas_slug),
                overwrite=bool(args.overwrite),
                task=str(args.task),
                task_manifest_root=args.task_manifest_root,
                dataset=args.dataset,
                tsnr_input_template=args.tsnr_input_template,
                tsnr_threshold=str(args.tsnr_threshold),
                resume=bool(args.resume),
            ): s
            for s in todo
        }
        done = 0
        for fut in as_completed(futs):
            done += 1
            subject, ok, msg = fut.result()
            if ok:
                print(f"[done {done}/{len(todo)}] sub-{subject}", flush=True)
            else:
                failed.append(subject)
                print(f"[FAIL {done}/{len(todo)}] sub-{subject}\n{msg}", flush=True)

    if failed:
        print(f"[error] failed_subjects={len(failed)} {failed}", flush=True)
        return 1
    print(f"[ok] completed={len(todo)} jobs={jobs}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
