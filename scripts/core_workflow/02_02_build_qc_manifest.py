#!/usr/bin/env python3
from __future__ import annotations

"""Build the accepted participant/visit QC roster for the selected dataset/task.

Apply frame-weighted physical-run Power FD <=0.50 mm, bilateral HippUnfold
Dice >=0.70, and <=20 unusable NIfTI-sampled vertices per consumed den-512
hemisphere, in that order. Missing inputs are unassessable, not quality failures.
No predetermined passing subject count is applied.

Shared BOLD/motion inputs are declared by --qc-run-inputs. HCP7T rest also
supports its --motion-manifest table (subject, run_label, filename, lss_path). Coverage is
read from shared-root/sub-ID/surface/raw/sub-ID_coverage_summary.json and must
match the declared physical runs and actual consumed 512-node hemispheres.
Counts use the current raw sampled signals and NIfTI support, with
finite/nonconstant validity in every run.
The participant QC report and latest pass list are downstream interfaces.
"""
import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np

from lib.output_layout import (
    DEFAULT_HIPPUNFOLD_BATCH_ROOT,
    DEFAULT_KONG2022_MANIFEST_ROOT,
    DEFAULT_KONG2022_QC_ROOT,
    DEFAULT_KONG2022_SURFACE_ROOT,
)
from lib.workflow_utils import file_meta, git_commit, utc_now_iso, write_json
from lib.io_utils import load_subject_ids, write_tsv
from lib.surface_sampling import NIFTI_COVERAGE_POLICY, load_hippocampal_coverage_validity
from lib.participant_qc import load_qc_run_inputs, pass_roster_name, read_motion, read_hippunfold_completion
from lib.task_quality import QC_DATASETS


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_BATCH_ROOT = DEFAULT_HIPPUNFOLD_BATCH_ROOT
DEFAULT_SHARED_ROOT = DEFAULT_KONG2022_SURFACE_ROOT
DEFAULT_OUT_DIR = DEFAULT_KONG2022_QC_ROOT
DEFAULT_MANIFEST_DIR = DEFAULT_KONG2022_MANIFEST_ROOT
DEFAULT_MOTION_MANIFEST = REPO_ROOT / "manifests/hcp7t_motion_transfer_20261019.tsv"
HCP_RUN_IDS = {
    "rfMRI_REST1_7T_PA": 1,
    "rfMRI_REST2_7T_AP": 2,
    "rfMRI_REST3_7T_PA": 3,
    "rfMRI_REST4_7T_AP": 4,
}
COVERAGE_POLICY = NIFTI_COVERAGE_POLICY


def load_subjects(batch_root: Path, explicit: list[str] | None) -> list[str]:
    if explicit:
        return sorted({s.removeprefix("sub-") for s in explicit})
    return sorted(p.name.removeprefix("sub-") for p in batch_root.glob("sub-*") if p.is_dir())


def read_dice(batch_root: Path, subject: str, hemi: str) -> tuple[float | None, Path]:
    path = (batch_root / f"sub-{subject}/hippunfold/sub-{subject}/qc"
            / f"sub-{subject}_hemi-{hemi}_desc-unetf3d_dice.tsv")
    if not path.is_file():
        return None, path
    lines = path.read_text(encoding="utf-8").strip().splitlines()
    try:
        value = float(lines[0].strip())
    except (ValueError, IndexError) as exc:
        raise ValueError(f"Malformed Dice value: {path}") from exc
    if not math.isfinite(value) or not 0 <= value <= 1:
        raise ValueError(f"Invalid Dice value {value}: {path}")
    return value, path


def load_motion_declarations(path: Path) -> dict[str, list[dict[str, object]]]:
    """Read exact declared physical-run motion paths; never search for a substitute."""
    records: dict[str, list[dict[str, object]]] = {}
    with path.open(encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream, delimiter="\t")
        required = {"subject", "run_label", "filename", "lss_path"}
        if not required.issubset(reader.fieldnames or []):
            raise ValueError(f"Motion manifest requires {sorted(required)}: {path}")
        seen: set[tuple[str, int]] = set()
        for row in reader:
            if row["filename"] != "Movement_Regressors.txt":
                continue
            subject = row["subject"].removeprefix("sub-")
            label = row["run_label"]
            if not subject or label not in HCP_RUN_IDS:
                raise ValueError(f"Unsupported HCP subject/run declaration {subject}/{label}: {path}")
            run_id = HCP_RUN_IDS[label]
            key = (subject, run_id)
            if key in seen:
                raise ValueError(f"Duplicate motion run declaration sub-{subject} {label}: {path}")
            seen.add(key)
            motion = Path(row["lss_path"])
            if not motion.is_absolute() or motion.name != "Movement_Regressors.txt":
                raise ValueError(f"Motion declaration requires exact absolute Movement_Regressors.txt path: {path}: {motion}")
            records.setdefault(subject, []).append({"run_id": run_id, "run_label": label, "path": str(motion)})
    for runs in records.values():
        runs.sort(key=lambda run: int(run["run_id"]))
    return records



def coverage_summary_path(shared_root: Path, subject: str) -> Path:
    return shared_root / f"sub-{subject}/surface/raw/sub-{subject}_coverage_summary.json"


def read_coverage(path: Path, subject: str, runs: list[dict[str, object]]) -> tuple[dict[str, int] | None, list[Path]]:
    if not path.is_file():
        return None, []
    obj = json.loads(path.read_text(encoding="utf-8"))
    if obj.get("subject") != f"sub-{subject}" or obj.get("coverage_policy") != COVERAGE_POLICY:
        raise ValueError(f"Coverage subject/policy mismatch: {path}")
    if obj.get("sampling_space") != "MNI152NLin6Asym":
        raise ValueError(f"Coverage sampling-space mismatch: {path}")
    ids = [run["run_id"] for run in runs]
    lengths = [run.get('n_timepoints', run["n_frames"]) for run in runs]
    labels = [run["run_label"] for run in runs]
    if obj.get("run_ids") != ids or obj.get("run_lengths") != lengths or obj.get("run_labels") != labels:
        raise ValueError(f"Coverage physical-run mismatch: {path}; expected run_ids={ids}, run_lengths={lengths}, run_labels={labels}")
    if runs and 'bold_path' in runs[0] and obj.get('bold_inputs') != [run['bold_path'] for run in runs]:
        raise ValueError(f'Coverage BOLD input mismatch: {path}')
    evidence = obj.get("density_evidence", {})
    counts = evidence.get("n_vertices", {})
    hemispheres = obj.get("hemispheres", {})
    unsupported = {}
    inputs = [path]
    for hemi in ("L", "R"):
        item = hemispheres.get(hemi, {})
        if evidence.get("density") != "512" or counts.get(hemi) != 512 or item.get("n_vertices_total") != 512:
            raise ValueError(f"Coverage requires actual consumed den-512 hemisphere {hemi}: density_evidence={evidence}, path={path}")
        valid, source_paths = load_hippocampal_coverage_validity(path.parent, subject, hemi, obj)
        unsupported[hemi] = int(np.count_nonzero(~valid))
        inputs.extend(source_paths)
    return unsupported, inputs


def assess_subject(subject: str, batch_root: Path, shared_root: Path, declared_runs: list[dict[str, object]],
                   *, fd_threshold: float, dice_threshold: float, max_unsupported_vertices: int,
                   required_hippunfold_version: str | None = None
                   ) -> tuple[dict[str, object], list[Path]]:
    row: dict[str, object] = {
        "subject": f"sub-{subject}", "overall_mean_power_fd_mm": "", "motion_runs": [],
        "dice_L": "", "dice_R": "", "unsupported_vertices_L": "", "unsupported_vertices_R": "",
        "pass_motion": None, "pass_dice": None, "pass_coverage": None,
        "pass_all": False, "qc_status": "unassessable", "exclude_reason": "", "unassessable_reason": "",
    }
    inputs: list[Path] = []
    mean_fd, runs, missing = read_motion(declared_runs)
    inputs.extend(Path(str(run["path"])) for run in declared_runs if Path(str(run["path"])).is_file())
    inputs.extend(Path(str(run['temporal_mask_path'])) for run in declared_runs
                  if run.get('temporal_mask_path') and Path(str(run['temporal_mask_path'])).is_file())
    row["motion_runs"] = runs
    if missing:
        row["unassessable_reason"] = ",".join(missing)
        return row, inputs
    row["overall_mean_power_fd_mm"] = mean_fd
    row["pass_motion"] = mean_fd <= fd_threshold
    if not row["pass_motion"]:
        row.update(qc_status="fail", exclude_reason="motion_gt_threshold")
        return row, inputs
    if required_hippunfold_version is not None:
        completed, check = read_hippunfold_completion(batch_root, subject, required_hippunfold_version)
        if completed is None:
            row['unassessable_reason'] = f'missing_hippunfold_completion:{check}'
            return row, inputs
        inputs.append(check)
        if not completed:
            row.update(qc_status='fail', exclude_reason='hippunfold_incomplete')
            return row, inputs
    missing = []
    for hemi in ("L", "R"):
        value, path = read_dice(batch_root, subject, hemi)
        if value is None:
            missing.append(f"missing_dice_{hemi}:{path}")
        else:
            row[f"dice_{hemi}"] = value
            inputs.append(path)
    if missing:
        row["unassessable_reason"] = ",".join(missing)
        return row, inputs
    row["pass_dice"] = all(float(row[f"dice_{h}"]) >= dice_threshold for h in ("L", "R"))
    if not row["pass_dice"]:
        row.update(qc_status="fail", exclude_reason="dice_lt_threshold")
        return row, inputs
    if any('bold_path' in run and not run['bold_path'] for run in declared_runs):
        row['unassessable_reason'] = 'missing_bold_nifti_declaration:qc-run-inputs'
        return row, inputs
    missing_bold = [str(run['bold_path']) for run in declared_runs
                    if 'bold_path' in run and not Path(str(run['bold_path'])).is_file()]
    if missing_bold:
        row['unassessable_reason'] = 'missing_bold_nifti:' + ','.join(missing_bold)
        return row, inputs
    path = coverage_summary_path(shared_root, subject)
    try:
        unsupported, coverage_inputs = read_coverage(path, subject, runs)
    except FileNotFoundError as exc:
        row["unassessable_reason"] = f"missing_coverage_input:{exc}"
        return row, inputs
    if unsupported is None:
        row["unassessable_reason"] = f"missing_coverage:{path}"
        return row, inputs
    inputs.extend(coverage_inputs)
    for hemi in ("L", "R"):
        row[f"unsupported_vertices_{hemi}"] = unsupported[hemi]
    row["pass_coverage"] = all(unsupported[h] <= max_unsupported_vertices for h in ("L", "R"))
    if not row["pass_coverage"]:
        row.update(qc_status="fail", exclude_reason="unsupported_vertices_gt_threshold")
        return row, inputs
    row.update(pass_all=True, qc_status="pass")
    return row, inputs


def main() -> int:
    parser = argparse.ArgumentParser(description="Build shared physical-run Power FD, Dice and NIfTI coverage QC rosters.")
    parser.add_argument('--dataset', choices=QC_DATASETS, default='hcp')
    parser.add_argument('--task', choices=('rest', 'emotion', 'wm', 'relational'), default='rest')
    parser.add_argument('--qc-run-inputs', type=Path, help='Explicit physical-run BOLD and motion TSV; shared across datasets/tasks.')
    parser.add_argument('--require-hippunfold-version')
    parser.add_argument("--hippunfold-batch-root", default=str(DEFAULT_BATCH_ROOT))
    parser.add_argument("--shared-root", default=str(DEFAULT_SHARED_ROOT))
    parser.add_argument("--motion-manifest", default=str(DEFAULT_MOTION_MANIFEST))
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    parser.add_argument("--manifest-dir", default=str(DEFAULT_MANIFEST_DIR))
    parser.add_argument("--subjects", nargs="*", default=None)
    parser.add_argument("--subjects-manifest", type=Path, help="Complete participant or visit roster, one ID per line.")
    parser.add_argument("--dice-threshold", type=float, default=0.70)
    parser.add_argument("--fd-threshold", type=float, default=0.50)
    parser.add_argument("--max-unsupported-vertices", type=int, default=10)
    args = parser.parse_args()
    if args.qc_run_inputs and not (args.subjects or args.subjects_manifest):
        parser.error('--qc-run-inputs requires the complete roster via --subjects or --subjects-manifest')
    if args.subjects and args.subjects_manifest:
        parser.error('Choose either --subjects or --subjects-manifest')
    if (args.dataset != 'hcp' or args.task != 'rest') and args.qc_run_inputs is None:
        parser.error('This dataset/task requires --qc-run-inputs with its exact BOLD and motion declarations')
    batch_root = Path(args.hippunfold_batch_root).resolve()
    shared_root = Path(args.shared_root).resolve()
    out_dir = Path(args.out_dir).resolve()
    manifest_dir = Path(args.manifest_dir).resolve()
    if (args.dataset != 'hcp' or args.task != 'rest') and any((
        shared_root == DEFAULT_SHARED_ROOT.resolve(), out_dir == DEFAULT_OUT_DIR.resolve(),
        manifest_dir == DEFAULT_MANIFEST_DIR.resolve(),
    )):
        parser.error('Non-HCP-rest QC requires separate --shared-root, --out-dir and --manifest-dir; cannot write HCP rest destinations')
    if shared_root != DEFAULT_SHARED_ROOT.resolve() and (
        out_dir == DEFAULT_OUT_DIR.resolve() or manifest_dir == DEFAULT_MANIFEST_DIR.resolve()
    ):
        parser.error(f"Non-default --shared-root {shared_root} requires separate --out-dir and --manifest-dir; cannot write HCP rest QC destinations {DEFAULT_OUT_DIR.resolve()} or {DEFAULT_MANIFEST_DIR.resolve()}.")
    motion_manifest = Path(args.qc_run_inputs or args.motion_manifest).resolve()
    declarations = load_qc_run_inputs(motion_manifest) if args.qc_run_inputs else load_motion_declarations(motion_manifest)
    subjects = (sorted(load_subject_ids(args.subjects_manifest)) if args.subjects_manifest
                else load_subjects(batch_root, args.subjects))
    if not subjects:
        parser.error(f"No subjects declared or found under {batch_root}")
    rows = []
    inputs = [file_meta(motion_manifest)]
    for subject in subjects:
        row, paths = assess_subject(subject, batch_root, shared_root, declarations.get(subject, []),
            fd_threshold=args.fd_threshold, dice_threshold=args.dice_threshold,
            max_unsupported_vertices=args.max_unsupported_vertices,
            required_hippunfold_version=args.require_hippunfold_version)
        rows.append(row)
        inputs.extend(file_meta(path) for path in paths)
    pass_subjects = [str(row["subject"]).removeprefix("sub-") for row in rows if row["pass_all"]]
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_dir.mkdir(parents=True, exist_ok=True)
    qc_tsv = out_dir / "t1only_hippunfold_qc.tsv"
    qc_json = out_dir / "t1only_hippunfold_qc.json"
    latest_manifest = manifest_dir / pass_roster_name(args.dataset, args.task)
    fields = ["subject", "overall_mean_power_fd_mm", "dice_L", "dice_R", "unsupported_vertices_L", "unsupported_vertices_R",
              "pass_motion", "pass_dice", "pass_coverage", "pass_all", "qc_status", "exclude_reason", "unassessable_reason"]
    write_tsv(qc_tsv, rows, fields)
    text = "\n".join(pass_subjects) + ("\n" if pass_subjects else "")
    latest_manifest.write_text(text, encoding="utf-8")
    motion_formats = {run.get('motion_format', 'hcp_movement_degrees')
                      for subject in subjects for run in declarations.get(subject, [])}
    payload = {
        "stage": "t1only_qc_manifest", "status": "incomplete_inputs" if any(row['qc_status'] == 'unassessable' for row in rows) else "done", "timestamp_utc": utc_now_iso(), "git_commit": git_commit(),
        'dataset': args.dataset, 'task': args.task, 'required_hippunfold_version': args.require_hippunfold_version,
        "hippunfold_batch_root": str(batch_root), "shared_root": str(shared_root), "motion_manifest": str(motion_manifest),
        "dice_threshold": args.dice_threshold, "fd_threshold_mm": args.fd_threshold,
        "max_unsupported_vertices_per_hemisphere": args.max_unsupported_vertices,
        "coverage_policy": COVERAGE_POLICY,
        "motion_rotation_units": "degrees" if motion_formats == {'hcp_movement_degrees'} else None,
        'motion_formats': sorted(motion_formats), "power_fd_radius_mm": 50,
        "power_fd_first_frame": 0, "motion_aggregation": "physical_run_frame_weighted_mean",
        "n_subjects_total": len(subjects), "n_subjects_pass": len(pass_subjects),
        "n_subjects_fail": sum(row["qc_status"] == "fail" for row in rows),
        "n_subjects_unassessable": sum(row["qc_status"] == "unassessable" for row in rows),
        "manifest": str(latest_manifest), "qc_tsv": str(qc_tsv), "rows": rows, "inputs": inputs,
        "outputs": [str(qc_tsv), str(qc_json), str(latest_manifest)],
    }
    write_json(qc_json, payload)
    print(json.dumps({"n_total": len(subjects), "n_pass": len(pass_subjects), "n_fail": payload["n_subjects_fail"],
                      "n_unassessable": payload["n_subjects_unassessable"], "manifest": str(latest_manifest)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
