#!/usr/bin/env python3
from __future__ import annotations

"""Build the accepted HCP7T participant QC roster.

Apply frame-weighted physical-run Power FD <=0.30 mm, bilateral HippUnfold
Dice >=0.70, and <=20 unsupported functional-GM vertices per consumed den-512
hemisphere, in that order. Missing inputs are unassessable, not quality failures.
No tSNR exclusion threshold or predetermined passing subject count is applied.

Motion is read only from exact Movement_Regressors.txt paths declared by
--motion-manifest (TSV: subject, run_label, filename, lss_path). Coverage is
read from shared-root/sub-ID/surface/raw/sub-ID_coverage_summary.json and must
match the declared physical runs and actual consumed 512-node hemispheres.
The established t1only_hippunfold_tsnr_qc report filenames and latest pass-list
filename remain the downstream interface; their names do not imply a tSNR gate.
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
from lib.io_utils import write_tsv


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
COVERAGE_POLICY = "cifti_hippocampal_common_run_gm_trilinear"


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


def read_motion(runs: list[dict[str, object]]) -> tuple[float | None, list[dict[str, object]], list[str]]:
    if not runs:
        return None, [], ["missing_motion_declaration"]
    missing = [f"missing_motion:{run['path']}" for run in runs if not Path(str(run["path"])).is_file()]
    if missing:
        return None, [], missing
    measured = []
    total_fd = 0.0
    total_frames = 0
    for run in runs:
        path = Path(str(run["path"]))
        motion = np.loadtxt(path, ndmin=2)
        if motion.shape[0] < 1 or motion.shape[1] != 12 or not np.isfinite(motion).all():
            raise ValueError(f"Expected finite physical-run HCP motion with 12 columns: {path}, shape={motion.shape}")
        differences = np.abs(np.diff(motion[:, :6], axis=0))
        # HCP first three parameters are mm, next three degrees. The remaining
        # six derivative columns are deliberately not used.
        fd_sum = float(differences[:, :3].sum() + differences[:, 3:].sum() * (math.pi / 180) * 50)
        frames = int(motion.shape[0])
        total_fd += fd_sum
        total_frames += frames  # First physical-run frame contributes zero FD.
        measured.append({**run, "n_frames": frames, "mean_power_fd_mm": fd_sum / frames})
    return total_fd / total_frames, measured, []


def coverage_summary_path(shared_root: Path, subject: str) -> Path:
    return shared_root / f"sub-{subject}/surface/raw/sub-{subject}_coverage_summary.json"


def read_coverage(path: Path, subject: str, runs: list[dict[str, object]]) -> dict[str, int] | None:
    if not path.is_file():
        return None
    obj = json.loads(path.read_text(encoding="utf-8"))
    if obj.get("subject") != f"sub-{subject}" or obj.get("coverage_policy") != COVERAGE_POLICY:
        raise ValueError(f"Coverage subject/policy mismatch: {path}")
    if obj.get("sampling_space") != "MNI152NLin6Asym":
        raise ValueError(f"Coverage sampling-space mismatch: {path}")
    ids = [run["run_id"] for run in runs]
    lengths = [run["n_frames"] for run in runs]
    labels = [run["run_label"] for run in runs]
    if obj.get("run_ids") != ids or obj.get("run_lengths") != lengths or obj.get("run_labels") != labels:
        raise ValueError(f"Coverage physical-run mismatch: {path}; expected run_ids={ids}, run_lengths={lengths}, run_labels={labels}")
    evidence = obj.get("density_evidence", {})
    counts = evidence.get("n_vertices", {})
    hemispheres = obj.get("hemispheres", {})
    unsupported = {}
    for hemi in ("L", "R"):
        item = hemispheres.get(hemi, {})
        if evidence.get("density") != "512" or counts.get(hemi) != 512 or item.get("n_vertices_total") != 512:
            raise ValueError(f"Coverage requires actual consumed den-512 hemisphere {hemi}: density_evidence={evidence}, path={path}")
        count = item.get("n_vertices_unsupported")
        if isinstance(count, bool) or not isinstance(count, int) or not 0 <= count <= 512:
            raise ValueError(f"Malformed unsupported-vertex count for hemisphere {hemi}: {path}")
        unsupported[hemi] = count
    return unsupported


def assess_subject(subject: str, batch_root: Path, shared_root: Path, declared_runs: list[dict[str, object]],
                   *, fd_threshold: float, dice_threshold: float, max_unsupported_vertices: int
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
    row["motion_runs"] = runs
    if missing:
        row["unassessable_reason"] = ",".join(missing)
        return row, inputs
    row["overall_mean_power_fd_mm"] = mean_fd
    row["pass_motion"] = mean_fd <= fd_threshold
    if not row["pass_motion"]:
        row.update(qc_status="fail", exclude_reason="motion_gt_threshold")
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
    path = coverage_summary_path(shared_root, subject)
    unsupported = read_coverage(path, subject, runs)
    if unsupported is None:
        row["unassessable_reason"] = f"missing_coverage:{path}"
        return row, inputs
    inputs.append(path)
    for hemi in ("L", "R"):
        row[f"unsupported_vertices_{hemi}"] = unsupported[hemi]
    row["pass_coverage"] = all(unsupported[h] <= max_unsupported_vertices for h in ("L", "R"))
    if not row["pass_coverage"]:
        row.update(qc_status="fail", exclude_reason="unsupported_vertices_gt_threshold")
        return row, inputs
    row.update(pass_all=True, qc_status="pass")
    return row, inputs


def main() -> int:
    parser = argparse.ArgumentParser(description="Build HCP7T physical-run Power FD, Dice and functional-GM coverage QC roster.")
    parser.add_argument("--hippunfold-batch-root", default=str(DEFAULT_BATCH_ROOT))
    parser.add_argument("--shared-root", default=str(DEFAULT_SHARED_ROOT))
    parser.add_argument("--motion-manifest", default=str(DEFAULT_MOTION_MANIFEST))
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    parser.add_argument("--manifest-dir", default=str(DEFAULT_MANIFEST_DIR))
    parser.add_argument("--subjects", nargs="*", default=None)
    parser.add_argument("--dice-threshold", type=float, default=0.70)
    parser.add_argument("--fd-threshold", type=float, default=0.30)
    parser.add_argument("--max-unsupported-vertices", type=int, default=20)
    args = parser.parse_args()
    batch_root = Path(args.hippunfold_batch_root).resolve()
    shared_root = Path(args.shared_root).resolve()
    out_dir = Path(args.out_dir).resolve()
    manifest_dir = Path(args.manifest_dir).resolve()
    if shared_root != DEFAULT_SHARED_ROOT.resolve() and (
        out_dir == DEFAULT_OUT_DIR.resolve() or manifest_dir == DEFAULT_MANIFEST_DIR.resolve()
    ):
        parser.error(f"Non-default --shared-root {shared_root} requires separate --out-dir and --manifest-dir; cannot write HCP rest QC destinations {DEFAULT_OUT_DIR.resolve()} or {DEFAULT_MANIFEST_DIR.resolve()}.")
    motion_manifest = Path(args.motion_manifest).resolve()
    declarations = load_motion_declarations(motion_manifest)
    subjects = load_subjects(batch_root, args.subjects)
    if not subjects:
        parser.error(f"No subjects declared or found under {batch_root}")
    rows = []
    inputs = [file_meta(motion_manifest)]
    for subject in subjects:
        row, paths = assess_subject(subject, batch_root, shared_root, declarations.get(subject, []),
            fd_threshold=args.fd_threshold, dice_threshold=args.dice_threshold,
            max_unsupported_vertices=args.max_unsupported_vertices)
        rows.append(row)
        inputs.extend(file_meta(path) for path in paths)
    pass_subjects = [str(row["subject"]).removeprefix("sub-") for row in rows if row["pass_all"]]
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_dir.mkdir(parents=True, exist_ok=True)
    qc_tsv = out_dir / "t1only_hippunfold_tsnr_qc.tsv"
    qc_json = out_dir / "t1only_hippunfold_tsnr_qc.json"
    manifest = manifest_dir / f"hcp_7t_hippocampus_fmri_qc_pass_t1only_{len(pass_subjects)}.txt"
    latest_manifest = manifest_dir / "hcp_7t_hippocampus_fmri_qc_pass_t1only_latest.txt"
    fields = ["subject", "overall_mean_power_fd_mm", "dice_L", "dice_R", "unsupported_vertices_L", "unsupported_vertices_R",
              "pass_motion", "pass_dice", "pass_coverage", "pass_all", "qc_status", "exclude_reason", "unassessable_reason"]
    write_tsv(qc_tsv, rows, fields)
    text = "\n".join(pass_subjects) + ("\n" if pass_subjects else "")
    manifest.write_text(text, encoding="utf-8")
    latest_manifest.write_text(text, encoding="utf-8")
    payload = {
        "stage": "t1only_qc_manifest", "status": "done", "timestamp_utc": utc_now_iso(), "git_commit": git_commit(),
        "hippunfold_batch_root": str(batch_root), "shared_root": str(shared_root), "motion_manifest": str(motion_manifest),
        "dice_threshold": args.dice_threshold, "fd_threshold_mm": args.fd_threshold,
        "max_unsupported_vertices_per_hemisphere": args.max_unsupported_vertices,
        "coverage_policy": COVERAGE_POLICY, "motion_rotation_units": "degrees", "power_fd_radius_mm": 50,
        "power_fd_first_frame": 0, "motion_aggregation": "physical_run_frame_weighted_mean",
        "n_subjects_total": len(subjects), "n_subjects_pass": len(pass_subjects),
        "n_subjects_fail": sum(row["qc_status"] == "fail" for row in rows),
        "n_subjects_unassessable": sum(row["qc_status"] == "unassessable" for row in rows),
        "manifest": str(manifest), "latest_manifest": str(latest_manifest), "qc_tsv": str(qc_tsv), "rows": rows, "inputs": inputs,
        "outputs": [str(qc_tsv), str(qc_json), str(manifest), str(latest_manifest)],
    }
    write_json(qc_json, payload)
    print(json.dumps({"n_total": len(subjects), "n_pass": len(pass_subjects), "n_fail": payload["n_subjects_fail"],
                      "n_unassessable": payload["n_subjects_unassessable"], "manifest": str(manifest)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
