#!/usr/bin/env python3
from __future__ import annotations

"""
Process Step Description:
Build quality control (QC) manifest. Generates a manifest file containing information on all passed/failed subjects so that subsequent steps only include QC-passed data.

Main Flow Path:
- 4th step in the numbered pipeline sequence:
  01_01_run_hippunfold_t1w_local.py -> 01_02_check_hippunfold_outputs.py -> 02_01_prepare_group_inputs.py -> 02_02_build_qc_manifest.py (this step) -> 03_01/03_02 reference building -> 04_01/04_02/04_03 FC materialization -> 05_01 gradient computation -> 05_02 group atlas parcellation -> 06_xx presentation rendering -> 07_01 group matrix overview.

Inputs:
- Command-line arguments:
  - --hippunfold-batch-root: Batch output root containing sub-*/hippunfold outputs (default: `outputs_core/inputs/hcp7t_hippocampus/hippunfold/dense_corobl_batch`).
  - --shared-root: Path to output shared processed surface files (default: `outputs_core/inputs/hcp7t_hippocampus/kong2022/shared_inputs/surface_151`).
  - --out-dir: Output directory for the QC report TSV (default: `outputs_core/inputs/hcp7t_hippocampus/kong2022/qc`).
  - --manifest-dir: Output directory for the generated subjects manifest (default: `outputs_core/inputs/hcp7t_hippocampus/kong2022/manifests`).
  - --subjects: Space-separated list of explicit subject IDs.
  - --dice-threshold: Dice overlap QC threshold (default: 0.70).
  - --tsnr-invalid-threshold: Maximum allowed fraction of invalid (low tSNR/sd) vertices (default: 0.20).
- Physical inputs:
  - Subject Dice TSV files: `{hippunfold_batch_root}/sub-<subject>/hippunfold/sub-<subject>/qc/sub-<subject>_hemi-{L|R}_desc-unetf3d_dice.tsv`.
  - Subject tSNR gate summaries: `{shared_root}/sub-<subject>/surface/tsnr_gate/sub-<subject>_hemi-{L|R}_tsnr_gate_summary.json`.

Outputs:
- Command-line output / file outputs:
  - QC summary TSV file: `{out_dir}/hcp_7t_hippocampus_fmri_qc_summary_t1only.tsv` detailing Dice scores and invalid tSNR fractions per subject.
  - QC-passed subjects list manifest: `{manifest_dir}/hcp_7t_hippocampus_fmri_qc_pass_t1only_151.txt` (the default 151 subjects cohort list).
  - Rolling pointer symlink/manifest: `{manifest_dir}/hcp_7t_hippocampus_fmri_qc_pass_t1only_latest.txt` (pointing to the 151 subjects list).
  - QC run traceability manifest: `{manifest_dir}/build_qc_manifest_run_manifest.json`.
"""
import argparse
import csv
import json
from pathlib import Path

from lib.output_layout import (
    DEFAULT_HIPPUNFOLD_BATCH_ROOT,
    DEFAULT_KONG2022_MANIFEST_ROOT,
    DEFAULT_KONG2022_QC_ROOT,
    DEFAULT_KONG2022_SURFACE_ROOT,
)
from lib.workflow_utils import file_meta, git_commit, utc_now_iso, write_json


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_BATCH_ROOT = DEFAULT_HIPPUNFOLD_BATCH_ROOT
DEFAULT_SHARED_ROOT = DEFAULT_KONG2022_SURFACE_ROOT
DEFAULT_OUT_DIR = DEFAULT_KONG2022_QC_ROOT
DEFAULT_MANIFEST_DIR = DEFAULT_KONG2022_MANIFEST_ROOT


def load_subjects(batch_root: Path, explicit: list[str] | None) -> list[str]:
    if explicit:
        return sorted({s.replace("sub-", "") for s in explicit})
    return sorted(p.name.replace("sub-", "") for p in batch_root.glob("sub-*") if p.is_dir())


def read_dice(batch_root: Path, subject: str, hemi: str) -> tuple[float | None, Path]:
    path = (
        batch_root
        / f"sub-{subject}"
        / "hippunfold"
        / f"sub-{subject}"
        / "qc"
        / f"sub-{subject}_hemi-{hemi}_desc-unetf3d_dice.tsv"
    )
    if not path.exists():
        return None, path
    text = path.read_text(encoding="utf-8").strip().splitlines()[0].strip()
    return float(text), path


def read_tsnr_fraction(shared_root: Path, subject: str, hemi: str) -> tuple[float | None, Path]:
    path = shared_root / f"sub-{subject}" / "surface" / "tsnr_gate" / f"sub-{subject}_hemi-{hemi}_tsnr_gate_summary.json"
    if not path.exists():
        return None, path
    obj = json.loads(path.read_text(encoding="utf-8"))
    if "invalid_fraction" in obj:
        return float(obj["invalid_fraction"]), path
    if "ratio_low" in obj:
        return float(obj["ratio_low"]), path
    total = int(obj.get("n_vertices_total", obj.get("n_total", 0)))
    bad = int(obj.get("n_vertices_invalid", obj.get("n_vertices_low", obj.get("n_low", 0))))
    if total <= 0:
        return None, path
    return float(bad / total), path


def write_tsv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "subject",
        "dice_L",
        "dice_R",
        "hipp_tsnr_invalid_fraction_L",
        "hipp_tsnr_invalid_fraction_R",
        "pass_dice",
        "pass_tsnr",
        "pass_all",
        "exclude_reason",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description="Build the T1only HippUnfold+tSNR QC pass manifest.")
    parser.add_argument("--hippunfold-batch-root", default=str(DEFAULT_BATCH_ROOT))
    parser.add_argument("--shared-root", default=str(DEFAULT_SHARED_ROOT))
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    parser.add_argument("--manifest-dir", default=str(DEFAULT_MANIFEST_DIR))
    parser.add_argument("--subjects", nargs="*", default=None)
    parser.add_argument("--dice-threshold", type=float, default=0.70)
    parser.add_argument("--tsnr-invalid-threshold", type=float, default=0.20)
    args = parser.parse_args()

    batch_root = Path(args.hippunfold_batch_root).resolve()
    shared_root = Path(args.shared_root).resolve()
    out_dir = Path(args.out_dir).resolve()
    manifest_dir = Path(args.manifest_dir).resolve()
    subjects = load_subjects(batch_root, args.subjects)

    rows: list[dict[str, object]] = []
    pass_subjects: list[str] = []
    inputs: list[dict[str, object]] = []

    for subject in subjects:
        dice: dict[str, float | None] = {}
        tsnr: dict[str, float | None] = {}
        missing: list[str] = []
        for hemi in ("L", "R"):
            val, path = read_dice(batch_root, subject, hemi)
            dice[hemi] = val
            if path.exists():
                inputs.append(file_meta(path))
            else:
                missing.append(f"missing_dice_{hemi}")
            frac, tpath = read_tsnr_fraction(shared_root, subject, hemi)
            tsnr[hemi] = frac
            if tpath.exists():
                inputs.append(file_meta(tpath))
            else:
                missing.append(f"missing_tsnr_{hemi}")

        pass_dice = all(dice[h] is not None and float(dice[h]) >= float(args.dice_threshold) for h in ("L", "R"))
        pass_tsnr = all(
            tsnr[h] is not None and float(tsnr[h]) < float(args.tsnr_invalid_threshold) for h in ("L", "R")
        )
        reasons = list(missing)
        if not pass_dice:
            reasons.append("dice_lt_threshold")
        if not pass_tsnr:
            reasons.append("hipp_tsnr_invalid_fraction_ge_threshold")
        pass_all = bool(pass_dice and pass_tsnr)
        if pass_all:
            pass_subjects.append(subject)
        rows.append(
            {
                "subject": f"sub-{subject}",
                "dice_L": "" if dice["L"] is None else f"{float(dice['L']):.8f}",
                "dice_R": "" if dice["R"] is None else f"{float(dice['R']):.8f}",
                "hipp_tsnr_invalid_fraction_L": "" if tsnr["L"] is None else f"{float(tsnr['L']):.8f}",
                "hipp_tsnr_invalid_fraction_R": "" if tsnr["R"] is None else f"{float(tsnr['R']):.8f}",
                "pass_dice": pass_dice,
                "pass_tsnr": pass_tsnr,
                "pass_all": pass_all,
                "exclude_reason": ",".join(sorted(set(reasons))),
            }
        )

    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_dir.mkdir(parents=True, exist_ok=True)
    qc_tsv = out_dir / "t1only_hippunfold_tsnr_qc.tsv"
    qc_json = out_dir / "t1only_hippunfold_tsnr_qc.json"
    manifest = manifest_dir / f"hcp_7t_hippocampus_fmri_qc_pass_t1only_{len(pass_subjects)}.txt"
    latest_manifest = manifest_dir / "hcp_7t_hippocampus_fmri_qc_pass_t1only_latest.txt"

    write_tsv(qc_tsv, rows)
    manifest.write_text("\n".join(pass_subjects) + ("\n" if pass_subjects else ""), encoding="utf-8")
    latest_manifest.write_text(manifest.read_text(encoding="utf-8"), encoding="utf-8")
    payload = {
        "stage": "t1only_qc_manifest",
        "status": "done",
        "timestamp_utc": utc_now_iso(),
        "git_commit": git_commit(),
        "hippunfold_batch_root": str(batch_root),
        "shared_root": str(shared_root),
        "dice_threshold": float(args.dice_threshold),
        "tsnr_invalid_threshold": float(args.tsnr_invalid_threshold),
        "n_subjects_total": int(len(subjects)),
        "n_subjects_pass": int(len(pass_subjects)),
        "manifest": str(manifest),
        "latest_manifest": str(latest_manifest),
        "qc_tsv": str(qc_tsv),
        "rows": rows,
        "inputs": inputs,
        "outputs": [str(qc_tsv), str(qc_json), str(manifest), str(latest_manifest)],
    }
    write_json(qc_json, payload)
    print(json.dumps({"n_total": len(subjects), "n_pass": len(pass_subjects), "manifest": str(manifest)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
