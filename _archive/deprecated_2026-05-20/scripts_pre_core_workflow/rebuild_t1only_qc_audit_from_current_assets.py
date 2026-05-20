#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
EXP_ROOT = REPO_ROOT / "outputs_experiments" / "hippunfold_recommended_t1w_20260423"
BATCH_ROOT = EXP_ROOT / "dense_corobl_batch"
KONG2022_ROOT = EXP_ROOT / "kong2022"
SURFACE_166 = EXP_ROOT / "vertex_qc_t1only" / "shared_inputs_166"
SOURCE_PASS_151 = KONG2022_ROOT / "manifests" / "hcp_7t_hippocampus_fmri_qc_pass_t1only_151.txt"
VERTEX_QC_ROOT = EXP_ROOT / "vertex_qc_t1only"
MANIFEST_DIR = VERTEX_QC_ROOT / "manifests"
QC_DIR = VERTEX_QC_ROOT / "qc"

OUT_TSV = QC_DIR / "hcp7t_t1only_vertex_qc_166_audit.tsv"
OUT_JSON = QC_DIR / "hcp7t_t1only_vertex_qc_166_audit.json"
OUT_PASS_151 = MANIFEST_DIR / "hcp_7t_hippocampus_fmri_qc_pass_t1only_151.txt"
OUT_PASS_LATEST = MANIFEST_DIR / "hcp_7t_hippocampus_fmri_qc_pass_t1only_latest.txt"
OUT_EXCL = MANIFEST_DIR / "hcp_7t_hippocampus_fmri_qc_excluded_t1only_15.txt"
OUT_EXCL_LATEST = MANIFEST_DIR / "hcp_7t_hippocampus_fmri_qc_excluded_t1only_latest.txt"

EXCLUDED_15 = [
    "sub-102816",
    "sub-109123",
    "sub-155938",
    "sub-158136",
    "sub-164131",
    "sub-173334",
    "sub-182739",
    "sub-198653",
    "sub-550439",
    "sub-680957",
    "sub-783462",
    "sub-878776",
    "sub-878877",
    "sub-898176",
    "sub-901442",
]


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def git_commit() -> str | None:
    proc = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=REPO_ROOT, text=True, capture_output=True)
    return proc.stdout.strip() if proc.returncode == 0 else None


def read_pass_set(path: Path) -> set[str]:
    ids = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        s = line.strip()
        if not s:
            continue
        ids.add(s if s.startswith("sub-") else f"sub-{s}")
    return ids


def read_dice(subject: str, hemi: str) -> float | None:
    p = BATCH_ROOT / subject / "hippunfold" / subject / "qc" / f"{subject}_hemi-{hemi}_desc-unetf3d_dice.tsv"
    if not p.exists():
        return None
    return float(p.read_text(encoding="utf-8").strip().splitlines()[0].strip())


def read_tsnr_invalid_fraction(subject: str, hemi: str) -> float | None:
    p = SURFACE_166 / subject / "surface" / "tsnr_gate" / f"{subject}_hemi-{hemi}_tsnr_gate_summary.json"
    if not p.exists():
        return None
    obj = json.loads(p.read_text(encoding="utf-8"))
    if "invalid_fraction" in obj:
        return float(obj["invalid_fraction"])
    if "ratio_low" in obj:
        return float(obj["ratio_low"])
    return None


def fmt(v: float | None) -> str:
    return "" if v is None else f"{float(v):.8f}"


def main() -> int:
    QC_DIR.mkdir(parents=True, exist_ok=True)
    MANIFEST_DIR.mkdir(parents=True, exist_ok=True)

    pass_set = read_pass_set(SOURCE_PASS_151)
    excluded_set = set(EXCLUDED_15)
    all_subjects = sorted(p.name for p in BATCH_ROOT.glob("sub-*") if p.is_dir())

    rows: list[dict[str, object]] = []
    n_tsnr_missing = 0
    n_mismatch = 0
    n_dice_fail = 0
    n_tsnr_fail_with_evidence = 0

    for subject in all_subjects:
        dice_l = read_dice(subject, "L")
        dice_r = read_dice(subject, "R")
        tsnr_l = read_tsnr_invalid_fraction(subject, "L")
        tsnr_r = read_tsnr_invalid_fraction(subject, "R")

        pass_dice = bool(
            dice_l is not None and dice_r is not None and float(dice_l) >= 0.70 and float(dice_r) >= 0.70
        )
        pass_tsnr_derived = bool(
            tsnr_l is not None and tsnr_r is not None and float(tsnr_l) < 0.20 and float(tsnr_r) < 0.20
        )
        tsnr_evidence_available = bool(tsnr_l is not None and tsnr_r is not None)
        in_pass_151_manifest = subject in pass_set
        in_pinned_excluded15 = subject in excluded_set

        if not tsnr_evidence_available:
            n_tsnr_missing += 1
        if not pass_dice:
            n_dice_fail += 1
        if tsnr_evidence_available and not pass_tsnr_derived:
            n_tsnr_fail_with_evidence += 1

        reasons: list[str] = []
        if not pass_dice:
            reasons.append("dice_lt_0.70")
        if not tsnr_evidence_available:
            reasons.append("tsnr_summary_missing_in_surface_151")
        elif not pass_tsnr_derived:
            reasons.append("hipp_tsnr_invalid_fraction_ge_0.20")
        if in_pinned_excluded15:
            reasons.append("in_pinned_excluded15")

        if tsnr_evidence_available:
            manifest_vs_derived = "match" if (in_pass_151_manifest == (pass_dice and pass_tsnr_derived)) else "mismatch"
            if manifest_vs_derived == "mismatch":
                n_mismatch += 1
        else:
            manifest_vs_derived = "insufficient_tsnr_evidence"

        rows.append(
            {
                "subject": subject,
                "dice_L": fmt(dice_l),
                "dice_R": fmt(dice_r),
                "hipp_tsnr_invalid_fraction_L": fmt(tsnr_l),
                "hipp_tsnr_invalid_fraction_R": fmt(tsnr_r),
                "pass_dice_derived": pass_dice,
                "pass_tsnr_derived": pass_tsnr_derived if tsnr_evidence_available else "",
                "tsnr_evidence_available": tsnr_evidence_available,
                "in_pass_151_manifest": in_pass_151_manifest,
                "in_pinned_excluded15": in_pinned_excluded15,
                "manifest_vs_derived": manifest_vs_derived,
                "exclude_reason_trace": ",".join(reasons),
            }
        )

    fields = [
        "subject",
        "dice_L",
        "dice_R",
        "hipp_tsnr_invalid_fraction_L",
        "hipp_tsnr_invalid_fraction_R",
        "pass_dice_derived",
        "pass_tsnr_derived",
        "tsnr_evidence_available",
        "in_pass_151_manifest",
        "in_pinned_excluded15",
        "manifest_vs_derived",
        "exclude_reason_trace",
    ]
    with OUT_TSV.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)

    pass_lines = [s.replace("sub-", "") for s in sorted(pass_set)]
    pass_txt = "\n".join(pass_lines) + ("\n" if pass_lines else "")
    OUT_PASS_151.write_text(pass_txt, encoding="utf-8")
    OUT_PASS_LATEST.write_text(pass_txt, encoding="utf-8")

    bare_excluded = [s.replace("sub-", "") for s in EXCLUDED_15]
    excl_txt = "\n".join(bare_excluded) + "\n"
    OUT_EXCL.write_text(excl_txt, encoding="utf-8")
    OUT_EXCL_LATEST.write_text(excl_txt, encoding="utf-8")

    payload = {
        "stage": "rebuild_hcp7t_t1only_vertex_qc_audit_from_current_assets",
        "status": "done",
        "timestamp_utc": now_utc(),
        "git_commit": git_commit(),
        "inputs": {
            "experiment_root": str(EXP_ROOT),
            "batch_root": str(BATCH_ROOT),
            "surface_166": str(SURFACE_166),
            "pass_151_manifest_source": str(SOURCE_PASS_151),
            "pinned_excluded15": EXCLUDED_15,
        },
        "outputs": {
            "vertex_qc_root": str(VERTEX_QC_ROOT),
            "qc_tsv": str(OUT_TSV),
            "qc_json": str(OUT_JSON),
            "pass151_manifest": str(OUT_PASS_151),
            "pass151_latest": str(OUT_PASS_LATEST),
            "excluded15_manifest": str(OUT_EXCL),
            "excluded15_latest": str(OUT_EXCL_LATEST),
        },
        "counts": {
            "n_subjects_total": len(all_subjects),
            "n_in_pass_151_manifest": len(pass_set),
            "n_in_pinned_excluded15": len(excluded_set),
            "n_tsnr_evidence_missing": n_tsnr_missing,
            "n_dice_fail_derived": n_dice_fail,
            "n_tsnr_fail_with_evidence_derived": n_tsnr_fail_with_evidence,
            "n_manifest_vs_derived_mismatch_on_evidence": n_mismatch,
        },
        "notes": [
            "This rebuild uses current on-disk assets only.",
            "For subjects outside surface_151, tSNR evidence may be missing.",
            "Pass-151 membership is sourced from current kong2022 manifest, then exported to vertex_qc_t1only manifests.",
        ],
    }
    OUT_JSON.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload["counts"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
