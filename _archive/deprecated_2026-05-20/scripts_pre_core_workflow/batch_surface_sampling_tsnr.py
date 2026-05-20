#!/usr/bin/env python3
"""
Batch hippocampal surface sampling + tSNR<25 statistics.

For each subject:
  1. Run sample_hipp_surface_timeseries.py  (wb_command -volume-to-surface-mapping -trilinear)
     Input  : rsfmri_volume/<id>/func_uncompressed.nii
     Surface: dense_corobl_batch/sub-<id>/hippunfold  (den-512, space-corobl)
     Output : hipp_group_inputs_shared/sub-<id>/surface/raw/*_label-hipp_bold.func.gii  (L + R)
  2. Compute tSNR = 10000 / std(t) per vertex, record ratio < 25.

Concurrency: capped at 10 (hard limit per CLAUDE.md / AGENTS.md).
"""
from __future__ import annotations

import argparse
import concurrent.futures
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import nibabel as nib
import numpy as np

# ── paths ────────────────────────────────────────────────────────────────────
REPO_ROOT      = Path(__file__).resolve().parents[1]
BOLD_ROOT      = Path("/home/yojiao/LSS/jianglab/3_Data_Working/HCP_7T_Hippocampus/rsfmri_volume")
HIPP_BATCH_DIR = REPO_ROOT / "outputs_migration" / "dense_corobl_batch"
SHARED_ROOT    = REPO_ROOT / "outputs_migration" / "hipp_group_inputs_shared"
LOG_DIR        = REPO_ROOT / "outputs_migration" / "hipp_group_inputs_shared_logs_sampling"
SAMPLE_SCRIPT  = REPO_ROOT / "scripts" / "common" / "sample_hipp_surface_timeseries.py"

DENSITY        = "512"
SPACE          = "corobl"
MAPPING_METHOD = "trilinear"
HCP_MODE       = 10000.0
TSNR_THRESHOLD = 25.0
MAX_CONCURRENCY = 10   # HARD CAP — do not raise above 10 (CLAUDE.md)

# ── helpers ──────────────────────────────────────────────────────────────────

def all_subjects() -> list[str]:
    """Return sorted list of subject IDs present in both bold root and hipp batch."""
    bold_ids  = {p.name for p in BOLD_ROOT.iterdir() if p.is_dir()}
    hipp_ids  = {p.name.removeprefix("sub-") for p in HIPP_BATCH_DIR.iterdir()
                 if p.is_dir() and p.name.startswith("sub-")}
    return sorted(bold_ids & hipp_ids)


def subject_raw_surface_dir(subject: str) -> Path:
    """Canonical path: hipp_group_inputs_shared/sub-{id}/surface/raw."""
    return SHARED_ROOT / f"sub-{subject}" / "surface" / "raw"


def output_done(subject: str) -> bool:
    """True if both hemi func.gii + summary json already exist (skip/resume)."""
    out = subject_raw_surface_dir(subject)
    for hemi in ("L", "R"):
        gii = out / f"sub-{subject}_hemi-{hemi}_space-{SPACE}_den-{DENSITY}_label-hipp_bold.func.gii"
        if not gii.exists():
            return False
    return (out / "surface_sampling_summary.json").exists()


def run_sampling(subject: str) -> dict:
    """Run sample_hipp_surface_timeseries.py for one subject. Returns result dict."""
    import subprocess
    bold       = BOLD_ROOT / subject / "func_uncompressed.nii"
    hipp_dir   = HIPP_BATCH_DIR / f"sub-{subject}" / "hippunfold"
    out_dir    = subject_raw_surface_dir(subject)
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path   = LOG_DIR / f"sub-{subject}.log"

    cmd = [
        sys.executable, str(SAMPLE_SCRIPT),
        "--bold",            str(bold),
        "--hippunfold-dir",  str(hipp_dir),
        "--subject",         subject,
        "--density",         DENSITY,
        "--space",           SPACE,
        "--mapping-method",  MAPPING_METHOD,
        "--smooth-iters",    "0",
        "--outdir",          str(out_dir),
    ]

    t0 = time.monotonic()
    proc = subprocess.run(
        cmd,
        text=True, capture_output=True,
        cwd=str(SAMPLE_SCRIPT.parent),  # so hipp_density_assets import works
    )
    elapsed = time.monotonic() - t0
    log_path.write_text(proc.stdout + proc.stderr, encoding="utf-8")

    if proc.returncode != 0:
        return {"subject": subject, "status": "failed", "rc": proc.returncode,
                "elapsed": elapsed, "error": proc.stderr[-500:]}

    return {"subject": subject, "status": "ok", "rc": 0, "elapsed": elapsed}


def compute_tsnr_stats(subject: str) -> dict:
    """Load L+R func.gii, compute tSNR, return per-hemi stats dict."""
    out_dir = subject_raw_surface_dir(subject)
    result  = {"subject": subject}
    for hemi in ("L", "R"):
        gii_path = out_dir / f"sub-{subject}_hemi-{hemi}_space-{SPACE}_den-{DENSITY}_label-hipp_bold.func.gii"
        img  = nib.load(str(gii_path))
        data = np.stack([d.data for d in img.darrays], axis=1).astype(np.float64)  # (V, T)
        sd   = data.std(axis=1, ddof=1)
        tsnr = np.where(sd > 0, HCP_MODE / sd, np.nan)
        n_total   = int(np.isfinite(tsnr).sum())
        n_low     = int((tsnr < TSNR_THRESHOLD).sum())
        ratio_low = float(n_low / n_total) if n_total > 0 else float("nan")
        result[hemi] = {
            "n_vertices_total": n_total,
            "n_vertices_low":   n_low,
            "ratio_low":        ratio_low,
            "tsnr_mean":        float(np.nanmean(tsnr)),
            "tsnr_median":      float(np.nanmedian(tsnr)),
        }
    return result


# ── main ─────────────────────────────────────────────────────────────────────

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--concurrency", type=int, default=MAX_CONCURRENCY,
                        help=f"Parallel workers (default {MAX_CONCURRENCY}, hard cap {MAX_CONCURRENCY})")
    parser.add_argument("--tsnr-only", action="store_true",
                        help="Skip sampling; only recompute tSNR stats from existing func.gii files")
    parser.add_argument("--out-json", default=str(REPO_ROOT / "outputs_migration" / "hipp_tsnr_lt25_surface.json"))
    args = parser.parse_args()

    concurrency = min(args.concurrency, MAX_CONCURRENCY)
    if args.concurrency > MAX_CONCURRENCY:
        print(f"[warn] concurrency capped at {MAX_CONCURRENCY} (requested {args.concurrency})", flush=True)

    SHARED_ROOT.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    subjects = all_subjects()
    print(f"[info] {len(subjects)} subjects found", flush=True)

    # ── surface sampling ────────────────────────────────────────────────────
    if not args.tsnr_only:
        todo    = [s for s in subjects if not output_done(s)]
        already = len(subjects) - len(todo)
        print(f"[info] sampling: {already} already done, {len(todo)} to run  concurrency={concurrency}", flush=True)

        ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        manager_log = LOG_DIR / f"manager_{ts}_conc{concurrency}.log"

        failures: list[str] = []
        completed = 0
        with manager_log.open("w", encoding="utf-8") as mlog:
            def logline(msg: str) -> None:
                line = f"[{datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}] {msg}"
                print(line, flush=True)
                mlog.write(line + "\n")
                mlog.flush()

            logline(f"manager start concurrency={concurrency} todo={len(todo)}")
            with concurrent.futures.ProcessPoolExecutor(max_workers=concurrency) as pool:
                futs = {pool.submit(run_sampling, s): s for s in todo}
                for fut in concurrent.futures.as_completed(futs):
                    r = fut.result()
                    completed += 1
                    if r["status"] == "ok":
                        logline(f"done sub-{r['subject']} elapsed={r['elapsed']:.1f}s ({completed}/{len(todo)})")
                    else:
                        logline(f"FAILED sub-{r['subject']} rc={r['rc']}")
                        failures.append(r["subject"])

            logline(f"manager finished failures={len(failures)}")
            if failures:
                logline("failed subjects: " + " ".join(failures))

        if failures:
            print(f"[error] {len(failures)} subjects failed — see {manager_log}", flush=True)
            return 1

    # ── tSNR statistics ─────────────────────────────────────────────────────
    print("[info] computing tSNR statistics …", flush=True)
    stats_subjects = [s for s in subjects if output_done(s)]
    print(f"[info] {len(stats_subjects)} subjects with complete func.gii output", flush=True)

    stats: list[dict] = []
    for s in stats_subjects:
        try:
            stats.append(compute_tsnr_stats(s))
        except Exception as exc:
            print(f"[warn] tSNR failed for {s}: {exc}", flush=True)

    out_json = Path(args.out_json)
    out_json.write_text(
        json.dumps(
            {
                "surface_sampling_root": str(SHARED_ROOT),
                "density":  DENSITY,
                "space":    SPACE,
                "mapping_method": MAPPING_METHOD,
                "threshold": TSNR_THRESHOLD,
                "tsnr_definition": "10000/std",
                "n_subjects": len(stats),
                "subjects": stats,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"[info] tSNR stats saved → {out_json}", flush=True)

    # ── summary printout ────────────────────────────────────────────────────
    all_ratios_L = [s["L"]["ratio_low"] for s in stats if "L" in s and np.isfinite(s["L"]["ratio_low"])]
    all_ratios_R = [s["R"]["ratio_low"] for s in stats if "R" in s and np.isfinite(s["R"]["ratio_low"])]
    print(f"\n=== tSNR < {TSNR_THRESHOLD} summary ({len(stats)} subjects) ===")
    if all_ratios_L:
        print(f"  L hemi  mean={np.mean(all_ratios_L)*100:.2f}%  median={np.median(all_ratios_L)*100:.2f}%")
    if all_ratios_R:
        print(f"  R hemi  mean={np.mean(all_ratios_R)*100:.2f}%  median={np.median(all_ratios_R)*100:.2f}%")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
