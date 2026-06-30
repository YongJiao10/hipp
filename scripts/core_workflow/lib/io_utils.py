from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import numpy as np


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_tsv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f, delimiter="\t"))


def write_tsv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter="\t", extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def normalize_subject_id(raw: str) -> str:
    return str(raw).strip().replace("sub-", "")


def load_subject_ids(path: Path, *, require_nonempty: bool = True) -> list[str]:
    out: list[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        s = normalize_subject_id(line)
        if s:
            out.append(s)
    if require_nonempty and not out:
        raise RuntimeError(f"No subjects in manifest: {path}")
    return out


def compute_cortex_tsnr_gate(
    *,
    data: np.ndarray,
    outdir: Path,
    hemi: str,
    threshold: float,
    hcp_mode: float = 10000.0,
) -> Path:
    sd = np.nanstd(np.asarray(data, dtype=np.float64), axis=1, ddof=1)
    tsnr = np.where(sd > 0, hcp_mode / sd, np.nan).astype(np.float32)
    valid = np.isfinite(tsnr) & (tsnr >= float(threshold))
    outdir.mkdir(parents=True, exist_ok=True)
    np.save(outdir / f"cortex_{hemi.lower()}_tsnr.npy", tsnr)
    mask_path = outdir / f"cortex_{hemi.lower()}_valid_mask.npy"
    np.save(mask_path, valid.astype(bool))
    
    # Write json directly
    gate_summary = {
        "hemi": hemi,
        "threshold": float(threshold),
        "tsnr_definition": f"{int(hcp_mode)}/std",
        "n_grayordinates_total": int(valid.size),
        "n_grayordinates_valid": int(valid.sum()),
        "n_grayordinates_invalid": int(valid.size - valid.sum()),
        "invalid_fraction": float(1.0 - (valid.sum() / max(1, valid.size))),
    }
    summary_path = outdir / f"cortex_{hemi.lower()}_tsnr_gate_summary.json"
    summary_path.write_text(json.dumps(gate_summary, indent=2) + "\n", encoding="utf-8")
    return mask_path


def parse_run_segments(
    run_lengths: list[int] | None,
    expected_total_tp: int,
    run_ids: list[int] = (1, 2, 3, 4),
) -> list[dict[str, object]]:
    if run_lengths is None:
        n_runs = len(run_ids)
        base = int(expected_total_tp) // n_runs
        remainder = int(expected_total_tp) % n_runs
        # Distribute remainder across the first `remainder` runs (one extra tp each)
        run_lengths = [base + (1 if i < remainder else 0) for i in range(n_runs)]
    if len(run_lengths) != len(run_ids):
        raise ValueError(f"Expected {len(run_ids)} run lengths, got {run_lengths}")
    segments: list[dict[str, object]] = []
    offset = 0
    for rid, run_len in zip(run_ids, run_lengths, strict=True):
        run_len = int(run_len)
        if run_len <= 0:
            raise ValueError(f"Invalid run length for run-{rid}: {run_len}")
        run_start = int(offset)
        run_stop = int(offset + run_len)
        segments.append(
            {
                "run_id": int(rid),
                "run_label": f"run-{rid}",
                "run_start": int(run_start),
                "run_stop": int(run_stop),
                "run_length": int(run_len),
            }
        )
        offset = run_stop
    if offset != int(expected_total_tp):
        raise RuntimeError(
            f"Run lengths do not sum to concat length: sum={offset} concat={expected_total_tp}"
        )
    return segments

