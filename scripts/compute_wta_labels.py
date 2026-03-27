#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def zscore_rows(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float32, copy=False)
    mean = np.nanmean(x, axis=1, keepdims=True)
    centered = x - mean
    centered[~np.isfinite(centered)] = 0.0
    std = np.nanstd(x, axis=1, keepdims=True)
    z = centered / np.clip(std, 1e-12, None)
    z[~np.isfinite(z)] = 0.0
    return z


def main() -> int:
    parser = argparse.ArgumentParser(description="Compute winner-takes-all labels from hippocampal surface timeseries")
    parser.add_argument("--hipp-ts", required=True, help="n_vertices x n_timepoints .npy")
    parser.add_argument("--network-ts", required=True, help="n_networks x n_timepoints .npy")
    parser.add_argument("--outdir", required=True)
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    hipp = np.load(args.hipp_ts).astype(np.float32)
    nets = np.load(args.network_ts).astype(np.float32)
    hipp_z = zscore_rows(hipp)
    nets_z = zscore_rows(nets)
    corr = (hipp_z @ nets_z.T) / hipp.shape[1]

    order = np.argsort(corr, axis=1)
    best = order[:, -1]
    second = order[:, -2] if corr.shape[1] > 1 else order[:, -1]
    labels = best + 1
    confidence = corr[np.arange(corr.shape[0]), best] - corr[np.arange(corr.shape[0]), second]

    np.save(outdir / "hipp_wta_labels.npy", labels.astype(np.int16))
    np.save(outdir / "hipp_wta_confidence.npy", confidence.astype(np.float32))
    np.save(outdir / "hipp_to_network_correlations.npy", corr.astype(np.float32))
    print(json.dumps({"n_vertices": int(hipp.shape[0]), "n_networks": int(nets.shape[0])}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
