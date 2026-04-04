#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import nibabel as nib
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import eigsh


def zscore_rows(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float32, copy=False)
    mean = np.nanmean(x, axis=1, keepdims=True)
    centered = x - mean
    centered[~np.isfinite(centered)] = 0.0
    std = np.nanstd(x, axis=1, keepdims=True)
    z = centered / np.clip(std, 1e-12, None)
    z[~np.isfinite(z)] = 0.0
    return z


def corrcoef_rows(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a_z = zscore_rows(a)
    b_z = zscore_rows(b)
    return (a_z @ b_z.T) / max(1, a.shape[1])


def build_sparse_affinity(features: np.ndarray, sparsity: float) -> sparse.csr_matrix:
    unit = features.astype(np.float32, copy=False)
    norms = np.linalg.norm(unit, axis=1, keepdims=True)
    unit = unit / np.clip(norms, 1e-12, None)
    sim = unit @ unit.T
    sim = np.clip(sim, -1.0, 1.0)
    aff = (sim + 1.0) * 0.5
    np.fill_diagonal(aff, 1.0)

    n_vertices = aff.shape[0]
    n_keep = max(10, int(round((n_vertices - 1) * sparsity)))
    n_keep = min(n_vertices, n_keep + 1)
    keep_idx = np.argpartition(aff, kth=n_vertices - n_keep, axis=1)[:, -n_keep:]
    rows = np.repeat(np.arange(n_vertices), n_keep)
    cols = keep_idx.reshape(-1)
    vals = aff[rows, cols]

    graph = sparse.csr_matrix((vals, (rows, cols)), shape=(n_vertices, n_vertices), dtype=np.float32)
    graph = graph.maximum(graph.T)
    graph.setdiag(1.0)
    graph.eliminate_zeros()
    return graph


def diffusion_map_embedding(affinity: sparse.csr_matrix, n_components: int) -> tuple[np.ndarray, np.ndarray]:
    degree = np.asarray(affinity.sum(axis=1)).ravel().astype(np.float32)
    inv_sqrt = 1.0 / np.sqrt(np.clip(degree, 1e-12, None))
    d_inv = sparse.diags(inv_sqrt)
    sym = d_inv @ affinity @ d_inv

    n_eigs = min(sym.shape[0] - 1, n_components + 1)
    if n_eigs < 2:
        raise RuntimeError("Not enough vertices to compute diffusion-map gradients")

    eigvals, eigvecs = eigsh(sym.astype(np.float32), k=n_eigs, which="LA")
    order = np.argsort(eigvals)[::-1]
    eigvals = eigvals[order]
    eigvecs = eigvecs[:, order]

    gradients = eigvecs[:, 1 : n_components + 1] * eigvals[1 : n_components + 1]
    return gradients.astype(np.float32), eigvals.astype(np.float32)


def orient_gradients(gradients: np.ndarray, surface_path: Path | None) -> np.ndarray:
    out = gradients.copy()
    if out.size == 0:
        return out

    if surface_path is not None:
        coords = nib.load(str(surface_path)).agg_data("pointset").astype(np.float32)
        coords = coords - coords.mean(axis=0, keepdims=True)
        _u, _s, vt = np.linalg.svd(coords, full_matrices=False)
        geom_axis = coords @ vt[0].astype(np.float32)
        corr = np.corrcoef(out[:, 0], geom_axis)[0, 1]
        if np.isfinite(corr) and corr < 0:
            out[:, 0] *= -1.0

    for idx in range(out.shape[1]):
        arr = out[:, idx]
        if abs(float(arr.min())) > abs(float(arr.max())):
            out[:, idx] *= -1.0
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description="Compute hippocampal FC gradients from vertex-to-parcel connectivity")
    parser.add_argument("--hipp-ts", required=True, help="n_vertices x n_timepoints .npy")
    parser.add_argument("--parcel-ts", required=True, help="n_parcels x n_timepoints .npy")
    parser.add_argument("--surface", default=None, help="optional surface for deterministic sign orientation")
    parser.add_argument("--n-components", type=int, default=3)
    parser.add_argument("--sparsity", type=float, default=0.1)
    parser.add_argument("--outdir", required=True)
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    hipp = np.load(args.hipp_ts).astype(np.float32)
    parcels = np.load(args.parcel_ts).astype(np.float32)
    if hipp.ndim != 2 or parcels.ndim != 2:
        raise ValueError("Both hippocampal and parcel timeseries must be 2D arrays")
    if hipp.shape[1] != parcels.shape[1]:
        raise ValueError(
            f"Timepoints mismatch: hipp={hipp.shape[1]} vs parcels={parcels.shape[1]}"
        )

    fc = corrcoef_rows(hipp, parcels)
    affinity = build_sparse_affinity(fc, sparsity=float(args.sparsity))
    gradients, eigvals = diffusion_map_embedding(affinity, n_components=int(args.n_components))
    gradients = orient_gradients(gradients, Path(args.surface) if args.surface else None)

    np.save(outdir / "hipp_vertex_to_parcel_fc.npy", fc.astype(np.float32))
    np.save(outdir / "hipp_fc_gradients.npy", gradients.astype(np.float32))
    np.save(outdir / "hipp_fc_gradient_eigenvalues.npy", eigvals.astype(np.float32))
    for idx in range(gradients.shape[1]):
        np.save(outdir / f"hipp_fc_gradient{idx + 1}.npy", gradients[:, idx].astype(np.float32))

    summary = {
        "n_vertices": int(hipp.shape[0]),
        "n_parcels": int(parcels.shape[0]),
        "n_timepoints": int(hipp.shape[1]),
        "n_components": int(gradients.shape[1]),
        "sparsity": float(args.sparsity),
        "surface": str(args.surface) if args.surface else None,
    }
    (outdir / "gradient_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
