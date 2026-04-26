#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SHARED_ROOT = REPO_ROOT / "outputs_migration" / "hipp_group_inputs_shared"
DEFAULT_SUBJECTS = REPO_ROOT / "manifests" / "hcp_7t_hippocampus_fmri_qc_pass_153.txt"
DEFAULT_CROSS_ATLAS_MERGE = REPO_ROOT / "config" / "cross_atlas_network_merge.json"

SMOOTHINGS = ["2mm", "4mm"]
HEMIS = ["L", "R"]
RUN_IDS = [1, 2, 3, 4]
KONG7 = "kong2019"
KONG8 = "kong2019_8net"


def load_subjects(path: Path) -> list[str]:
    subjects: list[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        s = line.strip()
        if s:
            subjects.append(s.replace("sub-", ""))
    if not subjects:
        raise RuntimeError(f"No subjects found in {path}")
    return subjects


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def save_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f, delimiter="\t"))


def write_networks_tsv(path: Path, names: list[str], counts: list[int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["index", "canonical_network", "n_parent_networks"], delimiter="\t")
        writer.writeheader()
        for i, (name, count) in enumerate(zip(names, counts, strict=True), start=1):
            writer.writerow({"index": i, "canonical_network": name, "n_parent_networks": int(count)})


def corrcoef_rows(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    x = np.asarray(a, dtype=np.float32)
    y = np.asarray(b, dtype=np.float32)
    x_mu = np.nanmean(x, axis=1, keepdims=True)
    y_mu = np.nanmean(y, axis=1, keepdims=True)
    x_sd = np.nanstd(x, axis=1, keepdims=True, ddof=1)
    y_sd = np.nanstd(y, axis=1, keepdims=True, ddof=1)
    x_z = (x - x_mu) / np.clip(x_sd, 1e-12, None)
    y_z = (y - y_mu) / np.clip(y_sd, 1e-12, None)
    x_z[~np.isfinite(x_z)] = 0.0
    y_z[~np.isfinite(y_z)] = 0.0
    corr = (x_z @ y_z.T) / float(max(1, x.shape[1] - 1))
    corr = np.clip(corr, -0.999999, 0.999999)
    corr[~np.isfinite(corr)] = 0.0
    return corr.astype(np.float32)


def one_sided_network_partial_corr_rows(rows: np.ndarray, network_ts: np.ndarray) -> np.ndarray:
    x = np.asarray(rows, dtype=np.float32)
    y = np.asarray(network_ts, dtype=np.float32)
    if x.shape[1] != y.shape[1]:
        raise ValueError(f"Time dimension mismatch: {x.shape} vs {y.shape}")
    x_center = x - np.nanmean(x, axis=1, keepdims=True)
    y_center = y - np.nanmean(y, axis=1, keepdims=True)
    x_center[~np.isfinite(x_center)] = 0.0
    y_center[~np.isfinite(y_center)] = 0.0
    out = np.zeros((x.shape[0], y.shape[0]), dtype=np.float32)
    for j in range(y.shape[0]):
        if y.shape[0] <= 1:
            resid = x_center
        else:
            keep = [i for i in range(y.shape[0]) if i != j]
            controls = y_center[np.asarray(keep, dtype=np.int32), :]
            beta = (x_center @ controls.T) @ np.linalg.pinv(controls @ controls.T, rcond=1e-6)
            resid = x_center - beta @ controls
        out[:, j] = corrcoef_rows(resid, y[[j], :])[:, 0]
    return out.astype(np.float32)


def compact_active_vertices(full: np.ndarray, valid_mask: np.ndarray) -> np.ndarray:
    return np.asarray(full[valid_mask, :], dtype=np.float32)


def canonical_spec(atlas: str) -> tuple[list[str], dict[str, str]]:
    payload = load_json(DEFAULT_CROSS_ATLAS_MERGE)
    mapping = {str(k): str(v) for k, v in payload["atlases"][atlas]["mapping"].items()}
    order_all = [str(x) for x in payload["canonical_network_order"]]
    names = [name for name in order_all if name in set(mapping.values()) and name not in set(payload.get("exclude_labels", []))]
    return names, mapping


def build_canonical_from_parent(parent_tsv: Path, parent_ts: Path, atlas: str) -> tuple[list[str], np.ndarray, list[int]]:
    names, mapping = canonical_spec(atlas)
    rows = read_tsv(parent_tsv)
    ts = np.load(parent_ts).astype(np.float32)
    if ts.shape[0] != len(rows):
        raise RuntimeError(f"Parent network row mismatch: {parent_tsv} rows={len(rows)} ts={ts.shape}")
    parent_names = []
    for row in rows:
        parent_names.append(str(row.get("parent_network") or row.get("network") or row.get("name") or row.get("canonical_network")))
    out = np.zeros((len(names), ts.shape[1]), dtype=np.float32)
    counts: list[int] = []
    for i, name in enumerate(names):
        idx = [j for j, parent in enumerate(parent_names) if mapping.get(parent) == name]
        if not idx:
            raise RuntimeError(f"No parent networks matched {name} in {parent_tsv}")
        out[i, :] = np.nanmean(ts[np.asarray(idx, dtype=np.int32), :], axis=0).astype(np.float32)
        counts.append(len(idx))
    out[~np.isfinite(out)] = 0.0
    return names, out, counts


def materialize_reference(shared_root: Path, subject: str) -> None:
    src = shared_root / f"sub-{subject}" / "reference" / KONG7
    dst = shared_root / f"sub-{subject}" / "reference" / KONG8
    parent_tsv = src / "cortex_parent_networks.tsv"
    parent_ts = src / "cortex_parent_network_timeseries.npy"
    names, ts, counts = build_canonical_from_parent(parent_tsv, parent_ts, KONG8)
    dst.mkdir(parents=True, exist_ok=True)
    np.save(dst / "cortex_canonical_network_timeseries.npy", ts.astype(np.float32))
    write_networks_tsv(dst / "cortex_canonical_networks.tsv", names, counts)
    for name in [
        "cortex_parent_networks.tsv",
        "cortex_parent_network_timeseries.npy",
    ]:
        src_file = src / name
        if src_file.exists():
            shutil.copy2(src_file, dst / name)
    if (src / "tsnr_gate").exists() and not (dst / "tsnr_gate").exists():
        shutil.copytree(src / "tsnr_gate", dst / "tsnr_gate")
    save_json(
        dst / "reference_summary.json",
        {
            "subject": f"sub-{subject}",
            "atlas_slug": KONG8,
            "source_atlas_slug": KONG7,
            "definition": "Kong2019 8net canonical networks; TemporalParietal is independent from Default",
            "n_networks_used": int(len(names)),
            "networks": names,
            "n_parent_networks_per_canonical": {name: int(count) for name, count in zip(names, counts, strict=True)},
        },
    )
    save_json(dst / "stage_manifest.json", {"stage": "materialize_kong2019_8net_reference", "inputs": [str(parent_tsv), str(parent_ts)]})
    offset = 0
    for rid in RUN_IDS:
        run_src = src / "runs" / f"run-{rid}"
        run_dst = dst / "runs" / f"run-{rid}"
        run_len = int(np.load(run_src / "cortex_canonical_network_timeseries.npy", mmap_mode="r").shape[1])
        run_ts = ts[:, offset : offset + run_len].astype(np.float32)
        if run_ts.shape[1] != run_len:
            raise RuntimeError(f"Cannot slice Kong8 runwise reference for sub-{subject} run-{rid}: offset={offset} len={run_len}")
        offset += run_len
        run_dst.mkdir(parents=True, exist_ok=True)
        np.save(run_dst / "cortex_canonical_network_timeseries.npy", run_ts.astype(np.float32))
        save_json(
            run_dst / "reference_summary.json",
            {
                "subject": f"sub-{subject}",
                "atlas_slug": KONG8,
                "source_atlas_slug": KONG7,
                "run_id": int(rid),
                "n_networks_used": int(len(names)),
                "networks": names,
            },
        )
        save_json(run_dst / "stage_manifest.json", {"stage": "materialize_kong2019_8net_run_reference", "run_id": int(rid)})
    if offset != int(ts.shape[1]):
        raise RuntimeError(f"Run lengths do not sum to full Kong8 time series for sub-{subject}: {offset} vs {ts.shape[1]}")


def materialize_fc(shared_root: Path, subject: str, atlas: str, smoothing: str, hemi: str) -> None:
    fc_dir = shared_root / f"sub-{subject}" / "fc" / atlas / smoothing / f"hemi_{hemi}"
    fc_dir.mkdir(parents=True, exist_ok=True)
    ts = np.load(shared_root / f"sub-{subject}" / "surface" / smoothing / f"sub-{subject}_hemi-{hemi}_timeseries.npy").astype(np.float32)
    valid_mask = np.load(shared_root / f"sub-{subject}" / "surface" / "tsnr_gate" / f"sub-{subject}_hemi-{hemi}_valid_mask.npy").astype(bool)
    net = np.load(shared_root / f"sub-{subject}" / "reference" / atlas / "cortex_canonical_network_timeseries.npy").astype(np.float32)
    if ts.shape[1] != net.shape[1]:
        raise RuntimeError(f"Time mismatch for sub-{subject} {atlas} {smoothing} hemi-{hemi}: {ts.shape} vs {net.shape}")
    pearson_valid = corrcoef_rows(ts[valid_mask, :], net)
    pearson_full = np.full((ts.shape[0], net.shape[0]), np.nan, dtype=np.float32)
    pearson_full[valid_mask, :] = pearson_valid
    np.save(fc_dir / "hipp_vertex_to_network_fc.npy", pearson_full.astype(np.float32))
    np.save(fc_dir / "hipp_vertex_to_network_fc_active.npy", compact_active_vertices(pearson_full, valid_mask))
    partial_valid = one_sided_network_partial_corr_rows(ts[valid_mask, :], net)
    partial_full = np.full((ts.shape[0], net.shape[0]), np.nan, dtype=np.float32)
    partial_full[valid_mask, :] = partial_valid
    np.save(fc_dir / "hipp_vertex_to_network_partial_fc.npy", partial_full.astype(np.float32))
    np.save(fc_dir / "hipp_vertex_to_network_partial_fc_active.npy", compact_active_vertices(partial_full, valid_mask))
    save_json(
        fc_dir / "partial_fc_summary.json",
        {
            "subject": f"sub-{subject}",
            "atlas_slug": atlas,
            "smoothing": smoothing,
            "hemi": hemi,
            "partial_fc_definition": "one_sided_networks_only: residualize hippocampal vertex time series by other canonical network time series; correlate residual with raw target network",
            "target_network_residualized": False,
            "hippocampal_vertex_covariates": False,
            "n_vertices_total": int(ts.shape[0]),
            "n_vertices_valid_high_tsnr": int(valid_mask.sum()),
            "n_networks_used": int(net.shape[0]),
            "n_timepoints": int(ts.shape[1]),
        },
    )
    offset = 0
    for rid in RUN_IDS:
        run_net = np.load(
            shared_root / f"sub-{subject}" / "reference" / atlas / "runs" / f"run-{rid}" / "cortex_canonical_network_timeseries.npy"
        ).astype(np.float32)
        run_len = int(run_net.shape[1])
        next_offset = offset + run_len
        run_ts = ts[:, offset:next_offset].astype(np.float32, copy=False)
        if run_ts.shape[1] != run_len:
            raise RuntimeError(
                f"Runwise time mismatch for sub-{subject} {atlas} {smoothing} hemi-{hemi} run-{rid}: "
                f"slice_len={run_ts.shape[1]} expected={run_len}"
            )
        offset = next_offset
        if run_ts.shape[1] != run_net.shape[1]:
            raise RuntimeError(f"Runwise time mismatch for sub-{subject} {atlas} {smoothing} hemi-{hemi} run-{rid}")
        run_dir = fc_dir / "runs" / f"run-{rid}"
        run_dir.mkdir(parents=True, exist_ok=True)
        run_pearson_valid = corrcoef_rows(run_ts[valid_mask, :], run_net)
        np.save(run_dir / "hipp_vertex_to_network_fc_active.npy", run_pearson_valid.astype(np.float32))
        run_partial_valid = one_sided_network_partial_corr_rows(run_ts[valid_mask, :], run_net)
        np.save(run_dir / "hipp_vertex_to_network_partial_fc_active.npy", run_partial_valid.astype(np.float32))
    if offset != int(ts.shape[1]):
        raise RuntimeError(
            f"Run lengths do not sum to full hippocampal timeseries for sub-{subject} {atlas} {smoothing} hemi-{hemi}: "
            f"{offset} vs {ts.shape[1]}"
        )


def process_subject(
    subject: str,
    shared_root_str: str,
    atlases: list[str],
    smoothings: list[str],
    hemis: list[str],
) -> dict[str, object]:
    shared_root = Path(shared_root_str)
    n_fc = 0
    if KONG8 in atlases:
        materialize_reference(shared_root, subject)
    for smoothing in smoothings:
        for hemi in hemis:
            for atlas in atlases:
                materialize_fc(shared_root, subject, atlas, smoothing, hemi)
                n_fc += 1
    return {"subject": f"sub-{subject}", "fc_combos": int(n_fc)}


def main() -> int:
    parser = argparse.ArgumentParser(description="Materialize Kong2019 8net and one-sided networks-only partial FC caches.")
    parser.add_argument("--shared-root", default=str(DEFAULT_SHARED_ROOT))
    parser.add_argument("--subjects-manifest", default=str(DEFAULT_SUBJECTS))
    parser.add_argument("--subjects", nargs="*", default=None)
    parser.add_argument("--atlases", nargs="+", default=[KONG7, KONG8], choices=[KONG7, KONG8])
    parser.add_argument("--smoothings", nargs="+", default=SMOOTHINGS, choices=SMOOTHINGS)
    parser.add_argument("--hemis", nargs="+", default=HEMIS, choices=HEMIS)
    parser.add_argument("--jobs", type=int, default=1, help="Subject-level parallel workers; hard-capped at 10.")
    args = parser.parse_args()

    shared_root = Path(args.shared_root).resolve()
    subjects = [s.replace("sub-", "") for s in args.subjects] if args.subjects else load_subjects(Path(args.subjects_manifest).resolve())
    jobs = max(1, min(int(args.jobs), 10, len(subjects)))
    n_fc = 0
    if jobs == 1:
        for subject in subjects:
            obj = process_subject(subject, str(shared_root), list(args.atlases), list(args.smoothings), list(args.hemis))
            n_fc += int(obj["fc_combos"])
            print(f"[done] {obj['subject']}", flush=True)
    else:
        with ProcessPoolExecutor(max_workers=jobs) as ex:
            futs = [
                ex.submit(process_subject, s, str(shared_root), list(args.atlases), list(args.smoothings), list(args.hemis))
                for s in subjects
            ]
            for fut in as_completed(futs):
                obj = fut.result()
                n_fc += int(obj["fc_combos"])
                print(f"[done] {obj['subject']}", flush=True)
    print(json.dumps({"subjects": len(subjects), "fc_combos": int(n_fc), "jobs": int(jobs), "status": "ok"}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
