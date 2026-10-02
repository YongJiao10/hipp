#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import matplotlib.image as mpimg
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.iowa7t.run_branch_downstream import DEFAULT_OUTPUT_ROOT, resolve_branch_job  # noqa: E402
from scripts.iowa7t.run_branch_overviews import resolve_branch_overview_paths  # noqa: E402

DEFAULT_JOBS = (
    ("04172026", "multiband"),
    ("06112026", "multiband"),
    ("06112026", "multi_echo"),
    ("06122026", "multiband"),
    ("06122026", "multi_echo"),
)
GRADIENT_METHOD_VARIANTS = {
    "gradient_spectral_k3": "fisherz",
    "gradient_count": "normalized_top20_count",
    "gradient_weighted_fisherz": "weighted_fisherz",
}
OVERVIEW_STEMS = {
    "gradient_spectral_k3": "gradient_family17_fisherz",
    "gradient_count": "gradient_family17_count",
    "gradient_weighted_fisherz": "gradient_family17_weighted_fisherz",
}
SMOOTHINGS = ("2mm", "4mm")
HEMIS = ("L", "R")


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def check_png_nonblank(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Missing overview PNG: {path}")
    if path.stat().st_size <= 0:
        raise RuntimeError(f"Empty overview PNG: {path}")
    arr = np.asarray(mpimg.imread(path), dtype=np.float32)
    if arr.size == 0:
        raise RuntimeError(f"Unreadable/blank overview PNG: {path}")
    if float(np.nanstd(arr)) <= 1e-6:
        raise RuntimeError(f"Blank overview PNG has near-zero pixel variance: {path}")


def check_fisher_profile_status(
    profile: np.ndarray,
    status_payload: dict[str, Any],
    network_names: list[str],
    cluster_ids: list[int],
    *,
    context: str,
    require_finite: bool = False,
) -> None:
    arr = np.asarray(profile, dtype=np.float32)
    by_cluster = {int(row["cluster_id"]): row for row in status_payload.get("items", []) if "cluster_id" in row}
    for row_idx, cluster_id in enumerate(cluster_ids):
        row = by_cluster.get(int(cluster_id))
        if row is None or "top1_count" in row:
            continue
        prof_row = arr[int(row_idx), :]
        if not np.isfinite(prof_row).any():
            if require_finite:
                raise RuntimeError(f"Fisher-z profile has no finite Fisher-z values in {context} cluster_id={cluster_id}")
            continue
        top_idx = int(np.nanargmax(prof_row))
        expected_network = str(network_names[top_idx])
        got_network = str(row.get("top1_network", ""))
        if got_network != expected_network:
            raise RuntimeError(f"Fisher-z network mismatch in {context} cluster_id={cluster_id}: profile={expected_network} status={got_network}")
        expected_value = float(prof_row[top_idx])
        got_value = float(row.get("top1_fisher_z", 0.0))
        if abs(expected_value - got_value) > 1e-4:
            raise RuntimeError(f"Fisher-z mismatch in {context} cluster_id={cluster_id}: profile={expected_value:.6g} status={got_value:.6g}")


def check_normalized_count_profile_status(
    profile: np.ndarray,
    status_payload: dict[str, Any],
    network_names: list[str],
    cluster_ids: list[int],
    *,
    context: str,
) -> None:
    arr = np.asarray(profile, dtype=np.float32)
    by_cluster = {int(row["cluster_id"]): row for row in status_payload.get("items", []) if "cluster_id" in row}
    for row_idx, cluster_id in enumerate(cluster_ids):
        row = by_cluster.get(int(cluster_id))
        if row is None or "top1_normalized_count" not in row:
            continue
        top_network = str(row.get("top1_network", ""))
        if top_network in {"", "none", "unassigned"}:
            continue
        if top_network not in network_names:
            raise RuntimeError(f"Normalized count network mismatch in {context} cluster_id={cluster_id}: status={top_network}")
        prof_row = arr[int(row_idx), :]
        top_idx = int(network_names.index(top_network))
        profile_value = float(prof_row[top_idx])
        profile_max = float(np.nanmax(prof_row)) if np.isfinite(prof_row).any() else 0.0
        status_value = float(row.get("top1_normalized_count", 0.0))
        if abs(profile_value - status_value) > 1e-4 or status_value < profile_max - 1e-4:
            raise RuntimeError(
                f"Normalized count mismatch in {context} cluster_id={cluster_id}: "
                f"profile_for_status={profile_value:.6g} profile_max={profile_max:.6g} status={status_value:.6g}"
            )


def adapter_run_dir(cache_root: Path, subject: str, method: str, smoothing: str, hemi: str) -> Path:
    return (
        Path(cache_root)
        / "overview_core_adapter"
        / method
        / f"sub-{subject}"
        / "gradient-spectral"
        / "kong2022"
        / smoothing
        / "cov_0p80"
        / "g1g2g3"
        / f"hemi_{hemi}"
    )


def audit_job(subject: str, branch: str, *, output_root: Path) -> list[dict[str, Any]]:
    job = resolve_branch_job(subject, branch, output_root=output_root)
    paths = resolve_branch_overview_paths(job, network_column="family17")
    rows: list[dict[str, Any]] = []
    sub = f"sub-{job.subject}"
    for method, variant in GRADIENT_METHOD_VARIANTS.items():
        png = paths.final_gradient_figures / f"{sub}_{OVERVIEW_STEMS[method]}_overview.png"
        check_png_nonblank(png)
        for smoothing in SMOOTHINGS:
            for hemi in HEMIS:
                run_dir = adapter_run_dir(paths.cache_root, job.subject, method, smoothing, hemi)
                summary = read_json(run_dir / "group_feature_summary.json")
                network_names = [str(x) for x in summary["network_names"]]
                if len(network_names) != 17:
                    raise RuntimeError(f"Expected 17 family networks in {run_dir}, got {len(network_names)}")
                if str(summary.get("network_column")) != "family17":
                    raise RuntimeError(f"Expected network_column=family17 in {run_dir}")
                profile = np.load(run_dir / "cluster_network_profile.npy").astype(np.float32)
                status = read_json(run_dir / f"cluster_assignment_status_{variant}.json")
                cluster_ids = list(range(1, int(profile.shape[0]) + 1))
                context = f"sub-{job.subject} {branch} {method} {smoothing} hemi-{hemi}"
                if variant == "normalized_top20_count":
                    check_normalized_count_profile_status(profile, status, network_names, cluster_ids, context=context)
                else:
                    check_fisher_profile_status(profile, status, network_names, cluster_ids, context=context, require_finite=(variant == "weighted_fisherz"))
                rows.append({"subject": sub, "branch": branch, "method": method, "smoothing": smoothing, "hemi": hemi, "status": "ok"})
    return rows


def parse_job(value: str) -> tuple[str, str]:
    if ":" not in value:
        raise argparse.ArgumentTypeError("Expected --job value as <subject>:<multiband|multi_echo>")
    subject, branch = value.split(":", 1)
    return subject.replace("sub-", ""), branch


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Audit Iowa7T family17 gradient overview PNGs and source profile/status consistency.")
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--job", action="append", type=parse_job, default=None)
    return parser


def main() -> int:
    args = build_arg_parser().parse_args()
    jobs = args.job if args.job else list(DEFAULT_JOBS)
    rows: list[dict[str, Any]] = []
    errors: list[dict[str, str]] = []
    for subject, branch in jobs:
        try:
            rows.extend(audit_job(subject, branch, output_root=Path(args.output_root)))
        except Exception as exc:
            errors.append({"subject": f"sub-{subject}", "branch": branch, "error_type": exc.__class__.__name__, "error": str(exc)})
    payload = {"status": "failed" if errors else "ok", "checked": len(rows), "errors": len(errors), "items": rows, "error_items": errors}
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
