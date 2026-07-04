#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
CORE_WORKFLOW_DIR = REPO_ROOT / "scripts" / "core_workflow"
if str(CORE_WORKFLOW_DIR) not in sys.path:
    sys.path.insert(0, str(CORE_WORKFLOW_DIR))

from lib.fc_visualization import plot_combined_fc
from lib.output_layout import IOWA7T_BRANCHES, IOWA7T_VARIANTS, resolve_iowa7t_hipp_outputs_root, validate_iowa7t_branch, validate_iowa7t_subject, validate_iowa7t_variant

DEFAULT_SUBJECTS = ["06112026", "06122026"]
DEFAULT_VARIANTS = ["noxcpd", "36p"]
DEFAULT_BRANCHES = list(IOWA7T_BRANCHES)
DEFAULT_NET_TYPES = ["8net", "17net"]
VARIANT_LABELS = {
    "0p": "0P",
    "32p": "32P",
    "36p": "36P",
    "noxcpd": "NoXCPD",
}


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate Iowa7T branch FC heatmap/chord PNGs.")
    parser.add_argument("--subject", dest="subjects", action="append", default=None, help="Subject ID, e.g. 06122026. Repeatable.")
    parser.add_argument("--variant", dest="variants", action="append", choices=IOWA7T_VARIANTS, default=None, help="Iowa7T variant. Repeatable.")
    parser.add_argument("--branch", dest="branches", action="append", choices=IOWA7T_BRANCHES, default=None, help="Functional branch. Repeatable.")
    parser.add_argument("--net-type", dest="net_types", action="append", choices=DEFAULT_NET_TYPES, default=None, help="Network taxonomy to render. Repeatable.")
    parser.add_argument("--base-root", default=None, help="Optional Jianglab root override for tests; defaults to detected Jianglab root.")
    parser.add_argument("--skip-missing", action="store_true", default=False, help="Skip jobs with missing cortex time series or parcel TSV instead of failing.")
    return parser


def normalized_args(args: argparse.Namespace) -> argparse.Namespace:
    args.subjects = [validate_iowa7t_subject(s) for s in (args.subjects or DEFAULT_SUBJECTS)]
    args.variants = [validate_iowa7t_variant(v) for v in (args.variants or DEFAULT_VARIANTS)]
    args.branches = [validate_iowa7t_branch(b) for b in (args.branches or DEFAULT_BRANCHES)]
    args.net_types = list(args.net_types or DEFAULT_NET_TYPES)
    return args


def iter_jobs(args: argparse.Namespace):
    args = normalized_args(args)
    for subject in args.subjects:
        for variant in args.variants:
            for branch in args.branches:
                yield subject, variant, branch


def branch_root(subject: str, variant: str, branch: str, base_root: str | None = None) -> Path:
    jianglab_root = Path(base_root) if base_root is not None else None
    output_root = resolve_iowa7t_hipp_outputs_root(
        variant=variant,
        jianglab_root=jianglab_root,
        require_exists=False,
    )
    return output_root / "subjects" / f"sub-{subject}" / branch


def load_fc_matrix(timeseries_path: Path) -> np.ndarray:
    timeseries = np.load(timeseries_path)
    r = np.corrcoef(timeseries)
    r_clipped = np.clip(r, -0.9999, 0.9999)
    fc_matrix = np.arctanh(r_clipped)
    np.fill_diagonal(fc_matrix, np.nan)
    return fc_matrix


def render_job(subject: str, variant: str, branch: str, *, net_types: list[str], base_root: str | None, skip_missing: bool) -> int:
    root = branch_root(subject, variant, branch, base_root)
    ts_path = root / "cortex" / "kong2022" / "cortex_roi_parcel_timeseries.npy"
    tsv_path = root / "cortex" / "kong2022" / "cortex_roi_parcels.tsv"
    if not ts_path.exists() or not tsv_path.exists():
        message = f"Missing cortex files for sub-{subject} {variant} {branch}: {ts_path} / {tsv_path}"
        if skip_missing:
            print(f"Warning: {message}. Skipping.")
            return 0
        raise FileNotFoundError(message)

    print(f"Processing sub-{subject} | Pipeline: {VARIANT_LABELS[variant]} | Scan: {branch}")
    fc_matrix = load_fc_matrix(ts_path)
    parcels_df = pd.read_csv(tsv_path, sep="\t")
    output_dir = root / "figures"

    count = 0
    for net_type in net_types:
        save_path = output_dir / f"fc_heatmap_{net_type}"
        title = f"Subject {subject} ({VARIANT_LABELS[variant]} {branch}) FC Heatmap & Chord ({net_type})"
        plot_combined_fc(
            fc_matrix=fc_matrix,
            parcels_df=parcels_df,
            net_type=net_type,
            dark_mode=True,
            title=title,
            scale=0.7,
            cb_title="Fisher's $z$",
            save_path=str(save_path),
        )
        count += 1
    return count


def main(argv: list[str] | None = None) -> int:
    args = normalized_args(build_arg_parser().parse_args(argv))
    total_images = 0
    total_jobs = 0
    for subject, variant, branch in iter_jobs(args):
        total_jobs += 1
        total_images += render_job(
            subject,
            variant,
            branch,
            net_types=args.net_types,
            base_root=args.base_root,
            skip_missing=bool(args.skip_missing),
        )
    print(f"Successfully generated combined plots. Total configurations: {total_jobs}, total images: {total_images}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
