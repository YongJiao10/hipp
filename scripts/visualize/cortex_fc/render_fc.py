"""Render cortical ROI connectivity from explicitly selected files."""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import numpy as np
import pandas as pd

from .fc_visualization import compute_network_fc, plot_chord, plot_combined_fc, plot_fc_heatmap

NETWORK_COLUMNS = {"7net": "canonical7", "8net": "canonical8", "17net": "family17"}


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument("--fc-matrix", type=Path, help="Square ROI-by-ROI NPY matrix; values are plotted unchanged.")
    inputs.add_argument("--timeseries", type=Path, help="ROI-by-time NPY array; compute Pearson correlation and Fisher-z FC.")
    parser.add_argument("--parcels", type=Path, required=True, help="ROI TSV in exactly the same row order as the input array.")
    parser.add_argument("--net-type", choices=NETWORK_COLUMNS, default="7net")
    parser.add_argument("--view", choices=("combined", "heatmap", "chord"), default="combined")
    parser.add_argument("--output", type=Path, required=True, help="Exact output PNG filename.")
    parser.add_argument("--title", default="")
    parser.add_argument("--colorbar-title", default="Fisher's $z$", help="Label for the supplied matrix's units.")
    parser.add_argument("--light-background", action="store_true")
    return parser


def load_fc_matrix(*, fc_matrix: Path | None, timeseries: Path | None) -> np.ndarray:
    if timeseries is not None:
        values = np.load(timeseries, allow_pickle=False)
        if values.ndim != 2 or min(values.shape) < 2:
            raise ValueError(f"Expected ROI-by-time data with at least two ROIs and timepoints: {timeseries}")
        matrix = np.arctanh(np.clip(np.corrcoef(values), -0.9999, 0.9999))
    else:
        matrix = np.load(fc_matrix, allow_pickle=False).astype(float)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1] or matrix.shape[0] < 2:
        raise ValueError("FC must be a square ROI-by-ROI matrix with at least two ROIs")
    np.fill_diagonal(matrix, np.nan)
    return matrix


def render(args: argparse.Namespace) -> Path:
    matrix = load_fc_matrix(fc_matrix=args.fc_matrix, timeseries=args.timeseries)
    parcels = pd.read_csv(args.parcels, sep="\t")
    if len(parcels) != matrix.shape[0]:
        raise ValueError(f"ROI TSV has {len(parcels)} rows but FC has {matrix.shape[0]} ROIs: {args.parcels}")
    required = {"parcel_id", "family17", "hemisphere", NETWORK_COLUMNS[args.net_type]}
    missing = sorted(required.difference(parcels.columns))
    if missing:
        raise ValueError(f"ROI TSV is missing columns {missing}: {args.parcels}")
    output = args.output
    if output.suffix != ".png":
        raise ValueError(f"Output must be a PNG filename: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    save_path = str(output.with_suffix(""))
    if args.view == "chord":
        network_fc, network_names = compute_network_fc(matrix, parcels, net_type=args.net_type)
        plot_chord(network_fc, node_names=network_names, save_path=save_path,
                   title=args.title, cb_title=args.colorbar_title,
                   dark_mode=not args.light_background)
    else:
        plot = plot_combined_fc if args.view == "combined" else plot_fc_heatmap
        plot(matrix, parcels_df=parcels, net_type=args.net_type,
             dark_mode=not args.light_background, title=args.title,
             cb_title=args.colorbar_title, save_path=save_path)
    return output


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    print(render(args))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
