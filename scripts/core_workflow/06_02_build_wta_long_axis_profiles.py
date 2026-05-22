#!/usr/bin/env python3
from __future__ import annotations

"""
Process Step Description:
Build winner-takes-all (WTA) long-axis network profiles. Computes and summarizes the WTA distribution proportion of each functional network along the hippocampal long axis (anterior to posterior).

Main Flow Path:
- 9th step in the numbered pipeline sequence (second stage of presentation output generation, specific to WTA-top method):
  01_01_run_hippunfold_t1w_local.py -> 01_02_check_hippunfold_outputs.py -> 02_01_prepare_group_inputs.py -> 02_02_build_qc_manifest.py -> 03_01/03_02 reference building -> 04_01/04_02/04_03 FC materialization -> 05_01 gradient computation -> 05_02 group atlas parcellation -> 06_02_build_wta_long_axis_profiles.py (this step) -> 07_01 group matrix overview.

Inputs:
- Command-line arguments:
  - --group-root: Directory where group-atlas intermediate files are saved (default: `outputs_core/work/group_atlas/routea_full_k3_fisherz`).
  - --present-dir: Directory where final presentation figures will be saved (default: `outputs_core/present/overviews`).
  - --atlas: Cortical atlas to process (default: "kong2022").
  - --route-tag: Workflow branch tag to process (default: "routea").
  - --smoothings: Smoothing levels to process: "2mm", "4mm".
  - --hemis: Hemispheres to process: "L", "R".
  - --bins: Long-axis bins count values to compute (default: [20]).
  - --coverage-threshold: Selected coverage threshold for overview outputs (default: 0.80).
- Physical inputs:
  - Group WTA-top label arrays: `{group_root}/wta-top/smooth-{smoothing}_hemi-{hemi}_cov-{threshold}/group_functional_atlas_labels.npy`.
  - Long axis AP coordinates of hippocampal template.

Outputs:
- Command-line output / file outputs:
  - Long axis network proportion profile line plots (PNG): `{present_dir}/{atlas}_wta-top_long_axis_network_profile_{route_tag}_{smoothing}_bins{n_bins}_cov{threshold_tag}.png`.
  - Detailed bins network proportions summary table (TSV): `{present_dir}/{atlas}_wta-top_long_axis_network_profile_{route_tag}_{smoothing}_bins{n_bins}_cov{threshold_tag}.tsv`.
  - Run output manifest JSON: `{present_dir}/{atlas}_wta-top_long_axis_network_profile_{route_tag}_bins{n_bins}_manifest.json`.
"""
import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from lib.output_layout import DEFAULT_OVERVIEWS_DIR, DEFAULT_ROUTEA_GROUP_ATLAS_ROOT
from lib.workflow_utils import write_json
from lib.io_utils import load_json, write_tsv
from lib.surface_utils import long_axis_order
from lib.traceability import StageSpec, run_stage


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_GROUP_ROOT = DEFAULT_ROUTEA_GROUP_ATLAS_ROOT
DEFAULT_PRESENT_DIR = DEFAULT_OVERVIEWS_DIR
DEFAULT_ROUTE_TAG = "routea"
DEFAULT_ATLAS = "kong2022"
DEFAULT_METHOD = "wta-top"
DEFAULT_SMOOTHINGS = ("2mm", "4mm")
DEFAULT_HEMIS = ("L", "R")
DEFAULT_BINS = (20, 25)
DEFAULT_COVERAGE_THRESHOLD = 0.80
DEFAULT_CROSS_ATLAS_MERGE = REPO_ROOT / "config" / "cross_atlas_network_merge.json"
NON_NETWORK_NAME = "__non_network_or_null__"
TSV_FIELDS = [
    "smoothing",
    "hemi",
    "bin_index",
    "bin_vertex_count",
    "network_name",
    "network_label_key",
    "network_vertex_count",
    "percent_of_bin_vertices",
]



def coverage_tag(threshold: float) -> str:
    return f"cov_{float(threshold):.2f}".replace(".", "p")


def profile_output_paths(
    *,
    present_dir: Path,
    atlas: str,
    method: str,
    route_tag: str,
    smoothing: str,
    n_bins: int,
) -> dict[str, Path]:
    stem = f"{atlas}_{method}_long_axis_network_profile_{route_tag}_{smoothing}_bins{int(n_bins)}"
    return {
        "png": Path(present_dir) / f"{stem}.png",
        "tsv": Path(present_dir) / f"{stem}.tsv",
    }


def manifest_output_path(*, present_dir: Path, atlas: str, method: str, route_tag: str, n_bins: int) -> Path:
    return Path(present_dir) / f"{atlas}_{method}_long_axis_network_profile_{route_tag}_bins{int(n_bins)}_manifest.json"


def load_network_colors(path: Path = DEFAULT_CROSS_ATLAS_MERGE) -> dict[str, str]:
    colors: dict[str, str] = {}
    if not path.exists():
        return colors
    payload = load_json(path)
    for name, rgba in payload.get("shared_colors_rgba", {}).items():
        if isinstance(rgba, list) and len(rgba) >= 3:
            colors[str(name)] = "#%02x%02x%02x" % (int(rgba[0]), int(rgba[1]), int(rgba[2]))
    return colors


def compute_profile_rows(
    *,
    labels_by_hemi: dict[str, np.ndarray],
    order_by_hemi: dict[str, np.ndarray],
    network_names: list[str],
    network_label_keys: dict[str, int],
    smoothing: str,
    n_bins: int,
) -> tuple[list[dict[str, object]], dict[str, list[int]]]:
    rows: list[dict[str, object]] = []
    bin_sizes: dict[str, list[int]] = {}
    for hemi in sorted(labels_by_hemi):
        labels = np.asarray(labels_by_hemi[hemi]).reshape(-1)
        order = np.asarray(order_by_hemi[hemi], dtype=np.int64).reshape(-1)
        if labels.size != order.size:
            raise ValueError(f"Label/order size mismatch for hemi-{hemi}: labels={labels.size} order={order.size}")
        bins = [np.asarray(x, dtype=np.int64) for x in np.array_split(order, int(n_bins))]
        bin_sizes[hemi] = [int(x.size) for x in bins]
        network_keys = {int(network_label_keys[name]) for name in network_names}
        for bin_idx, vertices in enumerate(bins, start=1):
            bin_labels = labels[vertices] if vertices.size else np.asarray([], dtype=labels.dtype)
            denom = int(bin_labels.size)
            for network_name in network_names:
                key = int(network_label_keys[network_name])
                count = int(np.sum(bin_labels == key)) if denom else 0
                pct = (100.0 * float(count) / float(denom)) if denom else 0.0
                rows.append(
                    {
                        "smoothing": str(smoothing),
                        "hemi": str(hemi),
                        "bin_index": int(bin_idx),
                        "bin_vertex_count": int(denom),
                        "network_name": str(network_name),
                        "network_label_key": int(key),
                        "network_vertex_count": int(count),
                        "percent_of_bin_vertices": f"{pct:.6f}",
                    }
                )
            non_network_count = int(np.sum(~np.isin(bin_labels, list(network_keys)))) if denom else 0
            non_network_pct = (100.0 * float(non_network_count) / float(denom)) if denom else 0.0
            rows.append(
                {
                    "smoothing": str(smoothing),
                    "hemi": str(hemi),
                    "bin_index": int(bin_idx),
                    "bin_vertex_count": int(denom),
                    "network_name": NON_NETWORK_NAME,
                    "network_label_key": "NA",
                    "network_vertex_count": int(non_network_count),
                    "percent_of_bin_vertices": f"{non_network_pct:.6f}",
                }
            )
    return rows, bin_sizes


def plot_profile_rows(
    *,
    rows: list[dict[str, object]],
    network_names: list[str],
    output_png: Path,
    title: str,
    colors: dict[str, str] | None = None,
) -> Path:
    colors = colors or {}
    hemis = sorted({str(r["hemi"]) for r in rows})
    fig, axes = plt.subplots(1, len(hemis), figsize=(6.8 * max(1, len(hemis)), 4.8), sharey=True)
    if len(hemis) == 1:
        axes = [axes]
    for ax, hemi in zip(axes, hemis):
        hemi_rows = [r for r in rows if str(r["hemi"]) == hemi]
        for network_name in network_names:
            pts = sorted(
                [r for r in hemi_rows if str(r["network_name"]) == network_name],
                key=lambda r: int(r["bin_index"]),
            )
            if not pts:
                continue
            xs = [int(r["bin_index"]) for r in pts]
            ys = [float(r["percent_of_bin_vertices"]) for r in pts]
            ax.plot(xs, ys, marker="o", linewidth=1.5, markersize=3.2, label=network_name, color=colors.get(network_name))
        non = sorted(
            [r for r in hemi_rows if str(r["network_name"]) == NON_NETWORK_NAME],
            key=lambda r: int(r["bin_index"]),
        )
        if any(float(r["percent_of_bin_vertices"]) > 0.0 for r in non):
            ax.plot(
                [int(r["bin_index"]) for r in non],
                [float(r["percent_of_bin_vertices"]) for r in non],
                linestyle="--",
                linewidth=1.0,
                color="#777777",
                label="non-network/null",
            )
        ax.set_title(f"hemi-{hemi}")
        ax.set_xlabel("Long-axis bin")
        ax.grid(True, axis="y", color="#dddddd", linewidth=0.6)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("Vertices in bin (%)")
    handles, labels = axes[-1].get_legend_handles_labels()
    fig.legend(handles, labels, loc="center left", bbox_to_anchor=(1.002, 0.5), frameon=False)
    fig.suptitle(title)
    fig.tight_layout(rect=(0, 0, 0.86, 0.95))
    output_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_png, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return output_png


def load_wta_inputs(
    *,
    group_root: Path,
    atlas: str,
    smoothing: str,
    coverage: str,
    hemis: list[str],
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], dict[str, Any], dict[str, dict[str, Any]]]:
    labels_by_hemi: dict[str, np.ndarray] = {}
    order_by_hemi: dict[str, np.ndarray] = {}
    manifests: dict[str, dict[str, Any]] = {}
    label_map: dict[str, Any] | None = None
    for hemi in hemis:
        run_dir = Path(group_root) / DEFAULT_METHOD / atlas / smoothing / coverage / f"hemi_{hemi}"
        manifest_path = run_dir / "group_atlas_manifest.json"
        if not manifest_path.exists():
            raise FileNotFoundError(f"Missing WTA group atlas manifest: {manifest_path}")
        manifest = load_json(manifest_path)
        manifests[hemi] = manifest
        labels_path = run_dir / "group_functional_atlas_labels.npy"
        map_path = run_dir / "group_functional_atlas_label_map.json"
        if not labels_path.exists() or not map_path.exists():
            raise FileNotFoundError(f"Missing WTA functional labels or label map under {run_dir}")
        labels_by_hemi[hemi] = np.load(labels_path).astype(np.int32)
        this_map = load_json(map_path)
        if label_map is None:
            label_map = this_map
        elif this_map.get("network_label_keys") != label_map.get("network_label_keys"):
            raise ValueError(f"Network label key mismatch at {map_path}")
        surface_path = Path(str(manifest.get("template_left_or_right_surface", "")))
        if not surface_path.exists():
            raise FileNotFoundError(f"Missing template surface recorded in manifest: {surface_path}")
        order_by_hemi[hemi] = long_axis_order(surface_path)
    if label_map is None:
        raise ValueError("No WTA inputs loaded")
    return labels_by_hemi, order_by_hemi, label_map, manifests


def build_profiles(
    *,
    group_root: Path,
    present_dir: Path,
    atlas: str,
    route_tag: str,
    smoothings: list[str],
    hemis: list[str],
    n_bins: int,
    coverage_threshold: float,
) -> dict[str, Any]:
    coverage = coverage_tag(coverage_threshold)
    colors = load_network_colors()
    runs: list[dict[str, Any]] = []
    present_dir.mkdir(parents=True, exist_ok=True)
    for smoothing in smoothings:
        labels_by_hemi, order_by_hemi, label_map, manifests = load_wta_inputs(
            group_root=group_root,
            atlas=atlas,
            smoothing=smoothing,
            coverage=coverage,
            hemis=hemis,
        )
        network_names = [str(x) for x in label_map["network_names"]]
        network_label_keys = {str(k): int(v) for k, v in label_map["network_label_keys"].items()}
        rows, bin_sizes = compute_profile_rows(
            labels_by_hemi=labels_by_hemi,
            order_by_hemi=order_by_hemi,
            network_names=network_names,
            network_label_keys=network_label_keys,
            smoothing=smoothing,
            n_bins=int(n_bins),
        )
        paths = profile_output_paths(
            present_dir=present_dir,
            atlas=atlas,
            method=DEFAULT_METHOD,
            route_tag=route_tag,
            smoothing=smoothing,
            n_bins=int(n_bins),
        )
        write_tsv(paths["tsv"], rows, TSV_FIELDS)
        plot_profile_rows(
            rows=rows,
            network_names=network_names,
            output_png=paths["png"],
            title=f"{atlas} {DEFAULT_METHOD} long-axis network profile | {route_tag} | {smoothing} | bins={int(n_bins)}",
            colors=colors,
        )
        runs.append(
            {
                "smoothing": str(smoothing),
                "coverage_tag": coverage,
                "n_bins": int(n_bins),
                "input": {
                    "group_root": str(Path(group_root).resolve()),
                    "manifests": {hemi: str((Path(group_root) / DEFAULT_METHOD / atlas / smoothing / coverage / f"hemi_{hemi}" / "group_atlas_manifest.json").resolve()) for hemi in hemis},
                    "labels_npy": {hemi: str((Path(group_root) / DEFAULT_METHOD / atlas / smoothing / coverage / f"hemi_{hemi}" / "group_functional_atlas_labels.npy").resolve()) for hemi in hemis},
                    "surfaces": {hemi: str(Path(str(manifests[hemi]["template_left_or_right_surface"])).resolve()) for hemi in hemis},
                },
                "network_names": network_names,
                "network_label_keys": network_label_keys,
                "bin_sizes": bin_sizes,
                "outputs": {"png": str(paths["png"].resolve()), "tsv": str(paths["tsv"].resolve())},
            }
        )
    manifest = {
        "atlas": atlas,
        "method": DEFAULT_METHOD,
        "route_tag": route_tag,
        "group_root": str(Path(group_root).resolve()),
        "present_dir": str(Path(present_dir).resolve()),
        "coverage_tag": coverage,
        "n_bins": int(n_bins),
        "profile_definition": "For each hemisphere, sort vertices by template long-axis coordinate, split into equal-count bins, and report percent of bin vertices assigned to each WTA-top network label.",
        "smoothings_requested": [str(x) for x in smoothings],
        "runs": runs,
    }
    out_manifest = manifest_output_path(
        present_dir=present_dir,
        atlas=atlas,
        method=DEFAULT_METHOD,
        route_tag=route_tag,
        n_bins=int(n_bins),
    )
    write_json(out_manifest, manifest, trailing_newline=True)
    return {"manifest": str(out_manifest.resolve()), "count": len(runs)}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build WTA-top hippocampal long-axis network-profile line plots.")
    parser.add_argument("--group-root", default=str(DEFAULT_GROUP_ROOT))
    parser.add_argument("--present-dir", default=str(DEFAULT_PRESENT_DIR))
    parser.add_argument("--atlas", default=DEFAULT_ATLAS)
    parser.add_argument("--route-tag", default=DEFAULT_ROUTE_TAG)
    parser.add_argument("--smoothings", nargs="+", default=list(DEFAULT_SMOOTHINGS))
    parser.add_argument("--hemis", nargs="+", choices=list(DEFAULT_HEMIS), default=list(DEFAULT_HEMIS))
    parser.add_argument("--bins", nargs="+", type=int, default=list(DEFAULT_BINS))
    parser.add_argument("--coverage-threshold", type=float, default=DEFAULT_COVERAGE_THRESHOLD)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    outputs = []
    for n_bins in args.bins:
        outputs.append(
            build_profiles(
                group_root=Path(args.group_root).resolve(),
                present_dir=Path(args.present_dir).resolve(),
                atlas=str(args.atlas),
                route_tag=str(args.route_tag),
                smoothings=[str(x) for x in args.smoothings],
                hemis=[str(x) for x in args.hemis],
                n_bins=int(n_bins),
                coverage_threshold=float(args.coverage_threshold),
            )
        )
    print(json.dumps({"status": "ok", "outputs": outputs}, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(
        run_stage(
            StageSpec(
                step_id="06_02_build_wta_long_axis_profiles",
                purpose="Render WTA-top hippocampal long-axis network-profile presentation products.",
                stage_main=main,
                upstream_defaults=("05_02_run_group_atlas",),
            )
        )
    )
