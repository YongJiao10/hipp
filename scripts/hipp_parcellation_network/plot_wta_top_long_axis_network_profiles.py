#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from surface_utils import long_axis_order


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_GROUP_ROOT = (
    REPO_ROOT
    / "outputs_experiments"
    / "hippunfold_recommended_t1w_20260423"
    / "kong2022"
    / "hipp_group_merged_canonical_atlas_routea_10perc"
)
DEFAULT_PRESENT_DIR = REPO_ROOT / "present_group_t1only"
DEFAULT_CROSS_ATLAS_MERGE = REPO_ROOT / "config" / "cross_atlas_network_merge.json"


@dataclass(frozen=True)
class HemiProfile:
    hemi: str
    network_names: list[str]
    network_keys: list[int]
    profile_percent: np.ndarray
    bin_sizes: list[int]
    surface_path: Path
    labels_path: Path
    label_map_path: Path


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def save_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def load_network_hex_colors(path: Path) -> dict[str, str]:
    out: dict[str, str] = {}
    payload = load_json(path)
    shared = payload.get("shared_colors_rgba", {})
    if not isinstance(shared, dict):
        return out
    for name, rgba in shared.items():
        if not isinstance(rgba, list) or len(rgba) != 4:
            continue
        r, g, b = int(rgba[0]), int(rgba[1]), int(rgba[2])
        out[str(name)] = f"#{r:02x}{g:02x}{b:02x}"
    return out


def load_network_keys(label_map: dict) -> tuple[list[str], list[int]]:
    raw = label_map.get("network_label_keys")
    if not isinstance(raw, dict) or len(raw) == 0:
        raise ValueError("Missing or empty network_label_keys in label map.")
    rows: list[tuple[int, str]] = []
    for name, key in raw.items():
        rows.append((int(key), str(name)))
    rows.sort(key=lambda x: x[0])
    keys = [k for k, _ in rows]
    names = [n for _, n in rows]
    return names, keys


def load_hemi_profile(manifest_path: Path, n_bins: int) -> tuple[HemiProfile, list[dict[str, str]]]:
    manifest = load_json(manifest_path)
    hemi = str(manifest["hemi"])
    labels_path = Path(str(manifest["group_functional_atlas_labels"])).resolve()
    label_map_path = Path(str(manifest["group_functional_atlas_label_map"])).resolve()
    surface_path = Path(str(manifest["template_left_or_right_surface"])).resolve()

    labels = np.load(labels_path).astype(np.int32, copy=False)
    label_map = load_json(label_map_path)
    network_names, network_keys = load_network_keys(label_map)

    order = long_axis_order(surface_path)
    if labels.ndim != 1:
        raise ValueError(f"Expected 1D label vector: {labels_path}")
    if order.ndim != 1:
        raise ValueError(f"Expected 1D long-axis order: {surface_path}")
    if labels.shape[0] != order.shape[0]:
        raise ValueError(
            f"Label/surface vertex mismatch for hemi {hemi}: labels={labels.shape[0]}, surface={order.shape[0]}"
        )

    bins = np.array_split(order, int(n_bins))
    profile = np.zeros((len(network_keys), int(n_bins)), dtype=np.float64)
    rows: list[dict[str, str]] = []
    bin_sizes: list[int] = []
    for b_idx, vertex_idx in enumerate(bins, start=1):
        bin_labels = labels[np.asarray(vertex_idx, dtype=np.int64)]
        denom = int(bin_labels.size)
        if denom <= 0:
            raise ValueError(f"Encountered empty bin for hemi {hemi} at bin {b_idx}")
        bin_sizes.append(denom)
        network_sum = 0
        for n_idx, (network_name, key) in enumerate(zip(network_names, network_keys)):
            count = int(np.count_nonzero(bin_labels == int(key)))
            network_sum += count
            pct = 100.0 * (float(count) / float(denom))
            profile[n_idx, b_idx - 1] = pct
            rows.append(
                {
                    "hemi": hemi,
                    "bin_index": str(b_idx),
                    "bin_vertex_count": str(denom),
                    "network_name": network_name,
                    "network_label_key": str(int(key)),
                    "network_vertex_count": str(count),
                    "percent_of_bin_vertices": f"{pct:.6f}",
                }
            )
        null_or_other = denom - network_sum
        rows.append(
            {
                "hemi": hemi,
                "bin_index": str(b_idx),
                "bin_vertex_count": str(denom),
                "network_name": "__non_network_or_null__",
                "network_label_key": "NA",
                "network_vertex_count": str(int(null_or_other)),
                "percent_of_bin_vertices": f"{100.0 * float(null_or_other) / float(denom):.6f}",
            }
        )

    return (
        HemiProfile(
            hemi=hemi,
            network_names=network_names,
            network_keys=network_keys,
            profile_percent=profile.astype(np.float32),
            bin_sizes=bin_sizes,
            surface_path=surface_path,
            labels_path=labels_path,
            label_map_path=label_map_path,
        ),
        rows,
    )


def write_tsv(path: Path, rows: list[dict[str, str]]) -> None:
    if len(rows) == 0:
        raise ValueError(f"No rows to write: {path}")
    fieldnames = [
        "smoothing",
        "hemi",
        "bin_index",
        "bin_vertex_count",
        "network_name",
        "network_label_key",
        "network_vertex_count",
        "percent_of_bin_vertices",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def build_figure(
    *,
    out_png: Path,
    atlas: str,
    route_tag: str,
    smoothing: str,
    n_bins: int,
    left: HemiProfile,
    right: HemiProfile,
    network_colors: dict[str, str],
) -> None:
    if left.network_names != right.network_names or left.network_keys != right.network_keys:
        raise ValueError("Left/right hemisphere network label map mismatch.")
    x = np.arange(1, int(n_bins) + 1, dtype=np.int32)
    fig, axes = plt.subplots(1, 2, figsize=(14.0, 5.2), sharey=True)
    for ax, item in zip(axes, [left, right]):
        for idx, name in enumerate(item.network_names):
            color = network_colors.get(name, "#444444")
            ax.plot(
                x,
                item.profile_percent[idx, :],
                color=color,
                linewidth=1.8,
                marker="o",
                markersize=3.5,
                label=name,
            )
        ax.set_xlim(1, int(n_bins))
        ax.set_xticks([1, 5, 10, 15, 20, 25] if n_bins == 25 else x)
        ax.set_ylim(0.0, 100.0)
        ax.grid(True, alpha=0.25, linewidth=0.6)
        ax.set_xlabel("Long-axis bin (head -> tail)")
        ax.set_title(f"Hemisphere {item.hemi}")
    axes[0].set_ylabel("% of bin vertices")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=4, frameon=False, bbox_to_anchor=(0.5, 1.03))
    fig.suptitle(f"{atlas} wta-top long-axis network profile ({route_tag}, {smoothing})", y=1.08)
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=220, bbox_inches="tight")
    plt.close(fig)


def run_one_smoothing(
    *,
    group_root: Path,
    present_dir: Path,
    atlas: str,
    route_tag: str,
    smoothing: str,
    coverage_tag: str,
    n_bins: int,
    network_colors: dict[str, str],
) -> dict:
    combo_root = group_root / "wta-top" / atlas / smoothing / coverage_tag
    manifest_l = combo_root / "hemi_L" / "group_atlas_manifest.json"
    manifest_r = combo_root / "hemi_R" / "group_atlas_manifest.json"
    for p in [manifest_l, manifest_r]:
        if not p.exists():
            raise FileNotFoundError(f"Missing manifest: {p}")

    left, left_rows = load_hemi_profile(manifest_l, n_bins)
    right, right_rows = load_hemi_profile(manifest_r, n_bins)
    out_png = present_dir / f"{atlas}_wta-top_long_axis_network_profile_{route_tag}_{smoothing}_bins{n_bins}.png"
    out_tsv = present_dir / f"{atlas}_wta-top_long_axis_network_profile_{route_tag}_{smoothing}_bins{n_bins}.tsv"

    rows = []
    for row in left_rows + right_rows:
        row_with_smoothing = dict(row)
        row_with_smoothing["smoothing"] = smoothing
        rows.append(row_with_smoothing)
    write_tsv(out_tsv, rows)

    build_figure(
        out_png=out_png,
        atlas=atlas,
        route_tag=route_tag,
        smoothing=smoothing,
        n_bins=n_bins,
        left=left,
        right=right,
        network_colors=network_colors,
    )

    return {
        "smoothing": smoothing,
        "coverage_tag": coverage_tag,
        "n_bins": int(n_bins),
        "input": {
            "group_root": str(group_root.resolve()),
            "left_manifest": str(manifest_l.resolve()),
            "right_manifest": str(manifest_r.resolve()),
            "left_surface": str(left.surface_path),
            "right_surface": str(right.surface_path),
            "left_labels_npy": str(left.labels_path),
            "right_labels_npy": str(right.labels_path),
            "left_label_map": str(left.label_map_path),
            "right_label_map": str(right.label_map_path),
        },
        "network_names": left.network_names,
        "network_label_keys": {name: int(key) for name, key in zip(left.network_names, left.network_keys)},
        "bin_sizes": {"L": left.bin_sizes, "R": right.bin_sizes},
        "outputs": {"png": str(out_png.resolve()), "tsv": str(out_tsv.resolve())},
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Plot wta-top long-axis network percentage profiles for each hemisphere.")
    parser.add_argument("--group-root", default=str(DEFAULT_GROUP_ROOT))
    parser.add_argument("--present-dir", default=str(DEFAULT_PRESENT_DIR))
    parser.add_argument("--atlas", default="kong2022")
    parser.add_argument("--route-tag", default="routea")
    parser.add_argument("--coverage-tag", default="cov_0p80")
    parser.add_argument("--smoothings", nargs="+", default=["2mm", "4mm"])
    parser.add_argument("--n-bins", type=int, default=25)
    parser.add_argument("--cross-atlas-merge-json", default=str(DEFAULT_CROSS_ATLAS_MERGE))
    args = parser.parse_args()

    if int(args.n_bins) <= 0:
        raise ValueError("--n-bins must be positive.")

    group_root = Path(args.group_root).resolve()
    present_dir = Path(args.present_dir).resolve()
    cross_atlas_merge_json = Path(args.cross_atlas_merge_json).resolve()
    if not group_root.exists():
        raise FileNotFoundError(f"Missing group root: {group_root}")
    if not cross_atlas_merge_json.exists():
        raise FileNotFoundError(f"Missing cross-atlas merge json: {cross_atlas_merge_json}")

    network_colors = load_network_hex_colors(cross_atlas_merge_json)
    present_dir.mkdir(parents=True, exist_ok=True)

    runs: list[dict] = []
    for smoothing in args.smoothings:
        run_info = run_one_smoothing(
            group_root=group_root,
            present_dir=present_dir,
            atlas=str(args.atlas),
            route_tag=str(args.route_tag),
            smoothing=str(smoothing),
            coverage_tag=str(args.coverage_tag),
            n_bins=int(args.n_bins),
            network_colors=network_colors,
        )
        runs.append(run_info)

    out_manifest = (
        present_dir / f"{args.atlas}_wta-top_long_axis_network_profile_{args.route_tag}_bins{args.n_bins}_manifest.json"
    )
    save_json(
        out_manifest,
        {
            "atlas": str(args.atlas),
            "method": "wta-top",
            "route_tag": str(args.route_tag),
            "group_root": str(group_root),
            "present_dir": str(present_dir),
            "coverage_tag": str(args.coverage_tag),
            "n_bins": int(args.n_bins),
            "cross_atlas_merge_json": str(cross_atlas_merge_json),
            "smoothings_requested": [str(x) for x in args.smoothings],
            "runs": runs,
        },
    )
    print(str(out_manifest))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
