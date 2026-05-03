#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib
import nibabel as nib
import numpy as np
from nibabel.gifti import GiftiDataArray, GiftiImage, GiftiLabel, GiftiLabelTable
from scipy import stats

matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT / "scripts" / "cortex") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "scripts" / "cortex"))
if str(REPO_ROOT / "scripts" / "hipp_parcellation_network") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "scripts" / "hipp_parcellation_network"))
from kong2022_label_utils import KONG2022_DLABEL_PATH, load_kong2022_bundle
from surface_utils import render_locked_grid


DEFAULT_GROUP_ROOT = REPO_ROOT / "outputs_migration" / "hipp_group_merged_canonical_atlas"
DEFAULT_SHARED_ROOT = REPO_ROOT / "outputs_migration" / "hipp_group_inputs_shared"
DEFAULT_PRESENT_DIR = REPO_ROOT / "present_group"
DEFAULT_DATA_ROOT = REPO_ROOT / "data" / "hippunfold_input"
DEFAULT_CORTEX_ASSETS_ROOT = REPO_ROOT / "outputs_migration" / "cortex_pfm"
DEFAULT_PUBLIC_TEMPLATE_ROOT = REPO_ROOT / "external"
DEFAULT_SCENE = REPO_ROOT / "config" / "manual_wb_scenes" / "cortex_manual.scene"
DEFAULT_HIPP_SCENE = REPO_ROOT / "config" / "wb_locked_native_view_lateral_medial.scene"

METHOD_CHOICES = [
    "group-network-spectral",
    "group-network-partial-spectral",
    "group-intrinsic-spectral",
    "group-intrinsic-partial-spectral",
    "group-network-top",
    "group-network-all",
]
METHODS_DEFAULT = ["group-intrinsic-spectral", "group-network-spectral"]
SMOOTHINGS = ["2mm", "4mm"]
HEMIS = ["L", "R"]
VIEW_SPECS = [
    {"name": "lateral", "axis": None, "deg": 0.0},
    {"name": "medial", "axis": "y", "deg": -90.0},
    {"name": "dorsal", "axis": "x", "deg": 90.0},
    {"name": "ventral", "axis": "x", "deg": -90.0},
]
CLUSTERLESS_METHODS = {"group-network-top", "group-network-all"}


@dataclass(frozen=True)
class Combo:
    atlas: str
    method: str
    smoothing: str
    hemi: str


@dataclass(frozen=True)
class ROIStat:
    roi_id: int
    roi_name: str
    hemi: str
    family17: str
    canonical8: str
    mean_z: float
    t_stat: float
    p_positive: float
    p_negative: float
    q_positive: float
    q_negative: float
    n_subjects: int
    is_sig_positive: int
    is_sig_negative: int
    is_selected: int
    selected_sign: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build standalone cluster-to-cortex top-ROI figures (independent from overview) "
            "for Kong2022 group-atlas combos."
        )
    )
    parser.add_argument("--group-root", default=str(DEFAULT_GROUP_ROOT))
    parser.add_argument("--shared-root", default=str(DEFAULT_SHARED_ROOT))
    parser.add_argument("--present-dir", default=str(DEFAULT_PRESENT_DIR))
    parser.add_argument(
        "--runtime-root",
        default=None,
        help="Root for all outputs (cluster figures, manifests, quicklook index). Defaults to <present-dir>/runtime.",
    )
    parser.add_argument("--atlas", default="kong2022", choices=["kong2022"])
    parser.add_argument("--methods", nargs="+", default=METHODS_DEFAULT, choices=METHOD_CHOICES)
    parser.add_argument("--smoothings", nargs="+", default=SMOOTHINGS)
    parser.add_argument("--hemis", nargs="+", default=HEMIS)
    parser.add_argument("--coverage-threshold", type=float, default=0.80)
    parser.add_argument("--fdr-q", type=float, default=0.05)
    parser.add_argument("--max-rois-per-sign", type=int, default=20)
    parser.add_argument("--cortex-template-subject", default="100610")
    parser.add_argument("--data-root", default=str(DEFAULT_DATA_ROOT))
    parser.add_argument("--cortex-assets-root", default=str(DEFAULT_CORTEX_ASSETS_ROOT))
    parser.add_argument("--scene", default=str(DEFAULT_SCENE))
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def save_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def read_tsv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f, delimiter="\t"))


def write_tsv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter="\t", extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def run_cmd(cmd: list[str]) -> None:
    proc = subprocess.run(cmd, text=True, capture_output=True)
    if proc.returncode != 0:
        raise RuntimeError(
            f"Command failed ({proc.returncode}): {' '.join(cmd)}\n"
            f"STDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}"
        )


def parse_roi_registry(path: Path) -> list[dict[str, Any]]:
    rows = read_tsv_rows(path)
    out: list[dict[str, Any]] = []
    for row in rows:
        out.append(
            {
                "roi_id": int(row["roi_id"]),
                "roi_name": str(row["roi_name"]),
                "hemi": str(row["hemi"]),
                "family17": str(row["family17"]),
                "subregion": str(row.get("subregion", "")),
                "rank": int(row.get("rank", "0")),
                "canonical8": str(row["canonical8"]),
            }
        )
    if len(out) != 400:
        raise RuntimeError(f"Expected 400 ROI registry rows, got {len(out)} from {path}")
    return out


def read_subject_roi_tsv(path: Path) -> list[dict[str, Any]]:
    rows = read_tsv_rows(path)
    out: list[dict[str, Any]] = []
    for row in rows:
        out.append(
            {
                "parcel_id": int(row["parcel_id"]),
                "parcel_name": str(row["parcel_name"]),
                "family17": str(row["family17"]),
                "canonical8": str(row["canonical8"]),
                "hemisphere": str(row["hemisphere"]),
            }
        )
    return out


def load_shared_network_colors() -> dict[str, tuple[int, int, int, int]]:
    cfg = REPO_ROOT / "config" / "cross_atlas_network_merge.json"
    payload = load_json(cfg)
    out: dict[str, tuple[int, int, int, int]] = {}
    for name, rgba in payload.get("shared_colors_rgba", {}).items():
        if isinstance(rgba, list) and len(rgba) == 4:
            out[str(name)] = tuple(int(v) for v in rgba)
    return out


def neg_color_rgba(rgba: tuple[int, int, int, int], blend: float = 0.72) -> tuple[int, int, int, int]:
    """Lighten network color toward white to signal negative FC."""
    r, g, b, a = rgba
    rr = int(round(r + (255 - r) * blend))
    gg = int(round(g + (255 - g) * blend))
    bb = int(round(b + (255 - b) * blend))
    return rr, gg, bb, a


def display_color_for_sign(base: tuple[int, int, int, int], sign: str) -> tuple[int, int, int, int]:
    if sign == "positive":
        return base
    if sign == "negative":
        return neg_color_rgba(base)
    return base


def corrcoef_rows(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    x = np.asarray(a, dtype=np.float32)
    y = np.asarray(b, dtype=np.float32)
    x_mu = np.nanmean(x, axis=1, keepdims=True)
    y_mu = np.nanmean(y, axis=1, keepdims=True)
    x_sd = np.nanstd(x, axis=1, keepdims=True, ddof=1)
    y_sd = np.nanstd(y, axis=1, keepdims=True, ddof=1)
    xz = (x - x_mu) / np.clip(x_sd, 1e-12, None)
    yz = (y - y_mu) / np.clip(y_sd, 1e-12, None)
    xz[~np.isfinite(xz)] = 0.0
    yz[~np.isfinite(yz)] = 0.0
    corr = (xz @ yz.T) / float(max(1, x.shape[1] - 1))
    corr = np.clip(corr, -0.999999, 0.999999)
    corr[~np.isfinite(corr)] = 0.0
    return corr.astype(np.float32)


def bh_fdr(p_values: np.ndarray) -> np.ndarray:
    p = np.asarray(p_values, dtype=np.float64)
    q = np.full_like(p, np.nan, dtype=np.float64)
    finite = np.isfinite(p)
    m = int(np.sum(finite))
    if m == 0:
        return q
    p_f = p[finite]
    order = np.argsort(p_f)
    sorted_p = p_f[order]
    ranks = np.arange(1, m + 1, dtype=np.float64)
    adjusted = sorted_p * float(m) / ranks
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    adjusted = np.clip(adjusted, 0.0, 1.0)
    q_f = np.empty_like(sorted_p)
    q_f[order] = adjusted
    q[finite] = q_f
    return q


def one_sample_one_sided_stats(x: np.ndarray) -> tuple[float, float, float, float, int]:
    vals = np.asarray(x, dtype=np.float64)
    vals = vals[np.isfinite(vals)]
    n = int(vals.size)
    if n == 0:
        return float("nan"), float("nan"), float("nan"), float("nan"), 0
    mean = float(np.mean(vals))
    if n < 2:
        return mean, float("nan"), float("nan"), float("nan"), n
    sd = float(np.std(vals, ddof=1))
    if not np.isfinite(sd) or sd <= 1e-12:
        if mean > 0.0:
            return mean, float("inf"), 0.0, 1.0, n
        if mean < 0.0:
            return mean, float("-inf"), 1.0, 0.0, n
        return mean, 0.0, 0.5, 0.5, n
    t_stat = mean / (sd / math.sqrt(float(n)))
    p_pos = float(stats.t.sf(t_stat, df=n - 1))
    p_neg = float(stats.t.cdf(t_stat, df=n - 1))
    return mean, float(t_stat), p_pos, p_neg, n


def make_label_gifti(labels: np.ndarray, label_items: list[tuple[int, str, tuple[int, int, int, int]]]) -> GiftiImage:
    table = GiftiLabelTable()
    for key, name, rgba in label_items:
        g = GiftiLabel(
            key=int(key),
            red=float(rgba[0]) / 255.0,
            green=float(rgba[1]) / 255.0,
            blue=float(rgba[2]) / 255.0,
            alpha=float(rgba[3]) / 255.0,
        )
        g.label = str(name)
        table.labels.append(g)
    da = GiftiDataArray(data=labels.astype(np.int32), intent="NIFTI_INTENT_LABEL", datatype="NIFTI_TYPE_INT32")
    return GiftiImage(darrays=[da], labeltable=table)


def resolve_cortex_assets(subject: str, data_root: Path, assets_root: Path, scene: Path) -> dict[str, Path]:
    anat = data_root / f"sub-{subject}" / "anat"
    assets = assets_root / f"sub-{subject}" / "assets"
    public_root = DEFAULT_PUBLIC_TEMPLATE_ROOT.resolve()
    fastans_tpl = public_root / "FASTANS" / "resources" / "templates"

    def first_existing(paths: list[Path]) -> Path | None:
        for p in paths:
            if p.exists():
                return p.resolve()
        return None

    left_inflated = first_existing(
        [
            anat / f"sub-{subject}_hemi-L_space-fsLR_den-32k_desc-MSMAll_inflated.surf.gii",
            assets_root / f"sub-{subject}_hemi-L_space-fsLR_den-32k_desc-MSMAll_inflated.surf.gii",
            public_root / f"sub-{subject}_hemi-L_space-fsLR_den-32k_desc-MSMAll_inflated.surf.gii",
        ]
    )
    right_inflated = first_existing(
        [
            anat / f"sub-{subject}_hemi-R_space-fsLR_den-32k_desc-MSMAll_inflated.surf.gii",
            assets_root / f"sub-{subject}_hemi-R_space-fsLR_den-32k_desc-MSMAll_inflated.surf.gii",
            public_root / f"sub-{subject}_hemi-R_space-fsLR_den-32k_desc-MSMAll_inflated.surf.gii",
        ]
    )
    sulc_dscalar = first_existing(
        [
            anat / f"sub-{subject}_space-fsLR_den-32k_desc-MSMAll_sulc.dscalar.nii",
            assets_root / f"sub-{subject}_space-fsLR_den-32k_desc-MSMAll_sulc.dscalar.nii",
            public_root / f"sub-{subject}_space-fsLR_den-32k_desc-MSMAll_sulc.dscalar.nii",
        ]
    )
    left_sulc = first_existing(
        [
            assets / f"sub-{subject}_hemi-L_space-fsLR_den-32k_desc-MSMAll_sulc.func.gii",
            assets_root / f"sub-{subject}_hemi-L_space-fsLR_den-32k_desc-MSMAll_sulc.func.gii",
            fastans_tpl / "CORTEX_LEFT.32k_fs_LR..func.gii",
        ]
    )
    right_sulc = first_existing(
        [
            assets / f"sub-{subject}_hemi-R_space-fsLR_den-32k_desc-MSMAll_sulc.func.gii",
            assets_root / f"sub-{subject}_hemi-R_space-fsLR_den-32k_desc-MSMAll_sulc.func.gii",
            fastans_tpl / "CORTEX_RIGHT.32k_fs_LR.func.gii",
        ]
    )
    if (left_sulc is None or right_sulc is None) and sulc_dscalar is not None:
        wb_cmd = (REPO_ROOT / "scripts" / "wb_command").resolve()
        cache_dir = (REPO_ROOT / "tmp" / "cortex_template_cache" / f"sub-{subject}").resolve()
        cache_dir.mkdir(parents=True, exist_ok=True)
        left_gen = cache_dir / f"sub-{subject}_hemi-L_space-fsLR_den-32k_desc-MSMAll_sulc.func.gii"
        right_gen = cache_dir / f"sub-{subject}_hemi-R_space-fsLR_den-32k_desc-MSMAll_sulc.func.gii"
        if not left_gen.exists() or not right_gen.exists():
            run_cmd(
                [
                    str(wb_cmd),
                    "-cifti-separate",
                    str(sulc_dscalar),
                    "COLUMN",
                    "-metric",
                    "CORTEX_LEFT",
                    str(left_gen),
                    "-metric",
                    "CORTEX_RIGHT",
                    str(right_gen),
                ]
            )
        if left_sulc is None and left_gen.exists():
            left_sulc = left_gen.resolve()
        if right_sulc is None and right_gen.exists():
            right_sulc = right_gen.resolve()

    out: dict[str, Path | None] = {
        "left_inflated": left_inflated,
        "right_inflated": right_inflated,
        "sulc_dscalar": sulc_dscalar,
        "left_sulc": left_sulc,
        "right_sulc": right_sulc,
        "scene": scene.resolve() if scene.exists() else None,
        "dlabel": KONG2022_DLABEL_PATH.resolve() if KONG2022_DLABEL_PATH.exists() else None,
    }
    missing = [name for name, path in out.items() if path is None]
    if missing:
        details = {
            "missing_keys": missing,
            "subject": f"sub-{subject}",
            "data_root": str(data_root),
            "assets_root": str(assets_root),
            "public_template_root": str(public_root),
        }
        raise FileNotFoundError("Missing cortex rendering assets:\n" + json.dumps(details, indent=2))
    return {k: v for k, v in out.items() if v is not None}


def compute_subject_cluster_roi_profiles(
    *,
    subjects: list[str],
    shared_root: Path,
    smoothing: str,
    hemi: str,
    group_labels: np.ndarray,
    support_mask: np.ndarray,
    roi_registry: list[dict[str, Any]],
) -> tuple[np.ndarray, list[str]]:
    k_final = int(np.max(group_labels))
    col_by_roi_id = {int(row["roi_id"]): idx for idx, row in enumerate(roi_registry)}
    reg_by_id = {int(row["roi_id"]): row for row in roi_registry}

    all_profiles: list[np.ndarray] = []
    used_subjects: list[str] = []
    for subject in subjects:
        ts_path = shared_root / f"sub-{subject}" / "surface" / smoothing / f"sub-{subject}_hemi-{hemi}_timeseries.npy"
        roi_path = shared_root / f"sub-{subject}" / "reference" / "kong2022" / "cortex_roi_parcel_timeseries.npy"
        roi_tsv = shared_root / f"sub-{subject}" / "reference" / "kong2022" / "cortex_roi_parcels.tsv"
        valid_path = shared_root / f"sub-{subject}" / "surface" / "tsnr_gate" / f"sub-{subject}_hemi-{hemi}_valid_mask.npy"
        if not (ts_path.exists() and roi_path.exists() and roi_tsv.exists() and valid_path.exists()):
            continue

        ts = np.load(ts_path).astype(np.float32)
        roi = np.load(roi_path).astype(np.float32)
        rows = read_subject_roi_tsv(roi_tsv)
        valid_mask = np.load(valid_path).astype(bool)

        if ts.ndim != 2 or roi.ndim != 2:
            continue
        if ts.shape[1] != roi.shape[1]:
            continue
        if int(ts.shape[0]) != int(group_labels.size):
            continue
        if int(valid_mask.size) != int(group_labels.size):
            continue
        if int(roi.shape[0]) != int(len(rows)):
            continue

        vertex_valid = support_mask & valid_mask
        cluster_ts = np.full((k_final, ts.shape[1]), np.nan, dtype=np.float32)
        for cluster_id in range(1, k_final + 1):
            mask = (group_labels == int(cluster_id)) & vertex_valid
            if int(np.sum(mask)) == 0:
                continue
            with np.errstate(invalid="ignore"):
                cluster_ts[cluster_id - 1, :] = np.nanmean(ts[mask, :], axis=0).astype(np.float32)

        corr = corrcoef_rows(cluster_ts, roi)
        fisher = np.arctanh(np.clip(corr, -0.999999, 0.999999)).astype(np.float32)
        fisher[~np.isfinite(fisher)] = np.nan

        scatter = np.full((k_final, 400), np.nan, dtype=np.float32)
        for local_col, row in enumerate(rows):
            roi_id = int(row["parcel_id"])
            reg = reg_by_id.get(roi_id)
            if reg is None:
                raise KeyError(f"Unexpected ROI id={roi_id} in sub-{subject}")
            if str(row["parcel_name"]) != str(reg["roi_name"]):
                raise RuntimeError(
                    f"ROI name mismatch for sub-{subject}, roi_id={roi_id}: "
                    f"{row['parcel_name']} != {reg['roi_name']}"
                )
            col = int(col_by_roi_id[roi_id])
            scatter[:, col] = fisher[:, int(local_col)]

        all_profiles.append(scatter)
        used_subjects.append(subject)

    if not all_profiles:
        raise RuntimeError(
            f"No valid subject profiles for smoothing={smoothing} hemi={hemi}; "
            "required timeseries/ROI/mask artifacts are missing or inconsistent."
        )
    return np.stack(all_profiles, axis=0).astype(np.float32), used_subjects


def analyze_cluster(
    *,
    cluster_values: np.ndarray,
    roi_registry: list[dict[str, Any]],
    fdr_q: float,
    max_rois_per_sign: int,
) -> tuple[list[ROIStat], list[dict[str, Any]], list[dict[str, Any]]]:
    n_roi = int(cluster_values.shape[1])
    mean_z = np.full((n_roi,), np.nan, dtype=np.float64)
    t_stat = np.full((n_roi,), np.nan, dtype=np.float64)
    p_pos = np.full((n_roi,), np.nan, dtype=np.float64)
    p_neg = np.full((n_roi,), np.nan, dtype=np.float64)
    n_subj = np.zeros((n_roi,), dtype=np.int32)

    for ridx in range(n_roi):
        mz, ts_val, pp, pn, n = one_sample_one_sided_stats(cluster_values[:, ridx])
        mean_z[ridx] = mz
        t_stat[ridx] = ts_val
        p_pos[ridx] = pp
        p_neg[ridx] = pn
        n_subj[ridx] = int(n)

    q_pos = bh_fdr(p_pos)
    q_neg = bh_fdr(p_neg)
    sig_pos = np.isfinite(q_pos) & (q_pos < float(fdr_q)) & np.isfinite(mean_z) & (mean_z > 0.0)
    sig_neg = np.isfinite(q_neg) & (q_neg < float(fdr_q)) & np.isfinite(mean_z) & (mean_z < 0.0)

    pos_idx = np.flatnonzero(sig_pos)
    neg_idx = np.flatnonzero(sig_neg)
    pos_order = pos_idx[np.argsort(-mean_z[pos_idx])] if pos_idx.size else np.array([], dtype=np.int32)
    neg_order = neg_idx[np.argsort(mean_z[neg_idx])] if neg_idx.size else np.array([], dtype=np.int32)
    pos_take = pos_order[: int(max_rois_per_sign)]
    neg_take = neg_order[: int(max_rois_per_sign)]

    selected_sign = np.full((n_roi,), "", dtype=object)
    selected_sign[pos_take] = "positive"
    selected_sign[neg_take] = "negative"

    rows: list[ROIStat] = []
    selected_pos: list[dict[str, Any]] = []
    selected_neg: list[dict[str, Any]] = []
    for ridx, reg in enumerate(roi_registry):
        stat_row = ROIStat(
            roi_id=int(reg["roi_id"]),
            roi_name=str(reg["roi_name"]),
            hemi=str(reg["hemi"]),
            family17=str(reg["family17"]),
            canonical8=str(reg["canonical8"]),
            mean_z=float(mean_z[ridx]) if np.isfinite(mean_z[ridx]) else float("nan"),
            t_stat=float(t_stat[ridx]) if np.isfinite(t_stat[ridx]) else float("nan"),
            p_positive=float(p_pos[ridx]) if np.isfinite(p_pos[ridx]) else float("nan"),
            p_negative=float(p_neg[ridx]) if np.isfinite(p_neg[ridx]) else float("nan"),
            q_positive=float(q_pos[ridx]) if np.isfinite(q_pos[ridx]) else float("nan"),
            q_negative=float(q_neg[ridx]) if np.isfinite(q_neg[ridx]) else float("nan"),
            n_subjects=int(n_subj[ridx]),
            is_sig_positive=int(bool(sig_pos[ridx])),
            is_sig_negative=int(bool(sig_neg[ridx])),
            is_selected=int(selected_sign[ridx] != ""),
            selected_sign=str(selected_sign[ridx]),
        )
        rows.append(stat_row)
        if selected_sign[ridx] == "positive":
            selected_pos.append(
                {
                    "roi_id": stat_row.roi_id,
                    "roi_name": stat_row.roi_name,
                    "canonical8": stat_row.canonical8,
                    "family17": stat_row.family17,
                    "mean_z": stat_row.mean_z,
                    "q": stat_row.q_positive,
                    "sign": "positive",
                }
            )
        elif selected_sign[ridx] == "negative":
            selected_neg.append(
                {
                    "roi_id": stat_row.roi_id,
                    "roi_name": stat_row.roi_name,
                    "canonical8": stat_row.canonical8,
                    "family17": stat_row.family17,
                    "mean_z": stat_row.mean_z,
                    "q": stat_row.q_negative,
                    "sign": "negative",
                }
            )

    return rows, selected_pos, selected_neg


def build_cluster_label_files(
    *,
    selected_items: list[dict[str, Any]],
    cluster_dir: Path,
    vertex_arrays: dict[str, tuple[np.ndarray, np.ndarray]],
    network_colors: dict[str, tuple[int, int, int, int]],
) -> tuple[Path, Path, list[dict[str, Any]]]:
    by_roi_id = {int(item["roi_id"]): item for item in selected_items}
    label_items: list[tuple[int, str, tuple[int, int, int, int]]] = [(0, "Null", (158, 158, 158, 255))]
    legend_items: list[dict[str, Any]] = []
    for item in selected_items:
        roi_id = int(item["roi_id"])
        canonical = str(item["canonical8"])
        base = network_colors.get(canonical, (128, 128, 128, 255))
        sign = str(item["sign"])
        color = display_color_for_sign(base, sign)
        sign_tag = "+" if sign == "positive" else "-"
        label_name = f"{sign_tag} {item['roi_name']}"
        label_items.append((roi_id, label_name, color))
        legend_items.append({"name": f"{sign_tag} {item['roi_name']} ({canonical})", "rgba": color})

    cluster_dir.mkdir(parents=True, exist_ok=True)
    wb_dir = cluster_dir / "workbench_assets"
    wb_dir.mkdir(parents=True, exist_ok=True)

    label_paths: dict[str, Path] = {}
    for hemi in ["L", "R"]:
        vertex_idx, parcel_ids = vertex_arrays[hemi]
        n_vertices = int(np.max(vertex_idx)) + 1
        values = np.zeros((n_vertices,), dtype=np.int32)
        if by_roi_id:
            selected_ids = np.asarray(sorted(by_roi_id), dtype=np.int32)
            keep = np.isin(parcel_ids.astype(np.int32), selected_ids)
            values[vertex_idx[keep].astype(np.int32)] = parcel_ids[keep].astype(np.int32)
        img = make_label_gifti(values, label_items)
        suffix = "L" if hemi == "L" else "R"
        out_path = wb_dir / f"cluster_selected_rois.{suffix}.label.gii"
        nib.save(img, str(out_path))
        label_paths[hemi] = out_path

    return label_paths["L"], label_paths["R"], legend_items


def render_cortex_single_image(
    *,
    cluster_dir: Path,
    left_label: Path,
    right_label: Path,
    cortex_assets: dict[str, Path],
) -> Path:
    views_dir = cluster_dir / "views"
    views_dir.mkdir(parents=True, exist_ok=True)
    render_scene_script = REPO_ROOT / "scripts" / "cortex" / "render_cortex_pfm_scene.py"
    view = VIEW_SPECS[0]
    out_scene = views_dir / f"cluster_rois_{view['name']}.scene"
    out_png_single = views_dir / f"cluster_rois_{view['name']}.png"
    cmd = [
        sys.executable,
        str(render_scene_script),
        "--source-scene",
        str(cortex_assets["scene"]),
        "--left-surface",
        str(cortex_assets["left_inflated"]),
        "--right-surface",
        str(cortex_assets["right_inflated"]),
        "--left-label",
        str(left_label),
        "--right-label",
        str(right_label),
        "--left-sulc",
        str(cortex_assets["left_sulc"]),
        "--right-sulc",
        str(cortex_assets["right_sulc"]),
        "--sulc-dscalar",
        str(cortex_assets["sulc_dscalar"]),
        "--dlabel",
        str(cortex_assets["dlabel"]),
        "--out-scene",
        str(out_scene),
        "--out-png",
        str(out_png_single),
        "--width",
        "1000",
        "--height",
        "700",
    ]
    if view["axis"] is not None:
        cmd.extend(["--rotation-axis", str(view["axis"]), "--rotation-deg", str(float(view["deg"]))])
    if not out_png_single.exists():
        run_cmd(cmd)
    return out_png_single


def dedupe_selected_items_from_global_top(global_top: dict[str, Any]) -> list[dict[str, Any]]:
    best_by_roi: dict[int, dict[str, Any]] = {}
    for sign_key in ["top_positive", "top_negative"]:
        sign = "positive" if sign_key == "top_positive" else "negative"
        for row in global_top.get(sign_key, []):
            roi_id = int(row["roi_idx"])
            cur = {
                "roi_id": roi_id,
                "roi_name": str(row["roi_name"]),
                "canonical8": str(row["canonical8"]),
                "sign": sign,
                "fisher_z": float(row["fisher_z"]),
            }
            old = best_by_roi.get(roi_id)
            if old is None or abs(cur["fisher_z"]) > abs(float(old["fisher_z"])):
                best_by_roi[roi_id] = cur
    return [best_by_roi[k] for k in sorted(best_by_roi)]


def surface_vertex_count(path: Path) -> int:
    img = nib.load(str(path))
    coords = np.asarray(img.agg_data("pointset"), dtype=np.float32)
    return int(coords.shape[0])


def build_global_top_hippomaps_label_files(
    *,
    cluster_dir: Path,
    combo_hemi: str,
    global_top: dict[str, Any],
    support_mask: np.ndarray,
    left_surface: Path,
    right_surface: Path,
) -> tuple[Path, Path, dict[str, int]]:
    n_left = surface_vertex_count(left_surface)
    n_right = surface_vertex_count(right_surface)
    labels_left = np.zeros((n_left,), dtype=np.int32)
    labels_right = np.zeros((n_right,), dtype=np.int32)

    # Keep one sign per vertex (strongest absolute FC among global-top edges).
    best_by_vertex: dict[int, tuple[int, float]] = {}
    for sign_key, sign_label in [("top_positive", 1), ("top_negative", 2)]:
        for row in global_top.get(sign_key, []):
            vidx = int(row["vertex_index"])
            val = float(row["fisher_z"])
            old = best_by_vertex.get(vidx)
            if old is None or abs(val) > abs(float(old[1])):
                best_by_vertex[vidx] = (int(sign_label), float(val))

    if combo_hemi == "L":
        n_vertices = int(labels_left.size)
        target = labels_left
    elif combo_hemi == "R":
        n_vertices = int(labels_right.size)
        target = labels_right
    else:
        raise ValueError(f"Unexpected combo_hemi={combo_hemi}")

    if int(support_mask.size) != n_vertices:
        raise RuntimeError(
            f"Support-mask size mismatch for hemi-{combo_hemi}: support={support_mask.size}, surface={n_vertices}"
        )

    selected_pos = 0
    selected_neg = 0
    for vidx, (sign_label, _val) in best_by_vertex.items():
        if 0 <= int(vidx) < n_vertices and bool(support_mask[int(vidx)]):
            target[int(vidx)] = int(sign_label)
            if int(sign_label) == 1:
                selected_pos += 1
            elif int(sign_label) == 2:
                selected_neg += 1

    label_items = [
        (0, "Null", (158, 158, 158, 255)),
        (1, "+ Top FC vertex", (220, 70, 70, 255)),
        (2, "- Top FC vertex", (79, 120, 255, 255)),
    ]

    wb_dir = cluster_dir / "hippomaps_workbench_assets"
    wb_dir.mkdir(parents=True, exist_ok=True)
    out_left = wb_dir / "global_top_vertices.L.label.gii"
    out_right = wb_dir / "global_top_vertices.R.label.gii"
    nib.save(make_label_gifti(labels_left, label_items), str(out_left))
    nib.save(make_label_gifti(labels_right, label_items), str(out_right))

    return out_left, out_right, {"positive": int(selected_pos), "negative": int(selected_neg)}


def render_clusterless_global_figure(
    *,
    cluster_dir: Path,
    title: str,
    left_label: Path,
    right_label: Path,
    cortex_assets: dict[str, Path],
    network_colors: dict[str, tuple[int, int, int, int]],
    global_top: dict[str, Any],
    support_mask: np.ndarray,
    combo_hemi: str,
    template_subject: str,
    template_spec_path: Path,
    left_surface: Path,
    right_surface: Path,
) -> Path:
    cortex_png = render_cortex_single_image(
        cluster_dir=cluster_dir,
        left_label=left_label,
        right_label=right_label,
        cortex_assets=cortex_assets,
    )
    hipp_left_label, hipp_right_label, vertex_counts = build_global_top_hippomaps_label_files(
        cluster_dir=cluster_dir,
        combo_hemi=combo_hemi,
        global_top=global_top,
        support_mask=support_mask,
        left_surface=left_surface,
        right_surface=right_surface,
    )
    hipp_render = render_locked_grid(
        subject=str(template_subject),
        scene=DEFAULT_HIPP_SCENE,
        outdir=cluster_dir / "hippomaps_render",
        name=f"global_top_vertices_hemi-{combo_hemi}",
        title=f"{title} | top FC vertices",
        left_labels=hipp_left_label,
        right_labels=hipp_right_label,
        left_surface=left_surface,
        right_surface=right_surface,
        spec_path=template_spec_path,
    )
    hipp_png = Path(str(hipp_render["native_scene_png"]))

    out_png = cluster_dir / "global_top_connections_cortex_hippomaps.png"
    fig = plt.figure(figsize=(18.0, 7.2), dpi=180, constrained_layout=True)
    gs = fig.add_gridspec(1, 3, width_ratios=[3.2, 3.2, 2.1], wspace=0.05)
    ax0 = fig.add_subplot(gs[0, 0])
    ax0.imshow(plt.imread(cortex_png))
    ax0.axis("off")
    ax0.set_title(f"{title} | cortex ROI", fontsize=14)

    ax1 = fig.add_subplot(gs[0, 1])
    ax1.imshow(plt.imread(hipp_png))
    ax1.axis("off")
    ax1.set_title(f"{title} | hippomaps vertex", fontsize=14)

    ax_leg = fig.add_subplot(gs[0, 2])
    ax_leg.axis("off")
    pos_n = int(len(global_top.get("top_positive", [])))
    neg_n = int(len(global_top.get("top_negative", [])))
    top_n_cap = int(global_top.get("top_n_cap", 20))
    q_threshold = float(global_top.get("q_threshold", 0.05))
    roi_pos = {int(x["roi_idx"]) for x in global_top.get("top_positive", [])}
    roi_neg = {int(x["roi_idx"]) for x in global_top.get("top_negative", [])}
    vertex_pos = {int(x["vertex_index"]) for x in global_top.get("top_positive", [])}
    vertex_neg = {int(x["vertex_index"]) for x in global_top.get("top_negative", [])}
    ax_leg.text(
        0.0,
        1.00,
        f"global-top selection (q<{q_threshold:.2f}); top {top_n_cap}/sign",
        ha="left",
        va="top",
        fontsize=13,
        transform=ax_leg.transAxes,
    )
    ax_leg.text(
        0.0,
        0.90,
        f"Selected FC edges: total={int(pos_n + neg_n)}  (+{pos_n} / -{neg_n})",
        ha="left",
        va="top",
        fontsize=15,
        fontweight="bold",
        transform=ax_leg.transAxes,
    )
    ax_leg.text(
        0.0,
        0.83,
        (
            f"Unique ROI: {len(roi_pos | roi_neg)}  "
            f"(+{len(roi_pos)} / -{len(roi_neg)})"
        ),
        ha="left",
        va="top",
        fontsize=14,
        transform=ax_leg.transAxes,
    )
    ax_leg.text(
        0.0,
        0.78,
        (
            f"Unique vertices: {int(vertex_counts['positive'] + vertex_counts['negative'])}  "
            f"(+{int(vertex_counts['positive'])} / -{int(vertex_counts['negative'])})"
        ),
        ha="left",
        va="top",
        fontsize=14,
        transform=ax_leg.transAxes,
    )
    ax_leg.text(0.0, 0.71, "Sign coding:", ha="left", va="top", fontsize=15, fontweight="bold", transform=ax_leg.transAxes)
    ax_leg.text(0.0, 0.66, "+ : network color (saturated)", ha="left", va="top", fontsize=14, transform=ax_leg.transAxes)
    ax_leg.text(0.0, 0.61, "- : pale tint of network color", ha="left", va="top", fontsize=14, transform=ax_leg.transAxes)
    ax_leg.text(0.0, 0.56, "Hippomap markers: red=+FC, blue=-FC", ha="left", va="top", fontsize=13, transform=ax_leg.transAxes)

    net_counts: dict[str, dict[str, int]] = {}
    for sign_key, sign in [("top_positive", "positive"), ("top_negative", "negative")]:
        for row in global_top.get(sign_key, []):
            net = str(row.get("canonical8", "Unknown"))
            bucket = net_counts.setdefault(net, {"positive": 0, "negative": 0})
            bucket[sign] += 1
    if net_counts:
        ax_leg.text(0.0, 0.53, "Network legend:", ha="left", va="top", fontsize=15, fontweight="bold", transform=ax_leg.transAxes)
        y = 0.49
        step = 0.050
        for net in sorted(net_counts.keys())[:12]:
            base = network_colors.get(net, (128, 128, 128, 255))
            neg = neg_color_rgba(base)
            c_pos = tuple(x / 255.0 for x in base[:3]) + (1.0,)
            c_neg = tuple(x / 255.0 for x in neg[:3]) + (1.0,)
            ax_leg.add_patch(
                plt.Rectangle((0.0, y - 0.018), 0.030, 0.026, transform=ax_leg.transAxes, facecolor=c_pos, edgecolor="none")
            )
            ax_leg.add_patch(
                plt.Rectangle((0.038, y - 0.018), 0.030, 0.026, transform=ax_leg.transAxes, facecolor=c_neg, edgecolor="none")
            )
            c = net_counts[net]
            ax_leg.text(
                0.076,
                y,
                f"{net} (+{int(c['positive'])}/-{int(c['negative'])})",
                ha="left",
                va="center",
                fontsize=14,
                transform=ax_leg.transAxes,
            )
            y -= step
            if y < 0.02:
                break
    else:
        ax_leg.text(
            0.0,
            0.53,
            "No global-top edges selected.",
            ha="left",
            va="top",
            fontsize=14,
            transform=ax_leg.transAxes,
        )

    ax_leg.text(
        0.0,
        0.01,
        f"Global top connections: +{pos_n} / -{neg_n}",
        ha="left",
        va="bottom",
        fontsize=11,
        transform=ax_leg.transAxes,
    )

    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return out_png


def render_cluster_figure(
    *,
    cluster_dir: Path,
    title: str,
    subtitle: str,
    left_label: Path,
    right_label: Path,
    selected_items: list[dict[str, Any]],
    network_colors: dict[str, tuple[int, int, int, int]],
    n_selected_positive: int,
    n_selected_negative: int,
    cortex_assets: dict[str, Path],
) -> Path:
    views_dir = cluster_dir / "views"
    views_dir.mkdir(parents=True, exist_ok=True)
    render_scene_script = REPO_ROOT / "scripts" / "cortex" / "render_cortex_pfm_scene.py"

    # Keep only the first scene view. In this scene template the first render already
    # contains the desired 4-panel cortex layout, avoiding 3x/4x multi-view expansion.
    view = VIEW_SPECS[0]
    out_scene = views_dir / f"cluster_rois_{view['name']}.scene"
    out_png_single = views_dir / f"cluster_rois_{view['name']}.png"
    cmd = [
        sys.executable,
        str(render_scene_script),
        "--source-scene",
        str(cortex_assets["scene"]),
        "--left-surface",
        str(cortex_assets["left_inflated"]),
        "--right-surface",
        str(cortex_assets["right_inflated"]),
        "--left-label",
        str(left_label),
        "--right-label",
        str(right_label),
        "--left-sulc",
        str(cortex_assets["left_sulc"]),
        "--right-sulc",
        str(cortex_assets["right_sulc"]),
        "--sulc-dscalar",
        str(cortex_assets["sulc_dscalar"]),
        "--dlabel",
        str(cortex_assets["dlabel"]),
        "--out-scene",
        str(out_scene),
        "--out-png",
        str(out_png_single),
        "--width",
        "1000",
        "--height",
        "700",
    ]
    if view["axis"] is not None:
        cmd.extend(["--rotation-axis", str(view["axis"]), "--rotation-deg", str(float(view["deg"]))])
    if not out_png_single.exists():
        run_cmd(cmd)

    out_png = cluster_dir / "cluster_cortex_toproi_4view.png"
    img = plt.imread(out_png_single)
    fig = plt.figure(figsize=(15.0, 8.8), dpi=180, constrained_layout=True)
    gs = fig.add_gridspec(1, 2, width_ratios=[3.5, 3.5], wspace=0.04)
    ax_img = fig.add_subplot(gs[0, 0])
    ax_img.imshow(img)
    ax_img.axis("off")
    ax_img.set_title(title, fontsize=16, loc="left", pad=7.0)

    ax_leg = fig.add_subplot(gs[0, 1])
    ax_leg.axis("off")
    total_selected = int(n_selected_positive) + int(n_selected_negative)
    ax_leg.text(0.0, 1.00, subtitle, ha="left", va="top", fontsize=13, wrap=True)
    ax_leg.text(
        0.0,
        0.90,
        f"Selected ROI: total={total_selected}  (+{int(n_selected_positive)} / -{int(n_selected_negative)})",
        ha="left",
        va="top",
        fontsize=15,
        fontweight="bold",
    )
    ax_leg.text(0.0, 0.83, "Sign coding:", ha="left", va="top", fontsize=15, fontweight="bold")
    ax_leg.text(0.0, 0.78, "+ : network color (saturated)", ha="left", va="top", fontsize=14)
    ax_leg.text(0.0, 0.73, "- : pale tint of network color", ha="left", va="top", fontsize=14)

    net_counts: dict[str, dict[str, int]] = {}
    for item in selected_items:
        net = str(item.get("canonical8", "Unknown"))
        sign = str(item.get("sign", ""))
        row = net_counts.setdefault(net, {"positive": 0, "negative": 0})
        if sign == "positive":
            row["positive"] += 1
        elif sign == "negative":
            row["negative"] += 1

    if not net_counts:
        ax_leg.text(0.0, 0.65, "No significant ROI selected.", ha="left", va="top", fontsize=15)
    else:
        ax_leg.text(0.0, 0.65, "Network legend:", ha="left", va="top", fontsize=15, fontweight="bold")
        y = 0.61
        step = 0.050
        ordered_nets = sorted(net_counts.keys())
        for net in ordered_nets[:12]:
            base = network_colors.get(net, (128, 128, 128, 255))
            neg = neg_color_rgba(base)
            c_pos = tuple(x / 255.0 for x in base[:3]) + (1.0,)
            c_neg = tuple(x / 255.0 for x in neg[:3]) + (1.0,)
            ax_leg.add_patch(plt.Rectangle(
                (0.0, y - 0.018), 0.030, 0.026,
                transform=ax_leg.transAxes, facecolor=c_pos, edgecolor="none"))
            ax_leg.add_patch(plt.Rectangle(
                (0.038, y - 0.018), 0.030, 0.026,
                transform=ax_leg.transAxes, facecolor=c_neg, edgecolor="none"))
            c = net_counts[net]
            label = f"{net} (+{int(c['positive'])}/-{int(c['negative'])})"
            ax_leg.text(0.076, y, label, ha="left", va="center", fontsize=14, transform=ax_leg.transAxes)
            y -= step
            if y < 0.02:
                break

    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return out_png


def roi_stats_to_rows(stats_rows: list[ROIStat]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for row in stats_rows:
        out.append(
            {
                "roi_id": row.roi_id,
                "roi_name": row.roi_name,
                "hemi": row.hemi,
                "family17": row.family17,
                "canonical8": row.canonical8,
                "mean_z": row.mean_z,
                "t_stat": row.t_stat,
                "p_positive": row.p_positive,
                "p_negative": row.p_negative,
                "q_positive": row.q_positive,
                "q_negative": row.q_negative,
                "n_subjects": row.n_subjects,
                "is_sig_positive": row.is_sig_positive,
                "is_sig_negative": row.is_sig_negative,
                "is_selected": row.is_selected,
                "selected_sign": row.selected_sign,
            }
        )
    return out


def coverage_tag(tau: float) -> str:
    return f"cov_{tau:.2f}".replace(".", "p")


def combo_tag(combo: Combo) -> str:
    return f"{combo.atlas}__{combo.method}__{combo.smoothing}__hemi-{combo.hemi}"


def write_viewer_readme(
    *,
    viewer_root: Path,
    present_dir: Path,
    runtime_root: Path,
    items: list[dict[str, Any]],
    skipped: list[dict[str, Any]],
    grouped_overviews: list[dict[str, Any]],
) -> None:
    lines = [
        "# cluster_roi_cortex quicklook",
        "",
        "This directory is a shallow viewer root.",
        f"- runtime_root: `{runtime_root}`",
        f"- present_root: `{present_dir}`",
        f"- rendered combos: {len(items)}",
        f"- skipped combos: {len(skipped)}",
        "",
        "Quick entry points:",
        "- `present_root/cortex_*_overview.png` collages per atlas/method (clusterless merges 2mm+4mm into one no-smoothing filename; cluster methods remain per smoothing)",
        "- `cluster_roi_cortex_manifest.json` index linking viewer and runtime manifests",
        "",
    ]
    if grouped_overviews:
        lines.append("Top-level merged-hemisphere overviews:")
        for row in grouped_overviews:
            lines.append(f"- `{Path(str(row['overview_png'])).name}`")
        lines.append("")
    (viewer_root / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def compose_combo_overview(
    *,
    cluster_previews: list[dict[str, Any]],
    out_png: Path,
    title: str,
) -> None:
    if not cluster_previews:
        fig, ax = plt.subplots(1, 1, figsize=(8.0, 4.5), dpi=180)
        ax.axis("off")
        ax.text(0.5, 0.5, "No cluster preview", ha="center", va="center", fontsize=12)
        ax.set_title(title, fontsize=12)
        out_png.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_png, dpi=220, bbox_inches="tight")
        plt.close(fig)
        return

    n_panels = int(len(cluster_previews))
    if n_panels <= 2:
        cols = n_panels
    elif n_panels <= 4:
        cols = 2
    elif n_panels <= 9:
        cols = 3
    else:
        cols = 4
    rows = int(math.ceil(n_panels / float(cols)))

    # Preserve source aspect ratio to avoid vertical squeezing.
    ref_img = plt.imread(Path(str(cluster_previews[0]["panel_png"])))
    ref_aspect = float(ref_img.shape[1]) / float(ref_img.shape[0])  # w / h
    cell_w = 7.0
    cell_h = cell_w / max(ref_aspect, 1e-6)
    suptitle_h = 0.6
    fig_w = cols * cell_w
    fig_h = rows * cell_h + suptitle_h
    fig = plt.figure(figsize=(fig_w, fig_h), dpi=180, constrained_layout=True)
    gs = fig.add_gridspec(rows, cols, wspace=0.02, hspace=0.06)

    for idx, item in enumerate(cluster_previews):
        if idx >= rows * cols:
            break
        rr = idx // cols
        cc = idx % cols
        ax = fig.add_subplot(gs[rr, cc])
        png = Path(str(item["panel_png"]))
        img = plt.imread(png)
        ax.imshow(img, interpolation="nearest")
        ax.set_aspect("equal")
        label = str(item.get("panel_label", f"C{int(item['cluster_id']):02d}"))
        n_pos = int(item.get("n_selected_positive", 0))
        n_neg = int(item.get("n_selected_negative", 0))
        n_tot = int(n_pos + n_neg)
        ax.set_title(f"{label} | n={n_tot} (+{n_pos}/-{n_neg})", fontsize=14)
        ax.axis("off")

    fig.suptitle(title, fontsize=18)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    args = parse_args()

    group_root = Path(args.group_root).resolve()
    shared_root = Path(args.shared_root).resolve()
    present_dir = Path(args.present_dir).resolve()
    present_dir.mkdir(parents=True, exist_ok=True)
    runtime_root = Path(args.runtime_root).resolve() if args.runtime_root else (present_dir / "runtime")
    viewer_root = runtime_root
    runtime_root.mkdir(parents=True, exist_ok=True)
    viewer_combo_manifest_dir = runtime_root / "combo_manifests"
    viewer_combo_manifest_dir.mkdir(parents=True, exist_ok=True)

    data_root = Path(args.data_root).resolve()
    cortex_assets_root = Path(args.cortex_assets_root).resolve()
    scene = Path(args.scene).resolve()
    cortex_assets = resolve_cortex_assets(str(args.cortex_template_subject), data_root, cortex_assets_root, scene)

    tau_tag = coverage_tag(float(args.coverage_threshold))
    network_colors = load_shared_network_colors()
    bundle = load_kong2022_bundle()
    vertex_arrays = {
        "L": (bundle["vertex_arrays"]["L"][0].astype(np.int32), bundle["vertex_arrays"]["L"][1].astype(np.int32)),
        "R": (bundle["vertex_arrays"]["R"][0].astype(np.int32), bundle["vertex_arrays"]["R"][1].astype(np.int32)),
    }

    combos: list[Combo] = []
    for method in args.methods:
        for smoothing in args.smoothings:
            for hemi in args.hemis:
                combos.append(Combo(atlas=str(args.atlas), method=str(method), smoothing=str(smoothing), hemi=str(hemi)))

    items: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    grouped_previews: dict[tuple[str, str, str], list[dict[str, Any]]] = {}

    for combo in combos:
        run_dir = group_root / combo.method / combo.atlas / combo.smoothing / tau_tag / f"hemi_{combo.hemi}"
        manifest_path = run_dir / "group_atlas_manifest.json"
        labels_path = run_dir / "group_labels_final.npy"
        support_path = run_dir / "group_support_mask.npy"
        registry_path = run_dir / "cluster_roi_registry.tsv"
        vertex_status_path = run_dir / "vertex_assignment_status.json"
        global_top_path = run_dir / "global_top_connections.json"
        combo_id = combo_tag(combo)
        clusterless = combo.method in CLUSTERLESS_METHODS

        if not manifest_path.exists():
            skipped.append(
                {
                    "atlas": combo.atlas,
                    "method": combo.method,
                    "smoothing": combo.smoothing,
                    "hemi": combo.hemi,
                    "reason": "missing_group_manifest",
                    "run_dir": str(run_dir),
                }
            )
            continue

        manifest = load_json(manifest_path)
        group_status = str(manifest.get("status", "ok"))
        if group_status != "ok":
            skipped.append(
                {
                    "atlas": combo.atlas,
                    "method": combo.method,
                    "smoothing": combo.smoothing,
                    "hemi": combo.hemi,
                    "reason": "group_combo_not_ok",
                    "group_status": group_status,
                    "group_skip_reason": manifest.get("skip_reason"),
                    "run_dir": str(run_dir),
                }
            )
            continue

        required_ok = (
            (support_path.exists() and registry_path.exists() and vertex_status_path.exists() and global_top_path.exists())
            if clusterless
            else (labels_path.exists() and support_path.exists() and registry_path.exists())
        )
        if not required_ok:
            skipped.append(
                {
                    "atlas": combo.atlas,
                    "method": combo.method,
                    "smoothing": combo.smoothing,
                    "hemi": combo.hemi,
                    "reason": "missing_required_group_atlas_artifacts",
                    "run_dir": str(run_dir),
                }
            )
            continue

        subjects = [str(s).replace("sub-", "") for s in manifest.get("subjects", [])]
        if not subjects:
            skipped.append(
                {
                    "atlas": combo.atlas,
                    "method": combo.method,
                    "smoothing": combo.smoothing,
                    "hemi": combo.hemi,
                    "reason": "empty_subjects_in_manifest",
                    "run_dir": str(run_dir),
                }
            )
            continue

        if clusterless:
            support = np.load(support_path).astype(bool)
            if support.ndim != 1:
                raise RuntimeError(f"Invalid support shape in {run_dir}: {support.shape}")
            roi_registry = parse_roi_registry(registry_path)
            global_top = load_json(global_top_path)
            selected_items = dedupe_selected_items_from_global_top(global_top)
            combo_runtime_out = runtime_root / combo.atlas / combo.method / combo.smoothing / tau_tag / f"hemi_{combo.hemi}"
            combo_runtime_out.mkdir(parents=True, exist_ok=True)
            panel_dir = combo_runtime_out / "global_top"
            other_hemi = "R" if combo.hemi == "L" else "L"
            other_manifest_path = (
                group_root / combo.method / combo.atlas / combo.smoothing / tau_tag / f"hemi_{other_hemi}" / "group_atlas_manifest.json"
            )
            if not other_manifest_path.exists():
                raise FileNotFoundError(
                    f"Missing opposite-hemi manifest required for hippomaps rendering: {other_manifest_path}"
                )
            other_manifest = load_json(other_manifest_path)
            template_spec_raw = manifest.get("template_spec_path")
            if template_spec_raw is None or str(template_spec_raw).strip().lower() in {"", "none", "null"}:
                skipped.append(
                    {
                        "atlas": combo.atlas,
                        "method": combo.method,
                        "smoothing": combo.smoothing,
                        "hemi": combo.hemi,
                        "reason": "missing_template_spec_path_subject_consensus",
                        "run_dir": str(run_dir),
                    }
                )
                continue
            this_surface = Path(str(manifest["template_left_or_right_surface"]))
            other_surface = Path(str(other_manifest["template_left_or_right_surface"]))
            left_surface = this_surface if combo.hemi == "L" else other_surface
            right_surface = this_surface if combo.hemi == "R" else other_surface
            left_label, right_label, _legend_items = build_cluster_label_files(
                selected_items=selected_items,
                cluster_dir=panel_dir,
                vertex_arrays=vertex_arrays,
                network_colors=network_colors,
            )
            figure_png = render_clusterless_global_figure(
                cluster_dir=panel_dir,
                title=f"{combo.atlas} | {combo.method} | {combo.smoothing} | hemi-{combo.hemi}",
                left_label=left_label,
                right_label=right_label,
                cortex_assets=cortex_assets,
                network_colors=network_colors,
                global_top=global_top,
                support_mask=support,
                combo_hemi=combo.hemi,
                template_subject=str(manifest.get("template_subject", "")).replace("sub-", ""),
                template_spec_path=Path(str(template_spec_raw)),
                left_surface=left_surface,
                right_surface=right_surface,
            )
            n_pos = int(len(global_top.get("top_positive", [])))
            n_neg = int(len(global_top.get("top_negative", [])))
            cluster_previews = [
                {
                    "atlas": combo.atlas,
                    "method": combo.method,
                    "smoothing": combo.smoothing,
                    "hemi": combo.hemi,
                    "cluster_id": 0,
                    "runtime_png": str(figure_png),
                    "n_selected_positive": n_pos,
                    "n_selected_negative": n_neg,
                    "panel_label": f"{combo.smoothing}-{combo.hemi}-GlobalTop",
                }
            ]
            group_key = (combo.atlas, combo.method, "2mm+4mm")
            grouped_previews.setdefault(group_key, []).extend(cluster_previews)

            combo_manifest = {
                "atlas": combo.atlas,
                "method": combo.method,
                "smoothing": combo.smoothing,
                "hemi": combo.hemi,
                "coverage_threshold": float(args.coverage_threshold),
                "selection_policy": "global_top_connections_2N",
                "top_n_cap_per_sign": int(global_top.get("top_n_cap", 20)),
                "q_threshold": float(global_top.get("q_threshold", 0.05)),
                "n_subjects_used": int(len(subjects)),
                "subjects_used": [f"sub-{s}" for s in subjects],
                "panel_count": 1,
                "global_top": {
                    "n_positive": n_pos,
                    "n_negative": n_neg,
                    "figure_png": str(figure_png),
                    "source_json": str(global_top_path),
                },
            }
            combo_manifest_path = combo_runtime_out / "cluster_roi_cortex_combo_manifest.json"
            save_json(combo_manifest_path, combo_manifest)
            combo_view_manifest_path = viewer_combo_manifest_dir / f"{combo_id}.json"
            save_json(
                combo_view_manifest_path,
                {
                    "atlas": combo.atlas,
                    "method": combo.method,
                    "smoothing": combo.smoothing,
                    "hemi": combo.hemi,
                    "combo_id": combo_id,
                    "runtime_combo_manifest": str(combo_manifest_path),
                    "clusters": cluster_previews,
                },
            )
            items.append(
                {
                    "atlas": combo.atlas,
                    "method": combo.method,
                    "smoothing": combo.smoothing,
                    "hemi": combo.hemi,
                    "combo_manifest_runtime": str(combo_manifest_path),
                    "combo_manifest_viewer": str(combo_view_manifest_path),
                    "cluster_count": 0,
                    "n_subjects_used": int(len(subjects)),
                }
            )
            continue

        labels = np.load(labels_path).astype(np.int32)
        support = np.load(support_path).astype(bool)
        if labels.ndim != 1 or support.ndim != 1 or int(labels.size) != int(support.size):
            raise RuntimeError(f"Invalid labels/support shape in {run_dir}")

        roi_registry = parse_roi_registry(registry_path)
        profile_3d, used_subjects = compute_subject_cluster_roi_profiles(
            subjects=subjects,
            shared_root=shared_root,
            smoothing=combo.smoothing,
            hemi=combo.hemi,
            group_labels=labels,
            support_mask=support,
            roi_registry=roi_registry,
        )

        k_final = int(np.max(labels))
        combo_runtime_out = runtime_root / combo.atlas / combo.method / combo.smoothing / tau_tag / f"hemi_{combo.hemi}"
        combo_runtime_out.mkdir(parents=True, exist_ok=True)

        cluster_items: list[dict[str, Any]] = []
        cluster_previews: list[dict[str, Any]] = []
        for cluster_id in range(1, k_final + 1):
            cluster_dir = combo_runtime_out / f"cluster_C{cluster_id:02d}"
            cluster_vals = profile_3d[:, cluster_id - 1, :]
            stats_rows, selected_pos, selected_neg = analyze_cluster(
                cluster_values=cluster_vals,
                roi_registry=roi_registry,
                fdr_q=float(args.fdr_q),
                max_rois_per_sign=int(args.max_rois_per_sign),
            )
            selected_items = selected_pos + selected_neg
            left_label, right_label, _legend_items = build_cluster_label_files(
                selected_items=selected_items,
                cluster_dir=cluster_dir,
                vertex_arrays=vertex_arrays,
                network_colors=network_colors,
            )

            title = (
                f"cluster C{cluster_id:02d} | {combo.atlas} | {combo.method} | "
                f"{combo.smoothing} | hemi-{combo.hemi}"
            )
            subtitle = (
                f"one-sided one-sample t-test + BH-FDR(q<{float(args.fdr_q):.2f}); "
                f"sig-only; max {int(args.max_rois_per_sign)}/sign; N={len(used_subjects)}"
            )
            figure_png = render_cluster_figure(
                cluster_dir=cluster_dir,
                title=title,
                subtitle=subtitle,
                left_label=left_label,
                right_label=right_label,
                selected_items=selected_items,
                network_colors=network_colors,
                n_selected_positive=int(len(selected_pos)),
                n_selected_negative=int(len(selected_neg)),
                cortex_assets=cortex_assets,
            )

            rows_dict = roi_stats_to_rows(stats_rows)
            stats_tsv = cluster_dir / "cluster_roi_stats.tsv"
            stats_json = cluster_dir / "cluster_roi_stats.json"
            write_tsv(
                stats_tsv,
                rows_dict,
                [
                    "roi_id",
                    "roi_name",
                    "hemi",
                    "family17",
                    "canonical8",
                    "mean_z",
                    "t_stat",
                    "p_positive",
                    "p_negative",
                    "q_positive",
                    "q_negative",
                    "n_subjects",
                    "is_sig_positive",
                    "is_sig_negative",
                    "is_selected",
                    "selected_sign",
                ],
            )
            save_json(
                stats_json,
                {
                    "cluster_id": cluster_id,
                    "selected_positive": selected_pos,
                    "selected_negative": selected_neg,
                    "n_selected_total": int(len(selected_items)),
                    "rows": rows_dict,
                    },
                )

            cluster_previews.append(
                {
                    "atlas": combo.atlas,
                    "method": combo.method,
                    "smoothing": combo.smoothing,
                    "hemi": combo.hemi,
                    "cluster_id": int(cluster_id),
                    "runtime_png": str(figure_png),
                    "n_selected_positive": int(len(selected_pos)),
                    "n_selected_negative": int(len(selected_neg)),
                }
            )

            cluster_items.append(
                {
                    "cluster_id": cluster_id,
                    "figure_png": str(figure_png),
                    "stats_tsv": str(stats_tsv),
                    "stats_json": str(stats_json),
                    "n_selected_positive": int(len(selected_pos)),
                    "n_selected_negative": int(len(selected_neg)),
                }
            )

        group_key = (combo.atlas, combo.method, combo.smoothing)
        grouped_previews.setdefault(group_key, []).extend(cluster_previews)

        combo_manifest = {
            "atlas": combo.atlas,
            "method": combo.method,
            "smoothing": combo.smoothing,
            "hemi": combo.hemi,
            "coverage_threshold": float(args.coverage_threshold),
            "fdr_q": float(args.fdr_q),
            "selection_policy": "sig_only_topN_per_sign",
            "max_rois_per_sign": int(args.max_rois_per_sign),
            "n_subjects_used": int(len(used_subjects)),
            "subjects_used": [f"sub-{s}" for s in used_subjects],
            "cluster_count": int(k_final),
            "clusters": cluster_items,
        }
        combo_manifest_path = combo_runtime_out / "cluster_roi_cortex_combo_manifest.json"
        save_json(combo_manifest_path, combo_manifest)
        combo_view_manifest_path = viewer_combo_manifest_dir / f"{combo_id}.json"
        save_json(
            combo_view_manifest_path,
            {
                "atlas": combo.atlas,
                "method": combo.method,
                "smoothing": combo.smoothing,
                "hemi": combo.hemi,
                "combo_id": combo_id,
                "runtime_combo_manifest": str(combo_manifest_path),
                "clusters": cluster_previews,
            },
        )

        items.append(
            {
                "atlas": combo.atlas,
                "method": combo.method,
                "smoothing": combo.smoothing,
                "hemi": combo.hemi,
                "combo_manifest_runtime": str(combo_manifest_path),
                "combo_manifest_viewer": str(combo_view_manifest_path),
                "cluster_count": int(k_final),
                "n_subjects_used": int(len(used_subjects)),
            }
        )

    grouped_overviews: list[dict[str, Any]] = []
    smoothing_rank = {"2mm": 0, "4mm": 1, "2mm+4mm": 2}
    for (atlas, method, smoothing), rows in sorted(grouped_previews.items(), key=lambda x: x[0]):
        merged_rows = sorted(
            rows,
            key=lambda r: (
                int(smoothing_rank.get(str(r.get("smoothing", "")), 99)),
                str(r.get("hemi", "")),
                int(r.get("cluster_id", 0)),
            ),
        )
        previews_for_plot: list[dict[str, Any]] = []
        for r in merged_rows:
            previews_for_plot.append(
                {
                    "cluster_id": int(r["cluster_id"]),
                    "panel_png": str(r["runtime_png"]),
                    "panel_label": str(r.get("panel_label", f"{str(r['hemi'])}-C{int(r['cluster_id']):02d}")),
                    "n_selected_positive": int(r.get("n_selected_positive", 0)),
                    "n_selected_negative": int(r.get("n_selected_negative", 0)),
                }
            )
        if str(smoothing) == "2mm+4mm":
            overview_png = present_dir / f"cortex_{atlas}_{method}_overview.png"
            title_txt = f"{atlas} | {method} | 2mm+4mm merged | L+R merged"
        else:
            overview_png = present_dir / f"cortex_{atlas}_{method}_{smoothing}_overview.png"
            title_txt = f"{atlas} | {method} | {smoothing} | L+R merged"
        compose_combo_overview(
            cluster_previews=previews_for_plot,
            out_png=overview_png,
            title=title_txt,
        )
        grouped_overviews.append(
            {
                "atlas": atlas,
                "method": method,
                "smoothing": smoothing,
                "overview_png": str(overview_png),
                "n_panels": int(len(previews_for_plot)),
            }
        )

    runtime_manifest = {
        "stage": "cluster_roi_cortex_independent",
        "atlas": str(args.atlas),
        "coverage_threshold": float(args.coverage_threshold),
        "fdr_q": float(args.fdr_q),
        "max_rois_per_sign": int(args.max_rois_per_sign),
        "runtime_root": str(runtime_root),
        "viewer_root": str(viewer_root),
        "count": int(len(items)),
        "items": items,
        "skipped": skipped,
        "grouped_overviews": grouped_overviews,
    }
    runtime_manifest_path = runtime_root / "cluster_roi_cortex_runtime_manifest.json"
    save_json(runtime_manifest_path, runtime_manifest)

    viewer_manifest = dict(runtime_manifest)
    viewer_manifest["runtime_manifest"] = str(runtime_manifest_path)
    viewer_manifest["viewer_layout"] = {
        "present_root_overview_pattern": str(present_dir / "cortex_*_overview.png"),
        "combo_manifest_dir": str(viewer_combo_manifest_dir),
    }
    viewer_manifest_path = viewer_root / "cluster_roi_cortex_manifest.json"
    save_json(viewer_manifest_path, viewer_manifest)
    present_overview_manifest_path = present_dir / "cluster_roi_cortex_overview_manifest.json"
    save_json(
        present_overview_manifest_path,
        {
            "stage": "cluster_roi_cortex_overview",
            "viewer_manifest": str(viewer_manifest_path),
            "runtime_manifest": str(runtime_manifest_path),
            "grouped_overviews": grouped_overviews,
            "count": int(len(grouped_overviews)),
        },
    )
    write_viewer_readme(
        viewer_root=viewer_root,
        present_dir=present_dir,
        runtime_root=runtime_root,
        items=items,
        skipped=skipped,
        grouped_overviews=grouped_overviews,
    )
    print(
        json.dumps(
            {
                "runtime_root": str(runtime_root),
                "viewer_root": str(viewer_root),
                "present_root": str(present_dir),
                "count": len(items),
                "skipped": len(skipped),
                "grouped_overviews": len(grouped_overviews),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
