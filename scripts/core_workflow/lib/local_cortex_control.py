"""Local-cortex control diagnostics for hippocampal-cortical FC.

These helpers quantify how hippocampus-to-Kong2022-ROI FC and the
normalized top-20 winner-take-all (WTA) assignment respond to two local
cortex controls, without materializing a signal branch:

* ctxreg: regress the mean of valid cortical vertices within a radius out of
  each hippocampal vertex time series, separately inside every run.
* ctxdrop: exclude, per hippocampal vertex, cortical vertices within a radius
  from the ROI averages used as FC targets.

They also describe the hippocampus-cortex boundary (distance-ring
correlations and boundary identity). All series are run-mean-centered before
correlation, matching the pooled FC used by the formal workflow.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence, cast

import nibabel as nib
import numpy as np
from scipy import sparse

from .extract_cortex_kong2022_route_a_roi_timeseries import (
    align_surface_labels_to_dt_vertices,
    load_surface_label_vector,
    route_a_label_paths,
)
from .extract_cortex_merged_network_timeseries import extract_structure_data
from .path_templates import format_subject_path_template
from .surface_coordinates import load_surface_coords

RING_EDGES_MM = (0.0, 3.0, 6.0, 10.0, 15.0, 20.0)
CONTROL_RADII_MM = (3.0, 6.0, 10.0, 20.0)
TOP_N = 20


@dataclass(frozen=True)
class CortexInputs:
    coords: np.ndarray        # (n_cortex, 3) in the dtseries/surface physical space
    ts: np.ndarray            # (n_cortex, T)
    roi_ids: np.ndarray       # (n_cortex,) Kong2022 parcel id per cortical grayordinate
    counts: dict[str, int]    # grayordinates per cortical structure
    paths: dict[str, Path]


def load_cortex_with_roi_labels(
    *,
    data_root: Path,
    subject: str,
    route_a_root: Path,
    left_cortex_surface_template: str,
    right_cortex_surface_template: str,
    dtseries_template: str,
) -> CortexInputs:
    """Load cortical grayordinates, midthickness coordinates and individual Kong2022 parcel ids."""
    subject = str(subject).replace("sub-", "")
    dt_path = format_subject_path_template(dtseries_template, data_root=data_root, subject=subject)
    img = cast(Any, nib.load(str(dt_path)))
    data = np.asarray(img.dataobj, dtype=np.float32)
    if data.ndim != 2:
        raise ValueError(f"Expected 2D dtseries data: {dt_path} shape={data.shape}")
    axis = img.header.get_axis(1)
    left_vertices, left_ts = extract_structure_data(axis, data.T, "CORTEX_LEFT")
    right_vertices, right_ts = extract_structure_data(axis, data.T, "CORTEX_RIGHT")
    lh_labels, rh_labels = route_a_label_paths(route_a_root=route_a_root, subject=subject)
    left_ids = align_surface_labels_to_dt_vertices(load_surface_label_vector(lh_labels), left_vertices, hemi="L")
    right_ids = align_surface_labels_to_dt_vertices(load_surface_label_vector(rh_labels), right_vertices, hemi="R")
    surfaces = {
        "L": format_subject_path_template(left_cortex_surface_template, data_root=data_root, subject=subject, hemi="L"),
        "R": format_subject_path_template(right_cortex_surface_template, data_root=data_root, subject=subject, hemi="R"),
    }
    left_xyz = load_surface_coords(surfaces["L"])
    right_xyz = load_surface_coords(surfaces["R"])
    if int(np.max(left_vertices)) >= left_xyz.shape[0] or int(np.max(right_vertices)) >= right_xyz.shape[0]:
        raise ValueError(f"dtseries vertex index exceeds surface rows: {dt_path} vs {surfaces}")
    return CortexInputs(
        coords=np.vstack([left_xyz[left_vertices], right_xyz[right_vertices]]).astype(np.float64),
        ts=np.vstack([left_ts, right_ts]).astype(np.float32),
        roi_ids=np.concatenate([left_ids, right_ids]).astype(np.int32),
        counts={"CORTEX_LEFT": int(left_ts.shape[0]), "CORTEX_RIGHT": int(right_ts.shape[0])},
        paths={"dtseries": Path(dt_path).resolve(), "left_surface": surfaces["L"], "right_surface": surfaces["R"],
               "route_a_labels": lh_labels.parent.resolve()},
    )


def zscore_rows(x: np.ndarray) -> np.ndarray:
    """Center and unit-norm each row (dot product of two rows = Pearson r); constant rows -> NaN."""
    x = np.asarray(x, dtype=np.float64)
    x = x - x.mean(axis=1, keepdims=True)
    norm = np.linalg.norm(x, axis=1, keepdims=True)
    with np.errstate(invalid="ignore", divide="ignore"):
        return x / np.where(norm > 0, norm, np.nan)


def runwise_residual(y: np.ndarray, x: np.ndarray, run_lengths: Sequence[int]) -> np.ndarray:
    """Per row and per run, OLS residual of run-centered y on run-centered x (intercept implied).

    Equivalent to 04_00's per-run ``y ~ x + intercept`` fit applied to
    run-centered inputs. Rows with an all-zero regressor are returned unchanged.
    """
    y = np.asarray(y, dtype=np.float64)
    x = np.asarray(x, dtype=np.float64)
    if y.shape != x.shape or sum(int(n) for n in run_lengths) != y.shape[1]:
        raise ValueError(f"Residualization shape mismatch: y={y.shape} x={x.shape} runs={list(run_lengths)}")
    out = np.empty_like(y)
    start = 0
    for n in run_lengths:
        window = slice(start, start + int(n))
        yy = y[:, window] - y[:, window].mean(axis=1, keepdims=True)
        xx = x[:, window] - x[:, window].mean(axis=1, keepdims=True)
        denom = np.sum(xx * xx, axis=1)
        beta = np.divide(np.sum(yy * xx, axis=1), denom, out=np.zeros_like(denom), where=denom > 0)
        out[:, window] = yy - beta[:, None] * xx
        start += int(n)
    return out


def normalized_top20_winner(fc: np.ndarray, roi_network: np.ndarray, network_names: Sequence[str], top_n: int = TOP_N) -> np.ndarray:
    """Top-1 network per row under the 05_02 ``normalized_top20_count`` rule; -1 = unassigned.

    The strongest ``top_n`` positive finite ROI values are counted per network,
    each count is divided by that row's finite available ROIs in the network,
    and ties are broken by summed FC, then count, then network name.
    """
    fc = np.asarray(fc, dtype=np.float64)
    roi_network = np.asarray(roi_network, dtype=np.int64)
    n_net = len(network_names)
    out = np.full(fc.shape[0], -1, dtype=np.int16)
    for v in range(fc.shape[0]):
        vals = fc[v]
        finite = np.isfinite(vals)
        pos = np.flatnonzero(finite & (vals > 0))
        if pos.size == 0:
            continue
        available = np.bincount(roi_network[finite], minlength=n_net)
        top = pos[np.argsort(-vals[pos], kind="stable")][:top_n]
        count = np.bincount(roi_network[top], minlength=n_net)
        score = np.bincount(roi_network[top], weights=vals[top], minlength=n_net)
        nets = np.flatnonzero(count > 0)
        norm = count[nets] / np.maximum(available[nets], 1)
        order = sorted(range(nets.size), key=lambda i: (-norm[i], -score[nets[i]], -count[nets[i]], network_names[nets[i]]))
        out[v] = nets[order[0]]
    return out


@dataclass(frozen=True)
class RoiBank:
    columns: np.ndarray       # (n_usable,) ROI column for each usable cortical row
    usable_rows: np.ndarray   # cortical row indices with a valid mask and a registered ROI
    sums: np.ndarray          # (n_roi, T)
    counts: np.ndarray        # (n_roi,)

    @property
    def means(self) -> np.ndarray:
        with np.errstate(invalid="ignore", divide="ignore"):
            return self.sums / self.counts[:, None]


def build_roi_bank(ctx: np.ndarray, roi_ids: np.ndarray, valid: np.ndarray, parcel_ids: Sequence[int]) -> RoiBank:
    col_of = {int(pid): i for i, pid in enumerate(parcel_ids)}
    cols = np.array([col_of.get(int(i), -1) for i in roi_ids], dtype=np.int64)
    usable = np.flatnonzero(np.asarray(valid, dtype=bool) & (cols >= 0))
    n_roi = len(parcel_ids)
    assign = sparse.csr_matrix((np.ones(usable.size), (cols[usable], np.arange(usable.size))), shape=(n_roi, usable.size))
    return RoiBank(
        columns=cols[usable],
        usable_rows=usable,
        sums=np.asarray(assign @ np.asarray(ctx, dtype=np.float64)[usable]),
        counts=np.bincount(cols[usable], minlength=n_roi).astype(np.float64),
    )


def ctxdrop_fc(
    *,
    hipp_z: np.ndarray,
    fc_standard: np.ndarray,
    ctx: np.ndarray,
    bank: RoiBank,
    drop_rows: list[np.ndarray],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """FC after removing ``drop_rows[v]`` (cortical rows) from the ROI averages of hippocampal vertex v.

    ``hipp_z`` must be ``zscore_rows`` of run-centered hippocampal series and
    ``fc_standard`` its FC to ``bank.means``. Only ROIs that lose vertices are
    recomputed; an ROI that loses every vertex becomes NaN for that vertex.
    Returns (fc, n_touched_rois, removed_fraction_of_touched_rois_median).
    """
    fc = np.array(fc_standard, dtype=np.float64, copy=True)
    n_touched = np.zeros(fc.shape[0], dtype=np.int32)
    removed_frac = np.full(fc.shape[0], np.nan)
    col_of_row = np.full(ctx.shape[0], -1, dtype=np.int64)
    col_of_row[bank.usable_rows] = bank.columns
    for v, rows in enumerate(drop_rows):
        rows = np.asarray(rows, dtype=np.int64)
        rows = rows[col_of_row[rows] >= 0] if rows.size else rows
        if rows.size == 0:
            continue
        cols, inverse = np.unique(col_of_row[rows], return_inverse=True)
        dropped = np.zeros((cols.size, ctx.shape[1]))
        np.add.at(dropped, inverse, ctx[rows])
        n_dropped = np.bincount(inverse, minlength=cols.size).astype(np.float64)
        remain = bank.counts[cols] - n_dropped
        with np.errstate(invalid="ignore", divide="ignore"):
            new_means = (bank.sums[cols] - dropped) / remain[:, None]
        new_means[remain <= 0] = np.nan
        fc[v, cols] = zscore_rows(new_means) @ hipp_z[v]
        n_touched[v] = cols.size
        removed_frac[v] = float(np.median(n_dropped / bank.counts[cols]))
    return fc, n_touched, removed_frac


def boundary_groups(
    *,
    cortex_to_hipp_mm: np.ndarray,
    hipp_to_cortex_mm: np.ndarray,
    cortex_is_dmn: np.ndarray,
    hipp_valid: np.ndarray,
    near_mm: float = 3.0,
    deep_mm: float = 4.0,
    dmn_mid_mm: tuple[float, float] = (20.0, 30.0),
    dmn_far_mm: float = 30.0,
) -> dict[str, np.ndarray]:
    """Index sets used to ask whether boundary signal behaves like cortex or hippocampus."""
    return {
        "cortex_near": np.flatnonzero(cortex_to_hipp_mm <= near_mm),
        "dmn_mid": np.flatnonzero(cortex_is_dmn & (cortex_to_hipp_mm > dmn_mid_mm[0]) & (cortex_to_hipp_mm <= dmn_mid_mm[1])),
        "dmn_far": np.flatnonzero(cortex_is_dmn & (cortex_to_hipp_mm > dmn_far_mm)),
        "hipp_edge": np.flatnonzero(hipp_valid & (hipp_to_cortex_mm <= near_mm)),
        "hipp_deep": np.flatnonzero(hipp_valid & (hipp_to_cortex_mm > deep_mm)),
    }


def pearson(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float64) - np.mean(a)
    b = np.asarray(b, dtype=np.float64) - np.mean(b)
    denom = float(np.sqrt((a @ a) * (b @ b)))
    return float(a @ b / denom) if denom > 0 else float("nan")
