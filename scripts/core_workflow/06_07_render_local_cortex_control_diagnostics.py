#!/usr/bin/env python3
"""
Process Step Description:
Quantify and render how hippocampus-to-cortex FC responds to local-cortex controls (HCP 7T, Kong2022 PFM,
canonical7). This is an FC/WTA-level diagnostic, not a signal branch: nothing here feeds group atlases.

For each subject and hemisphere it computes, at radii 3, 6, 10 and 20 mm (native T1w distances):
- ctxreg: per-run regression of the mean of valid cortex within the radius out of each hippocampal vertex;
- ctxdrop: per-vertex exclusion of cortex within the radius from the 400 Kong2022 ROI averages used as FC targets;
- the resulting 400-ROI FC change and the normalized top-20 WTA winner (05_02 rule) versus standard FC;
- the network make-up of the local cortex, hippocampus-to-cortex correlation by distance ring, and a
  boundary identity test (does cortex within 3 mm of the hippocampus behave like DMN cortex or like hippocampus?).

Stages:
- --stage subjects: per-subject arrays under `outputs_hcp/work/local_cortex_control/sub-<id>/`.
- --stage render: requires every roster subject, then writes the figure, summary TSV and manifest under
  `outputs_hcp/present/local_cortex_control/`. The figure shows the ctxreg diagnostics only (10 and 20 mm maps;
  regressor make-up, distance-ring coupling, network FC change and WTA share). Target-side exclusion (ctxdrop, a
  retired branch) and the boundary identity test are recorded in the summary TSV, not drawn.

Inputs: QC-pass roster; run-concat dtseries and MSMAll midthickness surfaces under `data/hippunfold_input`;
individual Kong2022 PFM labels; PFM reference (ROI registry, run metadata, cortex validity masks); hippocampal
surface series and validity masks under `shared_inputs/surface`; native T1w den-512 hippocampal surfaces.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import tempfile
import textwrap
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

from lib.fc_math import center_run_timeseries
from lib.io_utils import load_optional_cortex_valid_mask, load_subject_ids, segments_from_explicit_metadata
from lib.local_cortex_control import (
    CONTROL_RADII_MM,
    RING_EDGES_MM,
    boundary_groups,
    build_roi_bank,
    ctxdrop_fc,
    load_cortex_with_roi_labels,
    normalized_top20_winner,
    pearson,
    runwise_residual,
    zscore_rows,
)
from lib.output_layout import (
    DEFAULT_CORTEX_ROUTEA_ROOT,
    DEFAULT_HIPPUNFOLD_BATCH_ROOT,
    DEFAULT_KONG2022_MANIFEST_ROOT,
    DEFAULT_KONG2022_PFM_ROOT,
    DEFAULT_KONG2022_SURFACE_ROOT,
    DEFAULT_LOCAL_CORTEX_CONTROL_PRESENT_DIR,
    DEFAULT_LOCAL_CORTEX_CONTROL_WORK_ROOT,
)
from lib.surface_coordinates import MNI_SPACE, hipp_surface_path, load_surface_coords, native_cortex_coordinates
from lib.workflow_utils import capped_jobs, utc_now_iso, write_json

REPO_ROOT = Path(__file__).resolve().parents[2]
SCHEMA = "local_cortex_control_diagnostics.v1"
NETWORKS = ("Default", "Visual", "Somatomotor", "DorsalAttention", "VentralAttention", "Control", "Language")
SMOOTHINGS = ("2mm", "4mm")
HEMIS = ("L", "R")
MAP_RADII_MM = (10.0, 20.0)
MAP_SMOOTHING = "2mm"
DEFAULT_SUBJECTS = DEFAULT_KONG2022_MANIFEST_ROOT / "hcp_7t_hippocampus_fmri_qc_pass_t1only_latest.txt"
DEFAULT_DATA_ROOT = REPO_ROOT / "data" / "hippunfold_input"
DEFAULT_DTSERIES_TEMPLATE = "{data_root}/{sub_subject}/func/{sub_subject}_task-rest_run-concat.dtseries.nii"
DEFAULT_LEFT_SURFACE = "{data_root}/{sub_subject}/anat/{sub_subject}_hemi-L_space-fsLR_den-32k_desc-MSMAll_midthickness.surf.gii"
DEFAULT_RIGHT_SURFACE = "{data_root}/{sub_subject}/anat/{sub_subject}_hemi-R_space-fsLR_den-32k_desc-MSMAll_midthickness.surf.gii"
HIPPOCAMPAL_VERTEX_COUNT = 512


def rtag(r: float) -> str:
    return f"{int(r)}mm"


def subject_paths(work_root: Path, subject: str) -> dict[str, Path]:
    d = Path(work_root) / f"sub-{subject}"
    return {"dir": d, "npz": d / "local_cortex_control.npz", "json": d / "local_cortex_control.json"}


def fingerprint(path: Path) -> dict[str, object]:
    path = Path(path).resolve()
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    return {"path": str(path), "sha256": digest, "size_bytes": path.stat().st_size}


def read_parcels(reference_root: Path, subject: str) -> list[dict[str, str]]:
    path = Path(reference_root) / f"sub-{subject}" / "reference" / "kong2022" / "cortex_roi_parcels.tsv"
    with path.open(encoding="utf-8", newline="") as fh:
        rows = list(csv.DictReader(fh, delimiter="\t"))
    if len(rows) != 400:
        raise ValueError(f"Kong2022 ROI registry must list 400 parcels: {path} has {len(rows)}")
    unknown = sorted({r["canonical7"] for r in rows} - set(NETWORKS))
    if unknown:
        raise ValueError(f"Unexpected canonical7 networks {unknown} in {path}")
    return rows


def process_subject(subject: str, cfg: dict[str, str]) -> dict[str, object]:
    s = str(subject).replace("sub-", "")
    paths = subject_paths(Path(cfg["work_root"]), s)
    if paths["json"].is_file() and paths["npz"].is_file() and not cfg["overwrite"]:
        meta = json.loads(paths["json"].read_text(encoding="utf-8"))
        if meta.get("schema") == SCHEMA:
            return {"subject": f"sub-{s}", "status": "reused"}
        raise ValueError(f"Existing diagnostics use schema {meta.get('schema')!r}; rerun with --overwrite: {paths['json']}")
    reference_root, surface_root = Path(cfg["reference_root"]), Path(cfg["surface_root"])
    cortex = load_cortex_with_roi_labels(
        data_root=Path(cfg["data_root"]), subject=s, route_a_root=Path(cfg["route_a_root"]),
        left_cortex_surface_template=cfg["left_surface"], right_cortex_surface_template=cfg["right_surface"],
        dtseries_template=cfg["dtseries_template"],
    )
    coords = native_cortex_coordinates(cortex.coords, subject=s, space=MNI_SPACE)
    valid = load_optional_cortex_valid_mask(reference_root, s, cortex.counts)
    run_metadata = reference_root / f"sub-{s}" / "reference" / "kong2022" / "reference_summary.json"
    lengths = [int(seg["run_length"]) for seg in segments_from_explicit_metadata(run_metadata)]
    ctx = center_run_timeseries(cortex.ts, lengths).astype(np.float64)
    parcels = read_parcels(reference_root, s)
    parcel_ids = [int(r["parcel_id"]) for r in parcels]
    roi_net = np.array([NETWORKS.index(r["canonical7"]) for r in parcels])
    net_of_pid = dict(zip(parcel_ids, roi_net))
    vert_net = np.array([net_of_pid.get(int(i), -1) for i in cortex.roi_ids])
    bank = build_roi_bank(ctx, cortex.roi_ids, valid, parcel_ids)
    roi_z = zscore_rows(bank.means)
    valid_rows = np.flatnonzero(valid)
    tree = cKDTree(coords[valid_rows])
    is_dmn = vert_net == 0

    arrays: dict[str, np.ndarray] = {"roi_network": roi_net}
    inputs: dict[str, object] = {"dtseries": str(cortex.paths["dtseries"]), "run_metadata": fingerprint(run_metadata)}
    for hemi in HEMIS:
        hipp_surface = hipp_surface_path(Path(cfg["hippunfold_root"]), s, hemi)
        hc = load_surface_coords(hipp_surface).astype(np.float64)
        if hc.shape[0] != HIPPOCAMPAL_VERTEX_COUNT:
            raise ValueError(f"Expected den-512 hippocampal surface: {hipp_surface} has {hc.shape[0]} vertices")
        mask_path = surface_root / f"sub-{s}/surface/validity/sub-{s}_hemi-{hemi}_valid_mask.npy"
        hm = np.load(mask_path).astype(bool)
        inputs[f"hippocampal_surface_{hemi}"] = fingerprint(hipp_surface)
        inputs[f"hippocampal_valid_mask_{hemi}"] = fingerprint(mask_path)
        neighbours = tree.query_ball_point(hc, r=max(CONTROL_RADII_MM))
        n_r, n_ring = len(CONTROL_RADII_MM), len(RING_EDGES_MM) - 1
        disc_n = np.zeros((n_r, hc.shape[0]), np.int32)
        composition = np.full((n_r, hc.shape[0], len(NETWORKS) + 1), np.nan)
        disc_mean = np.zeros((n_r, hc.shape[0], ctx.shape[1]))
        ring_mean = np.full((n_ring, hc.shape[0], ctx.shape[1]), np.nan)
        drop_rows: list[list[np.ndarray]] = [[] for _ in CONTROL_RADII_MM]
        for v, row in enumerate(neighbours):
            rows = valid_rows[np.asarray(row, dtype=np.int64)]
            d = np.linalg.norm(coords[rows] - hc[v], axis=1)
            for i, r in enumerate(CONTROL_RADII_MM):
                sel = rows[d <= r]
                drop_rows[i].append(sel)
                disc_n[i, v] = sel.size
                if sel.size:
                    disc_mean[i, v] = ctx[sel].mean(axis=0)
                    nets = vert_net[sel]
                    composition[i, v] = np.bincount(np.where(nets < 0, len(NETWORKS), nets), minlength=len(NETWORKS) + 1) / sel.size
            for k in range(n_ring):
                sel = rows[(d >= RING_EDGES_MM[k]) & (d < RING_EDGES_MM[k + 1])]
                if sel.size:
                    ring_mean[k, v] = ctx[sel].mean(axis=0)
        arrays[f"{hemi}_valid"] = hm
        arrays[f"{hemi}_disc_n"] = disc_n
        arrays[f"{hemi}_composition"] = composition
        # boundary identity groups
        dist_c, _ = cKDTree(hc).query(coords[valid_rows], k=1)
        dist_h, _ = cKDTree(coords[valid_rows]).query(hc, k=1)
        groups = boundary_groups(cortex_to_hipp_mm=dist_c, hipp_to_cortex_mm=dist_h,
                                 cortex_is_dmn=is_dmn[valid_rows], hipp_valid=hm)
        arrays[f"{hemi}_boundary_group_sizes"] = np.array([groups[k].size for k in
                                                          ("cortex_near", "dmn_mid", "dmn_far", "hipp_edge", "hipp_deep")])
        for sm in SMOOTHINGS:
            ts_path = surface_root / f"sub-{s}/surface/{sm}/sub-{s}_hemi-{hemi}_timeseries.npy"
            inputs[f"hippocampal_timeseries_{hemi}_{sm}"] = fingerprint(ts_path)
            y = center_run_timeseries(np.load(ts_path), lengths).astype(np.float64)
            if y.shape[0] != HIPPOCAMPAL_VERTEX_COUNT:
                raise ValueError(f"Hippocampal series/surface mismatch: {ts_path} rows={y.shape[0]}")
            yz = zscore_rows(y)
            fc_std = np.where(hm[:, None], yz @ roi_z.T, np.nan)
            key = f"{hemi}_{sm}"
            arrays[f"{key}_standard_wta"] = normalized_top20_winner(fc_std, roi_net, NETWORKS)
            arrays[f"{key}_ring_r"] = np.array([np.sum(yz * zscore_rows(ring_mean[k]), axis=1) for k in range(n_ring)])
            for i, r in enumerate(CONTROL_RADII_MM):
                # ctxreg
                res = runwise_residual(y, disc_mean[i], lengths)
                fc = np.where(hm[:, None], zscore_rows(res) @ roi_z.T, np.nan)
                tag = f"{key}_ctxreg_{rtag(r)}"
                arrays[f"{tag}_wta"] = normalized_top20_winner(fc, roi_net, NETWORKS)
                arrays[f"{tag}_dfc_roi"] = np.nanmean((fc - fc_std)[hm], axis=0)
                has = disc_n[i] > 0
                local_r = np.full(y.shape[0], np.nan)
                local_r[has] = np.sum(yz[has] * zscore_rows(disc_mean[i][has]), axis=1)
                arrays[f"{tag}_local_r"] = local_r
                # ctxdrop
                fc, n_touched, removed = ctxdrop_fc(hipp_z=yz, fc_standard=fc_std, ctx=ctx, bank=bank, drop_rows=drop_rows[i])
                fc = np.where(hm[:, None], fc, np.nan)
                tag = f"{key}_ctxdrop_{rtag(r)}"
                arrays[f"{tag}_wta"] = normalized_top20_winner(fc, roi_net, NETWORKS)
                arrays[f"{tag}_dfc_roi"] = np.nanmean((fc - fc_std)[hm], axis=0)
                if sm == SMOOTHINGS[0]:
                    arrays[f"{hemi}_ctxdrop_{rtag(r)}_n_touched_rois"] = n_touched
                    arrays[f"{hemi}_ctxdrop_{rtag(r)}_removed_fraction"] = removed
            if all(groups[k].size for k in groups):
                cn, dm, df = (ctx[valid_rows[groups[k]]].mean(axis=0) for k in ("cortex_near", "dmn_mid", "dmn_far"))
                he, hd = y[groups["hipp_edge"]].mean(axis=0), y[groups["hipp_deep"]].mean(axis=0)
                arrays[f"{key}_boundary_identity"] = np.array([
                    [pearson(dm, df), pearson(dm, hd)],
                    [pearson(cn, df), pearson(cn, hd)],
                    [pearson(he, df), pearson(he, hd)],
                    [pearson(hd, df), np.nan],
                ])
                arrays[f"{key}_cortex_near_vs_hipp_edge_r"] = np.array(pearson(cn, he))
    paths["dir"].mkdir(parents=True, exist_ok=True)
    np.savez_compressed(paths["npz"], **arrays)
    write_json(paths["json"], {
        "schema": SCHEMA, "subject": f"sub-{s}", "timestamp_utc": utc_now_iso(),
        "radii_mm": list(CONTROL_RADII_MM), "ring_edges_mm": list(RING_EDGES_MM), "networks": list(NETWORKS),
        "smoothings": list(SMOOTHINGS), "distance_space": "T1w", "cortex_source_space": MNI_SPACE,
        "wta_rule": "05_02 normalized_top20_count top1 (finite available ROIs per network)",
        "n_runs": len(lengths), "run_lengths": lengths, "inputs": inputs,
        "boundary_identity_rows": ["dmn_cortex_20_30mm", "cortex_within_3mm", "hippocampus_edge_3mm", "hippocampus_deep_4mm"],
        "boundary_identity_columns": ["r_with_dmn_cortex_beyond_30mm", "r_with_deep_hippocampus"],
    }, trailing_newline=True)
    return {"subject": f"sub-{s}", "status": "computed"}


# ----------------------------------------------------------------------------- render


def load_cohort(work_root: Path, subjects: list[str]) -> dict[str, dict[str, np.ndarray]]:
    missing = [s for s in subjects if not subject_paths(work_root, s)["npz"].is_file()]
    if missing:
        raise FileNotFoundError(f"Missing local-cortex control diagnostics for {len(missing)} roster subjects under "
                                f"{work_root}: {missing[:5]}; run --stage subjects first")
    out = {}
    for s in subjects:
        meta = json.loads(subject_paths(work_root, s)["json"].read_text(encoding="utf-8"))
        if meta.get("schema") != SCHEMA:
            raise ValueError(f"Diagnostics schema mismatch for sub-{s}: {meta.get('schema')!r} != {SCHEMA!r}")
        out[s] = dict(np.load(subject_paths(work_root, s)["npz"]))
    return out


def summarize(Z: dict[str, dict[str, np.ndarray]]) -> list[dict[str, object]]:
    """Long table of every diagnostic, including the target-side (ctxdrop) and boundary evidence not drawn."""
    rows: list[dict[str, object]] = []
    zs = list(Z.values())
    roi_net = zs[0]["roi_network"]

    def row(control, radius, sm, metric, value):
        rows.append({"control": control, "radius_mm": "" if radius is None else int(radius), "smoothing": sm,
                     "metric": metric, "value": float(value)})

    def pooled_change(sm: str, tag: str) -> float:
        num = den = 0
        for z in zs:
            for h in HEMIS:
                v = z[f"{h}_valid"]
                num += int(np.sum(z[f"{h}_{sm}_{tag}_wta"][v] != z[f"{h}_{sm}_standard_wta"][v]))
                den += int(v.sum())
        return num / den

    def share(sm: str, tag: str) -> np.ndarray:
        lab = np.concatenate([z[f"{h}_{sm}_{tag}_wta"][z[f"{h}_valid"]] for z in zs for h in HEMIS])
        lab = lab[lab >= 0]
        return np.bincount(lab, minlength=len(NETWORKS)) / lab.size

    for sm in SMOOTHINGS:
        for k, n in enumerate(NETWORKS):
            row("standard", None, sm, f"wta_share_{n}", share(sm, "standard")[k])
        for control in ("ctxreg", "ctxdrop"):
            for r in CONTROL_RADII_MM:
                tag = f"{control}_{rtag(r)}"
                row(control, r, sm, "wta_change_fraction", pooled_change(sm, tag))
                sh = share(sm, tag)
                dfc = np.mean([[np.nanmean(z[f"{h}_{sm}_{tag}_dfc_roi"][roi_net == k]) for k in range(len(NETWORKS))]
                               for z in zs for h in HEMIS], axis=0)
                for k, n in enumerate(NETWORKS):
                    row(control, r, sm, f"wta_share_{n}", sh[k])
                    row(control, r, sm, f"mean_dfc_{n}", dfc[k])
                if control == "ctxreg":
                    lr = [np.nanmedian(z[f"{h}_{sm}_{tag}_local_r"][z[f"{h}_valid"]]) for z in zs for h in HEMIS]
                    row(control, r, sm, "median_r_hippocampus_regressor", np.nanmedian(lr))
        rings = np.array([np.nanmedian(z[f"{h}_{sm}_ring_r"][:, z[f"{h}_valid"]], axis=1) for z in zs for h in HEMIS])
        for k, (lo, hi) in enumerate(zip(RING_EDGES_MM[:-1], RING_EDGES_MM[1:])):
            row("boundary", None, sm, f"median_r_hippocampus_ring_{int(lo)}_{int(hi)}mm", np.nanmedian(rings[:, k]))
        ident = np.array([z[f"{h}_{sm}_boundary_identity"] for z in zs for h in HEMIS if f"{h}_{sm}_boundary_identity" in z])
        names = ("dmn_cortex_20_30mm", "cortex_within_3mm", "hippocampus_edge_3mm", "hippocampus_deep_4mm")
        for i, name in enumerate(names):
            row("boundary", None, sm, f"median_r_{name}_with_dmn_cortex_beyond_30mm", np.nanmedian(ident[:, i, 0]))
            if i < 3:
                row("boundary", None, sm, f"median_r_{name}_with_deep_hippocampus", np.nanmedian(ident[:, i, 1]))
    return rows


def render(Z: dict[str, dict[str, np.ndarray]], out_png: Path, hippomaps_root: Path | None) -> None:
    """ctxreg diagnostic figure (10 vs 20 mm); target-side and boundary evidence stays in the summary TSV."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import gridspec
    from matplotlib.colors import Normalize
    from PIL import Image

    from lib.hippomaps_adapter import DEFAULT_HIPPOMAPS_ROOT, capture_hippomaps_plot, load_hippomaps_plotter
    from lib.network_colors import NETWORK_DISPLAY_NAMES, load_network_hex_colors

    abbr = [NETWORK_DISPLAY_NAMES[n] for n in NETWORKS]
    colors = load_network_hex_colors()
    zs = list(Z.values())
    n_sub = len(zs)
    roi_net = zs[0]["roi_network"]
    ri = {r: i for i, r in enumerate(CONTROL_RADII_MM)}
    sm_map = MAP_SMOOTHING

    def gmap(fn, h):
        st = []
        for z in zs:
            x = np.asarray(fn(z, h), float).copy()
            x[~z[f"{h}_valid"]] = np.nan
            st.append(x)
        return np.nanmean(np.stack(st), axis=0)

    def changed(z, h, r):
        return (z[f"{h}_{sm_map}_ctxreg_{rtag(r)}_wta"] != z[f"{h}_{sm_map}_standard_wta"]).astype(float)

    lr_med = {r: float(np.median([np.nanmedian(z[f"{h}_{sm_map}_ctxreg_{rtag(r)}_local_r"][z[f"{h}_valid"]])
                                  for z in zs for h in HEMIS])) for r in MAP_RADII_MM}
    metrics = [
        ("count", "Size of the regressor: number of cortical vertices within the radius", "viridis",
         lambda z, h, r: z[f"{h}_disc_n"][ri[r]], None,
         "Valid cortical vertices within the radius (native T1w space) are averaged into one regressor per hippocampal vertex; "
         "color = vertex count. 10 mm counts track adjacency (posterior body/tail beside parahippocampal and isthmus cortex);\n"
         "20 mm counts track medial-lateral position (lateral vertices reach lateral temporal cortex; medial vertices face the "
         "non-cortical medial wall), so the two maps differ."),
        ("dmn", "What the regressor is made of: fraction of those cortical vertices in the DMN", "magma",
         lambda z, h, r: z[f"{h}_composition"][ri[r]][:, 0], (0.0, 1.0),
         "Each subject's Kong2022 PFM label of every averaged vertex; color = fraction labelled DMN (1 = regressor is pure DMN signal)."),
        ("localr", f"How much ctxreg removes: correlation of hippocampus with its regressor ({sm_map} hippocampal smoothing)", "viridis",
         lambda z, h, r: z[f"{h}_{sm_map}_ctxreg_{rtag(r)}_local_r"], None,
         "Pearson r between the hippocampal vertex time series and its local-cortex regressor (run-centered). "
         "Variance removed by ctxreg is r squared (r = 0.13 removes about 1.7%)."),
        ("wta", f"Effect on parcellation: how often the vertex's winning network changes ({sm_map})", "inferno",
         lambda z, h, r: changed(z, h, r), (0.0, 0.8),
         "Per subject, each vertex is labelled with its winning network (top-20 positive ROIs, counts normalized by available ROIs per "
         "network; 05_02 rule), once with standard and once with ctxreg time series. Color = fraction of subjects whose label differs."),
    ]
    plotter = load_hippomaps_plotter(Path(hippomaps_root) if hippomaps_root else DEFAULT_HIPPOMAPS_ROOT)
    render_tmp = tempfile.TemporaryDirectory(prefix="local_cortex_control_render_", dir=REPO_ROOT / "tmp")
    render_dir = Path(render_tmp.name)
    maps, ranges = {}, {}
    for key, _t, cmap, fn, rng, _e in metrics:
        vals = {(r, h): gmap(lambda z, hh, r=r: fn(z, hh, r), h) for r in MAP_RADII_MM for h in HEMIS}
        maps[key] = vals
        if rng is not None:
            ranges[key] = {r: rng for r in MAP_RADII_MM}
        elif key == "count":
            ranges[key] = {r: (0.0, float(np.nanmax([vals[(r, h)] for h in HEMIS]))) for r in MAP_RADII_MM}
        else:
            hi = float(np.nanpercentile(np.concatenate(list(vals.values())), 98))
            ranges[key] = {r: (0.0, hi) for r in MAP_RADII_MM}
        for r in MAP_RADII_MM:
            for h in HEMIS:
                capture_hippomaps_plot(
                    plotter, np.nan_to_num(vals[(r, h)]).astype(np.float32).reshape(-1, 1),
                    output_png=render_dir / f"{key}_{rtag(r)}_{h}.png", hemis=[h], labels=["hipp"], den="512",
                    unfoldAPrescale=False, tighten_cwindow=False, cmap=cmap, color_range=ranges[key][r], share="row",
                    color_bar=None, background=(1, 1, 1), transparent_bg=False, interactive=False, screenshot=True,
                    suppress_warnings=True, size=[450, 300], zoom=1.45)

    def crop(png: Path) -> np.ndarray:
        im = np.asarray(Image.open(png).convert("RGB"))
        keep = np.where((im < 250).any(axis=(0, 2)))[0]
        return im[:, keep.min(): keep.max() + 1]

    def panel(ax, header, explain):
        lines = textwrap.wrap(explain, 58)
        ax.set_title(header, fontweight="bold", fontsize=12, pad=12 + 13 * len(lines))
        ax.text(0.5, 1.02, "\n".join(lines), transform=ax.transAxes, ha="center", va="bottom", fontsize=9.5, color="0.25")

    fig = plt.figure(figsize=(24, 23))
    outer = gridspec.GridSpec(2, 1, height_ratios=[1.55, 0.52], hspace=0.2, top=0.915)
    top = gridspec.GridSpecFromSubplotSpec(len(metrics), 4, subplot_spec=outer[0], wspace=0.02, hspace=0.5)
    for row_i, (key, title, cmap, _f, _r, explain) in enumerate(metrics):
        for ci, r in enumerate(MAP_RADII_MM):
            for hi, h in enumerate(HEMIS):
                ax = fig.add_subplot(top[row_i, ci * 2 + hi])
                ax.imshow(crop(render_dir / f"{key}_{rtag(r)}_{h}.png"))
                ax.set_axis_off()
                ax.set_title(f"{int(r)} mm · {h}   (mean {float(np.nanmean(maps[key][(r, h)])):.3g})", fontsize=12)
                if hi == 1:
                    cax = ax.inset_axes([1.0, 0.1, 0.04, 0.8])
                    fig.colorbar(plt.cm.ScalarMappable(norm=Normalize(*ranges[key][r]), cmap=cmap), cax=cax)
        y1 = top[row_i, 0].get_position(fig).y1
        n_lines = explain.count("\n") + 1
        fig.text(0.5, y1 + 0.03 + 0.009 * (n_lines - 1), f"A{row_i + 1}. {title}", ha="center", fontsize=14, fontweight="bold")
        fig.text(0.5, y1 + 0.0145, explain, ha="center", va="bottom", fontsize=10.5, color="0.25")

    bot = gridspec.GridSpecFromSubplotSpec(1, 4, subplot_spec=outer[1], wspace=0.34)
    xr = np.arange(len(NETWORKS))

    ax = fig.add_subplot(bot[0, 0])
    for yi, r in enumerate(MAP_RADII_MM):
        comp = np.mean([np.nanmean(z[f"{h}_composition"][ri[r]], axis=0) for z in zs for h in HEMIS], axis=0)
        left = 0.0
        for k, n in enumerate(NETWORKS):
            ax.barh(yi, comp[k], left=left, color=colors[n], edgecolor="white")
            if comp[k] > 0.05:
                ax.text(left + comp[k] / 2, yi, f"{abbr[k]}\n{comp[k]:.2f}", ha="center", va="center", fontsize=9, color="white")
            left += comp[k]
    ax.set_yticks(range(len(MAP_RADII_MM)), [f"{int(r)} mm" for r in MAP_RADII_MM])
    ax.set_xlim(0, 1)
    ax.set_xlabel("fraction of local cortical vertices")
    panel(ax, "B1. Network make-up of the regressor",
          "Fraction of averaged cortical vertices in each network (all hippocampal vertices and subjects). "
          "10 mm is mostly DMN; 20 mm mixes in VN, SMN, DAN and VAN.")

    ax = fig.add_subplot(bot[0, 1])
    mids = [(a + b) / 2 for a, b in zip(RING_EDGES_MM[:-1], RING_EDGES_MM[1:])]
    labels = [f"{int(a)}-{int(b)}" for a, b in zip(RING_EDGES_MM[:-1], RING_EDGES_MM[1:])]
    for sm, ls in zip(SMOOTHINGS, ("-", "--")):
        vals = np.array([np.nanmedian(z[f"{h}_{sm}_ring_r"][:, z[f"{h}_valid"]], axis=1) for z in zs for h in HEMIS])
        ax.plot(mids, np.median(vals, 0), ls, marker="o", color="k", label=f"hipp smoothing {sm}")
        ax.fill_between(mids, np.percentile(vals, 25, 0), np.percentile(vals, 75, 0), color="grey", alpha=0.25)
    ax.set_xticks(mids, labels)
    ax.set_xlabel("distance ring from hippocampal vertex (mm)")
    ax.set_ylabel("r(hippocampus, ring cortex mean)")
    ax.legend(frameon=False)
    panel(ax, "B2. Shared signal vs distance",
          "r between each hippocampal vertex and the mean cortex in each distance ring (median; band = IQR); vertices without "
          "cortex in a ring are excluded. The jump at 0-3 mm suggests shared boundary voxels; beyond 3 mm r is small.")

    ax = fig.add_subplot(bot[0, 2])
    w = 0.38
    for j, r in enumerate(MAP_RADII_MM):
        d = np.array([[np.nanmean(z[f"{h}_{sm_map}_ctxreg_{rtag(r)}_dfc_roi"][roi_net == k]) for k in range(len(NETWORKS))]
                      for z in zs for h in HEMIS])
        ax.bar(xr + (j - 0.5) * w, d.mean(0), w, yerr=d.std(0) / np.sqrt(len(d)), color=[colors[n] for n in NETWORKS],
               alpha=1.0 if j == 0 else 0.55, hatch=None if j == 0 else "//", edgecolor="k",
               label=f"{int(r)} mm" + (" (solid)" if j == 0 else " (hatched)"))
    ax.axhline(0, color="k", lw=0.8)
    ax.set_xticks(xr, abbr)
    ax.set_ylabel(f"ΔFC = ctxreg − standard ({sm_map})")
    ax.legend(frameon=False)
    panel(ax, "B3. Change in hippocampus-network FC after ctxreg",
          "Mean (ctxreg - standard) FC per network (bar = SEM), about -r(hipp, regressor) x r(network, regressor). "
          f"10 mm: higher r(hipp, regressor) ({lr_med[10.0]:.2f} vs {lr_med[20.0]:.2f}) and a DMN-like regressor, so DMN loses "
          "twice as much. 20 mm: networks drop together, so rankings change less.")

    ax = fig.add_subplot(bot[0, 3])
    bars = []
    for sm in SMOOTHINGS:
        for tag, lab in [("standard", "std")] + [(f"ctxreg_{rtag(r)}", str(int(r))) for r in MAP_RADII_MM]:
            labs = np.concatenate([z[f"{h}_{sm}_{tag}_wta"][z[f"{h}_valid"]] for z in zs for h in HEMIS])
            labs = labs[labs >= 0]
            bars.append((f"{lab}\n{sm}", np.bincount(labs, minlength=len(NETWORKS)) / labs.size))
    for xi, (_lab, sh) in enumerate(bars):
        bottom = 0.0
        for k, n in enumerate(NETWORKS):
            ax.bar(xi, sh[k], bottom=bottom, color=colors[n], edgecolor="white", width=0.75)
            if sh[k] > 0.07:
                ax.text(xi, bottom + sh[k] / 2, f"{sh[k]:.2f}", ha="center", va="center", fontsize=8, color="white")
            bottom += sh[k]
    ax.set_xticks(range(len(bars)), [b[0] for b in bars], fontsize=8.5)
    ax.set_ylabel("share of hippocampal vertices")
    for k, n in enumerate(NETWORKS):
        ax.bar(0, 0, color=colors[n], label=abbr[k])
    ax.legend(frameon=False, fontsize=8, bbox_to_anchor=(1.0, 1.0), loc="upper left")
    panel(ax, "B4. Winning-network share",
          "Share of hippocampal vertices (all subjects) won by each network. std = standard; 10/20 = ctxreg radius in mm; "
          "2mm/4mm = hippocampal smoothing.")

    fig.suptitle(
        f"Local cortex regression (ctxreg) diagnostics — HCP 7T rest, Kong2022 PFM, canonical7 taxonomy; n_subjects={n_sub}\n"
        "DMN default mode, VN visual, SMN somatomotor, DAN dorsal attention, VAN ventral attention, CON control, LAN language; "
        "maps are group means on den-512 hippocampal surfaces (folded + unfolded)", fontsize=15, y=0.99)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=110, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    render_tmp.cleanup()


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0] if __doc__ else None)
    p.add_argument("--stage", choices=("subjects", "render", "all"), default="all")
    p.add_argument("--subjects-manifest", default=str(DEFAULT_SUBJECTS))
    p.add_argument("--subjects", nargs="*", default=None, help="Subset for --stage subjects; render always uses the full manifest.")
    p.add_argument("--data-root", default=str(DEFAULT_DATA_ROOT))
    p.add_argument("--dtseries-template", default=DEFAULT_DTSERIES_TEMPLATE)
    p.add_argument("--left-cortex-surface-template", default=DEFAULT_LEFT_SURFACE)
    p.add_argument("--right-cortex-surface-template", default=DEFAULT_RIGHT_SURFACE)
    p.add_argument("--route-a-root", default=str(DEFAULT_CORTEX_ROUTEA_ROOT), help="Individual Kong2022 PFM label root.")
    p.add_argument("--reference-root", default=str(DEFAULT_KONG2022_PFM_ROOT), help="PFM shared root with ROI registry, run metadata and cortex validity.")
    p.add_argument("--surface-root", default=str(DEFAULT_KONG2022_SURFACE_ROOT))
    p.add_argument("--hippunfold-batch-root", default=str(DEFAULT_HIPPUNFOLD_BATCH_ROOT))
    p.add_argument("--work-root", default=str(DEFAULT_LOCAL_CORTEX_CONTROL_WORK_ROOT))
    p.add_argument("--present-dir", default=str(DEFAULT_LOCAL_CORTEX_CONTROL_PRESENT_DIR))
    p.add_argument("--hippomaps-root", default=None)
    p.add_argument("--overwrite", action="store_true", help="Recompute existing per-subject diagnostics.")
    p.add_argument("--jobs", type=int, default=1, help="Subject-level workers; hard-capped at 10.")
    return p


def main() -> int:
    args = build_arg_parser().parse_args()
    manifest = Path(args.subjects_manifest).resolve()
    roster = load_subject_ids(manifest)
    work_root = Path(args.work_root).resolve()
    if args.stage in ("subjects", "all"):
        subjects = [s.replace("sub-", "") for s in args.subjects] if args.subjects else roster
        cfg = {
            "work_root": str(work_root), "overwrite": bool(args.overwrite),
            "data_root": str(Path(args.data_root).resolve()), "dtseries_template": args.dtseries_template,
            "left_surface": args.left_cortex_surface_template, "right_surface": args.right_cortex_surface_template,
            "route_a_root": str(Path(args.route_a_root).resolve()), "reference_root": str(Path(args.reference_root).resolve()),
            "surface_root": str(Path(args.surface_root).resolve()), "hippunfold_root": str(Path(args.hippunfold_batch_root).resolve()),
        }
        jobs = capped_jobs(int(args.jobs), len(subjects))
        if jobs == 1:
            for s in subjects:
                print(json.dumps(process_subject(s, cfg)), flush=True)
        else:
            with ProcessPoolExecutor(max_workers=jobs) as ex:
                for fut in as_completed([ex.submit(process_subject, s, cfg) for s in subjects]):
                    print(json.dumps(fut.result()), flush=True)
    if args.stage in ("render", "all"):
        present = Path(args.present_dir).resolve()
        Z = load_cohort(work_root, roster)
        if len(Z) != len(roster):
            raise ValueError(f"n_subjects mismatch: loaded {len(Z)} vs roster {len(roster)}")
        out_png = present / "local_cortex_control_diagnostics.png"
        render(Z, out_png, args.hippomaps_root)
        rows = summarize(Z)
        tsv = present / "local_cortex_control_summary.tsv"
        with tsv.open("w", encoding="utf-8", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=["control", "radius_mm", "smoothing", "metric", "value"], delimiter="\t")
            writer.writeheader()
            writer.writerows(rows)
        write_json(present / "local_cortex_control_manifest.json", {
            "schema": SCHEMA, "stage": "06_07_render_local_cortex_control_diagnostics", "timestamp_utc": utc_now_iso(),
            "subjects_manifest": str(manifest), "n_subjects": len(roster), "work_root": str(work_root),
            "radii_mm": list(CONTROL_RADII_MM), "map_radii_mm": list(MAP_RADII_MM), "map_smoothing": MAP_SMOOTHING,
            "outputs": {"png": str(out_png), "summary_tsv": str(tsv)},
        }, trailing_newline=True)
        print(json.dumps({"status": "ok", "n_subjects": len(roster), "png": str(out_png)}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
