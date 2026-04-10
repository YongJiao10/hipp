#!/usr/bin/env python3
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import scipy.io as sio
from nibabel.gifti import GiftiDataArray, GiftiImage, GiftiLabel, GiftiLabelTable

REPO_ROOT = Path(__file__).resolve().parents[1]
CORTEX_DIR = REPO_ROOT / "scripts" / "cortex"
if str(CORTEX_DIR) not in sys.path:
    sys.path.insert(0, str(CORTEX_DIR))

import run_cortex_pfm_subject as cortex
from render_group_priors_canonical_merged import load_merge_config, render_shape_figure

OUT_ROOT = REPO_ROOT / "outputs" / "group_priors" / "kong2022_areal"
PRESENT_ROOT = REPO_ROOT / "present"
PRIOR_MAT = OUT_ROOT / "Params_Final.mat"
LABEL_TXT = OUT_ROOT / "Schaefer2018_400Parcels_Kong2022_17Networks_order_info.txt"
MERGE_CONFIG = REPO_ROOT / "config" / "cross_atlas_network_merge.json"
SOURCE_SCENE = REPO_ROOT / "config" / "manual_wb_scenes" / "cortex_manual.scene"

def load_label_rows(path: Path) -> list[dict[str, object]]:
    lines = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    rows: list[dict[str, object]] = []
    for idx in range(0, len(lines), 2):
        name = lines[idx]
        parts = lines[idx + 1].split()
        rows.append(
            {
                "key": int(parts[0]),
                "name": name,
                "rgba": tuple(int(v) for v in parts[1:5]),
            }
        )
    return rows

def make_label_gifti(labels: np.ndarray, label_rows: list[dict[str, object]]) -> GiftiImage:
    table = GiftiLabelTable()
    unknown = GiftiLabel(key=0, red=0.0, green=0.0, blue=0.0, alpha=0.0)
    unknown.label = "???"
    table.labels.append(unknown)
    for row in label_rows:
        rgba = row["rgba"]
        item = GiftiLabel(
            key=int(row["key"]),
            red=float(rgba[0]) / 255.0,
            green=float(rgba[1]) / 255.0,
            blue=float(rgba[2]) / 255.0,
            alpha=float(rgba[3]) / 255.0,
        )
        item.label = str(row["name"])
        table.labels.append(item)
    arr = GiftiDataArray(data=labels.astype(np.int32), intent="NIFTI_INTENT_LABEL", datatype="NIFTI_TYPE_INT32")
    return GiftiImage(darrays=[arr], labeltable=table)

def run(cmd: list[str]) -> None:
    proc = subprocess.run(cmd, text=True, capture_output=True)
    if proc.returncode != 0:
        raise RuntimeError(f"Command failed ({proc.returncode}): {' '.join(cmd)}\nSTDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}")

def get_canonical_from_schaefer(name: str) -> str:
    # E.g. "17networks_LH_DefaultA_FPole_1" -> "DefaultA"
    parts = name.split('_')
    if len(parts) >= 3:
        net_17 = parts[2]
        if net_17.startswith("Vis"): return "Visual"
        if net_17.startswith("SomMot"): return "Somatomotor"
        if net_17.startswith("DorsAttn"): return "DorsalAttention"
        if net_17.startswith("SalVenAttn"): return "VentralAttention"
        if net_17.startswith("Cont"): return "Control"
        if net_17.startswith("Default"): return "Default"
        if net_17.startswith("Language"): return "Language"
        if net_17.startswith("Aud"): return "Auditory"
    return "Noise"

def main() -> int:
    canonical_order, _, _, shared_colors = load_merge_config(MERGE_CONFIG)
    
    mat = sio.loadmat(str(PRIOR_MAT))
    # theta shape is (64984, 400)
    theta = mat["Params"]["theta"][0,0].astype(np.float64)
    # transpose to (400, 64984)
    priors = theta.T
    
    label_rows = load_label_rows(LABEL_TXT)
    if len(label_rows) != 400:
        raise ValueError(f"Expected 400 labels, got {len(label_rows)}")
        
    canonical_sum = {name: np.zeros(priors.shape[1], dtype=np.float64) for name in canonical_order}
    
    for idx, row in enumerate(label_rows):
        canonical = get_canonical_from_schaefer(row["name"])
        if canonical in canonical_sum:
            canonical_sum[canonical] += priors[idx]
            
    used_canonical = [name for name in canonical_order if canonical_sum[name].sum() > 0]
    merged_priors = np.asarray([canonical_sum[name] for name in used_canonical], dtype=np.float64)
    
    outdir = OUT_ROOT / "canonical_merged" / "workbench_assets"
    outdir.mkdir(parents=True, exist_ok=True)
    
    canonical_label_rows: list[dict[str, object]] = []
    for idx, name in enumerate(used_canonical, start=1):
        canonical_label_rows.append({"key": idx, "name": name, "rgba": shared_colors[name]})
        
    prob_sum = merged_priors.sum(axis=0)
    labels = np.zeros(64984, dtype=np.int32)
    nz = prob_sum > 0
    labels[nz] = merged_priors[:, nz].argmax(axis=0).astype(np.int32) + 1
    
    left_labels = labels[:32492]
    right_labels = labels[32492:]
    
    stem = "Kong2022Areal_group_prior_canonical_merged_wta"
    left_path = outdir / f"{stem}.L.label.gii"
    right_path = outdir / f"{stem}.R.label.gii"
    dlabel_path = outdir / f"{stem}.dlabel.nii"
    
    nib.save(make_label_gifti(left_labels, canonical_label_rows), str(left_path))
    nib.save(make_label_gifti(right_labels, canonical_label_rows), str(right_path))
    run([
        cortex.WB_COMMAND, "-cifti-create-label", str(dlabel_path),
        "-left-label", str(left_path), "-right-label", str(right_path)
    ])
    
    nonzero_vertices = int(nz.sum())
    shape_png = render_shape_figure(
        source_scene=SOURCE_SCENE,
        atlas_name="Kong2022Areal_gMSHBM_beta50",
        assets={"dlabel": dlabel_path, "left_label": left_path, "right_label": right_path},
        label_rows=canonical_label_rows,
        outdir=OUT_ROOT / "canonical_merged",
        nonzero_vertices=nonzero_vertices,
        n_canonical=len(used_canonical),
    )
    
    PRESENT_ROOT.mkdir(parents=True, exist_ok=True)
    present_png = PRESENT_ROOT / "Kong2022Areal_gMSHBM_beta50_group_priors_canonical_merged_shape.png"
    present_png.write_bytes(shape_png.read_bytes())
    
    print(f"Rendered: {present_png}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())