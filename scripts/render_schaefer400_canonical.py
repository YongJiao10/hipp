#!/usr/bin/env python3
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
from nibabel.gifti import GiftiDataArray, GiftiImage, GiftiLabel, GiftiLabelTable

REPO_ROOT = Path(__file__).resolve().parents[1]
CORTEX_DIR = REPO_ROOT / "scripts" / "cortex"
if str(CORTEX_DIR) not in sys.path:
    sys.path.insert(0, str(CORTEX_DIR))

import run_cortex_pfm_subject as cortex
from render_group_priors_canonical_merged import load_merge_config, render_shape_figure

OUT_ROOT = REPO_ROOT / "outputs_migration" / "group_priors" / "kong2022_areal"
PRESENT_ROOT = REPO_ROOT / "present"
DLABEL_FILE = OUT_ROOT / "Schaefer2018_400Parcels_Kong2022_17Networks_order.dlabel.nii"
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
        if net_17.startswith("TempPar"): return "TemporalParietal"
    return "Noise"

def main() -> int:
    canonical_order, _, _, shared_colors = load_merge_config(MERGE_CONFIG)
    
    # Load dlabel file to extract vertices array
    dlabel_img = nib.load(str(DLABEL_FILE))
    # data is typically 1 x 64984 (if only cortex) or 91282
    data = dlabel_img.get_fdata().squeeze()
    # we only care about the first 64984 vertices
    if data.shape[0] < 64984:
        raise ValueError(f"Dlabel array too small: {data.shape}")
    cortex_labels = data[:64984].astype(np.int32)
    
    label_rows = load_label_rows(LABEL_TXT)
    
    used_canonical_names = set()
    # Pass used_canonical_names_list down into name_to_canonical_key so the labels align correctly
    
    key_mapping = {0: 0}
    for row in label_rows:
        orig_key = row["key"]
        canonical_name = get_canonical_from_schaefer(row["name"])
        if canonical_name != "Noise": used_canonical_names.add(canonical_name)
        
    used_canonical_names_list = []
    for name in canonical_order:
        if name in used_canonical_names and name in shared_colors:
            used_canonical_names_list.append(name)
            
    name_to_canonical_key = {name: idx + 1 for idx, name in enumerate(used_canonical_names_list)}
    
    for row in label_rows:
        orig_key = row["key"]
        canonical_name = get_canonical_from_schaefer(row["name"])
        if canonical_name != "Noise" and canonical_name in name_to_canonical_key:
            key_mapping[orig_key] = name_to_canonical_key[canonical_name]
        else:
            key_mapping[orig_key] = 0
            
    mapped_labels = np.zeros_like(cortex_labels)
    for orig_key, new_key in key_mapping.items():
        mapped_labels[cortex_labels == orig_key] = new_key
        
    outdir = OUT_ROOT / "schaefer_canonical_merged" / "workbench_assets"
    outdir.mkdir(parents=True, exist_ok=True)
    
    used_canonical_names_list = []
    for name in canonical_order:
        if name in used_canonical_names and name in shared_colors:
            used_canonical_names_list.append(name)
            
    canonical_label_rows: list[dict[str, object]] = []
    for idx, name in enumerate(used_canonical_names_list, start=1):
        canonical_label_rows.append({"key": idx, "name": name, "rgba": shared_colors[name]})
        
    left_labels = mapped_labels[:32492]
    right_labels = mapped_labels[32492:]
    
    stem = "Schaefer400_Kong2022Areal_canonical_merged"
    left_path = outdir / f"{stem}.L.label.gii"
    right_path = outdir / f"{stem}.R.label.gii"
    out_dlabel_path = outdir / f"{stem}.dlabel.nii"
    
    nib.save(make_label_gifti(left_labels, canonical_label_rows), str(left_path))
    nib.save(make_label_gifti(right_labels, canonical_label_rows), str(right_path))
    run([
        cortex.WB_COMMAND, "-cifti-create-label", str(out_dlabel_path),
        "-left-label", str(left_path), "-right-label", str(right_path)
    ])
    
    nz = mapped_labels > 0
    nonzero_vertices = int(nz.sum())
        
    shape_png = render_shape_figure(
        source_scene=SOURCE_SCENE,
        atlas_name="Schaefer400_Kong2022_Deterministic",
        assets={"dlabel": out_dlabel_path, "left_label": left_path, "right_label": right_path},
        label_rows=canonical_label_rows,
        outdir=OUT_ROOT / "schaefer_canonical_merged",
        nonzero_vertices=nonzero_vertices,
        n_canonical=len(used_canonical_names_list),
    )
    
    PRESENT_ROOT.mkdir(parents=True, exist_ok=True)
    present_png = PRESENT_ROOT / "Schaefer400_Kong2022_Deterministic_canonical_merged_shape.png"
    present_png.write_bytes(shape_png.read_bytes())
    
    print(f"Rendered: {present_png}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())