"""Apply the four formal Kong2022 canonical7 network annotations to fixed partitions."""
from __future__ import annotations

import csv
import importlib
import json
from pathlib import Path

import nibabel as nib
import numpy as np

from lib.network_colors import require_network_colors
from lib.network_ovr import export_network_ovr, fit_network_ovr_from_network_profiles, network_ts_ovr_variant
from lib.network_ts_fisherz import (
    assignment_status_from_weighted_network_profile,
    subjectwise_cluster_network_fisherz_profile,
)

atlas = importlib.import_module("05_02_run_group_atlas")
gradients = importlib.import_module("05_01_compute_hippomaps_gradients")
TAXONOMY = "canonical7"
OVR_ROI = "ovr_logistic_canonical7"
OVR_NET_TS = network_ts_ovr_variant(TAXONOMY)
VARIANTS = {
    "network_ts_fisherz": "Fisher-z (weighted network time series)",
    "fisherz": "Fisher-z (ROI)",
    OVR_NET_TS: "OVR (weighted network time series)",
    OVR_ROI: "OVR (ROI)",
}


def network_label_path(run: Path, variant: str) -> Path:
    return run / f"group_functional_atlas_labels_{variant}.label.gii"


def annotate_partition(run, subjects, ts_root, reference_root, masks, smoothing, hemi, network_names, registry, colors, *, observed_profiles=None):
    partitions = np.load(run / "group_labels_final.npy")
    cluster_ids = sorted(int(key) for key in np.unique(partitions) if key > 0)
    roi_mean = np.load(run / "cluster_roi_profile.npy")
    roi_model = json.loads((run / "network_ovr_canonical7.json").read_text())
    if (
        roi_model["subject_ids"] != subjects or roi_model["cluster_ids"] != cluster_ids
        or roi_model["network_names"] != network_names or roi_model["taxonomy"] != TAXONOMY
    ):
        raise ValueError(f"ROI OVR input order differs from the fitted partition: {run}")

    _, roi_status, _ = atlas.assignment_status_from_kong2022_roi_profile(
        roi_mean, registry, network_names, network_column=TAXONOMY,
    )
    profiles, mean_profile, used_subjects = subjectwise_cluster_network_fisherz_profile(
        group_labels_final=partitions, support_mask=partitions > 0, subjects=subjects,
        surface_root=ts_root, reference_root=reference_root, smoothing=smoothing,
        hemi=hemi, valid_masks_by_subject=masks, network_names=network_names,
        network_column=TAXONOMY,
    )
    if used_subjects != subjects:
        raise ValueError(f"Weighted network FC input order differs from the fitted partition: {run}")
    np.save(run / "cluster_network_ts_profile.npy", mean_profile)
    network_status, _ = assignment_status_from_weighted_network_profile(mean_profile, network_names)
    statuses = {"network_ts_fisherz": network_status, "fisherz": roi_status}
    network_model = fit_network_ovr_from_network_profiles(
        profiles, used_subjects, cluster_ids, network_names, taxonomy=TAXONOMY,
        observed_profiles=observed_profiles,
    )
    if roi_model.get("excluded_subject_cluster_rows", []) != network_model.get("excluded_subject_cluster_rows", []):
        raise ValueError(f"ROI and weighted time-series OVR used different subject–parcel rows: {run}")
    export_network_ovr(network_model, run, prefix=OVR_NET_TS)
    statuses[OVR_NET_TS] = network_model["assignment_rows"]
    statuses[OVR_ROI] = roi_model["assignment_rows"]
    for variant, rows in statuses.items():
        if [int(row["cluster_id"]) for row in rows] != cluster_ids:
            raise ValueError(f"Cluster order differs for {variant}: {run}")
        labels, _, label_items = atlas.functional_atlas_from_clusters(
            group_labels_final=partitions, status_rows=rows, network_names=network_names, color_map=colors,
        )
        nib.save(atlas.make_named_label_gifti(labels, label_items), network_label_path(run, variant))
        status_path = run / f"cluster_assignment_status_{variant}.json"
        status_path.write_text(json.dumps(
            {"network_taxonomy": TAXONOMY, "network_names": network_names, "items": rows},
            indent=2,
        ) + "\n")
    return statuses


def annotate_comparison(out, routes, combinations, methods, mask_root):
    network_names = list(atlas.canonical_families_7_order())
    registry = atlas.load_kong2022_roi_registry()
    colors = require_network_colors(network_names)
    table = []
    for route in routes:
        route_root = out / route
        settings = json.loads((route_root / "run_settings.json").read_text())
        subjects = settings["subjects"]
        if settings["route"] != route or settings["n_subjects"] != len(subjects):
            raise ValueError(f"Route or roster mismatch in {route_root / 'run_settings.json'}")
        ts_root = Path(settings["hippocampal_timeseries_root"])
        reference_root = Path(settings["reference_root"])
        masks = {
            hemi: {subject: gradients.subject_valid_mask(mask_root, subject, hemi) for subject in subjects}
            for hemi in ("L", "R")
        }
        for smoothing, hemi in combinations:
            for method in methods:
                print(f"[{route} {smoothing} {hemi}] Labelling {method}", flush=True)
                run = route_root / smoothing / f"hemi_{hemi}" / method
                variants = annotate_partition(
                    run, subjects, ts_root, reference_root, masks[hemi], smoothing, hemi,
                    network_names, registry, colors,
                )
                for variant, rows in variants.items():
                    for row in rows:
                        table.append({
                            "route": route, "smoothing": smoothing, "hemisphere": hemi,
                            "method": method, "variant": variant, "cluster_id": row["cluster_id"],
                            "network": row["top1_network"], "status": row["status"],
                            "score": row.get("top1_beta" if variant.startswith("ovr_") else "top1_fisher_z"),
                            "score_units": "raw_beta" if variant.startswith("ovr_") else "Fisher_z",
                        })
    with (out / "network_labels.tsv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(table[0]), delimiter="\t")
        writer.writeheader()
        writer.writerows(table)
