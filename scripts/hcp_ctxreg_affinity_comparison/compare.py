#!/usr/bin/env python3
"""Compare averaging Fisher-z FC and averaging within-subject Pearson similarity."""
from __future__ import annotations

import argparse
import csv
import importlib
import json
import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "core_workflow"))

from lib import native_threads  # noqa: F401  Keep native pools single-threaded.

import nibabel as nib
import numpy as np
from brainspace.gradient.kernels import compute_affinity
from scipy.sparse import triu
from sklearn.metrics import adjusted_rand_score

from lib.gradient_alignment import align_gradient_components_to_reference
from lib.gradient_profiles import get_gradient_profile
from lib.hippomaps_adapter import (
    diffusion_map_embedding,
    ev_fractions_from_lambdas,
    fit_diffusion_gradients,
    reconstruct_full_surface,
)
from lib.output_layout import REPO_ROOT, resolve_hcp_kong400_control_shared_root

gradients = importlib.import_module("05_01_compute_hippomaps_gradients")
atlas = importlib.import_module("05_02_run_group_atlas")

ROUTES = {"pfm": "Standard", "pfm_ctxreg_10mm": "CtxReg 10 mm", "pfm_ctxreg": "CtxReg 20 mm"}
COMBOS = [(smoothing, hemi) for smoothing in ("2mm", "4mm") for hemi in ("L", "R")]
METHODS = ("mean_fc", "mean_similarity")
TOPSIM = get_gradient_profile("topsim")
COVERAGE = 0.80
K = 3
N_COMPONENTS = 5
N_SPLITS = 5
CLUSTER_COLORS = [(158, 158, 158, 255), (43, 111, 166, 255), (232, 137, 60, 255), (71, 155, 120, 255)]


def subject_similarity(fisherz: np.ndarray) -> np.ndarray:
    """Pearson over jointly observed cortical ROIs for each pair of vertices."""
    out = np.full((len(fisherz), len(fisherz)), np.nan, dtype=np.float32)
    supported = np.isfinite(fisherz).any(axis=1)
    rows = fisherz[supported].astype(np.float64)
    rows = rows[:, np.isfinite(rows).any(axis=0)]
    if rows.shape[1] < 2:
        return out
    if np.isfinite(rows).all():
        rows -= rows.mean(axis=1, keepdims=True)
        norm = np.linalg.norm(rows, axis=1)
        supported[supported] = norm > 0
        unit = rows[norm > 0] / norm[norm > 0, None]
        similarity = unit @ unit.T
    else:
        valid = np.isfinite(rows).astype(np.float64)
        rows[~np.isfinite(rows)] = 0
        n = valid @ valid.T
        safe_n = np.maximum(n, 1)
        sums = rows @ valid.T
        variances = rows**2 @ valid.T - sums**2 / safe_n
        covariance = rows @ rows.T - sums * sums.T / safe_n
        denom = np.sqrt(np.maximum(variances, 0) * np.maximum(variances.T, 0))
        similarity = np.full_like(covariance, np.nan)
        np.divide(covariance, denom, out=similarity, where=(n >= 2) & (denom > 0))
    out[np.ix_(supported, supported)] = np.clip(similarity, -1, 1)
    return out


def fit_gradients(data: np.ndarray, support: np.ndarray, method: str):
    if method == "mean_fc":
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="Mean of empty slice", category=RuntimeWarning)
            mean_fc = np.nanmean(data[:, support, :], axis=0)
        mean_fc[~np.isfinite(mean_fc)] = 0
        active, values, ev = fit_diffusion_gradients(
            mean_fc, n_components=N_COMPONENTS, feature_scaling=TOPSIM.feature_scaling,
            sparsify_stage=TOPSIM.sparsify_stage, sparsity=TOPSIM.sparsity,
        )
    else:
        active_pairs = data[:, support, :][:, :, support]
        if not np.isfinite(active_pairs).any(axis=0).all():
            raise ValueError("A supported pair has no defined within-subject Pearson similarity")
        similarity = np.nanmean(active_pairs, axis=0, dtype=np.float64)
        np.fill_diagonal(similarity, 1)
        affinity = compute_affinity(similarity, kernel=None, sparsity=TOPSIM.sparsity, non_negative=True)
        affinity = (affinity + affinity.T) / 2
        active, values = diffusion_map_embedding(affinity, n_components=N_COMPONENTS)
        ev = ev_fractions_from_lambdas(values)
    active = active.astype(np.float32)
    return active, reconstruct_full_surface(active, support, support.size), values, ev


def fit_partitions(bank, mask_bank, adjacency, anterior, train, method):
    fitted = {}
    for combo in COMBOS:
        support = mask_bank[combo[1]][train].mean(axis=0) >= COVERAGE
        fitted[combo] = (support, fit_gradients(bank[combo][train], support, method))
    anchor = fitted[("2mm", "L")][1][1]
    component_labels = [f"G{component + 1}" for component in range(N_COMPONENTS)]
    partitions = {}
    for (smoothing, hemi), (support, fit) in fitted.items():
        active, full, values, ev = fit
        aligned, aligned_full, _, _, _ = align_gradient_components_to_reference(
            active=active, full=full, lambdas=values, ev_fractions=ev,
            reference_full=anchor, component_labels=component_labels,
        )
        _, full_labels = atlas.cluster_from_precomputed_group_features(
            method="gradient-ward", group_features=aligned[:, :3], support_mask=support,
            adjacency_full=adjacency[hemi], k=K, seed=20260420,
        )
        partitions[(smoothing, hemi)] = (
            atlas.reorder_labels_head_to_tail(full_labels, anterior[hemi]), aligned_full
        )
    return partitions


def evaluate_split_halves(banks, masks, adjacency, anterior, edges, subjects):
    individual_rows, split_rows = [], []
    for split in range(N_SPLITS):
        seed = atlas.split_seed_for_index(split)
        order = np.random.RandomState(seed).permutation(len(subjects))
        halves = np.array_split(order, 2)
        fitted = {
            method: [fit_partitions(banks[method], masks, adjacency, anterior, half, method) for half in halves]
            for method in METHODS
        }
        for combo in COMBOS:
            smoothing, hemi = combo
            a, b = edges[hemi]
            for method, (first, second) in fitted.items():
                labels_a, labels_b = first[combo][0], second[combo][0]
                overlap = (labels_a > 0) & (labels_b > 0)
                split_rows.append({
                    "smoothing": smoothing, "hemisphere": hemi, "seed": seed, "method": method,
                    "split_half_ari": adjusted_rand_score(labels_a[overlap], labels_b[overlap]),
                })
            for half in range(2):
                first = fitted["mean_fc"][half][combo][0]
                second = fitted["mean_similarity"][half][combo][0]
                common_edges = (first[a] > 0) & (first[b] > 0) & (second[a] > 0) & (second[b] > 0)
                for index in halves[1 - half]:
                    sim = banks["mean_similarity"][combo][index, a, b]
                    valid = common_edges & np.isfinite(sim)
                    scores = []
                    for labels in (first, second):
                        within = labels[a] == labels[b]
                        scores.append(
                            sim[valid & within].mean() - sim[valid & ~within].mean()
                            if np.any(valid & within) and np.any(valid & ~within) else np.nan
                        )
                    individual_rows.append({
                        "smoothing": smoothing, "hemisphere": hemi, "seed": seed,
                        "subject": subjects[index], "mean_fc": float(scores[0]),
                        "mean_similarity": float(scores[1]),
                    })
        print(f"  held-out split {split + 1}/{N_SPLITS}", flush=True)
    return individual_rows, split_rows


def write_table(path, rows):
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def annotate(out, labels, subjects, shared_root, ts_root, ref_root, masks, smoothing, hemi, *, observed_profiles=None):
    profiles, mean_profile, roi_registry, used_subjects = atlas.cluster_roi_profile_subjectwise(
        group_labels_final=labels, support_mask=labels > 0, subjects=subjects,
        shared_root=shared_root, surface_root=ts_root, reference_root=ref_root,
        smoothing=smoothing, hemi=hemi, valid_masks_by_subject=masks,
    )
    np.save(out / "cluster_roi_profile.npy", mean_profile)
    network_names = atlas.canonical_families_7_order()
    _, _, dominant = atlas.assignment_status_from_kong2022_roi_profile(mean_profile, roi_registry, network_names)
    ovr = atlas.build_ovr_assignment_specs(
        roi_profile_3d=profiles, subjects=used_subjects,
        cluster_ids=sorted(int(label) for label in np.unique(labels) if label > 0),
        roi_registry=roi_registry, variants=["ovr_logistic_canonical7"], out_dir=out,
        observed_profiles=observed_profiles,
    )
    return dominant, ovr["ovr_logistic_canonical7"]["dominant"]


def run_route(route, out, subjects, ctxreg_10mm_fc_root, masks, surfaces, adjacency, anterior, edges):
    ref_root = gradients.resolve_gradient_shared_root(gradients.DEFAULT_SHARED_INPUTS, "pfm")
    ts_root = gradients.DEFAULT_SURFACE_ROOT if route == "pfm" else resolve_hcp_kong400_control_shared_root(route)
    shared_root = {
        "pfm": ref_root, "pfm_ctxreg": ts_root, "pfm_ctxreg_10mm": ctxreg_10mm_fc_root,
    }[route]
    fc_bank, sim_bank = {}, {}
    for smoothing, hemi in COMBOS:
        print(f"[{route} {smoothing} {hemi}] Loading FC and calculating subject similarities", flush=True)
        fc = np.stack([
            gradients.load_subject_fc(
                shared_root, subject, "kong2022", smoothing, hemi, mask, reference_root=ref_root,
            )[0] for subject, mask in zip(subjects, masks[hemi], strict=True)
        ])
        fc[~masks[hemi]] = np.nan
        fisherz = gradients.fisher_z_transform_fc(fc)
        fc_bank[(smoothing, hemi)] = fisherz
        sim_bank[(smoothing, hemi)] = np.stack([subject_similarity(matrix) for matrix in fisherz])

    banks = dict(zip(METHODS, (fc_bank, sim_bank), strict=True))
    full = {
        method: fit_partitions(banks[method], masks, adjacency, anterior, slice(None), method)
        for method in METHODS
    }
    individual, splits = evaluate_split_halves(banks, masks, adjacency, anterior, edges, subjects)
    out.mkdir(parents=True, exist_ok=True)
    write_table(out / "held_out_subject_scores.tsv", individual)
    write_table(out / "split_half_ari.tsv", splits)
    summary = []
    for combo in COMBOS:
        smoothing, hemi = combo
        labels_fc, labels_sim = (full[method][combo][0] for method in METHODS)
        support = labels_fc > 0
        values_by_subject = {}
        for row in individual:
            if (row["smoothing"], row["hemisphere"]) == combo and np.isfinite([row[m] for m in METHODS]).all():
                values_by_subject.setdefault(row["subject"], []).append([row[m] for m in METHODS])
        scores = np.array([np.mean(values, axis=0) for values in values_by_subject.values()])
        improvement = scores[:, 1] - scores[:, 0]
        sizes = {}
        network_labels = {}
        for method in METHODS:
            print(f"[{route} {smoothing} {hemi}] Exporting and annotating {method}", flush=True)
            destination = out / smoothing / f"hemi_{hemi}" / method
            destination.mkdir(parents=True, exist_ok=True)
            labels, embedding = full[method][combo]
            np.save(destination / "group_labels_final.npy", labels)
            np.save(destination / "gradients_full.npy", embedding)
            label_gifti = atlas.make_cluster_label_gifti(labels, K, CLUSTER_COLORS)
            nib.save(label_gifti, destination / "group_labels_final.label.gii")
            sizes[method] = ",".join(str(n) for n in np.bincount(labels, minlength=K + 1)[1:])
            fisherz, ovr = annotate(
                destination, labels, subjects, shared_root, ts_root, ref_root,
                dict(zip(subjects, masks[hemi], strict=True)), smoothing, hemi,
            )
            network_labels[method] = [",".join(label or "unassigned" for label in row) for row in (fisherz, ovr)]
        a, b = edges[hemi]
        valid = support[a] & support[b]
        cut_a = (labels_fc[a] != labels_fc[b]) & valid
        cut_b = (labels_sim[a] != labels_sim[b]) & valid
        summary.append({
            "route": route, "smoothing": smoothing, "hemisphere": hemi,
            "n_vertices": int(support.sum()),
            "between_method_ari": adjusted_rand_score(labels_fc[support], labels_sim[support]),
            "changed_ap_ordered_vertices": int(np.sum(labels_fc[support] != labels_sim[support])),
            "boundary_jaccard": np.sum(cut_a & cut_b) / np.sum(cut_a | cut_b),
            **{f"{m}_cluster_sizes": sizes[m] for m in METHODS},
            **{f"{m}_{label}": network_labels[m][i] for m in METHODS for i, label in enumerate(("fisherz", "ovr"))},
            "held_out_n_subjects": len(scores), "held_out_gap_mean_fc": scores[:, 0].mean(),
            "held_out_gap_mean_similarity": scores[:, 1].mean(),
            "held_out_gap_delta": improvement.mean(),
            "held_out_fraction_improved": (improvement > 0).mean(),
            **{
                f"split_half_ari_{method}": np.mean([
                    r["split_half_ari"] for r in splits
                    if (r["smoothing"], r["hemisphere"], r["method"]) == (*combo, method)
                ])
                for method in METHODS
            },
        })
    write_table(out / "comparison.tsv", summary)
    (out / "run_settings.json").write_text(json.dumps({
        "route": route, "n_subjects": len(subjects), "subjects": subjects,
        "input_shape_by_combination": {f"{sm}/{hemi}": list(values.shape)
                                       for (sm, hemi), values in fc_bank.items()},
        "density_evidence": "corobl den-512 group-average midthickness surfaces and validated FC rows",
        "fc_root": str(shared_root), "hippocampal_timeseries_root": str(ts_root), "reference_root": str(ref_root),
        "surfaces": {hemi: str(path) for hemi, path in surfaces.items()},
        "coverage": COVERAGE, "k": K, "n_fitted_gradients": N_COMPONENTS,
        "components_clustered": [1, 2, 3], "alignment_anchor": "2mm/L",
        "row_sparsity": TOPSIM.sparsity, "feature_scaling": TOPSIM.feature_scaling,
        "split_seeds": [atlas.split_seed_for_index(i) for i in range(N_SPLITS)],
    }, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--ctxreg-10mm-fc-root", type=Path)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--render-only", action="store_true", help="Render existing partitions and network labels.")
    mode.add_argument("--annotate-only", action="store_true", help="Add network labels to existing partitions.")
    parser.add_argument("--mask-root", type=Path, help="Shared valid-vertex masks used by the existing partitions.")
    parser.add_argument("--kong2022-dlabel", type=Path, help="Explicit Kong2022 400-ROI atlas asset.")
    parser.add_argument("--routes", nargs="+", choices=tuple(ROUTES), default=list(ROUTES))
    args = parser.parse_args()
    out = args.out.resolve()
    if args.kong2022_dlabel is not None:
        atlas.KONG2022_DLABEL_OVERRIDE = args.kong2022_dlabel.resolve(strict=True)
    if args.render_only:
        from figure import render_comparison

        render_comparison(out, {route: ROUTES[route] for route in args.routes}, CLUSTER_COLORS)
        return
    if args.annotate_only:
        from network_annotation import annotate_comparison

        if args.mask_root is None:
            parser.error("--annotate-only requires the experiment's --mask-root")
        annotate_comparison(out, args.routes, COMBOS, METHODS, args.mask_root.resolve())
        return
    if not out.is_relative_to(REPO_ROOT / "tmp"):
        parser.error("--out must be under the repository tmp directory")

    subjects = gradients.load_subject_ids(gradients.DEFAULT_SUBJECTS)
    surfaces = {
        hemi: gradients.DEFAULT_GROUP_AVG_SURFACES_DIR
        / f"group-average_hemi-{hemi}_space-corobl_den-512_label-hipp_midthickness.surf.gii"
        for hemi in ("L", "R")
    }
    if "pfm_ctxreg_10mm" in args.routes and args.ctxreg_10mm_fc_root is None:
        parser.error("--ctxreg-10mm-fc-root is required when fitting pfm_ctxreg_10mm")
    ctxreg_10mm_fc_root = args.ctxreg_10mm_fc_root.resolve() if args.ctxreg_10mm_fc_root else None
    masks, adjacency, anterior, edges = {}, {}, {}, {}
    for hemi, path in surfaces.items():
        coords, faces = atlas.load_surface(path)
        adjacency[hemi] = atlas.build_surface_adjacency(faces, len(coords))
        anterior[hemi] = atlas.template_anterior_coordinate(path, hemi)
        edge_matrix = triu(adjacency[hemi], k=1).tocoo()
        edges[hemi] = (edge_matrix.row, edge_matrix.col)
        masks[hemi] = np.stack([
            gradients.subject_valid_mask(gradients.DEFAULT_SURFACE_ROOT, subject, hemi)
            for subject in subjects
        ])
    for route in args.routes:
        run_route(route, out / route, subjects, ctxreg_10mm_fc_root,
                  masks, surfaces, adjacency, anterior, edges)

    from network_annotation import annotate_comparison

    annotate_comparison(out, args.routes, COMBOS, METHODS, gradients.DEFAULT_SURFACE_ROOT)


if __name__ == "__main__":
    main()
