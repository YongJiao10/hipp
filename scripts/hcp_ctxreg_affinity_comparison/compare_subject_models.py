#!/usr/bin/env python3
"""Compare joint subject fits with matched group-only hippocampal parcellation."""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import compare as baseline
import nibabel as nib
import numpy as np
from scipy.sparse import triu
from scipy.sparse.csgraph import connected_components
from sklearn.metrics import adjusted_rand_score

from lib import kong2022_label_utils
from subject_models import (
    EM_TOLERANCE, HBP_REVISION, MAX_ITER, N_STARTS, SCREEN_ITER, SEED, Fit, FitError,
    fit_directional, mean_graph_coordinates, shen_discretize, subject_graphs, unit_directions,
)

K = baseline.K
atlas = baseline.atlas
gradients = baseline.gradients
BASELINES = {**dict.fromkeys(baseline.METHODS, "gradient-ward"), "gradient_spectral": "gradient-spectral"}
SHEN_METHODS = ("shen_joint", "shen_mean_graph")
VMF_METHODS = ("vmf_mean", "vmf_participants", "hierarch_bayes")
METHODS = (*BASELINES, *SHEN_METHODS, *VMF_METHODS)
SENSITIVITY = "hierarch_bayes_uncentered"
CONTRASTS = {
    "shen_joint_minus_mean_graph": SHEN_METHODS,
    "vmf_participants_minus_mean": ("vmf_participants", "vmf_mean"),
    "hierarchy_minus_participant_vmf": ("hierarch_bayes", "vmf_participants"),
    "hierarchy_minus_mean_vmf": ("hierarch_bayes", "vmf_mean"),
    "shen_joint_minus_avg_fc_ward": ("shen_joint", "mean_fc"),
    "hierarchy_minus_avg_fc_ward": ("hierarch_bayes", "mean_fc"),
    "uncentered_minus_centered": (SENSITIVITY, "hierarch_bayes"),
}


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def training_sets(n_subjects):
    yield "all", np.arange(n_subjects), np.array([], dtype=int)
    for split in range(baseline.N_SPLITS):
        seed = atlas.split_seed_for_index(split)
        halves = np.array_split(np.random.RandomState(seed).permutation(n_subjects), 2)
        for half in (0, 1):
            yield f"{split}:{half}", halves[half], halves[1 - half]


def load_condition(settings, subjects, mask, smoothing, hemi):
    fc_root = Path(settings["fc_root"])
    reference_root = Path(settings["reference_root"])
    profiles = []
    for subject, observed in zip(subjects, mask, strict=True):
        fc, path = gradients.load_subject_fc(
            fc_root, subject, "kong2022", smoothing, hemi, observed, reference_root=reference_root,
        )
        if fc.shape != (observed.size, 400) or not np.isfinite(fc[observed]).all():
            raise ValueError(f"Expected 400 finite FC values per observed vertex in {path}, got {fc.shape}")
        fc[~observed] = np.nan
        profiles.append(gradients.fisher_z_transform_fc(fc))
    fisher_z = np.stack(profiles)
    similarity = np.stack([baseline.subject_similarity(profile) for profile in fisher_z])
    return fisher_z, similarity


def baseline_fit(method, support, anchor, adjacency, stage):
    active, full, values, ev = stage
    aligned, _, _, _, _ = baseline.align_gradient_components_to_reference(
        active=active, full=full, lambdas=values, ev_fractions=ev,
        reference_full=anchor, component_labels=[f"G{i + 1}" for i in range(baseline.N_COMPONENTS)],
    )
    _, labels = atlas.cluster_from_precomputed_group_features(
        method=BASELINES[method], group_features=aligned[:, :3], support_mask=support,
        adjacency_full=adjacency, k=K, seed=SEED,
    )
    return Fit(labels, {})


def expand_fit(fit, support):
    labels = np.zeros(support.size, dtype=int)
    labels[support] = fit.labels
    fit.labels = labels
    if fit.individual is not None:
        labels = np.zeros((fit.individual.shape[0], support.size), dtype=int)
        labels[:, support] = fit.individual
        fit.individual = labels
    probability = np.full((support.size, K), np.nan)
    probability[support] = fit.probability
    fit.probability = probability
    if fit.posterior is not None:
        posterior = np.full((fit.posterior.shape[0], support.size, K), np.nan, dtype=np.float32)
        posterior[:, support] = fit.posterior
        fit.posterior = posterior
    return fit


def export_fit(fit, run, anterior, adjacency):
    run.mkdir(parents=True, exist_ok=True)
    labels = atlas.reorder_labels_head_to_tail(fit.labels, anterior)
    parcel_order = np.asarray(atlas.cluster_ids_head_to_tail(fit.labels, anterior))
    lut = np.zeros(K + 1, dtype=np.int32)
    lut[parcel_order] = np.arange(1, K + 1)
    np.save(run / "group_labels_final.npy", labels)
    nib.save(atlas.make_cluster_label_gifti(labels, K, baseline.CLUSTER_COLORS), run / "group_labels_final.label.gii")
    if fit.individual is not None:
        np.save(run / "individual_labels.npy", lut[fit.individual])
    if fit.probability is not None:
        np.save(run / "group_probability.npy", fit.probability[:, parcel_order - 1])
    if fit.posterior is not None:
        np.save(run / "individual_probability.npy", fit.posterior[:, :, parcel_order - 1])
    if fit.directions is not None:
        np.save(run / "emission_directions.npy", fit.directions[:, parcel_order - 1])
    parcels = []
    for label in range(1, K + 1):
        vertices = np.flatnonzero(labels == label)
        _, components = connected_components(adjacency[vertices][:, vertices], directed=False)
        parcels.append(np.bincount(components).tolist())
    return labels, parcels


def score_fold(labels, similarity, test, edge_pairs, subjects, fold, condition):
    a, b = edge_pairs
    common = np.logical_and.reduce([(x[a] > 0) & (x[b] > 0) for x in labels.values()])
    rows = []
    for i in test:
        sim = similarity[i, a, b]
        observed = common & np.isfinite(sim)
        for method, y in labels.items():
            inside = y[a] == y[b]
            within, boundary = sim[observed & inside], sim[observed & ~inside]
            gap = float(within.mean(dtype=np.float64) - boundary.mean(dtype=np.float64)) if within.size and boundary.size else np.nan
            rows.append({
                **condition, "fold": fold, "subject": subjects[i], "method": method,
                "n_within_edges": within.size, "n_boundary_edges": boundary.size, "gap": gap,
            })
    return rows


def record_failure(rows, condition, fold, method, exc):
    error = {**condition, "fold": fold, "method": method, "error": str(exc)}
    print(f"  {fold} {method}: {exc}", flush=True)
    rows.append(error)


def run_condition(out, condition, profiles, similarity, mask, subjects, surface, anchors, training, sensitivity):
    adjacency, anterior, edge_pairs = surface
    smoothing, hemi = condition["smoothing"], condition["hemisphere"]
    condition_root = out / condition["route"] / smoothing / f"hemi_{hemi}"
    condition_root.mkdir(parents=True, exist_ok=True)
    centered = unit_directions(profiles, mask, centered=True)
    uncentered = unit_directions(profiles, mask, centered=False) if sensitivity else None
    scores, diagnostics, failures, split_stability, parcel_rows = [], [], [], [], []
    full_maps, half_maps = {}, {}
    graph_cache = {}

    for fold, train, test in training:
        print(f"[{condition['route']} {smoothing} {hemi}] {fold}", flush=True)
        support = mask[train].mean(axis=0) >= baseline.COVERAGE
        fits = {}
        for profile_name, source in (("mean_fc", profiles), ("mean_similarity", similarity)):
            stage = baseline.fit_gradients(source[train], support, profile_name)
            key = fold, profile_name
            if (smoothing, hemi) == ("2mm", "L"):
                anchors[key] = stage[1]
            fits[profile_name] = baseline_fit(profile_name, support, anchors[key], adjacency, stage)
            if profile_name == "mean_fc":
                fits["gradient_spectral"] = baseline_fit(
                    "gradient_spectral", support, anchors[key], adjacency, stage,
                )
        # Graph construction only uses a participant's own input and the
        # training-derived mask. Cache that operation for recurring supports.
        cache = graph_cache.setdefault(support.tobytes(), {})
        try:
            for i in train:
                if i not in cache:
                    bank, coords = subject_graphs(similarity[i:i + 1], mask[i:i + 1], support, [subjects[i]], K)
                    cache[i] = bank[0], coords[0]
            graphs, embeddings = zip(*(cache[i] for i in train), strict=True)
        except FitError as exc:
            record_failure(failures, condition, fold, "shen_graphs", exc)
        else:
            for method in SHEN_METHODS:
                try:
                    sample = embeddings if method == "shen_joint" else [mean_graph_coordinates(graphs, support, K)]
                    fits[method] = shen_discretize(sample, support, k=K)
                except FitError as exc:
                    record_failure(failures, condition, fold, method, exc)
        for method in (*VMF_METHODS, *((SENSITIVITY,) if sensitivity else ())):
            try:
                data = uncentered if method == SENSITIVITY else centered
                fit = fit_directional(
                    data[train][:, support], mask[train][:, support], k=K,
                    spatial=method in {"hierarch_bayes", SENSITIVITY}, mean_only=method == "vmf_mean",
                    save_individual=fold == "all" and method == "hierarch_bayes",
                )
                fits[method] = expand_fit(fit, support)
            except (FitError, NameError) as exc:
                record_failure(failures, condition, fold, method, exc)
        diagnostics.extend({
            **condition, "fold": fold, "method": method, **fit.diagnostics,
        } for method, fit in fits.items() if fit.diagnostics)
        if fold == "all":
            for method, fit in fits.items():
                full_maps[method], components = export_fit(fit, condition_root / method, anterior, adjacency)
                for cid, sizes in enumerate(components, start=1):
                    parcel_rows.append({
                        **condition, "method": method, "cluster_id": cid,
                        "n_vertices": sum(sizes), "n_components": len(sizes),
                        "component_sizes": ",".join(map(str, sorted(sizes, reverse=True))),
                    })
            continue

        labels = {method: fit.labels for method, fit in fits.items()}
        scores.extend(score_fold(labels, similarity, test, edge_pairs, subjects, fold, condition))
        if fold.endswith(":0"):
            half_maps = labels
        else:
            for method in half_maps.keys() & labels.keys():
                first, second = half_maps[method], labels[method]
                common = (first > 0) & (second > 0)
                split_stability.append({
                    **condition, "split": int(fold.split(":")[0]), "method": method,
                    "ari": adjusted_rand_score(first[common], second[common]),
                })

    baseline.write_table(condition_root / "held_out_scores.tsv", scores)
    write_json(condition_root / "fit_diagnostics.json", diagnostics)
    write_json(condition_root / "fit_failures.json", failures)
    return full_maps, scores, split_stability, parcel_rows, failures


def compare_heldout(rows):
    means, folds = [], []
    conditions = ("route", "smoothing", "hemisphere")
    by_condition = {}
    for row in rows:
        key = tuple(row[field] for field in conditions)
        by_condition.setdefault(key, {})[(row["fold"], row["subject"], row["method"])] = row["gap"]
    for key, scores in by_condition.items():
        meta = dict(zip(conditions, key, strict=True))
        for contrast, (method, reference) in CONTRASTS.items():
            by_fold, by_subject = {}, {}
            for (fold, subject, name), gap in scores.items():
                if name != method:
                    continue
                delta = gap - scores.get((fold, subject, reference), np.nan)
                if np.isfinite(delta):
                    by_fold.setdefault(fold, []).append(delta)
                    by_subject.setdefault(subject, []).append(delta)
            if not by_subject:
                continue
            values = np.array([np.mean(values) for values in by_subject.values()])
            for fold, deltas in sorted(by_fold.items()):
                folds.append({
                    **meta, "contrast": contrast, "fold": fold,
                    "mean_delta": np.mean(deltas), "n_subjects": len(deltas),
                })
            for split in sorted({fold.split(":")[0] for fold in by_fold}):
                deltas = by_fold.get(f"{split}:0", []) + by_fold.get(f"{split}:1", [])
                folds.append({
                    **meta, "contrast": contrast, "fold": split,
                    "mean_delta": np.mean(deltas), "n_subjects": len(deltas),
                })
            means.append({
                **meta, "contrast": contrast, "mean_delta": values.mean(),
                "fraction_improved": (values > 0).mean(), "n_subjects": len(values), "n_folds": len(by_fold),
                "min_fold_delta": min(np.mean(v) for v in by_fold.values()),
                "max_fold_delta": max(np.mean(v) for v in by_fold.values()),
            })
    return means, folds


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", required=True, type=Path, help="Checkout containing authoritative input assets")
    parser.add_argument("--reference-run", required=True, type=Path, help="Previous Avg FC/Avg Sim comparison directory")
    parser.add_argument("--bayes-source", required=True, type=Path, help="Pinned HierarchBayesParcel Git checkout")
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--routes", nargs="+", choices=tuple(baseline.ROUTES), default=list(baseline.ROUTES))
    parser.add_argument("--smoothing", nargs="+", choices=("2mm", "4mm"), default=["2mm", "4mm"])
    args = parser.parse_args()
    out = args.out.resolve()
    if not out.is_relative_to(baseline.REPO_ROOT / "tmp"):
        parser.error("--out must be under this worktree's tmp directory")
    bayes_source = args.bayes_source.resolve(strict=True)
    revision = subprocess.check_output(["git", "-C", str(bayes_source), "rev-parse", "HEAD"], text=True).strip()
    if revision != HBP_REVISION:
        parser.error(f"Expected HierarchBayesParcel {HBP_REVISION}, got {revision} at {bayes_source}")
    sys.path.insert(0, str(bayes_source))
    import torch
    torch.set_num_threads(1)
    if args.device == "cuda":
        if not torch.cuda.is_available():
            parser.error("--device cuda requires a compatible visible GPU")
        # The pinned upstream release creates some tensors with torch.normal.
        # This factory needs the legacy default tensor type to follow CUDA.
        torch.set_default_tensor_type(torch.cuda.FloatTensor)
        torch.set_float32_matmul_precision("highest")

    input_root = args.input_root.resolve(strict=True)
    reference_run = args.reference_run.resolve(strict=True)
    subjects_file = input_root / gradients.DEFAULT_SUBJECTS.relative_to(baseline.REPO_ROOT)
    mask_root = input_root / gradients.DEFAULT_SURFACE_ROOT.relative_to(baseline.REPO_ROOT)
    atlas.KONG2022_DLABEL_OVERRIDE = (
        input_root / kong2022_label_utils.KONG2022_DLABEL_PATH.relative_to(baseline.REPO_ROOT)
    ).resolve(strict=True)
    kong2022_label_utils.KONG2022_DLABEL_PATH = atlas.KONG2022_DLABEL_OVERRIDE
    subjects = gradients.load_subject_ids(subjects_file)
    masks = {h: np.stack([gradients.subject_valid_mask(mask_root, s, h) for s in subjects]) for h in ("L", "R")}
    training = list(training_sets(len(subjects)))
    failures, scores, split_stability, parcels, bilateral = [], [], [], [], []
    out.mkdir(parents=True, exist_ok=False)
    for route in args.routes:
        source = reference_run / route / "run_settings.json"
        settings = json.loads(source.read_text())
        if (settings["route"], settings["n_subjects"], settings["subjects"]) != (route, len(subjects), subjects):
            raise ValueError(f"Route or authoritative subject order mismatch: {source}")
        route_root = out / route
        route_root.mkdir()
        write_json(route_root / "run_settings.json", {
            **settings, "comparison_source": str(source), "mask_root": str(mask_root),
            "subjects_manifest": str(subjects_file), "kong2022_dlabel": str(atlas.KONG2022_DLABEL_OVERRIDE),
            "smoothing": args.smoothing, "density_evidence": "Consumed corobl surface, validity masks and FC row counts",
            "hierarch_bayes_revision": revision, "torch_version": torch.__version__,
            "bayes_device": args.device,
            "random_seed": SEED, "n_initializations": N_STARTS, "max_iterations": MAX_ITER,
            "screen_iterations": SCREEN_ITER, "em_tolerance": EM_TOLERANCE,
        })
        surfaces = {}
        for h, path in settings["surfaces"].items():
            path = Path(path)
            coords, faces = atlas.load_surface(path)
            if len(coords) != masks[h].shape[1] or f"_den-{len(coords)}_" not in path.name:
                raise ValueError(f"Density mismatch between FC, surface and validity masks: {path}")
            adjacency = atlas.build_surface_adjacency(faces, len(coords))
            edges = triu(adjacency, k=1).tocoo()
            surfaces[h] = adjacency, atlas.template_anterior_coordinate(path, h), (edges.row, edges.col)
        if (surfaces["L"][0] != surfaces["R"][0]).nnz:
            raise ValueError("The left and right surfaces do not share the homologous template topology")

        anchors = {}
        if "2mm" not in args.smoothing:
            fc, sim = load_condition(settings, subjects, masks["L"], "2mm", "L")
            for fold, train, _ in training:
                support = masks["L"][train].mean(axis=0) >= baseline.COVERAGE
                for method, bank in zip(baseline.METHODS, (fc, sim), strict=True):
                    anchors[fold, method] = baseline.fit_gradients(bank[train], support, method)[1]
            del fc, sim
        for smoothing in ("2mm", "4mm"):
            if smoothing not in args.smoothing:
                continue
            maps = {}
            for h in ("L", "R"):
                condition = {"route": route, "smoothing": smoothing, "hemisphere": h}
                profiles, similarity = load_condition(settings, subjects, masks[h], smoothing, h)
                maps[h], rows, stability, parcel_rows, failed = run_condition(
                    out, condition, profiles, similarity, masks[h], subjects, surfaces[h], anchors, training,
                    sensitivity=(route == "pfm" and smoothing == "2mm"),
                )
                scores.extend(rows)
                split_stability.extend(stability)
                parcels.extend(parcel_rows)
                failures.extend(failed)
                del profiles, similarity
            for method in sorted(maps["L"].keys() & maps["R"].keys()):
                left, right = maps["L"][method], maps["R"][method]
                common = (left > 0) & (right > 0)
                bilateral.append({
                    "route": route, "smoothing": smoothing, "method": method,
                    "ari": adjusted_rand_score(left[common], right[common]), "n_vertices": common.sum(),
                })
    summary, folds = compare_heldout(scores)
    for filename, rows in (
        ("held_out_scores.tsv", scores), ("paired_summary.tsv", summary), ("paired_folds.tsv", folds),
        ("split_half_ari.tsv", split_stability), ("bilateral_ari.tsv", bilateral), ("parcel_topology.tsv", parcels),
    ):
        baseline.write_table(out / filename, rows)
    write_json(out / "fit_failures.json", failures)


if __name__ == "__main__":
    main()
