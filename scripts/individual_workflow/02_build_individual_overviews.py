#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import importlib.util
import os
import shutil
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import nibabel as nib
import numpy as np

from routea_individual_gradients_wta import (
    CORE_WORKFLOW_DIR,
    DEFAULT_HEMIS,
    DEFAULT_INDIVIDUAL_OVERVIEW_ROOT,
    DEFAULT_JOBS,
    DEFAULT_OUT_ROOT,
    DEFAULT_SMOOTHINGS,
    DEFAULT_SUBJECTS_MANIFEST,
    DEFAULT_KONG2022_ROUTEA_ROOT,
    DEFAULT_KONG2022_SURFACE_ROOT,
    DEFAULT_HIPPUNFOLD_BATCH_ROOT,
    OVERVIEW_METHODS,
    OVERVIEW_METHOD_ALIASES,
    THREAD_ENV_OVERRIDES,
    SubjectCombo,
    finite_fisher_z,
    git_commit,
    gradient_combo_output_dir,
    load_subject_ids,
    normalize_overview_method,
    normalize_subject,
    overview_output_path,
    read_json,
    roi_fc_path,
    subject_spec_path,
    subject_surface_path,
    utc_now_iso,
    valid_mask_path,
    write_json,
    wta_combo_output_dir,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ADAPTER_ROOT = DEFAULT_OUT_ROOT / "_overview_core_adapter"
DEFAULT_ADAPTER_PRESENT_ROOT = DEFAULT_ADAPTER_ROOT / "present_tmp"
DEFAULT_SCENE = REPO_ROOT / "config" / "wb_locked_native_view_lateral_medial.scene"
COVERAGE_TAG = "cov_0p80"
COVERAGE_THRESHOLD = "0.80"
ATLAS = "kong2022"
ROUTE_TAG = "routea"
GRADIENT_CORE_METHOD = "gradient-spectral"
WTA_CORE_METHOD = "wta-top"
GRADIENT_COMP_TAG = "g1g2g3"

K_SELECTION_FIELDS = [
    "method", "atlas", "smoothing", "hemi", "coverage_threshold", "k", "n_subjects",
    "n_support_vertices", "V_min_effective", "group_ari_mean", "group_ari_se",
    "subject_split_requested", "subject_split_used", "subject_split_skipped_overlap",
    "subject_split_skipped_empty", "run_aware_ari_mean", "run_aware_ari_se", "run_aware_n_pairs",
    "run_aware_n_used", "ARI_null_mean", "ARI_null_5pct", "ARI_weak_null_mean",
    "B_null_requested", "B_null_used_strong", "B_null_used_weak", "min_cluster_size",
    "max_components_per_cluster", "min_profile_coherence", "mean_profile_coherence",
    "nontrivial_min_size_ok", "nontrivial_connectedness_ok", "nontrivial_coherence_ok",
    "local_max", "within_1se_best", "selected_nontrivial", "max_profile_similarity",
    "redundant_pair_count", "nonredundant_ok", "reject_nontrivial_fail", "reject_redundant",
    "reject_reason_primary", "selected_final", "within_1se_survivor_pool", "survivor_pool_argmax_k",
    "k_final_decision_path",
]


def load_module(path: Path, module_name: str):
    if str(CORE_WORKFLOW_DIR) not in sys.path:
        sys.path.insert(0, str(CORE_WORKFLOW_DIR))
    if module_name in sys.modules:
        return sys.modules[module_name]
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load module: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def core_overview_module():
    return load_module(CORE_WORKFLOW_DIR / "06_01_build_group_overview.py", "_individual_adapter_core_overview")


def core_atlas_module():
    return load_module(CORE_WORKFLOW_DIR / "05_02_run_group_atlas.py", "_individual_adapter_core_atlas")


def write_tsv(path: Path, rows: list[dict[str, object]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, delimiter="\t", extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def fixed_k_selection_row(*, method: str, smoothing: str, hemi: str, n_support: int) -> dict[str, object]:
    row = {field: "" for field in K_SELECTION_FIELDS}
    row.update(
        {
            "method": method,
            "atlas": ATLAS,
            "smoothing": smoothing,
            "hemi": hemi,
            "coverage_threshold": COVERAGE_THRESHOLD,
            "k": 3,
            "n_subjects": 1,
            "n_support_vertices": int(n_support),
            "V_min_effective": 1,
            "group_ari_mean": 1.0,
            "group_ari_se": 0.0,
            "subject_split_requested": 0,
            "subject_split_used": 0,
            "ARI_null_mean": 0.0,
            "ARI_null_5pct": 0.0,
            "min_cluster_size": int(n_support),
            "max_components_per_cluster": 1,
            "min_profile_coherence": 1.0,
            "mean_profile_coherence": 1.0,
            "nontrivial_min_size_ok": 1,
            "nontrivial_connectedness_ok": 1,
            "nontrivial_coherence_ok": 1,
            "local_max": 1,
            "within_1se_best": 1,
            "selected_nontrivial": 1,
            "max_profile_similarity": 0.0,
            "redundant_pair_count": 0,
            "nonredundant_ok": 1,
            "reject_nontrivial_fail": 0,
            "reject_redundant": 0,
            "reject_reason_primary": "fixed_individual_k3",
            "selected_final": 1,
            "within_1se_survivor_pool": 1,
            "survivor_pool_argmax_k": 3,
            "k_final_decision_path": "individual_fixed_k3",
        }
    )
    return row


def adapter_run_dir(adapter_root: Path, subject: str, method: str, smoothing: str, hemi: str) -> Path:
    if method == GRADIENT_CORE_METHOD:
        return adapter_root / f"sub-{subject}" / method / ATLAS / smoothing / COVERAGE_TAG / GRADIENT_COMP_TAG / f"hemi_{hemi}"
    return adapter_root / f"sub-{subject}" / method / ATLAS / smoothing / COVERAGE_TAG / f"hemi_{hemi}"


def materialize_gradient_core_inputs(*, subject: str, work_root: Path, adapter_root: Path, shared_root: Path, surface_root: Path, hippunfold_batch_root: Path, assignment_variant: str = "fisherz") -> None:
    core = core_atlas_module()
    network_names = core.verify_canonical_network_order_consistency(shared_root, [subject], ATLAS, network_taxonomy="canonical8")
    color_map = core.load_canonical_network_color_map()
    for smoothing in DEFAULT_SMOOTHINGS:
        for hemi in DEFAULT_HEMIS:
            combo = SubjectCombo(subject, smoothing, hemi)
            src = gradient_combo_output_dir(work_root, combo)
            dst = adapter_run_dir(adapter_root, subject, GRADIENT_CORE_METHOD, smoothing, hemi)
            dst.mkdir(parents=True, exist_ok=True)
            labels = np.load(src / "cluster_labels_k3.npy").astype(np.int32)
            valid_mask = np.load(valid_mask_path(surface_root, subject, hemi)).astype(bool)
            profile = core.cluster_network_profile(
                group_labels_final=labels,
                support_mask=valid_mask,
                subjects=[subject],
                shared_root=shared_root,
                surface_root=surface_root,
                reference_root=shared_root,
                atlas=ATLAS,
                smoothing=smoothing,
                hemi=hemi,
                valid_masks_by_subject={subject: valid_mask},
                profile_fc_mode="pearson",
                network_taxonomy="canonical8",
            )
            roi_profile, roi_registry = core.cluster_roi_profile(
                group_labels_final=labels,
                support_mask=valid_mask,
                subjects=[subject],
                shared_root=shared_root,
                surface_root=surface_root,
                reference_root=shared_root,
                smoothing=smoothing,
                hemi=hemi,
                valid_masks_by_subject={subject: valid_mask},
            )
            assignment_variants = core.cluster_assignment_variants_from_kong2022_roi_profile(
                roi_profile=roi_profile,
                roi_profile_3d=roi_profile[np.newaxis, :, :],
                roi_registry=roi_registry,
                network_names=network_names,
                q_thresh=0.05,
                top_n_cap=20,
                network_column="canonical8",
            )
            functional_variant_payloads: dict[str, dict[str, object]] = {}
            profile_for_render = profile
            if assignment_variant == "weighted_fisherz" and (src / "cluster_network_profile_weighted_fisherz.npy").exists():
                profile_for_render = np.load(src / "cluster_network_profile_weighted_fisherz.npy").astype(np.float32)
            np.save(dst / "group_labels_final.npy", labels.astype(np.int32))
            np.save(dst / "cluster_network_profile.npy", profile_for_render.astype(np.float32))
            np.save(dst / "cluster_roi_profile.npy", roi_profile.astype(np.float32))
            write_json(dst / "group_feature_summary.json", {"network_names": network_names, "N_ref": int(len(network_names))})
            for variant_name, variant_spec in assignment_variants.items():
                variant_rows = [dict(x) for x in variant_spec["status_rows"]]
                functional_labels, functional_meta, label_items = core.functional_atlas_from_clusters(
                    group_labels_final=labels,
                    status_rows=variant_rows,
                    network_names=network_names,
                    color_map=color_map,
                )
                func_label = dst / "template" / "workbench_assets" / f"sub-{subject}_hemi-{hemi}_space-corobl_den-512_label-individual_functional_atlas_gradient-spectral_{smoothing}_{variant_name}.label.gii"
                func_label.parent.mkdir(parents=True, exist_ok=True)
                nib.save(core.make_named_label_gifti(functional_labels.astype(np.int32), label_items), str(func_label))
                np.save(dst / f"group_functional_atlas_labels_{variant_name}.npy", functional_labels.astype(np.int32))
                status_payload = {
                    "items": variant_rows,
                    "assignment_metric": str(variant_spec["assignment_metric"]),
                    "q_threshold": variant_spec["q_threshold"],
                    "top_n_cap": variant_spec["top_n_cap"],
                    "individual_level": True,
                }
                write_json(dst / f"cluster_assignment_status_{variant_name}.json", status_payload)
                write_json(dst / f"group_functional_atlas_label_map_{variant_name}.json", functional_meta)
                functional_variant_payloads[str(variant_name)] = {
                    "cluster_assignment_status": str((dst / f"cluster_assignment_status_{variant_name}.json").resolve()),
                    "group_functional_atlas_labels": str((dst / f"group_functional_atlas_labels_{variant_name}.npy").resolve()),
                    "group_functional_atlas_label_map": str((dst / f"group_functional_atlas_label_map_{variant_name}.json").resolve()),
                    "template_functional_atlas_label": str(func_label.resolve()),
                }
            weighted_status = src / "cluster_assignment_status_weighted_fisherz.json"
            weighted_labels = src / "group_functional_atlas_labels_weighted_fisherz.npy"
            weighted_map = src / "group_functional_atlas_label_map_weighted_fisherz.json"
            weighted_label_candidates = sorted((src / "workbench_assets").glob("*_weighted_fisherz.label.gii"))
            if weighted_status.exists() and weighted_labels.exists() and weighted_map.exists() and weighted_label_candidates:
                local_weighted_label = dst / "template" / "workbench_assets" / weighted_label_candidates[0].name
                local_weighted_label.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(weighted_label_candidates[0], local_weighted_label)
                np.save(dst / "group_functional_atlas_labels_weighted_fisherz.npy", np.load(weighted_labels).astype(np.int32))
                write_json(dst / "cluster_assignment_status_weighted_fisherz.json", read_json(weighted_status))
                write_json(dst / "group_functional_atlas_label_map_weighted_fisherz.json", read_json(weighted_map))
                functional_variant_payloads["weighted_fisherz"] = {
                    "cluster_assignment_status": str((dst / "cluster_assignment_status_weighted_fisherz.json").resolve()),
                    "group_functional_atlas_labels": str((dst / "group_functional_atlas_labels_weighted_fisherz.npy").resolve()),
                    "group_functional_atlas_label_map": str((dst / "group_functional_atlas_label_map_weighted_fisherz.json").resolve()),
                    "template_functional_atlas_label": str(local_weighted_label.resolve()),
                }
            dominant = [str(row.get("top1_network", "unassigned")) for row in assignment_variants["fisherz"]["status_rows"]]
            write_json(dst / "cluster_dominant_network.json", {"dominant": dominant, "network_names": network_names, "assignment_variant": "fisherz"})
            write_json(dst / "cluster_dominant_network_count.json", {"dominant": [str(row.get("top1_network", "unassigned")) for row in assignment_variants["count"]["status_rows"]], "network_names": network_names, "assignment_variant": "count"})
            if (src / "cluster_dominant_network_weighted_fisherz.json").exists():
                write_json(dst / "cluster_dominant_network_weighted_fisherz.json", read_json(src / "cluster_dominant_network_weighted_fisherz.json"))
            write_tsv(dst / "k_selection_summary.tsv", [fixed_k_selection_row(method=GRADIENT_CORE_METHOD, smoothing=smoothing, hemi=hemi, n_support=int(np.sum(valid_mask)))], K_SELECTION_FIELDS)
            surf = subject_surface_path(hippunfold_batch_root, subject, hemi)
            default_variant = assignment_variant if assignment_variant in functional_variant_payloads else "fisherz"
            manifest = {
                "status": "ok",
                "method": GRADIENT_CORE_METHOD,
                "atlas": ATLAS,
                "smoothing": smoothing,
                "hemi": hemi,
                "chosen_k_final": 3,
                "n_subjects": 1,
                "template_subject": subject,
                "template_left_or_right_surface": str(surf.resolve()),
                "template_spec_path": str(subject_spec_path(surf, subject).resolve()),
                "assignment_mode": "cluster_level",
                "default_assignment_variant": default_variant,
                "group_functional_atlas_labels": functional_variant_payloads[default_variant]["group_functional_atlas_labels"],
                "group_functional_atlas_label_map": functional_variant_payloads[default_variant]["group_functional_atlas_label_map"],
                "template_functional_atlas_label": functional_variant_payloads[default_variant]["template_functional_atlas_label"],
                "assignment_variants": functional_variant_payloads,
            }
            write_json(dst / "group_atlas_manifest.json", manifest, sort_keys=True)


def wta_variant_suffix(assignment_variant: str) -> str:
    if assignment_variant == "weighted_fisherz":
        return "_weighted_fisherz"
    if assignment_variant == "normalized_top20_count":
        return "_normalized_top20_count"
    if assignment_variant == "family17_normalized_top20_count":
        return "_family17_normalized_top20_count"
    return ""


def resolve_wta_label_gifti(*, src: Path, subject: str, smoothing: str, hemi: str, suffix: str) -> Path:
    expected = (
        Path(src)
        / "workbench_assets"
        / f"sub-{subject}_hemi-{hemi}_space-corobl_den-512_label-individual_wta_top_{smoothing}{suffix}.label.gii"
    )
    if expected.exists():
        return expected
    if suffix:
        raise FileNotFoundError(f"Missing WTA label GIFTI for suffix {suffix!r}: {expected}")
    candidates = sorted((Path(src) / "workbench_assets").glob("*.label.gii"))
    if not candidates:
        raise FileNotFoundError(f"Missing WTA label GIFTI under {Path(src) / 'workbench_assets'}")
    return candidates[0]


def materialize_wta_core_inputs(*, subject: str, work_root: Path, adapter_root: Path, hippunfold_batch_root: Path, assignment_variant: str = "fisherz") -> None:
    for smoothing in DEFAULT_SMOOTHINGS:
        for hemi in DEFAULT_HEMIS:
            combo = SubjectCombo(subject, smoothing, hemi)
            src = wta_combo_output_dir(work_root, combo)
            dst = adapter_run_dir(adapter_root, subject, WTA_CORE_METHOD, smoothing, hemi)
            dst.mkdir(parents=True, exist_ok=True)
            suffix = wta_variant_suffix(assignment_variant)
            labels = np.load(src / f"wta_top_labels{suffix}.npy").astype(np.int32)
            label_map = read_json(src / f"wta_top_label_map{suffix}.json")
            status = read_json(src / f"vertex_assignment_status{suffix}.json")
            src_label = resolve_wta_label_gifti(src=src, subject=subject, smoothing=smoothing, hemi=hemi, suffix=suffix)
            local_label = dst / "template" / "workbench_assets" / src_label.name
            local_label.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src_label, local_label)
            np.save(dst / "group_functional_atlas_labels.npy", labels)
            write_json(dst / "group_functional_atlas_label_map.json", label_map)
            write_json(dst / "vertex_assignment_status.json", status)
            surf = subject_surface_path(hippunfold_batch_root, subject, hemi)
            write_json(
                dst / "group_atlas_manifest.json",
                {
                    "status": "ok",
                    "method": WTA_CORE_METHOD,
                    "atlas": ATLAS,
                    "smoothing": smoothing,
                    "hemi": hemi,
                    "n_subjects": 1,
                    "template_subject": subject,
                    "template_left_or_right_surface": str(surf.resolve()),
                    "template_spec_path": str(subject_spec_path(surf, subject).resolve()),
                    "template_functional_atlas_label": str(local_label.resolve()),
                    "group_functional_atlas_labels": str((dst / "group_functional_atlas_labels.npy").resolve()),
                    "group_functional_atlas_label_map": str((dst / "group_functional_atlas_label_map.json").resolve()),
                    "assignment_variant": assignment_variant,
                    "confidence_metric": assignment_variant if assignment_variant == "weighted_fisherz" else "topn_count",
                },
                sort_keys=True,
            )


def render_via_core(*, subject: str, method: str, work_root: Path, overview_root: Path, adapter_root: Path, adapter_present_root: Path, shared_root: Path, surface_root: Path, hippunfold_batch_root: Path, scene: Path) -> Path:
    core_overview = core_overview_module()
    method_key = normalize_overview_method(method)
    subject_root = adapter_root / f"sub-{subject}"
    if method_key in {"gradient_spectral_k3", "gradient_count", "gradient_weighted_fisherz"}:
        gradient_variant = "weighted_fisherz" if method_key == "gradient_weighted_fisherz" else "fisherz"
        materialize_gradient_core_inputs(subject=subject, work_root=work_root, adapter_root=adapter_root, shared_root=shared_root, surface_root=surface_root, hippunfold_batch_root=hippunfold_batch_root, assignment_variant=gradient_variant)
        meta = core_overview.render_combo(
            group_root=subject_root,
            present_dir=adapter_present_root / f"sub-{subject}" / "gradient",
            method=GRADIENT_CORE_METHOD,
            atlas=ATLAS,
            coverage_tag=COVERAGE_TAG,
            scene=scene,
            route_tag=ROUTE_TAG,
            mode_tag=("weighted_fisherz" if method_key == "gradient_weighted_fisherz" else None),
            assignment_variant=("weighted_fisherz" if method_key == "gradient_weighted_fisherz" else ("count" if method_key == "gradient_count" else "fisherz")),
            gradient_comp_tag=GRADIENT_COMP_TAG,
        )
    else:
        wta_variant_by_method = {
            "wta_weighted_fisherz": "weighted_fisherz",
            "wta_normalized_top20_count": "normalized_top20_count",
            "wta_family17_normalized_top20_count": "family17_normalized_top20_count",
        }
        wta_variant = wta_variant_by_method.get(method_key, "fisherz")
        materialize_wta_core_inputs(subject=subject, work_root=work_root, adapter_root=adapter_root, hippunfold_batch_root=hippunfold_batch_root, assignment_variant=wta_variant)
        mode_tag = None if wta_variant == "fisherz" else wta_variant
        meta = core_overview.render_combo(
            group_root=subject_root,
            present_dir=adapter_present_root / f"sub-{subject}" / "wta",
            method=WTA_CORE_METHOD,
            atlas=ATLAS,
            coverage_tag=COVERAGE_TAG,
            scene=scene,
            route_tag=ROUTE_TAG,
            mode_tag=mode_tag,
            assignment_variant=("weighted_fisherz" if mode_tag else "fisherz"),
            gradient_comp_tag="",
        )
    final = overview_output_path(overview_root, method_key, subject)
    final.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(Path(str(meta["overview_png"])), final)
    return final


def overview_worker(payload: dict[str, object]) -> dict[str, object]:
    for key, value in THREAD_ENV_OVERRIDES.items():
        os.environ[key] = value
    started = time.time()
    subject = normalize_subject(str(payload["subject"]))
    method = normalize_overview_method(str(payload["method"]))
    try:
        out_path = render_via_core(
            subject=subject,
            method=method,
            work_root=Path(str(payload["work_root"])),
            overview_root=Path(str(payload["overview_root"])),
            adapter_root=Path(str(payload["adapter_root"])),
            adapter_present_root=Path(str(payload["adapter_present_root"])),
            shared_root=Path(str(payload["shared_root"])),
            surface_root=Path(str(payload["surface_root"])),
            hippunfold_batch_root=Path(str(payload["hippunfold_batch_root"])),
            scene=Path(str(payload["scene"])),
        )
        return {"status": "ok", "subject": f"sub-{subject}", "method": method, "task_dir": OVERVIEW_METHODS[method]["task_dir"], "output": str(out_path), "elapsed_seconds": time.time() - started}
    except Exception as exc:
        return {"status": "failed", "subject": f"sub-{subject}", "method": method, "task_dir": OVERVIEW_METHODS[method]["task_dir"], "error_type": exc.__class__.__name__, "error": str(exc), "elapsed_seconds": time.time() - started}


def write_status_tsv(path: Path, rows: list[dict[str, object]]) -> None:
    fields = ["status", "subject", "method", "task_dir", "elapsed_seconds", "output", "error_type", "error"]
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = ["\t".join(fields)]
    for row in rows:
        lines.append("\t".join(str(row.get(field, "")).replace("\t", " ").replace("\n", " ") for field in fields))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build individual overviews by adapting subject-level products to core_workflow/06_01_build_group_overview.py inputs.")
    parser.add_argument("--work-root", default=str(DEFAULT_OUT_ROOT))
    parser.add_argument("--overview-root", default=str(DEFAULT_INDIVIDUAL_OVERVIEW_ROOT))
    parser.add_argument("--adapter-root", default=str(DEFAULT_ADAPTER_ROOT))
    parser.add_argument("--adapter-present-root", default=str(DEFAULT_ADAPTER_PRESENT_ROOT))
    parser.add_argument("--shared-root", default=str(DEFAULT_KONG2022_ROUTEA_ROOT))
    parser.add_argument("--surface-root", default=str(DEFAULT_KONG2022_SURFACE_ROOT))
    parser.add_argument("--hippunfold-batch-root", default=str(DEFAULT_HIPPUNFOLD_BATCH_ROOT))
    parser.add_argument("--scene", default=str(DEFAULT_SCENE))
    parser.add_argument("--subjects-manifest", default=str(DEFAULT_SUBJECTS_MANIFEST))
    parser.add_argument("--subjects", nargs="*", default=None)
    parser.add_argument("--max-subjects", type=int, default=None)
    parser.add_argument("--methods", nargs="+", default=["gradient_spectral_k3", "wta_top"], choices=["gradient", "count", "wta", "gradient_spectral_k3", "gradient_count", "gradient_weighted_fisherz", "wta_top", "wta_weighted_fisherz", "wta_normalized_top20_count", "wta_family17_normalized_top20_count"])
    parser.add_argument("--jobs", type=int, default=DEFAULT_JOBS)
    parser.add_argument("--resume", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    work_root = Path(args.work_root).resolve()
    overview_root = Path(args.overview_root).resolve()
    adapter_root = Path(args.adapter_root).resolve()
    adapter_present_root = Path(args.adapter_present_root).resolve()
    subjects = [normalize_subject(s) for s in args.subjects] if args.subjects else load_subject_ids(Path(args.subjects_manifest))
    if args.max_subjects is not None:
        subjects = subjects[: int(args.max_subjects)]
    methods = [normalize_overview_method(method) for method in args.methods]

    tasks: list[dict[str, object]] = []
    rows: list[dict[str, object]] = []
    for subject in subjects:
        for method in methods:
            expected = overview_output_path(overview_root, method, subject)
            if args.resume and expected.exists():
                rows.append({"status": "ok", "subject": f"sub-{subject}", "method": method, "task_dir": OVERVIEW_METHODS[method]["task_dir"], "output": str(expected), "elapsed_seconds": 0.0, "resumed": True})
                continue
            tasks.append({"work_root": str(work_root), "overview_root": str(overview_root), "adapter_root": str(adapter_root), "adapter_present_root": str(adapter_present_root), "shared_root": str(Path(args.shared_root).resolve()), "surface_root": str(Path(args.surface_root).resolve()), "hippunfold_batch_root": str(Path(args.hippunfold_batch_root).resolve()), "scene": str(Path(args.scene).resolve()), "subject": subject, "method": method})

    jobs = max(1, min(int(args.jobs), 10, len(tasks) if tasks else 1))
    started = time.time()
    if tasks:
        with ProcessPoolExecutor(max_workers=jobs) as executor:
            future_to_task = {executor.submit(overview_worker, task): task for task in tasks}
            for future in as_completed(future_to_task):
                rows.append(future.result())
                write_status_tsv(overview_root / "overview_status.tsv", sorted(rows, key=lambda x: (str(x.get("method", "")), str(x.get("subject", "")))))
    rows = sorted(rows, key=lambda x: (str(x.get("method", "")), str(x.get("subject", ""))))
    write_status_tsv(overview_root / "overview_status.tsv", rows)
    summary = {"status": "failed" if any(str(row.get("status")) == "failed" for row in rows) else "ok", "workflow": "individual_core_06_01_overview_adapter", "work_root": str(work_root), "adapter_root": str(adapter_root), "overview_root": str(overview_root), "subjects": len(subjects), "methods": methods, "total": len(rows), "ok": sum(1 for row in rows if str(row.get("status")) == "ok"), "failed": sum(1 for row in rows if str(row.get("status")) == "failed"), "jobs": jobs, "elapsed_seconds": time.time() - started, "git_commit": git_commit(short=True), "updated_at": utc_now_iso(zulu=True), "status_tsv": str((overview_root / "overview_status.tsv").resolve())}
    write_json(overview_root / "overview_manifest.json", {"summary": summary, "items": rows}, sort_keys=True)
    print(f"status={summary['status']} ok={summary['ok']} failed={summary['failed']} total={summary['total']}")
    print(f"overview_root={overview_root}")
    print(f"adapter_root={adapter_root}")
    print(f"status_tsv={overview_root / 'overview_status.tsv'}")
    print(f"manifest={overview_root / 'overview_manifest.json'}")
    return 1 if summary["failed"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
