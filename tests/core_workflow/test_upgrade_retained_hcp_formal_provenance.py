from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = REPO_ROOT / "scripts/core_workflow/upgrade_retained_hcp_formal_provenance.py"


def load_module():
    spec = importlib.util.spec_from_file_location("upgrade_retained_hcp", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write(path: Path, data: bytes = b"artifact") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return path.resolve()


def build_fixture(root: Path) -> tuple[Path, list[Path], list[Path]]:
    hcp = root / "outputs_hcp"
    subjects = [f"sub-{100000 + index}" for index in range(151)]
    subjects_manifest = hcp / "inputs/hcp/kong2022/manifests/hcp_7t_hippocampus_fmri_qc_pass_t1only_latest.txt"
    _write(subjects_manifest, ("\n".join(subjects) + "\n").encode())
    shared = hcp / "inputs/hcp/kong2022/shared_inputs/routea"
    surface = hcp / "inputs/hcp/kong2022/shared_inputs/surface"
    gradients = hcp / "work/hippomaps_gradients/routea"
    for directory in (shared, surface, gradients):
        directory.mkdir(parents=True, exist_ok=True)
    manifests: list[Path] = []
    binaries: list[Path] = []
    for branch, route in (
        ("routea_full_k3_fisherz", "routea"),
        ("routea_17net_full_k3_fisherz", "routea_17net"),
    ):
        for smoothing in ("2mm", "4mm"):
            for hemi in ("L", "R"):
                run = hcp / "work/group_atlas" / branch / "gradient-spectral/kong2022" / smoothing / "cov_0p80/g1g2g3" / f"hemi_{hemi}"
                group_labels = _write(run / "group_labels_final.npy", b"labels")
                support = _write(run / "group_support_mask.npy", b"support")
                gradient_features = _write(run / "group_gradient_features_active.npy", b"features")
                profile = _write(run / "cluster_network_profile.npy", b"profile")
                consumed = _write(shared / f"fixture_{branch}_{smoothing}_{hemi}.npy")
                binaries.extend((group_labels, support, gradient_features, profile, consumed))
                variants: dict[str, object] = {}
                for variant in ("fisherz", "count"):
                    variants[variant] = {
                        "cluster_assignment_status": str(_write(run / f"cluster_assignment_status_{variant}.json")),
                        "group_functional_atlas_labels": str(_write(run / f"group_functional_atlas_labels_{variant}.npy")),
                        "group_functional_atlas_label_map": str(_write(run / f"group_functional_atlas_label_map_{variant}.json")),
                        "template_functional_atlas_label": str(_write(run / "template/workbench_assets" / f"{variant}.label.gii")),
                    }
                payload = {
                    "status": "ok",
                    "method": "gradient-spectral",
                    "atlas": "kong2022",
                    "smoothing": smoothing,
                    "hemi": hemi,
                    "network_taxonomy": "canonical8" if route == "routea" else "family17",
                    "surface_affinity": "shifted-cosine",
                    "affinity_description": "((1+cosine_similarity)/2)_hadamard_surface_adjacency_diag0",
                    "clustering_algorithm": "spectral_clustering_precomputed_affinity_shifted_cosine_diag0",
                    "candidate_k_range_requested": [2, 8],
                    "chosen_k_final": 3,
                    "affinity_graph_diagnostics": {"n_connected_components": 1},
                    "shared_root": str(shared.resolve()),
                    "surface_root": str(surface.resolve()),
                    "profile_surface_root": str(surface.resolve()),
                    "reference_root": str(shared.resolve()),
                    "group_gradient_features_active": str(gradient_features),
                    "group_labels_final": str(group_labels),
                    "group_support_mask": str(support),
                    "cluster_network_profile": str(profile),
                    "actual_consumed_input_paths": {"fixture": str(consumed)},
                    "subjects": subjects,
                    "n_subjects": 151,
                    "assignment_variants": variants,
                    "resume_input_signature_payload": {
                        "route_tag": route,
                        "gradient_root": str(gradients.resolve()),
                        "subjects_manifest": str(subjects_manifest.resolve()),
                        "input_category_hashes": {"primary": "b" * 64},
                    },
                }
                payload["resume_input_signature"] = hashlib.sha256(
                    json.dumps(
                        payload["resume_input_signature_payload"],
                        sort_keys=True,
                        separators=(",", ":"),
                        ensure_ascii=True,
                    ).encode()
                ).hexdigest()
                manifest = run / "group_atlas_manifest.json"
                manifest.parent.mkdir(parents=True, exist_ok=True)
                manifest.write_text(json.dumps(payload), encoding="utf-8")
                manifests.append(manifest.resolve())
    return subjects_manifest.resolve(), manifests, binaries


def test_upgrade_enriches_exact_eight_without_mutating_binary_assets(tmp_path: Path) -> None:
    module = load_module()
    subjects_manifest, manifests, binaries = build_fixture(tmp_path)
    before = {path: hashlib.sha256(path.read_bytes()).hexdigest() for path in binaries}

    report = module.upgrade_retained_manifests(tmp_path, subjects_manifest)

    assert report["status"] == "ok"
    assert report["updated_count"] == 8
    for manifest in manifests:
        payload = json.loads(manifest.read_text())
        assert payload["dataset"] == "outputs_hcp"
        assert payload["expected_source_root"] == str((tmp_path / "outputs_hcp").resolve())
        assert payload["provenance_upgrade"]["artifact_sha256"]
    assert before == {path: hashlib.sha256(path.read_bytes()).hexdigest() for path in binaries}


@pytest.mark.parametrize("fault", ["legacy", "cross-root", "truncated", "fingerprint"])
def test_upgrade_rejects_invalid_input_without_partial_writes(tmp_path: Path, fault: str) -> None:
    module = load_module()
    subjects_manifest, manifests, _binaries = build_fixture(tmp_path)
    if fault == "legacy":
        payload = json.loads(manifests[-1].read_text())
        payload["surface_affinity"] = "legacy-affinity"
        manifests[-1].write_text(json.dumps(payload))
    elif fault == "cross-root":
        payload = json.loads(manifests[-1].read_text())
        wrong = _write(tmp_path / "outputs_stanford/input.npy")
        payload["actual_consumed_input_paths"]["fixture"] = str(wrong)
        manifests[-1].write_text(json.dumps(payload))
    elif fault == "truncated":
        subjects_manifest.write_text("\n".join(f"sub-{100000 + index}" for index in range(150)) + "\n")
    else:
        payload = json.loads(manifests[-1].read_text())
        payload["resume_input_signature_payload"]["force_final_k"] = 4
        manifests[-1].write_text(json.dumps(payload))

    with pytest.raises((ValueError, FileNotFoundError)):
        module.upgrade_retained_manifests(tmp_path, subjects_manifest)

    assert all("dataset" not in json.loads(path.read_text()) for path in manifests)
