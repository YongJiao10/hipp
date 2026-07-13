#!/usr/bin/env python3
from __future__ import annotations

import importlib.util
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
MODULE_PATH = REPO_ROOT / "scripts/core_workflow/audit_formal_gradient_spectral_release.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("formal_gradient_release_audit", MODULE_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load module: {MODULE_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


module = _load_module()


@dataclass(frozen=True)
class FormalFixture:
    contract: dict[str, object]
    manifests: tuple[Path, ...]
    group_manifests: dict[tuple[str, str, str], tuple[Path, ...]]
    overview_manifest: Path


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def build_formal_fixture(
    repo_root: Path,
    *,
    surface_affinity: str,
) -> FormalFixture:
    contract = module.default_contract()
    manifest_paths: list[Path] = []
    group_manifests: dict[tuple[str, str, str], tuple[Path, ...]] = {}
    overview_items: list[dict[str, object]] = []
    for group in contract["groups"]:
        identity = (
            str(group["output_root"]),
            str(group["branch"]),
            str(group["atlas"]),
        )
        group_root = (
            repo_root / identity[0] / "work/group_atlas" / identity[1]
        )
        group_paths: list[Path] = []
        for smoothing in ("2mm", "4mm"):
            for hemi in ("L", "R"):
                manifest = (
                    group_root
                    / "gradient-spectral"
                    / identity[2]
                    / smoothing
                    / "cov_0p80/g1g2g3"
                    / f"hemi_{hemi}/group_atlas_manifest.json"
                )
                payload: dict[str, object] = {
                    "status": "ok",
                    "method": "gradient-spectral",
                    "atlas": identity[2],
                    "smoothing": smoothing,
                    "hemi": hemi,
                    "surface_affinity": surface_affinity,
                    "affinity_description": module.SHIFTED_COSINE_DESCRIPTION,
                    "clustering_algorithm": (
                        "spectral_clustering_precomputed_affinity_"
                        "shifted_cosine_diag0_kmeans_n_init_50"
                    ),
                    "candidate_k_range_requested": [2, 8],
                    "chosen_k_final": 3,
                    "affinity_graph_diagnostics": {"n_connected_components": 1},
                    "shared_root": str((repo_root / "inputs/shared").resolve()),
                    "surface_root": str((repo_root / "inputs/surface").resolve()),
                    "reference_root": str((repo_root / "inputs/reference").resolve()),
                    "group_gradient_features_active": str(
                        (manifest.parent / "group_gradient_features_active.npy").resolve()
                    ),
                    "actual_consumed_input_paths": {
                        "sub-001": str((repo_root / "inputs/features.npy").resolve())
                    },
                    "default_assignment_variant": group["assignment_variant"],
                    "assignment_variants": {
                        str(group["assignment_variant"]): {"status": "ok"}
                    },
                    "resume_input_signature_payload": {
                        "route_tag": group["route_tag"],
                        "gradient_root": str((repo_root / "inputs/gradients").resolve()),
                    },
                }
                if group["assignment_variant"] == "weighted_fisherz":
                    payload.update(
                        {
                            "clustering_surface_affinity": "shifted-cosine",
                            "clustering_affinity_description": (
                                module.SHIFTED_COSINE_DESCRIPTION
                            ),
                            "weighted_assignment_only": True,
                        }
                    )
                _write_json(manifest, payload)
                manifest_paths.append(manifest.resolve())
                group_paths.append(manifest.resolve())
        group_manifests[identity] = tuple(group_paths)
        overview_items.append(
            {
                "atlas": identity[2],
                "method": "gradient-spectral",
                "assignment_variant": group["assignment_variant"],
                "source_group_root": str(group_root.resolve()),
                "surface_affinity": surface_affinity,
                "affinity_description": module.SHIFTED_COSINE_DESCRIPTION,
                "source_group_manifests": [str(path) for path in group_paths],
                "overview_png": str(
                    (
                        repo_root
                        / identity[0]
                        / "present/overviews/formal"
                        / f"{identity[1]}.png"
                    ).resolve()
                ),
            }
        )

    overview_manifest = repo_root / "outputs_hcp/present/overviews/formal/group_overview_manifest.json"
    _write_json(
        overview_manifest,
        {
            "count": len(overview_items),
            "items": overview_items,
        },
    )
    return FormalFixture(
        contract,
        tuple(manifest_paths),
        group_manifests,
        overview_manifest,
    )


def test_audit_rejects_legacy_group_manifest(tmp_path: Path) -> None:
    fixture = build_formal_fixture(tmp_path, surface_affinity="relu-cosine")
    report = module.audit_release(tmp_path, fixture.contract)
    assert report["status"] == "failed"
    assert any("surface_affinity" in item for item in report["failures"])


def test_audit_rejects_overview_with_missing_lineage(tmp_path: Path) -> None:
    fixture = build_formal_fixture(tmp_path, surface_affinity="shifted-cosine")
    fixture.overview_manifest.unlink()
    report = module.audit_release(tmp_path, fixture.contract)
    assert report["status"] == "failed"


def test_audit_accepts_complete_release(tmp_path: Path) -> None:
    fixture = build_formal_fixture(tmp_path, surface_affinity="shifted-cosine")
    report = module.audit_release(tmp_path, fixture.contract)
    assert report["status"] == "ok"
    assert report["group_combinations"] == 36
    assert report["overview_items"] == 9
    assert report["failures"] == []


def test_audit_ignores_nonformal_assignment_item_from_same_group(tmp_path: Path) -> None:
    fixture = build_formal_fixture(tmp_path, surface_affinity="shifted-cosine")
    payload = json.loads(fixture.overview_manifest.read_text(encoding="utf-8"))
    count_item = dict(payload["items"][0])
    count_item["assignment_variant"] = "count"
    count_item["surface_affinity"] = "unknown"
    payload["items"].append(count_item)
    _write_json(fixture.overview_manifest, payload)

    assert module.audit_release(tmp_path, fixture.contract)["status"] == "ok"


def test_default_contract_uses_canonical_lynch_route_tags() -> None:
    groups = {
        str(group["atlas"]): group
        for group in module.default_contract()["groups"]
        if str(group["atlas"]).startswith("lynch")
    }

    assert groups["lynch2024_20net"]["route_tag"] == "lynch2024_20net"
    assert groups["lynch2024_8net"]["route_tag"] == "lynch2024_8net"
    assert groups["lynch2024_20net"]["assignment_variant"] == "fisherz"
    assert groups["lynch2024_8net"]["assignment_variant"] == "fisherz"


def test_default_contract_uses_routea_for_stanford_subgroups() -> None:
    groups = [
        group
        for group in module.default_contract()["groups"]
        if group["output_root"] == "outputs_stanford"
    ]

    assert {group["route_tag"] for group in groups} == {"routea"}


def test_default_contract_has_exact_formal_group_identities() -> None:
    identities = {
        (group["output_root"], group["branch"], group["atlas"])
        for group in module.default_contract()["groups"]
    }

    assert identities == set(module.FORMAL_GROUPS)
    assert len(module.default_contract()["groups"]) == len(module.FORMAL_GROUPS)


def test_audit_rejects_empty_subset_extra_and_out_of_scope_contracts(
    tmp_path: Path,
) -> None:
    fixture = build_formal_fixture(tmp_path, surface_affinity="shifted-cosine")
    valid_groups = fixture.contract["groups"]
    invalid_groups = (
        [],
        valid_groups[:-1],
        [*valid_groups, valid_groups[0]],
        [
            *valid_groups[:-1],
            {
                "output_root": "outputs_hcp",
                "branch": "routeb_full_k3_fisherz",
                "atlas": "kong2022",
                "route_tag": "routeb",
                "assignment_variant": "fisherz",
            },
        ],
        [
            *valid_groups[:-1],
            {
                "output_root": "outputs_hcp",
                "branch": "routea_ctxreg_full_k3_fisherz",
                "atlas": "kong2022",
                "route_tag": "routea_ctxreg",
                "assignment_variant": "fisherz",
            },
        ],
    )

    for groups in invalid_groups:
        contract = {**fixture.contract, "groups": groups}
        report = module.audit_release(tmp_path, contract)
        assert report["status"] == "failed"
        assert any("formal group identities" in item for item in report["failures"])


def test_audit_rejects_missing_or_wrong_weighted_affinity_description(
    tmp_path: Path,
) -> None:
    fixture = build_formal_fixture(tmp_path, surface_affinity="shifted-cosine")
    weighted_identity = (
        "outputs_hcp",
        "routea_weighted_fisherz_full_k3",
        "kong2022",
    )
    manifest = fixture.group_manifests[weighted_identity][0]
    original = json.loads(manifest.read_text(encoding="utf-8"))
    for invalid in (None, "wrong-description"):
        payload = dict(original)
        if invalid is None:
            payload.pop("clustering_affinity_description")
        else:
            payload["clustering_affinity_description"] = invalid
        _write_json(manifest, payload)
        report = module.audit_release(tmp_path, fixture.contract)
        assert report["status"] == "failed"
        assert any(
            "clustering_affinity_description" in item
            for item in report["failures"]
        )


def test_audit_reports_all_manifest_faults_in_one_pass(tmp_path: Path) -> None:
    fixture = build_formal_fixture(tmp_path, surface_affinity="shifted-cosine")
    first = json.loads(fixture.manifests[0].read_text(encoding="utf-8"))
    first["candidate_k_range_requested"] = [3, 8]
    _write_json(fixture.manifests[0], first)
    fixture.manifests[1].unlink()

    failures = module.audit_release(tmp_path, fixture.contract)["failures"]

    assert any("candidate_k_range_requested" in failure for failure in failures)
    assert any(str(fixture.manifests[1]) in failure for failure in failures)


def test_audit_rejects_cross_branch_overview_lineage(tmp_path: Path) -> None:
    fixture = build_formal_fixture(tmp_path, surface_affinity="shifted-cosine")
    payload = json.loads(fixture.overview_manifest.read_text(encoding="utf-8"))
    payload["items"][0]["source_group_manifests"][0] = str(
        (tmp_path / "outputs_hcp/work/group_atlas/other/group_atlas_manifest.json").resolve()
    )
    _write_json(fixture.overview_manifest, payload)

    report = module.audit_release(tmp_path, fixture.contract)

    assert report["status"] == "failed"
    assert any("source_group_manifests" in item for item in report["failures"])


def test_audit_rejects_missing_upstream_gradient_source(tmp_path: Path) -> None:
    fixture = build_formal_fixture(tmp_path, surface_affinity="shifted-cosine")
    payload = json.loads(fixture.manifests[0].read_text(encoding="utf-8"))
    payload["resume_input_signature_payload"].pop("gradient_root")
    _write_json(fixture.manifests[0], payload)

    report = module.audit_release(tmp_path, fixture.contract)

    assert report["status"] == "failed"
    assert any("gradient_root" in item for item in report["failures"])


def test_allow_incomplete_only_changes_cli_exit_status(tmp_path: Path) -> None:
    fixture = build_formal_fixture(tmp_path, surface_affinity="shifted-cosine")
    fixture.overview_manifest.unlink()
    output = tmp_path / "reports/audit.json"

    exit_code = module.main(
        [
            "--repo-root",
            str(tmp_path),
            "--output-json",
            str(output),
            "--allow-incomplete",
        ]
    )

    assert exit_code == 0
    assert json.loads(output.read_text(encoding="utf-8"))["status"] == "failed"


def test_cli_rejects_unplanned_contract_override() -> None:
    with pytest.raises(SystemExit):
        module._parser().parse_args(["--contract-json", "{}"])
