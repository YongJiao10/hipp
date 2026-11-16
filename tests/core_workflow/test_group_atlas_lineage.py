from __future__ import annotations

import json
from lineage_fixtures import make_run
from pathlib import Path
import numpy as np
import pytest

from scripts.core_workflow.lib.group_atlas_lineage import (
    validate_group_atlas_lineage,
    repair_self_referencing_manifest,
)


def test_consumed_upstream_gradients_do_not_claim_local_recomputation(tmp_path):
    from scripts.core_workflow.lib import group_atlas_lineage as lineage
    path = make_run(tmp_path / "consumed")
    summary = json.loads((path.parent / "group_gradient_feature_summary.json").read_text())
    fields = lineage.gradient_computation_provenance(summary, fixed_cluster_reuse=None)
    assert fields["gradient_recomputed"] is False
    assert fields["clustering_recomputed"] is True
    manifest = json.loads(path.read_text())
    manifest.update(fields)
    path.write_text(json.dumps(manifest))
    assert validate_group_atlas_lineage(path)["mode"] == "upstream_gradients_consumed"
    manifest["gradient_recomputed"] = True
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="Consumed upstream"):
        validate_group_atlas_lineage(path)


def test_family17_upstream_gradient_lineage_requires_selected_taxonomy(tmp_path):
    from scripts.core_workflow.lib import group_atlas_lineage as lineage

    path = make_run(tmp_path / "family17", fields={"network_taxonomy": "family17"})
    summary = json.loads((path.parent / "group_gradient_feature_summary.json").read_text())
    manifest = json.loads(path.read_text())
    manifest.update(lineage.gradient_computation_provenance(summary, fixed_cluster_reuse=None))
    path.write_text(json.dumps(manifest))

    with pytest.raises(ValueError, match="canonical7"):
        validate_group_atlas_lineage(path)
    result = validate_group_atlas_lineage(path, expected_network_taxonomy="family17")
    assert result["mode"] == "upstream_gradients_consumed"
    assert result["network_taxonomy"] == "family17"


def test_family17_reuses_canonical7_parent_labels(tmp_path):
    parent = make_run(tmp_path / "canonical7")
    child = make_run(tmp_path / "family17", fields={
        "network_taxonomy": "family17",
        "gradient_recomputed": False,
        "clustering_recomputed": False,
        "k_selection_recomputed": False,
        "fixed_cluster_reuse": {
            "source_manifest": str(parent),
            "source_network_taxonomy": "canonical7",
        },
    })
    result = validate_group_atlas_lineage(child, expected_network_taxonomy="family17")
    assert result["mode"] == "active_parent_declared"
    assert result["source_manifest"] == str(parent)


def test_gradient_lineage_validates_declared_unit_sd_transform(tmp_path):
    from scripts.core_workflow.lib.hippomaps_adapter import scale_gradient_features
    path = make_run(tmp_path / "scaled")
    source_path = path.parent / "group_gradient_feature_summary.json"
    source = json.loads(source_path.read_text())
    features = np.load(path.parent / "group_gradient_features_active.npy")
    scaled, transform = scale_gradient_features(features, scaling="unit_sd")
    np.save(path.parent / "group_gradient_features_active.npy", scaled)
    source["clustering_feature_transform"] = transform
    source_path.write_text(json.dumps(source))
    assert validate_group_atlas_lineage(path)["status"] == "ok"


def _create_mock_run(run_dir: Path, manifest_data: dict, with_artifacts: bool = True) -> Path:
    fields = {"gradient_recomputed": False, "clustering_recomputed": False, **manifest_data}
    if "historical_source" in fields:
        fields["historical_source"] = {
            "source_run_id": "verified-fixture-run",
            "archive_role": "historical_only_not_runtime_input", **fields["historical_source"]}
    return make_run(run_dir, fields=fields, with_artifacts=with_artifacts)


def test_self_reference_rejected(tmp_path: Path):
    run_a = tmp_path / "run_a"
    manifest_a = run_a / "group_atlas_manifest.json"
    data_a = {
        "status": "ok",
        "network_taxonomy": "canonical7",
        "source_group_atlas_manifest": str(manifest_a),
    }
    _create_mock_run(run_a, data_a)

    with pytest.raises(ValueError, match="Self-referencing"):
        validate_group_atlas_lineage(manifest_a)


def test_cyclic_lineage_rejected(tmp_path: Path):
    run_a = tmp_path / "run_a"
    run_b = tmp_path / "run_b"
    manifest_a = run_a / "group_atlas_manifest.json"
    manifest_b = run_b / "group_atlas_manifest.json"

    data_a = {
        "status": "ok",
        "network_taxonomy": "canonical7",
        "source_group_atlas_manifest": str(manifest_b),
    }
    data_b = {
        "status": "ok",
        "network_taxonomy": "canonical7",
        "source_group_atlas_manifest": str(manifest_a),
    }

    _create_mock_run(run_a, data_a)
    _create_mock_run(run_b, data_b)

    with pytest.raises(ValueError, match="Cyclic manifest lineage"):
        validate_group_atlas_lineage(manifest_a)


def test_legitimate_weighted_to_unweighted_passes(tmp_path: Path):
    run_unweighted = tmp_path / "unweighted"
    run_weighted = tmp_path / "weighted"
    manifest_unweighted = run_unweighted / "group_atlas_manifest.json"
    manifest_weighted = run_weighted / "group_atlas_manifest.json"

    data_unweighted = {
        "status": "ok",
        "network_taxonomy": "canonical7",
        "source_group_atlas_manifest": None,
        "historical_source": {
            "kind": "canonical8_fixed_cluster_reuse",
            "source_network_taxonomy": "canonical8",
            "archive_role": "historical_only_not_runtime_input",
        },
    }
    data_weighted = {
        "status": "ok",
        "network_taxonomy": "canonical7",
        "source_group_atlas_manifest": str(manifest_unweighted),
    }

    _create_mock_run(run_unweighted, data_unweighted)
    _create_mock_run(run_weighted, data_weighted)

    res = validate_group_atlas_lineage(manifest_weighted)
    assert res["status"] == "ok"
    assert res["mode"] == "active_parent_declared"


def test_missing_active_parent_rejected(tmp_path: Path):
    run_a = tmp_path / "run_a"
    missing_parent = tmp_path / "non_existent_parent" / "group_atlas_manifest.json"
    manifest_a = run_a / "group_atlas_manifest.json"
    data_a = {
        "status": "ok",
        "network_taxonomy": "canonical7",
        "source_group_atlas_manifest": str(missing_parent),
    }
    _create_mock_run(run_a, data_a)

    with pytest.raises(FileNotFoundError, match="Declared parent"):
        validate_group_atlas_lineage(manifest_a)


def test_historical_mode_without_artifacts_rejected(tmp_path: Path):
    run_a = tmp_path / "run_a"
    manifest_a = run_a / "group_atlas_manifest.json"
    data_a = {
        "status": "ok",
        "network_taxonomy": "canonical7",
        "source_group_atlas_manifest": None,
        "historical_source": {
            "kind": "canonical8_fixed_cluster_reuse",
            "source_network_taxonomy": "canonical8",
        },
    }
    _create_mock_run(run_a, data_a, with_artifacts=False)

    with pytest.raises(FileNotFoundError, match="cluster_roi_profile"):
        validate_group_atlas_lineage(manifest_a)


def test_historical_mode_with_nonexistent_historical_reference_succeeds(tmp_path: Path):
    run_a = tmp_path / "run_a"
    manifest_a = run_a / "group_atlas_manifest.json"
    non_existent_historical_archive = "/tmp/archive_does_not_exist_2026/group_atlas_manifest.json"

    data_a = {
        "status": "ok",
        "network_taxonomy": "canonical7",
        "source_group_atlas_manifest": None,
        "historical_source": {
            "kind": "canonical8_fixed_cluster_reuse",
            "source_network_taxonomy": "canonical8",
            "archive_role": "historical_only_not_runtime_input",
            "historical_reference_path": non_existent_historical_archive,
        },
    }
    _create_mock_run(run_a, data_a, with_artifacts=True)

    # Validates without needing historical archive to exist!
    res = validate_group_atlas_lineage(manifest_a)
    assert res["status"] == "ok"
    assert res["mode"] == "historical_fixed_cluster_reuse"


def test_claims_canonical8_source_pointing_to_canonical7_self_rejected(tmp_path: Path):
    run_a = tmp_path / "run_a"
    manifest_a = run_a / "group_atlas_manifest.json"
    data_a = {
        "status": "ok",
        "network_taxonomy": "canonical7",
        "source_network_taxonomy": "canonical8",
        "source_group_atlas_manifest": str(manifest_a),
    }
    _create_mock_run(run_a, data_a)

    with pytest.raises(ValueError, match="Self-referencing"):
        validate_group_atlas_lineage(manifest_a)


def test_recomputed_flag_conflicting_with_reuse_rejected(tmp_path: Path):
    run_a = tmp_path / "run_a"
    manifest_a = run_a / "group_atlas_manifest.json"
    data_a = {
        "status": "ok",
        "network_taxonomy": "canonical7",
        "gradient_recomputed": True,
        "clustering_recomputed": True,
        "historical_source": {
            "kind": "canonical8_fixed_cluster_reuse",
            "source_network_taxonomy": "canonical8",
        },
    }
    _create_mock_run(run_a, data_a)

    with pytest.raises(ValueError, match="Conflict in manifest provenance"):
        validate_group_atlas_lineage(manifest_a)


def test_repair_self_referencing_manifest(tmp_path: Path):
    run_a = tmp_path / "run_a"
    manifest_a = run_a / "group_atlas_manifest.json"
    data_a = {
        "status": "ok",
        "network_taxonomy": "canonical7",
        "source_network_taxonomy": "canonical8",
        "source_group_atlas_manifest": str(manifest_a),
        "source_group_run": str(run_a),
        "source_path_repair": {
            "original_missing_declarations": {
                "source_group_atlas_manifest": "/path/to/old/8net/manifest.json"
            }
        },
        "taxonomy_transform": {
            "source_group_manifest": str(manifest_a)
        }
    }
    _create_mock_run(run_a, data_a, with_artifacts=True)

    repaired = repair_self_referencing_manifest(manifest_a, historical_source={
        "kind": "canonical8_fixed_cluster_reuse", "source_network_taxonomy": "canonical8",
        "archive_role": "historical_only_not_runtime_input",
        "historical_reference_path": "/path/to/old/8net/manifest.json",
    })
    assert repaired is True

    # Now validate_group_atlas_lineage must pass!
    res = validate_group_atlas_lineage(manifest_a)
    assert res["status"] == "ok"
    assert res["mode"] == "historical_fixed_cluster_reuse"
    assert res["source_manifest"] is None

    reloaded = json.loads(manifest_a.read_text(encoding="utf-8"))
    assert reloaded["source_group_atlas_manifest"] is None
    assert reloaded["historical_source"]["source_network_taxonomy"] == "canonical8"
    assert reloaded["historical_source"]["historical_reference_path"] == "/path/to/old/8net/manifest.json"
    assert reloaded["taxonomy_transform"]["source_group_manifest"] == "/path/to/old/8net/manifest.json"
