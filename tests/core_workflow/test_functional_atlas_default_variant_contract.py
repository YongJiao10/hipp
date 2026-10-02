#!/usr/bin/env python3
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
CORE_DIR = REPO_ROOT / "scripts" / "core_workflow"
if str(CORE_DIR) not in sys.path:
    sys.path.insert(0, str(CORE_DIR))


def _load_numbered_module(script_name: str, alias: str):
    module_path = CORE_DIR / script_name
    spec = importlib.util.spec_from_file_location(alias, module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load module: {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[alias] = module
    spec.loader.exec_module(module)
    return module


overview_mod = _load_numbered_module("06_01_build_group_overview.py", "core_06_01_default_variant_contract")

from lib.gifti_labels import materialize_combo, resolve_functional_atlas_array_paths  # noqa: E402
from lib.workflow_utils import write_json  # noqa: E402


def _colors() -> dict[str, tuple[int, int, int, int]]:
    from lib.surface_utils import network_color_map

    canonical = network_color_map()
    return {
        "Default": canonical["Default"],
        "Visual": canonical["Visual"],
    }


def test_materialize_combo_writes_default_variant_without_unsuffixed_duplicate(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    asset_dir = run_dir / "template" / "workbench_assets"
    asset_dir.mkdir(parents=True)
    np.save(run_dir / "group_labels_final.npy", np.asarray([1, 1, 2, 2], dtype=np.int32))
    write_json(run_dir / "group_feature_summary.json", {"network_names": ["Default", "Visual"]})
    write_json(
        run_dir / "cluster_assignment_status_normalized_top20_count.json",
        {
            "assignment_metric": "top20_count_normalized_by_network_roi_count",
            "items": [
                {
                    "cluster_id": 1,
                    "status": "dominant",
                    "top1_network": "Default",
                    "normalized_count_by_network": {"Default": 1.0},
                },
                {
                    "cluster_id": 2,
                    "status": "dominant",
                    "top1_network": "Visual",
                    "normalized_count_by_network": {"Visual": 1.0},
                },
            ]
        },
    )
    manifest_path = run_dir / "group_atlas_manifest.json"
    write_json(
        manifest_path,
        {
            "status": "ok",
            "assignment_mode": "cluster_level",
            "default_assignment_variant": "normalized_top20_count",
            "template_cluster_label": str(
                asset_dir / "demo_label-group_cluster_gradient-spectral_kong2022_2mm_cov-0p80.label.gii"
            ),
            "assignment_variants": {
                "normalized_top20_count": {
                    "cluster_assignment_status": str((run_dir / "cluster_assignment_status_normalized_top20_count.json").resolve()),
                    "group_functional_atlas_labels": str(
                        (run_dir / "group_functional_atlas_labels_normalized_top20_count.npy").resolve()
                    ),
                    "group_functional_atlas_label_map": str(
                        (run_dir / "group_functional_atlas_label_map_normalized_top20_count.json").resolve()
                    ),
                    "template_functional_atlas_label": str(
                        (
                            asset_dir
                            / "demo_label-group_functional_atlas_gradient-spectral_kong2022_2mm_cov-0p80_normalized_top20_count.label.gii"
                        ).resolve()
                    ),
                }
            },
        },
    )

    assert materialize_combo(manifest_path, _colors()) is True

    refreshed = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert Path(refreshed["group_functional_atlas_labels"]).name == "group_functional_atlas_labels_normalized_top20_count.npy"
    assert Path(refreshed["group_functional_atlas_label_map"]).name == "group_functional_atlas_label_map_normalized_top20_count.json"
    assert not (run_dir / "group_functional_atlas_labels.npy").exists()
    assert not (run_dir / "group_functional_atlas_label_map.json").exists()
    np.testing.assert_array_equal(
        np.load(refreshed["group_functional_atlas_labels"]),
        np.asarray([1, 1, 2, 2], dtype=np.int32),
    )


def test_default_functional_array_resolution_requires_manifest_variant_for_cluster_level(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    labels_path = run_dir / "group_functional_atlas_labels_normalized_top20_count.npy"
    map_path = run_dir / "group_functional_atlas_label_map_normalized_top20_count.json"
    run_dir.mkdir()
    np.save(labels_path, np.asarray([1, 2], dtype=np.int32))
    write_json(map_path, {"network_names": ["Default", "Visual"]})
    manifest = {
        "assignment_mode": "cluster_level",
        "default_assignment_variant": "normalized_top20_count",
        "assignment_variants": {
            "normalized_top20_count": {
                "group_functional_atlas_labels": str(labels_path.resolve()),
                "group_functional_atlas_label_map": str(map_path.resolve()),
            }
        },
    }

    labels_got, map_got = resolve_functional_atlas_array_paths(manifest, run_dir)

    assert labels_got == labels_path.resolve()
    assert map_got == map_path.resolve()


def test_functional_array_resolution_rejects_declared_path_outside_run(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    outside = tmp_path / "outside.npy"
    manifest = {
        "assignment_mode": "cluster_level",
        "default_assignment_variant": "normalized_top20_count",
        "assignment_variants": {
            "normalized_top20_count": {
                "group_functional_atlas_labels": str(outside.resolve()),
                "group_functional_atlas_label_map": str(
                    (run_dir / "map.json").resolve()
                ),
            }
        },
    }

    with pytest.raises(ValueError, match="outside run directory"):
        resolve_functional_atlas_array_paths(manifest, run_dir)


def test_materialize_combo_requires_exact_declared_variant_label_path(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    asset_dir = run_dir / "template" / "workbench_assets"
    asset_dir.mkdir(parents=True)
    np.save(run_dir / "group_labels_final.npy", np.asarray([1, 1], dtype=np.int32))
    write_json(run_dir / "group_feature_summary.json", {"network_names": ["Default", "Visual"]})
    write_json(
        run_dir / "status.json",
        {
            "items": [
                {"cluster_id": 1, "status": "dominant", "top1_network": "Default"}
            ]
        },
    )
    manifest_path = run_dir / "group_atlas_manifest.json"
    write_json(
        manifest_path,
        {
            "status": "ok",
            "assignment_mode": "cluster_level",
            "default_assignment_variant": "fisherz",
            "template_cluster_label": str(
                (asset_dir / "cluster.label.gii").resolve()
            ),
            "assignment_variants": {
                "fisherz": {
                    "cluster_assignment_status": str(
                        (run_dir / "status.json").resolve()
                    ),
                    "group_functional_atlas_labels": str(
                        (run_dir / "labels.npy").resolve()
                    ),
                    "group_functional_atlas_label_map": str(
                        (run_dir / "map.json").resolve()
                    ),
                }
            },
        },
    )

    with pytest.raises(ValueError, match="template_functional_atlas_label"):
        materialize_combo(manifest_path, _colors())


def test_overview_wta_profile_loads_manifest_default_variant_not_unsuffixed_name(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    labels_path = run_dir / "group_functional_atlas_labels_normalized_top20_count.npy"
    map_path = run_dir / "group_functional_atlas_label_map_normalized_top20_count.json"
    run_dir.mkdir()
    np.save(labels_path, np.asarray([1, 2], dtype=np.int32))
    write_json(map_path, {"network_names": ["Default", "Visual"]})
    manifest = {
        "assignment_mode": "cluster_level",
        "default_assignment_variant": "normalized_top20_count",
        "assignment_variants": {
            "normalized_top20_count": {
                "group_functional_atlas_labels": str(labels_path.resolve()),
                "group_functional_atlas_label_map": str(map_path.resolve()),
            }
        },
    }

    loaded = overview_mod.load_default_functional_labels(run_dir, manifest)

    np.testing.assert_array_equal(loaded, np.asarray([1, 2], dtype=np.int32))
