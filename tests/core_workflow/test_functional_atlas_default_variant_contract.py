#!/usr/bin/env python3
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np

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
    return {
        "Default": (231, 76, 60, 255),
        "Visual": (94, 45, 145, 255),
    }


def test_materialize_combo_writes_default_variant_without_unsuffixed_duplicate(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    asset_dir = run_dir / "template" / "workbench_assets"
    asset_dir.mkdir(parents=True)
    np.save(run_dir / "group_labels_final.npy", np.asarray([1, 1, 2, 2], dtype=np.int32))
    write_json(run_dir / "group_feature_summary.json", {"network_names": ["Default", "Visual"]})
    write_json(
        run_dir / "cluster_assignment_status_fisherz.json",
        {
            "items": [
                {"cluster_id": 1, "status": "dominant", "top1_network": "Default"},
                {"cluster_id": 2, "status": "dominant", "top1_network": "Visual"},
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
                asset_dir / "demo_label-group_cluster_gradient-spectral_kong2022_2mm_cov-0p80.label.gii"
            ),
            "assignment_variants": {
                "fisherz": {
                    "cluster_assignment_status": str((run_dir / "cluster_assignment_status_fisherz.json").resolve()),
                    "group_functional_atlas_labels": str(
                        (run_dir / "group_functional_atlas_labels_fisherz.npy").resolve()
                    ),
                    "group_functional_atlas_label_map": str(
                        (run_dir / "group_functional_atlas_label_map_fisherz.json").resolve()
                    ),
                }
            },
        },
    )

    assert materialize_combo(manifest_path, _colors()) is True

    refreshed = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert Path(refreshed["group_functional_atlas_labels"]).name == "group_functional_atlas_labels_fisherz.npy"
    assert Path(refreshed["group_functional_atlas_label_map"]).name == "group_functional_atlas_label_map_fisherz.json"
    assert not (run_dir / "group_functional_atlas_labels.npy").exists()
    assert not (run_dir / "group_functional_atlas_label_map.json").exists()
    np.testing.assert_array_equal(
        np.load(refreshed["group_functional_atlas_labels"]),
        np.asarray([1, 1, 2, 2], dtype=np.int32),
    )


def test_default_functional_array_resolution_requires_manifest_variant_for_cluster_level(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    labels_path = run_dir / "group_functional_atlas_labels_fisherz.npy"
    map_path = run_dir / "group_functional_atlas_label_map_fisherz.json"
    run_dir.mkdir()
    np.save(labels_path, np.asarray([1, 2], dtype=np.int32))
    write_json(map_path, {"network_names": ["Default", "Visual"]})
    manifest = {
        "assignment_mode": "cluster_level",
        "default_assignment_variant": "fisherz",
        "assignment_variants": {
            "fisherz": {
                "group_functional_atlas_labels": str(labels_path.resolve()),
                "group_functional_atlas_label_map": str(map_path.resolve()),
            }
        },
    }

    labels_got, map_got = resolve_functional_atlas_array_paths(manifest, run_dir)

    assert labels_got == labels_path.resolve()
    assert map_got == map_path.resolve()


def test_overview_wta_profile_loads_manifest_default_variant_not_unsuffixed_name(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    labels_path = run_dir / "group_functional_atlas_labels_fisherz.npy"
    map_path = run_dir / "group_functional_atlas_label_map_fisherz.json"
    run_dir.mkdir()
    np.save(labels_path, np.asarray([1, 2], dtype=np.int32))
    write_json(map_path, {"network_names": ["Default", "Visual"]})
    manifest = {
        "assignment_mode": "cluster_level",
        "default_assignment_variant": "fisherz",
        "assignment_variants": {
            "fisherz": {
                "group_functional_atlas_labels": str(labels_path.resolve()),
                "group_functional_atlas_label_map": str(map_path.resolve()),
            }
        },
    }

    loaded = overview_mod.load_default_functional_labels(run_dir, manifest)

    np.testing.assert_array_equal(loaded, np.asarray([1, 2], dtype=np.int32))




def test_migration_plan_archives_verified_unsuffixed_duplicates_and_rewrites_manifest(tmp_path: Path) -> None:
    migrate_mod = _load_numbered_module("migrate_default_variant_outputs.py", "core_migrate_default_variant_outputs_test")
    root = tmp_path / "outputs_core" / "work" / "group_atlas"
    run_dir = root / "gradient-spectral" / "kong2022" / "2mm" / "cov-0p80" / "hemi_L"
    asset_dir = run_dir / "template" / "workbench_assets"
    asset_dir.mkdir(parents=True)

    unsuffixed_labels = run_dir / "group_functional_atlas_labels.npy"
    suffixed_labels = run_dir / "group_functional_atlas_labels_fisherz.npy"
    unsuffixed_map = run_dir / "group_functional_atlas_label_map.json"
    suffixed_map = run_dir / "group_functional_atlas_label_map_fisherz.json"
    unsuffixed_status = run_dir / "cluster_assignment_status.json"
    suffixed_status = run_dir / "cluster_assignment_status_fisherz.json"
    unsuffixed_label = asset_dir / "demo_label-group_functional_atlas_gradient-spectral_kong2022_2mm_cov-0p80.label.gii"
    suffixed_label = asset_dir / "demo_label-group_functional_atlas_gradient-spectral_kong2022_2mm_cov-0p80_fisherz.label.gii"

    labels = np.asarray([1, 2, 1, 2], dtype=np.int32)
    np.save(unsuffixed_labels, labels)
    np.save(suffixed_labels, labels)
    map_payload = {"network_names": ["Default", "Visual"], "network_label_keys": {"Default": 1, "Visual": 2}}
    status_payload = {"items": [{"cluster_id": 1, "status": "dominant", "top1_network": "Default"}]}
    write_json(unsuffixed_map, map_payload)
    write_json(suffixed_map, map_payload)
    write_json(unsuffixed_status, status_payload)
    write_json(suffixed_status, status_payload)
    label_img = overview_mod.make_named_label_gifti(
        labels,
        [(0, "Null", (158, 158, 158, 255)), (1, "Default", (231, 76, 60, 255)), (2, "Visual", (94, 45, 145, 255))],
    )
    import nibabel as nib

    nib.save(label_img, str(unsuffixed_label))
    nib.save(label_img, str(suffixed_label))
    manifest_path = run_dir / "group_atlas_manifest.json"
    write_json(
        manifest_path,
        {
            "status": "ok",
            "assignment_variants": {
                "fisherz": {
                    "cluster_assignment_status": str(suffixed_status.resolve()),
                    "group_functional_atlas_labels": str(suffixed_labels.resolve()),
                    "group_functional_atlas_label_map": str(suffixed_map.resolve()),
                    "template_functional_atlas_label": str(unsuffixed_label.resolve()),
                    "template_functional_atlas_label_variant": str(suffixed_label.resolve()),
                }
            },
            "group_functional_atlas_labels": str(unsuffixed_labels.resolve()),
            "group_functional_atlas_label_map": str(unsuffixed_map.resolve()),
            "cluster_assignment_status": str(unsuffixed_status.resolve()),
            "template_functional_atlas_label": str(unsuffixed_label.resolve()),
        },
    )

    plans = migrate_mod.build_plan([root], tmp_path)

    assert len(plans) == 1
    plan = plans[0]
    assert plan.updates["default_assignment_variant"] == "fisherz"
    assert plan.updates["assignment_mode"] == "cluster_level"
    assert plan.updates["group_functional_atlas_labels"] == str(suffixed_labels.resolve())
    assert plan.updates["template_functional_atlas_label"] == str(suffixed_label.resolve())
    assert set(plan.archive_targets) == {unsuffixed_labels.resolve(), unsuffixed_map.resolve(), unsuffixed_status.resolve(), unsuffixed_label.resolve()}
