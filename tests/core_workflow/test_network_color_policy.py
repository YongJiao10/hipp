#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
CORE_DIR = REPO_ROOT / "scripts" / "core_workflow"
if str(CORE_DIR) not in sys.path:
    sys.path.insert(0, str(CORE_DIR))

from lib.network_colors import (  # noqa: E402
    GLOBAL_NETWORK_STYLE_JSON,
    load_network_colors,
    require_network_colors,
    rgba_to_hex,
)


CANONICAL_8 = [
    "Default",
    "Visual",
    "Somatomotor",
    "DorsalAttention",
    "VentralAttention",
    "Control",
    "Auditory",
    "Language",
]

FAMILY_17 = [
    "DefaultA",
    "DefaultB",
    "DefaultC",
    "ContA",
    "ContB",
    "ContC",
    "SalVenAttnA",
    "SalVenAttnB",
    "DorsAttnA",
    "DorsAttnB",
    "SomMotA",
    "SomMotB",
    "VisualA",
    "VisualB",
    "VisualC",
    "Aud",
    "Language",
]

EXPECTED_17_RGBA = {
    "DefaultA": (231, 76, 60, 255),
    "DefaultB": (190, 49, 93, 255),
    "DefaultC": (124, 35, 53, 255),
    "ContA": (247, 172, 74, 255),
    "ContB": (232, 196, 64, 255),
    "ContC": (178, 104, 34, 255),
    "SalVenAttnA": (220, 92, 255, 255),
    "SalVenAttnB": (142, 55, 201, 255),
    "DorsAttnA": (44, 126, 17, 255),
    "DorsAttnB": (133, 188, 66, 255),
    "SomMotA": (47, 119, 190, 255),
    "SomMotB": (42, 204, 164, 255),
    "VisualA": (94, 45, 145, 255),
    "VisualB": (172, 75, 183, 255),
    "VisualC": (62, 52, 142, 255),
    "Aud": (24, 183, 201, 255),
    "Language": (138, 71, 24, 255),
}


def test_global_network_style_has_explicit_8net_and_17net_colors() -> None:
    colors = load_network_colors()
    required = CANONICAL_8 + FAMILY_17
    assert require_network_colors(required) == {name: colors[name] for name in required}
    assert {name: colors[name] for name in FAMILY_17} == EXPECTED_17_RGBA


def test_color_loader_rejects_missing_networks_without_fallback() -> None:
    try:
        require_network_colors(["DefaultA", "__missing_network__"])
    except KeyError as exc:
        assert "__missing_network__" in str(exc)
    else:
        raise AssertionError("Expected missing network color to raise KeyError")


def test_rgba_to_hex_uses_global_color_values() -> None:
    colors = load_network_colors()
    assert rgba_to_hex(colors["DefaultA"]) == "#e74c3c"
    assert rgba_to_hex(colors["ContB"]) == "#e8c440"
    assert GLOBAL_NETWORK_STYLE_JSON == REPO_ROOT / "config" / "hipp_network_style.json"


def test_cross_atlas_merge_no_longer_stores_network_colors() -> None:
    merge_path = REPO_ROOT / "config" / "cross_atlas_network_merge.json"
    payload = json.loads(merge_path.read_text(encoding="utf-8"))
    assert "shared_colors_rgba" not in payload


def test_core_workflow_does_not_read_shared_color_overrides() -> None:
    offenders = []
    for path in sorted((REPO_ROOT / "scripts" / "core_workflow").rglob("*.py")):
        text = path.read_text(encoding="utf-8")
        if "shared_colors_rgba" in text:
            offenders.append(path.relative_to(REPO_ROOT).as_posix())
    assert offenders == []


def test_materialize_combo_refreshes_assignment_variant_label_tables(tmp_path: Path) -> None:
    import nibabel as nib
    import numpy as np
    from lib.gifti_labels import materialize_combo
    from lib.workflow_utils import write_json

    run_dir = tmp_path / "run"
    asset_dir = run_dir / "template" / "workbench_assets"
    asset_dir.mkdir(parents=True)
    np.save(run_dir / "group_labels_final.npy", np.asarray([1, 1, 2, 2], dtype=np.int32))
    write_json(run_dir / "group_feature_summary.json", {"network_names": ["DefaultA", "ContB"]})
    write_json(
        run_dir / "cluster_assignment_status.json",
        {
            "items": [
                {"cluster_id": 1, "status": "dominant", "top1_network": "DefaultA"},
                {"cluster_id": 2, "status": "dominant", "top1_network": "ContB"},
            ]
        },
    )
    write_json(
        run_dir / "cluster_assignment_status_count.json",
        {
            "items": [
                {"cluster_id": 1, "status": "dominant", "top1_network": "ContB"},
                {"cluster_id": 2, "status": "dominant", "top1_network": "DefaultA"},
            ]
        },
    )
    manifest_path = run_dir / "group_atlas_manifest.json"
    cluster_label = asset_dir / "demo_label-group_cluster_method_kong2022_2mm_cov-0p80.label.gii"
    write_json(
        manifest_path,
        {
            "status": "ok",
            "template_cluster_label": str(cluster_label),
            "assignment_variants": {
                "count": {
                    "cluster_assignment_status": str((run_dir / "cluster_assignment_status_count.json").resolve()),
                    "group_functional_atlas_labels": str((run_dir / "group_functional_atlas_labels_count.npy").resolve()),
                    "group_functional_atlas_label_map": str((run_dir / "group_functional_atlas_label_map_count.json").resolve()),
                }
            },
        },
    )

    assert materialize_combo(manifest_path, load_network_colors()) is True

    refreshed = json.loads(manifest_path.read_text(encoding="utf-8"))
    variant = refreshed["assignment_variants"]["count"]
    count_label = Path(variant["template_functional_atlas_label"])
    assert count_label.exists()
    labels = {lab.label: tuple(round(x * 255) for x in (lab.red, lab.green, lab.blue, lab.alpha)) for lab in nib.load(str(count_label)).labeltable.labels}
    assert labels["DefaultA"] == EXPECTED_17_RGBA["DefaultA"]
    assert labels["ContB"] == EXPECTED_17_RGBA["ContB"]


def test_materialize_combo_refreshes_existing_vertex_level_label_table(tmp_path: Path) -> None:
    import nibabel as nib
    import numpy as np
    from lib.gifti_labels import make_named_label_gifti, materialize_combo
    from lib.workflow_utils import write_json

    run_dir = tmp_path / "wta"
    asset_dir = run_dir / "template" / "workbench_assets"
    asset_dir.mkdir(parents=True)
    labels_path = run_dir / "group_functional_atlas_labels.npy"
    map_path = run_dir / "group_functional_atlas_label_map.json"
    label_path = asset_dir / "demo_label-group_functional_atlas_wta-top_kong2022_2mm_cov-0p80.label.gii"
    np.save(labels_path, np.asarray([1, 2, 1, 2], dtype=np.int32))
    write_json(
        map_path,
        {
            "network_names": ["DefaultA", "ContB"],
            "network_label_keys": {"DefaultA": 1, "ContB": 2},
            "unassigned_label_key": 3,
        },
    )
    old_label = make_named_label_gifti(
        np.asarray([1, 2, 1, 2], dtype=np.int32),
        [(0, "Null", (158, 158, 158, 255)), (1, "DefaultA", (1, 2, 3, 255)), (2, "ContB", (4, 5, 6, 255))],
    )
    nib.save(old_label, str(label_path))
    manifest_path = run_dir / "group_atlas_manifest.json"
    write_json(
        manifest_path,
        {
            "status": "ok",
            "group_functional_atlas_labels": str(labels_path.resolve()),
            "group_functional_atlas_label_map": str(map_path.resolve()),
            "template_functional_atlas_label": str(label_path.resolve()),
        },
    )

    assert materialize_combo(manifest_path, load_network_colors()) is True

    table = {lab.label: tuple(round(x * 255) for x in (lab.red, lab.green, lab.blue, lab.alpha)) for lab in nib.load(str(label_path)).labeltable.labels}
    assert table["DefaultA"] == EXPECTED_17_RGBA["DefaultA"]
    assert table["ContB"] == EXPECTED_17_RGBA["ContB"]
