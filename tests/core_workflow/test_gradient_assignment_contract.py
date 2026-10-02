from __future__ import annotations

import importlib
import importlib.util
import json
import sys
from pathlib import Path

import nibabel as nib
import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[2]
CORE_DIR = ROOT / "scripts" / "core_workflow"
if str(CORE_DIR) not in sys.path:
    sys.path.insert(0, str(CORE_DIR))


def _load_contract():
    assert importlib.util.find_spec("lib.gradient_assignment_contract") is not None, (
        "formal gradient assignment contract module is missing"
    )
    return importlib.import_module("lib.gradient_assignment_contract")


@pytest.fixture
def normalized_artifact(tmp_path: Path) -> tuple[Path, dict[str, object]]:
    from lib.surface_utils import make_network_label_gifti, network_color_map

    names = ["Default", "Somatomotor", "VentralAttention"]
    clusters = np.asarray([1, 1, 2, 2, 3, 3], dtype=np.int32)
    labels = np.asarray([3, 3, 1, 1, 1, 1], dtype=np.int32)
    np.save(tmp_path / "group_labels_final.npy", clusters)
    np.save(tmp_path / "labels.npy", labels)
    nib.save(
        make_network_label_gifti(
            labels=labels,
            network_names=names,
            network_colors=network_color_map(),
        ),
        tmp_path / "labels.label.gii",
    )

    status = {
        "assignment_metric": "top20_count_normalized_by_network_roi_count",
        "items": [
            {
                "cluster_id": 1,
                "status": "dominant",
                "top1_network": "VentralAttention",
                "normalized_count_by_network": {
                    "Default": 0.02,
                    "Somatomotor": 0.07,
                    "VentralAttention": 0.12,
                },
            },
            {
                "cluster_id": 2,
                "status": "dominant",
                "top1_network": "Default",
                "normalized_count_by_network": {
                    "Default": 0.20,
                    "Somatomotor": 0.01,
                    "VentralAttention": 0.03,
                },
            },
            {
                "cluster_id": 3,
                "status": "dominant",
                "top1_network": "Default",
                "normalized_count_by_network": {
                    "Default": 0.30,
                    "Somatomotor": 0.02,
                    "VentralAttention": 0.01,
                },
            },
        ],
    }
    (tmp_path / "status.json").write_text(json.dumps(status), encoding="utf-8")
    mapping = {
        "network_names": names,
        "network_label_keys": dict(zip(names, [1, 2, 3])),
        "cluster_to_functional_label_key": {"1": 3, "2": 1, "3": 1},
    }
    (tmp_path / "map.json").write_text(json.dumps(mapping), encoding="utf-8")
    manifest = {
        "assignment_variants": {
            "normalized_top20_count": {
                "cluster_assignment_status": str(tmp_path / "status.json"),
                "group_functional_atlas_labels": str(tmp_path / "labels.npy"),
                "group_functional_atlas_label_map": str(tmp_path / "map.json"),
                "template_functional_atlas_label": str(tmp_path / "labels.label.gii"),
            }
        }
    }
    return tmp_path, manifest


def test_normalized_assignment_requires_all_declared_artifacts_to_agree(
    normalized_artifact: tuple[Path, dict[str, object]],
) -> None:
    run_dir, manifest = normalized_artifact
    result = _load_contract().validate_gradient_assignment(
        run_dir, manifest, "normalized_top20_count"
    )

    assert result["winners"] == {
        1: "VentralAttention",
        2: "Default",
        3: "Default",
    }
    assert result["compositions"][1]["VentralAttention"] == pytest.approx(
        12 / (2 + 7 + 12) * 100
    )
    assert result["label_path"] == run_dir / "labels.label.gii"
    assert str(run_dir / "labels.npy") in result["asset_fingerprints"]


def test_declared_assignment_artifact_outside_run_directory_is_rejected(
    normalized_artifact: tuple[Path, dict[str, object]],
) -> None:
    run_dir, manifest = normalized_artifact
    outside = run_dir.parent / "status.json"
    outside.write_text((run_dir / "status.json").read_text(encoding="utf-8"), encoding="utf-8")
    manifest["assignment_variants"]["normalized_top20_count"][
        "cluster_assignment_status"
    ] = str(outside)

    with pytest.raises(ValueError, match="outside"):
        _load_contract().validate_gradient_assignment(
            run_dir, manifest, "normalized_top20_count"
        )


def test_npy_and_gifti_assignment_labels_cannot_disagree(
    normalized_artifact: tuple[Path, dict[str, object]],
) -> None:
    run_dir, manifest = normalized_artifact
    labels = np.load(run_dir / "labels.npy")
    labels[0] = 1
    np.save(run_dir / "labels.npy", labels)

    with pytest.raises(ValueError, match="NPY/GIFTI"):
        _load_contract().validate_gradient_assignment(
            run_dir, manifest, "normalized_top20_count"
        )
