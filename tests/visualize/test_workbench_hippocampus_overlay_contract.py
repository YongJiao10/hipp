from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
RENDERER_PATH = (
    REPO_ROOT
    / "scripts/visualize/multidataset_workbench_hippocampus_overlay"
    / "render_multidataset_workbench_hippocampus_overlay.py"
)


@pytest.fixture
def renderer():
    spec = importlib.util.spec_from_file_location(
        "formal_workbench_hippocampus_overlay_renderer", RENDERER_PATH
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _assignment_fixture(tmp_path: Path) -> tuple[Path, dict, dict[str, Path]]:
    run_dir = tmp_path / "hemi_L"
    run_dir.mkdir()
    paths = {
        "cluster_assignment_status": run_dir / "status_fisherz.json",
        "group_functional_atlas_labels": run_dir / "labels_fisherz.npy",
        "group_functional_atlas_label_map": run_dir / "label_map_fisherz.json",
        "template_functional_atlas_label": run_dir / "canonical_fisherz.label.gii",
        "deprecated_label": run_dir / "deprecated_alias.label.gii",
        "group_labels_final": run_dir / "group_labels_final.npy",
    }
    for path in paths.values():
        path.write_bytes(b"fixture")
    manifest = {
        "assignment_variants": {
            "fisherz": {
                key: str(paths[key])
                for key in (
                    "cluster_assignment_status",
                    "group_functional_atlas_labels",
                    "group_functional_atlas_label_map",
                    "template_functional_atlas_label",
                )
            }
        }
    }
    return run_dir, manifest, paths


def test_renderer_uses_the_declared_fisherz_label_not_deprecated_alias(
    renderer, tmp_path
):
    run_dir, manifest, paths = _assignment_fixture(tmp_path)
    manifest["assignment_variants"]["fisherz"][
        "template_functional_atlas_label_variant"
    ] = str(paths["deprecated_label"])

    resolved = renderer._resolve_fisherz_assignment(run_dir, manifest)

    assert resolved["label_path"] == paths["template_functional_atlas_label"]
    assert resolved["status_path"] == paths["cluster_assignment_status"]
    assert paths["group_functional_atlas_labels"] in resolved["paths"]
    assert paths["group_functional_atlas_label_map"] in resolved["paths"]
    assert paths["deprecated_label"] not in resolved["paths"]


def test_renderer_requires_canonical_group_labels_final_without_legacy_fallback(
    renderer, tmp_path
):
    run_dir, manifest, paths = _assignment_fixture(tmp_path)
    paths["group_labels_final"].unlink()
    (run_dir / "group_labels.npy").write_bytes(b"legacy")

    with pytest.raises(FileNotFoundError, match="group_labels_final.npy"):
        renderer._resolve_fisherz_assignment(run_dir, manifest)


def test_renderer_rejects_assignment_artifacts_outside_the_run_directory(
    renderer, tmp_path
):
    run_dir, manifest, paths = _assignment_fixture(tmp_path)
    outside = tmp_path / "outside.label.gii"
    outside.write_bytes(b"outside")
    manifest["assignment_variants"]["fisherz"][
        "template_functional_atlas_label"
    ] = str(outside)

    with pytest.raises(ValueError, match="outside"):
        renderer._resolve_fisherz_assignment(run_dir, manifest)
