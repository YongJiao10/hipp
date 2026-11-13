"""The negative stacked renderer must use validated assignment labels."""

import json
from importlib import import_module
from pathlib import Path

import numpy as np
from PIL import Image


def test_high_resolution_surface_keeps_all_four_views():
    module = import_module(
        "scripts.visualize.multidataset_cluster_confidence_negative_stacked."
        "plot_multidataset_cluster_negative_stacked"
    )
    pixels = np.full((2048, 3200, 3), 255, dtype=np.uint8)
    colors = [(180, 20, 20), (20, 180, 20), (20, 20, 180), (120, 20, 120)]
    for (y, x), color in zip(((100, 100), (100, 1700), (1200, 100), (1200, 1700)), colors):
        pixels[y:y + 100, x:x + 100] = color
    result = np.asarray(module.build_tight_4view_composite(Image.fromarray(pixels)))
    for color in colors:
        assert np.any(np.all(result[:, :, :3] == color, axis=2))


def test_workbench_overlay_uses_validated_fisherz_labels(tmp_path, monkeypatch):
    module = import_module(
        "scripts.visualize.multidataset_cluster_confidence_negative_stacked."
        "plot_multidataset_cluster_negative_stacked"
    )
    atlas_dir = tmp_path / "cohort"
    grad_dir = (
        atlas_dir
        / "gradient-spectral/kong2022/2mm/cov_0p80/g1g2g3"
    )
    captured: dict[str, Path] = {}
    native_png = tmp_path / "native.png"
    Image.fromarray(
        np.full((1024, 1600, 3), 255, dtype=np.uint8), mode="RGB"
    ).save(native_png)

    for hemi in ("L", "R"):
        hemi_dir = grad_dir / f"hemi_{hemi}"
        hemi_dir.mkdir(parents=True)
        raw_label = tmp_path / f"raw_{hemi}.gii"
        spec_path = tmp_path / f"spec_{hemi}.json"
        (hemi_dir / "group_atlas_manifest.json").write_text(
            json.dumps(
                {
                    "template_left_or_right_surface": str(
                        tmp_path / f"surface_{hemi}.gii"
                    ),
                    "assignment_variants": {
                        "fisherz": {
                            "template_functional_atlas_label_variant": str(
                                raw_label
                            )
                        }
                    },
                    "template_subject": "sub-123",
                    "template_spec_path": str(spec_path),
                }
            ),
            encoding="utf-8",
        )

    def fake_validate(run_dir, _manifest, variant):
        assert variant == "fisherz"
        hemi = Path(run_dir).name[-1]
        return {"label_path": tmp_path / f"validated_{hemi}.gii"}

    def fake_render_hippocampal_grid(**kwargs):
        captured["left_labels"] = Path(kwargs["left_labels"])
        captured["right_labels"] = Path(kwargs["right_labels"])
        return {"surface_png": str(native_png)}

    monkeypatch.setattr(
        module, "validate_gradient_assignment", fake_validate, raising=False
    )
    monkeypatch.setattr(module, "render_hippocampal_grid", fake_render_hippocampal_grid)

    module.load_cohort_wb_overlay(
        atlas_dir,
        tmp_path / "render",
        tmp_path / "scene.scene",
    )

    assert captured == {
        "left_labels": tmp_path / "validated_L.gii",
        "right_labels": tmp_path / "validated_R.gii",
    }
