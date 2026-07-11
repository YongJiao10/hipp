from __future__ import annotations

import csv
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
MODULE_PATH = (
    REPO_ROOT
    / "scripts"
    / "individual_workflow"
    / "03_build_crossfit_robustness_overview.py"
)


def load_module():
    spec = importlib.util.spec_from_file_location(
        "crossfit_robustness_overview",
        MODULE_PATH,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_pairwise_metrics_and_vertex_agreement_detect_one_discordant_fold():
    overview = load_module()
    labels = np.asarray(
        [[1, 1, 2, 3], [1, 1, 2, 3], [1, 2, 2, 3]],
        dtype=np.int32,
    )
    support = np.ones_like(labels, dtype=bool)

    metrics = overview.pairwise_label_metrics(labels, support, k=3)

    assert np.allclose(np.diag(metrics["macro_dice"]), 1.0)
    assert metrics["macro_dice"][0, 1] == pytest.approx(1.0)
    assert metrics["ari"][0, 1] == pytest.approx(1.0)
    assert metrics["adjusted_rand_index"][0, 1] == pytest.approx(1.0)
    agreement = overview.vertex_agreement(labels)
    assert agreement.tolist() == pytest.approx([1.0, 2.0 / 3.0, 1.0, 1.0])


def test_pairwise_label_metrics_reject_empty_pairwise_support():
    overview = load_module()
    labels = np.asarray([[1, 2], [1, 2]], dtype=np.int32)
    support = np.asarray([[1, 0], [0, 1]], dtype=bool)

    with pytest.raises(ValueError, match="empty comparison support"):
        overview.pairwise_label_metrics(labels, support, k=3)


def test_prototype_correlations_use_only_jointly_finite_features():
    overview = load_module()
    profiles = np.asarray(
        [
            [[1.0, 2.0, np.nan, 4.0], [4.0, 2.0, 1.0, 0.0]],
            [[2.0, 4.0, 99.0, 8.0], [0.0, 1.0, 2.0, 4.0]],
        ],
        dtype=np.float64,
    )

    correlations = overview.pairwise_prototype_correlations(profiles)

    assert correlations.shape == (2, 2, 2)
    assert correlations[0, 1, 0] == pytest.approx(1.0)
    assert correlations[0, 1, 1] < -0.9
    assert np.allclose(correlations[0, 1], correlations[1, 0])


def test_renderer_writes_quantitative_outputs_for_five_synthetic_folds(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    overview = load_module()
    crossfit_root = tmp_path / "crossfit"
    template_surfaces_dir = tmp_path / "template_surfaces"
    template_surfaces_dir.mkdir(parents=True)
    surface_path = template_surfaces_dir / (
        "group-average_hemi-L_space-corobl_den-512_"
        "label-hipp_midthickness.surf.gii"
    )
    surface_path.write_bytes(b"synthetic surface")
    base_labels = np.asarray(
        [1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3],
        dtype=np.int32,
    )
    base_profiles = np.asarray(
        [
            [1.0, 2.0, 3.0, 4.0, 5.0],
            [5.0, 3.0, 1.0, 2.0, 4.0],
            [2.0, 5.0, 4.0, 1.0, 3.0],
        ],
        dtype=np.float32,
    )
    for fold_index in range(5):
        fold_dir = (
            crossfit_root
            / "templates"
            / f"fold-{fold_index + 1:02d}"
            / "2mm"
            / "hemi_L"
        )
        fold_dir.mkdir(parents=True)
        labels = base_labels.copy()
        if fold_index:
            labels[fold_index - 1] = 2
        np.save(fold_dir / "group_labels_k3.npy", labels)
        np.save(fold_dir / "group_support_mask.npy", np.ones(labels.size, dtype=np.uint8))
        np.save(
            fold_dir / "template_profiles.npy",
            base_profiles * float(fold_index + 1) + float(fold_index),
        )
        (fold_dir / "template_manifest.json").write_text(
            json.dumps(
                {
                    "status": "ok",
                    "fold_index": fold_index,
                    "fold_id": f"fold-{fold_index + 1:02d}",
                    "smoothing": "2mm",
                    "hemi": "L",
                    "density_evidence": {
                        "consumed_asset": str(surface_path.resolve()),
                        "evidence_source": "consumed_template_surface_path",
                    },
                }
            ),
            encoding="utf-8",
        )

    monkeypatch.setattr(overview, "long_axis_order", lambda path: np.arange(12))
    output_dir = tmp_path / "overview"
    outputs = overview.render_robustness_overview(
        crossfit_root=crossfit_root,
        template_surfaces_dir=template_surfaces_dir,
        output_dir=output_dir,
        smoothings=["2mm"],
        hemis=["L"],
        n_folds=5,
    )

    assert outputs["png"].stat().st_size > 0
    with outputs["summary_tsv"].open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream, delimiter="\t"))
    assert len(rows) == 1
    assert rows[0]["smoothing"] == "2mm"
    assert rows[0]["hemi"] == "L"
    metric_fields = [
        "mean_pairwise_macro_dice",
        "min_pairwise_macro_dice",
        "mean_pairwise_ari",
        "min_pairwise_ari",
        "mean_pairwise_prototype_correlation",
        "min_pairwise_prototype_correlation",
        "mean_vertex_agreement",
        "min_vertex_agreement",
    ]
    assert all(np.isfinite(float(rows[0][field])) for field in metric_fields)
    manifest = json.loads(outputs["manifest"].read_text(encoding="utf-8"))
    assert len(manifest["combinations"]) == 1
    assert len(manifest["combinations"][0]["source_folds"]) == 5
    assert all(
        Path(source["labels"]).is_absolute()
        for source in manifest["combinations"][0]["source_folds"]
    )
