from __future__ import annotations

import ast
import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from scipy import sparse


REPO_ROOT = Path(__file__).resolve().parents[2]
CORE_WORKFLOW_DIR = REPO_ROOT / "scripts" / "core_workflow"
MODULE_PATH = (
    REPO_ROOT
    / "scripts"
    / "individual_workflow"
    / "routea_crossfit_template_matching.py"
)
DOC_PATH = (
    REPO_ROOT
    / "docs"
    / "hipp_parcellation_network"
    / "crossfit_template_matching.md"
)
if str(CORE_WORKFLOW_DIR) not in sys.path:
    sys.path.insert(0, str(CORE_WORKFLOW_DIR))

from lib import output_layout  # noqa: E402


def load_module():
    spec = importlib.util.spec_from_file_location(
        "routea_crossfit_template_matching",
        MODULE_PATH,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_grouped_folds_are_deterministic_balanced_and_keep_families_together():
    workflow = load_module()
    subjects = ["01", "02", "03", "04", "05", "06"]
    families = {"01": "a", "02": "a", "03": "b", "04": "c", "05": "d", "06": "e"}
    first = workflow.assign_grouped_folds(subjects, families, n_folds=3, seed=17)
    second = workflow.assign_grouped_folds(subjects, families, n_folds=3, seed=17)
    assert first == second
    assert first["01"] == first["02"]
    counts = np.bincount(list(first.values()), minlength=3)
    assert int(counts.max() - counts.min()) <= 1


def test_grouped_folds_normalize_subject_prefixes_and_reject_missing_families():
    workflow = load_module()

    got = workflow.assign_grouped_folds(
        ["sub-01", "02"],
        {"01": "a", "sub-02": "b"},
        n_folds=2,
        seed=17,
    )

    assert set(got) == {"01", "02"}
    with pytest.raises(ValueError, match="family"):
        workflow.assign_grouped_folds(["01", "02"], {"01": "a"}, n_folds=2, seed=17)


def test_validate_fold_assignment_rejects_families_split_across_folds():
    workflow = load_module()
    subjects = ["sub-01", "02", "03"]
    families = {"01": "a", "sub-02": "a", "03": "b"}

    workflow.validate_fold_assignment(subjects, families, {"01": 0, "02": 0, "03": 1}, n_folds=2)

    with pytest.raises(ValueError, match="Family"):
        workflow.validate_fold_assignment(
            subjects,
            families,
            {"01": 0, "02": 1, "03": 0},
            n_folds=2,
        )


def test_crossfit_output_roots_are_isolated_from_existing_individual_atlas():
    assert output_layout.resolve_crossfit_template_matching_root().name == "crossfit_template_matching"
    assert output_layout.resolve_crossfit_template_matching_overview_root().name == "crossfit_template_matching"


def test_cluster_prototypes_average_only_supported_ordered_cluster_vertices():
    workflow = load_module()
    fc = np.asarray(
        [[1.0, 0.0], [3.0, 2.0], [9.0, 9.0], [0.0, 4.0]],
        dtype=np.float32,
    )
    labels = np.asarray([1, 1, 0, 2], dtype=np.int32)
    support = np.asarray([1, 1, 0, 1], dtype=bool)

    got = workflow.build_cluster_prototypes(fc, labels, support, k=2)

    assert np.allclose(got, [[2.0, 1.0], [0.0, 4.0]])


def test_cluster_prototypes_reject_missing_cluster():
    workflow = load_module()

    with pytest.raises(ValueError, match="cluster 2"):
        workflow.build_cluster_prototypes(
            np.asarray([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32),
            np.asarray([1, 1], dtype=np.int32),
            np.asarray([1, 1], dtype=bool),
            k=2,
        )


def test_vertex_matching_returns_labels_scores_and_top1_top2_margin():
    workflow = load_module()
    fc = np.asarray(
        [[1.0, 0.0, -1.0], [-1.0, 0.0, 1.0], [5.0, 5.0, 5.0]],
        dtype=np.float32,
    )
    valid = np.asarray([1, 1, 1], dtype=bool)
    prototypes = np.asarray(
        [[1.0, 0.0, -1.0], [-1.0, 0.0, 1.0]],
        dtype=np.float32,
    )

    labels, scores, margin = workflow.match_vertices_to_prototypes(
        fc,
        valid,
        prototypes,
    )

    assert labels.tolist() == [1, 2, 0]
    assert scores.shape == (3, 2)
    assert np.allclose(margin[:2], [2.0, 2.0])
    assert np.isnan(margin[2])


def test_vertex_matching_uses_jointly_finite_features_and_keeps_invalid_vertices_null():
    workflow = load_module()
    fc = np.asarray(
        [[1.0, np.nan, -1.0, 0.0], [1.0, 0.0, -1.0, 0.0]],
        dtype=np.float32,
    )
    prototypes = np.asarray(
        [[1.0, 9.0, -1.0, 0.0], [-1.0, 9.0, 1.0, 0.0]],
        dtype=np.float32,
    )

    labels, scores, margin = workflow.match_vertices_to_prototypes(
        fc,
        np.asarray([1, 0], dtype=bool),
        prototypes,
    )

    assert labels.tolist() == [1, 0]
    assert scores[0, 0] == pytest.approx(1.0)
    assert scores[0, 1] == pytest.approx(-1.0)
    assert np.all(np.isnan(scores[1]))
    assert margin[0] == pytest.approx(2.0)
    assert np.isnan(margin[1])


def test_hcp_family_map_is_required_without_explicit_waiver(tmp_path: Path):
    workflow = load_module()
    args = workflow.build_arg_parser().parse_args(
        ["--subjects-manifest", str(tmp_path / "subjects.txt")]
    )

    with pytest.raises(ValueError, match="family"):
        workflow.resolve_family_map(args, ["01", "02"])


def test_subject_only_family_map_requires_and_records_explicit_waiver(tmp_path: Path):
    workflow = load_module()
    args = workflow.build_arg_parser().parse_args(
        [
            "--subjects-manifest",
            str(tmp_path / "subjects.txt"),
            "--allow-subject-only-folds",
        ]
    )

    family_map, metadata = workflow.resolve_family_map(args, ["sub-01", "02"])

    assert family_map == {"01": "01", "02": "02"}
    assert metadata == {
        "family_aware": False,
        "subject_only_waiver": True,
        "family_map_path": None,
    }


def test_family_map_tsv_requires_subject_and_family_id_columns(tmp_path: Path):
    workflow = load_module()
    family_path = tmp_path / "families.tsv"
    family_path.write_text("subject\tfamily_id\nsub-01\tfam-a\n02\tfam-b\n", encoding="utf-8")
    args = workflow.build_arg_parser().parse_args(
        [
            "--subjects-manifest",
            str(tmp_path / "subjects.txt"),
            "--family-map",
            str(family_path),
        ]
    )

    family_map, metadata = workflow.resolve_family_map(args, ["01", "02"])

    assert family_map == {"01": "fam-a", "02": "fam-b"}
    assert metadata["family_aware"] is True
    assert metadata["subject_only_waiver"] is False
    assert metadata["family_map_path"] == str(family_path.resolve())


def test_family_map_rejects_duplicate_normalized_subject_rows(tmp_path: Path):
    workflow = load_module()
    family_path = tmp_path / "families.tsv"
    family_path.write_text(
        "subject\tfamily_id\nsub-01\tfam-a\n01\tfam-a\n",
        encoding="utf-8",
    )
    args = workflow.build_arg_parser().parse_args(
        ["--family-map", str(family_path)]
    )

    with pytest.raises(ValueError, match="Duplicate family row"):
        workflow.resolve_family_map(args, ["01"])


def test_fold_manifest_has_disjoint_nonempty_training_and_heldout_subjects():
    workflow = load_module()

    payload = workflow.fold_manifest_payload(
        {"01": 0, "02": 1, "03": 0, "04": 1},
        n_folds=2,
    )

    assert len(payload["folds"]) == 2
    for row in payload["folds"]:
        assert row["training_subjects"]
        assert row["heldout_subjects"]
        assert set(row["training_subjects"]).isdisjoint(row["heldout_subjects"])
        assert row["train_test_disjoint"] is True


def test_fold_manifest_rejects_empty_fold():
    workflow = load_module()

    with pytest.raises(ValueError, match="empty"):
        workflow.fold_manifest_payload({"01": 0, "02": 0}, n_folds=2)


def test_cluster_prototypes_treat_infinite_values_as_missing():
    workflow = load_module()
    prototypes = workflow.build_cluster_prototypes(
        np.asarray([[1.0, np.inf], [3.0, 4.0], [0.0, 2.0]], dtype=np.float32),
        np.asarray([1, 1, 2], dtype=np.int32),
        np.asarray([1, 1, 1], dtype=bool),
        k=2,
    )

    assert np.allclose(prototypes, [[2.0, 4.0], [0.0, 2.0]])


def _write_fold_subject_inputs(
    *,
    shared_root: Path,
    surface_root: Path,
    subject: str,
    offset: float,
) -> None:
    fc_dir = shared_root / f"sub-{subject}" / "fc" / "kong2022" / "2mm" / "hemi_L"
    fc_dir.mkdir(parents=True)
    base = np.asarray(
        [
            [0.10, 0.20, 0.30, 0.40],
            [0.15, 0.25, 0.35, 0.45],
            [0.40, 0.30, 0.20, 0.10],
            [0.45, 0.35, 0.25, 0.15],
            [-0.10, -0.20, -0.30, -0.40],
            [-0.15, -0.25, -0.35, -0.45],
        ],
        dtype=np.float32,
    )
    np.save(fc_dir / "hipp_vertex_to_network_fc.npy", base + offset)
    mask_dir = surface_root / f"sub-{subject}" / "surface" / "tsnr_gate"
    mask_dir.mkdir(parents=True)
    np.save(mask_dir / f"sub-{subject}_hemi-L_valid_mask.npy", np.ones(6, dtype=bool))


def test_fit_fold_template_uses_only_training_subjects_and_records_density(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    workflow = load_module()
    shared_root = tmp_path / "routea"
    surface_root = tmp_path / "surface"
    for subject, offset in [("01", 0.00), ("02", 0.01)]:
        _write_fold_subject_inputs(
            shared_root=shared_root,
            surface_root=surface_root,
            subject=subject,
            offset=offset,
        )
    template_dir = tmp_path / "group_average_surfaces"
    template_dir.mkdir()
    template_surface = (
        template_dir
        / "group-average_hemi-L_space-corobl_den-512_label-hipp_midthickness.surf.gii"
    )
    template_surface.write_bytes(b"surface")

    gradients = np.asarray(
        [
            [3.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
            [0.0, 2.0, 0.0],
            [0.0, 3.0, 0.0],
            [0.0, 0.0, 2.0],
            [0.0, 0.0, 3.0],
        ],
        dtype=np.float32,
    )
    monkeypatch.setattr(
        workflow,
        "fit_gradientmaps_pearson",
        lambda matrix, n_components, random_state: (
            gradients,
            np.asarray([3.0, 2.0, 1.0]),
            np.asarray([0.5, 1.0 / 3.0, 1.0 / 6.0]),
        ),
    )
    fake_core = SimpleNamespace(
        load_surface=lambda path: (
            np.zeros((6, 3), dtype=np.float32),
            np.asarray([[0, 1, 2], [3, 4, 5]], dtype=np.int32),
        ),
        build_surface_adjacency=lambda faces, n_vertices: sparse.eye(
            n_vertices,
            format="csr",
            dtype=np.float32,
        ),
        cluster_from_precomputed_group_features=lambda **kwargs: (
            np.asarray([1, 1, 2, 2, 3, 3], dtype=np.int32),
            np.asarray([1, 1, 2, 2, 3, 3], dtype=np.int32),
        ),
        reorder_labels_by_long_axis=lambda labels, order: labels,
    )
    monkeypatch.setattr(workflow, "load_group_atlas_module", lambda: fake_core)
    monkeypatch.setattr(workflow, "long_axis_order", lambda path: np.arange(6))
    config = workflow.CrossfitConfig(
        shared_root=shared_root,
        surface_root=surface_root,
        template_surfaces_dir=template_dir,
        hippunfold_batch_root=tmp_path / "hippunfold",
        out_root=tmp_path / "crossfit",
        fold_by_subject={"01": 0, "02": 0, "03": 1},
        coverage_threshold=0.80,
        n_components=3,
        gradient_random_state=0,
        cluster_seed=0,
    )

    result = workflow.fit_fold_template(config, 1, "2mm", "L")

    assert result.training_subjects == ("01", "02")
    assert result.heldout_subjects == ("03",)
    assert "03" not in result.fc_paths
    assert result.prototypes.shape == (3, 4)
    assert set(np.unique(result.labels_full)) == {1, 2, 3}
    assert result.density_evidence["consumed_asset"].endswith(
        "den-512_label-hipp_midthickness.surf.gii"
    )
    assert (result.template_dir / "template_profiles.npy").exists()
    manifest = json.loads((result.template_dir / "template_manifest.json").read_text())
    assert manifest["training_subjects"] == ["sub-01", "sub-02"]
    assert manifest["heldout_subjects"] == ["sub-03"]
    assert manifest["train_test_disjoint"] is True


def test_write_heldout_subject_outputs_saves_labels_scores_margin_and_provenance(
    tmp_path: Path,
):
    workflow = load_module()
    shared_root = tmp_path / "routea"
    surface_root = tmp_path / "surface"
    _write_fold_subject_inputs(
        shared_root=shared_root,
        surface_root=surface_root,
        subject="03",
        offset=0.0,
    )
    subject_surface = (
        tmp_path
        / "hippunfold"
        / "sub-03"
        / "hippunfold"
        / "sub-03"
        / "surf"
        / "sub-03_hemi-L_space-corobl_den-512_label-hipp_midthickness.surf.gii"
    )
    subject_surface.parent.mkdir(parents=True)
    subject_surface.write_bytes(b"surface")
    template_surface = (
        tmp_path
        / "group-average_hemi-L_space-corobl_den-512_label-hipp_midthickness.surf.gii"
    )
    template_surface.write_bytes(b"surface")
    config = workflow.CrossfitConfig(
        shared_root=shared_root,
        surface_root=surface_root,
        template_surfaces_dir=tmp_path,
        hippunfold_batch_root=tmp_path / "hippunfold",
        out_root=tmp_path / "crossfit",
        fold_by_subject={"01": 0, "02": 0, "03": 1},
    )
    template = workflow.FoldTemplateResult(
        fold_index=1,
        smoothing="2mm",
        hemi="L",
        training_subjects=("01", "02"),
        heldout_subjects=("03",),
        surface_path=template_surface,
        support_mask=np.ones(6, dtype=bool),
        coverage_count=np.full(6, 2, dtype=np.int32),
        coverage_fraction=np.ones(6, dtype=np.float32),
        group_fc_full=np.zeros((6, 4), dtype=np.float32),
        gradients_full=np.zeros((6, 3), dtype=np.float32),
        labels_full=np.asarray([1, 1, 2, 2, 3, 3], dtype=np.int32),
        prototypes=np.asarray(
            [
                [0.10, 0.20, 0.30, 0.40],
                [0.40, 0.30, 0.20, 0.10],
                [-0.10, -0.20, -0.30, -0.40],
            ],
            dtype=np.float32,
        ),
        fc_paths={"01": "train-01.npy", "02": "train-02.npy"},
        density_evidence={
            "consumed_asset": str(template_surface),
            "filename_density": "512",
            "evidence_source": "consumed_template_surface_path",
        },
        template_dir=tmp_path / "crossfit" / "templates" / "fold-02" / "2mm" / "hemi_L",
    )

    outputs = workflow.write_heldout_subject_outputs(config, template, "sub-03")

    combo_dir = tmp_path / "crossfit" / "subjects" / "sub-03" / "2mm" / "hemi_L"
    expected = {
        "individual_labels_k3.npy",
        "template_correlations.npy",
        "correlation_margin.npy",
        "combo_manifest.json",
    }
    assert expected.issubset({path.name for path in combo_dir.iterdir()})
    assert Path(outputs["labels"]) == (combo_dir / "individual_labels_k3.npy").resolve()
    manifest = json.loads((combo_dir / "combo_manifest.json").read_text())
    assert manifest["subject_was_in_template"] is False
    assert manifest["template_fold"] == 1
    assert manifest["template_training_subjects"] == ["sub-01", "sub-02"]
    assert manifest["density_evidence"]["consumed_subject_surface"] == str(
        subject_surface.resolve()
    )


def test_dry_run_audits_inputs_and_reports_exact_workload_without_writing_outputs(
    tmp_path: Path,
):
    workflow = load_module()
    shared_root = tmp_path / "routea"
    surface_root = tmp_path / "surface"
    hippunfold_root = tmp_path / "hippunfold"
    template_dir = tmp_path / "templates"
    out_root = tmp_path / "crossfit"
    subjects_manifest = tmp_path / "subjects.txt"
    subjects_manifest.write_text("01\n02\n", encoding="utf-8")
    for subject in ["01", "02"]:
        _write_fold_subject_inputs(
            shared_root=shared_root,
            surface_root=surface_root,
            subject=subject,
            offset=0.0,
        )
        subject_surface = (
            hippunfold_root
            / f"sub-{subject}"
            / "hippunfold"
            / f"sub-{subject}"
            / "surf"
            / f"sub-{subject}_hemi-L_space-corobl_den-512_label-hipp_midthickness.surf.gii"
        )
        subject_surface.parent.mkdir(parents=True)
        subject_surface.write_bytes(b"surface")
    template_dir.mkdir()
    (
        template_dir
        / "group-average_hemi-L_space-corobl_den-512_label-hipp_midthickness.surf.gii"
    ).write_bytes(b"surface")
    raw_args = [
            "--subjects-manifest",
            str(subjects_manifest),
            "--allow-subject-only-folds",
            "--n-folds",
            "2",
            "--shared-root",
            str(shared_root),
            "--surface-root",
            str(surface_root),
            "--hippunfold-batch-root",
            str(hippunfold_root),
            "--template-surfaces-dir",
            str(template_dir),
            "--out-root",
            str(out_root),
            "--smoothings",
            "2mm",
            "--hemis",
            "L",
            "--dry-run",
        ]
    args = workflow.build_arg_parser().parse_args(raw_args)
    fold_by_subject = workflow.assign_grouped_folds(
        ["01", "02"],
        {"01": "01", "02": "02"},
        n_folds=2,
        seed=17,
    )

    payload = workflow.dry_run_payload(args, ["01", "02"], fold_by_subject)

    assert payload["status"] == "ok"
    assert payload["template_fits"] == 2
    assert payload["subject_combo_mappings"] == 2
    assert payload["train_test_overlap_count"] == 0
    assert payload["missing_inputs"] == []
    assert not out_root.exists()

    assert workflow.main(raw_args) == 0
    assert not out_root.exists()


def test_main_full_execution_maps_each_subject_once_and_writes_run_manifests(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    workflow = load_module()
    shared_root = tmp_path / "routea"
    surface_root = tmp_path / "surface"
    hippunfold_root = tmp_path / "hippunfold"
    template_dir = tmp_path / "templates"
    out_root = tmp_path / "crossfit"
    subjects_manifest = tmp_path / "subjects.txt"
    subjects_manifest.write_text("01\n02\n", encoding="utf-8")
    for subject in ["01", "02"]:
        _write_fold_subject_inputs(
            shared_root=shared_root,
            surface_root=surface_root,
            subject=subject,
            offset=0.0,
        )
        subject_surface = (
            hippunfold_root
            / f"sub-{subject}"
            / "hippunfold"
            / f"sub-{subject}"
            / "surf"
            / f"sub-{subject}_hemi-L_space-corobl_den-512_label-hipp_midthickness.surf.gii"
        )
        subject_surface.parent.mkdir(parents=True)
        subject_surface.write_bytes(b"surface")
    template_dir.mkdir()
    template_surface = (
        template_dir
        / "group-average_hemi-L_space-corobl_den-512_label-hipp_midthickness.surf.gii"
    )
    template_surface.write_bytes(b"surface")
    fit_calls: list[tuple[int, str, str]] = []
    mapping_calls: list[tuple[int, str]] = []

    def fake_fit(config, fold_index, smoothing, hemi):
        training, heldout = workflow._fold_partitions(config.fold_by_subject, fold_index)
        fit_calls.append((fold_index, smoothing, hemi))
        return SimpleNamespace(
            fold_index=fold_index,
            smoothing=smoothing,
            hemi=hemi,
            training_subjects=training,
            heldout_subjects=heldout,
            template_dir=out_root / "templates" / f"fold-{fold_index + 1:02d}" / smoothing / f"hemi_{hemi}",
        )

    def fake_write(config, template, subject):
        mapping_calls.append((template.fold_index, subject))
        return {"manifest": str(out_root / "fake" / subject / "combo_manifest.json")}

    monkeypatch.setattr(workflow, "ensure_hippomaps_on_path", lambda path: path)
    monkeypatch.setattr(workflow, "fit_fold_template", fake_fit)
    monkeypatch.setattr(workflow, "write_heldout_subject_outputs", fake_write)
    raw_args = [
        "--subjects-manifest",
        str(subjects_manifest),
        "--allow-subject-only-folds",
        "--n-folds",
        "2",
        "--shared-root",
        str(shared_root),
        "--surface-root",
        str(surface_root),
        "--hippunfold-batch-root",
        str(hippunfold_root),
        "--template-surfaces-dir",
        str(template_dir),
        "--out-root",
        str(out_root),
        "--smoothings",
        "2mm",
        "--hemis",
        "L",
    ]

    assert workflow.main(raw_args) == 0

    assert sorted(fit_calls) == [(0, "2mm", "L"), (1, "2mm", "L")]
    assert sorted(subject for _fold, subject in mapping_calls) == ["01", "02"]
    assert len(mapping_calls) == 2
    assert (out_root / "folds" / "fold_assignments.tsv").exists()
    fold_manifest = json.loads((out_root / "folds" / "fold_manifest.json").read_text())
    assert all(row["train_test_disjoint"] for row in fold_manifest["folds"])
    run_manifest = json.loads((out_root / "crossfit_manifest.json").read_text())
    assert run_manifest["status"] == "ok"
    assert run_manifest["template_fits_completed"] == 2
    assert run_manifest["subject_combo_mappings_completed"] == 2
    assert run_manifest["family"]["subject_only_waiver"] is True


def test_resume_template_validation_rejects_parameter_or_training_mismatch(tmp_path: Path):
    workflow = load_module()
    surface = (
        tmp_path
        / "group-average_hemi-L_space-corobl_den-512_label-hipp_midthickness.surf.gii"
    )
    surface.write_bytes(b"surface")
    config = workflow.CrossfitConfig(
        shared_root=tmp_path / "routea",
        surface_root=tmp_path / "surface",
        template_surfaces_dir=tmp_path,
        hippunfold_batch_root=tmp_path / "hippunfold",
        out_root=tmp_path / "crossfit",
        fold_by_subject={"01": 0, "02": 0, "03": 1},
        coverage_threshold=0.80,
        n_components=5,
        gradient_random_state=0,
        cluster_seed=0,
    )
    manifest = {
        "status": "ok",
        "fold_index": 1,
        "smoothing": "2mm",
        "hemi": "L",
        "training_subjects": ["sub-01", "sub-02"],
        "heldout_subjects": ["sub-03"],
        "train_test_disjoint": True,
        "coverage_threshold": 0.80,
        "n_components": 5,
        "gradient_random_state": 0,
        "cluster_seed": 0,
        "k_final": 3,
        "density_evidence": {"consumed_asset": str(surface.resolve())},
        "fc_paths": {
            "sub-01": str(
                (tmp_path / "routea" / "sub-01" / "fc" / "kong2022" / "2mm" / "hemi_L" / "hipp_vertex_to_network_fc.npy").resolve()
            ),
            "sub-02": str(
                (tmp_path / "routea" / "sub-02" / "fc" / "kong2022" / "2mm" / "hemi_L" / "hipp_vertex_to_network_fc.npy").resolve()
            ),
        },
        "valid_mask_paths": {
            "sub-01": str(
                (tmp_path / "surface" / "sub-01" / "surface" / "tsnr_gate" / "sub-01_hemi-L_valid_mask.npy").resolve()
            ),
            "sub-02": str(
                (tmp_path / "surface" / "sub-02" / "surface" / "tsnr_gate" / "sub-02_hemi-L_valid_mask.npy").resolve()
            ),
        },
    }

    workflow.validate_template_manifest(manifest, config, 1, "2mm", "L")

    wrong_parameter = dict(manifest, coverage_threshold=0.75)
    with pytest.raises(ValueError, match="Resume template mismatch"):
        workflow.validate_template_manifest(wrong_parameter, config, 1, "2mm", "L")
    wrong_training = dict(manifest, training_subjects=["sub-01", "sub-03"])
    with pytest.raises(ValueError, match="Resume template mismatch"):
        workflow.validate_template_manifest(wrong_training, config, 1, "2mm", "L")
    wrong_fc = dict(manifest, fc_paths=dict(manifest["fc_paths"], **{"sub-01": "/wrong/fc.npy"}))
    with pytest.raises(ValueError, match="Resume template mismatch"):
        workflow.validate_template_manifest(wrong_fc, config, 1, "2mm", "L")


def test_fit_or_load_fold_template_prefers_valid_saved_template_on_resume(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    workflow = load_module()
    config = workflow.CrossfitConfig(
        shared_root=tmp_path,
        surface_root=tmp_path,
        template_surfaces_dir=tmp_path,
        hippunfold_batch_root=tmp_path,
        out_root=tmp_path,
        fold_by_subject={"01": 0, "02": 1},
    )
    manifest_path = (
        tmp_path / "templates" / "fold-01" / "2mm" / "hemi_L" / "template_manifest.json"
    )
    manifest_path.parent.mkdir(parents=True)
    manifest_path.write_text("{}\n", encoding="utf-8")
    sentinel = object()
    monkeypatch.setattr(workflow, "load_fold_template", lambda *args: sentinel)
    monkeypatch.setattr(
        workflow,
        "fit_fold_template",
        lambda *args: pytest.fail("fit should not run when valid resume template exists"),
    )

    assert workflow.fit_or_load_fold_template(config, 0, "2mm", "L", resume=True) is sentinel


def test_resume_subject_validation_rejects_wrong_template_fold(tmp_path: Path):
    workflow = load_module()
    config = workflow.CrossfitConfig(
        shared_root=tmp_path / "routea",
        surface_root=tmp_path / "surface",
        template_surfaces_dir=tmp_path / "templates",
        hippunfold_batch_root=tmp_path / "hippunfold",
        out_root=tmp_path / "crossfit",
        fold_by_subject={"01": 0, "02": 0, "03": 1},
    )
    template = SimpleNamespace(
        fold_index=1,
        smoothing="2mm",
        hemi="L",
        training_subjects=("01", "02"),
        heldout_subjects=("03",),
        template_dir=tmp_path / "crossfit" / "templates" / "fold-02" / "2mm" / "hemi_L",
    )
    manifest = {
        "status": "ok",
        "subject": "sub-03",
        "smoothing": "2mm",
        "hemi": "L",
        "template_fold": 1,
        "template_training_subjects": ["sub-01", "sub-02"],
        "subject_was_in_template": False,
        "inputs": {
            "fc": str(
                (tmp_path / "routea" / "sub-03" / "fc" / "kong2022" / "2mm" / "hemi_L" / "hipp_vertex_to_network_fc.npy").resolve()
            ),
            "valid_mask": str(
                (tmp_path / "surface" / "sub-03" / "surface" / "tsnr_gate" / "sub-03_hemi-L_valid_mask.npy").resolve()
            ),
            "subject_surface": str(
                (tmp_path / "hippunfold" / "sub-03" / "hippunfold" / "sub-03" / "surf" / "sub-03_hemi-L_space-corobl_den-512_label-hipp_midthickness.surf.gii").resolve()
            ),
            "template_profiles": str(
                (template.template_dir / "template_profiles.npy").resolve()
            ),
        },
    }

    workflow.validate_subject_manifest(manifest, config, template, "03")

    wrong = dict(manifest, template_fold=0)
    with pytest.raises(ValueError, match="Resume subject mismatch"):
        workflow.validate_subject_manifest(wrong, config, template, "03")


def test_write_or_load_heldout_outputs_prefers_valid_saved_combo_on_resume(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    workflow = load_module()
    config = workflow.CrossfitConfig(
        shared_root=tmp_path,
        surface_root=tmp_path,
        template_surfaces_dir=tmp_path,
        hippunfold_batch_root=tmp_path,
        out_root=tmp_path,
        fold_by_subject={"01": 0, "02": 1},
    )
    template = SimpleNamespace(fold_index=0, smoothing="2mm", hemi="L")
    manifest_path = (
        tmp_path / "subjects" / "sub-02" / "2mm" / "hemi_L" / "combo_manifest.json"
    )
    manifest_path.parent.mkdir(parents=True)
    manifest_path.write_text("{}\n", encoding="utf-8")
    sentinel = {"manifest": manifest_path.resolve()}
    monkeypatch.setattr(
        workflow,
        "load_existing_heldout_subject_outputs",
        lambda *args: sentinel,
    )
    monkeypatch.setattr(
        workflow,
        "write_heldout_subject_outputs",
        lambda *args: pytest.fail("writer should not run for valid resumed combo"),
    )

    got = workflow.write_or_load_heldout_subject_outputs(
        config,
        template,
        "02",
        resume=True,
    )

    assert got is sentinel


def test_existing_run_manifest_requires_resume(tmp_path: Path):
    workflow = load_module()
    out_root = tmp_path / "crossfit"
    out_root.mkdir()
    (out_root / "crossfit_manifest.json").write_text('{"status": "ok"}\n', encoding="utf-8")

    with pytest.raises(FileExistsError, match="--resume"):
        workflow.validate_output_root_policy(out_root, resume=False)

    workflow.validate_output_root_policy(out_root, resume=True)


def test_resume_run_manifest_rejects_changed_input_root(tmp_path: Path):
    workflow = load_module()
    args = workflow.build_arg_parser().parse_args(
        [
            "--subjects-manifest",
            str(tmp_path / "subjects.txt"),
            "--allow-subject-only-folds",
            "--shared-root",
            str(tmp_path / "routea"),
            "--out-root",
            str(tmp_path / "crossfit"),
        ]
    )
    subjects = ["01", "02", "03", "04", "05"]
    family = {
        "family_aware": False,
        "subject_only_waiver": True,
        "family_map_path": None,
    }
    folds = {subject: index for index, subject in enumerate(subjects)}
    run_config = workflow.run_config_payload(args, subjects, family, folds)
    manifest = {"status": "ok", "run_config": run_config}

    workflow.validate_run_manifest_for_resume(manifest, run_config)

    changed = dict(run_config, shared_root=str((tmp_path / "other-routea").resolve()))
    with pytest.raises(ValueError, match="Resume run mismatch"):
        workflow.validate_run_manifest_for_resume(manifest, changed)


def test_main_guard_follows_all_function_definitions():
    tree = ast.parse(MODULE_PATH.read_text(encoding="utf-8"))
    function_lines = [node.lineno for node in tree.body if isinstance(node, ast.FunctionDef)]
    guard_lines = [
        node.lineno
        for node in tree.body
        if isinstance(node, ast.If)
        and isinstance(node.test, ast.Compare)
        and isinstance(node.test.left, ast.Name)
        and node.test.left.id == "__name__"
    ]

    assert len(guard_lines) == 1
    assert guard_lines[0] > max(function_lines)


def test_crossfit_docs_name_family_gate_outputs_and_overview():
    text = DOC_PATH.read_text(encoding="utf-8")
    for token in [
        "subject,family_id",
        "--allow-subject-only-folds",
        "individual_labels_k3.npy",
        "robustness_summary.tsv",
    ]:
        assert token in text
