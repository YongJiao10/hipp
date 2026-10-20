from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
TARGET_DIR = REPO_ROOT / "scripts/visualize/multidataset_subfield_composition_2mm_2col"


def _load_module(module_name: str, path: Path):
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


RENDERER = _load_module(
    "subfield_composition_renderer_under_test",
    TARGET_DIR / "render_multidataset_subfield_composition_2mm_2col.py",
)
ANALYZER = _load_module(
    "stanford_subfield_difference_analyzer_under_test",
    TARGET_DIR / "analyze_and_render_stanford_subfield_differences.py",
)


def test_comparison_signal_branch_resolves_formal_roots_without_reference_worktree():
    spec = RENDERER.GridSpec("2mm", "36p", atlas="7net")
    standard_columns = RENDERER.get_comparison_visual_columns(spec)

    standard = RENDERER.resolve_comparison_signal_branch_cohort(
        standard_columns[0][0], spec, "standard"
    )
    assert standard["signal_branch"] == "standard"
    assert standard["group_root"] == standard_columns[0][0]["group_root"]

    ctxreg = RENDERER.resolve_comparison_signal_branch_cohort(
        standard_columns[0][0], spec, "ctxreg"
    )
    assert ctxreg["signal_branch"] == "ctxreg"
    assert ctxreg["group_root"] == REPO_ROOT / "outputs_hcp/work/group_atlas/routea_ctxreg"
    assert ctxreg["grad_subpath"] == (
        REPO_ROOT
        / "outputs_hcp/work/group_atlas/routea_ctxreg/gradient-spectral/kong2022/2mm/cov_0p80/g1g2g3"
    )
    assert "ctxreg_worktree" not in str(ctxreg["group_root"])

    iowa = RENDERER.resolve_comparison_signal_branch_cohort(
        standard_columns[0][1], spec, "ctxreg"
    )
    assert iowa["group_root"] == (
        REPO_ROOT / "outputs_iowa_jj_tms/work/group_atlas/routea_ctxreg/hc"
    )


def test_manifest_label_selection_is_exact_and_rejects_conflicting_declarations():
    manifest = {
        "assignment_variants": {
            "fisherz": {
                "template_functional_atlas_label": "/formal/canonical.label.gii",
                "template_functional_atlas_label_variant": "/formal/canonical.label.gii",
            }
        }
    }
    assert RENDERER.select_manifest_label_path(manifest, assignment_variant="fisherz") == Path(
        "/formal/canonical.label.gii"
    )

    conflicting = {
        "assignment_variants": {
            "fisherz": {
                "template_functional_atlas_label": "/formal/canonical.label.gii",
                "template_functional_atlas_label_variant": "/wrong/variant.label.gii",
            }
        }
    }
    with pytest.raises(ValueError, match="conflicting label paths"):
        RENDERER.select_manifest_label_path(conflicting, assignment_variant="fisherz")


def test_composition_fails_closed_when_manifest_subject_input_is_missing(tmp_path):
    group_root = tmp_path / "group"
    grad_subpath = group_root / "gradient-spectral/kong2022/2mm/cov_0p80/g1g2g3"
    hemi_dir = grad_subpath / "hemi_L"
    hemi_dir.mkdir(parents=True)
    (hemi_dir / "group_atlas_manifest.json").write_text(
        json.dumps({"cluster_roi_profile_subjects_used": ["sub-001"]}),
        encoding="utf-8",
    )
    np.save(hemi_dir / "group_labels_final.npy", np.ones(4, dtype=np.int32))
    np.save(hemi_dir / "group_support_mask.npy", np.ones(4, dtype=bool))

    cinfo = {
        "id": "fixture",
        "group_root": group_root,
        "grad_subpath": grad_subpath,
        "hippunfold_root": tmp_path / "hippunfold",
        "signal_branch": "standard",
    }
    expected = (
        tmp_path
        / "hippunfold/sub-001/hippunfold/sub-001/metric/sub-001_hemi-L_den-512_label-hipp_atlas-multihist7_subfields.label.gii"
    )
    with pytest.raises(FileNotFoundError, match=str(expected)):
        RENDERER.compute_cohort_subfield_composition(
            cinfo, "v2_cohort_specific_atlas", smoothing="2mm"
        )


def test_analyzer_loads_only_canonical_grid_exports(tmp_path, monkeypatch):
    stats_dir = tmp_path / "stats"
    stats_dir.mkdir()
    (stats_dir / "stanford_subfield_group_differences_stats.tsv").write_text(
        "placeholder\n", encoding="utf-8"
    )
    (stats_dir / "stanford_subfield_subject_level_long.tsv").write_text(
        "placeholder\n", encoding="utf-8"
    )
    monkeypatch.setattr(ANALYZER, "STATS_DIR", stats_dir)
    monkeypatch.setattr(ANALYZER, "OUTPUT_DIR", tmp_path / "output")

    with pytest.raises(FileNotFoundError, match="stanford_subfield_group_differences_stats_2mm_36p.tsv"):
        ANALYZER.load_exported_render_inputs(
            ANALYZER.TARGET_ATLAS_MODE,
            spec=ANALYZER.GridSpec("2mm", "36p"),
        )


def test_analyzer_fails_closed_on_missing_subject_inputs(tmp_path, monkeypatch):
    group_root = tmp_path / "group"
    grad_root = group_root / "gradient-spectral/kong2022/2mm/cov_0p80/g1g2g3"
    for hemi in ("L", "R"):
        hemi_dir = grad_root / f"hemi_{hemi}"
        hemi_dir.mkdir(parents=True)
        (hemi_dir / "group_atlas_manifest.json").write_text(
            json.dumps({"cluster_roi_profile_subjects_used": ["sub-001"]}),
            encoding="utf-8",
        )
        np.save(hemi_dir / "group_labels_final.npy", np.ones(4, dtype=np.int32))
        np.save(hemi_dir / "group_support_mask.npy", np.ones(4, dtype=bool))
    cinfo = {
        "id": "stanford_fixture",
        "name": "Fixture",
        "stem": "fixture",
        "group_root": group_root,
        "hippunfold_root": tmp_path / "hippunfold",
        "n": 1,
    }
    monkeypatch.setattr(ANALYZER, "get_diff_cohorts", lambda _spec: [cinfo])

    expected = (
        tmp_path
        / "hippunfold/sub-001/hippunfold/sub-001/metric/"
        "sub-001_hemi-L_den-512_label-hipp_atlas-multihist7_subfields.label.gii"
    )
    with pytest.raises(FileNotFoundError, match=str(expected)):
        ANALYZER.load_all_subject_data(
            "v2_cohort_specific_atlas", spec=ANALYZER.GridSpec("2mm", "36p")
        )


def test_analyzer_fails_closed_on_missing_individual_template_label(tmp_path, monkeypatch):
    group_root = tmp_path / "group"
    grad_root = group_root / "gradient-spectral/kong2022/2mm/cov_0p80/g1g2g3"
    for hemi in ("L", "R"):
        hemi_dir = grad_root / f"hemi_{hemi}"
        hemi_dir.mkdir(parents=True)
        (hemi_dir / "group_atlas_manifest.json").write_text(
            json.dumps({"cluster_roi_profile_subjects_used": ["sub-001"]}),
            encoding="utf-8",
        )

    hipp_root = tmp_path / "hippunfold"
    for hemi in ("L", "R"):
        metric_dir = hipp_root / "sub-001/hippunfold/sub-001/metric"
        surf_dir = hipp_root / "sub-001/hippunfold/sub-001/surf"
        metric_dir.mkdir(parents=True, exist_ok=True)
        surf_dir.mkdir(parents=True, exist_ok=True)
        (metric_dir / f"sub-001_hemi-{hemi}_den-512_label-hipp_atlas-multihist7_subfields.label.gii").touch()
        (surf_dir / f"sub-001_hemi-{hemi}_space-T1w_den-512_label-hipp_midthickness.surf.gii").touch()

    cinfo = {
        "id": "stanford_fixture",
        "name": "Fixture",
        "stem": "fixture",
        "group_root": group_root,
        "hippunfold_root": hipp_root,
        "n": 1,
    }
    monkeypatch.setattr(ANALYZER, "get_diff_cohorts", lambda _spec: [cinfo])
    monkeypatch.setattr(ANALYZER, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(ANALYZER, "get_subfield_labels", lambda _path: np.ones(4, dtype=np.int32))
    monkeypatch.setattr(ANALYZER, "vertex_area_weights", lambda _path: np.ones(4, dtype=np.float64))

    expected = (
        tmp_path
        / "outputs_stanford/work/hcp151_fixed_template_matching/subjects/sub-001/"
        "matching/2mm/hemi-L/labels.npy"
    )
    with pytest.raises(FileNotFoundError, match=str(expected)):
        ANALYZER.load_all_subject_data(
            "v3_individual_template_matching", spec=ANALYZER.GridSpec("2mm", "36p")
        )
