#!/usr/bin/env python3
# pyright: reportMissingImports=false
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
CORE_DIR = REPO_ROOT / "scripts" / "core_workflow"
if str(CORE_DIR) not in sys.path:
    sys.path.insert(0, str(CORE_DIR))


def _load_overview_module():
    path = CORE_DIR / "06_01_build_group_overview.py"
    spec = importlib.util.spec_from_file_location("group_overview_affinity_provenance", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load module: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


overview_mod = _load_overview_module()


def test_heatmap_text_contrasts_with_actual_sequential_cell_color():
    import numpy as np
    fig, ax = plt.subplots()
    try:
        values = np.array([[0.0, 1.0]])
        ax.imshow(values, cmap="magma", vmin=0, vmax=1)
        overview_mod.annotate_profile_values(ax, values, 1.0)
        assert [t.get_color() for t in ax.texts] == ["white", "#222222"]
    finally:
        plt.close(fig)


def test_confidence_panel_renders_both_assignment_bars_and_margin_annotations():
    fig, ax = plt.subplots(figsize=(4, 3))
    rows = [dict(label=f"L C{i}", metric="normalized_count", top1_network="Default",
                 top2_network="Control", top1_value=.33, top2_value=.1,
                 margin=.23, ambiguous=False) for i in range(1, 7)]
    try:
        overview_mod.plot_confidence_panel(ax, rows, "Confidence")
        assert len(ax.patches) == 2 * len(rows)
        assert len(ax.texts) == len(rows)
        assert ax.get_legend() is not None
    finally:
        plt.close(fig)


def test_confidence_panel_matches_the_routea_reference_layout():
    fig, ax = plt.subplots(figsize=(4, 3))
    rows = [dict(label=f"L C{i}", metric="fisher",
                 top1_network="Default", top2_network="Control",
                 top1_value=.16, top2_value=.13, margin=.03,
                 ambiguous=False) for i in range(1, 3)]
    try:
        overview_mod.plot_confidence_panel(ax, rows, "Confidence")
        legend = ax.get_legend()
        assert legend._ncols == 1
        assert legend.get_frame_on() is True
        assert ax.get_xlim()[1] == pytest.approx(.16 * 1.50)
        assert ax.get_ylim() == pytest.approx((1.391, -0.391))
    finally:
        plt.close(fig)


def _shifted_manifest() -> dict[str, object]:
    return {
        "surface_affinity": overview_mod.GRADIENT_SPECTRAL_AFFINITY,
        "affinity_description": overview_mod.GRADIENT_SPECTRAL_AFFINITY_DESCRIPTION,
    }


def _ctxdrop_manifest(smoothing: str, hemi: str) -> dict[str, object]:
    return {
        **_shifted_manifest(),
        "smoothing": smoothing,
        "hemi": hemi,
        "profile_fc_mode": "cluster_mean_timeseries",
        "profile_source_mode": "canonical",
        "profile_source_root": None,
        "profile_source_registry": None,
        "profile_surface_root": "/outputs_hcp/inputs/kong2022/shared_inputs/surface",
        "reference_root": "/outputs_hcp/inputs/kong2022/shared_inputs/pfm_ctxdrop",
        "resume_input_signature_payload": {"route_tag": "pfm_ctxdrop"},
    }


def _ctxreg_manifest(smoothing: str, hemi: str) -> dict[str, object]:
    return {
        **_ctxdrop_manifest(smoothing, hemi),
        "profile_surface_root": "/outputs_hcp/inputs/kong2022/shared_inputs/pfm_ctxreg",
        "reference_root": "/outputs_hcp/inputs/kong2022/shared_inputs/pfm_ctxreg",
        "resume_input_signature_payload": {"route_tag": "pfm_ctxreg"},
    }


def test_gradient_overview_item_records_four_shifted_cosine_sources() -> None:
    paths = [Path(f"/group/run-{idx}/group_atlas_manifest.json") for idx in range(4)]
    manifests = [_shifted_manifest() for _ in paths]

    item = overview_mod.overview_source_provenance(
        method="gradient-spectral",
        manifest_paths=paths,
        manifests=manifests,
    )
    rendered_title = overview_mod.overview_title(
        atlas="kong2022",
        method="gradient-spectral",
        route_tag="pfm",
        mode_tag=None,
        assignment_variant="fisherz",
        surface_affinity=item["surface_affinity"],
    )

    assert item["surface_affinity"] == "shifted-cosine"
    assert item["affinity_description"] == overview_mod.GRADIENT_SPECTRAL_AFFINITY_DESCRIPTION
    assert item["source_group_manifests"] == [str(path) for path in paths]
    assert "shifted-cosine" in rendered_title


def test_ctxdrop_gradient_provenance_records_original_fc_annotation() -> None:
    pairs = [("2mm", "L"), ("2mm", "R"), ("4mm", "L"), ("4mm", "R")]
    paths = [
        Path(f"/group/pfm_ctxdrop/{smoothing}/{hemi}/group_atlas_manifest.json")
        for smoothing, hemi in pairs
    ]
    manifests = [_ctxdrop_manifest(smoothing, hemi) for smoothing, hemi in pairs]

    provenance = overview_mod.overview_source_provenance(
        method="gradient-spectral",
        manifest_paths=paths,
        manifests=manifests,
    )

    assert provenance["cluster_annotation_signal"] == "original_timeseries"
    assert provenance["cluster_annotation_note"] == (
        "Original-FC annotation of cortex-controlled clusters"
    )
    assert provenance["cluster_annotation_profile_fc_mode"] == "cluster_mean_timeseries"
    assert provenance["cluster_annotation_profile_surface_root"].endswith(
        "/shared_inputs/surface"
    )


def test_standard_and_ctxreg_gradient_provenance_do_not_get_ctxdrop_raw_note() -> None:
    pairs = [("2mm", "L"), ("2mm", "R"), ("4mm", "L"), ("4mm", "R")]
    standard_paths = [Path(f"/group/pfm/{smoothing}/{hemi}/group_atlas_manifest.json") for smoothing, hemi in pairs]
    standard = [
        {
            **_shifted_manifest(),
            "smoothing": smoothing,
            "hemi": hemi,
            "resume_input_signature_payload": {"route_tag": "pfm"},
        }
        for smoothing, hemi in pairs
    ]
    ctxreg_paths = [Path(f"/group/pfm_ctxreg/{smoothing}/{hemi}/group_atlas_manifest.json") for smoothing, hemi in pairs]
    ctxreg = [_ctxreg_manifest(smoothing, hemi) for smoothing, hemi in pairs]

    for paths, manifests in ((standard_paths, standard), (ctxreg_paths, ctxreg)):
        provenance = overview_mod.overview_source_provenance(
            method="gradient-spectral",
            manifest_paths=paths,
            manifests=manifests,
        )
        assert "cluster_annotation_signal" not in provenance
        assert "cluster_annotation_note" not in provenance


@pytest.mark.parametrize(
    "mutator",
    [
        lambda manifest: manifest.update({"profile_surface_root": "/outputs_hcp/inputs/kong2022/shared_inputs/pfm_ctxreg"}),
        lambda manifest: manifest["resume_input_signature_payload"].update({"route_tag": "pfm_ctxreg"}),
        lambda manifest: manifest.update({"profile_fc_mode": "pearson"}),
    ],
)
def test_ctxdrop_gradient_provenance_rejects_mismatched_annotation_source(mutator) -> None:
    pairs = [("2mm", "L"), ("2mm", "R"), ("4mm", "L"), ("4mm", "R")]
    paths = [Path(f"/group/pfm_ctxdrop/{smoothing}/{hemi}/group_atlas_manifest.json") for smoothing, hemi in pairs]
    manifests = [_ctxdrop_manifest(smoothing, hemi) for smoothing, hemi in pairs]
    mutator(manifests[-1])

    with pytest.raises(ValueError, match="ctxdrop original-FC annotation provenance"):
        overview_mod.overview_source_provenance(
            method="gradient-spectral",
            manifest_paths=paths,
            manifests=manifests,
        )


def test_overview_manifest_item_preserves_rendered_title() -> None:
    title = "Group Overview | kong2022 | gradient-spectral | shifted-cosine"
    item = overview_mod.overview_manifest_item(
        base={"atlas": "kong2022", "method": "gradient-spectral"},
        meta={
            "rendered_title": title,
            "source_group_manifests": ["/group/manifest.json"],
            "surface_affinity": "shifted-cosine",
            "affinity_description": overview_mod.GRADIENT_SPECTRAL_AFFINITY_DESCRIPTION,
            "cluster_annotation_signal": "original_timeseries",
            "cluster_annotation_note": "Original-FC annotation of cortex-controlled clusters",
            "assignment_annotation_note": "Counts derived from original-FC ranking",
        },
    )

    assert item["rendered_title"] == title
    assert item["source_group_manifests"] == ["/group/manifest.json"]
    assert item["cluster_annotation_signal"] == "original_timeseries"
    assert item["assignment_annotation_note"] == "Counts derived from original-FC ranking"


def test_individual_overview_title_is_not_labeled_as_group() -> None:
    title = overview_mod.overview_title(
        atlas="kong2022",
        method="gradient-spectral",
        route_tag="routea",
        mode_tag=None,
        assignment_variant="fisherz",
        surface_affinity="shifted-cosine",
        overview_scope="Individual",
    )

    assert title.startswith("Individual Overview | ")
    assert "Group Overview" not in title


def test_individual_ari_skip_preserves_panel_scaffold_without_lines() -> None:
    fig, ax = plt.subplots()
    try:
        overview_mod.plot_subject_split_ari_skipped(ax, smoothing="2mm")

        assert ax.axison is True
        assert ax.get_xlim() == (1.5, 8.5)
        assert ax.get_ylim() == (-0.02, 1.02)
        assert list(ax.lines) == []
        assert "ARI skipped" in ax.get_title()
    finally:
        plt.close(fig)


@pytest.mark.parametrize("invalid_affinity", ["relu-cosine", None])
def test_gradient_overview_rejects_inconsistent_affinity_provenance(
    invalid_affinity: str | None,
) -> None:
    manifests = [_shifted_manifest() for _ in range(4)]
    if invalid_affinity is None:
        manifests[-1].pop("surface_affinity")
    else:
        manifests[-1]["surface_affinity"] = invalid_affinity

    with pytest.raises(
        ValueError,
        match="^Inconsistent gradient-spectral affinity provenance$",
    ):
        overview_mod.gradient_affinity_provenance(manifests)


def test_wta_overview_item_has_sources_without_gradient_affinity_fields() -> None:
    paths = [Path(f"/group/wta-{idx}/group_atlas_manifest.json") for idx in range(4)]

    item = overview_mod.overview_source_provenance(
        method="wta-top",
        manifest_paths=paths,
        manifests=[{} for _ in paths],
    )

    assert item == {"source_group_manifests": [str(path) for path in paths]}
    assert "surface_affinity" not in item
    assert "affinity_description" not in item


def test_invalid_gradient_provenance_fails_before_figure_creation(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    invalid = _shifted_manifest()
    invalid["surface_affinity"] = "relu-cosine"
    monkeypatch.setattr(overview_mod, "load_json", lambda _path: dict(invalid))

    def fail_if_figure_created(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("figure created before provenance validation")

    monkeypatch.setattr(overview_mod.plt, "figure", fail_if_figure_created)

    with pytest.raises(
        ValueError,
        match="^Inconsistent gradient-spectral affinity provenance$",
    ):
        overview_mod.render_combo(
            group_root=tmp_path / "group",
            present_dir=tmp_path / "present",
            method="gradient-spectral",
            atlas="kong2022",
            coverage_tag="cov_0p80",
            route_tag="routea",
            mode_tag=None,
            assignment_variant="fisherz",
            gradient_comp_tag="g1g2g3",
        )


def test_overview_title_with_group_name_and_n_subjects() -> None:
    title = overview_mod.overview_title(
        atlas="kong2022",
        method="gradient-spectral",
        route_tag="routea",
        mode_tag=None,
        assignment_variant="fisherz",
        surface_affinity="shifted-cosine",
        group_name="NTHC",
        n_subjects=51,
    )
    assert title == "Group Overview | kong2022 | gradient-spectral | routea | fisherz | NTHC (N=51)"

    title_wta = overview_mod.overview_title(
        atlas="kong2022",
        method="wta-top",
        route_tag="routea",
        mode_tag=None,
        assignment_variant=None,
        group_name="TEHC",
        n_subjects=40,
    )
    assert title_wta == "Group Overview | kong2022 | wta-top | routea | TEHC (N=40)"


def test_infer_group_name_from_path() -> None:
    assert overview_mod.infer_group_name_from_path("outputs_stanford/present/overviews/routea_nthc") == "NTHC"
    assert overview_mod.infer_group_name_from_path("outputs_stanford/present/overviews/routea_tehc") == "TEHC"
    assert overview_mod.infer_group_name_from_path("outputs_stanford/present/overviews/routea_nts") == "NTS"
    assert overview_mod.infer_group_name_from_path("outputs_stanford/present/overviews/routea_tis") == "TIS"
    assert overview_mod.infer_group_name_from_path("outputs_stanford/present/overviews/routea_hc") == "HC"
    assert overview_mod.infer_group_name_from_path("outputs_stanford/present/overviews/routea_symp") == "Symp"
    assert overview_mod.infer_group_name_from_path("outputs_stanford/present/overviews/routea_trauma") == "Trauma"
    assert overview_mod.infer_group_name_from_path("outputs_stanford/present/overviews/routea_nontrauma") == "Non-Trauma"
