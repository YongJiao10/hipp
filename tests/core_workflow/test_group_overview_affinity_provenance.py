#!/usr/bin/env python3
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


def _shifted_manifest() -> dict[str, object]:
    return {
        "surface_affinity": overview_mod.GRADIENT_SPECTRAL_AFFINITY,
        "affinity_description": overview_mod.GRADIENT_SPECTRAL_AFFINITY_DESCRIPTION,
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
        route_tag="routea",
        mode_tag=None,
        assignment_variant="fisherz",
        surface_affinity=item["surface_affinity"],
    )

    assert item["surface_affinity"] == "shifted-cosine"
    assert item["affinity_description"] == overview_mod.GRADIENT_SPECTRAL_AFFINITY_DESCRIPTION
    assert item["source_group_manifests"] == [str(path) for path in paths]
    assert "shifted-cosine" in rendered_title


def test_overview_manifest_item_preserves_rendered_title() -> None:
    title = "Group Overview | kong2022 | gradient-spectral | shifted-cosine"
    item = overview_mod.overview_manifest_item(
        base={"atlas": "kong2022", "method": "gradient-spectral"},
        meta={
            "rendered_title": title,
            "source_group_manifests": ["/group/manifest.json"],
            "surface_affinity": "shifted-cosine",
            "affinity_description": overview_mod.GRADIENT_SPECTRAL_AFFINITY_DESCRIPTION,
        },
    )

    assert item["rendered_title"] == title
    assert item["source_group_manifests"] == ["/group/manifest.json"]


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
            scene=tmp_path / "scene",
            route_tag="routea",
            mode_tag=None,
            assignment_variant="fisherz",
            gradient_comp_tag="g1g2g3",
        )
