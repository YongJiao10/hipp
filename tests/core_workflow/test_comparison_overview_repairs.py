"""Regression coverage for the two-interpretation comparison overview."""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import numpy as np
import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
CORE_DIR = REPO_ROOT / "scripts" / "core_workflow"
if str(CORE_DIR) not in sys.path:
    sys.path.insert(0, str(CORE_DIR))
SPEC = importlib.util.spec_from_file_location(
    "comparison_overview_repairs", CORE_DIR / "06_01_build_group_overview.py"
)
assert SPEC is not None and SPEC.loader is not None
overview = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = overview
SPEC.loader.exec_module(overview)

VARIANTS = ["fisherz", "ovr_logistic_canonical7"]


@pytest.fixture
def render_inputs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Render real panels from synthetic validated data without Workbench."""
    group_root = tmp_path / "group"
    figures: list[Figure] = []
    surface_titles: list[str] = []
    scratch_paths: list[Path] = []
    image_path = tmp_path / "surface.png"
    plt.imsave(image_path, np.ones((2, 2, 3)))
    monkeypatch.setattr(overview.tempfile, "tempdir", str(tmp_path))

    def save_figure(fig, path, **_kwargs):
        figures.append(fig)
        Path(path).write_bytes(image_path.read_bytes())

    def render_surface(**kwargs):
        assert kwargs["surface_cache_root"] == overview.REPO_ROOT / "tmp" / "overview_hippocampal_surfaces"
        surface_titles.append(kwargs["title"])
        scratch_paths.append(kwargs["outdir"])
        return {"biglegend_png": str(image_path)}

    monkeypatch.setattr(Figure, "savefig", save_figure)
    monkeypatch.setattr(overview, "render_hippocampal_grid", render_surface)
    monkeypatch.setattr(overview, "resolve_template_pair_assets", lambda *_a, **_k: (
        tmp_path / "left.surf.gii", tmp_path / "right.surf.gii", "template", tmp_path / "template.spec"
    ))
    monkeypatch.setattr(overview, "long_axis_order", lambda _path: np.arange(4))
    # Assignment and lineage contracts have separate coverage; these tests
    # exercise their validated data at the figure boundary.
    monkeypatch.setattr(overview, "validate_group_atlas_lineage", lambda *_a, **_k: {})
    monkeypatch.setattr(overview, "ensure_gradient_assignment_variants", lambda _p, manifest: manifest)

    def validated_assignment(run_dir, _manifest, variant):
        return {
            "network_names": ["Visual", "Default"],
            "profile": np.load(run_dir / f"{variant}_profile.npy"),
        }

    monkeypatch.setattr(overview, "validate_gradient_assignment", validated_assignment)

    def render(*, amplitude=1.0, layout="dual-interpretation", **kwargs):
        for smoothing in ("2mm", "4mm"):
            for hemi in ("L", "R"):
                run_dir = group_root / "gradient-spectral" / "kong2022" / smoothing / "cov_0p80" / f"hemi_{hemi}"
                run_dir.mkdir(parents=True, exist_ok=True)
                profile = np.array([[-0.2, 0.4], [0.6, -0.1]], dtype=np.float32) * amplitude
                np.save(run_dir / "cluster_network_profile.npy", profile)
                np.save(run_dir / "group_labels_final.npy", np.array([1, 1, 2, 2], dtype=np.int32))
                assignments = {}
                for variant in VARIANTS:
                    status = run_dir / f"{variant}_status.json"
                    label = run_dir / f"{variant}.label.gii"
                    label.touch()
                    status.write_text(json.dumps({"items": [
                        {"cluster_id": 1, "top1_network": "Default", "top2_network": "Visual",
                         "top1_fisher_z": 0.4 * amplitude, "top2_fisher_z": -0.2 * amplitude},
                        {"cluster_id": 2, "top1_network": "Visual", "top2_network": "Default",
                         "top1_fisher_z": 0.6 * amplitude, "top2_fisher_z": -0.1 * amplitude},
                    ]}), encoding="utf-8")
                    assignments[variant] = {
                        "cluster_assignment_status": str(status),
                        "template_functional_atlas_label": str(label),
                    }
                    values = profile if variant == "fisherz" else np.array([[-1.5, 2.0], [0.5, -0.25]]) * amplitude
                    np.save(run_dir / f"{variant}_profile.npy", values)
                (run_dir / "group_atlas_manifest.json").write_text(json.dumps({
                    "chosen_k_final": 2, "n_subjects": 3,
                    "surface_affinity": overview.GRADIENT_SPECTRAL_AFFINITY,
                    "affinity_description": overview.GRADIENT_SPECTRAL_AFFINITY_DESCRIPTION,
                    "assignment_variants": assignments,
                }), encoding="utf-8")
                (run_dir / "group_feature_summary.json").write_text(
                    json.dumps({"network_names": ["Visual", "Default"]}), encoding="utf-8"
                )
                (run_dir / "k_selection_summary.tsv").write_text(
                    "k\tmax_profile_similarity\n2\t0.5\n", encoding="utf-8"
                )
        meta = overview.render_combo(
            group_root=group_root, present_dir=tmp_path / "present",
            method="gradient-spectral", atlas="kong2022", coverage_tag="cov_0p80",
            route_tag="pfm", mode_tag=None,
            assignment_variant="fisherz", skip_subject_split_ari=True, layout=layout,
            **kwargs,
        )
        return meta, figures[-1], surface_titles, scratch_paths

    yield render
    for figure in figures:
        plt.close(figure)


def _profile_images(figure: Figure):
    return [image for axis in figure.axes for image in axis.images if image.get_array().ndim == 2]


def test_dual_interpretation_labels_rows_surface_and_combined_metadata(render_inputs):
    meta, figure, surface_titles, scratch_paths = render_inputs()

    assert [row["assignment_variant"] for row in meta["rows"]] == VARIANTS
    assert [row["row_label"] for row in meta["rows"]] == ["2mm Fisher-z", "2mm canonical7 OVR"]
    assert surface_titles[0].endswith("2mm fisherz")
    assert surface_titles[1].endswith("2mm ovr logistic canonical7")
    titles = [axis.get_title().replace("\n", " | ") for axis in figure.axes]
    assert "2mm Fisher-z | functional atlas surface" in titles
    assert "2mm canonical7 OVR | functional atlas surface" in titles
    assert "Fisher-z" in figure._suptitle.get_text()
    assert "canonical7 OVR" in figure._suptitle.get_text()
    assert meta["rendered_title"] == figure._suptitle.get_text()
    assert meta["assignment_variants"] == VARIANTS
    assert len(meta["source_group_manifests"]) == 2
    sidecar = json.loads(Path(meta["overview_source_sidecar"]).read_text(encoding="utf-8"))
    assert sidecar["assignment_variants"] == VARIANTS
    assert sidecar["assignment_variant"] == "dual-interpretation"
    item = overview.overview_manifest_item(base={"assignment_variant": "fisherz"}, meta=meta)
    assert item["assignment_variants"] == VARIANTS
    assert item["assignment_variant"] == "dual-interpretation"
    assert all(path.is_relative_to(REPO_ROOT / "tmp") for path in scratch_paths)


@pytest.mark.parametrize("amplitude", [1, 2, 3, 4, 5, 6, 7, 8])
def test_shared_profile_ranges_hold_across_comparison_arms(render_inputs, amplitude):
    meta, figure, _, _ = render_inputs(
        amplitude=amplitude,
        profile_vmax_by_variant={"fisherz": 6.0, "ovr_logistic_canonical7": 20.0},
    )

    assert [image.get_clim() for image in _profile_images(figure)] == [(-6.0, 6.0), (-20.0, 20.0)]
    assert [(row["profile_vmin"], row["profile_vmax"]) for row in meta["rows"]] == [(-6.0, 6.0), (-20.0, 20.0)]


def test_dual_interpretation_panel_titles_do_not_overlap(render_inputs):
    _, figure, _, _ = render_inputs()
    figure.canvas.draw()
    renderer = figure.canvas.get_renderer()
    titles = [axis.title.get_window_extent(renderer) for axis in figure.axes if axis.get_title()]
    for i, first in enumerate(titles):
        for second in titles[i + 1:]:
            assert not first.overlaps(second), 'Adjacent panel titles overlap'


@pytest.mark.parametrize("layout, expected", [
    ("dual-interpretation", [(-0.6, 0.6), (-2.0, 2.0)]),
    ("dual-smoothing", [(-0.6, 0.6), (-0.6, 0.6)]),
])
def test_omitted_shared_ranges_keep_local_scaling_and_legacy_layout(render_inputs, layout, expected):
    _, figure, surface_titles, _ = render_inputs(layout=layout)

    assert np.asarray([image.get_clim() for image in _profile_images(figure)]) == pytest.approx(np.asarray(expected))
    if layout == "dual-smoothing":
        assert "2mm | functional atlas surface" in [axis.get_title() for axis in figure.axes]
        assert "4mm | functional atlas surface" in [axis.get_title() for axis in figure.axes]
        assert all(title.endswith("fisherz") for title in surface_titles)
        assert "ovr" not in figure._suptitle.get_text().lower()


def _write_invalid_lineage_pair(root: Path) -> list[Path]:
    paths = []
    for hemi in ("L", "R"):
        run_dir = root / "gradient-spectral" / "kong2022" / "2mm" / "cov_0p80" / f"hemi_{hemi}"
        run_dir.mkdir(parents=True)
        path = run_dir / "group_atlas_manifest.json"
        path.write_text(json.dumps({
            "status": "ok", "atlas": "kong2022", "method": "gradient-spectral",
            "network_taxonomy": "canonical8",
            "surface_affinity": overview.GRADIENT_SPECTRAL_AFFINITY,
            "affinity_description": overview.GRADIENT_SPECTRAL_AFFINITY_DESCRIPTION,
            "assignment_variants": {},
        }), encoding="utf-8")
        np.save(run_dir / "group_labels_final.npy", np.array([1], dtype=np.int32))
        paths.append(path)
    return paths


def test_dual_interpretation_preflight_rejects_both_invalid_source_lineages(tmp_path):
    paths = _write_invalid_lineage_pair(tmp_path / "group")

    with pytest.raises(RuntimeError, match="invalid_group_atlas_lineage") as caught:
        overview.ensure_combo_ready(
            group_root=tmp_path / "group", method="gradient-spectral", atlas="kong2022",
            coverage_tag="cov_0p80", route_tag="pfm", smoothings=["2mm"],
            layout="dual-interpretation",
        )

    assert all(str(path) in str(caught.value) for path in paths)


def test_dual_interpretation_render_rejects_invalid_lineage_before_plotting(tmp_path, monkeypatch):
    _write_invalid_lineage_pair(tmp_path / "group")
    monkeypatch.setattr(overview, "ensure_gradient_assignment_variants", lambda _p, manifest: manifest)

    def fail_if_plotting_starts(*_args, **_kwargs):
        pytest.fail("invalid lineage reached figure construction")

    monkeypatch.setattr(overview.plt, "figure", fail_if_plotting_starts)
    with pytest.raises(ValueError, match="Expected active canonical7"):
        overview.render_combo(
            group_root=tmp_path / "group", present_dir=tmp_path / "present",
            method="gradient-spectral", atlas="kong2022", coverage_tag="cov_0p80",
            route_tag="pfm", mode_tag=None,
            assignment_variant="fisherz", layout="dual-interpretation",
        )
