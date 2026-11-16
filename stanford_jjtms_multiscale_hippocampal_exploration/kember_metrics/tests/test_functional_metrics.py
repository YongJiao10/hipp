"""Contracts for the Kember article-metric template-matching workflow."""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

import numpy as np


SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))


def _kember_module():
    return importlib.import_module("feature_build")


def test_feature_builder_exposes_stable_combined_methods() -> None:
    module = _kember_module()

    assert module.METHODS == ("hcp_template_matching", "hcp_fixed")


def test_paper_metrics_match_hand_checked_surface_area_segregation_and_sharpness() -> None:
    """Breaks if a Kember metric uses a custom transition/geometry substitute."""

    module = _kember_module()
    base = np.array([1.0, -1.0, 1.0, -1.0])
    orthogonal = np.array([1.0, 1.0, -1.0, -1.0])
    time_series = np.vstack(
        [
            orthogonal,
            -0.5 * base + orthogonal,
            -0.25 * base + orthogonal,
            base + orthogonal,
            base - 0.5 * orthogonal,
            base - 0.5 * orthogonal,
        ]
    )
    labels = np.array([1, 1, 2, 3, 3, 3])
    coordinates = np.column_stack([np.arange(6, dtype=float), np.zeros((6, 2))])
    faces = np.array([[0, 1, 3], [1, 2, 3], [2, 3, 4], [3, 4, 5]])
    vertex_area = np.array([1.0, 1.0, 1.0, 2.0, 2.0, 2.0])

    metrics = module.compute_kember_metrics(
        labels=labels,
        time_series=time_series,
        native_vertex_area_mm2=vertex_area,
        native_valid_mask=np.ones(labels.size, dtype=bool),
        coordinates=coordinates,
        faces=faces,
        vertex_area_metric_path="native_surfareacorobl.shape.gii",
        boundary_radius_mm=10.0,
    )
    posterior = metrics["posterior"]

    assert posterior["surface_area_mm2"] == 6.0
    assert np.isclose(posterior["surface_area_fraction"], 2.0 / 3.0)
    assert np.isclose(posterior["functional_segregation_d"], 6.040188312727962)
    assert np.isclose(posterior["boundary_sharpness_d"], 6.040188312727962)
    assert posterior["boundary_vertex_count"] == 2
    assert posterior["boundary_vertex_valid_count"] == 2


def test_article_metric_endpoint_contract_has_432_units_and_no_transition_metrics() -> None:
    """Breaks if the active direct workflow reintroduces custom endpoints."""

    direct = importlib.import_module("contrasts")
    endpoints = direct.enumerate_direct_endpoints()

    assert len(endpoints) == 432
    assert endpoints.groupby("metric_family").size().to_dict() == {
        "surface_area": 144,
        "functional_segregation": 144,
        "boundary_sharpness": 144,
    }
    assert set(endpoints["metric"]) == {
        "surface_area_mm2",
        "functional_segregation_d",
        "boundary_sharpness_d",
    }
    assert not endpoints["metric"].str.contains("transition|vertex_fraction|ap_centroid").any()


def test_consistent_smoothing_selects_only_2mm_for_figures() -> None:
    """Breaks if a redundant 4-mm figure is rendered despite matching conclusions."""

    visualization = importlib.import_module("plot_contrasts")
    effects = np.array(
        [
            ("hcp_template_matching", "surface_area", "surface_area_mm2", "posterior", "TEHC−NTHC", "L", 2, -0.40, 0.031),
            ("hcp_template_matching", "surface_area", "surface_area_mm2", "posterior", "TEHC−NTHC", "L", 4, -0.22, 0.042),
            ("hcp_template_matching", "functional_segregation", "functional_segregation_d", "posterior", "TEHC−NTHC", "L", 2, 0.33, 0.210),
            ("hcp_template_matching", "functional_segregation", "functional_segregation_d", "posterior", "TEHC−NTHC", "L", 4, 0.19, 0.440),
        ],
        dtype=[
            ("method", "U32"),
            ("metric_family", "U32"),
            ("metric", "U32"),
            ("system", "U32"),
            ("contrast", "U32"),
            ("hemisphere", "U1"),
            ("smoothing_mm", "i4"),
            ("cohens_d", "f8"),
            ("p_value", "f8"),
        ],
    )

    assert visualization.select_figure_smoothings(effects) == (2,)
