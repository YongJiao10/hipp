#!/usr/bin/env python3
from __future__ import annotations

import importlib.util
import math
import sys
from pathlib import Path

import nibabel as nib
import numpy as np
from nibabel.gifti import GiftiDataArray, GiftiImage


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _load_numbered_module(script_name: str, alias: str):
    module_path = REPO_ROOT / "scripts" / "core_workflow" / script_name
    spec = importlib.util.spec_from_file_location(alias, module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load module: {module_path}")
    module = importlib.util.module_from_spec(spec)
    core_dir = (REPO_ROOT / "scripts" / "core_workflow").resolve()
    if str(core_dir) not in sys.path:
        sys.path.insert(0, str(core_dir))
    sys.modules[alias] = module
    spec.loader.exec_module(module)
    return module


mod = _load_numbered_module("06_08_render_hcp_subfield_composition.py", "core_06_08_hcp_subfields")


def _write_surface(path: Path) -> None:
    coords = np.asarray(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [1.0, 1.0, 0.0],
        ],
        dtype=np.float32,
    )
    faces = np.asarray([[0, 1, 2], [1, 3, 2]], dtype=np.int32)
    img = GiftiImage(
        darrays=[
            GiftiDataArray(data=coords, intent="NIFTI_INTENT_POINTSET"),
            GiftiDataArray(data=faces, intent="NIFTI_INTENT_TRIANGLE"),
        ]
    )
    nib.save(img, str(path))


def test_vertex_area_weights_split_triangle_area_across_vertices(tmp_path: Path) -> None:
    surface = tmp_path / "midthickness.surf.gii"
    _write_surface(surface)

    got = mod.vertex_area_weights(surface)

    assert got.shape == (4,), got
    assert np.allclose(got, [1.0 / 6.0, 1.0 / 3.0, 1.0 / 3.0, 1.0 / 6.0])
    assert math.isclose(float(got.sum()), 1.0, rel_tol=1e-7, abs_tol=1e-7)


def test_subject_cluster_subfield_proportions_are_area_weighted_and_exclude_non_anatomic_labels() -> None:
    cluster_labels = np.asarray([1, 1, 2, 2], dtype=np.int32)
    support_mask = np.asarray([1, 1, 1, 1], dtype=bool)
    subfields = np.asarray([2, 6, 8, 0], dtype=np.int32)
    vertex_weights = np.asarray([1.0, 3.0, 5.0, 7.0], dtype=np.float64)

    rows, qc = mod.subject_cluster_subfield_rows(
        subject="100610",
        smoothing="2mm",
        hemi="L",
        cluster_labels=cluster_labels,
        support_mask=support_mask,
        subfield_labels=subfields,
        vertex_weights=vertex_weights,
    )

    by_key = {(row["cluster"], row["subfield"]): row for row in rows}
    assert math.isclose(by_key[(1, "CA1")]["proportion"], 0.25)
    assert math.isclose(by_key[(1, "DG")]["proportion"], 0.75)
    assert math.isclose(by_key[(2, "Sub")]["proportion"], 0.0)
    assert qc["excluded_cyst_area"] == 5.0
    assert qc["excluded_zero_area"] == 7.0
    assert qc["cluster_valid_area"][1] == 4.0
    assert qc["cluster_valid_area"][2] == 0.0


def test_summary_averages_subject_normalized_proportions_and_reports_sem() -> None:
    rows = [
        {"smoothing": "2mm", "hemi": "L", "cluster": 1, "cluster_name": "Anterior", "subfield": "CA1", "proportion": 0.25},
        {"smoothing": "2mm", "hemi": "L", "cluster": 1, "cluster_name": "Anterior", "subfield": "CA1", "proportion": 0.75},
        {"smoothing": "2mm", "hemi": "L", "cluster": 1, "cluster_name": "Anterior", "subfield": "DG", "proportion": 0.75},
        {"smoothing": "2mm", "hemi": "L", "cluster": 1, "cluster_name": "Anterior", "subfield": "DG", "proportion": 0.25},
    ]

    summary = mod.summarize_subject_rows(rows)

    by_subfield = {row["subfield"]: row for row in summary}
    assert math.isclose(by_subfield["CA1"]["mean_proportion"], 0.5)
    assert math.isclose(by_subfield["DG"]["mean_proportion"], 0.5)
    assert math.isclose(by_subfield["CA1"]["sem_proportion"], 0.25)
    assert by_subfield["CA1"]["n_subjects"] == 2


def test_default_group_root_uses_formal_routea_branch() -> None:
    args = mod.parse_args([])
    assert "routea_full_k3_fisherz" in str(args.group_root)
    assert "weighted_fisherz" not in str(args.group_root)
    assert args.smoothings == ["2mm", "4mm"]
    assert args.hemis == ["L", "R"]

def test_active_subfield_indices_omit_zero_contribution_subfields_from_plot() -> None:
    rows = [
        {"subfield_index": 1, "mean_proportion": 0.2},
        {"subfield_index": 2, "mean_proportion": 0.8},
        {"subfield_index": 6, "mean_proportion": 0.0},
        {"subfield_index": 7, "mean_proportion": 0.0},
    ]

    assert mod.active_subfield_indices(rows) == [1, 2]
