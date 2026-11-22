#!/usr/bin/env python3
"""Unfolded hippocampal panel: exact spec-declared asset selection and grid validation."""
from __future__ import annotations

import sys
from pathlib import Path

import nibabel as nib
import numpy as np
import pytest
from nibabel.gifti import GiftiDataArray, GiftiImage, GiftiLabel, GiftiLabelTable

REPO_ROOT = Path(__file__).resolve().parents[2]
CORE_DIR = REPO_ROOT / "scripts" / "core_workflow"
if str(CORE_DIR) not in sys.path:
    sys.path.insert(0, str(CORE_DIR))

from lib.surface_utils import (  # noqa: E402
    _draw_unfolded_map,
    cluster_ids_head_to_tail,
    resolve_unfolded_assets,
    unfolded_anterior_coordinate,
    unfolded_grid,
)

N_COLS, N_ROWS = 4, 6
SUB = "sub-01"


def _grid_coords(hemi: str) -> np.ndarray:
    xs = np.arange(N_COLS, dtype=np.float32) * 1.25 + (181.25 if hemi == "L" else -200.0)
    ys = -np.arange(N_ROWS, dtype=np.float32) * 1.25
    xx, yy = np.meshgrid(xs, ys)
    return np.c_[xx.ravel(), yy.ravel(), np.full(xx.size, 1.2, dtype=np.float32)].astype(np.float32)


def _faces() -> np.ndarray:
    faces = []
    for r in range(N_ROWS - 1):
        for c in range(N_COLS - 1):
            v = r * N_COLS + c
            faces += [[v, v + 1, v + N_COLS], [v + 1, v + N_COLS + 1, v + N_COLS]]
    return np.asarray(faces, dtype=np.int32)


def _save_surface(path: Path, coords: np.ndarray) -> Path:
    img = GiftiImage(darrays=[
        GiftiDataArray(coords.astype(np.float32), intent="NIFTI_INTENT_POINTSET"),
        GiftiDataArray(_faces(), intent="NIFTI_INTENT_TRIANGLE"),
    ])
    nib.save(img, str(path))
    return path


def _save_labels(path: Path, values: np.ndarray, names: dict[int, tuple[str, tuple[float, ...]]]) -> Path:
    table = GiftiLabelTable()
    for key, (name, rgba) in names.items():
        lab = GiftiLabel(key=key, red=rgba[0], green=rgba[1], blue=rgba[2], alpha=rgba[3])
        lab.label = name
        table.labels.append(lab)
    nib.save(GiftiImage(darrays=[GiftiDataArray(values.astype(np.int32), intent="NIFTI_INTENT_LABEL",
                                                datatype="NIFTI_TYPE_INT32")], labeltable=table), str(path))
    return path


def _write_spec(surf_dir: Path, entries: list[tuple[str, str, str]]) -> Path:
    body = "".join(
        f'   <DataFile Structure="{st}" DataFileType="{kind}" Selected="true">\n      {name}\n   </DataFile>\n'
        for st, kind, name in entries
    )
    spec = surf_dir / f"{SUB}_den-512_label-hipp_surfaces.spec"
    spec.write_text(f'<?xml version="1.0" encoding="UTF-8"?>\n<CaretSpecFile Version="1.0">\n{body}</CaretSpecFile>\n')
    return spec


def _hippunfold_tree(tmp_path: Path, *, extra_entries: list[tuple[str, str, str]] = ()) -> Path:
    surf_dir = tmp_path / "surf"
    metric_dir = tmp_path / "metric"
    surf_dir.mkdir()
    metric_dir.mkdir()
    entries = []
    subfields = {1: ("Subiculum", (0, 0, 1, 1)), 2: ("CA1", (0.5, 0.9, 1, 1))}
    for hemi, structure in (("L", "HippocampusLeft"), ("R", "HippocampusRight")):
        unfold = f"{SUB}_hemi-{hemi}_space-unfold_den-512_label-hipp_midthickness.surf.gii"
        _save_surface(surf_dir / unfold, _grid_coords(hemi))
        label = f"{SUB}_hemi-{hemi}_den-512_label-hipp_atlas-multihist7sgi_subfields.label.gii"
        values = np.where(np.arange(N_ROWS * N_COLS) % N_COLS < 2, 2, 1)
        _save_labels(metric_dir / label, values, subfields)
        entries += [(structure, "SURFACE", unfold), (structure, "LABEL", f"../metric/{label}")]
    return _write_spec(surf_dir, entries + list(extra_entries))


def test_resolve_unfolded_assets_selects_spec_declared_files(tmp_path):
    spec = _hippunfold_tree(tmp_path)
    assets = resolve_unfolded_assets(spec, density="512")
    assert assets.subfield_atlas == "multihist7sgi"
    assert assets.unfold_surface["L"].name == f"{SUB}_hemi-L_space-unfold_den-512_label-hipp_midthickness.surf.gii"
    assert assets.subfield_labels["R"].parent.name == "metric"
    assert assets.provenance()["density"] == "512"


def test_resolve_unfolded_assets_rejects_ambiguous_or_missing_entries(tmp_path):
    dup = ("HippocampusLeft", "LABEL", f"../metric/{SUB}_hemi-L_den-512_label-hipp_atlas-other_subfields.label.gii")
    spec = _hippunfold_tree(tmp_path, extra_entries=[dup])
    with pytest.raises(ValueError, match="exactly one hemi-L den-512 hipp subfield LABEL"):
        resolve_unfolded_assets(spec, density="512")
    with pytest.raises(ValueError, match="exactly one hemi-L den-2k"):
        resolve_unfolded_assets(spec, density="2k")


def test_unfolded_grid_orders_anterior_first_and_validates_orientation(tmp_path):
    unfold = _save_surface(tmp_path / "unfold.surf.gii", _grid_coords("L"))
    folded_coords = _grid_coords("L").copy()
    folded_coords[:, 2] = np.linspace(0, 3, folded_coords.shape[0])
    folded = _save_surface(tmp_path / "folded.surf.gii", folded_coords)
    row, col, shape = unfolded_grid(unfold, folded)
    assert shape == (N_ROWS, N_COLS)
    assert row[0] == 0 and col[0] == 0  # y = 0 is the most anterior row
    assert row[-1] == N_ROWS - 1

    flipped = folded_coords.copy()
    flipped[:, 1] = -flipped[:, 1]
    with pytest.raises(ValueError, match="anterior axis"):
        unfolded_grid(unfold, _save_surface(tmp_path / "flipped.surf.gii", flipped))

    holes = _grid_coords("L")
    holes[0, 0] += 0.3
    with pytest.raises(ValueError, match="regular grid"):
        unfolded_grid(_save_surface(tmp_path / "holes.surf.gii", holes), folded)


def test_draw_unfolded_map_rejects_label_vertex_mismatch(tmp_path):
    import matplotlib.pyplot as plt

    spec = _hippunfold_tree(tmp_path)
    assets = resolve_unfolded_assets(spec, density="512")
    folded = _save_surface(tmp_path / "folded.surf.gii", _grid_coords("L"))
    bad = _save_labels(tmp_path / "func.label.gii", np.ones(N_ROWS * N_COLS - 1, dtype=int), {1: ("DMN", (1, 0, 0, 1))})
    fig, ax = plt.subplots()
    try:
        with pytest.raises(ValueError, match="functional labels vertex mismatch"):
            _draw_unfolded_map(ax, hemi="L", labels_path=bad, folded_surface=folded, assets=assets)
        good = _save_labels(tmp_path / "func_ok.label.gii", np.ones(N_ROWS * N_COLS, dtype=int), {1: ("DMN", (1, 0, 0, 1))})
        _draw_unfolded_map(ax, hemi="L", labels_path=good, folded_surface=folded, assets=assets)
        assert len(ax.collections) == 2  # halo + coloured subfield boundaries
    finally:
        plt.close(fig)


def test_unfolded_anterior_coordinate_is_the_unfolded_y_axis(tmp_path):
    unfold = _save_surface(tmp_path / "unfold.surf.gii", _grid_coords("L"))
    folded_coords = _grid_coords("L") * np.array([1.0, 2.0, 1.0], dtype=np.float32)
    folded = _save_surface(tmp_path / "folded.surf.gii", folded_coords)
    np.testing.assert_allclose(unfolded_anterior_coordinate(unfold, folded), _grid_coords("L")[:, 1])


def test_unfolded_anterior_coordinate_rejects_a_sheet_pointing_posterior(tmp_path):
    unfold = _save_surface(tmp_path / "unfold.surf.gii", _grid_coords("L"))
    folded = _save_surface(tmp_path / "folded.surf.gii", _grid_coords("L") * np.array([1.0, -1.0, 1.0], dtype=np.float32))
    with pytest.raises(ValueError, match="anterior axis"):
        unfolded_anterior_coordinate(unfold, folded)


def test_head_cluster_is_c1_even_when_a_thin_strip_reaches_further_on_average():
    # Rows run head (y=0) to tail (y=-6.25). Cluster 2 fills the head rows; cluster 1 is a thin
    # strip along one column from head to tail; cluster 3 fills the tail.
    coordinate = _grid_coords("L")[:, 1]
    rows = np.repeat(np.arange(N_ROWS), N_COLS)
    cols = np.tile(np.arange(N_COLS), N_ROWS)
    labels = np.where(rows >= 4, 3, 2)
    labels[cols == 0] = 1
    assert cluster_ids_head_to_tail(labels, coordinate) == [2, 1, 3]
    assert cluster_ids_head_to_tail(np.where(labels == 0, 0, labels), coordinate)[0] == 2
