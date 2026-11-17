from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from PIL import Image

from scripts.visualize.cortex_fc.fc_visualization import compute_network_fc
from scripts.visualize.cortex_fc.render_fc import build_arg_parser, load_fc_matrix, main


def test_timeseries_input_computes_fisher_z_and_excludes_diagonal(tmp_path):
    timeseries = np.array([[0, 1, 2, 3], [0, 2, 1, 3], [3, 2, 1, 0]], dtype=float)
    path = tmp_path / "timeseries.npy"
    np.save(path, timeseries)

    matrix = load_fc_matrix(fc_matrix=None, timeseries=path)

    assert np.isnan(matrix.diagonal()).all()
    assert matrix[0, 1] == pytest.approx(np.arctanh(0.8))
    assert matrix[0, 2] == pytest.approx(np.arctanh(-0.9999))


def test_precomputed_matrix_preserves_off_diagonal_values(tmp_path):
    path = tmp_path / "fc.npy"
    np.save(path, [[1, -0.2], [-0.2, 1]])

    matrix = load_fc_matrix(fc_matrix=path, timeseries=None)

    assert np.isnan(matrix.diagonal()).all()
    assert matrix[0, 1] == -0.2


def test_network_means_use_roi_positions_not_dataframe_index_labels():
    matrix = np.array([[np.nan, 0.5, 0.1, 0.1], [0.5, np.nan, 0.3, 0.3],
                       [0.1, 0.3, np.nan, 0.7], [0.1, 0.3, 0.7, np.nan]])
    parcels = pd.DataFrame({"canonical7": ["Visual", "Visual", "Default", "Default"]},
                           index=[10, 20, 30, 40])

    network_fc, order = compute_network_fc(matrix, parcels)

    assert network_fc[order.index("Visual"), order.index("Visual")] == pytest.approx(0.5)
    assert network_fc[order.index("Visual"), order.index("Default")] == pytest.approx(0.2)


@pytest.mark.parametrize("net_type,view", [("7net", "combined"), ("17net", "heatmap"), ("7net", "chord")])
def test_cli_renders_explicit_files_without_dataset_configuration(tmp_path, monkeypatch, net_type, view):
    rows = []
    mapping = [("VisualA", "Visual"), ("SomMotA", "Somatomotor"),
               ("DorsAttnA", "DorsalAttention"), ("SalVenAttnA", "VentralAttention"),
               ("ContA", "Control"), ("Language", "Language"), ("DefaultA", "Default")]
    for family, canonical in mapping:
        for hemisphere in ("L", "R"):
            rows.append({"parcel_id": len(rows) + 1, "family17": family,
                         "canonical7": canonical, "hemisphere": hemisphere})
    parcels = tmp_path / "parcels.tsv"
    pd.DataFrame(rows).to_csv(parcels, sep="\t", index=False)
    fc = tmp_path / "fc.npy"
    values = np.eye(len(rows)) + np.full((len(rows), len(rows)), 0.2)
    np.save(fc, values)
    monkeypatch.chdir(tmp_path)
    output = tmp_path / "fc.png"

    assert main(["--fc-matrix", str(fc), "--parcels", str(parcels),
                 "--net-type", net_type, "--view", view, "--output", "fc.png"]) == 0

    with Image.open(output) as image:
        assert image.format == "PNG"
        assert min(image.size) > 300


def test_cli_requires_one_explicit_input():
    with pytest.raises(SystemExit):
        build_arg_parser().parse_args(["--parcels", "parcels.tsv", "--output", "fc.png"])
