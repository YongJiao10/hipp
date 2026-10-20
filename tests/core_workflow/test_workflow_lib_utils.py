#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

import nibabel as nib
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
CORE_DIR = (REPO_ROOT / "scripts" / "core_workflow").resolve()
if str(CORE_DIR) not in sys.path:
    sys.path.insert(0, str(CORE_DIR))

from lib.fc_math import compact_active_vertices, corrcoef_rows, fisher_z_transform_fc
from lib.gifti_labels import make_cluster_label_gifti, make_named_label_gifti
from lib.io_utils import load_json, load_subject_ids, normalize_subject_id, write_tsv


def test_io_utils_json_tsv_and_subject_manifest() -> None:
    with tempfile.TemporaryDirectory(prefix="cw_io_utils_") as td:
        tmp = Path(td)
        j = tmp / "a.json"
        payload = {"a": 1, "b": [2, 3]}
        j.write_text(json.dumps(payload), encoding="utf-8")
        got = load_json(j)
        assert got == payload

        tsv = tmp / "rows.tsv"
        rows = [{"k": "x", "v": 1}, {"k": "y", "v": 2}]
        write_tsv(tsv, rows, ["k", "v"])
        raw = tsv.read_text(encoding="utf-8")
        assert "k\tv" in raw
        assert "x\t1" in raw

        manifest = tmp / "subjects.txt"
        manifest.write_text("sub-100610\n 100307\n\nsub-102311\n", encoding="utf-8")
        subjects = load_subject_ids(manifest)
        assert subjects == ["100610", "100307", "102311"]
        assert normalize_subject_id("sub-123") == "123"


def test_fc_math_helpers() -> None:
    x = np.array([[0.0, 0.5, -0.5], [np.nan, 0.2, -0.2]], dtype=np.float32)
    z = fisher_z_transform_fc(x)
    assert z.shape == x.shape
    assert np.isfinite(z[0, 1])
    assert np.isnan(z[1, 0])

    a = np.array([[1.0, 2.0, 3.0], [3.0, 2.0, 1.0]], dtype=np.float32)
    b = np.array([[1.0, 2.0, 3.0], [1.0, 1.0, 1.0]], dtype=np.float32)
    corr = corrcoef_rows(a, b)
    assert corr.shape == (2, 2)
    assert np.all(np.isfinite(corr[:, 0]))
    assert np.isnan(corr[:, 1]).all()
    assert corr[0, 0] > 0.99

    values = np.arange(12, dtype=np.float32).reshape(4, 3)
    support = np.array([True, False, True, False], dtype=bool)
    compact = compact_active_vertices(values, support)
    assert compact.shape == (2, 3)
    assert np.array_equal(compact[:, 0], np.array([0.0, 6.0], dtype=np.float32))


def test_gifti_label_builders() -> None:
    labels = np.array([0, 1, 2, 1], dtype=np.int32)
    named = make_named_label_gifti(
        labels,
        [
            (0, "Null", (120, 120, 120, 255)),
            (1, "NetA", (200, 10, 10, 255)),
            (2, "NetB", (10, 10, 200, 255)),
        ],
    )
    assert isinstance(named, nib.gifti.GiftiImage)
    assert int(named.darrays[0].data.shape[0]) == 4
    assert len(named.labeltable.labels) == 3

    cluster = make_cluster_label_gifti(labels, 2, [(255, 0, 0, 255), (0, 255, 0, 255), (0, 0, 255, 255)])
    assert isinstance(cluster, nib.gifti.GiftiImage)
    assert len(cluster.labeltable.labels) == 3


def main() -> int:
    test_io_utils_json_tsv_and_subject_manifest()
    test_fc_math_helpers()
    test_gifti_label_builders()
    print("ok: core workflow lib utility tests passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


def test_hippomaps_screenshot_adapter_handles_supported_plotter_results(tmp_path):
    from PIL import Image
    from lib import hippomaps_adapter
    for mode in ("none", "path", "figure", "screenshot"):
        output = tmp_path / f"{mode}.png"
        def plotter(data, **kwargs):
            assert data.shape == (2, 1)
            assert kwargs["cmap"] == "viridis"
            target = Path(kwargs["filename"])
            if mode in {"none", "path"}:
                Image.new("RGB", (4, 3), "red").save(target)
                return str(target) if mode == "path" else None
            if mode == "figure":
                class Figure:
                    def savefig(self, path, **options):
                        Image.new("RGB", (4, 3), "green").save(path)
                return Figure()
            class Screenshot:
                def screenshot(self, path):
                    Image.new("RGB", (4, 3), "blue").save(path)
            return Screenshot()
        result = hippomaps_adapter.capture_hippomaps_plot(
            plotter, np.ones((2, 1)), output_png=output, cmap="viridis")
        assert result == output
        assert Image.open(output).size == (4, 3)


def test_hippomaps_screenshot_adapter_rejects_missing_render(tmp_path):
    import pytest
    from lib import hippomaps_adapter
    with pytest.raises(TypeError, match="did not create"):
        hippomaps_adapter.capture_hippomaps_plot(
            lambda data, **kwargs: None, np.ones((2, 1)), output_png=tmp_path / "missing.png")
