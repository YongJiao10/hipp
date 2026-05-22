from __future__ import annotations

import numpy as np
from nibabel.gifti import GiftiDataArray, GiftiImage, GiftiLabel, GiftiLabelTable


def make_cluster_label_gifti(labels: np.ndarray, k: int, colors: list[tuple[int, int, int, int]]) -> GiftiImage:
    tab = GiftiLabelTable()
    null = GiftiLabel(key=0, red=0.62, green=0.62, blue=0.62, alpha=1.0)
    null.label = "Null"
    tab.labels.append(null)
    for idx in range(1, int(k) + 1):
        rgba = colors[idx % len(colors)]
        g = GiftiLabel(
            key=int(idx),
            red=float(rgba[0]) / 255.0,
            green=float(rgba[1]) / 255.0,
            blue=float(rgba[2]) / 255.0,
            alpha=float(rgba[3]) / 255.0,
        )
        g.label = f"Cluster{idx}"
        tab.labels.append(g)
    da = GiftiDataArray(data=labels.astype(np.int32), intent="NIFTI_INTENT_LABEL", datatype="NIFTI_TYPE_INT32")
    return GiftiImage(darrays=[da], labeltable=tab)


def make_named_label_gifti(
    labels: np.ndarray,
    label_table_items: list[tuple[int, str, tuple[int, int, int, int]]],
) -> GiftiImage:
    tab = GiftiLabelTable()
    for key, name, rgba in label_table_items:
        g = GiftiLabel(
            key=int(key),
            red=float(rgba[0]) / 255.0,
            green=float(rgba[1]) / 255.0,
            blue=float(rgba[2]) / 255.0,
            alpha=float(rgba[3]) / 255.0,
        )
        g.label = str(name)
        tab.labels.append(g)
    da = GiftiDataArray(data=labels.astype(np.int32), intent="NIFTI_INTENT_LABEL", datatype="NIFTI_TYPE_INT32")
    return GiftiImage(darrays=[da], labeltable=tab)
