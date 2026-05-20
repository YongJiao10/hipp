#!/usr/bin/env python3
from __future__ import annotations

import nibabel as nib
import numpy as np


def extract_structure_data(
    dt_axis: nib.cifti2.cifti2_axes.BrainModelAxis,
    dt_data_t: np.ndarray,
    structure_name: str,
) -> tuple[np.ndarray, np.ndarray]:
    for name, slc, subaxis in dt_axis.iter_structures():
        if name == structure_name or name.endswith(structure_name):
            if getattr(subaxis, "vertex", None) is None:
                raise RuntimeError(f"Structure {structure_name} does not expose vertices")
            return np.asarray(subaxis.vertex, dtype=np.int32), dt_data_t[slc, :]
    raise RuntimeError(f"Could not find structure {structure_name} in dtseries")


def load_required_mask(path: str, expected_length: int, label: str) -> np.ndarray:
    mask = np.load(path).astype(bool)
    if mask.ndim != 1 or mask.shape[0] != expected_length:
        raise ValueError(f"{label} mask shape mismatch: expected ({expected_length},), got {mask.shape}")
    return mask
