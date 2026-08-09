from __future__ import annotations

from pathlib import Path

import nibabel as nib
import numpy as np


def _data_arrays(path: Path) -> list[np.ndarray]:
    image = nib.load(str(Path(path)))
    arrays = [np.asarray(data_array.data) for data_array in image.darrays]
    if not arrays:
        raise ValueError(f"GIFTI contains no data arrays: {path}")
    return arrays


def gifti_logical_data(path: Path) -> np.ndarray:
    """Return a GIFTI's logical array without guessing an orientation."""
    arrays = _data_arrays(Path(path))
    if len(arrays) == 1:
        return arrays[0]
    if all(array.ndim == 1 for array in arrays):
        return np.column_stack(arrays)
    raise ValueError(f"GIFTI has ambiguous logical arrays: {path}")


def gifti_logical_shape_dtype(path: Path) -> tuple[tuple[int, ...], np.dtype]:
    data = gifti_logical_data(Path(path))
    return tuple(int(value) for value in data.shape), np.dtype(data.dtype)


def validate_gifti_logical_contract(
    path: Path,
    *,
    expected_shape: tuple[int, ...],
    expected_dtype: np.dtype | type,
    context: str | None = None,
) -> np.ndarray:
    data = gifti_logical_data(Path(path))
    expected = np.dtype(expected_dtype)
    label = context or str(path)
    if tuple(int(value) for value in data.shape) != tuple(expected_shape):
        raise ValueError(
            f"GIFTI logical shape mismatch for {label}: "
            f"{data.shape} != {expected_shape}"
        )
    if np.dtype(data.dtype) != expected:
        raise ValueError(
            f"GIFTI logical dtype mismatch for {label}: "
            f"{data.dtype} != {expected}"
        )
    return np.asarray(data)


def load_gifti_timeseries(
    path: Path,
    *,
    expected_vertices: int,
    expected_timepoints: int | None = None,
    expected_dtype: np.dtype | type = np.float32,
) -> np.ndarray:
    arrays = _data_arrays(Path(path))
    if any(array.ndim != 1 for array in arrays):
        raise ValueError(f"Expected one-dimensional GIFTI time-series arrays: {path}")
    data = np.column_stack(arrays) if len(arrays) > 1 else arrays[0][:, None]
    timepoints = int(data.shape[1])
    if expected_timepoints is not None and timepoints != int(expected_timepoints):
        raise ValueError(
            f"GIFTI timepoint mismatch for {path}: "
            f"{timepoints} != {int(expected_timepoints)}"
        )
    return validate_gifti_logical_contract(
        path,
        expected_shape=(int(expected_vertices), timepoints),
        expected_dtype=expected_dtype,
        context=str(path),
    )


def validate_gifti_vector(
    path: Path,
    *,
    expected_length: int,
    expected_dtype: np.dtype | type,
    context: str | None = None,
) -> np.ndarray:
    return validate_gifti_logical_contract(
        path,
        expected_shape=(int(expected_length),),
        expected_dtype=expected_dtype,
        context=context,
    )


def validate_gifti_pointset(
    path: Path,
    *,
    expected_vertices: int | None = None,
    expected_dtype: np.dtype | type = np.float32,
) -> np.ndarray:
    image = nib.load(str(Path(path)))
    points = np.asarray(image.agg_data("pointset"))
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(f"GIFTI pointset shape mismatch for {path}: {points.shape}")
    if expected_vertices is not None and int(points.shape[0]) != int(expected_vertices):
        raise ValueError(
            f"GIFTI pointset vertex mismatch for {path}: "
            f"{points.shape[0]} != {int(expected_vertices)}"
        )
    expected = np.dtype(expected_dtype)
    if np.dtype(points.dtype) != expected:
        raise ValueError(
            f"GIFTI pointset dtype mismatch for {path}: "
            f"{points.dtype} != {expected}"
        )
    return points
