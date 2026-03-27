#!/opt/miniconda3/envs/py314/bin/python
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

import nibabel as nib
import numpy as np
from scipy.ndimage import map_coordinates


def run_real_wb_command(wb_app: str, wb_args: list[str]) -> int:
    cmd = [wb_app, *wb_args]
    if sys.platform == "darwin" and os.uname().machine == "arm64":
        cmd = ["arch", "-x86_64", *cmd]
    proc = subprocess.run(cmd)
    return proc.returncode


def looks_like_scalar_shape(path: Path) -> bool:
    return path.name.endswith(".shape.gii")


def load_surface_points(path: Path) -> np.ndarray:
    return np.asarray(nib.load(str(path)).agg_data("NIFTI_INTENT_POINTSET"), dtype=np.float64)


def sample_volume_at_surface(volume_path: Path, surface_points_world: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    volume_img = nib.load(str(volume_path))
    volume = np.asanyarray(volume_img.dataobj, dtype=np.float32)
    vox = nib.affines.apply_affine(np.linalg.inv(volume_img.affine), surface_points_world)
    coords = np.vstack([vox[:, 0], vox[:, 1], vox[:, 2]])
    values = map_coordinates(volume, coords, order=1, mode="constant", cval=0.0)
    return values.astype(np.float32), vox


def replace_gifti_data(output_path: Path, values: np.ndarray) -> None:
    img = nib.load(str(output_path))
    darray = img.darrays[0]
    new_darray = nib.gifti.GiftiDataArray(
        data=values,
        intent=darray.intent,
        datatype="NIFTI_TYPE_FLOAT32",
        meta=darray.meta,
    )
    out_img = nib.gifti.GiftiImage(meta=img.meta, labeltable=img.labeltable)
    out_img.add_gifti_data_array(new_darray)
    nib.save(out_img, str(output_path))


def maybe_apply_negative_x_fallback(volume_path: Path, surface_path: Path, output_path: Path) -> bool:
    if not looks_like_scalar_shape(output_path):
        return False

    out_img = nib.load(str(output_path))
    out_values = np.asarray(out_img.agg_data(), dtype=np.float32)
    if np.count_nonzero(out_values) != 0:
        return False

    volume_img = nib.load(str(volume_path))
    if volume_img.affine[0, 0] >= 0:
        return False

    surface_points = load_surface_points(surface_path)

    direct_values, _ = sample_volume_at_surface(volume_path, surface_points)
    if np.count_nonzero(direct_values) != 0:
        return False

    shifted_points = surface_points.copy()
    shifted_points[:, 0] -= volume_img.shape[0] * abs(float(volume_img.affine[0, 0]))
    shifted_values, shifted_vox = sample_volume_at_surface(volume_path, shifted_points)

    if np.count_nonzero(shifted_values) == 0:
        return False

    inside = (
        (shifted_vox[:, 0] >= 0)
        & (shifted_vox[:, 0] <= volume_img.shape[0] - 1)
        & (shifted_vox[:, 1] >= 0)
        & (shifted_vox[:, 1] <= volume_img.shape[1] - 1)
        & (shifted_vox[:, 2] >= 0)
        & (shifted_vox[:, 2] <= volume_img.shape[2] - 1)
    )
    if inside.mean() < 0.95:
        return False

    replace_gifti_data(output_path, shifted_values)
    print(
        f"[wb_command fallback] Replaced zero-mapped output with negative-x corrected sampling: {output_path}",
        file=sys.stderr,
    )
    return True


def parse_args() -> tuple[str, list[str]]:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--wb-app", required=True)
    ns, remaining = parser.parse_known_args()
    return ns.wb_app, remaining


def main() -> int:
    wb_app, wb_args = parse_args()
    if not wb_args:
        return run_real_wb_command(wb_app, wb_args)

    rc = run_real_wb_command(wb_app, wb_args)
    if rc != 0:
        return rc

    if wb_args[0] != "-volume-to-surface-mapping" or len(wb_args) < 4:
        return 0

    volume_path = Path(wb_args[1])
    surface_path = Path(wb_args[2])
    output_path = Path(wb_args[3])

    try:
        maybe_apply_negative_x_fallback(volume_path, surface_path, output_path)
    except Exception as exc:
        print(f"[wb_command fallback] Validation skipped after error: {exc}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
