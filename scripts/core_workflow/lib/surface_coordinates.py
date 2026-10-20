"""Native hippocampal geometry and explicit native-to-BOLD point registration."""
from __future__ import annotations

import csv
from dataclasses import dataclass
import os
from pathlib import Path
import socket
import subprocess
import tempfile

import nibabel as nib
import numpy as np
from scipy.ndimage import map_coordinates
from scipy.spatial import cKDTree

from lib.workbench import run_workbench, workbench_env

REPO_ROOT = Path(__file__).resolve().parents[3]
MNI_SPACE = 'MNI152NLin6Asym'
METRIC_SPACE = 'T1w'


def load_surface_coords(path: Path) -> np.ndarray:
    img = nib.load(str(path))
    coords = None
    for darray in img.darrays:
        if int(darray.intent) == 1008:
            coords = np.asarray(darray.data, dtype=np.float32)
            break
    if coords is None:
        data = img.agg_data()
        coords = np.asarray(data[0] if isinstance(data, tuple) else data, dtype=np.float32)
    if coords.ndim != 2 or coords.shape[1] != 3:
        raise ValueError(f"Expected surface POINTSET coordinates with shape (N, 3): {path} shape={coords.shape}")
    return coords.astype(np.float32, copy=False)


def hipp_surface_path(batch_root: Path, subject: str, hemi: str) -> Path:
    subject = str(subject).replace("sub-", "")
    return (Path(batch_root) / f"sub-{subject}" / "hippunfold" / f"sub-{subject}" / "surf"
            / f"sub-{subject}_hemi-{hemi}_space-T1w_den-512_label-hipp_midthickness.surf.gii")


def query_local_cortex_neighbors(
    *, hipp_coords: np.ndarray, cortex_coords: np.ndarray,
    cortex_valid_mask: np.ndarray, distance_mm: float,
) -> list[np.ndarray]:
    """Query valid cortical vertices in the caller's common physical space."""
    valid = np.asarray(cortex_valid_mask, dtype=bool).reshape(-1)
    if valid.size != int(np.asarray(cortex_coords).shape[0]):
        raise ValueError(f"Cortex valid mask size mismatch: mask={valid.size} coords={np.asarray(cortex_coords).shape[0]}")
    valid_idx = np.flatnonzero(valid)
    if valid_idx.size == 0:
        return [np.asarray([], dtype=np.int64) for _ in range(int(np.asarray(hipp_coords).shape[0]))]
    tree = cKDTree(np.asarray(cortex_coords, dtype=np.float32)[valid_idx, :])
    local = tree.query_ball_point(np.asarray(hipp_coords, dtype=np.float32), r=float(distance_mm))
    return [valid_idx[np.asarray(row, dtype=np.int64)] for row in local]


@dataclass(frozen=True)
class SurfaceRegistration:
    method: str
    image_transform: Path
    source_image: Path | None = None
    inverse_warp: Path | None = None
    target_space: str = MNI_SPACE


def jianglab_root() -> Path:
    """Select the system's prefix, without filesystem-existence fallback."""
    system = os.environ.get('SYSTEM_NAME', socket.gethostname()).lower()
    return Path('/Shared/jianglab' if 'argon' in system else '/home/yojiao/LSS/jianglab')


def dataset_registration(dataset: str, subject: str, *, root: Path | None = None) -> SurfaceRegistration:
    root = (Path(root) if root is not None else jianglab_root()) / '3_Data_Working'
    subject = str(subject).removeprefix('sub-')
    sub = f'sub-{subject}'
    if dataset == 'hcp':
        source = root / f'HCP_7T_Hippocampus/subjects/{subject}/source'
        return SurfaceRegistration('fnirt', source/'acpc_dc2standard.nii.gz',
                                   source/'T1w_acpc_dc_restore.nii.gz')
    if dataset in {'banda', 'pdc'}:
        if dataset == 'banda':
            directory = subject + '_MR'
        else:
            mapping = REPO_ROOT/'outputs_pdc/inputs/pdc/manifests/session_mapping.tsv'
            with mapping.open() as stream:
                rows = [row for row in csv.DictReader(stream, delimiter='\t')
                        if row['bids_subject_id'] == subject]
            if len(rows) != 1:
                raise ValueError(f'Expected exactly one PDC session for {subject}: {mapping}')
            directory = rows[0]['original_session_dir']
        xfms = root / f'Depression/{dataset.upper()}/imaging/{directory}/MNINonLinear/xfms'
        return SurfaceRegistration('fnirt-pair', xfms/'acpc_dc2standard.nii.gz',
                                   inverse_warp=xfms/'standard2acpc_dc.nii.gz')
    anat_roots = {
        'stanford': 'fmriprep_processed_CausalConnectome/rest_fmriprep/fmriprep',
        'iowa_jj_tms': 'fmriprep_processed_JJ_TMS/derivatives_surf',
        'iowa7t': 'fmriprep_processed_Iowa7T/derivatives',
    }
    if dataset not in anat_roots:
        raise ValueError(f'No declared surface registration for dataset={dataset!r}')
    # Image registration names describe resampling direction. ANTs point
    # mapping needs the opposite image transform: MNI->T1w maps T1w points to MNI.
    transform = root / anat_roots[dataset] / sub / 'anat' / f'{sub}_from-{MNI_SPACE}_to-T1w_mode-image_xfm.h5'
    return SurfaceRegistration('ants', transform)


def invert_world_displacement(native: np.ndarray, world_warp: Path) -> np.ndarray:
    """Solve x + d(x) = native for a Workbench-converted FNIRT pullback field."""
    image = nib.load(world_warp)
    data = np.asarray(image.dataobj, dtype=np.float64)
    if data.ndim != 4 or data.shape[-1] != 3:
        raise ValueError(f'Expected a three-component world displacement field: {world_warp}')
    inverse_affine = np.linalg.inv(image.affine)

    def displacement(points):
        voxels = nib.affines.apply_affine(inverse_affine, points)
        return np.column_stack([map_coordinates(data[..., axis], voxels.T, order=1,
                                                mode='constant', cval=np.nan) for axis in range(3)])

    points = np.asarray(native, dtype=np.float64).copy()
    for _ in range(20):
        residual = points + displacement(points) - native
        if not np.isfinite(residual).all():
            raise ValueError(f'Native-to-MNI point inversion left the warp grid: {world_warp}')
        if np.linalg.norm(residual, axis=1).max() < .001:
            return points
        jacobian = np.stack([displacement(points + np.eye(3)[axis] * .5)
                             - displacement(points - np.eye(3)[axis] * .5)
                             for axis in range(3)], axis=-1) + np.eye(3)
        if not np.isfinite(jacobian).all():
            raise ValueError(f'Native-to-MNI derivative left the warp grid: {world_warp}')
        points -= np.linalg.solve(jacobian, residual[..., None])[..., 0]
    raise ValueError(f'Native-to-MNI point inversion did not converge within .001 mm: {world_warp}')


def apply_world_displacement(points: np.ndarray, world_warp: Path) -> np.ndarray:
    """Map MNI points to native using the image pullback field directly."""
    image = nib.load(world_warp)
    data = np.asarray(image.dataobj, dtype=np.float64)
    voxels = nib.affines.apply_affine(np.linalg.inv(image.affine), points)
    delta = np.column_stack([map_coordinates(data[..., axis], voxels.T, order=1,
                                             mode='constant', cval=np.nan) for axis in range(3)])
    native = np.asarray(points) + delta
    if not np.isfinite(native).all():
        raise ValueError(f'MNI cortex points fall outside the declared native warp grid: {world_warp}')
    return native


def native_cortex_coordinates(points: np.ndarray, *, subject: str, space: str) -> np.ndarray:
    """HCP default cortex is in MNI; caller-selected native cortex needs no warp."""
    if space == METRIC_SPACE:
        return points
    if space != MNI_SPACE:
        raise ValueError(f'Unsupported cortical physical space: {space}')
    registration = dataset_registration('hcp', subject)
    for path in (registration.image_transform, registration.source_image):
        if not path.is_file():
            raise FileNotFoundError(f'Missing HCP cortex coordinate transform: {path}; native cortical inputs require --cortex-coordinate-space T1w')
    scratch = REPO_ROOT/'tmp'
    scratch.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='cortex_native_', dir=scratch) as temporary:
        world = Path(temporary)/'mni_to_native_world.nii.gz'
        run_workbench(['-convert-warpfield', '-from-fnirt', str(registration.image_transform),
                       str(registration.source_image), '-to-world', str(world)])
        return apply_world_displacement(points, world)


def ants_native_to_mni(points: np.ndarray, transform: Path, *, workdir: Path) -> np.ndarray:
    lps = np.asarray(points, dtype=np.float64).copy()
    lps[:, :2] *= -1
    source, destination = workdir/'native_lps.csv', workdir/'mni_lps.csv'
    np.savetxt(source, lps, delimiter=',', header='x,y,z', comments='')
    executable = Path(os.environ.get('CORE_WORKFLOW_ANTS_POINTS',
                         str(REPO_ROOT/'.cache/conda_envs/hippo2/bin/antsApplyTransformsToPoints')))
    if not executable.is_file():
        raise FileNotFoundError(f'ANTs points executable is missing: {executable}; set CORE_WORKFLOW_ANTS_POINTS')
    subprocess.run([str(executable), '-d', '3', '-i', str(source), '-o', str(destination),
                    '-t', str(transform)], check=True, capture_output=True, text=True, env=workbench_env())
    rows = np.genfromtxt(destination, delimiter=',', names=True)
    mapped = np.column_stack([np.atleast_1d(rows[axis]) for axis in ('x', 'y', 'z')])
    mapped[:, :2] *= -1
    return mapped


def registered_sampling_surfaces(native_surfaces: dict[str, Path], *,
                                 registration: SurfaceRegistration, outdir: Path,
                                 wb_command: Path | None = None) -> dict[str, Path]:
    inputs = [registration.image_transform]
    if registration.method == 'fnirt':
        if registration.source_image is None:
            raise ValueError('FNIRT conversion requires its native source_image')
        inputs.append(registration.source_image)
    elif registration.method == 'fnirt-pair':
        if registration.inverse_warp is None:
            raise ValueError('FNIRT surface mapping requires the declared inverse_warp')
        inputs.append(registration.inverse_warp)
    elif registration.method != 'ants':
        raise ValueError(f'Unknown registration method: {registration.method}')
    for path in inputs:
        if not Path(path).is_file():
            raise FileNotFoundError(f'Missing declared {registration.method} registration input: {path}')
    images = {hemi: nib.load(path) for hemi, path in native_surfaces.items()}
    for hemi, path in native_surfaces.items():
        if '_space-T1w_' not in path.name:
            raise ValueError(f'Physical registration requires a native T1w surface: {path}')
    outdir.mkdir(parents=True, exist_ok=True)
    outputs = {hemi: outdir/path.name.replace('_space-T1w_', f'_space-{registration.target_space}_')
               for hemi, path in native_surfaces.items()}
    if registration.method == 'fnirt-pair':
        for hemi, source in native_surfaces.items():
            run_workbench(['-surface-apply-warpfield', str(source), str(registration.inverse_warp),
                           str(outputs[hemi]), '-fnirt', str(registration.image_transform)], wb_command=wb_command)
    else:
        arrays = [np.asarray(image.agg_data('pointset'), dtype=float) for image in images.values()]
        native = np.vstack(arrays)
        # Intermediate warp/CSV files are temporary project work, not production products.
        scratch = REPO_ROOT/'tmp'
        scratch.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(prefix='surface_registration_', dir=scratch) as temporary:
            workdir = Path(temporary)
            if registration.method == 'fnirt':
                world = workdir/'mni_to_native_world.nii.gz'
                run_workbench(['-convert-warpfield', '-from-fnirt', str(registration.image_transform),
                               str(registration.source_image), '-to-world', str(world)], wb_command=wb_command)
                mapped = invert_world_displacement(native, world)
            else:
                mapped = ants_native_to_mni(native, registration.image_transform, workdir=workdir)
        if mapped.shape != native.shape or not np.isfinite(mapped).all():
            raise ValueError(f'Invalid {registration.method} transformed surface coordinates')
        offset = 0
        for (hemi, image), array in zip(images.items(), arrays):
            # Keep triangles, vertex order and anatomical correspondence unchanged.
            image.get_arrays_from_intent('pointset')[0].data = mapped[offset:offset+len(array)].astype(np.float32)
            image.get_arrays_from_intent('pointset')[0].coordsys = nib.gifti.GiftiCoordSystem(dataspace=4, xformspace=4)
            nib.save(image, outputs[hemi])
            offset += len(array)
    for hemi, path in outputs.items():
        if not np.isfinite(nib.load(path).agg_data('pointset')).all():
            raise ValueError(f'Nonfinite registered surface: {path}')
    return outputs


def require_registered_sampling(summary: object, *, context: Path) -> None:
    """An old corobl signal cache cannot be resumed as registered sampling."""
    records = summary if isinstance(summary, list) else summary.get('sampling_surfaces', []) if isinstance(summary, dict) else []
    if not records or any(r.get('space') != METRIC_SPACE or r.get('sampling_space') != MNI_SPACE
                          or '_space-T1w_' not in str(r.get('source_surface', '')) for r in records):
        raise ValueError(f'Unregistered/legacy corobl sampling cache: {context}; rebuild raw, QC and smoothing with --no-resume/--overwrite')
