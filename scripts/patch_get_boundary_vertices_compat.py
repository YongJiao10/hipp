#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path


PATCH_MARKER = "HIPPOMAPS_MACOS_COMPAT_BOUNDARY_IDS"
OLD = """# Count the number of points in each component
component_sizes = np.bincount(region_ids)

logger.info(f"Found {num_components} connected components.")

# Identify the largest component
largest_component_id = component_sizes.argmax()
largest_component_mask = region_ids == largest_component_id

# Create final scalar array for label output
boundary_scalars = np.zeros(surface.n_points, dtype=np.int32)

# Compute hole radii for smaller components
hole_radii = []

for region_id, size in enumerate(component_sizes):
    logger.info(f"Component {region_id}: {size} vertices")

    if region_id == largest_component_id:
        continue  # Skip largest component

    # Mask of points in this region
    region_mask = region_ids == region_id
    point_ids = connected_sub_mesh.point_data["vtkOriginalPointIds"][region_mask]
    boundary_scalars[point_ids] = 2
    coords = surface.points[point_ids]

    # Estimate hole radius as max distance from centroid
    centroid = coords.mean(axis=0)
    dists = np.linalg.norm(coords - centroid, axis=1)
    radius = dists.max()
    hole_radii.append(radius)

    logger.info(f"  → Estimated radius of component {region_id}: {radius:.3f}")

# Map back to original surface point indices
largest_component_indices = connected_sub_mesh.point_data["vtkOriginalPointIds"][
    largest_component_mask
]

boundary_scalars[largest_component_indices] = 1
"""

NEW = """# Count the number of points in each component
component_sizes = np.bincount(region_ids)

logger.info(f"Found {num_components} connected components.")

# Identify the largest component
largest_component_id = component_sizes.argmax()
largest_component_mask = region_ids == largest_component_id

# Create final scalar array for label output
boundary_scalars = np.zeros(surface.n_points, dtype=np.int32)

# Some pyvista versions drop vtkOriginalPointIds during connectivity().
# Recover them from the connected mesh when available, otherwise fall back
# to the sub-mesh or the original boolean boundary mask.
original_point_ids = connected_sub_mesh.point_data.get("vtkOriginalPointIds")
if original_point_ids is None:
    original_point_ids = sub_mesh.point_data.get("vtkOriginalPointIds")
if original_point_ids is None:
    candidate_ids = np.flatnonzero(boundary_indices >= 0)
    candidate_ids = boundary_indices
    if len(candidate_ids) == connected_sub_mesh.n_points:
        original_point_ids = candidate_ids
    else:
        raise KeyError("vtkOriginalPointIds")
original_point_ids = np.asarray(original_point_ids)

# Compute hole radii for smaller components
hole_radii = []

for region_id, size in enumerate(component_sizes):
    logger.info(f"Component {region_id}: {size} vertices")

    if region_id == largest_component_id:
        continue  # Skip largest component

    # Mask of points in this region
    region_mask = region_ids == region_id
    point_ids = original_point_ids[region_mask]
    boundary_scalars[point_ids] = 2
    coords = surface.points[point_ids]

    # Estimate hole radius as max distance from centroid
    centroid = coords.mean(axis=0)
    dists = np.linalg.norm(coords - centroid, axis=1)
    radius = dists.max()
    hole_radii.append(radius)

    logger.info(f"  → Estimated radius of component {region_id}: {radius:.3f}")

# Map back to original surface point indices
largest_component_indices = original_point_ids[largest_component_mask]

boundary_scalars[largest_component_indices] = 1

# HIPPOMAPS_MACOS_COMPAT_BOUNDARY_IDS: recover original boundary ids across pyvista connectivity variants
"""


def patch_file(path: Path) -> bool:
    if not path.exists():
        return False
    text = path.read_text(encoding="utf-8")
    if PATCH_MARKER in text:
        return False
    if OLD not in text:
        return False
    path.write_text(text.replace(OLD, NEW), encoding="utf-8")
    return True


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Patch HippUnfold boundary extraction for pyvista variants that drop vtkOriginalPointIds"
    )
    parser.add_argument("--hippunfold-site-root", required=True)
    parser.add_argument("--runtime-source-cache", required=True)
    args = parser.parse_args()

    site_root = Path(args.hippunfold_site_root)
    runtime_source_cache = Path(args.runtime_source_cache)
    candidates = [
        site_root / "workflow" / "scripts" / "get_boundary_vertices.py",
        runtime_source_cache
        / "file"
        / site_root.relative_to("/")
        / "workflow"
        / "scripts"
        / "get_boundary_vertices.py",
    ]

    patched = []
    skipped = []
    for path in candidates:
        if not path.exists():
            continue
        try:
            if patch_file(path):
                patched.append(str(path))
        except PermissionError:
            skipped.append(f"{path} (permission denied)")

    if patched:
        print("Patched boundary vertex compatibility files:")
        for path in patched:
            print(path)
    if skipped:
        print("Skipped boundary vertex compatibility files:")
        for path in skipped:
            print(path)
    if not patched and not skipped:
        print("No boundary vertex compatibility patches were needed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
