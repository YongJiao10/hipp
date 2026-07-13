#!/usr/bin/env python3
"""Atomically enrich the eight retained HCP RouteA formal group manifests."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import uuid
from pathlib import Path
from typing import Callable, NamedTuple

CORE_WORKFLOW_DIR = Path(__file__).resolve().parent
if str(CORE_WORKFLOW_DIR) not in sys.path:
    sys.path.insert(0, str(CORE_WORKFLOW_DIR))

from lib.formal_cohorts import load_formal_subjects, subject_list_sha256, validate_recorded_subjects


REPO_ROOT = Path(__file__).resolve().parents[2]
BRANCHES = ("routea_full_k3_fisherz", "routea_17net_full_k3_fisherz")
SMOOTHINGS = ("2mm", "4mm")
HEMIS = ("L", "R")
SHIFTED_COSINE = "shifted-cosine"
SHIFTED_COSINE_DESCRIPTION = "((1+cosine_similarity)/2)_hadamard_surface_adjacency_diag0"
DEFAULT_SUBJECTS = (
    REPO_ROOT / "outputs_hcp/inputs/hcp/kong2022/manifests"
    / "hcp_7t_hippocampus_fmri_qc_pass_t1only_latest.txt"
)
_ROOT_FIELDS = ("shared_root", "surface_root", "profile_surface_root", "reference_root")
_LEAF_FIELDS = (
    "group_gradient_features_active", "group_labels_final", "group_support_mask",
    "cluster_network_profile",
)
_ASSIGNMENT_FIELDS = (
    "cluster_assignment_status", "group_functional_atlas_labels",
    "group_functional_atlas_label_map", "template_functional_atlas_label",
)


class PreparedManifestUpdate(NamedTuple):
    path: Path
    original_bytes: bytes
    updated_bytes: bytes


def _within(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


def _resolved(value: object, manifest: Path) -> Path:
    path = Path(str(value)).expanduser()
    return path.resolve() if path.is_absolute() else (manifest.parent / path).resolve()


def _require_path(value: object, *, field: str, manifest: Path, root: Path, leaf: bool) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"Missing {field} in {manifest}")
    path = _resolved(value, manifest)
    if not _within(path, root):
        raise ValueError(f"{field} outside canonical HCP root {root}: {path}")
    if not path.exists() or (leaf and not path.is_file()):
        raise FileNotFoundError(f"Missing required {field}: {path}")
    if {part.lower() for part in path.parts} & {"archive", "archives", "archived"}:
        raise ValueError(f"Archived {field} is not eligible: {path}")
    return path


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_json_sha256(payload: object) -> str:
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def retained_manifest_paths(repo_root: Path) -> list[Path]:
    hcp = Path(repo_root).resolve() / "outputs_hcp/work/group_atlas"
    return [
        hcp / branch / "gradient-spectral/kong2022" / smoothing
        / "cov_0p80/g1g2g3" / f"hemi_{hemi}/group_atlas_manifest.json"
        for branch in BRANCHES for smoothing in SMOOTHINGS for hemi in HEMIS
    ]


def _validated_payload(
    path: Path, hcp_root: Path, subjects_manifest: Path, subjects: list[str],
    *, original_bytes: bytes | None = None,
) -> dict[str, object]:
    try:
        raw = path.read_bytes() if original_bytes is None else original_bytes
        payload = json.loads(raw.decode("utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"Cannot read retained manifest {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"Retained manifest is not an object: {path}")
    required = {
        "status": "ok", "method": "gradient-spectral", "atlas": "kong2022",
        "surface_affinity": SHIFTED_COSINE,
        "affinity_description": SHIFTED_COSINE_DESCRIPTION,
        "candidate_k_range_requested": [2, 8],
        "chosen_k_final": 3,
    }
    for field, expected in required.items():
        if payload.get(field) != expected:
            raise ValueError(f"Invalid retained manifest {path}: {field}={payload.get(field)!r}")
    algorithm = payload.get("clustering_algorithm")
    if not isinstance(algorithm, str) or "shifted_cosine" not in algorithm:
        raise ValueError(f"Invalid retained shifted-cosine clustering algorithm: {path}")
    branch = path.parents[6].name
    expected_route = "routea" if branch == BRANCHES[0] else "routea_17net"
    expected_taxonomy = "canonical8" if branch == BRANCHES[0] else "family17"
    signature_payload = payload.get("resume_input_signature_payload")
    if not isinstance(signature_payload, dict):
        raise ValueError(f"Missing retained resume signature payload: {path}")
    if signature_payload.get("route_tag") != expected_route:
        raise ValueError(f"Invalid retained route tag for {branch}: {path}")
    if payload.get("network_taxonomy") not in (None, expected_taxonomy):
        raise ValueError(f"Invalid retained network taxonomy for {branch}: {path}")
    if payload.get("smoothing") != path.parents[3].name or payload.get("hemi") != path.parent.name.removeprefix("hemi_"):
        raise ValueError(f"Retained combo fields do not match manifest path: {path}")
    graph = payload.get("affinity_graph_diagnostics")
    if not isinstance(graph, dict) or graph.get("n_connected_components") != 1:
        raise ValueError(f"Retained manifest graph is not connected: {path}")
    validate_recorded_subjects(payload, subjects, source=path)
    signature = payload.get("resume_input_signature")
    digest_pattern = re.compile(r"[0-9a-f]{64}")
    if not isinstance(signature, str) or digest_pattern.fullmatch(signature) is None or not isinstance(signature_payload, dict):
        raise ValueError(f"Missing retained resume artifact fingerprint contract: {path}")
    if signature != _canonical_json_sha256(signature_payload):
        raise ValueError(f"Retained resume artifact fingerprint mismatch: {path}")
    hashes = signature_payload.get("input_category_hashes")
    if not isinstance(hashes, dict) or not hashes or any(
        not isinstance(value, str) or digest_pattern.fullmatch(value) is None
        for value in hashes.values()
    ):
        raise ValueError(f"Invalid retained input-category fingerprints: {path}")
    recorded_subject_manifest = _resolved(signature_payload.get("subjects_manifest"), path)
    if recorded_subject_manifest != subjects_manifest:
        raise ValueError(
            f"Retained subjects_manifest differs from approved HCP 151 manifest: {recorded_subject_manifest}"
        )
    artifacts: set[Path] = set()
    routea_root = (hcp_root / "inputs/hcp/kong2022/shared_inputs/routea").resolve()
    surface_root = (hcp_root / "inputs/hcp/kong2022/shared_inputs/surface").resolve()
    expected_roots = {
        "shared_root": routea_root,
        "surface_root": surface_root,
        "profile_surface_root": surface_root,
        "reference_root": routea_root,
    }
    for field in _ROOT_FIELDS:
        source_root = _require_path(
            payload.get(field), field=field, manifest=path, root=hcp_root, leaf=False
        )
        if source_root != expected_roots[field]:
            raise ValueError(
                f"{field} is not the exact canonical HCP RouteA root "
                f"{expected_roots[field]}: {source_root}"
            )
    for field in _LEAF_FIELDS:
        artifacts.add(_require_path(payload.get(field), field=field, manifest=path, root=hcp_root, leaf=True))
    gradient_root = _require_path(
        signature_payload.get("gradient_root"), field="gradient_root", manifest=path,
        root=hcp_root, leaf=False,
    )
    canonical_gradient_root = (hcp_root / "work/hippomaps_gradients").resolve()
    if gradient_root != canonical_gradient_root and canonical_gradient_root not in gradient_root.parents:
        raise ValueError(
            f"gradient_root outside canonical HCP gradient prefix {canonical_gradient_root}: {gradient_root}"
        )
    consumed = payload.get("actual_consumed_input_paths")
    if not isinstance(consumed, dict) or not consumed:
        raise ValueError(f"Missing actual_consumed_input_paths: {path}")
    for name, value in consumed.items():
        artifacts.add(_require_path(value, field=f"actual_consumed_input_paths[{name!r}]", manifest=path, root=hcp_root, leaf=True))
    variants = payload.get("assignment_variants")
    if not isinstance(variants, dict) or not variants:
        raise ValueError(f"Missing assignment_variants: {path}")
    for variant, row in variants.items():
        if not isinstance(row, dict):
            raise ValueError(f"Invalid assignment variant {variant!r}: {path}")
        for field in _ASSIGNMENT_FIELDS:
            artifacts.add(_require_path(row.get(field), field=f"assignment_variants[{variant!r}].{field}", manifest=path, root=hcp_root, leaf=True))
    payload["dataset"] = "outputs_hcp"
    payload["expected_source_root"] = str(hcp_root)
    payload["provenance_upgrade"] = {
        "utility": "upgrade_retained_hcp_formal_provenance",
        "non_computational": True,
        "subjects_manifest": str(subjects_manifest),
        "subjects_sha256": subject_list_sha256(subjects),
        "artifact_sha256": {str(artifact): _sha256(artifact) for artifact in sorted(artifacts, key=str)},
    }
    return payload


def prepare_retained_manifest_updates(
    repo_root: Path, subjects_manifest: Path
) -> tuple[PreparedManifestUpdate, ...]:
    """Validate and render all eight updates without modifying the filesystem."""
    root = Path(repo_root).resolve()
    hcp_root = (root / "outputs_hcp").resolve()
    approved_subjects = Path(subjects_manifest).resolve()
    if not _within(approved_subjects, hcp_root):
        raise ValueError(f"HCP 151 subject manifest outside canonical HCP root: {approved_subjects}")
    subjects = load_formal_subjects(approved_subjects, "hcp151")
    paths = retained_manifest_paths(root)
    prepared: list[PreparedManifestUpdate] = []
    for path in paths:
        original = path.read_bytes()
        payload = _validated_payload(
            path, hcp_root, approved_subjects, subjects, original_bytes=original
        )
        updated = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode("utf-8")
        prepared.append(PreparedManifestUpdate(path, original, updated))
    return tuple(prepared)


def _staged_path(path: Path, role: str) -> Path:
    return path.with_name(
        f".{path.name}.{os.getpid()}.{uuid.uuid4().hex}.{role}.tmp"
    )


def commit_prepared_updates(
    prepared: tuple[PreparedManifestUpdate, ...],
    *, replace: Callable[[Path, Path], None] = os.replace,
) -> None:
    """Commit every prepared JSON or restore every exact original byte string."""
    staged: dict[Path, Path] = {}
    cleanup: set[Path] = set()
    try:
        for update in prepared:
            temporary = _staged_path(update.path, "new")
            temporary.write_bytes(update.updated_bytes)
            staged[update.path] = temporary
            cleanup.add(temporary)
        try:
            for update in prepared:
                replace(staged[update.path], update.path)
                cleanup.discard(staged[update.path])
        except Exception as commit_error:
            rollback_errors: list[Exception] = []
            for update in prepared:
                try:
                    if update.path.read_bytes() == update.original_bytes:
                        continue
                    rollback = _staged_path(update.path, "rollback")
                    cleanup.add(rollback)
                    rollback.write_bytes(update.original_bytes)
                    os.replace(rollback, update.path)
                    cleanup.discard(rollback)
                except Exception as exc:  # pragma: no cover - catastrophic storage failure
                    rollback_errors.append(exc)
            if rollback_errors:
                raise RuntimeError(
                    f"Provenance transaction failed and rollback was incomplete: {rollback_errors}"
                ) from commit_error
            raise
    finally:
        for temporary in cleanup:
            temporary.unlink(missing_ok=True)


def upgrade_retained_manifests(repo_root: Path, subjects_manifest: Path) -> dict[str, object]:
    prepared = prepare_retained_manifest_updates(repo_root, subjects_manifest)
    commit_prepared_updates(prepared)
    return {
        "status": "ok",
        "updated_count": len(prepared),
        "manifests": [str(update.path) for update in prepared],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    parser.add_argument("--subjects-manifest", type=Path, default=DEFAULT_SUBJECTS)
    args = parser.parse_args()
    print(json.dumps(upgrade_retained_manifests(args.repo_root, args.subjects_manifest), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
