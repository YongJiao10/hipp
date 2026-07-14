#!/usr/bin/env python3
"""Read-only audit of the formal gradient-spectral release provenance."""

from __future__ import annotations

import argparse
import json
import os
import sys
import uuid
from pathlib import Path
from typing import Iterable

CORE_WORKFLOW_DIR = Path(__file__).resolve().parent
if str(CORE_WORKFLOW_DIR) not in sys.path:
    sys.path.insert(0, str(CORE_WORKFLOW_DIR))

from lib.formal_cohorts import (
    cohort_key_for_group,
    load_formal_subjects,
    validate_recorded_subjects,
)


SHIFTED_COSINE = "shifted-cosine"
SHIFTED_COSINE_DESCRIPTION = (
    "((1+cosine_similarity)/2)_hadamard_surface_adjacency_diag0"
)
SMOOTHINGS = ("2mm", "4mm")
HEMIS = ("L", "R")

FORMAL_GROUPS = (
    ("outputs_hcp", "routea_full_k3_fisherz", "kong2022"),
    ("outputs_hcp", "routea_17net_full_k3_fisherz", "kong2022"),
    ("outputs_hcp", "routea_weighted_fisherz_full_k3", "kong2022"),
    (
        "outputs_hcp",
        "lynch2024_20net_nsi043_full_k3_normalized_wta",
        "lynch2024_20net",
    ),
    (
        "outputs_hcp",
        "lynch2024_8net_nsi043_full_k3_normalized_wta",
        "lynch2024_8net",
    ),
    ("outputs_stanford", "routea_hc_full_k3_fisherz", "kong2022"),
    ("outputs_stanford", "routea_symp_full_k3_fisherz", "kong2022"),
    (
        "outputs_stanford",
        "routea_hc_weighted_fisherz_full_k3",
        "kong2022",
    ),
    (
        "outputs_stanford",
        "routea_symp_weighted_fisherz_full_k3",
        "kong2022",
    ),
)

FORMAL_OVERVIEW_MANIFESTS = (
    "outputs_hcp/present/overviews/routea_group_overview_manifest.json",
    "outputs_hcp/present/overviews/routea_17net_group_overview_manifest.json",
    "outputs_hcp/present/overviews/weighted_fisherz/weighted_fisherz_group_overview_manifest.json",
    "outputs_hcp/present/overviews/lynch2024_20net/fisherz_group_overview_manifest.json",
    "outputs_hcp/present/overviews/lynch2024_8net/fisherz_group_overview_manifest.json",
    "outputs_stanford/present/overviews/routea_hc/fisherz_group_overview_manifest.json",
    "outputs_stanford/present/overviews/routea_symp/fisherz_group_overview_manifest.json",
    "outputs_stanford/present/overviews/routea_hc_weighted_fisherz/weighted_fisherz_group_overview_manifest.json",
    "outputs_stanford/present/overviews/routea_symp_weighted_fisherz/weighted_fisherz_group_overview_manifest.json",
)

_GROUP_SETTINGS: dict[str, tuple[str, str]] = {
    "routea_full_k3_fisherz": ("routea", "fisherz"),
    "routea_17net_full_k3_fisherz": ("routea_17net", "fisherz"),
    "routea_weighted_fisherz_full_k3": ("routea", "weighted_fisherz"),
    "lynch2024_20net_nsi043_full_k3_normalized_wta": (
        "lynch2024_20net",
        "fisherz",
    ),
    "lynch2024_8net_nsi043_full_k3_normalized_wta": (
        "lynch2024_8net",
        "fisherz",
    ),
    "routea_hc_full_k3_fisherz": ("routea", "fisherz"),
    "routea_symp_full_k3_fisherz": ("routea", "fisherz"),
    "routea_hc_weighted_fisherz_full_k3": (
        "routea",
        "weighted_fisherz",
    ),
    "routea_symp_weighted_fisherz_full_k3": (
        "routea",
        "weighted_fisherz",
    ),
}

_SOURCE_PATH_FIELDS = (
    "shared_root",
    "surface_root",
    "profile_surface_root",
    "reference_root",
    "group_gradient_features_active",
    "template_left_or_right_surface",
    "template_spec_path",
)
_ARCHIVE_PARTS = {"archive", "archives", "archived"}


def default_contract() -> dict[str, object]:
    groups: list[dict[str, object]] = []
    for output_root, branch, atlas in FORMAL_GROUPS:
        route_tag, assignment_variant = _GROUP_SETTINGS[branch]
        groups.append(
            {
                "output_root": output_root,
                "branch": branch,
                "atlas": atlas,
                "route_tag": route_tag,
                "assignment_variant": assignment_variant,
            }
        )
    return {
        "groups": groups,
        "overview_manifests": list(FORMAL_OVERVIEW_MANIFESTS),
    }


def _is_archived(path: Path) -> bool:
    return bool({part.lower() for part in path.parts} & _ARCHIVE_PARTS)


def _resolve(repo_root: Path, value: object) -> Path:
    path = Path(str(value))
    if not path.is_absolute():
        path = repo_root / path
    return path.resolve()


def _is_within(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


def _audit_source_path(
    *, value: object, field: str, manifest_path: Path, expected_root: Path,
    failures: list[str],
) -> None:
    if not isinstance(value, str) or not value.strip():
        failures.append(f"invalid: {manifest_path}: missing source path {field}")
        return
    source = _resolve(manifest_path.parent, value)
    if not source.exists():
        failures.append(f"incomplete: {manifest_path}: {field} does not exist: {source}")
    if not _is_within(source, expected_root):
        failures.append(
            f"invalid: {manifest_path}: {field} outside expected source root "
            f"{expected_root}: {source}"
        )
    if _is_archived(source):
        failures.append(f"invalid: {manifest_path}: archived source path in {field}: {source}")


def _read_json(path: Path, failures: list[str]) -> dict[str, object] | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        failures.append(f"invalid: cannot read JSON {path}: {exc}")
        return None
    if not isinstance(payload, dict):
        failures.append(f"invalid: JSON root is not an object: {path}")
        return None
    return payload


def _overview_manifest_paths(repo_root: Path, contract: dict[str, object]) -> list[Path]:
    explicit = contract.get("overview_manifests")
    if isinstance(explicit, list):
        return sorted({_resolve(repo_root, value) for value in explicit})

    found: set[Path] = set()
    roots = contract.get("overview_roots", [])
    if not isinstance(roots, list):
        return []
    for value in roots:
        root = _resolve(repo_root, value)
        if not root.is_dir() or _is_archived(root):
            continue
        for directory, names, files in os.walk(root, topdown=True):
            names[:] = [
                name
                for name in names
                if name.lower() not in _ARCHIVE_PARTS
            ]
            for filename in files:
                if filename.endswith("group_overview_manifest.json"):
                    found.add(Path(directory).resolve() / filename)
    return sorted(found)


def _group_manifest_path(
    repo_root: Path,
    group: dict[str, object],
    smoothing: str,
    hemi: str,
) -> Path:
    group_root = (
        repo_root
        / str(group["output_root"])
        / "work/group_atlas"
        / str(group["branch"])
    )
    return (
        group_root
        / "gradient-spectral"
        / str(group["atlas"])
        / smoothing
        / "cov_0p80/g1g2g3"
        / f"hemi_{hemi}/group_atlas_manifest.json"
    ).resolve()


def _require_equal(
    manifest: dict[str, object],
    field: str,
    expected: object,
    path: Path,
    failures: list[str],
) -> None:
    if manifest.get(field) != expected:
        failures.append(
            f"invalid: {path}: {field}={manifest.get(field)!r}, expected {expected!r}"
        )


def _manifest_route_tag(manifest: dict[str, object]) -> object:
    if "route_tag" in manifest:
        return manifest["route_tag"]
    signature = manifest.get("resume_input_signature_payload")
    if isinstance(signature, dict):
        return signature.get("route_tag")
    return None


def _resume_source(manifest: dict[str, object], field: str) -> object:
    value = manifest.get(field)
    if value is not None:
        return value
    signature = manifest.get("resume_input_signature_payload")
    if isinstance(signature, dict):
        return signature.get(field)
    return None


def _audit_group_manifest(
    *,
    path: Path,
    manifest: dict[str, object],
    group: dict[str, object],
    smoothing: str,
    hemi: str,
    failures: list[str],
) -> None:
    expected_root = path.parents[9].resolve()
    _require_equal(manifest, "dataset", group["output_root"], path, failures)
    _require_equal(
        manifest, "expected_source_root", str(expected_root), path, failures
    )
    _require_equal(manifest, "status", "ok", path, failures)
    _require_equal(manifest, "method", "gradient-spectral", path, failures)
    _require_equal(manifest, "atlas", group["atlas"], path, failures)
    _require_equal(manifest, "smoothing", smoothing, path, failures)
    _require_equal(manifest, "hemi", hemi, path, failures)
    _require_equal(manifest, "surface_affinity", SHIFTED_COSINE, path, failures)
    _require_equal(
        manifest,
        "affinity_description",
        SHIFTED_COSINE_DESCRIPTION,
        path,
        failures,
    )
    _require_equal(manifest, "candidate_k_range_requested", [2, 8], path, failures)
    _require_equal(manifest, "chosen_k_final", 3, path, failures)

    algorithm = manifest.get("clustering_algorithm")
    if not isinstance(algorithm, str) or "shifted_cosine" not in algorithm:
        failures.append(
            f"invalid: {path}: clustering_algorithm does not identify shifted-cosine"
        )
    graph = manifest.get("affinity_graph_diagnostics")
    if not isinstance(graph, dict) or graph.get("n_connected_components") != 1:
        failures.append(
            f"invalid: {path}: affinity_graph_diagnostics must record one connected component"
        )
    route_tag = _manifest_route_tag(manifest)
    if route_tag != group["route_tag"]:
        failures.append(
            f"invalid: {path}: route_tag={route_tag!r}, expected {group['route_tag']!r}"
        )

    for field in _SOURCE_PATH_FIELDS:
        _audit_source_path(
            value=manifest.get(field), field=field, manifest_path=path,
            expected_root=expected_root, failures=failures,
        )

    gradient_root = _resume_source(manifest, "gradient_root")
    if not isinstance(gradient_root, str) or not gradient_root.strip():
        failures.append(f"invalid: {path}: missing source path gradient_root")
    else:
        _audit_source_path(
            value=gradient_root, field="gradient_root", manifest_path=path,
            expected_root=expected_root, failures=failures,
        )
    subjects_manifest = _resume_source(manifest, "subjects_manifest")
    cohort_path = (
        _resolve(path.parent, subjects_manifest)
        if isinstance(subjects_manifest, str) and subjects_manifest.strip()
        else None
    )
    _audit_source_path(
        value=subjects_manifest, field="subjects_manifest", manifest_path=path,
        expected_root=expected_root, failures=failures,
    )
    if cohort_path is not None:
        try:
            cohort_subjects = load_formal_subjects(
                cohort_path,
                cohort_key_for_group(str(group["output_root"]), str(group["branch"])),
            )
            validate_recorded_subjects(manifest, cohort_subjects, source=path)
        except (ValueError, FileNotFoundError) as exc:
            failures.append(f"invalid: {path}: formal cohort validation failed: {exc}")
    consumed = manifest.get("actual_consumed_input_paths")
    if not isinstance(consumed, dict) or not consumed or not all(
        isinstance(value, str) and value.strip() for value in consumed.values()
    ):
        failures.append(
            f"invalid: {path}: actual_consumed_input_paths must contain source paths"
        )
    else:
        for name, value in consumed.items():
            _audit_source_path(
                value=value, field=f"actual_consumed_input_paths[{name!r}]",
                manifest_path=path, expected_root=expected_root, failures=failures,
            )

    expected_variant = str(group["assignment_variant"])
    _require_equal(
        manifest,
        "default_assignment_variant",
        expected_variant,
        path,
        failures,
    )
    variants = manifest.get("assignment_variants")
    if not isinstance(variants, dict) or expected_variant not in variants:
        failures.append(
            f"invalid: {path}: assignment_variants lacks {expected_variant!r}"
        )

    if expected_variant == "weighted_fisherz":
        _require_equal(
            manifest,
            "clustering_surface_affinity",
            SHIFTED_COSINE,
            path,
            failures,
        )
        _require_equal(
            manifest,
            "clustering_affinity_description",
            SHIFTED_COSINE_DESCRIPTION,
            path,
            failures,
        )
        _require_equal(manifest, "weighted_assignment_only", True, path, failures)
        source = manifest.get("weighted_fisherz_source_provenance")
        if not isinstance(source, dict):
            failures.append(
                f"invalid: {path}: missing weighted_fisherz_source_provenance"
            )
        else:
            _require_equal(source, "dataset", group["output_root"], path, failures)
            _require_equal(
                source, "expected_source_root", str(expected_root), path, failures
            )
            _require_equal(
                source, "subjects_manifest",
                str(cohort_path) if cohort_path is not None else None,
                path, failures,
            )
            _require_equal(
                source, "n_subjects", manifest.get("n_subjects"), path, failures
            )
            source_manifest = source.get("source_group_manifest")
            _audit_source_path(
                value=source_manifest, field="weighted source_group_manifest",
                manifest_path=path, expected_root=expected_root, failures=failures,
            )
            if isinstance(source_manifest, str) and source_manifest.strip():
                source_path = _resolve(path.parent, source_manifest)
                source_payload = _read_json(source_path, failures) if source_path.is_file() else None
                if source_payload is not None and source_payload.get("subjects") != manifest.get("subjects"):
                    failures.append(
                        f"invalid: {path}: weighted source subjects differ from weighted manifest"
                    )


def _overview_items(
    repo_root: Path,
    contract: dict[str, object],
    failures: list[str],
) -> list[tuple[Path, dict[str, object]]]:
    items: list[tuple[Path, dict[str, object]]] = []
    paths = _overview_manifest_paths(repo_root, contract)
    for path in paths:
        if _is_archived(path):
            failures.append(f"invalid: archived overview manifest configured: {path}")
            continue
        if not path.is_file():
            failures.append(f"incomplete: missing overview manifest: {path}")
            continue
        payload = _read_json(path, failures)
        if payload is None:
            continue
        raw_items = payload.get("items")
        if not isinstance(raw_items, list):
            failures.append(f"invalid: {path}: items must be a list")
            continue
        for index, item in enumerate(raw_items):
            if not isinstance(item, dict):
                failures.append(f"invalid: {path}: items[{index}] must be an object")
                continue
            if item.get("method") == "gradient-spectral":
                items.append((path, item))
    return items


def _item_group_root(repo_root: Path, item: dict[str, object]) -> Path | None:
    value = item.get("source_group_root")
    if not isinstance(value, str) or not value:
        return None
    return _resolve(repo_root, value)


def _audit_group_overview(
    *,
    repo_root: Path,
    group: dict[str, object],
    expected_manifests: set[Path],
    items: Iterable[tuple[Path, dict[str, object]]],
    failures: list[str],
) -> int:
    group_root = (
        repo_root
        / str(group["output_root"])
        / "work/group_atlas"
        / str(group["branch"])
    ).resolve()
    candidates = [
        entry
        for entry in items
        if _item_group_root(repo_root, entry[1]) == group_root
        and entry[1].get("atlas") == group["atlas"]
        and entry[1].get("assignment_variant") == group["assignment_variant"]
    ]
    if not candidates:
        failures.append(
            f"incomplete: no gradient overview lineage for formal group {group_root}"
        )
        return 0

    valid_count = 0
    for manifest_path, item in candidates:
        context = f"{manifest_path}: formal group {group_root}"
        item_failed = False
        checks = (
            ("atlas", group["atlas"]),
            ("assignment_variant", group["assignment_variant"]),
            ("surface_affinity", SHIFTED_COSINE),
            ("affinity_description", SHIFTED_COSINE_DESCRIPTION),
        )
        for field, expected in checks:
            if item.get(field) != expected:
                failures.append(
                    f"invalid: {context}: {field}={item.get(field)!r}, expected {expected!r}"
                )
                item_failed = True
        overview_png = item.get("overview_png")
        expected_present = (
            repo_root / str(group["output_root"]) / "present"
        ).resolve()
        if not isinstance(overview_png, str) or not overview_png.strip():
            failures.append(f"invalid: {context}: missing overview_png")
            item_failed = True
        else:
            png = _resolve(repo_root, overview_png)
            if not _is_within(png, expected_present):
                failures.append(
                    f"invalid: {context}: overview_png outside expected formal "
                    f"presentation directory {expected_present}: {png}"
                )
                item_failed = True
            if not png.is_file() or png.stat().st_size == 0:
                failures.append(f"incomplete: {context}: missing or empty overview_png: {png}")
                item_failed = True
        title = item.get("rendered_title")
        if not isinstance(title, str) or SHIFTED_COSINE not in title:
            failures.append(
                f"invalid: {context}: rendered_title must visibly contain {SHIFTED_COSINE!r}"
            )
            item_failed = True
        sources = item.get("source_group_manifests")
        if not isinstance(sources, list):
            failures.append(f"invalid: {context}: source_group_manifests must be a list")
            item_failed = True
        else:
            resolved_sources = {_resolve(repo_root, value) for value in sources}
            if resolved_sources != expected_manifests or len(sources) != 4:
                failures.append(
                    f"invalid: {context}: source_group_manifests do not equal the four formal combinations"
                )
                item_failed = True
            for source in resolved_sources:
                if _is_archived(source):
                    failures.append(
                        f"invalid: {context}: source_group_manifests references archive: {source}"
                    )
                    item_failed = True
                elif not source.is_file():
                    failures.append(
                        f"incomplete: {context}: source manifest does not exist: {source}"
                    )
                    item_failed = True
        if not item_failed:
            valid_count += 1
    return valid_count


def _group_identity(raw_group: object) -> tuple[str, str, str] | None:
    if isinstance(raw_group, dict):
        fields = ("output_root", "branch", "atlas")
        if any(field not in raw_group for field in fields):
            return None
        return (
            str(raw_group["output_root"]),
            str(raw_group["branch"]),
            str(raw_group["atlas"]),
        )
    if isinstance(raw_group, (list, tuple)) and len(raw_group) == 3:
        return (str(raw_group[0]), str(raw_group[1]), str(raw_group[2]))
    return None


def _canonical_group(identity: tuple[str, str, str]) -> dict[str, object]:
    output_root, branch, atlas = identity
    route_tag, assignment_variant = _GROUP_SETTINGS[branch]
    return {
        "output_root": output_root,
        "branch": branch,
        "atlas": atlas,
        "route_tag": route_tag,
        "assignment_variant": assignment_variant,
    }


def audit_release(repo_root: Path, contract: dict[str, object]) -> dict[str, object]:
    """Audit every formal group combination and its presentation lineage."""
    root = Path(repo_root).resolve()
    failures: list[str] = []
    groups = contract.get("groups")
    if not isinstance(groups, list):
        return {
            "status": "failed",
            "group_combinations": 0,
            "overview_items": 0,
            "failures": ["invalid: contract groups must be a list"],
        }

    identities = [_group_identity(group) for group in groups]
    expected_identities = set(FORMAL_GROUPS)
    if (
        any(identity is None for identity in identities)
        or len(identities) != len(FORMAL_GROUPS)
        or set(identities) != expected_identities
    ):
        return {
            "status": "failed",
            "group_combinations": 0,
            "overview_items": 0,
            "failures": [
                "invalid: contract formal group identities must equal FORMAL_GROUPS exactly"
            ],
        }

    overview_items = _overview_items(root, contract, failures)
    audited_combinations = 0
    valid_overview_items = 0
    for identity in identities:
        assert identity is not None
        raw_group = _canonical_group(identity)
        expected_paths: set[Path] = set()
        for smoothing in SMOOTHINGS:
            for hemi in HEMIS:
                path = _group_manifest_path(root, raw_group, smoothing, hemi)
                expected_paths.add(path)
                audited_combinations += 1
                if not path.is_file():
                    failures.append(f"incomplete: missing group manifest: {path}")
                    continue
                manifest = _read_json(path, failures)
                if manifest is not None:
                    _audit_group_manifest(
                        path=path,
                        manifest=manifest,
                        group=raw_group,
                        smoothing=smoothing,
                        hemi=hemi,
                        failures=failures,
                    )
        valid_overview_items += _audit_group_overview(
            repo_root=root,
            group=raw_group,
            expected_manifests=expected_paths,
            items=overview_items,
            failures=failures,
        )

    return {
        "status": "ok" if not failures else "failed",
        "group_combinations": audited_combinations,
        "overview_items": valid_overview_items,
        "failures": failures,
    }


def _atomic_write_json(path: Path, payload: dict[str, object]) -> None:
    destination = path.resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(
        f".{destination.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp"
    )
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, destination)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--output-json", type=Path)
    parser.add_argument("--allow-incomplete", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    contract = default_contract()
    report = audit_release(args.repo_root, contract)
    if args.output_json is not None:
        _atomic_write_json(args.output_json, report)
    print(json.dumps(report, indent=2))
    if report["status"] == "ok":
        return 0
    failures = report["failures"]
    if args.allow_incomplete and isinstance(failures, list) and failures and all(
        str(failure).startswith("incomplete:") for failure in failures
    ):
        return 0
    return 1


if __name__ == "__main__":
    sys.exit(main())
