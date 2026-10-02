#!/usr/bin/env python3
"""Repair the explicit default pointers in existing cluster-level runs.

This command is intentionally metadata-only. It requires each caller to pass
the exact run directories to repair, validates the manifest-declared Fisher-z
artifacts, and then points the manifest/default dominant-network JSON at the
already-generated ``fisherz`` variant. The normalized Top-20 variant is never
removed or overwritten.
"""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Any

from lib.workflow_utils import utc_now_iso, write_json


DEFAULT_ASSIGNMENT_VARIANT = "fisherz"
EXPECTED_ASSIGNMENT_METRIC = "top1_fisher_z"
REQUIRED_VARIANT_FIELDS = (
    "assignment_metric",
    "cluster_assignment_status",
    "group_functional_atlas_labels",
    "group_functional_atlas_label_map",
    "template_functional_atlas_label",
)


def _load_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Cannot read JSON artifact: {path}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"JSON artifact must contain an object: {path}")
    return payload


def _require_declared_path(
    *,
    run_dir: Path,
    variant: dict[str, Any],
    field: str,
    expected_name: str | None = None,
) -> Path:
    raw = variant.get(field)
    if not isinstance(raw, str) or not raw.strip():
        raise ValueError(f"Fisher-z variant is missing declared {field!r}: {run_dir}")
    path = Path(raw).expanduser()
    if not path.is_absolute():
        raise ValueError(
            f"Fisher-z variant must declare an absolute {field} path; "
            f"run_dir={run_dir} value={raw!r}"
        )
    resolved = path.resolve()
    if not resolved.is_relative_to(run_dir):
        raise ValueError(
            f"Fisher-z {field} path escapes its run directory: "
            f"run_dir={run_dir} path={resolved}"
        )
    if expected_name is not None and resolved.name != expected_name:
        raise ValueError(
            f"Fisher-z {field} must name {expected_name!r}; "
            f"run_dir={run_dir} path={resolved}"
        )
    if not resolved.is_file():
        raise FileNotFoundError(f"Manifest-declared Fisher-z {field} is missing: {resolved}")
    return resolved


def _fisherz_repair_plan(run_dir_input: Path) -> dict[str, Any]:
    run_dir = run_dir_input.expanduser().resolve()
    if not run_dir.is_dir():
        raise FileNotFoundError(f"Run directory does not exist: {run_dir}")

    manifest_path = run_dir / "group_atlas_manifest.json"
    dominant_path = run_dir / "cluster_dominant_network.json"
    status_path = run_dir / "cluster_assignment_status_fisherz.json"
    manifest = _load_json(manifest_path)
    if manifest.get("status") != "ok":
        raise ValueError(f"Group-atlas manifest is not ok: {manifest_path}")
    if manifest.get("assignment_mode") != "cluster_level":
        raise ValueError(
            f"Repair applies only to cluster-level manifests: {manifest_path} "
            f"assignment_mode={manifest.get('assignment_mode')!r}"
        )

    variants = manifest.get("assignment_variants")
    if not isinstance(variants, dict) or not isinstance(variants.get(DEFAULT_ASSIGNMENT_VARIANT), dict):
        raise ValueError(
            f"Manifest lacks the required {DEFAULT_ASSIGNMENT_VARIANT!r} assignment variant: "
            f"{manifest_path}"
        )
    fisherz_variant = dict(variants[DEFAULT_ASSIGNMENT_VARIANT])
    missing_fields = [field for field in REQUIRED_VARIANT_FIELDS if field not in fisherz_variant]
    if missing_fields:
        raise ValueError(
            f"Fisher-z variant is missing fields {missing_fields}: {manifest_path}"
        )
    if fisherz_variant.get("assignment_metric") != EXPECTED_ASSIGNMENT_METRIC:
        raise ValueError(
            f"Unexpected Fisher-z assignment metric in {manifest_path}: "
            f"{fisherz_variant.get('assignment_metric')!r}"
        )

    declared_status_path = _require_declared_path(
        run_dir=run_dir,
        variant=fisherz_variant,
        field="cluster_assignment_status",
        expected_name=status_path.name,
    )
    _require_declared_path(
        run_dir=run_dir,
        variant=fisherz_variant,
        field="group_functional_atlas_labels",
        expected_name="group_functional_atlas_labels_fisherz.npy",
    )
    _require_declared_path(
        run_dir=run_dir,
        variant=fisherz_variant,
        field="group_functional_atlas_label_map",
        expected_name="group_functional_atlas_label_map_fisherz.json",
    )
    template_label_path = _require_declared_path(
        run_dir=run_dir,
        variant=fisherz_variant,
        field="template_functional_atlas_label",
    )
    if "fisherz" not in template_label_path.name:
        raise ValueError(
            f"Fisher-z template label declaration does not contain the variant name: "
            f"{template_label_path}"
        )
    if declared_status_path != status_path.resolve():
        raise ValueError(
            f"Fisher-z status declaration must point to the exact run-local artifact: "
            f"{manifest_path} declared={declared_status_path} expected={status_path.resolve()}"
        )

    status_payload = _load_json(status_path)
    if status_payload.get("assignment_metric") != EXPECTED_ASSIGNMENT_METRIC:
        raise ValueError(
            f"Unexpected Fisher-z status metric in {status_path}: "
            f"{status_payload.get('assignment_metric')!r}"
        )
    items = status_payload.get("items")
    if not isinstance(items, list) or not items:
        raise ValueError(f"Fisher-z status has no assignment items: {status_path}")
    cluster_ids: list[int] = []
    dominant: list[str] = []
    for item in items:
        if not isinstance(item, dict):
            raise ValueError(f"Malformed Fisher-z status item in {status_path}: {item!r}")
        try:
            cluster_id = int(item["cluster_id"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"Fisher-z status item lacks a valid cluster_id: {status_path}") from exc
        top1_network = str(item.get("top1_network", "")).strip()
        if not top1_network:
            raise ValueError(f"Fisher-z status item lacks top1_network: {status_path}")
        cluster_ids.append(cluster_id)
        dominant.append(top1_network)
    if sorted(cluster_ids) != list(range(1, len(cluster_ids) + 1)):
        raise ValueError(
            f"Fisher-z cluster IDs must be unique and contiguous from 1: {status_path} "
            f"ids={cluster_ids}"
        )
    dominant_by_cluster = [
        name for _, name in sorted(zip(cluster_ids, dominant), key=lambda pair: pair[0])
    ]

    if not dominant_path.is_file():
        raise FileNotFoundError(f"Missing cluster dominant-network artifact: {dominant_path}")
    dominant_payload = _load_json(dominant_path)
    network_names = dominant_payload.get("network_names")
    if not isinstance(network_names, list) or not network_names:
        raise ValueError(f"Dominant-network artifact lacks network_names: {dominant_path}")
    unknown = [name for name in dominant_by_cluster if name not in {str(x) for x in network_names}]
    if unknown:
        raise ValueError(
            f"Fisher-z dominant networks are absent from network_names: "
            f"{dominant_path} unknown={unknown}"
        )

    repaired_manifest = copy.deepcopy(manifest)
    previous_default = repaired_manifest.get("default_assignment_variant")
    repaired_manifest["default_assignment_variant"] = DEFAULT_ASSIGNMENT_VARIANT
    repaired_manifest["cluster_assignment_metric"] = EXPECTED_ASSIGNMENT_METRIC
    repaired_manifest["cluster_assignment_status"] = fisherz_variant["cluster_assignment_status"]
    repaired_manifest["group_functional_atlas_labels"] = fisherz_variant["group_functional_atlas_labels"]
    repaired_manifest["group_functional_atlas_label_map"] = fisherz_variant["group_functional_atlas_label_map"]
    repaired_manifest["template_functional_atlas_label"] = fisherz_variant["template_functional_atlas_label"]
    repaired_manifest["default_assignment_repair"] = {
        "status": "explicit_cluster_level_fisherz_default_repair",
        "assignment_variant": DEFAULT_ASSIGNMENT_VARIANT,
        "previous_default_assignment_variant": previous_default,
        "source_variant": "assignment_variants.fisherz",
        "repaired_at": utc_now_iso(),
    }

    repaired_dominant = copy.deepcopy(dominant_payload)
    repaired_dominant["dominant"] = dominant_by_cluster
    repaired_dominant["assignment_variant"] = DEFAULT_ASSIGNMENT_VARIANT
    return {
        "run_dir": run_dir,
        "manifest_path": manifest_path,
        "dominant_path": dominant_path,
        "manifest": repaired_manifest,
        "dominant": repaired_dominant,
        "previous_default_assignment_variant": previous_default,
        "dominant_networks": dominant_by_cluster,
    }


def repair_run_dirs(run_dirs: list[Path], *, dry_run: bool = False) -> list[dict[str, Any]]:
    """Validate every exact run-dir first, then optionally write its repair."""
    if not run_dirs:
        raise ValueError("At least one exact --run-dir is required")
    plans = [_fisherz_repair_plan(run_dir) for run_dir in run_dirs]
    if not dry_run:
        for plan in plans:
            write_json(plan["manifest_path"], plan["manifest"])
            write_json(plan["dominant_path"], plan["dominant"])
    return plans


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Explicit metadata-only repair of cluster-level Fisher-z defaults."
    )
    parser.add_argument(
        "--run-dir",
        action="append",
        type=Path,
        required=True,
        help="Exact existing group-atlas run directory; repeat once per target.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate all exact targets and print the planned changes without writing.",
    )
    return parser


def main() -> int:
    args = build_arg_parser().parse_args()
    plans = repair_run_dirs(args.run_dir, dry_run=bool(args.dry_run))
    action = "would repair" if args.dry_run else "repaired"
    for plan in plans:
        print(
            f"{action}: {plan['run_dir']} | "
            f"{plan['previous_default_assignment_variant']!r} -> 'fisherz' | "
            f"dominant={plan['dominant_networks']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
