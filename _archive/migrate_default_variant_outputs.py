#!/usr/bin/env python3
from __future__ import annotations

"""
Migrate cluster-level functional atlas outputs to the strict default-variant contract.

The script treats `assignment_variants[default_assignment_variant]` as the only
cluster-level default source. It never deletes by default. Use dry-run output as
the deletion target list for user confirmation, then archive before deletion.
"""

import argparse
import json
import tarfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import nibabel as nib
import numpy as np

from lib.output_layout import (
    DEFAULT_DATASET,
    resolve_group_atlas_root,
    resolve_work_root,
)


def _default_roots(dataset: str = DEFAULT_DATASET) -> list[Path]:
    return [
        resolve_group_atlas_root(dataset),
        resolve_work_root(dataset) / "individual_atlas",
    ]


DEFAULT_ROOTS = _default_roots()
DEFAULT_FIELDS = {
    "group_functional_atlas_labels": "group_functional_atlas_labels.npy",
    "group_functional_atlas_label_map": "group_functional_atlas_label_map.json",
    "cluster_assignment_status": "cluster_assignment_status.json",
}


@dataclass
class ManifestPlan:
    manifest: Path
    default_variant: str
    updates: dict[str, str] = field(default_factory=dict)
    variant_updates: dict[str, dict[str, str]] = field(default_factory=dict)
    archive_targets: list[Path] = field(default_factory=list)
    skipped: list[str] = field(default_factory=list)

    def to_json(self) -> dict[str, Any]:
        return {
            "manifest": str(self.manifest),
            "default_variant": self.default_variant,
            "updates": self.updates,
            "variant_updates": self.variant_updates,
            "archive_targets": [str(p) for p in self.archive_targets],
            "skipped": self.skipped,
        }


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def path_from(raw: Any) -> Path | None:
    if not raw:
        return None
    return Path(str(raw)).resolve()


def npy_equal(left: Path, right: Path) -> bool:
    return bool(np.array_equal(np.load(left), np.load(right)))


def json_equal(left: Path, right: Path) -> bool:
    return load_json(left) == load_json(right)


def gifti_equal(left: Path, right: Path) -> bool:
    left_img = nib.load(str(left))
    right_img = nib.load(str(right))
    if len(left_img.darrays) != len(right_img.darrays):
        return False
    for left_da, right_da in zip(left_img.darrays, right_img.darrays, strict=True):
        if not np.array_equal(np.asarray(left_da.data), np.asarray(right_da.data)):
            return False
    left_labels = [(lab.key, lab.label, lab.red, lab.green, lab.blue, lab.alpha) for lab in left_img.labeltable.labels]
    right_labels = [(lab.key, lab.label, lab.red, lab.green, lab.blue, lab.alpha) for lab in right_img.labeltable.labels]
    return left_labels == right_labels


def semantically_equal(left: Path, right: Path) -> bool:
    if not left.exists() or not right.exists():
        return False
    if left.suffix == ".npy" and right.suffix == ".npy":
        return npy_equal(left, right)
    if left.suffix == ".json" and right.suffix == ".json":
        return json_equal(left, right)
    if left.name.endswith(".label.gii") and right.name.endswith(".label.gii"):
        return gifti_equal(left, right)
    return left.read_bytes() == right.read_bytes()


def require_within_cwd(path: Path, cwd: Path) -> Path:
    resolved = path.resolve()
    resolved.relative_to(cwd.resolve())
    return resolved


def add_duplicate_if_verified(plan: ManifestPlan, old_path: Path, new_path: Path, cwd: Path, reason: str) -> None:
    old_path = require_within_cwd(old_path, cwd)
    new_path = require_within_cwd(new_path, cwd)
    if old_path == new_path or not old_path.exists():
        return
    if not new_path.exists():
        plan.skipped.append(f"missing_default_variant_target:{reason}:{new_path}")
        return
    if semantically_equal(old_path, new_path):
        plan.archive_targets.append(old_path)
    else:
        plan.skipped.append(f"not_semantically_equal:{reason}:{old_path}:{new_path}")


def build_manifest_plan(manifest_path: Path, cwd: Path) -> ManifestPlan | None:
    manifest = load_json(manifest_path)
    variants = manifest.get("assignment_variants")
    if not isinstance(variants, dict) or not variants:
        return None
    default_variant = str(manifest.get("default_assignment_variant") or "fisherz")
    default_spec = variants.get(default_variant)
    if not isinstance(default_spec, dict):
        return None

    run_dir = manifest_path.parent
    plan = ManifestPlan(manifest=manifest_path.resolve(), default_variant=default_variant)
    for field, default_name in DEFAULT_FIELDS.items():
        variant_path = path_from(default_spec.get(field))
        if variant_path is None:
            plan.skipped.append(f"missing_variant_field:{field}")
            continue
        old_path = path_from(manifest.get(field)) or (run_dir / default_name).resolve()
        plan.updates[field] = str(variant_path)
        add_duplicate_if_verified(plan, old_path, variant_path, cwd, field)

    variant_label = path_from(default_spec.get("template_functional_atlas_label_variant"))
    if variant_label is not None:
        old_label = path_from(manifest.get("template_functional_atlas_label"))
        if old_label is not None:
            plan.updates["template_functional_atlas_label"] = str(variant_label)
            add_duplicate_if_verified(plan, old_label, variant_label, cwd, "template_functional_atlas_label")
        default_variant_updates = plan.variant_updates.setdefault(default_variant, {})
        default_variant_updates["template_functional_atlas_label"] = str(variant_label)

    if "default_assignment_variant" not in manifest:
        plan.updates["default_assignment_variant"] = default_variant
    if manifest.get("assignment_mode") != "cluster_level":
        plan.updates["assignment_mode"] = "cluster_level"
    plan.archive_targets = sorted(set(plan.archive_targets))
    return plan


def build_plan(roots: list[Path], cwd: Path) -> list[ManifestPlan]:
    plans: list[ManifestPlan] = []
    for root in roots:
        if not root.exists():
            continue
        for manifest_path in sorted(root.rglob("group_atlas_manifest.json")):
            manifest_path = require_within_cwd(manifest_path, cwd)
            plan = build_manifest_plan(manifest_path, cwd)
            if plan is not None and (plan.updates or plan.variant_updates or plan.archive_targets or plan.skipped):
                plans.append(plan)
    return plans


def write_manifest_updates(plans: list[ManifestPlan]) -> None:
    for plan in plans:
        manifest = load_json(plan.manifest)
        for key, value in plan.updates.items():
            manifest[key] = value
        variants = manifest.setdefault("assignment_variants", {})
        for variant, updates in plan.variant_updates.items():
            row = variants.setdefault(variant, {})
            for key, value in updates.items():
                row[key] = value
        write_json(plan.manifest, manifest)


def archive_targets(plans: list[ManifestPlan], archive_path: Path, cwd: Path) -> None:
    targets = sorted({require_within_cwd(path, cwd) for plan in plans for path in plan.archive_targets})
    archive_path.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive_path, "w:gz") as tf:
        for target in targets:
            tf.add(target, arcname=target.relative_to(cwd.resolve()))


def delete_targets_from_plan(plan_json: Path, cwd: Path) -> None:
    payload = load_json(plan_json)
    targets = sorted({require_within_cwd(Path(path), cwd) for item in payload["plans"] for path in item["archive_targets"]})
    for target in targets:
        target.unlink()


def main() -> int:
    parser = argparse.ArgumentParser(description="Migrate cluster-level default variant outputs without unsuffixed duplicates.")
    parser.add_argument("--dataset", default=DEFAULT_DATASET, choices=("hcp", "iowa", "stanford"),
                        help="Dataset key (default: hcp)")
    parser.add_argument("--roots", nargs="*", default=[str(p) for p in _default_roots()])
    parser.add_argument("--plan-out", default=str(resolve_work_root() / "default_variant_migration_plan.json"))
    parser.add_argument("--write-manifests", action="store_true", help="Rewrite manifests to point default fields at suffixed default variants.")
    parser.add_argument("--archive", default=None, help="Create a tar.gz archive containing verified duplicate targets.")
    parser.add_argument("--delete-plan", default=None, help="Delete targets from a previously reviewed plan JSON.")
    args = parser.parse_args()

    cwd = Path.cwd().resolve()
    if args.delete_plan:
        delete_targets_from_plan(require_within_cwd(Path(args.delete_plan), cwd), cwd)
        return 0

    roots = [require_within_cwd(Path(raw), cwd) for raw in args.roots]
    plans = build_plan(roots, cwd)
    payload = {"plans": [plan.to_json() for plan in plans]}
    plan_out = require_within_cwd(Path(args.plan_out), cwd)
    plan_out.parent.mkdir(parents=True, exist_ok=True)
    write_json(plan_out, payload)

    if args.write_manifests:
        write_manifest_updates(plans)
    if args.archive:
        archive_targets(plans, require_within_cwd(Path(args.archive), cwd), cwd)

    print(json.dumps({"plan_out": str(plan_out), "manifests": len(plans), "archive_targets": sum(len(p.archive_targets) for p in plans)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
