from __future__ import annotations

import argparse
import filecmp
import json
import shutil
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.core_workflow.lib.output_layout import (
    resolve_iowa7t_hipp_outputs_root,
    validate_iowa7t_subject,
)
from scripts.core_workflow.lib.workflow_utils import utc_timestamp_for_filename


@dataclass(frozen=True)
class Mapping:
    source: Path
    destination: Path
    kind: str

    def to_json(self) -> dict[str, str]:
        return {key: str(value) for key, value in asdict(self).items()}


def _is_migration_payload(path: Path) -> bool:
    if any(part == ".ipynb_checkpoints" for part in path.parts):
        return False
    if path.name == ".DS_Store" or path.name.startswith("._"):
        return False
    return path.is_file() or path.is_symlink()


def _iter_payload_files(root: Path) -> list[Path]:
    if not root.exists():
        return []
    if root.is_file() or root.is_symlink():
        return [root] if _is_migration_payload(root) else []
    return sorted(path for path in root.rglob("*") if _is_migration_payload(path))


def _map_tree(source_root: Path, destination_root: Path) -> list[Mapping]:
    return [
        Mapping(
            source=path,
            destination=destination_root / path.relative_to(source_root),
            kind="symlink" if path.is_symlink() else "file",
        )
        for path in _iter_payload_files(source_root)
    ]


def build_mappings(hipp_outputs_root: Path, subjects: Iterable[str]) -> list[Mapping]:
    root = Path(hipp_outputs_root)
    mappings: list[Mapping] = []
    for raw_subject in subjects:
        subject = validate_iowa7t_subject(raw_subject)
        sub = f"sub-{subject}"
        mappings.extend(
            _map_tree(
                root / "inputs/hippunfold/dense_corobl_batch" / sub / "hippunfold",
                root / "subjects" / sub / "hippunfold",
            )
        )
        mappings.extend(
            _map_tree(
                root / "inputs/kong2022/shared_inputs/surface" / sub / "surface",
                root / "subjects" / sub / "single_echo" / "surface",
            )
        )
        mappings.extend(
            _map_tree(
                root / "inputs/kong2022/shared_inputs/routea" / sub / "reference",
                root / "subjects" / sub / "single_echo" / "cortex",
            )
        )
        mappings.extend(
            _map_tree(
                root / "inputs/kong2022/shared_inputs/routea" / sub / "fc",
                root / "subjects" / sub / "single_echo" / "fc",
            )
        )
        mappings.extend(
            _map_tree(
                root / "work/individual_atlas" / sub / "gradient",
                root / "subjects" / sub / "single_echo" / "atlas" / "gradient",
            )
        )
        mappings.extend(
            _map_tree(
                root / "work/individual_atlas" / sub / "wta",
                root / "subjects" / sub / "single_echo" / "atlas" / "wta",
            )
        )
        mappings.extend(
            _map_tree(
                root / "present/overviews/individual/gradient",
                root / "subjects" / sub / "single_echo" / "figures" / "gradient",
            )
        )
        mappings.extend(
            _map_tree(
                root / "present/overviews/individual/wta",
                root / "subjects" / sub / "single_echo" / "figures" / "wta",
            )
        )
    mappings.extend(
        _map_tree(
            root / "work/individual_atlas/_overview_core_adapter",
            root / "runs/cache/overview_core_adapter",
        )
    )
    mappings.extend(_map_tree(root / "work/individual_atlas/_runs", root / "runs"))
    mappings.extend(
        _map_tree(
            root / "inputs/kong2022/templates/hipp_group_average_t1only_151",
            root / "group/hippocampal_template",
        )
    )
    mappings.extend(_map_tree(root / "inputs/kong2022/qc", root / "group/qc"))
    mappings.extend(
        _map_tree(root / "inputs/kong2022/manifests", root / "group/manifests")
    )
    return sorted(mappings, key=lambda mapping: str(mapping.destination))


def _same_payload(source: Path, destination: Path) -> bool:
    if source.is_symlink() or destination.is_symlink():
        return source.resolve(strict=True) == destination.resolve(strict=True)
    return source.stat().st_size == destination.stat().st_size and filecmp.cmp(
        source,
        destination,
        shallow=False,
    )


def validate_mappings(mappings: list[Mapping]) -> None:
    seen: dict[Path, Path] = {}
    for mapping in mappings:
        if mapping.destination in seen and seen[mapping.destination] != mapping.source:
            raise FileExistsError(
                "Multiple sources map to one destination: "
                f"{seen[mapping.destination]} and {mapping.source} -> "
                f"{mapping.destination}"
            )
        seen[mapping.destination] = mapping.source
        if mapping.destination.exists() and not _same_payload(
            mapping.source,
            mapping.destination,
        ):
            raise FileExistsError(
                f"Destination exists with different content: {mapping.destination}"
            )


def apply_mappings(
    mappings: list[Mapping],
    *,
    mode: str,
    write_manifest: bool = True,
    manifest_root: Path | None = None,
) -> Path | None:
    if mode not in {"dry-run", "copy"}:
        raise ValueError(f"Unsupported mode: {mode}")
    validate_mappings(mappings)
    manifest_path: Path | None = None
    if write_manifest and manifest_root is not None:
        manifest_root.mkdir(parents=True, exist_ok=True)
        manifest_path = (
            manifest_root
            / f"iowa7t_layout_migration_{utc_timestamp_for_filename()}_{mode}.json"
        )
        manifest_path.write_text(
            json.dumps([m.to_json() for m in mappings], indent=2) + "\n",
            encoding="utf-8",
        )
    if mode == "dry-run":
        return manifest_path
    for mapping in mappings:
        if mapping.destination.exists():
            continue
        mapping.destination.parent.mkdir(parents=True, exist_ok=True)
        if mapping.source.is_symlink():
            mapping.destination.symlink_to(mapping.source.resolve(strict=True))
        else:
            shutil.copy2(mapping.source, mapping.destination)
    return manifest_path


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Migrate Iowa7T 36P hippocampal outputs to subject-first layout."
    )
    parser.add_argument(
        "--hipp-outputs-root",
        type=Path,
        default=resolve_iowa7t_hipp_outputs_root(),
    )
    parser.add_argument(
        "--subjects",
        default="04172026",
        help="Comma-separated subject IDs",
    )
    parser.add_argument("--mode", choices=("dry-run", "copy"), default="dry-run")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    subjects = tuple(s.strip() for s in args.subjects.split(",") if s.strip())
    mappings = build_mappings(args.hipp_outputs_root, subjects=subjects)
    manifest_root = args.hipp_outputs_root / "runs" / "manifests"
    manifest = apply_mappings(
        mappings,
        mode=args.mode,
        write_manifest=True,
        manifest_root=manifest_root,
    )
    print(
        json.dumps(
            {
                "mode": args.mode,
                "mapping_count": len(mappings),
                "manifest": str(manifest) if manifest else None,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
