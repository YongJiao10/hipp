#!/usr/bin/env python3
"""Safely migrate HCP Kong2022 shared-input directories to canonical names."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import tempfile
from typing import Sequence


DIRECTORY_MAPPINGS: tuple[tuple[str, str], ...] = (
    ("routea_151", "routea"),
    ("surface_151", "surface"),
    ("routeb_151", "routeb"),
    ("routea_ctxreg_151", "routea_ctxreg"),
    ("routea_ctxdrop_151", "routea_ctxdrop"),
)
MANIFEST_NAME = "path_migration_20260711.json"


@dataclass(frozen=True)
class PathMove:
    source: Path
    destination: Path


def build_migration_plan(shared_root: Path) -> list[PathMove]:
    """Return the five ordered shared-input directory moves."""
    root = Path(shared_root)
    return [PathMove(root / old, root / new) for old, new in DIRECTORY_MAPPINGS]


def _preflight(plan: Sequence[PathMove]) -> None:
    problems: list[str] = []
    for move in plan:
        if not move.source.is_dir():
            problems.append(f"source directory is missing: {move.source}")
        if move.destination.exists():
            problems.append(f"destination already exists: {move.destination}")
    if problems:
        raise RuntimeError("migration preflight failed:\n" + "\n".join(problems))


def _write_text_atomically(path: Path, text: str) -> None:
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
        stream.write(text)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary_name, path)


def _manifest(
    repo_root: Path,
    plan: Sequence[PathMove],
) -> dict[str, object]:
    return {
        "status": "ok",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "mappings": [
            {"source": move.source.name, "destination": move.destination.name}
            for move in plan
        ],
        "affected_files": [],
        "replacement_count": 0,
        "validation": {
            "move_count": len(plan),
            "all_sources_absent": all(not move.source.exists() for move in plan),
            "all_destinations_present": all(move.destination.is_dir() for move in plan),
        },
    }


def migrate(repo_root: Path, *, apply: bool) -> dict[str, object]:
    """Preflight a migration, optionally applying it."""
    root = Path(repo_root).resolve()
    shared_root = root / "outputs_hcp" / "inputs" / "hcp" / "kong2022" / "shared_inputs"
    plan = build_migration_plan(shared_root)
    _preflight(plan)

    if not apply:
        return {
            "status": "dry-run",
            "move_count": len(plan),
            "affected_files": [],
            "replacement_count": 0,
        }

    for move in plan:
        move.source.rename(move.destination)

    manifest = _manifest(root, plan)
    if not all(manifest["validation"].values()):
        raise RuntimeError("post-migration validation failed")
    manifest_path = shared_root / MANIFEST_NAME
    _write_text_atomically(manifest_path, json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return manifest


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, required=True)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--dry-run", action="store_true")
    mode.add_argument("--apply", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    try:
        result = migrate(args.repo_root, apply=args.apply)
    except (OSError, RuntimeError) as exc:
        print(str(exc), file=os.sys.stderr)
        return 1
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
