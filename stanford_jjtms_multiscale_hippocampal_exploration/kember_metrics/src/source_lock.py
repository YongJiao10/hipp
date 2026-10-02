"""Validate the local checkout of the Kember paper's pinned v1 release."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any


def load_source_lock(path: Path) -> dict[str, Any]:
    """Load a tracked author-source provenance lock."""

    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def validate_author_checkout(lock: dict[str, Any], checkout: Path) -> None:
    """Require every non-archive author file needed by the locked source audit."""

    required_files = lock.get("required_files")
    if not isinstance(required_files, list):
        raise ValueError("required_files must be a list")
    if any(checkout.glob("data/Figure *_extracted")):
        raise ValueError("unapproved Figure archive extraction present")

    for relative_path in required_files:
        candidate = checkout / str(relative_path)
        if not candidate.is_file():
            raise ValueError(f"missing required author file: {relative_path}")

    expected_commit = lock.get("commit")
    if not isinstance(expected_commit, str):
        raise ValueError("commit must be a string")
    resolved = subprocess.run(
        ["git", "-C", str(checkout), "rev-parse", "HEAD"],
        check=False,
        capture_output=True,
        text=True,
    )
    if resolved.returncode != 0:
        raise ValueError("unable to read author checkout commit")
    actual_commit = resolved.stdout.strip()
    if actual_commit != expected_commit:
        raise ValueError(f"unexpected author checkout commit: {actual_commit}")
