from __future__ import annotations

import importlib.metadata
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[3]
REPO_CONCURRENCY_LIMIT = 10


def utc_now_iso(*, zulu: bool = False) -> str:
    """Return a repository-standard UTC timestamp for manifests."""
    now = datetime.now(timezone.utc)
    if zulu:
        return now.strftime("%Y-%m-%dT%H:%M:%SZ")
    return now.isoformat(timespec="seconds")


def utc_timestamp_for_filename() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")


def utc_timestamp_compact_z() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def git_commit(*, short: bool = True, repo_root: Path = REPO_ROOT) -> str | None:
    cmd = ["git", "rev-parse", "--short", "HEAD"] if short else ["git", "rev-parse", "HEAD"]
    proc = subprocess.run(cmd, cwd=repo_root, text=True, capture_output=True, check=False)
    if proc.returncode != 0:
        return None
    return proc.stdout.strip()


RUNTIME_VERSION_PACKAGES = ("numpy", "scipy", "scikit-learn", "brainspace", "nibabel")


def runtime_versions() -> dict[str, str]:
    """Installed versions of the numerical stack, for stage manifests."""
    versions = {"python": platform.python_version()}
    for package in RUNTIME_VERSION_PACKAGES:
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = "missing"
    return versions


def code_provenance() -> dict[str, object]:
    """Commit and numerical-stack versions recorded in work-stage manifests."""
    return {"git_commit": git_commit(short=False), "runtime_versions": runtime_versions()}


def option_was_given(option: str, argv: list[str] | None = None) -> bool:
    """Whether a long option appears on the command line, as `--opt value` or `--opt=value`."""
    arguments = sys.argv[1:] if argv is None else argv
    return any(argument.split("=", 1)[0] == option for argument in arguments)


def file_meta(path: Path, *, resolve: bool = True) -> dict[str, object]:
    st = path.stat()
    out_path = path.resolve() if resolve else path
    return {"path": str(out_path), "size": int(st.st_size), "mtime_ns": int(st.st_mtime_ns)}


def file_meta_unresolved(path: Path) -> dict[str, object]:
    return file_meta(path, resolve=False)


def write_json(
    path: Path,
    payload: Any,
    *,
    indent: int = 2,
    sort_keys: bool = False,
    trailing_newline: bool = False,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(payload, indent=indent, sort_keys=sort_keys)
    if trailing_newline:
        text += "\n"
    path.write_text(text, encoding="utf-8")


def read_subject_ids(path: Path) -> list[str]:
    return [line.strip().replace("sub-", "") for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def capped_jobs(requested: int, item_count: int, *, limit: int = REPO_CONCURRENCY_LIMIT) -> int:
    return max(1, min(int(requested), int(limit), int(item_count)))
