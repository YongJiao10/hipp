from __future__ import annotations

import json
import subprocess
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
    cmd = ["git", "rev-parse", "--short" if short else "HEAD"]
    proc = subprocess.run(cmd, cwd=repo_root, text=True, capture_output=True, check=False)
    if proc.returncode != 0:
        return None
    return proc.stdout.strip()


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
