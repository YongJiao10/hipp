#!/usr/bin/env python3
from __future__ import annotations

import os
import subprocess
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_WB_COMMAND = REPO_ROOT / "software" / "workbench" / "bin_linux64" / "wb_command"
ENV_WB_COMMAND = "CORE_WORKFLOW_WB_COMMAND"


def resolve_wb_command(override: str | Path | None = None) -> Path:
    candidate: Path
    if override is not None:
        candidate = Path(override).expanduser().resolve()
    else:
        env_value = os.environ.get(ENV_WB_COMMAND, "").strip()
        candidate = Path(env_value).expanduser().resolve() if env_value else DEFAULT_WB_COMMAND.resolve()
    if not candidate.exists():
        raise FileNotFoundError(
            f"Workbench command not found: {candidate}\n"
            f"Expected default: {DEFAULT_WB_COMMAND}\n"
            f"Override with ${ENV_WB_COMMAND} or resolve_wb_command(override=...)."
        )
    if not os.access(candidate, os.X_OK):
        raise PermissionError(f"Workbench command is not executable: {candidate}")
    return candidate


def run_workbench(
    args: list[str],
    *,
    env: dict[str, str] | None = None,
    cwd: Path | None = None,
) -> subprocess.CompletedProcess[str]:
    cmd = [str(resolve_wb_command()), *args]
    proc = subprocess.run(
        cmd,
        cwd=str(cwd) if cwd is not None else None,
        env=env,
        text=True,
        capture_output=True,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            f"wb_command failed ({proc.returncode}): {' '.join(cmd)}\n"
            f"STDOUT:\n{proc.stdout}\n"
            f"STDERR:\n{proc.stderr}"
        )
    return proc
