#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path
import platform
import shutil
import subprocess
import sys


def which(path: str) -> str | None:
    return shutil.which(path)


def run_text(cmd: list[str]) -> str:
    proc = subprocess.run(cmd, text=True, capture_output=True)
    if proc.returncode != 0:
        return ""
    return proc.stdout.strip()


def main() -> int:
    wb = which("wb_command")
    if wb is None:
        app_wb = Path("/Applications/wb_view.app/Contents/usr/bin/wb_command")
        wb = str(app_wb) if app_wb.exists() else None

    wb_arch_test = ""
    if wb is not None:
        cmd = [wb, "-help"]
        if platform.system() == "Darwin" and platform.machine() == "arm64":
            cmd = ["arch", "-x86_64"] + cmd
        wb_arch_test = run_text(cmd)

    report = {
        "platform": run_text(["uname", "-a"]),
        "architecture": run_text(["uname", "-m"]),
        "commands": {
            "python3": which("python3"),
            "wb_command": wb,
            "docker": which("docker"),
            "hippunfold": which("hippunfold"),
            "conda": which("conda"),
        },
        "wb_command_help_prefix": wb_arch_test.splitlines()[0] if wb_arch_test else "",
        "notes": [],
    }

    if report["commands"]["docker"] is None:
        report["notes"].append("docker missing")
    if report["commands"]["hippunfold"] is None:
        report["notes"].append("hippunfold missing")
    if report["commands"]["wb_command"] is None:
        report["notes"].append("wb_command missing")
    elif platform.system() == "Darwin" and platform.machine() == "arm64":
        report["notes"].append("wb_command should be invoked via arch -x86_64 on this machine")

    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
