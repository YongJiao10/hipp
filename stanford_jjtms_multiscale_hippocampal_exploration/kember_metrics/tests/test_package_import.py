"""Public package-import contracts for the unified Kember workflow."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]


def test_run_feature_build_imports_through_public_package_path() -> None:
    environment = os.environ.copy()
    environment.pop("PYTHONPATH", None)
    command = (
        "from stanford_jjtms_multiscale_hippocampal_exploration."
        "kember_metrics.src.feature_build import run_feature_build; "
        "print(run_feature_build.__name__)"
    )

    completed = subprocess.run(
        [sys.executable, "-c", command],
        cwd=REPO_ROOT,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    assert completed.stdout.strip() == "run_feature_build"
