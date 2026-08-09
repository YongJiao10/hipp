import importlib.util
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
CHECKER_PATH = REPO_ROOT / "scripts/core_workflow/01_02_check_hippunfold_outputs.py"


def load_checker():
    sys.path.insert(0, str(REPO_ROOT / "scripts/core_workflow"))
    spec = importlib.util.spec_from_file_location("hippunfold_output_checker", CHECKER_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_checker_accepts_hippunfold_2_completion_logs() -> None:
    checker = load_checker()

    assert checker.is_successful_log("56 of 56 steps (100%) done")
    assert checker.is_successful_log("41 of 41 steps (100%) done")
    assert checker.is_successful_log(
        "Nothing to be done (all requested files are present and up to date)."
    )
    assert not checker.is_successful_log(
        "WorkflowError: At least one job did not complete successfully."
    )
