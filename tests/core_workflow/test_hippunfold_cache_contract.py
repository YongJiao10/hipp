"""Contract tests for the locked HippUnfold shared asset store."""

from __future__ import annotations

from pathlib import Path

from scripts.core_workflow.lib.hippunfold_run_contract import (
    CANONICAL_CACHE_DIR,
    CANONICAL_MODEL_PATH,
    CANONICAL_RUNTIME_SOURCE,
    build_run_environment,
    resolve_run_contract,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
CORE_ENTRY = REPO_ROOT / "scripts/core_workflow/01_01_run_hippunfold_t1w_local.py"


def test_contract_owns_shared_asset_store_and_runtime_root() -> None:
    contract = resolve_run_contract("stanford", "1001")
    env = build_run_environment(contract, base_env={"HIPPUNFOLD_CACHE_DIR": "/tmp/agent-choice"})

    assert env["HIPPUNFOLD_CACHE_DIR"] == CANONICAL_CACHE_DIR
    assert env["HIPPUNFOLD_RUNTIME_SOURCE_CACHE"] == CANONICAL_RUNTIME_SOURCE
    assert env["HIPPUNFOLD_MODEL_PATH"] == CANONICAL_MODEL_PATH
    assert env["HIPPUNFOLD_CACHE_ROOT"] == str(contract.runtime_root)


def test_entrypoint_has_no_positional_path_or_model_override() -> None:
    text = CORE_ENTRY.read_text(encoding="utf-8")

    assert '"--dataset"' in text
    assert '"--dry-run"' in text
    assert 'parser.add_argument("input_dir"' not in text
    assert 'parser.add_argument("out_dir"' not in text
    assert 'parser.add_argument("--model-path"' not in text
    assert "HIPPUNFOLD_EXTRA_ARGS" not in text
    assert CANONICAL_MODEL_PATH not in text
