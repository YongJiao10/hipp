"""Contract tests for HippUnfold shared asset-store enforcement."""

from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
BASE_WRAPPER = REPO_ROOT / "scripts/core_workflow/lib/hippunfold_shell/run_hippunfold_local.sh"
T1W_WRAPPER = REPO_ROOT / "scripts/core_workflow/lib/hippunfold_shell/run_hippunfold_local_t1w_isolated_cache.sh"
CANONICAL_ASSET_ROOT = "/home/yojiao/hpchome/hipp/runtime/hippunfold_asset_store_shared"
CANONICAL_CACHE_DIR = f"{CANONICAL_ASSET_ROOT}/hippunfold_assets"
CANONICAL_RUNTIME_SOURCE = f"{CANONICAL_ASSET_ROOT}/runtime_source_assets"


def test_t1w_wrapper_forces_fixed_shared_asset_store() -> None:
    text = T1W_WRAPPER.read_text(encoding="utf-8")

    assert f'ASSET_ROOT="{CANONICAL_ASSET_ROOT}"' in text
    assert f'export HIPPUNFOLD_CACHE_DIR="{CANONICAL_CACHE_DIR}"' in text
    assert f'export HIPPUNFOLD_RUNTIME_SOURCE_CACHE="{CANONICAL_RUNTIME_SOURCE}"' in text
    assert "HIPPUNFOLD_ASSET_STORE_ROOT:-" not in text
    assert "HIPPUNFOLD_CACHE_DIR:-" not in text
    assert "HIPPUNFOLD_RUNTIME_SOURCE_CACHE:-" not in text


def test_base_wrapper_rejects_noncanonical_cache_overrides() -> None:
    text = BASE_WRAPPER.read_text(encoding="utf-8")

    assert f'CANONICAL_HIPPUNFOLD_CACHE_DIR="{CANONICAL_CACHE_DIR}"' in text
    assert f'CANONICAL_HIPPUNFOLD_RUNTIME_SOURCE_CACHE="{CANONICAL_RUNTIME_SOURCE}"' in text
    assert "Refusing noncanonical HIPPUNFOLD_CACHE_DIR" in text
    assert "Refusing noncanonical HIPPUNFOLD_RUNTIME_SOURCE_CACHE" in text
    assert 'export HIPPUNFOLD_CACHE_DIR="${CANONICAL_HIPPUNFOLD_CACHE_DIR}"' in text
    assert 'export HIPPUNFOLD_RUNTIME_SOURCE_CACHE="${CANONICAL_HIPPUNFOLD_RUNTIME_SOURCE_CACHE}"' in text
    assert 'export HIPPUNFOLD_CACHE_DIR="${HIPPUNFOLD_CACHE_DIR:-${CACHE_ROOT}/hippunfold_cache}"' not in text
    assert 'export HIPPUNFOLD_RUNTIME_SOURCE_CACHE="${HIPPUNFOLD_RUNTIME_SOURCE_CACHE:-${CACHE_ROOT}/runtime_source_cache}"' not in text
