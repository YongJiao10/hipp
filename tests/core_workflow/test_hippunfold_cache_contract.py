"""Contract tests for HippUnfold shared asset-store enforcement."""

from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
CORE_ENTRY = REPO_ROOT / "scripts/core_workflow/01_01_run_hippunfold_t1w_local.py"
BASE_WRAPPER = REPO_ROOT / "scripts/core_workflow/lib/hippunfold_shell/run_hippunfold_local.sh"
T1W_WRAPPER = REPO_ROOT / "scripts/core_workflow/lib/hippunfold_shell/run_hippunfold_local_t1w_isolated_cache.sh"
CANONICAL_ASSET_ROOT = "/home/yojiao/hpchome/hipp/runtime/hippunfold_asset_store_shared"
CANONICAL_CACHE_DIR = f"{CANONICAL_ASSET_ROOT}/hippunfold_assets"
CANONICAL_RUNTIME_SOURCE = f"{CANONICAL_ASSET_ROOT}/runtime_source_assets"
CANONICAL_MODEL_PATH = (
    f"{CANONICAL_CACHE_DIR}/model/"
    "trained_model.3d_fullres.Task101_hcp1200_T1w.nnUNetTrainerV2.model_best.tar"
)


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


def test_core_entry_exposes_explicit_model_path() -> None:
    text = CORE_ENTRY.read_text(encoding="utf-8")

    assert 'parser.add_argument("--model-path"' in text
    assert CANONICAL_MODEL_PATH in text
    assert 'str(args.model_path)' in text


def test_t1w_wrapper_passes_and_reports_model_path() -> None:
    text = T1W_WRAPPER.read_text(encoding="utf-8")

    assert 'MODEL_PATH="${4:-' in text
    assert 'export HIPPUNFOLD_MODEL_PATH="${MODEL_PATH}"' in text


def test_base_wrapper_requires_local_model_and_omits_remote_model_rule() -> None:
    text = BASE_WRAPPER.read_text(encoding="utf-8")

    assert f'CANONICAL_HIPPUNFOLD_MODEL_PATH="{CANONICAL_MODEL_PATH}"' in text
    assert 'if [[ ! -f "${HIPPUNFOLD_MODEL_PATH}" ]]' in text
    assert 'HippUnfold model path:' in text
    assert '--omit-from' in text
    assert 'download_nnunet_model' in text
    assert 'HIPPUNFOLD_T1W_FILTER="${HIPPUNFOLD_T1W_FILTER:-}"' in text
    assert 'cmd+=(--filter-T1w "${HIPPUNFOLD_T1W_FILTER}")' in text
    assert '--filter-T1w extension=.nii' not in text


def test_base_wrapper_retains_runtime_cache_by_default() -> None:
    text = BASE_WRAPPER.read_text(encoding="utf-8")

    assert 'HIPPUNFOLD_CLEANUP_RUNTIME="${HIPPUNFOLD_CLEANUP_RUNTIME:-0}"' in text
    assert 'if [[ "${HIPPUNFOLD_CLEANUP_RUNTIME}" == "1" ]]' in text
    assert 'echo "run_hippunfold_local.sh: Retaining runtime cache:' in text
