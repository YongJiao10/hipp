# HippUnfold Shared Asset Store Guard Design

## Goal

Prevent future HippUnfold runs in this repository from using per-run model caches
or downloading duplicate nnUNet/template assets. All local HippUnfold runs must
use the project-level shared asset store.

## Fixed Paths

The canonical local shared asset store is fixed:

```text
/home/yojiao/hpchome/hipp/runtime/hippunfold_asset_store_shared
```

The only valid HippUnfold cache paths are:

```text
/home/yojiao/hpchome/hipp/runtime/hippunfold_asset_store_shared/hippunfold_assets
/home/yojiao/hpchome/hipp/runtime/hippunfold_asset_store_shared/runtime_source_assets
```

## Behavior

`run_hippunfold_local_t1w_isolated_cache.sh` must export those fixed paths and
must not allow `HIPPUNFOLD_CACHE_DIR`, `HIPPUNFOLD_RUNTIME_SOURCE_CACHE`, or
`HIPPUNFOLD_ASSET_STORE_ROOT` from the caller to redirect model assets.

`run_hippunfold_local.sh` must fail closed if a caller directly passes a
different `HIPPUNFOLD_CACHE_DIR` or `HIPPUNFOLD_RUNTIME_SOURCE_CACHE`. If the
variables are unset, the script sets them to the fixed shared paths.

Per-run scratch such as `XDG_CACHE_HOME`, `TMPDIR`, `PIP_CACHE_DIR`, and
`HIPPUNFOLD_CACHE_ROOT` may remain run-specific. Reusable model/template assets
must not be run-specific.

## Verification

Tests must assert that the wrappers contain the fixed shared asset paths, reject
or ignore external cache overrides, and do not default `HIPPUNFOLD_CACHE_DIR` to
`${CACHE_ROOT}/hippunfold_cache`.
