#!/usr/bin/env python3
from __future__ import annotations

from _launcher import run_with_spec


if __name__ == "__main__":
    raise SystemExit(
        run_with_spec(
            step_id="02_01_run_hippunfold_t1w_local",
            purpose="Run HippUnfold for the active T1w workflow surface.",
            target_relpath="scripts/run_hippunfold_local_t1w_isolated_cache.sh",
            target_type="shell",
        )
    )
