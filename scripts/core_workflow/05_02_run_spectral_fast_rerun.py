#!/usr/bin/env python3
from __future__ import annotations

from _launcher import run_with_spec


if __name__ == "__main__":
    raise SystemExit(
        run_with_spec(
            step_id="05_02_run_spectral_fast_rerun",
            purpose="Run urgent spectral rerun orchestration across active combos.",
            target_relpath="scripts/hipp_parcellation_network/run_t1only_spectral_fast_rerun.py",
        )
    )
