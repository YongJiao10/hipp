#!/usr/bin/env python3
from __future__ import annotations

from _launcher import run_with_spec


if __name__ == "__main__":
    raise SystemExit(
        run_with_spec(
            step_id="03_01_build_kong2022_reference",
            purpose="Build cortex Kong2022 references for the shared active workflow.",
            target_relpath="scripts/cortex/run_cortex_kong2022_reference_batch.py",
        )
    )
