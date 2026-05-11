#!/usr/bin/env python3
from __future__ import annotations

from _launcher import run_with_spec


if __name__ == "__main__":
    raise SystemExit(
        run_with_spec(
            step_id="03_02_build_kong2022_routea_reference",
            purpose="Build cortex reference assets for the Kong2022 RouteA branch.",
            target_relpath="scripts/cortex/run_cortex_kong2022_route_a_reference_batch.py",
        )
    )
