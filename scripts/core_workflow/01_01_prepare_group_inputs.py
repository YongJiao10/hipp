#!/usr/bin/env python3
from __future__ import annotations

from _launcher import run_with_spec


if __name__ == "__main__":
    raise SystemExit(
        run_with_spec(
            step_id="01_01_prepare_group_inputs",
            purpose="Build shared subject surfaces, QC artifacts, and group-average templates.",
            target_relpath="scripts/hipp_parcellation_network/prepare_t1only_group_inputs.py",
        )
    )
