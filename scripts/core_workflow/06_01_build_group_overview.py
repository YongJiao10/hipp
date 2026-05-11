#!/usr/bin/env python3
from __future__ import annotations

from _launcher import run_with_spec


if __name__ == "__main__":
    raise SystemExit(
        run_with_spec(
            step_id="06_01_build_group_overview",
            purpose="Render group overview products for atlas outputs.",
            target_relpath="scripts/hipp_parcellation_network/build_present_group_overview_merged_canonical.py",
        )
    )
