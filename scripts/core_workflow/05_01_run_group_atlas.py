#!/usr/bin/env python3
from __future__ import annotations

from _launcher import run_with_spec


if __name__ == "__main__":
    raise SystemExit(
        run_with_spec(
            step_id="05_01_run_group_atlas",
            purpose="Run merged-canonical hippocampal group-atlas construction.",
            target_relpath="scripts/hipp_parcellation_network/run_merged_canonical_group_atlas.py",
        )
    )
