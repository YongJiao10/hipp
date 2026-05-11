#!/usr/bin/env python3
from __future__ import annotations

from _launcher import run_with_spec


if __name__ == "__main__":
    raise SystemExit(
        run_with_spec(
            step_id="06_04_build_cluster_cortex_panels",
            purpose="Build cortex/hippocampal cluster or clusterless overview figure panels.",
            target_relpath="scripts/hipp_parcellation_network/build_cluster_roi_cortex_independent.py",
        )
    )
