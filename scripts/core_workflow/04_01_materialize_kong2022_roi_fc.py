#!/usr/bin/env python3
from __future__ import annotations

from _launcher import run_with_spec


if __name__ == "__main__":
    raise SystemExit(
        run_with_spec(
            step_id="04_01_materialize_kong2022_roi_fc",
            purpose="Materialize hippocampal vertex-to-ROI FC caches for Kong2022.",
            target_relpath="scripts/hipp_parcellation_network/materialize_kong2022_vertex_to_roi_fc.py",
        )
    )
