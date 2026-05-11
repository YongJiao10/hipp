#!/usr/bin/env python3
from __future__ import annotations

from _launcher import run_with_spec


if __name__ == "__main__":
    raise SystemExit(
        run_with_spec(
            step_id="04_02_materialize_intrinsic_fc",
            purpose="Materialize intrinsic hippocampal vertex-to-vertex FC caches.",
            target_relpath="scripts/hipp_parcellation_network/materialize_intrinsic_vertex_to_vertex_fc.py",
        )
    )
