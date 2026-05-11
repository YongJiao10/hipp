#!/usr/bin/env python3
from __future__ import annotations

from _launcher import run_with_spec


if __name__ == "__main__":
    raise SystemExit(
        run_with_spec(
            step_id="06_03_materialize_group_labels",
            purpose="Materialize final functional-atlas labels from assignment status artifacts.",
            target_relpath="scripts/hipp_parcellation_network/materialize_group_functional_atlas_labels.py",
        )
    )
