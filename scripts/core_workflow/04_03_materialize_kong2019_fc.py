#!/usr/bin/env python3
from __future__ import annotations

from _launcher import run_with_spec


if __name__ == "__main__":
    raise SystemExit(
        run_with_spec(
            step_id="04_03_materialize_kong2019_fc",
            purpose="Materialize Kong2019/Kong2022 FC banks used by Kong2019 branch runs.",
            target_relpath="scripts/hipp_parcellation_network/materialize_kong7_kong8_pearson_partial_fc.py",
        )
    )
