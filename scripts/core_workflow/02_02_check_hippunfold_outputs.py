#!/usr/bin/env python3
from __future__ import annotations

from _launcher import run_with_spec


if __name__ == "__main__":
    raise SystemExit(
        run_with_spec(
            step_id="02_02_check_hippunfold_outputs",
            purpose="Validate HippUnfold outputs before downstream cortex/FC stages.",
            target_relpath="scripts/check_hippunfold_output_integrity.py",
        )
    )
