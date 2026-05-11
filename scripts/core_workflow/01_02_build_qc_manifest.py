#!/usr/bin/env python3
from __future__ import annotations

from _launcher import run_with_spec


if __name__ == "__main__":
    raise SystemExit(
        run_with_spec(
            step_id="01_02_build_qc_manifest",
            purpose="Build the hippocampal QC-pass manifest used by downstream stages.",
            target_relpath="scripts/hipp_parcellation_network/build_t1only_qc_manifest.py",
        )
    )
