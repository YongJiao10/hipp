#!/usr/bin/env python3
"""Standalone entry point for cortical network area analysis."""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.visualize.cortex_network_area.analysis import main


if __name__ == "__main__":
    raise SystemExit(main())
