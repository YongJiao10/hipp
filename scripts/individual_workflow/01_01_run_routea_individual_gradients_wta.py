#!/usr/bin/env python3
"""Numbered entrypoint for the shared RouteA individual gradient/WTA runner."""

from __future__ import annotations

import sys
from pathlib import Path


LIB_DIR = Path(__file__).resolve().parent / "lib"
if str(LIB_DIR) not in sys.path:
    sys.path.insert(0, str(LIB_DIR))

from routea_individual_gradients_wta import main  # noqa: E402


if __name__ == "__main__":
    raise SystemExit(main())
