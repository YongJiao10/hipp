#!/usr/bin/env python3
"""Retired locked-scene group gradient heatmaps; use the maintained 06_04 step."""
from __future__ import annotations

import argparse
import json
import sys


def main() -> int:
    if any(arg in {"-h", "--help"} for arg in sys.argv[1:]):
        argparse.ArgumentParser(description=__doc__).parse_args()
        return 0
    print(json.dumps({
        "status": "deprecated",
        "step": "06_05_render_group_gradient_heatmaps",
        "message": "This legacy locked-scene heatmap output is retired and must not be refreshed.",
        "replacement": "Use scripts/core_workflow/06_04_render_hippomaps_gradient_products.py.",
    }, indent=2), flush=True)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
