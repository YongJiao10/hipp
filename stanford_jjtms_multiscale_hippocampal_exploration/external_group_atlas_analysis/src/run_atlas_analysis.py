"""Compatibility CLI for the Stanford group-atlas geometry family.

The authoritative execution path is the unified
cross_dataset_rest_fc/src/run_primary.py run. This entry point is compatibility
only and may rerun atlas diagnostics in isolation; it is not an experimental
branch scheduler or an authoritative result path.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from atlas_compare import run_analysis


def main() -> None:
    package_root = Path(__file__).resolve().parents[1]
    repo_root = package_root.parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-root",
        type=Path,
        default=repo_root / "outputs_stanford" / "work" / "group_atlas",
    )
    parser.add_argument("--analysis-root", type=Path, default=package_root)
    parser.add_argument("--run-id", default="atlas_descriptive")
    args = parser.parse_args()
    paths = run_analysis(input_root=args.input_root, analysis_root=args.analysis_root, run_id=args.run_id)
    print(f"completed run: {args.run_id}")
    for name, path in paths.items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()
