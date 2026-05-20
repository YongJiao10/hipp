#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_KONG2022_DIR = REPO_ROOT / "external" / "Schaefer2018_Kong2022"
DEFAULT_KONG2022_PARAMS = DEFAULT_KONG2022_DIR / "Params_Final.mat"
DEFAULT_KONG2022_DLABEL = DEFAULT_KONG2022_DIR / "Schaefer2018_400Parcels_Kong2022_17Networks_order.dlabel.nii"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Route A placeholder for Kong2022 individualized PFM. "
            "This stub validates required assets and exits with NotImplementedError."
        )
    )
    p.add_argument("--subject", required=True)
    p.add_argument("--dtseries", required=True)
    p.add_argument("--left-midthickness", required=True)
    p.add_argument("--right-midthickness", required=True)
    p.add_argument("--left-inflated", required=True)
    p.add_argument("--right-inflated", required=True)
    p.add_argument("--sulc-dscalar", required=True)
    p.add_argument("--out-root", default=str(REPO_ROOT / "outputs_migration" / "cortex_pfm"))
    p.add_argument("--kong2022-params", default=str(DEFAULT_KONG2022_PARAMS))
    p.add_argument("--kong2022-dlabel", default=str(DEFAULT_KONG2022_DLABEL))
    return p.parse_args()


def must_exist(path: Path, label: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Missing required {label}: {path}")


def main() -> int:
    args = parse_args()
    must_exist(Path(args.kong2022_params).resolve(), "Kong2022 Params_Final.mat")
    must_exist(Path(args.kong2022_dlabel).resolve(), "Kong2022 dlabel")

    payload = {
        "status": "not_implemented",
        "atlas_slug": "kong2022",
        "route": "A",
        "reason": (
            "Route A (individualized PFM with Kong2022 priors) is intentionally stubbed in this pass. "
            "Use Route B (public dlabel extraction) for runnable cortex reference generation."
        ),
        "todo_doc": str(
            (
                REPO_ROOT
                / "docs"
                / "hipp_parcellation_network"
                / "merged_canonical_group_atlas_plan.md"
            ).resolve()
        ),
    }
    print(json.dumps(payload, indent=2))
    raise NotImplementedError(
        "run_cortex_pfm_subject_kong2022.py is a Route A stub. "
        "Planned Kong2019->Kong2022 deltas: Params_Final.mat priors, 400 ROI outputs, "
        "subject-specific Kong2022 dlabel generation, and a dedicated individualized extractor."
    )


if __name__ == "__main__":
    raise SystemExit(main())
