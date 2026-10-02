#!/usr/bin/env python3
from __future__ import annotations

"""
Process Step Description:
Materialize group-level labels. Converts group-level clustering results into standard functional network label files (e.g., GIFTI format) for visualization and downstream analysis.

Main Flow Path:
- 9th step in the numbered pipeline sequence (first stage of presentation output generation, run in parallel with 06_01, 06_03, 06_04, 06_05, 06_06, 06_07):
  01_01_run_hippunfold_t1w_local.py -> 01_02_check_hippunfold_outputs.py -> 02_01_prepare_group_inputs.py -> 02_02_build_qc_manifest.py -> 03_01/03_02 reference building -> 04_01/04_02 FC materialization -> 05_01 gradient computation -> 05_02 group atlas parcellation -> 06_02_materialize_group_labels.py (this step) -> 07_01 group matrix overview.

Inputs:
- Command-line arguments:
  - --out-root: Root directory containing group atlas results (default: `outputs_hcp/work/group_atlas`).
- Physical inputs:
  - Subdirectory manifests: `{out_root}/**/group_atlas_manifest.json`
  - From subdirectories containing the manifest:
    - Group-level parcellation clusters: `group_labels_final.npy`
    - Cluster assignment status: manifest `assignment_variants[<variant>].cluster_assignment_status`
    - Group feature summary: `group_feature_summary.json`
  - Style configuration files:
    - `config/hipp_network_style.json`
  
Outputs:
- File outputs:
  - Group functional atlas label arrays and maps written to suffixed variant files such as `group_functional_atlas_labels_fisherz.npy` and `group_functional_atlas_label_map_fisherz.json` for cluster-level methods.
  - Vertex-level methods keep their single manifest-declared label array and label map.
  - Standard GIFTI label image: `*label-group_functional_atlas_*.surf.gii` in template assets.
  - Updated manifest file: `group_atlas_manifest.json` updated in-place with new label keys. Every cluster-level assignment variant must declare its exact in-run status, NPY, label-map, and GIFTI paths; missing or conflicting assets fail closed and no fallback filename is generated.
"""
import argparse
import json
from pathlib import Path

from lib.output_layout import DEFAULT_ROUTEA_GROUP_ATLAS_ROOT
from lib.surface_utils import network_color_map
from lib.gifti_labels import materialize_combo

DEFAULT_OUT_ROOT = DEFAULT_ROUTEA_GROUP_ATLAS_ROOT


def main() -> int:
    parser = argparse.ArgumentParser(description="Backfill group functional atlas label products for merged-canonical outputs.")
    parser.add_argument("--out-root", default=str(DEFAULT_OUT_ROOT))
    args = parser.parse_args()

    root = Path(args.out_root).resolve()
    manifests = sorted(root.rglob("group_atlas_manifest.json"))
    cmap = network_color_map()
    n_done = 0
    for m in manifests:
        try:
            if materialize_combo(m, cmap):
                n_done += 1
        except Exception as e:  # pragma: no cover
            print(f"[warn] failed {m}: {e}", flush=True)
    print(json.dumps({"out_root": str(root), "manifests": len(manifests), "updated": int(n_done)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
