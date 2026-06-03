from __future__ import annotations

from pathlib import Path

from .workflow_utils import utc_timestamp_for_filename


REPO_ROOT = Path(__file__).resolve().parents[3]
OUTPUTS_CORE_ROOT = REPO_ROOT / "outputs_core"
DEFAULT_INPUTS_ROOT = OUTPUTS_CORE_ROOT / "inputs" / "hcp7t_hippocampus"
DEFAULT_HIPPUNFOLD_BATCH_ROOT = DEFAULT_INPUTS_ROOT / "hippunfold" / "dense_corobl_batch"
DEFAULT_QC_ROOT = DEFAULT_INPUTS_ROOT / "qc" / "vertex_qc_t1only"
DEFAULT_KONG2022_ROOT = DEFAULT_INPUTS_ROOT / "kong2022"
DEFAULT_KONG2022_SHARED_INPUTS_ROOT = DEFAULT_KONG2022_ROOT / "shared_inputs"
DEFAULT_KONG2022_SURFACE_ROOT = DEFAULT_KONG2022_SHARED_INPUTS_ROOT / "surface_151"
DEFAULT_KONG2022_ROUTEA_ROOT = DEFAULT_KONG2022_SHARED_INPUTS_ROOT / "routea_151"
DEFAULT_KONG2022_ROUTEA_CTXREG_ROOT = DEFAULT_KONG2022_SHARED_INPUTS_ROOT / "routea_ctxreg_151"
DEFAULT_KONG2022_ROUTEA_CTXDROP_ROOT = DEFAULT_KONG2022_SHARED_INPUTS_ROOT / "routea_ctxdrop_151"
DEFAULT_KONG2022_ROUTEB_ROOT = DEFAULT_KONG2022_SHARED_INPUTS_ROOT / "routeb_151"
DEFAULT_KONG2022_TEMPLATE_ROOT = DEFAULT_KONG2022_ROOT / "templates" / "hipp_group_average_t1only_151"
DEFAULT_KONG2022_MANIFEST_ROOT = DEFAULT_KONG2022_ROOT / "manifests"
DEFAULT_KONG2022_QC_ROOT = DEFAULT_KONG2022_ROOT / "qc"
DEFAULT_CORTEX_ROUTEA_ROOT = DEFAULT_KONG2022_ROOT / "cortex_references" / "routea_areal_mshbm"
DEFAULT_KONG2019_ROOT = DEFAULT_INPUTS_ROOT / "kong2019_fast_pfm_t1only"
DEFAULT_WORK_ROOT = OUTPUTS_CORE_ROOT / "work"
DEFAULT_GROUP_ATLAS_ROOT = DEFAULT_WORK_ROOT / "group_atlas"
DEFAULT_ROUTEA_GROUP_ATLAS_ROOT = DEFAULT_GROUP_ATLAS_ROOT / "routea_full_k3_fisherz"
DEFAULT_ROUTEA_17NET_GROUP_ATLAS_ROOT = DEFAULT_GROUP_ATLAS_ROOT / "routea_17net_full_k3_fisherz"
DEFAULT_ROUTEA_CTXREG_GROUP_ATLAS_ROOT = DEFAULT_GROUP_ATLAS_ROOT / "routea_ctxreg_full_k3_fisherz"
DEFAULT_ROUTEA_CTXDROP_GROUP_ATLAS_ROOT = DEFAULT_GROUP_ATLAS_ROOT / "routea_ctxdrop_full_k3_fisherz"
DEFAULT_ROUTEB_GROUP_ATLAS_ROOT = DEFAULT_GROUP_ATLAS_ROOT / "routeb_full_k3_fisherz"
DEFAULT_HIPPOMAPS_GRADIENT_ROOT = DEFAULT_WORK_ROOT / "hippomaps_gradients"
DEFAULT_CORTEX_ASSETS_ROOT = DEFAULT_WORK_ROOT / "cortex_pfm"
SMOKE_RUNS_ROOT = OUTPUTS_CORE_ROOT / "smoke_runs"
RERUNS_ROOT = OUTPUTS_CORE_ROOT / "reruns"
DEFAULT_PRESENT_ROOT = OUTPUTS_CORE_ROOT / "present"
DEFAULT_OVERVIEWS_DIR = DEFAULT_PRESENT_ROOT / "overviews"
DEFAULT_GRADIENTS_DIR = DEFAULT_PRESENT_ROOT / "gradients"
DEFAULT_CORTEX_PANELS_DIR = DEFAULT_PRESENT_ROOT / "cortex_panels"
DEFAULT_LOCAL_CORTEX_NEIGHBORS_DIR = DEFAULT_PRESENT_ROOT / "local_cortex_neighbors"


def timestamped_run_id(prefix: str = "smoke5") -> str:
    stamp = utc_timestamp_for_filename()
    clean_prefix = str(prefix).strip().replace("/", "_") or "run"
    return f"{clean_prefix}_{stamp}"


def run_root(run_id: str) -> Path:
    clean = str(run_id).strip()
    if not clean or "/" in clean or clean in {".", ".."}:
        raise ValueError(f"Invalid run_id: {run_id!r}")
    return SMOKE_RUNS_ROOT / clean


def run_layout(run_id: str) -> dict[str, Path]:
    base = run_root(run_id)
    present = base / "present"
    return {
        "run_root": base,
        "inputs": base / "inputs",
        "logs": base / "logs",
        "manifests": base / "manifests",
        "hippunfold": base / "hippunfold",
        "features": base / "features",
        "group_atlas": base / "group_atlas",
        "gradients": base / "gradients",
        "present": present,
        "overviews": present / "overviews",
        "cortex_panels": present / "cortex_panels",
        "gradient_products": present / "gradients",
    }
