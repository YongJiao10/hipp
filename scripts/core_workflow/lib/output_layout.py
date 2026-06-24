from __future__ import annotations

from pathlib import Path

from .workflow_utils import utc_timestamp_for_filename


REPO_ROOT = Path(__file__).resolve().parents[3]

DATASET_KEYS = ("hcp", "iowa", "stanford")
DEFAULT_DATASET = "hcp"


def _validate_dataset(dataset: str) -> str:
    """Normalize and validate a dataset key."""
    key = str(dataset).strip().lower()
    if key not in DATASET_KEYS:
        raise ValueError(
            f"Unknown dataset: {dataset!r}; expected one of {DATASET_KEYS}"
        )
    return key


# ── dataset-scoped roots ──────────────────────────────────────────────

def resolve_outputs_root(dataset: str = DEFAULT_DATASET) -> Path:
    """Return the dataset-scoped outputs root directory (e.g. outputs_hcp)."""
    return REPO_ROOT / f"outputs_{_validate_dataset(dataset)}"


def resolve_inputs_root(dataset: str = DEFAULT_DATASET) -> Path:
    """Return the dataset-scoped inputs root (e.g. outputs_hcp/inputs/hcp)."""
    return resolve_outputs_root(dataset) / "inputs" / _validate_dataset(dataset)


def resolve_work_root(dataset: str = DEFAULT_DATASET) -> Path:
    return resolve_outputs_root(dataset) / "work"


def resolve_present_root(dataset: str = DEFAULT_DATASET) -> Path:
    return resolve_outputs_root(dataset) / "present"


def resolve_reruns_root(dataset: str = DEFAULT_DATASET) -> Path:
    return resolve_outputs_root(dataset) / "reruns"


def resolve_smoke_runs_root(dataset: str = DEFAULT_DATASET) -> Path:
    return resolve_outputs_root(dataset) / "smoke_runs"


# ── inputs subtree resolvers ──────────────────────────────────────────

def resolve_hippunfold_batch_root(dataset: str = DEFAULT_DATASET) -> Path:
    return resolve_inputs_root(dataset) / "hippunfold" / "dense_corobl_batch"


def resolve_qc_root(dataset: str = DEFAULT_DATASET) -> Path:
    return resolve_inputs_root(dataset) / "qc" / "vertex_qc_t1only"


def resolve_kong2022_root(dataset: str = DEFAULT_DATASET) -> Path:
    return resolve_inputs_root(dataset) / "kong2022"


def resolve_kong2022_shared_inputs_root(dataset: str = DEFAULT_DATASET) -> Path:
    return resolve_kong2022_root(dataset) / "shared_inputs"


def resolve_kong2022_surface_root(dataset: str = DEFAULT_DATASET) -> Path:
    return resolve_kong2022_shared_inputs_root(dataset) / "surface"


def resolve_kong2022_routea_root(dataset: str = DEFAULT_DATASET) -> Path:
    return resolve_kong2022_shared_inputs_root(dataset) / "routea"


def resolve_kong2022_routea_ctxreg_root(dataset: str = DEFAULT_DATASET) -> Path:
    return resolve_kong2022_shared_inputs_root(dataset) / "routea_ctxreg_151"


def resolve_kong2022_routea_ctxdrop_root(dataset: str = DEFAULT_DATASET) -> Path:
    return resolve_kong2022_shared_inputs_root(dataset) / "routea_ctxdrop_151"


def resolve_kong2022_routeb_root(dataset: str = DEFAULT_DATASET) -> Path:
    return resolve_kong2022_shared_inputs_root(dataset) / "routeb_151"


def resolve_kong2022_template_root(dataset: str = DEFAULT_DATASET) -> Path:
    return resolve_kong2022_root(dataset) / "templates" / "hipp_group_average_t1only_151"


def resolve_kong2022_manifest_root(dataset: str = DEFAULT_DATASET) -> Path:
    return resolve_kong2022_root(dataset) / "manifests"


def resolve_kong2022_qc_root(dataset: str = DEFAULT_DATASET) -> Path:
    return resolve_kong2022_root(dataset) / "qc"


def resolve_cortex_routea_root(dataset: str = DEFAULT_DATASET) -> Path:
    return resolve_kong2022_root(dataset) / "cortex_references" / "routea_areal_mshbm"


def resolve_kong2019_root(dataset: str = DEFAULT_DATASET) -> Path:
    return resolve_inputs_root(dataset) / "kong2019_fast_pfm_t1only"


# ── work subtree resolvers ────────────────────────────────────────────

def resolve_group_atlas_root(dataset: str = DEFAULT_DATASET) -> Path:
    return resolve_work_root(dataset) / "group_atlas"


def resolve_routea_group_atlas_root(dataset: str = DEFAULT_DATASET) -> Path:
    return resolve_group_atlas_root(dataset) / "routea_full_k3_fisherz"


def resolve_routea_17net_group_atlas_root(dataset: str = DEFAULT_DATASET) -> Path:
    return resolve_group_atlas_root(dataset) / "routea_17net_full_k3_fisherz"


def resolve_routea_ctxreg_group_atlas_root(dataset: str = DEFAULT_DATASET) -> Path:
    return resolve_group_atlas_root(dataset) / "routea_ctxreg_full_k3_fisherz"


def resolve_routea_ctxdrop_group_atlas_root(dataset: str = DEFAULT_DATASET) -> Path:
    return resolve_group_atlas_root(dataset) / "routea_ctxdrop_full_k3_fisherz"


def resolve_routeb_group_atlas_root(dataset: str = DEFAULT_DATASET) -> Path:
    return resolve_group_atlas_root(dataset) / "routeb_full_k3_fisherz"


def resolve_hippomaps_gradient_root(dataset: str = DEFAULT_DATASET) -> Path:
    return resolve_work_root(dataset) / "hippomaps_gradients"


def resolve_cortex_assets_root(dataset: str = DEFAULT_DATASET) -> Path:
    return resolve_work_root(dataset) / "cortex_pfm"


# ── present subtree resolvers ─────────────────────────────────────────

def resolve_overviews_dir(dataset: str = DEFAULT_DATASET) -> Path:
    return resolve_present_root(dataset) / "overviews"


def resolve_gradients_dir(dataset: str = DEFAULT_DATASET) -> Path:
    return resolve_present_root(dataset) / "gradients"


def resolve_cortex_panels_dir(dataset: str = DEFAULT_DATASET) -> Path:
    return resolve_present_root(dataset) / "cortex_panels"


def resolve_local_cortex_neighbors_dir(dataset: str = DEFAULT_DATASET) -> Path:
    return resolve_present_root(dataset) / "local_cortex_neighbors"


# ── backward-compatible module-level constants (resolved for DEFAULT_DATASET) ──

OUTPUTS_ROOT = resolve_outputs_root()
DEFAULT_INPUTS_ROOT = resolve_inputs_root()
DEFAULT_HIPPUNFOLD_BATCH_ROOT = resolve_hippunfold_batch_root()
DEFAULT_QC_ROOT = resolve_qc_root()
DEFAULT_KONG2022_ROOT = resolve_kong2022_root()
DEFAULT_KONG2022_SHARED_INPUTS_ROOT = resolve_kong2022_shared_inputs_root()
DEFAULT_KONG2022_SURFACE_ROOT = resolve_kong2022_surface_root()
DEFAULT_KONG2022_ROUTEA_ROOT = resolve_kong2022_routea_root()
DEFAULT_KONG2022_ROUTEA_CTXREG_ROOT = resolve_kong2022_routea_ctxreg_root()
DEFAULT_KONG2022_ROUTEA_CTXDROP_ROOT = resolve_kong2022_routea_ctxdrop_root()
DEFAULT_KONG2022_ROUTEB_ROOT = resolve_kong2022_routeb_root()
DEFAULT_KONG2022_TEMPLATE_ROOT = resolve_kong2022_template_root()
DEFAULT_KONG2022_MANIFEST_ROOT = resolve_kong2022_manifest_root()
DEFAULT_KONG2022_QC_ROOT = resolve_kong2022_qc_root()
DEFAULT_CORTEX_ROUTEA_ROOT = resolve_cortex_routea_root()
DEFAULT_KONG2019_ROOT = resolve_kong2019_root()
DEFAULT_WORK_ROOT = resolve_work_root()
DEFAULT_GROUP_ATLAS_ROOT = resolve_group_atlas_root()
DEFAULT_ROUTEA_GROUP_ATLAS_ROOT = resolve_routea_group_atlas_root()
DEFAULT_ROUTEA_17NET_GROUP_ATLAS_ROOT = resolve_routea_17net_group_atlas_root()
DEFAULT_ROUTEA_CTXREG_GROUP_ATLAS_ROOT = resolve_routea_ctxreg_group_atlas_root()
DEFAULT_ROUTEA_CTXDROP_GROUP_ATLAS_ROOT = resolve_routea_ctxdrop_group_atlas_root()
DEFAULT_ROUTEB_GROUP_ATLAS_ROOT = resolve_routeb_group_atlas_root()
DEFAULT_HIPPOMAPS_GRADIENT_ROOT = resolve_hippomaps_gradient_root()
DEFAULT_CORTEX_ASSETS_ROOT = resolve_cortex_assets_root()
SMOKE_RUNS_ROOT = resolve_smoke_runs_root()
RERUNS_ROOT = resolve_reruns_root()
DEFAULT_PRESENT_ROOT = resolve_present_root()
DEFAULT_OVERVIEWS_DIR = resolve_overviews_dir()
DEFAULT_GRADIENTS_DIR = resolve_gradients_dir()
DEFAULT_CORTEX_PANELS_DIR = resolve_cortex_panels_dir()
DEFAULT_LOCAL_CORTEX_NEIGHBORS_DIR = resolve_local_cortex_neighbors_dir()


# ── run helpers ───────────────────────────────────────────────────────

def timestamped_run_id(prefix: str = "smoke5") -> str:
    stamp = utc_timestamp_for_filename()
    clean_prefix = str(prefix).strip().replace("/", "_") or "run"
    return f"{clean_prefix}_{stamp}"


def run_root(run_id: str, dataset: str = DEFAULT_DATASET) -> Path:
    clean = str(run_id).strip()
    if not clean or "/" in clean or clean in {".", ".."}:
        raise ValueError(f"Invalid run_id: {run_id!r}")
    return resolve_smoke_runs_root(dataset) / clean


def run_layout(run_id: str, dataset: str = DEFAULT_DATASET) -> dict[str, Path]:
    base = run_root(run_id, dataset=dataset)
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
