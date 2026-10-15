from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .workflow_utils import utc_timestamp_for_filename


REPO_ROOT = Path(__file__).resolve().parents[3]

DATASET_KEYS = (
    "hcp",
    "iowa",
    "iowa_estt",
    "iowa_jj_tms",
    "stanford",
    "iowa_hcp500",
    "banda",
    "pdc",
)
DEFAULT_DATASET = "hcp"

IOWA7T_BRANCHES = ("multiband", "multi_echo")
IOWA7T_VARIANTS = ("0p", "22p", "32p", "36p", "noxcpd")
LOCAL_JIANGLAB_ROOT = Path("/home/yojiao/LSS/jianglab")
ARGON_JIANGLAB_ROOT = Path("/Shared/jianglab")
IOWA7T_0P_RELATIVE_ROOT = Path(
    "3_Data_Working/fmriprep_processed_Iowa7T/DataOrg_PFM_0P"
)
IOWA7T_22P_RELATIVE_ROOT = Path(
    "3_Data_Working/fmriprep_processed_Iowa7T/DataOrg_PFM_22P"
)
IOWA7T_32P_RELATIVE_ROOT = Path(
    "3_Data_Working/fmriprep_processed_Iowa7T/DataOrg_PFM_32P"
)
IOWA7T_36P_RELATIVE_ROOT = Path(
    "3_Data_Working/fmriprep_processed_Iowa7T/DataOrg_PFM_36P"
)
IOWA7T_NOXCPD_RELATIVE_ROOT = Path(
    "3_Data_Working/fmriprep_processed_Iowa7T/DataOrg_PFM_NoXCPD"
)
IOWA7T_VARIANT_RELATIVE_ROOTS = {
    "0p": IOWA7T_0P_RELATIVE_ROOT,
    "22p": IOWA7T_22P_RELATIVE_ROOT,
    "32p": IOWA7T_32P_RELATIVE_ROOT,
    "36p": IOWA7T_36P_RELATIVE_ROOT,
    "noxcpd": IOWA7T_NOXCPD_RELATIVE_ROOT,
}


@dataclass(frozen=True)
class Iowa7TBranchLayout:
    subject: str
    branch: str
    root: Path
    cortex: Path
    surface: Path
    fc: Path
    atlas: Path
    qc: Path
    figures: Path
    provenance: Path


@dataclass(frozen=True)
class Iowa7TSubjectLayout:
    subject: str
    subject_root: Path
    hippunfold_root: Path

    def branch_root(self, branch: str) -> Path:
        return self.subject_root / validate_iowa7t_branch(branch)


def validate_iowa7t_subject(subject: str) -> str:
    raw = str(subject).strip()
    subject_id = raw.removeprefix("sub-")
    if not (len(subject_id) == 8 and subject_id.isdigit()):
        raise ValueError(
            f"Invalid Iowa7T subject: {subject!r}; "
            "expected MMDDYYYY digits with optional sub- prefix"
        )
    return subject_id


def validate_iowa7t_branch(branch: str) -> str:
    key = str(branch).strip().lower().replace("-", "_")
    if key not in IOWA7T_BRANCHES:
        raise ValueError(
            f"Invalid Iowa7T branch: {branch!r}; expected one of {IOWA7T_BRANCHES}"
        )
    return key


def validate_iowa7t_variant(variant: str) -> str:
    key = str(variant).strip().lower().replace("-", "_")
    if key not in IOWA7T_VARIANTS:
        raise ValueError(
            f"Invalid Iowa7T variant: {variant!r}; expected one of {IOWA7T_VARIANTS}"
        )
    return key


def resolve_jianglab_root(*, require_exists: bool = True) -> Path:
    if LOCAL_JIANGLAB_ROOT.exists():
        return LOCAL_JIANGLAB_ROOT
    if ARGON_JIANGLAB_ROOT.exists():
        return ARGON_JIANGLAB_ROOT
    if require_exists:
        raise FileNotFoundError(
            "Jianglab root not found. Local uses /home/yojiao/LSS/jianglab; "
            "Argon uses /Shared/jianglab."
        )
    return LOCAL_JIANGLAB_ROOT


def resolve_iowa7t_data_root(
    *,
    variant: str = "36p",
    jianglab_root: Path | None = None,
    require_exists: bool = True,
) -> Path:
    variant_key = validate_iowa7t_variant(variant)
    root = (
        Path(jianglab_root)
        if jianglab_root is not None
        else resolve_jianglab_root(require_exists=require_exists)
    )
    return root / IOWA7T_VARIANT_RELATIVE_ROOTS[variant_key]


def resolve_iowa7t_36p_root(
    *,
    jianglab_root: Path | None = None,
    require_exists: bool = True,
) -> Path:
    return resolve_iowa7t_data_root(
        variant="36p",
        jianglab_root=jianglab_root,
        require_exists=require_exists,
    )


def resolve_iowa7t_hipp_outputs_root(
    *,
    variant: str = "36p",
    jianglab_root: Path | None = None,
    require_exists: bool = True,
) -> Path:
    return (
        resolve_iowa7t_data_root(
            variant=variant,
            jianglab_root=jianglab_root,
            require_exists=require_exists,
        )
        / "hipp_outputs"
    )


def iowa7t_subject_layout(
    subject: str,
    *,
    variant: str = "36p",
    hipp_outputs_root: Path | None = None,
) -> Iowa7TSubjectLayout:
    subject_id = validate_iowa7t_subject(subject)
    root = (
        Path(hipp_outputs_root)
        if hipp_outputs_root is not None
        else resolve_iowa7t_hipp_outputs_root(variant=variant)
    )
    subject_root = root / "subjects" / f"sub-{subject_id}"
    return Iowa7TSubjectLayout(
        subject=subject_id,
        subject_root=subject_root,
        hippunfold_root=subject_root / "hippunfold",
    )


def iowa7t_branch_layout(
    subject: str,
    branch: str,
    *,
    variant: str = "36p",
    hipp_outputs_root: Path | None = None,
) -> Iowa7TBranchLayout:
    subject_layout = iowa7t_subject_layout(
        subject,
        variant=variant,
        hipp_outputs_root=hipp_outputs_root,
    )
    branch_key = validate_iowa7t_branch(branch)
    root = subject_layout.branch_root(branch_key)
    return Iowa7TBranchLayout(
        subject=subject_layout.subject,
        branch=branch_key,
        root=root,
        cortex=root / "cortex",
        surface=root / "surface",
        fc=root / "fc",
        atlas=root / "atlas",
        qc=root / "qc",
        figures=root / "figures",
        provenance=root / "provenance.json",
    )



def _validate_dataset(dataset: str) -> str:
    """Normalize and validate a dataset key."""
    key = str(dataset).strip().lower()
    if key == "iowa":
        key = "iowa_estt"
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


KONG2022_VARIANTS = ("36p", "22p")


def _validate_kong2022_variant(variant: str) -> str:
    key = str(variant).strip().lower().replace("-", "_")
    if key not in KONG2022_VARIANTS:
        raise ValueError(
            f"Invalid Kong2022 variant: {variant!r}; expected one of {KONG2022_VARIANTS}"
        )
    return key


def resolve_kong2022_root(dataset: str = DEFAULT_DATASET) -> Path:
    return resolve_inputs_root(dataset) / "kong2022"


def resolve_kong2022_variant_root(
    dataset: str = DEFAULT_DATASET,
    variant: str = "36p",
) -> Path:
    dataset_key = _validate_dataset(dataset)
    variant_key = _validate_kong2022_variant(variant)
    if variant_key == "36p":
        return resolve_inputs_root(dataset_key) / "kong2022"
    if dataset_key == "stanford":
        return resolve_inputs_root(dataset_key) / "kong2022_22p"
    if dataset_key == "iowa_jj_tms":
        return resolve_inputs_root(dataset_key) / "kong2022"
    raise ValueError(
        f"Kong2022 {variant_key} has no supported native input layout for "
        f"dataset={dataset_key!r}"
    )


def resolve_kong2022_shared_inputs_root(
    dataset: str = DEFAULT_DATASET,
    variant: str = "36p",
) -> Path:
    dataset_key = _validate_dataset(dataset)
    root = resolve_kong2022_variant_root(dataset_key, variant)
    if _validate_kong2022_variant(variant) == "22p" and dataset_key == "iowa_jj_tms":
        return root / "shared_inputs_22p"
    return root / "shared_inputs"


def resolve_kong2022_surface_root(
    dataset: str = DEFAULT_DATASET,
    variant: str = "36p",
) -> Path:
    return resolve_kong2022_shared_inputs_root(dataset, variant) / "surface"


def resolve_kong2022_routea_root(
    dataset: str = DEFAULT_DATASET,
    variant: str = "36p",
) -> Path:
    return resolve_kong2022_shared_inputs_root(dataset, variant) / "routea"


def resolve_kong2022_routea_ctxreg_root(dataset: str = DEFAULT_DATASET) -> Path:
    return resolve_kong2022_shared_inputs_root(dataset) / "routea_ctxreg"


def resolve_kong2022_routea_ctxdrop_root(dataset: str = DEFAULT_DATASET) -> Path:
    return resolve_kong2022_shared_inputs_root(dataset) / "routea_ctxdrop"


def resolve_kong2022_routeb_root(dataset: str = DEFAULT_DATASET) -> Path:
    return resolve_kong2022_shared_inputs_root(dataset) / "routeb"


def resolve_kong2022_template_root(
    dataset: str = DEFAULT_DATASET,
    variant: str = "36p",
) -> Path:
    dataset_key = _validate_dataset(dataset)
    variant_key = _validate_kong2022_variant(variant)
    if variant_key == "22p" and dataset_key == "stanford":
        return (
            resolve_kong2022_shared_inputs_root(dataset_key, variant_key)
            / "template"
        )
    return (
        resolve_kong2022_variant_root(dataset_key, variant_key)
        / "templates"
        / "hipp_group_average_t1only_151"
    )


def resolve_kong2022_manifest_root(dataset: str = DEFAULT_DATASET) -> Path:
    return resolve_kong2022_root(dataset) / "manifests"


def resolve_kong2022_qc_root(dataset: str = DEFAULT_DATASET) -> Path:
    return resolve_kong2022_root(dataset) / "qc"


def resolve_cortex_routea_root(dataset: str = DEFAULT_DATASET) -> Path:
    return resolve_kong2022_root(dataset) / "cortex_references" / "routea_areal_mshbm"


def resolve_lynch2024_root(dataset: str = DEFAULT_DATASET) -> Path:
    return resolve_inputs_root(dataset) / "lynch2024_fast_pfm_t1only"


def resolve_lynch2024_manifest_root(dataset: str = DEFAULT_DATASET) -> Path:
    return resolve_lynch2024_root(dataset) / "manifests"


def resolve_lynch2024_shared_inputs_base(dataset: str = DEFAULT_DATASET) -> Path:
    return resolve_lynch2024_root(dataset) / "shared_inputs"


def resolve_lynch2024_shared_inputs_root(dataset: str = DEFAULT_DATASET) -> Path:
    """Return the current all-base-subject Lynch2024 shared-input root."""
    return resolve_lynch2024_shared_inputs_base(dataset) / "pfm_hcp151"


def resolve_lynch2024_nsi043_shared_inputs_root(dataset: str = DEFAULT_DATASET) -> Path:
    """Return the historical 78-subject input root in the legacy namespace."""
    return resolve_lynch2024_shared_inputs_base(dataset) / "pfm_nsi043_78"


# ── work subtree resolvers ────────────────────────────────────────────

def resolve_group_atlas_root(dataset: str = DEFAULT_DATASET) -> Path:
    return resolve_work_root(dataset) / "group_atlas"


def resolve_routea_group_atlas_root(dataset: str = DEFAULT_DATASET) -> Path:
    return resolve_group_atlas_root(dataset) / "routea_full_k3_fisherz"



def resolve_routea_17net_group_atlas_root(dataset: str = DEFAULT_DATASET) -> Path:
    return resolve_group_atlas_root(dataset) / "routea_17net_full_k3_fisherz"


def resolve_routea_ctxreg_group_atlas_root(dataset: str = DEFAULT_DATASET) -> Path:
    dataset_key = _validate_dataset(dataset)
    if dataset_key == "hcp":
        return resolve_group_atlas_root(dataset_key) / "routea_ctxreg"
    return resolve_group_atlas_root(dataset_key) / "routea_ctxreg_full_k3_fisherz"


def resolve_routea_ctxdrop_group_atlas_root(dataset: str = DEFAULT_DATASET) -> Path:
    dataset_key = _validate_dataset(dataset)
    if dataset_key == "hcp":
        return resolve_group_atlas_root(dataset_key) / "routea_ctxdrop"
    return resolve_group_atlas_root(dataset_key) / "routea_ctxdrop_full_k3_fisherz"



def resolve_routeb_group_atlas_root(dataset: str = DEFAULT_DATASET) -> Path:
    return resolve_group_atlas_root(dataset) / "routeb_full_k3_fisherz"


def resolve_lynch2024_20net_group_atlas_root(dataset: str = DEFAULT_DATASET) -> Path:
    return resolve_group_atlas_root(dataset) / "lynch2024_20net_hcp151_full_k3_fisherz"


def resolve_lynch2024_7net_group_atlas_root(dataset: str = DEFAULT_DATASET) -> Path:
    return resolve_group_atlas_root(dataset) / "lynch2024_7net_hcp151_full_k3_fisherz"


def resolve_hippomaps_gradient_root(dataset: str = DEFAULT_DATASET) -> Path:
    return resolve_work_root(dataset) / "hippomaps_gradients"


def resolve_cortex_assets_root(dataset: str = DEFAULT_DATASET) -> Path:
    return resolve_work_root(dataset) / "cortex_pfm"


def resolve_crossfit_template_matching_root(dataset: str = DEFAULT_DATASET) -> Path:
    return resolve_work_root(dataset) / "crossfit_template_matching"


# ── present subtree resolvers ─────────────────────────────────────────

def resolve_overviews_dir(dataset: str = DEFAULT_DATASET) -> Path:
    return resolve_present_root(dataset) / "overviews"


def resolve_gradients_dir(dataset: str = DEFAULT_DATASET) -> Path:
    return resolve_present_root(dataset) / "gradients"


def resolve_cortex_panels_dir(dataset: str = DEFAULT_DATASET) -> Path:
    return resolve_present_root(dataset) / "cortex_panels"


def resolve_local_cortex_neighbors_dir(dataset: str = DEFAULT_DATASET) -> Path:
    return resolve_present_root(dataset) / "local_cortex_neighbors"


def resolve_crossfit_template_matching_overview_root(dataset: str = DEFAULT_DATASET) -> Path:
    return resolve_overviews_dir(dataset) / "crossfit_template_matching"


def resolve_crossfit_template_matching_individual_overview_root(
    dataset: str = DEFAULT_DATASET,
) -> Path:
    """Return the canonical per-subject cross-fit overview directory."""
    return resolve_crossfit_template_matching_overview_root(dataset) / "individual"


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
DEFAULT_LYNCH2024_ROOT = resolve_lynch2024_root()
DEFAULT_LYNCH2024_MANIFEST_ROOT = resolve_lynch2024_manifest_root()
DEFAULT_LYNCH2024_SHARED_INPUTS_BASE = resolve_lynch2024_shared_inputs_base()
DEFAULT_LYNCH2024_SHARED_INPUTS_ROOT = resolve_lynch2024_shared_inputs_root()
DEFAULT_WORK_ROOT = resolve_work_root()
DEFAULT_GROUP_ATLAS_ROOT = resolve_group_atlas_root()
DEFAULT_ROUTEA_GROUP_ATLAS_ROOT = resolve_routea_group_atlas_root()
DEFAULT_ROUTEA_17NET_GROUP_ATLAS_ROOT = resolve_routea_17net_group_atlas_root()
DEFAULT_ROUTEA_CTXREG_GROUP_ATLAS_ROOT = resolve_routea_ctxreg_group_atlas_root()
DEFAULT_ROUTEA_CTXDROP_GROUP_ATLAS_ROOT = resolve_routea_ctxdrop_group_atlas_root()
DEFAULT_ROUTEB_GROUP_ATLAS_ROOT = resolve_routeb_group_atlas_root()
DEFAULT_LYNCH2024_20NET_GROUP_ATLAS_ROOT = resolve_lynch2024_20net_group_atlas_root()
DEFAULT_LYNCH2024_7NET_GROUP_ATLAS_ROOT = resolve_lynch2024_7net_group_atlas_root()
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
KONG400_CONTROL_ROUTES = ("routea_ctxreg", "routea_ctxdrop")




def resolve_hcp_kong400_control_shared_root(
    route: str,
    dataset: str = DEFAULT_DATASET,
) -> Path:
    """Return the formal HCP 400-ROI cortex-control shared-input branch."""
    route_key = str(route).strip().lower().replace("-", "_")
    if route_key not in KONG400_CONTROL_ROUTES:
        raise ValueError(
            f"Unsupported Kong400 control route: {route!r}; expected one of {KONG400_CONTROL_ROUTES}"
        )
    dataset_key = _validate_dataset(dataset)
    if dataset_key != "hcp":
        raise ValueError(
            "The formal Kong400 control roots are only defined for dataset='hcp'"
        )
    return resolve_kong2022_shared_inputs_root(dataset_key) / route_key


DEFAULT_KONG400_CTXREG_SHARED_ROOT = resolve_hcp_kong400_control_shared_root("routea_ctxreg")
DEFAULT_KONG400_CTXDROP_SHARED_ROOT = resolve_hcp_kong400_control_shared_root("routea_ctxdrop")
