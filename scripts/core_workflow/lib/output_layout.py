from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]
OUTPUTS_CORE_ROOT = REPO_ROOT / "outputs_core" / "hcp7t_hippocampus"
RUNS_ROOT = OUTPUTS_CORE_ROOT / "runs"
DEFAULT_MANUAL_RUN_ID = "manual"
DEFAULT_MANUAL_RUN_ROOT = RUNS_ROOT / DEFAULT_MANUAL_RUN_ID
DEFAULT_PRESENT_ROOT = DEFAULT_MANUAL_RUN_ROOT / "present"
DEFAULT_OVERVIEWS_DIR = DEFAULT_PRESENT_ROOT / "overviews"
DEFAULT_GRADIENTS_DIR = DEFAULT_PRESENT_ROOT / "gradients"
DEFAULT_CORTEX_PANELS_DIR = DEFAULT_PRESENT_ROOT / "cortex_panels"


def timestamped_run_id(prefix: str = "smoke5") -> str:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    clean_prefix = str(prefix).strip().replace("/", "_") or "run"
    return f"{clean_prefix}_{stamp}"


def run_root(run_id: str) -> Path:
    clean = str(run_id).strip()
    if not clean or "/" in clean or clean in {".", ".."}:
        raise ValueError(f"Invalid run_id: {run_id!r}")
    return RUNS_ROOT / clean


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
