"""Closed launch contract for the repository's local HippUnfold stage."""

from __future__ import annotations

import os
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Mapping


REPO_ROOT = Path(__file__).resolve().parents[3]
HIPPO2_PREFIX = Path("/home/yojiao/hpchome/hipp/.cache/conda_envs/hippo2")
HIPPUNFOLD_EXECUTABLE = HIPPO2_PREFIX / "bin" / "hippunfold"
CANONICAL_ASSET_ROOT = "/home/yojiao/hpchome/hipp/runtime/hippunfold_asset_store_shared"
CANONICAL_CACHE_DIR = f"{CANONICAL_ASSET_ROOT}/hippunfold_assets"
CANONICAL_RUNTIME_SOURCE = f"{CANONICAL_ASSET_ROOT}/runtime_source_assets"
CANONICAL_MODEL_PATH = (
    f"{CANONICAL_CACHE_DIR}/model/"
    "trained_model.3d_fullres.Task101_hcp1200_T1w.nnUNetTrainerV2.model_best.tar"
)
RUNTIME_WORK_ROOT = REPO_ROOT / "runtime" / "hippunfold_workdirs"
_SAFE_SUBJECT = re.compile(r"^[0-9A-Za-z]+$")


@dataclass(frozen=True)
class DatasetPreset:
    """Dataset paths and the per-dataset N4 preprocessing setting."""

    bids_dir: Path
    output_root: Path
    t1w_filter: str
    skip_preproc: bool = False


DATASET_PRESETS = MappingProxyType({
    "hcp": DatasetPreset(
        bids_dir=REPO_ROOT / "data" / "hippunfold_input",
        output_root=REPO_ROOT
        / "outputs_hcp"
        / "inputs"
        / "hcp"
        / "hippunfold"
        / "dense_corobl_batch",
        t1w_filter="extension=.nii.gz",
        skip_preproc=False,
    ),
    "stanford": DatasetPreset(
        bids_dir=Path(
            "/home/yojiao/LSS/jianglab/3_Data_Working/"
            "fmriprep_processed_CausalConnectome/rest_fmriprep/DataOrg_PFM_36P"
        ),
        output_root=REPO_ROOT
        / "outputs_stanford"
        / "inputs"
        / "stanford"
        / "hippunfold"
        / "dense_corobl_batch",
        t1w_filter="extension=.nii.gz",
        skip_preproc=False,
    ),
    "iowa_jj_tms": DatasetPreset(
        bids_dir=Path(
            "/home/yojiao/LSS/jianglab/3_Data_Working/"
            "fmriprep_processed_JJ_TMS/raw"
        ),
        output_root=REPO_ROOT
        / "outputs_iowa_jj_tms"
        / "inputs"
        / "iowa_jj_tms"
        / "hippunfold"
        / "dense_corobl_batch",
        t1w_filter="extension=.nii",
        skip_preproc=False,
    ),
    "banda": DatasetPreset(
        bids_dir=Path(
            "/home/yojiao/LSS/jianglab/3_Data_Working/Depression/BANDA/bids_input"
        ),
        output_root=REPO_ROOT
        / "outputs_banda"
        / "inputs"
        / "banda"
        / "hippunfold"
        / "dense_corobl_batch",
        t1w_filter="extension=.nii.gz",
        skip_preproc=True,
    ),
    "pdc": DatasetPreset(
        bids_dir=Path(
            "/home/yojiao/LSS/jianglab/3_Data_Working/Depression/PDC/bids_input"
        ),
        output_root=REPO_ROOT
        / "outputs_pdc"
        / "inputs"
        / "pdc"
        / "hippunfold"
        / "dense_corobl_batch",
        t1w_filter="extension=.nii.gz",
        skip_preproc=True,
    ),
})


@dataclass(frozen=True)
class HippUnfoldRunContract:
    dataset: str
    subject: str

    def __post_init__(self) -> None:
        if self.dataset not in DATASET_PRESETS:
            raise ValueError(
                f"Unknown HippUnfold dataset {self.dataset!r}; "
                f"expected one of {tuple(DATASET_PRESETS)}"
            )
        if not _SAFE_SUBJECT.fullmatch(self.subject):
            raise ValueError(
                f"HippUnfold subject must be alphanumeric: {self.subject!r}"
            )

    @property
    def preset(self) -> DatasetPreset:
        return DATASET_PRESETS[self.dataset]

    @property
    def bids_dir(self) -> Path:
        return self.preset.bids_dir

    @property
    def output_dir(self) -> Path:
        return self.preset.output_root / f"sub-{self.subject}" / "hippunfold"

    @property
    def t1w_filter(self) -> str:
        return self.preset.t1w_filter

    @property
    def hippunfold_argv(self) -> tuple[str, ...]:
        argv = (
            str(HIPPUNFOLD_EXECUTABLE),
            str(self.bids_dir),
            str(self.output_dir),
            "participant",
            "--modality",
            "T1w",
            "--output-density",
            "512",
            "--output-spaces",
            "corobl",
            "--autotop_labels",
            "hipp",
            "dentate",
            "--participant-label",
            self.subject,
            "--runtime-source-cache-path",
            CANONICAL_RUNTIME_SOURCE,
            "--workflow-profile",
            "none",
            "--sdm",
            "env-modules",
            "--cores",
            "1",
            "--notemp",
            "--filter-T1w",
            self.t1w_filter,
            "--use_gpu",
        )
        return argv + (("--skip-preproc",) if self.preset.skip_preproc else ())

    @property
    def runtime_root(self) -> Path:
        return RUNTIME_WORK_ROOT / self.dataset / f"sub-{self.subject}"

    def as_json(self) -> dict[str, object]:
        return {
            "dataset": self.dataset,
            "subject": self.subject,
            "bids_dir": str(self.bids_dir),
            "output_dir": str(self.output_dir),
            "modality": "T1w",
            "skip_preproc": self.preset.skip_preproc,
            "output_density": "512",
            "output_spaces": ["corobl"],
            "expected_surface_spaces": ["corobl", "T1w", "unfold"],
            "t1_reg_template": False,
            "unfolded_registration": True,
            "t1w_filter": self.t1w_filter,
            "hippunfold_argv": list(self.hippunfold_argv),
            "runtime_root": str(self.runtime_root),
        }


def resolve_run_contract(dataset: str, subject: str) -> HippUnfoldRunContract:
    return HippUnfoldRunContract(
        dataset=str(dataset).strip().lower(),
        subject=str(subject).strip(),
    )


def _join_path(*values: str) -> str:
    return os.pathsep.join(
        part
        for value in values
        for part in str(value).split(os.pathsep)
        if part
    )


def build_run_environment(
    contract: HippUnfoldRunContract,
    *,
    base_env: Mapping[str, str] | None = None,
) -> dict[str, str]:
    """Build a clean environment with all HippUnfold choices fixed."""
    env = dict(base_env if base_env is not None else os.environ)
    for key in tuple(env):
        if key.startswith("HIPPUNFOLD_"):
            del env[key]
    env.update(
        {
            "HIPPUNFOLD_CACHE_DIR": CANONICAL_CACHE_DIR,
            "HIPPUNFOLD_RUNTIME_SOURCE_CACHE": CANONICAL_RUNTIME_SOURCE,
            "HIPPUNFOLD_MODEL_PATH": CANONICAL_MODEL_PATH,
            "HIPPUNFOLD_CACHE_ROOT": str(contract.runtime_root),
            "XDG_CACHE_HOME": str(contract.runtime_root / "xdg_cache"),
            "PIP_CACHE_DIR": str(contract.runtime_root / "pip_cache"),
            "TMPDIR": str(contract.runtime_root / "tmp"),
            "PATH": _join_path(
                str(REPO_ROOT / "scripts"), str(HIPPO2_PREFIX / "bin"), env.get("PATH", "")
            ),
            "LD_LIBRARY_PATH": _join_path(
                str(HIPPO2_PREFIX / "lib"), env.get("LD_LIBRARY_PATH", "")
            ),
            "OMP_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "VECLIB_MAXIMUM_THREADS": "1",
            "nnUNet_n_proc_DA": "1",
            "KMP_DUPLICATE_LIB_OK": "TRUE",
            "TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD": "1",
        }
    )
    return env


def validate_local_assets() -> None:
    """Require the repository's validated hippo2 CLI before a real run."""
    if not HIPPUNFOLD_EXECUTABLE.is_file():
        raise FileNotFoundError(
            f"Validated HippUnfold executable is missing: {HIPPUNFOLD_EXECUTABLE}"
        )

    resource_adapter = subprocess.run(
        [
            str(HIPPO2_PREFIX / "bin" / "python"),
            "-c",
            "from hippunfold.workflow.lib.local_assets import "
            "template_is_cached, atlas_copy_is_cached, atlas_extract_is_cached, "
            "model_is_unpacked",
        ],
        capture_output=True,
        text=True,
        check=False,
        timeout=30,
    )
    if resource_adapter.returncode != 0:
        raise RuntimeError(
            "hippo2 local-asset adaptation is missing. Apply "
            "scripts/core_workflow/lib/hippunfold_local_assets.patch to the installed "
            "HippUnfold package, as documented in "
            "docs/environment/hippo2_reproducibility_manifest.md. "
            f"stderr={resource_adapter.stderr.strip()!r}"
        )

    activation = Path("/opt/conda/bin/activate")
    which = subprocess.run(
        [
            "/bin/bash",
            "-lc",
            f"source {activation} hippo2 >/dev/null 2>&1 && which hippunfold",
        ],
        capture_output=True,
        text=True,
        check=False,
        timeout=30,
    )
    resolved = Path(which.stdout.strip()) if which.stdout.strip() else None
    if which.returncode != 0 or resolved is None or resolved.resolve() != HIPPUNFOLD_EXECUTABLE.resolve():
        raise RuntimeError(
            "hippo2 gate failed: `which hippunfold` did not resolve to "
            f"{HIPPUNFOLD_EXECUTABLE}; stdout={which.stdout.strip()!r}; "
            f"stderr={which.stderr.strip()!r}"
        )

    version = subprocess.run(
        [str(resolved), "--version"],
        capture_output=True,
        text=True,
        check=False,
        timeout=30,
    )
    version_text = f"{version.stdout} {version.stderr}"
    if version.returncode != 0 or not re.search(r"\b2\.0\.0\b", version_text):
        raise RuntimeError(
            "hippo2 gate failed: HippUnfold CLI must report 2.0.0; "
            f"stdout={version.stdout.strip()!r}; stderr={version.stderr.strip()!r}"
        )

    help_run = subprocess.run(
        [str(resolved), "--modality", "T1w", "--output_density", "512", "--help"],
        capture_output=True,
        text=True,
        check=False,
        timeout=30,
    )
    help_text = f"{help_run.stdout} {help_run.stderr}"
    if help_run.returncode != 0 or "512" not in help_text:
        raise RuntimeError(
            "hippo2 gate failed: T1w/512 help probe did not succeed; "
            f"stdout={help_run.stdout.strip()!r}; stderr={help_run.stderr.strip()!r}"
        )
