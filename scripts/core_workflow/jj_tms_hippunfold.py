#!/home/yojiao/.conda/envs/py314/bin/python
"""Fixed JJ_TMS HippUnfold 2.0 corobl rerun and acceptance contract.

The module deliberately keeps the subject roster and expected outputs outside
the current output tree.  Existing output files can therefore never expand or
shrink the set of required subjects or redefine what "complete" means.
"""

from __future__ import annotations

import hashlib
import nibabel as nib
import json
import os
import re
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


REPO_ROOT = Path(__file__).resolve().parents[2]
PYTHON_EXE = Path("/home/yojiao/.conda/envs/py314/bin/python")
CORE_ENTRY = REPO_ROOT / "scripts/core_workflow/01_01_run_hippunfold_t1w_local.py"
DAG_PREFLIGHT_ENTRY = REPO_ROOT / "scripts/core_workflow/preflight_jj_tms_hippunfold_dag.py"
DAG_PREFLIGHT_SUBJECT = "1001"
DEFAULT_INPUT_ROOT = Path(
    "/home/yojiao/LSS/jianglab/3_Data_Working/"
    "fmriprep_processed_JJ_TMS/raw"
)
DEFAULT_BATCH_ROOT = (
    REPO_ROOT
    / "outputs_iowa_jj_tms/inputs/iowa_jj_tms/hippunfold/dense_corobl_batch"
)
DEFAULT_SUBJECTS_FILE = REPO_ROOT / "manifests/jj_tms_hippunfold_subjects.txt"
DEFAULT_RUN_ROOT_NAME = "jj_tms_hippunfold_full_corobl"
CANONICAL_MODEL_PATH = Path(
    "/home/yojiao/hpchome/hipp/runtime/hippunfold_asset_store_shared/"
    "hippunfold_assets/model/"
    "trained_model.3d_fullres.Task101_hcp1200_T1w.nnUNetTrainerV2.model_best.tar"
)
CANONICAL_MODEL_DIR = Path(str(CANONICAL_MODEL_PATH).removesuffix(".tar"))
CANONICAL_CACHE_DIR = Path(
    "/home/yojiao/hpchome/hipp/runtime/hippunfold_asset_store_shared/"
    "hippunfold_assets"
)
CANONICAL_RUNTIME_SOURCE_CACHE = Path(
    "/home/yojiao/hpchome/hipp/runtime/hippunfold_asset_store_shared/"
    "runtime_source_assets"
)
WORKDIR_ROOT = REPO_ROOT / "runtime/hippunfold_workdirs"
EXPECTED_SUBJECT_COUNT = 70
CONCURRENCY = 4
OUTPUT_SPACES = ("corobl",)
HEMIS = ("L", "R")
LABELS = ("hipp", "dentate")
EXPECTED_SURFACE_VERTICES = {"hipp": 512, "dentate": 128}
SURFACE_NAMES = ("midthickness", "inner", "outer")
SPEC_SUFFIXES = (
    "den-512_surfaces.spec",
    "den-512_label-hipp_surfaces.spec",
    "den-512_label-dentate_surfaces.spec",
    "hemi-L_den-512_label-hipp_surfaces.spec",
    "hemi-R_den-512_label-hipp_surfaces.spec",
    "hemi-L_den-512_label-dentate_surfaces.spec",
    "hemi-R_den-512_label-dentate_surfaces.spec",
)

SUCCESS_MARKER = re.compile(r"Finished jobid:\s*0\s+\(Rule:\s*all\)")
FAILURE_PATTERNS = (
    "SIGINT",
    "RuleException",
    "At least one job did not complete successfully",
    "Error in rule",
    "WorkflowError",
    "MissingOutputException",
    "CalledProcessError",
    "non-zero exit",
)
REMOTE_RULE = "download_nnunet_model"
LOCAL_WORKFLOW_MARKER = "HippUnfold local-model workflow:"
REQUIRED_DAG_RULE_COUNTS = {
    "gen_native_mesh": 4,
    "resample_native_surf_to_atlas_density": 12,
    "create_spec_file": 4,
}
DOWNSTREAM_SURFACE_REF = re.compile(
    r"[^\s\"']*space-(?:unfold|corobl)_den-512[^\s\"']*"
)


def normalize_subject(subject: str) -> str:
    """Return a bare numeric subject ID, rejecting ambiguous values."""

    normalized = subject.strip()
    if normalized.startswith("sub-"):
        normalized = normalized[4:]
    if not normalized or not normalized.isdigit():
        raise ValueError(f"Invalid JJ_TMS subject ID: {subject!r}")
    return normalized


def read_subject_manifest(path: Path = DEFAULT_SUBJECTS_FILE) -> list[str]:
    """Read the explicit fixed roster; never derive it from HippUnfold outputs."""

    if not path.is_file():
        raise FileNotFoundError(f"Subject manifest not found: {path}")
    subjects: list[str] = []
    for line_number, raw_line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        try:
            subject = normalize_subject(line)
        except ValueError as exc:
            raise ValueError(f"Invalid subject manifest line {line_number}: {exc}") from exc
        subjects.append(subject)
    if len(subjects) != len(set(subjects)):
        duplicates = sorted({s for s in subjects if subjects.count(s) > 1})
        raise ValueError(f"Duplicate subjects in manifest: {duplicates}")
    return subjects


def read_production_subject_manifest(path: Path | None = None) -> list[str]:
    """Require the requested production roster to equal the fixed manifest."""

    canonical = read_subject_manifest(DEFAULT_SUBJECTS_FILE)
    requested = read_subject_manifest(path or DEFAULT_SUBJECTS_FILE)
    if requested != canonical:
        raise ValueError(
            "production subject manifest does not match the canonical fixed roster exactly"
        )
    return requested


def expected_surface_paths(subject: str) -> list[str]:
    """Return every required corobl den-512 surface relative to batch root."""

    subject = normalize_subject(subject)
    return [
        (
            f"sub-{subject}/hippunfold/sub-{subject}/surf/"
            f"sub-{subject}_hemi-{hemi}_space-corobl_den-512_"
            f"label-{label}_{surface}.surf.gii"
        )
        for hemi in HEMIS
        for label in LABELS
        for surface in SURFACE_NAMES
    ]


def expected_spec_paths(subject: str) -> list[str]:
    """Return the seven den-512 surface specification files HippUnfold emits."""

    subject = normalize_subject(subject)
    return [
        f"sub-{subject}/hippunfold/sub-{subject}/surf/sub-{subject}_{suffix}"
        for suffix in SPEC_SUFFIXES
    ]


def _subject_output_root(batch_root: Path, subject: str) -> Path:
    return batch_root / f"sub-{normalize_subject(subject)}" / "hippunfold"


def _latest_log(output_root: Path, run_started_at: float | None) -> Path | None:
    log_root = output_root / ".snakemake" / "log"
    if not log_root.is_dir():
        return None
    logs = [p for p in log_root.glob("*.snakemake.log") if p.is_file()]
    if run_started_at is not None:
        recent = [p for p in logs if p.stat().st_mtime >= run_started_at - 5.0]
        logs = recent
    if not logs:
        return None
    return max(logs, key=lambda p: (p.stat().st_mtime, p.name))


def _read_log_info(output_root: Path, run_started_at: float | None) -> dict[str, Any]:
    log_path = _latest_log(output_root, run_started_at)
    if log_path is None:
        return {
            "path": None,
            "all_success": False,
            "all_target_seen": False,
            "failure_markers": ["missing current snakemake log"],
        }
    text = log_path.read_text(encoding="utf-8", errors="ignore")
    failures = [pattern for pattern in FAILURE_PATTERNS if pattern in text]
    return {
        "path": str(log_path),
        "all_success": bool(SUCCESS_MARKER.search(text)) and not failures,
        "all_target_seen": "Rule: all" in text or "localrule all:" in text,
        "failure_markers": failures,
    }


def _read_overlay_info(snakefile_value: str | None) -> dict[str, Any]:
    errors: list[str] = []
    if not snakefile_value:
        return {
            "valid": False,
            "snakefile": None,
            "manifest_path": None,
            "model_path": None,
            "errors": ["missing --snakefile value"],
        }

    snakefile = Path(snakefile_value)
    manifest_path = snakefile.parent.parent / "local_model_overlay.json"
    payload: dict[str, Any] = {}
    if not snakefile.is_file():
        errors.append(f"overlay Snakefile is missing: {snakefile}")
    if not manifest_path.is_file():
        errors.append(f"overlay manifest is missing: {manifest_path}")
    else:
        try:
            payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as exc:
            errors.append(f"overlay manifest is unreadable: {exc}")

    expected = {
        "status": "ready",
        "model_path": str(CANONICAL_MODEL_PATH),
        "snakefile": str(snakefile),
        "remote_rule": "removed from overlay",
        "local_unpack_rule": "retained",
        "model_binding": "explicit get_model_tar return",
    }
    for key, value in expected.items():
        if payload.get(key) != value:
            label = "canonical model" if key == "model_path" else key
            errors.append(f"overlay {label} mismatch: {payload.get(key)!r}")
    source_root_value = payload.get("source_root")
    source_nnunet = (
        Path(source_root_value) / "workflow/rules/nnunet.smk"
        if isinstance(source_root_value, str) and source_root_value
        else None
    )
    overlay_nnunet = snakefile.parent / "rules/nnunet.smk"
    hash_targets = (
        ("source_nnunet_sha256", source_nnunet),
        ("overlay_nnunet_sha256", overlay_nnunet),
    )
    for key, path in hash_targets:
        if path is None or not path.is_file():
            errors.append(f"overlay hash source is missing for {key}: {path}")
            continue
        actual_hash = hashlib.sha256(path.read_bytes()).hexdigest()
        if payload.get(key) != actual_hash:
            errors.append(
                f"overlay {key} mismatch: recorded={payload.get(key)!r}, "
                f"actual={actual_hash}"
            )
    if overlay_nnunet.is_file():
        overlay_text = overlay_nnunet.read_text(encoding="utf-8", errors="ignore")
        expected_binding = f"return {json.dumps(str(CANONICAL_MODEL_PATH))}"
        if expected_binding not in overlay_text:
            errors.append("overlay nnunet.smk lacks the canonical get_model_tar binding")
        if REMOTE_RULE in overlay_text:
            errors.append("overlay nnunet.smk still references the remote model rule")
        if "rule unpack_nnunet_model:" not in overlay_text:
            errors.append("overlay nnunet.smk lacks unpack_nnunet_model")
    return {
        "valid": not errors,
        "snakefile": str(snakefile),
        "manifest_path": str(manifest_path),
        "model_path": payload.get("model_path"),
        "errors": errors,
    }


def _read_config_info(output_root: Path) -> dict[str, Any]:
    config_path = output_root / "config" / "snakebids.yml"
    if not config_path.is_file():
        return {
            "path": None,
            "output_spaces": [],
            "output_spaces_ok": False,
            "local_model_workflow": False,
            "overlay": _read_overlay_info(None),
        }

    lines = config_path.read_text(encoding="utf-8", errors="ignore").splitlines()
    output_spaces: list[str] = []
    for index, line in enumerate(lines):
        if line.strip() != "output_spaces:":
            continue
        values: list[str] = []
        for candidate in lines[index + 1 :]:
            stripped = candidate.strip()
            if not stripped.startswith("-"):
                break
            values.append(stripped[1:].strip().strip("'\""))
        if values:
            output_spaces = values

    snakefile_value: str | None = None
    for index, line in enumerate(lines):
        if line.strip() != "- --snakefile" or index + 1 >= len(lines):
            continue
        candidate = lines[index + 1].strip()
        if candidate.startswith("-"):
            snakefile_value = candidate[1:].strip().strip("\'\"")
            break
    overlay_info = _read_overlay_info(snakefile_value)
    return {
        "path": str(config_path),
        "output_spaces": output_spaces,
        "output_spaces_ok": output_spaces == list(OUTPUT_SPACES),
        "local_model_workflow": (
            bool(snakefile_value)
            and "hippunfold_local_model_overlay" in snakefile_value
            and overlay_info["valid"]
        ),
        "overlay": overlay_info,
    }


def _metadata_files(output_root: Path, run_started_at: float | None) -> list[Path]:
    metadata_root = output_root / ".snakemake" / "metadata"
    if not metadata_root.is_dir():
        return []
    files = [p for p in metadata_root.iterdir() if p.is_file()]
    if run_started_at is not None:
        recent = [p for p in files if p.stat().st_mtime >= run_started_at - 5.0]
        files = recent
    return files


def _read_metadata_info(output_root: Path, run_started_at: float | None) -> dict[str, Any]:
    files = _metadata_files(output_root, run_started_at)
    remote_files = [
        str(path)
        for path in files
        if REMOTE_RULE in path.read_text(encoding="utf-8", errors="ignore")
    ]
    return {
        "files_checked": len(files),
        "remote_model_rule_found": bool(remote_files),
        "remote_model_rule_files": remote_files,
    }


def _surface_info(batch_root: Path, subject: str) -> dict[str, Any]:
    missing: list[str] = []
    invalid: list[str] = []
    vertex_counts: dict[str, int] = {}
    for relative in expected_surface_paths(subject):
        label = next(label for label in LABELS if f"_label-{label}_" in relative)
        path = batch_root / relative
        if not path.is_file() or path.stat().st_size == 0:
            missing.append(relative)
            continue
        try:
            image = nib.load(path)
            pointsets = image.get_arrays_from_intent("NIFTI_INTENT_POINTSET")
        except (OSError, ValueError, IndexError) as exc:
            invalid.append(f"{relative}: unreadable GIFTI ({exc})")
            continue
        if len(pointsets) != 1 or pointsets[0].data.ndim != 2:
            invalid.append(f"{relative}: missing unique pointset array")
            continue
        vertices = int(pointsets[0].data.shape[0])
        vertex_counts[relative] = vertices
        expected_vertices = EXPECTED_SURFACE_VERTICES[label]
        if vertices != expected_vertices:
            invalid.append(f"{relative}: {vertices} vertices, expected {expected_vertices}")
    return {
        "expected_vertices": dict(EXPECTED_SURFACE_VERTICES),
        "missing_files": missing,
        "invalid_files": invalid,
        "vertex_counts": vertex_counts,
        "complete": not missing and not invalid,
    }


def _spec_info(batch_root: Path, subject: str) -> dict[str, Any]:
    missing: list[str] = []
    invalid: list[str] = []
    for relative in expected_spec_paths(subject):
        path = batch_root / relative
        if not path.is_file() or path.stat().st_size == 0:
            missing.append(relative)
            continue
        text = path.read_text(encoding="utf-8", errors="ignore")
        if "space-corobl_den-512" not in text:
            invalid.append(relative)
    return {
        "missing_files": missing,
        "invalid_files": invalid,
        "complete": not missing and not invalid,
    }


def _downstream_info(downstream_root: Path | None) -> dict[str, Any]:
    if downstream_root is None:
        return {
            "checked": False,
            "forbidden_references": [],
            "files_checked": 0,
        }
    if not downstream_root.is_dir():
        return {
            "checked": True,
            "forbidden_references": [f"missing downstream root: {downstream_root}"],
            "files_checked": 0,
        }

    forbidden: list[str] = []
    files_checked = 0
    for path in downstream_root.rglob("*"):
        if not path.is_file() or path.stat().st_size > 16 * 1024 * 1024:
            continue
        if path.suffix.lower() not in {".json", ".yaml", ".yml", ".txt", ".tsv", ".log"}:
            continue
        text = path.read_text(encoding="utf-8", errors="ignore")
        files_checked += 1
        for reference in DOWNSTREAM_SURFACE_REF.findall(text):
            if "space-unfold_den-512" in reference:
                forbidden.append(f"{path}: {reference}")
    return {
        "checked": True,
        "forbidden_references": forbidden,
        "files_checked": files_checked,
    }


def check_subject(
    batch_root: Path,
    subject: str,
    *,
    run_started_at: float | None = None,
    downstream_root: Path | None = None,
) -> dict[str, Any]:
    """Check one subject without inferring required paths from current files."""

    subject = normalize_subject(subject)
    output_root = _subject_output_root(batch_root, subject)
    missing_files = [
        relative
        for relative in [*expected_surface_paths(subject), *expected_spec_paths(subject)]
        if not (batch_root / relative).is_file() or (batch_root / relative).stat().st_size == 0
    ]
    log_info = _read_log_info(output_root, run_started_at)
    config_info = _read_config_info(output_root)
    metadata_info = _read_metadata_info(output_root, run_started_at)
    surface_info = _surface_info(batch_root, subject)
    spec_info = _spec_info(batch_root, subject)
    downstream_info = _downstream_info(downstream_root)

    errors: list[str] = []
    if not output_root.is_dir():
        errors.append("missing output root")
    if not surface_info["complete"]:
        errors.append("corobl surface set is missing, unreadable, or not 512 vertices")
    if missing_files:
        errors.append(f"missing {len(missing_files)} required corobl/spec files")
    if not log_info["all_success"]:
        errors.append("latest run has no successful all target")
    if log_info["failure_markers"]:
        errors.append("failure markers in latest run log")
    if not config_info["output_spaces_ok"]:
        errors.append("output_spaces is not exactly corobl")
    if not config_info["local_model_workflow"]:
        errors.append(
            "validated local-model workflow/canonical model check failed: "
            + "; ".join(config_info["overlay"]["errors"])
        )
    if metadata_info["remote_model_rule_found"]:
        errors.append("new metadata references download_nnunet_model")
    if not spec_info["complete"]:
        errors.append("surface spec set is missing or lacks corobl den-512 refs")
    if downstream_info["forbidden_references"]:
        errors.append("downstream consumes a space-unfold den-512 surface")

    return {
        "subject": subject,
        "status": "complete" if not errors else "incomplete",
        "errors": errors,
        "missing_files": missing_files,
        "log": log_info,
        "config": config_info,
        "surface": surface_info,
        "metadata": metadata_info,
        "spec": spec_info,
        "downstream": downstream_info,
    }


def build_batch_report(
    batch_root: Path,
    subjects: Sequence[str],
    *,
    expected_subject_count: int = EXPECTED_SUBJECT_COUNT,
    run_started_at: float | None = None,
    downstream_root: Path | None = None,
) -> dict[str, Any]:
    """Build a report whose complete status requires the entire fixed roster."""

    normalized = [normalize_subject(subject) for subject in subjects]
    duplicate_subjects = sorted({s for s in normalized if normalized.count(s) > 1})
    checks = [
        check_subject(
            batch_root,
            subject,
            run_started_at=run_started_at,
            downstream_root=downstream_root,
        )
        for subject in normalized
    ]
    complete_count = sum(row["status"] == "complete" for row in checks)
    roster_ok = len(normalized) == expected_subject_count and not duplicate_subjects
    status = "complete" if roster_ok and complete_count == expected_subject_count else "incomplete"
    return {
        "schema_version": 1,
        "dataset": "jj_tms",
        "status": status,
        "model_path": str(CANONICAL_MODEL_PATH),
        "model_dir": str(CANONICAL_MODEL_DIR),
        "output_spaces": list(OUTPUT_SPACES),
        "expected_subject_count": expected_subject_count,
        "subject_count": len(normalized),
        "complete_count": complete_count,
        "duplicate_subjects": duplicate_subjects,
        "subjects": checks,
    }


def validate_dag_output(text: str) -> list[str]:
    """Return errors for a fresh standard-entry dry-run transcript."""

    errors: list[str] = []
    if not re.search(r"(?:local)?rule\s+all:", text, re.IGNORECASE):
        errors.append("DAG is missing all target")
    for rule, expected_count in REQUIRED_DAG_RULE_COUNTS.items():
        count = len(re.findall(rf"(?:local)?rule\s+{re.escape(rule)}:", text))
        if count != expected_count:
            errors.append(
                f"DAG rule {rule} count is {count}; expected {expected_count}"
            )
    if LOCAL_WORKFLOW_MARKER not in text:
        errors.append("DAG did not use the validated local-model workflow")
    if re.search(r"(?:local)?rule\s+download_nnunet_model\s*:", text, re.IGNORECASE):
        errors.append("DAG includes the remote model download rule")
    return errors


def build_subject_command(
    subject: str,
    *,
    input_root: Path = DEFAULT_INPUT_ROOT,
    batch_root: Path = DEFAULT_BATCH_ROOT,
) -> list[str]:
    """Construct the only allowed full-run command for one subject."""

    subject = normalize_subject(subject)
    output_dir = batch_root / f"sub-{subject}" / "hippunfold"
    return [
        str(PYTHON_EXE),
        str(CORE_ENTRY),
        subject,
        str(input_root),
        str(output_dir),
        "--model-path",
        str(CANONICAL_MODEL_PATH),
    ]


def build_subject_environment(subject: str, base: Mapping[str, str] | None = None) -> dict[str, str]:
    """Pin cache/model/target settings for a single standard-entry process."""

    subject = normalize_subject(subject)
    environment = dict(base or os.environ)
    cache_root = WORKDIR_ROOT / f"jj_tms_sub-{subject}"
    environment.update(
        {
            "HIPPUNFOLD_MODEL_PATH": str(CANONICAL_MODEL_PATH),
            "HIPPUNFOLD_CACHE_DIR": str(CANONICAL_CACHE_DIR),
            "HIPPUNFOLD_RUNTIME_SOURCE_CACHE": str(CANONICAL_RUNTIME_SOURCE_CACHE),
            "HIPPUNFOLD_CACHE_ROOT": str(cache_root),
            "HIPPUNFOLD_CLEANUP_RUNTIME": "0",
            "HIPPUNFOLD_CORES": "1",
            "HIPPUNFOLD_USE_GPU": "1",
            "HIPPUNFOLD_USE_CONDA": "0",
            "HIPPUNFOLD_WORKFLOW_PROFILE": "none",
            "HIPPUNFOLD_SDM": "env-modules",
            "HIPPUNFOLD_T1W_FILTER": "extension=.nii",
            "HIPPUNFOLD_EXTRA_ARGS": "",
            "XDG_CACHE_HOME": str(cache_root / "xdg_cache"),
            "PIP_CACHE_DIR": str(cache_root / "pip_cache"),
            "TMPDIR": str(cache_root / "tmp"),
        }
    )
    return environment


def _input_t1w_files(input_root: Path, subject: str) -> list[Path]:
    subject = normalize_subject(subject)
    anat = input_root / f"sub-{subject}" / "anat"
    return sorted(
        path
        for path in anat.glob("*_T1w.nii")
        if path.is_file() and path.name.endswith("_T1w.nii")
    )


def preflight(
    input_root: Path,
    subjects: Sequence[str],
    *,
    expected_subject_count: int = EXPECTED_SUBJECT_COUNT,
) -> dict[str, Any]:
    """Check fixed inputs and assets before any subprocess is started."""

    normalized = [normalize_subject(subject) for subject in subjects]
    errors: list[str] = []
    if len(normalized) != expected_subject_count:
        errors.append(
            f"expected {expected_subject_count} manifest subjects, got {len(normalized)}"
        )
    if not input_root.is_dir():
        errors.append(f"input root missing: {input_root}")
    missing_inputs: dict[str, str] = {}
    for subject in normalized:
        files = _input_t1w_files(input_root, subject)
        if not files:
            missing_inputs[subject] = f"{input_root}/sub-{subject}/anat/*_T1w.nii*"
    if missing_inputs:
        errors.append(f"missing T1w input for {len(missing_inputs)} subjects")
    if not CORE_ENTRY.is_file():
        errors.append(f"standard core entry missing: {CORE_ENTRY}")
    if not CANONICAL_MODEL_PATH.is_file() or CANONICAL_MODEL_PATH.stat().st_size == 0:
        errors.append(f"canonical model missing: {CANONICAL_MODEL_PATH}")
    if (
        not CANONICAL_MODEL_DIR.is_dir()
        or next(CANONICAL_MODEL_DIR.iterdir(), None) is None
    ):
        errors.append(f"canonical unpacked model missing or empty: {CANONICAL_MODEL_DIR}")
    return {
        "status": "pass" if not errors else "fail",
        "subject_count": len(normalized),
        "expected_subject_count": expected_subject_count,
        "input_root": str(input_root),
        "model_path": str(CANONICAL_MODEL_PATH),
        "model_dir": str(CANONICAL_MODEL_DIR),
        "missing_inputs": missing_inputs,
        "errors": errors,
    }


def run_representative_dag_preflight(
    input_root: Path, subject: str = DAG_PREFLIGHT_SUBJECT
) -> dict[str, Any]:
    """Run and capture the mandatory unique-output, non-executing DAG gate."""

    command = [
        str(PYTHON_EXE),
        str(DAG_PREFLIGHT_ENTRY),
        "--subject",
        normalize_subject(subject),
        "--input-root",
        str(input_root),
    ]
    process = subprocess.run(
        command,
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
        check=False,
    )
    try:
        result = json.loads(process.stdout)
    except json.JSONDecodeError:
        return {
            "status": "fail",
            "errors": ["DAG preflight did not emit valid JSON"],
            "command": command,
            "returncode": process.returncode,
            "stdout": process.stdout[-4000:],
            "stderr": process.stderr[-4000:],
        }
    result["entry"] = str(DAG_PREFLIGHT_ENTRY)
    result["launcher_returncode"] = process.returncode
    if process.returncode != 0 or result.get("status") != "pass":
        result["status"] = "fail"
        result.setdefault("errors", []).append(
            f"DAG preflight entry returned {process.returncode}"
        )
        if process.stderr:
            result["stderr_tail"] = process.stderr[-4000:]
    return result


def _run_one_subject(
    subject: str,
    *,
    input_root: Path,
    batch_root: Path,
    launcher_log: Path,
) -> dict[str, Any]:
    command = build_subject_command(
        subject, input_root=input_root, batch_root=batch_root
    )
    launcher_log.parent.mkdir(parents=True, exist_ok=True)
    with launcher_log.open("w", encoding="utf-8") as stream:
        process = subprocess.run(
            command,
            cwd=str(REPO_ROOT),
            env=build_subject_environment(subject),
            stdout=stream,
            stderr=subprocess.STDOUT,
            check=False,
        )
    return {
        "subject": normalize_subject(subject),
        "returncode": int(process.returncode),
        "command": command,
        "launcher_log": str(launcher_log),
    }


def run_batch(
    *,
    input_root: Path = DEFAULT_INPUT_ROOT,
    batch_root: Path = DEFAULT_BATCH_ROOT,
    subjects_file: Path = DEFAULT_SUBJECTS_FILE,
    concurrency: int = CONCURRENCY,
    downstream_root: Path | None = None,
) -> tuple[int, dict[str, Any]]:
    """Run the fixed 70-subject full DAG and write a strict final manifest."""

    if concurrency != CONCURRENCY:
        raise ValueError(f"JJ_TMS concurrency is fixed at {CONCURRENCY}")
    subjects = read_production_subject_manifest(subjects_file)
    preflight_report = preflight(input_root, subjects)
    if preflight_report["status"] != "pass":
        return 2, {"status": "preflight_failed", "preflight": preflight_report}
    dag_preflight_report = run_representative_dag_preflight(input_root)
    if dag_preflight_report["status"] != "pass":
        return 2, {
            "status": "dag_preflight_failed",
            "preflight": preflight_report,
            "dag_preflight": dag_preflight_report,
        }

    run_started_at = time.time()
    run_id = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime(run_started_at))
    run_root = batch_root / "_batch_logs" / DEFAULT_RUN_ROOT_NAME / run_id
    launcher_log_root = run_root / "launcher_logs"
    launcher_log_root.mkdir(parents=True, exist_ok=True)
    initial = {
        "schema_version": 1,
        "status": "running",
        "run_id": run_id,
        "started_at": run_started_at,
        "input_root": str(input_root),
        "batch_root": str(batch_root),
        "subjects_file": str(subjects_file),
        "subjects": subjects,
        "concurrency": concurrency,
        "core_entry": str(CORE_ENTRY),
        "model_path": str(CANONICAL_MODEL_PATH),
        "model_dir": str(CANONICAL_MODEL_DIR),
        "snakemake_remote_model_rule": "removed_from_validated_overlay",
        "snakemake_omit_from": False,
        "dag_preflight": dag_preflight_report,
    }
    (run_root / "run_manifest.json").write_text(
        json.dumps(initial, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    results: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=concurrency) as executor:
        futures = {
            executor.submit(
                _run_one_subject,
                subject,
                input_root=input_root,
                batch_root=batch_root,
                launcher_log=launcher_log_root / f"sub-{subject}.log",
            ): subject
            for subject in subjects
        }
        for future in as_completed(futures):
            results.append(future.result())
    results.sort(key=lambda row: row["subject"])

    report = build_batch_report(
        batch_root,
        subjects,
        expected_subject_count=EXPECTED_SUBJECT_COUNT,
        run_started_at=run_started_at,
        downstream_root=downstream_root,
    )
    result_by_subject = {row["subject"]: row for row in results}
    for row in report["subjects"]:
        launcher = result_by_subject[row["subject"]]
        row["launcher"] = launcher
        if launcher["returncode"] != 0:
            row["status"] = "incomplete"
            row["errors"].append(
                f"standard core entry returned {launcher['returncode']}"
            )
    if any(row["status"] != "complete" for row in report["subjects"]):
        report["status"] = "incomplete"
        report["complete_count"] = sum(
            row["status"] == "complete" for row in report["subjects"]
        )
    final = {
        **report,
        "run_id": run_id,
        "started_at": run_started_at,
        "ended_at": time.time(),
        "input_root": str(input_root),
        "batch_root": str(batch_root),
        "subjects_file": str(subjects_file),
        "concurrency": concurrency,
        "core_entry": str(CORE_ENTRY),
        "model_path": str(CANONICAL_MODEL_PATH),
        "model_dir": str(CANONICAL_MODEL_DIR),
        "launcher_results": results,
        "preflight": preflight_report,
        "dag_preflight": dag_preflight_report,
    }
    (run_root / "final_manifest.json").write_text(
        json.dumps(final, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return (0 if final["status"] == "complete" else 1), final


def write_report(report: Mapping[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
