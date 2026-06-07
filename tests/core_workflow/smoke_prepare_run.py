#!/usr/bin/env python3
from __future__ import annotations

"""
Process Step Description:
Prepare smoke run files and commands. Prepares an isolated, sub-selected subject cohort directory (default: 5 subjects) and builds the sequential command manifests (shell script and JSON command plans) to run a complete, isolated smoke test execution of the entire core workflow pipeline.

Main Flow Path:
- None (utility/helper script for isolated smoke test runs; not part of the mainline sequence).

Inputs:
- Command-line arguments:
  - --run-id: Unique identifier for the smoke run directory (default: `smoke5_<timestamp>`).
  - --source-subjects: Text file list containing bare source subject IDs to choose from (default: `outputs_experiments/hippunfold_recommended_t1w_20260423/vertex_qc_t1only/manifests/hcp_7t_hippocampus_fmri_qc_pass_t1only_151.txt`).
  - --subjects: Explicit space-separated subject list to override --source-subjects.
  - --subject-count: Number of subjects to include in the smoke run (default: 5).
  - --python-exe: Explicit python interpreter path for commands (default: `/home/yojiao/.conda/envs/py314/bin/python`).
  - --input-dir: Target HippUnfold inputs folder (default: `data/hippunfold_input`).
  - --bold-root: Root directory of fMRI volumes.
  - --hippunfold-asset-store-root: Optional directory path to a shared local HippUnfold model asset cache.
- Physical inputs:
  - Subject manifest file: `{source_subjects}`.
  - Core directory layouts and environment references.

Outputs:
- File outputs (generated under `outputs_hcp/smoke_runs/{run_id}/`):
  - Selected subjects list: `inputs/subjects_smoke5.txt`
  - Sequential pipeline shell script: `manifests/run_smoke_commands.sh`
  - Sequential pipeline command manifest JSON: `manifests/smoke_command_manifest.json`
  - Dispatched intermediate directories for logs, inputs, features, group_atlas, gradients, and overviews.
"""

import argparse
import json
from pathlib import Path
from typing import Any
import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
CORE_WORKFLOW_DIR = REPO_ROOT / "scripts" / "core_workflow"
if str(CORE_WORKFLOW_DIR) not in sys.path:
    sys.path.insert(0, str(CORE_WORKFLOW_DIR))

from lib.output_layout import DEFAULT_QC_ROOT, run_layout, timestamped_run_id
from lib.workflow_utils import git_commit, utc_now_iso, write_json
from lib.io_utils import normalize_subject_id, load_subject_ids


PYTHON_EXE = "/home/yojiao/.conda/envs/py314/bin/python"
DEFAULT_SOURCE_SUBJECTS = DEFAULT_QC_ROOT / "manifests" / "hcp_7t_hippocampus_fmri_qc_pass_t1only_151.txt"
DEFAULT_FALLBACK_SMOKE_SUBJECTS = ("100610", "102311", "104416", "105923", "108323")
DEFAULT_INPUT_DIR = REPO_ROOT / "data" / "hippunfold_input"
DEFAULT_BOLD_ROOT = Path("/home/yojiao/LSS/jianglab/3_Data_Working/HCP_7T_Hippocampus/rsfmri_volume")
DEFAULT_HIPPUNFOLD_ASSET_STORE_ROOT = REPO_ROOT / "runtime" / "hippunfold_asset_store_shared"
DEFAULT_HIPPOMAPS_ROOT = REPO_ROOT / "external" / "hippomaps"


def load_subjects(path: Path, count: int, explicit: list[str] | None = None) -> tuple[list[str], str]:
    if explicit:
        subjects = []
        seen = set()
        for raw in explicit:
            subject = normalize_subject_id(raw)
            if subject and subject not in seen:
                subjects.append(subject)
                seen.add(subject)
        return subjects[:count], "explicit"
    if path.exists():
        subjects = load_subject_ids(path)
        if len(subjects) < count:
            raise RuntimeError(f"Requested {count} subjects but only found {len(subjects)} in {path}")
        return subjects[:count], str(path.resolve())
    fallback = list(DEFAULT_FALLBACK_SMOKE_SUBJECTS)[:count]
    if len(fallback) < count:
        raise RuntimeError(f"No source subject manifest and fallback only has {len(fallback)} subjects")
    return fallback, "built_in_fallback_subjects_manifest_missing"


def cmd(*parts: object) -> list[str]:
    return [str(p) for p in parts]


def command_record(stage: str, command: list[str], *, note: str = "", env: dict[str, str] | None = None) -> dict[str, Any]:
    return {"stage": stage, "command": command, "note": note, "env": {} if env is None else dict(env)}


def build_command_plan(*, subjects: list[str], layout: dict[str, Path], python_exe: str, input_dir: Path, bold_root: Path, hippunfold_asset_store_root: Path | None = None, kong2022_dlabel: Path | None = None, hippomaps_root: Path | None = None) -> list[dict[str, Any]]:
    subject_manifest = layout["inputs"] / "subjects_smoke5.txt"
    shared_inputs = layout["features"] / "shared_inputs"
    surface_root = shared_inputs / "surface"
    routea_root = shared_inputs / "routea_151"
    routeb_root = shared_inputs / "routeb_151"
    template_root = layout["features"] / "templates" / "hipp_group_average"
    group_surfaces = template_root / "group_average_surfaces"
    group_root_base = layout["group_atlas"]
    gradient_root = layout["gradients"]

    hippunfold_env = (
        {"HIPPUNFOLD_ASSET_STORE_ROOT": str(Path(hippunfold_asset_store_root).resolve())}
        if hippunfold_asset_store_root is not None
        else {}
    )
    kong2022_dlabel_args = (
        ["--kong2022-dlabel", str(Path(kong2022_dlabel).resolve())]
        if kong2022_dlabel is not None
        else []
    )
    hippomaps_root_args = (
        ["--hippomaps-root", str(Path(hippomaps_root).resolve())]
        if hippomaps_root is not None
        else []
    )

    records: list[dict[str, Any]] = [
        command_record(
            "00_env_preflight_py314",
            cmd(python_exe, "-c", "import sys,nibabel; print(sys.executable); print(nibabel.__version__)"),
        ),
        command_record(
            "00_env_preflight_informational_bare_python",
            ["python", "-c", "import sys; print(sys.executable)"],
            note="Informational only; do not use bare python for repository tasks.",
        ),
    ]
    for subject in subjects:
        records.append(
            command_record(
                f"01_01_run_hippunfold_t1w_local_sub-{subject}",
                cmd(
                    python_exe,
                    REPO_ROOT / "scripts/core_workflow/01_01_run_hippunfold_t1w_local.py",
                    subject,
                    input_dir,
                    layout["hippunfold"] / f"sub-{subject}",
                ),
                env=hippunfold_env,
            )
        )
    jobs = str(min(5, max(1, len(subjects))))
    records.extend(
        [
            command_record(
                "02_01_prepare_group_inputs",
                cmd(
                    python_exe,
                    REPO_ROOT / "scripts/core_workflow/02_01_prepare_group_inputs.py",
                    "--hippunfold-batch-root",
                    layout["hippunfold"],
                    "--route-root",
                    layout["features"] / "kong2022",
                    "--shared-root",
                    surface_root,
                    "--template-root",
                    template_root,
                    "--bold-root",
                    bold_root,
                    "--subjects-manifest",
                    subject_manifest,
                    "--jobs",
                    jobs,
                    "--stage",
                    "all",
                ),
            ),
            command_record(
                "03_01_build_kong2022_routeb_reference",
                cmd(
                    python_exe,
                    REPO_ROOT / "scripts/core_workflow/03_01_build_kong2022_reference.py",
                    "--shared-root",
                    routeb_root,
                    "--subjects-manifest",
                    subject_manifest,
                    "--data-root",
                    input_dir,
                    *kong2022_dlabel_args,
                    "--jobs",
                    jobs,
                ),
            ),
            command_record(
                "03_02_build_kong2022_routea_reference",
                cmd(
                    python_exe,
                    REPO_ROOT / "scripts/core_workflow/03_02_build_kong2022_routea_reference.py",
                    "--shared-root",
                    routea_root,
                    "--subjects-manifest",
                    subject_manifest,
                    "--data-root",
                    input_dir,
                    *kong2022_dlabel_args,
                    "--jobs",
                    jobs,
                ),
            ),
        ]
    )
    for route, shared_root in (("routea", routea_root), ("routeb", routeb_root)):
        records.extend(
            [
                command_record(
                    f"04_01_materialize_kong2022_roi_fc_{route}",
                    cmd(
                        python_exe,
                        REPO_ROOT / "scripts/core_workflow/04_01_materialize_kong2022_roi_fc.py",
                        "--shared-root",
                        shared_root,
                        "--surface-root",
                        surface_root,
                        "--subjects-manifest",
                        subject_manifest,
                        "--jobs",
                        jobs,
                    ),
                ),
                command_record(
                    f"04_02_materialize_intrinsic_fc_{route}",
                    cmd(
                        python_exe,
                        REPO_ROOT / "scripts/core_workflow/04_02_materialize_intrinsic_fc.py",
                        "--shared-root",
                        shared_root,
                        "--surface-root",
                        surface_root,
                        "--subjects-manifest",
                        subject_manifest,
                        "--jobs",
                        jobs,
                    ),
                ),
            ]
        )
    records.append(
        command_record(
            "05_01_compute_hippomaps_gradients",
            cmd(
                python_exe,
                REPO_ROOT / "scripts/core_workflow/05_01_compute_hippomaps_gradients.py",
                "--routes",
                "routea",
                "routeb",
                "--shared-inputs-root",
                shared_inputs,
                "--surface-root",
                surface_root,
                "--subjects-manifest",
                subject_manifest,
                "--group-average-surfaces-dir",
                group_surfaces,
                "--out-root",
                gradient_root,
                *hippomaps_root_args,
                "--n-components",
                "5",
                "--selected-components",
                "1,2,3",
                "--max-subjects",
                str(len(subjects)),
            ),
        )
    )
    for route, shared_root in (("routea", routea_root), ("routeb", routeb_root)):
        group_root = group_root_base / f"{route}_full_k3_fisherz"
        records.extend(
            [
                command_record(
                    f"05_02_run_group_atlas_{route}_full_fixedk3",
                    cmd(
                        python_exe,
                        REPO_ROOT / "scripts/core_workflow/05_02_run_group_atlas.py",
                        "--shared-root",
                        shared_root,
                        "--surface-root",
                        surface_root,
                        "--out-root",
                        group_root,
                        "--gradient-root",
                        gradient_root,
                        *hippomaps_root_args,
                        "--route-tag",
                        route,
                        "--subjects-manifest",
                        subject_manifest,
                        "--hippunfold-batch-root",
                        layout["hippunfold"],
                        *kong2022_dlabel_args,
                        "--group-average-surfaces-dir",
                        group_surfaces,
                        "--methods",
                        "intrinsic-spectral",
                        "network-spectral",
                        "gradient-spectral",
                        "wta-top",
                        "--atlases",
                        "kong2022",
                        "--coverage-thresholds",
                        "0.80",
                        "--k-min",
                        "2",
                        "--k-max",
                        "10",
                        "--force-final-k",
                        "3",
                        "--instability-resamples",
                        "5",
                        "--null-resamples",
                        "5",
                        "--jobs",
                        jobs,
                    ),
                ),
                command_record(
                    f"06_01_build_group_overview_{route}",
                    cmd(
                        python_exe,
                        REPO_ROOT / "scripts/core_workflow/06_01_build_group_overview.py",
                        "--group-root",
                        group_root,
                        "--present-dir",
                        layout["overviews"],
                        "--methods",
                        "intrinsic-spectral",
                        "network-spectral",
                        "gradient-spectral",
                        "wta-top",
                        "--atlases",
                        "kong2022",
                        "--coverage-threshold",
                        "0.80",
                        "--route-tag",
                        route,
                    ),
                ),
            ]
        )
    records.append(
        command_record(
            "07_01_build_group_matrix",
            cmd(
                python_exe,
                REPO_ROOT / "scripts/core_workflow/07_01_build_group_matrix.py",
                "--group-atlas-root",
                group_root_base,
                "--present-dir",
                layout["overviews"],
            ),
        )
    )
    return records


def write_command_script(path: Path, records: list[dict[str, Any]]) -> None:
    lines = ["#!/usr/bin/env bash", "set -euo pipefail", "", f"cd {REPO_ROOT}", ""]
    for record in records:
        stage = str(record["stage"])
        command = [str(x) for x in record["command"]]
        env = {str(k): str(v) for k, v in dict(record.get("env", {})).items()}
        lines.append(f"echo '[stage] {stage}'")
        prefix = " ".join(f"{k}={json.dumps(v)}" for k, v in sorted(env.items()))
        rendered = " ".join(json.dumps(part) for part in command)
        lines.append(f"{prefix} {rendered}".strip())
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Prepare an isolated 5-subject core-workflow smoke run directory and command manifest.")
    parser.add_argument("--run-id", default=None, help="Run ID under outputs_hcp/smoke_runs/. Defaults to smoke5_<UTC timestamp>.")
    parser.add_argument("--source-subjects", default=str(DEFAULT_SOURCE_SUBJECTS))
    parser.add_argument("--subjects", nargs="*", default=None, help="Explicit smoke subject IDs. Overrides --source-subjects.")
    parser.add_argument("--subject-count", type=int, default=5)
    parser.add_argument("--python-exe", default=PYTHON_EXE)
    parser.add_argument("--input-dir", default=str(DEFAULT_INPUT_DIR))
    parser.add_argument("--bold-root", default=str(DEFAULT_BOLD_ROOT))
    parser.add_argument(
        "--hippunfold-asset-store-root",
        default=str(DEFAULT_HIPPUNFOLD_ASSET_STORE_ROOT) if DEFAULT_HIPPUNFOLD_ASSET_STORE_ROOT.exists() else None,
        help="Optional shared HippUnfold asset store containing hippunfold_assets/ and runtime_source_assets/ to avoid repeated model/template downloads.",
    )
    args = parser.parse_args()

    if int(args.subject_count) < 1:
        raise ValueError("--subject-count must be >= 1")
    run_id = str(args.run_id or timestamped_run_id("smoke5"))
    layout = run_layout(run_id)
    for key in ("inputs", "logs", "manifests", "hippunfold", "features", "group_atlas", "gradients", "overviews", "cortex_panels", "gradient_products"):
        layout[key].mkdir(parents=True, exist_ok=True)

    subjects, subject_source = load_subjects(Path(args.source_subjects).resolve(), int(args.subject_count), args.subjects)
    subject_manifest = layout["inputs"] / "subjects_smoke5.txt"
    subject_manifest.write_text("\n".join(subjects) + "\n", encoding="utf-8")

    records = build_command_plan(
        subjects=subjects,
        layout=layout,
        python_exe=str(args.python_exe),
        input_dir=Path(args.input_dir).resolve(),
        bold_root=Path(args.bold_root).resolve(),
        hippunfold_asset_store_root=(None if args.hippunfold_asset_store_root is None else Path(args.hippunfold_asset_store_root).resolve()),
        kong2022_dlabel=(None if args.kong2022_dlabel is None else Path(args.kong2022_dlabel).resolve()),
        hippomaps_root=(None if args.hippomaps_root is None else Path(args.hippomaps_root).resolve()),
    )
    commands_json = layout["manifests"] / "smoke_command_manifest.json"
    commands_sh = layout["manifests"] / "run_smoke_commands.sh"
    payload = {
        "schema_version": "1.0",
        "stage": "smoke_prepare_run",
        "status": "prepared",
        "timestamp_utc": utc_now_iso(),
        "git_commit": git_commit(),
        "run_id": run_id,
        "subject_source": subject_source,
        "subjects": subjects,
        "run_root": str(layout["run_root"].resolve()),
        "layout": {k: str(v.resolve()) for k, v in layout.items()},
        "policy": {
            "subject_count": int(args.subject_count),
            "max_jobs_in_generated_commands": min(5, max(1, len(subjects))),
            "fixed_final_k": 3,
            "auto_k_default": False,
            "presentation_split": "none",
            "overview_dir_contains_all_methods": str(layout["overviews"].resolve()),
            "hippunfold_asset_store_root": None if args.hippunfold_asset_store_root is None else str(Path(args.hippunfold_asset_store_root).resolve()),
            "kong2022_dlabel": None if args.kong2022_dlabel is None else str(Path(args.kong2022_dlabel).resolve()),
            "hippomaps_root": None if args.hippomaps_root is None else str(Path(args.hippomaps_root).resolve()),
        },
        "commands": records,
    }
    commands_json.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    write_command_script(commands_sh, records)
    commands_sh.chmod(0o755)

    print(json.dumps({"run_id": run_id, "run_root": str(layout["run_root"].resolve()), "subjects": subjects, "manifest": str(commands_json.resolve())}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
