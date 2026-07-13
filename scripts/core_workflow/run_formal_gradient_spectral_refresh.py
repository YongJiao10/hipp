#!/usr/bin/env python3
"""Launch the fixed, sequential formal gradient-spectral refresh."""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

CORE_WORKFLOW_DIR = Path(__file__).resolve().parent
if str(CORE_WORKFLOW_DIR) not in sys.path:
    sys.path.insert(0, str(CORE_WORKFLOW_DIR))

from lib.formal_cohorts import cohort_key_for_manifest, load_formal_subjects


REPO_ROOT = Path(__file__).resolve().parents[2]
PY314 = Path("/home/yojiao/.conda/envs/py314/bin/python")
THREAD_ENV = {
    "OMP_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1",
    "VECLIB_MAXIMUM_THREADS": "1",
}
FORMAL_GROUPS = (
    ("outputs_hcp", "routea_full_k3_fisherz", "kong2022"),
    ("outputs_hcp", "routea_17net_full_k3_fisherz", "kong2022"),
    ("outputs_hcp", "routea_weighted_fisherz_full_k3", "kong2022"),
    ("outputs_hcp", "lynch2024_20net_nsi043_full_k3_normalized_wta", "lynch2024_20net"),
    ("outputs_hcp", "lynch2024_8net_nsi043_full_k3_normalized_wta", "lynch2024_8net"),
    ("outputs_stanford", "routea_hc_full_k3_fisherz", "kong2022"),
    ("outputs_stanford", "routea_symp_full_k3_fisherz", "kong2022"),
    ("outputs_stanford", "routea_hc_weighted_fisherz_full_k3", "kong2022"),
    ("outputs_stanford", "routea_symp_weighted_fisherz_full_k3", "kong2022"),
)


def validate_release_settings(*, run_id: str, jobs: int) -> None:
    if jobs != 1:
        raise ValueError("The formal release runner is fixed at --jobs 1")
    if re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}", run_id) is None:
        raise ValueError(f"Unsafe --run-id: {run_id!r}")


def build_stage_plan(
    *, repo_root: Path, python_exe: Path, jobs: int, run_id: str = "formal"
) -> list[dict[str, object]]:
    validate_release_settings(run_id=run_id, jobs=jobs)
    root = Path(repo_root).resolve()
    python = str(Path(python_exe))
    scripts = root / "scripts/core_workflow"

    hcp = root / "outputs_hcp"
    hcp_routea = hcp / "inputs/hcp/kong2022/shared_inputs/routea"
    hcp_surface = hcp / "inputs/hcp/kong2022/shared_inputs/surface"
    hcp_template = hcp / "inputs/hcp/kong2022/templates/hipp_group_average_t1only_151/group_average_surfaces"
    hcp_batch = hcp / "inputs/hcp/hippunfold/dense_corobl_batch"
    hcp_gradient = hcp / "work/hippomaps_gradients"
    hcp_group = hcp / "work/group_atlas"
    hcp_present = hcp / "present/overviews"

    lynch_base = hcp / "inputs/hcp/lynch2024_fast_pfm_t1only"
    lynch_shared_parent = lynch_base / "shared_inputs"
    lynch_shared = lynch_shared_parent / "pfm_nsi043_78"
    lynch_manifest = lynch_base / "manifests/hcp_7t_lynch2024_nsi043_pass_78.txt"
    lynch20_group = hcp_group / "lynch2024_20net_nsi043_full_k3_normalized_wta"
    lynch8_group = hcp_group / "lynch2024_8net_nsi043_full_k3_normalized_wta"

    stanford = root / "outputs_stanford"
    stanford_inputs = stanford / "inputs/stanford/kong2022"
    stanford_routea = stanford_inputs / "shared_inputs/routea"
    stanford_surface = stanford_inputs / "shared_inputs/surface"
    stanford_template = stanford_inputs / "templates/hipp_group_average_t1only_151/group_average_surfaces"
    stanford_batch = stanford / "inputs/stanford/hippunfold/dense_corobl_batch"
    stanford_gradient = stanford / "work/hippomaps_gradients"
    stanford_group = stanford / "work/group_atlas"
    stanford_present = stanford / "present/overviews"
    stanford_template_spec = (
        stanford_batch
        / "sub-1001/hippunfold/sub-1001/surf/sub-1001_den-512_label-hipp_surfaces.spec"
    )
    subgroup_manifests = {
        "hc": stanford_inputs / "manifests/stanford_hc_1xxx2xxx_qc_pass_t1only_91.txt",
        "symp": stanford_inputs / "manifests/stanford_symp_3xxx4xxx_qc_pass_t1only_103.txt",
    }

    def command(script: str, *args: object) -> list[str]:
        return [python, str(scripts / script), *(str(value) for value in args)]

    def group_command(
        *, shared: Path, surface: Path, reference: Path, gradients: Path,
        out: Path, route: str, taxonomy: str, atlas: str, manifest: Path,
        batch: Path, template: Path, dataset: str,
    ) -> list[str]:
        return command(
            "05_02_run_group_atlas.py",
            "--shared-root", shared,
            "--surface-root", surface,
            "--reference-root", reference,
            "--gradient-root", gradients,
            "--out-root", out,
            "--dataset", dataset,
            "--route-tag", route,
            "--network-taxonomy", taxonomy,
            "--subjects-manifest", manifest,
            "--hippunfold-batch-root", batch,
            "--group-average-surfaces-dir", template,
            "--methods", "gradient-spectral",
            "--atlases", atlas,
            "--k-min", 2,
            "--k-max", 8,
            "--force-final-k", 3,
            "--jobs", jobs,
            "--resume",
        )

    def overview_command(
        *, group: Path, present: Path, route: str, atlas: str, variant: str,
        mode: str | None = None, manifest_name: str | None = None,
        output_name: str | None = None,
    ) -> list[str]:
        args: list[object] = [
            "--group-root", group,
            "--present-dir", present,
            "--route-tag", route,
            "--methods", "gradient-spectral",
            "--atlases", atlas,
            "--assignment-variant", variant,
            "--scene", root / "config/wb_locked_native_view_lateral_medial.scene",
        ]
        if mode is not None:
            args.extend(["--mode-tag", mode])
        if manifest_name is not None:
            args.extend(["--manifest-name", manifest_name])
        if output_name is not None:
            args.extend(["--output-name", output_name])
        return command("06_01_build_group_overview.py", *args)

    def weighted_command(
        *, source: Path, out: Path, surface: Path, reference: Path,
        present: Path, template: Path, template_spec: Path,
    ) -> list[str]:
        return command(
            "05_05_build_weighted_fisherz_branch.py",
            "--source-root", source,
            "--out-root", out,
            "--surface-root", surface,
            "--reference-root", reference,
            "--present-dir", present,
            "--template-surface-dir", template,
            "--template-spec", template_spec,
            "--network-style", root / "config/hipp_network_style.json",
            "--methods", "gradient-spectral",
        )

    def stanford_stages(subgroup: str) -> list[dict[str, object]]:
        source = stanford_group / f"routea_{subgroup}_full_k3_fisherz"
        weighted = stanford_group / f"routea_{subgroup}_weighted_fisherz_full_k3"
        plain_present = stanford_present / f"routea_{subgroup}"
        weighted_present = stanford_present / f"routea_{subgroup}_weighted_fisherz"
        plain_overview = overview_command(
            group=source, present=plain_present, route="routea", atlas="kong2022",
            variant="fisherz", manifest_name="fisherz_group_overview_manifest.json",
            output_name="kong2022_gradient-spectral_group_overview_g1g2g3_routea_fisherz.png",
        )
        weighted_overview = overview_command(
            group=weighted, present=weighted_present, route="routea", atlas="kong2022",
            variant="weighted_fisherz", mode="weighted_fisherz",
            manifest_name="weighted_fisherz_group_overview_manifest.json",
            output_name="kong2022_gradient-spectral_group_overview_g1g2g3_routea_weighted_fisherz_weighted_fisherz.png",
        )
        return [
            {
                "name": f"stanford_{subgroup}_group",
                "command": group_command(
                    shared=stanford_routea, surface=stanford_surface,
                    reference=stanford_routea, gradients=stanford_gradient,
                    out=source, route="routea", taxonomy="canonical8", atlas="kong2022",
                    manifest=subgroup_manifests[subgroup], batch=stanford_batch,
                    template=stanford_template,
                    dataset="outputs_stanford",
                ),
            },
            {
                "name": f"stanford_{subgroup}_weighted",
                "command": weighted_command(
                    source=source, out=weighted, surface=stanford_surface,
                    reference=stanford_routea, present=weighted_present,
                    template=stanford_template, template_spec=stanford_template_spec,
                ),
            },
            {
                "name": f"stanford_{subgroup}_overviews",
                "command": plain_overview,
                "additional_commands": [weighted_overview],
            },
        ]

    hcp_weighted = hcp_group / "routea_weighted_fisherz_full_k3"
    hcp_weighted_present = hcp_present / "weighted_fisherz"
    hcp_template_spec = (
        hcp_batch
        / "sub-100610/hippunfold/sub-100610/surf/sub-100610_den-512_label-hipp_surfaces.spec"
    )
    stages: list[dict[str, object]] = [
        {
            "name": "lynch20_reference",
            "command": command(
                "03_03_build_lynch2024_reference.py",
                "--shared-root", lynch_shared,
                "--subjects-manifest", lynch_manifest,
                "--cortex-pfm-root", hcp / "work/cortex_pfm",
                "--cross-atlas-merge-json", root / "config/cross_atlas_network_merge.json",
                "--jobs", jobs,
            ),
        },
        {
            "name": "lynch20_fc",
            "command": command(
                "04_01_materialize_kong2022_roi_fc.py",
                "--shared-root", lynch_shared,
                "--surface-root", hcp_surface,
                "--reference-root", lynch_shared,
                "--subjects-manifest", lynch_manifest,
                "--atlas", "lynch2024_20net",
                "--jobs", jobs,
            ),
        },
        {
            "name": "lynch8_fc",
            "command": command(
                "04_01_materialize_kong2022_roi_fc.py",
                "--shared-root", lynch_shared,
                "--surface-root", hcp_surface,
                "--reference-root", lynch_shared,
                "--subjects-manifest", lynch_manifest,
                "--atlas", "lynch2024_8net",
                "--jobs", jobs,
            ),
        },
        {
            "name": "lynch20_gradient_bank",
            "command": command(
                "05_01_compute_hippomaps_gradients.py",
                "--routes", "lynch2024_20net",
                "--shared-inputs-root", lynch_shared_parent,
                "--surface-root", lynch_shared,
                "--subjects-manifest", lynch_manifest,
                "--group-average-surfaces-dir", hcp_template,
                "--out-root", hcp_gradient,
                "--n-components", 5,
                "--gradient-random-state", 0,
                "--selected-components", "1,2,3",
                "--force",
            ),
        },
        {
            "name": "lynch20_group",
            "command": group_command(
                shared=lynch_shared, surface=lynch_shared, reference=lynch_shared,
                gradients=hcp_gradient, out=lynch20_group, route="lynch2024_20net",
                taxonomy="lynch20", atlas="lynch2024_20net", manifest=lynch_manifest,
                batch=hcp_batch, template=hcp_template,
                dataset="outputs_hcp",
            ),
        },
        {
            "name": "lynch20_overview",
            "command": overview_command(
                group=lynch20_group, present=hcp_present / "lynch2024_20net",
                route="lynch2024_20net", atlas="lynch2024_20net", variant="fisherz",
                manifest_name="fisherz_group_overview_manifest.json",
                output_name="lynch2024_20net_gradient-spectral_group_overview_g1g2g3_lynch2024_20net_fisherz.png",
            ),
        },
        {
            "name": "lynch8_group",
            "command": group_command(
                shared=lynch_shared, surface=lynch_shared, reference=lynch_shared,
                gradients=hcp_gradient, out=lynch8_group, route="lynch2024_8net",
                taxonomy="lynch8", atlas="lynch2024_8net", manifest=lynch_manifest,
                batch=hcp_batch, template=hcp_template,
                dataset="outputs_hcp",
            ),
        },
        {
            "name": "lynch8_overview",
            "command": overview_command(
                group=lynch8_group, present=hcp_present / "lynch2024_8net",
                route="lynch2024_8net", atlas="lynch2024_8net", variant="fisherz",
                manifest_name="fisherz_group_overview_manifest.json",
                output_name="lynch2024_8net_gradient-spectral_group_overview_g1g2g3_lynch2024_8net_fisherz.png",
            ),
        },
        {
            "name": "hcp_weighted_branch",
            "command": command(
                "upgrade_retained_hcp_formal_provenance.py",
                "--repo-root", root,
                "--subjects-manifest",
                hcp / "inputs/hcp/kong2022/manifests/hcp_7t_hippocampus_fmri_qc_pass_t1only_latest.txt",
            ),
            "additional_commands": [
                weighted_command(
                    source=hcp_group / "routea_full_k3_fisherz", out=hcp_weighted,
                    surface=hcp / "inputs/hcp/kong2019_fast_pfm_t1only/shared_inputs/pfm_151",
                    reference=hcp_routea, present=hcp_weighted_present,
                    template=hcp_template, template_spec=hcp_template_spec,
                )
            ],
        },
        {
            "name": "hcp_weighted_overview",
            "command": overview_command(
                group=hcp_weighted, present=hcp_weighted_present, route="routea",
                atlas="kong2022", variant="weighted_fisherz", mode="weighted_fisherz",
                manifest_name="weighted_fisherz_group_overview_manifest.json",
                output_name="kong2022_gradient-spectral_group_overview_g1g2g3_routea_weighted_fisherz_weighted_fisherz.png",
            ),
        },
    ]
    stages.extend(stanford_stages("hc"))
    stages.extend(stanford_stages("symp"))
    stages.extend(
        [
            {
                "name": "hcp_routea_overviews",
                "command": overview_command(
                    group=hcp_group / "routea_full_k3_fisherz", present=hcp_present,
                    route="routea", atlas="kong2022", variant="both",
                    manifest_name="routea_group_overview_manifest.json",
                ),
            },
            {
                "name": "hcp_family17_overviews",
                "command": overview_command(
                    group=hcp_group / "routea_17net_full_k3_fisherz", present=hcp_present,
                    route="routea_17net", atlas="kong2022", variant="both",
                    manifest_name="routea_17net_group_overview_manifest.json",
                ),
            },
            {
                "name": "formal_release_audit",
                "command": command(
                    "audit_formal_gradient_spectral_release.py",
                    "--repo-root", root,
                    "--output-json", hcp_group / "logs" / f"{run_id}.audit.json",
                ),
            },
        ]
    )
    return stages


def run_paths(repo_root: Path, run_id: str) -> dict[str, Path]:
    validate_release_settings(run_id=run_id, jobs=1)
    directory = Path(repo_root).resolve() / "outputs_hcp/work/group_atlas/logs"
    return {
        "status": directory / f"{run_id}.status.json",
        "stdout": directory / f"{run_id}.stdout.log",
        "stderr": directory / f"{run_id}.stderr.log",
        "pid": directory / f"{run_id}.pid",
        "audit": directory / f"{run_id}.audit.json",
    }


def _write_status(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def _stage_commands(stage: dict[str, object]) -> list[list[str]]:
    commands = [list(stage["command"])]
    commands.extend(list(command) for command in stage.get("additional_commands", []))
    return commands


def _subject_ids(manifest: Path) -> list[str]:
    if not manifest.is_file():
        return []
    return [
        subject.removeprefix("sub-")
        for subject in load_formal_subjects(manifest, cohort_key_for_manifest(manifest))
    ]


def derive_formal_group_identities(
    stages: list[dict[str, object]], repo_root: Path
) -> tuple[tuple[str, str, str], ...]:
    """Derive formal identities from actual consumer/producer commands."""
    root = Path(repo_root).resolve()
    identities: set[tuple[str, str, str]] = set()
    for stage in stages:
        for command in _stage_commands(stage):
            script = Path(command[1]).name
            if script == "05_02_run_group_atlas.py":
                group_root = Path(command[command.index("--out-root") + 1]).resolve()
                atlas = command[command.index("--atlases") + 1]
                dataset = command[command.index("--dataset") + 1]
            elif script == "05_05_build_weighted_fisherz_branch.py":
                group_root = Path(command[command.index("--out-root") + 1]).resolve()
                atlas = "kong2022"
                dataset = group_root.relative_to(root).parts[0]
            elif script == "06_01_build_group_overview.py":
                group_root = Path(command[command.index("--group-root") + 1]).resolve()
                atlas = command[command.index("--atlases") + 1]
                dataset = group_root.relative_to(root).parts[0]
            else:
                continue
            branch = group_root.relative_to(root / dataset / "work/group_atlas").parts[0]
            identities.add((dataset, branch, atlas))
    return tuple(identity for identity in FORMAL_GROUPS if identity in identities)


def _group_manifest_path(group_root: Path, smoothing: str, hemi: str) -> Path:
    return (
        group_root
        / "gradient-spectral/kong2022"
        / smoothing
        / "cov_0p80/g1g2g3"
        / f"hemi_{hemi}"
        / "group_atlas_manifest.json"
    )


def _group_manifest_subjects(manifest: Path) -> list[str]:
    if not manifest.is_file():
        return []
    try:
        payload = json.loads(manifest.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return []
    subjects = payload.get("subjects", []) if isinstance(payload, dict) else []
    return [str(subject).removeprefix("sub-") for subject in subjects]


def _recorded_path(repo_root: Path, raw: object) -> Path:
    path = Path(str(raw)).expanduser()
    return path.resolve() if path.is_absolute() else (repo_root / path).resolve()


def _existing_group_run_inputs(repo_root: Path, manifest: Path) -> set[Path]:
    """Enumerate exact leaves read by weighted refreshes and overview rendering."""
    run_dir = manifest.parent
    paths = {
        manifest,
        run_dir / "cluster_network_profile.npy",
        run_dir / "group_feature_summary.json",
        run_dir / "group_labels_final.npy",
        run_dir / "k_selection_summary.tsv",
    }
    if not manifest.is_file():
        return paths
    try:
        payload = json.loads(manifest.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return paths
    if not isinstance(payload, dict):
        return paths
    for key in ("template_left_or_right_surface", "template_spec_path"):
        raw = payload.get(key)
        if raw not in (None, "", "none", "null"):
            paths.add(_recorded_path(repo_root, raw))
    variants = payload.get("assignment_variants", {})
    if isinstance(variants, dict):
        for variant in ("fisherz", "count"):
            row = variants.get(variant)
            if not isinstance(row, dict):
                continue
            for key in (
                "cluster_assignment_status",
                "template_functional_atlas_label",
                "template_functional_atlas_label_variant",
                "group_functional_atlas_labels",
                "group_functional_atlas_label_map",
            ):
                raw = row.get(key)
                if raw:
                    paths.add(_recorded_path(repo_root, raw))
    return paths


def _subject_surface_inputs(
    *, surface_root: Path, subject: str, include_masks: bool = True
) -> set[Path]:
    paths: set[Path] = set()
    for smoothing in ("2mm", "4mm"):
        for hemi in ("L", "R"):
            paths.add(
                surface_root
                / f"sub-{subject}/surface/{smoothing}"
                / f"sub-{subject}_hemi-{hemi}_timeseries.npy"
            )
    if include_masks:
        for hemi in ("L", "R"):
            paths.add(
                surface_root
                / f"sub-{subject}/surface/tsnr_gate"
                / f"sub-{subject}_hemi-{hemi}_valid_mask.npy"
            )
    return paths


def _hippunfold_surface_inputs(batch_root: Path, subjects: list[str]) -> set[Path]:
    paths: set[Path] = set()
    for subject in subjects:
        surface_dir = batch_root / f"sub-{subject}/hippunfold/sub-{subject}/surf"
        for hemi in ("L", "R"):
            paths.add(
                surface_dir
                / f"sub-{subject}_hemi-{hemi}_space-corobl_den-512_label-hipp_midthickness.surf.gii"
            )
    if subjects:
        first = subjects[0]
        paths.add(
            batch_root
            / f"sub-{first}/hippunfold/sub-{first}/surf"
            / f"sub-{first}_den-512_label-hipp_surfaces.spec"
        )
    return paths


def required_preexisting_paths(repo_root: Path) -> list[Path]:
    """Return exact leaf inputs that must exist before the 19-stage worker starts."""
    root = Path(repo_root).resolve()
    hcp = root / "outputs_hcp"
    stanford = root / "outputs_stanford"
    required: set[Path] = {
        PY314,
        root / "config/cross_atlas_network_merge.json",
        root / "config/hipp_network_style.json",
        root / "config/wb_locked_native_view_lateral_medial.scene",
        root / "external/hippomaps/hippomaps/__init__.py",
    }
    wb_override = os.environ.get("CORE_WORKFLOW_WB_COMMAND", "").strip()
    required.add(
        Path(wb_override).expanduser().resolve()
        if wb_override
        else root / "software/workbench/bin_linux64/wb_command"
    )

    hcp_template = (
        hcp
        / "inputs/hcp/kong2022/templates/hipp_group_average_t1only_151"
        / "group_average_surfaces"
    )
    for hemi in ("L", "R"):
        required.add(
            hcp_template
            / f"group-average_hemi-{hemi}_space-corobl_den-512_label-hipp_midthickness.surf.gii"
        )
    hcp_batch = hcp / "inputs/hcp/hippunfold/dense_corobl_batch"
    required.add(
        hcp_batch
        / "sub-100610/hippunfold/sub-100610/surf"
        / "sub-100610_den-512_label-hipp_surfaces.spec"
    )

    hcp_group = hcp / "work/group_atlas"
    hcp_151_manifest = (
        hcp / "inputs/hcp/kong2022/manifests"
        / "hcp_7t_hippocampus_fmri_qc_pass_t1only_latest.txt"
    )
    required.add(hcp_151_manifest)
    hcp_151_subjects = _subject_ids(hcp_151_manifest)
    hcp_routea_group = hcp_group / "routea_full_k3_fisherz"
    hcp_family17_group = hcp_group / "routea_17net_full_k3_fisherz"
    hcp_weighted_subjects: set[str] = set()
    for group_root in (hcp_routea_group, hcp_family17_group):
        for smoothing in ("2mm", "4mm"):
            for hemi in ("L", "R"):
                manifest = _group_manifest_path(group_root, smoothing, hemi)
                required.update(_existing_group_run_inputs(root, manifest))
                if group_root == hcp_routea_group:
                    run_dir = manifest.parent
                    required.update(
                        {
                            run_dir / "group_support_mask.npy",
                            run_dir / "group_functional_atlas_labels_fisherz.npy",
                        }
                    )
                    hcp_weighted_subjects.update(_group_manifest_subjects(manifest))
    if hcp_weighted_subjects and hcp_weighted_subjects != set(hcp_151_subjects):
        raise ValueError(
            "Retained HCP RouteA group manifests do not match the canonical 151-subject cohort"
        )
    hcp_weighted_surface = (
        hcp / "inputs/hcp/kong2019_fast_pfm_t1only/shared_inputs/pfm_151"
    )
    hcp_routea = hcp / "inputs/hcp/kong2022/shared_inputs/routea"
    for subject in hcp_weighted_subjects:
        required.update(
            _subject_surface_inputs(
                surface_root=hcp_weighted_surface,
                subject=subject,
                include_masks=False,
            )
        )
        reference = hcp_routea / f"sub-{subject}/reference/kong2022"
        required.update(
            {
                reference / "cortex_roi_parcel_timeseries.npy",
                reference / "cortex_roi_parcels.tsv",
            }
        )

    lynch_manifest = (
        hcp
        / "inputs/hcp/lynch2024_fast_pfm_t1only/manifests"
        / "hcp_7t_lynch2024_nsi043_pass_78.txt"
    )
    required.add(lynch_manifest)
    lynch_subjects = _subject_ids(lynch_manifest)
    lynch_shared = (
        hcp
        / "inputs/hcp/lynch2024_fast_pfm_t1only/shared_inputs/pfm_nsi043_78"
    )
    hcp_surface = hcp / "inputs/hcp/kong2022/shared_inputs/surface"
    for subject in lynch_subjects:
        cortex_pfm = hcp / f"work/cortex_pfm/sub-{subject}/lynch2024"
        required.update(
            {
                cortex_pfm / "cortex_roi_component_timeseries.npy",
                cortex_pfm / "cortex_roi_components.tsv",
            }
        )
        required.update(
            _subject_surface_inputs(surface_root=hcp_surface, subject=subject)
        )
        required.update(
            _subject_surface_inputs(surface_root=lynch_shared, subject=subject)
        )
    required.update(_hippunfold_surface_inputs(hcp_batch, lynch_subjects))

    stanford_inputs = stanford / "inputs/stanford/kong2022"
    stanford_routea = stanford_inputs / "shared_inputs/routea"
    stanford_surface = stanford_inputs / "shared_inputs/surface"
    stanford_template = (
        stanford_inputs
        / "templates/hipp_group_average_t1only_151/group_average_surfaces"
    )
    for hemi in ("L", "R"):
        required.add(
            stanford_template
            / f"group-average_hemi-{hemi}_space-corobl_den-512_label-hipp_midthickness.surf.gii"
        )
    stanford_gradient = stanford / "work/hippomaps_gradients/routea"
    for smoothing in ("2mm", "4mm"):
        for hemi in ("L", "R"):
            gradient_run = stanford_gradient / smoothing / f"hemi_{hemi}"
            required.update(
                {
                    gradient_run / "hippomaps_gradient_manifest.json",
                    gradient_run / "hippomaps_gradients_full.npy",
                }
            )
    stanford_batch = stanford / "inputs/stanford/hippunfold/dense_corobl_batch"
    required.add(
        stanford_batch
        / "sub-1001/hippunfold/sub-1001/surf"
        / "sub-1001_den-512_label-hipp_surfaces.spec"
    )
    stanford_manifests = (
        stanford_inputs / "manifests/stanford_hc_1xxx2xxx_qc_pass_t1only_91.txt",
        stanford_inputs / "manifests/stanford_symp_3xxx4xxx_qc_pass_t1only_103.txt",
    )
    for manifest in stanford_manifests:
        required.add(manifest)
        subjects = _subject_ids(manifest)
        required.update(_hippunfold_surface_inputs(stanford_batch, subjects))
        for subject in subjects:
            required.update(
                _subject_surface_inputs(
                    surface_root=stanford_surface,
                    subject=subject,
                )
            )
            reference = stanford_routea / f"sub-{subject}/reference/kong2022"
            required.update(
                {
                    reference / "cortex_roi_parcel_timeseries.npy",
                    reference / "cortex_roi_parcels.tsv",
                    reference / "cortex_canonical_network_timeseries.npy",
                    reference / "cortex_canonical_networks.tsv",
                }
            )
            for smoothing in ("2mm", "4mm"):
                for hemi in ("L", "R"):
                    fc_run = (
                        stanford_routea
                        / f"sub-{subject}/fc/kong2022/{smoothing}/hemi_{hemi}"
                    )
                    required.add(fc_run / "hipp_vertex_to_network_fc.npy")
                    for run_id in (1, 2, 3, 4):
                        required.add(
                            fc_run
                            / f"runs/run-{run_id}/hipp_vertex_to_network_fc_active.npy"
                        )
    return sorted(required, key=str)


def preflight(repo_root: Path, stages: list[dict[str, object]]) -> None:
    root = Path(repo_root).resolve()
    required = set(required_preexisting_paths(root))
    for stage in stages:
        for command in _stage_commands(stage):
            required.add(Path(command[1]))
    missing = sorted(str(path) for path in required if not path.exists())
    if missing:
        raise FileNotFoundError("Missing formal refresh inputs:\n" + "\n".join(missing))
    if len(stages) != 19:
        raise ValueError(f"Expected exactly 19 formal refresh stages, got {len(stages)}")


def build_worker_command(
    *, script_path: Path, repo_root: Path, run_id: str, jobs: int
) -> list[str]:
    return [
        str(PY314),
        str(Path(script_path).resolve()),
        "--worker",
        "--repo-root", str(Path(repo_root).resolve()),
        "--run-id", run_id,
        "--jobs", str(jobs),
    ]


def run_worker(*, repo_root: Path, run_id: str, jobs: int) -> int:
    validate_release_settings(run_id=run_id, jobs=jobs)
    root = Path(repo_root).resolve()
    paths = run_paths(root, run_id)
    stages = build_stage_plan(repo_root=root, python_exe=PY314, jobs=jobs, run_id=run_id)
    env = dict(os.environ)
    env.update(THREAD_ENV)
    completed: list[str] = []
    current_stage: str | None = None
    base = {"run_id": run_id, "pid": os.getpid(), "jobs": jobs, "stages": stages}
    try:
        preflight(root, stages)
        _write_status(paths["status"], {**base, "status": "running", "completed_stages": completed})
        for stage in stages:
            current_stage = str(stage["name"])
            for command in _stage_commands(stage):
                subprocess.run(command, cwd=root, env=env, check=True)
            completed.append(current_stage)
            _write_status(paths["status"], {**base, "status": "running", "completed_stages": completed})
        _write_status(
            paths["status"],
            {
                **base,
                "status": "ok",
                "completed_stages": completed,
                "completed_utc": datetime.now(timezone.utc).isoformat(),
                "audit_json": str(paths["audit"]),
            },
        )
        return 0
    except Exception as exc:
        _write_status(
            paths["status"],
            {
                **base,
                "status": "failed",
                "completed_stages": completed,
                "failed_stage": current_stage,
                "failed_utc": datetime.now(timezone.utc).isoformat(),
                "error": f"{type(exc).__name__}: {exc}",
            },
        )
        return 1


def launch(*, repo_root: Path, run_id: str, jobs: int) -> dict[str, object]:
    validate_release_settings(run_id=run_id, jobs=jobs)
    root = Path(repo_root).resolve()
    paths = run_paths(root, run_id)
    stages = build_stage_plan(repo_root=root, python_exe=PY314, jobs=jobs, run_id=run_id)
    preflight(root, stages)
    collisions = [str(path) for path in paths.values() if path.exists()]
    if collisions:
        raise FileExistsError("Run record collision:\n" + "\n".join(collisions))
    _write_status(paths["status"], {"status": "launching", "run_id": run_id, "jobs": jobs, "stages": stages})
    env = dict(os.environ)
    env.update(THREAD_ENV)
    worker = build_worker_command(
        script_path=Path(__file__), repo_root=root, run_id=run_id, jobs=jobs
    )
    with paths["stdout"].open("w", encoding="utf-8") as stdout, paths["stderr"].open("w", encoding="utf-8") as stderr:
        process = subprocess.Popen(
            worker,
            cwd=root,
            env=env,
            stdin=subprocess.DEVNULL,
            stdout=stdout,
            stderr=stderr,
            start_new_session=True,
        )
    paths["pid"].write_text(f"{process.pid}\n", encoding="utf-8")
    return {
        "status": "launched",
        "run_id": run_id,
        "pid": process.pid,
        **{f"{key}_path": str(path) for key, path in paths.items()},
    }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--launch", action="store_true")
    mode.add_argument("--worker", action="store_true")
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    parser.add_argument("--run-id")
    parser.add_argument("--jobs", type=int, default=1)
    return parser


def main() -> int:
    args = build_arg_parser().parse_args()
    run_id = args.run_id or f"formal_gradient_refresh_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
    validate_release_settings(run_id=run_id, jobs=args.jobs)
    if args.worker:
        return run_worker(repo_root=args.repo_root, run_id=run_id, jobs=args.jobs)
    print(json.dumps(launch(repo_root=args.repo_root, run_id=run_id, jobs=args.jobs), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
