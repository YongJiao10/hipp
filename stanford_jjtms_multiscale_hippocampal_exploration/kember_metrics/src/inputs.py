"""Rest-fMRI-only artifact and cohort inventory for the Stanford workflow.

The inventory deliberately uses explicit, known paths. It never performs a
recursive search of the repository or of an input artifact root. The first
pass samples one representative subject for concrete schemas and checks the
full frozen roster by constructing expected paths for each subject, smoothing,
and hemisphere.
"""

from __future__ import annotations

import argparse
import copy
import datetime as dt
import json
import re
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd


WORKFLOW_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = WORKFLOW_ROOT.parents[1]
DEFAULT_CONFIG_PATH = WORKFLOW_ROOT / "config" / "analysis_config.json"

GROUP_CODING = {
    "NTHC": (0, 0),
    "TEHC": (1, 0),
    "NTS": (0, 1),
    "TIS": (1, 1),
}
EXPECTED_CELL_COUNTS = {"NTHC": 51, "TEHC": 40, "NTS": 56, "TIS": 47}
GROUP_ORDER = tuple(GROUP_CODING)

# Normalize case and every separator before matching so that spaces,
# hyphens, and underscores cannot evade the rest-only scope gate.
_PROHIBITED_NORMALIZED_MARKERS = (
    "beta",
    "archive",
    "_archive",
    "con_image",
    "conimage",
    "jjtms",
    "native_stimulation_mapping",
    "native_surface_comparison",
    "native_mapping",
    "outputs_native_surface_comparison",
)
_PROHIBITED_JJ_TMS = re.compile(r"(?<![a-z])jj[_]?tms(?![a-z])")
_PROHIBITED_TMS = re.compile(r"(?<![a-z])tms(?:_|$|response|contrast)")
_PROJECT_COMPONENT = "stanford_jjtms_multiscale_hippocampal_exploration"


def validate_rest_only_path(path: Path) -> None:
    """Reject paths that would leave the rest-fMRI-only analysis scope."""

    candidate = str(path).replace("\\", "/")
    normalized_parts = [
        re.sub(r"[^a-z0-9]+", "_", part.lower()).strip("_")
        for part in Path(candidate).parts
    ]
    normalized = "_".join(
        part for part in normalized_parts if part and part != _PROJECT_COMPONENT
    )
    marker_hit = any(marker in normalized for marker in _PROHIBITED_NORMALIZED_MARKERS)
    if marker_hit or _PROHIBITED_JJ_TMS.search(normalized) or _PROHIBITED_TMS.search(normalized):
        raise ValueError(
            "rest-fMRI-only scope rejects prohibited path: " f"{path}"
        )


def validate_workflow_path(
    path: str | Path,
    *,
    root: str | Path = WORKFLOW_ROOT,
    label: str = "path",
) -> Path:
    """Resolve and validate one path against the rest-only workflow root.

    This is the common fail-closed gate for public writers and provenance
    consumers. Resolution happens before containment testing so symlinks and
    ``..`` traversal cannot escape the canonical root.
    """

    candidate = Path(path).expanduser().resolve()
    root_path = Path(root).expanduser().resolve()
    validate_rest_only_path(candidate)
    if not candidate.is_relative_to(root_path):
        raise ValueError(
            f"{label} must be contained by WORKFLOW_ROOT: {candidate}"
        )
    return candidate


def _resolve_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = REPO_ROOT / path
    return path


def _load_config(config_path: Path = DEFAULT_CONFIG_PATH) -> dict[str, Any]:
    path = _resolve_path(config_path)
    validate_rest_only_path(path)
    with path.open(encoding="utf-8") as handle:
        config = json.load(handle)
    if not isinstance(config, dict):
        raise ValueError(f"inventory config must be an object: {path}")
    return config


def _configured_path(config: dict[str, Any], key: str) -> Path:
    try:
        value = config["paths"][key]
    except (KeyError, TypeError) as exc:
        raise KeyError(f"missing inventory path configuration: paths.{key}") from exc
    path = _resolve_path(value)
    validate_rest_only_path(path)
    return path


def _configured_roster_path(config: dict[str, Any], key: str) -> Path:
    try:
        value = config["rosters"][key]
    except (KeyError, TypeError) as exc:
        raise KeyError(f"missing roster configuration: rosters.{key}") from exc
    path = _resolve_path(value)
    validate_rest_only_path(path)
    return path


def _normalize_subject(value: str | int) -> str:
    raw = str(value).strip()
    if not raw:
        raise ValueError("empty subject identifier")
    if raw.lower().startswith("sub-"):
        raw = raw[4:]
    if not raw.isdigit():
        raise ValueError(f"invalid Stanford subject identifier: {value!r}")
    return f"sub-{int(raw):04d}"


def _read_subject_manifest(path: Path) -> set[str]:
    validate_rest_only_path(path)
    if not path.is_file():
        raise FileNotFoundError(f"subject manifest is missing: {path}")
    subjects: set[str] = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        token = line.strip()
        if not token or token.startswith("#"):
            continue
        subject = _normalize_subject(token.split()[0])
        if subject in subjects:
            raise ValueError(f"duplicate subject {subject} in {path}")
        subjects.add(subject)
    if not subjects:
        raise ValueError(f"subject manifest is empty: {path}")
    return subjects


def build_factorial_roster(config: dict[str, Any] | None = None) -> pd.DataFrame:
    """Build the exact mutually exclusive Stanford 2x2 roster.

    The four cells are derived from intersections of the existing QC manifests,
    preserving the manifest-defined cohort rather than inferring a group from
    a subject-number pattern.
    """

    config = _load_config() if config is None else copy.deepcopy(config)
    healthy = _read_subject_manifest(_configured_roster_path(config, "healthy"))
    nontrauma = _read_subject_manifest(_configured_roster_path(config, "nontrauma"))
    symptomatic = _read_subject_manifest(
        _configured_roster_path(config, "symptomatic")
    )
    trauma = _read_subject_manifest(_configured_roster_path(config, "trauma"))
    all_subjects = _read_subject_manifest(
        _configured_roster_path(config, "all_subjects")
    )

    cell_sets = {
        "NTHC": healthy & nontrauma,
        "TEHC": healthy & trauma,
        "NTS": symptomatic & nontrauma,
        "TIS": symptomatic & trauma,
    }
    union = set().union(*cell_sets.values())
    if len(union) != sum(EXPECTED_CELL_COUNTS.values()):
        raise ValueError(f"factorial roster union has {len(union)} subjects, expected 194")
    if not union <= all_subjects:
        missing = sorted(union - all_subjects)
        raise ValueError(
            "factorial roster contains subjects absent from all-subject manifest: "
            f"{missing[:5]}"
        )
    overlaps = [
        (first, second)
        for index, first in enumerate(GROUP_ORDER)
        for second in GROUP_ORDER[index + 1 :]
        if cell_sets[first] & cell_sets[second]
    ]
    if overlaps:
        raise ValueError(f"factorial roster cells overlap: {overlaps}")
    observed_counts = {group: len(subjects) for group, subjects in cell_sets.items()}
    if observed_counts != EXPECTED_CELL_COUNTS:
        raise ValueError(
            f"factorial roster counts {observed_counts} do not match "
            f"{EXPECTED_CELL_COUNTS}"
        )

    rows: list[dict[str, Any]] = []
    for group in GROUP_ORDER:
        trauma_binary, symptomatic_binary = GROUP_CODING[group]
        for subject in sorted(cell_sets[group], key=lambda value: int(value[4:])):
            rows.append(
                {
                    "subject": subject,
                    "trauma": trauma_binary,
                    "symptomatic": symptomatic_binary,
                    "four_group": group,
                }
            )
    return pd.DataFrame(rows, columns=["subject", "trauma", "symptomatic", "four_group"])


def _array_schema(path: Path) -> dict[str, Any]:
    result: dict[str, Any] = {"path": str(path), "exists": path.is_file()}
    if not path.is_file():
        result.update({"shape": None, "dtype": None})
        return result
    array = np.load(path, mmap_mode="r", allow_pickle=False)
    result.update(
        {
            "shape": [int(value) for value in array.shape],
            "dtype": str(array.dtype),
            "ndim": int(array.ndim),
        }
    )
    return result


def _json_schema(path: Path) -> dict[str, Any]:
    result: dict[str, Any] = {"path": str(path), "exists": path.is_file()}
    if not path.is_file():
        result["keys"] = []
        return result
    with path.open(encoding="utf-8") as handle:
        value = json.load(handle)
    result["keys"] = list(value) if isinstance(value, dict) else []
    result["value_type"] = type(value).__name__
    return result


def _table_schema(path: Path) -> dict[str, Any]:
    result: dict[str, Any] = {"path": str(path), "exists": path.is_file()}
    if not path.is_file():
        result.update({"columns": [], "shape": None, "dtypes": {}})
        return result
    table = pd.read_csv(path, sep="\t")
    result.update(
        {
            "columns": [str(column) for column in table.columns],
            "shape": [int(value) for value in table.shape],
            "dtypes": {str(column): str(dtype) for column, dtype in table.dtypes.items()},
        }
    )
    return result


def _surface_schema(path: Path) -> dict[str, Any]:
    result: dict[str, Any] = {"path": str(path), "exists": path.is_file()}
    if not path.is_file():
        result.update({"point_shape": None, "point_dtype": None, "face_shape": None})
        return result
    try:
        import nibabel as nib
    except ImportError as exc:  # pragma: no cover - py314 includes nibabel
        raise RuntimeError("nibabel is required for surface schema inspection") from exc
    image = nib.load(str(path))
    point_data = np.asarray(image.darrays[0].data)
    face_data = np.asarray(image.darrays[1].data) if len(image.darrays) > 1 else None
    result.update(
        {
            "point_shape": [int(value) for value in point_data.shape],
            "point_dtype": str(point_data.dtype),
            "face_shape": None
            if face_data is None
            else [int(value) for value in face_data.shape],
            "face_dtype": None if face_data is None else str(face_data.dtype),
        }
    )
    return result


def _artifact_paths(
    config: dict[str, Any], subject: str, smoothing_mm: int, hemi: str
) -> dict[str, Path]:
    subject = _normalize_subject(subject)
    smooth = f"{int(smoothing_mm)}mm"
    if hemi not in {"L", "R"}:
        raise ValueError(f"hemisphere must be L or R: {hemi!r}")
    routea_subject = _configured_path(config, "routea_fc_root") / subject
    surface_subject = _configured_path(config, "routea_surface_root") / subject
    matching_root = _configured_path(config, "template_matching_root")
    matching_hemi = (
        matching_root / "subjects" / subject / "matching" / smooth / f"hemi-{hemi}"
    )
    template_root = _configured_path(config, "hcp151_template_root")
    hippunfold_root = _configured_path(config, "hippunfold_root")
    surf_path = (
        hippunfold_root
        / subject
        / "hippunfold"
        / subject
        / "surf"
        / f"{subject}_hemi-{hemi}_space-corobl_den-512_label-hipp_midthickness.surf.gii"
    )
    return {
        "fc_vertex_to_network": routea_subject
        / "fc"
        / "kong2022"
        / smooth
        / f"hemi_{hemi}"
        / "hipp_vertex_to_network_fc.npy",
        "fc_vertex_to_vertex": routea_subject
        / "fc"
        / "intrinsic"
        / smooth
        / f"hemi_{hemi}"
        / "hipp_vertex_to_vertex_fc_active.npy",
        "surface_timeseries": surface_subject
        / "surface"
        / smooth
        / f"{subject}_hemi-{hemi}_timeseries.npy",
        "surface_gifti": surf_path,
        "fixed_labels": template_root / f"hcp151_{smooth}_hemi-{hemi}_labels.npy",
        "fixed_support": template_root / f"hcp151_{smooth}_hemi-{hemi}_support.npy",
        "fixed_prototypes": template_root / f"hcp151_{smooth}_hemi-{hemi}_prototypes.npy",
        "fixed_manifest": _configured_path(config, "hcp151_template_manifest"),
        "matching_labels": matching_hemi / "labels.npy",
        "matching_scores": matching_hemi / "template_correlations.npy",
        "matching_margin": matching_hemi / "correlation_margin.npy",
        "matching_valid_mask": matching_hemi / "valid_mask.npy",
        "matching_manifest": matching_hemi / "combo_manifest.json",
        "fc_summary": routea_subject
        / "fc"
        / "kong2022"
        / smooth
        / f"hemi_{hemi}"
        / "fc_summary.json",
    }


def _coverage_row(
    method: str,
    subject: str,
    smoothing_mm: int,
    hemi: str,
    paths: dict[str, Path],
    required: Iterable[str],
    input_only: bool = False,
) -> dict[str, Any]:
    required_keys = list(required)
    missing = [key for key in required_keys if not paths[key].is_file()]
    if missing:
        status = "missing"
    elif input_only:
        status = "input_available"
    else:
        status = "present"
    return {
        "subject": subject,
        "method": method,
        "smoothing_mm": int(smoothing_mm),
        "hemisphere": hemi,
        "status": status,
        "required_count": len(required_keys),
        "present_count": len(required_keys) - len(missing),
        "missing_components": ",".join(missing),
        "fc_path": str(paths["fc_vertex_to_network"]),
        "label_path": str(
            paths["fixed_labels"]
            if method == "hcp151_fixed"
            else paths["matching_labels"]
            if method == "hcp151_template_matching"
            else ""
        ),
        "surface_timeseries_path": str(paths["surface_timeseries"]),
        "surface_midthickness_path": str(paths["surface_gifti"]),
        "manifest_path": str(
            paths["matching_manifest"]
            if method == "hcp151_template_matching"
            else "not_applicable"
        ),
        "gradient_status": "not_applicable",
    }


def _method_row(
    config: dict[str, Any], subject: str, smoothing_mm: int, hemi: str, method: str
) -> dict[str, Any]:
    paths = _artifact_paths(config, subject, smoothing_mm, hemi)
    if method == "hcp151_fixed":
        row = _coverage_row(
            method,
            subject,
            smoothing_mm,
            hemi,
            paths,
            (
                "fc_vertex_to_network",
                "fixed_labels",
                "fixed_support",
                "fixed_prototypes",
                "fixed_manifest",
                "surface_timeseries",
                "surface_gifti",
            ),
        )
        row["manifest_path"] = str(_configured_path(config, "hcp151_template_manifest"))
        return row
    if method == "hcp151_template_matching":
        return _coverage_row(
            method,
            subject,
            smoothing_mm,
            hemi,
            paths,
            (
                "fc_vertex_to_network",
                "matching_labels",
                "matching_scores",
                "matching_margin",
                "matching_valid_mask",
                "matching_manifest",
                "surface_timeseries",
                "surface_gifti",
            ),
        )
    raise ValueError(f"unknown inventory method: {method}")


def _representative_schema(config: dict[str, Any], subject: str) -> dict[str, Any]:
    paths_l2 = _artifact_paths(config, subject, 2, "L")
    roster = build_factorial_roster(config)
    all_subjects = _read_subject_manifest(_configured_roster_path(config, "all_subjects"))
    schema = {
        "representative_subject": subject,
        "fc_vertex_to_network": _array_schema(paths_l2["fc_vertex_to_network"]),
        "fc_vertex_to_vertex": _array_schema(paths_l2["fc_vertex_to_vertex"]),
        "surface_timeseries": _array_schema(paths_l2["surface_timeseries"]),
        "surface_midthickness": _surface_schema(paths_l2["surface_gifti"]),
        "hcp151_fixed_labels": _array_schema(paths_l2["fixed_labels"]),
        "hcp151_fixed_support": _array_schema(paths_l2["fixed_support"]),
        "hcp151_fixed_prototypes": _array_schema(paths_l2["fixed_prototypes"]),
        "template_matching_labels": _array_schema(paths_l2["matching_labels"]),
        "template_matching_scores": _array_schema(paths_l2["matching_scores"]),
        "template_matching_margin": _array_schema(paths_l2["matching_margin"]),
        "template_matching_valid_mask": _array_schema(paths_l2["matching_valid_mask"]),
        "template_matching_manifest": _json_schema(paths_l2["matching_manifest"]),
        "fc_summary_manifest": _json_schema(paths_l2["fc_summary"]),
        "hcp151_template_manifest": _json_schema(
            _configured_path(config, "hcp151_template_manifest")
        ),
        "template_matching_run_manifest": _json_schema(
            _configured_path(config, "template_matching_run_manifest")
        ),
        "template_matching_validation_manifest": _json_schema(
            _configured_path(config, "template_matching_validation_manifest")
        ),
        "network_registry": _table_schema(_configured_path(config, "network_registry")),
    }
    schema["cohort"] = {
        "factorial_subjects": int(roster["subject"].nunique()),
        "all_subject_manifest_subjects": len(all_subjects),
        "excluded_from_factorial": sorted(all_subjects - set(roster["subject"])),
        "factorial_cell_counts": {
            str(group): int(count)
            for group, count in roster.groupby("four_group").size().items()
        },
    }
    return schema


def inspect_rest_artifacts(config: dict[str, Any]) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Inspect explicit representative rest-FC artifacts and return schemas.

    The returned table contains one row per configured method × smoothing ×
    hemisphere for the configured representative subject. Full-cohort path
    coverage is produced by :func:`write_inventory_report`.
    """

    if not isinstance(config, dict):
        raise TypeError("inventory config must be a dictionary")
    for section in ("paths", "rosters"):
        values = config.get(section, {})
        if not isinstance(values, dict):
            raise ValueError(f"inventory config section must be an object: {section}")
        for value in values.values():
            validate_rest_only_path(_resolve_path(value))

    representative_subject = _normalize_subject(
        config.get("representative_subject", "sub-1001")
    )
    smoothings = [int(value) for value in config.get("smoothings_mm", [2, 4])]
    hemispheres = [str(value) for value in config.get("hemispheres", ["L", "R"])]
    methods = [str(value) for value in config.get("methods", [])]
    rows = [
        _method_row(config, representative_subject, smoothing, hemi, method)
        for method in methods
        for smoothing in smoothings
        for hemi in hemispheres
    ]
    return pd.DataFrame(rows), _representative_schema(config, representative_subject)


def _build_subject_coverage(config: dict[str, Any], roster: pd.DataFrame) -> pd.DataFrame:
    methods = [str(value) for value in config.get("methods", [])]
    smoothings = [int(value) for value in config.get("smoothings_mm", [2, 4])]
    hemispheres = [str(value) for value in config.get("hemispheres", ["L", "R"])]
    rows: list[dict[str, Any]] = []
    for record in roster.to_dict("records"):
        for method in methods:
            for smoothing in smoothings:
                for hemi in hemispheres:
                    row = _method_row(config, record["subject"], smoothing, hemi, method)
                    row.update(
                        {
                            "trauma": int(record["trauma"]),
                            "symptomatic": int(record["symptomatic"]),
                            "four_group": record["four_group"],
                        }
                    )
                    rows.append(row)
    columns = [
        "subject",
        "trauma",
        "symptomatic",
        "four_group",
        "method",
        "smoothing_mm",
        "hemisphere",
        "status",
        "required_count",
        "present_count",
        "missing_components",
        "gradient_status",
        "fc_path",
        "label_path",
        "surface_timeseries_path",
        "surface_midthickness_path",
        "manifest_path",
    ]
    return pd.DataFrame(rows).reindex(columns=columns).fillna("")


def _safe_run_id(value: str | None) -> str:
    if value is None:
        return "inventory"
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", value):
        raise ValueError(
            "run_id must contain only letters, numbers, dot, underscore, or hyphen"
        )
    return value


def write_inventory_report(config: dict[str, Any], run_id: str | None = None) -> Path:
    """Write the inventory artifacts below the workflow reports directory."""

    run_name = _safe_run_id(run_id)
    report_dir = WORKFLOW_ROOT / "reports" / run_name
    validate_rest_only_path(report_dir)
    report_dir.mkdir(parents=True, exist_ok=True)

    artifact_inventory, schema = inspect_rest_artifacts(config)
    roster = build_factorial_roster(config)
    subject_coverage = _build_subject_coverage(config, roster)
    artifact_inventory.to_csv(report_dir / "artifact_inventory.tsv", sep="\t", index=False)
    subject_coverage.to_csv(report_dir / "subject_coverage.tsv", sep="\t", index=False)
    with (report_dir / "schema_examples.json").open("w", encoding="utf-8") as handle:
        json.dump(schema, handle, indent=2, sort_keys=True)
        handle.write("\n")
    return report_dir


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--run-id", default=None)
    args = parser.parse_args(argv)
    config = _load_config(args.config)
    report_dir = write_inventory_report(config, args.run_id)
    print(report_dir)
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised by the real inventory command
    raise SystemExit(main())
