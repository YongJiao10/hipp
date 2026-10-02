"""Fail-closed inventory for the cross-dataset resting-state FC analysis.

Only paths named by ``analysis_config.json`` are inspected.  In particular,
this module never globs, walks, or recursively discovers candidate inputs.
The path gate runs before ``is_file`` or any file read, and it deliberately
allows the legal dataset names ``JJTMS`` and ``JJ_TMS`` while rejecting
explicit beta, stimulation-response, con-image, native-mapping, traversal,
and archive components.
"""

from __future__ import annotations

import csv
import json
import re
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

import pandas as pd


WORKFLOW_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = WORKFLOW_ROOT.parent
REPO_ROOT = PROJECT_ROOT.parent
DEFAULT_CONFIG_PATH = WORKFLOW_ROOT / "config" / "analysis_config.json"

STANFORD_GROUPS = ("NTHC", "TEHC", "NTS", "TIS")
STANFORD_CODING = {
    "NTHC": (0, 0),
    "TEHC": (1, 0),
    "NTS": (0, 1),
    "TIS": (1, 1),
}
JJTMS_GROUPS = ("healthy", "patient")
REGISTERED_FC_ESTIMATOR = "fisher_z"

_ARCHIVE_SUFFIXES = frozenset(
    {
        ".7z",
        ".rar",
        ".tar",
        ".tgz",
        ".zip",
    }
)
_TMS_RESPONSE_RE = re.compile(
    r"(?:^|_)tms_(?:response|contrast|mapping|activation)(?:_|$)"
)
_TMS_RESPONSE_COMPACT_RE = re.compile(
    r"(?:^|_)tms(?:response|contrast|mapping|activation)(?:_|$)"
)
_NATIVE_MAPPING_RE = re.compile(r"(?:^|_)native_(?:mapping|stimulation)(?:_|$)")


def _normalise_component(component: str) -> str:
    # Split both ordinary camelCase and acronym-to-word transitions while
    # preserving the legal all-caps dataset token JJTMS.
    camel_split = re.sub(r"(?<=[A-Z])(?=[A-Z][a-z])", "_", component)
    camel_split = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", "_", camel_split)
    return re.sub(r"[^a-z0-9]+", "_", camel_split.lower()).strip("_")


def _scope_error_for_component(component: str) -> str | None:
    """Return a scope reason for one raw path component, if any."""

    normalised = _normalise_component(component)
    tokens = tuple(token for token in normalised.split("_") if token)
    if "beta" in tokens:
        return "beta"
    # ``con`` is checked as a token, so ordinary words such as connection or
    # reconstruction remain valid while con/con-image assets do not.
    if "con" in tokens:
        return "con"
    if _NATIVE_MAPPING_RE.search(normalised) or "native_mapping" in normalised:
        return "native_mapping"
    if _TMS_RESPONSE_RE.search(normalised) or _TMS_RESPONSE_COMPACT_RE.search(
        normalised
    ):
        return "TMS-response"
    compact_markers = (
        ("betamap", "beta"),
        ("conimage", "con"),
        ("nativemapping", "native_mapping"),
        ("nativestimulation", "native_mapping"),
        ("tmsresponse", "TMS-response"),
        ("tmscontrast", "TMS-response"),
        ("tmsmapping", "TMS-response"),
        ("tmsactivation", "TMS-response"),
    )
    for marker, reason in compact_markers:
        if normalised == marker or normalised.startswith(marker):
            return reason
    if "archive" in tokens:
        return "archive"
    suffixes = {suffix.lower() for suffix in Path(component).suffixes}
    if suffixes & _ARCHIVE_SUFFIXES or ".tar." in component.lower():
        return "archive"
    return None


def validate_rest_only_path(
    path: Path, allowed_roots: tuple[Path, ...]
) -> Path:
    """Validate and resolve a configured path before opening or probing it.

    ``allowed_roots`` is intentionally explicit.  A path is accepted only if
    its resolved form is below one of those roots.  Raw ``..`` components are
    rejected even when they happen to resolve back inside a root, because a
    traversal-shaped configuration is not an auditable explicit input.
    """

    candidate = Path(path).expanduser()
    raw_parts = candidate.parts
    if ".." in raw_parts:
        raise ValueError(f"rest-fMRI-only scope rejects traversal path: {path}")
    for component in raw_parts:
        reason = _scope_error_for_component(component)
        if reason is not None:
            raise ValueError(
                "rest-fMRI-only scope rejects prohibited path "
                f"component ({reason}): {path}"
            )
    if not allowed_roots:
        raise ValueError("rest-fMRI-only inventory requires at least one allowed root")

    try:
        resolved = candidate.resolve(strict=False)
        roots = tuple(Path(root).expanduser().resolve(strict=False) for root in allowed_roots)
    except OSError as exc:
        raise ValueError(f"could not resolve configured rest path: {path}") from exc

    for root in roots:
        if _scope_error_for_component(root.name) is not None:
            raise ValueError(
                f"rest-fMRI-only scope rejects prohibited allowed root: {root}"
            )
        if resolved == root or resolved.is_relative_to(root):
            return resolved
    roots_display = ", ".join(str(root) for root in roots)
    raise ValueError(
        f"rest-fMRI-only path is outside allowed root(s): {resolved}; "
        f"allowed roots: {roots_display}"
    )


def _allowed_roots(config: Mapping[str, Any]) -> tuple[Path, ...]:
    raw = config.get("allowed_roots", (".",))
    if isinstance(raw, (str, Path)):
        raw = (raw,)
    if not isinstance(raw, Sequence):
        raise TypeError("allowed_roots must be a sequence of paths")
    roots: list[Path] = []
    for value in raw:
        candidate = Path(value).expanduser()
        if not candidate.is_absolute():
            candidate = REPO_ROOT / candidate
        # Validate the root itself for prohibited components, but do not use
        # the path gate recursively because that would require a parent root.
        for component in candidate.parts:
            reason = _scope_error_for_component(component)
            if reason is not None:
                raise ValueError(
                    "rest-fMRI-only scope rejects prohibited allowed root "
                    f"component ({reason}): {candidate}"
                )
        roots.append(candidate.resolve(strict=False))
    if not roots:
        raise ValueError("rest-fMRI-only inventory requires at least one allowed root")
    return tuple(roots)


def _resolve_config_path(value: str | Path, allowed_roots: tuple[Path, ...]) -> Path:
    candidate = Path(value).expanduser()
    if not candidate.is_absolute():
        candidate = REPO_ROOT / candidate
    return validate_rest_only_path(candidate, allowed_roots)


def _normalise_subject(value: object) -> str:
    raw = str(value).strip()
    if raw.lower().startswith("sub-"):
        raw = raw[4:]
    if not raw or not raw.isdigit():
        raise ValueError(f"invalid subject identifier: {value!r}")
    return f"sub-{int(raw):04d}"


def _manifest_rows(path: Path, allowed_roots: tuple[Path, ...]) -> list[dict[str, str]]:
    """Read one explicitly configured text/CSV/TSV/JSON subject manifest."""

    resolved = validate_rest_only_path(path, allowed_roots)
    if not resolved.is_file():
        raise FileNotFoundError(f"configured subject manifest is missing: {resolved}")

    suffix = resolved.suffix.lower()
    if suffix == ".json":
        with resolved.open(encoding="utf-8") as handle:
            payload = json.load(handle)
        if isinstance(payload, Mapping):
            if "subjects" in payload:
                payload = payload["subjects"]
            elif "rows" in payload:
                payload = payload["rows"]
        if not isinstance(payload, list):
            raise ValueError(f"subject manifest JSON must contain a list: {resolved}")
        result: list[dict[str, str]] = []
        for item in payload:
            if isinstance(item, Mapping):
                if "subject" not in item:
                    raise ValueError(f"subject is missing from manifest row: {resolved}")
                row = {str(key): str(value) for key, value in item.items()}
                row["subject"] = _normalise_subject(row["subject"])
                result.append(row)
            else:
                result.append({"subject": _normalise_subject(item)})
        return result

    if suffix in {".csv", ".tsv"}:
        delimiter = "," if suffix == ".csv" else "\t"
        with resolved.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle, delimiter=delimiter)
            if not reader.fieldnames or "subject" not in reader.fieldnames:
                raise ValueError(f"manifest table needs a subject column: {resolved}")
            rows = []
            for raw in reader:
                row = {str(key): str(value) for key, value in raw.items() if key is not None}
                row["subject"] = _normalise_subject(row["subject"])
                rows.append(row)
            return rows

    rows = []
    with resolved.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            subject = stripped.split()[0]
            if subject.lower() == "subject":
                continue
            try:
                rows.append({"subject": _normalise_subject(subject)})
            except ValueError as exc:
                raise ValueError(
                    f"invalid subject manifest line {line_number} in {resolved}: {exc}"
                ) from exc
    if not rows:
        raise ValueError(f"subject manifest is empty: {resolved}")
    return rows


def _read_subject_set(
    path: str | Path,
    allowed_roots: tuple[Path, ...],
    *,
    expected_group: str | None = None,
) -> set[str]:
    rows = _manifest_rows(Path(path), allowed_roots)
    subjects: list[str] = []
    for row in rows:
        if expected_group is not None and "group" in row:
            observed = row["group"].strip().lower()
            if observed and observed != expected_group.lower():
                raise ValueError(
                    f"manifest label {observed!r} does not match {expected_group!r}"
                )
        subjects.append(row["subject"])
    if len(subjects) != len(set(subjects)):
        raise ValueError(f"duplicate subject in manifest: {path}")
    return set(subjects)


def _empty_roster(dataset: str, *, reason: str = "") -> pd.DataFrame:
    frame = pd.DataFrame(
        columns=[
            "dataset",
            "subject",
            "group",
            "four_group",
            "trauma",
            "symptomatic",
            "source_manifest",
            "source_manifest_primary",
            "source_manifest_secondary",
            "clinical_manifest",
            "trauma_manifest",
        ]
    )
    frame.attrs.update({"immutable": True, "input_gate_status": "blocked", "blocker": reason})
    frame.flags.allows_duplicate_labels = False
    return frame


def _freeze_roster(frame: pd.DataFrame, *, status: str = "ready") -> pd.DataFrame:
    frame = frame.copy(deep=True)
    frame.flags.allows_duplicate_labels = False
    frame.attrs["immutable"] = True
    frame.attrs["input_gate_status"] = status
    return frame


def _rows_to_frame(
    dataset: str,
    group_subjects: Mapping[str, Iterable[str]],
    source_manifests: Mapping[str, str | Sequence[str]],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for group, subjects in group_subjects.items():
        if dataset == "stanford":
            trauma, symptomatic = STANFORD_CODING[group]
        else:
            trauma, symptomatic = pd.NA, pd.NA
        configured_sources = source_manifests[group]
        if isinstance(configured_sources, str):
            source_paths = (configured_sources,)
        else:
            source_paths = tuple(str(path) for path in configured_sources)
        if not source_paths:
            raise ValueError(f"no source manifests configured for {dataset}/{group}")
        primary_manifest = source_paths[0]
        secondary_manifest = source_paths[1] if len(source_paths) > 1 else ""
        clinical_manifest = primary_manifest if dataset == "stanford" else primary_manifest
        trauma_manifest = secondary_manifest if dataset == "stanford" else ""
        for subject in sorted(subjects, key=lambda value: int(value[4:])):
            rows.append(
                {
                    "dataset": dataset,
                    "subject": subject,
                    "group": group,
                    "four_group": group if dataset == "stanford" else "",
                    "trauma": trauma,
                    "symptomatic": symptomatic,
                    "source_manifest": ";".join(source_paths),
                    "source_manifest_primary": primary_manifest,
                    "source_manifest_secondary": secondary_manifest,
                    "clinical_manifest": clinical_manifest,
                    "trauma_manifest": trauma_manifest,
                }
            )
    frame = pd.DataFrame(
        rows,
        columns=[
            "dataset",
            "subject",
            "group",
            "four_group",
            "trauma",
            "symptomatic",
            "source_manifest",
            "source_manifest_primary",
            "source_manifest_secondary",
            "clinical_manifest",
            "trauma_manifest",
        ],
    )
    if dataset == "jjtms":
        frame["trauma"] = frame["trauma"].astype("Int64")
        frame["symptomatic"] = frame["symptomatic"].astype("Int64")
    return _freeze_roster(frame)


def _check_group_sets(
    group_subjects: Mapping[str, set[str]],
    expected_counts: Mapping[str, Any] | None,
    *,
    require_union: int | None = None,
) -> None:
    groups = tuple(group_subjects)
    overlaps = [
        (first, second)
        for index, first in enumerate(groups)
        for second in groups[index + 1 :]
        if group_subjects[first] & group_subjects[second]
    ]
    if overlaps:
        raise ValueError(f"roster group overlap: {overlaps}")
    observed = {group: len(values) for group, values in group_subjects.items()}
    if expected_counts is not None:
        expected = {str(group): int(count) for group, count in expected_counts.items()}
        if observed != expected:
            raise ValueError(f"roster group counts {observed} do not match {expected}")
    if require_union is not None and len(set().union(*group_subjects.values())) != require_union:
        raise ValueError(
            "roster union has "
            f"{len(set().union(*group_subjects.values()))} subjects, expected {require_union}"
        )


def _build_dataset_roster(
    dataset: str, dataset_config: Mapping[str, Any], allowed_roots: tuple[Path, ...]
) -> pd.DataFrame:
    gate = dataset_config.get("input_gate", {})
    gate_status = str(gate.get("status", "ready")) if isinstance(gate, Mapping) else "ready"
    rosters = dataset_config.get("rosters", {})
    if not isinstance(rosters, Mapping):
        raise TypeError(f"{dataset}.rosters must be an object")
    if gate_status == "blocked" and not rosters:
        reason = str(gate.get("reason", "no explicit rest-only input"))
        return _empty_roster(dataset, reason=reason)

    expected_counts = dataset_config.get("expected_group_counts")
    if expected_counts is not None and not isinstance(expected_counts, Mapping):
        raise TypeError(f"{dataset}.expected_group_counts must be an object")
    roster_type = str(dataset_config.get("roster_type", "")).lower()

    if dataset == "stanford" and (
        roster_type in {"stanford_four_cell_intersections", "four_cell_intersections"}
        or set(STANFORD_GROUPS).issubset(rosters)
    ):
        if set(STANFORD_GROUPS).issubset(rosters) and "all_subjects" not in rosters:
            groups = {
                group: _read_subject_set(rosters[group], allowed_roots, expected_group=group)
                for group in STANFORD_GROUPS
            }
            _check_group_sets(groups, expected_counts)
            return _rows_to_frame(
                dataset,
                groups,
                {group: str(rosters[group]) for group in STANFORD_GROUPS},
            )
        required = ("all_subjects", "healthy", "nontrauma", "symptomatic", "trauma")
        missing = [name for name in required if name not in rosters]
        if missing:
            raise ValueError(f"Stanford roster is missing manifests: {missing}")
        all_subjects = _read_subject_set(rosters["all_subjects"], allowed_roots)
        healthy = _read_subject_set(rosters["healthy"], allowed_roots)
        nontrauma = _read_subject_set(rosters["nontrauma"], allowed_roots)
        symptomatic = _read_subject_set(rosters["symptomatic"], allowed_roots)
        trauma = _read_subject_set(rosters["trauma"], allowed_roots)
        groups = {
            "NTHC": healthy & nontrauma,
            "TEHC": healthy & trauma,
            "NTS": symptomatic & nontrauma,
            "TIS": symptomatic & trauma,
        }
        _check_group_sets(groups, expected_counts, require_union=None)
        union = set().union(*groups.values())
        if not union <= all_subjects:
            missing_from_all = sorted(union - all_subjects)
            raise ValueError(
                "Stanford roster contains subjects absent from all_subjects: "
                f"{missing_from_all[:5]}"
            )
        return _rows_to_frame(
            dataset,
            groups,
            {
                "NTHC": (str(rosters["healthy"]), str(rosters["nontrauma"])),
                "TEHC": (str(rosters["healthy"]), str(rosters["trauma"])),
                "NTS": (str(rosters["symptomatic"]), str(rosters["nontrauma"])),
                "TIS": (str(rosters["symptomatic"]), str(rosters["trauma"])),
            },
        )

    if dataset == "jjtms" and (
        roster_type in {"binary_labels", "binary"}
        or {"healthy", "patient"}.issubset(rosters)
    ):
        missing = [group for group in JJTMS_GROUPS if group not in rosters]
        if missing:
            raise ValueError(f"JJTMS roster is missing manifests: {missing}")
        groups = {
            group: _read_subject_set(
                rosters[group], allowed_roots, expected_group=group
            )
            for group in JJTMS_GROUPS
        }
        _check_group_sets(groups, expected_counts)
        if "all_subjects" in rosters:
            all_subjects = _read_subject_set(rosters["all_subjects"], allowed_roots)
            union = set().union(*groups.values())
            if not union <= all_subjects:
                raise ValueError("JJTMS labels contain subjects absent from all_subjects")
        return _rows_to_frame(
            dataset,
            groups,
            {group: str(rosters[group]) for group in JJTMS_GROUPS},
        )

    if "manifest" in rosters:
        rows = _manifest_rows(Path(str(rosters["manifest"])), allowed_roots)
        group_subjects: dict[str, set[str]] = {group: set() for group in JJTMS_GROUPS}
        for row in rows:
            group = row.get("group", "").strip().lower()
            if group not in group_subjects:
                raise ValueError(
                    f"{dataset} manifest group must be healthy or patient: {group!r}"
                )
            group_subjects[group].add(row["subject"])
        _check_group_sets(group_subjects, expected_counts)
        return _rows_to_frame(
            dataset,
            group_subjects,
            {group: str(rosters["manifest"]) for group in group_subjects},
        )

    raise ValueError(f"unsupported {dataset} roster configuration")


def build_rosters(config: Mapping[str, Any]) -> dict[str, pd.DataFrame]:
    """Build deterministic, mutually exclusive subject/group tables.

    The returned frames are fresh snapshots of the explicitly listed manifests
    and carry ``attrs['immutable'] = True`` as a contract for downstream
    consumers.  No path other than a configured manifest is opened.
    """

    if not isinstance(config, Mapping):
        raise TypeError("inventory config must be a mapping")
    allowed_roots = _allowed_roots(config)
    datasets = config.get("datasets")
    if not isinstance(datasets, Mapping):
        raise ValueError("inventory config needs a datasets object")
    result: dict[str, pd.DataFrame] = {}
    for dataset in ("stanford", "jjtms"):
        if dataset not in datasets:
            continue
        dataset_config = datasets[dataset]
        if not isinstance(dataset_config, Mapping):
            raise TypeError(f"{dataset} configuration must be an object")
        result[dataset] = _build_dataset_roster(dataset, dataset_config, allowed_roots)
    if not result:
        raise ValueError("inventory config contains no supported datasets")
    return result


def _format_input_path(
    root: Path,
    pattern: str,
    *,
    subject: str,
    smoothing_mm: int,
    hemisphere: str,
    allowed_roots: tuple[Path, ...],
) -> Path:
    subject_id = subject.removeprefix("sub-")
    values = {
        "subject": subject,
        "subject_id": subject_id,
        "smoothing": int(smoothing_mm),
        "smoothing_mm": int(smoothing_mm),
        "smoothing_label": f"{int(smoothing_mm)}mm",
        "hemisphere": hemisphere,
        "hemi": hemisphere,
    }
    try:
        rendered = pattern.format(**values)
    except (KeyError, ValueError) as exc:
        raise ValueError(f"invalid FC input pattern {pattern!r}: {exc}") from exc
    candidate = Path(rendered).expanduser()
    if not candidate.is_absolute():
        candidate = root / candidate
    return validate_rest_only_path(candidate, allowed_roots)


def _blocked_coverage_row(
    dataset: str,
    *,
    reason: str,
    missing_components: str,
    configured_estimator: str,
) -> dict[str, Any]:
    return {
        "dataset": dataset,
        "subject": "",
        "group": "",
        "smoothing_mm": pd.NA,
        "hemisphere": "",
        "status": "blocked",
        "missing_components": missing_components,
        "reason": reason,
        "fc_estimator": configured_estimator,
        "registered_fc_estimator": REGISTERED_FC_ESTIMATOR,
        "units": "Fisher-z FC",
        "fisher_z_path": "",
        "provenance_path": "",
        "path_opened": False,
    }


def inspect_fc_inputs(
    config: Mapping[str, Any], rosters: Mapping[str, pd.DataFrame]
) -> pd.DataFrame:
    """Return explicit FC/provenance coverage without opening asset files.

    Each row corresponds to one configured subject × smoothing × hemisphere
    combination.  ``path_opened`` is always false: this stage proves only
    explicit path presence and leaves array/header inspection to the tensor
    task.  Missing JJTMS inputs therefore remain visible as a concrete
    blocker instead of triggering guessed discovery.
    """

    if not isinstance(config, Mapping) or not isinstance(rosters, Mapping):
        raise TypeError("config and rosters must be mappings")
    allowed_roots = _allowed_roots(config)
    datasets = config.get("datasets")
    if not isinstance(datasets, Mapping):
        raise ValueError("inventory config needs a datasets object")
    smoothing_values = config.get("smoothings_mm", config.get("smoothing_mm", (2, 4)))
    hemispheres = config.get("hemispheres", ("L", "R"))
    if isinstance(smoothing_values, (str, int)):
        smoothing_values = (smoothing_values,)
    if isinstance(hemispheres, str):
        hemispheres = (hemispheres,)
    smoothings = tuple(int(value) for value in smoothing_values)
    hemis = tuple(str(value) for value in hemispheres)
    if not smoothings or not hemis:
        raise ValueError("inventory must configure at least one smoothing and hemisphere")

    rows: list[dict[str, Any]] = []
    for dataset, roster in rosters.items():
        if dataset not in datasets:
            raise ValueError(f"roster has no matching dataset configuration: {dataset}")
        dataset_config = datasets[dataset]
        if not isinstance(dataset_config, Mapping):
            raise TypeError(f"{dataset} configuration must be an object")
        gate = dataset_config.get("input_gate", {})
        gate_status = str(gate.get("status", "ready")) if isinstance(gate, Mapping) else "ready"
        fc_config = dataset_config.get("fc_inputs", {})
        if not isinstance(fc_config, Mapping):
            raise TypeError(f"{dataset}.fc_inputs must be an object")
        configured_estimator = str(
            fc_config.get("fc_estimator", REGISTERED_FC_ESTIMATOR)
        )
        if configured_estimator != REGISTERED_FC_ESTIMATOR:
            rows.append(
                _blocked_coverage_row(
                    dataset,
                    reason=(
                        f"unsupported fc_estimator {configured_estimator!r}; "
                        f"only registered {REGISTERED_FC_ESTIMATOR!r} is allowed"
                    ),
                    missing_components="fc_estimator",
                    configured_estimator=configured_estimator,
                )
            )
            continue
        pattern = fc_config.get("fisher_z_pattern")
        root_value = fc_config.get("root")
        provenance_pattern = fc_config.get("provenance_pattern")
        if gate_status == "blocked" or not pattern or root_value is None:
            reason = "no explicit JJTMS rest-only input is available"
            if isinstance(gate, Mapping) and gate.get("reason"):
                reason = str(gate["reason"])
            rows.append(
                _blocked_coverage_row(
                    dataset,
                    reason=reason,
                    missing_components="fisher_z_path,provenance_path",
                    configured_estimator=configured_estimator,
                )
            )
            continue
        if not provenance_pattern:
            rows.append(
                _blocked_coverage_row(
                    dataset,
                    reason=(
                        "provenance_pattern is required for a ready rest-FC "
                        "input; no provenance manifest can be verified"
                    ),
                    missing_components="provenance_pattern",
                    configured_estimator=configured_estimator,
                )
            )
            continue
        root = _resolve_config_path(str(root_value), allowed_roots)
        for record in roster.to_dict("records"):
            subject = _normalise_subject(record["subject"])
            for smoothing_mm in smoothings:
                for hemisphere in hemis:
                    fc_path = _format_input_path(
                        root,
                        str(pattern),
                        subject=subject,
                        smoothing_mm=smoothing_mm,
                        hemisphere=hemisphere,
                        allowed_roots=allowed_roots,
                    )
                    provenance_path = None
                    if provenance_pattern:
                        provenance_path = _format_input_path(
                            root,
                            str(provenance_pattern),
                            subject=subject,
                            smoothing_mm=smoothing_mm,
                            hemisphere=hemisphere,
                            allowed_roots=allowed_roots,
                        )
                    required = {
                        "fisher_z_path": fc_path,
                        "provenance_path": provenance_path,
                    }
                    missing = [name for name, value in required.items() if not value.is_file()]
                    status = "present" if not missing else "missing"
                    rows.append(
                        {
                            "dataset": dataset,
                            "subject": subject,
                            "group": record.get("group", ""),
                            "smoothing_mm": int(smoothing_mm),
                            "hemisphere": hemisphere,
                            "status": status,
                            "missing_components": ",".join(missing),
                            "reason": "" if not missing else "explicit rest-FC input is missing",
                            "fc_estimator": configured_estimator,
                            "registered_fc_estimator": REGISTERED_FC_ESTIMATOR,
                            "units": str(fc_config.get("units", "Fisher-z FC")),
                            "fisher_z_path": str(fc_path),
                            "provenance_path": "" if provenance_path is None else str(provenance_path),
                            "path_opened": False,
                        }
                    )
    columns = [
        "dataset",
        "subject",
        "group",
        "smoothing_mm",
        "hemisphere",
        "status",
        "missing_components",
        "reason",
        "fc_estimator",
        "registered_fc_estimator",
        "units",
        "fisher_z_path",
        "provenance_path",
        "path_opened",
    ]
    frame = pd.DataFrame(rows, columns=columns)
    if not frame.empty:
        frame["path_opened"] = frame["path_opened"].astype(object)
    frame.attrs["immutable"] = True
    frame.flags.allows_duplicate_labels = False
    return frame


def load_config(path: Path = DEFAULT_CONFIG_PATH) -> dict[str, Any]:
    """Load and JSON-decode one explicitly configured analysis config."""

    candidate = Path(path).expanduser()
    if not candidate.is_absolute():
        candidate = REPO_ROOT / candidate
    # The config is itself an input and is checked before opening.
    resolved = validate_rest_only_path(candidate, (REPO_ROOT,))
    with resolved.open(encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"analysis config must be a JSON object: {resolved}")
    return value


__all__ = [
    "DEFAULT_CONFIG_PATH",
    "build_rosters",
    "inspect_fc_inputs",
    "load_config",
    "validate_rest_only_path",
]
