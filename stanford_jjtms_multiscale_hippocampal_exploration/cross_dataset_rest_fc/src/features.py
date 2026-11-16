"""Auditable subject-level resting-state Fisher-z FC tensors.

This module is deliberately a small adapter around the explicit inventory
produced by :mod:`inventory`.  It reads only configured FC, provenance,
region-label/weight, and parcel-registry paths.  It never discovers a
replacement asset and it does not perform any group-level inference.

The FC arrays consumed here are vertex-by-parcel matrices.  A frozen parcel
registry assigns every parcel to one family-17 network and, through the same
row, to one canonical-8 network.  The 8-network values are therefore derived
from the very same parcel values as the 17-network values; no label-text
inference or second atlas is involved.
"""

from __future__ import annotations

import json
import re
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

try:  # Support both ``import features`` from src and package imports.
    from .inventory import REPO_ROOT, REGISTERED_FC_ESTIMATOR, _allowed_roots
    from .inventory import build_rosters, load_config, validate_rest_only_path
except ImportError:  # pragma: no cover - exercised by the focused tests
    from inventory import REPO_ROOT, REGISTERED_FC_ESTIMATOR, _allowed_roots
    from inventory import build_rosters, load_config, validate_rest_only_path


REGIONS = ("anterior", "middle", "posterior")
NETWORK_RESOLUTIONS = (17, 8)
HEMISPHERE_MODES = ("bilateral", "L", "R", "L-R")
OUTPUT_COLUMNS = [
    "dataset",
    "subject",
    "group",
    "trauma",
    "symptomatic",
    "method",
    "smoothing_mm",
    "hemisphere",
    "region",
    "network_resolution",
    "network",
    "fc_estimator",
    "fisher_z",
    "region_weight",
    "valid_vertex_count",
    "network_coverage",
    "status",
    "failure_reason",
]

# These are the names in the frozen Kong2022 parcel tables.  A synthetic or
# future registry may use another explicit vocabulary, in which case it must
# provide ``network_order_17`` and ``network_order_8`` in the configuration.
DEFAULT_NETWORK_ORDER_17 = (
    "DefaultA",
    "DefaultB",
    "DefaultC",
    "ContA",
    "ContB",
    "ContC",
    "SalVenAttnA",
    "SalVenAttnB",
    "DorsAttnA",
    "DorsAttnB",
    "SomMotA",
    "SomMotB",
    "VisualA",
    "VisualB",
    "VisualC",
    "Aud",
    "Language",
)
DEFAULT_NETWORK_ORDER_8 = (
    "Default",
    "Visual",
    "Somatomotor",
    "DorsalAttention",
    "VentralAttention",
    "Control",
    "Auditory",
    "Language",
)


def _normalise_subject(value: object) -> str:
    raw = str(value).strip()
    if raw.lower().startswith("sub-"):
        raw = raw[4:]
    if not raw or not raw.isdigit():
        raise ValueError(f"invalid subject identifier: {value!r}")
    return f"sub-{int(raw):04d}"


def _sort_subject_key(value: object) -> tuple[int, str]:
    text = str(value)
    suffix = text[4:] if text.lower().startswith("sub-") else text
    try:
        return int(suffix), text
    except ValueError:
        return (10**12, text)


def _as_sequence(value: object, *, name: str) -> tuple[Any, ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise TypeError(f"{name} must be a sequence")
    return tuple(value)


def _first(mapping: Mapping[str, Any], *keys: str) -> Any:
    for key in keys:
        if key in mapping and mapping[key] is not None:
            return mapping[key]
    return None


def _feature_config(
    config: Mapping[str, Any], dataset: str, dataset_config: Mapping[str, Any]
) -> Mapping[str, Any]:
    candidate = dataset_config.get("features")
    if candidate is None:
        candidate = config.get("features")
    if candidate is None:
        candidate = {}
    if not isinstance(candidate, Mapping):
        raise TypeError(f"{dataset}.features must be an object")
    return candidate


def _parcellation_specs(
    config: Mapping[str, Any], dataset: str, dataset_config: Mapping[str, Any]
) -> tuple[dict[str, Any], ...]:
    features = _feature_config(config, dataset, dataset_config)
    raw = dataset_config.get("parcellations")
    if raw is None:
        raw = features.get("parcellations")
    if raw is None:
        raw = features.get("methods")
    if raw is None:
        raw = config.get("parcellations")
    if raw is None:
        raise ValueError(
            f"{dataset} needs an explicit parcellations registry for FC tensor construction"
        )

    specs: list[dict[str, Any]] = []
    if isinstance(raw, Mapping):
        # A mapping can either be one spec (with a method key) or a deterministic
        # method-name -> spec registry.
        if any(key in raw for key in ("method", "labels_pattern", "registry_path", "registry")):
            raw = (raw,)
        else:
            entries = []
            for method, value in raw.items():
                if isinstance(value, Mapping):
                    item = dict(value)
                else:
                    item = {"labels_pattern": value}
                item.setdefault("method", str(method))
                entries.append(item)
            raw = entries
    elif isinstance(raw, Sequence) and not isinstance(raw, (str, bytes)):
        raw = tuple(raw)
    else:
        raise TypeError(f"{dataset}.parcellations must be a mapping or sequence")

    for item in raw:
        if isinstance(item, str):
            specs.append({"method": item})
        elif isinstance(item, Mapping):
            spec = dict(item)
            method = str(spec.get("method", "")).strip()
            if not method:
                raise ValueError(f"{dataset} parcellation entry has no method")
            spec["method"] = method
            specs.append(spec)
        else:
            raise TypeError(f"{dataset} parcellation entries must be objects")
    if not specs:
        raise ValueError(f"{dataset} parcellations registry is empty")
    methods = [spec["method"] for spec in specs]
    if len(methods) != len(set(methods)):
        raise ValueError(f"{dataset} parcellations contain duplicate methods")
    return tuple(specs)


def _network_orders(
    config: Mapping[str, Any],
    dataset_config: Mapping[str, Any],
    features: Mapping[str, Any],
    spec: Mapping[str, Any],
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    raw17 = _first(spec, "network_order_17", "networks_17")
    if raw17 is None:
        raw17 = _first(dataset_config, "network_order_17", "networks_17")
    if raw17 is None:
        raw17 = _first(features, "network_order_17", "networks_17")
    if raw17 is None:
        raw17 = _first(config, "network_order_17", "networks_17")
    raw8 = _first(spec, "network_order_8", "networks_8")
    if raw8 is None:
        raw8 = _first(dataset_config, "network_order_8", "networks_8")
    if raw8 is None:
        raw8 = _first(features, "network_order_8", "networks_8")
    if raw8 is None:
        raw8 = _first(config, "network_order_8", "networks_8")
    order17 = (
        tuple(str(value).strip() for value in _as_sequence(raw17, name="network_order_17"))
        if raw17 is not None
        else DEFAULT_NETWORK_ORDER_17
    )
    order8 = (
        tuple(str(value).strip() for value in _as_sequence(raw8, name="network_order_8"))
        if raw8 is not None
        else DEFAULT_NETWORK_ORDER_8
    )
    if len(order17) != 17 or len(set(order17)) != 17 or any(not value for value in order17):
        raise ValueError("network_order_17 must contain exactly 17 unique non-empty labels")
    if len(order8) != 8 or len(set(order8)) != 8 or any(not value for value in order8):
        raise ValueError("network_order_8 must contain exactly 8 unique non-empty labels")
    return order17, order8


def _allowed_roots_for(config: Mapping[str, Any]) -> tuple[Path, ...]:
    try:
        return _allowed_roots(config)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"invalid rest-FC allowed_roots: {exc}") from exc


def _render_path(
    value: str | Path,
    *,
    root: Path,
    subject: str,
    smoothing_mm: int,
    hemisphere: str,
    method: str,
    allowed_roots: tuple[Path, ...],
) -> Path:
    values = {
        "subject": subject,
        "subject_id": subject.removeprefix("sub-"),
        "smoothing": int(smoothing_mm),
        "smoothing_mm": int(smoothing_mm),
        "smoothing_label": f"{int(smoothing_mm)}mm",
        "hemisphere": hemisphere,
        "hemi": hemisphere,
        "method": method,
    }
    try:
        rendered = str(value).format(**values)
    except (KeyError, ValueError) as exc:
        raise ValueError(f"invalid feature input path pattern {value!r}: {exc}") from exc
    candidate = Path(rendered).expanduser()
    if not candidate.is_absolute():
        candidate = root / candidate
    return validate_rest_only_path(candidate, allowed_roots)


def _resolve_static_path(value: str | Path, allowed_roots: tuple[Path, ...]) -> Path:
    candidate = Path(value).expanduser()
    if not candidate.is_absolute():
        candidate = REPO_ROOT / candidate
    return validate_rest_only_path(candidate, allowed_roots)


def _configured_root(
    value: str | Path | None,
    *,
    fallback: Path,
    allowed_roots: tuple[Path, ...],
) -> Path:
    """Resolve an explicit sidecar root relative to the repository.

    FC roots and static sidecar roots are intentionally separate: a labels or
    provenance sidecar may live beside a frozen template rather than beside
    the subject FC array.  Both roots still pass through the same rest-only
    allow-list gate before they are used.
    """

    if value is None:
        return fallback
    candidate = Path(str(value)).expanduser()
    if not candidate.is_absolute():
        candidate = REPO_ROOT / candidate
    return validate_rest_only_path(candidate, allowed_roots)


def _asset_root(
    spec: Mapping[str, Any],
    key: str,
    *,
    fallback: Path,
    allowed_roots: tuple[Path, ...],
) -> Path:
    return _configured_root(
        spec.get(key), fallback=fallback, allowed_roots=allowed_roots
    )


def _select_combo_value(value: Any, *, subject: str, smoothing_mm: int, hemisphere: str) -> Any:
    """Select an optional inline value keyed by a known combination."""

    if not isinstance(value, Mapping):
        return value
    keys = (
        (subject, int(smoothing_mm), hemisphere),
        f"{subject}|{int(smoothing_mm)}|{hemisphere}",
        f"{subject}/{int(smoothing_mm)}mm/{hemisphere}",
        f"{subject}/{int(smoothing_mm)}mm/hemi-{hemisphere}",
        hemisphere,
        str(smoothing_mm),
        "default",
    )
    for key in keys:
        if key in value:
            return value[key]
    return value


def _load_json(path: Path) -> Mapping[str, Any]:
    try:
        with path.open(encoding="utf-8") as handle:
            value = json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"could not read provenance JSON {path}: {exc}") from exc
    if not isinstance(value, Mapping):
        raise ValueError(f"provenance JSON must be an object: {path}")
    return value


def _shape_from_provenance(provenance: Mapping[str, Any]) -> tuple[int, ...] | None:
    candidates: list[Any] = [
        provenance.get("fc_shape"),
        provenance.get("active_fc_shape"),
    ]
    outputs = provenance.get("outputs")
    if isinstance(outputs, Mapping):
        fisher = outputs.get("fisherz_fc")
        if isinstance(fisher, Mapping):
            candidates.append(fisher.get("shape"))
    for candidate in candidates:
        if candidate is None:
            continue
        if isinstance(candidate, Sequence) and not isinstance(candidate, (str, bytes)):
            try:
                return tuple(int(value) for value in candidate)
            except (TypeError, ValueError):
                return None
    return None


def _iter_provenance_fields(
    value: Any, path: tuple[str, ...] = ()
) -> Sequence[tuple[tuple[str, ...], Any]]:
    """Return deterministic key/value leaves plus their mapping paths."""

    fields: list[tuple[tuple[str, ...], Any]] = []
    if isinstance(value, Mapping):
        for key, child in value.items():
            child_path = path + (str(key),)
            fields.append((child_path, child))
            fields.extend(_iter_provenance_fields(child, child_path))
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        for index, child in enumerate(value):
            fields.extend(_iter_provenance_fields(child, path + (str(index),)))
    return fields


def _provenance_text(value: Any) -> str:
    return str(value).strip().lower()


_TASK_ENTITY_RE = re.compile(
    r"(?:^|[_./-])task[-_]([a-z0-9]+)(?=$|[_./-])"
)


def _task_entities(value: Any) -> tuple[str, ...]:
    if not isinstance(value, str):
        return ()
    text = _provenance_text(value).replace("–", "-").replace("—", "-")
    return tuple(
        re.sub(r"[^a-z0-9]+", "_", entity).strip("_")
        for entity in _TASK_ENTITY_RE.findall(text)
    )


def _provenance_forbidden_text(value: Any) -> str | None:
    text = _provenance_text(value).replace("–", "-").replace("—", "-")
    task_entities = _task_entities(value)
    if task_entities and any(entity != "rest" for entity in task_entities):
        return "non-rest task"
    if re.search(r"(^|[^a-z])beta(?:map)?([^a-z]|$)", text):
        return "beta"
    if re.search(r"(^|[^a-z])tms[ _-]?(?:response|contrast|mapping|activation)([^a-z]|$)", text):
        return "TMS-response"
    if re.search(r"(^|[^a-z])con[ _-]?(?:image|map|contrast)([^a-z]|$)", text):
        return "CON"
    if re.fullmatch(r"con", text):
        return "CON"
    if re.search(r"(^|[^a-z])contrast([^a-z]|$)", text):
        return "contrast"
    if re.search(r"(^|[^a-z])native[ _-]?(?:mapping|stimulation)([^a-z]|$)", text):
        return "native mapping"
    return None


_ACQUISITION_KEYS = frozenset(
    {"task", "task_name", "acquisition", "acquisition_task", "modality", "modality_name"}
)
_ROLE_KEYS = frozenset(
    {"data_role", "role", "fc_role", "fc", "fc_input", "fc_definition", "data_type"}
)
_ESTIMATOR_KEYS = frozenset(
    {"measure", "metric", "fc_estimator", "estimator", "fc_transform", "fc_space"}
)
_REST_ACQUISITION_ALIASES = frozenset(
    {"rest", "rest_fmri", "resting_state", "resting_state_fmri"}
)
_FC_ROLE_ALIASES = frozenset(
    {
        "fc",
        "functional_connectivity",
        "rest_fc",
        "resting_state_fc",
        "resting_state_functional_connectivity",
        "vertex_to_roi_fc",
        "vertex_to_roi_pearson",
        "pearson_vertex_to_kong2022_roi_timeseries",
        "precomputed_vertex_to_kong2022_roi_pearson_fc",
    }
)
_FISHER_Z_ALIASES = frozenset(
    {
        "fisher_z",
        "fisherz",
        "fisher_z_fc",
        "fisher_z_scale",
        "fisher_z_estimator",
        "arctanh_clipped_to_0_999999",
        "subject_mean_fisher_z_vertex_to_kong2022_roi_fc",
        "fisher_z_vertex_to_kong2022_roi_fc",
    }
)


def _semantic_normalize(value: Any) -> str:
    return re.sub(r"[^a-z0-9]+", "_", _provenance_text(value)).strip("_")


def _validate_semantic_alias(
    category: str,
    value: Any,
    *,
    path: Path,
    field_path: tuple[str, ...],
) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(
            f"provenance {category} field {'.'.join(field_path)!r} is not an explicit string: {path}"
        )
    normalized = _semantic_normalize(value)
    if category == "acquisition":
        allowed = _REST_ACQUISITION_ALIASES
    elif category == "role":
        allowed = _FC_ROLE_ALIASES
    elif category == "estimator":
        allowed = _FISHER_Z_ALIASES
    else:  # pragma: no cover - schema is internal and fixed
        raise ValueError(f"unknown provenance semantic category: {category!r}")
    if normalized not in allowed:
        raise ValueError(
            f"provenance {category} field {'.'.join(field_path)!r} has unsupported value {value!r}"
        )
    if category == "acquisition":
        return "rest"
    if category == "role":
        return "functional_connectivity"
    return "fisher_z"


def _extract_record_semantics(
    fields: Sequence[tuple[tuple[str, ...], Any]],
    *,
    provenance_path: Path,
) -> dict[str, set[str]]:
    """Validate recognized semantic aliases within one provenance record."""

    semantics = {"acquisition": set(), "role": set(), "estimator": set()}
    for field_path, value in fields:
        key = field_path[-1].lower() if field_path else ""
        if key in {"fisherz_fc", "fisher_z_fc", "fisher-z_fc"}:
            semantics["estimator"].add("fisher_z")
            continue
        if key in _ACQUISITION_KEYS:
            semantics["acquisition"].add(
                _validate_semantic_alias(
                    "acquisition",
                    value,
                    path=provenance_path,
                    field_path=field_path,
                )
            )
        elif key in _ROLE_KEYS:
            semantics["role"].add(
                _validate_semantic_alias(
                    "role",
                    value,
                    path=provenance_path,
                    field_path=field_path,
                )
            )
        elif key in _ESTIMATOR_KEYS:
            semantics["estimator"].add(
                _validate_semantic_alias(
                    "estimator",
                    value,
                    path=provenance_path,
                    field_path=field_path,
                )
            )
    return semantics


def _nested_provenance_value(
    record: Mapping[str, Any], field_path: str
) -> tuple[tuple[str, ...], Any]:
    current: Any = record
    parts = tuple(part for part in str(field_path).split(".") if part)
    if not parts:
        raise ValueError("provenance semantic schema has an empty field path")
    for part in parts:
        if not isinstance(current, Mapping) or part not in current:
            raise ValueError(
                f"provenance semantic field is missing: {field_path!r}"
            )
        current = current[part]
    return parts, current


def _apply_semantic_schema_extractor(value: Any, descriptor: Mapping[str, Any]) -> Any:
    extractor = descriptor.get("extract")
    if extractor is None:
        return value
    if extractor == "task_entity":
        if not isinstance(value, str):
            raise ValueError("task_entity provenance extractor requires a string field")
        entities = _task_entities(value)
        if not entities:
            raise ValueError(
                f"task_entity provenance extractor found no task entity in {value!r}"
            )
        if any(entity != "rest" for entity in entities):
            raise ValueError(
                "task_entity provenance extractor requires every task entity "
                f"to be rest, found {entities!r} in {value!r}"
            )
        return "rest"
    raise ValueError(f"unknown provenance semantic extractor: {extractor!r}")


def _schema_source_record(
    records: Sequence[tuple[Path, Mapping[str, Any]]], source: Any
) -> tuple[Path, Mapping[str, Any]]:
    if source == "primary":
        return records[0]
    text = str(source)
    if not text.startswith("sidecar:"):
        raise ValueError(f"provenance semantic schema has invalid source: {source!r}")
    try:
        index = int(text.split(":", 1)[1])
    except (TypeError, ValueError) as exc:
        raise ValueError(f"provenance semantic schema has invalid source: {source!r}") from exc
    if index < 0 or index >= len(records) - 1:
        raise ValueError(f"provenance semantic schema sidecar index is out of range: {source!r}")
    return records[index + 1]


def _validate_semantic_schema(
    records: Sequence[tuple[Path, Mapping[str, Any]]],
    schema: Mapping[str, Any],
) -> dict[str, set[str]]:
    """Validate an explicit adapter for known multi-file provenance bundles.

    Generic records cannot use partial semantic evidence.  This adapter is
    only for shipped manifests whose schema deliberately stores acquisition,
    FC role, and Fisher-z scale in named fields across named records.  Each
    descriptor is checked independently; values are never discovered by
    searching arbitrary strings.
    """

    if not isinstance(schema, Mapping):
        raise ValueError("provenance_semantics must be an object")
    required = ("acquisition", "role", "estimator")
    result: dict[str, set[str]] = {}
    for category in required:
        descriptor = schema.get(category)
        if not isinstance(descriptor, Mapping):
            raise ValueError(f"provenance semantic schema lacks {category!r} descriptor")
        source_path, record = _schema_source_record(records, descriptor.get("source"))
        field_path, value = _nested_provenance_value(record, str(descriptor.get("field", "")))
        schema_value = _apply_semantic_schema_extractor(value, descriptor)
        normalized = _semantic_normalize(schema_value)
        allowed_exact = descriptor.get("allowed_exact")
        if isinstance(allowed_exact, (str, bytes)) or not isinstance(allowed_exact, Sequence):
            raise ValueError(
                f"provenance {category} allowed_exact must be a sequence"
            )
        expected = {_semantic_normalize(item) for item in allowed_exact}
        if normalized not in expected:
            raise ValueError(
                f"provenance {category} schema value {schema_value!r} is not allowed: {source_path}"
            )
        result[category] = {
            "rest" if category == "acquisition" else
            "functional_connectivity" if category == "role" else "fisher_z"
        }
        # The source/field itself must be a scalar semantic value.  A mapping
        # can otherwise pass an empty normalized representation accidentally.
        if not isinstance(schema_value, str) or not schema_value.strip():
            raise ValueError(
                f"provenance {category} schema field {'.'.join(field_path)!r} is not a string"
            )
    return result


def _require_complete_record_semantics(
    semantics: Mapping[str, set[str]], provenance_path: Path
) -> None:
    missing = [category for category in ("acquisition", "role", "estimator") if not semantics.get(category)]
    if missing:
        raise ValueError(
            f"provenance record lacks explicit semantic field(s) {missing}: {provenance_path}"
        )


def _check_semantic_consistency(
    semantic_records: Sequence[tuple[Path, Mapping[str, set[str]]]],
    schema_semantics: Mapping[str, set[str]] | None = None,
) -> None:
    observed: dict[str, set[str]] = {"acquisition": set(), "role": set(), "estimator": set()}
    for _, semantics in semantic_records:
        for category, values in semantics.items():
            observed[category].update(values)
    if schema_semantics is not None:
        for category, values in schema_semantics.items():
            observed[category].update(values)
    for category, values in observed.items():
        if len(values) > 1:
            raise ValueError(
                f"provenance records disagree on semantic category {category!r}: {sorted(values)}"
            )
    missing = [category for category, values in observed.items() if not values]
    if missing:
        raise ValueError(f"provenance lacks explicit semantic field(s): {missing}")


def _normalise_identifier_text(value: Any) -> str:
    return re.sub(r"[^a-z0-9]+", "_", _provenance_text(value)).strip("_")


def _check_provenance_identity(
    fields: Sequence[tuple[tuple[str, ...], Any]],
    *,
    dataset: str,
    method: str,
) -> None:
    dataset_keys = {"dataset", "study", "cohort"}
    method_keys = {"method", "parcellation_method", "workflow"}
    expected_dataset = _normalise_identifier_text(dataset)
    expected_method = _normalise_identifier_text(method)
    for path, value in fields:
        if not isinstance(value, str):
            continue
        key = path[-1].lower() if path else ""
        text = _normalise_identifier_text(value)
        if key in dataset_keys:
            if expected_dataset == "stanford" and any(token in text for token in ("jjtms", "jj_tms")):
                raise ValueError(f"provenance dataset mismatch: {value!r} != {dataset!r}")
            if expected_dataset in {"jjtms", "jj_tms"} and "stanford" in text:
                raise ValueError(f"provenance dataset mismatch: {value!r} != {dataset!r}")
        if key not in method_keys:
            continue
        if expected_method == "synthetic_registry":
            if "synthetic_registry" not in text:
                raise ValueError(f"provenance method mismatch: {value!r} != {method!r}")
        elif expected_method == "hcp_fixed":
            if not any(token in text for token in ("hcp", "fixed", "template_matching")):
                raise ValueError(f"provenance method mismatch: {value!r} != {method!r}")
        elif expected_method == "hcp_template_matching":
            if not any(token in text for token in ("template_matching", "hcp_fixed")):
                raise ValueError(f"provenance method mismatch: {value!r} != {method!r}")
        elif expected_method not in text:
            raise ValueError(f"provenance method mismatch: {value!r} != {method!r}")


def _check_provenance(
    provenance_path: Path,
    provenance: Mapping[str, Any],
    *,
    dataset: str,
    method: str,
    estimator: str,
    subject: str,
    smoothing_mm: int,
    hemisphere: str,
    fc_shape: tuple[int, ...],
    additional_provenance: Sequence[tuple[Path, Mapping[str, Any]]] = (),
    semantic_schema: Mapping[str, Any] | None = None,
) -> None:
    expected_estimator = _normalise_identifier_text(estimator)
    if expected_estimator != _normalise_identifier_text(REGISTERED_FC_ESTIMATOR):
        raise ValueError(
            f"provenance estimator cannot be checked for unsupported estimator {estimator!r}"
        )
    records = [(provenance_path, provenance), *additional_provenance]
    semantic_records: list[tuple[Path, Mapping[str, set[str]]]] = []
    for path, record in records:
        fields = _iter_provenance_fields(record)
        for field_path, value in fields:
            for candidate in (".".join(field_path), value):
                forbidden = _provenance_forbidden_text(candidate)
                if forbidden is not None:
                    raise ValueError(
                        f"provenance contains prohibited {forbidden} evidence: {path}"
                    )
        semantics = _extract_record_semantics(fields, provenance_path=path)
        semantic_records.append((path, semantics))
        observed_shape = _shape_from_provenance(record)
        if observed_shape is not None and observed_shape != fc_shape:
            raise ValueError(
                f"provenance shape mismatch: manifest={observed_shape}, FC={fc_shape}"
            )
        for key in ("subject", "subject_id"):
            if key in record and record[key] not in (None, ""):
                if _normalise_subject(record[key]) != subject:
                    raise ValueError(
                        f"provenance subject mismatch: {record[key]!r} != {subject!r}"
                    )
                break
        observed_hemi = record.get("hemi", record.get("hemisphere"))
        if observed_hemi not in (None, "") and str(observed_hemi).upper() != hemisphere:
            raise ValueError(
                f"provenance hemisphere mismatch: {observed_hemi!r} != {hemisphere!r}"
            )
        observed_smoothing = record.get("smoothing", record.get("smooth"))
        if observed_smoothing not in (None, ""):
            text = str(observed_smoothing).lower().removesuffix("mm")
            try:
                if int(float(text)) != int(smoothing_mm):
                    raise ValueError(
                        f"provenance smoothing mismatch: {observed_smoothing!r} != {smoothing_mm}mm"
                    )
            except ValueError as exc:
                if "mismatch" in str(exc):
                    raise
                raise ValueError(f"invalid provenance smoothing: {observed_smoothing!r}") from exc
        status = record.get("status")
        if status not in (None, "", "ok", "done", "present", "success", "complete"):
            raise ValueError(f"provenance status is not usable: {status!r}")
        _check_provenance_identity(fields, dataset=dataset, method=method)
    if semantic_schema is None:
        for path, semantics in semantic_records:
            _require_complete_record_semantics(semantics, path)
        schema_semantics = None
    else:
        schema_semantics = _validate_semantic_schema(records, semantic_schema)
    _check_semantic_consistency(semantic_records, schema_semantics)


def _load_array(value: Any, *, path_label: str) -> np.ndarray:
    if isinstance(value, (str, Path)):
        path = Path(value)
        if not path.is_file():
            raise FileNotFoundError(f"{path_label} is missing: {path}")
        try:
            return np.asarray(np.load(path, allow_pickle=False))
        except (OSError, ValueError) as exc:
            raise ValueError(f"could not load {path_label}: {path}: {exc}") from exc
    try:
        return np.asarray(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"could not coerce inline {path_label}") from exc


def _load_table(value: Any, *, allowed_roots: tuple[Path, ...]) -> pd.DataFrame:
    if isinstance(value, pd.DataFrame):
        return value.copy(deep=True)
    if isinstance(value, Mapping):
        # A mapping with explicit row records is the least surprising inline
        # representation; a column mapping is accepted by DataFrame as well.
        try:
            return pd.DataFrame(value)
        except (TypeError, ValueError) as exc:
            raise ValueError("could not coerce inline parcel registry") from exc
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        try:
            return pd.DataFrame(value)
        except (TypeError, ValueError) as exc:
            raise ValueError("could not coerce inline parcel registry") from exc
    if isinstance(value, (str, Path)):
        path = _resolve_static_path(value, allowed_roots)
        if not path.is_file():
            raise FileNotFoundError(f"parcel registry is missing: {path}")
        suffix = path.suffix.lower()
        try:
            return pd.read_csv(path, sep="\t" if suffix in {".tsv", ".txt"} else ",")
        except (OSError, pd.errors.ParserError, UnicodeError) as exc:
            raise ValueError(f"could not read parcel registry {path}: {exc}") from exc
    raise ValueError("parcel registry must be a path, table, or row sequence")


def _registry_value(spec: Mapping[str, Any]) -> Any:
    value = _first(spec, "registry", "registry_path", "registry_pattern", "parcel_registry")
    if value is None:
        raise ValueError("parcellation needs an explicit parcel registry")
    return value


def _registry_columns(
    registry: pd.DataFrame,
    *,
    order17: tuple[str, ...],
    order8: tuple[str, ...],
    spec: Mapping[str, Any],
) -> tuple[np.ndarray, np.ndarray]:
    aliases17 = ("family17", "network17", "network_17", "network")
    aliases8 = ("canonical8", "network8", "network_8")
    col17 = next((column for column in aliases17 if column in registry.columns), None)
    col8 = next((column for column in aliases8 if column in registry.columns), None)
    if col17 is None or col8 is None:
        raise ValueError(
            "parcel registry needs family17 and canonical8 columns for deterministic 17/8 mapping"
        )
    index_values: Any = None
    index_label = ""
    configured_column = spec.get("fc_column_index_column")
    inline_index = spec.get("fc_column_index")
    if configured_column is not None:
        index_label = str(configured_column)
        if index_label not in registry.columns:
            raise ValueError(
                f"parcel registry is missing configured FC-column index column {index_label!r}"
            )
        index_values = registry[index_label].to_numpy()
    elif inline_index is not None and not isinstance(inline_index, str):
        index_label = "fc_column_index"
        index_values = np.asarray(inline_index)
        if index_values.size != len(registry):
            raise ValueError(
                "inline FC-column index must have one value per parcel registry row"
            )
    else:
        direct_index = next(
            (
                column
                for column in (
                    "fc_column",
                    "fc_column_index",
                    "fc_index",
                    "column_index",
                )
                if column in registry.columns
            ),
            None,
        )
        if direct_index is None:
            raise ValueError(
                "parcel registry needs an explicit integer FC-column index"
            )
        index_label = direct_index
        index_values = registry[direct_index].to_numpy()
    base_raw = spec.get("fc_column_index_base", 0)
    if isinstance(base_raw, bool):
        raise ValueError("FC-column index base must be integer 0 or 1")
    try:
        base = int(base_raw)
    except (TypeError, ValueError) as exc:
        raise ValueError("FC-column index base must be integer 0 or 1") from exc
    if base not in (0, 1) or (isinstance(base_raw, float) and base_raw != base):
        raise ValueError("FC-column index base must be integer 0 or 1")
    coerced: list[int] = []
    for value in np.asarray(index_values).reshape(-1):
        if isinstance(value, (bool, np.bool_)) or value is None or pd.isna(value):
            raise ValueError(f"FC-column index {index_label!r} contains a non-integer value")
        try:
            numeric = float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"FC-column index {index_label!r} contains a non-integer value"
            ) from exc
        if not np.isfinite(numeric) or numeric != np.floor(numeric):
            raise ValueError(
                f"FC-column index {index_label!r} contains a non-integer value"
            )
        coerced.append(int(numeric) - base)
    expected_indices = set(range(len(registry)))
    observed_indices = set(coerced)
    if len(coerced) != len(registry) or len(observed_indices) != len(coerced):
        raise ValueError(
            f"FC-column index {index_label!r} must be unique and contiguous 0..{len(registry) - 1}"
        )
    if observed_indices != expected_indices:
        raise ValueError(
            f"FC-column index {index_label!r} must be unique and contiguous 0..{len(registry) - 1}"
        )
    registry = registry.iloc[np.argsort(np.asarray(coerced), kind="stable")].reset_index(drop=True)
    for id_column in ("roi_id", "parcel_id", "column", "index"):
        if id_column in registry.columns:
            values = registry[id_column]
            if values.isna().any() or values.duplicated().any():
                raise ValueError(f"parcel registry has duplicate labels in {id_column}")
            break
    values17 = registry[col17].astype(object).to_numpy()
    values8 = registry[col8].astype(object).to_numpy()
    if values17.ndim != 1 or values8.ndim != 1 or values17.size != values8.size:
        raise ValueError("parcel registry network columns must be one-dimensional and aligned")
    if values17.size == 0:
        raise ValueError("parcel registry is empty")
    if any(value is None or pd.isna(value) or not str(value).strip() for value in values17):
        raise ValueError("parcel registry contains an empty family17 network label")
    if any(value is None or pd.isna(value) or not str(value).strip() for value in values8):
        raise ValueError("parcel registry contains an empty canonical8 network label")
    labels17 = np.asarray([str(value).strip() for value in values17], dtype=object)
    labels8 = np.asarray([str(value).strip() for value in values8], dtype=object)
    unexpected17 = sorted(set(labels17) - set(order17))
    unexpected8 = sorted(set(labels8) - set(order8))
    if unexpected17 or unexpected8:
        unexpected = unexpected17 + unexpected8
        raise ValueError(f"unknown network label(s) in parcel registry: {unexpected}")
    observed17 = set(labels17)
    observed8 = set(labels8)
    if observed17 != set(order17) or len(observed17) != 17:
        raise ValueError(
            f"parcel registry must contain exactly the configured 17 networks; observed {sorted(observed17)}"
        )
    if observed8 != set(order8) or len(observed8) != 8:
        raise ValueError(
            f"parcel registry must contain exactly the configured 8 networks; observed {sorted(observed8)}"
        )
    mapping: dict[str, set[str]] = {}
    for family17, canonical8 in zip(labels17, labels8):
        mapping.setdefault(family17, set()).add(canonical8)
    ambiguous = sorted(family17 for family17, parents in mapping.items() if len(parents) != 1)
    if ambiguous:
        raise ValueError(f"17-network labels map to multiple 8-network labels: {ambiguous}")
    return labels17, labels8


def _region_mapping(spec: Mapping[str, Any]) -> dict[str, str]:
    raw = spec.get("region_labels", spec.get("region_registry"))
    if raw is None:
        raw = {"1": "anterior", "2": "middle", "3": "posterior"}
    if not isinstance(raw, Mapping):
        raise TypeError("region_labels must map machine labels to anterior/middle/posterior")
    result: dict[str, str] = {}
    for machine, display in raw.items():
        label = str(display).strip().lower()
        if label not in REGIONS:
            raise ValueError(f"unknown hippocampal region label: {display!r}")
        key = str(machine).strip()
        if not key or key in result:
            raise ValueError(f"duplicate hippocampal region label: {machine!r}")
        result[key] = label
    if set(result.values()) != set(REGIONS) or len(result) != 3:
        raise ValueError("region registry must define exactly anterior, middle, and posterior")
    return result


def _map_regions(labels: np.ndarray, mapping: Mapping[str, str]) -> tuple[np.ndarray, np.ndarray]:
    raw = np.asarray(labels).reshape(-1)
    mapped = np.full(raw.size, "", dtype=object)
    valid = np.ones(raw.size, dtype=bool)
    for index, value in enumerate(raw):
        if value is None or (isinstance(value, str) and not value.strip()):
            valid[index] = False
            continue
        try:
            if pd.isna(value):
                valid[index] = False
                continue
        except (TypeError, ValueError):
            pass
        key = str(value).strip()
        if key in {"0", "-1", "nan", "NaN"}:
            valid[index] = False
            continue
        if key not in mapping:
            raise ValueError(f"unknown hippocampal region machine label: {value!r}")
        mapped[index] = mapping[key]
    return mapped, valid


def _empty_rows(
    metadata: Mapping[str, Any],
    *,
    network_order17: tuple[str, ...],
    network_order8: tuple[str, ...],
    network_resolutions: Sequence[int] = NETWORK_RESOLUTIONS,
    status: str,
    failure_reason: str,
    region_weights: Mapping[str, float] | None = None,
    valid_counts: Mapping[str, int] | None = None,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    region_weights = region_weights or {}
    valid_counts = valid_counts or {}
    network_orders = {17: network_order17, 8: network_order8}
    requested_resolutions = tuple(int(value) for value in network_resolutions)
    if any(resolution not in network_orders for resolution in requested_resolutions):
        raise ValueError(f"unsupported network resolution: {requested_resolutions!r}")
    for region in REGIONS:
        for resolution in requested_resolutions:
            networks = network_orders[resolution]
            for network in networks:
                rows.append(
                    {
                        **metadata,
                        "region": region,
                        "network_resolution": resolution,
                        "network": network,
                        "fisher_z": np.nan,
                        "region_weight": float(region_weights.get(region, 0.0)),
                        "valid_vertex_count": int(valid_counts.get(region, 0)),
                        "network_coverage": 0.0,
                        "status": status,
                        "failure_reason": failure_reason,
                    }
                )
    return rows


def _read_optional_combo_array(
    spec: Mapping[str, Any],
    keys: tuple[str, ...],
    *,
    root: Path,
    subject: str,
    smoothing_mm: int,
    hemisphere: str,
    method: str,
    allowed_roots: tuple[Path, ...],
    label: str,
    root_key: str | None = None,
) -> tuple[np.ndarray | None, str | None]:
    raw = _first(spec, *keys)
    if raw is None:
        return None, None
    selected = _select_combo_value(raw, subject=subject, smoothing_mm=smoothing_mm, hemisphere=hemisphere)
    if isinstance(selected, (str, Path)):
        selected_root = root
        if root_key is not None:
            selected_root = _asset_root(
                spec, root_key, fallback=root, allowed_roots=allowed_roots
            )
        path = _render_path(
            selected,
            root=selected_root,
            subject=subject,
            smoothing_mm=smoothing_mm,
            hemisphere=hemisphere,
            method=method,
            allowed_roots=allowed_roots,
        )
        if not path.is_file():
            return None, f"missing {label}: {path}"
        selected = path
    try:
        return _load_array(selected, path_label=label), None
    except (ValueError, FileNotFoundError) as exc:
        return None, str(exc)


def _load_registry_for_combo(
    spec: Mapping[str, Any],
    *,
    root: Path,
    subject: str,
    smoothing_mm: int,
    hemisphere: str,
    method: str,
    allowed_roots: tuple[Path, ...],
) -> pd.DataFrame:
    value = _registry_value(spec)
    selected = _select_combo_value(value, subject=subject, smoothing_mm=smoothing_mm, hemisphere=hemisphere)
    if isinstance(selected, (str, Path)) and any(token in str(selected) for token in ("{", "}")):
        selected = _render_path(
            selected,
            root=root,
            subject=subject,
            smoothing_mm=smoothing_mm,
            hemisphere=hemisphere,
            method=method,
            allowed_roots=allowed_roots,
        )
    return _load_table(selected, allowed_roots=allowed_roots)


def _method_root(
    spec: Mapping[str, Any], fc_root: Path, allowed_roots: tuple[Path, ...]
) -> Path:
    raw = spec.get("root")
    if raw is None:
        return fc_root
    candidate = Path(str(raw)).expanduser()
    if not candidate.is_absolute():
        candidate = fc_root / candidate
    return validate_rest_only_path(candidate, allowed_roots)


def _metadata(
    dataset: str,
    record: Mapping[str, Any],
    *,
    method: str,
    smoothing_mm: int,
    hemisphere: str,
    estimator: str,
) -> dict[str, Any]:
    return {
        "dataset": dataset,
        "subject": _normalise_subject(record.get("subject", "")),
        "group": record.get("group", ""),
        "trauma": record.get("trauma", pd.NA),
        "symptomatic": record.get("symptomatic", pd.NA),
        "method": method,
        "smoothing_mm": int(smoothing_mm),
        "hemisphere": hemisphere,
        "fc_estimator": estimator,
    }


def _combo_rows(
    metadata: Mapping[str, Any],
    fc: np.ndarray,
    labels: np.ndarray | None,
    vertex_weights: np.ndarray | None,
    registry: pd.DataFrame,
    *,
    spec: Mapping[str, Any],
    order17: tuple[str, ...],
    order8: tuple[str, ...],
    network_resolutions: Sequence[int] = NETWORK_RESOLUTIONS,
) -> list[dict[str, Any]]:
    values = np.asarray(fc)
    if values.ndim != 2:
        raise ValueError(f"FC array must be two-dimensional, got shape {values.shape}")
    if not np.issubdtype(values.dtype, np.number):
        raise ValueError(f"FC array must be numeric, got dtype {values.dtype}")
    labels_array: np.ndarray
    if labels is None:
        if values.shape[0] != 3:
            raise ValueError(
                "region labels sidecar is required unless FC has exactly three region rows"
            )
        labels_array = np.asarray(["anterior", "middle", "posterior"], dtype=object)
        region_map = {region: region for region in REGIONS}
    else:
        labels_array = np.asarray(labels).reshape(-1)
        region_map = _region_mapping(spec)
    if values.shape[0] != labels_array.size:
        raise ValueError(
            f"FC/region-label shape mismatch: FC has {values.shape[0]} rows, labels has {labels_array.size}"
        )
    mapped_regions, label_valid = _map_regions(labels_array, region_map)
    if vertex_weights is None:
        weights = np.ones(labels_array.size, dtype=np.float64)
    else:
        weights = np.asarray(vertex_weights).reshape(-1)
        if weights.size != labels_array.size:
            raise ValueError(
                f"FC/region-weight shape mismatch: FC has {labels_array.size} rows, weights has {weights.size}"
            )
        if not np.issubdtype(weights.dtype, np.number):
            raise ValueError("region weights must be numeric")
        weights = weights.astype(np.float64, copy=False)
    valid_mask_raw = spec.get("valid_mask")
    if valid_mask_raw is None:
        valid_mask = np.ones(labels_array.size, dtype=bool)
    else:
        valid_mask = np.asarray(valid_mask_raw).reshape(-1)
        if valid_mask.size != labels_array.size:
            raise ValueError("valid mask does not match FC vertex count")
        valid_mask = valid_mask.astype(bool, copy=False)
    if not np.isfinite(weights[label_valid & valid_mask]).all() or np.any(
        weights[label_valid & valid_mask] < 0
    ):
        raise ValueError("valid region weights must be finite and non-negative")
    if values.shape[1] != len(registry):
        raise ValueError(
            f"FC/parcel-registry shape mismatch: FC has {values.shape[1]} parcels, registry has {len(registry)}"
        )
    labels17, labels8 = _registry_columns(
        registry, order17=order17, order8=order8, spec=spec
    )
    rows: list[dict[str, Any]] = []
    finite_vertex_mask = np.isfinite(values).all(axis=1)
    for region in REGIONS:
        configured_region_mask = label_valid & valid_mask & (mapped_regions == region)
        region_mask = configured_region_mask & finite_vertex_mask
        valid_count = int(np.sum(region_mask))
        region_weight = float(np.sum(weights[region_mask])) if valid_count else 0.0
        network_orders = {
            17: (order17, labels17),
            8: (order8, labels8),
        }
        requested_resolutions = tuple(int(value) for value in network_resolutions)
        if any(resolution not in network_orders for resolution in requested_resolutions):
            raise ValueError(f"unsupported network resolution: {requested_resolutions!r}")
        for resolution in requested_resolutions:
            network_order, network_labels = network_orders[resolution]
            for network in network_order:
                parcel_mask = network_labels == network
                value = np.nan
                coverage = 0.0
                status = "present"
                reason = ""
                if valid_count == 0 or region_weight <= 0:
                    status = "missing"
                    reason = f"no finite valid vertices for region {region!r}"
                elif not parcel_mask.any():
                    status = "missing"
                    reason = f"network {network!r} has no registered parcels"
                else:
                    configured_component = values[
                        np.ix_(configured_region_mask, parcel_mask)
                    ]
                    if configured_component.size:
                        coverage = float(
                            np.sum(np.isfinite(configured_component))
                            / configured_component.size
                        )
                    component = values[np.ix_(region_mask, parcel_mask)]
                    finite = np.isfinite(component)
                    if not finite.all():
                        status = "failed"
                        reason = (
                            "nonfinite Fisher-z values for "
                            f"region={region!r}, network={network!r}"
                        )
                    else:
                        value = float(np.mean(component, dtype=np.float64))
                rows.append(
                    {
                        **metadata,
                        "region": region,
                        "network_resolution": resolution,
                        "network": network,
                        "fisher_z": value,
                        "region_weight": region_weight,
                        "valid_vertex_count": valid_count,
                        "network_coverage": coverage,
                        "status": status,
                        "failure_reason": reason,
                    }
                )
    return rows


def _rows_for_combo_failure(
    metadata: Mapping[str, Any],
    *,
    order17: tuple[str, ...],
    order8: tuple[str, ...],
    network_resolutions: Sequence[int] = NETWORK_RESOLUTIONS,
    status: str,
    reason: str,
    region_weights: Mapping[str, float] | None = None,
    valid_counts: Mapping[str, int] | None = None,
) -> list[dict[str, Any]]:
    return _empty_rows(
        metadata,
        network_order17=order17,
        network_order8=order8,
        network_resolutions=network_resolutions,
        status=status,
        failure_reason=reason,
        region_weights=region_weights,
        valid_counts=valid_counts,
    )


def _fc_inputs(
    dataset_config: Mapping[str, Any], features: Mapping[str, Any]
) -> Mapping[str, Any]:
    candidate = features.get("fc_inputs")
    if candidate is None:
        candidate = dataset_config.get("fc_inputs")
    if not isinstance(candidate, Mapping):
        raise ValueError("feature tensor needs the Task 1 fc_inputs mapping")
    return candidate


def _patterns_for_combo(
    fc_inputs: Mapping[str, Any], spec: Mapping[str, Any]
) -> tuple[Any, Any, Any]:
    root = _first(spec, "fc_root", "input_root")
    if root is None:
        root = fc_inputs.get("root")
    fc_pattern = _first(spec, "fisher_z_pattern", "fc_pattern", "fisher_z_path_pattern")
    if fc_pattern is None:
        fc_pattern = fc_inputs.get("fisher_z_pattern")
    provenance_pattern = _first(spec, "provenance_pattern", "provenance_path_pattern")
    if provenance_pattern is None:
        provenance_pattern = fc_inputs.get("provenance_pattern")
    return root, fc_pattern, provenance_pattern


def _provenance_sidecar_paths(
    spec: Mapping[str, Any],
    *,
    root: Path,
    subject: str,
    smoothing_mm: int,
    hemisphere: str,
    method: str,
    allowed_roots: tuple[Path, ...],
) -> tuple[Path, ...]:
    raw = spec.get("provenance_sidecar_patterns", ())
    if raw in (None, (), []):
        return ()
    if isinstance(raw, Mapping):
        raw = (raw,)
    elif isinstance(raw, (str, Path)):
        raw = (raw,)
    elif not isinstance(raw, Sequence):
        raise TypeError("provenance_sidecar_patterns must be a sequence")
    paths: list[Path] = []
    for item in raw:
        sidecar_root = root
        pattern: Any = item
        if isinstance(item, Mapping):
            pattern = _first(item, "pattern", "path", "provenance_pattern")
            if pattern is None:
                raise ValueError("provenance sidecar entry needs a path pattern")
            sidecar_root = _configured_root(
                item.get("root"), fallback=root, allowed_roots=allowed_roots
            )
        if not isinstance(pattern, (str, Path)):
            raise TypeError("provenance sidecar patterns must be paths")
        paths.append(
            _render_path(
                pattern,
                root=sidecar_root,
                subject=subject,
                smoothing_mm=smoothing_mm,
                hemisphere=hemisphere,
                method=method,
                allowed_roots=allowed_roots,
            )
        )
    if len(paths) != len(set(paths)):
        raise ValueError("provenance sidecar patterns contain duplicate paths")
    return tuple(paths)


def _sort_tensor(
    frame: pd.DataFrame,
    network_ordinals: Mapping[tuple[str, int, str], int] | None = None,
) -> pd.DataFrame:
    if frame.empty:
        return frame.reindex(columns=OUTPUT_COLUMNS)
    resolution_order = {17: 0, 8: 1}
    region_order = {name: index for index, name in enumerate(REGIONS)}
    frame = frame.copy()
    frame["_subject_order"] = frame["subject"].map(_sort_subject_key)
    frame["_resolution_order"] = frame["network_resolution"].map(resolution_order).fillna(99)
    frame["_region_order"] = frame["region"].map(region_order).fillna(99)
    ordinals = network_ordinals or {}
    frame["_network_order"] = [
        ordinals.get((str(method), int(resolution), str(network)), 10**6)
        for method, resolution, network in zip(
            frame["method"], frame["network_resolution"], frame["network"]
        )
    ]
    frame = frame.sort_values(
        [
            "_subject_order",
            "method",
            "smoothing_mm",
            "hemisphere",
            "_region_order",
            "_resolution_order",
            "_network_order",
        ],
        kind="stable",
    ).drop(columns=["_subject_order", "_resolution_order", "_region_order", "_network_order"])
    return frame.reindex(columns=OUTPUT_COLUMNS).reset_index(drop=True)


def build_fc_tensor(
    dataset: str,
    roster: pd.DataFrame,
    config: Mapping[str, Any],
    *,
    methods: Sequence[str] | None = None,
    smoothings_mm: Sequence[int] | None = None,
    network_resolutions: Sequence[int] | None = None,
) -> pd.DataFrame:
    """Build deterministic subject-level FC rows for selected analysis axes.

    Missing or incompatible subject combinations remain represented by rows
    with ``status`` and ``failure_reason`` rather than being replaced by a
    guessed path or a zero.  Configuration/schema errors that would make the
    registry itself ambiguous raise ``ValueError`` immediately.
    """

    if not isinstance(config, Mapping):
        raise TypeError("feature config must be a mapping")
    if not isinstance(roster, pd.DataFrame):
        raise TypeError("roster must be a pandas DataFrame")
    dataset = str(dataset)
    datasets = config.get("datasets")
    if not isinstance(datasets, Mapping) or dataset not in datasets:
        raise ValueError(f"feature config has no dataset {dataset!r}")
    dataset_config = datasets[dataset]
    if not isinstance(dataset_config, Mapping):
        raise TypeError(f"{dataset} dataset configuration must be an object")
    if "subject" not in roster.columns or "group" not in roster.columns:
        raise ValueError("roster needs subject and group columns")
    roster_records: list[dict[str, Any]] = []
    seen_subjects: set[str] = set()
    for raw in roster.to_dict("records"):
        record = dict(raw)
        record["subject"] = _normalise_subject(record["subject"])
        if record["subject"] in seen_subjects:
            raise ValueError("roster contains duplicate subjects after normalization")
        seen_subjects.add(record["subject"])
        if "dataset" in record and pd.notna(record["dataset"]) and str(record["dataset"]) != dataset:
            raise ValueError("roster dataset does not match requested dataset")
        roster_records.append(record)
    roster_records.sort(key=lambda record: _sort_subject_key(record["subject"]))

    allowed_roots = _allowed_roots_for(config)
    features = _feature_config(config, dataset, dataset_config)
    fc_inputs = _fc_inputs(dataset_config, features)
    estimator = str(fc_inputs.get("fc_estimator", REGISTERED_FC_ESTIMATOR))
    specs = _parcellation_specs(config, dataset, dataset_config)
    configured_methods = tuple(str(spec["method"]) for spec in specs)
    if methods is not None:
        requested_methods = tuple(
            str(value) for value in _as_sequence(methods, name="methods")
        )
        if not requested_methods:
            raise ValueError("methods must not be empty")
        if len(requested_methods) != len(set(requested_methods)):
            raise ValueError("methods contains duplicates")
        unknown_methods = sorted(set(requested_methods) - set(configured_methods))
        if unknown_methods:
            raise ValueError(f"methods are not configured for {dataset!r}: {unknown_methods}")
        specs = tuple(spec for spec in specs if str(spec["method"]) in requested_methods)
    smoothings_raw = config.get("smoothings_mm", config.get("smoothing_mm", (2, 4)))
    if isinstance(smoothings_raw, (str, int)):
        smoothings_raw = (smoothings_raw,)
    smoothings = tuple(sorted({int(value) for value in _as_sequence(smoothings_raw, name="smoothings_mm")}))
    if smoothings_mm is not None:
        requested_smoothings = tuple(
            int(value) for value in _as_sequence(smoothings_mm, name="smoothings_mm")
        )
        if not requested_smoothings:
            raise ValueError("smoothings_mm must not be empty")
        if len(requested_smoothings) != len(set(requested_smoothings)):
            raise ValueError("smoothings_mm contains duplicates")
        unknown_smoothings = sorted(set(requested_smoothings) - set(smoothings))
        if unknown_smoothings:
            raise ValueError(
                f"smoothings_mm are not configured for {dataset!r}: {unknown_smoothings}"
            )
        smoothings = tuple(value for value in smoothings if value in set(requested_smoothings))
    configured_resolutions = tuple(NETWORK_RESOLUTIONS)
    if network_resolutions is None:
        selected_network_resolutions = configured_resolutions
    else:
        requested_resolutions = tuple(
            int(value)
            for value in _as_sequence(network_resolutions, name="network_resolutions")
        )
        if not requested_resolutions:
            raise ValueError("network_resolutions must not be empty")
        if len(requested_resolutions) != len(set(requested_resolutions)):
            raise ValueError("network_resolutions contains duplicates")
        unknown_resolutions = sorted(
            set(requested_resolutions) - set(configured_resolutions)
        )
        if unknown_resolutions:
            raise ValueError(f"unsupported network_resolutions: {unknown_resolutions}")
        selected_network_resolutions = tuple(
            value for value in configured_resolutions if value in set(requested_resolutions)
        )
    hemis_raw = config.get("hemispheres", ("L", "R"))
    if isinstance(hemis_raw, str):
        hemis_raw = (hemis_raw,)
    hemispheres = tuple(str(value) for value in _as_sequence(hemis_raw, name="hemispheres"))
    if not smoothings or not hemispheres:
        raise ValueError("feature config needs at least one smoothing and hemisphere")
    if len(hemispheres) != len(set(hemispheres)):
        raise ValueError("feature config contains duplicate configured hemispheres")
    if any(hemi not in {"L", "R"} for hemi in hemispheres):
        raise ValueError("feature tensor supports only L/R input hemispheres")
    if estimator != REGISTERED_FC_ESTIMATOR:
        # The registry is still validated below where possible; no non-Fisher
        # values are ever read into a usable row.
        estimator_reason = (
            f"unsupported fc_estimator {estimator!r}; only registered "
            f"{REGISTERED_FC_ESTIMATOR!r} is allowed"
        )
    else:
        estimator_reason = ""

    rows: list[dict[str, Any]] = []
    network_ordinals: dict[tuple[str, int, str], int] = {}
    for spec in specs:
        method = str(spec["method"])
        order17, order8 = _network_orders(config, dataset_config, features, spec)
        for resolution, order in ((17, order17), (8, order8)):
            for ordinal, network in enumerate(order):
                key = (method, resolution, network)
                if key in network_ordinals and network_ordinals[key] != ordinal:
                    raise ValueError(f"conflicting configured network ordinal for {key!r}")
                network_ordinals[key] = ordinal
        availability = str(spec.get("availability", "available")).strip().lower()
        if availability not in {"", "available"}:
            status = availability if availability in {"unavailable", "missing", "blocked"} else "blocked"
            reason = str(
                spec.get(
                    "availability_reason",
                    f"parcellation method {method!r} is configured as {availability!r}",
                )
            )
            for record in roster_records:
                for smoothing_mm in smoothings:
                    for hemisphere in hemispheres:
                        metadata = _metadata(
                            dataset,
                            record,
                            method=method,
                            smoothing_mm=smoothing_mm,
                            hemisphere=hemisphere,
                            estimator=estimator,
                        )
                        rows.extend(
                            _rows_for_combo_failure(
                                metadata,
                                order17=order17,
                                order8=order8,
                                network_resolutions=selected_network_resolutions,
                                status=status,
                                reason=reason,
                            )
                        )
            continue
        root_value, fc_pattern, provenance_pattern = _patterns_for_combo(fc_inputs, spec)
        if root_value is None or fc_pattern is None or provenance_pattern is None:
            reason = (
                "feature input requires explicit root, fisher_z_pattern, and "
                "provenance_pattern"
            )
            for record in roster_records:
                for smoothing_mm in smoothings:
                    for hemisphere in hemispheres:
                        metadata = _metadata(
                            dataset,
                            record,
                            method=method,
                            smoothing_mm=smoothing_mm,
                            hemisphere=hemisphere,
                            estimator=estimator,
                        )
                        rows.extend(
                            _rows_for_combo_failure(
                                metadata,
                                order17=order17,
                                order8=order8,
                                network_resolutions=selected_network_resolutions,
                                status="blocked",
                                reason=reason,
                            )
                        )
            continue
        root_candidate = Path(str(root_value)).expanduser()
        if not root_candidate.is_absolute():
            root_candidate = REPO_ROOT / root_candidate
        root = validate_rest_only_path(root_candidate, allowed_roots)
        for record in roster_records:
            subject = _normalise_subject(record["subject"])
            for smoothing_mm in smoothings:
                for hemisphere in hemispheres:
                    metadata = _metadata(
                        dataset,
                        record,
                        method=method,
                        smoothing_mm=smoothing_mm,
                        hemisphere=hemisphere,
                        estimator=estimator,
                    )
                    if estimator_reason:
                        rows.extend(
                            _rows_for_combo_failure(
                                metadata,
                                order17=order17,
                                order8=order8,
                                network_resolutions=selected_network_resolutions,
                                status="blocked",
                                reason=estimator_reason,
                            )
                        )
                        continue
                    try:
                        fc_path = _render_path(
                            fc_pattern,
                            root=root,
                            subject=subject,
                            smoothing_mm=smoothing_mm,
                            hemisphere=hemisphere,
                            method=method,
                            allowed_roots=allowed_roots,
                        )
                        provenance_path = _render_path(
                            provenance_pattern,
                            root=root,
                            subject=subject,
                            smoothing_mm=smoothing_mm,
                            hemisphere=hemisphere,
                            method=method,
                            allowed_roots=allowed_roots,
                        )
                        provenance_sidecar_paths = _provenance_sidecar_paths(
                            spec,
                            root=root,
                            subject=subject,
                            smoothing_mm=smoothing_mm,
                            hemisphere=hemisphere,
                            method=method,
                            allowed_roots=allowed_roots,
                        )
                    except (TypeError, ValueError) as exc:
                        rows.extend(
                            _rows_for_combo_failure(
                                metadata,
                                order17=order17,
                                order8=order8,
                                network_resolutions=selected_network_resolutions,
                                status="failed",
                                reason=str(exc),
                            )
                        )
                        continue
                    missing_sidecars = [
                        f"provenance sidecar {path}"
                        for path in provenance_sidecar_paths
                        if not path.is_file()
                    ]
                    if not fc_path.is_file() or not provenance_path.is_file() or missing_sidecars:
                        missing = []
                        if not fc_path.is_file():
                            missing.append(f"FC array {fc_path}")
                        if not provenance_path.is_file():
                            missing.append(f"provenance sidecar {provenance_path}")
                        missing.extend(missing_sidecars)
                        rows.extend(
                            _rows_for_combo_failure(
                                metadata,
                                order17=order17,
                                order8=order8,
                                network_resolutions=selected_network_resolutions,
                                status="missing",
                                reason="missing " + "; ".join(missing),
                            )
                        )
                        continue
                    try:
                        fc = _load_array(fc_path, path_label="Fisher-z FC array")
                        if fc.ndim != 2:
                            raise ValueError(f"FC array must be two-dimensional, got shape {fc.shape}")
                        provenance = _load_json(provenance_path)
                        additional_provenance = tuple(
                            (path, _load_json(path)) for path in provenance_sidecar_paths
                        )
                        _check_provenance(
                            provenance_path,
                            provenance,
                            dataset=dataset,
                            method=method,
                            estimator=estimator,
                            subject=subject,
                            smoothing_mm=smoothing_mm,
                            hemisphere=hemisphere,
                            fc_shape=tuple(int(value) for value in fc.shape),
                            additional_provenance=additional_provenance,
                            semantic_schema=spec.get("provenance_semantics"),
                        )
                        registry = _load_registry_for_combo(
                            spec,
                            root=root,
                            subject=subject,
                            smoothing_mm=smoothing_mm,
                            hemisphere=hemisphere,
                            method=method,
                            allowed_roots=allowed_roots,
                        )
                        labels, labels_reason = _read_optional_combo_array(
                            spec,
                            ("labels_pattern", "region_labels_pattern", "label_pattern", "labels_path", "region_labels_path", "labels"),
                            root=root,
                            subject=subject,
                            smoothing_mm=smoothing_mm,
                            hemisphere=hemisphere,
                            method=method,
                            allowed_roots=allowed_roots,
                            label="region labels",
                            root_key="labels_root",
                        )
                        if labels_reason:
                            raise FileNotFoundError(labels_reason)
                        weights, weights_reason = _read_optional_combo_array(
                            spec,
                            ("region_weight_pattern", "region_weights_pattern", "vertex_weight_pattern", "weights_pattern", "weights_path", "region_weights"),
                            root=root,
                            subject=subject,
                            smoothing_mm=smoothing_mm,
                            hemisphere=hemisphere,
                            method=method,
                            allowed_roots=allowed_roots,
                            label="region weights",
                            root_key="region_weight_root",
                        )
                        if weights_reason:
                            raise FileNotFoundError(weights_reason)
                        valid_mask, valid_reason = _read_optional_combo_array(
                            spec,
                            ("valid_mask_pattern", "valid_vertices_pattern", "valid_mask_path", "valid_mask"),
                            root=root,
                            subject=subject,
                            smoothing_mm=smoothing_mm,
                            hemisphere=hemisphere,
                            method=method,
                            allowed_roots=allowed_roots,
                            label="valid-vertex mask",
                            root_key="valid_mask_root",
                        )
                        if valid_reason:
                            raise FileNotFoundError(valid_reason)
                        spec_for_combo = dict(spec)
                        if valid_mask is not None:
                            spec_for_combo["valid_mask"] = valid_mask
                        rows.extend(
                            _combo_rows(
                                metadata,
                                fc,
                                labels,
                                weights,
                                registry,
                                spec=spec_for_combo,
                                order17=order17,
                                order8=order8,
                                network_resolutions=selected_network_resolutions,
                            )
                        )
                    except FileNotFoundError as exc:
                        rows.extend(
                            _rows_for_combo_failure(
                                metadata,
                                order17=order17,
                                order8=order8,
                                network_resolutions=selected_network_resolutions,
                                status="missing",
                                reason=str(exc),
                            )
                        )
                    except (OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
                        rows.extend(
                            _rows_for_combo_failure(
                                metadata,
                                order17=order17,
                                order8=order8,
                                network_resolutions=selected_network_resolutions,
                                status="failed",
                                reason=f"{type(exc).__name__}: {exc}",
                            )
                        )
    frame = pd.DataFrame(rows, columns=OUTPUT_COLUMNS)
    frame.attrs["immutable"] = True
    frame.attrs["registered_fc_estimator"] = REGISTERED_FC_ESTIMATOR
    frame.attrs["network_ordinals"] = dict(network_ordinals)
    frame.flags.allows_duplicate_labels = False
    result = _sort_tensor(frame, network_ordinals)
    result.attrs.update(frame.attrs)
    return result


def run_config_smoke(
    config_or_path: Mapping[str, Any] | str | Path,
    *,
    subjects_per_dataset: int = 1,
) -> pd.DataFrame:
    """Read a bounded subject sample from the shipped config and summarize it.

    This is intentionally a read-only compatibility check.  It opens only the
    configured roster, FC, provenance, label, mask, and registry paths and
    returns one deterministic status row per dataset × registered method.  It
    never writes tensors or scientific result files.
    """

    if isinstance(config_or_path, (str, Path)):
        config = load_config(Path(config_or_path))
    elif isinstance(config_or_path, Mapping):
        config = config_or_path
    else:
        raise TypeError("config_or_path must be a config mapping or JSON path")
    if isinstance(subjects_per_dataset, bool) or int(subjects_per_dataset) != subjects_per_dataset:
        raise ValueError("subjects_per_dataset must be a positive integer")
    subjects_per_dataset = int(subjects_per_dataset)
    if subjects_per_dataset < 1:
        raise ValueError("subjects_per_dataset must be a positive integer")
    rosters = build_rosters(config)
    datasets = config.get("datasets")
    if not isinstance(datasets, Mapping):
        raise ValueError("config needs a datasets mapping")
    rows: list[dict[str, Any]] = []
    for dataset in ("stanford", "jjtms"):
        if dataset not in rosters or dataset not in datasets:
            continue
        roster = rosters[dataset].copy(deep=True)
        roster["_subject_order"] = roster["subject"].map(_sort_subject_key)
        roster = roster.sort_values("_subject_order", kind="stable").drop(columns="_subject_order")
        sample = roster.head(subjects_per_dataset).reset_index(drop=True)
        tensor = build_fc_tensor(dataset, sample, config)
        features = _feature_config(config, dataset, datasets[dataset])
        specs = _parcellation_specs(config, dataset, datasets[dataset])
        for spec in specs:
            method = str(spec["method"])
            subset = tensor.loc[tensor["method"] == method]
            status_counts = subset["status"].value_counts() if not subset.empty else pd.Series(dtype=int)
            tensor_rows = int(len(subset))
            present_rows = int(status_counts.get("present", 0))
            unavailable_rows = int(status_counts.get("unavailable", 0))
            missing_rows = int(status_counts.get("missing", 0))
            failed_rows = int(
                status_counts.get("failed", 0)
                + status_counts.get("blocked", 0)
            )
            if present_rows:
                summary_status = "available"
            elif tensor_rows and unavailable_rows == tensor_rows:
                summary_status = "unavailable"
            elif tensor_rows and missing_rows == tensor_rows:
                summary_status = "missing"
            else:
                summary_status = "failed"
            rows.append(
                {
                    "dataset": dataset,
                    "method": method,
                    "subject_count": int(len(sample)),
                    "tensor_rows": tensor_rows,
                    "present_rows": present_rows,
                    "unavailable_rows": unavailable_rows,
                    "missing_rows": missing_rows,
                    "failed_rows": failed_rows,
                    "status": summary_status,
                }
            )
    columns = [
        "dataset",
        "method",
        "subject_count",
        "tensor_rows",
        "present_rows",
        "unavailable_rows",
        "missing_rows",
        "failed_rows",
        "status",
    ]
    frame = pd.DataFrame(rows, columns=columns)
    frame.attrs["immutable"] = True
    frame.flags.allows_duplicate_labels = False
    return frame


def _valid_row(row: Mapping[str, Any]) -> bool:
    return (
        str(row.get("status", "")) == "present"
        and pd.notna(row.get("fisher_z"))
        and pd.notna(row.get("region_weight"))
        and float(row.get("region_weight", 0.0)) > 0
    )


def _view_metadata(base: Mapping[str, Any], hemisphere: str, mode: str) -> dict[str, Any]:
    result = dict(base)
    result["hemisphere"] = hemisphere
    result["hemisphere_mode"] = mode
    return result


def build_hemisphere_views(tensor: pd.DataFrame) -> pd.DataFrame:
    """Add raw L/R, weighted bilateral, and paired L-minus-R views.

    Bilateral values are weighted means of the two subject-level Fisher-z
    values using their recorded ``region_weight``.  L-minus-R is a paired
    within-subject subtraction in Fisher-z units.  Either view is marked
    missing when one required side is invalid; no zero is inserted.
    """

    if not isinstance(tensor, pd.DataFrame):
        raise TypeError("tensor must be a pandas DataFrame")
    missing = [column for column in OUTPUT_COLUMNS if column not in tensor.columns]
    if missing:
        raise ValueError(f"tensor is missing columns: {missing}")
    source = tensor[OUTPUT_COLUMNS].copy(deep=True)
    network_ordinals = tensor.attrs.get("network_ordinals", {})
    key_columns = [
        "dataset",
        "subject",
        "group",
        "trauma",
        "symptomatic",
        "method",
        "smoothing_mm",
        "region",
        "network_resolution",
        "network",
        "fc_estimator",
    ]
    identity = key_columns + ["hemisphere"]
    if source.duplicated(identity).any():
        raise ValueError("tensor contains duplicate subject/hemisphere/network keys")
    if not set(source["hemisphere"].dropna().astype(str)).issubset({"L", "R"}):
        raise ValueError("tensor hemisphere values must be L or R")

    rows: list[dict[str, Any]] = []
    for record in source.to_dict("records"):
        rows.append(_view_metadata(record, str(record["hemisphere"]), str(record["hemisphere"])))

    group_columns = key_columns
    grouped = source.groupby(group_columns, dropna=False, sort=False)
    for group_key, group in grouped:
        base = dict(zip(group_columns, group_key))
        side = {str(row["hemisphere"]): row for row in group.to_dict("records")}
        left = side.get("L")
        right = side.get("R")
        for mode in ("bilateral", "L-R"):
            metadata = dict(base)
            metadata["hemisphere"] = mode
            metadata["hemisphere_mode"] = mode
            if left is None or right is None:
                reason = f"{mode} requires valid left and right rows"
                metadata.update(
                    {
                        "fisher_z": np.nan,
                        "region_weight": 0.0,
                        "valid_vertex_count": 0,
                        "network_coverage": 0.0,
                        "status": "missing",
                        "failure_reason": reason,
                    }
                )
                rows.append(metadata)
                continue
            if not _valid_row(left) or not _valid_row(right):
                invalid_side = "left" if not _valid_row(left) else "right"
                reason = f"{mode} requires valid {invalid_side} row; no zero imputation"
                metadata.update(
                    {
                        "fisher_z": np.nan,
                        "region_weight": 0.0,
                        "valid_vertex_count": 0,
                        "network_coverage": 0.0,
                        "status": "missing",
                        "failure_reason": reason,
                    }
                )
                rows.append(metadata)
                continue
            left_weight = float(left["region_weight"])
            right_weight = float(right["region_weight"])
            if mode == "bilateral":
                weight = left_weight + right_weight
                value = (
                    float(left["fisher_z"]) * left_weight
                    + float(right["fisher_z"]) * right_weight
                ) / weight
                valid_count = int(left["valid_vertex_count"]) + int(right["valid_vertex_count"])
                coverage = (
                    float(left["network_coverage"]) * left_weight
                    + float(right["network_coverage"]) * right_weight
                ) / weight
            else:
                weight = min(left_weight, right_weight)
                value = float(left["fisher_z"]) - float(right["fisher_z"])
                valid_count = min(int(left["valid_vertex_count"]), int(right["valid_vertex_count"]))
                coverage = min(float(left["network_coverage"]), float(right["network_coverage"]))
            metadata.update(
                {
                    "fisher_z": value,
                    "region_weight": weight,
                    "valid_vertex_count": valid_count,
                    "network_coverage": coverage,
                    "status": "present",
                    "failure_reason": "",
                }
            )
            rows.append(metadata)
    columns = OUTPUT_COLUMNS[:8] + ["hemisphere_mode"] + OUTPUT_COLUMNS[8:]
    frame = pd.DataFrame(rows)
    # The mode is an additional key, not a replacement for the source
    # hemisphere column.  Reorder to keep the contract easy to inspect.
    frame = frame.reindex(columns=columns)
    if frame.empty:
        frame.attrs["immutable"] = True
        frame.flags.allows_duplicate_labels = False
        return frame
    mode_order = {mode: index for index, mode in enumerate(HEMISPHERE_MODES)}
    region_order = {name: index for index, name in enumerate(REGIONS)}
    frame["_subject_order"] = frame["subject"].map(_sort_subject_key)
    frame["_mode_order"] = frame["hemisphere_mode"].map(mode_order).fillna(99)
    frame["_region_order"] = frame["region"].map(region_order).fillna(99)
    frame["_resolution_order"] = frame["network_resolution"].map({17: 0, 8: 1}).fillna(99)
    frame["_network_order"] = [
        network_ordinals.get((str(method), int(resolution), str(network)), 10**6)
        for method, resolution, network in zip(
            frame["method"], frame["network_resolution"], frame["network"]
        )
    ]
    frame = frame.sort_values(
        [
            "_subject_order",
            "method",
            "smoothing_mm",
            "_mode_order",
            "_region_order",
            "_resolution_order",
            "_network_order",
        ],
        kind="stable",
    ).drop(columns=["_subject_order", "_mode_order", "_region_order", "_resolution_order", "_network_order"])
    frame.attrs["immutable"] = True
    frame.attrs["network_ordinals"] = network_ordinals
    frame.flags.allows_duplicate_labels = False
    return frame.reset_index(drop=True)


__all__ = ["build_fc_tensor", "build_hemisphere_views", "run_config_smoke"]
