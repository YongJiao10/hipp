"""Strict provenance checks for the Kong2022 400-ROI feature bank.

The historical ``hipp_vertex_to_network_fc*.npy`` names are retained for
compatibility, but under the Kong2022 gradient/WTA contract they contain one
column per ordered Kong parcel.  Canonical network summaries are separate
artifacts and are never accepted as an implicit substitute for this bank.
"""
from __future__ import annotations

import csv
import hashlib
import json
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np

from .kong2022_label_utils import load_kong2022_bundle


KONG2022_ROI_COUNT = 400


@lru_cache(maxsize=1)
def _cached_authoritative_kong2022_roi_registry() -> tuple[dict[str, Any], ...]:
    """Load the immutable shipped registry once per interpreter.

    FC validation runs once per subject/smoothing/hemi combination. Re-reading
    the same dlabel for every validation is both needlessly expensive and
    particularly costly on the shared filesystem; callers still receive fresh
    dictionaries from :func:`authoritative_kong2022_roi_registry` below.
    """
    rows = list(load_kong2022_bundle()["roi_registry"])
    if len(rows) != KONG2022_ROI_COUNT:
        raise ValueError(f"Authoritative Kong2022 registry must have 400 rows, got {len(rows)}")
    return tuple(dict(row) for row in rows)


def authoritative_kong2022_roi_registry() -> list[dict[str, Any]]:
    """Return the canonical ordered 400-row registry from the shipped dlabel."""
    return [dict(row) for row in _cached_authoritative_kong2022_roi_registry()]


def _row_value(row: dict[str, Any], *names: str, default: Any = "") -> Any:
    for name in names:
        if name in row:
            return row[name]
    return default


def read_kong2022_roi_registry(path: Path) -> list[dict[str, Any]]:
    """Read a per-subject registry TSV without accepting alternate ordering."""
    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    if len(rows) != KONG2022_ROI_COUNT:
        raise ValueError(f"Kong2022 ROI registry must have 400 rows: {path} rows={len(rows)}")
    out: list[dict[str, Any]] = []
    for index, row in enumerate(rows, start=1):
        out.append(
            {
                "roi_id": int(_row_value(row, "roi_id", "parcel_id", default=index)),
                "roi_name": str(_row_value(row, "roi_name", "parcel_name", default="")),
                "hemi": str(_row_value(row, "hemi", "hemisphere", default="")),
                "family17": str(_row_value(row, "family17", default="")),
                "subregion": str(_row_value(row, "subregion", default="")),
                "rank": int(_row_value(row, "rank", default=index) or index),
                "canonical8": str(_row_value(row, "canonical8", default="")),
                "canonical7": str(_row_value(row, "canonical7", default="")),
                # Normalize identity aliases so parcel_id/parcel_name TSVs are
                # checked against the authoritative ROI order as strictly as
                # roi_id/roi_name TSVs.
                "_provided_fields": (
                    set(row)
                    | ({"roi_id"} if any(key in row for key in ("roi_id", "parcel_id")) else set())
                    | ({"roi_name"} if any(key in row for key in ("roi_name", "parcel_name")) else set())
                ),
            }
        )
    return out


def validate_kong2022_roi_registry(
    rows: list[dict[str, Any]],
    *,
    path: Path | None = None,
) -> list[str]:
    """Validate exact IDs and names; return the ordered ROI names."""
    expected = authoritative_kong2022_roi_registry()
    if len(rows) != KONG2022_ROI_COUNT:
        raise ValueError(f"Kong2022 ROI order contract requires 400 rows, got {len(rows)}")
    mismatches: list[str] = []
    for index, (got, want) in enumerate(zip(rows, expected, strict=True)):
        # IDs and names define the feature-column order. Optional registry
        # annotations are checked whenever the TSV carries them, while small
        # contract fixtures may intentionally contain only the two identity
        # columns.
        provided = set(got.get("_provided_fields", ()))
        fields = ("roi_id", "roi_name") + tuple(
            field for field in ("hemi", "family17", "subregion", "rank", "canonical8", "canonical7")
            if field in provided and str(got.get(field, "")) != ""
        )
        for field in fields:
            if str(got.get(field, "")) != str(want.get(field, "")):
                mismatches.append(
                    f"row={index} field={field} got={got.get(field)!r} expected={want.get(field)!r}"
                )
    if mismatches:
        location = f" at {Path(path)}" if path is not None else ""
        raise ValueError("Kong2022 ROI order/metadata mismatch" + location + ": " + "; ".join(mismatches[:5]))
    return [str(row["roi_name"]) for row in expected]


def kong2022_registry_digest(rows: list[dict[str, Any]] | None = None) -> str:
    payload = rows if rows is not None else authoritative_kong2022_roi_registry()
    normalized = [
        {str(k): v for k, v in row.items() if str(k) != "_provided_fields"}
        for row in payload
    ]
    canonical = json.dumps(normalized, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def validate_kong2022_roi_fc(
    fc_path: Path,
    *,
    valid_mask: np.ndarray | None = None,
    reference_dir: Path | None = None,
) -> dict[str, Any]:
    """Validate a Kong FC array and its adjacent reference registry.

    The check intentionally requires the explicit ROI reference.  A seven
    column network cache, stale stage manifest, or missing registry fails
    closed even when the array happens to have a plausible number of rows.
    """
    path = Path(fc_path)
    arr = np.load(path, mmap_mode="r")
    if arr.ndim != 2:
        raise ValueError(f"Kong2022 FC must be 2D: {path} shape={arr.shape}")
    if int(arr.shape[1]) != KONG2022_ROI_COUNT:
        raise ValueError(
            f"Kong2022 gradient/WTA FC must contain exactly 400 ordered ROI columns: "
            f"{path} shape={arr.shape}"
        )
    if valid_mask is not None:
        mask = np.asarray(valid_mask, dtype=bool).reshape(-1)
        if arr.shape[0] not in {mask.size, int(np.sum(mask))}:
            raise ValueError(
                f"Kong2022 FC row count mismatch: {path} rows={arr.shape[0]} "
                f"vertex_count={mask.size} valid_count={int(np.sum(mask))}"
            )

    if reference_dir is None:
        # sub-*/fc/kong2022/<smooth>/hemi_* / file -> sub-*/reference/kong2022
        try:
            reference_dir = path.parents[4] / "reference" / "kong2022"
        except IndexError as exc:  # pragma: no cover - defensive path contract
            raise ValueError(f"Cannot infer Kong2022 reference directory from {path}") from exc
    reference_dir = Path(reference_dir)
    registry_path = reference_dir / "cortex_roi_parcels.tsv"
    timeseries_path = reference_dir / "cortex_roi_parcel_timeseries.npy"
    if not registry_path.exists():
        raise FileNotFoundError(f"Missing Kong2022 400-ROI registry for {path}: {registry_path}")
    if not timeseries_path.exists():
        raise FileNotFoundError(f"Missing Kong2022 400-ROI reference time series for {path}: {timeseries_path}")
    rows = read_kong2022_roi_registry(registry_path)
    roi_names = validate_kong2022_roi_registry(rows, path=registry_path)
    timeseries = np.load(timeseries_path, mmap_mode="r")
    if timeseries.ndim != 2 or int(timeseries.shape[0]) != KONG2022_ROI_COUNT:
        raise ValueError(
            f"Kong2022 ROI reference must have shape (400,time): {timeseries_path} shape={timeseries.shape}"
        )
    stage_manifest_path = path.parent / "stage_manifest.json"
    stage_manifest: dict[str, Any] = {}
    if stage_manifest_path.exists():
        try:
            stage_manifest = json.loads(stage_manifest_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise ValueError(f"Malformed FC stage manifest: {stage_manifest_path}") from exc
    summary_path = path.parent / "fc_summary.json"
    summary: dict[str, Any] = {}
    if summary_path.exists():
        try:
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise ValueError(f"Malformed FC summary: {summary_path}") from exc
        declared = summary.get("fc_shape")
        if isinstance(declared, list) and len(declared) == 2 and int(declared[1]) != KONG2022_ROI_COUNT:
            raise ValueError(f"FC summary declares non-400 Kong FC: {summary_path} fc_shape={declared}")
    return {
        "fc_path": str(path.resolve()),
        "reference_dir": str(reference_dir.resolve()),
        "registry_path": str(registry_path.resolve()),
        "timeseries_path": str(timeseries_path.resolve()),
        "roi_names": roi_names,
        "registry_digest": kong2022_registry_digest(rows),
        "stage_manifest": stage_manifest,
        "summary": summary,
        "shape": [int(v) for v in arr.shape],
    }
