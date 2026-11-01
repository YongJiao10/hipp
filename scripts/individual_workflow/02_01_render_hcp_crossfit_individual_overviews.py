#!/usr/bin/env python3
"""Render compact 2 mm/4 mm per-subject cross-fit hippocampal overviews."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sys
import tempfile
import time
import uuid
from collections.abc import Mapping, Sequence
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import nibabel as nib
import numpy as np
from nibabel.gifti import GiftiDataArray, GiftiImage, GiftiLabel, GiftiLabelTable
from PIL import Image, ImageDraw

REPO_ROOT = Path(__file__).resolve().parents[2]
CORE_WORKFLOW_DIR = REPO_ROOT / "scripts" / "core_workflow"
if str(CORE_WORKFLOW_DIR) not in sys.path:
    sys.path.insert(0, str(CORE_WORKFLOW_DIR))

from lib.kong2022_label_utils import load_kong2022_bundle  # noqa: E402
from lib.network_colors import require_network_colors  # noqa: E402
from lib.output_layout import (  # noqa: E402
    resolve_crossfit_template_matching_individual_overview_root,
    resolve_crossfit_template_matching_root,
)
from lib.surface_utils import (  # noqa: E402
    infer_spec_path,
    read_surface_vertex_count,
    render_locked_grid,
)
from lib.workbench_grid import load_font, wrap_text  # noqa: E402

DEFAULT_SMOOTHINGS = ("2mm", "4mm")
DEFAULT_HEMIS = ("L", "R")
DEFAULT_JOBS = 4
DEFAULT_SCENE = REPO_ROOT / "config" / "wb_locked_native_view_lateral_medial.scene"
LABEL_SEMANTICS = {"1": "Anterior", "2": "Middle", "3": "Posterior"}
DISPLAY_NULL_LABEL = (0, "Null", (158, 158, 158, 255))
FISHERZ_NETWORK_DISPLAY_SPECS = {
    # Keep compact legend aliases, but index colors by canonical Kong2022 networks.
    "VentralAttention": {"display_name": "VAN", "style_name": "VentralAttention"},
    "Somatomotor": {"display_name": "SMN", "style_name": "Somatomotor"},
    "Default": {"display_name": "DMN", "style_name": "Default"},
}
VISUAL_LABEL_CONTRACT_SCHEMA_VERSION = 1
WORKFLOW_NAME = "hcp_routea_crossfit_k3_individual_overviews"
THREAD_ENV_OVERRIDES = {
    "OMP_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1",
}
STATUS_FIELDS = (
    "status",
    "subject",
    "output",
    "resumed",
    "elapsed_seconds",
    "error_type",
    "error",
)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return payload


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(
        json.dumps(dict(payload), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _fingerprint(path: Path) -> dict[str, object]:
    digest = hashlib.sha256()
    size = 0
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
            size += len(chunk)
    return {"sha256": digest.hexdigest(), "size_bytes": int(size)}


def _normalize_subject(value: str) -> str:
    subject = str(value).strip().removeprefix("sub-")
    if not subject or not subject.isdigit():
        raise ValueError(
            f"Invalid subject {value!r}; expected numeric ID with optional sub-"
        )
    return subject


def _combo_dir(root: Path, subject: str, smoothing: str, hemi: str) -> Path:
    return (
        Path(root)
        / "subjects"
        / f"sub-{_normalize_subject(subject)}"
        / str(smoothing)
        / f"hemi_{hemi}"
    )


def subject_overview_png_path(output_dir: Path, subject: str) -> Path:
    return (
        Path(output_dir)
        / f"sub-{_normalize_subject(subject)}_crossfit_template_matching_k3_overview.png"
    )


def _validate_fingerprint(recorded: object, path: Path, context: str) -> None:
    if not isinstance(recorded, dict) or dict(recorded) != _fingerprint(path):
        raise ValueError(f"{context} fingerprint mismatch")


def _duplicates(values: Sequence[str]) -> list[str]:
    seen: set[str] = set()
    duplicates: list[str] = []
    for value in values:
        if value in seen and value not in duplicates:
            duplicates.append(value)
        seen.add(value)
    return duplicates


def _require_canonical_smoothings(requested: Sequence[str]) -> list[str]:
    selected = tuple(str(value) for value in requested)
    if selected != DEFAULT_SMOOTHINGS:
        raise ValueError(
            "Individual cross-fit overview layout requires exactly 2mm followed by 4mm"
        )
    return list(DEFAULT_SMOOTHINGS)


def _load_labels(path: Path) -> np.ndarray:
    image = nib.load(str(path))
    if not image.darrays:
        raise ValueError(f"Cross-fit label GIFTI has no data arrays: {path}")
    labels = np.asarray(image.darrays[0].data)
    if labels.ndim != 1 or labels.dtype != np.int32:
        raise ValueError(
            f"Cross-fit labels must be native int32 1D: {path}; "
            f"shape={labels.shape} dtype={labels.dtype}"
        )
    invalid = labels[(labels < 0) | (labels > 3)]
    if invalid.size:
        raise ValueError(f"Cross-fit labels must be in [0, 3]: {path}")
    return labels


def _canonical_json_sha256(payload: Mapping[str, object]) -> str:
    serialized = json.dumps(
        dict(payload),
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(serialized).hexdigest()


def _visual_label_contract_fingerprint(contract: Mapping[str, object]) -> str:
    payload = {
        str(key): value
        for key, value in dict(contract).items()
        if str(key) != "contract_fingerprint"
    }
    return _canonical_json_sha256(payload)


def _as_rgba(raw: object, *, context: str) -> tuple[int, int, int, int]:
    if not isinstance(raw, (list, tuple)) or len(raw) != 4:
        raise ValueError(f"{context} must be an RGBA sequence of length 4")
    rgba = tuple(int(value) for value in raw)
    if any(value < 0 or value > 255 for value in rgba):
        raise ValueError(f"{context} contains an out-of-range channel: {rgba}")
    return rgba  # type: ignore[return-value]


def _network_columns_from_registry(
    roi_registry: Sequence[Mapping[str, object]],
) -> dict[str, list[int]]:
    if len(roi_registry) != 400:
        raise ValueError(
            f"Expected a 400-ROI Kong2022 registry, got {len(roi_registry)}"
        )
    columns: dict[str, list[int]] = {}
    for index, row in enumerate(roi_registry):
        network = str(row.get("canonical7", "")).strip()
        if not network:
            raise ValueError(f"Kong2022 registry row {index} has no canonical7 network")
        columns.setdefault(network, []).append(int(index))
    if len(columns) < 3:
        raise ValueError("Kong2022 registry has fewer than three canonical networks")
    return columns


def _single_fisherz_winner(
    scores: Mapping[str, float],
    *,
    context: str,
) -> str:
    finite = {
        str(network): float(score)
        for network, score in scores.items()
        if np.isfinite(float(score))
    }
    if not finite:
        raise ValueError(f"No finite Fisher-z network means for {context}")
    maximum = max(finite.values())
    winners = sorted(
        network
        for network, score in finite.items()
        if np.isclose(score, maximum, rtol=0.0, atol=1e-12)
    )
    if len(winners) != 1:
        raise ValueError(
            f"Tied Fisher-z network winners for {context}: {winners}; scores={finite}"
        )
    return winners[0]


def summarize_fisherz_network_profiles(
    *,
    profile_records: Sequence[tuple[Mapping[str, object], np.ndarray]],
    roi_registry: Sequence[Mapping[str, object]],
    network_colors: Mapping[str, Sequence[int]],
) -> dict[str, object]:
    if not profile_records:
        raise ValueError("At least one Fisher-z template profile is required")
    columns_by_network = _network_columns_from_registry(roi_registry)
    network_names = sorted(columns_by_network)
    template_evidence: list[dict[str, object]] = []
    scores_by_label: dict[str, list[dict[str, float]]] = {
        str(label): [] for label in range(1, 4)
    }
    winners_by_label: dict[str, list[str]] = {str(label): [] for label in range(1, 4)}

    for record_index, (raw_evidence, raw_profile) in enumerate(profile_records):
        evidence = {str(key): value for key, value in dict(raw_evidence).items()}
        profile = np.asarray(raw_profile, dtype=np.float64)
        if profile.ndim != 2 or profile.shape != (3, len(roi_registry)):
            raise ValueError(
                "Template Fisher-z profile must have shape "
                f"(3, {len(roi_registry)}), got {profile.shape} for record {record_index}"
            )
        means_by_label: dict[str, dict[str, float]] = {}
        top_by_label: dict[str, str] = {}
        for label_index, label in enumerate(("1", "2", "3")):
            scores: dict[str, float] = {}
            for network in network_names:
                values = profile[label_index, columns_by_network[network]]
                finite_values = values[np.isfinite(values)]
                if finite_values.size == 0:
                    raise ValueError(
                        "Template Fisher-z profile has no finite values for "
                        f"label {label}, network {network}, record {record_index}"
                    )
                scores[network] = float(np.mean(finite_values))
            winner = _single_fisherz_winner(
                scores,
                context=f"template record {record_index}, label {label}",
            )
            means_by_label[label] = scores
            top_by_label[label] = winner
            scores_by_label[label].append(scores)
            winners_by_label[label].append(winner)
        template_evidence.append(
            {
                **evidence,
                "mean_fisherz_by_label": means_by_label,
                "top_network_by_label": top_by_label,
            }
        )

    label_mapping: dict[str, dict[str, object]] = {}
    aggregate_means: dict[str, dict[str, float]] = {}
    for label in ("1", "2", "3"):
        stable_winners = sorted(set(winners_by_label[label]))
        if len(stable_winners) != 1:
            raise ValueError(
                "Fisher-z network identity is not stable across templates for "
                f"label {label}: {stable_winners}"
            )
        canonical_network = stable_winners[0]
        display_spec = FISHERZ_NETWORK_DISPLAY_SPECS.get(canonical_network)
        if display_spec is None:
            raise ValueError(
                "Fisher-z winner has no approved display alias/color mapping for "
                f"label {label}: {canonical_network}"
            )
        style_name = str(display_spec["style_name"])
        if style_name not in network_colors:
            raise KeyError(
                f"Missing project network color for {style_name} while displaying label {label}"
            )
        rgba = _as_rgba(
            network_colors[style_name], context=f"Network color {style_name}"
        )
        label_mapping[label] = {
            "source_label": LABEL_SEMANTICS[label],
            "canonical_network": canonical_network,
            "display_name": str(display_spec["display_name"]),
            "style_network": style_name,
            "rgba": list(rgba),
        }
        aggregate_means[label] = {
            network: float(
                np.mean([scores[network] for scores in scores_by_label[label]])
            )
            for network in network_names
        }

    if len({str(row["canonical_network"]) for row in label_mapping.values()}) != 3:
        raise ValueError(
            "Fisher-z display mapping must resolve three distinct networks"
        )
    return {
        "schema_version": VISUAL_LABEL_CONTRACT_SCHEMA_VERSION,
        "method": "mean_fisherz_by_canonical7_kong2022_roi",
        "source_label_semantics": dict(LABEL_SEMANTICS),
        "template_count": len(template_evidence),
        "label_mapping": label_mapping,
        "aggregate_mean_fisherz_by_label": aggregate_means,
        "template_evidence": template_evidence,
    }


def derive_fisherz_network_label_contract(
    *,
    crossfit_root: Path,
    smoothings: Sequence[str] = DEFAULT_SMOOTHINGS,
) -> dict[str, object]:
    root = Path(crossfit_root).resolve()
    run_manifest = _run_manifest(root)
    selected_smoothings = _require_canonical_smoothings(smoothings)
    try:
        n_folds = int(run_manifest["n_folds"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("Cross-fit run manifest has no valid n_folds value") from exc
    if n_folds < 1:
        raise ValueError(f"Cross-fit n_folds must be positive, got {n_folds}")

    bundle = load_kong2022_bundle()
    raw_registry = bundle.get("roi_registry")
    if not isinstance(raw_registry, list):
        raise ValueError("Kong2022 bundle has no ROI registry list")
    roi_registry = [dict(row) for row in raw_registry]
    _network_columns_from_registry(roi_registry)
    profile_records: list[tuple[Mapping[str, object], np.ndarray]] = []
    for fold_number in range(1, n_folds + 1):
        for smoothing in selected_smoothings:
            for hemi in DEFAULT_HEMIS:
                template_dir = (
                    root
                    / "templates"
                    / f"fold-{fold_number:02d}"
                    / str(smoothing)
                    / f"hemi_{hemi}"
                )
                manifest_path = template_dir / "template_manifest.json"
                if not manifest_path.exists():
                    raise FileNotFoundError(
                        f"Missing cross-fit template manifest: {manifest_path}"
                    )
                manifest = _load_json(manifest_path)
                expected = {
                    "status": "ok",
                    "fold_index": fold_number - 1,
                    "smoothing": str(smoothing),
                    "hemi": str(hemi),
                    "cluster_label_semantics": LABEL_SEMANTICS,
                }
                mismatches = [
                    key for key, value in expected.items() if manifest.get(key) != value
                ]
                if mismatches:
                    raise ValueError(
                        "Cross-fit template manifest mismatch for "
                        f"fold-{fold_number:02d} {smoothing} hemi-{hemi}: {mismatches}"
                    )
                profile_path = template_dir / "template_profiles.npy"
                if not profile_path.exists():
                    raise FileNotFoundError(
                        f"Missing template Fisher-z profiles: {profile_path}"
                    )
                artifact_fingerprints = manifest.get("artifact_fingerprints")
                if not isinstance(artifact_fingerprints, Mapping):
                    raise ValueError(
                        f"Template artifact fingerprints are missing: {manifest_path}"
                    )
                _validate_fingerprint(
                    artifact_fingerprints.get("template_profiles.npy"),
                    profile_path,
                    f"template Fisher-z profiles {template_dir}",
                )
                profile = np.load(profile_path, allow_pickle=False)
                if profile.dtype != np.float32 or profile.shape != (
                    3,
                    len(roi_registry),
                ):
                    raise ValueError(
                        "Template Fisher-z profile type/shape mismatch for "
                        f"{profile_path}: shape={profile.shape} dtype={profile.dtype}"
                    )
                profile_records.append(
                    (
                        {
                            "fold": fold_number,
                            "smoothing": str(smoothing),
                            "hemi": str(hemi),
                            "template_manifest": str(manifest_path.resolve()),
                            "template_manifest_fingerprint": _fingerprint(
                                manifest_path
                            ),
                            "template_profiles": str(profile_path.resolve()),
                            "template_profiles_fingerprint": _fingerprint(profile_path),
                        },
                        profile,
                    )
                )

    network_colors = require_network_colors(
        ["VentralAttention", "Somatomotor", "Default"]
    )
    contract = summarize_fisherz_network_profiles(
        profile_records=profile_records,
        roi_registry=roi_registry,
        network_colors=network_colors,
    )
    contract["crossfit_root"] = str(root)
    contract["contract_fingerprint"] = _visual_label_contract_fingerprint(contract)
    return contract


def _display_label_items(
    visual_label_contract: Mapping[str, object],
) -> list[tuple[int, str, tuple[int, int, int, int]]]:
    raw_mapping = visual_label_contract.get("label_mapping")
    if not isinstance(raw_mapping, Mapping):
        raise ValueError("Visual-label contract has no label_mapping object")
    items = [DISPLAY_NULL_LABEL]
    for code in ("1", "2", "3"):
        raw_row = raw_mapping.get(code)
        if not isinstance(raw_row, Mapping):
            raise ValueError(f"Visual-label contract has no mapping for code {code}")
        canonical_network = str(raw_row.get("canonical_network", ""))
        display_spec = FISHERZ_NETWORK_DISPLAY_SPECS.get(canonical_network)
        if display_spec is None:
            raise ValueError(
                f"Visual-label contract has an unsupported canonical network for {code}: "
                f"{canonical_network}"
            )
        display_name = str(raw_row.get("display_name", ""))
        if display_name != str(display_spec["display_name"]):
            raise ValueError(
                f"Visual-label contract display alias mismatch for {code}: {display_name}"
            )
        rgba = _as_rgba(raw_row.get("rgba"), context=f"Visual-label color {code}")
        items.append((int(code), display_name, rgba))
    return items


def _display_label_semantics(
    visual_label_contract: Mapping[str, object],
) -> tuple[dict[str, str], dict[str, list[int]]]:
    items = _display_label_items(visual_label_contract)
    names = {str(key): str(name) for key, name, _rgba in items if key != 0}
    colors = {str(key): list(rgba) for key, _name, rgba in items if key != 0}
    return names, colors


def _validate_visual_label_contract(
    visual_label_contract: Mapping[str, object],
) -> dict[str, object]:
    contract = {str(key): value for key, value in dict(visual_label_contract).items()}
    recorded = contract.get("contract_fingerprint")
    if not isinstance(recorded, str) or not recorded:
        raise ValueError("Visual-label contract has no fingerprint")
    expected = _visual_label_contract_fingerprint(contract)
    if recorded != expected:
        raise ValueError("Visual-label contract fingerprint mismatch")
    _display_label_items(contract)
    return contract


def write_network_display_label_gifti(
    *,
    source_label_path: Path,
    output_path: Path,
    visual_label_contract: Mapping[str, object],
) -> Path:
    labels = _load_labels(source_label_path)
    label_table = GiftiLabelTable()
    for key, name, rgba in _display_label_items(visual_label_contract):
        label = GiftiLabel(
            key=int(key),
            red=float(rgba[0]) / 255.0,
            green=float(rgba[1]) / 255.0,
            blue=float(rgba[2]) / 255.0,
            alpha=float(rgba[3]) / 255.0,
        )
        label.label = str(name)
        label_table.labels.append(label)
    image = GiftiImage(
        darrays=[
            GiftiDataArray(
                data=np.ascontiguousarray(labels, dtype=np.int32),
                intent="NIFTI_INTENT_LABEL",
            )
        ],
        labeltable=label_table,
    )
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    nib.save(image, str(destination))
    rendered_labels = _load_labels(destination)
    if not np.array_equal(rendered_labels, labels):
        raise ValueError(
            f"Display GIFTI label codes changed while writing {destination}"
        )
    return destination


def load_subject_combo(
    *,
    crossfit_root: Path,
    subject: str,
    smoothing: str,
    hemi: str,
) -> dict[str, object]:
    """Load a held-out label map and validate its recorded input/output evidence."""
    subject_id = _normalize_subject(subject)
    combo_dir = _combo_dir(crossfit_root, subject_id, smoothing, hemi)
    manifest_path = combo_dir / "combo_manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Missing cross-fit combo manifest: {manifest_path}")
    manifest = _load_json(manifest_path)
    expected = {
        "status": "ok",
        "subject": f"sub-{subject_id}",
        "smoothing": str(smoothing),
        "hemi": str(hemi),
        "subject_was_in_template": False,
    }
    mismatches = [key for key, value in expected.items() if manifest.get(key) != value]
    if manifest.get("label_semantics") != LABEL_SEMANTICS:
        mismatches.append("label_semantics")
    if mismatches:
        raise ValueError(
            f"Cross-fit combo manifest mismatch for sub-{subject_id} "
            f"{smoothing} hemi-{hemi}: {', '.join(mismatches)}"
        )

    outputs = manifest.get("outputs")
    fingerprints = manifest.get("output_fingerprints")
    density = manifest.get("density_evidence")
    if not isinstance(outputs, dict) or not isinstance(fingerprints, dict):
        raise ValueError(
            f"Cross-fit combo output provenance is missing: {manifest_path}"
        )
    if not isinstance(density, dict):
        raise ValueError(
            f"Cross-fit combo density evidence is missing: {manifest_path}"
        )

    label_raw = outputs.get("label_gifti")
    if label_raw is None:
        raise ValueError(f"Cross-fit combo label GIFTI is missing: {manifest_path}")
    label_path = Path(str(label_raw)).resolve()
    if not label_path.exists():
        raise FileNotFoundError(f"Missing cross-fit label GIFTI: {label_path}")
    try:
        label_path.relative_to(combo_dir.resolve())
    except ValueError as exc:
        raise ValueError(
            f"Cross-fit label GIFTI must be inside its combo directory: {label_path}"
        ) from exc
    _validate_fingerprint(
        fingerprints.get("label_gifti"),
        label_path,
        f"sub-{subject_id} {smoothing} hemi-{hemi} label GIFTI",
    )

    surface_raw = density.get("consumed_subject_surface")
    if surface_raw is None:
        raise ValueError(f"Cross-fit subject surface is missing: {manifest_path}")
    surface_path = Path(str(surface_raw)).resolve()
    if not surface_path.exists():
        raise FileNotFoundError(f"Missing consumed subject surface: {surface_path}")
    if (
        "_den-512_" not in surface_path.name
        or str(density.get("filename_density")) != "512"
    ):
        raise ValueError(f"Cross-fit density evidence is not den-512: {manifest_path}")
    expected_surface_name = (
        f"sub-{subject_id}_hemi-{hemi}_space-corobl_den-512_"
        "label-hipp_midthickness.surf.gii"
    )
    if surface_path.name != expected_surface_name:
        raise ValueError(
            "Cross-fit consumed surface identity mismatch for "
            f"sub-{subject_id} hemi-{hemi}: expected {expected_surface_name}, "
            f"got {surface_path.name}"
        )

    inputs = manifest.get("inputs")
    input_fingerprints = manifest.get("input_fingerprints")
    if not isinstance(inputs, dict) or not isinstance(input_fingerprints, dict):
        raise ValueError(
            f"Cross-fit subject-surface provenance is missing: {manifest_path}"
        )
    input_surface_raw = inputs.get("subject_surface")
    if input_surface_raw is None:
        raise ValueError(f"Cross-fit input subject surface is missing: {manifest_path}")
    input_surface_path = Path(str(input_surface_raw)).resolve()
    if input_surface_path != surface_path:
        raise ValueError(
            "Cross-fit consumed/input subject surface paths disagree for "
            f"sub-{subject_id} {smoothing} hemi-{hemi}"
        )
    _validate_fingerprint(
        input_fingerprints.get("subject_surface"),
        surface_path,
        f"sub-{subject_id} {smoothing} hemi-{hemi} subject surface",
    )

    labels = _load_labels(label_path)
    n_vertices = read_surface_vertex_count(surface_path)
    if labels.size != n_vertices:
        raise ValueError(
            f"Cross-fit label/surface vertex mismatch for sub-{subject_id} "
            f"{smoothing} hemi-{hemi}: labels={labels.size} surface={n_vertices}"
        )
    spec_path = infer_spec_path(subject_id, surface_path).resolve()
    if not spec_path.exists():
        raise FileNotFoundError(f"Missing subject surface spec: {spec_path}")
    return {
        "subject": f"sub-{subject_id}",
        "smoothing": str(smoothing),
        "hemi": str(hemi),
        "combo_manifest": str(manifest_path.resolve()),
        "label_gifti": str(label_path),
        "subject_surface": str(surface_path),
        "spec_path": str(spec_path),
        "label_fingerprint": dict(fingerprints["label_gifti"]),
        "label_counts": {str(key): int(np.sum(labels == key)) for key in range(4)},
        "density_evidence": dict(density),
    }


def _run_manifest(root: Path) -> dict[str, Any]:
    path = Path(root) / "crossfit_manifest.json"
    if not path.exists():
        raise FileNotFoundError(f"Missing cross-fit run manifest: {path}")
    manifest = _load_json(path)
    if manifest.get("status") != "ok":
        raise ValueError(f"Cross-fit run is not complete: {path}")
    return manifest


def _subjects(
    manifest: Mapping[str, Any], requested: Sequence[str] | None
) -> list[str]:
    config = manifest.get("run_config")
    source = config.get("subjects") if isinstance(config, dict) else None
    if not isinstance(source, list) or not source:
        raise ValueError("Cross-fit run manifest has no run_config.subjects list")
    available = [_normalize_subject(str(value)) for value in source]
    duplicate_available = _duplicates(available)
    if duplicate_available:
        raise ValueError(
            "Cross-fit run manifest has duplicate subjects: "
            + ", ".join(f"sub-{value}" for value in duplicate_available)
        )
    if requested is None:
        return available
    selected = [_normalize_subject(value) for value in requested]
    duplicate_selected = _duplicates(selected)
    if duplicate_selected:
        raise ValueError(
            "Requested subjects contain duplicates: "
            + ", ".join(f"sub-{value}" for value in duplicate_selected)
        )
    unknown = sorted(set(selected) - set(available))
    if unknown:
        raise ValueError(
            "Requested subjects are absent from the cross-fit run: "
            + ", ".join(f"sub-{value}" for value in unknown)
        )
    return selected


def _smoothings(manifest: Mapping[str, Any], requested: Sequence[str]) -> list[str]:
    selected = _require_canonical_smoothings(requested)
    available = manifest.get("smoothings")
    if not isinstance(available, list) or any(
        value not in available for value in selected
    ):
        raise ValueError("Requested smoothing is absent from the cross-fit run")
    if manifest.get("hemis") != list(DEFAULT_HEMIS):
        raise ValueError("Cross-fit run must have the canonical L/R hemisphere pair")
    return selected


def _fit_width(image: Image.Image, width: int) -> Image.Image:
    if image.width == width:
        return image
    height = max(1, round(image.height * width / image.width))
    return image.resize((width, height), Image.Resampling.LANCZOS)


def _validated_png_fingerprint(path: Path) -> dict[str, object]:
    output_path = Path(path)
    if not output_path.is_file() or output_path.stat().st_size <= 0:
        raise ValueError(
            f"Individual cross-fit overview PNG is missing or empty: {output_path}"
        )
    try:
        with Image.open(output_path) as image:
            if image.format != "PNG":
                raise ValueError(f"Expected PNG, found {image.format!r}")
            image.verify()
        with Image.open(output_path) as image:
            image.load()
            if image.width <= 0 or image.height <= 0:
                raise ValueError("PNG has non-positive dimensions")
    except (OSError, SyntaxError, ValueError) as exc:
        raise ValueError(
            f"Individual cross-fit overview PNG is not valid: {output_path}"
        ) from exc
    return _fingerprint(output_path)


def _subject_overview_title_lines(
    subject: str, *, max_width: int, title: str | None = None
) -> list[str]:
    if title is None:
        title = (
            "HCP RouteA cross-fit Fisher-z network matching | "
            f"sub-{_normalize_subject(subject)}"
        )
    draw = ImageDraw.Draw(Image.new("RGB", (8, 8)))
    return wrap_text(draw, title, load_font(38), max_width)


def compose_stacked_rows(
    *,
    subject: str,
    rows: Sequence[tuple[str, Path]],
    output_path: Path,
    title: str | None = None,
) -> None:
    """Stack standard locked-Workbench rows beneath one subject title."""
    if not rows:
        raise ValueError("At least one rendered smoothing row is required")
    images = [Image.open(path).convert("RGB") for _smoothing, path in rows]
    width = max(image.width for image in images)
    images = [_fit_width(image, width) for image in images]
    pad, minimum_title_h, row_h, gap = 34, 88, 46, 34
    title_font = load_font(38)
    title_line_h = 46
    title_lines = _subject_overview_title_lines(subject, max_width=width, title=title)
    title_h = max(minimum_title_h, 16 + title_line_h * len(title_lines))
    height = (
        pad * 2
        + title_h
        + sum(row_h + image.height for image in images)
        + gap * (len(images) - 1)
    )
    canvas = Image.new("RGB", (width + pad * 2, height), (0, 0, 0))
    draw = ImageDraw.Draw(canvas)
    for line_index, line in enumerate(title_lines):
        draw.text(
            (pad, pad + 8 + line_index * title_line_h),
            line,
            fill=(255, 255, 255),
            font=title_font,
        )
    y = pad + title_h
    for index, ((smoothing, _path), image) in enumerate(zip(rows, images, strict=True)):
        draw.text(
            (pad, y), f"{smoothing} smoothing", fill=(230, 230, 230), font=load_font(30)
        )
        y += row_h
        canvas.paste(image, (pad + (width - image.width) // 2, y))
        y += image.height + (gap if index + 1 < len(images) else 0)
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(
        f".{destination.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp"
    )
    canvas.save(temporary, format="PNG", dpi=(300, 300))
    _validated_png_fingerprint(temporary)
    os.replace(temporary, destination)


def _source_rows(
    *,
    smoothings: Sequence[str],
    combos: Mapping[tuple[str, str], Mapping[str, object]],
) -> list[dict[str, object]]:
    return [
        {
            "smoothing": smoothing,
            "left": dict(combos[(smoothing, "L")]),
            "right": dict(combos[(smoothing, "R")]),
            "label_gifti": str(combos[(smoothing, "L")]["label_gifti"]),
        }
        for smoothing in smoothings
    ]


def _row(
    *,
    subject: str,
    output_path: Path,
    resumed: bool,
    elapsed_seconds: float,
    smoothings: Sequence[str],
    combos: Mapping[tuple[str, str], Mapping[str, object]],
    output_fingerprint: Mapping[str, object],
    scene_path: Path,
    scene_fingerprint: Mapping[str, object],
    visual_label_contract: Mapping[str, object],
) -> dict[str, object]:
    display_label_semantics, display_label_colors = _display_label_semantics(
        visual_label_contract
    )
    return {
        "status": "ok",
        "subject": f"sub-{_normalize_subject(subject)}",
        "output": str(output_path.resolve()),
        "output_fingerprint": dict(output_fingerprint),
        "scene": str(Path(scene_path).resolve()),
        "scene_fingerprint": dict(scene_fingerprint),
        "visual_label_contract_fingerprint": str(
            visual_label_contract["contract_fingerprint"]
        ),
        "display_label_semantics": display_label_semantics,
        "display_label_colors": display_label_colors,
        "resumed": bool(resumed),
        "elapsed_seconds": float(elapsed_seconds),
        "rows": _source_rows(smoothings=smoothings, combos=combos),
    }


def _load_existing_overview_items(output_dir: Path) -> dict[str, dict[str, object]]:
    manifest_path = Path(output_dir) / "individual_overview_manifest.json"
    if not manifest_path.exists():
        return {}
    payload = _load_json(manifest_path)
    items = payload.get("items")
    if not isinstance(items, list):
        raise ValueError(
            f"Existing individual overview manifest has no items list: {manifest_path}"
        )

    records: dict[str, dict[str, object]] = {}
    for index, item in enumerate(items):
        if not isinstance(item, dict):
            raise ValueError(
                f"Existing individual overview manifest item {index} is not an object: "
                f"{manifest_path}"
            )
        try:
            subject = _normalize_subject(str(item.get("subject", "")))
        except ValueError as exc:
            raise ValueError(
                f"Existing individual overview manifest item {index} has an invalid subject: "
                f"{manifest_path}"
            ) from exc
        if subject in records:
            raise ValueError(
                f"Existing individual overview manifest has duplicate subject sub-{subject}: "
                f"{manifest_path}"
            )
        records[subject] = dict(item)
    return records


def _validate_resume_reuse(
    *,
    subject: str,
    output_path: Path,
    scene_path: Path,
    source_rows: Sequence[Mapping[str, object]],
    previous_item: Mapping[str, object] | None,
    visual_label_contract: Mapping[str, object],
    allow_visual_contract_mismatch: bool = False,
) -> dict[str, object]:
    output_fingerprint = _validated_png_fingerprint(output_path)
    if previous_item is None:
        raise ValueError(
            f"Existing individual overview has no provenance record for sub-{subject}; "
            "refuse --resume reuse"
        )
    if previous_item.get("status") != "ok":
        raise ValueError(
            f"Existing individual overview provenance is not ok for sub-{subject}"
        )
    try:
        previous_subject = _normalize_subject(str(previous_item.get("subject", "")))
    except ValueError as exc:
        raise ValueError(
            f"Existing individual overview provenance has an invalid subject for sub-{subject}"
        ) from exc
    if previous_subject != subject:
        raise ValueError(
            f"Existing individual overview provenance subject mismatch for sub-{subject}"
        )
    expected_scene = str(Path(scene_path).resolve())
    if previous_item.get("scene") != expected_scene:
        raise ValueError(
            f"Existing individual overview scene mismatch for sub-{subject}"
        )
    _validate_fingerprint(
        previous_item.get("scene_fingerprint"),
        scene_path,
        f"Existing individual overview Workbench scene for sub-{subject}",
    )
    if previous_item.get("rows") != [dict(row) for row in source_rows]:
        raise ValueError(
            f"Existing individual overview source provenance changed for sub-{subject}"
        )

    recorded_fingerprint = previous_item.get("output_fingerprint")
    if recorded_fingerprint is None:
        raise ValueError(
            f"Existing individual overview has no output fingerprint for sub-{subject}; "
            "refuse --resume reuse"
        )
    _validate_fingerprint(
        recorded_fingerprint,
        output_path,
        f"Existing individual overview PNG for sub-{subject}",
    )
    expected_visual_contract = str(visual_label_contract["contract_fingerprint"])
    if (
        previous_item.get("visual_label_contract_fingerprint")
        != expected_visual_contract
    ):
        if not allow_visual_contract_mismatch:
            raise ValueError(
                "Existing individual overview visual-label contract differs for "
                f"sub-{subject}; use --resume --refresh-visuals to replace it"
            )
    return output_fingerprint


def render_subject_overview(
    *,
    crossfit_root: Path,
    output_dir: Path,
    scene: Path,
    subject: str,
    smoothings: Sequence[str] = DEFAULT_SMOOTHINGS,
    resume: bool = False,
    previous_item: Mapping[str, object] | None = None,
    visual_label_contract: Mapping[str, object] | None = None,
    refresh_visuals: bool = False,
) -> dict[str, object]:
    # Validate all immutable source provenance before rendering any display labels.
    started = time.time()
    subject_id = _normalize_subject(subject)
    selected = _require_canonical_smoothings(smoothings)
    if refresh_visuals and not resume:
        raise ValueError("--refresh-visuals requires --resume")
    scene_path = Path(scene).resolve()
    if not scene_path.exists():
        raise FileNotFoundError(f"Missing locked Workbench scene: {scene_path}")
    scene_fingerprint = _fingerprint(scene_path)
    combos: dict[tuple[str, str], dict[str, object]] = {}
    for smoothing in selected:
        for hemi in DEFAULT_HEMIS:
            combos[(smoothing, hemi)] = load_subject_combo(
                crossfit_root=Path(crossfit_root),
                subject=subject_id,
                smoothing=smoothing,
                hemi=hemi,
            )
        if Path(str(combos[(smoothing, "L")]["spec_path"])) != Path(
            str(combos[(smoothing, "R")]["spec_path"])
        ):
            raise ValueError(
                f"L/R surface specs disagree for sub-{subject_id} {smoothing}"
            )

    source_rows = _source_rows(smoothings=selected, combos=combos)
    output_path = subject_overview_png_path(output_dir, subject_id)
    if output_path.exists() and not resume:
        raise FileExistsError(
            f"Refusing to overwrite existing individual overview: {output_path}; use --resume"
        )
    if visual_label_contract is None:
        contract = derive_fisherz_network_label_contract(
            crossfit_root=Path(crossfit_root),
            smoothings=selected,
        )
    else:
        contract = _validate_visual_label_contract(visual_label_contract)

    if output_path.exists():
        prior_item = previous_item
        if prior_item is None:
            prior_item = _load_existing_overview_items(output_dir).get(subject_id)
        output_fingerprint = _validate_resume_reuse(
            subject=subject_id,
            output_path=output_path,
            scene_path=scene_path,
            source_rows=source_rows,
            previous_item=prior_item,
            visual_label_contract=contract,
            allow_visual_contract_mismatch=refresh_visuals,
        )
        if not refresh_visuals:
            return _row(
                subject=subject_id,
                output_path=output_path,
                resumed=True,
                elapsed_seconds=time.time() - started,
                smoothings=selected,
                combos=combos,
                output_fingerprint=output_fingerprint,
                scene_path=scene_path,
                scene_fingerprint=scene_fingerprint,
                visual_label_contract=contract,
            )

    rendered_rows: list[tuple[str, Path]] = []
    with tempfile.TemporaryDirectory(prefix="crossfit_individual_overview_") as scratch:
        scratch_root = Path(scratch)
        for smoothing in selected:
            left, right = combos[(smoothing, "L")], combos[(smoothing, "R")]
            display_root = scratch_root / "display_labels" / smoothing
            left_display = write_network_display_label_gifti(
                source_label_path=Path(str(left["label_gifti"])),
                output_path=display_root / "hemi-L.label.gii",
                visual_label_contract=contract,
            )
            right_display = write_network_display_label_gifti(
                source_label_path=Path(str(right["label_gifti"])),
                output_path=display_root / "hemi-R.label.gii",
                visual_label_contract=contract,
            )
            rendered = render_locked_grid(
                subject=subject_id,
                scene=scene_path,
                outdir=scratch_root / smoothing,
                name=f"crossfit_template_matching_k3_{smoothing}",
                title=(
                    f"Cross-fit Fisher-z network matching | sub-{subject_id} | {smoothing}"
                ),
                left_labels=left_display,
                right_labels=right_display,
                left_surface=Path(str(left["subject_surface"])),
                right_surface=Path(str(right["subject_surface"])),
                spec_path=Path(str(left["spec_path"])),
            )
            image_path = Path(str(rendered["biglegend_png"]))
            if not image_path.exists():
                raise FileNotFoundError(
                    f"Locked surface renderer did not write {image_path}"
                )
            rendered_rows.append((smoothing, image_path))
        compose_stacked_rows(
            subject=subject_id, rows=rendered_rows, output_path=output_path
        )
    output_fingerprint = _validated_png_fingerprint(output_path)
    return _row(
        subject=subject_id,
        output_path=output_path,
        resumed=False,
        elapsed_seconds=time.time() - started,
        smoothings=selected,
        combos=combos,
        output_fingerprint=output_fingerprint,
        scene_path=scene_path,
        scene_fingerprint=scene_fingerprint,
        visual_label_contract=contract,
    )


def _worker(payload: Mapping[str, object]) -> dict[str, object]:
    for key, value in THREAD_ENV_OVERRIDES.items():
        os.environ[key] = value
    subject = _normalize_subject(str(payload["subject"]))
    started = time.time()
    try:
        previous_item = payload.get("previous_item")
        if previous_item is not None and not isinstance(previous_item, Mapping):
            raise ValueError("Worker previous_item must be a mapping")
        visual_label_contract = payload.get("visual_label_contract")
        if not isinstance(visual_label_contract, Mapping):
            raise ValueError("Worker visual_label_contract must be a mapping")
        return render_subject_overview(
            crossfit_root=Path(str(payload["crossfit_root"])),
            output_dir=Path(str(payload["output_dir"])),
            scene=Path(str(payload["scene"])),
            subject=subject,
            smoothings=[str(value) for value in payload["smoothings"]],
            resume=bool(payload["resume"]),
            previous_item=previous_item,
            visual_label_contract=visual_label_contract,
            refresh_visuals=bool(payload.get("refresh_visuals", False)),
        )
    except Exception as exc:
        return {
            "status": "failed",
            "subject": f"sub-{subject}",
            "output": "",
            "resumed": False,
            "elapsed_seconds": time.time() - started,
            "error_type": exc.__class__.__name__,
            "error": str(exc),
            "rows": [],
        }


def _run_tasks(
    tasks: Sequence[Mapping[str, object]], jobs: int
) -> list[dict[str, object]]:
    if not tasks:
        return []
    if jobs == 1:
        return [_worker(task) for task in tasks]
    with ProcessPoolExecutor(max_workers=jobs) as executor:
        return [
            future.result()
            for future in as_completed(
                [executor.submit(_worker, task) for task in tasks]
            )
        ]


def _write_status(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with Path(path).open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(
            stream, fieldnames=STATUS_FIELDS, delimiter="\t", lineterminator="\n"
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    field: str(row.get(field, "")).replace("\t", " ").replace("\n", " ")
                    for field in STATUS_FIELDS
                }
            )


def render_individual_overviews(
    *,
    crossfit_root: Path,
    output_dir: Path,
    scene: Path,
    subjects: Sequence[str] | None = None,
    smoothings: Sequence[str] = DEFAULT_SMOOTHINGS,
    jobs: int = DEFAULT_JOBS,
    resume: bool = False,
    refresh_visuals: bool = False,
) -> dict[str, Path]:
    # Render canonical per-subject PNGs and aggregate their immutable/display provenance.
    started = time.time()
    root = Path(crossfit_root).resolve()
    destination = Path(output_dir).resolve()
    scene_path = Path(scene).resolve()
    if refresh_visuals and not resume:
        raise ValueError("--refresh-visuals requires --resume")
    if not scene_path.exists():
        raise FileNotFoundError(f"Missing locked Workbench scene: {scene_path}")
    scene_fingerprint = _fingerprint(scene_path)
    manifest = _run_manifest(root)
    selected_subjects = _subjects(manifest, subjects)
    selected_smoothings = _smoothings(manifest, smoothings)
    visual_label_contract = _validate_visual_label_contract(
        derive_fisherz_network_label_contract(
            crossfit_root=root,
            smoothings=selected_smoothings,
        )
    )
    previous_items = _load_existing_overview_items(destination)
    tasks = [
        {
            "crossfit_root": str(root),
            "output_dir": str(destination),
            "scene": str(scene_path),
            "subject": subject,
            "smoothings": selected_smoothings,
            "resume": resume,
            "refresh_visuals": refresh_visuals,
            "visual_label_contract": visual_label_contract,
            "previous_item": previous_items.get(subject),
        }
        for subject in selected_subjects
    ]
    effective_jobs = max(1, min(int(jobs), 10, len(tasks) or 1))
    requested_rows = sorted(
        _run_tasks(tasks, effective_jobs),
        key=lambda row: str(row["subject"]),
    )
    merged_rows = dict(previous_items)
    for row in requested_rows:
        subject = _normalize_subject(str(row["subject"]))
        if row["status"] == "ok" or subject not in merged_rows:
            merged_rows[subject] = row
    rows = sorted(merged_rows.values(), key=lambda row: str(row["subject"]))

    status_path = destination / "individual_overview_status.tsv"
    mapping_path = destination / "network_label_mapping.json"
    manifest_path = destination / "individual_overview_manifest.json"
    _write_status(status_path, requested_rows)
    _write_json(
        mapping_path,
        {
            "workflow": WORKFLOW_NAME,
            "crossfit_root": str(root),
            "visual_label_contract": visual_label_contract,
            "written_at": _now(),
        },
    )
    mapping_fingerprint = _fingerprint(mapping_path)
    attempt_failed = [row for row in requested_rows if row["status"] != "ok"]
    aggregate_failed = [row for row in rows if row["status"] != "ok"]
    summary: dict[str, object] = {
        "status": "failed" if attempt_failed else "ok",
        "workflow": WORKFLOW_NAME,
        "crossfit_root": str(root),
        "output_dir": str(destination),
        "scene": str(scene_path),
        "scene_fingerprint": scene_fingerprint,
        "visual_label_contract_fingerprint": str(
            visual_label_contract["contract_fingerprint"]
        ),
        "network_label_mapping": str(mapping_path.resolve()),
        "network_label_mapping_fingerprint": mapping_fingerprint,
        "refresh_visuals": bool(refresh_visuals),
        "requested_subjects": len(selected_subjects),
        "subjects": len(rows),
        "smoothings": selected_smoothings,
        "hemis": list(DEFAULT_HEMIS),
        "jobs": effective_jobs,
        "total": len(requested_rows),
        "ok": len(requested_rows) - len(attempt_failed),
        "failed": len(attempt_failed),
        "aggregate_total": len(rows),
        "aggregate_ok": len(rows) - len(aggregate_failed),
        "aggregate_failed": len(aggregate_failed),
        "elapsed_seconds": time.time() - started,
        "updated_at": _now(),
        "status_tsv": str(status_path.resolve()),
    }
    _write_json(
        manifest_path,
        {
            "summary": summary,
            "items": rows,
            "attempt_items": requested_rows,
        },
    )
    return {
        "status_tsv": status_path.resolve(),
        "network_label_mapping": mapping_path.resolve(),
        "manifest": manifest_path.resolve(),
    }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Render one standard-Workbench 2mm/4mm cross-fit hippocampal PNG per subject."
    )
    parser.add_argument(
        "--crossfit-root", type=Path, default=resolve_crossfit_template_matching_root()
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=resolve_crossfit_template_matching_individual_overview_root(),
    )
    parser.add_argument("--scene", type=Path, default=DEFAULT_SCENE)
    parser.add_argument("--subjects", nargs="*", default=None)
    parser.add_argument(
        "--smoothings",
        nargs="+",
        default=list(DEFAULT_SMOOTHINGS),
        help="Fixed top-to-bottom layout; must be exactly: 2mm 4mm",
    )
    parser.add_argument("--jobs", type=int, default=DEFAULT_JOBS)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--refresh-visuals",
        action="store_true",
        help=(
            "With --resume, atomically replace existing PNGs whose visual-label contract "
            "is intentionally being updated"
        ),
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_arg_parser().parse_args(list(argv) if argv is not None else None)
    outputs = render_individual_overviews(
        crossfit_root=args.crossfit_root,
        output_dir=args.output_dir,
        scene=args.scene,
        subjects=args.subjects,
        smoothings=args.smoothings,
        jobs=args.jobs,
        resume=args.resume,
        refresh_visuals=args.refresh_visuals,
    )
    summary = _load_json(outputs["manifest"])["summary"]
    print(
        f"status={summary['status']} ok={summary['ok']} failed={summary['failed']} total={summary['total']}"
    )
    print(f"output_dir={summary['output_dir']}")
    print(f"status_tsv={outputs['status_tsv']}")
    print(f"manifest={outputs['manifest']}")
    return 0 if summary["status"] == "ok" else 1


if __name__ == "__main__":
    raise SystemExit(main())
