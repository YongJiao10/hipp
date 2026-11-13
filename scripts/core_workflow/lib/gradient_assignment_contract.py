"""Validate exact gradient assignment artifacts before a scientific render."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

import nibabel as nib
import numpy as np
from .network_ovr import ALL_OVR_VARIANTS, NETWORK_TS_OVR_VARIANT, OVR_VARIANTS, OVR_ASSIGNMENT_METRIC


ASSIGNMENT_VARIANTS = (
    "normalized_top20_count",
    "fisherz",
    "network_ts_fisherz",
    *ALL_OVR_VARIANTS,
)
EXPECTED_METRICS = {
    "normalized_top20_count": "top20_count_normalized_by_network_roi_count",
    "fisherz": "top1_fisher_z",
    "network_ts_fisherz": "top1_network_ts_fisher_z",
    **{variant: OVR_ASSIGNMENT_METRIC for variant in ALL_OVR_VARIANTS},
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _declared_path(
    row: Mapping[str, Any], field: str, run_dir: Path, variant: str
) -> Path:
    raw = row.get(field)
    if not isinstance(raw, str) or not raw.strip():
        raise ValueError(
            f"Gradient assignment contract ({variant}): missing {field}: {run_dir}"
        )
    path = Path(raw).expanduser()
    if not path.is_absolute():
        raise ValueError(
            f"Gradient assignment contract ({variant}): {field} must declare "
            f"an absolute path: {path}: {run_dir}"
        )
    path = path.resolve()
    try:
        path.relative_to(run_dir)
    except ValueError as exc:
        raise ValueError(
            f"Gradient assignment contract ({variant}): {field} points "
            f"outside its hemisphere run directory: {path}: {run_dir}"
        ) from exc
    if not path.is_file():
        raise FileNotFoundError(
            f"Gradient assignment {variant} missing {field}: {path}"
        )
    return path


def _read_json(path: Path, *, label: str, variant: str, run_dir: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(
            f"Gradient assignment contract ({variant}): invalid {label} "
            f"{path}: {run_dir}"
        ) from exc


def _load_mapping(path: Path, variant: str, run_dir: Path) -> dict[str, Any]:
    mapping = _read_json(path, label="label map", variant=variant, run_dir=run_dir)
    if not isinstance(mapping, dict):
        raise ValueError(
            f"Gradient assignment contract ({variant}): label map must be "
            f"an object: {path}: {run_dir}"
        )
    return mapping


def _load_integer_vector(
    path: Path, *, label: str, variant: str, run_dir: Path
) -> np.ndarray:
    try:
        raw = np.load(path, allow_pickle=False)
    except (OSError, ValueError) as exc:
        raise ValueError(
            f"Gradient assignment contract ({variant}): cannot load {label} "
            f"{path}: {run_dir}"
        ) from exc
    array = np.asarray(raw)
    if array.ndim != 1 or not np.issubdtype(array.dtype, np.integer):
        raise ValueError(
            f"Gradient assignment contract ({variant}): {label} must be a "
            f"one-dimensional integer array: {path}: {run_dir}"
        )
    return array.astype(np.int32, copy=False)


def _validate_label_table(
    gifti: nib.gifti.GiftiImage,
    *,
    names: list[str],
    keys: dict[str, int],
    unassigned_key: int | None,
    variant: str,
    run_dir: Path,
) -> None:
    labels = list(gifti.labeltable.labels) if gifti.labeltable is not None else []
    by_key: dict[int, Any] = {}
    by_name: dict[str, int] = {}
    for entry in labels:
        key = int(entry.key)
        name = str(entry.label)
        if key in by_key or name in by_name:
            raise ValueError(
                f"Gradient assignment contract ({variant}): duplicate GIFTI "
                f"label key/name: {run_dir}"
            )
        by_key[key] = entry
        by_name[name] = key

    expected_keys = {0, *[int(value) for value in keys.values()]}
    if unassigned_key is not None:
        expected_keys.add(int(unassigned_key))
    if set(by_key) != expected_keys:
        raise ValueError(
            f"Gradient assignment contract ({variant}): GIFTI label keys "
            f"{sorted(by_key)} disagree with expected {sorted(expected_keys)}: "
            f"{run_dir}"
        )

    from lib.surface_utils import network_color_map

    colors = network_color_map()
    expected_colors: dict[int, tuple[int, int, int, int]] = {
        0: (158, 158, 158, 255)
    }
    expected_names: dict[int, str] = {0: "Null"}
    if unassigned_key is not None:
        expected_colors[int(unassigned_key)] = (20, 20, 20, 255)
        expected_names[int(unassigned_key)] = "unassigned"
    for name, key in keys.items():
        expected_names[int(key)] = str(name)
        if str(name) not in colors:
            raise ValueError(
                f"Gradient assignment contract ({variant}): unknown network "
                f"{name!r}: {run_dir}"
            )
        expected_colors[int(key)] = colors[str(name)]

    for key, expected_name in expected_names.items():
        entry = by_key.get(int(key))
        if entry is None or str(entry.label) != expected_name:
            raise ValueError(
                f"Gradient assignment contract ({variant}): GIFTI label "
                f"table key {key} has "
                f"{None if entry is None else entry.label!r}, expected "
                f"{expected_name!r}: {run_dir}"
            )
        rgba = np.asarray(entry.rgba, dtype=float)
        expected_rgba = np.asarray(expected_colors[key], dtype=float) / 255.0
        if rgba.shape != (4,) or not np.allclose(
            rgba, expected_rgba, atol=1 / 255.0, rtol=0
        ):
            raise ValueError(
                f"Gradient assignment contract ({variant}): GIFTI color for "
                f"{expected_name!r} disagrees with canonical network colors: "
                f"{run_dir}"
            )


def _validate_status_items(
    status: Mapping[str, Any],
    *,
    clusters: np.ndarray,
    labels: np.ndarray,
    mapping: Mapping[str, Any],
    profile: np.ndarray | None,
    profile_names: list[str] | None,
    variant: str,
    run_dir: Path,
) -> tuple[dict[int, str], dict[int, dict[str, float]]]:
    items = status.get("items")
    if not isinstance(items, list) or any(
        not isinstance(item, dict) for item in items
    ):
        raise ValueError(
            f"Gradient assignment contract ({variant}): status items must be "
            f"objects: {run_dir}"
        )

    ids = [item.get("cluster_id") for item in items]
    if any(
        isinstance(value, bool)
        or not isinstance(value, (int, np.integer))
        for value in ids
    ):
        raise ValueError(
            f"Gradient assignment contract ({variant}): status cluster IDs "
            f"must be integers: {run_dir}"
        )
    actual_ids = sorted(
        int(value) for value in np.unique(clusters) if int(value) > 0
    )
    if (
        len(ids) != len(set(int(value) for value in ids))
        or set(int(value) for value in ids) != set(actual_ids)
    ):
        raise ValueError(
            f"Gradient assignment contract ({variant}): status cluster IDs "
            f"{ids} disagree with surface {actual_ids}: {run_dir}"
        )

    names = [str(value) for value in mapping["network_names"]]
    key_by_name = {
        str(name): int(value)
        for name, value in mapping["network_label_keys"].items()
    }
    cluster_map = {
        str(key): int(value)
        for key, value in mapping["cluster_to_functional_label_key"].items()
    }
    winners: dict[int, str] = {}
    compositions: dict[int, dict[str, float]] = {}

    for item in items:
        cid = int(item["cluster_id"])
        if variant == "normalized_top20_count":
            raw_values = item.get("normalized_count_by_network")
            if not isinstance(raw_values, dict) or not raw_values:
                raise ValueError(
                    f"Gradient assignment contract ({variant}): cluster "
                    f"{cid} has missing normalized counts: {run_dir}"
                )
            if set(str(key) for key in raw_values) - set(names):
                raise ValueError(
                    f"Gradient assignment contract ({variant}): cluster "
                    f"{cid} has unknown normalized networks: {run_dir}"
                )
            try:
                values = {
                    str(name): float(value)
                    for name, value in raw_values.items()
                }
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"Gradient assignment contract ({variant}): cluster "
                    f"{cid} has nonnumeric normalized counts: {run_dir}"
                ) from exc
            if any(
                not np.isfinite(value) or value < 0
                for value in values.values()
            ):
                raise ValueError(
                    f"Gradient assignment contract ({variant}): cluster "
                    f"{cid} has invalid normalized counts: {run_dir}"
                )
        elif variant in ALL_OVR_VARIANTS:
            if profile is None or profile_names is None:
                raise ValueError(f"OVR missing exact coefficients: {run_dir}")
            values = dict(zip(profile_names, map(float, profile[cid - 1]), strict=True))
            reported = item.get("beta_by_network")
            if not isinstance(reported, dict) or set(reported) != set(values) or any(
                not np.isclose(float(reported[name]), value, rtol=0, atol=1e-10)
                for name, value in values.items()
            ):
                raise ValueError(f"OVR status coefficients disagree: cluster={cid}: {run_dir}")
        elif variant in {"fisherz", "network_ts_fisherz"}:
            if profile is None or profile_names is None:
                raise ValueError(
                    f"Gradient assignment contract ({variant}): missing "
                    f"profile for cluster {cid}: {run_dir}"
                )
            if cid < 1 or cid > len(ids):
                raise ValueError(
                    f"Gradient assignment contract ({variant}): profile "
                    f"cluster IDs must be contiguous and one-based: {run_dir}"
                )
            values = dict(zip(profile_names, map(float, profile[cid - 1])))
            reported = item.get("top1_fisher_z")
            if reported is None or not np.isfinite(float(reported)):
                raise ValueError(
                    f"Gradient assignment contract ({variant}): cluster "
                    f"{cid} is missing finite top1_fisher_z: {run_dir}"
                )
            top2 = str(item.get("top2_network", ""))
            top2_value = item.get("top2_fisher_z")
            if (
                top2 in values
                and top2_value is not None
                and np.isfinite(float(top2_value))
                and not np.isclose(
                    float(top2_value), values[top2], rtol=0, atol=1e-6
                )
            ):
                raise ValueError(
                    f"Gradient assignment contract ({variant}): cluster "
                    f"{cid} status top2_fisher_z disagrees with profile "
                    f"value: {run_dir}"
                )
        else:
            raise ValueError(
                f"Gradient assignment contract ({variant}): unsupported "
                f"profile variant: {run_dir}"
            )

        top = str(item.get("top1_network", ""))
        positive = {
            name: value for name, value in values.items() if value > 0
        }
        try:
            maximum = max(values.values())
        except ValueError as exc:
            raise ValueError(
                f"Gradient assignment contract ({variant}): cluster {cid} "
                f"has no profile values: {run_dir}"
            ) from exc

        if top in {"unassigned", "none", ""}:
            if "unassigned_label_key" not in mapping:
                raise ValueError(
                    f"Gradient assignment contract ({variant}): unassigned "
                    f"winner requires explicit unassigned_label_key: {run_dir}"
                )
            expected_key = int(mapping["unassigned_label_key"])
            if top not in {"unassigned", "none"} or positive:
                raise ValueError(
                    f"Gradient assignment contract ({variant}): cluster "
                    f"{cid} has an unassigned winner with positive profile "
                    f"values: {run_dir}"
                )
        else:
            minimum_positive = variant == "normalized_top20_count" or variant in OVR_VARIANTS
            if (
                top not in values
                or (minimum_positive and values[top] <= 0)
                or not np.isclose(
                    values[top], maximum, rtol=0, atol=1e-10
                )
            ):
                raise ValueError(
                    f"Gradient assignment contract ({variant}): cluster "
                    f"{cid} status winner {top!r} disagrees with "
                    f"profile/count maximum: {run_dir}"
                )
            expected_key = int(key_by_name[top])

        if variant == "fisherz":
            reported = item.get("top1_fisher_z")
            if reported is None or not np.isfinite(float(reported)):
                raise ValueError(
                    f"Gradient assignment contract ({variant}): cluster "
                    f"{cid} status is missing finite top1_fisher_z: {run_dir}"
                )
            if top not in values or not np.isclose(
                float(reported), values[top], rtol=0, atol=1e-6
            ):
                raise ValueError(
                    f"Gradient assignment contract ({variant}): cluster "
                    f"{cid} status top1_fisher_z disagrees with profile "
                    f"value: {run_dir}"
                )

        if (
            str(cid) not in cluster_map
            or int(cluster_map[str(cid)]) != expected_key
        ):
            raise ValueError(
                f"Gradient assignment contract ({variant}): cluster {cid} "
                f"label-map winner disagrees with {top}: {run_dir}"
            )
        if not np.all(labels[clusters == cid] == expected_key):
            raise ValueError(
                f"Gradient assignment contract ({variant}): cluster {cid} "
                f"surface labels disagree with {top}: {run_dir}"
            )

        total = sum(positive.values())
        compositions[cid] = {} if variant in OVR_VARIANTS else (
            {name: 100.0 * value / total for name, value in positive.items()}
            if total > 0
            else {}
        )
        winners[cid] = top

    return winners, compositions


def assignment_asset_fingerprints(
    validated: Mapping[str, Any],
) -> dict[str, object]:
    """Return JSON-safe fingerprints for the exact assets used by a render."""
    paths = validated.get("paths", [])
    out: dict[str, object] = {}
    for raw in paths:
        path = Path(raw).resolve()
        out[str(path)] = {
            "sha256": sha256_file(path),
            "size": int(path.stat().st_size),
        }
    label_path = validated.get("label_path")
    if label_path is not None:
        path = Path(label_path).resolve()
        if str(path) not in out:
            out[str(path)] = {
                "sha256": sha256_file(path),
                "size": int(path.stat().st_size),
            }
    return out


def validate_gradient_assignment(
    run_dir: Path, manifest: Mapping[str, Any], variant: str
) -> dict[str, Any]:
    """Return validated compositions and exact render assets."""
    run_dir = Path(run_dir).resolve()
    if variant not in ASSIGNMENT_VARIANTS:
        raise ValueError(
            f"Gradient assignment contract ({variant}): unsupported variant "
            f"{variant!r}: {run_dir}"
        )

    variants = manifest.get("assignment_variants")
    row = variants.get(variant) if isinstance(variants, dict) else None
    if not isinstance(row, dict):
        raise ValueError(
            f"Gradient assignment contract ({variant}): missing "
            f"assignment_variants entry: {run_dir}"
        )

    fields = (
        "cluster_assignment_status",
        "group_functional_atlas_labels",
        "group_functional_atlas_label_map",
        "template_functional_atlas_label",
    )
    paths = {
        field: _declared_path(row, field, run_dir, variant)
        for field in fields
    }

    cluster_path = run_dir / "group_labels_final.npy"
    if not cluster_path.is_file():
        raise FileNotFoundError(
            f"Gradient assignment missing group_labels_final.npy: {cluster_path}"
        )
    clusters = _load_integer_vector(
        cluster_path,
        label="cluster labels",
        variant=variant,
        run_dir=run_dir,
    )
    labels = _load_integer_vector(
        paths["group_functional_atlas_labels"],
        label="functional labels",
        variant=variant,
        run_dir=run_dir,
    )
    if not np.any(clusters > 0) or np.any(clusters < 0):
        raise ValueError(
            f"Gradient assignment contract ({variant}): cluster labels must "
            f"contain at least one positive ID and no negative IDs: {run_dir}"
        )

    try:
        gifti = nib.load(str(paths["template_functional_atlas_label"]))
    except (OSError, ValueError) as exc:
        raise ValueError(
            f"Gradient assignment contract ({variant}): cannot load GIFTI "
            f"label asset: {paths['template_functional_atlas_label']}"
        ) from exc
    arrays = gifti.get_arrays_from_intent("NIFTI_INTENT_LABEL")
    if len(arrays) != 1:
        raise ValueError(
            f"Gradient assignment contract ({variant}): GIFTI must contain "
            f"exactly one label array: {run_dir}"
        )
    raw_gii_labels = np.asarray(arrays[0].data)
    if raw_gii_labels.ndim != 1 or not np.issubdtype(
        raw_gii_labels.dtype, np.integer
    ):
        raise ValueError(
            f"Gradient assignment contract ({variant}): GIFTI labels must "
            f"be a one-dimensional integer array: {run_dir}"
        )
    gii_labels = raw_gii_labels.astype(np.int32, copy=False)
    if labels.shape != clusters.shape or gii_labels.shape != labels.shape:
        raise ValueError(
            f"Gradient assignment contract ({variant}): cluster/NPY/GIFTI "
            f"shape mismatch: {run_dir}"
        )
    if not np.array_equal(labels, gii_labels):
        raise ValueError(
            f"Gradient assignment contract ({variant}): NPY/GIFTI labels "
            f"disagree: {paths['template_functional_atlas_label']}"
        )

    mapping = _load_mapping(
        paths["group_functional_atlas_label_map"], variant, run_dir
    )
    names = mapping.get("network_names")
    keys = mapping.get("network_label_keys")
    cluster_map = mapping.get("cluster_to_functional_label_key")
    unassigned_key = mapping.get("unassigned_label_key")
    if (
        not isinstance(names, list)
        or not names
        or any(not isinstance(value, str) or not value for value in names)
        or len(set(names)) != len(names)
    ):
        raise ValueError(
            f"Gradient assignment contract ({variant}): ambiguous network "
            f"names: {run_dir}"
        )
    if (
        not isinstance(keys, dict)
        or set(str(key) for key in keys) != set(names)
    ):
        raise ValueError(
            f"Gradient assignment contract ({variant}): network label map "
            f"names disagree: {run_dir}"
        )
    try:
        key_values = [int(value) for value in keys.values()]
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"Gradient assignment contract ({variant}): network label keys "
            f"must be integers: {run_dir}"
        ) from exc
    if len(set(key_values)) != len(key_values) or sorted(key_values) != list(
        range(1, len(names) + 1)
    ):
        raise ValueError(
            f"Gradient assignment contract ({variant}): network label keys "
            f"must be contiguous: {run_dir}"
        )
    if unassigned_key is not None:
        try:
            unassigned_key = int(unassigned_key)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"Gradient assignment contract ({variant}): invalid "
                f"unassigned_label_key: {run_dir}"
            ) from exc
        if unassigned_key != len(names) + 1:
            raise ValueError(
                f"Gradient assignment contract ({variant}): invalid "
                f"unassigned_label_key: {run_dir}"
            )

    inferred_unassigned_key = len(names) + 1
    if unassigned_key is None and (
        np.any(labels == inferred_unassigned_key)
        or (
            isinstance(cluster_map, dict)
            and any(
                int(value) == inferred_unassigned_key
                for value in cluster_map.values()
            )
        )
    ):
        raise ValueError(
            f"Gradient assignment contract ({variant}): unassigned labels "
            f"require explicit unassigned_label_key: {run_dir}"
        )
    if not isinstance(cluster_map, dict):
        raise ValueError(
            f"Gradient assignment contract ({variant}): "
            f"cluster_to_functional_label_key must be an object: {run_dir}"
        )
    actual_ids = sorted(
        int(value) for value in np.unique(clusters) if int(value) > 0
    )
    if set(str(value) for value in cluster_map) != set(
        str(value) for value in actual_ids
    ):
        raise ValueError(
            f"Gradient assignment contract ({variant}): cluster label map "
            f"IDs disagree with surface: {run_dir}"
        )
    try:
        cluster_keys = [int(value) for value in cluster_map.values()]
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"Gradient assignment contract ({variant}): cluster label map "
            f"keys must be integers: {run_dir}"
        ) from exc
    allowed_keys = set(key_values) | (
        {int(unassigned_key)} if unassigned_key is not None else set()
    )
    if any(value not in allowed_keys for value in cluster_keys):
        raise ValueError(
            f"Gradient assignment contract ({variant}): cluster label map "
            f"contains unknown network key: {run_dir}"
        )

    _validate_label_table(
        gifti,
        names=[str(value) for value in names],
        keys={str(key): int(value) for key, value in keys.items()},
        unassigned_key=unassigned_key,
        variant=variant,
        run_dir=run_dir,
    )

    status = _read_json(
        paths["cluster_assignment_status"],
        label="status",
        variant=variant,
        run_dir=run_dir,
    )
    if (
        not isinstance(status, dict)
        or status.get("assignment_metric") != EXPECTED_METRICS[variant]
    ):
        raise ValueError(
            f"Gradient assignment contract ({variant}): wrong "
            f"assignment_metric; expected {EXPECTED_METRICS[variant]!r}: "
            f"{run_dir}"
        )

    profile: np.ndarray | None = None
    profile_names: list[str] | None = None
    dependencies: list[Path] = [cluster_path, *paths.values()]
    if variant in ALL_OVR_VARIANTS:
        coefficient_path = _declared_path(row, "network_profile", run_dir, variant)
        metadata_path = _declared_path(row, "ovr_model_metadata", run_dir, variant)
        model = _read_json(metadata_path, label="OVR model", variant=variant, run_dir=run_dir)
        taxonomy = "canonical7" if variant == NETWORK_TS_OVR_VARIANT else variant.removeprefix("ovr_logistic_")
        if model.get("taxonomy") != taxonomy or model.get("variant") != variant:
            raise ValueError(f"OVR taxonomy/variant mismatch: {metadata_path}")
        if (model.get("cluster_ids") != actual_ids or model.get("network_names") != names
                or row.get("profile_units") != "raw_beta_per_Fisher_z_unit"):
            raise ValueError(f"OVR class/network order or units mismatch: {metadata_path}")
        if Path(str(model.get("artifacts", {}).get("beta", ""))).resolve() != coefficient_path:
            raise ValueError(f"OVR exact coefficient path mismatch: {metadata_path}")
        profile = np.asarray(np.load(coefficient_path, allow_pickle=False), dtype=float)
        profile_names = list(names)
        if profile.shape != (len(actual_ids), len(names)) or not np.isfinite(profile).all():
            raise ValueError(f"OVR coefficient shape/values mismatch: {coefficient_path}")
        dependencies.extend([coefficient_path, metadata_path])
    if variant in {"fisherz", "network_ts_fisherz"}:
        if variant == "network_ts_fisherz":
            profile_raw = manifest.get("cluster_network_ts_profile") or str(run_dir / "cluster_network_ts_profile.npy")
            declaration = "network-ts"
        else:
            profile_raw = manifest.get("cluster_network_profile")
            declaration = "top-level ordinary-run"
        if not isinstance(profile_raw, str) or not profile_raw.strip():
            raise ValueError(
                f"Gradient assignment contract ({variant}): missing exact "
                f"{declaration} cluster_network_profile declaration: {run_dir}"
            )
        profile_path = Path(profile_raw).expanduser().resolve()
        try:
            profile_path.relative_to(run_dir)
        except ValueError as exc:
            raise ValueError(
                f"Gradient assignment contract ({variant}): declared profile "
                f"is outside hemisphere run directory: {profile_path}"
            ) from exc
        summary_path = run_dir / "group_feature_summary.json"
        if not profile_path.is_file() or not summary_path.is_file():
            raise FileNotFoundError(
                f"Gradient assignment contract ({variant}): missing exact "
                f"Fisher-z profile/summary: {run_dir}"
            )
        try:
            profile = np.asarray(
                np.load(profile_path, allow_pickle=False), dtype=float
            )
        except (OSError, ValueError) as exc:
            raise ValueError(
                f"Gradient assignment contract ({variant}): invalid "
                f"cluster_network_profile: {profile_path}"
            ) from exc
        summary = _read_json(
            summary_path,
            label="group feature summary",
            variant=variant,
            run_dir=run_dir,
        )
        profile_names_raw = (
            summary.get("network_names")
            if isinstance(summary, dict)
            else None
        )
        if (
            not isinstance(profile_names_raw, list)
            or any(not isinstance(value, str) for value in profile_names_raw)
            or len(set(profile_names_raw)) != len(profile_names_raw)
            or set(profile_names_raw) != set(str(value) for value in names)
        ):
            raise ValueError(
                f"Gradient assignment contract ({variant}): profile network "
                f"names disagree with label map: {run_dir}"
            )
        profile_names = [str(value) for value in profile_names_raw]
        if (
            profile.shape != (len(actual_ids), len(profile_names))
            or not np.isfinite(profile).all()
        ):
            raise ValueError(
                f"Gradient assignment contract ({variant}): invalid profile "
                f"shape or nonfinite values: {run_dir}"
            )
        if variant == "fisherz" and actual_ids != list(
            range(1, len(actual_ids) + 1)
        ):
            raise ValueError(
                f"Gradient assignment contract ({variant}): profile cluster "
                f"IDs must be contiguous and one-based: {run_dir}"
            )
        dependencies.extend([profile_path, summary_path])

    winners, compositions = _validate_status_items(
        status,
        clusters=clusters,
        labels=labels,
        mapping=mapping,
        profile=profile,
        profile_names=profile_names,
        variant=variant,
        run_dir=run_dir,
    )
    validated: dict[str, Any] = {
        "profile": profile,
        "network_names": profile_names,
        "winners": winners,
        "compositions": compositions,
        "paths": dependencies,
        "label_path": paths["template_functional_atlas_label"],
        "variant": variant,
    }
    validated["asset_fingerprints"] = assignment_asset_fingerprints(validated)
    return validated
