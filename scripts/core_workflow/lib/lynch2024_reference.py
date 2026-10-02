from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import numpy as np

from .io_utils import write_tsv
from .kong2022_label_utils import canonical_families_7_order


LYNCH2024_ATLAS_KEY = "lynch2024"
LYNCH2024_20NET_ATLAS = "lynch2024_20net"
LYNCH2024_7NET_ATLAS = "lynch2024_7net"


def _load_merge_payload(merge_config: Path) -> dict[str, object]:
    return json.loads(Path(merge_config).read_text(encoding="utf-8"))


def lynch2024_mapping(merge_config: Path) -> dict[str, str]:
    payload = _load_merge_payload(merge_config)
    mapping = payload["atlases"][LYNCH2024_ATLAS_KEY]["mapping"]
    return {str(k): str(v) for k, v in mapping.items()}


def lynch2024_exclude_labels(merge_config: Path) -> set[str]:
    payload = _load_merge_payload(merge_config)
    return {str(x) for x in payload.get("exclude_labels", [])}


def lynch2024_20net_order(merge_config: Path) -> list[str]:
    mapping = lynch2024_mapping(merge_config)
    excluded = lynch2024_exclude_labels(merge_config)
    names = [
        network
        for network in mapping
        if network != "Noise"
        and mapping[network] != "Noise"
        and network not in excluded
        and mapping[network] not in excluded
    ]
    if len(names) != 20 or len(set(names)) != 20:
        raise ValueError(
            "Lynch2024 formal taxonomy must contain exactly 20 unique non-Noise networks: "
            f"count={len(names)} names={names}"
        )
    return names


def lynch2024_7net_order(merge_config: Path) -> list[str]:
    mapping = lynch2024_mapping(merge_config)
    excluded = lynch2024_exclude_labels(merge_config)
    present = {target for source, target in mapping.items() if source not in excluded and target not in excluded}
    expected = canonical_families_7_order()
    if present != set(expected):
        raise ValueError(
            "Lynch2024 canonical7 taxonomy must match Kong2022 canonical7 exactly: "
            f"present={sorted(present)} expected={expected}"
        )
    return expected


def lynch2024_merge_mapping_sha256(merge_config: Path) -> str:
    payload = {
        "mapping": lynch2024_mapping(merge_config),
        "exclude_labels": sorted(lynch2024_exclude_labels(merge_config)),
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def lynch2024_7net_reference_contract(merge_config: Path) -> dict[str, object]:
    mapping = lynch2024_mapping(merge_config)
    excluded = sorted(lynch2024_exclude_labels(merge_config))
    order = lynch2024_7net_order(merge_config)
    return {
        "network_taxonomy": "lynch7",
        "network_count": 7,
        "network_registry_column": "canonical7",
        "network_order": order,
        "network_reference_tsv": "cortex_canonical7_networks.tsv",
        "network_reference_timeseries": "cortex_canonical7_network_timeseries.npy",
        "network_name_column": "canonical7_network",
        "source_network_taxonomy": "lynch20",
        "source_provenance": "lynch20_components_weighted_by_n_grayordinates_used",
        "noise_excluded": True,
        "merge_mapping": mapping,
        "exclude_labels": excluded,
        "merge_mapping_sha256": lynch2024_merge_mapping_sha256(merge_config),
    }


def read_component_rows(path: Path) -> list[dict[str, object]]:
    with Path(path).open("r", encoding="utf-8", newline="") as f:
        rows = [dict(row) for row in csv.DictReader(f, delimiter="\t")]
    if not rows:
        raise RuntimeError(f"Empty Lynch component TSV: {path}")
    for idx, row in enumerate(rows):
        row.setdefault("component_id", str(idx + 1))
        row.setdefault("component_name", row.get("name", row["component_id"]))
        if "network" not in row:
            if "lynch20_network" in row:
                row["network"] = row["lynch20_network"]
            elif "parent_network" in row:
                row["network"] = row["parent_network"]
            else:
                raise RuntimeError(
                    f"Lynch component TSV requires a network/lynch20_network/parent_network column: {path}"
                )
        row.setdefault("hemisphere", "")
        row.setdefault("n_grayordinates_used", "")
    return rows


def _component_id(row: dict[str, object], fallback: int) -> int:
    raw = row.get("component_id", fallback)
    try:
        return int(raw)
    except ValueError:
        return int(fallback)


def _component_weight(row: dict[str, object]) -> float:
    raw = row.get("n_grayordinates_used", "")
    try:
        value = float(raw)
    except (TypeError, ValueError):
        return 1.0
    return value if np.isfinite(value) and value > 0 else 1.0


def build_lynch_network_references(
    *,
    component_timeseries: np.ndarray,
    component_tsv: Path,
    out_dir: Path,
    merge_config: Path,
) -> dict[str, object]:
    arr = np.asarray(component_timeseries, dtype=np.float32)
    if arr.ndim != 2:
        raise ValueError(f"Expected 2D component_timeseries, got shape={arr.shape}")
    rows_all = read_component_rows(component_tsv)
    if arr.shape[0] != len(rows_all):
        raise RuntimeError(
            f"Component TSV/time-series row mismatch: ts_rows={arr.shape[0]} tsv_rows={len(rows_all)}"
        )
    mapping = lynch2024_mapping(merge_config)
    excluded = lynch2024_exclude_labels(merge_config)
    order20 = lynch2024_20net_order(merge_config)
    order7 = lynch2024_7net_order(merge_config)
    order20_set = set(order20)

    kept: list[tuple[int, dict[str, object]]] = []
    for idx, row in enumerate(rows_all):
        net = str(row["network"])
        if net in excluded or mapping.get(net) in excluded:
            continue
        if net not in order20_set:
            raise KeyError(f"Unknown Lynch2024 network {net!r} in {component_tsv}")
        kept.append((idx, row))
    if not kept:
        raise RuntimeError("No Lynch2024 components remained after excluding Noise")

    kept_sorted = sorted(kept, key=lambda item: (order20.index(str(item[1]["network"])), item[0]))
    present_order = list(dict.fromkeys(str(row["network"]) for _idx, row in kept_sorted))
    if present_order != order20:
        missing = [name for name in order20 if name not in present_order]
        raise ValueError(
            "Lynch2024 component table must cover exactly the 20 formal networks in canonical order: "
            f"path={component_tsv} missing={missing} present={present_order}"
        )

    out_root = Path(out_dir)
    lynch20_dir = out_root / LYNCH2024_20NET_ATLAS
    lynch7_dir = out_root / LYNCH2024_7NET_ATLAS
    lynch20_dir.mkdir(parents=True, exist_ok=True)
    lynch7_dir.mkdir(parents=True, exist_ok=True)

    kept_idx = [idx for idx, _row in kept_sorted]
    kept_rows = [row for _idx, row in kept_sorted]
    lynch20_ts = arr[np.asarray(kept_idx, dtype=np.int32), :].astype(np.float32)
    lynch20_rows: list[dict[str, object]] = []
    for out_idx, (src_idx, row) in enumerate(kept_sorted, start=1):
        net = str(row["network"])
        lynch20_rows.append(
            {
                "parcel_id": _component_id(row, src_idx + 1),
                "parcel_name": str(row.get("component_name", f"component_{src_idx + 1}")),
                "lynch20_network": net,
                "canonical7": mapping[net],
                "hemisphere": str(row.get("hemisphere", "")),
                "component_order": int(out_idx),
                "n_grayordinates_used": row.get("n_grayordinates_used", ""),
            }
        )

    np.save(lynch20_dir / "cortex_roi_parcel_timeseries.npy", lynch20_ts)
    write_tsv(
        lynch20_dir / "cortex_roi_parcels.tsv",
        lynch20_rows,
        [
            "parcel_id",
            "parcel_name",
            "lynch20_network",
            "canonical7",
            "hemisphere",
            "component_order",
            "n_grayordinates_used",
        ],
    )
    network20_rows: list[dict[str, object]] = []
    network20_ts: list[np.ndarray] = []
    for network in order20:
        idx = [i for i, row in enumerate(kept_rows) if str(row["network"]) == network]
        if not idx:
            continue
        weights = np.asarray([_component_weight(kept_rows[i]) for i in idx], dtype=np.float64)
        ts = np.average(lynch20_ts[np.asarray(idx, dtype=np.int32), :], axis=0, weights=weights).astype(np.float32)
        network20_ts.append(ts)
        network20_rows.append(
            {
                "lynch20_network": network,
                "canonical7": mapping[network],
                "n_components_merged": int(len(idx)),
                "n_grayordinates_used_total": float(np.sum(weights)),
            }
        )
    np.save(lynch20_dir / "cortex_lynch20_network_timeseries.npy", np.asarray(network20_ts, dtype=np.float32))
    write_tsv(
        lynch20_dir / "cortex_lynch20_networks.tsv",
        network20_rows,
        ["lynch20_network", "canonical7", "n_components_merged", "n_grayordinates_used_total"],
    )

    network7_rows: list[dict[str, object]] = []
    network7_ts: list[np.ndarray] = []
    for network in order7:
        idx = [i for i, row in enumerate(kept_rows) if mapping[str(row["network"])] == network]
        if not idx:
            continue
        weights = np.asarray([_component_weight(kept_rows[i]) for i in idx], dtype=np.float64)
        ts = np.average(lynch20_ts[np.asarray(idx, dtype=np.int32), :], axis=0, weights=weights).astype(np.float32)
        network7_ts.append(ts)
        parents = [str(kept_rows[i]["network"]) for i in idx]
        network7_rows.append(
            {
                "canonical7_network": network,
                "n_components_merged": int(len(idx)),
                "original_parent_networks": ",".join(dict.fromkeys(parents)),
                "n_grayordinates_used_total": float(np.sum(weights)),
            }
        )
    if [str(row["canonical7_network"]) for row in network7_rows] != order7:
        raise RuntimeError(
            "Lynch2024 canonical7 reference did not produce the exact Kong canonical7 order: "
            f"got={[str(row['canonical7_network']) for row in network7_rows]} expected={order7}"
        )
    np.save(lynch7_dir / "cortex_canonical7_network_timeseries.npy", np.asarray(network7_ts, dtype=np.float32))
    write_tsv(
        lynch7_dir / "cortex_canonical7_networks.tsv",
        network7_rows,
        ["canonical7_network", "n_components_merged", "original_parent_networks", "n_grayordinates_used_total"],
    )
    np.save(lynch7_dir / "cortex_roi_parcel_timeseries.npy", lynch20_ts)
    write_tsv(
        lynch7_dir / "cortex_roi_parcels.tsv",
        lynch20_rows,
        [
            "parcel_id",
            "parcel_name",
            "lynch20_network",
            "canonical7",
            "hemisphere",
            "component_order",
            "n_grayordinates_used",
        ],
    )

    return {
        "lynch20_order": [str(row["lynch20_network"]) for row in network20_rows],
        "lynch7_order": [str(row["canonical7_network"]) for row in network7_rows],
        "lynch20_timeseries": lynch20_dir / "cortex_roi_parcel_timeseries.npy",
        "lynch20_tsv": lynch20_dir / "cortex_roi_parcels.tsv",
        "lynch20_network_timeseries": lynch20_dir / "cortex_lynch20_network_timeseries.npy",
        "lynch20_network_tsv": lynch20_dir / "cortex_lynch20_networks.tsv",
        "lynch7_timeseries": lynch7_dir / "cortex_canonical7_network_timeseries.npy",
        "lynch7_tsv": lynch7_dir / "cortex_canonical7_networks.tsv",
        "lynch7_roi_timeseries": lynch7_dir / "cortex_roi_parcel_timeseries.npy",
        "lynch7_roi_tsv": lynch7_dir / "cortex_roi_parcels.tsv",
        "lynch7_reference_contract": lynch2024_7net_reference_contract(merge_config),
    }
