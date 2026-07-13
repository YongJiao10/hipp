from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

from .io_utils import write_tsv


LYNCH2024_ATLAS_KEY = "lynch2024"
LYNCH2024_20NET_ATLAS = "lynch2024_20net"
LYNCH2024_8NET_ATLAS = "lynch2024_8net"
LYNCH2024_CANONICAL8_ORDER = [
    "Default",
    "Visual",
    "Somatomotor",
    "DorsalAttention",
    "VentralAttention",
    "Control",
    "Auditory",
    "Language",
]


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
    names = [network for network in mapping if network not in excluded and mapping[network] not in excluded]
    if len(names) != 20 or len(set(names)) != 20:
        raise ValueError(
            "Lynch2024 formal taxonomy must contain exactly 20 unique non-Noise networks: "
            f"count={len(names)} names={names}"
        )
    return names


def lynch2024_8net_order(merge_config: Path) -> list[str]:
    mapping = lynch2024_mapping(merge_config)
    excluded = lynch2024_exclude_labels(merge_config)
    present = {target for source, target in mapping.items() if source not in excluded and target not in excluded}
    return [name for name in LYNCH2024_CANONICAL8_ORDER if name in present]


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
    order8 = lynch2024_8net_order(merge_config)
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

    out_root = Path(out_dir)
    lynch20_dir = out_root / LYNCH2024_20NET_ATLAS
    lynch8_dir = out_root / LYNCH2024_8NET_ATLAS
    lynch20_dir.mkdir(parents=True, exist_ok=True)
    lynch8_dir.mkdir(parents=True, exist_ok=True)

    kept_sorted = sorted(kept, key=lambda item: (order20.index(str(item[1]["network"])), item[0]))
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
                "canonical8": mapping[net],
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
            "canonical8",
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
                "canonical8": mapping[network],
                "n_components_merged": int(len(idx)),
                "n_grayordinates_used_total": float(np.sum(weights)),
            }
        )
    np.save(lynch20_dir / "cortex_lynch20_network_timeseries.npy", np.asarray(network20_ts, dtype=np.float32))
    write_tsv(
        lynch20_dir / "cortex_lynch20_networks.tsv",
        network20_rows,
        ["lynch20_network", "canonical8", "n_components_merged", "n_grayordinates_used_total"],
    )

    network8_rows: list[dict[str, object]] = []
    network8_ts: list[np.ndarray] = []
    for network in order8:
        idx = [i for i, row in enumerate(kept_rows) if mapping[str(row["network"])] == network]
        if not idx:
            continue
        weights = np.asarray([_component_weight(kept_rows[i]) for i in idx], dtype=np.float64)
        ts = np.average(lynch20_ts[np.asarray(idx, dtype=np.int32), :], axis=0, weights=weights).astype(np.float32)
        network8_ts.append(ts)
        parents = [str(kept_rows[i]["network"]) for i in idx]
        network8_rows.append(
            {
                "canonical_network": network,
                "n_components_merged": int(len(idx)),
                "original_parent_networks": ",".join(dict.fromkeys(parents)),
                "n_grayordinates_used_total": float(np.sum(weights)),
            }
        )
    np.save(lynch8_dir / "cortex_canonical_network_timeseries.npy", np.asarray(network8_ts, dtype=np.float32))
    write_tsv(
        lynch8_dir / "cortex_canonical_networks.tsv",
        network8_rows,
        ["canonical_network", "n_components_merged", "original_parent_networks", "n_grayordinates_used_total"],
    )
    np.save(lynch8_dir / "cortex_roi_parcel_timeseries.npy", lynch20_ts)
    write_tsv(
        lynch8_dir / "cortex_roi_parcels.tsv",
        lynch20_rows,
        [
            "parcel_id",
            "parcel_name",
            "lynch20_network",
            "canonical8",
            "hemisphere",
            "component_order",
            "n_grayordinates_used",
        ],
    )

    return {
        "lynch20_order": [str(row["lynch20_network"]) for row in network20_rows],
        "lynch8_order": [str(row["canonical_network"]) for row in network8_rows],
        "lynch20_timeseries": lynch20_dir / "cortex_roi_parcel_timeseries.npy",
        "lynch20_tsv": lynch20_dir / "cortex_roi_parcels.tsv",
        "lynch20_network_timeseries": lynch20_dir / "cortex_lynch20_network_timeseries.npy",
        "lynch20_network_tsv": lynch20_dir / "cortex_lynch20_networks.tsv",
        "lynch8_timeseries": lynch8_dir / "cortex_canonical_network_timeseries.npy",
        "lynch8_tsv": lynch8_dir / "cortex_canonical_networks.tsv",
        "lynch8_roi_timeseries": lynch8_dir / "cortex_roi_parcel_timeseries.npy",
        "lynch8_roi_tsv": lynch8_dir / "cortex_roi_parcels.tsv",
    }
