#!/usr/bin/env python3
from __future__ import annotations

import json
import re
from pathlib import Path

import nibabel as nib
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[3]
KONG2022_DLABEL_PATH = (
    REPO_ROOT
    / "external"
    / "Schaefer2018_Kong2022"
    / "Schaefer2018_400Parcels_Kong2022_17Networks_order.dlabel.nii"
)
KONG2022_MERGE_CONFIG = REPO_ROOT / "config" / "cross_atlas_network_merge.json"

KNOWN_FAMILIES_17: frozenset[str] = frozenset({
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
})

_ROI_NAME_RE = re.compile(
    r"^17networks_(?P<hemi>LH|RH)_(?P<family>[A-Za-z0-9]+)(?:_(?P<subregion>.+))?_(?P<rank>\d+)$"
)


def parse_kong2022_roi_name(name: str) -> tuple[str, str, str, int]:
    m = _ROI_NAME_RE.fullmatch(name)
    if m is None:
        raise ValueError(f"Label does not match Kong2022 pattern: {name!r}")
    family17 = str(m.group("family"))
    if family17 not in KNOWN_FAMILIES_17:
        raise ValueError(f"Unknown Kong2022 family token in label: {name!r}")
    hemi = "L" if m.group("hemi") == "LH" else "R"
    subregion = str(m.group("subregion") or "")
    rank = int(m.group("rank"))
    return hemi, family17, subregion, rank


def canonical_families_8_order() -> list[str]:
    return [
        "Default",
        "Visual",
        "Somatomotor",
        "DorsalAttention",
        "VentralAttention",
        "Control",
        "Auditory",
        "Language",
    ]


def _label_name_from_axis_value(value: object) -> str:
    if isinstance(value, tuple) and value:
        return str(value[0])
    return str(value)


def load_kong2022_merge_mapping(config_path: Path | None = None) -> dict[str, str]:
    path = config_path if config_path is not None else KONG2022_MERGE_CONFIG
    payload = json.loads(path.read_text(encoding="utf-8"))
    mapping = {str(k): str(v) for k, v in payload["atlases"]["kong2022"]["mapping"].items()}
    expected = set(KNOWN_FAMILIES_17)
    if set(mapping.keys()) != expected:
        missing = sorted(expected - set(mapping.keys()))
        extra = sorted(set(mapping.keys()) - expected)
        raise KeyError(
            f"Unexpected Kong2022 merge mapping keys. missing={missing} extra={extra}"
        )
    allowed_targets = set(canonical_families_8_order())
    invalid_targets = sorted({v for v in mapping.values() if v not in allowed_targets})
    if invalid_targets:
        raise ValueError(f"Invalid Kong2022 canonical targets: {invalid_targets}")
    return mapping


def load_kong2022_dlabel(
    dlabel_path: Path | None = None,
) -> tuple[nib.Cifti2Image, dict[int, str]]:
    path = dlabel_path if dlabel_path is not None else KONG2022_DLABEL_PATH
    img = nib.load(str(path))
    label_axis = img.header.get_axis(0)
    # axis 0 of a dlabel is the LabelAxis; its .label attribute is a list of
    # dicts (one per scalar map, here just one map at index 0).
    label_map: dict[int, str] = {}
    for parcel_id, label_value in label_axis.label[0].items():
        key = int(parcel_id)
        if key == 0:
            continue
        label_map[key] = _label_name_from_axis_value(label_value)
    if len(label_map) != 400:
        raise ValueError(f"Expected 400 non-zero parcels, got {len(label_map)}")
    for roi_id, roi_name in label_map.items():
        try:
            parse_kong2022_roi_name(roi_name)
        except Exception as exc:
            raise ValueError(f"Invalid Kong2022 ROI name for id={roi_id}: {roi_name}") from exc
    return img, label_map


def split_vertex_label_arrays(
    img: nib.Cifti2Image,
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    data = img.get_fdata(dtype=np.float32)  # shape (1, n_grayordinates_total)
    data = data[0, :]
    geom_axis = img.header.get_axis(1)
    result: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for name, slc, subaxis in geom_axis.iter_structures():
        if "CORTEX_LEFT" in name:
            hemi_key = "L"
        elif "CORTEX_RIGHT" in name:
            hemi_key = "R"
        else:
            continue
        vertex_indices = np.asarray(subaxis.vertex, dtype=np.int32)
        parcel_ids = np.asarray(data[slc], dtype=np.int32)
        result[hemi_key] = (vertex_indices, parcel_ids)
    return result


def build_roi_registry(
    label_map: dict[int, str],
    mapping_17_to_8: dict[str, str],
) -> list[dict]:
    rows: list[dict] = []
    for roi_id in sorted(label_map):
        roi_name = label_map[roi_id]
        hemi, family17, subregion, rank = parse_kong2022_roi_name(roi_name)
        canonical8 = mapping_17_to_8[family17]
        rows.append(
            {
                "roi_id": roi_id,
                "roi_name": roi_name,
                "hemi": hemi,
                "family17": family17,
                "subregion": subregion,
                "rank": rank,
                "canonical8": canonical8,
            }
        )
    if len(rows) != 400:
        raise ValueError(f"Expected 400 registry rows, got {len(rows)}")
    return rows


def load_kong2022_bundle(
    dlabel_path: Path | None = None,
    config_path: Path | None = None,
) -> dict:
    img, label_map = load_kong2022_dlabel(dlabel_path)
    mapping_17_to_8 = load_kong2022_merge_mapping(config_path)
    roi_registry = build_roi_registry(label_map, mapping_17_to_8)
    vertex_arrays = split_vertex_label_arrays(img)
    return {
        "img": img,
        "label_map": label_map,
        "roi_registry": roi_registry,
        "mapping_17_to_8": mapping_17_to_8,
        "canonical8_order": canonical_families_8_order(),
        "vertex_arrays": vertex_arrays,
    }


if __name__ == "__main__":
    bundle = load_kong2022_bundle()
    print(f"Total parcels: {len(bundle['label_map'])}")
    print(f"Registry rows: {len(bundle['roi_registry'])}")
    print("First 3 registry rows:")
    for row in bundle["roi_registry"][:3]:
        print(" ", row)
    print("Canonical 8 families:", bundle["canonical8_order"])
    vL, vR = bundle["vertex_arrays"]["L"], bundle["vertex_arrays"]["R"]
    print(f"Vertex arrays — L: {vL[0].shape[0]} grayordinates, R: {vR[0].shape[0]} grayordinates")
