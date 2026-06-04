from __future__ import annotations

import numpy as np
from nibabel.gifti import GiftiDataArray, GiftiImage, GiftiLabel, GiftiLabelTable
from pathlib import Path
import nibabel as nib
from lib.io_utils import load_json
from lib.workflow_utils import write_json



def make_cluster_label_gifti(labels: np.ndarray, k: int, colors: list[tuple[int, int, int, int]]) -> GiftiImage:
    tab = GiftiLabelTable()
    null = GiftiLabel(key=0, red=0.62, green=0.62, blue=0.62, alpha=1.0)
    null.label = "Null"
    tab.labels.append(null)
    for idx in range(1, int(k) + 1):
        rgba = colors[idx % len(colors)]
        g = GiftiLabel(
            key=int(idx),
            red=float(rgba[0]) / 255.0,
            green=float(rgba[1]) / 255.0,
            blue=float(rgba[2]) / 255.0,
            alpha=float(rgba[3]) / 255.0,
        )
        g.label = f"Cluster{idx}"
        tab.labels.append(g)
    da = GiftiDataArray(data=labels.astype(np.int32), intent="NIFTI_INTENT_LABEL", datatype="NIFTI_TYPE_INT32")
    return GiftiImage(darrays=[da], labeltable=tab)


def make_named_label_gifti(
    labels: np.ndarray,
    label_table_items: list[tuple[int, str, tuple[int, int, int, int]]],
) -> GiftiImage:
    tab = GiftiLabelTable()
    for key, name, rgba in label_table_items:
        g = GiftiLabel(
            key=int(key),
            red=float(rgba[0]) / 255.0,
            green=float(rgba[1]) / 255.0,
            blue=float(rgba[2]) / 255.0,
            alpha=float(rgba[3]) / 255.0,
        )
        g.label = str(name)
        tab.labels.append(g)
    da = GiftiDataArray(data=labels.astype(np.int32), intent="NIFTI_INTENT_LABEL", datatype="NIFTI_TYPE_INT32")
    return GiftiImage(darrays=[da], labeltable=tab)


def resolve_cluster_label_path(manifest: dict, run_dir: Path) -> Path:
    raw = manifest.get("template_cluster_label")
    if raw is None:
        raise ValueError(f"Missing template_cluster_label in manifest: {run_dir / 'group_atlas_manifest.json'}")
    raw_path = Path(str(raw))
    local_candidate = run_dir / "template" / "workbench_assets" / raw_path.name
    return local_candidate.resolve()


def _functional_from_status(
    *,
    labels: np.ndarray,
    status_items: list[dict[str, object]],
    network_names: list[str],
) -> tuple[np.ndarray, dict[str, object]]:
    net_to_key = {str(n): int(i + 1) for i, n in enumerate(network_names)}
    key_unassigned = int(len(network_names) + 1)
    cluster_to_key: dict[int, int] = {}
    for row in status_items:
        cid = int(row["cluster_id"])
        status = str(row["status"])
        top1 = str(row["top1_network"])
        if status == "dominant" or status == "mixed":
            cluster_to_key[cid] = int(net_to_key[top1])
        elif status == "unassigned":
            cluster_to_key[cid] = key_unassigned
        else:
            raise ValueError(f"Unknown cluster status: {status}")

    functional = np.zeros_like(labels, dtype=np.int32)
    for cid, key in cluster_to_key.items():
        functional[labels == int(cid)] = int(key)

    label_map = {
        "network_names": [str(x) for x in network_names],
        "network_label_keys": {str(n): int(i + 1) for i, n in enumerate(network_names)},
        "unassigned_label_key": int(key_unassigned),
        "cluster_to_functional_label_key": {str(k): int(v) for k, v in sorted(cluster_to_key.items())},
    }
    return functional.astype(np.int32), label_map


def _functional_label_items(
    network_names: list[str],
    color_map: dict[str, tuple[int, int, int, int]],
) -> list[tuple[int, str, tuple[int, int, int, int]]]:
    label_items: list[tuple[int, str, tuple[int, int, int, int]]] = [(0, "Null", (158, 158, 158, 255))]
    for i, n in enumerate(network_names, start=1):
        name = str(n)
        if name not in color_map:
            raise KeyError(f"Missing network color for {name!r}")
        label_items.append((int(i), name, color_map[name]))
    label_items.append((int(len(network_names) + 1), "unassigned", (20, 20, 20, 255)))
    return label_items


def _label_items_from_label_map(
    label_map: dict[str, object],
    color_map: dict[str, tuple[int, int, int, int]],
) -> list[tuple[int, str, tuple[int, int, int, int]]]:
    raw_keys = label_map.get("network_label_keys", {})
    if not isinstance(raw_keys, dict):
        raise ValueError("group_functional_atlas_label_map.network_label_keys must be an object")
    label_items: list[tuple[int, str, tuple[int, int, int, int]]] = [(0, "Null", (158, 158, 158, 255))]
    for name, key in sorted(((str(k), int(v)) for k, v in raw_keys.items()), key=lambda x: x[1]):
        if name not in color_map:
            raise KeyError(f"Missing network color for {name!r}")
        label_items.append((int(key), name, color_map[name]))
    unassigned_key = label_map.get("unassigned_label_key")
    if unassigned_key is not None:
        label_items.append((int(unassigned_key), "unassigned", (20, 20, 20, 255)))
    return label_items


def _refresh_existing_functional_label_table(
    manifest: dict,
    run_dir: Path,
    color_map: dict[str, tuple[int, int, int, int]],
) -> bool:
    try:
        labels_path, map_path = resolve_functional_atlas_array_paths(manifest, run_dir)
    except ValueError:
        return False
    label_raw = manifest.get("template_functional_atlas_label")
    if not label_raw:
        return False
    label_path = Path(str(label_raw)).resolve()
    if not labels_path.exists() or not map_path.exists():
        return False
    labels = np.load(labels_path).astype(np.int32)
    label_map = load_json(map_path)
    func_lbl = make_named_label_gifti(labels, _label_items_from_label_map(label_map, color_map))
    label_path.parent.mkdir(parents=True, exist_ok=True)
    nib.save(func_lbl, str(label_path))
    return True


def resolve_functional_atlas_array_paths(
    manifest: dict,
    run_dir: Path,
    *,
    variant: str | None = None,
) -> tuple[Path, Path]:
    variants = manifest.get("assignment_variants", {})
    has_variants = isinstance(variants, dict) and bool(variants)
    is_cluster_level = str(manifest.get("assignment_mode", "")) == "cluster_level" or has_variants
    if is_cluster_level:
        selected = str(variant or manifest.get("default_assignment_variant") or "")
        if not selected:
            raise ValueError(f"Missing default_assignment_variant in manifest: {run_dir / 'group_atlas_manifest.json'}")
        if not isinstance(variants, dict) or selected not in variants or not isinstance(variants[selected], dict):
            raise ValueError(
                f"Missing assignment_variants.{selected} in manifest: {run_dir / 'group_atlas_manifest.json'}"
            )
        row = variants[selected]
        labels_raw = row.get("group_functional_atlas_labels")
        map_raw = row.get("group_functional_atlas_label_map")
        if not labels_raw or not map_raw:
            raise ValueError(
                f"Missing functional atlas array paths for variant {selected}: {run_dir / 'group_atlas_manifest.json'}"
            )
        return Path(str(labels_raw)).resolve(), Path(str(map_raw)).resolve()

    labels_raw = manifest.get("group_functional_atlas_labels")
    map_raw = manifest.get("group_functional_atlas_label_map")
    if not labels_raw or not map_raw:
        raise ValueError(f"Missing functional atlas array paths in manifest: {run_dir / 'group_atlas_manifest.json'}")
    return Path(str(labels_raw)).resolve(), Path(str(map_raw)).resolve()


def _write_functional_variant(
    *,
    labels: np.ndarray,
    status_path: Path,
    network_names: list[str],
    color_map: dict[str, tuple[int, int, int, int]],
    labels_path: Path,
    map_path: Path,
    label_path: Path,
) -> None:
    status_items = load_json(status_path)["items"]
    functional, label_map = _functional_from_status(labels=labels, status_items=status_items, network_names=network_names)
    np.save(labels_path, functional.astype(np.int32))
    write_json(map_path, label_map)
    func_lbl = make_named_label_gifti(functional.astype(np.int32), _functional_label_items(network_names, color_map))
    label_path.parent.mkdir(parents=True, exist_ok=True)
    nib.save(func_lbl, str(label_path))


def materialize_combo(manifest_path: Path, color_map: dict[str, tuple[int, int, int, int]]) -> bool:
    manifest = load_json(manifest_path)
    if str(manifest.get("status", "ok")) != "ok":
        return False
    run_dir = manifest_path.parent
    if not (run_dir / "group_labels_final.npy").exists():
        refreshed = _refresh_existing_functional_label_table(manifest, run_dir, color_map)
        if refreshed:
            write_json(manifest_path, manifest)
        return refreshed
    labels = np.load(run_dir / "group_labels_final.npy").astype(np.int32)
    network_names = [str(x) for x in load_json(run_dir / "group_feature_summary.json")["network_names"]]
    cluster_lbl_path = resolve_cluster_label_path(manifest, run_dir)
    func_lbl_path = cluster_lbl_path.with_name(cluster_lbl_path.name.replace("label-group_cluster_", "label-group_functional_atlas_"))
    func_lbl_path.parent.mkdir(parents=True, exist_ok=True)

    manifest["template_cluster_label"] = str(cluster_lbl_path)

    variants = manifest.get("assignment_variants", {})
    if isinstance(variants, dict):
        for variant_name, raw_spec in variants.items():
            if not isinstance(raw_spec, dict):
                continue
            status_raw = raw_spec.get("cluster_assignment_status")
            if not status_raw:
                continue
            status_path = Path(str(status_raw)).resolve()
            if not status_path.exists():
                raise FileNotFoundError(f"Missing assignment variant status: {status_path}")
            if str(variant_name) == "fisherz":
                variant_label_path = func_lbl_path
            else:
                variant_label_path = Path(str(raw_spec.get("template_functional_atlas_label", ""))).resolve() if raw_spec.get("template_functional_atlas_label") else func_lbl_path.with_name(f"{func_lbl_path.stem}_{variant_name}{func_lbl_path.suffix}").resolve()
            labels_path = Path(str(raw_spec.get("group_functional_atlas_labels", run_dir / f"group_functional_atlas_labels_{variant_name}.npy"))).resolve()
            map_path = Path(str(raw_spec.get("group_functional_atlas_label_map", run_dir / f"group_functional_atlas_label_map_{variant_name}.json"))).resolve()
            _write_functional_variant(
                labels=labels,
                status_path=status_path,
                network_names=network_names,
                color_map=color_map,
                labels_path=labels_path,
                map_path=map_path,
                label_path=variant_label_path,
            )
            suffixed_label_path = func_lbl_path.with_name(f"{func_lbl_path.stem}_{variant_name}{func_lbl_path.suffix}").resolve()
            if suffixed_label_path != variant_label_path:
                _write_functional_variant(
                    labels=labels,
                    status_path=status_path,
                    network_names=network_names,
                    color_map=color_map,
                    labels_path=labels_path,
                    map_path=map_path,
                    label_path=suffixed_label_path,
                )
            raw_spec["group_functional_atlas_labels"] = str(labels_path)
            raw_spec["group_functional_atlas_label_map"] = str(map_path)
            raw_spec["template_functional_atlas_label"] = str(variant_label_path)
            raw_spec["template_functional_atlas_label_variant"] = str(suffixed_label_path)

    default_variant = str(manifest.get("default_assignment_variant") or "")
    if default_variant:
        default_labels_path, default_map_path = resolve_functional_atlas_array_paths(
            manifest,
            run_dir,
            variant=default_variant,
        )
        manifest["group_functional_atlas_labels"] = str(default_labels_path)
        manifest["group_functional_atlas_label_map"] = str(default_map_path)
        default_spec = variants.get(default_variant, {}) if isinstance(variants, dict) else {}
        if isinstance(default_spec, dict) and default_spec.get("template_functional_atlas_label"):
            manifest["template_functional_atlas_label"] = str(Path(str(default_spec["template_functional_atlas_label"])).resolve())

    write_json(manifest_path, manifest)
    return True
