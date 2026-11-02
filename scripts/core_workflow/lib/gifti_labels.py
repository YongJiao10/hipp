from __future__ import annotations

from .network_ovr import OVR_VARIANTS
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


def _declared_path(raw: object, *, field: str, run_dir: Path, require_file: bool) -> Path:
    if not isinstance(raw, str) or not raw.strip():
        raise ValueError(f"Missing exact {field} declaration: {run_dir / 'group_atlas_manifest.json'}")
    path = Path(raw).expanduser()
    if not path.is_absolute():
        raise ValueError(f"Exact {field} must be absolute: {path}")
    path = path.resolve()
    try:
        path.relative_to(run_dir.resolve())
    except ValueError as exc:
        raise ValueError(f"Exact {field} points outside run directory: {path}") from exc
    if require_file and not path.is_file():
        raise FileNotFoundError(f"Missing exact {field}: {path}")
    return path


def resolve_cluster_label_path(manifest: dict, run_dir: Path) -> Path:
    return _declared_path(
        manifest.get("template_cluster_label"),
        field="template_cluster_label",
        run_dir=run_dir,
        require_file=False,
    )


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
    variants = manifest.get("assignment_variants")
    if str(manifest.get("assignment_mode", "")) == "cluster_level" or (
        isinstance(variants, dict) and bool(variants)
    ):
        raise FileNotFoundError(
            "Cluster-level manifest is missing group_labels_final.npy; refusing to refresh "
            "from an unqualified top-level label asset: "
            f"{run_dir / 'group_labels_final.npy'}"
        )
    labels_path, map_path = resolve_functional_atlas_array_paths(manifest, run_dir)
    label_path = _declared_path(
        manifest.get("template_functional_atlas_label"),
        field="template_functional_atlas_label",
        run_dir=run_dir,
        require_file=False,
    )
    if not labels_path.is_file() or not map_path.is_file():
        raise FileNotFoundError(
            "Declared functional atlas arrays are missing; no alternate asset is allowed: "
            f"labels={labels_path} map={map_path}"
        )
    labels = np.asarray(np.load(labels_path), dtype=np.int32)
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
        raw_selected = variant if variant is not None else manifest.get("default_assignment_variant")
        if not isinstance(raw_selected, str) or not raw_selected.strip():
            raise ValueError(f"Missing exact default_assignment_variant: {run_dir / 'group_atlas_manifest.json'}")
        selected = str(raw_selected)
        if not isinstance(variants, dict) or selected not in variants or not isinstance(variants[selected], dict):
            raise ValueError(
                f"Missing exact assignment_variants.{selected}: {run_dir / 'group_atlas_manifest.json'}"
            )
        row = variants[selected]
        labels_path = _declared_path(
            row.get("group_functional_atlas_labels"),
            field=f"assignment_variants.{selected}.group_functional_atlas_labels",
            run_dir=run_dir,
            require_file=False,
        )
        map_path = _declared_path(
            row.get("group_functional_atlas_label_map"),
            field=f"assignment_variants.{selected}.group_functional_atlas_label_map",
            run_dir=run_dir,
            require_file=False,
        )
        return labels_path, map_path

    labels_path = _declared_path(
        manifest.get("group_functional_atlas_labels"),
        field="group_functional_atlas_labels",
        run_dir=run_dir,
        require_file=False,
    )
    map_path = _declared_path(
        manifest.get("group_functional_atlas_label_map"),
        field="group_functional_atlas_label_map",
        run_dir=run_dir,
        require_file=False,
    )
    return labels_path, map_path


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
    status_payload = load_json(status_path)
    status_items = status_payload.get("items")
    if not isinstance(status_items, list):
        raise ValueError(f"Exact assignment status has no items: {status_path}")
    functional, label_map = _functional_from_status(
        labels=labels,
        status_items=status_items,
        network_names=network_names,
    )
    labels_path.parent.mkdir(parents=True, exist_ok=True)
    map_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(labels_path, functional.astype(np.int32))
    write_json(map_path, label_map)
    func_lbl = make_named_label_gifti(functional.astype(np.int32), _functional_label_items(network_names, color_map))
    label_path.parent.mkdir(parents=True, exist_ok=True)
    nib.save(func_lbl, str(label_path))


def materialize_combo(manifest_path: Path, color_map: dict[str, tuple[int, int, int, int]]) -> bool:
    """Materialize every declared variant and validate the complete asset tuple."""
    manifest = load_json(manifest_path)
    if str(manifest.get("status", "ok")) != "ok":
        return False
    run_dir = Path(manifest_path).parent.resolve()
    labels_path = run_dir / "group_labels_final.npy"
    if not labels_path.is_file():
        refreshed = _refresh_existing_functional_label_table(manifest, run_dir, color_map)
        if refreshed:
            write_json(manifest_path, manifest)
        return refreshed
    variants = manifest.get("assignment_variants")
    if not isinstance(variants, dict) or not variants:
        raise ValueError(
            "Cluster-level materialization requires explicit assignment_variants; "
            f"{manifest_path}"
        )
    labels = np.asarray(np.load(labels_path), dtype=np.int32).reshape(-1)
    feature_summary_path = run_dir / "group_feature_summary.json"
    if not feature_summary_path.is_file():
        raise FileNotFoundError(f"Missing exact group_feature_summary.json: {feature_summary_path}")
    network_names = [str(x) for x in load_json(feature_summary_path)["network_names"]]

    cluster_lbl_path = _declared_path(
        manifest.get("template_cluster_label"),
        field="template_cluster_label",
        run_dir=run_dir,
        require_file=False,
    )
    manifest["template_cluster_label"] = str(cluster_lbl_path)
    allowed = {"fisherz", "normalized_top20_count", "weighted_fisherz", *OVR_VARIANTS}
    unknown = sorted(set(str(key) for key in variants) - allowed)
    if unknown:
        raise ValueError(f"Unsupported assignment variant(s) in manifest: {unknown}: {manifest_path}")

    for variant_name in sorted(variants):
        raw_spec = variants[variant_name]
        if not isinstance(raw_spec, dict):
            raise ValueError(f"assignment_variants.{variant_name} must be an object: {manifest_path}")
        status_path = _declared_path(
            raw_spec.get("cluster_assignment_status"),
            field=f"assignment_variants.{variant_name}.cluster_assignment_status",
            run_dir=run_dir,
            require_file=True,
        )
        variant_label_path = _declared_path(
            raw_spec.get("template_functional_atlas_label"),
            field=f"assignment_variants.{variant_name}.template_functional_atlas_label",
            run_dir=run_dir,
            require_file=False,
        )
        variant_labels_path = _declared_path(
            raw_spec.get("group_functional_atlas_labels"),
            field=f"assignment_variants.{variant_name}.group_functional_atlas_labels",
            run_dir=run_dir,
            require_file=False,
        )
        variant_map_path = _declared_path(
            raw_spec.get("group_functional_atlas_label_map"),
            field=f"assignment_variants.{variant_name}.group_functional_atlas_label_map",
            run_dir=run_dir,
            require_file=False,
        )
        _write_functional_variant(
            labels=labels,
            status_path=status_path,
            network_names=([str(x) for x in raw_spec["network_names"]] if variant_name in OVR_VARIANTS else network_names),
            color_map=color_map,
            labels_path=variant_labels_path,
            map_path=variant_map_path,
            label_path=variant_label_path,
        )
        raw_spec["cluster_assignment_status"] = str(status_path)
        raw_spec["group_functional_atlas_labels"] = str(variant_labels_path)
        raw_spec["group_functional_atlas_label_map"] = str(variant_map_path)
        raw_spec["template_functional_atlas_label"] = str(variant_label_path)
        raw_spec["template_functional_atlas_label_variant"] = str(variant_label_path)

    default_variant = manifest.get("default_assignment_variant")
    if not isinstance(default_variant, str) or not default_variant.strip() or default_variant not in variants:
        raise ValueError(f"Manifest default_assignment_variant does not name an exact variant: {manifest_path}")
    default_labels_path, default_map_path = resolve_functional_atlas_array_paths(
        manifest,
        run_dir,
        variant=default_variant,
    )
    manifest["group_functional_atlas_labels"] = str(default_labels_path)
    manifest["group_functional_atlas_label_map"] = str(default_map_path)
    manifest["template_functional_atlas_label"] = str(
        _declared_path(
            variants[default_variant].get("template_functional_atlas_label"),
            field=f"assignment_variants.{default_variant}.template_functional_atlas_label",
            run_dir=run_dir,
            require_file=True,
        )
    )

    from lib.gradient_assignment_contract import validate_gradient_assignment
    for variant_name in sorted(variants):
        validate_gradient_assignment(run_dir, manifest, variant_name)
    write_json(manifest_path, manifest)
    return True
