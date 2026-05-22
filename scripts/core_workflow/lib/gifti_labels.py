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


def materialize_combo(manifest_path: Path, color_map: dict[str, tuple[int, int, int, int]]) -> bool:
    manifest = load_json(manifest_path)
    if str(manifest.get("status", "ok")) != "ok":
        return False
    run_dir = manifest_path.parent
    if not (run_dir / "group_labels_final.npy").exists():
        return False
    if not (run_dir / "cluster_assignment_status.json").exists():
        return False
    labels = np.load(run_dir / "group_labels_final.npy").astype(np.int32)
    status_items = load_json(run_dir / "cluster_assignment_status.json")["items"]
    network_names = load_json(run_dir / "group_feature_summary.json")["network_names"]

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

    np.save(run_dir / "group_functional_atlas_labels.npy", functional.astype(np.int32))
    label_map = {
        "network_names": [str(x) for x in network_names],
        "network_label_keys": {str(n): int(i + 1) for i, n in enumerate(network_names)},
        "unassigned_label_key": int(key_unassigned),
        "cluster_to_functional_label_key": {str(k): int(v) for k, v in sorted(cluster_to_key.items())},
    }
    write_json(run_dir / "group_functional_atlas_label_map.json", label_map)

    label_items: list[tuple[int, str, tuple[int, int, int, int]]] = [(0, "Null", (158, 158, 158, 255))]
    for i, n in enumerate(network_names, start=1):
        label_items.append((int(i), str(n), color_map.get(str(n), (80, 80, 80, 255))))
    label_items.append((key_unassigned, "unassigned", (20, 20, 20, 255)))
    func_lbl = make_named_label_gifti(functional.astype(np.int32), label_items)

    cluster_lbl_path = resolve_cluster_label_path(manifest, run_dir)
    func_lbl_path = cluster_lbl_path.with_name(cluster_lbl_path.name.replace("label-group_cluster_", "label-group_functional_atlas_"))
    func_lbl_path.parent.mkdir(parents=True, exist_ok=True)
    nib.save(func_lbl, str(func_lbl_path))

    manifest["template_cluster_label"] = str(cluster_lbl_path)
    manifest["group_functional_atlas_labels"] = str((run_dir / "group_functional_atlas_labels.npy").resolve())
    manifest["group_functional_atlas_label_map"] = str((run_dir / "group_functional_atlas_label_map.json").resolve())
    manifest["template_functional_atlas_label"] = str(func_lbl_path.resolve())
    write_json(manifest_path, manifest)
    return True

