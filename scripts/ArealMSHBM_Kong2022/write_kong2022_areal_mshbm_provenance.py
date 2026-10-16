#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import nibabel as nib
import numpy as np
from nibabel.gifti import GiftiDataArray, GiftiImage


REPO_ROOT = Path(__file__).resolve().parents[2]
CORE_WORKFLOW_DIR = REPO_ROOT / "scripts" / "core_workflow"
if str(CORE_WORKFLOW_DIR) not in sys.path:
    sys.path.insert(0, str(CORE_WORKFLOW_DIR))

from lib.kong2022_label_utils import load_kong2022_bundle  # noqa: E402
from lib.network_colors import load_network_colors  # noqa: E402


def read_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def file_meta(path: Path) -> dict[str, object]:
    stat = path.stat()
    return {"path": str(path), "size": int(stat.st_size), "mtime_ns": int(stat.st_mtime_ns)}


def read_labels(path: Path, *, subject: str, hemi: str) -> np.ndarray:
    values = np.loadtxt(str(path), dtype=np.int32).reshape(-1)
    if values.shape != (32492,):
        raise ValueError(f"Expected 32492 {hemi} labels for sub-{subject}, got {values.shape}")
    if np.any(values < 0) or np.any(values > 400):
        raise ValueError(f"Out-of-range {hemi} label value for sub-{subject}")
    expected = set(range(1, 201)) if hemi == "L" else set(range(201, 401))
    observed = {int(x) for x in np.unique(values) if int(x) != 0}
    unexpected = sorted(observed - expected)
    if unexpected:
        raise ValueError(f"Unexpected {hemi} parcel IDs for sub-{subject}: {unexpected[:10]}")
    return values


def make_metric_gifti(values: np.ndarray) -> GiftiImage:
    data = GiftiDataArray(
        data=values.astype(np.float32),
        intent="NIFTI_INTENT_SHAPE",
        datatype="NIFTI_TYPE_FLOAT32",
    )
    return GiftiImage(darrays=[data])


def run_workbench(wb_command: Path, args: list[str]) -> None:
    subprocess.run([str(wb_command), *args], check=True)


def write_label_table(path: Path, registry: list[dict], colors: dict[str, tuple[int, int, int, int]]) -> None:
    lines: list[str] = []
    for row in registry:
        name = str(row["roi_name"])
        family = str(row["family17"])
        rgba = colors.get(family, (158, 158, 158, 255))
        lines.extend(
            [
                name,
                f"{int(row['roi_id'])} {int(rgba[0])} {int(rgba[1])} {int(rgba[2])} {int(rgba[3])}",
            ]
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def label_stats(left: np.ndarray, right: np.ndarray) -> dict[str, object]:
    combined = np.concatenate([left, right])
    values, counts = np.unique(combined, return_counts=True)
    return {
        "combined_unique_count": int(values.size),
        "positive_unique_count": int(np.count_nonzero(values > 0)),
        "missing_positive_labels": sorted(set(range(1, 401)) - {int(x) for x in values if int(x) > 0}),
        "left_unique_count": int(np.unique(left).size),
        "right_unique_count": int(np.unique(right).size),
        "counts": {str(int(value)): int(count) for value, count in zip(values, counts)},
    }


def materialize_label_assets(args: argparse.Namespace, subject: str) -> tuple[dict[str, str], dict[str, object]]:
    required = {
        "kong2022_dlabel": args.kong2022_dlabel,
        "left_inflated": args.left_inflated,
        "right_inflated": args.right_inflated,
        "left_midthickness": args.left_midthickness,
        "right_midthickness": args.right_midthickness,
        "sulc_dscalar": args.sulc_dscalar,
        "run_lengths": args.run_lengths,
        "wb_command": args.wb_command,
        "matlab_log": args.matlab_log,
    }
    missing = [name for name, value in required.items() if value in (None, "")]
    if missing:
        raise ValueError(f"Missing ESTT materialization arguments: {', '.join(missing)}")

    out = args.output_dir.resolve()
    labels_dir = out / "labels"
    labels_dir.mkdir(parents=True, exist_ok=True)
    left = read_labels(out / "lh_labels.txt", subject=subject, hemi="L")
    right = read_labels(out / "rh_labels.txt", subject=subject, hemi="R")
    bundle = load_kong2022_bundle(Path(args.kong2022_dlabel).resolve())
    colors = load_network_colors()

    label_table = labels_dir / "kong2022_label_table.txt"
    write_label_table(label_table, bundle["roi_registry"], colors)
    left_metric = labels_dir / f"sub-{subject}_atlas-kong2022_areal_mshbm.L.func.gii"
    right_metric = labels_dir / f"sub-{subject}_atlas-kong2022_areal_mshbm.R.func.gii"
    left_label = labels_dir / f"sub-{subject}_atlas-kong2022_areal_mshbm.L.label.gii"
    right_label = labels_dir / f"sub-{subject}_atlas-kong2022_areal_mshbm.R.label.gii"
    dlabel = out / f"sub-{subject}_atlas-kong2022_areal_mshbm.dlabel.nii"

    nib.save(make_metric_gifti(left), str(left_metric))
    nib.save(make_metric_gifti(right), str(right_metric))
    run_workbench(
        Path(args.wb_command).resolve(),
        ["-metric-label-import", str(left_metric), str(label_table), str(left_label), "-discard-others"],
    )
    run_workbench(
        Path(args.wb_command).resolve(),
        ["-metric-label-import", str(right_metric), str(label_table), str(right_label), "-discard-others"],
    )
    run_workbench(
        Path(args.wb_command).resolve(),
        ["-cifti-create-label", str(dlabel), "-left-label", str(left_label), "-right-label", str(right_label)],
    )

    outputs = {
        "left_metric": str(left_metric),
        "right_metric": str(right_metric),
        "left_label": str(left_label),
        "right_label": str(right_label),
        "dlabel": str(dlabel),
        "label_table_txt": str(label_table),
    }
    return outputs, label_stats(left, right)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Write provenance for a completed Kong2022 ArealMSHBM subject branch."
    )
    parser.add_argument("--input-manifest", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--provenance", required=True, type=Path)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--branch", default="")
    parser.add_argument("--subject", required=True)
    parser.add_argument("--job-id", required=True)
    parser.add_argument("--task-id", required=True)
    parser.add_argument("--host", required=True)
    parser.add_argument("--matlab-script", required=True)
    parser.add_argument("--cbig-code-dir", required=True)
    parser.add_argument("--group-prior", required=True, type=Path)
    parser.add_argument("--wb-command", required=True)
    parser.add_argument("--matlab-mode", required=True)
    parser.add_argument("--overwrite", required=True)
    parser.add_argument("--run-start-utc", required=True)
    parser.add_argument("--run-end-utc", required=True)
    parser.add_argument("--runtime-seconds", required=True, type=float)
    parser.add_argument("--materialize-label-assets", action="store_true")
    parser.add_argument("--kong2022-dlabel", type=Path)
    parser.add_argument("--left-inflated", type=Path)
    parser.add_argument("--right-inflated", type=Path)
    parser.add_argument("--left-midthickness", type=Path)
    parser.add_argument("--right-midthickness", type=Path)
    parser.add_argument("--sulc-dscalar", type=Path)
    parser.add_argument("--run-lengths", nargs=4, type=int)
    parser.add_argument("--matlab-log", type=Path)
    args = parser.parse_args()

    manifest = read_json(args.input_manifest)
    runs = manifest.get("runs", [])
    input_ciftis = [row["dtseries"] for row in runs]
    timepoints = [int(row["timepoints"]) for row in runs]
    source_dtseries = manifest.get("source_dtseries")
    subject = args.subject.removeprefix("sub-")

    asset_outputs: dict[str, str] = {}
    asset_label_stats: dict[str, object] | None = None
    if args.materialize_label_assets:
        asset_outputs, asset_label_stats = materialize_label_assets(args, subject)

    payload = {
        "schema_version": 1,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset": args.dataset,
        "branch": args.branch or None,
        "subject": args.subject,
        "input_cifti": source_dtseries or (input_ciftis[0] if len(input_ciftis) == 1 else input_ciftis),
        "input_ciftis": input_ciftis,
        "output_dir": str(args.output_dir),
        "labels": {
            "lh": str(args.output_dir / "lh_labels.txt"),
            "rh": str(args.output_dir / "rh_labels.txt"),
        },
        "qsub": {
            "job_id": args.job_id,
            "task_id": args.task_id,
            "host": args.host,
        },
        "matlab": {
            "script": args.matlab_script,
            "mode": args.matlab_mode,
        },
        "parameters": {
            "target_mesh": "fs_LR_32k",
            "model": "gMSHBM",
            "num_ROIs": "400",
            "w": "30",
            "c": "30",
            "beta": "50",
            "group_prior": str(args.group_prior),
            "cbig_code_dir": args.cbig_code_dir,
            "wb_command": args.wb_command,
            "overwrite": args.overwrite,
        },
        "timepoints": timepoints[0] if len(timepoints) == 1 else timepoints,
        "runs": runs,
        "run_list": manifest.get("run_list"),
        "censor_list": manifest.get("censor_list"),
        "timing": {
            "run_start_utc": args.run_start_utc,
            "run_end_utc": args.run_end_utc,
            "runtime_seconds": args.runtime_seconds,
        },
    }
    if manifest.get("variant") is not None:
        payload["variant"] = manifest["variant"]
    if args.dataset in ("pdc", "banda"):
        for key in ("source_subject", "cohort_manifest", "cohort_manifest_sha256",
                    "source_manifest", "source_manifest_sha256", "session_policy"):
            payload[key] = manifest[key]
        for key in ("base_subject_id", "session_number"):
            if key in manifest:
                payload[key] = manifest[key]
        payload["label_stats"] = label_stats(
            read_labels(args.output_dir / "lh_labels.txt", subject=subject, hemi="L"),
            read_labels(args.output_dir / "rh_labels.txt", subject=subject, hemi="R"),
        )
    if asset_outputs:
        payload["assets"] = asset_outputs
        payload["label_stats"] = asset_label_stats

    args.provenance.parent.mkdir(parents=True, exist_ok=True)
    args.provenance.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    if args.materialize_label_assets:
        params = {
            "model": "gMSHBM",
            "num_rois": 400,
            "w": 30,
            "c": 30,
            "beta": 50,
            "num_sessions": 4,
            "run_lengths": [int(x) for x in args.run_lengths],
        }
        input_paths = ([Path(source_dtseries).resolve()] if source_dtseries else []) + [
            Path(row["dtseries"]).resolve() for row in runs
        ] + [
            Path(args.left_inflated).resolve(),
            Path(args.right_inflated).resolve(),
            Path(args.left_midthickness).resolve(),
            Path(args.right_midthickness).resolve(),
            Path(args.sulc_dscalar).resolve(),
            Path(args.group_prior).resolve(),
            Path(args.kong2022_dlabel).resolve(),
        ]
        stage_outputs = [
            str(args.output_dir / "summary.json"),
            str(args.output_dir / "stage_manifest.json"),
            str(args.provenance),
        ]
        if args.matlab_log is not None:
            stage_outputs.append(str(args.matlab_log))
        stage_outputs.extend(asset_outputs.values())
        timestamp = datetime.now(timezone.utc).isoformat()
        stage = {
            "stage": "kong2022_areal_mshbm_individualize",
            "status": "done",
            "timestamp_utc": timestamp,
            "variant": manifest["variant"] if "variant" in manifest else None,
            "params": params,
            "inputs": [file_meta(path) for path in input_paths],
            "outputs": stage_outputs,
            "qsub": {"job_id": str(args.job_id), "task_id": str(args.task_id), "host": str(args.host)},
        }
        summary = {
            "subject": subject,
            "atlas_slug": "kong2022_areal_mshbm",
            "status": "done",
            "timestamp_utc": timestamp,
            "variant": manifest["variant"] if "variant" in manifest else None,
            "params": params,
            "inputs": {
                "dtseries": source_dtseries or input_ciftis,
                "session_dtseries": input_ciftis,
                "left_inflated": str(args.left_inflated.resolve()),
                "right_inflated": str(args.right_inflated.resolve()),
                "left_midthickness": str(args.left_midthickness.resolve()),
                "right_midthickness": str(args.right_midthickness.resolve()),
                "sulc_dscalar": str(args.sulc_dscalar.resolve()),
                "group_prior": str(args.group_prior.resolve()),
                "kong2022_dlabel": str(args.kong2022_dlabel.resolve()),
            },
            "outputs": {
                "lh_labels_txt": str(args.output_dir / "lh_labels.txt"),
                "rh_labels_txt": str(args.output_dir / "rh_labels.txt"),
                **asset_outputs,
            },
            "label_stats": asset_label_stats,
            "qsub": {"job_id": str(args.job_id), "task_id": str(args.task_id), "host": str(args.host)},
        }
        (args.output_dir / "summary.json").write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        (args.output_dir / "stage_manifest.json").write_text(
            json.dumps(stage, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
