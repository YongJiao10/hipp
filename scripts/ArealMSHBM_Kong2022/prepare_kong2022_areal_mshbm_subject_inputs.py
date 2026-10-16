#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import nibabel as nib


DATASET_SPECS = {
    "jj_tms": {
        "subject_prefix": "sub-",
        "relative_globs": ["func/*_task-rest_run-*_space-fsLR_den-91k_desc-denoised_bold.dtseries.nii"],
    },
    "stanford": {
        "subject_prefix": "sub-",
        "relative_globs": ["func/*_task-rest_space-fsLR_den-91k_desc-denoised_bold.dtseries.nii"],
    },
    "iowa7t": {
        "subject_prefix": "sub-",
        "relative_globs": ["func/*_task-rest_run-concat_space-fsLR_den-91k_desc-denoised_bold.dtseries.nii"],
        "branch_relative_globs": {
            "multiband": [
                "func/{subject}_task-rest_dir-concat_space-fsLR_den-91k_desc-denoised_bold.dtseries.nii"
            ],
            "multi_echo": [
                "func/{subject}_task-rest_run-concat_space-fsLR_den-32k_desc-MEICA_XCPD36P_denoised_bold.dtseries.nii"
            ],
            "noxcpd_multiband": [
                "func/{subject}_task-rest_dir-concat_space-fsLR_den-91k_desc-NoXCPD_bold.dtseries.nii"
            ],
            "noxcpd_multi_echo": [
                "func/{subject}_task-rest_run-concat_space-fsLR_den-91k_desc-MEICA_NoXCPD_bold.dtseries.nii"
            ],
        },
    },
    "iowa_hcp500": {
        "subject_prefix": "",
        "relative_globs": [
            "hcp/{subject_id}/MNINonLinear/Results/REST/"
            "REST_Atlas_MSMAll_hp0_clean_rclean.dtseries.nii"
        ],
    },
}

HCP_VARIANT = "MSMAll_hp0_clean"
HCP_RUNS = ("rfMRI_REST1_AP", "rfMRI_REST1_PA", "rfMRI_REST2_AP", "rfMRI_REST2_PA")
HCP_METADATA = {
    "pdc": ("datastructure_manifest.txt", "associated_file"),
    "banda": ("datastructure_manifest.txt", "associated_file"),
}


def normalize_dataset(value: str) -> str:
    return value.strip().lower().replace("-", "_")


def natural_key(path: Path) -> tuple:
    parts = re.split(r"(\d+)", path.name)
    return tuple(int(part) if part.isdigit() else part for part in parts)


def dtseries_n_timepoints(path: Path) -> int:
    img = nib.load(str(path))
    if len(img.shape) != 2:
        raise ValueError(f"Expected 2D CIFTI dtseries, got shape={img.shape}: {path}")
    return int(img.shape[0])


def write_ones(path: Path, count: int) -> None:
    with path.open("w", encoding="utf-8") as f:
        for _ in range(count):
            f.write("1\n")


def hcp_identity(dataset: str, source_subject: str) -> dict:
    if dataset == "pdc":
        match = re.fullmatch(r"([a-z]\d{4})_(\d{2})_MR", source_subject)
        if not match:
            raise ValueError(f"Invalid PDC session: {source_subject}")
        base, session = match.groups()
        return {"subject": f"sub-{base}s{session}", "source_subject": source_subject,
                "base_subject_id": base, "session_number": session}
    if dataset == "banda" and re.fullmatch(r"BANDA\d{3}_MR", source_subject):
        return {"subject": f"sub-{source_subject.removesuffix('_MR')}",
                "source_subject": source_subject}
    raise ValueError(f"Invalid {dataset} source identity: {source_subject}")


def hcp_run_path(root: Path, source_subject: str, run: str) -> Path:
    return root / source_subject / "MNINonLinear" / "Results" / run / f"{run}_Atlas_{HCP_VARIANT}.dtseries.nii"


def check_hcp_file(row: dict) -> None:
    path = Path(row["dtseries"])
    stat = path.stat()
    if not path.is_file() or stat.st_size <= 0:
        raise ValueError(f"Missing/empty canonical {HCP_VARIANT} input: {path}")
    if stat.st_size != row["size_bytes"] or stat.st_mtime_ns != row["mtime_ns"]:
        raise ValueError(f"Stale {HCP_VARIANT} declaration: {path}; size or mtime changed")


def build_hcp_manifest(dataset: str, root: Path) -> dict:
    metadata_name, column = HCP_METADATA[dataset]
    source = root.parent / "metadata" / metadata_name
    source_bytes = source.read_bytes()
    by_subject: dict[str, dict] = {}
    seen = set()
    concatenated = set()
    reader = csv.DictReader(source_bytes.decode("utf-8-sig").splitlines(), delimiter="\t")
    for row in reader:
        raw = row[column]
        match = re.search(r"(?:^|/)([^/]+)/MNINonLinear/Results/(rfMRI_REST[^/]*)/([^/]+\.dtseries\.nii)$", raw)
        if not match:
            continue
        source_subject, run, filename = match.groups()
        if filename != f"{run}_Atlas_{HCP_VARIANT}.dtseries.nii" or run not in (*HCP_RUNS, "rfMRI_REST"):
            raise ValueError(f"Undeclared resting CIFTI variant/run in {source}: {raw}")
        if run == "rfMRI_REST":
            concatenated.add(source_subject)
            continue
        path = hcp_run_path(root, source_subject, run)
        if str(path) in seen:
            raise ValueError(f"Duplicate canonical {HCP_VARIANT} declaration in {source}: {path}")
        seen.add(str(path))
        stat = path.stat()
        if not path.is_file() or stat.st_size <= 0:
            raise ValueError(f"Missing/empty canonical {HCP_VARIANT} input: {path}")
        entry = by_subject.setdefault(source_subject, {**hcp_identity(dataset, source_subject), "runs": []})
        entry["runs"].append({"run_id": run, "dtseries": str(path),
                              "size_bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns})
    if not by_subject:
        raise ValueError(f"No declared run-level {HCP_VARIANT} inputs in {source}")
    subject_pattern = r"[a-z]\d{4}_\d{2}_MR" if dataset == "pdc" else r"BANDA\d{3}_MR"
    imaging_subjects = {p.name for p in root.iterdir()
                        if re.fullmatch(subject_pattern, p.name) and p.is_dir()}
    missing_dirs = set(by_subject) - imaging_subjects
    if missing_dirs:
        raise ValueError(f"Metadata subjects missing from {root}: {sorted(missing_dirs)}")
    excluded = [{**hcp_identity(dataset, sid), "reason": "no_run_level_rest_in_source_metadata"}
                for sid in sorted(imaging_subjects - set(by_subject))]
    subjects = []
    for sid in sorted(by_subject):
        entry = by_subject[sid]
        entry["runs"].sort(key=lambda item: HCP_RUNS.index(item["run_id"]))
        subjects.append(entry)
    return {
        "schema_version": 1, "dataset": dataset, "variant": HCP_VARIANT,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "data_root": str(root), "source_manifest": str(source),
        "source_manifest_sha256": hashlib.sha256(source_bytes).hexdigest(),
        "session_policy": "each_metadata_declared_rest_run_is_one_Kong2022_session",
        "concatenated_rest_policy": "excluded_to_avoid_duplicate_frames",
        "source_concatenated_rest_subject_count": len(concatenated),
        "subjects": subjects, "excluded_subjects": excluded,
        "subject_count": len(subjects), "run_count": len(seen),
        "runs_per_subject": dict(sorted(Counter(len(item["runs"]) for item in subjects).items())),
    }


def load_hcp_manifest(path: Path, dataset: str, root: Path) -> dict:
    manifest = json.loads(path.read_text())
    expected_values = (
        ("schema_version", 1), ("dataset", dataset), ("variant", HCP_VARIANT),
        ("data_root", str(root)),
        ("source_manifest", str(root.parent / "metadata" / HCP_METADATA[dataset][0])),
        ("session_policy", "each_metadata_declared_rest_run_is_one_Kong2022_session"),
        ("concatenated_rest_policy", "excluded_to_avoid_duplicate_frames"),
    )
    for key, expected in expected_values:
        if manifest.get(key) != expected:
            raise ValueError(f"Cohort manifest mismatch at {path}: {key}={manifest.get(key)!r}, expected {expected!r}")
    if not re.fullmatch(r"[a-f0-9]{64}", manifest.get("source_manifest_sha256", "")):
        raise ValueError(f"Missing/malformed source metadata hash in {path}")
    subjects = manifest.get("subjects", [])
    seen_subjects, seen_paths = set(), set()
    for entry in subjects:
        expected_identity = hcp_identity(dataset, entry["source_subject"])
        if any(entry.get(key) != value for key, value in expected_identity.items()):
            raise ValueError(f"Identity mismatch in {path}: {entry}")
        subject = entry["subject"]
        if subject in seen_subjects:
            raise ValueError(f"Duplicate subject in {path}: {subject}")
        seen_subjects.add(subject)
        runs = entry["runs"]
        if not runs or len({item["run_id"] for item in runs}) != len(runs):
            raise ValueError(f"Empty/duplicate run declarations in {path}: {subject}")
        if any(item["run_id"] not in HCP_RUNS for item in runs):
            raise ValueError(f"Unsupported rest run in {path}: {subject}")
        if [item["run_id"] for item in runs] != sorted((item["run_id"] for item in runs), key=HCP_RUNS.index):
            raise ValueError(f"Run order mismatch in {path}: {subject}")
        for item in runs:
            expected = str(hcp_run_path(root, entry["source_subject"], item["run_id"]))
            if item["dtseries"] != expected or expected in seen_paths:
                raise ValueError(f"Canonical {HCP_VARIANT} path mismatch/duplicate in {path}: {item['dtseries']}; expected {expected}")
            if item["size_bytes"] <= 0 or item["mtime_ns"] <= 0:
                raise ValueError(f"Invalid input fingerprint in {path}: {expected}")
            seen_paths.add(expected)
    if not subjects or len(subjects) != manifest["subject_count"] or len(seen_paths) != manifest["run_count"]:
        raise ValueError(f"Cohort counts mismatch in {path}")
    return manifest


def prepare_hcp_subject(manifest_path: Path, dataset: str, root: Path, subject: str, output: Path) -> None:
    manifest = load_hcp_manifest(manifest_path, dataset, root)
    subject = f"sub-{subject.removeprefix('sub-')}"
    matches = [entry for entry in manifest["subjects"] if entry["subject"] == subject]
    if len(matches) != 1:
        raise ValueError(f"Expected one declaration for {subject} in {manifest_path}, got {len(matches)}")
    rows = []
    for index, declared in enumerate(matches[0]["runs"], 1):
        check_hcp_file(declared)
        img = nib.load(declared["dtseries"])
        if not isinstance(img, nib.Cifti2Image) or len(img.shape) != 2:
            raise ValueError(f"Not a 2D CIFTI: {declared['dtseries']}")
        series, brain = img.header.get_axis(0), img.header.get_axis(1)
        if not isinstance(series, nib.cifti2.SeriesAxis) or not isinstance(brain, nib.cifti2.BrainModelAxis) or img.shape[0] < 1:
            raise ValueError(f"Invalid dtseries axes: {declared['dtseries']}")
        for hemi in ("LEFT", "RIGHT"):
            name = f"CIFTI_STRUCTURE_CORTEX_{hemi}"
            if brain.nvertices.get(name) != 32492 or not (brain.name == name).any():
                raise ValueError(f"Expected consumed fsLR32k {name} vertices in {declared['dtseries']}; got {brain.nvertices}")
        rows.append({**declared, "run_index": index, "timepoints": int(img.shape[0]),
                     "density_evidence": {"asset": declared["dtseries"], "cortex_vertices_per_hemisphere": 32492}})
    inputs = output / "inputs"
    censors = inputs / "censors"
    censors.mkdir(parents=True, exist_ok=True)
    for row in rows:
        path = censors / f"{subject}_{row['run_id']}_censor.txt"
        path.write_text("1\n" * row["timepoints"])
        row["censor"] = str(path)
    run_list, censor_list = inputs / "task_rest_dtseries.txt", inputs / "task_rest_censors.txt"
    run_list.write_text("".join(row["dtseries"] + "\n" for row in rows))
    censor_list.write_text("".join(row["censor"] + "\n" for row in rows))
    entry = matches[0]
    payload = {**entry, "dataset": dataset, "variant": HCP_VARIANT, "data_root": str(root),
               "subject_dir": str(root / entry["source_subject"]), "runs": rows,
               "branch": None, "run_count": len(rows), "run_list": str(run_list), "censor_list": str(censor_list),
               "cohort_manifest": str(manifest_path),
               "cohort_manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
               "source_manifest": manifest["source_manifest"],
               "source_manifest_sha256": manifest["source_manifest_sha256"],
               "session_policy": manifest["session_policy"]}
    (inputs / "input_manifest.json").write_text(json.dumps(payload, indent=2) + "\n")
    print(f"Prepared {subject}: {len(rows)} declared {HCP_VARIANT} runs, timepoints={[row['timepoints'] for row in rows]}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Prepare task-rest dtseries inputs for one Kong2022 ArealMSHBM SGE task."
    )
    parser.add_argument(
        "--action",
        choices=("prepare", "build-cohort", "list-cohort", "prepare-cohort"),
        default="prepare",
        help="Operation. Default: prepare one subject using the existing dataset pattern contract.",
    )
    parser.add_argument(
        "--dataset",
        default="jj_tms",
        help="Dataset preset: jj_tms, stanford, iowa7t, iowa_hcp500, pdc, or banda.",
    )
    parser.add_argument("--subject", help="Subject ID with or without sub- prefix.")
    parser.add_argument("--data-root", type=Path, help="Dataset root (imaging/ for PDC/BANDA).")
    parser.add_argument("--subject-output-dir", type=Path)
    parser.add_argument("--cohort-manifest", type=Path)
    parser.add_argument(
        "--branch",
        default="",
        help="Dataset branch selector. For iowa7t, use multiband or multi_echo.",
    )
    parser.add_argument(
        "--dtseries-glob",
        action="append",
        default=None,
        help="Override func/ dtseries glob. Repeat to allow multiple patterns.",
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    dataset = normalize_dataset(args.dataset)
    if args.action in ("build-cohort", "list-cohort", "prepare-cohort"):
        if dataset not in HCP_METADATA:
            parser.error(f"--action {args.action} requires --dataset pdc or banda")
        if args.data_root is None or not args.data_root.is_absolute():
            parser.error(f"--action {args.action} requires an absolute --data-root")
        if args.cohort_manifest is None:
            parser.error(f"--action {args.action} requires --cohort-manifest")
        if args.action == "build-cohort":
            if args.cohort_manifest.exists():
                parser.error(f"Refusing to replace existing cohort manifest: {args.cohort_manifest}")
            cohort = build_hcp_manifest(dataset, args.data_root)
            args.cohort_manifest.parent.mkdir(parents=True, exist_ok=True)
            args.cohort_manifest.write_text(json.dumps(cohort, indent=2) + "\n")
            print(json.dumps({key: cohort[key] for key in ("dataset", "subject_count", "run_count", "runs_per_subject", "excluded_subjects")}))
            return 0
        cohort = load_hcp_manifest(args.cohort_manifest, dataset, args.data_root)
        if args.action == "list-cohort":
            print("\n".join(entry["subject"].removeprefix("sub-") for entry in cohort["subjects"]))
            return 0
        if not args.subject or args.subject_output_dir is None:
            parser.error("prepare-cohort requires --subject and --subject-output-dir")
        prepare_hcp_subject(args.cohort_manifest, dataset, args.data_root, args.subject,
                            args.subject_output_dir)
        return 0

    if dataset not in DATASET_SPECS:
        known = ", ".join(sorted(DATASET_SPECS))
        raise SystemExit(f"Unsupported dataset: {args.dataset}. Expected one of: {known}")
    spec = DATASET_SPECS[dataset]
    if args.subject is None or args.data_root is None or args.subject_output_dir is None:
        parser.error("prepare requires --subject, --data-root, and --subject-output-dir")
    branch = args.branch.strip()
    branch_globs = spec.get("branch_relative_globs", {})
    if branch:
        if branch not in branch_globs:
            known = ", ".join(sorted(branch_globs)) or "<none>"
            raise SystemExit(f"Unsupported branch for dataset {dataset}: {branch}. Expected one of: {known}")
        dtseries_globs = args.dtseries_glob or branch_globs[branch]
    else:
        dtseries_globs = args.dtseries_glob or spec["relative_globs"]

    subject_id = args.subject.removeprefix("sub-")
    subject = f"sub-{subject_id}"
    subject_dir = args.data_root / f"{spec['subject_prefix']}{subject_id}"
    if not subject_dir.is_dir():
        raise SystemExit(f"Missing subject directory: {subject_dir}")

    dtseries_by_path = {}
    for pattern in dtseries_globs:
        resolved_pattern = pattern.format(subject_id=subject_id, subject=subject)
        for path in subject_dir.glob(resolved_pattern):
            dtseries_by_path[path] = path
    dtseries = sorted(dtseries_by_path, key=natural_key)
    if not dtseries:
        patterns = ", ".join(dtseries_globs)
        raise SystemExit(f"No task-rest fsLR dtseries found under {subject_dir}; patterns={patterns}")

    inputs_dir = args.subject_output_dir / "inputs"
    censor_dir = inputs_dir / "censors"
    inputs_dir.mkdir(parents=True, exist_ok=True)
    censor_dir.mkdir(parents=True, exist_ok=True)

    run_list_path = inputs_dir / "task_rest_dtseries.txt"
    censor_list_path = inputs_dir / "task_rest_censors.txt"
    rows = []
    censor_paths = []
    for idx, dt_path in enumerate(dtseries, start=1):
        n_tp = dtseries_n_timepoints(dt_path)
        censor_path = censor_dir / f"{subject}_task-rest_run-{idx:02d}_censor.txt"
        if args.overwrite or not censor_path.exists():
            write_ones(censor_path, n_tp)
        rows.append({"run_index": idx, "dtseries": str(dt_path), "timepoints": n_tp, "censor": str(censor_path)})
        censor_paths.append(censor_path)

    run_list_path.write_text("\n".join(str(p) for p in dtseries) + "\n", encoding="utf-8")
    censor_list_path.write_text("\n".join(str(p) for p in censor_paths) + "\n", encoding="utf-8")

    manifest = {
        "dataset": dataset,
        "branch": branch or None,
        "subject": subject,
        "data_root": str(args.data_root),
        "dtseries_globs": dtseries_globs,
        "subject_dir": str(subject_dir),
        "run_count": len(rows),
        "runs": rows,
        "run_list": str(run_list_path),
        "censor_list": str(censor_list_path),
    }
    manifest_path = inputs_dir / "input_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
