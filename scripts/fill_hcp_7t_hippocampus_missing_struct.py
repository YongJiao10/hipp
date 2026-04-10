#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import gzip
import json
import shlex
import shutil
import subprocess
from pathlib import Path

SEED_FILES = ["DG_CA4.nii.gz", "CA2_3.nii.gz", "CA1.nii.gz", "SC.nii.gz"]
STRUCT_MEMBERS = {
    "T1w_acpc_dc_restore.nii.gz": "{sid}/T1w/T1w_acpc_dc_restore.nii.gz",
    "T2w_acpc_dc_restore.nii.gz": "{sid}/T1w/T2w_acpc_dc_restore.nii.gz",
    "acpc_dc2standard.nii.gz": "{sid}/MNINonLinear/xfms/acpc_dc2standard.nii.gz",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Backfill missing structural subjects into HCP_7T_Hippocampus root.")
    parser.add_argument("--project-root", default="/Volumes/yojiao/HCP_7T_Hippocampus")
    parser.add_argument("--remote-host", default="yojiao@192.168.0.113")
    parser.add_argument(
        "--remote-struct-root",
        default="/Volumes/Elements/HCP-YA-2025/Structural Preprocessed Recommended for 3T and 7T",
    )
    parser.add_argument("--subject-id", action="append", default=None, help="Optional subject id to backfill; repeatable")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def is_valid_gzip(path: Path) -> bool:
    if not path.exists() or path.stat().st_size == 0:
        return False
    try:
        with gzip.open(path, "rb") as stream:
            stream.read(1)
        return True
    except Exception:
        return False


def read_master_rows(master_tsv: Path) -> list[dict[str, str]]:
    with master_tsv.open(newline="") as stream:
        return list(csv.DictReader(stream, delimiter="\t"))


def infer_missing_subjects(rows: list[dict[str, str]]) -> list[str]:
    missing = []
    for row in rows:
        if row.get("struct_zip_status", "").strip() != "complete":
            subject_id = row.get("subject_id", "").strip()
            if subject_id:
                missing.append(subject_id)
    return sorted(set(missing))


def ensure_remote_zip_exists(remote_host: str, remote_zip: str) -> None:
    command = ["ssh", remote_host, f"test -f {shlex.quote(remote_zip)}"]
    subprocess.run(command, check=True)


def ensure_from_remote_zip(remote_host: str, remote_zip: str, member: str, dst: Path) -> None:
    if is_valid_gzip(dst):
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_suffix(dst.suffix + ".tmp")
    if tmp.exists():
        tmp.unlink()
    remote_cmd = f"unzip -p {shlex.quote(remote_zip)} {shlex.quote(member)}"
    with tmp.open("wb") as out:
        proc = subprocess.run(["ssh", remote_host, remote_cmd], stdout=out, stderr=subprocess.PIPE)
    if proc.returncode != 0:
        if tmp.exists():
            tmp.unlink()
        stderr = proc.stderr.decode("utf-8", errors="replace")
        raise RuntimeError(f"Failed to stream {member} from {remote_zip}:\n{stderr}")
    if not is_valid_gzip(tmp):
        if tmp.exists():
            tmp.unlink()
        raise RuntimeError(f"Extracted invalid gzip: {tmp}")
    tmp.replace(dst)


def clean_notes(notes: str) -> str:
    tokens = [token.strip() for token in notes.split(";") if token.strip()]
    filtered = [
        token
        for token in tokens
        if token not in {"rest7t_zip_status=missing", "rest7t_zip_status=partial"}
    ]
    return " ; ".join(filtered)


def stage_subject(
    *,
    project_root: Path,
    remote_host: str,
    remote_struct_root: str,
    subject_id: str,
    dry_run: bool,
) -> dict[str, object]:
    remote_zip = f"{remote_struct_root.rstrip('/')}/{subject_id}_StructuralRecommended.zip"
    subject_dir = project_root / "subjects" / subject_id
    source_dir = subject_dir / "source"
    labels_dir = subject_dir / "labels_native"
    manifest_path = subject_dir / "subject_manifest.json"

    if dry_run:
        return {
            "subject_id": subject_id,
            "remote_zip": remote_zip,
            "subject_dir": str(subject_dir),
            "dry_run": True,
        }

    ensure_remote_zip_exists(remote_host, remote_zip)
    source_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)

    for out_name, member_template in STRUCT_MEMBERS.items():
        ensure_from_remote_zip(
            remote_host=remote_host,
            remote_zip=remote_zip,
            member=member_template.format(sid=subject_id),
            dst=source_dir / out_name,
        )

    manifest = {
        "subject_id": subject_id,
        "mode": "struct_only_backfill_from_remote",
        "struct_zip": remote_zip,
        "source_files": sorted(path.name for path in source_dir.glob("*.nii.gz")),
        "labels_native_available": sorted(path.name for path in labels_dir.glob("*.nii.gz")),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest


def rebuild_ashs_subjects(project_root: Path, dry_run: bool) -> list[str]:
    subjects_root = project_root / "subjects"
    staged = []
    for subject_dir in sorted(path for path in subjects_root.iterdir() if path.is_dir() and path.name.isdigit()):
        source_dir = subject_dir / "source"
        needed = [source_dir / name for name in STRUCT_MEMBERS]
        if all(path.exists() for path in needed):
            staged.append(subject_dir.name)
    if not dry_run:
        (project_root / "master" / "ashs_subjects.txt").write_text(
            "".join(f"{subject_id}\n" for subject_id in staged),
            encoding="utf-8",
        )
    return staged


def rewrite_master_files(project_root: Path, rows: list[dict[str, str]], staged_subjects: list[str], dry_run: bool) -> dict[str, object]:
    staged_set = set(staged_subjects)
    for row in rows:
        subject_id = row["subject_id"].strip()
        if subject_id in staged_set:
            row["struct_zip_status"] = "complete"
            row["rest7t_zip_status"] = "complete"
            if row.get("ashs_status", "").strip() != "complete":
                row["ashs_status"] = "missing"
            if row.get("seed_status", "").strip() != "done":
                row["seed_status"] = "pending_ashs"
            row["notes"] = clean_notes(row.get("notes", ""))

    summary: dict[str, object] = {"official_total": len(rows)}
    status_field = {
        "struct": "struct_zip_status",
        "rest": "rest7t_zip_status",
        "ashs": "ashs_status",
        "seed": "seed_status",
    }
    for prefix, field_name in status_field.items():
        for row in rows:
            key = f"{prefix}_{row[field_name]}"
            summary[key] = int(summary.get(key, 0)) + 1

    labels_complete = []
    subjects_root = project_root / "subjects"
    for subject_id in staged_subjects:
        labels_dir = subjects_root / subject_id / "labels_native"
        if labels_dir.exists() and all((labels_dir / name).exists() for name in SEED_FILES):
            labels_complete.append(subject_id)
    bundle_summary = {
        "official_total": len(rows),
        "struct_complete_staged": len(staged_subjects),
        "struct_missing": len(rows) - len(staged_subjects),
        "struct_missing_subjects": sorted({row["subject_id"] for row in rows} - set(staged_subjects)),
        "label_subjects_count": len(labels_complete),
        "label_subjects": labels_complete,
        "out_root": str(project_root),
    }

    if not dry_run:
        master_tsv = project_root / "master" / "rest7t_175_master.tsv"
        with master_tsv.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(
                stream,
                fieldnames=[
                    "subject_id",
                    "official_rest7t",
                    "struct_zip_status",
                    "rest7t_zip_status",
                    "ashs_status",
                    "seed_status",
                    "notes",
                ],
                delimiter="\t",
            )
            writer.writeheader()
            writer.writerows(rows)
        (project_root / "master" / "rest7t_175_summary.json").write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        (project_root / "master" / "bundle_summary.json").write_text(
            json.dumps(bundle_summary, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    return {"rest_summary": summary, "bundle_summary": bundle_summary}


def main() -> None:
    args = parse_args()
    project_root = Path(args.project_root).expanduser().resolve()
    master_tsv = project_root / "master" / "rest7t_175_master.tsv"
    if not master_tsv.exists():
        raise FileNotFoundError(f"Missing master TSV: {master_tsv}")

    rows = read_master_rows(master_tsv)
    subject_ids = sorted(set(args.subject_id)) if args.subject_id else infer_missing_subjects(rows)
    if not subject_ids:
        print(json.dumps({"project_root": str(project_root), "staged_subjects": [], "message": "Nothing to backfill."}))
        return

    staged_manifests = []
    for subject_id in subject_ids:
        staged_manifests.append(
            stage_subject(
                project_root=project_root,
                remote_host=args.remote_host,
                remote_struct_root=args.remote_struct_root,
                subject_id=subject_id,
                dry_run=args.dry_run,
            )
        )

    staged_subjects = rebuild_ashs_subjects(project_root, dry_run=args.dry_run)
    summaries = rewrite_master_files(project_root, rows, staged_subjects, dry_run=args.dry_run)

    payload = {
        "project_root": str(project_root),
        "dry_run": bool(args.dry_run),
        "requested_subjects": subject_ids,
        "staged_subjects": staged_subjects,
        "staged_count": len(subject_ids),
        "stage_records": staged_manifests,
        **summaries,
    }
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
