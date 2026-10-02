#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.core_workflow.lib.io_utils import load_run_artifact_manifest
from scripts.core_workflow.lib.task_design import TASK_RUN_LABELS
from scripts.core_workflow.lib.task_manifests import (
    build_task_run_artifact_manifest,
    find_task_run_files,
    task_run_labels,
    task_run_manifest_path,
)


def _fake_staged_tree(root: Path, subject: str = "115017") -> Path:
    data_root = root / "HCP_tast_3T"
    for label in ("tfMRI_EMOTION_LR", "tfMRI_EMOTION_RL"):
        run_dir = data_root / f"sub-{subject}" / "func" / label
        run_dir.mkdir(parents=True)
        (run_dir / f"{label}_hp0_clean_rclean_tclean.nii.gz").write_bytes(b"fakevol")
        (run_dir / f"{label}_Atlas_MSMAll_hp0_clean_rclean_tclean.dtseries.nii").write_bytes(
            b"fakedt"
        )
        (run_dir / "Movement_Regressors.txt").write_text("0 " * 12 + "\n")
    return data_root


def test_task_run_labels_match_expected_order() -> None:
    assert task_run_labels("emotion") == TASK_RUN_LABELS["emotion"]
    assert task_run_labels("EMOTION") == ("tfMRI_EMOTION_LR", "tfMRI_EMOTION_RL")


def test_manifest_path_layout(tmp_path: Path) -> None:
    path = task_run_manifest_path(tmp_path, subject="sub-115017", task="wm")
    assert path == tmp_path.resolve() / "task-wm" / "sub-115017.json"


def test_find_task_run_files_rejects_ambiguous(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "a_hp0_clean_rclean_tclean.nii.gz").write_bytes(b"x")
    (run_dir / "b_hp0_clean_rclean_tclean.nii.gz").write_bytes(b"x")
    (run_dir / "a_Atlas_MSMAll_hp0_clean_rclean_tclean.dtseries.nii").write_bytes(b"x")
    with pytest.raises(ValueError):
        find_task_run_files(run_dir)


def test_build_and_reload_manifest_roundtrip(tmp_path: Path) -> None:
    data_root = _fake_staged_tree(tmp_path)
    manifest_root = tmp_path / "manifests"
    out = build_task_run_artifact_manifest(
        subject="115017",
        task="emotion",
        task_data_root=data_root,
        manifest_root=manifest_root,
    )
    assert out.is_file()
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert payload["subject"] == "sub-115017"
    assert [run["run_id"] for run in payload["runs"]] == [1, 2]
    records = load_run_artifact_manifest(
        out, subject="115017", required_artifacts=("bold", "dtseries")
    )
    assert [int(record["run_id"]) for record in records] == [1, 2]
    assert all("bold" in record["artifacts"] for record in records)


def test_build_manifest_refuses_overwrite_without_flag(tmp_path: Path) -> None:
    data_root = _fake_staged_tree(tmp_path)
    manifest_root = tmp_path / "manifests"
    build_task_run_artifact_manifest(
        subject="115017",
        task="emotion",
        task_data_root=data_root,
        manifest_root=manifest_root,
    )
    with pytest.raises(FileExistsError):
        build_task_run_artifact_manifest(
            subject="115017",
            task="emotion",
            task_data_root=data_root,
            manifest_root=manifest_root,
        )


def test_build_manifest_fails_closed_on_missing_run(tmp_path: Path) -> None:
    data_root = _fake_staged_tree(tmp_path)
    with pytest.raises(FileNotFoundError):
        build_task_run_artifact_manifest(
            subject="999999",
            task="emotion",
            task_data_root=data_root,
            manifest_root=tmp_path / "manifests",
        )


def _manifest_runs() -> list[dict[str, object]]:
    return [
        {"run_id": 1, "run_dir": Path("/data/tfMRI_EMOTION_LR"),
         "bold": Path("/data/tfMRI_EMOTION_LR/bold.nii.gz"),
         "dtseries": Path("/data/tfMRI_EMOTION_LR/data.dtseries.nii")},
        {"run_id": 2, "run_dir": Path("/data/tfMRI_EMOTION_RL"),
         "bold": Path("/data/tfMRI_EMOTION_RL/bold.nii.gz"),
         "dtseries": Path("/data/tfMRI_EMOTION_RL/data.dtseries.nii")},
    ]


def test_verify_cached_runs_match(tmp_path: Path) -> None:
    from scripts.core_workflow.lib.task_manifests import verify_cached_runs_against_manifest

    receipt = tmp_path / "summary.json"
    receipt.write_text("{}")
    out = verify_cached_runs_against_manifest(
        subject="001", task="emotion", cached_kind="surface sampling",
        cached_summary_path=receipt,
        cached_run_ids=[1, 2], cached_run_lengths=[176, 176],
        cached_source_paths=[
            "/data/tfMRI_EMOTION_LR/bold.nii.gz",
            "/data/tfMRI_EMOTION_RL/bold.nii.gz",
        ],
        manifest_runs=_manifest_runs(), manifest_source_key="bold",
        manifest_lengths=[176, 176],
    )
    assert out["match"] is True
    assert out["run_ids"] == [1, 2]


def test_verify_cached_runs_fails_on_order_swap(tmp_path: Path) -> None:
    from scripts.core_workflow.lib.task_manifests import verify_cached_runs_against_manifest

    receipt = tmp_path / "summary.json"
    receipt.write_text("{}")
    runs = _manifest_runs()[::-1]  # stale manifest reordered LR/RL
    with pytest.raises(ValueError, match="run order drift"):
        verify_cached_runs_against_manifest(
            subject="001", task="emotion", cached_kind="surface sampling",
            cached_summary_path=receipt,
            cached_run_ids=[1, 2], cached_run_lengths=[176, 176],
            cached_source_paths=[
                "/data/tfMRI_EMOTION_LR/bold.nii.gz",
                "/data/tfMRI_EMOTION_RL/bold.nii.gz",
            ],
            manifest_runs=runs, manifest_source_key="bold",
            manifest_lengths=[176, 176],
        )


def test_verify_cached_runs_fails_on_length_and_source_drift(tmp_path: Path) -> None:
    from scripts.core_workflow.lib.task_manifests import verify_cached_runs_against_manifest

    receipt = tmp_path / "summary.json"
    receipt.write_text("{}")
    with pytest.raises(ValueError, match="run length drift"):
        verify_cached_runs_against_manifest(
            subject="001", task="emotion", cached_kind="surface sampling",
            cached_summary_path=receipt,
            cached_run_ids=[1, 2], cached_run_lengths=[176, 176],
            cached_source_paths=[
                "/data/tfMRI_EMOTION_LR/bold.nii.gz",
                "/data/tfMRI_EMOTION_RL/bold.nii.gz",
            ],
            manifest_runs=_manifest_runs(), manifest_source_key="bold",
            manifest_lengths=[176, 405],
        )
    with pytest.raises(ValueError, match="source path drift"):
        verify_cached_runs_against_manifest(
            subject="001", task="emotion", cached_kind="surface sampling",
            cached_summary_path=receipt,
            cached_run_ids=[1, 2], cached_run_lengths=[176, 176],
            cached_source_paths=["/elsewhere/bold.nii.gz", "/elsewhere/bold2.nii.gz"],
            manifest_runs=_manifest_runs(), manifest_source_key="bold",
            manifest_lengths=[176, 176],
        )
    with pytest.raises(FileNotFoundError):
        verify_cached_runs_against_manifest(
            subject="001", task="emotion", cached_kind="surface sampling",
            cached_summary_path=tmp_path / "missing.json",
            cached_run_ids=[1, 2], cached_run_lengths=[176, 176],
            cached_source_paths=[], manifest_runs=_manifest_runs(),
            manifest_source_key="bold", manifest_lengths=[176, 176],
        )
