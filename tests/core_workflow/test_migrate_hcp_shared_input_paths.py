from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = REPO_ROOT / "scripts" / "core_workflow" / "migrate_hcp_shared_input_paths.py"
OLD_TO_NEW = {
    "routea_151": "routea",
    "surface_151": "surface",
    "routeb_151": "routeb",
    "routea_ctxreg_151": "routea_ctxreg",
    "routea_ctxdrop_151": "routea_ctxdrop",
}


def _load_module():
    spec = importlib.util.spec_from_file_location("migrate_hcp_shared_input_paths", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


MODULE = _load_module()


def _make_repo(tmp_path: Path) -> tuple[Path, Path, dict[str, tuple[int, str]]]:
    repo_root = tmp_path / "repo"
    shared_root = repo_root / "outputs_hcp" / "inputs" / "hcp" / "kong2022" / "shared_inputs"
    sentinels: dict[str, tuple[int, str]] = {}
    for old_name in OLD_TO_NEW:
        source = shared_root / old_name
        source.mkdir(parents=True)
        sentinel = source / "sentinel.txt"
        content = f"content for {old_name}\n"
        sentinel.write_text(content, encoding="utf-8")
        sentinels[old_name] = (sentinel.stat().st_ino, content)
    return repo_root, shared_root, sentinels


def _run_cli(repo_root: Path, mode: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(SCRIPT), "--repo-root", str(repo_root), mode],
        text=True,
        capture_output=True,
        check=False,
    )


def _contains_family_data(value: object) -> bool:
    if isinstance(value, dict):
        return any("family" in str(key).lower() or _contains_family_data(item) for key, item in value.items())
    if isinstance(value, list):
        return any(_contains_family_data(item) for item in value)
    return isinstance(value, str) and "family" in value.lower()


def test_build_plan(tmp_path: Path) -> None:
    shared_root = tmp_path / "shared_inputs"

    plan = MODULE.build_migration_plan(shared_root)
    assert [(move.source.name, move.destination.name) for move in plan] == list(OLD_TO_NEW.items())


def test_dry_run_makes_no_changes(tmp_path: Path) -> None:
    repo_root, shared_root, sentinels = _make_repo(tmp_path)
    provenance = repo_root / "outputs_hcp" / "runs" / "provenance.json"
    provenance.parent.mkdir(parents=True)
    original = '{"path":"shared_inputs/routea_151/sub-1"}\n'
    provenance.write_text(original, encoding="utf-8")

    result = _run_cli(repo_root, "--dry-run")

    assert result.returncode == 0, result.stderr
    assert not (shared_root / "path_migration_20260711.json").exists()
    assert provenance.read_text(encoding="utf-8") == original
    for old_name, (inode, content) in sentinels.items():
        sentinel = shared_root / old_name / "sentinel.txt"
        assert sentinel.stat().st_ino == inode
        assert sentinel.read_text(encoding="utf-8") == content
        assert not (shared_root / OLD_TO_NEW[old_name]).exists()


def test_destination_collision_fails_before_any_rename(tmp_path: Path) -> None:
    repo_root, shared_root, sentinels = _make_repo(tmp_path)
    (shared_root / "routeb").mkdir()

    result = _run_cli(repo_root, "--apply")

    assert result.returncode != 0
    assert "destination" in result.stderr.lower()
    for old_name, (inode, _) in sentinels.items():
        assert (shared_root / old_name / "sentinel.txt").stat().st_ino == inode
    assert not (shared_root / "routea").exists()
    assert not (shared_root / "path_migration_20260711.json").exists()


def test_apply_renames_in_place_rewrites_only_provenance_and_writes_private_manifest(tmp_path: Path) -> None:
    repo_root, shared_root, sentinels = _make_repo(tmp_path)
    outputs = repo_root / "outputs_hcp"
    provenance = outputs / "runs" / "provenance.md"
    provenance.parent.mkdir(parents=True)
    provenance.write_text(
        "shared_inputs/routea_151/sub-1\nshared_inputs/routea_151_backup/sub-2\n",
        encoding="utf-8",
    )
    family_source = outputs / "metadata" / "family_assignments.csv"
    family_source.parent.mkdir(parents=True)
    family_source.write_text("family,path\nF1,shared_inputs/routeb_151/sub-1\n", encoding="utf-8")
    kinship_source = outputs / "metadata" / "kinship.csv"
    kinship_source.write_text("group,path\nK1,shared_inputs/routeb_151/sub-1\n", encoding="utf-8")
    nested_provenance = shared_root / "routea_151" / "provenance.json"
    nested_provenance.write_text(
        '{"path":"shared_inputs/routea_151/sub-1"}\n', encoding="utf-8"
    )
    binary = outputs / "runs" / "array.npy"
    binary.write_bytes(b"shared_inputs/routeb_151")
    archive_text = outputs / "archive" / "old.json"
    archive_text.parent.mkdir(parents=True)
    archive_text.write_text('{"path":"shared_inputs/routeb_151"}\n', encoding="utf-8")
    underscore_archive = outputs / "_archive" / "old.json"
    underscore_archive.parent.mkdir(parents=True)
    underscore_archive.write_text(
        '{"path":"shared_inputs/routeb_151"}\n', encoding="utf-8"
    )

    result = _run_cli(repo_root, "--apply")

    assert result.returncode == 0, result.stderr
    for old_name, (inode, content) in sentinels.items():
        assert not (shared_root / old_name).exists()
        sentinel = shared_root / OLD_TO_NEW[old_name] / "sentinel.txt"
        assert sentinel.stat().st_ino == inode
        assert sentinel.read_text(encoding="utf-8") == content
    assert provenance.read_text(encoding="utf-8") == (
        "shared_inputs/routea_151/sub-1\nshared_inputs/routea_151_backup/sub-2\n"
    )
    assert family_source.read_text(encoding="utf-8").endswith("shared_inputs/routeb_151/sub-1\n")
    assert kinship_source.read_text(encoding="utf-8").endswith("shared_inputs/routeb_151/sub-1\n")
    assert (shared_root / "routea" / "provenance.json").read_text(encoding="utf-8") == (
        '{"path":"shared_inputs/routea_151/sub-1"}\n'
    )
    assert binary.read_bytes() == b"shared_inputs/routeb_151"
    assert archive_text.read_text(encoding="utf-8") == '{"path":"shared_inputs/routeb_151"}\n'
    assert underscore_archive.read_text(encoding="utf-8") == (
        '{"path":"shared_inputs/routeb_151"}\n'
    )

    manifest_path = shared_root / "path_migration_20260711.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["status"] == "ok"
    assert manifest["affected_files"] == []
    assert manifest["replacement_count"] == 0
    assert manifest["validation"]["all_sources_absent"] is True
    assert manifest["validation"]["all_destinations_present"] is True
    assert not _contains_family_data(manifest)


def test_cli_requires_exactly_one_mode(tmp_path: Path) -> None:
    repo_root, _, _ = _make_repo(tmp_path)
    neither = subprocess.run(
        [sys.executable, str(SCRIPT), "--repo-root", str(repo_root)],
        text=True,
        capture_output=True,
        check=False,
    )
    both = subprocess.run(
        [sys.executable, str(SCRIPT), "--repo-root", str(repo_root), "--dry-run", "--apply"],
        text=True,
        capture_output=True,
        check=False,
    )
    assert neither.returncode != 0
    assert both.returncode != 0
