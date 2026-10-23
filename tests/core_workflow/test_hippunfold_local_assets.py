"""Behavior of the local-asset adapter installed in the canonical hippo2 runtime."""

from __future__ import annotations

import importlib.util
import os
import select
import shlex
import subprocess
import sys
from pathlib import Path

import pytest

from scripts.core_workflow.lib.hippunfold_run_contract import HIPPO2_PREFIX


ADAPTER = (
    HIPPO2_PREFIX
    / "lib/python3.13/site-packages/hippunfold/workflow/lib/local_assets.py"
)
CACHE = Path("/hippunfold-unit-assets")


def test_gpu_slot_serializes_commands_and_releases_on_exit(assets):
    prefix = shlex.split(assets.gpu_inference_prefix(f'hippunfold-test-{os.getpid()}'))
    first = subprocess.Popen(
        prefix + [sys.executable, '-c', 'print("held", flush=True); input()'],
        stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True,
    )
    second = None
    try:
        assert select.select([first.stdout], [], [], 10)[0], 'GPU lock acquisition stalled'
        assert first.stdout.readline().strip() == 'held'
        second = subprocess.Popen(
            prefix + [sys.executable, '-c', 'print("acquired", flush=True)'],
            stdout=subprocess.PIPE, text=True,
        )
        assert not select.select([second.stdout], [], [], 0.3)[0]
        first.communicate('\n', timeout=10)
        stdout, _ = second.communicate(timeout=10)
        assert first.returncode == second.returncode == 0
        assert stdout.strip() == 'acquired'
    finally:
        for process in (first, second):
            if process is not None and process.poll() is None:
                process.terminate()
                process.wait(timeout=10)


def test_two_gpu_slots_admit_two_commands_and_queue_the_third(assets):
    prefix = shlex.split(assets.gpu_inference_prefix(f'hippunfold-pair-test-{os.getpid()}', slots=2))
    processes = []
    try:
        for _ in range(2):
            process = subprocess.Popen(
                prefix + [sys.executable, '-c', 'print("held", flush=True); input()'],
                stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True,
            )
            processes.append(process)
            assert select.select([process.stdout], [], [], 10)[0], 'GPU slot acquisition stalled'
            assert process.stdout.readline().strip() == 'held'
        third = subprocess.Popen(
            prefix + [sys.executable, '-c', 'print("acquired", flush=True)'],
            stdout=subprocess.PIPE, text=True,
        )
        processes.append(third)
        assert not select.select([third.stdout], [], [], 0.3)[0]
        processes[0].communicate('\n', timeout=10)
        stdout, _ = third.communicate(timeout=10)
        assert third.returncode == 0
        assert stdout.strip() == 'acquired'
    finally:
        for process in processes:
            if process.poll() is None:
                process.terminate()
                process.wait(timeout=10)


@pytest.fixture
def assets():
    spec = importlib.util.spec_from_file_location("hippunfold_local_assets", ADAPTER)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def stub_cache(monkeypatch, assets, directories, files):
    original_exists = Path.exists
    monkeypatch.setattr(
        Path,
        "exists",
        lambda path: path in directories
        if path.is_relative_to(CACHE)
        else original_exists(path),
    )
    monkeypatch.setattr(assets, "file_is_present", lambda path: Path(path) in files)


def test_cached_template_and_missing_template_are_distinct(assets, monkeypatch):
    cfg = {
        "template_files": {
            "example": {
                "T1w": "T1w.nii.gz",
                "crop_ref": "crop_hemi-{hemi}.nii.gz",
                "hemi_wildcards": ["L", "R"],
            }
        },
        "autotop_labels": ["hipp"],
    }
    folder = CACHE / "template/example"
    files = {folder / "T1w.nii.gz", folder / "crop_hemi-L.nii.gz", folder / "crop_hemi-R.nii.gz"}
    stub_cache(monkeypatch, assets, {folder}, files)
    assert assets.template_is_cached(CACHE, "example", cfg)

    stub_cache(monkeypatch, assets, set(), set())
    assert not assets.template_is_cached(CACHE, "example", cfg)


def test_incomplete_template_names_the_missing_file(assets, monkeypatch):
    cfg = {"template_files": {"example": {"T1w": "T1w.nii.gz", "hemi_wildcards": ["L"]}}, "autotop_labels": ["hipp"]}
    folder = CACHE / "template/example"
    stub_cache(monkeypatch, assets, {folder}, set())
    with pytest.raises(FileNotFoundError, match="T1w.nii.gz"):
        assets.template_is_cached(CACHE, "example", cfg)


def test_atlas_accepts_required_surfaces_and_registration_metrics(assets, monkeypatch):
    cfg = {
        "hemi": ["L"],
        "autotop_labels": ["hipp"],
        "output_density": ["512"],
        "unfoldreg_density": "18k",
        "atlas_metadata": {"example": {"metric_wildcards": ["curvature"]}},
    }
    folder = CACHE / "atlases/tpl-example"
    names = {
        "tpl-example_hemi-L_space-unfold_den-512_label-hipp_inner.surf.gii",
        "tpl-example_hemi-L_space-unfold_den-512_label-hipp_midthickness.surf.gii",
        "tpl-example_hemi-L_space-unfold_den-512_label-hipp_outer.surf.gii",
        "tpl-example_hemi-L_space-unfold_den-18k_label-hipp_inner.surf.gii",
        "tpl-example_hemi-L_space-unfold_den-18k_label-hipp_midthickness.surf.gii",
        "tpl-example_hemi-L_space-unfold_den-18k_label-hipp_outer.surf.gii",
        "tpl-example_hemi-L_den-18k_label-hipp_curvature.shape.gii",
        "tpl-example_hemi-L_den-512_label-hipp_dseg.label.gii",
        "tpl-example_hemi-L_den-18k_label-hipp_dseg.label.gii",
    }
    files = {folder / name for name in names}
    stub_cache(monkeypatch, assets, {folder}, files)
    assert assets.atlas_copy_is_cached(CACHE, "example", cfg)
    files.remove(folder / "tpl-example_hemi-L_den-512_label-hipp_dseg.label.gii")
    assert not assets.atlas_copy_is_cached(CACHE, "example", cfg)
    files.add(folder / "tpl-example_hemi-L_den-512_label-hipp_dseg.label.gii")
    files.remove(folder / "tpl-example_hemi-L_space-unfold_den-512_label-hipp_outer.surf.gii")
    assert not assets.atlas_copy_is_cached(CACHE, "example", cfg)


def test_missing_atlas_extraction_uses_original_producer(assets, monkeypatch):
    cfg = {"hemi": ["L"], "autotop_labels": ["hipp"], "output_density": ["512"], "unfoldreg_density": "18k", "atlas_metadata": {"example": {"metric_wildcards": ["curvature"]}}}
    stub_cache(monkeypatch, assets, set(), set())
    assert not assets.atlas_extract_is_cached(CACHE, "example", cfg)


def test_missing_model_directory_is_not_a_cached_model(assets, monkeypatch):
    stub_cache(monkeypatch, assets, set(), set())
    assert not assets.model_is_unpacked(CACHE / "model/example", {"arch_version": "nnunet_v1", "task": "Task101", "trainer": "nnUNetTrainerV2", "checkpoint": "model_best"})


def test_runtime_source_cache_refreshes_changed_local_rules():
    code = """
from types import SimpleNamespace
from unittest.mock import Mock
from snakemake.sourcecache import SourceCache, LocalSourceFile
cache = SourceCache.__new__(SourceCache)
entry = Mock()
entry.exists.return_value = True
entry.stat.return_value = SimpleNamespace(st_mtime=1.0)
cache._cache_entry = Mock(return_value=entry)
cache._do_cache = Mock()
source = LocalSourceFile('/hippunfold-source-test/download.smk')
source.mtime = Mock(return_value=2.0)
cache._cache(source)
cache._do_cache.assert_called_once_with(source, entry)
cache._do_cache.reset_mock()
source.mtime.return_value = 1.0
cache._cache(source)
cache._do_cache.assert_not_called()
remote = Mock()
remote.mtime.side_effect = AssertionError('Do not query remote source metadata')
cache._cache(remote)
remote.mtime.assert_not_called()
cache._do_cache.assert_not_called()
"""
    result = subprocess.run(
        [str(HIPPO2_PREFIX / "bin/python"), "-c", code],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
