#!/usr/bin/env python3
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
CORE_DIR = REPO_ROOT / "scripts" / "core_workflow"
if str(CORE_DIR) not in sys.path:
    sys.path.insert(0, str(CORE_DIR))


def _load_numbered_module(script_name: str, alias: str):
    module_path = CORE_DIR / script_name
    spec = importlib.util.spec_from_file_location(alias, module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load module: {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[alias] = module
    spec.loader.exec_module(module)
    return module


nsi_mod = _load_numbered_module("02_03_build_lynch2024_nsi_manifest.py", "core_02_03_lynch_nsi_manifest")


def test_nsi_manifest_filters_normalized_subjects_and_writes_exclusions(tmp_path: Path) -> None:
    base = tmp_path / "base_subjects.txt"
    base.write_text("sub-1001\n1002\nsub-1003\n1004\n", encoding="utf-8")
    nsi = tmp_path / "nsi.tsv"
    nsi.write_text(
        "subject\tnsi\n"
        "1001\t0.43\n"
        "sub-1002\t0.429\n"
        "1003\t0.7\n"
        "sub-9999\t0.9\n",
        encoding="utf-8",
    )
    out_dir = tmp_path / "manifests"

    result = nsi_mod.build_lynch2024_nsi_manifest(
        base_subjects_manifest=base,
        nsi_tsv=nsi,
        out_dir=out_dir,
        nsi_threshold=0.43,
        expected_count=2,
    )

    assert result.pass_subjects == ["1001", "1003"]
    assert result.excluded_subjects == ["1002", "1004"]
    assert result.pass_manifest.read_text(encoding="utf-8").splitlines() == ["sub-1001", "sub-1003"]
    excluded_text = result.excluded_tsv.read_text(encoding="utf-8")
    assert "sub-1002\t0.429\tbelow_threshold" in excluded_text
    assert "sub-1004\t\tmissing_nsi" in excluded_text
    run_manifest = nsi_mod.json.loads(result.run_manifest.read_text(encoding="utf-8"))
    assert run_manifest["nsi_threshold"] == 0.43
    assert run_manifest["n_pass"] == 2
    assert run_manifest["expected_count"] == 2


def test_nsi_manifest_enforces_expected_count_unless_overridden(tmp_path: Path) -> None:
    base = tmp_path / "base_subjects.txt"
    base.write_text("1001\n1002\n", encoding="utf-8")
    nsi = tmp_path / "nsi.csv"
    nsi.write_text("subject,nsi\nsub-1001,0.44\nsub-1002,0.45\n", encoding="utf-8")

    with pytest.raises(RuntimeError, match="expected 78"):
        nsi_mod.build_lynch2024_nsi_manifest(
            base_subjects_manifest=base,
            nsi_tsv=nsi,
            out_dir=tmp_path / "strict",
            nsi_threshold=0.43,
            expected_count=78,
        )

    result = nsi_mod.build_lynch2024_nsi_manifest(
        base_subjects_manifest=base,
        nsi_tsv=nsi,
        out_dir=tmp_path / "override",
        nsi_threshold=0.43,
        expected_count=78,
        allow_count_mismatch=True,
    )
    assert result.pass_subjects == ["1001", "1002"]
