#!/usr/bin/env python3
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

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


def test_nsi_manifest_keeps_all_base_subjects_and_writes_nsi_audit(tmp_path: Path) -> None:
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
    )

    assert result.pass_subjects == ["1001", "1002", "1003", "1004"]
    assert result.excluded_subjects == []
    assert result.pass_manifest.name == "hcp_7t_lynch2024_pass_latest.txt"
    assert result.pass_manifest.read_text(encoding="utf-8").splitlines() == [
        "sub-1001",
        "sub-1002",
        "sub-1003",
        "sub-1004",
    ]
    audit_text = result.audit_tsv.read_text(encoding="utf-8")
    assert "sub-1001\t0.43\tincluded" in audit_text
    assert "sub-1002\t0.429\tincluded" in audit_text
    assert "sub-1004\t\tmissing_nsi" in audit_text
    run_manifest = nsi_mod.json.loads(result.run_manifest.read_text(encoding="utf-8"))
    assert run_manifest["selection_rule"] == "all_base_subjects"
    assert run_manifest["n_base_subjects"] == 4
    assert run_manifest["n_pass"] == 4
    assert run_manifest["n_excluded"] == 0
    assert run_manifest["n_missing_nsi"] == 1
    assert "nsi_threshold" not in run_manifest
    assert "expected_count" not in run_manifest


def test_nsi_manifest_does_not_require_an_expected_count(tmp_path: Path) -> None:
    base = tmp_path / "base_subjects.txt"
    base.write_text("1001\n1002\n", encoding="utf-8")
    nsi = tmp_path / "nsi.csv"
    nsi.write_text("subject,nsi\nsub-1001,0.44\nsub-1002,0.45\n", encoding="utf-8")

    result = nsi_mod.build_lynch2024_nsi_manifest(
        base_subjects_manifest=base,
        nsi_tsv=nsi,
        out_dir=tmp_path / "all_subjects",
    )
    assert result.pass_subjects == ["1001", "1002"]
    assert result.excluded_subjects == []
