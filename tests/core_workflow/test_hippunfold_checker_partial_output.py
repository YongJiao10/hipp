"""Filesystem regression test for the partial space-unfold batch failure."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
CHECKER_PATH = REPO_ROOT / "scripts/core_workflow/01_02_check_hippunfold_outputs.py"
sys.path.insert(0, str(REPO_ROOT / "scripts/core_workflow"))
SPEC = importlib.util.spec_from_file_location("hippunfold_output_checker_partial", CHECKER_PATH)
assert SPEC is not None and SPEC.loader is not None
checker = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(checker)


def test_checker_rejects_unfold_only_surface_pair(tmp_path: Path) -> None:
    batch_root = tmp_path / "batch"
    subject = "1028"
    output_root = batch_root / f"sub-{subject}" / "hippunfold" / f"sub-{subject}" / "surf"
    output_root.mkdir(parents=True)
    for hemi in ("L", "R"):
        filename = (
            f"sub-{subject}_hemi-{hemi}_space-unfold_den-512_"
            "label-hipp_midthickness.surf.gii"
        )
        (output_root / filename).write_text("template-only\n", encoding="utf-8")

    manifest = [
        "sub-{subject}/surf/sub-{subject}_hemi-L_space-unfold_den-512_label-hipp_midthickness.surf.gii",
        "sub-{subject}/surf/sub-{subject}_hemi-R_space-unfold_den-512_label-hipp_midthickness.surf.gii",
    ]
    report = checker.check_subject(batch_root, subject, manifest)

    assert report["status"] == "missing_outputs"
    assert any("space-corobl" in path for path in report["missing_outputs"])
    assert any("space-T1w" in path for path in report["missing_outputs"])
    assert any("surfaces.spec" in path for path in report["missing_outputs"])
