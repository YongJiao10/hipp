from __future__ import annotations

import re
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPTS_ROOT = REPO_ROOT / "scripts"


def _is_numbered(name: str) -> bool:
    return re.match(r"^[0-9]{2}(?:_[0-9]{2})?_", name) is not None


def test_protected_workflow_roots_have_numbered_entrypoints_only() -> None:
    core_root = SCRIPTS_ROOT / "core_workflow"
    individual_root = SCRIPTS_ROOT / "individual_workflow"

    unnumbered_core = [
        path.name for path in core_root.glob("*.py") if not _is_numbered(path.name)
    ]
    unnumbered_individual = [
        path.name
        for path in (*individual_root.glob("*.py"), *individual_root.glob("*.sh"))
        if not _is_numbered(path.name)
    ]

    assert unnumbered_core == []
    assert unnumbered_individual == []


def test_dataset_specific_and_individual_workflow_surfaces_are_explicit() -> None:
    assert (SCRIPTS_ROOT / "jj_tms" / "README.md").is_file()
    assert not (SCRIPTS_ROOT / "core_workflow" / "map_jj_tms_sites_to_kong2022.py").exists()
    assert not (SCRIPTS_ROOT / "individual_workflow" / "jj_tms_hcp151_fixed_template_matching.py").exists()

    readme = (SCRIPTS_ROOT / "individual_workflow" / "README.md").read_text(
        encoding="utf-8"
    )
    assert "family-aware crossfit" in readme
    assert "immutable HCP151 template" in readme
