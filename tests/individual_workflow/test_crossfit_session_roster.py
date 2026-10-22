from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest


def test_session_comparison_uses_explicit_qc_roster_not_reference_directories(tmp_path, monkeypatch):
    source = Path(__file__).resolve().parents[2] / "scripts/individual_workflow/01_04_render_hcp_runs_comparison.py"
    spec = importlib.util.spec_from_file_location("session_roster_comparison", source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    roster = tmp_path / "qc.txt"
    roster.write_text("100610\n102311\n")
    references = tmp_path / "references"
    for subject in ("100610", "102311", "999999"):
        (references / f"sub-{subject}").mkdir(parents=True)
    monkeypatch.setattr(module, "DEFAULT_KONG2022_ROUTEA_ROOT", references)
    monkeypatch.setattr(module, "resolve_crossfit_template_matching_root", lambda: tmp_path / "work")
    fake_crossfit = SimpleNamespace(load_group_atlas_module=lambda: None, load_group_gradient_module=lambda: None)
    monkeypatch.setattr(module, "load_module", lambda *args: fake_crossfit)

    class ReachedFit(Exception):
        pass

    def fit(**kwargs):
        assert kwargs["subjects"] == ["100610", "102311"]
        raise ReachedFit

    monkeypatch.setattr(module, "fit_split_half_template", fit)
    monkeypatch.setattr(sys, "argv", [str(source), "--subjects-manifest", str(roster)])
    with pytest.raises(ReachedFit):
        module.main()
