#!/usr/bin/env python3
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = (
    REPO_ROOT
    / "scripts"
    / "visualize"
    / "multidataset_subfield_composition_2mm_2col"
    / "analyze_and_render_stanford_subfield_differences.py"
)


spec = importlib.util.spec_from_file_location("stanford_subfield_fdr", SCRIPT)
if spec is None or spec.loader is None:
    raise RuntimeError(f"Failed to load module: {SCRIPT}")
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)


def test_bh_fdr_is_applied_within_each_case_cohort_family() -> None:
    results = [
        {
            "case_cohort": "Group A",
            "p_parametric": 0.001 if index == 0 else 1.0,
        }
        for index in range(30)
    ]
    results.extend(
        {
            "case_cohort": "Group B",
            "p_parametric": 1.0e-8,
        }
        for _ in range(30)
    )

    module.apply_bh_fdr_by_case_cohort(results)

    assert results[0]["fdr_q_parametric"] == pytest.approx(0.03)
    assert results[0]["fdr_n_tests"] == 30
    assert results[30]["fdr_q_parametric"] == pytest.approx(1.0e-8)
    assert all(row["fdr_n_tests"] == 30 for row in results)
