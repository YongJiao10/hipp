# pyright: reportMissingImports=false
from __future__ import annotations

from pathlib import Path
import sys

import pytest


ROOT = Path(__file__).resolve().parents[2]
LIB_DIR = ROOT / "scripts" / "visualize" / "lib"
if str(LIB_DIR) not in sys.path:
    sys.path.insert(0, str(LIB_DIR))

from multidataset_cohorts import (  # noqa: E402
    GRID_VARIANTS,
    GridSpec,
    get_comparison_cohorts,
    get_multidataset_cohorts,
    resolve_cohort_paths,
)


def test_36p_grid_registry() -> None:
    assert GRID_VARIANTS == ("2mm_36p", "4mm_36p")
    assert GridSpec("2mm", "36p").tag == "2mm_36p"
    assert len(get_multidataset_cohorts(GridSpec("4mm", "36p"))[0]) == 6


def test_ctxreg_is_an_explicit_additive_2mm_36p_branch() -> None:
    spec = GridSpec("2mm", "36p", signal_branch="ctxreg")

    assert spec.tag == "2mm_36p_ctxreg"
    assert GridSpec.from_tag(spec.tag).signal_branch == "ctxreg"
    assert "pfm_ctxreg" in str(
        resolve_cohort_paths("iowa_hc", spec)["group_root"]
    )
    assert "pfm_ctxreg" not in str(
        resolve_cohort_paths("iowa_hc", GridSpec("2mm", "36p"))["group_root"]
    )

    columns = get_multidataset_cohorts(spec)
    assert all(
        entry["signal_branch"] == "ctxreg"
        for column in columns
        for entry in column
    )


def test_ctxreg_rejects_noncanonical_grid_or_atlas() -> None:
    with pytest.raises(ValueError, match="ctxreg.*2mm/36p"):
        GridSpec("4mm", "36p", signal_branch="ctxreg")
    with pytest.raises(ValueError, match="canonical7"):
        GridSpec("2mm", "36p", atlas="canonical8", signal_branch="ctxreg")


def test_comparison_cohorts_keep_method_specific_ctxreg_roots() -> None:
    spec = GridSpec("2mm", "36p", signal_branch="ctxreg")
    methods = get_comparison_cohorts(spec)

    assert set(methods) == {"kong2022_7net", "lynch2024_7net"}
    assert all(
        entry["signal_branch"] == "ctxreg"
        and "pfm_ctxreg" in str(entry["group_root"])
        for entries in methods.values()
        for entry in entries
    )
