"""Document contracts for the frozen Stanford/JJTMS method decision."""

from __future__ import annotations

import re
from pathlib import Path


ANALYSIS_ROOT = Path(__file__).resolve().parents[2]
DOCS_ROOT = ANALYSIS_ROOT / "docs"


def _table_row(text: str, field: str) -> str:
    prefix = f"| `{field}` |"
    rows = [line.strip() for line in text.splitlines() if line.strip().startswith(prefix)]
    assert len(rows) == 1
    return rows[0]


def test_method_document_freezes_native_kember_area_source() -> None:
    text = (DOCS_ROOT / "kember_area_method.md").read_text(encoding="utf-8")

    assert "surfareacorobl" in text
    assert "not a Kember endpoint" in _table_row(text, "transition_area")
    assert "geometric-only" in _table_row(text, "compute_vertex_area")


def test_method_document_defines_area_totals_and_relationships() -> None:
    text = (DOCS_ROOT / "kember_area_method.md").read_text(encoding="utf-8")

    assert "sum of the consumed native per-vertex metric values" in _table_row(
        text, "total_surface_area_mm2"
    )
    assert "sum of consumed native per-vertex metric values at valid ROI vertices" in _table_row(
        text, "total_valid_surface_area_mm2"
    )
    assert "sum of the three system surface areas" in _table_row(
        text, "total_labeled_area_mm2"
    )
    unassigned = _table_row(text, "unassigned_surface_area_mm2")
    assert "total_surface_area_mm2 - total_labeled_area_mm2" in unassigned
    assert "valid label-0 and invalid/out-of-ROI area" in unassigned


def test_method_document_binds_native_area_to_required_provenance() -> None:
    text = (DOCS_ROOT / "kember_area_method.md").read_text(encoding="utf-8")
    rows = [
        line.strip()
        for line in text.splitlines()
        if line.strip().startswith("| Native provenance bundle |")
    ]

    assert len(rows) == 1
    provenance = rows[0]
    assert "recorded together" in provenance
    for artifact in (
        "consumed native surface",
        "native labels",
        "label-transfer method",
        "density evidence",
        "input hashes",
        "native area metric",
    ):
        assert artifact in provenance


def test_analysis_map_has_exactly_the_three_active_workflows() -> None:
    text = (DOCS_ROOT / "analysis_map.md").read_text(encoding="utf-8")
    active_names = re.findall(r"`([a-z][a-z0-9_]+)`", text)

    assert set(active_names) == {
        "cross_dataset_rest_fc",
        "kember_metrics",
        "external_group_atlas_analysis",
    }
    assert len(active_names) == 3
