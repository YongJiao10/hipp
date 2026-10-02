"""Contracts preventing the active workflow from depending on retired branches."""

from __future__ import annotations

from pathlib import Path


SOURCE_ROOT = Path(__file__).resolve().parents[1] / "src"


def test_active_kember_sources_do_not_reference_retired_packages() -> None:
    offenders = {}
    for path in sorted(SOURCE_ROOT.glob("*.py")):
        text = path.read_text(encoding="utf-8")
        hits = [marker for marker in ("fast_rest", "kember_area") if marker in text]
        if hits:
            offenders[path.name] = hits
    assert offenders == {}
