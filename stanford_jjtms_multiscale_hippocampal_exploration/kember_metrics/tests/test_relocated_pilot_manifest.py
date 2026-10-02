"""Contracts ensuring the pre-rebuild pilot does not remain an active output."""

from __future__ import annotations

from pathlib import Path


PILOT_ROOT = (
    Path(__file__).resolve().parents[1]
    / "results"
    / "pilot_sub-1001_2mm_native_area_20260828T000000Z"
)


def test_pre_rebuild_pilot_is_not_an_active_output() -> None:
    assert not PILOT_ROOT.exists()
