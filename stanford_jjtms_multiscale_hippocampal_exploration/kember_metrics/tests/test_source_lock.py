"""Contract tests for the pinned Kember v1 author-source checkout."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest


SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))

from source_lock import load_source_lock, validate_author_checkout  # noqa: E402


def test_validate_author_checkout_requires_the_v1_nonarchive_files(tmp_path: Path) -> None:
    checkout = tmp_path / "posterior_hippocampus_development"
    checkout.mkdir()
    lock = {
        "repository": "https://github.com/JonahKember/posterior_hippocampus_development.git",
        "tag": "v1",
        "commit": "5fd2858ca8979891216f518b45751d4517e037a0",
        "required_files": [
            "REPRODUCING.md",
            "manuscript_analyses.ipynb",
            "data/supplementary_data.csv",
        ],
    }

    with pytest.raises(ValueError, match="missing required author file"):
        validate_author_checkout(lock, checkout)


def test_validate_author_checkout_rejects_unapproved_figure_archive_extraction(tmp_path: Path) -> None:
    checkout = tmp_path / "posterior_hippocampus_development"
    (checkout / "data" / "Figure 4_extracted").mkdir(parents=True)

    with pytest.raises(ValueError, match="unapproved Figure archive extraction"):
        validate_author_checkout({"required_files": []}, checkout)


def test_tracked_lock_validates_the_pinned_author_checkout() -> None:
    analysis_root = Path(__file__).resolve().parents[1]
    checkout = Path(__file__).resolve().parents[3] / "external" / "posterior_hippocampus_development"
    lock = load_source_lock(analysis_root / "source_lock.json")

    validate_author_checkout(lock, checkout)

    assert lock["tag"] == "v1"


def test_validate_author_checkout_rejects_an_unpinned_commit() -> None:
    analysis_root = Path(__file__).resolve().parents[1]
    checkout = Path(__file__).resolve().parents[3] / "external" / "posterior_hippocampus_development"
    lock = load_source_lock(analysis_root / "source_lock.json")
    lock["commit"] = "0" * 40

    with pytest.raises(ValueError, match="unexpected author checkout commit"):
        validate_author_checkout(lock, checkout)
