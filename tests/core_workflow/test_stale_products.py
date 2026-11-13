#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
CORE_DIR = REPO_ROOT / "scripts" / "core_workflow"
if str(CORE_DIR) not in sys.path:
    sys.path.insert(0, str(CORE_DIR))

from lib.stale_products import check_present_products, remind_stale_figures  # noqa: E402


def _write(path: Path, payload: dict, *, age: int) -> Path:
    """Write a JSON file whose modification time is `age` steps in the past."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")
    _set_age(path, age)
    return path


def _set_age(path: Path, age: int) -> None:
    stamp = 1_800_000_000 - age * 100
    os.utime(path, (stamp, stamp))


def _dataset(tmp_path: Path) -> dict[str, Path]:
    """gradient (oldest) -> group atlas -> overview figure; gradient -> gradient figure."""
    work, present = tmp_path / "work", tmp_path / "present"
    gradient = _write(work / "hippomaps_gradients/pfm/hippomaps_gradient_manifest.json", {}, age=4)
    atlas = _write(work / "group_atlas/pfm/group_atlas_manifest.json",
                   {"gradient_source_manifest": str(gradient)}, age=3)
    overview = present / "overviews/pfm/overview.png"
    overview.parent.mkdir(parents=True)
    overview.write_bytes(b"png")
    _set_age(overview, 2)
    _write(present / "overviews/pfm/overview.png.sources.json",
           {"overview_png": str(overview), "source_manifests": [str(atlas)]}, age=2)
    gradient_figure = present / "gradients/pfm/gradients.png"
    gradient_figure.parent.mkdir(parents=True)
    gradient_figure.write_bytes(b"png")
    _set_age(gradient_figure, 2)
    _write(present / "gradients/pfm/hippomaps_gradient_products_pfm_2mm_bilateral_manifest.json",
           {"outputs": {"gradient_figure": str(gradient_figure)}, "source_manifests": {"L": str(gradient)}}, age=2)
    return {"present": present, "gradient": gradient, "atlas": atlas, "overview": overview,
            "gradient_figure": gradient_figure, "work": work}


def _stale(present: Path) -> dict[str, bool]:
    return {Path(item["figure"]).name: item["stale"] for item in check_present_products([present])}


def test_figures_newer_than_their_sources_are_current(tmp_path: Path) -> None:
    assert _stale(_dataset(tmp_path)["present"]) == {"overview.png": False, "gradients.png": False}


def test_rerun_gradient_outdates_gradient_figure_and_overview(tmp_path: Path) -> None:
    files = _dataset(tmp_path)
    _set_age(files["gradient"], 0)
    assert _stale(files["present"]) == {"overview.png": True, "gradients.png": True}


def test_rerendered_overview_on_an_outdated_group_atlas_stays_stale(tmp_path: Path) -> None:
    files = _dataset(tmp_path)
    _set_age(files["gradient"], 1)
    _set_age(files["overview"], 0)
    (result,) = [item for item in check_present_products([files["present"]]) if item["figure"].endswith("overview.png")]
    assert result["stale"] is True
    assert result["problems"] == [f"{files['gradient']} is newer than {files['atlas']}"]


def test_rerun_group_atlas_outdates_only_the_overview(tmp_path: Path) -> None:
    files = _dataset(tmp_path)
    _set_age(files["atlas"], 0)
    assert _stale(files["present"]) == {"overview.png": True, "gradients.png": False}


def test_reused_parent_atlas_is_followed(tmp_path: Path) -> None:
    files = _dataset(tmp_path)
    parent = _write(files["work"] / "group_atlas/pfm/parent/group_atlas_manifest.json", {}, age=0)
    _write(files["atlas"], {"fixed_cluster_reuse": {"source_manifest": str(parent)}}, age=3)
    assert _stale(files["present"])["overview.png"] is True


def test_missing_source_or_root_is_reported(tmp_path: Path) -> None:
    files = _dataset(tmp_path)
    files["atlas"].unlink()
    (result,) = [item for item in check_present_products([files["present"]]) if item["figure"].endswith("overview.png")]
    assert result["problems"] == [f"missing source: {files['atlas']}"]
    with pytest.raises(FileNotFoundError, match="Presentation root does not exist"):
        check_present_products([tmp_path / "absent"])


def test_work_stage_reminder_names_outdated_figures(tmp_path: Path, capsys) -> None:
    files = _dataset(tmp_path)
    remind_stale_figures(files["work"] / "hippomaps_gradients")
    assert capsys.readouterr().out == ""
    _set_age(files["gradient"], 0)
    remind_stale_figures(files["work"] / "hippomaps_gradients")
    assert "[stale] 2 presentation figure(s)" in capsys.readouterr().out
