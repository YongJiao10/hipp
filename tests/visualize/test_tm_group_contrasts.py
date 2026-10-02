"""Tests for the TM third column (template-matching dominant atlas).

Covers branch-exact dominant-vote computation, fail-closed cohort/branch
gates, TM contrast manifest contracts, renderer TM helpers (HCP blank),
and the 14-column figure layout with tiny synthetic tiles (no Workbench).
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest  # pyright: ignore[reportMissingImports]
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "visualize" / "lib"))
import tm_group_contrasts as tm  # pyright: ignore[reportMissingImports]

RENDERER_PATH = ROOT / (
    "scripts/visualize/multidataset_hipp_cortex_gradient_wta/"
    "render_multidataset_hipp_cortex_gradient_wta.py"
)


@pytest.fixture(scope="module")
def renderer():
    module_spec = importlib.util.spec_from_file_location(
        "test_tm_renderer", RENDERER_PATH
    )
    assert module_spec is not None and module_spec.loader is not None
    module = importlib.util.module_from_spec(module_spec)
    sys.modules[module_spec.name] = module
    module_spec.loader.exec_module(module)
    return module


def test_tm_dominant_counts_match_legacy_csv():
    import csv

    expected = {}
    with open(tm.TM_LEGACY_STATS_CSV, newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row["grid"] == "2mm_36p":
                expected[row["cohort_key"]] = (
                    int(row["van_ant_vtx_count"]),
                    int(row["smn_mid_vtx_count"]),
                    int(row["dmn_post_vtx_count"]),
                )
    assert set(expected) == set(tm.TM_COHORT_ROSTERS)
    for cohort_id, (van, smn, dmn) in expected.items():
        dominant, _ = tm.tm_dominant_labels(cohort_id, "standard")
        got = tuple(
            int((dominant["L"] == k).sum() + (dominant["R"] == k).sum())
            for k in (1, 2, 3)
        )
        assert got == (van, smn, dmn), cohort_id


def test_tm_ctxreg_dominant_is_deterministic_and_complete():
    first, _ = tm.tm_dominant_labels("iowa_hc", "ctxreg")
    second, _ = tm.tm_dominant_labels("iowa_hc", "ctxreg")
    for hemi in ("L", "R"):
        assert (first[hemi] == second[hemi]).all()
        assert set(first[hemi].tolist()).issubset({1, 2, 3})
    total = sum(int((first[h] == k).sum()) for h in ("L", "R") for k in (1, 2, 3))
    assert total == 1024


def test_tm_dominant_rejects_hcp_and_bad_branch():
    with pytest.raises(ValueError, match="HCP"):
        tm.tm_dominant_labels("hcp_7t", "standard")
    with pytest.raises(ValueError, match="[Ss]ignal branch"):
        tm.tm_dominant_labels("iowa_hc", "bogus")


def test_tm_gifti_contract_matches_branch_display():
    assert set(tm.TM_GIFTI_RGBA) == {1, 2, 3}
    assert set(tm.TM_SUBREGION_DISPLAY) == {1, 2, 3}
    assert tm.TM_GIFTI_RGBA[1] == (220, 92, 255, 255)
    assert tm.TM_GIFTI_RGBA[2] == (99, 150, 194, 255)
    assert tm.TM_GIFTI_RGBA[3] == (216, 92, 107, 255)


def test_render_tm_hippo_rejects_hcp_without_data(renderer, tmp_path):
    with pytest.raises(ValueError, match="HCP"):
        renderer.render_hippo_tm_biglegend(
            cid="hcp_7t",
            spec={},
            panels_dir=tmp_path,
            smoothing="2mm",
            variant="36p",
        )


def test_render_tm_hippo_rejects_bad_branch_and_grid(renderer, tmp_path):
    bad_branch = {"id": "iowa_hc", "signal_branch": "bogus", "n_subjects": 67}
    with pytest.raises(ValueError, match="[Ss]ignal branch"):
        renderer.render_hippo_tm_biglegend(
            cid="iowa_hc",
            spec=bad_branch,
            panels_dir=tmp_path,
            smoothing="2mm",
            variant="36p",
        )
    bad_grid = {"id": "iowa_hc", "signal_branch": "standard", "n_subjects": 67}
    with pytest.raises(ValueError, match="2mm/36p"):
        renderer.render_hippo_tm_biglegend(
            cid="iowa_hc",
            spec=bad_grid,
            panels_dir=tmp_path,
            smoothing="4mm",
            variant="36p",
        )


def test_render_tm_hippo_requires_grad_manifests(renderer, tmp_path):
    spec = {
        "id": "iowa_hc",
        "signal_branch": "standard",
        "n_subjects": 67,
        "grad_subpath": str(tmp_path / "missing-gradient"),
        "group_root_grad": str(tmp_path),
    }
    with pytest.raises(FileNotFoundError, match="[Gg]radient cohort manifests"):
        renderer.render_hippo_tm_biglegend(
            cid="iowa_hc",
            spec=spec,
            panels_dir=tmp_path,
            smoothing="2mm",
            variant="36p",
        )


def test_tm_cohort_data_blank_for_hcp(renderer, tmp_path):
    spec = {
        "id": "hcp_7t",
        "dataset_key": "hcp",
        "signal_branch": "standard",
    }
    out = renderer.load_tm_cohort_data(
        spec, None, tmp_path, {}, {}, smoothing="2mm", variant="36p"
    )
    assert out["has_tm"] is False
    assert out["surf_tm"] is None
    assert out["tm_cortex_pngs"] == {}
    assert tuple(out["tm_network_order"]) == tuple(renderer.TM_SUBREGIONS)


def test_tm_cohort_data_requires_mean(renderer, tmp_path):
    spec = {"id": "iowa_hc", "dataset_key": "iowa_jj_tms", "signal_branch": "standard"}
    with pytest.raises(ValueError, match="display mean"):
        renderer.load_tm_cohort_data(
            spec, None, tmp_path, {}, {}, smoothing="2mm", variant="36p"
        )


def test_append_tm_contrast_none_is_reference(renderer, tmp_path):
    cdata = {
        "spec": {"id": "iowa_hc"},
        "tm_cortex_pngs": {},
        "has_tm": False,
    }
    out = renderer.append_tm_contrast_maps(cdata, None, {}, {}, {}, tmp_path, None)
    assert out["has_tm_contrast"] is False


def test_tm_display_means_rejects_panel_mismatch(renderer):
    with pytest.raises(ValueError, match="panel order"):
        renderer.tm_display_means_from_receipt({"panels": ["X"]})


def test_extract_tm_hemisphere_requires_2mm36p(renderer):
    from multidataset_cohorts import (  # pyright: ignore[reportMissingImports]
        GridSpec,
    )

    grid = GridSpec("2mm", "22p", signal_branch="standard", atlas="7net")
    with pytest.raises(ValueError, match="2mm/36p"):
        tm.extract_tm_hemisphere(
            renderer, {"id": "iowa_hc"}, grid, "L", {}, "canonical"
        )


def test_load_tm_contrasts_fail_closed(tmp_path):
    from multidataset_cohorts import (  # pyright: ignore[reportMissingImports]
        GridSpec,
    )

    grid = GridSpec("2mm", "36p", signal_branch="standard", atlas="7net")

    bad_schema = tmp_path / "bad_schema.json"
    bad_schema.write_text(json.dumps({"schema": "nope", "version": "canonical"}))
    with pytest.raises(ValueError, match="contract mismatch"):
        tm.load_tm_contrasts(bad_schema, grid, "canonical")

    bad_panels = tmp_path / "bad_panels.json"
    bad_panels.write_text(
        json.dumps(
            {"schema": tm.SCHEMA, "version": "canonical", "panels": ["X"], "grids": {}}
        )
    )
    with pytest.raises(ValueError, match="panel order"):
        tm.load_tm_contrasts(bad_panels, grid, "canonical")

    missing_grid = tmp_path / "missing_grid.json"
    missing_grid.write_text(
        json.dumps(
            {
                "schema": tm.SCHEMA,
                "version": "canonical",
                "panels": list(tm.TM_PANELS),
                "grids": {},
            }
        )
    )
    with pytest.raises(ValueError, match="lacks grid"):
        tm.load_tm_contrasts(missing_grid, grid, "canonical")


def _tiny_png(path: Path, size=(60, 40), color="white") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", size, color).save(path)
    return path


def _figure_cdata(tmp_path: Path, renderer, *, with_tm: bool, cid: str):
    spec = {
        "id": cid,
        "dataset_name": "Iowa 3T" if cid != "hcp_7t" else "HCP 7T",
        "cohort_name": "HC"
        if cid == "iowa_hc"
        else ("HCP 7T" if cid == "hcp_7t" else "Patient"),
        "cohort_desc": "Healthy Controls",
        "n_subjects": 67 if cid == "iowa_hc" else 151,
        "bg_color": "#f7faff",
        "badge_bg": "#1d5b79",
        "badge_fg": "#ffffff",
        "col": 1,
    }

    def tile(name):
        return _tiny_png(tmp_path / f"{cid}_{name}.png")

    cdata = {
        "spec": spec,
        "surf_grad": Image.new("RGB", (60, 40), "white"),
        "surf_wta": Image.new("RGB", (60, 40), "white"),
        "grad_cortex_pngs": {
            s: tile(f"grad_{s}") for s in ("Anterior", "Body", "Posterior")
        },
        "wta_cortex_pngs": {
            n: tile(f"wta_{n}")
            for n in ("VentralAttention", "Somatomotor", "Language", "Default")
        },
        "wta_network_order": ("Default", "VentralAttention", "Somatomotor", "Language"),
        "has_contrast": cid != "hcp_7t" and cid != "iowa_hc",
    }
    if with_tm:
        cdata.update(
            {
                "surf_tm": None
                if cid == "hcp_7t"
                else Image.new("RGB", (60, 40), "white"),
                "tm_cortex_pngs": {}
                if cid == "hcp_7t"
                else {n: tile(f"tm_{n}") for n in renderer.TM_SUBREGIONS},
                "tm_network_order": tuple(renderer.TM_SUBREGIONS),
                "has_tm": cid != "hcp_7t",
            }
        )
    return cdata


def test_figure_14col_tm_layout(tmp_path, renderer):
    from multidataset_cohorts import (  # pyright: ignore[reportMissingImports]
        GridSpec,
    )

    grid = GridSpec("2mm", "36p", signal_branch="standard", atlas="7net")
    cohort_data = [
        _figure_cdata(tmp_path, renderer, with_tm=True, cid="hcp_7t"),
        _figure_cdata(tmp_path, renderer, with_tm=True, cid="iowa_hc"),
    ]
    out_png = tmp_path / "tm_14col.png"
    renderer.render_multidataset_hipp_cortex_figure_2col(
        cohort_data=cohort_data, output_png=out_png, spec=grid, dpi=25
    )
    assert out_png.is_file() and out_png.stat().st_size > 0
    with Image.open(out_png) as image:
        width, _ = image.size
    out_legacy = tmp_path / "legacy_10col.png"
    legacy_data = [
        _figure_cdata(tmp_path, renderer, with_tm=False, cid="hcp_7t"),
        _figure_cdata(tmp_path, renderer, with_tm=False, cid="iowa_hc"),
    ]
    renderer.render_multidataset_hipp_cortex_figure_2col(
        cohort_data=legacy_data, output_png=out_legacy, spec=grid, dpi=25
    )
    with Image.open(out_legacy) as image:
        legacy_width, _ = image.size
    assert width > legacy_width


def test_tm_subject_and_contrast_paths_are_branch_scoped(tmp_path):
    standard = tm.tm_subject_path(tmp_path, "canonical", "2mm_36p", "iowa_hc")
    ctxreg = tm.tm_subject_path(tmp_path, "canonical", "2mm_36p_ctxreg", "iowa_hc")
    assert standard != ctxreg
    assert "_tm_subjects.npz" in standard.name
    assert tm.tm_contrast_path(
        tmp_path, "canonical", "2mm_36p", "iowa_pat"
    ).name.endswith("_tm_vs_iowa_hc.npz")
