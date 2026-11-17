#!/usr/bin/env python3
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
CORE = REPO_ROOT / "scripts" / "core_workflow"
if str(CORE) not in sys.path:
    sys.path.insert(0, str(CORE))

from lib import local_cortex_control as lcc  # noqa: E402


def _load(relpath: str, alias: str):
    spec = importlib.util.spec_from_file_location(alias, REPO_ROOT / relpath)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[alias] = module
    spec.loader.exec_module(module)
    return module


atlas_mod = _load("scripts/core_workflow/05_02_run_group_atlas.py", "core_05_02_local_cortex_control_test")
ctxreg_mod = _load("scripts/core_workflow/04_00_materialize_cortex_regressed_surface_timeseries.py", "core_04_00_local_cortex_control_test")
diag_mod = _load("scripts/core_workflow/06_07_render_local_cortex_control_diagnostics.py", "core_06_07_local_cortex_control_test")

NETS = ("Default", "Visual", "Somatomotor", "DorsalAttention", "VentralAttention", "Control", "Language")


def test_winner_matches_group_atlas_normalized_top20_rule() -> None:
    rng = np.random.default_rng(3)
    roi_network = rng.integers(0, len(NETS), size=60)
    fc = rng.normal(0.02, 0.05, size=(80, 60))
    fc[rng.random(fc.shape) < 0.1] = np.nan
    fc[:2] = -1.0  # unassigned rows
    registry = [{"canonical7": NETS[k]} for k in roi_network]
    rows = atlas_mod.build_normalized_topn_count_vertex_rows(
        mean_profile=fc.astype(np.float32), support_indices=np.arange(fc.shape[0]),
        roi_registry=registry, top_n_cap=20,
    )
    expected = [NETS.index(r["top1_network"]) if r["top1_network"] != "unassigned" else -1 for r in rows]
    got = lcc.normalized_top20_winner(fc.astype(np.float32), roi_network, NETS)
    assert got.tolist() == expected


def test_ctxdrop_fc_matches_explicit_roi_recomputation() -> None:
    rng = np.random.default_rng(5)
    n_cortex, n_t = 40, 30
    ctx = rng.normal(size=(n_cortex, n_t))
    roi_ids = np.repeat([11, 12, 13, 14], 10)
    roi_ids[0] = 99  # unregistered vertex is never an ROI member
    valid = np.ones(n_cortex, bool)
    valid[5] = False
    bank = lcc.build_roi_bank(ctx, roi_ids, valid, [11, 12, 13, 14])
    hipp = rng.normal(size=(3, n_t))
    hz = lcc.zscore_rows(hipp)
    fc_std = hz @ lcc.zscore_rows(bank.means).T
    drops = [np.array([1, 2, 5, 0]), np.arange(10, 20), np.array([], dtype=int)]
    fc, n_touched, _ = lcc.ctxdrop_fc(hipp_z=hz, fc_standard=fc_std, ctx=ctx, bank=bank, drop_rows=drops)
    for v, drop in enumerate(drops):
        for c, pid in enumerate([11, 12, 13, 14]):
            members = [i for i in range(n_cortex) if roi_ids[i] == pid and valid[i] and i not in set(drop.tolist())]
            if not members:
                assert np.isnan(fc[v, c])
                continue
            expected = np.corrcoef(hipp[v], ctx[members].mean(axis=0))[0, 1]
            assert fc[v, c] == pytest.approx(expected, abs=1e-10)
    assert n_touched.tolist() == [1, 1, 0]


def test_runwise_residual_matches_formal_ctxreg_on_centered_series() -> None:
    rng = np.random.default_rng(7)
    lengths = [12, 9, 15]
    y = rng.normal(size=(4, sum(lengths))) + np.repeat([0.0, 50.0, -20.0], lengths)
    x = rng.normal(size=(4, sum(lengths))) + np.repeat([5.0, -5.0, 1.0], lengths)
    segments, start = [], 0
    for i, n in enumerate(lengths, start=1):
        segments.append({"run_id": i, "run_start": start, "run_stop": start + n, "run_length": n})
        start += n
    cortex = x  # one cortical vertex per hippocampal vertex
    formal = ctxreg_mod.regress_local_cortex_signal(
        hipp_ts=y.astype(np.float32), cortex_ts=cortex.astype(np.float32),
        neighbor_indices=[np.array([v]) for v in range(4)], run_segments=segments,
    ).controlled_ts.astype(np.float64)
    ours = lcc.runwise_residual(y, x, lengths)
    for seg in segments:
        w = slice(seg["run_start"], seg["run_stop"])
        np.testing.assert_allclose(ours[:, w], formal[:, w] - formal[:, w].mean(axis=1, keepdims=True), atol=1e-3)


def test_render_requires_every_roster_subject(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="Missing local-cortex control diagnostics"):
        diag_mod.load_cohort(tmp_path, ["100610"])


def test_diagnostic_outputs_use_neutral_formal_roots() -> None:
    assert diag_mod.DEFAULT_LOCAL_CORTEX_CONTROL_WORK_ROOT == REPO_ROOT / "outputs_hcp" / "work" / "local_cortex_control"
    assert diag_mod.DEFAULT_LOCAL_CORTEX_CONTROL_PRESENT_DIR == REPO_ROOT / "outputs_hcp" / "present" / "local_cortex_control"
    assert diag_mod.CONTROL_RADII_MM == (3.0, 6.0, 10.0, 20.0)
