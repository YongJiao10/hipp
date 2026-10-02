#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.core_workflow.lib.task_design import (
    TASK_RUN_LENGTHS,
    build_first_level_design,
    build_task_regressors,
    contrast_vector,
    load_nuisance_regressors,
    normalize_task_branch,
    parse_fsl_ev,
)
from scripts.core_workflow.lib.task_glm import (
    apply_contrast,
    background_residuals,
    fit_first_level_glm,
)


def test_normalize_task_branch_rejects_unknown() -> None:
    with pytest.raises(ValueError):
        normalize_task_branch("gambling")


def test_parse_fsl_ev_and_missing_stem_fails_closed(tmp_path: Path) -> None:
    ev = tmp_path / "fear.txt"
    ev.write_text("32.0 18 1\n74.0 18 1\n", encoding="utf-8")
    arr = parse_fsl_ev(ev)
    assert arr.shape == (2, 3)
    with pytest.raises(FileNotFoundError):
        build_task_regressors(tmp_path, task="emotion", n_frames=176)


def test_task_regressor_frame_contract(tmp_path: Path) -> None:
    for stem in ("fear", "neut"):
        (tmp_path / f"{stem}.txt").write_text("10.0 18 1\n", encoding="utf-8")
    names, design = build_task_regressors(tmp_path, task="emotion", n_frames=176)
    assert names == ["fear", "neut"]
    assert design.shape == (176, 2)
    with pytest.raises(ValueError):
        build_task_regressors(tmp_path, task="emotion", n_frames=175)


def test_nuisance_shape_contract(tmp_path: Path) -> None:
    mov = tmp_path / "Movement_Regressors.txt"
    np.savetxt(str(mov), np.zeros((176, 12)))
    assert load_nuisance_regressors(mov, n_frames=176).shape == (176, 12)
    with pytest.raises(ValueError):
        load_nuisance_regressors(mov, n_frames=175)


def test_glm_recovers_known_betas_and_contrast() -> None:
    rng = np.random.default_rng(0)
    n_frames, n_regs, n_vert = 176, 4, 8
    x = rng.normal(size=(n_frames, n_regs))
    true_beta = rng.normal(size=(n_vert, n_regs))
    y = true_beta @ x.T + 1e-6 * rng.normal(size=(n_vert, n_frames))
    betas, resid, r2 = fit_first_level_glm(y, x)
    np.testing.assert_allclose(betas, true_beta, atol=1e-3)
    assert np.all(r2 > 0.99)
    c = np.array([1.0, -1.0, 0.0, 0.0])
    np.testing.assert_allclose(apply_contrast(betas, c), true_beta[:, 0] - true_beta[:, 1], atol=1e-3)


def test_background_residuals_remove_task_variance() -> None:
    rng = np.random.default_rng(1)
    n_frames = 176
    task = rng.normal(size=(n_frames, 2))
    nuisance = rng.normal(size=(n_frames, 12))
    design = build_first_level_design(task, nuisance)
    y = rng.normal(size=(5, n_frames)) + (design[:, :2] @ rng.normal(size=(2, 5))).T
    resid = background_residuals(y, design, 2)
    assert resid.shape == y.shape
    # Residuals must be near-orthogonal to every design column.
    corr = np.abs(np.corrcoef(resid.ravel(), np.tile(design.ravel(), 5)[: resid.size])[0, 1])
    assert corr < 0.05


def test_task_run_lengths_match_staged_data() -> None:
    assert TASK_RUN_LENGTHS == {"emotion": 176, "wm": 405, "relational": 232}


def test_contrast_vector_rejects_unknown_regressor() -> None:
    with pytest.raises(ValueError):
        contrast_vector(["fear", "neut"], {"fear": 1.0, "bogus": -1.0})


def test_spm_hrf_matches_reference_shape() -> None:
    from scripts.core_workflow.lib.task_design import spm_hrf

    h = spm_hrf(0.72)
    assert abs(float(h.sum()) - 1.0) < 1e-9
    peak_s = float(np.argmax(h)) * 0.72
    assert 4.5 <= peak_s <= 5.5, peak_s
    assert float(h.min()) < 0.0
