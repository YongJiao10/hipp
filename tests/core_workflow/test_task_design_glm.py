#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pytest
from scipy.fft import idct

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


@pytest.mark.parametrize(
    "n_frames,tr,n_drift",
    [(176, 0.72, 1), (405, 0.72, 4), (232, 0.72, 2), (405, 0.36, 2)],
)
def test_design_uses_spm_128s_dct_drift(
    n_frames: int, tr: float, n_drift: int
) -> None:
    rng = np.random.default_rng(8)
    task = rng.normal(size=(n_frames, 2))
    nuisance = rng.normal(size=(n_frames, 12))
    design = build_first_level_design(task, nuisance, tr=tr)

    assert design.shape == (n_frames, 15 + n_drift)
    np.testing.assert_array_equal(design[:, :2], task)
    np.testing.assert_array_equal(design[:, 2:14], nuisance)
    np.testing.assert_array_equal(design[:, 14], np.ones(n_frames))
    # Independent DCT-II reference; omit the constant, which is already explicit.
    expected_drift = idct(np.eye(n_frames)[:, 1 : n_drift + 1], norm="ortho", axis=0)
    np.testing.assert_allclose(design[:, 15:], expected_drift, atol=1e-14)


@pytest.mark.parametrize("n_frames,n_drift", [(176, 1), (405, 4), (232, 2)])
def test_dct_glm_matches_filtering_both_data_and_model(
    n_frames: int, n_drift: int
) -> None:
    rng = np.random.default_rng(9)
    task = rng.normal(size=(n_frames, 2))
    nuisance = rng.normal(size=(n_frames, 12))
    base = np.column_stack([task, nuisance, np.ones(n_frames)])
    drift = idct(np.eye(n_frames)[:, 1 : n_drift + 1], norm="ortho", axis=0)
    y = base @ rng.normal(size=(15, 3))
    y += drift @ rng.normal(scale=20.0, size=(n_drift, 3))
    y += rng.normal(size=y.shape)

    # spm_filter: remove the same low-frequency subspace from Y and X.
    filtered_y = y - drift @ (drift.T @ y)
    filtered_base = base - drift @ (drift.T @ base)
    expected_beta = np.linalg.lstsq(filtered_base, filtered_y, rcond=None)[0]
    expected_residual = filtered_y - filtered_base @ expected_beta

    design = build_first_level_design(task, nuisance)
    beta, residual, _ = fit_first_level_glm(y.T, design)
    np.testing.assert_allclose(beta[:, :15], expected_beta.T, atol=1e-11)
    np.testing.assert_allclose(residual, expected_residual.T, atol=1e-11)
    np.testing.assert_allclose(residual @ design, 0.0, atol=1e-10)


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
    np.testing.assert_allclose(resid @ design, 0.0, atol=1e-10)


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
