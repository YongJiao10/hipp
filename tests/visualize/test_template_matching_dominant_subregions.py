"""Contract tests for standard and CtxReg dominant-subregion rendering."""

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
RENDERER_PATH = (
    REPO_ROOT
    / "scripts"
    / "visualize"
    / "template_matching_dominant_subregions"
    / "render_template_matching_dominant_subregions.py"
)
SPEC = importlib.util.spec_from_file_location("dominant_subregions_renderer", RENDERER_PATH)
assert SPEC is not None and SPEC.loader is not None
RENDERER = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = RENDERER
SPEC.loader.exec_module(RENDERER)


EXPECTED_DISPLAY_NAMES = {
    1: "Anterior / VAN",
    2: "Middle / SMN",
    3: "Posterior / DMN",
}


@pytest.mark.parametrize("signal_branch", ["standard", "ctxreg"])
def test_labels_use_exact_template_matching_semantics(signal_branch):
    assert RENDERER.load_branch_subregion_names(signal_branch) == EXPECTED_DISPLAY_NAMES


def test_branch_matching_roots_are_explicit_and_distinct(tmp_path):
    roots = RENDERER.branch_matching_roots(tmp_path)
    assert roots == {
        "standard": tmp_path / "outputs_iowa_jj_tms" / "work" / "hcp151_fixed_template_matching" / "subjects",
        "ctxreg": tmp_path / "outputs_iowa_jj_tms" / "work" / "hcp151_fixed_template_matching" / "pfm_ctxreg" / "subjects",
    }
    assert roots["standard"] != roots["ctxreg"]


def test_dual_stat_cols_replace_vertex_count():
    assert RENDERER.DUAL_STAT_COLS == ("area_pct_adj", "area_pct_welch")
    assert "count" not in RENDERER.DUAL_STAT_COLS


def test_stanford_cohort_counts_are_exact():
    cohorts = RENDERER.stanford_cohorts()
    assert [c["key"] for c in cohorts] == [
        "stan_nthc",
        "stan_tehc",
        "stan_nts",
        "stan_nts_any_mdd",
        "stan_nts_pure_mdd",
        "stan_tis",
        "stan_tis_ptsd",
    ]
    assert {c["key"]: len(c["subjects"]) for c in cohorts} == dict(
        RENDERER.STANFORD_EXPECTED_N
    )
    assert RENDERER.STAN_REF == "stan_nthc"


def test_stanford_roster_fully_covered_by_demo_and_tm():
    cohorts = RENDERER.stanford_cohorts()
    demo = RENDERER.load_stanford_age_sex()
    subjects = [s for c in cohorts for s in c["subjects"]]
    assert len(set(subjects)) == 194
    for sub in subjects:
        assert sub in demo
        age, female = demo[sub]
        assert np.isfinite(age) and female in (0.0, 1.0)


def test_iowa_demo_covers_branch_rosters():
    demo = RENDERER.load_iowa_age_sex()
    for cohort in RENDERER.COHORTS_DEF:
        for sub in cohort["subjects"]:
            assert sub in demo


def test_ols_group_test_detects_adjusted_effect():
    rng = np.random.default_rng(7)
    age_ctrl = rng.normal(33, 5, 60)
    age_case = rng.normal(33, 5, 40)
    # Pure group shift, no covariate confounding.
    ctrl = rng.normal(30, 2, 60)
    case = rng.normal(36, 2, 40)
    cov_ctrl = np.column_stack([age_ctrl, rng.binomial(1, 0.5, 60).astype(float)])
    cov_case = np.column_stack([age_case, rng.binomial(1, 0.5, 40).astype(float)])
    res = RENDERER.ols_group_test(case, ctrl, cov_case, cov_ctrl)
    assert res["t"] > 0
    assert res["p"] < 0.001


def test_ols_group_test_removes_confounded_effect():
    rng = np.random.default_rng(11)
    # y depends only on age; case group is older on average.
    age_ctrl = rng.normal(25, 3, 60)
    age_case = rng.normal(35, 3, 40)
    ctrl = 20 + 0.5 * age_ctrl + rng.normal(0, 1, 60)
    case = 20 + 0.5 * age_case + rng.normal(0, 1, 40)
    cov_ctrl = np.column_stack([age_ctrl, rng.binomial(1, 0.5, 60).astype(float)])
    cov_case = np.column_stack([age_case, rng.binomial(1, 0.5, 40).astype(float)])
    res = RENDERER.ols_group_test(case, ctrl, cov_case, cov_ctrl)
    assert res["p"] > 0.05


def test_ols_group_test_fails_closed():
    rng = np.random.default_rng(3)
    ctrl = rng.normal(0, 1, 10)
    case = rng.normal(0, 1, 8)
    cov = np.column_stack([rng.normal(0, 1, 10), np.zeros(10)])
    cov_case = np.column_stack([rng.normal(0, 1, 8), np.zeros(8)])
    with pytest.raises(ValueError):
        RENDERER.ols_group_test(
            np.array([1.0, np.nan]), ctrl[:2], cov_case[:2], cov[:2]
        )
    # Rank-deficient: constant covariate column after centering.
    const_ctrl = np.column_stack([np.ones(10), np.zeros(10)])
    const_case = np.column_stack([np.ones(8), np.zeros(8)])
    with pytest.raises(ValueError):
        RENDERER.ols_group_test(case, ctrl, const_case, const_ctrl)



def test_area_percentages_use_native_geometry_even_when_corobl_is_present(tmp_path, monkeypatch):
    import nibabel as nib
    from lib.workbench import resolve_wb_command

    monkeypatch.setattr(RENDERER, "WB_COMMAND", str(resolve_wb_command()))
    native = np.zeros((512, 3), dtype=np.float32)
    native[:6] = [[0, 0, 0], [2, 0, 0], [0, 1, 0],
                  [0, 0, 2], [0, 2, 2], [0, 0, 3]]
    faces = np.array([[0, 1, 2], [3, 4, 5]], dtype=np.int32)
    labels = np.full(512, 3, dtype=np.int32)
    labels[:3], labels[3:6] = 1, 2
    for hemi in ('L', 'R'):
        surf = tmp_path/'batch/sub-01/hippunfold/sub-01/surf'
        surf.mkdir(parents=True, exist_ok=True)
        for space, points in [('T1w', native), ('corobl', native * [3, 1, 1])]:
            nib.save(nib.gifti.GiftiImage(darrays=[
                nib.gifti.GiftiDataArray(np.asarray(points, dtype=np.float32), intent='NIFTI_INTENT_POINTSET'),
                nib.gifti.GiftiDataArray(faces, intent='NIFTI_INTENT_TRIANGLE'),
            ]), surf/f'sub-01_hemi-{hemi}_space-{space}_den-512_label-hipp_midthickness.surf.gii')
        combo = tmp_path/'subjects/sub-01/matching/2mm'/f'hemi-{hemi}'
        combo.mkdir(parents=True)
        np.save(combo/'labels.npy', labels)
    result = RENDERER.compute_subject_counts_areas(
        '2mm', ['sub-01'], tmp_path/'subjects', tmp_path/'cache',
        hippunfold_batch_root=tmp_path/'batch',
    )
    for hemi in ('L', 'R'):
        np.testing.assert_allclose(result['sub-01']['area_pct'][hemi], [50, 50, 0], atol=1e-5)
