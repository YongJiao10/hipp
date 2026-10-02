"""Algebraic tests for the frozen rest-FC group estimands."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd


SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))

from contrasts import (  # noqa: E402
    JJTMS_CELLS,
    JJTMS_ESTIMANDS,
    STANFORD_CELLS,
    STANFORD_ESTIMANDS,
    jjtms_contrast_matrix,
    stanford_contrast_matrix,
)


def _estimate(matrix: pd.DataFrame, means: dict[str, float]) -> pd.Series:
    """Evaluate each row against arbitrary cell means in table order."""

    return matrix.loc[:, list(means)].to_numpy(dtype=float) @ np.array(
        [means[cell] for cell in means], dtype=float
    )


def test_stanford_matrix_has_frozen_ids_cells_and_numeric_coefficients() -> None:
    matrix = stanford_contrast_matrix()

    assert list(matrix.columns) == ["estimand", *STANFORD_CELLS]
    assert matrix["estimand"].tolist() == list(STANFORD_ESTIMANDS)
    assert matrix["estimand"].is_unique
    assert matrix[list(STANFORD_CELLS)].dtypes.apply(
        lambda dtype: np.issubdtype(dtype, np.number)
    ).all()


def test_stanford_matrix_recovers_exact_formulas_from_non_symmetric_means() -> None:
    means = {"NTHC": 1.25, "TEHC": 8.5, "NTS": 23.75, "TIS": 101.0}
    matrix = stanford_contrast_matrix()
    estimates = dict(zip(matrix["estimand"], _estimate(matrix, means)))

    assert estimates["symptom_marginal"] == (
        0.5 * (means["NTS"] + means["TIS"])
        - 0.5 * (means["NTHC"] + means["TEHC"])
    )
    assert estimates["trauma_marginal"] == (
        0.5 * (means["TEHC"] + means["TIS"])
        - 0.5 * (means["NTHC"] + means["NTS"])
    )
    assert estimates["interaction"] == (
        (means["TIS"] - means["TEHC"])
        - (means["NTS"] - means["NTHC"])
    )
    assert estimates["NTS-NTHC"] == means["NTS"] - means["NTHC"]
    assert estimates["TIS-TEHC"] == means["TIS"] - means["TEHC"]
    assert estimates["TEHC-NTHC"] == means["TEHC"] - means["NTHC"]
    assert estimates["TIS-NTS"] == means["TIS"] - means["NTS"]


def test_stanford_contrasts_are_zero_sum_and_marginals_are_not_reference_effects() -> None:
    matrix = stanford_contrast_matrix().set_index("estimand")
    assert np.allclose(matrix.sum(axis=1).to_numpy(dtype=float), 0.0)

    # A deliberately asymmetric table makes an accidental reference-cell
    # coding visibly different from each marginal estimand.
    means = {"NTHC": 2.0, "TEHC": 11.0, "NTS": 37.0, "TIS": 109.0}
    estimates = _estimate(matrix.reset_index(), means)
    row_ids = matrix.index.tolist()
    values = dict(zip(row_ids, estimates))
    assert values["symptom_marginal"] != values["NTS-NTHC"]
    assert values["symptom_marginal"] != values["TIS-TEHC"]
    assert values["trauma_marginal"] != values["TEHC-NTHC"]
    assert values["trauma_marginal"] != values["TIS-NTS"]
    assert not np.array_equal(
        matrix.loc["symptom_marginal"].to_numpy(dtype=float),
        matrix.loc["NTS-NTHC"].to_numpy(dtype=float),
    )


def test_jjtms_matrix_has_frozen_ids_cells_and_patient_positive_sign() -> None:
    matrix = jjtms_contrast_matrix()

    assert list(matrix.columns) == ["estimand", *JJTMS_CELLS]
    assert matrix["estimand"].tolist() == list(JJTMS_ESTIMANDS)
    assert matrix["estimand"].is_unique
    assert matrix["patient"].iat[0] == 1.0
    assert matrix["healthy"].iat[0] == -1.0
    assert matrix.loc[:, list(JJTMS_CELLS)].dtypes.apply(
        lambda dtype: np.issubdtype(dtype, np.number)
    ).all()

    means = {"healthy": -3.5, "patient": 12.25}
    assert _estimate(matrix, means)[0] == means["patient"] - means["healthy"]
    assert np.allclose(matrix.loc[:, list(JJTMS_CELLS)].sum(axis=1), 0.0)


def test_contrast_matrix_calls_return_fresh_copy_safe_snapshots() -> None:
    first = stanford_contrast_matrix()
    pristine = stanford_contrast_matrix()
    assert first is not pristine
    pd.testing.assert_frame_equal(first, pristine)

    first.loc[0, "NTHC"] = 999.0
    first.loc[0, "estimand"] = "caller-mutation"
    second = stanford_contrast_matrix()
    pd.testing.assert_frame_equal(second, pristine)

    second.loc[0, "TEHC"] = -999.0
    third = stanford_contrast_matrix()
    pd.testing.assert_frame_equal(third, pristine)

    jjtms = jjtms_contrast_matrix()
    jjtms.loc[0, "patient"] = -999.0
    assert jjtms_contrast_matrix().loc[0, "patient"] == 1.0
