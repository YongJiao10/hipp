"""Frozen group estimands for the resting-state Fisher-z FC analysis.

The tables in this module are coefficient matrices only.  They do not read
subject data, fit models, calculate uncertainty, or perform multiplicity
correction.  Each public function returns a fresh DataFrame so a caller can
annotate or otherwise mutate its snapshot without changing a later call.
"""

from __future__ import annotations

from collections.abc import Sequence

import pandas as pd


STANFORD_CELLS = ("NTHC", "TEHC", "NTS", "TIS")
JJTMS_CELLS = ("healthy", "patient")

STANFORD_ESTIMANDS = (
    "symptom_marginal",
    "trauma_marginal",
    "interaction",
    "NTS-NTHC",
    "TIS-TEHC",
    "TEHC-NTHC",
    "TIS-NTS",
)
JJTMS_ESTIMANDS = ("patient-healthy",)

# Keep the scientific definitions as literal tuples rather than deriving one
# contrast from another.  This makes the frozen ordering and signs auditable.
_STANFORD_WEIGHTS = (
    (-0.5, -0.5, 0.5, 0.5),  # symptom_marginal
    (-0.5, 0.5, -0.5, 0.5),  # trauma_marginal
    (1.0, -1.0, -1.0, 1.0),  # interaction
    (-1.0, 0.0, 1.0, 0.0),  # NTS-NTHC
    (0.0, -1.0, 0.0, 1.0),  # TIS-TEHC
    (-1.0, 1.0, 0.0, 0.0),  # TEHC-NTHC
    (0.0, 0.0, -1.0, 1.0),  # TIS-NTS
)
_JJTMS_WEIGHTS = ((-1.0, 1.0),)  # patient-healthy


def _build_snapshot(
    estimands: Sequence[str],
    cells: Sequence[str],
    weights: Sequence[Sequence[float]],
) -> pd.DataFrame:
    """Build one independent, deterministically ordered coefficient table."""

    frame = pd.DataFrame(weights, columns=list(cells), dtype=float)
    frame.insert(0, "estimand", list(estimands))
    frame.attrs = {
        "immutable": True,
        "cell_order": tuple(cells),
        "estimand_order": tuple(estimands),
    }
    return frame


def stanford_contrast_matrix() -> pd.DataFrame:
    """Return the frozen Stanford four-cell coefficient matrix.

    The cell columns are ordered ``NTHC, TEHC, NTS, TIS`` and rows are ordered
    by :data:`STANFORD_ESTIMANDS`.  Positive values follow the first named
    group/higher factor minus its comparator.
    """

    return _build_snapshot(STANFORD_ESTIMANDS, STANFORD_CELLS, _STANFORD_WEIGHTS)


def jjtms_contrast_matrix() -> pd.DataFrame:
    """Return the frozen JJTMS patient-minus-healthy coefficient matrix."""

    return _build_snapshot(JJTMS_ESTIMANDS, JJTMS_CELLS, _JJTMS_WEIGHTS)


__all__ = [
    "JJTMS_CELLS",
    "JJTMS_ESTIMANDS",
    "STANFORD_CELLS",
    "STANFORD_ESTIMANDS",
    "jjtms_contrast_matrix",
    "stanford_contrast_matrix",
]
