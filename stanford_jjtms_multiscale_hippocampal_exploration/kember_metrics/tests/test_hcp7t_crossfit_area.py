"""Tests for HCP 7T held-out cross-fit area inputs."""

from pathlib import Path

import pandas as pd
import pytest

SRC = Path(__file__).resolve().parents[1] / "src"

import sys

sys.path.insert(0, str(SRC))

import hcp7t_crossfit_area as hcp_area  # noqa: E402


def _run_manifest(*, family_aware: bool = True, n_folds: int = 5) -> dict[str, object]:
    return {
        "n_subjects": 151,
        "n_folds": n_folds,
        "template_fits_completed": 20,
        "family": {
            "family_aware": family_aware,
            "subject_only_waiver": False,
        },
    }


def _combo_manifest(*, held_out: bool = True) -> dict[str, object]:
    return {
        "status": "ok",
        "subject": "sub-100610",
        "hemi": "L",
        "smoothing": "2mm",
        "subject_was_in_template": not held_out,
        "template_fold_id": "fold-02",
        "template_training_subjects": ["sub-105923", "sub-108323"],
        "label_semantics": {
            "1": "Anterior",
            "2": "Middle",
            "3": "Posterior",
        },
    }


def test_crossfit_run_manifest_requires_family_aware_five_fold_contract() -> None:
    with pytest.raises(ValueError, match="family-aware"):
        hcp_area.validate_crossfit_run_manifest(_run_manifest(family_aware=False))

    with pytest.raises(ValueError, match="five-fold"):
        hcp_area.validate_crossfit_run_manifest(_run_manifest(n_folds=4))


def test_crossfit_combo_manifest_requires_subject_to_be_held_out() -> None:
    validated = hcp_area.validate_crossfit_combo_manifest(
        _combo_manifest(), subject="sub-100610", hemisphere="L", smoothing_mm=2
    )
    assert validated["template_fold_id"] == "fold-02"

    with pytest.raises(ValueError, match="held out"):
        hcp_area.validate_crossfit_combo_manifest(
            _combo_manifest(held_out=False),
            subject="sub-100610",
            hemisphere="L",
            smoothing_mm=2,
        )


def test_hcp_reference_effect_is_explicitly_hcp_minus_nthc() -> None:
    def complete_table(cohort: str, subjects: list[str], values: list[float]) -> pd.DataFrame:
        rows = []
        for subject, value in zip(subjects, values):
            for hemisphere in hcp_area.HEMISPHERES:
                for system in hcp_area.SYSTEMS:
                    rows.append({
                        "subject": subject,
                        "cohort": cohort,
                        "hemisphere": hemisphere,
                        "system": system,
                        "surface_area_mm2": value,
                    })
        return pd.DataFrame(rows)

    hcp_features = complete_table("HCP 7T", ["sub-100610", "sub-100611"], [10.0, 12.0])
    nthc_features = complete_table("NTHC", ["sub-1001", "sub-1002"], [8.0, 9.0])

    effects = hcp_area.compute_hcp7t_reference_effect(hcp_features, nthc_features)

    row = effects.iloc[0]
    assert row["contrast"] == "HCP 7T−NTHC"
    assert row["case_group"] == "HCP 7T"
    assert row["reference_group"] == "NTHC"
    assert row["mean_difference"] == pytest.approx(2.5)
    assert row["case_n"] == 2
    assert row["reference_n"] == 2

def test_hcp_asset_fingerprint_changes_when_an_input_changes(tmp_path: Path) -> None:
    asset = tmp_path / "crossfit_label.npy"
    asset.write_bytes(b"first")
    first, first_records = hcp_area._fingerprint_assets([("crossfit_labels", asset)])

    asset.write_bytes(b"second")
    second, second_records = hcp_area._fingerprint_assets([("crossfit_labels", asset)])

    assert first != second
    assert first_records[0]["sha256"] != second_records[0]["sha256"]

