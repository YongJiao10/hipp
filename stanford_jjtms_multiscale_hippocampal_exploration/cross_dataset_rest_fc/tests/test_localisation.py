"""RED-first tests for primary region x network localisation."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy import stats


SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))

from contrasts import jjtms_contrast_matrix, stanford_contrast_matrix  # noqa: E402
from localisation import fit_local_effects  # noqa: E402


REGIONS = ("anterior", "middle", "posterior")
MODES = ("bilateral", "L", "R", "L-R")
GROUPS = {
    "NTHC": (0, 0),
    "TEHC": (1, 0),
    "NTS": (0, 1),
    "TIS": (1, 1),
}


def _subject_rows(dataset: str, group: str, index: int, resolution: int) -> list[dict[str, object]]:
    subject = f"sub-{index:04d}"
    trauma, symptomatic = GROUPS.get(group, (0, 0))
    base_values = {
        "NTHC": 1.0,
        "TEHC": 4.0,
        "NTS": 10.0,
        "TIS": 20.0,
        "healthy": -3.0,
        "patient": 6.0,
    }
    rows: list[dict[str, object]] = []
    networks = tuple(f"N{network:02d}" for network in range(1, resolution + 1))
    for mode in MODES:
        for region_index, region in enumerate(REGIONS):
            for network_index, network in enumerate(networks):
                # Non-symmetric values ensure that a reference-cell or pooled-
                # SD implementation cannot accidentally pass these tests.
                value = float(
                    base_values[group]
                    + 0.4 * index
                    + 0.7 * region_index
                    + 0.03 * network_index
                    + {"bilateral": 0.0, "L": 2.0, "R": -1.0, "L-R": 8.0}[mode]
                )
                rows.append(
                    {
                        "dataset": dataset,
                        "subject": subject,
                        "group": group,
                        "trauma": trauma,
                        "symptomatic": symptomatic,
                        "method": "hcp_fixed",
                        "smoothing_mm": 2,
                        "hemisphere": mode,
                        "hemisphere_mode": mode,
                        "region": region,
                        "network_resolution": resolution,
                        "network": network,
                        "fc_estimator": "fisher_z",
                        "fisher_z": value,
                        "status": "present",
                        "failure_reason": "",
                    }
                )
    return rows

def _stanford_views(*, resolutions: tuple[int, ...] = (8,)) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    index = 1
    for group in GROUPS:
        for _ in range(3):
            for resolution in resolutions:
                rows.extend(_subject_rows("stanford", group, index, resolution))
            index += 1
    frame = pd.DataFrame(rows)
    frame.attrs["global_p"] = 0.99
    frame.attrs["network_ordinals"] = {
        ("hcp_fixed", resolution, f"N{network:02d}"): network - 1
        for resolution in (8, 17)
        for network in range(1, resolution + 1)
    }
    return frame


def _jjtms_views(*, resolutions: tuple[int, ...] = (8,)) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    index = 1
    for group in ("healthy", "patient"):
        for _ in range(3):
            # Reuse the Stanford coding only to make the helper compact; the
            # localisation implementation must use the frozen JJTMS labels.
            rows.extend(_subject_rows("jjtms", group, index, resolutions[0]))
            index += 1
    frame = pd.DataFrame(rows)
    frame["trauma"] = pd.NA
    frame["symptomatic"] = pd.NA
    frame.attrs["network_ordinals"] = {
        ("hcp_fixed", resolution, f"N{network:02d}"): network - 1
        for resolution in (8, 17)
        for network in range(1, resolution + 1)
    }
    return frame


def test_stanford_cell_means_recover_all_frozen_contrasts_with_hc3_t_inference() -> None:
    tensor = _stanford_views()
    result = fit_local_effects(tensor, stanford_contrast_matrix())
    selected = result.query(
        "dataset == 'stanford' and resolution == 8 and hemisphere_mode == 'bilateral' "
        "and region == 'anterior' and network == 'N01'"
    ).set_index("estimand")

    means = {
        "NTHC": np.mean([1.0 + 0.4 * index for index in (1, 2, 3)]),
        "TEHC": np.mean([4.0 + 0.4 * index for index in (4, 5, 6)]),
        "NTS": np.mean([10.0 + 0.4 * index for index in (7, 8, 9)]),
        "TIS": np.mean([20.0 + 0.4 * index for index in (10, 11, 12)]),
    }
    weights = stanford_contrast_matrix().set_index("estimand")
    for estimand, row in selected.iterrows():
        assert row["coefficient"] == pytest.approx(
            float(weights.loc[estimand, list(GROUPS)] @ pd.Series(means))
        )
        assert np.isfinite(row["hc3_se"])
        assert np.isfinite(row["ci_low"])
        assert np.isfinite(row["ci_high"])
        assert np.isfinite(row["p_raw"])
        assert 0.0 <= row["p_raw"] <= 1.0
        assert np.isfinite(row["standardized_effect"])
        assert row["failure_reason"] == ""

    # The first contrast's HC3 value is checked against the direct cell-means
    # sandwich calculation and finite-residual-df t inference.
    first = selected.loc["symptom_marginal"]
    y_by_group = {
        group: np.asarray(
            [
                {"NTHC": 1.0, "TEHC": 4.0, "NTS": 10.0, "TIS": 20.0}[group]
                + 0.4 * index
                for index in range(
                    {"NTHC": 1, "TEHC": 4, "NTS": 7, "TIS": 10}[group],
                    {"NTHC": 4, "TEHC": 7, "NTS": 10, "TIS": 13}[group],
                )
            ],
            dtype=float,
        )
        for group in GROUPS
    }
    contrast = weights.loc["symptom_marginal", list(GROUPS)].to_numpy(dtype=float)
    variance = sum(
        contrast[index] ** 2
        * np.sum((values - values.mean()) ** 2)
        / ((1.0 - 1.0 / len(values)) ** 2 * len(values) ** 2)
        for index, values in enumerate(y_by_group.values())
    )
    expected_se = np.sqrt(variance)
    df_resid = sum(map(len, y_by_group.values())) - len(y_by_group)
    expected_t = float(first["coefficient"] / expected_se)
    expected_p = 2.0 * stats.t.sf(abs(expected_t), df_resid)
    expected_sd = np.sqrt(
        sum(np.sum((values - values.mean()) ** 2) for values in y_by_group.values())
        / df_resid
    )
    assert first["hc3_se"] == pytest.approx(expected_se)
    assert first["p_raw"] == pytest.approx(expected_p)
    assert first["standardized_effect"] == pytest.approx(first["coefficient"] / expected_sd)


def test_localisation_defers_multiplicity_until_analysis_set_is_frozen() -> None:
    result = fit_local_effects(_jjtms_views(), jjtms_contrast_matrix())

    assert result["q_bh"].isna().all()
    assert set(result["correction_status"]) == {
        "deferred_until_analysis_set_freeze"
    }
    assert result.attrs["multiplicity_correction"] == "deferred"
    assert result.attrs["analysis_object_set"] == "not_frozen"
    assert "planned_family_sizes" not in result.attrs


def test_jjtms_patient_minus_healthy_and_lr_passthrough_do_not_use_global_gate() -> None:
    tensor = _jjtms_views()
    tensor.attrs["global_p"] = 1.0
    tensor.loc[
        tensor["hemisphere_mode"].eq("L-R") & tensor["group"].eq("patient"),
        "fisher_z",
    ] += 5.0
    result = fit_local_effects(tensor, jjtms_contrast_matrix())
    selected = result.query(
        "dataset == 'jjtms' and resolution == 8 and region == 'anterior' and network == 'N01'"
    ).set_index("hemisphere_mode")
    # The L-R rows already contain within-subject differences.  Their model
    # coefficient must therefore be the patient-minus-healthy difference of
    # those values, with no laterality re-creation or group-mean subtraction.
    expected_healthy = np.mean([-3.0 + 0.4 * index + 8.0 for index in (1, 2, 3)])
    expected_patient = np.mean([6.0 + 0.4 * index + 8.0 + 5.0 for index in (4, 5, 6)])
    assert selected.loc["L-R", "coefficient"] == pytest.approx(
        expected_patient - expected_healthy
    )
    assert selected.loc["L-R", "failure_reason"] == ""
    assert selected.loc["bilateral", "coefficient"] > 0.0
    assert np.isfinite(selected.loc["bilateral", "p_raw"])


def test_network_labels_and_ordinals_must_match_task2_frozen_metadata() -> None:
    tensor = _jjtms_views()
    tensor.loc[tensor["network"].eq("N08"), "network"] = "replacement"
    with pytest.raises(ValueError, match="network metadata|network label|planned"):
        fit_local_effects(tensor, jjtms_contrast_matrix())

    tensor = _jjtms_views()
    metadata = dict(tensor.attrs["network_ordinals"])
    metadata[("hcp_fixed", 8, "N08")] = metadata[("hcp_fixed", 8, "N07")]
    tensor.attrs["network_ordinals"] = metadata
    with pytest.raises(ValueError, match="ordinal|metadata"):
        fit_local_effects(tensor, jjtms_contrast_matrix())


def test_unknown_status_fails_closed_instead_of_becoming_missing() -> None:
    tensor = _jjtms_views()
    tensor.loc[tensor.index[0], "status"] = "mystery"
    with pytest.raises(ValueError, match="status"):
        fit_local_effects(tensor, jjtms_contrast_matrix())


def test_subject_cannot_change_frozen_group_across_components() -> None:
    tensor = _jjtms_views()
    subject = tensor.loc[0, "subject"]
    first_subject_row = tensor.index[tensor["subject"].eq(subject)][0]
    tensor.loc[first_subject_row, "group"] = "patient"
    with pytest.raises(ValueError, match="group assignment|conflicting group"):
        fit_local_effects(tensor, jjtms_contrast_matrix())


def test_hemisphere_and_mode_must_match_task2_collapsed_representation() -> None:
    tensor = _jjtms_views()
    tensor.loc[tensor.index[0], "hemisphere"] = "R"
    with pytest.raises(ValueError, match="hemisphere"):
        fit_local_effects(tensor, jjtms_contrast_matrix())


def test_empty_tensor_fails_closed() -> None:
    with pytest.raises(ValueError, match="empty"):
        fit_local_effects(pd.DataFrame(), jjtms_contrast_matrix())


def test_failed_component_is_retained_without_provisional_correction() -> None:
    tensor = _jjtms_views()
    unavailable = (tensor["network"] == "N08") & (tensor["region"] == "posterior")
    tensor.loc[unavailable, "status"] = "unavailable"
    tensor.loc[unavailable, "fisher_z"] = np.nan
    tensor.loc[unavailable, "failure_reason"] = "audited unavailable component"
    result = fit_local_effects(tensor, jjtms_contrast_matrix())
    failed = result.query(
        "network == 'N08' and region == 'posterior' and hemisphere_mode == 'bilateral'"
    ).iloc[0]
    assert pd.isna(failed["p_raw"])
    assert pd.isna(failed["q_bh"])
    assert "unavailable" in failed["failure_reason"]



def test_duplicate_subject_key_after_normalization_fails_closed() -> None:
    tensor = _jjtms_views()
    duplicate = tensor.iloc[[0]].copy()
    duplicate["subject"] = "1"
    tensor = pd.concat([tensor, duplicate], ignore_index=True)
    with pytest.raises(ValueError, match="duplicate subject"):
        fit_local_effects(tensor, jjtms_contrast_matrix())


def test_missing_required_group_is_a_failure_row_not_a_dropped_component() -> None:
    tensor = _jjtms_views()
    target = (tensor["network"] == "N01") & (tensor["region"] == "middle")
    tensor = tensor.loc[~(target & tensor["group"].eq("patient"))].copy()
    result = fit_local_effects(tensor, jjtms_contrast_matrix())
    row = result.query(
        "network == 'N01' and region == 'middle' and hemisphere_mode == 'bilateral'"
    ).iloc[0]
    assert pd.isna(row["coefficient"])
    assert pd.isna(row["p_raw"])
    assert pd.isna(row["q_bh"])
    assert "patient" in row["failure_reason"]


@pytest.mark.parametrize("resolution, expected_m", [(8, 24), (17, 51)])
def test_localisation_family_keeps_three_regions_times_all_networks(
    resolution: int, expected_m: int
) -> None:
    tensor = _jjtms_views(resolutions=(resolution,))
    result = fit_local_effects(tensor, jjtms_contrast_matrix())
    family = result.query(
        "dataset == 'jjtms' and estimand == 'patient-healthy' "
        "and hemisphere_mode == 'bilateral' and method == 'hcp_fixed' "
        "and smoothing_mm == 2"
    )
    assert len(family) == expected_m
    assert family["q_bh"].isna().all()
