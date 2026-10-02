"""RED-first tests for auxiliary subject-profile omnibus distances."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))

from contrasts import jjtms_contrast_matrix, stanford_contrast_matrix  # noqa: E402
from global_profile import (  # noqa: E402
    _fit_reduced,
    _null_basis,
    run_global_tests,
)
from localisation import fit_local_effects  # noqa: E402


REGIONS = ("anterior", "body", "tail")
GROUPS = ("NTHC", "TEHC", "NTS", "TIS")


def _profile_rows(
    dataset: str = "jjtms",
    *,
    resolutions: tuple[int, ...] = (8,),
    methods: tuple[str, ...] = ("hcp151_fixed",),
    include_stanford: bool = False,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    groups = ("healthy", "patient") if dataset == "jjtms" else GROUPS
    group_base = {
        "healthy": 1.0,
        "patient": 5.0,
        "NTHC": 1.0,
        "TEHC": 4.0,
        "NTS": 9.0,
        "TIS": 15.0,
    }
    subject_number = 1
    for group_index, group in enumerate(groups):
        for replicate in range(3):
            subject = f"sub-{subject_number:04d}"
            subject_number += 1
            for method_index, method in enumerate(methods):
                for resolution in resolutions:
                    for region_index, region in enumerate(REGIONS):
                        for network_index in range(resolution):
                            # Shared subject variation across the complete
                            # profile makes cell-wise shuffling observably
                            # different from subject-row exchangeability.
                            value = (
                                group_base[group]
                                + 0.35 * (replicate + 1)
                                + 0.11 * group_index * (replicate + 1)
                                + 0.17 * region_index
                                + 0.013 * network_index
                                + 0.07 * method_index
                            )
                            trauma = {
                                "NTHC": 0,
                                "TEHC": 1,
                                "NTS": 0,
                                "TIS": 1,
                            }.get(group, pd.NA)
                            symptomatic = {
                                "NTHC": 0,
                                "TEHC": 0,
                                "NTS": 1,
                                "TIS": 1,
                            }.get(group, pd.NA)
                            rows.append(
                                {
                                    "dataset": dataset,
                                    "subject": subject,
                                    "group": group,
                                    "trauma": trauma,
                                    "symptomatic": symptomatic,
                                    "method": method,
                                    "smoothing_mm": 2,
                                    "hemisphere": "bilateral",
                                    "hemisphere_mode": "bilateral",
                                    "region": region,
                                    "network_resolution": resolution,
                                    "network": f"N{network_index + 1:02d}",
                                    "fc_estimator": "fisher_z",
                                    "fisher_z": value,
                                    "status": "present",
                                    "failure_reason": "",
                                }
                            )
    frame = pd.DataFrame(rows)
    frame.attrs["network_ordinals"] = {
        (method, resolution, f"N{network:02d}"): network - 1
        for method in methods
        for resolution in (8, 17)
        for network in range(1, resolution + 1)
    }
    if include_stanford:
        stanford = _profile_rows("stanford", resolutions=resolutions, methods=methods)
        jjtms_attrs = dict(frame.attrs)
        frame = pd.concat([frame, stanford], ignore_index=True)
        frame.attrs["network_ordinals"] = {
            **jjtms_attrs,
            **stanford.attrs["network_ordinals"],
        }
    return frame


def test_output_has_separate_subregion_and_whole_profile_dimensions() -> None:
    tensor = _profile_rows(resolutions=(8, 17))
    result = run_global_tests(tensor, jjtms_contrast_matrix(), seed=17, n_perm=5)

    assert list(result.columns) == [
        "dataset",
        "estimand",
        "method",
        "smoothing_mm",
        "hemisphere_mode",
        "network_resolution",
        "fc_estimator",
        "profile_scope",
        "region",
        "n_total",
        "n_complete",
        "effective_dimension",
        "statistic",
        "p_perm",
        "n_perm",
        "seed",
        "derived_seed",
        "failure_reason",
    ]
    assert set(result["profile_scope"]) == {
        "subregion_profile",
        "whole_hippocampus_profile",
    }
    assert set(result.loc[result["profile_scope"] == "subregion_profile", "region"]) == set(
        REGIONS
    )
    assert set(result.loc[result["profile_scope"] == "whole_hippocampus_profile", "region"]) == {
        "all"
    }
    dimensions = result.set_index(["network_resolution", "profile_scope"]).sort_index()[
        "effective_dimension"
    ]
    for resolution in (8, 17):
        assert set(dimensions.loc[(resolution, "subregion_profile")]) == {resolution}
        assert set(dimensions.loc[(resolution, "whole_hippocampus_profile")]) == {3 * resolution}


def test_exact_non_symmetric_contrast_distance_uses_component_residual_sd() -> None:
    tensor = _profile_rows(resolutions=(8,))
    result = run_global_tests(tensor, jjtms_contrast_matrix(), seed=3, n_perm=0)
    row = result.query(
        "profile_scope == 'subregion_profile' and region == 'anterior'"
    ).iloc[0]

    scope = tensor.loc[
        (tensor["region"] == "anterior") & (tensor["network_resolution"] == 8)
    ]
    component_values = scope.pivot(index="subject", columns="network", values="fisher_z")
    groups = scope[["subject", "group"]].drop_duplicates().set_index("subject")[
        "group"
    ].loc[component_values.index]
    effects: list[float] = []
    residual_sds: list[float] = []
    for network in sorted(component_values.columns):
        means = component_values.groupby(groups).mean(numeric_only=True)[network]
        effects.append(float(means["patient"] - means["healthy"]))
        residual_sum = sum(
            np.sum(
                (
                    component_values.loc[groups == group, network].to_numpy()
                    - means[group]
                )
                ** 2
            )
            for group in ("healthy", "patient")
        )
        residual_sds.append(np.sqrt(residual_sum / (len(component_values) - 2)))
    expected = float(np.sum((np.asarray(effects) / np.asarray(residual_sds)) ** 2))
    assert row["statistic"] == pytest.approx(expected)
    assert row["n_total"] == 6
    assert row["n_complete"] == 6
    assert row["failure_reason"] == ""


def test_freedman_lane_rows_match_reference_and_differ_from_cellwise_control() -> None:
    tensor = _profile_rows(resolutions=(8,))
    n_perm = 37
    user_seed = 29
    result = run_global_tests(tensor, jjtms_contrast_matrix(), seed=user_seed, n_perm=n_perm)
    row = result.query(
        "profile_scope == 'subregion_profile' and region == 'anterior'"
    ).iloc[0]

    scope = tensor.loc[
        (tensor["region"] == "anterior") & (tensor["network_resolution"] == 8)
    ].copy()
    pivot = scope.pivot(index="subject", columns="network", values="fisher_z").sort_index()
    labels = (
        scope[["subject", "group"]]
        .drop_duplicates()
        .set_index("subject")
        .loc[pivot.index]
    )
    label_values = labels["group"].to_numpy()
    x = np.column_stack(
        [
            (label_values == "healthy").astype(float),
            (label_values == "patient").astype(float),
        ]
    )
    y = pivot.to_numpy(dtype=float)
    c = np.asarray([-1.0, 1.0])
    scope_key = (
        "jjtms",
        "patient-healthy",
        "hcp151_fixed",
        2,
        "bilateral",
        8,
        "fisher_z",
        "subregion_profile",
        "anterior",
    )
    payload = json.dumps(
        {"seed": user_seed, "scope": list(scope_key)},
        ensure_ascii=True,
        separators=(",", ":"),
    ).encode("utf-8")
    derived_seed = int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")
    assert row["derived_seed"] == derived_seed

    def full_statistic(values: np.ndarray, contrast: np.ndarray = c) -> float:
        beta = np.linalg.lstsq(x, values, rcond=None)[0]
        residual = values - x @ beta
        residual_sd = np.sqrt(np.sum(residual**2, axis=0) / (len(values) - 2))
        standardized = (contrast @ beta) / residual_sd
        return float(np.dot(standardized, standardized))

    reduced = np.ones((len(y), 1), dtype=float)
    reduced_fit = reduced @ np.linalg.lstsq(reduced, y, rcond=None)[0]
    reduced_residual = y - reduced_fit
    observed_statistic = full_statistic(y)
    row_rng = np.random.default_rng(derived_seed)
    row_statistics = []
    for _ in range(n_perm):
        permutation = row_rng.permutation(len(y))
        row_statistics.append(
            full_statistic(reduced_fit + reduced_residual[permutation, :])
        )
    row_count = sum(statistic >= observed_statistic for statistic in row_statistics)
    row_p = (1.0 + row_count) / (n_perm + 1.0)

    # Negative control: independently shuffle each network residual column.
    # This breaks complete subject profiles and must not match production.
    cell_rng = np.random.default_rng(derived_seed)
    cell_statistics = []
    for _ in range(n_perm):
        cellwise = reduced_fit.copy()
        for column in range(y.shape[1]):
            cellwise[:, column] += reduced_residual[
                cell_rng.permutation(len(y)), column
            ]
        cell_statistics.append(full_statistic(cellwise))
    cell_count = sum(statistic >= observed_statistic for statistic in cell_statistics)
    cell_p = (1.0 + cell_count) / (n_perm + 1.0)

    assert row["statistic"] == pytest.approx(observed_statistic, rel=1e-12, abs=1e-12)
    assert row["p_perm"] == pytest.approx(row_p, rel=1e-12, abs=1e-12)
    assert row_count != cell_count
    assert not np.isclose(row_p, cell_p)


def test_seed_identity_and_branch_order_invariance_are_byte_identical() -> None:
    tensor = _profile_rows(resolutions=(8,), methods=("method_a", "method_b"))
    contrasts = jjtms_contrast_matrix()
    first = run_global_tests(tensor, contrasts, seed=101, n_perm=7)
    repeated = run_global_tests(tensor, contrasts, seed=101, n_perm=7)
    shuffled = run_global_tests(
        tensor.sample(frac=1.0, random_state=11).reset_index(drop=True),
        contrasts,
        seed=101,
        n_perm=7,
    )
    assert first.to_csv(index=False) == repeated.to_csv(index=False)
    assert first.to_csv(index=False) == shuffled.to_csv(index=False)


def test_failed_profiles_are_preserved_without_zero_imputation() -> None:
    tensor = _profile_rows(resolutions=(8,))
    failed = (tensor["region"] == "tail") & (tensor["network"] == "N08")
    tensor.loc[failed, "status"] = "unavailable"
    tensor.loc[failed, "fisher_z"] = np.nan
    tensor.loc[failed, "failure_reason"] = "audited unavailable component"
    result = run_global_tests(tensor, jjtms_contrast_matrix(), seed=7, n_perm=3)

    tail = result.query(
        "profile_scope == 'subregion_profile' and region == 'tail'"
    ).iloc[0]
    whole = result.query("profile_scope == 'whole_hippocampus_profile'").iloc[0]
    assert tail["n_total"] == 6
    assert tail["n_complete"] == 0
    assert whole["n_complete"] == 0
    assert pd.isna(tail["statistic"])
    assert pd.isna(tail["p_perm"])
    assert "audited unavailable component" in tail["failure_reason"]
    assert "audited unavailable component" in whole["failure_reason"]


def test_malformed_task2_schema_and_contrasts_fail_closed() -> None:
    tensor = _profile_rows()
    tensor.loc[tensor.index[0], "status"] = "mystery"
    with pytest.raises(ValueError, match="status"):
        run_global_tests(tensor, jjtms_contrast_matrix(), seed=1, n_perm=1)

    tensor = _profile_rows()
    metadata = dict(tensor.attrs["network_ordinals"])
    metadata[("hcp151_fixed", 8, "N08")] = 0
    tensor.attrs["network_ordinals"] = metadata
    with pytest.raises(ValueError, match="ordinal|metadata"):
        run_global_tests(tensor, jjtms_contrast_matrix(), seed=1, n_perm=1)

    tensor = _profile_rows()
    with pytest.raises(ValueError, match="contrast"):
        run_global_tests(tensor, stanford_contrast_matrix(), seed=1, n_perm=1)


def test_localisation_does_not_read_or_gate_on_global_profile_p_values() -> None:
    tensor = _profile_rows()
    contrasts = jjtms_contrast_matrix()
    before = fit_local_effects(tensor, contrasts)
    global_result = run_global_tests(tensor, contrasts, seed=5, n_perm=2)
    tensor.attrs["global_p"] = global_result["p_perm"].min()
    after = fit_local_effects(tensor, contrasts)
    pd.testing.assert_frame_equal(before, after)


def test_stanford_profile_public_p_matches_contrast_null_not_intercept_null() -> None:
    tensor = _profile_rows("stanford", resolutions=(8,))
    # Retain non-symmetric nuisance cell effects while making both tested
    # contrasts near their null, so reduced-model choice changes the tail.
    target_means = {"NTHC": 0.0, "TEHC": 10.0, "NTS": 0.0, "TIS": 10.0}
    base_means = {"NTHC": 1.0, "TEHC": 4.0, "NTS": 9.0, "TIS": 15.0}
    group_index = {group: index for index, group in enumerate(GROUPS)}
    for group in GROUPS:
        current_mean = base_means[group] + 0.7 + 0.22 * group_index[group]
        tensor.loc[tensor["group"] == group, "fisher_z"] += (
            target_means[group] - current_mean
        )

    contrasts = stanford_contrast_matrix()
    user_seed = 13
    n_perm = 37
    result = run_global_tests(tensor, contrasts, seed=user_seed, n_perm=n_perm)
    repeated = run_global_tests(tensor, contrasts, seed=user_seed, n_perm=n_perm)
    assert result.to_csv(index=False) == repeated.to_csv(index=False)

    scope = tensor.loc[
        (tensor["region"] == "anterior") & (tensor["network_resolution"] == 8)
    ]
    pivot = scope.pivot(index="subject", columns="network", values="fisher_z").sort_index()
    labels = (
        scope[["subject", "group"]]
        .drop_duplicates()
        .set_index("subject")
        .loc[pivot.index]["group"]
        .to_numpy()
    )
    values = pivot.to_numpy(dtype=float)
    design = np.column_stack([(labels == group).astype(float) for group in GROUPS])
    weights = contrasts.set_index("estimand")

    def full_statistic(contrast: np.ndarray, matrix: np.ndarray) -> float:
        beta = np.linalg.lstsq(design, matrix, rcond=None)[0]
        residual = matrix - design @ beta
        residual_sd = np.sqrt(np.sum(residual**2, axis=0) / (len(matrix) - len(GROUPS)))
        standardized = contrast @ beta / residual_sd
        return float(np.dot(standardized, standardized))

    for estimand in ("symptom_marginal", "interaction"):
        contrast = weights.loc[estimand, list(GROUPS)].to_numpy(dtype=float)
        pivot_index = int(np.flatnonzero(np.abs(contrast) > 1e-12)[0])
        reference_basis = []
        for column in range(contrast.size):
            if column == pivot_index:
                continue
            vector = np.zeros(contrast.size, dtype=float)
            vector[column] = 1.0
            vector[pivot_index] = -contrast[column] / contrast[pivot_index]
            reference_basis.append(vector)
        reference_basis = np.column_stack(reference_basis)
        reduced_design = design @ reference_basis
        reduced_fit = reduced_design @ np.linalg.lstsq(
            reduced_design, values, rcond=None
        )[0]
        intercept_design = np.ones((len(values), 1), dtype=float)
        intercept_fit = intercept_design @ np.linalg.lstsq(
            intercept_design, values, rcond=None
        )[0]
        reduced_residual = values - reduced_fit
        intercept_residual = values - intercept_fit

        scope_key = (
            "stanford",
            estimand,
            "hcp151_fixed",
            2,
            "bilateral",
            8,
            "fisher_z",
            "subregion_profile",
            "anterior",
        )
        payload = json.dumps(
            {"seed": user_seed, "scope": list(scope_key)},
            ensure_ascii=True,
            separators=(",", ":"),
        ).encode("utf-8")
        derived_seed = int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")
        rng = np.random.default_rng(derived_seed)
        permutations = [rng.permutation(len(values)) for _ in range(n_perm)]
        observed = full_statistic(contrast, values)
        exact_statistics = [
            full_statistic(contrast, reduced_fit + reduced_residual[permutation, :])
            for permutation in permutations
        ]
        intercept_statistics = [
            full_statistic(contrast, intercept_fit + intercept_residual[permutation, :])
            for permutation in permutations
        ]
        exact_count = sum(statistic >= observed for statistic in exact_statistics)
        intercept_count = sum(
            statistic >= observed for statistic in intercept_statistics
        )
        exact_p = (1.0 + exact_count) / (n_perm + 1.0)
        intercept_p = (1.0 + intercept_count) / (n_perm + 1.0)

        row = result.query(
            "estimand == @estimand and profile_scope == 'subregion_profile' "
            "and region == 'anterior'"
        ).iloc[0]
        assert row["derived_seed"] == derived_seed
        assert row["failure_reason"] == ""
        assert row["n_total"] == 12
        assert row["n_complete"] == 12
        assert row["statistic"] == pytest.approx(observed, rel=1e-12, abs=1e-12)
        assert row["p_perm"] == pytest.approx(exact_p, rel=1e-12, abs=1e-12)
        assert exact_count != intercept_count
        assert not np.isclose(exact_p, intercept_p)

        # Exercise the actual reduced/null helpers used by public execution.
        production_basis = _null_basis(contrast)
        production_fit, failure = _fit_reduced(values, design, contrast)
        assert failure == ""
        assert production_basis.shape == (len(GROUPS), len(GROUPS) - 1)
        assert np.linalg.matrix_rank(design @ production_basis) == len(GROUPS) - 1
        assert np.allclose(contrast @ production_basis, 0.0, atol=1e-12)
        assert np.linalg.norm(
            production_fit
            - design @ np.linalg.lstsq(design, production_fit, rcond=None)[0]
        ) < 1e-10
