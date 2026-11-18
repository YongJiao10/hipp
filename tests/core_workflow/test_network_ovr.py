from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts/core_workflow"))
from lib import network_ovr as ovr
from lib import kong2022_fc_contract as contract
from lib.kong2022_label_utils import FAMILIES_17_ORDER, load_kong2022_merge_mapping


def test_network_ts_ovr_variant_follows_taxonomy():
    assert ovr.network_ts_ovr_variant("canonical7") == "ovr_logistic_canonical7_network_ts"
    assert ovr.network_ts_ovr_variant("family17") == "ovr_logistic_family17_network_ts"


@pytest.fixture
def inputs(monkeypatch):
    mapping = load_kong2022_merge_mapping()
    registry = [{"roi_id": i + 1, "roi_name": f"roi-{i+1}",
                 "family17": FAMILIES_17_ORDER[i % 17],
                 "canonical7": mapping[FAMILIES_17_ORDER[i % 17]]} for i in range(400)]
    monkeypatch.setattr(contract, "authoritative_kong2022_roi_registry", lambda: registry)
    rng = np.random.default_rng(7)
    profiles = rng.normal(0, .3, (12, 3, 400))
    for c in range(3):
        profiles[:, c, c::7] += 1.
    return profiles, [f"sub-{i}" for i in range(12)], [9, 2, 5], registry


@pytest.mark.parametrize("taxonomy, count", [("canonical7", 7), ("family17", 17)])
def test_ovr_matches_direct_estimator_classes_and_raw_per_sd_units(inputs, taxonomy, count):
    profiles, subjects, clusters, registry = inputs
    original = profiles.copy()
    with warnings.catch_warnings(record=True) as caught:
        result = ovr.fit_network_ovr(profiles, subjects, clusters, registry, taxonomy=taxonomy)
    assert not any("penalty" in str(w.message) for w in caught)
    assert result["network_profiles"].shape == (12, 3, count)
    x = result["network_profiles"].reshape(-1, count)
    y = np.tile(clusters, len(subjects))
    direct = OneVsRestClassifier(LogisticRegression(solver="liblinear", C=1., tol=1e-4,
                                                   max_iter=1000)).fit(x, y)
    expected = np.stack([direct.estimators_[list(direct.classes_).index(c)].coef_[0] for c in clusters])
    assert np.allclose(result["beta"], expected)
    assert result["cluster_ids"] == clusters
    assert result["subject_ids"] == subjects
    assert np.allclose(result["odds_ratio"], np.exp(expected))
    assert np.allclose(result["feature_sd"], x.std(axis=0, ddof=0))
    assert np.allclose(result["standardized_beta"], expected * x.std(axis=0))
    assert np.allclose(result["standardized_odds_ratio"], np.exp(expected * x.std(axis=0)))
    assert np.array_equal(profiles, original)


def test_binary_classes_expand_negative_and_positive_estimator_directions(inputs):
    profiles, subjects, _, registry = inputs
    result = ovr.fit_network_ovr(profiles[:, :2], subjects, [9, 2], registry, taxonomy="canonical7")
    assert np.allclose(result["beta"][0], -result["beta"][1])
    assert np.allclose(result["intercept"][0], -result["intercept"][1])
    assert result["estimator_classes"] == [2, 9]


def test_missing_network_values_identify_subject_cluster_and_network(inputs):
    profiles, subjects, clusters, registry = inputs
    name = registry[0]["canonical7"]
    columns = [i for i, row in enumerate(registry) if row["canonical7"] == name]
    profiles[4, 1, columns] = np.nan
    with pytest.raises(ValueError, match=f"sub-4.*cluster=2.*{name}"):
        ovr.fit_network_ovr(profiles, subjects, clusters, registry, taxonomy="canonical7")


@pytest.mark.parametrize("feature_source", ["roi", "network_ts"])
def test_ovr_excludes_only_declared_missing_subject_cluster_rows(inputs, tmp_path, feature_source):
    import csv
    import json

    profiles, subjects, clusters, registry = inputs
    names = list(ovr.canonical_families_7_order())
    observed = np.ones(profiles.shape[:2], dtype=bool)
    observed[4, 1] = False
    profiles[4, 1] = np.nan
    if feature_source == "roi":
        result = ovr.fit_network_ovr(
            profiles, subjects, clusters, registry, taxonomy="canonical7", observed_profiles=observed,
        )
    else:
        network = profiles[..., :len(names)]
        result = ovr.fit_network_ovr_from_network_profiles(
            network, subjects, clusters, names, observed_profiles=observed,
        )

    x = result["network_profiles"][observed]
    y = np.broadcast_to(clusters, observed.shape)[observed]
    expected = OneVsRestClassifier(LogisticRegression(
        solver="liblinear", C=1., tol=1e-4, max_iter=1000,
    )).fit(x, y)
    np.testing.assert_allclose(
        result["beta"], [expected.estimators_[list(expected.classes_).index(c)].coef_[0] for c in clusters],
    )
    np.testing.assert_allclose(result["feature_sd"], x.std(axis=0))
    paths = ovr.export_network_ovr(result, tmp_path)
    with paths["input_table"].open() as handle:
        fitting_rows = list(csv.DictReader(handle, delimiter="\t"))
    assert [(row["subject_id"], int(row["cluster_id"])) for row in fitting_rows] == [
        (subjects[i], clusters[j]) for i, j in np.argwhere(observed)
    ]
    metadata = json.loads(paths["metadata"].read_text())
    assert metadata["n_observations"] == 35
    assert metadata["excluded_subject_cluster_rows"] == [{"subject_id": subjects[4], "cluster_id": clusters[1]}]
    assert np.isnan(np.load(paths["network_profiles"])[4, 1]).all()


def test_ovr_observation_mask_cannot_omit_measured_profiles():
    data = np.ones((3, 2, 2))
    observed = np.ones(data.shape[:2], dtype=bool)
    observed[0, 0] = False
    with pytest.raises(ValueError, match="excluded profiles must be wholly missing"):
        ovr.fit_network_ovr_from_network_profiles(
            data, ["s1", "s2", "s3"], [1, 2], ["a", "b"], observed_profiles=observed,
        )


def test_partial_roi_missingness_preserves_existing_nanmean_semantics(inputs):
    profiles, subjects, clusters, registry = inputs
    profiles[:, :, 0] = np.nan
    result = ovr.fit_network_ovr(profiles, subjects, clusters, registry, taxonomy="family17")
    names = result["network_names"]
    cols = [i for i, row in enumerate(registry) if row["family17"] == registry[0]["family17"]]
    assert np.allclose(result["network_profiles"][:, :, names.index(registry[0]["family17"])],
                       np.nanmean(profiles[:, :, cols], axis=2))


def test_annotation_uses_largest_strictly_positive_raw_beta():
    beta = np.array([[.5, .2], [-1., 0.], [.1, .4]])
    rows = ovr.assignment_rows(beta, [1, 2, 3], ["A", "B"])
    assert [r["top1_network"] for r in rows] == ["A", "unassigned", "B"]
    assert rows[1]["status"] == "unassigned"
    assert rows[0]["beta_by_network"] == {"A": .5, "B": .2}


def test_nonconvergence_is_fatal(inputs, monkeypatch):
    from sklearn.exceptions import ConvergenceWarning
    class FailedOVR:
        def __init__(self, estimator, **kwargs): pass
        def fit(self, x, y):
            warnings.warn("did not converge", ConvergenceWarning)
            return self
    monkeypatch.setattr(ovr, "OneVsRestClassifier", FailedOVR)
    with pytest.raises(RuntimeError, match="converg"):
        ovr.fit_network_ovr(*inputs, taxonomy="canonical7")


def test_export_contains_exact_observations_coefficients_and_population(inputs, tmp_path):
    result = ovr.fit_network_ovr(*inputs, taxonomy="canonical7")
    paths = ovr.export_network_ovr(result, tmp_path)
    import csv
    import json
    with paths["input_table"].open() as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    assert len(rows) == 12 * 3
    assert [(r["subject_id"], int(r["cluster_id"])) for r in rows] == [
        (s, c) for s in inputs[1] for c in inputs[2]]
    assert np.array_equal(np.load(paths["beta"]), result["beta"])
    metadata = json.loads(paths["metadata"].read_text())
    assert metadata["feature_sd_ddof"] == 0
    assert metadata["n_observations"] == 36
    assert metadata["taxonomy"] == "canonical7"
    assert "dependent" in metadata["interpretation"]


def test_unsupported_taxonomy_and_registry_mismatch_fail(inputs):
    with pytest.raises(ValueError, match="taxonomy"):
        ovr.fit_network_ovr(*inputs, taxonomy="lynch20")
    profiles, subjects, clusters, registry = inputs
    wrong = [dict(r) for r in registry]
    wrong[0]["roi_id"] = 999
    with pytest.raises(ValueError, match="ROI order"):
        ovr.fit_network_ovr(profiles, subjects, clusters, wrong, taxonomy="canonical7")


def test_atlas_ovr_variants_annotate_fixed_labels_and_both_keeps_its_meaning(inputs, tmp_path):
    import importlib.util
    path = Path(__file__).resolve().parents[2] / "scripts/core_workflow/05_02_run_group_atlas.py"
    spec = importlib.util.spec_from_file_location("network_ovr_atlas_test", path)
    core = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = core
    spec.loader.exec_module(core)
    assert core.requested_ovr_variants(["both"], atlas="kong2022", method="gradient-ward") == []
    with pytest.raises(ValueError, match="Kong2022.*cluster"):
        core.requested_ovr_variants(["ovr_logistic_family17"], atlas="lynch2024_20net", method="gradient-gmm")
    labels = np.array([1, 1, 2, 2, 3, 3])
    original = labels.copy()
    profiles, subjects, _, registry = inputs
    specs = core.build_ovr_assignment_specs(
        roi_profile_3d=profiles, subjects=subjects, cluster_ids=[1, 2, 3], roi_registry=registry,
        variants=list(ovr.OVR_VARIANTS), out_dir=tmp_path,
    )
    assert set(specs) == set(ovr.OVR_VARIANTS)
    for variant, count in zip(ovr.OVR_VARIANTS, [7, 17], strict=True):
        row = specs[variant]
        assert len(row["network_names"]) == count
        core.functional_atlas_from_clusters(
            group_labels_final=labels, status_rows=row["status_rows"], network_names=row["network_names"],
            color_map={n: (1, 2, 3, 255) for n in row["network_names"]},
        )
        assert np.array_equal(labels, original)
        assert Path(row["ovr_model_metadata"]).is_file()
        assert row["profile_units"] == "raw_beta_per_Fisher_z_unit"


def test_ovr_artifact_contract_checks_taxonomy_and_coefficients(inputs, tmp_path):
    import json
    import nibabel as nib
    from lib.gradient_assignment_contract import validate_gradient_assignment
    from lib.gifti_labels import make_named_label_gifti
    from lib.network_colors import load_network_colors
    import importlib.util
    path = Path(__file__).resolve().parents[2] / "scripts/core_workflow/05_02_run_group_atlas.py"
    spec = importlib.util.spec_from_file_location("ovr_artifact_atlas", path)
    core = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = core
    spec.loader.exec_module(core)
    profiles, subjects, _, registry = inputs
    result = ovr.fit_network_ovr(profiles, subjects, [1, 2, 3], registry, taxonomy="canonical7")
    paths = ovr.export_network_ovr(result, tmp_path)
    labels = np.array([1, 1, 2, 2, 3, 3], np.int32)
    np.save(tmp_path / "group_labels_final.npy", labels)
    functional, mapping, items = core.functional_atlas_from_clusters(
        group_labels_final=labels, status_rows=result["assignment_rows"],
        network_names=result["network_names"], color_map=load_network_colors(),
    )
    np.save(tmp_path / "functional.npy", functional)
    nib.save(make_named_label_gifti(functional, items), tmp_path / "functional.label.gii")
    (tmp_path / "mapping.json").write_text(json.dumps(mapping))
    (tmp_path / "status.json").write_text(json.dumps({"assignment_metric": ovr.OVR_ASSIGNMENT_METRIC,
                                                     "items": result["assignment_rows"]}))
    row = {"cluster_assignment_status": str(tmp_path / "status.json"),
           "group_functional_atlas_labels": str(tmp_path / "functional.npy"),
           "group_functional_atlas_label_map": str(tmp_path / "mapping.json"),
           "template_functional_atlas_label": str(tmp_path / "functional.label.gii"),
           "network_profile": str(paths["beta"]), "ovr_model_metadata": str(paths["metadata"]),
           "profile_units": "raw_beta_per_Fisher_z_unit"}
    manifest = {"assignment_variants": {"ovr_logistic_canonical7": row}}
    validated = validate_gradient_assignment(tmp_path, manifest, "ovr_logistic_canonical7")
    assert set(validated["winners"]) == {1, 2, 3}
    metadata = json.loads(paths["metadata"].read_text())
    metadata["taxonomy"] = "family17"
    paths["metadata"].write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match="taxonomy"):
        validate_gradient_assignment(tmp_path, manifest, "ovr_logistic_canonical7")


def test_coefficient_plots_use_zero_and_or_one_references_with_explicit_units(inputs):
    result = ovr.fit_network_ovr(*inputs, taxonomy="canonical7")
    fig = ovr.plot_network_ovr(result)
    try:
        assert len(fig.axes) == 4
        assert [float(ax.lines[0].get_ydata()[0]) for ax in fig.axes] == [0., 0., 1., 1.]
        assert "Fisher-z" in fig.axes[0].get_ylabel()
        assert "SD" in fig.axes[1].get_ylabel()
        assert "OR" in fig.axes[2].get_ylabel()
        assert "SD" in fig.axes[3].get_ylabel()
    finally:
        import matplotlib.pyplot as plt
        plt.close(fig)


def test_fit_network_ovr_from_network_profiles_coexistence(tmp_path):
    rng = np.random.default_rng(42)
    n_sub, n_clust, n_net = 16, 3, 7
    net_profiles = rng.normal(0, 0.4, (n_sub, n_clust, n_net))
    # Give cluster c distinct positive affinity for network c
    for c in range(n_clust):
        net_profiles[:, c, c] += 1.5

    subjects = [f"sub-{i+1:03d}" for i in range(n_sub)]
    clusters = [1, 2, 3]
    network_names = [f"Net{i+1}" for i in range(n_net)]

    res = ovr.fit_network_ovr_from_network_profiles(
        net_profiles,
        subjects,
        clusters,
        network_names,
        taxonomy="canonical7",
        feature_source="network_ts",
    )

    assert res["feature_source"] == "network_ts"
    assert res["prefix"] == "network_ovr_canonical7_network_ts"
    assert res["beta"].shape == (3, 7)
    assert len(res["assignment_rows"]) == 3
    # Check that winning networks match the expected strongest network
    for c in range(3):
        assert res["assignment_rows"][c]["top1_network"] == f"Net{c+1}"

    # Export and verify filenames don't collide with standard ROI OVR
    paths = ovr.export_network_ovr(res, tmp_path)
    assert "network_ovr_canonical7_network_ts_beta.npy" in str(paths["beta"])
    assert (tmp_path / "network_ovr_canonical7_network_ts_beta.npy").exists()
    assert (tmp_path / "network_ovr_canonical7_network_ts_inputs.tsv").exists()
    assert (tmp_path / "network_ovr_canonical7_network_ts.json").exists()
