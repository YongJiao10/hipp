"""Contract tests for subject-level cross-dataset rest-FC tensors."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))

import features  # noqa: E402

build_fc_tensor = features.build_fc_tensor
build_hemisphere_views = features.build_hemisphere_views


NETWORKS_17 = tuple(f"F{index:02d}" for index in range(1, 18))
NETWORKS_8 = tuple(f"C{index:02d}" for index in range(1, 9))
REGIONS = ("anterior", "middle", "posterior")
CONFIG_PATH = Path(__file__).resolve().parents[1] / "config" / "analysis_config.json"


def _write_fixture(tmp_path: Path, *, include_right: bool = True) -> dict[str, object]:
    tmp_path.mkdir(parents=True, exist_ok=True)
    root = tmp_path / "fc"
    registry_path = tmp_path / "registry.tsv"
    registry_rows = []
    for index, family17 in enumerate(NETWORKS_17):
        registry_rows.append(
            {
                "roi_id": index + 1,
                "fc_column": index,
                "family17": family17,
                "canonical8": NETWORKS_8[index % len(NETWORKS_8)],
            }
        )
    pd.DataFrame(registry_rows).to_csv(registry_path, sep="\t", index=False)

    labels = np.asarray([1, 1, 2, 2, 3, 3], dtype=np.int32)
    weights = {
        "L": np.asarray([1, 3, 2, 2, 4, 1], dtype=np.float64),
        "R": np.asarray([3, 1, 1, 3, 2, 4], dtype=np.float64),
    }
    for hemisphere in ("L", "R"):
        if hemisphere == "R" and not include_right:
            continue
        combo = root / "sub-0001" / "2mm" / f"hemi-{hemisphere}"
        combo.mkdir(parents=True)
        values = np.empty((labels.size, len(NETWORKS_17)), dtype=np.float64)
        for vertex, region_label in enumerate(labels):
            values[vertex] = float(region_label * 100) + np.arange(len(NETWORKS_17))
        if hemisphere == "R":
            values += 10.0
        np.save(combo / "fc.npy", values)
        np.save(combo / "labels.npy", labels)
        np.save(combo / "weights.npy", weights[hemisphere])
        (combo / "provenance.json").write_text(
            json.dumps(
                {
                    "subject": "sub-0001",
                    "hemi": hemisphere,
                    "smoothing": "2mm",
                    "fc_shape": list(values.shape),
                    "modality": "rest-fMRI",
                    "data_role": "resting-state functional connectivity",
                    "measure": "fisher_z",
                    "fc_estimator": "fisher_z",
                    "method": "synthetic_registry",
                    "dataset": "jjtms",
                }
            ),
            encoding="utf-8",
        )

    return {
        "allowed_roots": [str(tmp_path)],
        "smoothings_mm": [2],
        "hemispheres": ["L", "R"],
        "datasets": {
            "jjtms": {
                "fc_inputs": {
                    "root": str(root),
                    "fisher_z_pattern": (
                        "{subject}/{smoothing}mm/hemi-{hemisphere}/fc.npy"
                    ),
                    "provenance_pattern": (
                        "{subject}/{smoothing}mm/hemi-{hemisphere}/provenance.json"
                    ),
                    "fc_estimator": "fisher_z",
                },
                "parcellations": [
                    {
                        "method": "synthetic_registry",
                        "labels_pattern": (
                            "{subject}/{smoothing}mm/hemi-{hemisphere}/labels.npy"
                        ),
                        "region_weight_pattern": (
                            "{subject}/{smoothing}mm/hemi-{hemisphere}/weights.npy"
                        ),
                        "registry_path": str(registry_path),
                        "region_labels": {"1": "anterior", "2": "middle", "3": "posterior"},
                    }
                ],
            }
        },
        "network_order_17": list(reversed(NETWORKS_17)),
        "network_order_8": list(reversed(NETWORKS_8)),
    }


def _roster() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "dataset": "jjtms",
                "subject": "sub-0001",
                "group": "healthy",
                "trauma": pd.NA,
                "symptomatic": pd.NA,
            }
        ]
    )


def test_tensor_has_exact_regions_resolutions_and_deterministic_aggregation(
    tmp_path: Path,
) -> None:
    config = _write_fixture(tmp_path)
    tensor = build_fc_tensor("jjtms", _roster(), config)

    assert list(tensor.columns) == [
        "dataset",
        "subject",
        "group",
        "trauma",
        "symptomatic",
        "method",
        "smoothing_mm",
        "hemisphere",
        "region",
        "network_resolution",
        "network",
        "fc_estimator",
        "fisher_z",
        "region_weight",
        "valid_vertex_count",
        "network_coverage",
        "status",
        "failure_reason",
    ]
    assert set(tensor["region"]) == set(REGIONS)
    assert set(tensor["network_resolution"]) == {8, 17}
    assert tensor.groupby("network_resolution")["network"].nunique().to_dict() == {
        8: 8,
        17: 17,
    }
    assert tensor["fc_estimator"].eq("fisher_z").all()
    assert tensor["status"].eq("present").all()
    assert tensor["fisher_z"].notna().all()

    # The C01 value is derived from F01 and F09, not from a label-name guess.
    anterior_c01 = tensor.query(
        "hemisphere == 'L' and region == 'anterior' and network == 'C01'"
    ).iloc[0]
    expected_c01 = np.mean([100.0 + 0.0, 100.0 + 8.0, 100.0 + 16.0])
    assert anterior_c01["fisher_z"] == pytest.approx(expected_c01)

    repeated = build_fc_tensor("jjtms", _roster(), config)
    pd.testing.assert_frame_equal(tensor, repeated)


def test_build_fc_tensor_can_materialize_one_priority_resolution(tmp_path: Path) -> None:
    config = _write_fixture(tmp_path)
    tensor = build_fc_tensor(
        "jjtms",
        _roster(),
        config,
        methods=("synthetic_registry",),
        smoothings_mm=(2,),
        network_resolutions=(8,),
    )
    assert set(tensor["method"]) == {"synthetic_registry"}
    assert set(tensor["network_resolution"]) == {8}
    assert set(tensor["smoothing_mm"]) == {2}
    assert tensor["status"].eq("present").all()


def test_nonfinite_vertices_outside_configured_valid_mask_are_ignored(
    tmp_path: Path,
) -> None:
    config = _write_fixture(tmp_path)
    spec = config["datasets"]["jjtms"]["parcellations"][0]
    spec["valid_mask_pattern"] = (
        "{subject}/{smoothing}mm/hemi-{hemisphere}/valid_mask.npy"
    )
    for hemisphere in ("L", "R"):
        combo = tmp_path / "fc" / "sub-0001" / "2mm" / f"hemi-{hemisphere}"
        values = np.load(combo / "fc.npy")
        valid_mask = np.ones(values.shape[0], dtype=bool)
        if hemisphere == "L":
            values[0, :] = np.nan
            valid_mask[0] = False
            np.save(combo / "fc.npy", values)
        np.save(combo / "valid_mask.npy", valid_mask)

    tensor = build_fc_tensor("jjtms", _roster(), config)
    anterior_left = tensor.loc[
        tensor["hemisphere"].eq("L") & tensor["region"].eq("anterior")
    ]
    assert anterior_left["status"].eq("present").all()
    assert anterior_left["valid_vertex_count"].eq(1).all()
    assert anterior_left["region_weight"].eq(3.0).all()
    assert anterior_left["network_coverage"].eq(1.0).all()


def test_whole_row_nonfinite_vertices_inside_mask_are_excluded_not_zero_filled(
    tmp_path: Path,
) -> None:
    config = _write_fixture(tmp_path)
    spec = config["datasets"]["jjtms"]["parcellations"][0]
    spec["valid_mask_pattern"] = (
        "{subject}/{smoothing}mm/hemi-{hemisphere}/valid_mask.npy"
    )
    for hemisphere in ("L", "R"):
        combo = tmp_path / "fc" / "sub-0001" / "2mm" / f"hemi-{hemisphere}"
        values = np.load(combo / "fc.npy")
        if hemisphere == "L":
            values[0, :] = np.nan
            np.save(combo / "fc.npy", values)
        np.save(combo / "valid_mask.npy", np.ones(values.shape[0], dtype=bool))

    tensor = build_fc_tensor("jjtms", _roster(), config)
    anterior_left = tensor.loc[
        tensor["hemisphere"].eq("L") & tensor["region"].eq("anterior")
    ]
    assert anterior_left["status"].eq("present").all()
    assert anterior_left["valid_vertex_count"].eq(1).all()
    assert anterior_left["region_weight"].eq(3.0).all()
    assert anterior_left["network_coverage"].eq(0.5).all()
    expected_f01 = 100.0
    actual_f01 = anterior_left.loc[
        anterior_left["network_resolution"].eq(17)
        & anterior_left["network"].eq("F01"),
        "fisher_z",
    ].iloc[0]
    assert actual_f01 == pytest.approx(expected_f01)


def test_all_configured_vertices_nonfinite_emit_explicit_missing_rows(
    tmp_path: Path,
) -> None:
    config = _write_fixture(tmp_path)
    spec = config["datasets"]["jjtms"]["parcellations"][0]
    spec["valid_mask_pattern"] = (
        "{subject}/{smoothing}mm/hemi-{hemisphere}/valid_mask.npy"
    )
    for hemisphere in ("L", "R"):
        combo = tmp_path / "fc" / "sub-0001" / "2mm" / f"hemi-{hemisphere}"
        values = np.load(combo / "fc.npy")
        if hemisphere == "L":
            values[:2, :] = np.nan
            np.save(combo / "fc.npy", values)
        np.save(combo / "valid_mask.npy", np.ones(values.shape[0], dtype=bool))

    tensor = build_fc_tensor("jjtms", _roster(), config)
    anterior_left = tensor.loc[
        tensor["hemisphere"].eq("L") & tensor["region"].eq("anterior")
    ]
    assert anterior_left["status"].eq("missing").all()
    assert anterior_left["fisher_z"].isna().all()
    assert anterior_left["valid_vertex_count"].eq(0).all()
    assert anterior_left["region_weight"].eq(0.0).all()
    assert anterior_left["network_coverage"].eq(0.0).all()
    assert anterior_left["failure_reason"].str.contains(
        "no finite valid vertices"
    ).all()


def test_views_use_weighted_bilateral_and_paired_lr_without_zero_imputation(
    tmp_path: Path,
) -> None:
    config = _write_fixture(tmp_path)
    tensor = build_fc_tensor("jjtms", _roster(), config)
    views = build_hemisphere_views(tensor)

    assert set(views["hemisphere_mode"]) == {"bilateral", "L", "R", "L-R"}
    key = {
        "subject": "sub-0001",
        "method": "synthetic_registry",
        "smoothing_mm": 2,
        "region": "anterior",
        "network_resolution": 17,
        "network": "F01",
    }
    selected = views.loc[
        (views["subject"] == key["subject"])
        & (views["method"] == key["method"])
        & (views["smoothing_mm"] == key["smoothing_mm"])
        & (views["region"] == key["region"])
        & (views["network_resolution"] == key["network_resolution"])
        & (views["network"] == key["network"])
    ]
    bilateral = selected.loc[selected["hemisphere_mode"] == "bilateral"].iloc[0]
    left = selected.loc[selected["hemisphere_mode"] == "L"].iloc[0]
    right = selected.loc[selected["hemisphere_mode"] == "R"].iloc[0]
    lr = selected.loc[selected["hemisphere_mode"] == "L-R"].iloc[0]
    assert bilateral["region_weight"] == pytest.approx(8.0)
    assert bilateral["fisher_z"] == pytest.approx(
        (left["fisher_z"] * 4.0 + right["fisher_z"] * 4.0) / 8.0
    )
    assert lr["fisher_z"] == pytest.approx(left["fisher_z"] - right["fisher_z"])
    assert lr["fisher_z"] != 0.0

    missing_right = build_fc_tensor("jjtms", _roster(), _write_fixture(tmp_path / "missing", include_right=False))
    missing_views = build_hemisphere_views(missing_right)
    missing_lr = missing_views.loc[
        (missing_views["hemisphere_mode"] == "L-R")
        & (missing_views["network"] == "F01")
        & (missing_views["region"] == "anterior")
    ].iloc[0]
    missing_bilateral = missing_views.loc[
        (missing_views["hemisphere_mode"] == "bilateral")
        & (missing_views["network"] == "F01")
        & (missing_views["region"] == "anterior")
    ].iloc[0]
    assert pd.isna(missing_lr["fisher_z"])
    assert pd.isna(missing_bilateral["fisher_z"])
    assert "right" in missing_lr["failure_reason"].lower()
    assert "right" in missing_bilateral["failure_reason"].lower()


def test_tensor_rejects_unknown_or_ambiguous_registry_mapping(tmp_path: Path) -> None:
    config = _write_fixture(tmp_path)
    registry_path = Path(config["datasets"]["jjtms"]["parcellations"][0]["registry_path"])
    registry = pd.read_csv(registry_path, sep="\t")
    registry.loc[0, "canonical8"] = "unknown"
    registry.to_csv(registry_path, sep="\t", index=False)
    tensor = build_fc_tensor("jjtms", _roster(), config)
    assert tensor["status"].eq("failed").all()
    assert tensor["failure_reason"].str.contains("unknown network").all()


def test_tensor_fails_closed_on_provenance_shape_mismatch(tmp_path: Path) -> None:
    config = _write_fixture(tmp_path)
    provenance = (
        tmp_path
        / "fc"
        / "sub-0001"
        / "2mm"
        / "hemi-L"
        / "provenance.json"
    )
    payload = json.loads(provenance.read_text(encoding="utf-8"))
    payload["fc_shape"] = [999, 17]
    provenance.write_text(json.dumps(payload), encoding="utf-8")
    tensor = build_fc_tensor("jjtms", _roster(), config)
    left = tensor.loc[tensor["hemisphere"] == "L"]
    assert left["status"].eq("failed").all()
    assert left["failure_reason"].str.contains("provenance shape mismatch").all()


def test_tensor_fails_closed_when_provenance_is_not_rest_fisher_z(tmp_path: Path) -> None:
    config = _write_fixture(tmp_path)
    provenance = (
        tmp_path
        / "fc"
        / "sub-0001"
        / "2mm"
        / "hemi-L"
        / "provenance.json"
    )
    payload = json.loads(provenance.read_text(encoding="utf-8"))
    payload.update({"modality": "beta", "data_role": "TMS-response contrast", "measure": "beta"})
    provenance.write_text(json.dumps(payload), encoding="utf-8")
    tensor = build_fc_tensor("jjtms", _roster(), config)
    left = tensor.loc[tensor["hemisphere"] == "L"]
    assert left["status"].eq("failed").all()
    assert left["failure_reason"].str.contains("rest|Fisher-z|provenance", case=False).all()


def test_provenance_rejects_rest_plus_task_role_token_mixing(tmp_path: Path) -> None:
    config = _write_fixture(tmp_path)
    provenance = (
        tmp_path
        / "fc"
        / "sub-0001"
        / "2mm"
        / "hemi-L"
        / "provenance.json"
    )
    payload = json.loads(provenance.read_text(encoding="utf-8"))
    payload.update(
        {
            "modality": "rest-fMRI",
            "data_role": "task functional connectivity",
            "measure": "fisher_z",
            "fc_estimator": "fisher_z",
        }
    )
    provenance.write_text(json.dumps(payload), encoding="utf-8")
    tensor = build_fc_tensor("jjtms", _roster(), config)
    left = tensor.loc[tensor["hemisphere"] == "L"]
    assert left["status"].eq("failed").all()
    assert left["failure_reason"].str.contains("role|connectivity|provenance", case=False).all()


def test_provenance_does_not_union_partial_cross_sidecar_semantics(tmp_path: Path) -> None:
    config = _write_fixture(tmp_path)
    provenance = (
        tmp_path
        / "fc"
        / "sub-0001"
        / "2mm"
        / "hemi-L"
        / "provenance.json"
    )
    payload = json.loads(provenance.read_text(encoding="utf-8"))
    for key in ("modality", "data_role", "measure", "fc_estimator"):
        payload.pop(key, None)
    provenance.write_text(json.dumps(payload), encoding="utf-8")
    rest_role = tmp_path / "rest_role.json"
    rest_role.write_text(
        json.dumps(
            {
                "subject": "sub-0001",
                "hemi": "L",
                "smoothing": "2mm",
                "modality": "rest-fMRI",
                "data_role": "resting-state functional connectivity",
            }
        ),
        encoding="utf-8",
    )
    fisher_only = tmp_path / "fisher_only.json"
    fisher_only.write_text(
        json.dumps(
            {
                "subject": "sub-0001",
                "hemi": "L",
                "smoothing": "2mm",
                "fc_estimator": "fisher_z",
            }
        ),
        encoding="utf-8",
    )
    config["datasets"]["jjtms"]["parcellations"][0]["provenance_sidecar_patterns"] = [
        {"root": str(tmp_path), "pattern": rest_role.name},
        {"root": str(tmp_path), "pattern": fisher_only.name},
    ]
    tensor = build_fc_tensor("jjtms", _roster(), config)
    assert tensor["status"].eq("failed").all()
    assert tensor["failure_reason"].str.contains("record|provenance|semantic", case=False).all()


def test_provenance_rejects_contradictory_sidecar_estimator(tmp_path: Path) -> None:
    config = _write_fixture(tmp_path)
    contradictory = tmp_path / "contradictory.json"
    contradictory.write_text(
        json.dumps(
            {
                "subject": "sub-0001",
                "hemi": "L",
                "smoothing": "2mm",
                "modality": "rest-fMRI",
                "data_role": "resting-state functional connectivity",
                "fc_estimator": "pearson",
            }
        ),
        encoding="utf-8",
    )
    config["datasets"]["jjtms"]["parcellations"][0]["provenance_sidecar_patterns"] = [
        {"root": str(tmp_path), "pattern": contradictory.name}
    ]
    tensor = build_fc_tensor("jjtms", _roster(), config)
    assert tensor["status"].eq("failed").all()
    assert tensor["failure_reason"].str.contains("estimator|Fisher-z|provenance", case=False).all()


def test_provenance_accepts_one_explicit_rest_fc_fisher_z_record(tmp_path: Path) -> None:
    config = _write_fixture(tmp_path)
    tensor = build_fc_tensor("jjtms", _roster(), config)
    assert tensor["status"].eq("present").all()


def _task_adapter_records(dtseries: str) -> tuple[list[tuple[Path, dict[str, object]]], dict[str, object]]:
    records = [
        (
            Path("primary.json"),
            {
                "data_role": "resting-state functional connectivity",
                "measure": "fisher_z",
            },
        ),
        (Path("reference.json"), {"dtseries": dtseries}),
    ]
    schema = {
        "acquisition": {
            "source": "sidecar:0",
            "field": "dtseries",
            "extract": "task_entity",
            "allowed_exact": ["rest"],
        },
        "role": {
            "source": "primary",
            "field": "data_role",
            "allowed_exact": ["resting-state functional connectivity"],
        },
        "estimator": {
            "source": "primary",
            "field": "measure",
            "allowed_exact": ["fisher_z"],
        },
    }
    return records, schema


def test_stanford_task_adapter_rejects_mixed_task_entities() -> None:
    records, schema = _task_adapter_records(
        "/data/sub-0001_task-rest_task-memory_space-fsLR.dtseries.nii"
    )
    with pytest.raises(ValueError, match="task|rest"):
        features._validate_semantic_schema(records, schema)


def test_stanford_task_adapter_rejects_reversed_mixed_task_entities() -> None:
    records, schema = _task_adapter_records(
        "/data/sub-0001_task-memory_task-rest_space-fsLR.dtseries.nii"
    )
    with pytest.raises(ValueError, match="task|rest"):
        features._validate_semantic_schema(records, schema)


def test_stanford_task_adapter_accepts_single_or_repeated_rest_entities() -> None:
    for dtseries in (
        "/data/sub-0001_task-rest_space-fsLR.dtseries.nii",
        "/data/sub-0001_task-rest_task-rest_space-fsLR.dtseries.nii",
    ):
        records, schema = _task_adapter_records(dtseries)
        semantic = features._validate_semantic_schema(records, schema)
        assert semantic["acquisition"] == {"rest"}


def test_tensor_rejects_registry_without_contiguous_fc_column_index(tmp_path: Path) -> None:
    config = _write_fixture(tmp_path)
    registry_path = Path(config["datasets"]["jjtms"]["parcellations"][0]["registry_path"])
    registry = pd.read_csv(registry_path, sep="\t")
    registry.loc[0, "fc_column"] = 17
    registry.to_csv(registry_path, sep="\t", index=False)
    tensor = build_fc_tensor("jjtms", _roster(), config)
    assert tensor["status"].eq("failed").all()
    assert tensor["failure_reason"].str.contains("FC-column|contiguous|index", case=False).all()


def test_tensor_aligns_shuffled_registry_rows_by_fc_column_index(tmp_path: Path) -> None:
    config = _write_fixture(tmp_path)
    registry_path = Path(config["datasets"]["jjtms"]["parcellations"][0]["registry_path"])
    registry = pd.read_csv(registry_path, sep="\t").iloc[::-1]
    registry.to_csv(registry_path, sep="\t", index=False)
    tensor = build_fc_tensor("jjtms", _roster(), config)
    f17 = tensor.query(
        "hemisphere == 'L' and region == 'anterior' and network_resolution == 17 and network == 'F17'"
    ).iloc[0]
    assert f17["fisher_z"] == pytest.approx(116.0)


def test_tensor_preserves_configured_network_ordinals(tmp_path: Path) -> None:
    config = _write_fixture(tmp_path)
    tensor = build_fc_tensor("jjtms", _roster(), config)
    left_anterior_17 = tensor.loc[
        (tensor["hemisphere"] == "L")
        & (tensor["region"] == "anterior")
        & (tensor["network_resolution"] == 17)
    ]
    assert left_anterior_17["network"].tolist() == list(reversed(NETWORKS_17))


def test_tensor_normalizes_subjects_before_duplicate_check(tmp_path: Path) -> None:
    config = _write_fixture(tmp_path)
    roster = pd.DataFrame(
        [
            {"dataset": "jjtms", "subject": "sub-0001", "group": "healthy"},
            {"dataset": "jjtms", "subject": "1", "group": "healthy"},
        ]
    )
    with pytest.raises(ValueError, match="duplicate subjects"):
        build_fc_tensor("jjtms", roster, config)


def test_tensor_rejects_repeated_configured_hemispheres(tmp_path: Path) -> None:
    config = _write_fixture(tmp_path)
    config["hemispheres"] = ["L", "L"]
    with pytest.raises(ValueError, match="duplicate"):
        build_fc_tensor("jjtms", _roster(), config)


def test_ambiguous_17_to_8_mapping_is_failure_audited(tmp_path: Path) -> None:
    config = _write_fixture(tmp_path)
    registry_path = Path(config["datasets"]["jjtms"]["parcellations"][0]["registry_path"])
    registry = pd.read_csv(registry_path, sep="\t")
    extra = registry.iloc[[0]].copy()
    extra["roi_id"] = 401
    extra["fc_column"] = 17
    extra["canonical8"] = "C02"
    registry = pd.concat([registry, extra], ignore_index=True)
    for hemisphere in ("L", "R"):
        fc_path = (
            Path(config["datasets"]["jjtms"]["fc_inputs"]["root"])
            / "sub-0001"
            / "2mm"
            / f"hemi-{hemisphere}"
            / "fc.npy"
        )
        values = np.load(fc_path)
        np.save(fc_path, np.column_stack([values, values[:, 0]]))
        provenance = json.loads(fc_path.with_name("provenance.json").read_text(encoding="utf-8"))
        provenance["fc_shape"] = [int(values.shape[0]), int(values.shape[1] + 1)]
        fc_path.with_name("provenance.json").write_text(json.dumps(provenance), encoding="utf-8")
    registry.to_csv(registry_path, sep="\t", index=False)
    tensor = build_fc_tensor("jjtms", _roster(), config)
    assert tensor["status"].eq("failed").all()
    assert tensor["failure_reason"].str.contains("multiple 8-network", case=False).all()


def test_shipped_config_smoke_is_callable_and_reports_every_family() -> None:
    summary = features.run_config_smoke(CONFIG_PATH, subjects_per_dataset=1)
    assert set(summary["dataset"]) == {"stanford", "jjtms"}
    assert set(summary["method"]) == {
        "hcp_fixed",
        "hcp_template_matching",
    }
    assert summary["status"].isin({"available", "unavailable", "failed", "missing"}).all()
