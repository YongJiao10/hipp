#!/usr/bin/env python3
from __future__ import annotations

import importlib
import json
import sys
from types import SimpleNamespace
from pathlib import Path

import numpy as np
import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
CORE_DIR = REPO_ROOT / "scripts" / "core_workflow"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(CORE_DIR) not in sys.path:
    sys.path.insert(0, str(CORE_DIR))


label_utils = importlib.import_module("scripts.core_workflow.lib.kong2022_label_utils")
fc_contract = importlib.import_module("scripts.core_workflow.lib.kong2022_fc_contract")
formal_cohorts = importlib.import_module("scripts.core_workflow.lib.formal_cohorts")
route_tags = importlib.import_module("scripts.core_workflow.lib.route_tags")
output_layout = importlib.import_module("scripts.core_workflow.lib.output_layout")
atlas = importlib.import_module("scripts.core_workflow.05_02_run_group_atlas")
gradients = importlib.import_module("scripts.core_workflow.05_01_compute_hippomaps_gradients")
materializer = importlib.import_module("scripts.core_workflow.04_01_materialize_kong2022_roi_fc")
weighted = importlib.import_module("scripts.core_workflow.lib.network_ts_fisherz")
overview = importlib.import_module("scripts.core_workflow.06_01_build_group_overview")
reference_builder = importlib.import_module("scripts.core_workflow.03_02_build_kong2022_routea_reference")
template_assets = importlib.import_module("scripts.core_workflow.lib.template_assets")
template_profiles = importlib.import_module("scripts.core_workflow.lib.cortex_template_profiles")
extractor = importlib.import_module("scripts.core_workflow.lib.extract_cortex_kong2022_route_a_roi_timeseries")


def test_iowa_patient_manifest_excludes_sub_3003() -> None:
    subjects = formal_cohorts.load_formal_subjects(
        REPO_ROOT / "manifests" / "jj_tms_patient_subjects.txt",
        "jjtms_patient",
    )

    assert "sub-3003" not in subjects


def test_kong2022_canonical7_mapping_merges_auditory_into_somatomotor() -> None:
    mapping = label_utils.load_kong2022_merge_mapping(taxonomy="canonical7")

    assert label_utils.canonical_families_7_order() == [
        "Default",
        "Visual",
        "Somatomotor",
        "DorsalAttention",
        "VentralAttention",
        "Control",
        "Language",
    ]
    assert mapping["SomMotA"] == "Somatomotor"
    assert mapping["SomMotB"] == "Somatomotor"
    assert mapping["Aud"] == "Somatomotor"
    assert set(mapping.values()) == set(label_utils.canonical_families_7_order())


def test_canonical7_network_timeseries_uses_unweighted_roi_means() -> None:
    roi_timeseries = np.asarray(
        [
            [1.0, 2.0, 3.0],
            [3.0, 4.0, 5.0],
            [10.0, 20.0, 30.0],
            [14.0, 22.0, 34.0],
        ],
        dtype=np.float32,
    )
    rows = [
        {"canonical7": "Somatomotor"},
        {"canonical7": "Somatomotor"},
        {"canonical7": "Visual"},
        {"canonical7": "Visual"},
    ]

    got = label_utils.aggregate_kong2022_network_timeseries(
        roi_timeseries,
        rows,
        ["Somatomotor", "Visual"],
        network_column="canonical7",
    )

    assert np.allclose(
        got,
        np.asarray([[2.0, 3.0, 4.0], [12.0, 21.0, 32.0]], dtype=np.float32),
    )


def test_unsuffixed_routea_is_the_default_canonical7_branch() -> None:
    assert "pfm_7net" not in route_tags.ROUTE_TAG_CHOICES
    assert "canonical7" in route_tags.NETWORK_TAXONOMY_CHOICES
    assert (
        route_tags.resolve_route_tag_for_group_atlas_run(
            shared_root=Path("outputs_hcp/inputs/kong2022/shared_inputs/pfm"),
            explicit_route="pfm",
            network_taxonomy="canonical7",
        )
        == "pfm"
    )
    with pytest.raises(ValueError, match="canonical7"):
        route_tags.resolve_route_tag_for_group_atlas_run(
            shared_root=Path("outputs_hcp/inputs/kong2022/shared_inputs/pfm"),
            explicit_route="pfm",
            network_taxonomy="canonical8",
        )
    assert output_layout.resolve_routea_group_atlas_root("stanford").name == "pfm"



def test_hcp_canonical7_reference_is_explicitly_derived_from_400_roi_inputs(tmp_path: Path) -> None:
    subject = "100610"
    ref_dir = tmp_path / f"sub-{subject}" / "reference" / "kong2022"
    ref_dir.mkdir(parents=True)
    rows = fc_contract.authoritative_kong2022_roi_registry()
    lines = ["parcel_id\tparcel_name\tfamily17\tcanonical8"]
    for row in rows:
        lines.append(
            f'{row["roi_id"]}\t{row["roi_name"]}\t{row["family17"]}\t{row["canonical8"]}'
        )
    (ref_dir / "cortex_roi_parcels.tsv").write_text("\n".join(lines) + "\n", encoding="utf-8")
    roi_ts = np.arange(400 * 5, dtype=np.float32).reshape(400, 5)
    np.save(ref_dir / "cortex_roi_parcel_timeseries.npy", roi_ts)

    names, derived = atlas.load_kong2022_network_reference(
        tmp_path, subject, "kong2022", "canonical7"
    )
    assert names == label_utils.canonical_families_7_order()
    expected_rows = atlas.canonical7_registry_from_rows(
        fc_contract.read_kong2022_roi_registry(ref_dir / "cortex_roi_parcels.tsv")
    )
    expected = label_utils.aggregate_kong2022_network_timeseries(
        roi_ts, expected_rows, names, network_column="canonical7"
    )
    assert derived.shape == (7, 5)
    assert np.allclose(derived, expected)
    assert not (ref_dir / "cortex_canonical7_network_timeseries.npy").exists()


def test_04_01_derives_canonical7_from_legacy_400_roi_schema(tmp_path: Path) -> None:
    """The production 400-ROI TSV may omit the derived canonical7 column."""
    subject = "100610"
    ref_dir = tmp_path / f"sub-{subject}" / "reference" / "kong2022"
    ref_dir.mkdir(parents=True)
    source_rows = fc_contract.authoritative_kong2022_roi_registry()
    lines = ["parcel_id\tparcel_name\tfamily17\tcanonical8"]
    for row in source_rows:
        lines.append(
            f'{row["roi_id"]}\t{row["roi_name"]}\t{row["family17"]}\t{row["canonical8"]}'
        )
    tsv_path = ref_dir / "cortex_roi_parcels.tsv"
    tsv_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    roi_ts = np.arange(400 * 5, dtype=np.float32).reshape(400, 5)
    timeseries_path = ref_dir / "cortex_roi_parcel_timeseries.npy"
    np.save(timeseries_path, roi_ts)

    result = materializer.prepare_reference_profiles(
        ref_subj_root=tmp_path / f"sub-{subject}",
        atlas_slug="kong2022",
        target_tsv_path=tsv_path,
        target_timeseries_path=timeseries_path,
        roi_ts=roi_ts,
        roi_names=[str(row["roi_name"]) for row in source_rows],
    )
    _, _, _, _, network_ts, network_names, derived_rows, evidence = result
    assert network_names == label_utils.canonical_families_7_order()
    assert network_ts is not None and network_ts.shape == (7, 5)
    assert derived_rows is not None
    assert [int(row["roi_id"]) for row in derived_rows] == list(range(1, 401))
    assert {str(row["canonical7"]) for row in derived_rows} == set(network_names)
    assert evidence["feature_count"] == 400


def test_kong_canonical7_ignores_direct_seven_network_reference(tmp_path: Path) -> None:
    subject = "100610"
    ref_dir = tmp_path / f"sub-{subject}" / "reference" / "kong2022"
    ref_dir.mkdir(parents=True)
    rows = fc_contract.authoritative_kong2022_roi_registry()
    lines = ["parcel_id\tparcel_name\tfamily17\tcanonical8"]
    for row in rows:
        lines.append(
            f'{row["roi_id"]}\t{row["roi_name"]}\t{row["family17"]}\t{row["canonical8"]}'
        )
    (ref_dir / "cortex_roi_parcels.tsv").write_text("\n".join(lines) + "\n", encoding="utf-8")
    np.save(ref_dir / "cortex_roi_parcel_timeseries.npy", np.arange(400 * 5, dtype=np.float32).reshape(400, 5))
    (ref_dir / "cortex_canonical7_networks.tsv").write_text(
        "canonical7_network\n" + "Wrong\n" * 7, encoding="utf-8"
    )
    np.save(ref_dir / "cortex_canonical7_network_timeseries.npy", np.zeros((7, 5), dtype=np.float32))

    names, derived = atlas.load_kong2022_network_reference(
        tmp_path, subject, "kong2022", "canonical7"
    )
    expected_rows = atlas.canonical7_registry_from_rows(
        fc_contract.read_kong2022_roi_registry(ref_dir / "cortex_roi_parcels.tsv")
    )
    expected = label_utils.aggregate_kong2022_network_timeseries(
        np.load(ref_dir / "cortex_roi_parcel_timeseries.npy"),
        expected_rows,
        names,
        network_column="canonical7",
    )
    assert names == label_utils.canonical_families_7_order()
    assert np.allclose(derived, expected)


def test_group_atlas_and_weighted_helpers_accept_canonical7() -> None:
    assert atlas.resolve_kong2022_network_registry_column("canonical7") == "canonical7"
    assert atlas.network_reference_filenames("canonical7") == (
        "cortex_canonical7_networks.tsv",
        "cortex_canonical7_network_timeseries.npy",
    )
    assert atlas.method_allowed_for_atlas("gradient-spectral", "kong2022")
    assert atlas.method_uses_kong2022_topn_cluster_annotation("gradient-spectral", "kong2022")
    assert atlas.fc_mode_for_method("gradient-spectral", "kong2022") == (
        "pearson_vertex_to_kong2022_400roi_group_average_then_diffusion_gradients"
    )
    assert gradients.atlas_for_gradient_route("pfm") == "kong2022"
    assert atlas.gradient_route_for_route_tag("pfm") == "pfm"
    assert materializer.reference_fc_target_files("kong2022") == (
        "cortex_roi_parcels.tsv",
        "cortex_roi_parcel_timeseries.npy",
        "parcel_name",
    )
    assert materializer.roi_fc_filename_for_method("group-roi-sim-spectral") == "hipp_vertex_to_roi_fc.npy"
    assert materializer.roi_fc_filename_for_method("wta-top") == "hipp_vertex_to_roi_fc.npy"
    assert weighted._network_column("canonical7") == "canonical7"
    assert "kong2022" in overview.ATLASES
    assert "pfm" in overview.ROUTE_TAG_CHOICES
    assert "pfm_7net" not in overview.ROUTE_TAG_CHOICES


def test_kong2022_default_has_the_roi_profile_contract() -> None:
    assert atlas.atlas_produces_cluster_roi_profile("kong2022")
    assert not atlas.atlas_produces_cluster_roi_profile("kong2022_7net")
    assert not atlas.atlas_produces_cluster_roi_profile("kong2019")
    assert not atlas.atlas_produces_cluster_roi_profile("lynch2024_7net")

def test_taxonomy_switch_reuses_same_400_roi_cluster_profile() -> None:
    registry = fc_contract.authoritative_kong2022_roi_registry()
    rng = np.random.default_rng(20260910)
    profile_3d = rng.normal(size=(4, 3, 400)).astype(np.float32)
    profile_7 = atlas.cluster_assignment_variants_from_kong2022_roi_profile(
        roi_profile=np.nanmean(profile_3d, axis=0),
        roi_profile_3d=profile_3d,
        roi_registry=registry,
        network_names=label_utils.canonical_families_7_order(),
        q_thresh=0.05,
        top_n_cap=20,
        network_column="canonical7",
    )
    profile_8 = atlas.cluster_assignment_variants_from_kong2022_roi_profile(
        roi_profile=np.nanmean(profile_3d, axis=0),
        roi_profile_3d=profile_3d,
        roi_registry=registry,
        network_names=label_utils.canonical_families_8_order(),
        q_thresh=0.05,
        top_n_cap=20,
        network_column="canonical8",
    )
    assert profile_7["normalized_top20_count"]["profile"].shape == (3, 7)
    assert profile_8["normalized_top20_count"]["profile"].shape == (3, 8)
    # Raw top-20 ROI selection and Fisher-z statistics are shared; only the
    # network column used to aggregate and label them changes.
    selected_7 = profile_7["normalized_top20_count"]["status_rows"][0]["top_positive_selected"]
    selected_8 = profile_8["normalized_top20_count"]["status_rows"][0]["top_positive_selected"]
    assert [(row["roi_idx"], row["fisher_z"]) for row in selected_7] == [
        (row["roi_idx"], row["fisher_z"]) for row in selected_8
    ]
    assert profile_7["fisherz"]["assignment_metric"] == "top1_fisher_z"
    assert profile_8["fisherz"]["assignment_metric"] == "top1_fisher_z"


def test_routea_extractor_preserves_canonical7_metadata() -> None:
    registry = [
        {
            "roi_id": 1,
            "roi_name": "17networks_LH_Aud_1",
            "hemi": "L",
            "family17": "Aud",
            "subregion": "",
            "rank": 1,
            "canonical8": "Auditory",
            "canonical7": "Somatomotor",
        }
    ]
    _ts, kept, _empty, _stats = extractor.aggregate_hemi_roi_timeseries(
        hemi="L",
        hemi_dt=np.asarray([[1.0, 2.0, 3.0], [3.0, 4.0, 5.0]], dtype=np.float32),
        aligned_parcel_ids=np.asarray([1, 1], dtype=np.int32),
        valid_mask=np.asarray([True, True]),
        roi_registry=registry,
    )

    assert kept[0]["canonical7"] == "Somatomotor"


def test_kong2022_default_is_a_direct_canonical7_05_02_branch() -> None:
    args = atlas.build_arg_parser().parse_args(
        [
            "--atlases",
            "kong2022",
            "--network-taxonomy",
            "canonical7",
            "--route-tag",
            "pfm",
        ]
    )

    assert args.atlases == ["kong2022"]
    assert args.network_taxonomy == "canonical7"
    assert args.route_tag == "pfm"
    assert not hasattr(args, "reuse_source_root")


def test_default_parsers_select_canonical7_without_a_7net_suffix() -> None:
    assert reference_builder.DEFAULT_SHARED_ROOT.name == "pfm"
    assert extractor.build_arg_parser().parse_args(["--subject", "01", "--dataset", "hcp"]).atlas_slug == "kong2022"
    assert atlas.build_arg_parser().parse_args([]).network_taxonomy == "canonical7"


def test_canonical7_registry_is_derived_from_existing_kong2022_registry() -> None:
    rows = [
        {
            "roi_id": 1,
            "roi_name": "auditory",
            "hemi": "L",
            "family17": "Aud",
            "subregion": "",
            "rank": 1,
            "canonical8": "Auditory",
        },
        {
            "roi_id": 2,
            "roi_name": "somatomotor",
            "hemi": "R",
            "family17": "SomMotA",
            "subregion": "",
            "rank": 2,
            "canonical8": "Somatomotor",
        },
    ]

    got = atlas.canonical7_registry_from_rows(rows, require_all_networks=False)

    assert [row["canonical7"] for row in got] == ["Somatomotor", "Somatomotor"]
    assert [row["canonical8"] for row in got] == ["Auditory", "Somatomotor"]


def test_canonical7_template_profile_reads_shared_kong2022_registry() -> None:
    assert template_profiles.WTA_NETWORK_COLUMNS["canonical7"] == "canonical7"
    assert template_profiles.wta_run_dir(
        group_root=Path("outputs_hcp/work/group_atlas/routea_full_k3_fisherz"),
        smoothing="2mm",
        hemi="L",
        tau_tag="cov_0p80",
    ).parts[-5:] == ("wta-top", "kong2022", "2mm", "cov_0p80", "hemi_L")


def _write_minimal_wta_consumer_fixture(tmp_path: Path, manifest: dict[str, object]) -> Path:
    group_root = tmp_path / "group"
    run_dir = template_profiles.wta_run_dir(
        group_root=group_root,
        smoothing="2mm",
        hemi="L",
        tau_tag="cov_0p80",
    )
    run_dir.mkdir(parents=True)
    np.save(run_dir / "group_support_mask.npy", np.asarray([False], dtype=bool))
    np.save(run_dir / "group_functional_atlas_labels.npy", np.asarray([0], dtype=np.int32))
    (run_dir / "vertex_assignment_status.json").write_text(
        json.dumps({"items": [], "assignment_metric": "top20_count_normalized_by_network_roi_count"}),
        encoding="utf-8",
    )
    (run_dir / "group_atlas_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return group_root


def test_overview_rejects_all_unassigned_wta_support_domain(tmp_path: Path) -> None:
    group_root = tmp_path / "group"
    run_dir = template_profiles.wta_run_dir(
        group_root=group_root,
        smoothing="2mm",
        hemi="L",
        tau_tag="cov_0p80",
    )
    run_dir.mkdir(parents=True)
    (run_dir / "group_atlas_manifest.json").write_text(
        json.dumps(
            {
                "status": "ok",
                "network_taxonomy": "canonical7",
                "network_order": label_utils.canonical_families_7_order(),
                "n_support_vertices": 2,
            }
        ),
        encoding="utf-8",
    )
    (run_dir / "vertex_assignment_status.json").write_text(
        json.dumps(
            {
                "items": [
                    {"vertex_index": 0, "status": "dominant", "top1_network": ""},
                    {"vertex_index": 1, "status": "dominant", "top1_network": ""},
                ]
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(RuntimeError, match="unassigned"):
        overview.ensure_combo_ready(
            group_root=group_root,
            method="wta-top",
            atlas="kong2022",
            coverage_tag="cov_0p80",
            smoothings=["2mm"],
            hemis=["L"],
        )


def test_wta_consumer_requires_explicit_taxonomy_metadata(tmp_path: Path) -> None:
    group_root = _write_minimal_wta_consumer_fixture(
        tmp_path,
        {
            "status": "ok",
            "default_assignment_variant": "normalized_top20_count",
            "cluster_roi_registry": str(tmp_path / "registry.tsv"),
            "subjects": ["sub-01"],
        },
    )

    with pytest.raises(ValueError, match="taxonomy mismatch"):
        template_profiles.load_wta_run(
            group_root=group_root,
            smoothing="2mm",
            hemi="L",
            tau_tag="cov_0p80",
            taxonomy="canonical8",
        )


def test_wta_consumer_does_not_fallback_to_run_local_registry(tmp_path: Path) -> None:
    group_root = _write_minimal_wta_consumer_fixture(
        tmp_path,
        {
            "status": "ok",
            "default_assignment_variant": "normalized_top20_count",
            "network_column": "canonical8",
            "subjects": ["sub-01"],
        },
    )
    (group_root / "wta-top" / "kong2022" / "2mm" / "cov_0p80" / "hemi_L" / "cluster_roi_registry.tsv").write_text(
        "", encoding="utf-8"
    )

    with pytest.raises(ValueError, match="missing cluster_roi_registry"):
        template_profiles.load_wta_run(
            group_root=group_root,
            smoothing="2mm",
            hemi="L",
            tau_tag="cov_0p80",
            taxonomy="canonical8",
        )


def test_template_producer_canonicalizes_and_consumer_rejects_foreign_paths(
    tmp_path: Path,
    monkeypatch,
) -> None:
    dataset_root = tmp_path / "outputs_iowa_estt"
    template_root = dataset_root / "inputs" / "kong2022" / "templates" / "hipp_group_average"
    spec_root = dataset_root / "inputs" / "hippunfold" / "dense_corobl_batch"
    surface_name = "group-average_hemi-L_space-corobl_den-512_label-hipp_midthickness.surf.gii"
    spec_name = "sub-01_den-512_label-hipp_surfaces.spec"
    canonical_surface = template_root / "group_average_surfaces" / surface_name
    canonical_spec = spec_root / "sub-01" / "hippunfold" / "sub-01" / "surf" / spec_name
    canonical_surface.parent.mkdir(parents=True)
    canonical_spec.parent.mkdir(parents=True)
    canonical_surface.write_bytes(b"canonical")
    canonical_spec.write_text("spec fixture\n", encoding="utf-8")

    monkeypatch.setattr(
        template_assets,
        "resolve_kong2022_template_root",
        lambda dataset, variant="36p": template_root,
    )
    monkeypatch.setattr(
        template_assets,
        "resolve_hippunfold_batch_root",
        lambda dataset: spec_root,
    )

    manifest = {
        "dataset": "outputs_iowa_estt",
        "preprocessing_variant": "36p",
        "template_subject": "sub-01",
        "template_left_or_right_surface": str(tmp_path / "old_mount" / surface_name),
        "template_spec_path": str(tmp_path / "old_mount" / spec_name),
    }

    with pytest.raises(ValueError, match="not canonical"):
        template_assets.validate_template_assets(manifest, hemi="L", dataset="iowa_estt")

    canonicalized = template_assets.canonicalize_template_manifest(
        dict(manifest),
        hemi="L",
        dataset="iowa_estt",
    )
    assert Path(canonicalized["template_left_or_right_surface"]).resolve() == canonical_surface.resolve()
    assert Path(canonicalized["template_spec_path"]).resolve() == canonical_spec.resolve()
    assert canonicalized["template_subject"] == "sub-01"
    assert template_assets.validate_template_assets(
        canonicalized,
        hemi="L",
        dataset="iowa_estt",
    ) == (canonical_surface.resolve(), "01", canonical_spec.resolve())


def test_overview_panel_error_is_bounded() -> None:
    message = overview.format_panel_error(
        "Profile unavailable",
        FileNotFoundError("/a/very/long/legacy/path/that/must/not/fill/the/panel"),
    )

    assert len(message.splitlines()) <= 3
    assert all(len(line) <= 36 for line in message.splitlines())


def test_template_surface_validation_rejects_foreign_existing_path(
    tmp_path: Path,
    monkeypatch,
) -> None:
    canonical_root = tmp_path / "canonical"
    canonical_surface_dir = canonical_root / "group_average_surfaces"
    foreign_surface_dir = tmp_path / "foreign"
    canonical_surface_dir.mkdir(parents=True)
    foreign_surface_dir.mkdir(parents=True)
    surface_name = "group-average_hemi-L_space-corobl_den-512_label-hipp_midthickness.surf.gii"
    canonical_surface = canonical_surface_dir / surface_name
    foreign_surface = foreign_surface_dir / surface_name
    canonical_surface.write_bytes(b"canonical")
    foreign_surface.write_bytes(b"foreign")
    monkeypatch.setattr(
        template_assets,
        "resolve_kong2022_template_root",
        lambda dataset, variant="36p": canonical_root,
    )

    with pytest.raises(ValueError, match="not canonical"):
        template_assets.validate_template_surface_path(
            foreign_surface,
            dataset="stanford",
            hemi="L",
        )


def test_template_surface_validation_rejects_foreign_dataset_path(
    tmp_path: Path,
    monkeypatch,
) -> None:
    dataset_root = tmp_path / "outputs_iowa_jj_tms"
    canonical_root = dataset_root / "inputs" / "kong2022" / "templates" / "hipp_group_average"
    canonical_surface = canonical_root / "group_average_surfaces" / "group-average_hemi-L_space-corobl_den-512_label-hipp_midthickness.surf.gii"
    foreign_surface = tmp_path / "outputs_hcp" / "inputs" / "kong2022" / "templates" / "hipp_group_average" / "group_average_surfaces" / canonical_surface.name
    canonical_surface.parent.mkdir(parents=True)
    foreign_surface.parent.mkdir(parents=True)
    canonical_surface.write_bytes(b"iowa-canonical")
    foreign_surface.write_bytes(b"stale-hcp")
    monkeypatch.setattr(
        template_assets,
        "resolve_kong2022_template_root",
        lambda dataset, variant="36p": canonical_root,
    )

    with pytest.raises(ValueError, match="not canonical"):
        template_assets.validate_template_surface_path(
            foreign_surface,
            dataset="outputs_iowa_jj_tms",
            hemi="L",
        )




def test_05_02_group_atlas_rejects_retired_canonical8_taxonomy(monkeypatch) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "05_02_run_group_atlas.py",
            "--shared-root",
            "outputs_hcp/inputs/kong2022/shared_inputs/pfm",
            "--network-taxonomy",
            "canonical8",
            "--atlases",
            "kong2022",
        ],
    )
    with pytest.raises(ValueError, match="canonical8 taxonomy is retired"):
        atlas.main()


def test_wta_registry_derives_canonical7_from_legacy_400_roi_schema(tmp_path: Path) -> None:
    subject = "100610"
    ref_dir = tmp_path / f"sub-{subject}" / "reference" / "kong2022"
    ref_dir.mkdir(parents=True)
    source_rows = fc_contract.authoritative_kong2022_roi_registry()
    lines = ["parcel_id\tparcel_name\tfamily17\tcanonical8"]
    lines.extend(
        f'{row["roi_id"]}\t{row["roi_name"]}\t{row["family17"]}\t{row["canonical8"]}'
        for row in source_rows
    )
    (ref_dir / "cortex_roi_parcels.tsv").write_text("\n".join(lines) + "\n", encoding="utf-8")

    registry = atlas.read_roi_registry_for_atlas(tmp_path, subject, "kong2022")

    assert len(registry) == 400
    assert {str(row["canonical7"]) for row in registry} == set(label_utils.canonical_families_7_order())


def test_wta_rejects_empty_or_unknown_canonical7_registry_network() -> None:
    with pytest.raises(ValueError, match="canonical7"):
        atlas.build_normalized_topn_count_vertex_rows(
            mean_profile=np.asarray([[1.0]], dtype=np.float32),
            support_indices=np.asarray([0], dtype=np.int32),
            roi_registry=[{"canonical7": ""}],
            top_n_cap=1,
        )


def test_wta_rejects_dominant_network_that_has_no_functional_label() -> None:
    color_map = {name: (1, 2, 3, 255) for name in label_utils.canonical_families_7_order()}
    with pytest.raises(ValueError, match="dominant"):
        atlas.functional_atlas_from_vertices(
            n_vertices=1,
            status_rows=[
                {
                    "vertex_index": 0,
                    "status": "dominant",
                    "top1_network": "not-a-canonical7-network",
                }
            ],
            network_names=label_utils.canonical_families_7_order(),
            color_map=color_map,
        )
