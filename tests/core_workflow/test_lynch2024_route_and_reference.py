#!/usr/bin/env python3
from __future__ import annotations

import csv
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
CORE_DIR = REPO_ROOT / "scripts" / "core_workflow"
if str(CORE_DIR) not in sys.path:
    sys.path.insert(0, str(CORE_DIR))


def _load_numbered_module(script_name: str, alias: str):
    module_path = CORE_DIR / script_name
    spec = importlib.util.spec_from_file_location(alias, module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load module: {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[alias] = module
    spec.loader.exec_module(module)
    return module


atlas_mod = _load_numbered_module("05_02_run_group_atlas.py", "core_05_02_lynch_route")
compute_mod = _load_numbered_module("05_01_compute_hippomaps_gradients.py", "core_05_01_lynch_route")
overview_mod = _load_numbered_module("06_01_build_group_overview.py", "core_06_01_lynch_route")
fc_mod = _load_numbered_module("04_01_materialize_kong2022_roi_fc.py", "core_04_01_lynch_route")

from lib import route_tags  # noqa: E402
from lib import lynch2024_reference as reference_mod  # noqa: E402
from lib import output_layout  # noqa: E402
from lib.output_layout import (  # noqa: E402
    resolve_lynch2024_20net_group_atlas_root,
    resolve_lynch2024_nsi043_shared_inputs_root,
    resolve_lynch2024_shared_inputs_root,
)
from lib.lynch2024_reference import (  # noqa: E402
    build_lynch_network_references,
    lynch2024_20net_order,
)


def _write_component_tsv(path: Path) -> None:
    rows = [
        {"component_id": "1", "component_name": "Default_Parietal_L", "network": "Default_Parietal", "hemisphere": "L", "n_grayordinates_used": "3"},
        {"component_id": "2", "component_name": "Noise_L", "network": "Noise", "hemisphere": "L", "n_grayordinates_used": "5"},
        {"component_id": "3", "component_name": "Visual_V1_R", "network": "Visual_V1", "hemisphere": "R", "n_grayordinates_used": "2"},
        {"component_id": "4", "component_name": "Frontoparietal_R", "network": "Frontoparietal", "hemisphere": "R", "n_grayordinates_used": "4"},
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]), delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def test_lynch20_order_is_exactly_twenty_and_excludes_noise() -> None:
    assert hasattr(reference_mod, "lynch2024_20net_order")
    order = reference_mod.lynch2024_20net_order(REPO_ROOT / "config/cross_atlas_network_merge.json")
    assert len(order) == 20
    assert "Noise" not in order
    assert len(set(order)) == 20


def test_lynch20_order_excludes_noise_even_when_exclude_labels_is_empty(tmp_path: Path) -> None:
    source = REPO_ROOT / "config/cross_atlas_network_merge.json"
    payload = json.loads(source.read_text(encoding="utf-8"))
    payload["exclude_labels"] = []
    config = tmp_path / "merge.json"
    config.write_text(json.dumps(payload), encoding="utf-8")

    order = lynch2024_20net_order(config)

    assert len(order) == 20
    assert "Noise" not in order


def test_lynch20_runtime_names_are_removed() -> None:
    retired_route = f"lynch2024_{20 + 1}net"
    retired_taxonomy = f"lynch{20 + 1}"
    assert "lynch2024_20net" in route_tags.ROUTE_TAG_CHOICES
    assert retired_route not in route_tags.ROUTE_TAG_CHOICES
    assert "lynch20" in route_tags.NETWORK_TAXONOMY_CHOICES
    assert retired_taxonomy not in route_tags.NETWORK_TAXONOMY_CHOICES


def test_lynch7_reuses_lynch20_gradient_bank() -> None:
    assert atlas_mod.gradient_route_for_route_tag("lynch2024_7net") == "lynch2024_20net"


def test_lynch7_uses_kong_canonical7_order_and_merge_metadata() -> None:
    payload = json.loads((REPO_ROOT / "config/cross_atlas_network_merge.json").read_text(encoding="utf-8"))
    lynch = payload["atlases"]["lynch2024"]
    assert lynch["merged_functional_network_count"] == 7
    assert lynch["mapping"]["Auditory"] == "Somatomotor"
    assert getattr(reference_mod, "lynch2024_7net_order", None) is not None
    assert reference_mod.lynch2024_7net_order(REPO_ROOT / "config/cross_atlas_network_merge.json") == [
        "Default",
        "Visual",
        "Somatomotor",
        "DorsalAttention",
        "VentralAttention",
        "Control",
        "Language",
    ]


def test_legacy_lynch8_cli_is_rejected() -> None:
    with pytest.raises(SystemExit):
        atlas_mod.build_arg_parser().parse_args(["--route-tag", "lynch2024_" + "8net"])


def test_lynch_reference_orders_exclude_noise_and_merge_to_canonical7_weighted(tmp_path: Path) -> None:
    formal_order = lynch2024_20net_order(REPO_ROOT / "config/cross_atlas_network_merge.json")
    component_ts = np.asarray(
        [[float(i), float(i + 1), float(i + 2)] for i in range(20)]
        + [[100.0, 100.0, 100.0]],
        dtype=np.float32,
    )
    component_tsv = tmp_path / "components.tsv"
    weights = {network: "1" for network in formal_order}
    weights["Auditory"] = "9"
    rows = [
        {
            "component_id": str(i + 1),
            "component_name": f"{network}_component",
            "network": network,
            "hemisphere": "L",
            "n_grayordinates_used": weights[network],
        }
        for i, network in enumerate(formal_order)
    ]
    rows.append(
        {
            "component_id": "21",
            "component_name": "Noise_component",
            "network": "Noise",
            "hemisphere": "L",
            "n_grayordinates_used": "1",
        }
    )
    with component_tsv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]), delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)
    outputs = build_lynch_network_references(
        component_timeseries=component_ts,
        component_tsv=component_tsv,
        out_dir=tmp_path / "reference",
        merge_config=REPO_ROOT / "config" / "cross_atlas_network_merge.json",
    )

    assert "Noise" not in lynch2024_20net_order(REPO_ROOT / "config" / "cross_atlas_network_merge.json")
    assert reference_mod.lynch2024_7net_order(REPO_ROOT / "config" / "cross_atlas_network_merge.json") == [
        "Default",
        "Visual",
        "Somatomotor",
        "DorsalAttention",
        "VentralAttention",
        "Control",
        "Language",
    ]
    assert outputs["lynch20_order"] == formal_order
    assert outputs["lynch7_order"] == reference_mod.lynch2024_7net_order(
        REPO_ROOT / "config" / "cross_atlas_network_merge.json"
    )
    np.testing.assert_allclose(np.load(outputs["lynch20_timeseries"]), component_ts[:20])
    assert "Noise" not in outputs["lynch20_tsv"].read_text(encoding="utf-8")
    assert "Noise" not in outputs["lynch7_tsv"].read_text(encoding="utf-8")
    assert outputs["lynch7_tsv"].read_text(encoding="utf-8").splitlines()[0].startswith("canonical7_network\t")
    smn_idx = outputs["lynch7_order"].index("Somatomotor")
    expected_smn = np.average(
        component_ts[
            [formal_order.index(name) for name in formal_order if json.loads((REPO_ROOT / "config/cross_atlas_network_merge.json").read_text(encoding="utf-8"))["atlases"]["lynch2024"]["mapping"][name] == "Somatomotor"],
            :,
        ],
        axis=0,
        weights=[float(weights[name]) for name in formal_order if json.loads((REPO_ROOT / "config/cross_atlas_network_merge.json").read_text(encoding="utf-8"))["atlases"]["lynch2024"]["mapping"][name] == "Somatomotor"],
    )
    np.testing.assert_allclose(np.load(outputs["lynch7_timeseries"])[smn_idx], expected_smn)
    assert not np.allclose(np.load(outputs["lynch7_timeseries"])[smn_idx], np.mean(
        component_ts[
            [formal_order.index(name) for name in formal_order if json.loads((REPO_ROOT / "config/cross_atlas_network_merge.json").read_text(encoding="utf-8"))["atlases"]["lynch2024"]["mapping"][name] == "Somatomotor"],
            :,
        ],
        axis=0,
    ))


def test_lynch_reference_rejects_incomplete_formal_network_coverage(tmp_path: Path) -> None:
    component_tsv = tmp_path / "components.tsv"
    _write_component_tsv(component_tsv)
    out_dir = tmp_path / "reference"

    with pytest.raises(ValueError, match="exactly the 20 formal networks"):
        build_lynch_network_references(
            component_timeseries=np.ones((4, 3), dtype=np.float32),
            component_tsv=component_tsv,
            out_dir=out_dir,
            merge_config=REPO_ROOT / "config" / "cross_atlas_network_merge.json",
        )

    assert not out_dir.exists()


def test_lynch_routes_and_roots_are_distinct_from_kong_routea() -> None:
    shared = resolve_lynch2024_shared_inputs_root()
    assert shared.parts[-3:] == ("lynch2024_fast_pfm_t1only", "shared_inputs", "pfm_hcp151")
    assert resolve_lynch2024_nsi043_shared_inputs_root().name == "pfm_nsi043_78"
    assert resolve_lynch2024_20net_group_atlas_root().name == "lynch2024_20net_hcp151_full_k3_fisherz"
    assert route_tags.infer_route_tag_from_shared_root(shared) == "lynch2024_20net"
    assert route_tags.infer_route_tag_from_group_root(resolve_lynch2024_20net_group_atlas_root()) == "lynch2024_20net"
    lynch7_root = output_layout.resolve_lynch2024_7net_group_atlas_root()
    assert route_tags.infer_route_tag_from_group_root(lynch7_root) == "lynch2024_7net"
    route_tags.validate_group_atlas_root_for_route(lynch7_root, "lynch2024_7net")
    with pytest.raises(ValueError, match="Provenance discrepancy"):
        route_tags.validate_group_atlas_root_for_route(lynch7_root, "routea")


def test_05_01_resolves_lynch_shared_fc_roots_and_atlas_slug() -> None:
    shared_root = Path("/tmp/shared_inputs")
    route_shared = compute_mod.resolve_gradient_shared_root(shared_root, "lynch2024_7net")
    assert route_shared == shared_root / "pfm_hcp151"
    assert compute_mod.atlas_for_gradient_route("lynch2024_20net") == "lynch2024_20net"
    assert compute_mod.atlas_for_gradient_route("lynch2024_7net") == "lynch2024_20net"
    fc_path = compute_mod.subject_fc_path(route_shared, "1001", "lynch2024_20net", "2mm", "L")
    assert fc_path.as_posix().endswith(
        "/pfm_hcp151/sub-1001/fc/lynch2024_20net/2mm/hemi_L/hipp_vertex_to_network_fc.npy"
    )


def test_04_01_lynch20_fc_targets_formal_network_timeseries() -> None:
    assert fc_mod.reference_fc_target_files("lynch2024_20net") == (
        "cortex_lynch20_networks.tsv",
        "cortex_lynch20_network_timeseries.npy",
        "lynch20_network",
    )


def test_05_02_accepts_lynch_gradient_and_wta() -> None:
    parser = atlas_mod.build_arg_parser()
    args = parser.parse_args(
        [
            "--route-tag",
            "lynch2024_7net",
            "--network-taxonomy",
            "lynch7",
            "--profile-surface-root",
            "/tmp/profile_surface",
            "--atlases",
            "lynch2024_7net",
            "--methods",
            "gradient-spectral",
            "wta-top",
        ]
    )
    assert args.route_tag == "lynch2024_7net"
    assert args.network_taxonomy == "lynch7"
    assert args.profile_surface_root == "/tmp/profile_surface"
    assert "lynch2024_7net" in atlas_mod.ATLASES
    atlas_mod.validate_requested_methods_for_taxonomy(
        args.methods,
        args.network_taxonomy,
        args.atlases,
    )


def test_lynch_gradient_metadata_describes_network_fc_not_roi_fc() -> None:
    assert atlas_mod.feature_family_for_method(
        "gradient-spectral", "lynch2024_7net", (1, 2, 3)
    ) == "group_vertex_to_lynch2024_7net_network_fc_diffusion_gradients_g1g2g3"
    assert atlas_mod.fc_mode_for_method(
        "gradient-spectral", "lynch2024_7net"
    ) == "pearson_vertex_to_network_group_average_then_diffusion_gradients"
    assert "lynch2024_7net" not in atlas_mod.DEFAULT_ATLASES
    assert "lynch2024_7net" not in overview_mod.DEFAULT_ATLASES
    assert atlas_mod.method_allowed_for_atlas("gradient-spectral", "lynch2024_7net") is True
    assert atlas_mod.method_allowed_for_atlas("wta-top", "lynch2024_7net", network_taxonomy="lynch7") is True
    assert atlas_mod.method_allowed_for_atlas("network-spectral", "lynch2024_7net") is False
    assert atlas_mod.method_allowed_for_atlas("intrinsic-spectral", "lynch2024_20net") is False
    assert atlas_mod.method_allowed_for_atlas("wta-top", "kong2019") is False
    assert atlas_mod.method_allowed_for_atlas("gradient-spectral", "kong2019") is False
    assert overview_mod.resolve_render_assignment_variants("wta-top", "both") == ("normalized_top20_count",)
    assert overview_mod.resolve_render_assignment_variants(
        "wta-top", "both", atlas="lynch2024_7net"
    ) == ("fisherz",)
    assert overview_mod.resolve_render_assignment_variants("gradient-spectral", "both") == ("fisherz", "normalized_top20_count")
    assert overview_mod.resolve_render_assignment_variants(
        "gradient-spectral", "both", atlas="lynch2024_7net"
    ) == ("fisherz",)
    with pytest.raises(ValueError, match="Lynch gradient-spectral"):
        overview_mod.resolve_render_assignment_variants(
            "gradient-spectral", "normalized_top20_count", atlas="lynch2024_7net"
        )
    with pytest.raises(ValueError, match="normalized_top20_count"):
        overview_mod.resolve_render_assignment_variants(
            "wta-top", "normalized_top20_count", atlas="lynch2024_7net"
        )
    assert atlas_mod.feature_family_for_method(
        "wta-top", "lynch2024_7net", (1, 2, 3)
    ) == "vertex_to_lynch2024_7net_network_fc_fisherz"



def test_lynch_uses_same_vertex_fc_profile_redundancy_branch_as_kong() -> None:
    assert atlas_mod.redundancy_profile_applicable("lynch2024_20net") is True
    assert atlas_mod.redundancy_profile_applicable("lynch2024_7net") is True
    assert atlas_mod.redundancy_profile_applicable("kong2022") is True
    legacy_name = "_".join(["cluster", "roi", "profile", "from", "fc", "bank"])
    assert not hasattr(atlas_mod, legacy_name)


def test_default_k_is_separate_from_forced_final_k() -> None:
    args = atlas_mod.build_arg_parser().parse_args([])
    assert args.force_final_k is None
    assert args.default_k == 3
    assert hasattr(atlas_mod, "resolve_primary_k")


def test_primary_k_provenance_preserves_automatic_selection() -> None:
    assert hasattr(atlas_mod, "primary_k_provenance")
    assert atlas_mod.primary_k_provenance(
        force_final_k=None, default_k=3, automatic_k_final=5
    ) == {
        "chosen_k_final": 3,
        "automatic_k_final": 5,
        "primary_k_source": "default_k",
    }


def test_lynch_cluster_profile_uses_cluster_time_series_correlation(tmp_path: Path) -> None:
    subject = "1001"
    smoothing = "2mm"
    hemi = "L"
    profile_surface_root = tmp_path / "profile_surface"
    reference_root = tmp_path / "reference_root"
    ts_dir = profile_surface_root / f"sub-{subject}" / "surface" / smoothing
    ref_dir = reference_root / f"sub-{subject}" / "reference" / "lynch2024_7net"
    ts_dir.mkdir(parents=True)
    ref_dir.mkdir(parents=True)

    visual = np.asarray([-2.0, -1.0, 0.0, 1.0, 2.0], dtype=np.float32)
    default = np.asarray([2.0, 1.0, 0.0, -1.0, -2.0], dtype=np.float32)
    hippocampal_ts = np.stack(
        [
            visual,
            visual,
            default,
            default,
        ],
        axis=0,
    )
    network_ts = np.stack(
        [default, visual, default * 0.5, visual * 0.5, default * 0.25, visual * 0.25, default * 0.1],
        axis=0,
    )
    np.save(ts_dir / f"sub-{subject}_hemi-{hemi}_timeseries.npy", hippocampal_ts)
    np.save(ref_dir / "cortex_canonical7_network_timeseries.npy", network_ts)
    (ref_dir / "reference_summary.json").write_text(json.dumps({
        "run_ids": [1], "run_lengths": [network_ts.shape[1]],
    }))
    (ref_dir / "cortex_canonical7_networks.tsv").write_text(
        "canonical7_network\nDefault\nVisual\nSomatomotor\nDorsalAttention\nVentralAttention\nControl\nLanguage\n",
        encoding="utf-8",
    )
    (ref_dir / "stage_manifest.json").write_text(
        json.dumps(reference_mod.lynch2024_7net_reference_contract(REPO_ROOT / "config/cross_atlas_network_merge.json")),
        encoding="utf-8",
    )

    profile = atlas_mod.cluster_network_profile(
        group_labels_final=np.asarray([1, 1, 2, 2], dtype=np.int32),
        support_mask=np.asarray([True, True, True, True], dtype=bool),
        subjects=[subject],
        shared_root=tmp_path / "unused_shared",
        surface_root=profile_surface_root,
        reference_root=reference_root,
        atlas="lynch2024_7net",
        smoothing=smoothing,
        hemi=hemi,
        valid_masks_by_subject={subject: np.asarray([True, True, True, True], dtype=bool)},
        profile_fc_mode="pearson",
        network_taxonomy="lynch7",
    )

    assert profile.shape == (2, 7)
    assert profile[0, 1] > 6.0
    assert profile[1, 0] > 6.0
    assert profile[0, 0] < -6.0
    assert profile[1, 1] < -6.0


def test_lynch7_cluster_annotation_collapses_component_profile_by_registry() -> None:
    component_profile = np.asarray(
        [
            [0.9, 0.7, 0.1],
            [0.1, 0.2, 0.8],
        ],
        dtype=np.float32,
    )
    roi_registry = [
        {"roi_id": 1, "roi_name": "default_a", "canonical7": "Default", "lynch20_network": "Default_A"},
        {"roi_id": 2, "roi_name": "default_b", "canonical7": "Default", "lynch20_network": "Default_B"},
        {"roi_id": 3, "roi_name": "visual_a", "canonical7": "Visual", "lynch20_network": "Visual_A"},
    ]

    profile, status_rows, dominant = atlas_mod.assignment_status_from_kong2022_roi_profile(
        component_profile,
        roi_registry,
        ["Default", "Visual"],
        network_column="lynch7",
    )

    assert profile.shape == (2, 2)
    np.testing.assert_allclose(profile[:, 0], [0.8, 0.15], atol=1e-6)
    np.testing.assert_allclose(profile[:, 1], [0.1, 0.8], atol=1e-6)
    assert dominant == ["Default", "Visual"]
    assert status_rows[0]["top1_network"] == "Default"
    assert status_rows[1]["top1_network"] == "Visual"

def test_06_01_lynch7_overview_names_do_not_fall_back_to_kong() -> None:
    out = overview_mod.overview_png_path(
        present_dir=Path("/tmp/present"),
        atlas="lynch2024_7net",
        method="gradient-spectral",
        route_tag="lynch2024_7net",
        mode_tag=None,
        assignment_variant="fisherz",
        gradient_comp_tag="g1g2g3",
    )
    assert out.name == "lynch2024_7net_gradient-spectral_group_overview_g1g2g3_lynch2024_7net_fisherz.png"


def test_04_01_lynch7_materializes_canonical7_fc_not_component20(tmp_path: Path) -> None:
    subject = "1001"
    shared_root = tmp_path / "shared"
    surface_root = tmp_path / "surface"
    reference_root = tmp_path / "reference"
    mask_root = tmp_path / "mask"
    ref_dir = reference_root / f"sub-{subject}" / "reference" / "lynch2024_7net"
    common_run_dir = reference_root / f"sub-{subject}" / "reference" / "common" / "runs"
    surf_dir = surface_root / f"sub-{subject}" / "surface" / "2mm"
    mask_dir = mask_root / f"sub-{subject}" / "surface" / "tsnr_gate"
    ref_dir.mkdir(parents=True)
    surf_dir.mkdir(parents=True)
    mask_dir.mkdir(parents=True)
    t = np.arange(8, dtype=np.float32)
    np.save(ref_dir / "cortex_roi_parcel_timeseries.npy", np.vstack([t + i for i in range(20)]).astype(np.float32))
    names = ["Default", "Visual", "Somatomotor", "DorsalAttention", "VentralAttention", "Control", "Language"]
    np.save(ref_dir / "cortex_canonical7_network_timeseries.npy", np.vstack([t * (i + 1) for i in range(7)]).astype(np.float32))
    with (ref_dir / "cortex_roi_parcels.tsv").open("w", encoding="utf-8") as f:
        f.write("parcel_id\tparcel_name\n")
        for i in range(20):
            f.write(f"{i+1}\tcomponent_{i+1}\n")
    with (ref_dir / "cortex_canonical7_networks.tsv").open("w", encoding="utf-8") as f:
        f.write("canonical7_network\tn_components_merged\n")
        for name in names:
            f.write(f"{name}\t1\n")
    (ref_dir / "stage_manifest.json").write_text(
        json.dumps(reference_mod.lynch2024_7net_reference_contract(REPO_ROOT / "config/cross_atlas_network_merge.json")),
        encoding="utf-8",
    )
    for rid in range(1, 5):
        run_dir = common_run_dir / f"run-{rid}"
        run_dir.mkdir(parents=True)
        start = (rid - 1) * 2
        (run_dir / "reference_summary.json").write_text(
            f'{{"run_id": {rid}, "run_start": {start}, "run_stop": {start + 2}, "run_length": 2}}',
            encoding="utf-8",
        )
    hipp_ts = np.vstack([t, t[::-1], np.ones_like(t)]).astype(np.float32)
    np.save(surf_dir / f"sub-{subject}_hemi-L_timeseries.npy", hipp_ts)
    np.save(mask_dir / f"sub-{subject}_hemi-L_valid_mask.npy", np.asarray([True, True, False]))
    (ref_dir / "reference_summary.json").write_text(json.dumps({
        "run_ids": [1, 2, 3, 4], "run_lengths": [2, 2, 2, 2],
    }))

    fc_mod.materialize_fc(
        shared_root,
        surface_root,
        subject,
        "2mm",
        "L",
        reference_root=reference_root,
        mask_root=mask_root,
        atlas="lynch2024_7net",
        dataset="hcp",
    )

    out_dir = shared_root / f"sub-{subject}" / "fc" / "lynch2024_7net" / "2mm" / "hemi_L"
    fc = np.load(out_dir / "hipp_vertex_to_network_fc.npy")
    summary_text = (out_dir / "fc_summary.json").read_text(encoding="utf-8")
    manifest_text = (out_dir / "stage_manifest.json").read_text(encoding="utf-8")
    assert fc.shape == (3, 7)
    assert '"n_roi_targets_used": 7' in summary_text
    assert "cortex_canonical7_network_timeseries.npy" in manifest_text
    assert "cortex_roi_parcel_timeseries.npy" not in manifest_text


def test_05_02_lynch7_rejects_roi_registry_adapter(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="network references, not an ROI registry"):
        atlas_mod.read_roi_registry_for_atlas(tmp_path, "1001", "lynch2024_7net")


def test_05_02_lynch7_gradient_uses_shared_lynch20_component_bank() -> None:
    assert atlas_mod.gradient_route_for_route_tag("lynch2024_7net") == "lynch2024_20net"
    assert atlas_mod.gradient_route_for_route_tag("lynch2024_20net") == "lynch2024_20net"
