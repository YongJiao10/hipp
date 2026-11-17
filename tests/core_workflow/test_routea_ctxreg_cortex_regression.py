#!/usr/bin/env python3
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _load_numbered_module(relpath: str, alias: str):
    module_path = REPO_ROOT / relpath
    spec = importlib.util.spec_from_file_location(alias, module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load module: {module_path}")
    core_dir = (REPO_ROOT / "scripts" / "core_workflow").resolve()
    if str(core_dir) not in sys.path:
        sys.path.insert(0, str(core_dir))
    module = importlib.util.module_from_spec(spec)
    sys.modules[alias] = module
    spec.loader.exec_module(module)
    return module


ctxreg_mod = _load_numbered_module(
    "scripts/core_workflow/04_00_materialize_cortex_regressed_surface_timeseries.py",
    "core_04_00_materialize_cortex_regressed_surface_timeseries",
)
compute_mod = _load_numbered_module(
    "scripts/core_workflow/05_01_compute_hippomaps_gradients.py",
    "core_05_01_compute_hippomaps_gradients_ctxreg_test",
)
atlas_mod = _load_numbered_module(
    "scripts/core_workflow/05_02_run_group_atlas.py",
    "core_05_02_run_group_atlas_ctxreg_test",
)
overview_mod = _load_numbered_module(
    "scripts/core_workflow/06_01_build_group_overview.py",
    "core_06_01_build_group_overview_ctxreg_test",
)
grad_render_mod = _load_numbered_module(
    "scripts/core_workflow/06_04_render_hippomaps_gradient_products.py",
    "core_06_04_render_hippomaps_gradient_products_ctxreg_test",
)
matrix_mod = _load_numbered_module(
    "scripts/core_workflow/07_01_build_group_matrix.py",
    "core_07_01_build_group_matrix_ctxreg_test",
)
roi_fc_mod = _load_numbered_module(
    "scripts/core_workflow/04_01_materialize_kong2022_roi_fc.py",
    "core_04_01_materialize_kong2022_roi_fc_ctxreg_test",
)
intrinsic_mod = _load_numbered_module(
    "scripts/core_workflow/04_02_materialize_intrinsic_fc.py",
    "core_04_02_materialize_intrinsic_fc_ctxreg_test",
)
cortex_mod = _load_numbered_module(
    "scripts/core_workflow/06_03_build_cluster_cortex_panels.py",
    "core_06_03_build_cluster_cortex_panels_ctxreg_test",
)
diagnostics_mod = _load_numbered_module(
    "scripts/core_workflow/06_07_render_local_cortex_control_diagnostics.py",
    "core_06_07_render_local_cortex_control_diagnostics_ctxreg_test",
)


def _ols_residual(y: np.ndarray, x: np.ndarray) -> np.ndarray:
    design = np.column_stack([x, np.ones_like(x)])
    beta, *_ = np.linalg.lstsq(design, y, rcond=None)
    return y - design @ beta


def test_local_cortex_regression_matches_hand_computed_ols_residuals() -> None:
    hipp = np.asarray([[1.0, 2.0, 4.0, 8.0], [2.0, 3.0, 5.0, 7.0]], dtype=np.float32)
    cortex = np.asarray(
        [
            [1.0, 1.0, 3.0, 5.0],
            [3.0, 5.0, 7.0, 9.0],
            [10.0, 11.0, 12.0, 13.0],
        ],
        dtype=np.float32,
    )
    result = ctxreg_mod.regress_local_cortex_signal(
        hipp_ts=hipp,
        cortex_ts=cortex,
        neighbor_indices=[np.asarray([0, 1], dtype=np.int64), np.asarray([2], dtype=np.int64)],
        run_segments=[{"run_id": 1, "run_start": 0, "run_stop": 4, "run_length": 4}],
    )
    expected0 = _ols_residual(hipp[0], np.mean(cortex[[0, 1]], axis=0))
    expected1 = _ols_residual(hipp[1], cortex[2])
    assert np.allclose(result.controlled_ts[0], expected0, atol=1e-6)
    assert np.allclose(result.controlled_ts[1], expected1, atol=1e-6)
    assert result.neighbor_counts.tolist() == [2, 1]
    assert result.skipped_no_neighbor.tolist() == [False, False]


def test_local_cortex_regression_is_runwise_and_does_not_mix_segments() -> None:
    hipp = np.asarray([[0.0, 1.0, 2.0, 9.0, 8.0, 7.0]], dtype=np.float32)
    cortex = np.asarray([[0.0, 1.0, 2.0, 2.0, 1.0, 0.0]], dtype=np.float32)
    segments = [
        {"run_id": 1, "run_start": 0, "run_stop": 3, "run_length": 3},
        {"run_id": 2, "run_start": 3, "run_stop": 6, "run_length": 3},
    ]
    result = ctxreg_mod.regress_local_cortex_signal(
        hipp_ts=hipp,
        cortex_ts=cortex,
        neighbor_indices=[np.asarray([0], dtype=np.int64)],
        run_segments=segments,
    )
    expected = np.concatenate([
        _ols_residual(hipp[0, 0:3], cortex[0, 0:3]),
        _ols_residual(hipp[0, 3:6], cortex[0, 3:6]),
    ])
    global_residual = _ols_residual(hipp[0], cortex[0])
    assert np.allclose(result.controlled_ts[0], expected, atol=1e-6)
    assert not np.allclose(result.controlled_ts[0], global_residual, atol=1e-6)


def test_vertices_without_nearby_cortex_keep_original_signal_and_are_flagged() -> None:
    hipp = np.asarray([[1.0, 2.0, 3.0], [4.0, 6.0, 8.0]], dtype=np.float32)
    cortex = np.asarray([[0.0, 1.0, 0.0]], dtype=np.float32)
    result = ctxreg_mod.regress_local_cortex_signal(
        hipp_ts=hipp,
        cortex_ts=cortex,
        neighbor_indices=[np.asarray([0], dtype=np.int64), np.asarray([], dtype=np.int64)],
        run_segments=[{"run_id": 1, "run_start": 0, "run_stop": 3, "run_length": 3}],
    )
    assert np.allclose(result.controlled_ts[1], hipp[1], atol=1e-6)
    assert result.neighbor_counts.tolist() == [1, 0]
    assert result.skipped_no_neighbor.tolist() == [False, True]


def test_regression_preserves_invalid_hippocampal_vertices() -> None:
    hipp = np.asarray([[1.0, 2.0, 3.0], [4.0, 6.0, 8.0]], dtype=np.float32)
    cortex = np.asarray([[0.0, 1.0, 0.0]], dtype=np.float32)
    result = ctxreg_mod.regress_local_cortex_signal(
        hipp_ts=hipp,
        cortex_ts=cortex,
        neighbor_indices=[np.asarray([0]), np.asarray([0])],
        run_segments=[{"run_id": 1, "run_start": 0, "run_stop": 3, "run_length": 3}],
        hipp_valid_mask=np.asarray([True, False]),
    )
    assert np.allclose(result.controlled_ts[1], hipp[1], atol=1e-6)
    assert result.hipp_valid_mask.tolist() == [True, False]


def test_regression_rejects_all_nan_cortical_neighbor_signal() -> None:
    hipp = np.asarray([[1.0, 2.0, 3.0]], dtype=np.float32)
    cortex = np.asarray([[np.nan, np.nan, np.nan]], dtype=np.float32)
    try:
        ctxreg_mod.regress_local_cortex_signal(
            hipp_ts=hipp,
            cortex_ts=cortex,
            neighbor_indices=[np.asarray([0])],
            run_segments=[{"run_id": 1, "run_start": 0, "run_stop": 3, "run_length": 3}],
        )
    except ValueError as exc:
        assert "Unusable regression data" in str(exc)
    else:
        raise AssertionError("all-NaN cortical neighbors must fail closed")


def test_cortex_valid_masks_are_required_by_default(tmp_path: Path) -> None:
    try:
        ctxreg_mod.load_optional_cortex_valid_mask(
            tmp_path,
            "1",
            {"CORTEX_LEFT": 2, "CORTEX_RIGHT": 2},
        )
    except FileNotFoundError as exc:
        assert "explicit cortical validity mask" in str(exc)
    else:
        raise AssertionError("missing cortical masks must fail closed")


def test_explicit_run_metadata_preserves_nonconsecutive_ids(tmp_path: Path) -> None:
    metadata = tmp_path / "runs.json"
    metadata.write_text(
        '{"run_ids": [1, 3], "run_lengths": [2, 5], "concat_timepoints": 7}',
        encoding="utf-8",
    )
    assert ctxreg_mod.load_run_segments(tmp_path, "1", run_metadata=metadata) == [
        {"run_id": 1, "run_start": 0, "run_stop": 2, "run_length": 2},
        {"run_id": 3, "run_start": 2, "run_stop": 7, "run_length": 5},
    ]


def test_coordinate_gate_rejects_implausible_coordinate_spaces() -> None:
    hipp_coords = np.asarray([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=np.float32)
    cortex_coords = np.asarray([[1000000.0, 0.0, 0.0], [1000001.0, 0.0, 0.0]], dtype=np.float32)
    try:
        ctxreg_mod.validate_coordinate_space_compatibility(
            hipp_coords=hipp_coords,
            cortex_coords=cortex_coords,
            distance_mm=20.0,
        )
    except ValueError as exc:
        assert "coordinate-space" in str(exc)
    else:
        raise AssertionError("expected coordinate-space validation failure")


def test_diagnostic_default_cortex_surface_template_matches_bids_input_layout() -> None:
    left = ctxreg_mod.format_subject_template(
        diagnostics_mod.DEFAULT_LEFT_SURFACE,
        data_root=Path("/data"),
        subject="100610",
        hemi="L",
    )
    right = ctxreg_mod.format_subject_template(
        diagnostics_mod.DEFAULT_RIGHT_SURFACE,
        data_root=Path("/data"),
        subject="100610",
        hemi="R",
    )
    assert left.as_posix() == "/data/sub-100610/anat/sub-100610_hemi-L_space-fsLR_den-32k_desc-MSMAll_midthickness.surf.gii"
    assert right.as_posix() == "/data/sub-100610/anat/sub-100610_hemi-R_space-fsLR_den-32k_desc-MSMAll_midthickness.surf.gii"


def test_routea_ctx_control_route_contracts_and_paths() -> None:
    assert compute_mod.ROUTE_TO_SHARED_DIR["pfm_ctxreg"] == "pfm_ctxreg"
    assert atlas_mod.infer_route_tag(Path("/data/kong2022/shared_inputs/pfm_ctxreg"), "auto") == "pfm_ctxreg"
    assert "pfm_ctxreg" in overview_mod.ROUTE_TAG_CHOICES
    assert "pfm_ctxreg" in grad_render_mod.ROUTES
    assert "pfm_ctxreg" in matrix_mod.ROUTE_CHOICES
    for retired in (compute_mod.ROUTE_TO_SHARED_DIR, overview_mod.ROUTE_TAG_CHOICES, grad_render_mod.ROUTES, matrix_mod.ROUTE_CHOICES):
        assert "pfm_ctxdrop" not in retired

    overview_path = matrix_mod.expected_overview_path(
        Path("/tmp/present"),
        method="network-spectral",
        route_tag="pfm_ctxreg",
        mode_tag=None,
        assignment_variant="fisherz", smoothing="2mm",
    )
    assert overview_path.name == "kong2022_network-spectral_group_overview_pfm_ctxreg_fisherz.png"



def test_routea_17net_reuses_routea_inputs_and_has_shallow_output_contract() -> None:
    assert "pfm_17net" not in compute_mod.ROUTE_TO_SHARED_DIR
    assert atlas_mod.gradient_route_for_route_tag("pfm_17net") == "pfm"
    assert atlas_mod.gradient_route_for_route_tag("fixed_17net") == "fixed"
    assert "pfm_17net" in atlas_mod.ROUTE_TAG_CHOICES
    assert "fixed_17net" in atlas_mod.ROUTE_TAG_CHOICES
    assert "pfm_17net" in overview_mod.ROUTE_TAG_CHOICES
    assert "fixed_17net" in overview_mod.ROUTE_TAG_CHOICES
    assert "pfm_17net" in matrix_mod.ROUTE_CHOICES
    assert "fixed_17net" in matrix_mod.ROUTE_CHOICES

    route_roots = matrix_mod.canonical_group_roots(REPO_ROOT / "outputs_hcp" / "work" / "group_atlas")
    assert route_roots["pfm_17net"] == (
        REPO_ROOT / "outputs_hcp" / "work" / "group_atlas" / "pfm_17net"
    ).resolve()
    assert route_roots["fixed_17net"] == (
        REPO_ROOT / "outputs_hcp" / "work" / "group_atlas" / "fixed_17net"
    ).resolve()


def test_routea_ctxreg_fc_steps_use_separate_reference_and_mask_roots(tmp_path: Path, cortex_reference_factory) -> None:
    import json

    subject = "101"
    out_root = tmp_path / "routea_ctxreg"
    controlled_surface = tmp_path / "controlled_surface"
    routea_reference = tmp_path / "routea"
    source_masks = tmp_path / "surface"

    ts_dir = controlled_surface / f"sub-{subject}" / "surface" / "2mm"
    ts_dir.mkdir(parents=True)
    np.save(ts_dir / f"sub-{subject}_hemi-L_timeseries.npy", np.asarray([[1, 2, 3, 4], [2, 3, 4, 5]], dtype=np.float32))

    mask_dir = source_masks / f"sub-{subject}" / "surface" / "validity"
    mask_dir.mkdir(parents=True)
    np.save(mask_dir / f"sub-{subject}_hemi-L_valid_mask.npy", np.asarray([True, False], dtype=bool))

    ref_dir = routea_reference / f"sub-{subject}" / "reference" / "kong2022"
    ref_dir.mkdir(parents=True)
    registry = roi_fc_mod.authoritative_kong2022_roi_registry()
    np.save(
        ref_dir / "cortex_roi_parcel_timeseries.npy",
        np.asarray([[float(i), float(i + 1), float(i + 2), float(i + 3)] for i in range(400)], dtype=np.float32),
    )
    registry_lines = ["parcel_id\tparcel_name\tcanonical7"]
    registry_lines.extend(f'{row["roi_id"]}\t{row["roi_name"]}\t{row["canonical7"]}' for row in registry)
    (ref_dir / "cortex_roi_parcels.tsv").write_text("\n".join(registry_lines) + "\n", encoding="utf-8")
    np.save(ref_dir / "cortex_canonical7_network_timeseries.npy", np.asarray([[1, 2, 3, 4], [2, 3, 4, 5], [3, 4, 5, 6], [4, 5, 6, 7], [5, 6, 7, 8], [6, 7, 8, 9], [7, 8, 9, 10]], dtype=np.float32))
    (ref_dir / "cortex_canonical7_networks.tsv").write_text(
        "canonical7_network\nDefault\nVisual\nSomatomotor\nDorsalAttention\nVentralAttention\nControl\nLanguage\n",
        encoding="utf-8",
    )
    cortex_reference_factory(routea_reference, subject=subject, dataset="hcp", run_ids=(1, 2), run_lengths=(2, 2))
    for rid, start in enumerate([0, 2], start=1):
        run_dir = routea_reference / f"sub-{subject}" / "reference" / "common" / "runs" / f"run-{rid}"
        run_dir.mkdir(parents=True)
        (run_dir / "reference_summary.json").write_text(
            f'{{"run_id": {rid}, "run_start": {start}, "run_stop": {start + 2}, "run_length": 2}}',
            encoding="utf-8",
        )

    roi_fc_mod.materialize_fc(
        dataset="hcp",
        shared_root=out_root,
        surface_root=controlled_surface,
        reference_root=routea_reference,
        mask_root=source_masks,
        subject=subject,
        smoothing="2mm",
        hemi="L",
    )
    assert (out_root / f"sub-{subject}" / "fc" / "kong2022" / "2mm" / "hemi_L" / "hipp_vertex_to_network_fc.npy").exists()
    assert not (out_root / f"sub-{subject}" / "reference").exists()
    assert not (out_root / f"sub-{subject}" / "surface" / "validity").exists()

    intrinsic_mod.materialize_subject(
        shared_root=out_root,
        surface_root=controlled_surface,
        reference_root=routea_reference,
        mask_root=source_masks,
        subject=subject,
        smoothings=["2mm"],
        hemis=["L"],
    )
    assert (out_root / f"sub-{subject}" / "fc" / "intrinsic" / "2mm" / "hemi_L" / "hipp_vertex_to_vertex_fc_active.npy").exists()


def test_group_and_cortex_helpers_accept_reference_root_for_ctxreg() -> None:
    routea_ref = Path("/tmp/routea")
    ctxreg = Path("/tmp/routea_ctxreg")
    assert atlas_mod.reference_path(routea_ref, "101", "kong2022", "cortex_roi_parcels.tsv").as_posix().endswith(
        "/routea/sub-101/reference/kong2022/cortex_roi_parcels.tsv"
    )
    assert atlas_mod.reference_path(ctxreg, "101", "kong2022", "cortex_roi_parcels.tsv").as_posix().endswith(
        "/routea_ctxreg/sub-101/reference/kong2022/cortex_roi_parcels.tsv"
    )
    assert cortex_mod.subject_reference_path(routea_ref, "101", "kong2022", "cortex_roi_parcels.tsv").as_posix().endswith(
        "/routea/sub-101/reference/kong2022/cortex_roi_parcels.tsv"
    )


def test_hcp_control_writer_defaults_use_dated_400_roi_roots() -> None:
    expected_ctxreg = ctxreg_mod.DEFAULT_KONG400_CTXREG_SHARED_ROOT.resolve()
    assert ctxreg_mod.DEFAULT_OUT_SHARED_ROOT == expected_ctxreg


def test_hcp_gradient_control_routes_use_dated_400_roi_roots() -> None:
    assert compute_mod.resolve_gradient_shared_root(
        compute_mod.DEFAULT_SHARED_INPUTS,
        "pfm_ctxreg",
    ) == ctxreg_mod.DEFAULT_KONG400_CTXREG_SHARED_ROOT.resolve()


def test_local_cortex_corr_diagnostic_removes_run_offsets() -> None:
    rng = np.random.default_rng(7)
    noise_a = rng.standard_normal(40)
    noise_b = rng.standard_normal(40)
    segments = [
        {"run_id": 1, "run_start": 0, "run_stop": 20, "run_length": 20},
        {"run_id": 2, "run_start": 20, "run_stop": 40, "run_length": 20},
    ]
    offsets = np.r_[np.zeros(20), np.full(20, 50.0)]
    hipp = (noise_a + offsets)[None, :].astype(np.float32)
    cortex = (noise_b + offsets)[None, :].astype(np.float32)
    result = ctxreg_mod.regress_local_cortex_signal(
        hipp_ts=hipp,
        cortex_ts=cortex,
        neighbor_indices=[np.asarray([0])],
        run_segments=segments,
    )
    centered = [
        np.r_[noise_a[:20] - noise_a[:20].mean(), noise_a[20:] - noise_a[20:].mean()],
        np.r_[noise_b[:20] - noise_b[:20].mean(), noise_b[20:] - noise_b[20:].mean()],
    ]
    expected = float(np.corrcoef(centered[0], centered[1])[0, 1])
    assert abs(float(result.local_cortex_corr[0]) - expected) < 1e-5
    assert float(result.local_cortex_corr[0]) < 0.9


def test_group_atlas_profile_root_must_match_signal_branch(tmp_path: Path) -> None:
    shared = tmp_path / "shared_inputs" / "pfm_ctxreg"
    surface = tmp_path / "shared_inputs" / "surface"
    check = atlas_mod.validate_profile_surface_root_for_signal_branch
    assert check(shared, signal_branch="ctxreg", shared_root=shared) == shared.resolve()
    assert check(surface, signal_branch="standard", shared_root=tmp_path / "pfm") == surface.resolve()
    assert check(surface, signal_branch=None, shared_root=tmp_path / "lynch") == surface.resolve()
    with pytest.raises(ValueError, match="must read residualized"):
        check(surface, signal_branch="ctxreg", shared_root=shared)
    with pytest.raises(ValueError, match="Signal-branch mismatch"):
        check(shared, signal_branch=None, shared_root=tmp_path / "lynch")


def test_ctxreg_10mm_route_is_distinct_and_radius_bound() -> None:
    from scripts.core_workflow.lib import branch_config, route_tags

    assert route_tags.infer_route_tag_from_shared_root(Path("/x/shared_inputs/pfm_ctxreg_10mm")) == "pfm_ctxreg_10mm"
    assert route_tags.infer_route_tag_from_shared_root(Path("/x/shared_inputs/pfm_ctxreg")) == "pfm_ctxreg"
    assert branch_config.signal_branch_for_route("pfm_ctxreg_10mm") == "ctxreg_10mm"
    assert branch_config.CTXREG_ROUTE_DISTANCE_MM == {"pfm_ctxreg": 20.0, "pfm_ctxreg_10mm": 10.0}
    assert ctxreg_mod.resolve_route_distance_mm("pfm_ctxreg_10mm", None) == 10.0
    with pytest.raises(ValueError, match="defined at 10 mm"):
        ctxreg_mod.resolve_route_distance_mm("pfm_ctxreg_10mm", 20.0)
    root = Path("/x/shared_inputs/pfm_ctxreg_10mm")
    assert atlas_mod.standard_reference_root_for_signal_branch(root, "ctxreg_10mm") == Path("/x/shared_inputs/pfm")
    with pytest.raises(ValueError, match="Signal-branch mismatch"):
        atlas_mod.validate_profile_surface_root_for_signal_branch(root, signal_branch="standard", shared_root=Path("/x/pfm"))
    assert compute_mod.resolve_gradient_reference_root(Path("/x/shared_inputs"), root, "pfm_ctxreg_10mm") == Path("/x/shared_inputs/pfm").resolve()
