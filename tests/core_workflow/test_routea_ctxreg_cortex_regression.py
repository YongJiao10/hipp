#!/usr/bin/env python3
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np


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
neighbors_mod = _load_numbered_module(
    "scripts/core_workflow/06_07_render_ctxdrop_neighbor_counts.py",
    "core_06_07_render_ctxdrop_neighbor_counts_ctxreg_test",
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


def test_ctxdrop_default_cortex_surface_template_matches_bids_input_layout() -> None:
    left = roi_fc_mod.format_subject_template(
        roi_fc_mod.DEFAULT_LEFT_CORTEX_SURFACE_TEMPLATE,
        data_root=Path("/data"),
        subject="100610",
        hemi="L",
    )
    right = roi_fc_mod.format_subject_template(
        roi_fc_mod.DEFAULT_RIGHT_CORTEX_SURFACE_TEMPLATE,
        data_root=Path("/data"),
        subject="100610",
        hemi="R",
    )
    assert left.as_posix() == "/data/sub-100610/anat/sub-100610_hemi-L_space-fsLR_den-32k_desc-MSMAll_midthickness.surf.gii"
    assert right.as_posix() == "/data/sub-100610/anat/sub-100610_hemi-R_space-fsLR_den-32k_desc-MSMAll_midthickness.surf.gii"


def test_local_cortex_neighbor_renderer_uses_shallow_neutral_present_path() -> None:
    assert neighbors_mod.DEFAULT_OUT_DIR == REPO_ROOT / "outputs_core" / "present" / "local_cortex_neighbors"
    assert not any(part in {"gradients", "diagnostics", "ctxdrop", "ctxreg"} for part in neighbors_mod.DEFAULT_OUT_DIR.parts)

    ctxdrop_path = neighbors_mod.count_path(
        shared_root=Path("/tmp/shared/routea_ctxdrop_151"),
        subject="100610",
        hemi="L",
        source_branch="routea_ctxdrop",
    )
    ctxreg_path = neighbors_mod.count_path(
        shared_root=Path("/tmp/shared/routea_ctxreg_151"),
        subject="100610",
        hemi="L",
        source_branch="routea_ctxreg",
    )
    assert ctxdrop_path.as_posix() == "/tmp/shared/routea_ctxdrop_151/sub-100610/surface/local_cortex_neighbors/hemi_L/neighbor_counts.npy"
    assert ctxreg_path.as_posix() == "/tmp/shared/routea_ctxreg_151/sub-100610/surface/local_cortex_neighbors/hemi_L/neighbor_counts.npy"


def test_local_cortex_neighbor_manifest_omits_smoothing_semantics() -> None:
    payload = neighbors_mod.manifest_payload(
        summaries=[],
        subjects=[],
        subjects_manifest=Path("/tmp/subjects.txt"),
        source_branch="routea_ctxdrop",
        distance_mm=20.0,
        display_vmin=1.0,
        display_vmax=2.0,
        hippomaps_root=Path("/tmp/hippomaps"),
        outputs={},
    )
    assert "smoothing" not in payload
    assert "source_smoothing" not in payload
    assert "cache_smoothing" not in payload

    parser = neighbors_mod.parse_args_for_test([])
    assert not hasattr(parser, "smoothing")


def test_local_cortex_neighbor_output_names_are_smoothing_independent() -> None:
    paths = neighbors_mod.output_paths(Path("/tmp/present/local_cortex_neighbors"), distance_mm=20.0)
    assert paths["png"].name == "kong2022_routea_local_cortex_neighbors_mean20mm.png"
    assert paths["manifest"].name == "kong2022_routea_local_cortex_neighbors_mean20mm_manifest.json"
    assert paths["tsv"].name == "group_mean_local_cortex_neighbors.tsv"
    assert paths["mean_array"].name == "group_mean_local_cortex_neighbors_hemi-{hemi}.npy"
    assert "2mm" not in paths["png"].name
    assert "4mm" not in paths["png"].name


def test_local_cortex_neighbor_renderer_outputs_hippocampal_overlay(tmp_path: Path) -> None:
    from PIL import Image

    calls = []

    def fake_plotter(cdata, **kwargs):
        calls.append((np.asarray(cdata), kwargs))
        Image.new("RGB", (90, 50), (255, 255, 255)).save(kwargs["filename"])
        return None

    summaries = [
        neighbors_mod.HemiSummary(
            hemi="L",
            count_paths=[],
            mean=np.asarray([1.0, 2.0, 3.0], dtype=np.float32),
            sd=np.zeros(3, dtype=np.float32),
            n_subjects=np.asarray([2, 2, 2], dtype=np.int32),
        ),
        neighbors_mod.HemiSummary(
            hemi="R",
            count_paths=[],
            mean=np.asarray([4.0, 5.0, 6.0], dtype=np.float32),
            sd=np.zeros(3, dtype=np.float32),
            n_subjects=np.asarray([2, 2, 2], dtype=np.int32),
        ),
    ]
    out = tmp_path / "overlay.png"

    neighbors_mod.render_overlay_png(out, summaries, title="RouteA group mean cortex vertices within 20 mm", plotter=fake_plotter)

    assert out.exists()
    assert Image.open(out).size[0] > 180
    assert len(calls) == 2
    assert calls[0][1]["hemis"] == ["L"]
    assert calls[1][1]["hemis"] == ["R"]
    assert calls[0][1]["cmap"] == "viridis"
    assert calls[0][1]["color_range"] == (1.0, 6.0)
    assert calls[0][1]["cb__labelFormat"] == "%.0f"


def test_ctxdrop_vertex_to_roi_fc_excludes_only_each_hipp_vertex_local_cortex() -> None:
    hipp = np.asarray(
        [
            [0.0, 1.0, 0.0, 1.0],
            [0.0, 0.0, 1.0, 1.0],
            [1.0, 2.0, 1.0, 2.0],
        ],
        dtype=np.float32,
    )
    cortex = np.asarray(
        [
            [0.0, 0.0, 1.0, 1.0],
            [0.0, 1.0, 0.0, 1.0],
            [1.0, 2.0, 1.0, 2.0],
        ],
        dtype=np.float32,
    )
    result = roi_fc_mod.compute_ctxdrop_vertex_to_roi_fc(
        hipp_ts=hipp,
        cortex_ts=cortex,
        cortex_roi_ids=np.asarray([1, 1, 2], dtype=np.int32),
        roi_ids=[1, 2],
        cortex_valid_mask=np.asarray([True, True, True], dtype=bool),
        neighbor_indices=[
            np.asarray([0], dtype=np.int64),
            np.asarray([1], dtype=np.int64),
            np.asarray([2], dtype=np.int64),
        ],
    )
    assert result.excluded_neighbor_counts.tolist() == [1, 1, 1]
    assert result.roi_cortex_counts.tolist() == [[1, 1], [1, 1], [2, 0]]
    assert np.allclose(result.fc[0, 0], 1.0, atol=1e-6)
    assert np.allclose(result.fc[1, 0], 1.0, atol=1e-6)
    assert np.isnan(result.fc[2, 1])


def test_routea_ctx_control_route_contracts_and_paths() -> None:
    assert compute_mod.ROUTE_TO_SHARED_DIR["routea_ctxreg"] == "routea_ctxreg_151"
    assert compute_mod.ROUTE_TO_SHARED_DIR["routea_ctxdrop"] == "routea_ctxdrop_151"
    assert atlas_mod.infer_route_tag(Path("/tmp/kong2022/shared_inputs/routea_ctxreg_151"), "auto") == "routea_ctxreg"
    assert atlas_mod.infer_route_tag(Path("/tmp/kong2022/shared_inputs/routea_ctxdrop_151"), "auto") == "routea_ctxdrop"
    assert "routea_ctxreg" in overview_mod.ROUTE_TAG_CHOICES
    assert "routea_ctxdrop" in overview_mod.ROUTE_TAG_CHOICES
    assert "routea_ctxreg" in grad_render_mod.ROUTES
    assert "routea_ctxdrop" in grad_render_mod.ROUTES
    assert "routea_ctxreg" in matrix_mod.ROUTE_CHOICES
    assert "routea_ctxdrop" in matrix_mod.ROUTE_CHOICES

    overview_path = matrix_mod.expected_overview_path(
        Path("/tmp/present"),
        method="network-spectral",
        route_tag="routea_ctxreg",
        mode_tag=None,
        assignment_variant="fisherz",
    )
    assert overview_path.name == "kong2022_network-spectral_group_overview_routea_ctxreg_fisherz.png"

    ctxdrop_overview_path = matrix_mod.expected_overview_path(
        Path("/tmp/present"),
        method="gradient-spectral",
        route_tag="routea_ctxdrop",
        mode_tag=None,
        assignment_variant="fisherz",
    )
    assert ctxdrop_overview_path.name == "kong2022_gradient-spectral_group_overview_g1g2g3_routea_ctxdrop_fisherz.png"




def test_routea_17net_reuses_routea_inputs_and_has_shallow_output_contract() -> None:
    assert "routea_17net" not in compute_mod.ROUTE_TO_SHARED_DIR
    assert atlas_mod.gradient_route_for_route_tag("routea_17net") == "routea"
    assert atlas_mod.infer_route_tag(Path("/tmp/kong2022/shared_inputs/routea_151"), "routea_17net") == "routea_17net"
    assert "routea_17net" in atlas_mod.ROUTE_TAG_CHOICES
    assert "routea_17net" in overview_mod.ROUTE_TAG_CHOICES
    assert "routea_17net" in matrix_mod.ROUTE_CHOICES

    route_roots = matrix_mod.canonical_group_roots(REPO_ROOT / "outputs_core" / "work" / "group_atlas")
    assert route_roots["routea_17net"] == (
        REPO_ROOT / "outputs_core" / "work" / "group_atlas" / "routea_17net_full_k3_fisherz"
    ).resolve()

    jobs = matrix_mod.build_render_jobs(
        REPO_ROOT / "outputs_core" / "work" / "group_atlas",
        routes=("routea_17net",),
        methods=("network-spectral",),
    )
    assert {job.route_tag for job in jobs} == {"routea_17net"}
    assert {job.assignment_variant for job in jobs} == {"fisherz", "count"}
    assert all("routea_17net_full_k3_fisherz" in str(job.group_root) for job in jobs)

    overview_path = matrix_mod.expected_overview_path(
        Path("/tmp/present"),
        method="network-spectral",
        route_tag="routea_17net",
        mode_tag=None,
        assignment_variant="fisherz",
    )
    assert overview_path.name == "kong2022_network-spectral_group_overview_routea_17net_fisherz.png"


def test_ctxdrop_materializer_writes_standard_fc_and_diagnostics(tmp_path: Path) -> None:
    subject = "101"
    fc_dir = tmp_path / "routea_ctxdrop_151" / f"sub-{subject}" / "fc" / "kong2022" / "2mm" / "hemi_L"
    diag_dir = tmp_path / "routea_ctxdrop_151" / f"sub-{subject}" / "surface" / "local_cortex_neighbors" / "hemi_L"
    hipp = np.asarray(
        [
            [0.0, 1.0, 0.0, 1.0],
            [9.0, 9.0, 9.0, 9.0],
            [1.0, 2.0, 1.0, 2.0],
        ],
        dtype=np.float32,
    )
    cortex = np.asarray(
        [
            [0.0, 0.0, 1.0, 1.0],
            [0.0, 1.0, 0.0, 1.0],
            [1.0, 2.0, 1.0, 2.0],
        ],
        dtype=np.float32,
    )
    roi_rows = [
        {"parcel_id": 1, "parcel_name": "roi1"},
        {"parcel_id": 2, "parcel_name": "roi2"},
    ]
    run_segments = [
        {"run_id": 1, "run_start": 0, "run_stop": 1, "run_length": 1},
        {"run_id": 2, "run_start": 1, "run_stop": 2, "run_length": 1},
        {"run_id": 3, "run_start": 2, "run_stop": 3, "run_length": 1},
        {"run_id": 4, "run_start": 3, "run_stop": 4, "run_length": 1},
    ]

    roi_fc_mod.materialize_ctxdrop_fc_from_arrays(
        fc_dir=fc_dir,
        subject=subject,
        smoothing="2mm",
        hemi="L",
        hipp_ts=hipp,
        valid_mask=np.asarray([True, False, True], dtype=bool),
        cortex_ts=cortex,
        cortex_roi_ids=np.asarray([1, 1, 2], dtype=np.int32),
        roi_rows=roi_rows,
        neighbor_indices=[
            np.asarray([0], dtype=np.int64),
            np.asarray([1], dtype=np.int64),
            np.asarray([2], dtype=np.int64),
        ],
        run_segments=run_segments,
        distance_mm=20.0,
        local_neighbor_dir=diag_dir,
    )

    full = np.load(fc_dir / "hipp_vertex_to_network_fc.npy")
    active = np.load(fc_dir / "hipp_vertex_to_network_fc_active.npy")
    assert full.shape == (3, 2)
    assert active.shape == (2, 2)
    assert np.isnan(full[1]).all()
    assert not (fc_dir / "ctxdrop_neighbor_counts.npy").exists()
    assert not (fc_dir / "ctxdrop_roi_cortex_counts.npy").exists()
    assert np.load(diag_dir / "neighbor_counts.npy").tolist() == [1, 1, 1]
    assert np.load(diag_dir / "ctxdrop_roi_cortex_counts.npy").tolist() == [[1, 1], [1, 1], [2, 0]]
    summary = (fc_dir / "fc_summary.json").read_text(encoding="utf-8")
    assert "pearson_vertex_to_kong2022_roi_timeseries_ctxdrop_local20mm" in summary
    assert (fc_dir / "runs" / "run-1" / "hipp_vertex_to_network_fc_active.npy").exists()


def test_routea_ctxreg_fc_steps_use_separate_reference_and_mask_roots(tmp_path: Path) -> None:
    subject = "101"
    out_root = tmp_path / "routea_ctxreg_151"
    controlled_surface = tmp_path / "controlled_surface"
    routea_reference = tmp_path / "routea_151"
    source_masks = tmp_path / "surface_151"

    ts_dir = controlled_surface / f"sub-{subject}" / "surface" / "2mm"
    ts_dir.mkdir(parents=True)
    np.save(ts_dir / f"sub-{subject}_hemi-L_timeseries.npy", np.asarray([[1, 2, 3, 4], [2, 3, 4, 5]], dtype=np.float32))

    mask_dir = source_masks / f"sub-{subject}" / "surface" / "tsnr_gate"
    mask_dir.mkdir(parents=True)
    np.save(mask_dir / f"sub-{subject}_hemi-L_valid_mask.npy", np.asarray([True, False], dtype=bool))

    ref_dir = routea_reference / f"sub-{subject}" / "reference" / "kong2022"
    ref_dir.mkdir(parents=True)
    np.save(ref_dir / "cortex_roi_parcel_timeseries.npy", np.asarray([[1, 1, 2, 2], [4, 3, 2, 1]], dtype=np.float32))
    (ref_dir / "cortex_roi_parcels.tsv").write_text("parcel_name\nA\nB\n", encoding="utf-8")
    for rid, start in enumerate([0, 1, 2, 3], start=1):
        run_dir = routea_reference / f"sub-{subject}" / "reference" / "common" / "runs" / f"run-{rid}"
        run_dir.mkdir(parents=True)
        (run_dir / "reference_summary.json").write_text(
            f'{{"run_id": {rid}, "run_start": {start}, "run_stop": {start + 1}, "run_length": 1}}',
            encoding="utf-8",
        )

    roi_fc_mod.materialize_fc(
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
    assert not (out_root / f"sub-{subject}" / "surface" / "tsnr_gate").exists()

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
    routea_ref = Path("/tmp/routea_151")
    ctxreg = Path("/tmp/routea_ctxreg_151")
    assert atlas_mod.reference_path(routea_ref, "101", "kong2022", "cortex_roi_parcels.tsv").as_posix().endswith(
        "/routea_151/sub-101/reference/kong2022/cortex_roi_parcels.tsv"
    )
    assert atlas_mod.reference_path(ctxreg, "101", "kong2022", "cortex_roi_parcels.tsv").as_posix().endswith(
        "/routea_ctxreg_151/sub-101/reference/kong2022/cortex_roi_parcels.tsv"
    )
    assert cortex_mod.subject_reference_path(routea_ref, "101", "kong2022", "cortex_roi_parcels.tsv").as_posix().endswith(
        "/routea_151/sub-101/reference/kong2022/cortex_roi_parcels.tsv"
    )
