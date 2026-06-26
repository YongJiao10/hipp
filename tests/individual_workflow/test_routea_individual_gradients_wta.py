from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
MODULE_PATH = REPO_ROOT / "scripts" / "individual_workflow" / "routea_individual_gradients_wta.py"


def load_module():
    spec = importlib.util.spec_from_file_location("routea_individual_gradients_wta", MODULE_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_default_combo_enumeration_uses_all_subject_smoothing_hemi_pairs():
    workflow = load_module()

    combos = list(
        workflow.iter_subject_combos(
            subjects=["100610", "sub-102311"],
            smoothings=["2mm", "4mm"],
            hemis=["L", "R"],
        )
    )

    assert len(combos) == 8
    assert combos[0].subject == "100610"
    assert combos[0].smoothing == "2mm"
    assert combos[0].hemi == "L"
    assert combos[-1].subject == "102311"
    assert combos[-1].smoothing == "4mm"
    assert combos[-1].hemi == "R"


def test_method_output_dirs_are_direct_and_deterministic(tmp_path):
    workflow = load_module()
    combo = workflow.SubjectCombo(subject="100610", smoothing="4mm", hemi="R")

    gradient_dir = workflow.gradient_combo_output_dir(tmp_path, combo)
    wta_dir = workflow.wta_combo_output_dir(tmp_path, combo)

    assert gradient_dir == tmp_path / "sub-100610" / "gradient" / "4mm" / "R"
    assert wta_dir == tmp_path / "sub-100610" / "wta" / "4mm" / "R"


def test_parser_accepts_explicit_input_and_output_roots(tmp_path):
    workflow = load_module()
    shared_root = tmp_path / "routea"
    surface_root = tmp_path / "surface"
    hippunfold_root = tmp_path / "hippunfold"
    out_root = tmp_path / "individual"

    args = workflow.build_arg_parser().parse_args(
        [
            "--shared-root",
            str(shared_root),
            "--surface-root",
            str(surface_root),
            "--hippunfold-batch-root",
            str(hippunfold_root),
            "--out-root",
            str(out_root),
        ]
    )

    assert Path(args.shared_root) == shared_root
    assert Path(args.surface_root) == surface_root
    assert Path(args.hippunfold_batch_root) == hippunfold_root
    assert Path(args.out_root) == out_root


def test_parser_defaults_use_dataset_agnostic_shared_roots():
    workflow = load_module()

    args = workflow.build_arg_parser().parse_args([])

    assert Path(args.shared_root).name == "routea"
    assert Path(args.surface_root).name == "surface"


def test_subject_wta_assignments_use_single_subject_top_positive_roi_votes():
    workflow = load_module()
    roi_registry = [
        {"roi_id": 1, "roi_name": "roi_a", "canonical8": "A"},
        {"roi_id": 2, "roi_name": "roi_b1", "canonical8": "B"},
        {"roi_id": 3, "roi_name": "roi_b2", "canonical8": "B"},
    ]
    network_names = ["A", "B"]
    network_to_key = {"A": 1, "B": 2}
    fc_full = np.asarray(
        [
            [0.9, 0.1, -0.2],
            [np.nan, np.nan, np.nan],
            [0.2, 0.8, 0.7],
        ],
        dtype=np.float32,
    )
    valid_mask = np.asarray([True, False, True])

    labels, rows, meta = workflow.subject_wta_from_fc(
        fc_full=fc_full,
        valid_mask=valid_mask,
        roi_registry=roi_registry,
        network_names=network_names,
        network_to_key=network_to_key,
        network_column="canonical8",
        top_n_cap=20,
    )

    assert labels.tolist() == [1, 0, 2]
    assert [row["top1_network"] for row in rows] == ["A", "B"]
    assert rows[0]["top1_count"] == 1
    assert rows[1]["top1_count"] == 2
    assert meta["statistical_inference"] is False
    assert meta["assignment_definition"] == "single_subject_top_positive_roi_network_vote"



def test_monitor_summary_counts_ok_failed_running_and_stale(tmp_path):
    workflow = load_module()
    manifest_dir = tmp_path / "sub-100610" / "gradient" / "2mm" / "L"
    manifest_dir.mkdir(parents=True)
    workflow.write_json(
        manifest_dir / "combo_manifest.json",
        {"status": "ok", "subject": "100610", "smoothing": "2mm", "hemi": "L"},
    )
    heartbeat = tmp_path / "logs" / "run_a" / "heartbeat.json"
    workflow.write_json(
        heartbeat,
        {"status": "running", "updated_at_epoch": 1, "pid": 123, "active": 1},
    )

    summary = workflow.build_monitor_summary(
        out_root=tmp_path,
        heartbeat_path=heartbeat,
        stale_seconds=5,
        now_epoch=10,
    )

    assert summary["ok"] == 1
    assert summary["failed"] == 0
    assert summary["running"] == 1
    assert summary["heartbeat_stale"] is True

def test_monitor_summary_does_not_mark_completed_run_stale(tmp_path):
    workflow = load_module()
    manifest_dir = tmp_path / "sub-100610" / "gradient" / "2mm" / "L"
    manifest_dir.mkdir(parents=True)
    workflow.write_json(
        manifest_dir / "combo_manifest.json",
        {"status": "ok", "subject": "100610", "smoothing": "2mm", "hemi": "L"},
    )
    heartbeat = tmp_path / "logs" / "run_a" / "heartbeat.json"
    workflow.write_json(
        heartbeat,
        {"status": "ok", "updated_at_epoch": 1, "pid": None, "active": 0},
    )

    summary = workflow.build_monitor_summary(
        out_root=tmp_path,
        heartbeat_path=heartbeat,
        stale_seconds=5,
        now_epoch=10,
    )

    assert summary["status"] == "ok"
    assert summary["heartbeat_stale"] is False

def test_overview_output_paths_are_per_subject_per_task(tmp_path):
    workflow = load_module()
    overview_root = tmp_path / "present" / "overviews" / "individual"

    gradient_png = workflow.overview_output_path(overview_root, "gradient_spectral_k3", "sub-100610")
    wta_png = workflow.overview_output_path(overview_root, "wta_top", "100610")

    assert gradient_png == overview_root / "gradient" / "sub-100610_gradient_fisherz_overview.png"
    assert wta_png == overview_root / "wta" / "sub-100610_wta_overview.png"


def test_overview_output_path_supports_gradient_count_variant(tmp_path):
    workflow = load_module()
    overview_root = tmp_path / "present" / "overviews" / "individual"

    count_png = workflow.overview_output_path(overview_root, "gradient_count", "sub-04172026")

    assert count_png == overview_root / "gradient" / "sub-04172026_gradient_count_overview.png"


def test_individual_workflow_entrypoints_have_contiguous_numbering():
    workflow_dir = REPO_ROOT / "scripts" / "individual_workflow"

    assert not (workflow_dir / "01_run_routea_individual_gradients_wta.py").exists()
    assert (workflow_dir / "routea_individual_gradients_wta.py").exists()
    assert (workflow_dir / "01_monitor_individual_run.py").exists()
    assert (workflow_dir / "02_build_individual_overviews.py").exists()

    searched_paths = [
        workflow_dir / "run_routea_individual_gradients_wta_background.sh",
        workflow_dir / "README.md",
        REPO_ROOT / "docs" / "hipp_parcellation_network" / "individual_workflow_routea_gradients_wta.md",
        workflow_dir / "routea_individual_gradients_wta.py",
    ]
    combined_text = "\n".join(path.read_text(encoding="utf-8") for path in searched_paths)

    assert "01_run_routea_individual_gradients_wta.py" not in combined_text
    assert "02_monitor_individual_run.py" not in combined_text
    assert "03_build_individual_overviews.py" not in combined_text


def test_materialize_wta_core_inputs(tmp_path):
    import sys
    path = REPO_ROOT / "scripts" / "individual_workflow" / "02_build_individual_overviews.py"
    if str(path.parent) not in sys.path:
        sys.path.insert(0, str(path.parent))
    spec = importlib.util.spec_from_file_location("build_individual_overviews", path)
    assert spec is not None and spec.loader is not None
    build_overviews = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(build_overviews)

    subject = "100610"
    work_root = tmp_path / "work"
    adapter_root = tmp_path / "adapter"
    hippunfold_batch_root = tmp_path / "hippunfold"

    for smoothing in ["2mm", "4mm"]:
        for hemi in ["L", "R"]:
            src = work_root / f"sub-{subject}" / "wta" / smoothing / hemi
            src.mkdir(parents=True, exist_ok=True)
            np.save(src / "wta_top_labels.npy", np.zeros(10, dtype=np.int32))
            build_overviews.write_json(src / "wta_top_label_map.json", {"network_names": ["A", "B"]})
            build_overviews.write_json(src / "vertex_assignment_status.json", {})
            wb_assets = src / "workbench_assets"
            wb_assets.mkdir(parents=True, exist_ok=True)
            (wb_assets / "dummy.label.gii").write_text("dummy", encoding="utf-8")

    for hemi in ["L", "R"]:
        surf_dir = hippunfold_batch_root / f"sub-{subject}" / "hippunfold" / f"sub-{subject}" / "surf"
        surf_dir.mkdir(parents=True, exist_ok=True)
        (surf_dir / f"sub-{subject}_hemi-{hemi}_space-corobl_den-512_label-hipp_midthickness.surf.gii").write_text("surf", encoding="utf-8")
        (surf_dir / f"sub-{subject}_den-512_label-hipp_surfaces.spec").write_text("spec", encoding="utf-8")

    build_overviews.materialize_wta_core_inputs(
        subject=subject,
        work_root=work_root,
        adapter_root=adapter_root,
        hippunfold_batch_root=hippunfold_batch_root,
    )

    for smoothing in ["2mm", "4mm"]:
        for hemi in ["L", "R"]:
            dst = adapter_root / f"sub-{subject}" / "wta-top" / "kong2022" / smoothing / "cov_0p80" / f"hemi_{hemi}"
            manifest_path = dst / "group_atlas_manifest.json"
            assert manifest_path.exists()
            manifest = build_overviews.read_json(manifest_path)
            assert "group_functional_atlas_labels" in manifest
            assert "group_functional_atlas_label_map" in manifest
            assert Path(manifest["group_functional_atlas_labels"]).exists()
            assert Path(manifest["group_functional_atlas_label_map"]).exists()

