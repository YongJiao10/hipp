from __future__ import annotations

import importlib.util
import json
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
MODULE_PATH = REPO_ROOT / "scripts" / "individual_workflow" / "promote_crossfit_template_matching.py"


def load_module():
    spec = importlib.util.spec_from_file_location("promote_crossfit_template_matching", MODULE_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def build_source(root: Path, present: Path) -> None:
    write_json(
        root / "crossfit_manifest.json",
        {
            "status": "ok",
            "n_subjects": 151,
            "n_folds": 5,
            "template_fits_completed": 2,
            "subject_combo_mappings_completed": 3,
            "surface_affinity": "shifted-cosine",
            "out_root": str(root.resolve()),
        },
    )
    for index in range(2):
        write_json(
            root / f"templates/fold-{index + 1:02d}/2mm/hemi_L/template_manifest.json",
            {"status": "ok", "surface_affinity": "shifted-cosine", "root": str(root.resolve())},
        )
    for index in range(3):
        write_json(
            root / f"subjects/sub-{index:06d}/2mm/hemi_L/combo_manifest.json",
            {"status": "ok", "root": str(root.resolve())},
        )
    present.mkdir(parents=True, exist_ok=True)
    (present / "robustness_summary.tsv").write_text(
        "smoothing\themi\tmean_pairwise_macro_dice\n"
        "2mm\tL\t0.95\n2mm\tR\t0.91\n4mm\tL\t0.96\n4mm\tR\t0.92\n",
        encoding="utf-8",
    )
    write_json(
        present / "robustness_manifest.json",
        {
            "status": "ok",
            "crossfit_root": str(root.resolve()),
            "n_folds": 5,
            "robustness_gate": {
                "passed": True,
                "failures": [],
                "thresholds": {
                    "max_affinity_graph_components": 1,
                    "min_cluster_fraction": 0.1,
                    "min_pairwise_ari": 0.7,
                    "min_pairwise_macro_dice": 0.85,
                    "min_pairwise_prototype_correlation": 0.5,
                },
            },
            "combinations": [
                {
                    "smoothing": smoothing,
                    "hemi": hemi,
                    "pairwise_adjusted_rand_index": [[1.0] * 5 for _ in range(5)],
                }
                for smoothing in ("2mm", "4mm")
                for hemi in ("L", "R")
            ],
        },
    )
    (present / "hcp_routea_crossfit_k3_robustness_overview.png").write_bytes(b"png")


def test_promote_crossfit_copies_and_rewrites_only_canonical_copy(tmp_path: Path) -> None:
    module = load_module()
    source = tmp_path / "outputs_hcp/reruns/new/crossfit_template_matching"
    source_present = tmp_path / "outputs_hcp/reruns/new/present/crossfit_template_matching"
    destination = tmp_path / "outputs_hcp/work/crossfit_template_matching"
    destination_present = tmp_path / "outputs_hcp/present/overviews/crossfit_template_matching"
    build_source(source, source_present)

    manifest = module.promote_crossfit(
        repo_root=tmp_path,
        source_root=source,
        source_present=source_present,
        destination_root=destination,
        destination_present=destination_present,
        expected_template_fits=2,
        expected_mappings=3,
    )

    assert manifest["status"] == "ok"
    assert source.exists()
    assert destination.exists()
    copied_text = (destination / "crossfit_manifest.json").read_text(encoding="utf-8")
    assert str(source.resolve()) not in copied_text
    assert str(destination.resolve()) in copied_text
    assert str(source.resolve()) in (source / "crossfit_manifest.json").read_text(encoding="utf-8")
    assert (destination_present / "hcp_routea_crossfit_k3_robustness_overview.png").exists()
    assert (destination / "promotion_manifest.json").exists()


def test_promote_crossfit_rejects_incomplete_source(tmp_path: Path) -> None:
    module = load_module()
    source = tmp_path / "source"
    present = tmp_path / "present"
    build_source(source, present)
    payload = json.loads((source / "crossfit_manifest.json").read_text(encoding="utf-8"))
    payload["subject_combo_mappings_completed"] = 2
    write_json(source / "crossfit_manifest.json", payload)

    try:
        module.promote_crossfit(
            repo_root=tmp_path,
            source_root=source,
            source_present=present,
            destination_root=tmp_path / "dest",
            destination_present=tmp_path / "dest-present",
            expected_template_fits=2,
            expected_mappings=3,
        )
    except ValueError as exc:
        assert "mapping" in str(exc).lower()
    else:
        raise AssertionError("promotion must reject incomplete source")


def test_promote_crossfit_rejects_failed_robustness_gate(tmp_path: Path) -> None:
    module = load_module()
    source, present = tmp_path / "source", tmp_path / "present"
    build_source(source, present)
    payload = json.loads((present / "robustness_manifest.json").read_text(encoding="utf-8"))
    payload["robustness_gate"]["passed"] = False
    payload["robustness_gate"]["failures"] = ["min_pairwise_ari"]
    write_json(present / "robustness_manifest.json", payload)

    try:
        module.validate_source(source, present, expected_template_fits=2, expected_mappings=3)
    except ValueError as exc:
        assert "gate" in str(exc).lower()
    else:
        raise AssertionError("promotion must reject a failed robustness gate")


def test_promote_crossfit_rejects_mismatched_robustness_lineage(tmp_path: Path) -> None:
    module = load_module()
    source, present = tmp_path / "source", tmp_path / "present"
    build_source(source, present)
    payload = json.loads((present / "robustness_manifest.json").read_text(encoding="utf-8"))
    payload["crossfit_root"] = str((tmp_path / "some-other-run").resolve())
    write_json(present / "robustness_manifest.json", payload)

    try:
        module.validate_source(source, present, expected_template_fits=2, expected_mappings=3)
    except ValueError as exc:
        assert "crossfit_root" in str(exc)
    else:
        raise AssertionError("promotion must reject mismatched source lineage")
