from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
CONTRACT_PATH = (
    REPO_ROOT
    / "scripts"
    / "individual_workflow"
    / "lib"
    / "fixed_template_overview_contract.py"
)


def load_contract():
    assert CONTRACT_PATH.is_file(), "fixed-template overview contract module is missing"
    spec = importlib.util.spec_from_file_location(
        "fixed_template_overview_contract_under_test",
        CONTRACT_PATH,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def fingerprint(path: Path) -> dict[str, object]:
    return {
        "path": str(path.resolve()),
        "sha256": sha256(path),
        "size_bytes": path.stat().st_size,
    }


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_render_inputs(tmp_path: Path) -> dict[str, Path]:
    matching_root = tmp_path / "matching"
    template = tmp_path / "template_manifest.json"
    template.write_text("{}\n", encoding="utf-8")
    rosters = []
    for name, subject in (("nthc.txt", "1001"), ("tehc.txt", "2002")):
        roster = tmp_path / name
        roster.write_text(f"{subject}\n", encoding="utf-8")
        rosters.append(roster)
    run_manifest = matching_root / "runs" / "full" / "run_manifest.json"
    write_json(
        run_manifest,
        {
            "status": "complete",
            "scope": "full",
            "workflow": "stanford_hcp_fixed_template_matching",
            "canonical_roster_count": 2,
            "subject_count": 2,
            "complete_count": 2,
            "failed_count": 0,
            "subjects": ["sub-1001", "sub-2002"],
            "path_roots": {"out_root": str(matching_root.resolve())},
            "template_manifest": fingerprint(template),
        },
    )
    scene = tmp_path / "locked.scene"
    scene.write_text("scene-v1\n", encoding="utf-8")
    return {
        "matching_root": matching_root,
        "run_manifest": run_manifest,
        "roster_1": rosters[0],
        "roster_2": rosters[1],
        "scene": scene,
        "template": template,
    }


def write_subject_sources(inputs: dict[str, Path]) -> None:
    matching_root = inputs["matching_root"]
    template_record = fingerprint(inputs["template"])
    label_semantics = {
        "1": "Anterior/VAN",
        "2": "Middle/SMN",
        "3": "Posterior/DMN",
    }
    for subject in ("1001", "2002"):
        subject_dir = matching_root / "subjects" / f"sub-{subject}"
        surface_records: dict[str, dict[str, object]] = {}
        combinations: list[dict[str, object]] = []
        outputs: dict[str, dict[str, object]] = {}
        for hemi in ("L", "R"):
            surface = subject_dir / "surf" / (
                f"sub-{subject}_hemi-{hemi}_space-T1w_den-512_"
                "label-hipp_midthickness.surf.gii"
            )
            spec = surface.parent / f"sub-{subject}_den-512_label-hipp_surfaces.spec"
            surface.parent.mkdir(parents=True, exist_ok=True)
            surface.write_text(f"surface-{subject}-{hemi}\n", encoding="utf-8")
            spec.write_text(f"spec-{subject}\n", encoding="utf-8")
            surface_records[hemi] = fingerprint(surface)
            surface_records[hemi]["spec_path"] = str(spec.resolve())
            surface_records[hemi]["spec_fingerprint"] = fingerprint(spec)

        for smoothing in ("2mm", "4mm"):
            for hemi in ("L", "R"):
                combo_dir = subject_dir / "matching" / smoothing / f"hemi-{hemi}"
                label = combo_dir / (
                    f"sub-{subject}_hemi-{hemi}_space-corobl_den-512_"
                    f"label-hcp_fixed_template_k3_{smoothing}.label.gii"
                )
                combo_dir.mkdir(parents=True, exist_ok=True)
                label.write_text(
                    f"labels-{subject}-{smoothing}-{hemi}\n", encoding="utf-8"
                )
                combo = combo_dir / "combo_manifest.json"
                density = {
                    "consumed_asset": surface_records[hemi]["path"],
                    "n_vertices": 512,
                    "surface_fingerprint": {
                        key: value
                        for key, value in surface_records[hemi].items()
                        if key in {"path", "sha256", "size_bytes"}
                    },
                    "spec_path": surface_records[hemi]["spec_path"],
                    "spec_fingerprint": surface_records[hemi]["spec_fingerprint"],
                }
                label_record = fingerprint(label)
                write_json(
                    combo,
                    {
                        "status": "ok",
                        "workflow": "stanford_hcp_fixed_template_matching",
                        "subject": f"sub-{subject}",
                        "smoothing": smoothing,
                        "hemi": hemi,
                        "label_semantics": label_semantics,
                        "outputs": {"label_gifti": label_record},
                        "density_evidence": density,
                    },
                )
                combo_record = fingerprint(combo)
                prefix = f"{smoothing}_hemi-{hemi}"
                outputs[f"{prefix}_combo_manifest"] = combo_record
                outputs[f"{prefix}_label_gifti"] = label_record
                combinations.append(
                    {
                        "status": "ok",
                        "smoothing": smoothing,
                        "hemi": hemi,
                        "manifest": str(combo.resolve()),
                        "outputs": {
                            "combo_manifest": combo_record,
                            "label_gifti": label_record,
                        },
                        "density_evidence": density,
                    }
                )

        write_json(
            subject_dir / "subject_manifest.json",
            {
                "status": "ok",
                "workflow": "stanford_hcp_fixed_template_matching",
                "subject": f"sub-{subject}",
                "template_manifest": template_record,
                "inputs": {
                    f"surface_hemi-{hemi}": {
                        key: value
                        for key, value in surface_records[hemi].items()
                        if key in {"path", "sha256", "size_bytes"}
                    }
                    for hemi in ("L", "R")
                },
                "outputs": outputs,
                "combinations": combinations,
            },
        )


def test_preflight_locks_explicit_roster_to_matching_run(tmp_path: Path) -> None:
    contract = load_contract()
    inputs = write_render_inputs(tmp_path)

    preflight = contract.preflight_render_inputs(
        matching_root=inputs["matching_root"],
        run_manifest=inputs["run_manifest"],
        roster_paths=(inputs["roster_1"], inputs["roster_2"]),
        scene=inputs["scene"],
        expected_matching_workflow="stanford_hcp_fixed_template_matching",
        expected_subject_count=2,
    )

    assert preflight.subjects == ("1001", "2002")
    assert preflight.matching_run_manifest["path"] == str(inputs["run_manifest"].resolve())
    assert preflight.scene_fingerprint["sha256"] == sha256(inputs["scene"])


def test_preflight_rejects_roster_closure_drift(tmp_path: Path) -> None:
    contract = load_contract()
    inputs = write_render_inputs(tmp_path)
    inputs["roster_2"].write_text("2999\n", encoding="utf-8")

    with pytest.raises(ValueError, match="roster closure"):
        contract.preflight_render_inputs(
            matching_root=inputs["matching_root"],
            run_manifest=inputs["run_manifest"],
            roster_paths=(inputs["roster_1"], inputs["roster_2"]),
            scene=inputs["scene"],
            expected_matching_workflow="stanford_hcp_fixed_template_matching",
            expected_subject_count=2,
        )


def test_existing_receipt_rejects_changed_scene_before_reuse(tmp_path: Path) -> None:
    contract = load_contract()
    inputs = write_render_inputs(tmp_path)
    preflight = contract.preflight_render_inputs(
        matching_root=inputs["matching_root"],
        run_manifest=inputs["run_manifest"],
        roster_paths=(inputs["roster_1"], inputs["roster_2"]),
        scene=inputs["scene"],
        expected_matching_workflow="stanford_hcp_fixed_template_matching",
        expected_subject_count=2,
    )
    receipt = tmp_path / "fixed_template_overview_manifest.json"
    write_json(
        receipt,
        {
            "status": "complete",
            "workflow": "stanford_hcp_fixed_template_matching_overviews",
            "output_dir": str((tmp_path / "present").resolve()),
            "subjects": ["sub-1001", "sub-2002"],
            "subject_count": 2,
            "matching_run_manifest": dict(preflight.matching_run_manifest),
            "scene": str(inputs["scene"].resolve()),
            "scene_fingerprint": dict(preflight.scene_fingerprint),
        },
    )
    inputs["scene"].write_text("scene-v2\n", encoding="utf-8")

    with pytest.raises(ValueError, match="scene drift"):
        contract.validate_existing_receipt(
            receipt_path=receipt,
            preflight=preflight,
            expected_overview_workflow="stanford_hcp_fixed_template_matching_overviews",
            output_dir=tmp_path / "present",
        )


def test_subject_preflight_rejects_label_drift_before_rendering(tmp_path: Path) -> None:
    contract = load_contract()
    inputs = write_render_inputs(tmp_path)
    write_subject_sources(inputs)
    preflight = contract.preflight_render_inputs(
        matching_root=inputs["matching_root"],
        run_manifest=inputs["run_manifest"],
        roster_paths=(inputs["roster_1"], inputs["roster_2"]),
        scene=inputs["scene"],
        expected_matching_workflow="stanford_hcp_fixed_template_matching",
        expected_subject_count=2,
    )
    label = (
        inputs["matching_root"]
        / "subjects"
        / "sub-2002"
        / "matching"
        / "4mm"
        / "hemi-R"
        / "sub-2002_hemi-R_space-corobl_den-512_"
        "label-hcp_fixed_template_k3_4mm.label.gii"
    )
    label.write_text("drifted-label\n", encoding="utf-8")

    with pytest.raises(ValueError, match="label GIFTI fingerprint drift"):
        contract.preflight_subject_sources(
            preflight=preflight,
            expected_matching_workflow="stanford_hcp_fixed_template_matching",
            expected_label_semantics={
                "1": "Anterior/VAN",
                "2": "Middle/SMN",
                "3": "Posterior/DMN",
            },
            smoothings=("2mm", "4mm"),
            hemis=("L", "R"),
            expected_n_vertices=512,
        )


def test_subject_preflight_returns_ordered_exact_sources(tmp_path: Path) -> None:
    contract = load_contract()
    inputs = write_render_inputs(tmp_path)
    write_subject_sources(inputs)
    preflight = contract.preflight_render_inputs(
        matching_root=inputs["matching_root"],
        run_manifest=inputs["run_manifest"],
        roster_paths=(inputs["roster_1"], inputs["roster_2"]),
        scene=inputs["scene"],
        expected_matching_workflow="stanford_hcp_fixed_template_matching",
        expected_subject_count=2,
    )

    sources = contract.preflight_subject_sources(
        preflight=preflight,
        expected_matching_workflow="stanford_hcp_fixed_template_matching",
        expected_label_semantics={
            "1": "Anterior/VAN",
            "2": "Middle/SMN",
            "3": "Posterior/DMN",
        },
        smoothings=("2mm", "4mm"),
        hemis=("L", "R"),
        expected_n_vertices=512,
    )

    assert [source["subject"] for source in sources] == ["sub-1001", "sub-2002"]
    assert sources[0]["source_rows"][0]["smoothing"] == "2mm"
    assert sources[0]["source_rows"][0]["L"]["label_gifti"] == str(
        (
            inputs["matching_root"]
            / "subjects"
            / "sub-1001"
            / "matching"
            / "2mm"
            / "hemi-L"
            / "sub-1001_hemi-L_space-corobl_den-512_"
            "label-hcp_fixed_template_k3_2mm.label.gii"
        ).resolve()
    )
    assert not (tmp_path / "present").exists()

