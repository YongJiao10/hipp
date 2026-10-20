from __future__ import annotations

import importlib.util
import json
import os
import sys
from pathlib import Path

import nibabel as nib
import numpy as np
import pytest
from PIL import Image
from nibabel.gifti import GiftiDataArray, GiftiImage


REPO_ROOT = Path(__file__).resolve().parents[2]
MODULE_PATH = (
    REPO_ROOT
    / "scripts"
    / "jj_tms"
    / "render_jj_tms_hcp151_fixed_template_overviews.py"
)


def load_module():
    spec = importlib.util.spec_from_file_location(
        "render_jj_tms_hcp151_fixed_template_overviews",
        MODULE_PATH,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _fingerprint(path: Path) -> dict[str, object]:
    import hashlib

    data = path.read_bytes()
    return {"sha256": hashlib.sha256(data).hexdigest(), "size_bytes": len(data)}


def _write_fixed_subject(root: Path, subject: str = "1001") -> None:
    subject_dir = root / "subjects" / f"sub-{subject}"
    surface_dir = root / "source_surfaces" / subject
    surface_dir.mkdir(parents=True)
    for hemi in ("L", "R"):
        surface = surface_dir / (
            f"sub-{subject}_hemi-{hemi}_space-T1w_den-512_"
            "label-hipp_midthickness.surf.gii"
        )
        nib.save(
            GiftiImage(
                darrays=[
                    GiftiDataArray(
                        data=np.zeros((512, 3), dtype=np.float32),
                        intent="NIFTI_INTENT_POINTSET",
                    )
                ]
            ),
            str(surface),
        )
        (surface_dir / f"sub-{subject}_den-512_label-hipp_surfaces.spec").write_text(
            "spec\n", encoding="utf-8"
        )

    subject_inputs = {}
    subject_outputs = {}
    combinations = []
    for smoothing in ("2mm", "4mm"):
        for hemi in ("L", "R"):
            combo_dir = subject_dir / "matching" / smoothing / f"hemi-{hemi}"
            combo_dir.mkdir(parents=True)
            labels_path = combo_dir / (
                f"sub-{subject}_hemi-{hemi}_space-corobl_den-512_"
                f"label-hcp151_fixed_template_k3_{smoothing}.label.gii"
            )
            labels = (np.arange(512, dtype=np.int32) % 3) + 1
            nib.save(
                GiftiImage(
                    darrays=[
                        GiftiDataArray(data=labels, intent="NIFTI_INTENT_LABEL")
                    ]
                ),
                str(labels_path),
            )
            surface = surface_dir / (
                f"sub-{subject}_hemi-{hemi}_space-T1w_den-512_"
                "label-hipp_midthickness.surf.gii"
            )
            subject_inputs[f"surface_hemi-{hemi}"] = {
                "path": str(surface.resolve()),
                **_fingerprint(surface),
            }
            combo_manifest = combo_dir / "combo_manifest.json"
            combo_manifest.write_text(
                json.dumps(
                    {
                        "status": "ok",
                        "subject": f"sub-{subject}",
                        "smoothing": smoothing,
                        "hemi": hemi,
                        "label_semantics": {
                            "1": "Anterior/VAN",
                            "2": "Body/SMN",
                            "3": "Posterior/DMN",
                        },
                        "outputs": {
                            "label_gifti": {
                                "path": str(labels_path.resolve()),
                                **_fingerprint(labels_path),
                                "shape": [512],
                                "dtype": "int32",
                            }
                        },
                        "density_evidence": {
                            "consumed_asset": str(surface.resolve()),
                            "n_vertices": 512,
                        },
                    }
                ),
                encoding="utf-8",
            )
            combo_payload = json.loads(combo_manifest.read_text(encoding="utf-8"))
            subject_outputs[f"{smoothing}_hemi-{hemi}_combo_manifest"] = {
                "path": str(combo_manifest.resolve()),
                **_fingerprint(combo_manifest),
            }
            subject_outputs[f"{smoothing}_hemi-{hemi}_label_gifti"] = (
                combo_payload["outputs"]["label_gifti"]
            )
            combinations.append(
                {
                    "status": "ok",
                    "subject": f"sub-{subject}",
                    "smoothing": smoothing,
                    "hemi": hemi,
                    "manifest": str(combo_manifest.resolve()),
                }
            )
    (subject_dir / "subject_manifest.json").parent.mkdir(parents=True, exist_ok=True)
    (subject_dir / "subject_manifest.json").write_text(
        json.dumps(
            {
                "status": "ok",
                "subject": f"sub-{subject}",
                "inputs": subject_inputs,
                "outputs": subject_outputs,
                "combinations": combinations,
            }
        ),
        encoding="utf-8",
    )


def test_fixed_overview_renderer_writes_two_rows_and_provenance(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    workflow = load_module()
    matching_root = tmp_path / "matching"
    _write_fixed_subject(matching_root)
    scene = tmp_path / "scene.scene"
    scene.write_text("scene-v1\n", encoding="utf-8")

    def fake_render_locked_grid(**kwargs):
        output = Path(kwargs["outdir"]) / "fake.png"
        output.parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (120, 60), (20, 30, 40)).save(output, format="PNG")
        return {"biglegend_png": str(output)}

    monkeypatch.setattr(workflow, "render_locked_grid", fake_render_locked_grid)
    outputs = workflow.render_fixed_template_overviews(
        matching_root=matching_root,
        output_dir=tmp_path / "present",
        scene=scene,
        subjects=["1001"],
        jobs=1,
        allow_subset=True,
    )

    pngs = list((tmp_path / "present").glob("*.png"))
    assert len(pngs) == 1
    payload = json.loads(outputs["manifest"].read_text(encoding="utf-8"))
    assert payload["summary"]["status"] == "subset"
    assert payload["summary"]["scope"] == "subset"
    assert payload["summary"]["ok"] == 1
    assert payload["summary"]["total"] == 1
    assert payload["items"][0]["source_rows"][0]["smoothing"] == "2mm"
    assert payload["items"][0]["source_rows"][1]["smoothing"] == "4mm"
    assert outputs["status_tsv"].is_file()


def test_fixed_overview_resume_is_immutable_and_detects_input_drift(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    workflow = load_module()
    matching_root = tmp_path / "matching"
    _write_fixed_subject(matching_root)
    scene = tmp_path / "scene.scene"
    scene.write_text("scene-v1\n", encoding="utf-8")
    calls = {"count": 0}

    def fake_render_locked_grid(**kwargs):
        calls["count"] += 1
        output = Path(kwargs["outdir"]) / "fake.png"
        output.parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (120, 60), (20, 30, 40)).save(output, format="PNG")
        return {"biglegend_png": str(output)}

    monkeypatch.setattr(workflow, "render_locked_grid", fake_render_locked_grid)
    output_dir = tmp_path / "present"
    outputs = workflow.render_fixed_template_overviews(
        matching_root=matching_root,
        output_dir=output_dir,
        scene=scene,
        subjects=["1001"],
        jobs=1,
        allow_subset=True,
    )
    manifest_bytes = outputs["manifest"].read_bytes()
    png = next(output_dir.glob("*.png"))
    png_bytes = png.read_bytes()
    monkeypatch.setattr(
        workflow,
        "render_locked_grid",
        lambda **kwargs: (_ for _ in ()).throw(AssertionError("rendered on resume")),
    )

    workflow.render_fixed_template_overviews(
        matching_root=matching_root,
        output_dir=output_dir,
        scene=scene,
        subjects=["1001"],
        jobs=1,
        resume=True,
        allow_subset=True,
    )
    assert calls["count"] == 2
    assert outputs["manifest"].read_bytes() == manifest_bytes
    assert png.read_bytes() == png_bytes

    label = next(
        (matching_root / "subjects" / "sub-1001" / "matching").glob(
            "2mm/hemi-L/*.label.gii"
        )
    )
    label.write_bytes(label.read_bytes() + b"drift")
    with pytest.raises(ValueError, match="fingerprint"):
        workflow.render_fixed_template_overviews(
            matching_root=matching_root,
            output_dir=output_dir,
            scene=scene,
            subjects=["1001"],
            jobs=1,
            resume=True,
            allow_subset=True,
        )


def test_fixed_overview_rejects_duplicate_subject_request(tmp_path: Path):
    workflow = load_module()
    with pytest.raises(ValueError, match="duplicate"):
        workflow.normalize_subjects(["1001", "sub-1001"])



def test_fixed_overview_rejects_orphan_png_without_manifest(tmp_path: Path):
    workflow = load_module()
    matching_root = tmp_path / "matching"
    _write_fixed_subject(matching_root)
    scene = tmp_path / "scene.scene"
    scene.write_text("scene-v1\\n", encoding="utf-8")
    output_dir = tmp_path / "present"
    output_dir.mkdir()
    Image.new("RGB", (20, 20), (1, 2, 3)).save(
        output_dir / "sub-1001_hcp151_fixed_template_matching_overview.png",
        format="PNG",
    )

    with pytest.raises(ValueError, match="provenance"):
        workflow.render_fixed_template_overviews(
            matching_root=matching_root,
            output_dir=output_dir,
            scene=scene,
            subjects=["1001"],
            jobs=1,
            resume=True,
            allow_subset=True,
        )


def test_fixed_overview_rejects_validation_label_sha_drift(tmp_path: Path):
    workflow = load_module()
    matching_root = tmp_path / "matching"
    _write_fixed_subject(matching_root)
    run_manifest = tmp_path / "run.json"
    run_manifest.write_text(
        json.dumps({"status": "complete", "subjects": ["sub-1001"]}),
        encoding="utf-8",
    )
    table = tmp_path / "validation_items.tsv"
    header = (
        "subject\tsmoothing\themi\tstatus\tlabel_1_count\tlabel_2_count\t"
        "label_3_count\tmargin_min\tmargin_median\tmargin_mean\tmargin_max\t"
        "labels_sha256\tconsumed_surface\n"
    )
    rows = "".join(
        f"sub-1001\t{smoothing}\t{hemi}\tok\t1\t1\t1\t0.1\t0.2\t0.2\t0.3\t"
        f"{'bad' * 16}\t/tmp/surface.surf.gii\n"
        for smoothing in ("2mm", "4mm")
        for hemi in ("L", "R")
    )
    table.write_text(header + rows, encoding="utf-8")
    validation = tmp_path / "validation.json"
    validation.write_text(
        json.dumps(
            {
                "status": "complete",
                "subject_manifests": [
                    str(
                        (
                            matching_root
                            / "subjects"
                            / "sub-1001"
                            / "subject_manifest.json"
                        ).resolve()
                    )
                ],
                "run_manifest": workflow.fingerprint(run_manifest),
                "validation_items": workflow.fingerprint(table),
                "items": [
                    {
                        "status": "ok",
                        "subject": "sub-1001",
                        "smoothing": smoothing,
                        "hemi": hemi,
                    }
                    for smoothing in ("2mm", "4mm")
                    for hemi in ("L", "R")
                ],
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="label|validation"):
        workflow.load_source_manifest(
            matching_root=matching_root,
            subjects=["1001"],
            validation_manifest=validation,
        )


def test_fixed_overview_rejects_combo_label_metadata_drift(tmp_path: Path):
    workflow = load_module()
    matching_root = tmp_path / "matching"
    _write_fixed_subject(matching_root)
    combo = next(
        (matching_root / "subjects" / "sub-1001" / "matching").glob(
            "2mm/hemi-L/combo_manifest.json"
        )
    )
    payload = json.loads(combo.read_text(encoding="utf-8"))
    payload["outputs"]["label_gifti"]["shape"] = [511]
    combo.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="shape|metadata|provenance"):
        workflow.load_subject_sources(
            matching_root=matching_root,
            subject="1001",
        )


def test_fixed_overview_fallback_rejects_subject_output_record_drift(
    tmp_path: Path,
):
    workflow = load_module()
    matching_root = tmp_path / "matching"
    _write_fixed_subject(matching_root)
    manifest = matching_root / "subjects" / "sub-1001" / "subject_manifest.json"
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    payload["outputs"]["2mm_hemi-L_label_gifti"]["sha256"] = "f" * 64
    manifest.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="subject.*output|provenance|fingerprint"):
        workflow.load_subject_sources(
            matching_root=matching_root,
            subject="1001",
        )


def test_validation_backed_renderer_requires_full_roster_and_dependencies(
    tmp_path: Path,
):
    workflow = load_module()
    matching_root = tmp_path / "matching"
    _write_fixed_subject(matching_root)
    subject_dir = matching_root / "subjects" / "sub-1001"
    subject_payload = json.loads(
        (subject_dir / "subject_manifest.json").read_text(encoding="utf-8")
    )
    header = (
        "subject\tsmoothing\themi\tstatus\tlabel_1_count\tlabel_2_count\t"
        "label_3_count\tmargin_min\tmargin_median\tmargin_mean\tmargin_max\t"
        "labels_sha256\tconsumed_surface\n"
    )
    table_rows = []
    items = []
    for smoothing in ("2mm", "4mm"):
        for hemi in ("L", "R"):
            combo_path = (
                subject_dir
                / "matching"
                / smoothing
                / f"hemi-{hemi}"
                / "combo_manifest.json"
            )
            combo = json.loads(combo_path.read_text(encoding="utf-8"))
            label = combo["outputs"]["label_gifti"]
            surface = subject_payload["inputs"][f"surface_hemi-{hemi}"]["path"]
            label_sha = workflow.fingerprint(Path(label["path"]))["sha256"]
            table_rows.append(
                f"sub-1001\t{smoothing}\t{hemi}\tok\t1\t1\t1\t0.1\t0.2\t0.2\t0.3\t"
                f"{label_sha}\t{surface}\n"
            )
            items.append(
                {
                    "status": "ok",
                    "subject": "sub-1001",
                    "smoothing": smoothing,
                    "hemi": hemi,
                    "label_counts": {"1": 1, "2": 1, "3": 1},
                    "margin_summary": {
                        "min": 0.1,
                        "median": 0.2,
                        "mean": 0.2,
                        "max": 0.3,
                    },
                    "labels_sha256": label_sha,
                    "density_evidence": {"consumed_asset": surface},
                }
            )
    table = tmp_path / "validation_items.tsv"
    table.write_text(header + "".join(table_rows), encoding="utf-8")
    run_manifest = tmp_path / "run.json"
    run_manifest.write_text(
        json.dumps(
            {
                "status": "complete",
                "subjects": ["sub-1001"],
                "scope": "subset",
            }
        ),
        encoding="utf-8",
    )
    validation = tmp_path / "validation.json"
    validation.write_text(
        json.dumps(
            {
                "status": "complete",
                "subject_manifests": [
                    str((subject_dir / "subject_manifest.json").resolve())
                ],
                "run_manifest": {
                    "path": str(run_manifest.resolve()),
                    **workflow.fingerprint(run_manifest),
                },
                "validation_items": {
                    "path": str(table.resolve()),
                    **workflow.fingerprint(table),
                },
                "items": items,
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="full|70|root|parameter"):
        workflow.load_source_manifest(
            matching_root=matching_root,
            subjects=["1001"],
            validation_manifest=validation,
        )


def test_fixed_overview_rejects_worker_budget_above_process_cap(tmp_path: Path):
    workflow = load_module()

    with pytest.raises(ValueError, match="5"):
        workflow.render_fixed_template_overviews(
            matching_root=tmp_path / "matching",
            output_dir=tmp_path / "present",
            scene=tmp_path / "scene.scene",
            subjects=[],
            jobs=6,
            allow_subset=True,
        )


def test_fixed_overview_writes_in_progress_contract_before_workers(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    workflow = load_module()
    matching_root = tmp_path / "matching"
    _write_fixed_subject(matching_root)
    scene = tmp_path / "scene.scene"
    scene.write_text("scene-v1\n", encoding="utf-8")

    def crash_workers(**kwargs):
        manifest = Path(kwargs["output_dir"]) / (
            "fixed_template_overview_manifest.partial.json"
        )
        assert manifest.is_file()
        payload = json.loads(manifest.read_text(encoding="utf-8"))
        assert payload["summary"]["status"] == "in_progress"
        raise RuntimeError("simulated worker crash")

    monkeypatch.setattr(workflow, "_run_workers", crash_workers)
    with pytest.raises(RuntimeError, match="simulated worker crash"):
        workflow.render_fixed_template_overviews(
            matching_root=matching_root,
            output_dir=tmp_path / "present",
            scene=scene,
            subjects=["1001"],
            jobs=1,
            allow_subset=True,
        )

def test_fixed_overview_resume_validates_status_tsv_fingerprint(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    workflow = load_module()
    matching_root = tmp_path / "matching"
    _write_fixed_subject(matching_root)
    scene = tmp_path / "scene.scene"
    scene.write_text("scene-v1\n", encoding="utf-8")

    def fake_render_locked_grid(**kwargs):
        output = Path(kwargs["outdir"]) / "fake.png"
        output.parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (120, 60), (20, 30, 40)).save(output, format="PNG")
        return {"biglegend_png": str(output)}

    monkeypatch.setattr(workflow, "render_locked_grid", fake_render_locked_grid)
    output_dir = tmp_path / "present"
    workflow.render_fixed_template_overviews(
        matching_root=matching_root,
        output_dir=output_dir,
        scene=scene,
        subjects=["1001"],
        jobs=1,
        allow_subset=True,
    )
    status = output_dir / "fixed_template_overview_status.tsv"
    status.write_text(status.read_text(encoding="utf-8") + "drift\n", encoding="utf-8")

    with pytest.raises(ValueError, match="status|fingerprint"):
        workflow.render_fixed_template_overviews(
            matching_root=matching_root,
            output_dir=output_dir,
            scene=scene,
            subjects=["1001"],
            jobs=1,
            resume=True,
            allow_subset=True,
        )


def test_fixed_overview_recovers_atomic_aggregate_partials(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    workflow = load_module()
    matching_root = tmp_path / "matching"
    _write_fixed_subject(matching_root)
    scene = tmp_path / "scene.scene"
    scene.write_text("scene-v1\n", encoding="utf-8")

    def fake_render_locked_grid(**kwargs):
        output = Path(kwargs["outdir"]) / "fake.png"
        output.parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (120, 60), (20, 30, 40)).save(output, format="PNG")
        return {"biglegend_png": str(output)}

    monkeypatch.setattr(workflow, "render_locked_grid", fake_render_locked_grid)
    output_dir = tmp_path / "present"
    workflow.render_fixed_template_overviews(
        matching_root=matching_root,
        output_dir=output_dir,
        scene=scene,
        subjects=["1001"],
        jobs=1,
        allow_subset=True,
    )
    (output_dir / "fixed_template_overview_manifest.json").rename(
        output_dir / "fixed_template_overview_manifest.partial.json"
    )
    (output_dir / "fixed_template_overview_status.tsv").rename(
        output_dir / "fixed_template_overview_status.partial.tsv"
    )

    outputs = workflow.render_fixed_template_overviews(
        matching_root=matching_root,
        output_dir=output_dir,
        scene=scene,
        subjects=["1001"],
        jobs=1,
        resume=True,
        allow_subset=True,
    )

    assert outputs["manifest"].is_file()
    assert outputs["status_tsv"].is_file()


def test_fixed_overview_pins_native_thread_environment_before_native_import(
    monkeypatch: pytest.MonkeyPatch,
):
    for key in workflow_thread_keys():
        monkeypatch.delenv(key, raising=False)
    workflow = load_module()
    assert all(os.environ.get(key) == "1" for key in workflow_thread_keys())


def workflow_thread_keys() -> tuple[str, ...]:
    return (
        "CORE_WORKFLOW_WB_THREADS",
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
        "ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS",
    )
