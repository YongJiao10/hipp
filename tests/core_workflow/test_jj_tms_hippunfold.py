"""Tests for the fixed JJ_TMS HippUnfold corobl rerun contract."""

from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

import nibabel as nib
import numpy as np
import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
MODULE_PATH = REPO_ROOT / "scripts/core_workflow/jj_tms_hippunfold.py"
SUBJECT = "1001"
CANONICAL_MODEL_PATH = (
    "/home/yojiao/hpchome/hipp/runtime/hippunfold_asset_store_shared/"
    "hippunfold_assets/model/"
    "trained_model.3d_fullres.Task101_hcp1200_T1w.nnUNetTrainerV2.model_best.tar"
)


def load_workflow_module():
    if not MODULE_PATH.exists():
        pytest.fail(f"JJ_TMS workflow module has not been implemented: {MODULE_PATH}")
    spec = importlib.util.spec_from_file_location("jj_tms_hippunfold", MODULE_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _surface_name(subject: str, hemi: str, label: str, surface: str) -> str:
    return (
        f"sub-{subject}_hemi-{hemi}_space-corobl_den-512_"
        f"label-{label}_{surface}.surf.gii"
    )


def _write_gifti_surface(path: Path, vertices: int = 512) -> None:
    coords = np.zeros((vertices, 3), dtype=np.float32)
    faces = np.array([[0, 1, 2]], dtype=np.int32)
    image = nib.gifti.GiftiImage(darrays=[
        nib.gifti.GiftiDataArray(coords, intent="NIFTI_INTENT_POINTSET"),
        nib.gifti.GiftiDataArray(faces, intent="NIFTI_INTENT_TRIANGLE"),
    ])
    nib.save(image, path)



def _write_subject_tree(
    batch_root: Path,
    subject: str = SUBJECT,
    *,
    include_corobl: bool = True,
    include_specs: bool = True,
    log_text: str | None = None,
    metadata_text: str = '{"rule": "create_surface_spec", "incomplete": false}',
) -> Path:
    hipp_root = batch_root / f"sub-{subject}" / "hippunfold"
    subject_root = hipp_root / f"sub-{subject}"
    surf_root = subject_root / "surf"
    surf_root.mkdir(parents=True)

    if include_corobl:
        for hemi in ("L", "R"):
            for label in ("hipp", "dentate"):
                for surface in ("midthickness", "inner", "outer"):
                    vertices = 512 if label == "hipp" else 128
                    _write_gifti_surface(
                        surf_root / _surface_name(subject, hemi, label, surface),
                        vertices=vertices,
                    )

    if include_specs:
        spec_names = (
            f"sub-{subject}_den-512_surfaces.spec",
            f"sub-{subject}_den-512_label-hipp_surfaces.spec",
            f"sub-{subject}_den-512_label-dentate_surfaces.spec",
            f"sub-{subject}_hemi-L_den-512_label-hipp_surfaces.spec",
            f"sub-{subject}_hemi-R_den-512_label-hipp_surfaces.spec",
            f"sub-{subject}_hemi-L_den-512_label-dentate_surfaces.spec",
            f"sub-{subject}_hemi-R_den-512_label-dentate_surfaces.spec",
        )
        spec_text = "\n".join(
            _surface_name(subject, hemi, label, surface)
            for hemi in ("L", "R")
            for label in ("hipp", "dentate")
            for surface in ("midthickness", "inner", "outer")
        )
        for name in spec_names:
            (surf_root / name).write_text(spec_text + "\n", encoding="utf-8")

    overlay_root = (
        batch_root.parent / "runtime" / f"jj_tms_sub-{subject}"
        / "hippunfold_local_model_overlay"
    )
    snakefile = overlay_root / "workflow/Snakefile"
    snakefile.parent.mkdir(parents=True)
    snakefile.write_text("localrule all:\n", encoding="utf-8")
    source_root = batch_root.parent / "installed_hippunfold"
    source_nnunet = source_root / "workflow/rules/nnunet.smk"
    source_nnunet.parent.mkdir(parents=True)
    source_nnunet.write_text("stock nnunet workflow\n", encoding="utf-8")
    overlay_nnunet = overlay_root / "workflow/rules/nnunet.smk"
    overlay_nnunet.parent.mkdir(parents=True)
    overlay_nnunet.write_text(
        f"def get_model_tar():\n    return {json.dumps(CANONICAL_MODEL_PATH)}\n\n"
        "rule unpack_nnunet_model:\n    input: tar=get_model_tar()\n",
        encoding="utf-8",
    )
    (overlay_root / "local_model_overlay.json").write_text(
        json.dumps({
            "status": "ready",
            "source_root": str(source_root),
            "model_path": CANONICAL_MODEL_PATH,
            "snakefile": str(snakefile),
            "remote_rule": "removed from overlay",
            "local_unpack_rule": "retained",
            "model_binding": "explicit get_model_tar return",
            "source_nnunet_sha256": hashlib.sha256(source_nnunet.read_bytes()).hexdigest(),
            "overlay_nnunet_sha256": hashlib.sha256(overlay_nnunet.read_bytes()).hexdigest(),
        }),
        encoding="utf-8",
    )
    config_root = hipp_root / "config"
    config_root.mkdir()
    (config_root / "snakebids.yml").write_text(
        "output_spaces:\n- corobl\n"
        f"snakemake_args:\n- --snakefile\n- {snakefile}\n",
        encoding="utf-8",
    )
    metadata_root = hipp_root / ".snakemake" / "metadata"
    metadata_root.mkdir(parents=True)
    (metadata_root / "surface.json").write_text(metadata_text, encoding="utf-8")
    log_root = hipp_root / ".snakemake" / "log"
    log_root.mkdir(parents=True)
    (log_root / "2026-08-09T000000.snakemake.log").write_text(
        log_text
        or "localrule all:\n"
        "    input: sub-1001/surf/sub-1001_hemi-L_space-corobl_den-512_label-hipp_midthickness.surf.gii\n"
        "Finished jobid: 0 (Rule: all)\n",
        encoding="utf-8",
    )
    return hipp_root


def test_strict_checker_accepts_a_complete_corobl_subject(tmp_path: Path) -> None:
    workflow = load_workflow_module()
    batch_root = tmp_path / "batch"
    _write_subject_tree(batch_root)

    result = workflow.check_subject(batch_root, SUBJECT)

    assert result["status"] == "complete"
    assert result["missing_files"] == []
    assert result["log"]["all_success"] is True
    assert result["metadata"]["remote_model_rule_found"] is False


def test_strict_checker_rejects_wrong_overlay_model_path(tmp_path: Path) -> None:
    workflow = load_workflow_module()
    batch_root = tmp_path / "batch"
    _write_subject_tree(batch_root)
    manifest_path = (
        tmp_path / "runtime/jj_tms_sub-1001/hippunfold_local_model_overlay/"
        "local_model_overlay.json"
    )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["model_path"] = "/tmp/wrong-model.tar"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    result = workflow.check_subject(batch_root, SUBJECT)

    assert result["status"] == "incomplete"
    assert "canonical model" in " ".join(result["errors"])


def test_strict_checker_rejects_tampered_overlay_with_stale_hash(
    tmp_path: Path,
) -> None:
    workflow = load_workflow_module()
    batch_root = tmp_path / "batch"
    _write_subject_tree(batch_root)
    overlay_nnunet = (
        tmp_path / "runtime/jj_tms_sub-1001/hippunfold_local_model_overlay/"
        "workflow/rules/nnunet.smk"
    )
    overlay_nnunet.write_text(
        overlay_nnunet.read_text(encoding="utf-8")
        + "\nrule download_nnunet_model:\n    shell: \x27false\x27\n",
        encoding="utf-8",
    )

    result = workflow.check_subject(batch_root, SUBJECT)

    assert result["status"] == "incomplete"
    assert "overlay_nnunet_sha256" in " ".join(result["config"]["overlay"]["errors"])


def test_strict_checker_rejects_empty_or_wrong_density_surface(
    tmp_path: Path,
) -> None:
    workflow = load_workflow_module()
    batch_root = tmp_path / "batch"
    _write_subject_tree(batch_root)
    surface = (
        batch_root / "sub-1001/hippunfold/sub-1001/surf/"
        "sub-1001_hemi-L_space-corobl_den-512_label-hipp_inner.surf.gii"
    )
    surface.write_bytes(b"")
    empty_result = workflow.check_subject(batch_root, SUBJECT)
    assert empty_result["status"] == "incomplete"

    _write_gifti_surface(surface, vertices=511)
    wrong_density_result = workflow.check_subject(batch_root, SUBJECT)
    assert wrong_density_result["status"] == "incomplete"
    assert wrong_density_result["surface"]["invalid_files"]


def test_strict_checker_rejects_missing_corobl_surface(tmp_path: Path) -> None:
    workflow = load_workflow_module()
    batch_root = tmp_path / "batch"
    _write_subject_tree(batch_root, include_corobl=False)

    result = workflow.check_subject(batch_root, SUBJECT)

    assert result["status"] == "incomplete"
    assert (
        "sub-1001/hippunfold/sub-1001/surf/"
        "sub-1001_hemi-L_space-corobl_den-512_"
        "label-hipp_midthickness.surf.gii"
    ) in result["missing_files"]


def test_checker_rejects_noop_log_as_full_dag_evidence(tmp_path: Path) -> None:
    workflow = load_workflow_module()
    batch_root = tmp_path / "batch"
    _write_subject_tree(batch_root, log_text="Nothing to be done.\n")

    result = workflow.check_subject(batch_root, SUBJECT)

    assert result["status"] == "incomplete"
    assert result["log"]["all_success"] is False


def test_checker_rejects_remote_model_metadata_and_failed_all_log(tmp_path: Path) -> None:
    workflow = load_workflow_module()
    batch_root = tmp_path / "batch"
    _write_subject_tree(
        batch_root,
        log_text=(
            "localrule all:\n"
            "RuleException:\n"
            "SIGINT\n"
            "At least one job did not complete successfully.\n"
        ),
        metadata_text='{"rule": "download_nnunet_model"}',
    )

    result = workflow.check_subject(batch_root, SUBJECT)

    assert result["status"] == "incomplete"
    assert result["log"]["all_success"] is False
    assert result["log"]["failure_markers"]
    assert result["metadata"]["remote_model_rule_found"] is True


def test_run_batch_refuses_production_when_fresh_dag_gate_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    workflow = load_workflow_module()
    monkeypatch.setattr(
        workflow, "read_production_subject_manifest", lambda path: [SUBJECT]
    )
    monkeypatch.setattr(
        workflow, "preflight", lambda *args, **kwargs: {"status": "pass"}
    )
    monkeypatch.setattr(
        workflow, "run_representative_dag_preflight",
        lambda *args, **kwargs: {"status": "fail", "errors": ["incomplete DAG"]},
        raising=False,
    )

    status, report = workflow.run_batch(
        input_root=tmp_path / "raw",
        batch_root=tmp_path / "batch",
        subjects_file=tmp_path / "subjects.txt",
        concurrency=4,
    )

    assert status == 2
    assert report["status"] == "dag_preflight_failed"


def test_batch_command_uses_standard_entry_and_fixed_model(tmp_path: Path) -> None:
    workflow = load_workflow_module()

    command = workflow.build_subject_command(
        SUBJECT,
        input_root=tmp_path / "input",
        batch_root=tmp_path / "batch",
    )

    assert str(workflow.CORE_ENTRY) in command
    assert command[command.index("--model-path") + 1] == str(workflow.CANONICAL_MODEL_PATH)
    assert str(tmp_path / "batch" / "sub-1001" / "hippunfold") in command
    assert "--until" not in command
    assert "--touch" not in command
    assert "download_nnunet_model" not in command


def test_production_roster_must_exactly_match_canonical_manifest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    workflow = load_workflow_module()
    canonical = tmp_path / "canonical.txt"
    alternate = tmp_path / "alternate.txt"
    canonical.write_text("1001\n1002\n", encoding="utf-8")
    alternate.write_text("1002\n1001\n", encoding="utf-8")
    monkeypatch.setattr(workflow, "DEFAULT_SUBJECTS_FILE", canonical)

    with pytest.raises(ValueError, match="canonical fixed roster"):
        workflow.read_production_subject_manifest(alternate)

    assert workflow.read_production_subject_manifest(canonical) == ["1001", "1002"]


def test_input_preflight_matches_runtime_nii_filter(tmp_path: Path) -> None:
    workflow = load_workflow_module()
    anat = tmp_path / "sub-1001/anat"
    anat.mkdir(parents=True)
    (anat / "sub-1001_T1w.nii.gz").write_text("gzip\n", encoding="utf-8")
    assert workflow._input_t1w_files(tmp_path, SUBJECT) == []
    nii = anat / "sub-1001_T1w.nii"
    nii.write_text("nii\n", encoding="utf-8")
    assert workflow._input_t1w_files(tmp_path, SUBJECT) == [nii]


def test_preflight_reports_positive_dag_contract() -> None:
    text = (
        REPO_ROOT / "scripts/core_workflow/preflight_jj_tms_hippunfold_dag.py"
    ).read_text(encoding="utf-8")
    assert '"status": "pass" if not errors else "fail"' in text
    assert '"rule_counts": rule_counts' in text
    assert '"errors": errors' in text


def test_preflight_rejects_missing_unpacked_model_dir(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    workflow = load_workflow_module()
    input_root = tmp_path / "raw"
    anat = input_root / "sub-1001/anat"
    anat.mkdir(parents=True)
    (anat / "sub-1001_T1w.nii").write_text("input\n", encoding="utf-8")
    model_tar = tmp_path / "model.tar"
    model_tar.write_text("model\n", encoding="utf-8")
    monkeypatch.setattr(workflow, "CANONICAL_MODEL_PATH", model_tar)
    monkeypatch.setattr(workflow, "CANONICAL_MODEL_DIR", tmp_path / "missing-model-dir")

    report = workflow.preflight(input_root, [SUBJECT], expected_subject_count=1)

    assert report["status"] == "fail"
    assert "unpacked model" in " ".join(report["errors"])


def test_dag_preflight_creates_only_the_required_output_parent() -> None:
    text = (
        REPO_ROOT / "scripts/core_workflow/preflight_jj_tms_hippunfold_dag.py"
    ).read_text(encoding="utf-8")

    assert "probe_output.exists()" in text
    assert "probe_output.parent.mkdir(parents=True)" in text


def test_dag_contract_requires_exact_full_targets() -> None:
    workflow = load_workflow_module()

    dag_text = (
        "localrule all:\n"
        + "rule gen_native_mesh:\n" * 4
        + "rule resample_native_surf_to_atlas_density:\n" * 12
        + "rule create_spec_file:\n" * 4
        + "HippUnfold local-model workflow: /tmp/hippunfold_local_model_overlay/workflow/Snakefile\n"
    )

    assert workflow.validate_dag_output(dag_text) == []
    assert workflow.validate_dag_output(dag_text.replace("rule gen_native_mesh:\n", "", 1))
    assert workflow.validate_dag_output(
        dag_text + "localrule download_nnunet_model:\n"
    )



def test_batch_report_never_marks_partial_roster_complete(tmp_path: Path) -> None:
    workflow = load_workflow_module()
    batch_root = tmp_path / "batch"
    _write_subject_tree(batch_root)

    report = workflow.build_batch_report(batch_root, [SUBJECT], expected_subject_count=70)

    assert report["status"] == "incomplete"
    assert report["complete_count"] == 1
    assert report["subject_count"] == 1
    assert report["expected_subject_count"] == 70


def test_downstream_unfold_surface_reference_is_rejected(tmp_path: Path) -> None:
    workflow = load_workflow_module()
    batch_root = tmp_path / "batch"
    downstream_root = tmp_path / "downstream"
    _write_subject_tree(batch_root)
    downstream_root.mkdir()
    (downstream_root / "surface_manifest.json").write_text(
        json.dumps(
            {
                "surface": (
                    "sub-1001_hemi-L_space-unfold_den-512_"
                    "label-hipp_midthickness.surf.gii"
                )
            }
        ),
        encoding="utf-8",
    )

    result = workflow.check_subject(
        batch_root, SUBJECT, downstream_root=downstream_root
    )

    assert result["status"] == "incomplete"
    assert result["downstream"]["forbidden_references"]
def test_strict_checker_accepts_hipp_512_and_dentate_128_corobl_surfaces(
    tmp_path: Path,
) -> None:
    workflow = load_workflow_module()
    batch_root = tmp_path / "batch"
    _write_subject_tree(batch_root)

    result = workflow.check_subject(batch_root, SUBJECT)

    assert result["status"] == "complete"
    dentate_surface = (
        "sub-1001/hippunfold/sub-1001/surf/"
        "sub-1001_hemi-L_space-corobl_den-512_label-dentate_midthickness.surf.gii"
    )
    assert result["surface"]["vertex_counts"][dentate_surface] == 128
