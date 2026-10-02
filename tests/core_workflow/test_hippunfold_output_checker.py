import importlib.util
import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
CHECKER_PATH = REPO_ROOT / "scripts/core_workflow/01_02_check_hippunfold_outputs.py"


def load_checker():
    sys.path.insert(0, str(REPO_ROOT / "scripts/core_workflow"))
    spec = importlib.util.spec_from_file_location("hippunfold_output_checker", CHECKER_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_checker_accepts_hippunfold_2_completion_logs() -> None:
    checker = load_checker()

    assert checker.is_successful_log("56 of 56 steps (100%) done")
    assert checker.is_successful_log("41 of 41 steps (100%) done")
    assert checker.is_successful_log(
        "Nothing to be done (all requested files are present and up to date)."
    )
    assert not checker.is_successful_log(
        "WorkflowError: At least one job did not complete successfully."
    )


def _make_full_subject(checker, batch_root: Path, subject: str, templates: list[str]) -> None:
    output_root = batch_root / f"sub-{subject}" / "hippunfold"
    for template in checker.required_check_templates(templates):
        path = output_root / template.replace("{subject}", subject)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("test\n", encoding="utf-8")
    subject_root = output_root / f"sub-{subject}"
    for name in ("coords", "warps"):
        path = subject_root / name / "intermediate.nii.gz"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("intermediate\n", encoding="utf-8")


def test_complete_subject_is_pruned_once_and_receipt_allows_recheck(tmp_path: Path) -> None:
    checker = load_checker()
    subject = "1001"
    batch_root = tmp_path / "batch"
    manifest_cache = batch_root / "_batch_logs" / "output_integrity_manifest.json"
    templates = [
        "sub-{subject}/anat/final_T1w.nii.gz",
        "sub-{subject}/cifti/final.dscalar.nii",
        "sub-{subject}/metric/subfields.label.gii",
        "sub-{subject}/qc/dice.tsv",
        "sub-{subject}/surf/midthickness.surf.gii",
    ]
    _make_full_subject(checker, batch_root, subject, templates)

    report = checker.validate_and_cleanup_subject(
        batch_root,
        subject,
        templates,
        manifest_cache=manifest_cache,
        cleanup_enabled=True,
    )

    assert report["status"] == "complete_retained_outputs_after_cleanup"
    assert report["validation_mode"] == "retained_outputs_with_valid_cleanup_receipt"
    assert report["cleanup"]["performed"] is True
    subject_root = batch_root / f"sub-{subject}" / "hippunfold" / f"sub-{subject}"
    for name in checker.PRUNED_OUTPUT_DIRS:
        assert not (subject_root / name).exists()
    assert (subject_root / "metric/subfields.label.gii").is_file()
    assert (subject_root / "qc/dice.tsv").is_file()
    assert (subject_root / "surf/midthickness.surf.gii").is_file()

    receipt_path = checker.cleanup_receipt_path(batch_root, subject)
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    assert receipt["status"] == "cleanup_complete"
    assert receipt["approved_pruned_directories"] == list(checker.PRUNED_OUTPUT_DIRS)
    assert receipt["files_removed"] == 4

    second = checker.validate_and_cleanup_subject(
        batch_root,
        subject,
        templates,
        manifest_cache=manifest_cache,
        cleanup_enabled=True,
    )
    assert second["status"] == "complete_retained_outputs_after_cleanup"
    assert second["cleanup"]["performed"] is False
    assert second["cleanup"]["already_complete"] is True
    assert json.loads(receipt_path.read_text(encoding="utf-8")) == receipt


def test_incomplete_subject_is_not_pruned(tmp_path: Path) -> None:
    checker = load_checker()
    subject = "1002"
    batch_root = tmp_path / "batch"
    templates = [
        "sub-{subject}/anat/final_T1w.nii.gz",
        "sub-{subject}/surf/midthickness.surf.gii",
    ]
    _make_full_subject(checker, batch_root, subject, templates)
    missing = batch_root / f"sub-{subject}/hippunfold/sub-{subject}/surf/midthickness.surf.gii"
    missing.unlink()

    report = checker.validate_and_cleanup_subject(
        batch_root,
        subject,
        templates,
        manifest_cache=batch_root / "_batch_logs/output_integrity_manifest.json",
        cleanup_enabled=True,
    )

    assert report["status"] == "missing_outputs"
    subject_root = batch_root / f"sub-{subject}" / "hippunfold" / f"sub-{subject}"
    assert (subject_root / "anat").is_dir()
    assert (subject_root / "coords").is_dir()
    assert (subject_root / "warps").is_dir()
    assert not checker.cleanup_receipt_path(batch_root, subject).exists()


def test_list_batch_subjects_ignores_placeholder_and_missing_payload(tmp_path: Path) -> None:
    checker = load_checker()
    batch_root = tmp_path / "batch"
    (batch_root / "sub-1001/hippunfold/sub-1001").mkdir(parents=True)
    (batch_root / "sub-").mkdir(parents=True)
    (batch_root / "sub-1002/hippunfold").mkdir(parents=True)

    assert checker.list_batch_subjects(batch_root) == ["1001"]
