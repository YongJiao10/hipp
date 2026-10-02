"""Regression tests for the HippUnfold output-space completeness contract."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
CHECKER_PATH = REPO_ROOT / "scripts/core_workflow/01_02_check_hippunfold_outputs.py"
sys.path.insert(0, str(REPO_ROOT / "scripts/core_workflow"))
SPEC = importlib.util.spec_from_file_location("hippunfold_output_checker", CHECKER_PATH)
assert SPEC is not None and SPEC.loader is not None
checker = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(checker)


def test_partial_space_unfold_manifest_requires_all_fixed_space_anchors() -> None:
    partial_manifest = [
        "sub-{subject}/surf/sub-{subject}_hemi-L_space-unfold_den-512_label-hipp_midthickness.surf.gii",
        "sub-{subject}/surf/sub-{subject}_hemi-R_space-unfold_den-512_label-hipp_midthickness.surf.gii",
    ]

    required = checker.required_check_templates(partial_manifest)

    for space in ("corobl", "T1w", "unfold"):
        for hemi in ("L", "R"):
            expected = (
                f"sub-{{subject}}/surf/sub-{{subject}}_hemi-{hemi}_space-{space}_"
                "den-512_label-hipp_midthickness.surf.gii"
            )
            assert expected in required

    assert (
        "sub-{subject}/surf/sub-{subject}_den-512_label-hipp_surfaces.spec"
        in required
    )
    assert (
        "sub-{subject}/qc/sub-{subject}_hemi-L_desc-unetf3d_dice.tsv"
        in required
    )
    assert (
        "sub-{subject}/qc/sub-{subject}_hemi-R_desc-unetf3d_dice.tsv"
        in required
    )


def test_every_manifest_is_normalized_to_the_fixed_contract() -> None:
    generic = ["sub-{subject}/surf/midthickness.surf.gii"]
    required = checker.required_check_templates(generic)

    assert generic[0] in required
    assert all(anchor in required for anchor in checker.FIXED_HIPPUNFOLD_SPACE_ANCHORS)


def test_legacy_receipt_hash_cannot_bypass_fixed_contract(tmp_path: Path) -> None:
    batch_root = tmp_path / "batch"
    subject = "1001"
    output_root = batch_root / f"sub-{subject}" / "hippunfold"
    output_root.joinpath(f"sub-{subject}/surf/midthickness.surf.gii").parent.mkdir(
        parents=True
    )
    output_root.joinpath(f"sub-{subject}/surf/midthickness.surf.gii").write_text(
        "surface\n", encoding="utf-8"
    )

    raw_manifest = ["sub-{subject}/surf/midthickness.surf.gii"]
    legacy_hash = hashlib.sha256(
        json.dumps(raw_manifest, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    receipt_path = checker.cleanup_receipt_path(batch_root, subject)
    receipt_path.parent.mkdir(parents=True, exist_ok=True)
    receipt_path.write_text(
        json.dumps(
            {
                "schema_version": checker.CLEANUP_RECEIPT_SCHEMA_VERSION,
                "status": "cleanup_complete",
                "subject": subject,
                "manifest_contract_sha256": legacy_hash,
                "full_expected_outputs": 75,
                "retained_expected_outputs": 9,
                "approved_pruned_directories": list(checker.PRUNED_OUTPUT_DIRS),
                "allocated_bytes_removed": 4,
                "apparent_bytes_removed": 4,
                "files_removed": 4,
                "removed_directories": [
                    {
                        "directory": name,
                        "path": f"/old-root/{name}",
                        "existed": True,
                        "allocated_bytes_removed": 1,
                        "apparent_bytes_removed": 1,
                        "files_removed": 1,
                    }
                    for name in checker.PRUNED_OUTPUT_DIRS
                ],
            }
        ),
        encoding="utf-8",
    )

    report = checker.check_subject(batch_root, subject, raw_manifest)

    assert report["status"] == "missing_outputs"
    assert report["cleanup_receipt"] is None
    assert any("space-corobl" in path for path in report["missing_outputs"])


def test_historical_cleanup_receipt_is_reconciled_against_current_retained_contract(
    tmp_path: Path,
) -> None:
    batch_root = tmp_path / "batch"
    subject = "1001"
    output_root = batch_root / f"sub-{subject}" / "hippunfold"
    retained_paths = [
        "sub-{subject}/surf/sub-{subject}_hemi-L_space-corobl_den-512_label-hipp_midthickness.surf.gii",
        "sub-{subject}/surf/sub-{subject}_hemi-R_space-corobl_den-512_label-hipp_midthickness.surf.gii",
        "sub-{subject}/surf/sub-{subject}_hemi-L_space-T1w_den-512_label-hipp_midthickness.surf.gii",
        "sub-{subject}/surf/sub-{subject}_hemi-R_space-T1w_den-512_label-hipp_midthickness.surf.gii",
        "sub-{subject}/surf/sub-{subject}_hemi-L_space-unfold_den-512_label-hipp_midthickness.surf.gii",
        "sub-{subject}/surf/sub-{subject}_hemi-R_space-unfold_den-512_label-hipp_midthickness.surf.gii",
        "sub-{subject}/surf/sub-{subject}_den-512_label-hipp_surfaces.spec",
        "sub-{subject}/qc/sub-{subject}_hemi-L_desc-unetf3d_dice.tsv",
        "sub-{subject}/qc/sub-{subject}_hemi-R_desc-unetf3d_dice.tsv",
        "sub-{subject}/metric/subfields.label.gii",
    ]
    for template in retained_paths:
        path = output_root / template.replace("{subject}", subject)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("retained\n", encoding="utf-8")

    receipt_path = checker.cleanup_receipt_path(batch_root, subject)
    receipt_path.parent.mkdir(parents=True, exist_ok=True)
    receipt_path.write_text(
        json.dumps(
            {
                "schema_version": checker.CLEANUP_RECEIPT_SCHEMA_VERSION,
                "status": "cleanup_complete",
                "subject": subject,
                "manifest_contract_sha256": "0" * 64,
                "full_expected_outputs": 75,
                "retained_expected_outputs": 10,
                "approved_pruned_directories": list(checker.PRUNED_OUTPUT_DIRS),
                "allocated_bytes_removed": 4,
                "apparent_bytes_removed": 4,
                "files_removed": 4,
                "removed_directories": [
                    {
                        "directory": name,
                        "path": f"/old-root/{name}",
                        "existed": True,
                        "allocated_bytes_removed": 1,
                        "apparent_bytes_removed": 1,
                        "files_removed": 1,
                    }
                    for name in checker.PRUNED_OUTPUT_DIRS
                ],
            }
        ),
        encoding="utf-8",
    )

    current_manifest = [
        "sub-{subject}/anat/final_T1w.nii.gz",
        "sub-{subject}/metric/subfields.label.gii",
    ]

    report = checker.check_subject(batch_root, subject, current_manifest)

    assert report["status"] == "complete_retained_outputs_after_cleanup"
    assert report["validation_mode"] == "retained_outputs_with_legacy_cleanup_receipt"
    assert report["missing_outputs"] == []
    assert report["cleanup_receipt"] == str(receipt_path)


def test_historical_cleanup_receipt_does_not_hide_missing_retained_anchor(
    tmp_path: Path,
) -> None:
    batch_root = tmp_path / "batch"
    subject = "1002"
    output_root = batch_root / f"sub-{subject}" / "hippunfold"
    retained_paths = [
        "sub-{subject}/surf/sub-{subject}_hemi-L_space-corobl_den-512_label-hipp_midthickness.surf.gii",
        "sub-{subject}/surf/sub-{subject}_hemi-R_space-corobl_den-512_label-hipp_midthickness.surf.gii",
        "sub-{subject}/surf/sub-{subject}_hemi-L_space-T1w_den-512_label-hipp_midthickness.surf.gii",
        "sub-{subject}/surf/sub-{subject}_hemi-R_space-T1w_den-512_label-hipp_midthickness.surf.gii",
        "sub-{subject}/surf/sub-{subject}_hemi-L_space-unfold_den-512_label-hipp_midthickness.surf.gii",
        "sub-{subject}/surf/sub-{subject}_hemi-R_space-unfold_den-512_label-hipp_midthickness.surf.gii",
        "sub-{subject}/surf/sub-{subject}_den-512_label-hipp_surfaces.spec",
        "sub-{subject}/qc/sub-{subject}_hemi-L_desc-unetf3d_dice.tsv",
        "sub-{subject}/qc/sub-{subject}_hemi-R_desc-unetf3d_dice.tsv",
        "sub-{subject}/metric/subfields.label.gii",
    ]
    for template in retained_paths[:-1]:
        path = output_root / template.replace("{subject}", subject)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("retained\n", encoding="utf-8")

    receipt_path = checker.cleanup_receipt_path(batch_root, subject)
    receipt_path.parent.mkdir(parents=True, exist_ok=True)
    receipt_path.write_text(
        json.dumps(
            {
                "schema_version": checker.CLEANUP_RECEIPT_SCHEMA_VERSION,
                "status": "cleanup_complete",
                "subject": subject,
                "manifest_contract_sha256": "0" * 64,
                "full_expected_outputs": 75,
                "retained_expected_outputs": 10,
                "approved_pruned_directories": list(checker.PRUNED_OUTPUT_DIRS),
                "allocated_bytes_removed": 4,
                "apparent_bytes_removed": 4,
                "files_removed": 4,
                "removed_directories": [
                    {
                        "directory": name,
                        "path": f"/old-root/{name}",
                        "existed": True,
                        "allocated_bytes_removed": 1,
                        "apparent_bytes_removed": 1,
                        "files_removed": 1,
                    }
                    for name in checker.PRUNED_OUTPUT_DIRS
                ],
            }
        ),
        encoding="utf-8",
    )

    current_manifest = [
        "sub-{subject}/anat/final_T1w.nii.gz",
        "sub-{subject}/metric/subfields.label.gii",
    ]

    report = checker.check_subject(batch_root, subject, current_manifest)

    assert report["status"] == "missing_outputs"
    assert any("subfields.label.gii" in path for path in report["missing_outputs"])


def test_historical_cleanup_receipt_requires_complete_removal_evidence(tmp_path: Path) -> None:
    batch_root = tmp_path / "batch"
    subject = "1003"
    receipt_path = checker.cleanup_receipt_path(batch_root, subject)
    receipt_path.parent.mkdir(parents=True, exist_ok=True)
    receipt_path.write_text(
        json.dumps(
            {
                "schema_version": checker.CLEANUP_RECEIPT_SCHEMA_VERSION,
                "status": "cleanup_complete",
                "subject": subject,
                "manifest_contract_sha256": "1" * 64,
                "full_expected_outputs": 75,
                "retained_expected_outputs": 9,
                "approved_pruned_directories": list(checker.PRUNED_OUTPUT_DIRS),
                "removed_directories": [
                    {"directory": name} for name in checker.PRUNED_OUTPUT_DIRS
                ],
            }
        ),
        encoding="utf-8",
    )

    assert checker.load_legacy_cleanup_receipt(
        batch_root,
        subject,
        ["sub-{subject}/anat/final_T1w.nii.gz"],
    ) is None
