"""Tests for ``scripts.core_workflow.lib.path_templates``."""

from pathlib import Path

import pytest

from scripts.core_workflow.lib.path_templates import (
    format_subject_path_template,
    normalize_subject,
)


def test_normalize_subject_removes_sub_prefix():
    assert normalize_subject("sub-2001") == "2001"
    assert normalize_subject("2001") == "2001"


def test_normalize_subject_only_first_occurrence():
    assert normalize_subject("sub-sub-2001") == "sub-2001"


def test_format_subject_path_template_supports_stanford_dtseries():
    path = format_subject_path_template(
        "{data_root}/{sub_subject}/func/"
        "{sub_subject}_task-rest_space-fsLR_den-91k_desc-denoised_bold.dtseries.nii",
        data_root="/tmp/data",
        subject="sub-2001",
    )
    assert path == Path(
        "/tmp/data/sub-2001/func/"
        "sub-2001_task-rest_space-fsLR_den-91k_desc-denoised_bold.dtseries.nii"
    )


def test_format_subject_path_template_supports_legacy_bold_default():
    path = format_subject_path_template(
        "{bold_root}/{subject}/func_uncompressed.nii",
        bold_root="/tmp/bold",
        subject="sub-2001",
    )
    assert path == Path("/tmp/bold/2001/func_uncompressed.nii")


def test_format_subject_path_template_supports_hemi_surface():
    path = format_subject_path_template(
        "{data_root}/{sub_subject}/anat/"
        "{sub_subject}_hemi-{hemi}_space-fsLR_den-32k_midthickness.surf.gii",
        data_root="/tmp/data",
        subject="2001",
        hemi="L",
    )
    assert path == Path(
        "/tmp/data/sub-2001/anat/"
        "sub-2001_hemi-L_space-fsLR_den-32k_midthickness.surf.gii"
    )


def test_format_subject_path_template_hemi_lower():
    path = format_subject_path_template(
        "{data_root}/{sub_subject}/anat/"
        "{sub_subject}_hemi-{hemi_lower}h_midthickness.surf.gii",
        data_root="/tmp/data",
        subject="2001",
        hemi="R",
    )
    assert path == Path(
        "/tmp/data/sub-2001/anat/"
        "sub-2001_hemi-rh_midthickness.surf.gii"
    )


def test_format_subject_path_template_hemi_none_yields_empty():
    path = format_subject_path_template(
        "{data_root}/{sub_subject}/file_{hemi}.txt",
        data_root="/tmp",
        subject="2001",
    )
    assert path == Path("/tmp/sub-2001/file_.txt")


def test_format_subject_path_template_reports_unknown_variable():
    with pytest.raises(KeyError, match="Unknown path template variable"):
        format_subject_path_template(
            "{data_root}/{missing}/{sub_subject}.nii",
            data_root="/tmp/data",
            subject="2001",
        )


def test_format_subject_path_template_resolves_relative_roots():
    """Roots are resolved to absolute paths before substitution."""
    path = format_subject_path_template(
        "{data_root}/{sub_subject}/file.nii",
        data_root=Path("relative/data"),
        subject="2001",
    )
    assert path.is_absolute()
