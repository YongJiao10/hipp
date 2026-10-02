"""Tests for ``scripts.core_workflow.lib.path_templates``."""

from pathlib import Path

import pytest

from scripts.core_workflow.lib.path_templates import (
    format_subject_path_template,
    normalize_subject,
)
from scripts.core_workflow.lib.output_layout import (
    resolve_hcp_kong400_control_shared_root,
    resolve_kong2022_routea_root,
    resolve_kong2022_routea_ctxdrop_root,
    resolve_kong2022_routea_ctxreg_root,
    resolve_kong2022_routeb_root,
    resolve_kong2022_surface_root,
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


def test_kong2022_base_shared_roots_are_dataset_agnostic():
    routea_root = resolve_kong2022_routea_root("iowa")
    surface_root = resolve_kong2022_surface_root("iowa")

    assert routea_root.parts[-2:] == ("shared_inputs", "routea")
    assert surface_root.parts[-2:] == ("shared_inputs", "surface")


def test_kong2022_all_shared_input_roots_are_cohort_independent():
    expected = {
        resolve_kong2022_routea_root(): "routea",
        resolve_kong2022_surface_root(): "surface",
        resolve_kong2022_routeb_root(): "routeb",
        resolve_kong2022_routea_ctxreg_root(): "routea_ctxreg",
        resolve_kong2022_routea_ctxdrop_root(): "routea_ctxdrop",
    }

    assert {path.name for path in expected} == set(expected.values())


def test_hcp_kong400_control_roots_use_formal_route_namespace():
    assert resolve_hcp_kong400_control_shared_root("routea_ctxreg").parts[-2:] == ("shared_inputs", "routea_ctxreg")
    assert resolve_hcp_kong400_control_shared_root("routea_ctxdrop").parts[-2:] == ("shared_inputs", "routea_ctxdrop")


def test_hcp_kong400_control_roots_reject_non_hcp_dataset():
    with pytest.raises(ValueError, match="only defined for dataset='hcp'"):
        resolve_hcp_kong400_control_shared_root("routea_ctxreg", dataset="iowa")
