"""Tests for ``scripts.core_workflow.lib.path_templates``."""

from pathlib import Path

import pytest

from scripts.core_workflow.lib.path_templates import (
    format_subject_path_template,
    normalize_subject,
)
from scripts.core_workflow.lib.output_layout import (
    REPO_ROOT,
    load_legacy_wrong_path_mapping,
    resolve_cohort_gradient_root,
    resolve_cohort_group_atlas_root,
    resolve_cohort_overview_root,
    resolve_hcp_kong400_control_shared_root,
    resolve_kong2022_routea_root,
    resolve_kong2022_routea_ctxreg_root,
    resolve_kong2022_routeb_root,
    resolve_kong2022_surface_root,
    resolve_legacy_wrong_path,
)


def test_normalize_subject_removes_sub_prefix():
    assert normalize_subject("sub-2001") == "2001"
    assert normalize_subject("2001") == "2001"


def test_retired_unsupported_variants_are_rejected():
    from scripts.core_workflow.lib.output_layout import (
        resolve_kong2022_variant_root,
        validate_iowa7t_variant,
    )

    with pytest.raises(ValueError):
        resolve_kong2022_variant_root("stanford", "unsupported")
    with pytest.raises(ValueError):
        validate_iowa7t_variant("unsupported")
    assert resolve_kong2022_variant_root("stanford", "36p") == REPO_ROOT / "outputs_stanford/inputs/kong2022"
    assert validate_iowa7t_variant("36p") == "36p"


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
    pfm_root = resolve_kong2022_routea_root("iowa")
    surface_root = resolve_kong2022_surface_root("iowa")

    assert pfm_root.parts[-2:] == ("shared_inputs", "pfm")
    assert surface_root.parts[-2:] == ("shared_inputs", "surface")


def test_kong2022_all_shared_input_roots_are_cohort_independent():
    expected = {
        resolve_kong2022_routea_root(): "pfm",
        resolve_kong2022_surface_root(): "surface",
        resolve_kong2022_routeb_root(): "fixed",
        resolve_kong2022_routea_ctxreg_root(): "pfm_ctxreg",
    }

    assert {path.name for path in expected} == set(expected.values())


def test_hcp_kong400_control_roots_use_formal_route_namespace():
    assert resolve_hcp_kong400_control_shared_root("pfm_ctxreg").parts[-2:] == ("shared_inputs", "pfm_ctxreg")
    with pytest.raises(ValueError, match="Unsupported Kong400 control route"):
        resolve_hcp_kong400_control_shared_root("pfm_ctxdrop")


def test_hcp_kong400_control_roots_reject_non_hcp_dataset():
    with pytest.raises(ValueError, match="only defined for dataset='hcp'"):
        resolve_hcp_kong400_control_shared_root("pfm_ctxreg", dataset="iowa")


def test_cohort_resolvers_omit_redundant_cohort_layer_when_none():
    assert resolve_cohort_group_atlas_root("hcp", "pfm").parts[-3:] == (
        "work",
        "group_atlas",
        "pfm",
    )
    assert resolve_cohort_gradient_root("stanford", "pfm", cohort="nthc").parts[-4:] == (
        "work",
        "hippomaps_gradients",
        "pfm",
        "nthc",
    )
    assert resolve_cohort_overview_root("iowa_jj_tms", "pfm", cohort="patient").parts[-4:] == (
        "present",
        "overviews",
        "pfm",
        "patient",
    )


def test_legacy_wrong_path_mapping_resolves_all_declared_targets(tmp_path):
    mappings = load_legacy_wrong_path_mapping()
    assert mappings
    assert resolve_legacy_wrong_path("outputs_hcp/work/group_atlas/pfm", repo_root=tmp_path) == (
        tmp_path / "wrong/outputs_hcp/work/group_atlas/kong2022_7net_full_k3_fisherz"
    ).resolve()
    assert resolve_legacy_wrong_path(
        "outputs_hcp/work/group_atlas/pfm/gradient-spectral/kong2022/2mm/cov_0p80/g1g2g3/hemi_L",
        repo_root=tmp_path,
    ) == (
        tmp_path
        / "wrong/outputs_hcp/work/group_atlas/kong2022_7net_full_k3_fisherz/gradient-spectral/kong2022/2mm/cov_0p80/g1g2g3/hemi_L"
    ).resolve()
    assert resolve_legacy_wrong_path("outputs_stanford/work/group_atlas/pfm/nthc", repo_root=tmp_path) == (
        tmp_path / "wrong/outputs_stanford/work/group_atlas/routea_7net_nthc_full_k3_fisherz"
    ).resolve()
    for new_rel, old_rel in mappings.items():
        resolved = resolve_legacy_wrong_path(new_rel, repo_root=tmp_path)
        assert old_rel.startswith("wrong/outputs_")
        assert resolved == (tmp_path / old_rel).resolve()


@pytest.mark.parametrize("new_directory,old_directory,old_tag,old_atlas", [
    ("outputs_hcp/present/overviews/pfm", "wrong/outputs_hcp/present/overviews/kong2022_7net", "kong2022_7net", "kong2022_7net"),
    ("outputs_iowa_jj_tms/present/overviews/pfm/patient", "wrong/outputs_iowa_jj_tms/present/overviews/kong2022_7net/36p/patient", "kong2022_7net", "kong2022_7net"),
])
def test_legacy_mapping_resolves_renamed_overview_files(tmp_path, new_directory, old_directory, old_tag, old_atlas):
    new_name = "kong2022_gradient-spectral_group_overview_g1g2g3_pfm_fisherz.png"
    old_name = f"{old_atlas}_gradient-spectral_group_overview_g1g2g3_{old_tag}_fisherz.png"
    assert resolve_legacy_wrong_path(f"{new_directory}/{new_name}", repo_root=tmp_path) == tmp_path / old_directory / old_name


@pytest.mark.parametrize("retained_asset", [
    "outputs_hcp/inputs/kong2022/cortex_references/pfm_areal_mshbm",
    "outputs_hcp/inputs/kong2022/templates/hipp_group_average",
])
def test_legacy_mapping_excludes_retained_active_assets(tmp_path, retained_asset):
    with pytest.raises(KeyError, match="No declared legacy"):
        resolve_legacy_wrong_path(retained_asset, repo_root=tmp_path)
