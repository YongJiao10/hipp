from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]

NATIVE_BETA_FILES = (
    "compare_native_standard_surface_bad_rates.py",
    "copy_native_con_inputs.py",
    "run_native_surface_beta.py",
    "render_native_surface_beta_figures.py",
    "stanford_tms_surface_model.R",
    "stanford_tms_five_figures.py",
    "jj_tms_sptms_native_surface_model.R",
    "render_jj_tms_five_figures.py",
)


def test_native_beta_entrypoints_live_only_on_exploratory_surface():
    exploratory_dir = REPO_ROOT / "scripts" / "exploratory_beta"
    formal_dir = REPO_ROOT / "scripts" / "individual_workflow"

    assert all((exploratory_dir / name).is_file() for name in NATIVE_BETA_FILES)
    assert all(not (formal_dir / name).exists() for name in NATIVE_BETA_FILES)


def test_exploratory_status_and_active_paths_are_documented():
    exploratory_dir = REPO_ROOT / "scripts" / "exploratory_beta"
    readme = (exploratory_dir / "README.md").read_text(encoding="utf-8")
    assert "formal-workflow step" in readme
    assert "not automatically" in readme

    active_docs = (
        REPO_ROOT / "scripts" / "README.md",
        REPO_ROOT / "scripts" / "individual_workflow" / "README.md",
        REPO_ROOT / "docs" / "hipp_parcellation_network" / "archive_index.md",
        REPO_ROOT / "docs" / "hipp_parcellation_network" / "stanford_tms_surface_beta.md",
        REPO_ROOT / "docs" / "hipp_parcellation_network" / "jj_tms_sptms_cluster_beta.md",
    )
    combined = "\n".join(path.read_text(encoding="utf-8") for path in active_docs)

    for name in NATIVE_BETA_FILES:
        assert f"scripts/individual_workflow/{name}" not in combined
    assert "scripts/exploratory_beta/run_native_surface_beta.py" in combined
