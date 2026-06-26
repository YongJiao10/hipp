#!/usr/bin/env python3
from __future__ import annotations

import importlib
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
CORE_DIR = REPO_ROOT / "scripts" / "core_workflow"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(CORE_DIR) not in sys.path:
    sys.path.insert(0, str(CORE_DIR))


extract_mod = importlib.import_module("scripts.core_workflow.lib.extract_cortex_kong2022_route_a_roi_timeseries")
roi_fc_mod = importlib.import_module("scripts.core_workflow.04_01_materialize_kong2022_roi_fc")


def test_routea_label_paths_are_direct_flat_paths(tmp_path: Path) -> None:
    root = tmp_path / "routea_areal_mshbm"
    subject_dir = root / "sub-100610"
    subject_dir.mkdir(parents=True)
    (subject_dir / "lh_labels.txt").write_text("1\n", encoding="utf-8")
    (subject_dir / "rh_labels.txt").write_text("201\n", encoding="utf-8")

    lh, rh = extract_mod.route_a_label_paths(root, "sub-100610")

    assert lh == subject_dir / "lh_labels.txt"
    assert rh == subject_dir / "rh_labels.txt"


def test_routea_label_paths_reject_nested_only_outputs(tmp_path: Path) -> None:
    root = tmp_path / "routea_areal_mshbm"
    nested = root / "sub-100610" / "kong2022_areal_mshbm" / "project_dir" / "route_a_outputs"
    nested.mkdir(parents=True)
    (nested / "lh_labels.txt").write_text("1\n", encoding="utf-8")
    (nested / "rh_labels.txt").write_text("201\n", encoding="utf-8")

    try:
        extract_mod.route_a_label_paths(root, "100610")
    except FileNotFoundError as exc:
        msg = str(exc)
        assert "sub-100610/lh_labels.txt" in msg
        assert "nested" not in msg.lower()
        assert "fallback" not in msg.lower()
    else:
        raise AssertionError("nested-only RouteA outputs must not be accepted")


def test_routea_modules_no_longer_export_discovery_or_label_dir_override() -> None:
    assert not hasattr(extract_mod, "discover_route_a_outputs_dir")
    assert not hasattr(roi_fc_mod, "discover_route_a_outputs_dir")

    extract_parser = extract_mod.build_arg_parser()
    roi_parser = roi_fc_mod.build_arg_parser()
    assert "--route-a-label-dir" not in {opt for action in extract_parser._actions for opt in action.option_strings}
    assert "--route-a-label-dir" not in {opt for action in roi_parser._actions for opt in action.option_strings}


def test_routea_backend_no_longer_emits_mat_sidecar() -> None:
    checked = [
        REPO_ROOT / "scripts" / "core_workflow" / "sge_argon" / "matlab_kong2022_mshbm_subject_sge.sh",
        REPO_ROOT / "scripts" / "ArealMSHBM_Kong2022_Iowa_Stanford" / "run_kong2022_areal_mshbm_subject.m",
        REPO_ROOT / "docs" / "hipp_parcellation_network" / "core_workflow_artifact_lineage.md",
    ]
    offenders = []
    for path in checked:
        if "route_a_outputs.mat" in path.read_text(encoding="utf-8"):
            offenders.append(path.relative_to(REPO_ROOT).as_posix())
    assert offenders == []

def test_jj_tms_default_routea_contract_is_text_labels_only() -> None:
    checked = [
        REPO_ROOT / "scripts" / "ArealMSHBM_Kong2022_Iowa_Stanford" / "submit_kong2022_areal_mshbm.sh",
        REPO_ROOT / "scripts" / "ArealMSHBM_Kong2022_Iowa_Stanford" / "run_kong2022_areal_mshbm_subject_sge.sh",
        REPO_ROOT / "scripts" / "ArealMSHBM_Kong2022_Iowa_Stanford" / "report_kong2022_areal_mshbm_status.sh",
        REPO_ROOT / "scripts" / "ArealMSHBM_Kong2022_Iowa_Stanford" / "README.md",
        REPO_ROOT / "docs" / "hipp_parcellation_network" / "kong2022_route_a_smoke.md",
    ]
    forbidden = [
        "create_kong2022_areal_outputs.py",
        "atlas-kong2022_areal_mshbm",
        "summary.json",
        "stage_manifest.json",
        "KONG2022_DLABEL",
    ]
    offenders = []
    for path in checked:
        text = path.read_text(encoding="utf-8")
        for token in forbidden:
            if token in text:
                offenders.append(f"{path.relative_to(REPO_ROOT).as_posix()} contains {token}")
    assert offenders == []

def test_jj_tms_subject_output_dir_contains_flat_text_labels_only() -> None:
    wrapper = REPO_ROOT / "scripts" / "ArealMSHBM_Kong2022_Iowa_Stanford" / "run_kong2022_areal_mshbm_subject_sge.sh"
    report = REPO_ROOT / "scripts" / "ArealMSHBM_Kong2022_Iowa_Stanford" / "report_kong2022_areal_mshbm_status.sh"
    readme = REPO_ROOT / "scripts" / "ArealMSHBM_Kong2022_Iowa_Stanford" / "README.md"
    matlab = REPO_ROOT / "scripts" / "ArealMSHBM_Kong2022_Iowa_Stanford" / "run_kong2022_areal_mshbm_subject.m"

    wrapper_text = wrapper.read_text(encoding="utf-8")
    assert "${subject_out}/lh_labels.txt" in wrapper_text
    assert "${subject_out}/rh_labels.txt" in wrapper_text
    assert "project_dir/route_a_outputs" not in wrapper_text
    assert "${subject_out}/inputs" not in wrapper_text
    assert "matlab_wrapper.log" not in wrapper_text

    for path in [report, readme, matlab]:
        text = path.read_text(encoding="utf-8")
        assert "project_dir/route_a_outputs" not in text
        assert "route_a_outputs" not in text

def test_jj_tms_temp_workdir_is_cleaned_with_tmp_prefix_guard() -> None:
    wrapper = REPO_ROOT / "scripts" / "ArealMSHBM_Kong2022_Iowa_Stanford" / "run_kong2022_areal_mshbm_subject_sge.sh"
    text = wrapper.read_text(encoding="utf-8")
    assert "cleanup_tmp_root()" in text
    assert "trap cleanup_tmp_root EXIT" in text
    assert "/tmp/jj_tms_kong2022_*" in text
    assert "rm -rf -- \"${tmp_root}\"" in text

def test_jj_tms_launcher_does_not_write_runtime_state_under_script_dir() -> None:
    submit = REPO_ROOT / "scripts" / "ArealMSHBM_Kong2022_Iowa_Stanford" / "submit_kong2022_areal_mshbm.sh"
    report = REPO_ROOT / "scripts" / "ArealMSHBM_Kong2022_Iowa_Stanford" / "report_kong2022_areal_mshbm_status.sh"
    task = REPO_ROOT / "scripts" / "ArealMSHBM_Kong2022_Iowa_Stanford" / "run_kong2022_areal_mshbm_subject_sge.sh"

    submit_text = submit.read_text(encoding="utf-8")
    assert "${script_dir}/manifests" not in submit_text
    assert "${script_dir}/logs" not in submit_text
    assert "${out_root}/_manifests" in submit_text
    assert "${out_root}/_sge_logs" in submit_text

    report_text = report.read_text(encoding="utf-8")
    assert "${script_dir}/logs" not in report_text
    assert "${out_root}/_sge_logs" in report_text

    task_text = task.read_text(encoding="utf-8")
    assert "${script_dir}/logs" not in task_text


def test_iowa_hcp500_argon_preset_contract() -> None:
    script_dir = REPO_ROOT / "scripts" / "ArealMSHBM_Kong2022_Iowa_Stanford"
    checked = [
        script_dir / "submit_kong2022_areal_mshbm.sh",
        script_dir / "run_kong2022_areal_mshbm_subject_sge.sh",
        script_dir / "report_kong2022_areal_mshbm_status.sh",
    ]
    required = [
        "iowa_hcp500",
        "/old_Users/yojiao/HCPpipeline500/work/qunex_iowa7t_hcppipeline500/sessions",
        "outputs_iowa_hcp500/inputs/iowa_hcp500/kong2022/cortex_references/routea_areal_mshbm",
    ]
    for path in checked:
        text = path.read_text(encoding="utf-8")
        for token in required:
            assert token in text, f"{path.name} missing {token}"

    task_text = checked[1].read_text(encoding="utf-8")
    assert "/tmp/iowa_hcp500_kong2022_*" in task_text
    assert 'KEEP_TMP=1: retaining ${tmp_root}' in task_text
    assert '-file-information "${dtseries}" -only-number-of-maps' in task_text
    assert 'if [ "${dataset}" != "iowa_hcp500" ]; then' in task_text

    submit_text = checked[0].read_text(encoding="utf-8")
    assert "default_ui_tc=3" in submit_text
    assert "default_pinc_tc=7" in submit_text


def test_iowa7t_argon_preset_contract_uses_subject_first_36p_layout() -> None:
    script_dir = REPO_ROOT / "scripts" / "ArealMSHBM_Kong2022_Iowa_Stanford"
    checked = {
        script_dir / "submit_kong2022_areal_mshbm.sh",
        script_dir / "run_kong2022_areal_mshbm_subject_sge.sh",
        script_dir / "report_kong2022_areal_mshbm_status.sh",
        script_dir / "README.md",
    }
    combined = "\n".join(path.read_text(encoding="utf-8") for path in checked)
    required = [
        "DATASET=iowa7t",
        "IOWA7T_BRANCH",
        "/Shared/jianglab/3_Data_Working/fmriprep_processed_Iowa7T/DataOrg_PFM_36P",
        "hipp_outputs",
        "subjects/${subject}/${iowa7t_branch}/cortex/areal_mshbm",
        "runs/logs",
        "provenance.json",
    ]
    forbidden = [
        "/Shared/jianglab/3_Data_Working/fmriprep_processed_Iowa7T/ArealMSHBM_Kong2022",
        "outputs_iowa/inputs/iowa",
        "outputs_iowa_hcp500/inputs/iowa_hcp500/kong2022/cortex_references/routea_areal_mshbm",
    ]
    for token in required:
        assert token in combined, f"missing {token}"

    iowa7t_text = "\n".join(
        line for line in combined.splitlines() if "iowa7t" in line.lower() or "Iowa7T" in line
    )
    for token in forbidden:
        assert token not in iowa7t_text, f"Iowa7T contract contains forbidden {token}"

    task_text = (script_dir / "run_kong2022_areal_mshbm_subject_sge.sh").read_text(
        encoding="utf-8"
    )
    assert '[ "${dataset}" != "iowa_hcp500" ] && [ "${dataset}" != "iowa7t" ]' in task_text
    assert 'Prepared Iowa7T ${iowa7t_branch} dtseries=' in task_text
    assert 'wb_command": "$(json_escape "${wb_command}")"' in task_text
