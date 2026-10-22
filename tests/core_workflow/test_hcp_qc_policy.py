"""Accepted HCP QC gates use physical-run FD and actual den-512 GM coverage."""
from __future__ import annotations

import csv
import importlib
import json
from pathlib import Path
import sys

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts/core_workflow"))
QC = importlib.import_module("scripts.core_workflow.02_02_build_qc_manifest")


def prepare(tmp_path, monkeypatch, subjects=("100610",), *, increments=(0.8, 0.1)):
    batch, shared = tmp_path / "anatomy", tmp_path / "surface"
    motion_rows = []
    for subject in subjects:
        for hemi in ("L", "R"):
            path = batch / f"sub-{subject}/hippunfold/sub-{subject}/qc/sub-{subject}_hemi-{hemi}_desc-unetf3d_dice.tsv"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("0.70\n")
        for run, increment in enumerate(increments, 1):
            path = tmp_path / f"motion/{subject}/run-{run}/Movement_Regressors.txt"
            path.parent.mkdir(parents=True, exist_ok=True)
            data = np.zeros((2, 12))
            data[:, 0] = [100 * run, 100 * run + increment]
            np.savetxt(path, data)
            motion_rows.append({"subject": subject, "run_label": QC_RUN_LABELS[run], "filename": path.name, "lss_path": str(path)})
        coverage = shared / f"sub-{subject}/surface/raw/sub-{subject}_coverage_summary.json"
        coverage.parent.mkdir(parents=True, exist_ok=True)
        coverage.write_text(json.dumps({
            "subject": f"sub-{subject}",
            "coverage_policy": "cifti_hippocampal_common_run_gm_trilinear",
            "sampling_space": "MNI152NLin6Asym",
            "run_ids": list(range(1, len(increments) + 1)),
            "run_labels": [QC_RUN_LABELS[run] for run in range(1, len(increments) + 1)],
            "run_lengths": [2] * len(increments),
            "density_evidence": {"density": "512", "n_vertices": {"L": 512, "R": 512}},
            "hemispheres": {h: {"n_vertices_total": 512, "n_vertices_unsupported": 20} for h in ("L", "R")},
        }))
    motion = tmp_path / "motion.tsv"
    with motion.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=("subject", "run_label", "filename", "lss_path"), delimiter="\t")
        writer.writeheader()
        writer.writerows(motion_rows)
    out, manifests = tmp_path / "qc", tmp_path / "manifests"
    monkeypatch.setattr(QC, "DEFAULT_SHARED_ROOT", shared)
    monkeypatch.setattr(QC, "DEFAULT_OUT_DIR", out)
    monkeypatch.setattr(QC, "DEFAULT_MANIFEST_DIR", manifests)
    monkeypatch.setattr(sys, "argv", ["qc", "--hippunfold-batch-root", str(batch), "--shared-root", str(shared),
        "--motion-manifest", str(motion), "--out-dir", str(out), "--manifest-dir", str(manifests), "--subjects", *subjects])
    return batch, shared, motion, out, manifests


QC_RUN_LABELS = {1: "rfMRI_REST1_7T_PA", 2: "rfMRI_REST2_7T_AP", 3: "rfMRI_REST3_7T_PA", 4: "rfMRI_REST4_7T_AP"}


def result(out):
    return json.loads((out / "t1only_hippunfold_tsnr_qc.json").read_text())


def test_inclusive_bilateral_gates_and_physical_run_boundaries(tmp_path, monkeypatch):
    _, _, _, out, manifests = prepare(tmp_path, monkeypatch)
    assert QC.main() == 0
    payload = result(out)
    row = payload["rows"][0]
    assert float(row["overall_mean_power_fd_mm"]) == pytest.approx(0.225)
    assert row["qc_status"] == "pass"
    assert payload["n_subjects_pass"] == 1
    assert (manifests / "hcp_7t_hippocampus_fmri_qc_pass_t1only_latest.txt").read_text() == "100610\n"
    assert "tsnr_invalid_threshold" not in payload


def test_power_fd_uses_degrees_and_frame_weighting(tmp_path, monkeypatch):
    _, _, motion, out, _ = prepare(tmp_path, monkeypatch)
    rows = list(csv.DictReader(motion.open(), delimiter="\t"))
    first = np.zeros((2, 12)); first[1, 3] = 180 / np.pi / 50
    np.savetxt(rows[0]["lss_path"], first)
    second = np.zeros((4, 12)); second[:, 0] = [20, 20.1, 20.2, 20.3]
    np.savetxt(rows[1]["lss_path"], second)
    coverage = tmp_path / "surface/sub-100610/surface/raw/sub-100610_coverage_summary.json"
    payload = json.loads(coverage.read_text()); payload["run_lengths"] = [2, 4]
    coverage.write_text(json.dumps(payload))
    assert QC.main() == 0
    assert float(result(out)["rows"][0]["overall_mean_power_fd_mm"]) == pytest.approx(1.3 / 6)


def test_fd_threshold_is_inclusive_and_tsnr_is_not_a_gate(tmp_path, monkeypatch):
    _, shared, _, out, _ = prepare(tmp_path, monkeypatch, increments=(0.6,))
    for hemi in ("L", "R"):
        gate = shared / f"sub-100610/surface/tsnr_gate/sub-100610_hemi-{hemi}_tsnr_gate_summary.json"
        gate.parent.mkdir(parents=True, exist_ok=True)
        gate.write_text(json.dumps({"invalid_fraction": 1.0}))
    assert QC.main() == 0
    row = result(out)["rows"][0]
    assert float(row["overall_mean_power_fd_mm"]) == pytest.approx(0.30)
    assert row["qc_status"] == "pass"


@pytest.mark.parametrize("missing", ["motion", "dice"])
def test_missing_motion_or_anatomy_is_unassessable(tmp_path, monkeypatch, missing):
    _, _, motion, out, _ = prepare(tmp_path, monkeypatch)
    if missing == "motion":
        rows = list(csv.DictReader(motion.open(), delimiter="\t"))
        rows[0]["lss_path"] = str(tmp_path / "absent/Movement_Regressors.txt")
        with motion.open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=rows[0].keys(), delimiter="\t")
            writer.writeheader(); writer.writerows(rows)
    else:
        argv = sys.argv.copy()
        argv[argv.index("--hippunfold-batch-root") + 1] = str(tmp_path / "absent_anatomy")
        monkeypatch.setattr(sys, "argv", argv)
    assert QC.main() == 0
    row = result(out)["rows"][0]
    assert row["qc_status"] == "unassessable"
    assert row["exclude_reason"] == ""
    assert f"missing_{missing}" in row["unassessable_reason"]


@pytest.mark.parametrize("criterion", ["motion", "dice", "coverage"])
def test_quality_failures_are_sequential_and_bilateral(tmp_path, monkeypatch, criterion):
    batch, shared, _, out, _ = prepare(tmp_path, monkeypatch, increments=(1.6, 0.1) if criterion == "motion" else (0.1, 0.1))
    if criterion == "dice":
        (batch / "sub-100610/hippunfold/sub-100610/qc/sub-100610_hemi-R_desc-unetf3d_dice.tsv").write_text("0.69\n")
    if criterion == "coverage":
        path = shared / "sub-100610/surface/raw/sub-100610_coverage_summary.json"
        data = json.loads(path.read_text()); data["hemispheres"]["R"]["n_vertices_unsupported"] = 21
        path.write_text(json.dumps(data))
    assert QC.main() == 0
    row = result(out)["rows"][0]
    assert row["qc_status"] == "fail"
    assert row["exclude_reason"] == {"motion": "motion_gt_threshold", "dice": "dice_lt_threshold", "coverage": "unsupported_vertices_gt_threshold"}[criterion]


def test_missing_coverage_is_unassessable_without_quality_failure(tmp_path, monkeypatch):
    _, shared, _, out, _ = prepare(tmp_path, monkeypatch)
    # A second subject with no coverage declaration exercises absence without deleting fixtures.
    monkeypatch.setattr(QC, "coverage_summary_path", lambda root, subject: shared / "missing_coverage.json")
    assert QC.main() == 0
    payload = result(out); row = payload["rows"][0]
    assert row["qc_status"] == "unassessable"
    assert row["exclude_reason"] == ""
    assert "missing_coverage" in row["unassessable_reason"]
    assert payload["n_subjects_fail"] == 0


@pytest.mark.parametrize("mismatch", ["density", "run_lengths", "run_labels", "duplicate_motion"])
def test_mismatched_declared_inputs_fail_before_output(tmp_path, monkeypatch, mismatch):
    _, shared, motion, out, _ = prepare(tmp_path, monkeypatch)
    path = shared / "sub-100610/surface/raw/sub-100610_coverage_summary.json"
    data = json.loads(path.read_text())
    if mismatch == "density":
        data["density_evidence"]["n_vertices"]["R"] = 1024
    elif mismatch == "run_lengths":
        data["run_lengths"] = [4]
    elif mismatch == "run_labels":
        data["run_labels"] = ["rfMRI_REST2_7T_AP", "rfMRI_REST1_7T_PA"]
    else:
        lines = motion.read_text().splitlines(); motion.write_text("\n".join(lines + [lines[1]]) + "\n")
    path.write_text(json.dumps(data))
    with pytest.raises((ValueError, SystemExit), match="density|run|Duplicate"):
        QC.main()
    assert not out.exists()
