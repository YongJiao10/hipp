"""Task prep must not replace rest QC lists or the formal HCP151 surface."""
from __future__ import annotations

import importlib
import json
from pathlib import Path
import sys

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts/core_workflow"))
sys.path.insert(0, str(ROOT))
PREP = importlib.import_module("scripts.core_workflow.02_01_prepare_group_inputs")
QC = importlib.import_module("scripts.core_workflow.02_02_build_qc_manifest")
LATEST = "hcp_7t_hippocampus_fmri_qc_pass_t1only_latest.txt"
FORMAL = [str(subject) for subject in range(100001, 100152)]
OTHER_SUBJECTS = ["200001", "200002"]


@pytest.fixture
def roots(tmp_path, monkeypatch):
    rest = tmp_path / "rest"
    shared = rest / "surface"
    template = rest / "templates/hcp151"
    roster = tmp_path / "vertex_qc/manifests/hcp_7t_hippocampus_fmri_qc_pass_t1only_151.txt"
    roster.parent.mkdir(parents=True)
    roster.write_text("\n".join(FORMAL) + "\n")
    for module in (PREP, QC):
        monkeypatch.setattr(module, "DEFAULT_SHARED_ROOT", shared)
        monkeypatch.setattr(module, "FORMAL_SUBJECTS_MANIFEST", roster, raising=False)
    monkeypatch.setattr(PREP, "DEFAULT_ROUTE_ROOT", rest)
    monkeypatch.setattr(PREP, "DEFAULT_TEMPLATE_ROOT", template)
    monkeypatch.setattr(QC, "DEFAULT_OUT_DIR", rest / "qc")
    monkeypatch.setattr(QC, "DEFAULT_MANIFEST_DIR", rest / "manifests")
    latest = rest / "manifests" / LATEST
    latest.parent.mkdir(parents=True)
    latest.write_text(roster.read_text())
    return rest, shared, template, roster


def make_inputs(anatomy: Path, shared: Path, subjects: list[str]) -> None:
    """Tiny real surfaces and passing QC inputs, without external tools."""
    for index, subject in enumerate(subjects):
        for hemi in ("L", "R"):
            dice = anatomy / f"sub-{subject}/hippunfold/sub-{subject}/qc/sub-{subject}_hemi-{hemi}_desc-unetf3d_dice.tsv"
            dice.parent.mkdir(parents=True, exist_ok=True)
            dice.write_text("0.95\n")
            gate = shared / f"sub-{subject}/surface/tsnr_gate/sub-{subject}_hemi-{hemi}_tsnr_gate_summary.json"
            gate.parent.mkdir(parents=True, exist_ok=True)
            gate.write_text(json.dumps({"invalid_fraction": 0.0}))
            PREP.save_surface(
                PREP.surface_path(anatomy, subject, hemi),
                np.full((3, 3), index + 1, dtype=np.float32),
                np.array([[0, 1, 2]], dtype=np.int32),
            )


@pytest.mark.parametrize("task,dataset", [
    ("wm", "hcp"), ("emotion", "hcp"), ("relational", "hcp"),
    ("rest", "stanford"), ("rest", "iowa_jj_tms"),
])
@pytest.mark.parametrize("omitted", ["route-root", "shared-root"])
def test_task_qc_rejects_default_rest_destination_before_writing(tmp_path, monkeypatch, roots, task, dataset, omitted):
    rest, rest_shared, _, roster = roots
    route = tmp_path / "task"
    shared = tmp_path / "task/surface" if omitted != "shared-root" else rest_shared
    make_inputs(tmp_path / "anatomy", shared, OTHER_SUBJECTS)
    argv = ["prepare", "--task", task, "--dataset", dataset, "--stage", "qc", "--subjects", *OTHER_SUBJECTS,
            "--hippunfold-batch-root", str(tmp_path / "anatomy")]
    if omitted != "route-root":
        argv += ["--route-root", str(route)]
    if omitted != "shared-root":
        argv += ["--shared-root", str(shared)]
    monkeypatch.setattr(sys, "argv", argv)

    with pytest.raises(SystemExit) as exc:
        PREP.main()

    assert exc.value.code == 2
    assert (rest / "manifests" / LATEST).read_text() == roster.read_text()
    assert not (rest / "logs").exists()
    assert not (route / "logs").exists()


def test_task_group_average_rejects_formal_destination_before_writing(tmp_path, monkeypatch, roots):
    _, _, template, _ = roots
    subjects = OTHER_SUBJECTS
    route = tmp_path / "task"
    make_inputs(tmp_path / "anatomy", route / "surface", subjects)
    (route / "manifests").mkdir()
    (route / "manifests" / LATEST).write_text("\n".join(subjects) + "\n")
    monkeypatch.setattr(sys, "argv", [
        "prepare", "--task", "wm", "--stage", "group-average",
        "--hippunfold-batch-root", str(tmp_path / "anatomy"),
        "--route-root", str(route), "--shared-root", str(route / "surface"),
        "--subjects", *subjects,
    ])

    with pytest.raises(SystemExit) as exc:
        PREP.main()

    assert exc.value.code == 2
    assert not template.exists()
    assert not (route / "logs").exists()


@pytest.mark.parametrize("recorded", [False, True])
def test_formal_average_rejects_another_cohort_even_if_existing_roster_matches(tmp_path, roots, recorded):
    _, _, template, _ = roots
    subjects = OTHER_SUBJECTS
    make_inputs(tmp_path / "anatomy", tmp_path / "task/surface", subjects)
    out = template / "group_average_surfaces"
    sentinel = out / "group-average_hemi-L_space-corobl_den-512_label-hipp_midthickness.surf.gii"
    PREP.save_surface(sentinel, np.full((3, 3), 76, dtype=np.float32), np.array([[0, 1, 2]], dtype=np.int32))
    if recorded:
        (out / "stage_manifest.json").write_text(json.dumps({"subjects": subjects}))

    with pytest.raises(ValueError, match="HCP151"):
        PREP.build_group_average_surfaces(tmp_path / "anatomy", template, subjects)

    np.testing.assert_array_equal(PREP.load_surface(sentinel)[0], np.full((3, 3), 76))


def test_formal_average_checks_subject_identity_not_only_count(tmp_path, roots):
    _, _, template, _ = roots
    subjects = FORMAL[:-1] + ["199999"]
    make_inputs(tmp_path / "anatomy", tmp_path / "task/surface", subjects)
    with pytest.raises(ValueError, match="HCP151"):
        PREP.build_group_average_surfaces(tmp_path / "anatomy", template, subjects)
    assert not template.exists()


@pytest.mark.parametrize("formal", [False, True])
def test_average_builds_valid_formal_or_explicit_isolated_cohort(tmp_path, roots, formal):
    _, _, template, _ = roots
    subjects = FORMAL if formal else OTHER_SUBJECTS
    destination = template if formal else tmp_path / "task/template"
    make_inputs(tmp_path / "anatomy", tmp_path / "task/surface", subjects)

    PREP.build_group_average_surfaces(tmp_path / "anatomy", destination, subjects)

    out = destination / "group_average_surfaces"
    for hemi in ("L", "R"):
        coordinates, _ = PREP.load_surface(out / f"group-average_hemi-{hemi}_space-corobl_den-512_label-hipp_midthickness.surf.gii")
        np.testing.assert_array_equal(coordinates, np.full((3, 3), 76 if formal else 1.5))
    assert json.loads((out / "stage_manifest.json").read_text())["n_subjects"] == len(subjects)


@pytest.mark.parametrize("omitted", ["out-dir", "manifest-dir"])
def test_standalone_qc_rejects_task_input_with_rest_destination(tmp_path, monkeypatch, roots, omitted):
    rest, _, _, roster = roots
    shared = tmp_path / "task/surface"
    make_inputs(tmp_path / "anatomy", shared, OTHER_SUBJECTS)
    argv = ["qc", "--hippunfold-batch-root", str(tmp_path / "anatomy"),
            "--shared-root", str(shared), "--subjects", *OTHER_SUBJECTS]
    if omitted != "out-dir":
        argv += ["--out-dir", str(tmp_path / "task/qc")]
    if omitted != "manifest-dir":
        argv += ["--manifest-dir", str(tmp_path / "task/manifests")]
    monkeypatch.setattr(sys, "argv", argv)

    with pytest.raises(SystemExit) as exc:
        QC.main()

    assert exc.value.code == 2
    assert (rest / "manifests" / LATEST).read_text() == roster.read_text()
    assert not (rest / "qc").exists()


@pytest.mark.parametrize("case", [
    "incomplete_cohort", "wrong_identity", "qc_failure",
    "branch_neutral_wrong_identity",
])
def test_standalone_qc_preserves_formal_outputs_for_wrong_pass_cohort(tmp_path, monkeypatch, roots, capsys, case):
    rest, shared, _, roster = roots
    subjects = FORMAL[:-1] if case == "incomplete_cohort" else FORMAL.copy()
    if case.endswith("wrong_identity"):
        subjects[-1] = "199999"
    make_inputs(tmp_path / "anatomy", shared, subjects)
    if case == "qc_failure":
        gate = shared / f"sub-{subjects[-1]}/surface/tsnr_gate/sub-{subjects[-1]}_hemi-L_tsnr_gate_summary.json"
        gate.write_text(json.dumps({"invalid_fraction": 1.0}))
    out = rest / "qc"
    manifests = rest / "manifests"
    if case == "branch_neutral_wrong_identity":
        out = roster.parent.parent / "qc"
        manifests = roster.parent
    latest = manifests / LATEST
    latest.write_text("\n".join(subjects) + "\n" if case == "wrong_identity" else roster.read_text())
    report = out / "t1only_hippunfold_tsnr_qc.tsv"
    report.parent.mkdir(parents=True)
    report.write_text("Existing formal QC report\n")
    before_latest, before_roster = latest.read_text(), roster.read_text()
    monkeypatch.setattr(sys, "argv", [
        "qc", "--hippunfold-batch-root", str(tmp_path / "anatomy"),
        "--out-dir", str(out), "--manifest-dir", str(manifests),
        "--subjects", *subjects,
    ])

    with pytest.raises(SystemExit) as exc:
        QC.main()

    assert exc.value.code == 2
    error = capsys.readouterr().err
    assert "HCP151" in error and str(roster) in error
    assert latest.read_text() == before_latest
    assert roster.read_text() == before_roster
    assert report.read_text() == "Existing formal QC report\n"
    assert not (out / "t1only_hippunfold_tsnr_qc.json").exists()


@pytest.mark.parametrize("protected", ["qc", "manifest"])
def test_standalone_other_cohort_cannot_write_even_one_default_destination(tmp_path, monkeypatch, roots, protected):
    rest, shared, _, roster = roots
    make_inputs(tmp_path / "anatomy", shared, OTHER_SUBJECTS)
    out = rest / "qc" if protected == "qc" else tmp_path / "isolated/qc"
    manifests = rest / "manifests" if protected == "manifest" else tmp_path / "isolated/manifests"
    monkeypatch.setattr(sys, "argv", [
        "qc", "--hippunfold-batch-root", str(tmp_path / "anatomy"),
        "--out-dir", str(out), "--manifest-dir", str(manifests),
        "--subjects", *OTHER_SUBJECTS,
    ])

    with pytest.raises(SystemExit) as exc:
        QC.main()

    assert exc.value.code == 2
    assert (rest / "manifests" / LATEST).read_text() == roster.read_text()
    assert not (rest / "qc").exists()
    assert not (tmp_path / "isolated").exists()


def test_standalone_qc_filters_failed_subjects_from_the_complete_dataset(tmp_path, monkeypatch, roots):
    rest, shared, _, roster = roots
    subjects = FORMAL + ["199999"]
    make_inputs(tmp_path / "anatomy", shared, subjects)
    gate = shared / "sub-199999/surface/tsnr_gate/sub-199999_hemi-L_tsnr_gate_summary.json"
    gate.write_text(json.dumps({"invalid_fraction": 1.0}))
    monkeypatch.setattr(sys, "argv", [
        "qc", "--hippunfold-batch-root", str(tmp_path / "anatomy"),
    ])

    assert QC.main() == 0

    assert (rest / "manifests" / LATEST).read_text() == roster.read_text()
    result = json.loads((rest / "qc/t1only_hippunfold_tsnr_qc.json").read_text())
    assert result["n_subjects_total"] == 152
    assert result["n_subjects_pass"] == 151


def test_task_qc_updates_only_explicit_task_roots(tmp_path, monkeypatch, roots):
    rest, _, _, roster = roots
    route = tmp_path / "task"
    make_inputs(tmp_path / "anatomy", route / "surface", OTHER_SUBJECTS)
    monkeypatch.setattr(sys, "argv", [
        "prepare", "--task", "wm", "--stage", "qc",
        "--hippunfold-batch-root", str(tmp_path / "anatomy"),
        "--route-root", str(route), "--shared-root", str(route / "surface"),
        "--subjects", *OTHER_SUBJECTS,
    ])

    assert PREP.main() == 0
    assert (route / "manifests" / LATEST).read_text().splitlines() == OTHER_SUBJECTS
    assert (rest / "manifests" / LATEST).read_text() == roster.read_text()
    assert not (rest / "qc").exists()
