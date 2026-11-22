"""Task prep must not replace rest QC lists or the formal HCP surface."""
from __future__ import annotations

import importlib
import csv
import json
from pathlib import Path
import sys

import nibabel as nib
import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts/core_workflow"))
sys.path.insert(0, str(ROOT))
PREP = importlib.import_module("scripts.core_workflow.02_01_prepare_group_inputs")
QC = importlib.import_module("scripts.core_workflow.02_02_build_qc_manifest")
LATEST = "hcp_7t_hippocampus_fmri_qc_pass_t1only_latest.txt"
FORMAL = ["100001", "100002", "100003"]
OTHER_SUBJECTS = ["200001", "200002"]


@pytest.fixture
def roots(tmp_path, monkeypatch):
    rest = tmp_path / "rest"
    shared = rest / "surface"
    template = rest / "templates/hcp"
    roster = tmp_path / "vertex_qc/manifests/hcp_7t_hippocampus_fmri_qc_pass_t1only_latest.txt"
    roster.parent.mkdir(parents=True)
    roster.write_text("\n".join(FORMAL) + "\n")
    for module in (PREP, QC):
        monkeypatch.setattr(module, "DEFAULT_SHARED_ROOT", shared)
        monkeypatch.setattr(module, "FORMAL_SUBJECTS_MANIFEST", roster, raising=False)
    monkeypatch.setattr(PREP, "DEFAULT_ROUTE_ROOT", rest)
    monkeypatch.setattr(PREP, "DEFAULT_TEMPLATE_ROOT", template)
    monkeypatch.setattr(QC, "DEFAULT_OUT_DIR", rest / "qc")
    monkeypatch.setattr(QC, "DEFAULT_MANIFEST_DIR", rest / "manifests")
    monkeypatch.setattr(QC, "DEFAULT_MOTION_MANIFEST", tmp_path / "motion.tsv")
    latest = rest / "manifests" / LATEST
    latest.parent.mkdir(parents=True)
    latest.write_text(roster.read_text())
    return rest, shared, template, roster


def make_inputs(anatomy: Path, shared: Path, subjects: list[str]) -> None:
    """Tiny real surfaces and passing QC inputs, without external tools."""
    for index, subject in enumerate(subjects):
        raw = shared / f"sub-{subject}/surface/raw"
        raw.mkdir(parents=True, exist_ok=True)
        (raw / f"sub-{subject}_coverage_summary.json").write_text(json.dumps({
            "subject": f"sub-{subject}",
            "coverage_policy": "bold_nifti_runwise_trilinear_validity",
            "sampling_space": "MNI152NLin6Asym", "run_ids": [1], "run_lengths": [2], "run_labels": ["rfMRI_REST1_7T_PA"],
            "density_evidence": {"density": "512", "n_vertices": {"L": 512, "R": 512}},
            "hemispheres": {h: {"n_vertices_total": 512, "n_vertices_unsupported": 0,
                "nifti_support_path": str(raw / f"sub-{subject}_hemi-{h}_nifti_support.npy")}
                for h in ("L", "R")},
        }))
        for h in ("L", "R"):
            np.save(raw / f"sub-{subject}_hemi-{h}_nifti_support.npy", np.ones(512, dtype=np.float32))
            nib.save(PREP.make_func_gifti(np.tile([0., 1.], (512, 1))),
                raw / f"sub-{subject}_hemi-{h}_space-T1w_den-512_label-hipp_bold.func.gii")
        motion = anatomy.parent / f"motion/sub-{subject}/Movement_Regressors.txt"
        motion.parent.mkdir(parents=True, exist_ok=True)
        np.savetxt(motion, np.zeros((2, 12)))
        for hemi in ("L", "R"):
            dice = anatomy / f"sub-{subject}/hippunfold/sub-{subject}/qc/sub-{subject}_hemi-{hemi}_desc-unetf3d_dice.tsv"
            dice.parent.mkdir(parents=True, exist_ok=True)
            dice.write_text("0.95\n")
            gate = shared / f"sub-{subject}/surface/validity/sub-{subject}_hemi-{hemi}_validity_summary.json"
            gate.parent.mkdir(parents=True, exist_ok=True)
            gate.write_text(json.dumps({"invalid_fraction": 0.0}))
            PREP.save_surface(
                PREP.surface_path(anatomy, subject, hemi),
                np.full((3, 3), index + 1, dtype=np.float32),
                np.array([[0, 1, 2]], dtype=np.int32),
            )
            PREP.save_surface(
                PREP.surface_path(anatomy, subject, hemi, space='corobl'),
                np.full((3, 3), index + 1, dtype=np.float32),
                np.array([[0, 1, 2]], dtype=np.int32),
            )
            # HippUnfold's unfolded sheet and subfield labels are the same for every subject.
            PREP.save_surface(
                PREP.surface_path(anatomy, subject, hemi, space='unfold'),
                np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=np.float32),
                np.array([[0, 1, 2]], dtype=np.int32),
            )
            nib.save(nib.gifti.GiftiImage(darrays=[nib.gifti.GiftiDataArray(
                np.array([1, 1, 2], dtype=np.int32), intent="NIFTI_INTENT_LABEL", datatype="NIFTI_TYPE_INT32")]),
                str(PREP.surface_path(anatomy, subject, hemi).parent
                    / f"sub-{subject}_hemi-{hemi}_den-512_label-hipp_atlas-multihist7sgi_subfields.label.gii"))
        PREP.write_hipp_spec(
            PREP.surface_path(anatomy, subject, "L").parent / f"sub-{subject}_den-512_label-hipp_surfaces.spec",
            [(h, kind, name) for h in ("L", "R") for kind, name in (
                ("SURFACE", f"sub-{subject}_hemi-{h}_space-unfold_den-512_label-hipp_midthickness.surf.gii"),
                ("LABEL", f"sub-{subject}_hemi-{h}_den-512_label-hipp_atlas-multihist7sgi_subfields.label.gii"))],
        )
    with (anatomy.parent / "motion.tsv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=("subject", "run_label", "filename", "lss_path"), delimiter="\t")
        writer.writeheader()
        writer.writerows({"subject": s, "run_label": "rfMRI_REST1_7T_PA", "filename": "Movement_Regressors.txt",
            "lss_path": str(anatomy.parent / f"motion/sub-{s}/Movement_Regressors.txt")} for s in subjects)


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
def test_formal_average_uses_current_qc_cohort_instead_of_historical_roster(tmp_path, roots, recorded):
    _, _, template, _ = roots
    subjects = OTHER_SUBJECTS
    make_inputs(tmp_path / "anatomy", tmp_path / "task/surface", subjects)
    out = template / "group_average_surfaces"
    sentinel = out / "group-average_hemi-L_space-corobl_den-512_label-hipp_midthickness.surf.gii"
    PREP.save_surface(sentinel, np.full((3, 3), 76, dtype=np.float32), np.array([[0, 1, 2]], dtype=np.int32))
    if recorded:
        (out / "stage_manifest.json").write_text(json.dumps({"subjects": subjects}))

    PREP.build_group_average_surfaces(tmp_path / "anatomy", template, subjects)

    np.testing.assert_array_equal(PREP.load_surface(sentinel)[0], np.full((3, 3), 1.5))
    stage = json.loads((out / "stage_manifest.json").read_text())
    assert stage["subjects"] == [f"sub-{subject}" for subject in subjects]


def test_formal_average_records_the_selected_qc_subjects_without_a_fixed_count(tmp_path, roots):
    _, _, template, _ = roots
    subjects = FORMAL[:-1] + ["199999"]
    make_inputs(tmp_path / "anatomy", tmp_path / "task/surface", subjects)
    PREP.build_group_average_surfaces(tmp_path / "anatomy", template, subjects)
    stage = json.loads((template / "group_average_surfaces/stage_manifest.json").read_text())
    assert stage["n_subjects"] == len(subjects)
    assert stage["subjects"][-1] == "sub-199999"


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
        np.testing.assert_array_equal(coordinates, np.full((3, 3), 2 if formal else 1.5))
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
    "smaller_roster", "changed_identity", "qc_failure",
    "branch_neutral_changed_identity",
])
def test_standalone_qc_uses_measured_passes_without_fixed_count_or_ids(tmp_path, monkeypatch, roots, case):
    rest, shared, _, roster = roots
    subjects = FORMAL[:-1] if case == "smaller_roster" else FORMAL.copy()
    if case.endswith("changed_identity"):
        subjects[-1] = "199999"
    make_inputs(tmp_path / "anatomy", shared, subjects)
    if case == "qc_failure":
        dice = tmp_path / f"anatomy/sub-{subjects[-1]}/hippunfold/sub-{subjects[-1]}/qc/sub-{subjects[-1]}_hemi-L_desc-unetf3d_dice.tsv"
        dice.write_text("0.69\n")
    out = rest / "qc"
    manifests = rest / "manifests"
    if case == "branch_neutral_changed_identity":
        out = roster.parent.parent / "qc"
        manifests = roster.parent
    latest = manifests / LATEST
    latest.write_text(roster.read_text())
    report = out / "t1only_hippunfold_qc.tsv"
    report.parent.mkdir(parents=True)
    report.write_text("Existing formal QC report\n")
    before_roster = roster.read_text()
    monkeypatch.setattr(sys, "argv", [
        "qc", "--hippunfold-batch-root", str(tmp_path / "anatomy"),
        "--out-dir", str(out), "--manifest-dir", str(manifests),
        "--subjects", *subjects,
    ])

    assert QC.main() == 0
    expected = sorted(subjects[:-1] if case == "qc_failure" else subjects)
    assert latest.read_text().splitlines() == expected
    if manifests != roster.parent or len(expected) != len(FORMAL):
        assert roster.read_text() == before_roster
    result = json.loads((out / "t1only_hippunfold_qc.json").read_text())
    assert result["n_subjects_pass"] == len(expected)


@pytest.mark.parametrize("protected", ["qc", "manifest"])
def test_standalone_qc_accepts_current_input_roster_in_default_destinations(tmp_path, monkeypatch, roots, protected):
    rest, shared, _, roster = roots
    make_inputs(tmp_path / "anatomy", shared, OTHER_SUBJECTS)
    out = rest / "qc" if protected == "qc" else tmp_path / "isolated/qc"
    manifests = rest / "manifests" if protected == "manifest" else tmp_path / "isolated/manifests"
    monkeypatch.setattr(sys, "argv", [
        "qc", "--hippunfold-batch-root", str(tmp_path / "anatomy"),
        "--out-dir", str(out), "--manifest-dir", str(manifests),
        "--subjects", *OTHER_SUBJECTS,
    ])

    assert QC.main() == 0
    assert (manifests / LATEST).read_text().splitlines() == OTHER_SUBJECTS
    assert json.loads((out / "t1only_hippunfold_qc.json").read_text())["n_subjects_pass"] == len(OTHER_SUBJECTS)
    assert roster.read_text().splitlines() == FORMAL


def test_standalone_qc_filters_failed_subjects_from_the_complete_dataset(tmp_path, monkeypatch, roots):
    rest, shared, _, roster = roots
    subjects = FORMAL + ["199999"]
    make_inputs(tmp_path / "anatomy", shared, subjects)
    coverage = shared / "sub-199999/surface/raw/sub-199999_coverage_summary.json"
    support = np.ones(512, dtype=np.float32); support[:21] = 0
    np.save(coverage.parent / 'sub-199999_hemi-L_nifti_support.npy', support)
    monkeypatch.setattr(sys, "argv", [
        "qc", "--hippunfold-batch-root", str(tmp_path / "anatomy"),
    ])

    assert QC.main() == 0

    assert (rest / "manifests" / LATEST).read_text() == roster.read_text()
    result = json.loads((rest / "qc/t1only_hippunfold_qc.json").read_text())
    assert result["n_subjects_total"] == len(subjects)
    assert result["n_subjects_pass"] == len(FORMAL)


def test_task_qc_requires_explicit_run_inputs_before_writing(tmp_path, monkeypatch, roots, capsys):
    rest, _, _, roster = roots
    route = tmp_path / "task"
    make_inputs(tmp_path / "anatomy", route / "surface", OTHER_SUBJECTS)
    monkeypatch.setattr(sys, "argv", [
        "prepare", "--task", "wm", "--stage", "qc",
        "--hippunfold-batch-root", str(tmp_path / "anatomy"),
        "--route-root", str(route), "--shared-root", str(route / "surface"),
        "--subjects", *OTHER_SUBJECTS,
    ])

    with pytest.raises(SystemExit) as exc:
        PREP.main()
    assert exc.value.code == 2
    error = capsys.readouterr().err
    assert '--qc-run-inputs' in error and 'dataset/task' in error
    assert not (route / "manifests").exists()
    assert (rest / "manifests" / LATEST).read_text() == roster.read_text()
    assert not (rest / "qc").exists()


def test_hcp_rest_qc_updates_explicit_isolated_roots(tmp_path, monkeypatch, roots):
    rest, _, _, roster = roots
    route = tmp_path / "isolated_rest"
    make_inputs(tmp_path / "anatomy", route / "surface", OTHER_SUBJECTS)
    monkeypatch.setattr(sys, "argv", [
        "prepare", "--task", "rest", "--dataset", "hcp", "--stage", "qc", "--jobs", "1",
        "--hippunfold-batch-root", str(tmp_path / "anatomy"),
        "--route-root", str(route), "--shared-root", str(route / "surface"),
        "--motion-manifest", str(tmp_path / "motion.tsv"),
        "--subjects", *OTHER_SUBJECTS,
    ])
    assert PREP.main() == 0
    assert (route / "manifests" / LATEST).read_text().splitlines() == OTHER_SUBJECTS
    assert (rest / "manifests" / LATEST).read_text() == roster.read_text()
    assert not (rest / "qc").exists()
