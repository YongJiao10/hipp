from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import nibabel as nib
import numpy as np
import pytest
from nibabel.cifti2 import BrainModelAxis, SeriesAxis


REPO_ROOT = Path(__file__).resolve().parents[2]
MODULE_PATH = (
    REPO_ROOT
    / "scripts"
    / "individual_workflow"
    / "jj_tms_hcp151_fixed_template_matching.py"
)


def load_module():
    assert MODULE_PATH.is_file(), "JJ_TMS fixed-template adapter is missing"
    spec = importlib.util.spec_from_file_location(
        "jj_tms_hcp151_fixed_template_matching",
        MODULE_PATH,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _write_dtseries(path: Path, n_timepoints: int, *, n_vertices: int = 4) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    left = BrainModelAxis.from_mask(
        np.ones(n_vertices, dtype=bool),
        name="CIFTI_STRUCTURE_CORTEX_LEFT",
    )
    right = BrainModelAxis.from_mask(
        np.ones(n_vertices, dtype=bool),
        name="CIFTI_STRUCTURE_CORTEX_RIGHT",
    )
    series = SeriesAxis(start=0.0, step=1.0, size=n_timepoints)
    data = np.arange(
        n_timepoints * (n_vertices * 2),
        dtype=np.float32,
    ).reshape(n_timepoints, n_vertices * 2)
    nib.save(
        nib.Cifti2Image(data, header=nib.cifti2.Cifti2Header.from_axes((series, left + right))),
        str(path),
    )


def _write_functional_inputs(
    root: Path,
    *,
    subject: str,
    run_lengths: list[int],
    affine: np.ndarray | None = None,
) -> tuple[Path, Path, Path]:
    volume_root = root / "volumes"
    dtseries_root = root / "dtseries"
    concat_root = root / "concat"
    affine_value = np.eye(4, dtype=np.float64) if affine is None else affine
    for run_id, n_timepoints in enumerate(run_lengths, start=1):
        volume_path = (
            volume_root
            / f"sub-{subject}"
            / "func"
            / (
                f"sub-{subject}_task-rest_run-{run_id:02d}_space-"
                "MNI152NLin6Asym_res-2_desc-denoised_bold.nii.gz"
            )
        )
        volume_path.parent.mkdir(parents=True, exist_ok=True)
        nib.save(
            nib.Nifti1Image(
                np.zeros((2, 3, 4, n_timepoints), dtype=np.float32),
                affine_value,
            ),
            str(volume_path),
        )
        dt_path = (
            dtseries_root
            / f"sub-{subject}"
            / "func"
            / (
                f"sub-{subject}_task-rest_run-{run_id:02d}_space-fsLR_"
                "den-91k_desc-denoised_bold.dtseries.nii"
            )
        )
        _write_dtseries(dt_path, n_timepoints)
    concat_path = (
        concat_root
        / f"sub-{subject}"
        / "func"
        / (
            f"sub-{subject}_task-rest_run-concat_space-fsLR_den-91k_"
            "desc-denoised_bold.dtseries.nii"
        )
    )
    _write_dtseries(concat_path, sum(run_lengths))
    return volume_root, dtseries_root, concat_root


def test_empty_failed_partial_is_preserved_before_exact_resume(tmp_path: Path):
    workflow = load_module()
    subject_dir = tmp_path / "subjects" / "sub-1001"
    subject_dir.mkdir(parents=True)
    source = tmp_path / "source.txt"
    source.write_text("source-v1", encoding="utf-8")
    parameters = {"threshold": 25.0}
    partial = subject_dir / "subject_manifest.partial.json"
    partial.write_text(
        json.dumps(
            {
                "status": "failed",
                "parameters": parameters,
                "inputs": {"source": workflow.fingerprint(source)},
                "retained_outputs": {},
                "error": "RuntimeError: parent missing",
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    original = partial.read_bytes()

    history = workflow.prepare_empty_failed_subject_resume(
        subject_dir=subject_dir,
        expected_inputs={"source": source},
        expected_parameters=parameters,
    )

    history_path = Path(history["path"])
    assert history_path.name == "subject_manifest.attempt-001.failed.json"
    assert history_path.read_bytes() == original
    assert partial.read_bytes() == original
    assert history["sha256"] == workflow.fingerprint(history_path)["sha256"]


@pytest.mark.parametrize("drift", ["input", "parameter", "retained", "extra_entry"])
def test_failed_partial_resume_rejects_any_nonempty_or_inexact_state(
    tmp_path: Path, drift: str
):
    workflow = load_module()
    subject_dir = tmp_path / "subjects" / "sub-1001"
    subject_dir.mkdir(parents=True)
    source = tmp_path / "source.txt"
    source.write_text("source-v1", encoding="utf-8")
    parameters = {"threshold": 25.0}
    payload = {
        "status": "failed",
        "parameters": parameters,
        "inputs": {"source": workflow.fingerprint(source)},
        "retained_outputs": {},
    }
    partial = subject_dir / "subject_manifest.partial.json"
    partial.write_text(json.dumps(payload), encoding="utf-8")
    expected_parameters = dict(parameters)
    if drift == "input":
        source.write_text("source-v2", encoding="utf-8")
    elif drift == "parameter":
        expected_parameters["threshold"] = 30.0
    elif drift == "retained":
        payload["retained_outputs"] = {"partial": "/tmp/partial.npy"}
        partial.write_text(json.dumps(payload), encoding="utf-8")
    else:
        (subject_dir / "unexpected.txt").write_text("unexpected", encoding="utf-8")

    with pytest.raises(ValueError, match="fingerprint|parameter|retained|only"):
        workflow.prepare_empty_failed_subject_resume(
            subject_dir=subject_dir,
            expected_inputs={"source": source},
            expected_parameters=expected_parameters,
        )

@pytest.mark.parametrize(
    ("subject", "run_lengths"),
    [("1001", [3, 4, 5]), ("1016", [6, 7])],
)
def test_resolve_and_validate_accepts_canonical_three_and_two_run_inputs(
    tmp_path: Path,
    subject: str,
    run_lengths: list[int],
):
    workflow = load_module()
    volume_root, dtseries_root, concat_root = _write_functional_inputs(
        tmp_path,
        subject=subject,
        run_lengths=run_lengths,
    )

    inputs = workflow.resolve_subject_runs(
        subject=subject,
        volume_root=volume_root,
        dtseries_root=dtseries_root,
        concat_dtseries_root=concat_root,
    )
    evidence = workflow.validate_functional_headers(inputs)

    assert [row.run_id for row in inputs.runs] == list(range(1, len(run_lengths) + 1))
    assert evidence["run_lengths"] == run_lengths
    assert evidence["total_timepoints"] == sum(run_lengths)


def test_run_resolution_uses_numeric_order_and_rejects_missing_pairs(tmp_path: Path):
    workflow = load_module()
    volume_root, dtseries_root, concat_root = _write_functional_inputs(
        tmp_path,
        subject="1001",
        run_lengths=[2, 2, 2],
    )
    func = volume_root / "sub-1001" / "func"
    run3 = next(func.glob("*run-03_*.nii.gz"))
    run10 = Path(str(run3).replace("run-03", "run-10"))
    run3.rename(run10)

    with pytest.raises(ValueError, match="run IDs"):
        workflow.resolve_subject_runs(
            subject="1001",
            volume_root=volume_root,
            dtseries_root=dtseries_root,
            concat_dtseries_root=concat_root,
        )


@pytest.mark.parametrize(
    "mutation",
    ["grid", "affine", "run_timepoints", "concat_timepoints"],
)
def test_functional_header_validation_fails_immediately_on_mismatch(
    tmp_path: Path,
    mutation: str,
):
    workflow = load_module()
    volume_root, dtseries_root, concat_root = _write_functional_inputs(
        tmp_path,
        subject="1001",
        run_lengths=[2, 3, 4],
    )
    if mutation in {"grid", "affine"}:
        path = sorted((volume_root / "sub-1001" / "func").glob("*.nii.gz"))[1]
        shape = (3, 3, 4, 3) if mutation == "grid" else (2, 3, 4, 3)
        affine = np.eye(4)
        if mutation == "affine":
            affine[0, 3] = 2.0
        nib.save(nib.Nifti1Image(np.zeros(shape, dtype=np.float32), affine), str(path))
    elif mutation == "run_timepoints":
        path = sorted((dtseries_root / "sub-1001" / "func").glob("*.dtseries.nii"))[1]
        _write_dtseries(path, 9)
    else:
        path = next((concat_root / "sub-1001" / "func").glob("*.dtseries.nii"))
        _write_dtseries(path, 8)
    inputs = workflow.resolve_subject_runs(
        subject="1001",
        volume_root=volume_root,
        dtseries_root=dtseries_root,
        concat_dtseries_root=concat_root,
    )

    with pytest.raises(ValueError, match="grid|affine|timepoint"):
        workflow.validate_functional_headers(inputs)


def test_sample_runs_creates_workbench_output_parent_before_call(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    workflow = load_module()
    volume_root, dtseries_root, concat_root = _write_functional_inputs(
        tmp_path, subject="1001", run_lengths=[2, 2, 2]
    )
    functional = workflow.resolve_subject_runs(
        subject="1001",
        volume_root=volume_root,
        dtseries_root=dtseries_root,
        concat_dtseries_root=concat_root,
    )
    observed: list[Path] = []

    def fake_workbench(_command: Path, args: list[object]) -> None:
        assert args[0] == "-volume-to-surface-mapping"
        partial = Path(args[3])
        observed.append(partial)
        assert partial.parent.is_dir()
        n_timepoints = int(nib.load(str(args[1])).shape[3])
        workflow._write_func_gifti_exclusive(
            partial, np.zeros((512, n_timepoints), dtype=np.float32)
        )

    monkeypatch.setattr(workflow, "_run_workbench", fake_workbench)
    outputs: dict[str, Path] = {}
    raw = workflow._sample_runs(
        functional=functional,
        surfaces={
            "L": {"consumed_asset": tmp_path / "hemi-L.surf.gii"},
            "R": {"consumed_asset": tmp_path / "hemi-R.surf.gii"},
        },
        subject_dir=tmp_path / "subject-output",
        wb_command=tmp_path / "wb_command",
        outputs=outputs,
    )

    assert len(observed) == 6
    assert raw["L"].shape == (512, 6)
    assert raw["R"].shape == (512, 6)


def test_areal_labels_require_32k_lengths_and_fixed_global_roi_ids(tmp_path: Path):
    workflow = load_module()
    label_root = tmp_path / "labels"
    subject_dir = label_root / "sub-1001"
    subject_dir.mkdir(parents=True)
    np.savetxt(subject_dir / "lh_labels.txt", np.ones(32492, dtype=np.int32), fmt="%d")
    np.savetxt(subject_dir / "rh_labels.txt", np.full(32492, 201, dtype=np.int32), fmt="%d")

    labels = workflow.load_areal_labels(label_root=label_root, subject="1001")
    assert labels["L"].shape == (32492,)
    assert labels["R"].shape == (32492,)

    np.savetxt(subject_dir / "lh_labels.txt", np.ones(32491, dtype=np.int32), fmt="%d")
    with pytest.raises(ValueError, match="32492"):
        workflow.load_areal_labels(label_root=label_root, subject="1001")


def test_roi_registry_rejects_any_order_other_than_one_through_400(tmp_path: Path):
    workflow = load_module()
    registry = tmp_path / "registry.tsv"
    registry.write_text(
        "roi_id\troi_name\n"
        + "".join(f"{roi_id}\tROI_{roi_id}\n" for roi_id in range(1, 401)),
        encoding="utf-8",
    )
    assert [row["roi_id"] for row in workflow.read_roi_registry(registry)] == list(
        range(1, 401)
    )
    rows = registry.read_text(encoding="utf-8").splitlines()
    rows[1], rows[2] = rows[2], rows[1]
    registry.write_text("\n".join(rows) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="order"):
        workflow.read_roi_registry(registry)


def test_fixed_roi_timeseries_preserves_missing_columns_as_nan():
    workflow = load_module()
    cortex = {
        "L": np.asarray([[1.0, 2.0, 3.0], [3.0, 4.0, 5.0]], dtype=np.float32),
        "R": np.asarray([[5.0, 6.0, 7.0], [7.0, 8.0, 9.0]], dtype=np.float32),
    }
    aligned_labels = {
        "L": np.asarray([1, 1], dtype=np.int32),
        "R": np.asarray([400, 400], dtype=np.int32),
    }

    timeseries, counts = workflow.extract_fixed_roi_timeseries(
        cortex_by_hemi=cortex,
        aligned_labels_by_hemi=aligned_labels,
        roi_ids=list(range(1, 401)),
    )

    assert timeseries.shape == (400, 3)
    assert timeseries.dtype == np.float32
    assert np.allclose(timeseries[0], [2.0, 3.0, 4.0])
    assert np.isnan(timeseries[1:399]).all()
    assert np.allclose(timeseries[399], [6.0, 7.0, 8.0])
    assert counts[0] == 2 and counts[399] == 2


def test_hippocampal_tsnr_uses_10000_over_sample_std_and_threshold_25():
    workflow = load_module()
    timeseries = np.asarray(
        [
            [10000.0, 10020.0, 9980.0],
            [10000.0, 10500.0, 9500.0],
            [10000.0, 10000.0, 10000.0],
        ],
        dtype=np.float32,
    )

    tsnr, valid = workflow.compute_tsnr(timeseries, threshold=25.0)

    expected = 10000.0 / np.std(timeseries[:2], axis=1, ddof=1)
    assert np.allclose(tsnr[:2], expected.astype(np.float32))
    assert valid.tolist() == [True, False, False]
    assert np.isnan(tsnr[2])


def test_combo_validation_requires_nonzero_valid_labels_and_all_three_codes():
    workflow = load_module()
    valid = np.zeros(512, dtype=bool)
    valid[:3] = True
    scores = np.full((512, 3), np.nan, dtype=np.float32)
    scores[:3] = np.asarray(
        [[0.9, 0.1, -0.2], [0.1, 0.8, 0.2], [-0.1, 0.3, 0.7]],
        dtype=np.float32,
    )
    margin = np.full(512, np.nan, dtype=np.float32)
    margin[:3] = [0.8, 0.6, 0.4]
    labels = np.zeros(512, dtype=np.int32)
    labels[:3] = [1, 2, 3]
    workflow.validate_combo_arrays(
        labels=labels,
        scores=scores,
        margin=margin,
        valid_mask=valid,
    )

    missing_valid_label = labels.copy()
    missing_valid_label[1] = 0
    with pytest.raises(ValueError, match="nonzero"):
        workflow.validate_combo_arrays(
            labels=missing_valid_label,
            scores=scores,
            margin=margin,
            valid_mask=valid,
        )
    missing_cluster = labels.copy()
    missing_cluster[1] = 1
    with pytest.raises(ValueError, match="three"):
        workflow.validate_combo_arrays(
            labels=missing_cluster,
            scores=scores,
            margin=margin,
            valid_mask=valid,
        )


@pytest.mark.parametrize("drift", ["input", "parameter", "shape", "dtype"])
def test_resume_reuses_only_exact_fingerprints_shapes_dtypes_and_parameters(
    tmp_path: Path,
    drift: str,
):
    workflow = load_module()
    subject_dir = tmp_path / "subjects" / "sub-1001"
    subject_dir.mkdir(parents=True)
    source = tmp_path / "source.txt"
    source.write_text("source-v1", encoding="utf-8")
    output = subject_dir / "labels.npy"
    np.save(output, np.asarray([1, 2, 3], dtype=np.int32))
    parameters = {"threshold": 25.0, "smoothings": ["2mm", "4mm"]}
    manifest = workflow.build_subject_manifest(
        subject="1001",
        inputs={"source": source},
        parameters=parameters,
        outputs={"labels": output},
        combinations=[],
    )
    (subject_dir / "subject_manifest.json").write_text(
        json.dumps(manifest),
        encoding="utf-8",
    )
    exact = workflow.validate_reusable_subject(
        subject_dir=subject_dir,
        expected_inputs={"source": source},
        expected_parameters=parameters,
    )
    assert exact["status"] == "ok"
    expected_parameters = dict(parameters)
    if drift == "input":
        source.write_text("source-v2", encoding="utf-8")
    elif drift == "parameter":
        expected_parameters["threshold"] = 30.0
    elif drift == "shape":
        np.save(output, np.asarray([1, 2], dtype=np.int32))
    else:
        np.save(output, np.asarray([1, 2, 3], dtype=np.int16))

    if drift:
        with pytest.raises(ValueError, match="fingerprint|parameter|shape|dtype"):
            workflow.validate_reusable_subject(
                subject_dir=subject_dir,
                expected_inputs={"source": source},
                expected_parameters=expected_parameters,
            )


@pytest.mark.parametrize(
    "field", ["subjects", "path_roots", "template_manifest", "parameters"]
)
def test_run_resume_contract_rejects_immutable_context_drift(field: str):
    workflow = load_module()
    expected = {
        "workflow": "jj_tms_hcp151_fixed_template_matching",
        "run_id": "smoke",
        "scope": "subset",
        "subjects": ["sub-1001"],
        "subject_count": 1,
        "canonical_roster_count": 70,
        "template_manifest": {"sha256": "template"},
        "roi_registry": {"sha256": "registry"},
        "parameters": {"tsnr_threshold": 25.0},
        "path_roots": {"out_root": "/tmp/out"},
    }
    workflow.validate_run_resume_contract(dict(expected), expected)
    recorded = json.loads(json.dumps(expected))
    if field == "subjects":
        recorded[field] = ["sub-1002"]
    else:
        recorded[field] = {"drift": True}
    with pytest.raises(ValueError, match="resume contract mismatch"):
        workflow.validate_run_resume_contract(recorded, expected)


def test_cli_is_fixed_to_three_modes_and_has_no_overwrite_option():
    workflow = load_module()
    parser = workflow.build_arg_parser()
    args = parser.parse_args(
        ["--mode", "preflight", "--subjects", "1001", "1002", "--jobs", "2"]
    )
    assert args.mode == "preflight"
    assert args.jobs == 2
    assert not hasattr(args, "overwrite")
    with pytest.raises(SystemExit):
        parser.parse_args(["--mode", "run", "--overwrite"])
