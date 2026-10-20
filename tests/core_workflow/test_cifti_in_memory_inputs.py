#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import importlib
import json
import subprocess
import sys
from pathlib import Path

import nibabel as nib
import numpy as np
import pytest
from nibabel.cifti2 import BrainModelAxis, Cifti2Image, SeriesAxis


REPO_ROOT = Path(__file__).resolve().parents[2]
CORE_DIR = REPO_ROOT / "scripts" / "core_workflow"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(CORE_DIR) not in sys.path:
    sys.path.insert(0, str(CORE_DIR))


io_utils = importlib.import_module("scripts.core_workflow.lib.io_utils")
surface_sampling = importlib.import_module("scripts.core_workflow.lib.surface_sampling")
prepare_inputs_mod = importlib.import_module("scripts.core_workflow.02_01_prepare_group_inputs")
routea_batch = importlib.import_module("scripts.core_workflow.03_02_build_kong2022_routea_reference")
routeb_batch = importlib.import_module("scripts.core_workflow.03_01_build_kong2022_reference")
roi_fc_mod = importlib.import_module("scripts.core_workflow.04_01_materialize_kong2022_roi_fc")
group_atlas_mod = importlib.import_module("scripts.core_workflow.05_02_run_group_atlas")


def test_prepare_group_inputs_manifest_ignores_comments(tmp_path: Path) -> None:
    manifest = tmp_path / "subjects.txt"
    manifest.write_text("# cohort note\n1001\nsub-1002\n\n", encoding="utf-8")

    subjects = prepare_inputs_mod.load_subjects(tmp_path / "hippunfold", tmp_path / "bold", manifest, None)

    assert subjects == ["1001", "1002"]


def test_single_volume_sampling_records_volume_in_stage_manifest(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bold = tmp_path / "sub-01_bold.nii.gz"
    bold.write_bytes(b"volume fixture")
    log_dir = tmp_path / "logs"
    log_dir.mkdir()
    monkeypatch.setattr(
        prepare_inputs_mod,
        "sample_hipp_surface_timeseries",
        lambda **_kwargs: [{"hemi": "L"}, {"hemi": "R"}],
    )

    result = prepare_inputs_mod.run_raw_sampling(
        subject="01",
        batch_root=str(tmp_path / "batch"),
        shared_root=str(tmp_path / "shared"),
        bold_root=str(tmp_path),
        bold_template="{bold_root}/sub-{subject}_bold.nii.gz",
        log_dir=str(log_dir),
    )

    manifest_path = (
        tmp_path / "shared" / "sub-01" / "surface" / "raw" / "stage_manifest.json"
    )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert result["ok"] is True
    assert [item["path"] for item in manifest["inputs"]] == [str(bold.resolve())]


def test_persistent_sampled_metric_summary_omits_temporary_metric_path() -> None:
    sampled = [
        {
            "hemi": "L",
            "surface": "/data/sub-1027_hemi-L_midthickness.surf.gii",
            "metric_gii": "/tmp/sub-1027_run_surface_x/run-2/bold.func.gii",
            "n_vertices": 512,
        }
    ]

    persisted = surface_sampling.persistent_sampled_metric_summary(sampled)

    assert persisted == [
        {
            "hemi": "L",
            "surface": "/data/sub-1027_hemi-L_midthickness.surf.gii",
            "n_vertices": 512,
        }
    ]


def test_run_volume_sampling_concatenates_in_memory_and_removes_run_outputs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    surfaces: dict[str, Path] = {}
    for hemi in ("L", "R"):
        surface = tmp_path / f"sub-1027_hemi-{hemi}_midthickness.surf.gii"
        nib.save(
            nib.gifti.GiftiImage(
                darrays=[
                    nib.gifti.GiftiDataArray(
                        data=np.zeros((3, 3), dtype=np.float32),
                        intent="NIFTI_INTENT_POINTSET",
                    )
                ]
            ),
            surface,
        )
        surfaces[hemi] = surface

    runs: list[tuple[int, Path]] = []
    for run_id, n_timepoints in ((2, 2), (3, 3)):
        bold = tmp_path / f"sub-1027_run-{run_id:02d}_bold.nii.gz"
        nib.save(
            nib.Nifti1Image(
                np.zeros((2, 2, 2, n_timepoints), dtype=np.float32),
                np.eye(4),
            ),
            bold,
        )
        runs.append((run_id, bold))

    monkeypatch.setattr(surface_sampling, '_prepare_surfaces', lambda **kwargs: (surfaces, surfaces))

    def fake_sample(**kwargs: object) -> list[dict[str, object]]:
        bold = Path(kwargs["bold"])
        outdir = Path(kwargs["outdir"])
        outdir.mkdir(parents=True, exist_ok=True)
        n_timepoints = int(nib.load(str(bold)).shape[3])
        sampled: list[dict[str, object]] = []
        for hemi in ("L", "R"):
            metric = outdir / (
                f"sub-1027_hemi-{hemi}_space-T1w_den-512_label-hipp_bold.func.gii"
            )
            nib.save(
                nib.gifti.GiftiImage(
                    darrays=[
                        nib.gifti.GiftiDataArray(
                            data=np.full((3, n_timepoints), n_timepoints, dtype=np.float32),
                            intent="NIFTI_INTENT_TIME_SERIES",
                        )
                    ]
                ),
                metric,
            )
            sampled.append({"hemi": hemi, "metric_gii": str(metric)})
        return sampled

    monkeypatch.setattr(surface_sampling, "_sample_prepared_surfaces", fake_sample)
    outdir = tmp_path / "sampled"

    arrays, summary = surface_sampling.sample_hipp_surface_run_timeseries(
        runs=runs,
        hippunfold_dir=tmp_path,
        subject="1027",
        density="512",
        outdir=outdir,
        space="T1w",
        registration=surface_sampling.SurfaceRegistration('ants', tmp_path/'transform.h5'),
    )

    assert summary["run_ids"] == [2, 3]
    assert summary["run_lengths"] == [2, 3]
    assert arrays["L"].shape == (3, 5)
    assert arrays["R"].shape == (3, 5)
    assert list(outdir.iterdir()) == []


def test_prepare_group_inputs_help_exposes_no_resume() -> None:
    proc = subprocess.run(
        [sys.executable, str(CORE_DIR / "02_01_prepare_group_inputs.py"), "--help"],
        check=False,
        capture_output=True,
        text=True,
    )

    assert proc.returncode == 0
    assert "--no-resume" in proc.stdout


def test_prepare_group_inputs_cli_only_exposes_volume_surface_sampling() -> None:
    proc = subprocess.run(
        [sys.executable, str(CORE_DIR / "02_01_prepare_group_inputs.py"), "--help"],
        check=False,
        capture_output=True,
        text=True,
    )

    assert proc.returncode == 0, proc.stderr
    assert "--run-artifact-manifest-template" in proc.stdout
    assert "--run-bold-glob" not in proc.stdout
    assert "--run-dtseries-glob" not in proc.stdout
    assert "cifti-subcortex" not in proc.stdout


@pytest.mark.parametrize(
    "script_name",
    [
        "02_01_prepare_group_inputs.py",
        "03_01_build_kong2022_reference.py",
        "03_02_build_kong2022_routea_reference.py",
        "04_01_materialize_kong2022_roi_fc.py",
        "05_01_compute_hippomaps_gradients.py",
        "05_02_run_group_atlas.py",
        "05_05_build_weighted_fisherz_branch.py",
    ],
)
def test_core_workflow_cli_has_no_dataset_specific_source_guardrail(script_name: str) -> None:
    proc = subprocess.run(
        [sys.executable, str(CORE_DIR / script_name), "--help"],
        check=False,
        capture_output=True,
        text=True,
    )

    assert proc.returncode == 0, proc.stderr
    assert "--functional-source-variant" not in proc.stdout


def _write_test_dtseries(path: Path, data: np.ndarray) -> None:
    left = BrainModelAxis.from_surface(np.arange(4), 4, name="CORTEX_LEFT")
    right = BrainModelAxis.from_surface(np.arange(4), 4, name="CORTEX_RIGHT")
    image = Cifti2Image(
        np.asarray(data, dtype=np.float32),
        header=(SeriesAxis(0.0, 1.0, data.shape[0]), left + right),
    )
    image.to_filename(path)



def _artifact_receipt(path: Path) -> dict[str, object]:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return {
        "path": str(path.resolve()),
        "sha256": digest.hexdigest(),
        "size_bytes": path.stat().st_size,
    }

def test_run_raw_sampling_uses_manifest_declared_runs_and_receipts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bolds: dict[int, Path] = {}
    dtseries: dict[int, Path] = {}
    for run_id, n_timepoints in ((2, 2), (7, 3)):
        bold = tmp_path / f"bold-source-{run_id}.nii.gz"
        nib.save(
            nib.Nifti1Image(
                np.zeros((2, 2, 2, n_timepoints), dtype=np.float32),
                np.eye(4),
            ),
            bold,
        )
        dtseries_path = tmp_path / f"cifti-source-{run_id}.dtseries.nii"
        _write_test_dtseries(
            dtseries_path,
            np.zeros((n_timepoints, 8), dtype=np.float32),
        )
        bolds[run_id] = bold
        dtseries[run_id] = dtseries_path

    manifest = tmp_path / "sub-1027_run_artifacts.json"
    manifest.write_text(
        json.dumps(
            {
                "subject": "sub-1027",
                "runs": [
                    {
                        "run_id": run_id,
                        "bold": _artifact_receipt(bolds[run_id]),
                        "dtseries": _artifact_receipt(dtseries[run_id]),
                    }
                    for run_id in (2, 7)
                ],
            }
        ),
        encoding="utf-8",
    )

    captured_runs: list[tuple[int, Path]] = []

    def fake_sample(**kwargs: object) -> tuple[dict[str, np.ndarray], dict[str, object]]:
        captured_runs.extend(kwargs["runs"])
        return (
            {
                "L": np.ones((3, 5), dtype=np.float32),
                "R": np.ones((3, 5), dtype=np.float32),
            },
            {"run_ids": [2, 7], "run_lengths": [2, 3]},
        )

    monkeypatch.setattr(
        prepare_inputs_mod,
        "sample_hipp_surface_run_timeseries",
        fake_sample,
    )
    log_dir = tmp_path / "logs"
    log_dir.mkdir()

    result = prepare_inputs_mod.run_raw_sampling(
        subject="1027",
        batch_root=str(tmp_path / "batch"),
        shared_root=str(tmp_path / "shared"),
        bold_root=str(tmp_path),
        bold_template="{bold_root}/sub-{subject}_bold.nii.gz",
        log_dir=str(log_dir),
        run_artifact_manifest_template="{bold_root}/{sub_subject}_run_artifacts.json",
    )

    assert result["ok"] is True
    assert captured_runs == [
        (2, bolds[2].resolve()),
        (7, bolds[7].resolve()),
    ]
    summary_path = (
        tmp_path
        / "shared"
        / "sub-1027"
        / "surface"
        / "raw"
        / "surface_sampling_summary.json"
    )
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert summary["run_ids"] == [2, 7]
    assert summary["dtseries_inputs"] == [
        str(dtseries[2].resolve()),
        str(dtseries[7].resolve()),
    ]


def test_run_raw_sampling_missing_manifest_leaves_no_raw_output(tmp_path: Path) -> None:
    log_dir = tmp_path / "logs"
    log_dir.mkdir()
    shared_root = tmp_path / "shared"

    result = prepare_inputs_mod.run_raw_sampling(
        subject="1027",
        batch_root=str(tmp_path / "batch"),
        shared_root=str(shared_root),
        bold_root=str(tmp_path),
        bold_template="{bold_root}/sub-{subject}_bold.nii.gz",
        log_dir=str(log_dir),
        run_artifact_manifest_template="{bold_root}/{sub_subject}_run_artifacts.json",
    )

    assert result["ok"] is False
    assert "run artifact manifest" in str(result["error"])
    assert not shared_root.exists()


def test_load_run_artifact_manifest_preserves_declared_order_and_receipts(
    tmp_path: Path,
) -> None:
    run2 = tmp_path / "sub-1027_task-rest_run-02.dtseries.nii"
    run3 = tmp_path / "sub-1027_task-rest_run-03.dtseries.nii"
    run2.write_bytes(b"run-2")
    run3.write_bytes(b"run-3")
    manifest = tmp_path / "sub-1027_run_artifacts.json"
    manifest.write_text(
        json.dumps(
            {
                "subject": "sub-1027",
                "runs": [
                    {"run_id": 2, "dtseries": _artifact_receipt(run2)},
                    {"run_id": 3, "dtseries": _artifact_receipt(run3)},
                ],
            }
        ),
        encoding="utf-8",
    )

    runs = io_utils.load_run_artifact_manifest(
        manifest,
        subject="1027",
        required_artifacts=("dtseries",),
    )

    assert [row["run_id"] for row in runs] == [2, 3]
    assert [row["artifacts"]["dtseries"] for row in runs] == [
        run2.resolve(),
        run3.resolve(),
    ]
    assert runs[0]["artifact_receipts"]["dtseries"] == _artifact_receipt(run2)


def test_load_run_artifact_manifest_rejects_missing_or_duplicate_run_ids(
    tmp_path: Path,
) -> None:
    run = tmp_path / "sub-1027_task-rest_run-02.dtseries.nii"
    run.write_bytes(b"run-2")
    manifest = tmp_path / "sub-1027_run_artifacts.json"
    receipt = _artifact_receipt(run)

    manifest.write_text(
        json.dumps(
            {
                "subject": "sub-1027",
                "runs": [{"dtseries": receipt}],
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="positive explicit run_id"):
        io_utils.load_run_artifact_manifest(
            manifest,
            subject="1027",
            required_artifacts=("dtseries",),
        )

    manifest.write_text(
        json.dumps(
            {
                "subject": "sub-1027",
                "runs": [
                    {"run_id": 2, "dtseries": receipt},
                    {"run_id": 2, "dtseries": receipt},
                ],
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="strictly increasing"):
        io_utils.load_run_artifact_manifest(
            manifest,
            subject="1027",
            required_artifacts=("dtseries",),
        )

def test_load_cifti_dtseries_inputs_concatenates_in_memory(tmp_path: Path) -> None:
    run1 = tmp_path / "sub-01_task-rest_run-01.dtseries.nii"
    run2 = tmp_path / "sub-01_task-rest_run-02.dtseries.nii"
    first = np.arange(16, dtype=np.float32).reshape(2, 8)
    second = (100.0 + np.arange(24, dtype=np.float32)).reshape(3, 8)
    _write_test_dtseries(run1, first)
    _write_test_dtseries(run2, second)

    data, axis, lengths = io_utils.load_cifti_dtseries_inputs([run1, run2])

    assert data.shape == (5, 8)
    np.testing.assert_array_equal(data, np.vstack([first, second]))
    assert axis.size == 8
    assert set(axis.name.tolist()) == {
        "CIFTI_STRUCTURE_CORTEX_LEFT",
        "CIFTI_STRUCTURE_CORTEX_RIGHT",
    }
    assert lengths == [2, 3]
    assert run1.exists() and run2.exists()


def test_load_cifti_dtseries_inputs_rejects_incompatible_brain_model_axes(tmp_path: Path) -> None:
    run1 = tmp_path / "run-01.dtseries.nii"
    run2 = tmp_path / "run-02.dtseries.nii"
    _write_test_dtseries(run1, np.ones((2, 8), dtype=np.float32))

    left = BrainModelAxis.from_surface(np.arange(4), 4, name="CORTEX_LEFT")
    right = BrainModelAxis.from_surface(np.arange(1, 5), 5, name="CORTEX_RIGHT")
    image = Cifti2Image(
        np.ones((3, 8), dtype=np.float32),
        header=(SeriesAxis(0.0, 1.0, 3), left + right),
    )
    image.to_filename(run2)

    with pytest.raises(ValueError, match="BrainModelAxis"):
        io_utils.load_cifti_dtseries_inputs([run1, run2])


def test_run_artifact_manifest_does_not_sort_or_infer_paths(tmp_path: Path) -> None:
    first = tmp_path / "source-z.dtseries.nii"
    second = tmp_path / "source-a.dtseries.nii"
    first.write_bytes(b"first")
    second.write_bytes(b"second")
    manifest = tmp_path / "sub-01_run_artifacts.json"
    manifest.write_text(
        json.dumps(
            {
                "subject": "sub-01",
                "runs": [
                    {"run_id": 1, "dtseries": _artifact_receipt(first)},
                    {"run_id": 10, "dtseries": _artifact_receipt(second)},
                ],
            }
        ),
        encoding="utf-8",
    )

    records = io_utils.load_run_artifact_manifest(
        manifest,
        subject="01",
        required_artifacts=("dtseries",),
    )

    assert [record["run_id"] for record in records] == [1, 10]
    assert [record["artifacts"]["dtseries"] for record in records] == [
        first.resolve(),
        second.resolve(),
    ]
    with pytest.raises(FileNotFoundError, match="Missing explicit run artifact manifest"):
        io_utils.load_run_artifact_manifest(
            tmp_path / "missing.json",
            subject="01",
            required_artifacts=("dtseries",),
        )


def test_run_artifact_manifest_preserves_noncontiguous_explicit_run_ids(tmp_path: Path) -> None:
    run2 = tmp_path / "source-run-two.dtseries.nii"
    run7 = tmp_path / "source-run-seven.dtseries.nii"
    run2.write_bytes(b"run2")
    run7.write_bytes(b"run7")
    manifest = tmp_path / "sub-1027_run_artifacts.json"
    manifest.write_text(
        json.dumps(
            {
                "subject": "sub-1027",
                "runs": [
                    {"run_id": 2, "dtseries": _artifact_receipt(run2)},
                    {"run_id": 7, "dtseries": _artifact_receipt(run7)},
                ],
            }
        ),
        encoding="utf-8",
    )

    records = io_utils.load_run_artifact_manifest(
        manifest,
        subject="1027",
        required_artifacts=("dtseries",),
    )

    assert [record["run_id"] for record in records] == [2, 7]
    assert [record["artifacts"]["dtseries"] for record in records] == [
        run2.resolve(),
        run7.resolve(),
    ]


def test_run_level_source_summary_preserves_source_run_ids_and_lengths(tmp_path: Path) -> None:
    func = tmp_path / "dtseries" / "sub-1027" / "func"
    func.mkdir(parents=True)
    run2 = func / "sub-1027_task-rest_run-02_space-fsLR_desc-denoised_bold.dtseries.nii"
    run3 = func / "sub-1027_task-rest_run-03_space-fsLR_desc-denoised_bold.dtseries.nii"
    run2.write_bytes(b"run2")
    run3.write_bytes(b"run3")

    summary = io_utils.build_run_level_input_summary(
        [run2, run3],
        [209, 217],
        run_ids=[2, 3],
        input_mode="in_memory_run_concat",
    )

    assert summary == {
        "dtseries_input_mode": "in_memory_run_concat",
        "dtseries_inputs": [str(run2.resolve()), str(run3.resolve())],
        "run_ids": [2, 3],
        "run_lengths": [209, 217],
        "n_timepoints": 426,
    }


def test_surface_sampling_exposes_run_level_in_memory_entrypoint() -> None:
    source = Path(surface_sampling.__file__).read_text(encoding="utf-8")
    assert "sample_hipp_surface_run_timeseries" in source


def test_routea_and_routeb_batches_expose_the_same_no_persist_option() -> None:
    routea_parser_source = Path(routea_batch.__file__).read_text(encoding="utf-8")
    routeb_parser_source = Path(routeb_batch.__file__).read_text(encoding="utf-8")
    assert "--run-artifact-manifest-template" in routea_parser_source
    assert "--run-artifact-manifest-template" in routeb_parser_source
    assert "--run-dtseries-glob" not in routea_parser_source
    assert "--run-dtseries-glob" not in routeb_parser_source

@pytest.mark.parametrize(
    ("module", "extra_kwargs"),
    [
        (routeb_batch, {"cross_atlas_merge_json": None}),
        (
            routea_batch,
            {"atlas_slug": "kong2022", "overwrite": False},
        ),
    ],
)
def test_reference_batch_passes_manifest_declared_paths_and_ids(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    module: object,
    extra_kwargs: dict[str, object],
) -> None:
    first = tmp_path / "source-alpha.dtseries.nii"
    second = tmp_path / "source-beta.dtseries.nii"
    _write_test_dtseries(first, np.zeros((3, 8), dtype=np.float32))
    _write_test_dtseries(second, np.zeros((2, 8), dtype=np.float32))
    manifest = tmp_path / "sub-1027_run_artifacts.json"
    manifest.write_text(
        json.dumps(
            {
                "subject": "sub-1027",
                "runs": [
                    {"run_id": 2, "dtseries": _artifact_receipt(first)},
                    {"run_id": 7, "dtseries": _artifact_receipt(second)},
                ],
            }
        ),
        encoding="utf-8",
    )
    calls = []

    def fake_build(**kwargs):
        calls.append(kwargs["request"])

    monkeypatch.setattr(module, "build_kong2022_reference", fake_build)
    kwargs: dict[str, object] = {
        "subject": "1027",
        "shared_root": tmp_path / "shared",
        "data_root": tmp_path,
        "dtseries_template": "{data_root}/legacy-{subject}.dtseries.nii",
        "run_artifact_manifest_template": "{data_root}/sub-{subject}_run_artifacts.json",
        "kong2022_dlabel": None,
        **extra_kwargs,
    }
    if module is routea_batch:
        kwargs["route_a_root"] = tmp_path / "route-a"
        kwargs["dataset"] = "hcp"
    result = module.run_subject(**kwargs)

    assert result == ("1027", True, "ok")
    assert len(calls) == 1
    selected = calls[0]
    assert selected.run_dtseries == [first.resolve(), second.resolve()]
    assert selected.run_ids == [2, 7]
    assert selected.run_artifact_manifest == manifest.resolve()



def test_fc_run_loader_uses_current_reference_manifest_not_stale_run_directories(tmp_path: Path) -> None:
    subject_root = tmp_path / "sub-01"
    reference_dir = subject_root / "reference" / "kong2022"
    reference_dir.mkdir(parents=True)
    (reference_dir / "reference_summary.json").write_text(
        '{"run_ids": [1, 2, 3], "run_lengths": [2, 3, 4]}\n',
        encoding="utf-8",
    )
    stale_run = subject_root / "reference" / "common" / "runs" / "run-4"
    stale_run.mkdir(parents=True)
    (stale_run / "reference_summary.json").write_text(
        '{"run_id": "4", "run_start": 6, "run_stop": 99, "run_length": 93}\n',
        encoding="utf-8",
    )

    assert roi_fc_mod.load_run_segments(subject_root) == [
        {"run_id": 1, "run_start": 0, "run_stop": 2, "run_length": 2},
        {"run_id": 2, "run_start": 2, "run_stop": 5, "run_length": 3},
        {"run_id": 3, "run_start": 5, "run_stop": 9, "run_length": 4},
    ]


def test_fc_explicit_run_metadata_preserves_nonconsecutive_ids(tmp_path: Path) -> None:
    metadata = tmp_path / "runs.json"
    metadata.write_text(
        '{"run_ids": [1, 3], "run_lengths": [2, 5], "concat_timepoints": 7}',
        encoding="utf-8",
    )
    assert roi_fc_mod.load_run_segments(tmp_path / "sub-01", run_metadata=metadata) == [
        {"run_id": 1, "run_start": 0, "run_stop": 2, "run_length": 2},
        {"run_id": 3, "run_start": 2, "run_stop": 7, "run_length": 5},
    ]


def test_fc_cortex_valid_masks_are_required_by_default(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="explicit cortical validity mask"):
        roi_fc_mod.load_optional_cortex_valid_mask(
            tmp_path,
            "01",
            {"CORTEX_LEFT": 2, "CORTEX_RIGHT": 2},
        )


def test_cifti_reference_extractors_require_manifest_declared_run_contract() -> None:
    extractor_sources = [
        (
            REPO_ROOT / "scripts/core_workflow/lib/extract_cortex_kong2022_routeb_roi_timeseries.py"
        ).read_text(encoding="utf-8"),
        (
            REPO_ROOT
            / "scripts/core_workflow/lib/extract_cortex_kong2022_route_a_roi_timeseries.py"
        ).read_text(encoding="utf-8"),
    ]
    for source in extractor_sources:
        assert "--run-ids" in source
        assert "--run-artifact-manifest" in source
        assert "run_ids_from_paths" not in source

def test_group_atlas_runwise_loader_uses_current_fc_manifest_not_stale_run_directories(tmp_path: Path) -> None:
    from lib.kong2022_label_utils import load_kong2022_bundle
    subject = "01"
    reference = tmp_path / f"sub-{subject}" / "reference" / "kong2022"
    reference.mkdir(parents=True)
    rows = load_kong2022_bundle()["roi_registry"]
    (reference / "cortex_roi_parcels.tsv").write_text(
        "parcel_id\tparcel_name\n" + "".join(f"{r['roi_id']}\t{r['roi_name']}\n" for r in rows))
    np.save(reference / "cortex_roi_parcel_timeseries.npy", np.ones((400, 8), dtype=np.float32))
    fc_dir = tmp_path / f"sub-{subject}" / "fc" / "kong2022" / "2mm" / "hemi_L"
    fc_dir.mkdir(parents=True)
    (fc_dir / "fc_summary.json").write_text(
        '{"run_labels": ["run-1", "run-2", "run-3"], "run_lengths": [2, 3, 4]}\n',
        encoding="utf-8",
    )
    for rid in (1, 2, 3):
        run_dir = fc_dir / "runs" / f"run-{rid}"
        run_dir.mkdir(parents=True)
        np.save(run_dir / "hipp_vertex_to_network_fc_active.npy", np.full((2, 400), 0.1 * rid, dtype=np.float32))
    stale_run = fc_dir / "runs" / "run-4"
    stale_run.mkdir(parents=True)
    np.save(stale_run / "hipp_vertex_to_network_fc_active.npy", np.ones((2, 2), dtype=np.float32))

    mode = group_atlas_mod.SubjectFCMode(
        mode="full",
        mode_tag="full",
        top_percent=None,
        top_fraction=1.0,
        binarize=False,
    )
    bank = group_atlas_mod.runwise_feature_bank(
        method="gradient-spectral",
        shared_root=tmp_path,
        subjects=[subject],
        atlas="kong2022",
        smoothing="2mm",
        hemi="L",
        valid_masks={subject: np.ones(2, dtype=bool)},
        subject_fc_mode=mode,
        reference_root=tmp_path,
    )

    assert sorted(bank) == [1, 2, 3]
    for rid in (1, 2, 3):
        np.testing.assert_allclose(bank[rid][subject], np.full((2, 400), np.arctanh(0.1 * rid)), rtol=1e-6)


@pytest.mark.parametrize('module', [routea_batch, routeb_batch])
def test_reference_worker_preserves_outputs_and_isolates_concurrent_subject_failure(tmp_path, module):
    from concurrent.futures import ThreadPoolExecutor
    from lib.kong2022_label_utils import load_kong2022_bundle
    bundle = load_kong2022_bundle()
    ids = np.asarray([row['roi_id'] for row in bundle['roi_registry']], np.int32)
    left = nib.cifti2.BrainModelAxis.from_surface(np.arange(200), 32492, name='CORTEX_LEFT')
    right = nib.cifti2.BrainModelAxis.from_surface(np.arange(200), 32492, name='CORTEX_RIGHT')
    axis = left + right
    data = np.tile(np.asarray([10., 12., 14., 12.], np.float32)[:, None], (1, 400))
    source = tmp_path / 'sub-01.dtseries.nii'
    nib.save(nib.Cifti2Image(data, header=nib.Cifti2Header.from_axes([
        nib.cifti2.SeriesAxis(0, 1, 4), axis])), source)
    dlabel = tmp_path / 'atlas.dlabel.nii'
    nib.save(nib.Cifti2Image(ids[None, :], header=nib.Cifti2Header.from_axes([
        bundle['img'].header.get_axis(0), axis])), dlabel)
    labels = tmp_path / 'labels/sub-01'
    labels.mkdir(parents=True)
    for hemi, values in [('lh', ids[:200]), ('rh', ids[200:])]:
        full = np.zeros(32492, np.int32)
        full[:200] = values
        np.savetxt(labels / f'{hemi}_labels.txt', full, fmt='%d')
    metadata = tmp_path / 'runs.json'
    metadata.write_text(json.dumps({'run_ids': [1, 3], 'run_lengths': [2, 2]}))
    shared = tmp_path / 'shared'
    kwargs = dict(shared_root=shared, data_root=tmp_path,
        dtseries_template=str(tmp_path / 'sub-{subject}.dtseries.nii'),
        run_artifact_manifest_template=None, kong2022_dlabel=dlabel,
        run_metadata_template=str(metadata))
    if module is routea_batch:
        kwargs.update(route_a_root=labels.parent, atlas_slug='kong2022', overwrite=False, dataset='hcp')
    else:
        kwargs['cross_atlas_merge_json'] = None
    original_argv = sys.argv[:]
    with ThreadPoolExecutor(max_workers=2) as pool:
        good = pool.submit(module.run_subject, subject='01', **kwargs)
        bad = pool.submit(module.run_subject, subject='02', **kwargs)
        good_result, bad_result = good.result(), bad.result()
    assert good_result == ('01', True, 'ok')
    assert bad_result[:2] == ('02', False)
    assert 'sub-02.dtseries.nii' in bad_result[2]
    assert sys.argv == original_argv
    reference = shared / 'sub-01/reference/kong2022'
    np.testing.assert_array_equal(np.load(reference / 'cortex_roi_parcel_timeseries.npy'), data.T)
    summary = json.loads((reference / 'reference_summary.json').read_text())
    assert summary['run_ids'] == [1, 3]
    assert summary['run_lengths'] == [2, 2]
    assert summary['n_parcels_used_after_tsnr'] == 400


def test_routeb_batch_default_matches_its_public_label_extractor():
    import importlib
    direct = importlib.import_module('scripts.core_workflow.lib.extract_cortex_kong2022_routeb_roi_timeseries')
    assert routeb_batch.DEFAULT_SHARED_ROOT == direct.DEFAULT_SHARED_ROOT
    assert routeb_batch.DEFAULT_SHARED_ROOT.name == 'routeb'
    assert routeb_batch.DEFAULT_SHARED_ROOT != routea_batch.DEFAULT_SHARED_ROOT


def test_reference_request_preserves_routea_cli_label_root_default():
    from lib.kong2022_reference import ReferenceRequest
    from lib.output_layout import DEFAULT_CORTEX_ROUTEA_ROOT
    request = ReferenceRequest(subject='01', shared_root=Path('unused'))
    assert request.route_a_root == DEFAULT_CORTEX_ROUTEA_ROOT


@pytest.mark.parametrize('selected', [
    {'dataset': 'stanford'}, {'task': 'wm'}, {'tsnr_threshold': 'none'},
    {'tsnr_input_template': 'baseline-{hemi}.npz'}, {'atlas_slug': 'other'},
])
def test_routeb_callable_rejects_unsupported_policy_before_reading_inputs(tmp_path, selected):
    from lib.extract_cortex_kong2022_routeb_roi_timeseries import build_kong2022_reference
    from lib.kong2022_reference import ReferenceRequest
    request = ReferenceRequest(subject='01', shared_root=tmp_path / 'out', **selected)
    with pytest.raises(ValueError, match='Route B.*HCP rest.*25'):
        build_kong2022_reference(request=request)
    assert not (tmp_path / 'out').exists()
