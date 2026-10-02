"""Contracts for the Kember-method surface-area pilot runner."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pytest


SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))

from native_label_alignment import (  # noqa: E402
    _normalise_resampled_labels,
    evaluate_hemisphere_area,
    freeze_input_manifest,
    run_kember_method_area,
    write_area_run,
)
import native_label_alignment as runner_module  # noqa: E402


def _surface() -> tuple[np.ndarray, np.ndarray]:
    return (
        np.asarray(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [1.0, 1.0, 0.0],
            ]
        ),
        np.asarray([[0, 1, 2], [1, 3, 2]], dtype=np.int64),
    )


def _density_evidence(*, n_vertices: int = 4) -> dict[str, object]:
    return {
        "consumed_asset": "/frozen/sub-1001_hemi-L_midthickness.surf.gii",
        "n_vertices": n_vertices,
        "surface_fingerprint": {"sha256": "a" * 64},
        "evidence_source": "consumed_surface_path_and_gifti_point_count",
    }


def _record(*, hemisphere: str = "L", branch: str = "hcp151_fixed") -> dict[str, object]:
    return evaluate_hemisphere_area(
        branch=branch,
        subject="sub-1001",
        smoothing_mm=2,
        hemisphere=hemisphere,
        surface_path=f"/current/sub-1001_hemi-{hemisphere}_midthickness.surf.gii",
        vertex_area_mm2=np.asarray([0.2, 0.3, 0.4, 0.1], dtype=np.float64),
        labels=np.asarray([1, 1, 2, 3], dtype=np.int16),
        valid_mask=np.asarray([True, True, True, True]),
        density_evidence=_density_evidence(),
    )


def test_evaluate_hemisphere_area_requires_matching_asset_backed_density_evidence() -> None:
    """A label/surface calculation cannot proceed from an inferred vertex count."""

    with pytest.raises(ValueError, match="asset-backed density evidence"):
        evaluate_hemisphere_area(
            branch="hcp151_fixed",
            subject="sub-1001",
            smoothing_mm=2,
            hemisphere="L",
            surface_path="/current/surface.surf.gii",
            vertex_area_mm2=np.asarray([0.2, 0.3, 0.4, 0.1], dtype=np.float64),
            labels=np.asarray([1, 1, 2, 3], dtype=np.int16),
            valid_mask=None,
            density_evidence=_density_evidence(n_vertices=512),
        )


def test_evaluate_hemisphere_area_keeps_branch_identity_and_area_contract() -> None:
    """Per-hemisphere records retain branch provenance with Kember area output."""

    record = _record(branch="hcp151_template_matching")

    assert record["branch"] == "hcp151_template_matching"
    assert record["subject"] == "sub-1001"
    assert record["hemisphere"] == "L"
    assert record["density_evidence"] == _density_evidence()
    assert record["system_area_mm2"] == pytest.approx(
        {"anterior": 0.5, "body": 0.4, "posterior": 0.1}
    )
    assert record["total_valid_surface_area_mm2"] == pytest.approx(1.0)


def test_resampled_labels_are_zeroed_outside_the_native_valid_roi() -> None:
    """Workbench fill values outside the valid ROI cannot become a system label."""

    labels = _normalise_resampled_labels(
        np.asarray([1, 4, 2], dtype=np.int32),
        np.asarray([True, False, True]),
    )

    assert labels.tolist() == [1, 0, 2]
    with pytest.raises(ValueError, match="unsupported label"):
        _normalise_resampled_labels(
            np.asarray([1, 4, 2], dtype=np.int32),
            np.asarray([True, True, True]),
        )


def test_write_area_run_hashes_tabular_outputs_and_overwrites_current_run(tmp_path: Path) -> None:
    """The pilot writes a self-describing result directory that can be corrected in place."""

    output_dir = tmp_path / "pilot-001"
    result = write_area_run(
        output_dir,
        records=[_record(hemisphere="L"), _record(hemisphere="R")],
        input_contract={"common_subjects": ["sub-1001"]},
        source_lock={"tag": "v1", "commit": "5fd2858ca8979891216f518b45751d4517e037a0"},
        runtime_provenance={"mode": "small_dual_hemi_pilot"},
    )

    per_hemisphere = output_dir / "system_surface_area_per_hemisphere.csv"
    bilateral = output_dir / "system_surface_area_bilateral.csv"
    manifest = output_dir / "run_manifest.json"
    assert result["output_dir"] == str(output_dir)
    assert per_hemisphere.is_file()
    assert bilateral.is_file()
    assert manifest.is_file()
    assert "total_valid_surface_area_mm2" in per_hemisphere.read_text(encoding="utf-8").splitlines()[0]
    manifest_data = json.loads(manifest.read_text(encoding="utf-8"))
    assert manifest_data["source_lock"] == {
        "tag": "v1",
        "commit": "5fd2858ca8979891216f518b45751d4517e037a0",
    }
    assert manifest_data["hemisphere_aggregation"] == "arithmetic mean of left and right system values"
    assert manifest_data["outputs"][per_hemisphere.name]["sha256"] == hashlib.sha256(
        per_hemisphere.read_bytes()
    ).hexdigest()
    assert manifest_data["outputs"][bilateral.name]["sha256"] == hashlib.sha256(
        bilateral.read_bytes()
    ).hexdigest()
    assert "hcp151_fixed" in manifest_data["branches"]

    write_area_run(
        output_dir,
        records=[_record(hemisphere="L"), _record(hemisphere="R")],
        input_contract={"common_subjects": ["sub-1001"]},
        source_lock={"tag": "v1", "commit": "5fd2858ca8979891216f518b45751d4517e037a0"},
        runtime_provenance={"mode": "corrected_dual_hemi_pilot"},
    )
    updated_manifest = json.loads(manifest.read_text(encoding="utf-8"))
    assert updated_manifest["runtime_provenance"]["mode"] == "corrected_dual_hemi_pilot"


def test_freeze_input_manifest_precedes_calculation_and_overwrites_current_run(
    tmp_path: Path,
) -> None:
    """Every pilot run has a durable pre-calculation input/density record."""

    output_dir = tmp_path / "pilot-002"
    result = freeze_input_manifest(
        output_dir,
        source_lock={"tag": "v1", "commit": "5fd2858ca8979891216f518b45751d4517e037a0"},
        input_contract={"common_subjects": ["sub-1001"]},
        consumed_inputs={
            "hcp151_fixed": {
                "surface": {"density_evidence": _density_evidence()},
            }
        },
    )

    manifest_path = output_dir / "input_manifest.json"
    assert result["input_manifest_path"] == str(manifest_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["status"] == "frozen_precalculation"
    assert manifest["consumed_inputs"]["hcp151_fixed"]["surface"]["density_evidence"] == _density_evidence()
    freeze_input_manifest(
        output_dir,
        source_lock={"tag": "v1", "commit": "5fd2858ca8979891216f518b45751d4517e037a0"},
        input_contract={"common_subjects": ["sub-1002"]},
        consumed_inputs={},
    )
    updated_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert updated_manifest["input_contract"]["common_subjects"] == ["sub-1002"]

def test_small_dual_hemi_runner_freezes_inputs_before_writing_branch_results(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A real GIFTI-backed pilot retains fixed and template branches separately."""

    import nibabel as nib
    from nibabel.gifti import GiftiDataArray, GiftiImage

    def write_surface(path: Path) -> None:
        coordinates, faces = _surface()
        path.parent.mkdir(parents=True, exist_ok=True)
        image = GiftiImage(
            darrays=[
                GiftiDataArray(
                    data=coordinates.astype(np.float32),
                    intent="NIFTI_INTENT_POINTSET",
                ),
                GiftiDataArray(
                    data=faces.astype(np.int32),
                    intent="NIFTI_INTENT_TRIANGLE",
                ),
            ]
        )
        nib.save(image, str(path))

    def write_area_metric(path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        image = GiftiImage(
            darrays=[
                GiftiDataArray(
                    data=np.asarray([0.2, 0.3, 0.4, 0.1], dtype=np.float32),
                    intent="NIFTI_INTENT_SHAPE",
                )
            ]
        )
        nib.save(image, str(path))

    root = tmp_path
    config_path = root / "config.json"
    roster_path = root / "rosters" / "all_subjects.txt"
    roster_path.parent.mkdir(parents=True, exist_ok=True)
    roster_path.write_text("1001\n", encoding="utf-8")
    config_path.write_text(
        json.dumps(
            {
                "smoothings_mm": [2],
                "hemispheres": ["L", "R"],
                "rosters": {"all_subjects": "rosters/all_subjects.txt"},
                "paths": {
                    "hippunfold_root": "hippunfold",
                    "hcp151_template_root": "fixed",
                    "template_matching_root": "matching",
                },
            }
        ),
        encoding="utf-8",
    )

    labels = np.asarray([1, 1, 2, 3], dtype=np.int16)
    for hemisphere in ("L", "R"):
        surf = root / "hippunfold" / "sub-1001" / "hippunfold" / "sub-1001" / "surf"
        source_surface_path = (
            surf
            / f"sub-1001_hemi-{hemisphere}_space-corobl_den-512_label-hipp_midthickness.surf.gii"
        )
        for space, density in (
            ("corobl", "512"),
            ("corobl", "native"),
            ("unfold", "512"),
            ("unfold", "native"),
        ):
            write_surface(
                surf
                / f"sub-1001_hemi-{hemisphere}_space-{space}_den-{density}_label-hipp_midthickness.surf.gii"
            )
        write_area_metric(
            root
            / "hippunfold"
            / "sub-1001"
            / "hippunfold"
            / "sub-1001"
            / "metric"
            / f"sub-1001_hemi-{hemisphere}_den-native_label-hipp_surfareacorobl.shape.gii"
        )
        sha256 = hashlib.sha256(source_surface_path.read_bytes()).hexdigest()
        density_evidence = {
            "consumed_asset": str(source_surface_path),
            "n_vertices": 4,
            "surface_fingerprint": {"sha256": sha256},
            "evidence_source": "consumed_surface_path_and_gifti_point_count",
        }
        fixed = root / "fixed"
        fixed.mkdir(parents=True, exist_ok=True)
        np.save(fixed / f"hcp151_2mm_hemi-{hemisphere}_labels.npy", labels)
        np.save(fixed / f"hcp151_2mm_hemi-{hemisphere}_support.npy", np.ones(4, dtype=bool))
        matching = (
            root
            / "matching"
            / "subjects"
            / "sub-1001"
            / "matching"
            / "2mm"
            / f"hemi-{hemisphere}"
        )
        matching.mkdir(parents=True, exist_ok=True)
        np.save(matching / "labels.npy", labels)
        np.save(matching / "valid_mask.npy", np.ones(4, dtype=bool))
        (matching / "combo_manifest.json").write_text(
            json.dumps({"density_evidence": density_evidence}),
            encoding="utf-8",
        )

    def fake_resample(
        entry: dict[str, object],
        *,
        output_dir: Path,
        workbench_command: Path,
    ) -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
        del output_dir, workbench_command
        return (
            np.asarray(entry["labels"], dtype=np.int32),
            np.asarray(entry["valid_mask"], dtype=bool),
            {"method": "test_native_label_transfer"},
        )

    monkeypatch.setattr(runner_module, "align_labels_to_native", fake_resample)
    source_lock_path = root / "source_lock.json"
    source_lock_path.write_text(
        json.dumps(
            {
                "tag": "v1",
                "commit": "5fd2858ca8979891216f518b45751d4517e037a0",
                "kember_method_area_implementation": {
                    "next_action_requires_user_authorization": False
                },
            }
        ),
        encoding="utf-8",
    )
    output_dir = root / "results" / "pilot-003"
    result = run_kember_method_area(
        root=root,
        config_path=config_path,
        source_lock_path=source_lock_path,
        output_dir=output_dir,
        subject="sub-1001",
        smoothing_mm=2,
        workbench_command="/bin/true",
        branches=("hcp151_fixed", "hcp151_template_matching"),
    )

    assert result["output_dir"] == str(output_dir)
    assert (output_dir / "input_manifest.json").is_file()
    assert (output_dir / "run_manifest.json").is_file()
    assert len((output_dir / "system_surface_area_per_hemisphere.csv").read_text().splitlines()) == 5
    assert len((output_dir / "system_surface_area_bilateral.csv").read_text().splitlines()) == 3
    run_manifest = json.loads((output_dir / "run_manifest.json").read_text(encoding="utf-8"))
    assert run_manifest["branches"] == ["hcp151_fixed", "hcp151_template_matching"]
    assert run_manifest["vertex_area_convention"] == "HippUnfold native per-vertex surface-area metric in mm^2"
    assert len(run_manifest["runtime_provenance"]["native_label_resampling"]) == 4
