"""Small, explicit inputs shared by core-workflow regression tests."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

CORE = Path(__file__).resolve().parents[2] / "scripts" / "core_workflow"
if str(CORE) not in sys.path:
    sys.path.insert(0, str(CORE))


@pytest.fixture
def cortex_reference_factory():
    """Write the RouteA reference policy and run declarations, without signals."""
    def write(root, *, subject="01", dataset="hcp",
              run_ids=(1,), run_lengths=(4,)):
        root = Path(root)
        ref = root / f"sub-{subject}" / "reference" / "kong2022"
        ref.mkdir(parents=True, exist_ok=True)
        policy = {
            "dataset": dataset, "task": "rest",
            "validity_policy": "finite_cortex_signal",
        }
        (ref / "reference_summary.json").write_text(json.dumps({
            **policy, "subject": subject, "atlas_slug": "kong2022",
            "reference_label_source": "route_a_kong2022_areal_mshbm",
            "run_ids": list(run_ids),
            "run_lengths": list(run_lengths),
        }), encoding="utf-8")
        (ref / "stage_manifest.json").write_text(json.dumps({"status": "done", "params": policy}), encoding="utf-8")
        gate = ref.parent / "common" / "validity"
        gate.mkdir(parents=True, exist_ok=True)
        for side, hemi in (("left", "L"), ("right", "R")):
            (gate / f"cortex_{side}_validity_summary.json").write_text(json.dumps({
                **policy,
            }), encoding="utf-8")
        return ref
    return write


@pytest.fixture
def baseline_input_factory():
    """Write small baseline samples with the declared frame and vertex order."""
    def write(path, data, *, dataset, subject="1", hemi="L", run_lengths=None, vertex_ids=None):
        values = np.asarray(data)
        lengths = [values.shape[1]] if run_lengths is None else list(run_lengths)
        payload = {"dataset": dataset, "subject": subject, "hemi": hemi, "n_runs": len(lengths)}
        if vertex_ids is not None:
            payload["vertex_ids"] = np.asarray(vertex_ids)
        start = 0
        for index, length in enumerate(lengths):
            payload[f"run_{index}"] = values[:, start:start + length]
            payload[f"frames_{index}"] = np.arange(length)
            start += length
        np.savez(path, **payload)
        return Path(path)
    return write


@pytest.fixture
def emotion_run_paths(tmp_path):
    """Two tiny HCP emotion runs with real EV and nuisance files."""
    import nibabel as nib

    paths = []
    for run_id in (1, 2):
        directory = tmp_path / f"run-{run_id}"
        ev = directory / "EVs"
        ev.mkdir(parents=True)
        for stem, onset in (("fear", 10), ("neut", 40)):
            (ev / f"{stem}.txt").write_text(f"{onset} 18 1\n", encoding="utf-8")
        np.savetxt(directory / "Movement_Regressors.txt", np.zeros((176, 12)))
        path = directory / "bold.nii.gz"
        nib.save(nib.Nifti1Image(np.ones((1, 1, 1, 176), np.float32), np.eye(4)), path)
        paths.append(path)
    return paths
