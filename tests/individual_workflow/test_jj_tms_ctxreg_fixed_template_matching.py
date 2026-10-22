from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import nibabel as nib
import numpy as np
import pytest
from nibabel.gifti import GiftiDataArray, GiftiImage


REPO_ROOT = Path(__file__).resolve().parents[2]
MODULE_PATH = (
    REPO_ROOT
    / "scripts"
    / "jj_tms"
    / "jj_tms_ctxreg_fixed_template_matching.py"
)


def load_module():
    assert MODULE_PATH.is_file(), "JJ_TMS CtxReg fixed-template matcher is missing"
    spec = importlib.util.spec_from_file_location(
        "jj_tms_ctxreg_fixed_template_matching",
        MODULE_PATH,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _record(module, path: Path, *, array: np.ndarray | None = None):
    record = module.fingerprint(path)
    if array is not None:
        record.update(shape=list(array.shape), dtype=str(array.dtype))
    return record


def _write_pointset(path: Path, n_vertices: int = 512) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    nib.save(
        GiftiImage(
            darrays=[
                GiftiDataArray(
                    data=np.zeros((n_vertices, 3), dtype=np.float32),
                    intent="NIFTI_INTENT_POINTSET",
                )
            ]
        ),
        str(path),
    )


def _write_template_fixture(module, root: Path, *, template_id: str):
    root.mkdir(parents=True)
    source_root = root.parent / "template-sources"
    source_root.mkdir()
    registry = source_root / "cluster_roi_registry.tsv"
    registry.write_text(
        "roi_id\troi_name\n"
        + "".join(f"{roi_id}\tROI_{roi_id:03d}\n" for roi_id in range(1, 401)),
        encoding="utf-8",
    )

    subjects = [f"sub-{index:06d}" for index in range(151)]
    x = np.linspace(-0.45, 0.45, 400, dtype=np.float32)
    prototypes = np.stack(
        [
            np.sin(x * np.float32(9.0)),
            np.cos(x * np.float32(11.0)),
            np.sin(x * np.float32(17.0) + np.float32(0.7)),
        ],
        axis=0,
    ).astype(np.float32)
    labels = (np.arange(512, dtype=np.int32) % 3) + 1
    support = np.ones(512, dtype=np.uint8)
    combinations = []

    for smoothing in ("2mm", "4mm"):
        for hemi in ("L", "R"):
            combo_root = source_root / smoothing / f"hemi_{hemi}"
            combo_root.mkdir(parents=True)
            source_surface = combo_root / (
                f"group-average_hemi-{hemi}_space-corobl_den-512_"
                "label-hipp_midthickness.surf.gii"
            )
            _write_pointset(source_surface)
            source_labels = combo_root / "group_labels_final.npy"
            source_support = combo_root / "group_support_mask.npy"
            source_fc = combo_root / "group_mean_fisherz_fc_full.npy"
            np.save(source_labels, labels)
            np.save(source_support, support)
            np.save(source_fc, prototypes[labels - 1])

            group_manifest = combo_root / "group_atlas_manifest.json"
            group_manifest.write_text(
                json.dumps(
                    {
                        "status": "ok",
                        "method": "gradient-spectral",
                        "atlas": "kong2022",
                        "smoothing": smoothing,
                        "hemi": hemi,
                        "n_subjects": 151,
                        "subjects": subjects,
                        "chosen_k_final": 3,
                        "cluster_label_ordering": "long_axis_head_to_tail",
                        "density_evidence": {
                            "n_vertices": 512,
                            "surface_path": str(source_surface.resolve()),
                        },
                        "template_left_or_right_surface": str(
                            source_surface.resolve()
                        ),
                        "cluster_roi_registry": str(registry.resolve()),
                        "group_labels_final": str(source_labels.resolve()),
                        "group_support_mask": str(source_support.resolve()),
                        "resume_input_signature_payload": {
                            "route_tag": "pfm_ctxreg"
                        },
                    }
                ),
                encoding="utf-8",
            )
            gradient_manifest = combo_root / "hippomaps_gradient_manifest.json"
            gradient_manifest.write_text(
                json.dumps(
                    {
                        "status": "ok",
                        "smoothing": smoothing,
                        "hemi": hemi,
                        "n_subjects": 151,
                        "subjects": subjects,
                        "route": "pfm_ctxreg",
                    }
                ),
                encoding="utf-8",
            )

            labels_out = root / f"hcp151_{smoothing}_hemi-{hemi}_labels.npy"
            support_out = root / f"hcp151_{smoothing}_hemi-{hemi}_support.npy"
            prototypes_out = root / f"hcp151_{smoothing}_hemi-{hemi}_prototypes.npy"
            np.save(labels_out, labels)
            np.save(support_out, support)
            np.save(prototypes_out, prototypes)
            combinations.append(
                {
                    "smoothing": smoothing,
                    "hemi": hemi,
                    "source_files": {
                        "group_manifest": _record(module, group_manifest),
                        "gradient_manifest": _record(module, gradient_manifest),
                        "labels": _record(module, source_labels),
                        "support": _record(module, source_support),
                        "group_mean_fisherz_fc_full": _record(module, source_fc),
                        "roi_registry": _record(module, registry),
                        "consumed_surface": _record(module, source_surface),
                    },
                    "outputs": {
                        "labels": _record(module, labels_out, array=labels),
                        "support": _record(module, support_out, array=support),
                        "prototypes": _record(
                            module, prototypes_out, array=prototypes
                        ),
                    },
                    "density_evidence": {
                        "consumed_asset": str(source_surface.resolve()),
                        "evidence_source": "group_atlas_manifest.density_evidence",
                        "n_vertices": 512,
                    },
                }
            )

    manifest = {
        "status": "complete",
        "workflow": "build_hcp151_fixed_template",
        "template_id": template_id,
        "signal_branch": "ctxreg",
        "signal_control_route": "pfm_ctxreg",
        "n_subjects": 151,
        "density": 512,
        "k": 3,
        "n_rois": 400,
        "subjects": subjects,
        "roi_registry": _record(module, registry),
        "combinations": combinations,
    }
    (root / "template_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return root, prototypes


def _write_subject_inputs(module, root: Path, subject: str, prototypes: np.ndarray):
    shared_fc_root = root / "shared-fc"
    surface_root = root / "surface-inputs"
    hippunfold_root = root / "hippunfold"
    labels = (np.arange(512, dtype=np.int32) % 3) + 1
    raw_fc = np.tanh(prototypes[labels - 1]).astype(np.float32)
    for hemi in ("L", "R"):
        fc_path = (
            shared_fc_root
            / f"sub-{subject}"
            / "fc"
            / "kong2022"
            / "2mm"
            / f"hemi_{hemi}"
            / "hipp_vertex_to_roi_fc.npy"
        )
        fc_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(fc_path, raw_fc)
        mask_path = (
            surface_root
            / f"sub-{subject}"
            / "surface"
            / "tsnr_gate"
            / f"sub-{subject}_hemi-{hemi}_valid_mask.npy"
        )
        mask_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(mask_path, np.ones(512, dtype=bool))
        surface_path = (
            hippunfold_root
            / f"sub-{subject}"
            / "hippunfold"
            / f"sub-{subject}"
            / "surf"
            / (
                f"sub-{subject}_hemi-{hemi}_space-T1w_den-512_"
                "label-hipp_midthickness.surf.gii"
            )
        )
        _write_pointset(surface_path)
    return shared_fc_root, surface_root, hippunfold_root


def test_load_template_uses_a_hermetic_ctxreg_fixture(tmp_path: Path):
    module = load_module()
    template_root, _ = _write_template_fixture(
        module, tmp_path / "template", template_id=module.TEMPLATE_ID
    )

    info = module.load_template(template_root)

    assert info["manifest"]["template_id"] == "hcp151_routea_ctxreg"
    assert info["manifest_fingerprint"]["path"] == str(
        (template_root / "template_manifest.json").resolve()
    )
    assert set(info["prototypes"]) == {"L", "R"}


def test_load_template_rejects_non_ctxreg_identity(tmp_path: Path):
    module = load_module()
    template_root, _ = _write_template_fixture(
        module, tmp_path / "template", template_id="hcp151_routea_full_k3_fisherz"
    )

    with pytest.raises(ValueError, match="hcp151_routea_ctxreg"):
        module.load_template(template_root)


def test_default_roster_is_explicit_hc67_plus_patient26_without_union_manifest(
    tmp_path: Path,
):
    module = load_module()
    hc = tmp_path / "hc.txt"
    patient = tmp_path / "patient.txt"
    hc.write_text("1001\n1002\n", encoding="utf-8")
    patient.write_text("3001\n3028\n", encoding="utf-8")

    assert module.load_subjects(manifest_paths=[hc, patient]) == [
        "1001",
        "1002",
        "3001",
        "3028",
    ]


def test_run_preflights_all_subjects_before_creating_output_dirs(tmp_path: Path):
    module = load_module()
    template_root, prototypes = _write_template_fixture(
        module, tmp_path / "template", template_id=module.TEMPLATE_ID
    )
    shared_fc_root, surface_root, hippunfold_root = _write_subject_inputs(
        module, tmp_path, "1001", prototypes
    )
    out_root = tmp_path / "outputs"

    with pytest.raises(FileNotFoundError, match="sub-1002"):
        module.main(
            [
                "--mode",
                "run",
                "--subjects",
                "1001",
                "1002",
                "--template-root",
                str(template_root),
                "--shared-fc-root",
                str(shared_fc_root),
                "--surface-root",
                str(surface_root),
                "--hippunfold-root",
                str(hippunfold_root),
                "--out-root",
                str(out_root),
            ]
        )

    assert not out_root.exists()


def test_resume_rejects_input_fingerprint_drift(tmp_path: Path):
    module = load_module()
    template_root, prototypes = _write_template_fixture(
        module, tmp_path / "template", template_id=module.TEMPLATE_ID
    )
    shared_fc_root, surface_root, hippunfold_root = _write_subject_inputs(
        module, tmp_path, "1001", prototypes
    )
    evidence = module.preflight_subject(
        "1001",
        shared_fc_root=shared_fc_root,
        surface_root=surface_root,
        hippunfold_root=hippunfold_root,
    )
    template_info = module.load_template(template_root)
    out_root = tmp_path / "outputs"
    result = module.process_subject(
        "1001",
        template_info=template_info,
        evidence=evidence,
        shared_fc_root=shared_fc_root,
        surface_root=surface_root,
        hippunfold_root=hippunfold_root,
        out_root=out_root,
    )
    assert result["status"] == "ok"
    reused = module.process_subject(
        "1001",
        template_info=template_info,
        shared_fc_root=shared_fc_root,
        surface_root=surface_root,
        hippunfold_root=hippunfold_root,
        out_root=out_root,
        resume=True,
    )
    assert reused["status"] == "ok"

    changed_fc = (
        shared_fc_root
        / "sub-1001"
        / "fc"
        / "kong2022"
        / "2mm"
        / "hemi_L"
        / "hipp_vertex_to_roi_fc.npy"
    )
    changed = np.load(changed_fc)
    changed[0, 0] = np.float32(0.123)
    np.save(changed_fc, changed)

    with pytest.raises(ValueError, match="fingerprint"):
        module.process_subject(
            "1001",
            template_info=template_info,
            shared_fc_root=shared_fc_root,
            surface_root=surface_root,
            hippunfold_root=hippunfold_root,
            out_root=out_root,
            resume=True,
        )
