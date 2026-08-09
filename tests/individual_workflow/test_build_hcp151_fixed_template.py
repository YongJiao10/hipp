from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
MODULE_PATH = (
    REPO_ROOT
    / "scripts"
    / "individual_workflow"
    / "build_hcp151_fixed_template.py"
)


def load_module():
    assert MODULE_PATH.is_file(), "HCP151 fixed-template builder is missing"
    spec = importlib.util.spec_from_file_location(
        "build_hcp151_fixed_template",
        MODULE_PATH,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _combo_dir(group_root: Path, smoothing: str, hemi: str) -> Path:
    return (
        group_root
        / "gradient-spectral"
        / "kong2022"
        / smoothing
        / "cov_0p80"
        / "g1g2g3"
        / f"hemi_{hemi}"
    )


def _write_sources(
    root: Path,
    *,
    n_subjects: int = 151,
    k: int = 3,
    n_vertices: int = 512,
    manifest_smoothing_override: str | None = None,
    manifest_hemi_override: str | None = None,
) -> tuple[Path, Path]:
    group_root = root / "group_atlas" / "routea_full_k3_fisherz"
    gradient_root = root / "hippomaps_gradients" / "routea"
    subjects = [f"sub-{index:06d}" for index in range(n_subjects)]
    registry_text = "roi_id\troi_name\themi\tfamily17\tsubregion\trank\tcanonical8\n" + "".join(
        f"{roi_id}\tROI_{roi_id:03d}\t{'L' if roi_id <= 200 else 'R'}\tF\tS\t1\tDefault\n"
        for roi_id in range(1, 401)
    )
    template_surfaces = root / "template_surfaces"
    template_surfaces.mkdir(parents=True)
    for hemi in ("L", "R"):
        (template_surfaces / (
            f"group-average_hemi-{hemi}_space-corobl_den-512_"
            "label-hipp_midthickness.surf.gii"
        )).write_bytes(b"surface")

    labels = np.zeros(n_vertices, dtype=np.int32)
    for vertex_index in range(n_vertices):
        labels[vertex_index] = (vertex_index % max(k, 1)) + 1
    support = np.ones(n_vertices, dtype=np.uint8)
    base_profile = np.linspace(-0.8, 0.8, 400, dtype=np.float32)
    group_fc = np.stack(
        [base_profile * np.float32(labels[index]) for index in range(n_vertices)],
        axis=0,
    ).astype(np.float32)
    if n_vertices:
        group_fc[0, 0] = np.inf

    for smoothing in ("2mm", "4mm"):
        for hemi in ("L", "R"):
            group_run = _combo_dir(group_root, smoothing, hemi)
            group_run.mkdir(parents=True)
            gradient_run = gradient_root / smoothing / f"hemi_{hemi}"
            gradient_run.mkdir(parents=True)
            np.save(group_run / "group_labels_final.npy", labels)
            np.save(group_run / "group_support_mask.npy", support)
            np.save(
                group_run / "cluster_roi_profile.npy",
                np.full((max(k, 1), 400), -999.0, dtype=np.float32),
            )
            np.save(gradient_run / "group_mean_fisherz_fc_full.npy", group_fc)
            registry = group_run / "cluster_roi_registry.tsv"
            registry.write_text(registry_text, encoding="utf-8")
            surface = template_surfaces / (
                f"group-average_hemi-{hemi}_space-corobl_den-512_"
                "label-hipp_midthickness.surf.gii"
            )
            group_manifest = {
                "status": "ok",
                "method": "gradient-spectral",
                "atlas": "kong2022",
                "smoothing": manifest_smoothing_override or smoothing,
                "hemi": manifest_hemi_override or hemi,
                "n_subjects": n_subjects,
                "subjects": subjects,
                "chosen_k_final": k,
                "cluster_label_ordering": "long_axis_head_to_tail",
                "density_evidence": {
                    "n_vertices": n_vertices,
                    "surface_path": str(surface.resolve()),
                },
                "template_left_or_right_surface": str(surface.resolve()),
                "cluster_roi_registry": str(registry.resolve()),
                "group_labels_final": str(
                    (group_run / "group_labels_final.npy").resolve()
                ),
                "group_support_mask": str(
                    (group_run / "group_support_mask.npy").resolve()
                ),
            }
            (group_run / "group_atlas_manifest.json").write_text(
                json.dumps(group_manifest),
                encoding="utf-8",
            )
            gradient_manifest = {
                "status": "ok",
                "smoothing": manifest_smoothing_override or smoothing,
                "hemi": manifest_hemi_override or hemi,
                "n_subjects": n_subjects,
                "subjects": subjects,
                "fisher_z_preprocessing": (
                    "arctanh clipped to +/-0.999999 before subject averaging"
                ),
            }
            (gradient_run / "hippomaps_gradient_manifest.json").write_text(
                json.dumps(gradient_manifest),
                encoding="utf-8",
            )
    return group_root, gradient_root


def test_build_writes_four_strict_templates_from_group_mean_fisherz_fc(tmp_path: Path):
    workflow = load_module()
    group_root, gradient_root = _write_sources(tmp_path / "sources")
    out_root = tmp_path / "template"

    manifest = workflow.build_template(
        group_root=group_root,
        gradient_root=gradient_root,
        out_root=out_root,
    )

    assert manifest["status"] == "complete"
    assert manifest["n_subjects"] == 151
    assert manifest["density"] == 512
    assert manifest["k"] == 3
    assert manifest["n_rois"] == 400
    assert len(manifest["combinations"]) == 4
    registry_sha = manifest["roi_registry"]["sha256"]
    assert len(registry_sha) == 64
    for smoothing in ("2mm", "4mm"):
        for hemi in ("L", "R"):
            prefix = out_root / f"hcp151_{smoothing}_hemi-{hemi}"
            labels = np.load(prefix.with_name(prefix.name + "_labels.npy"))
            support = np.load(prefix.with_name(prefix.name + "_support.npy"))
            prototypes = np.load(prefix.with_name(prefix.name + "_prototypes.npy"))
            assert labels.shape == (512,)
            assert labels.dtype == np.int32
            assert support.shape == (512,)
            assert support.dtype == np.uint8
            assert prototypes.shape == (3, 400)
            assert prototypes.dtype == np.float32
            assert not np.any(prototypes == np.float32(-999.0))
    assert workflow.validate_template(out_root=out_root)["status"] == "complete"


@pytest.mark.parametrize(
    ("fixture_kwargs", "message"),
    [
        ({"n_subjects": 150}, "151"),
        ({"k": 2}, "chosen_k_final=3"),
        ({"n_vertices": 511}, "512"),
        ({"manifest_smoothing_override": "8mm"}, "smoothing"),
        ({"manifest_hemi_override": "X"}, "hemi"),
    ],
)
def test_preflight_rejects_noncanonical_hcp_template_sources(
    tmp_path: Path,
    fixture_kwargs: dict[str, object],
    message: str,
):
    workflow = load_module()
    group_root, gradient_root = _write_sources(
        tmp_path / "sources",
        **fixture_kwargs,
    )

    with pytest.raises(ValueError, match=message):
        workflow.preflight_sources(
            group_root=group_root,
            gradient_root=gradient_root,
        )


def test_validate_rejects_source_fingerprint_drift(tmp_path: Path):
    workflow = load_module()
    group_root, gradient_root = _write_sources(tmp_path / "sources")
    out_root = tmp_path / "template"
    workflow.build_template(
        group_root=group_root,
        gradient_root=gradient_root,
        out_root=out_root,
    )
    labels_path = _combo_dir(group_root, "2mm", "L") / "group_labels_final.npy"
    labels = np.load(labels_path)
    labels[0] = 2
    np.save(labels_path, labels)

    with pytest.raises(ValueError, match="fingerprint"):
        workflow.validate_template(out_root=out_root)


def test_validate_rejects_output_shape_or_dtype_drift(tmp_path: Path):
    workflow = load_module()
    group_root, gradient_root = _write_sources(tmp_path / "sources")
    out_root = tmp_path / "template"
    workflow.build_template(
        group_root=group_root,
        gradient_root=gradient_root,
        out_root=out_root,
    )
    labels_path = out_root / "hcp151_2mm_hemi-L_labels.npy"
    np.save(labels_path, np.ones(512, dtype=np.int16))

    with pytest.raises(ValueError, match="dtype"):
        workflow.validate_template(out_root=out_root)


def test_preflight_mode_does_not_create_output_directory(tmp_path: Path):
    workflow = load_module()
    group_root, gradient_root = _write_sources(tmp_path / "sources")
    out_root = tmp_path / "template"

    exit_code = workflow.main(
        [
            "--mode",
            "preflight",
            "--group-root",
            str(group_root),
            "--gradient-root",
            str(gradient_root),
            "--out-root",
            str(out_root),
        ]
    )

    assert exit_code == 0
    assert not out_root.exists()
