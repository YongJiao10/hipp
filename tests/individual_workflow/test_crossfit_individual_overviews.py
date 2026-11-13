from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import nibabel as nib
import numpy as np
import pytest
from PIL import Image, ImageDraw
from nibabel.gifti import GiftiDataArray, GiftiImage, GiftiLabel, GiftiLabelTable


REPO_ROOT = Path(__file__).resolve().parents[2]
MODULE_PATH = (
    REPO_ROOT
    / "scripts"
    / "individual_workflow"
    / "02_01_render_hcp_crossfit_individual_overviews.py"
)


def load_module():
    spec = importlib.util.spec_from_file_location(
        "crossfit_individual_overviews",
        MODULE_PATH,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def fingerprint(path: Path) -> dict[str, object]:
    data = path.read_bytes()
    return {"sha256": hashlib.sha256(data).hexdigest(), "size_bytes": len(data)}


def write_surface(path: Path) -> None:
    coordinates = np.asarray(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
        dtype=np.float32,
    )
    faces = np.asarray([[0, 1, 2], [0, 1, 3]], dtype=np.int32)
    image = GiftiImage(
        darrays=[
            GiftiDataArray(data=coordinates, intent="NIFTI_INTENT_POINTSET"),
            GiftiDataArray(data=faces, intent="NIFTI_INTENT_TRIANGLE"),
        ]
    )
    nib.save(image, str(path))


def write_crossfit_label(path: Path) -> None:
    labels = np.asarray([1, 1, 2, 3], dtype=np.int32)
    table = GiftiLabelTable()
    for key, name, rgba in (
        (0, "Null", (0.62, 0.62, 0.62, 1.0)),
        (1, "Anterior", (0.86, 0.31, 0.29, 1.0)),
        (2, "Middle", (0.92, 0.69, 0.27, 1.0)),
        (3, "Posterior", (0.27, 0.45, 0.75, 1.0)),
    ):
        label = GiftiLabel(key=key, red=rgba[0], green=rgba[1], blue=rgba[2], alpha=rgba[3])
        label.label = name
        table.labels.append(label)
    image = GiftiImage(
        darrays=[GiftiDataArray(data=labels, intent="NIFTI_INTENT_LABEL")],
        labeltable=table,
    )
    nib.save(image, str(path))


def build_crossfit_inputs(
    tmp_path: Path,
    subject: str = "100001",
    additional_subjects: tuple[str, ...] = (),
) -> tuple[Path, Path]:
    subject_ids = (subject, *additional_subjects)
    crossfit_root = tmp_path / "crossfit"
    scene = tmp_path / "locked.scene"
    scene.write_text("synthetic scene\n", encoding="utf-8")

    for subject_id in subject_ids:
        surface_root = tmp_path / "hippunfold" / f"sub-{subject_id}" / "surf"
        surface_root.mkdir(parents=True)
        for smoothing in ("2mm", "4mm"):
            for hemi in ("L", "R"):
                surface = surface_root / (
                    f"sub-{subject_id}_hemi-{hemi}_space-corobl_den-512_"
                    "label-hipp_midthickness.surf.gii"
                )
                write_surface(surface)
                spec = surface_root / f"sub-{subject_id}_den-512_label-hipp_surfaces.spec"
                spec.write_text("synthetic spec\n", encoding="utf-8")
                combo_dir = (
                    crossfit_root
                    / "subjects"
                    / f"sub-{subject_id}"
                    / smoothing
                    / f"hemi_{hemi}"
                )
                combo_dir.mkdir(parents=True)
                label = combo_dir / (
                    f"sub-{subject_id}_hemi-{hemi}_space-corobl_den-512_"
                    f"label-crossfit_template_matching_k3_{smoothing}.label.gii"
                )
                write_crossfit_label(label)
                manifest = {
                    "status": "ok",
                    "subject": f"sub-{subject_id}",
                    "smoothing": smoothing,
                    "hemi": hemi,
                    "subject_was_in_template": False,
                    "label_semantics": {"1": "Anterior", "2": "Middle", "3": "Posterior"},
                    "density_evidence": {
                        "consumed_subject_surface": str(surface.resolve()),
                        "filename_density": "512",
                        "evidence_source": "consumed_surface_paths",
                    },
                    "inputs": {"subject_surface": str(surface.resolve())},
                    "input_fingerprints": {"subject_surface": fingerprint(surface)},
                    "outputs": {"label_gifti": str(label.resolve())},
                    "output_fingerprints": {"label_gifti": fingerprint(label)},
                }
                (combo_dir / "combo_manifest.json").write_text(
                    json.dumps(manifest, indent=2) + "\n",
                    encoding="utf-8",
                )

    (crossfit_root / "crossfit_manifest.json").write_text(
        json.dumps(
            {
                "status": "ok",
                "n_subjects": len(subject_ids),
                "smoothings": ["2mm", "4mm"],
                "hemis": ["L", "R"],
                "subject_combo_mappings_completed": len(subject_ids) * 4,
                "run_config": {"subjects": [f"sub-{value}" for value in subject_ids]},
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return crossfit_root, scene


def fake_locked_grid(calls: list[dict[str, object]]):
    def render(**kwargs):
        call = dict(kwargs)
        for side in ("left", "right"):
            label_image = nib.load(str(kwargs[f"{side}_labels"]))
            call[f"{side}_display_label_table"] = {
                int(label.key): {
                    "name": label.label,
                    "rgba": [label.red, label.green, label.blue, label.alpha],
                }
                for label in label_image.labeltable.labels
            }
        calls.append(call)
        outdir = Path(str(kwargs["outdir"]))
        outdir.mkdir(parents=True, exist_ok=True)
        smoothing = "2mm" if "2mm" in str(kwargs["name"]) else "4mm"
        image = Image.new(
            "RGB",
            (640, 300),
            (70, 110, 180) if smoothing == "2mm" else (180, 110, 70),
        )
        output = outdir / f"{smoothing}_render.png"
        image.save(output)
        return {"surface_png": str(output), "biglegend_png": str(output)}

    return render


def _synthetic_kong2022_registry() -> list[dict[str, object]]:
    from scripts.core_workflow.lib.kong2022_fc_contract import authoritative_kong2022_roi_registry

    return authoritative_kong2022_roi_registry()



def _synthetic_template_profile(offset: float = 0.0) -> np.ndarray:
    registry = _synthetic_kong2022_registry()
    scores = {
        "1": {"VentralAttention": 0.08 + offset, "Somatomotor": 0.04, "Default": 0.01},
        "2": {"VentralAttention": 0.01, "Somatomotor": 0.09 + offset, "Default": 0.03},
        "3": {"VentralAttention": -0.01, "Somatomotor": 0.02, "Default": 0.11 + offset},
    }
    profile = np.zeros((3, 400), dtype=np.float32)
    for column, row in enumerate(registry):
        network = str(row["canonical7"])
        for label in range(1, 4):
            profile[label - 1, column] = scores[str(label)].get(network, -0.02)
    return profile


def _synthetic_network_colors() -> dict[str, tuple[int, int, int, int]]:
    # Legacy aliases deliberately differ, so canonical colors must be selected.
    return {
        "VAN": (231, 150, 156, 255),
        "DMN": (196, 156, 148, 255),
        "VentralAttention": (220, 92, 255, 255),
        "Somatomotor": (99, 150, 194, 255),
        "Default": (216, 92, 107, 255),
    }


def _synthetic_visual_label_contract(overview) -> dict[str, object]:
    contract = overview.summarize_fisherz_network_profiles(
        profile_records=[({"fold": 1}, _synthetic_template_profile())],
        roi_registry=_synthetic_kong2022_registry(),
        network_colors=_synthetic_network_colors(),
    )
    contract["crossfit_root"] = "/synthetic/crossfit"
    contract["contract_fingerprint"] = overview._visual_label_contract_fingerprint(contract)
    return contract


def _stub_network_contract(overview, monkeypatch: pytest.MonkeyPatch) -> dict[str, object]:
    contract = _synthetic_visual_label_contract(overview)
    monkeypatch.setattr(
        overview,
        "derive_fisherz_network_label_contract",
        lambda **_kwargs: json.loads(json.dumps(contract)),
    )
    return contract


def test_fisherz_profiles_define_stable_van_smn_dmn_display_contract():
    overview = load_module()
    profile_records = [
        ({"fold": 1, "smoothing": "2mm", "hemi": "L"}, _synthetic_template_profile()),
        ({"fold": 1, "smoothing": "4mm", "hemi": "R"}, _synthetic_template_profile(0.001)),
    ]

    contract = overview.summarize_fisherz_network_profiles(
        profile_records=profile_records,
        roi_registry=_synthetic_kong2022_registry(),
        network_colors=_synthetic_network_colors(),
    )

    assert {
        code: row["display_name"]
        for code, row in contract["label_mapping"].items()
    } == {"1": "VAN", "2": "SMN", "3": "DMN"}
    assert {
        code: row["canonical_network"]
        for code, row in contract["label_mapping"].items()
    } == {
        "1": "VentralAttention",
        "2": "Somatomotor",
        "3": "Default",
    }
    assert contract["template_count"] == 2
    assert {
        code: (row["style_network"], row["rgba"])
        for code, row in contract["label_mapping"].items()
    } == {
        "1": ("VentralAttention", [220, 92, 255, 255]),
        "2": ("Somatomotor", [99, 150, 194, 255]),
        "3": ("Default", [216, 92, 107, 255]),
    }


def test_network_display_gifti_keeps_codes_but_replaces_legend_labels_and_colors(
    tmp_path: Path,
):
    overview = load_module()
    source = tmp_path / "source.label.gii"
    output = tmp_path / "display.label.gii"
    write_crossfit_label(source)
    contract = overview.summarize_fisherz_network_profiles(
        profile_records=[({"fold": 1}, _synthetic_template_profile())],
        roi_registry=_synthetic_kong2022_registry(),
        network_colors=_synthetic_network_colors(),
    )

    overview.write_network_display_label_gifti(
        source_label_path=source,
        output_path=output,
        visual_label_contract=contract,
    )

    image = nib.load(str(output))
    assert np.array_equal(image.darrays[0].data, nib.load(str(source)).darrays[0].data)
    labels = {int(label.key): label for label in image.labeltable.labels}
    assert {key: labels[key].label for key in (1, 2, 3)} == {
        1: "VAN",
        2: "SMN",
        3: "DMN",
    }
    assert labels[1].red == pytest.approx(220 / 255)
    assert labels[2].green == pytest.approx(150 / 255)
    assert labels[3].red == pytest.approx(216 / 255)


def test_subject_overview_title_wraps_before_subject_id_is_clipped():
    overview = load_module()
    max_width = 1023

    lines = overview._subject_overview_title_lines("100610", max_width=max_width)

    assert lines == [
        "HCP RouteA cross-fit Fisher-z network matching |",
        "sub-100610",
    ]
    draw = ImageDraw.Draw(Image.new("RGB", (8, 8)))
    font = overview.load_font(38)
    assert all(
        draw.textbbox((0, 0), line, font=font)[2] <= max_width
        for line in lines
    )


def test_renderer_uses_network_display_labels_and_records_visual_contract(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    overview = load_module()
    contract = _stub_network_contract(overview, monkeypatch)
    crossfit_root, scene = build_crossfit_inputs(tmp_path)
    calls: list[dict[str, object]] = []
    monkeypatch.setattr(overview, "render_hippocampal_grid", fake_locked_grid(calls))
    output_dir = tmp_path / "present" / "individual"

    outputs = overview.render_individual_overviews(
        crossfit_root=crossfit_root,
        output_dir=output_dir,
        scene=scene,
        subjects=["100001"],
        jobs=1,
    )

    assert {key: row["name"] for key, row in calls[0]["left_display_label_table"].items() if key}
    assert {
        key: row["name"]
        for key, row in calls[0]["left_display_label_table"].items()
        if key
    } == {1: "VAN", 2: "SMN", 3: "DMN"}
    assert calls[0]["left_display_label_table"][1]["rgba"][0] == pytest.approx(220 / 255)
    manifest = json.loads(outputs["manifest"].read_text(encoding="utf-8"))
    item = manifest["items"][0]
    assert item["display_label_semantics"] == {"1": "VAN", "2": "SMN", "3": "DMN"}
    assert item["visual_label_contract_fingerprint"] == contract["contract_fingerprint"]


def test_renderer_requires_refresh_to_replace_legacy_visual_contract(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    overview = load_module()
    _stub_network_contract(overview, monkeypatch)
    crossfit_root, scene = build_crossfit_inputs(tmp_path)
    calls: list[dict[str, object]] = []
    monkeypatch.setattr(overview, "render_hippocampal_grid", fake_locked_grid(calls))
    output_dir = tmp_path / "present" / "individual"

    initial = overview.render_individual_overviews(
        crossfit_root=crossfit_root,
        output_dir=output_dir,
        scene=scene,
        subjects=["100001"],
        jobs=1,
    )
    payload = json.loads(initial["manifest"].read_text(encoding="utf-8"))
    payload["items"][0].pop("visual_label_contract_fingerprint", None)
    initial["manifest"].write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    stale = overview.render_individual_overviews(
        crossfit_root=crossfit_root,
        output_dir=output_dir,
        scene=scene,
        subjects=["100001"],
        jobs=1,
        resume=True,
    )
    stale_payload = json.loads(stale["manifest"].read_text(encoding="utf-8"))
    assert stale_payload["summary"]["status"] == "failed"
    assert "visual-label" in stale_payload["attempt_items"][0]["error"]

    refreshed = overview.render_individual_overviews(
        crossfit_root=crossfit_root,
        output_dir=output_dir,
        scene=scene,
        subjects=["100001"],
        jobs=1,
        resume=True,
        refresh_visuals=True,
    )
    refreshed_payload = json.loads(refreshed["manifest"].read_text(encoding="utf-8"))
    assert refreshed_payload["summary"]["status"] == "ok"
    assert refreshed_payload["items"][0]["resumed"] is False


def test_renderer_writes_one_compact_png_with_2mm_and_4mm_rows(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    overview = load_module()
    _stub_network_contract(overview, monkeypatch)
    crossfit_root, scene = build_crossfit_inputs(tmp_path)
    calls: list[dict[str, object]] = []
    monkeypatch.setattr(overview, "render_hippocampal_grid", fake_locked_grid(calls))
    output_dir = tmp_path / "present" / "individual"

    outputs = overview.render_individual_overviews(
        crossfit_root=crossfit_root,
        output_dir=output_dir,
        scene=scene,
        subjects=["100001"],
        smoothings=["2mm", "4mm"],
        jobs=1,
        resume=False,
    )

    png = output_dir / "sub-100001_crossfit_template_matching_k3_overview.png"
    assert png.exists()
    image = Image.open(png)
    assert image.height > image.width
    assert len(calls) == 2
    assert [str(call["name"]).rsplit("_", 1)[-1] for call in calls] == ["2mm", "4mm"]
    assert all(
        {key: row["name"] for key, row in call["left_display_label_table"].items() if key}
        == {1: "VAN", 2: "SMN", 3: "DMN"}
        for call in calls
    )
    assert all(
        {key: row["name"] for key, row in call["right_display_label_table"].items() if key}
        == {1: "VAN", 2: "SMN", 3: "DMN"}
        for call in calls
    )

    manifest = json.loads(outputs["manifest"].read_text(encoding="utf-8"))
    assert manifest["summary"]["status"] == "ok"
    assert manifest["summary"]["ok"] == 1
    assert manifest["items"][0]["scene"] == str(scene.resolve())
    assert manifest["items"][0]["scene_fingerprint"] == fingerprint(scene)
    assert [row["smoothing"] for row in manifest["items"][0]["rows"]] == ["2mm", "4mm"]
    assert all(row["label_gifti"].endswith(".label.gii") for row in manifest["items"][0]["rows"])


def test_renderer_rejects_stale_label_before_resume_reuses_png(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    overview = load_module()
    _stub_network_contract(overview, monkeypatch)
    crossfit_root, scene = build_crossfit_inputs(tmp_path)
    calls: list[dict[str, object]] = []
    monkeypatch.setattr(overview, "render_hippocampal_grid", fake_locked_grid(calls))
    output_dir = tmp_path / "present" / "individual"

    overview.render_individual_overviews(
        crossfit_root=crossfit_root,
        output_dir=output_dir,
        scene=scene,
        subjects=["100001"],
        smoothings=["2mm", "4mm"],
        jobs=1,
        resume=False,
    )

    with pytest.raises(FileExistsError, match="--resume"):
        overview.render_subject_overview(
            crossfit_root=crossfit_root,
            output_dir=output_dir,
            scene=scene,
            subject="100001",
            smoothings=["2mm", "4mm"],
            resume=False,
        )

    stale_label = (
        crossfit_root
        / "subjects"
        / "sub-100001"
        / "2mm"
        / "hemi_L"
        / "sub-100001_hemi-L_space-corobl_den-512_"
        "label-crossfit_template_matching_k3_2mm.label.gii"
    )
    stale_label.write_bytes(stale_label.read_bytes() + b"stale")
    with pytest.raises(ValueError, match="fingerprint"):
        overview.render_subject_overview(
            crossfit_root=crossfit_root,
            output_dir=output_dir,
            scene=scene,
            subject="100001",
            smoothings=["2mm", "4mm"],
            resume=True,
        )

@pytest.mark.parametrize("smoothings", (["4mm"], ["4mm", "2mm"]))
def test_renderer_requires_the_fixed_2mm_then_4mm_layout(
    tmp_path: Path,
    smoothings: list[str],
):
    overview = load_module()
    crossfit_root, _scene = build_crossfit_inputs(tmp_path)
    manifest = json.loads((crossfit_root / "crossfit_manifest.json").read_text(encoding="utf-8"))

    with pytest.raises(ValueError, match="2mm.*4mm"):
        overview._smoothings(manifest, smoothings)


def test_renderer_rejects_duplicate_requested_subjects(
    tmp_path: Path,
):
    overview = load_module()
    crossfit_root, scene = build_crossfit_inputs(tmp_path)

    with pytest.raises(ValueError, match="duplicate"):
        overview.render_individual_overviews(
            crossfit_root=crossfit_root,
            output_dir=tmp_path / "present" / "individual",
            scene=scene,
            subjects=["100001", "sub-100001"],
            jobs=1,
        )


def test_renderer_rejects_a_surface_for_the_wrong_hemisphere(tmp_path: Path):
    overview = load_module()
    crossfit_root, _scene = build_crossfit_inputs(tmp_path)
    right_manifest_path = (
        crossfit_root / "subjects" / "sub-100001" / "2mm" / "hemi_R" / "combo_manifest.json"
    )
    right_manifest = json.loads(right_manifest_path.read_text(encoding="utf-8"))
    left_surface = (
        crossfit_root
        .parent
        / "hippunfold"
        / "sub-100001"
        / "surf"
        / "sub-100001_hemi-L_space-corobl_den-512_label-hipp_midthickness.surf.gii"
    )
    right_manifest["density_evidence"]["consumed_subject_surface"] = str(left_surface)
    right_manifest_path.write_text(json.dumps(right_manifest, indent=2) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="surface identity"):
        overview.load_subject_combo(
            crossfit_root=crossfit_root,
            subject="100001",
            smoothing="2mm",
            hemi="R",
        )


def test_renderer_rejects_corrupt_png_during_resume(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    overview = load_module()
    _stub_network_contract(overview, monkeypatch)
    crossfit_root, scene = build_crossfit_inputs(tmp_path)
    calls: list[dict[str, object]] = []
    monkeypatch.setattr(overview, "render_hippocampal_grid", fake_locked_grid(calls))
    output_dir = tmp_path / "present" / "individual"

    overview.render_individual_overviews(
        crossfit_root=crossfit_root,
        output_dir=output_dir,
        scene=scene,
        subjects=["100001"],
        jobs=1,
    )
    png = output_dir / "sub-100001_crossfit_template_matching_k3_overview.png"
    png.write_bytes(b"not a PNG")

    outputs = overview.render_individual_overviews(
        crossfit_root=crossfit_root,
        output_dir=output_dir,
        scene=scene,
        subjects=["100001"],
        jobs=1,
        resume=True,
    )
    resumed_manifest = json.loads(outputs["manifest"].read_text(encoding="utf-8"))

    assert resumed_manifest["summary"]["status"] == "failed"
    assert "PNG" in resumed_manifest["attempt_items"][0]["error"]


def test_renderer_rejects_valid_but_modified_png_during_resume(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    overview = load_module()
    _stub_network_contract(overview, monkeypatch)
    crossfit_root, scene = build_crossfit_inputs(tmp_path)
    calls: list[dict[str, object]] = []
    monkeypatch.setattr(overview, "render_hippocampal_grid", fake_locked_grid(calls))
    output_dir = tmp_path / "present" / "individual"

    overview.render_individual_overviews(
        crossfit_root=crossfit_root,
        output_dir=output_dir,
        scene=scene,
        subjects=["100001"],
        jobs=1,
    )
    png = output_dir / "sub-100001_crossfit_template_matching_k3_overview.png"
    Image.new("RGB", (40, 40), (0, 0, 0)).save(png)

    outputs = overview.render_individual_overviews(
        crossfit_root=crossfit_root,
        output_dir=output_dir,
        scene=scene,
        subjects=["100001"],
        jobs=1,
        resume=True,
    )
    resumed_manifest = json.loads(outputs["manifest"].read_text(encoding="utf-8"))

    assert resumed_manifest["summary"]["status"] == "failed"
    assert "fingerprint" in resumed_manifest["attempt_items"][0]["error"]



def test_renderer_refuses_legacy_resume_without_an_output_fingerprint(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    overview = load_module()
    _stub_network_contract(overview, monkeypatch)
    crossfit_root, scene = build_crossfit_inputs(tmp_path)
    calls: list[dict[str, object]] = []
    monkeypatch.setattr(overview, "render_hippocampal_grid", fake_locked_grid(calls))
    output_dir = tmp_path / "present" / "individual"

    outputs = overview.render_individual_overviews(
        crossfit_root=crossfit_root,
        output_dir=output_dir,
        scene=scene,
        subjects=["100001"],
        jobs=1,
    )
    manifest = json.loads(outputs["manifest"].read_text(encoding="utf-8"))
    manifest["items"][0].pop("output_fingerprint")
    outputs["manifest"].write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    resumed = overview.render_individual_overviews(
        crossfit_root=crossfit_root,
        output_dir=output_dir,
        scene=scene,
        subjects=["100001"],
        jobs=1,
        resume=True,
    )
    resumed_manifest = json.loads(resumed["manifest"].read_text(encoding="utf-8"))

    assert resumed_manifest["summary"]["status"] == "failed"
    assert "output fingerprint" in resumed_manifest["attempt_items"][0]["error"]


def test_renderer_refuses_resume_with_a_different_workbench_scene(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    overview = load_module()
    _stub_network_contract(overview, monkeypatch)
    crossfit_root, scene = build_crossfit_inputs(tmp_path)
    calls: list[dict[str, object]] = []
    monkeypatch.setattr(overview, "render_hippocampal_grid", fake_locked_grid(calls))
    output_dir = tmp_path / "present" / "individual"

    overview.render_individual_overviews(
        crossfit_root=crossfit_root,
        output_dir=output_dir,
        scene=scene,
        subjects=["100001"],
        jobs=1,
    )
    other_scene = tmp_path / "other.scene"
    other_scene.write_text("a different synthetic scene\n", encoding="utf-8")

    resumed = overview.render_individual_overviews(
        crossfit_root=crossfit_root,
        output_dir=output_dir,
        scene=other_scene,
        subjects=["100001"],
        jobs=1,
        resume=True,
    )
    resumed_manifest = json.loads(resumed["manifest"].read_text(encoding="utf-8"))

    assert resumed_manifest["summary"]["status"] == "failed"
    assert "scene" in resumed_manifest["attempt_items"][0]["error"].lower()


def test_resume_subset_preserves_untouched_subject_provenance(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    overview = load_module()
    _stub_network_contract(overview, monkeypatch)
    crossfit_root, scene = build_crossfit_inputs(
        tmp_path,
        additional_subjects=("100002",),
    )
    calls: list[dict[str, object]] = []
    monkeypatch.setattr(overview, "render_hippocampal_grid", fake_locked_grid(calls))
    output_dir = tmp_path / "present" / "individual"

    initial = overview.render_individual_overviews(
        crossfit_root=crossfit_root,
        output_dir=output_dir,
        scene=scene,
        jobs=1,
    )
    initial_manifest = json.loads(initial["manifest"].read_text(encoding="utf-8"))
    initial_by_subject = {
        item["subject"]: item for item in initial_manifest["items"]
    }

    resumed = overview.render_individual_overviews(
        crossfit_root=crossfit_root,
        output_dir=output_dir,
        scene=scene,
        subjects=["100001"],
        jobs=1,
        resume=True,
    )
    resumed_manifest = json.loads(resumed["manifest"].read_text(encoding="utf-8"))
    resumed_by_subject = {
        item["subject"]: item for item in resumed_manifest["items"]
    }

    assert set(resumed_by_subject) == {"sub-100001", "sub-100002"}
    assert resumed_by_subject["sub-100002"] == initial_by_subject["sub-100002"]
    assert resumed_manifest["summary"]["requested_subjects"] == 1
    assert resumed_manifest["summary"]["subjects"] == 2




def test_failed_scene_resume_preserves_valid_provenance_and_can_retry(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    overview = load_module()
    _stub_network_contract(overview, monkeypatch)
    crossfit_root, scene = build_crossfit_inputs(tmp_path)
    calls: list[dict[str, object]] = []
    monkeypatch.setattr(overview, "render_hippocampal_grid", fake_locked_grid(calls))
    output_dir = tmp_path / "present" / "individual"

    initial = overview.render_individual_overviews(
        crossfit_root=crossfit_root,
        output_dir=output_dir,
        scene=scene,
        subjects=["100001"],
        jobs=1,
    )
    initial_manifest = json.loads(initial["manifest"].read_text(encoding="utf-8"))
    initial_item = initial_manifest["items"][0]
    other_scene = tmp_path / "other.scene"
    other_scene.write_text("a different synthetic scene\n", encoding="utf-8")

    failed = overview.render_individual_overviews(
        crossfit_root=crossfit_root,
        output_dir=output_dir,
        scene=other_scene,
        subjects=["100001"],
        jobs=1,
        resume=True,
    )
    failed_manifest = json.loads(failed["manifest"].read_text(encoding="utf-8"))

    assert failed_manifest["summary"]["status"] == "failed"
    assert failed_manifest["items"] == [initial_item]
    assert "scene" in failed_manifest["attempt_items"][0]["error"].lower()

    recovered = overview.render_individual_overviews(
        crossfit_root=crossfit_root,
        output_dir=output_dir,
        scene=scene,
        subjects=["100001"],
        jobs=1,
        resume=True,
    )
    recovered_manifest = json.loads(recovered["manifest"].read_text(encoding="utf-8"))

    assert recovered_manifest["summary"]["status"] == "ok"
    assert recovered_manifest["items"][0]["resumed"] is True


def test_failed_nonresume_subset_preserves_existing_aggregate_provenance(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    overview = load_module()
    _stub_network_contract(overview, monkeypatch)
    crossfit_root, scene = build_crossfit_inputs(
        tmp_path,
        additional_subjects=("100002",),
    )
    calls: list[dict[str, object]] = []
    monkeypatch.setattr(overview, "render_hippocampal_grid", fake_locked_grid(calls))
    output_dir = tmp_path / "present" / "individual"

    initial = overview.render_individual_overviews(
        crossfit_root=crossfit_root,
        output_dir=output_dir,
        scene=scene,
        jobs=1,
    )
    initial_manifest = json.loads(initial["manifest"].read_text(encoding="utf-8"))

    failed = overview.render_individual_overviews(
        crossfit_root=crossfit_root,
        output_dir=output_dir,
        scene=scene,
        subjects=["100001"],
        jobs=1,
        resume=False,
    )
    failed_manifest = json.loads(failed["manifest"].read_text(encoding="utf-8"))

    assert failed_manifest["summary"]["status"] == "failed"
    assert failed_manifest["items"] == initial_manifest["items"]
    assert len(failed_manifest["attempt_items"]) == 1
    assert failed_manifest["attempt_items"][0]["error_type"] == "FileExistsError"
def test_output_layout_resolver_places_subject_pngs_under_crossfit_individual_root():
    overview = load_module()

    root = overview.resolve_crossfit_template_matching_individual_overview_root()

    assert root.as_posix().endswith("outputs_hcp/present/overviews/crossfit_template_matching/individual")
