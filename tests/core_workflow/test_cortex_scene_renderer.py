"""The callable cortex renderer preserves scene editing and Workbench errors."""
from pathlib import Path
import sys
import xml.etree.ElementTree as ET

import numpy as np
import pytest

from lib import render_cortex_pfm_scene as module
from lib import cortex_template_profiles as profiles


@pytest.fixture
def cortex_asset_inputs(tmp_path):
    import nibabel as nib
    from nibabel.cifti2 import BrainModelAxis, ScalarAxis

    data_root = tmp_path / "data"
    anat = data_root / "sub-01" / "anat"
    anat.mkdir(parents=True)
    sources = {
        "left_inflated": anat / "sub-01_hemi-L_space-fsLR_den-32k_desc-MSMAll_inflated.surf.gii",
        "right_inflated": anat / "sub-01_hemi-R_space-fsLR_den-32k_desc-MSMAll_inflated.surf.gii",
        "sulc_dscalar": anat / "sub-01_space-fsLR_den-32k_desc-MSMAll_sulc.dscalar.nii",
        "scene": tmp_path / "template.scene",
        "dlabel": tmp_path / "atlas.dlabel.nii",
    }
    # Missing-input tests omit a file at creation, without deleting fixtures.
    sources["right_inflated"].touch()
    sources["scene"].write_text("<Scene/>")
    left = BrainModelAxis.from_surface(np.arange(3), 3, name="CortexLeft")
    right = BrainModelAxis.from_surface(np.arange(4), 4, name="CortexRight")
    nib.save(nib.Cifti2Image(
        np.array([[1, 2, 3, 10, 20, 30, 40]], dtype=np.float32),
        header=nib.Cifti2Header.from_axes((ScalarAxis(["sulc"]), left + right)),
    ), sources["sulc_dscalar"])
    legacy = tmp_path / "legacy_assets"
    legacy.mkdir()
    # These alternative files must never rescue or override a declared input.
    for name in (
        "sub-01_hemi-L_space-fsLR_den-32k_desc-MSMAll_inflated.surf.gii",
        "sub-01_hemi-R_space-fsLR_den-32k_desc-MSMAll_inflated.surf.gii",
        "sub-01_hemi-L_space-fsLR_den-32k_desc-MSMAll_sulc.func.gii",
        "sub-01_hemi-R_space-fsLR_den-32k_desc-MSMAll_sulc.func.gii",
        "sub-01_space-fsLR_den-32k_desc-MSMAll_sulc.dscalar.nii",
    ):
        (legacy / name).touch()
    return data_root, sources, legacy, tmp_path / "render" / "cortex_assets"


@pytest.mark.parametrize("missing", ["left_inflated", "dlabel"])
def test_cortex_assets_reject_missing_exact_input_despite_alternatives(cortex_asset_inputs, missing):
    import re

    data_root, sources, legacy, sulc_dir = cortex_asset_inputs
    if missing != "left_inflated":
        sources["left_inflated"].touch()
    if missing != "dlabel":
        sources["dlabel"].touch()
    with pytest.raises(FileNotFoundError, match=re.escape(str(sources[missing]))):
        profiles.resolve_cortex_assets(
            subject="01", data_root=data_root, sulc_dir=sulc_dir,
            scene=sources["scene"], dlabel=sources["dlabel"],
        )
    assert not sulc_dir.exists()


def test_cortex_assets_split_declared_sulc_and_refresh_existing_outputs(cortex_asset_inputs):
    import nibabel as nib

    data_root, sources, legacy, sulc_dir = cortex_asset_inputs
    sources["left_inflated"].touch()
    sources["dlabel"].touch()
    kwargs = dict(subject="01", data_root=data_root, sulc_dir=sulc_dir,
                  scene=sources["scene"], dlabel=sources["dlabel"])
    assets = profiles.resolve_cortex_assets(**kwargs)
    for name, path in sources.items():
        assert assets[name] == path
    assert assets["left_sulc"] == sulc_dir / "sub-01_hemi-L_space-fsLR_den-32k_desc-MSMAll_sulc.func.gii"
    assert assets["right_sulc"] == sulc_dir / "sub-01_hemi-R_space-fsLR_den-32k_desc-MSMAll_sulc.func.gii"
    np.testing.assert_array_equal(nib.load(assets["left_sulc"]).darrays[0].data, [1, 2, 3])
    np.testing.assert_array_equal(nib.load(assets["right_sulc"]).darrays[0].data, [10, 20, 30, 40])
    original = nib.load(sources["sulc_dscalar"])
    nib.save(nib.Cifti2Image(np.array([[4, 5, 6, 50, 60, 70, 80]], np.float32), header=original.header),
             sources["sulc_dscalar"])
    refreshed = profiles.resolve_cortex_assets(**kwargs)
    np.testing.assert_array_equal(nib.load(refreshed["left_sulc"]).darrays[0].data, [4, 5, 6])
    np.testing.assert_array_equal(nib.load(refreshed["right_sulc"]).darrays[0].data, [50, 60, 70, 80])


def test_callable_renderer_rewrites_assets_rotation_and_background(tmp_path, monkeypatch):
    source = tmp_path / "template.scene"
    source.write_text('''<Scene>
<Object Type="pathName">old_hemi-L_inflated.surf.gii</Object>
<Object Type="string">old.L.label.gii</Object>
<Object Class="BackgroundAndForegroundColorsSceneHelper">
<ObjectArray Name="m_Background"><Element>0</Element></ObjectArray>
<ObjectArray Name="m_Foreground"><Element>255</Element></ObjectArray>
</Object>
<Object Class="ViewingTransformations" Name="m_viewingTransformation">
<Object Name="m_rotationMatrix">''' + "".join(f"<Element>{int(i % 5 == 0)}</Element>" for i in range(16)) + "</Object></Object></Scene>")
    names = ["left_surface", "right_surface", "left_label", "right_label",
             "left_sulc", "right_sulc", "sulc_dscalar", "dlabel"]
    assets = {}
    for name in names:
        path = tmp_path / name
        path.touch()
        assets[name] = path
    calls = []
    monkeypatch.setattr(module, "capture_scene", lambda *args: calls.append(args))
    out_scene, out_png = tmp_path / "out.scene", tmp_path / "out.png"
    module.render_cortex_pfm_scene(
        source_scene=source, **assets, out_scene=out_scene, out_png=out_png,
        width=321, height=123, renderer="OSMesa", rotation_axis="x", rotation_deg=90,
    )
    root = ET.parse(out_scene).getroot()
    assert root.find("Object").text == str(assets["left_surface"])
    assert root.findall("Object")[1].text == assets["left_label"].name
    assert [node.text for node in root.findall(".//ObjectArray/Element")] == ["255", "0"]
    matrix = module.read_rotation_matrix(module.find_rotation_matrix_element(root))
    np.testing.assert_allclose(matrix[:3, :3], module.axis_rotation("x", 90), atol=1e-8)
    assert calls == [(out_scene, 1, out_png, 321, 123, "OSMesa")]


def test_callable_renderer_propagates_workbench_failure(tmp_path, monkeypatch):
    source = tmp_path / "template.scene"
    source.write_text("<Scene/>")
    def fail(*args):
        raise RuntimeError("Workbench failed")
    monkeypatch.setattr(module, "capture_scene", fail)
    assets = {key: source for key in ["left_surface", "right_surface", "left_label",
        "right_label", "left_sulc", "right_sulc", "sulc_dscalar", "dlabel"]}
    with pytest.raises(RuntimeError, match="Workbench failed"):
        module.render_cortex_pfm_scene(source_scene=source, **assets,
            out_scene=tmp_path / "out.scene", out_png=tmp_path / "out.png")


def test_cluster_panels_reuse_the_same_callable_scene_renderer(tmp_path, monkeypatch):
    import importlib
    from PIL import Image
    panels = importlib.import_module('scripts.core_workflow.06_03_build_cluster_cortex_panels')
    calls = []
    def render(**kwargs):
        calls.append(kwargs)
        Image.new('RGB', (64, 64), 'white').save(kwargs['out_png'])
    monkeypatch.setattr(panels, 'render_cortex_pfm_scene', render)
    assets = {key: tmp_path / key for key in ['scene', 'left_inflated', 'right_inflated',
        'left_sulc', 'right_sulc', 'sulc_dscalar', 'dlabel']}
    labels = dict(left_label=tmp_path / 'L.label.gii', right_label=tmp_path / 'R.label.gii')
    first = panels.render_cortex_single_image(cluster_dir=tmp_path / 'single',
        cortex_assets=assets, **labels)
    assert first.exists()
    figure = panels.render_cluster_figure(cluster_dir=tmp_path / 'figure', title='Test',
        cortex_assets=assets, **labels, selected_items=[], network_colors={},
        n_selected_positive=0, n_selected_negative=0)
    assert figure.exists()
    assert len(calls) == 2
    for call in calls:
        assert call['source_scene'] == assets['scene']
        assert call['left_surface'] == assets['left_inflated']
        assert call['right_surface'] == assets['right_inflated']
        assert call['left_label'] == labels['left_label']
        assert call['right_label'] == labels['right_label']
        assert (call['width'], call['height']) == (1000, 700)


def test_subject_cortex_views_preserve_assets_angles_and_capture_options(tmp_path, monkeypatch):
    import importlib
    views = importlib.import_module('scripts.core_workflow.06_09_plot_cortex_kong2022_mshbm')
    inputs = {name: tmp_path / name for name in ['left', 'right', 'sulc', 'scene']}
    for path in inputs.values():
        path.touch()
    labels = tmp_path / 'labels/sub-01'
    labels.mkdir(parents=True)
    for name in ['lh_labels.txt', 'rh_labels.txt']:
        (labels / name).touch()
    outputs = [tmp_path / name for name in ['L.label.gii','R.label.gii','labels.dlabel.nii', 'L.sulc.gii','R.sulc.gii']]
    monkeypatch.setattr(views, 'load_kong2022_bundle', lambda: {})
    monkeypatch.setattr(views, 'load_network_colors', lambda: {})
    monkeypatch.setattr(views, 'generate_gifti_labels', lambda **kwargs: tuple(outputs[:2]))
    monkeypatch.setattr(views, 'create_cifti_dlabel', lambda **kwargs: outputs[2])
    monkeypatch.setattr(views, 'separate_sulc', lambda **kwargs: tuple(outputs[3:]))
    monkeypatch.setattr(views, 'compose_overview', lambda **kwargs: None)
    calls = []
    monkeypatch.setattr(views, 'render_cortex_pfm_scene', lambda **kwargs: calls.append(kwargs))
    monkeypatch.setattr(sys, 'argv', ['render', '--subject', '01', '--routea-root', str(labels.parent),
        '--left-surface-template', str(inputs['left']), '--right-surface-template', str(inputs['right']),
        '--sulc-template', str(inputs['sulc']), '--source-scene', str(inputs['scene']),
        '--out-dir', str(tmp_path / 'out'), '--width', '321', '--height', '123', '--scene-index', '2'])
    assert views.run_render_stage() == 0
    assert [(c['rotation_axis'], c['rotation_deg']) for c in calls] == [
        (None, 0.), ('y', -90.), ('x', 90.), ('x', -90.)]
    for call in calls:
        assert call['source_scene'] == inputs['scene']
        assert call['left_surface'] == inputs['left']
        assert call['right_surface'] == inputs['right']
        assert call['left_label'] == outputs[0] and call['right_label'] == outputs[1]
        assert (call['width'], call['height'], call['scene_index'], call['renderer']) == (321, 123, 2, 'OSMesa')
